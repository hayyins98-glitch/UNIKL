"""
Rotatable 3D surface view of a volume and its tumor components.

Surfaces come from marching cubes rather than voxel rendering: a triangle mesh
stays interactive inside a notebook where a 96^3 point cloud would not. The
brain is drawn semi-transparent so the tumor inside stays visible, and each
component reuses the colour it has in the 2D component views, so a mass that is
red there is red here.

Volume axes follow BraTS NIfTI ordering: (L-R, A-P, S-I).
"""
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes

from brain_tumor_seg.utils.mask_analysis import (
    ComponentAnalysis,
    analyze_components,
    to_3d,
)
from brain_tumor_seg.visualization.components import _component_colour

AXIS_LABELS = ("L-R", "A-P", "S-I")

# Front-left and slightly above, with S-I pointing up: a recognisable
# three-quarter head view rather than an arbitrary starting angle.
DEFAULT_CAMERA = {
    "eye": {"x": 1.55, "y": -1.55, "z": 0.95},
    "up": {"x": 0.0, "y": 0.0, "z": 1.0},
}

BRAIN_COLOUR = "rgb(188, 190, 200)"

LIGHTING = {"ambient": 0.55, "diffuse": 0.8, "specular": 0.12, "roughness": 0.85}

# Every scene decoration off: no tick numbers, no gridlines, no zero lines, no
# axis panes or bounding box, no hover spikes, no axis titles. `visible: False`
# alone would do it, but the individual flags are spelled out so the intent is
# readable in the layout as well as here.
CLEAN_AXIS = {
    "visible": False,
    "showticklabels": False,
    "showgrid": False,
    "zeroline": False,
    "showline": False,
    "showbackground": False,
    "showspikes": False,
    "ticks": "",
    "title": {"text": ""},
}

# Plotly config for a genuinely interactive figure.
#   scrollZoom      - wheel zooms. plotly.js already defaults to "gl3d+geo+map",
#                     but stating it keeps a stricter global default from
#                     silently disabling the wheel.
#   displayModeBar  - always visible instead of on hover, so the Pan and
#                     orbit/turntable buttons are always reachable.
#   doubleClick     - double click restores the starting camera.
INTERACTION_CONFIG = {
    "scrollZoom": True,
    "displayModeBar": True,
    "displaylogo": False,
    "doubleClick": "reset+autosize",
}

# `vscode` emits the plotly mimetype bundle that Cursor/VS Code renders with
# plotly.js in the output cell, which is the only one of these that keeps the
# scene live without a network round trip or a scratch directory.
# `notebook_connected` is the classic-notebook fallback (needs the CDN), and
# `iframe` is the last resort that works anywhere but writes HTML next to the
# notebook.
PREFERRED_RENDERERS = ("vscode", "notebook_connected", "iframe")


def interaction_config(**overrides) -> Dict[str, Any]:
    """
    Copy of `INTERACTION_CONFIG`, with any key overridden.

    Returned fresh each time because Plotly stores the dict on the figure div
    and callers should not be able to mutate the module default.
    """
    config = dict(INTERACTION_CONFIG)
    config.update(overrides)
    return config


def resolve_renderer(preferred: Sequence[str] = PREFERRED_RENDERERS) -> str:
    """
    First renderer of `preferred` that Plotly knows about.

    `plotly.io.renderers.default` is autodetected from the environment and
    resolves to "browser" inside Cursor, which pops a separate tab instead of
    drawing in the output cell, so the default is deliberately not trusted here.
    """
    for name in preferred:
        if name in pio.renderers:
            return name
    return pio.renderers.default


def _scene_axes(clean: bool) -> Dict[str, dict]:
    """Per-axis scene settings: stripped bare, or anatomically labelled."""
    if clean:
        return {name: deepcopy(CLEAN_AXIS) for name in ("xaxis", "yaxis", "zaxis")}
    return {
        f"{letter}axis": {"title": {"text": label}}
        for letter, label in zip("xyz", AXIS_LABELS)
    }


def _rgb_string(colour: Sequence[float]) -> str:
    """Convert a 0-1 RGB tuple from the shared palette to a Plotly colour."""
    red, green, blue = (int(round(255 * channel)) for channel in colour[:3])
    return f"rgb({red}, {green}, {blue})"


def _select_modality(image, modality: int) -> np.ndarray:
    """Reduce an image to 3D, choosing which channel of a multi-modal volume."""
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()

    array = np.asarray(image)
    while array.ndim > 4 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 4:
        if not 0 <= modality < array.shape[0]:
            raise ValueError(
                f"modality {modality} out of range for {array.shape[0]} channel(s)"
            )
        array = array[modality]

    return to_3d(array)


def brain_surface_level(volume: np.ndarray) -> float:
    """
    Intensity that separates background from tissue, via Otsu's method.

    Volumes are z-scored per case, so the background sits at whatever negative
    constant the case happens to produce; a hardcoded cutoff would swallow the
    whole box for one case and miss the brain entirely for the next. Otsu picks
    the split between the background and tissue modes from the data itself.
    """
    return float(threshold_otsu(np.asarray(volume)))


def _mesh_trace(
    volume: np.ndarray,
    level: float,
    step_size: int,
    colour: str,
    opacity: float,
    name: str,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Optional[go.Mesh3d]:
    """Marching-cubes isosurface as a Mesh3d, or None if nothing crosses `level`."""
    if not volume.min() < level < volume.max():
        return None

    try:
        verts, faces, _, _ = marching_cubes(
            volume, level=level, step_size=step_size, spacing=spacing
        )
    except (ValueError, RuntimeError):
        return None

    if len(faces) == 0:
        return None

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=colour,
        opacity=opacity,
        name=name,
        showlegend=True,
        hoverinfo="name",
        flatshading=False,
        lighting=LIGHTING,
    )


def show_volume_3d(
    image,
    mask=None,
    modality: int = 0,
    level: Optional[float] = None,
    brain_step: int = 2,
    tumor_step: int = 1,
    brain_opacity: float = 0.18,
    smoothing: float = 1.0,
    show_brain: bool = True,
    analysis: Optional[ComponentAnalysis] = None,
    connectivity: int = 3,
    min_voxels: int = 1,
    max_components: Optional[int] = None,
    spacing: Optional[Tuple[float, float, float]] = None,
    title: str = "3D view",
    width: int = 820,
    height: int = 700,
    clean: bool = True,
    show_legend: bool = False,
    dragmode: Union[str, bool] = "turntable",
) -> go.Figure:
    """
    Build a rotatable 3D figure of the brain surface with the tumor inside it.

    Args:
        image:         3D or 4D array/tensor of MRI intensities.
        mask:          Optional tumor mask; ignored when `analysis` is given.
        modality:      Channel to render when the image is multi-modal.
        level:         Brain isosurface intensity. Defaults to `brain_surface_level`.
        brain_step:    Marching-cubes decimation for the brain (higher = coarser).
        tumor_step:    Decimation for tumor components; keep at 1 for detail.
        brain_opacity: Transparency of the brain shell.
        smoothing:     Gaussian sigma applied to the brain volume only, so the
                       shell is not a mass of interpolation speckle.
        show_brain:    Draw the brain shell at all.
        analysis:      Reuse an existing ComponentAnalysis instead of relabelling.
        min_voxels:    Drop components smaller than this. Worth raising above 1
                       for resized or predicted masks, which scatter one- and
                       two-voxel specks that would each become a mesh here.
        spacing:       Physical voxel size per axis, if the resized grid is not
                       isotropic. Defaults to unit voxels.
        clean:         Strip the scene to the surfaces alone: no tick numbers,
                       gridlines, axis panes or titles, on white. Pass False for
                       the anatomically labelled (L-R, A-P, S-I) axes instead.
        show_legend:   Draw the per-surface legend. Off by default to keep the
                       view uncluttered; the surfaces are still hoverable.
        dragmode:      Scene drag behaviour: "turntable" or "orbit" to rotate,
                       "pan", "zoom", or False to disable dragging.

    Returns a plotly Figure; pass it to `show_3d` to display it interactively.
    """
    volume = _select_modality(image, modality)
    voxel_spacing = (1.0, 1.0, 1.0) if spacing is None else tuple(float(s) for s in spacing)

    if level is None:
        level = brain_surface_level(volume)

    traces: List[go.Mesh3d] = []

    if show_brain:
        # Otsu is measured on the raw intensities; smoothing only tidies the shell.
        surface = gaussian_filter(volume, smoothing) if smoothing > 0 else volume
        brain = _mesh_trace(
            surface,
            level,
            brain_step,
            BRAIN_COLOUR,
            brain_opacity,
            f"brain surface (level {level:.2f})",
            spacing=voxel_spacing,
        )
        if brain is not None:
            traces.append(brain)

    if analysis is None and mask is not None:
        analysis = analyze_components(
            mask, connectivity=connectivity, min_voxels=min_voxels
        )

    components = analysis.components if analysis is not None else []
    if max_components is not None:
        components = components[:max_components]

    for comp in components:
        trace = _mesh_trace(
            (analysis.labels == comp.id).astype(np.float32),
            0.5,
            tumor_step,
            _rgb_string(_component_colour(comp.id)),
            1.0,
            f"#{comp.id}  {comp.n_voxels:,} vox ({comp.volume_fraction:.0%})",
            spacing=voxel_spacing,
        )
        if trace is not None:
            traces.append(trace)

    n_components = analysis.n_components if analysis is not None else 0
    scene = {
        "aspectmode": "data",
        "camera": DEFAULT_CAMERA,
        "dragmode": dragmode,
        **_scene_axes(clean),
    }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"{title}  -  {n_components} component(s)",
        width=width,
        height=height,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        scene=scene,
        showlegend=show_legend,
        legend={"itemsizing": "constant", "yanchor": "top", "y": 0.95},
        meta={"surface_level": level, "modality": modality, "n_components": n_components},
    )

    if clean:
        fig.update_layout(paper_bgcolor="white", scene_bgcolor="white")

    return fig


def show_3d(
    fig: go.Figure,
    renderer: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **config_overrides,
) -> None:
    """
    Display a 3D figure inline and interactively: drag to rotate, wheel to zoom.

    A bare `fig.show()` uses `plotly.io.renderers.default`, which is "browser"
    in this environment, so the scene opens in a separate tab. This picks a
    renderer that draws into the notebook output cell instead, and attaches the
    interaction config that a plain `.show()` leaves at plotly.js defaults.

    Args:
        fig:      Figure from `show_volume_3d`.
        renderer: Force one renderer, e.g. "notebook_connected" or "browser",
                  instead of trying `PREFERRED_RENDERERS` in order.
        config:   Replace the whole Plotly config; `None` uses
                  `INTERACTION_CONFIG`.
        **config_overrides: Individual config keys, e.g. `scrollZoom=False`.
    """
    settings = interaction_config(**config_overrides) if config is None else {
        **config,
        **config_overrides,
    }

    candidates = [renderer] if renderer else list(PREFERRED_RENDERERS)
    last_error: Optional[Exception] = None
    for name in candidates:
        if name not in pio.renderers:
            continue
        try:
            fig.show(renderer=name, config=settings)
            return
        except Exception as error:      # renderer present but unusable here
            last_error = error

    if last_error is not None:
        raise last_error
    raise ValueError(f"no usable renderer among {candidates}")


def save_figure_html(
    fig: go.Figure,
    path: Union[str, Path],
    include_plotlyjs: Union[bool, str] = True,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write the figure to an HTML file that opens in a browser without Jupyter.

    `include_plotlyjs=True` embeds the plotly.js bundle, which costs a few MB
    but makes the file work offline; pass "cdn" for a small file instead. The
    same interaction config as `show_3d` is written into the file, so the
    standalone page rotates, zooms and pans identically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(path),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config=interaction_config() if config is None else config,
    )
    return path
