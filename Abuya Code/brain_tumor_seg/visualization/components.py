"""
Component-coloured slice views.

Each connected 3D region of the mask gets its own colour, so two blobs that look
separate in one plane can be checked at a glance: same colour means they are the
same 3D object seen through an awkward cross-section.
"""
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact

from brain_tumor_seg.utils.mask_analysis import (
    ComponentAnalysis,
    analyze_components,
    diagnose_slice,
    to_3d,
)

# Visually distinct colours, cycled if there are more components than entries
PALETTE = [
    (1.00, 0.24, 0.24),   # red
    (0.25, 0.60, 1.00),   # blue
    (0.30, 0.90, 0.40),   # green
    (1.00, 0.80, 0.15),   # yellow
    (0.80, 0.40, 1.00),   # purple
    (0.20, 0.92, 0.92),   # cyan
    (1.00, 0.55, 0.15),   # orange
    (1.00, 0.50, 0.78),   # pink
]


def _component_colour(component_id: int) -> tuple:
    return PALETTE[(component_id - 1) % len(PALETTE)]


def _overlay(base_slice: np.ndarray, label_slice: np.ndarray, alpha: float) -> np.ndarray:
    """Grey MRI slice with each component blended in its own colour."""
    span = base_slice.max() - base_slice.min()
    base = (base_slice - base_slice.min()) / (span + 1e-8)
    rgb = np.stack([base, base, base], axis=-1)

    for component_id in np.unique(label_slice):
        if component_id == 0:
            continue
        selection = label_slice == component_id
        colour = np.array(_component_colour(int(component_id)))
        rgb[selection] = (1 - alpha) * rgb[selection] + alpha * colour

    return np.clip(rgb, 0.0, 1.0)


def _default_indices(analysis: ComponentAnalysis) -> tuple:
    """Centre the views on the largest component, falling back to mid-volume."""
    n_lr, n_ap, n_si = analysis.labels.shape
    if not analysis.components:
        return n_si // 2, n_ap // 2, n_lr // 2

    lr, ap, si = (int(round(c)) for c in analysis.components[0].centroid)
    return si, ap, lr


def plot_component_slices(
    image,
    mask,
    axial_idx: Optional[int] = None,
    coronal_idx: Optional[int] = None,
    sagittal_idx: Optional[int] = None,
    analysis: Optional[ComponentAnalysis] = None,
    connectivity: int = 3,
    min_voxels: int = 1,
    alpha: float = 0.55,
    title: str = "Connected components",
    figsize: tuple = (15, 5.5),
) -> ComponentAnalysis:
    """
    Plot axial, coronal and sagittal slices with components colour-coded.

    Each panel title reports how many separate blobs are visible in that slice
    and which 3D components they belong to, so a "split" view is immediately
    explained. Indices default to the centre of the largest component.

    Returns the ComponentAnalysis so it can be reused without relabelling.
    """
    volume = to_3d(image)
    if analysis is None:
        analysis = analyze_components(mask, connectivity=connectivity, min_voxels=min_voxels)

    if volume.shape != analysis.labels.shape:
        raise ValueError(
            f"image shape {volume.shape} does not match mask shape {analysis.labels.shape}"
        )

    default_axial, default_coronal, default_sagittal = _default_indices(analysis)
    axial_idx = default_axial if axial_idx is None else axial_idx
    coronal_idx = default_coronal if coronal_idx is None else coronal_idx
    sagittal_idx = default_sagittal if sagittal_idx is None else sagittal_idx

    n_lr, n_ap, n_si = analysis.labels.shape
    axial_idx = int(np.clip(axial_idx, 0, n_si - 1))
    coronal_idx = int(np.clip(coronal_idx, 0, n_ap - 1))
    sagittal_idx = int(np.clip(sagittal_idx, 0, n_lr - 1))

    panels = [
        ("axial", axial_idx, volume[:, :, axial_idx], analysis.labels[:, :, axial_idx]),
        ("coronal", coronal_idx, volume[:, coronal_idx, :], analysis.labels[:, coronal_idx, :]),
        ("sagittal", sagittal_idx, volume[sagittal_idx, :, :], analysis.labels[sagittal_idx, :, :]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, (plane, index, img_slice, label_slice) in zip(axes, panels):
        ax.imshow(_overlay(img_slice, label_slice, alpha))
        ax.axis("off")

        info = diagnose_slice(analysis, plane, index)
        if info["n_blobs_2d"] == 0:
            caption = "no mask in this slice"
        else:
            ids = ", ".join(f"#{i}" for i in info["component_ids"])
            caption = f"{info['n_blobs_2d']} blob(s) from component(s) {ids}"
            if info["split_components"]:
                caption += "\nsame 3D mass, split by this cut"
        ax.set_title(f"{plane.capitalize()}  (slice {index})\n{caption}", fontsize=10)

    if analysis.components:
        handles = [
            mpatches.Patch(
                color=_component_colour(comp.id),
                label=f"#{comp.id}  {comp.n_voxels:,} vox ({comp.volume_fraction:.0%})",
            )
            for comp in analysis.components[: len(PALETTE)]
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle(f"{title}  -  {analysis.n_components} component(s)", fontsize=13, y=0.99)
    plt.tight_layout(rect=(0, 0.08, 1, 0.94))
    plt.show()

    return analysis


def show_components_with_slider(
    image,
    mask,
    connectivity: int = 3,
    min_voxels: int = 1,
    title: str = "Connected components",
) -> ComponentAnalysis:
    """
    Interactive component viewer with one slider per anatomical plane.

    Labelling runs once up front, so dragging the sliders only redraws.
    """
    volume = to_3d(image)
    analysis = analyze_components(mask, connectivity=connectivity, min_voxels=min_voxels)
    n_lr, n_ap, n_si = analysis.labels.shape
    default_axial, default_coronal, default_sagittal = _default_indices(analysis)

    def update(axial_idx: int, coronal_idx: int, sagittal_idx: int):
        plt.close("all")
        plot_component_slices(
            volume,
            mask=None,
            axial_idx=axial_idx,
            coronal_idx=coronal_idx,
            sagittal_idx=sagittal_idx,
            analysis=analysis,
            title=title,
        )

    interact(
        update,
        axial_idx=IntSlider(
            min=0, max=n_si - 1, step=1, value=default_axial,
            description="Axial", continuous_update=False,
        ),
        coronal_idx=IntSlider(
            min=0, max=n_ap - 1, step=1, value=default_coronal,
            description="Coronal", continuous_update=False,
        ),
        sagittal_idx=IntSlider(
            min=0, max=n_lr - 1, step=1, value=default_sagittal,
            description="Sagittal", continuous_update=False,
        ),
    )

    return analysis
