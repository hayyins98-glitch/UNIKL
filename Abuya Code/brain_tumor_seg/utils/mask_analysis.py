"""
3D connected-component analysis for segmentation masks.

Answers "is this one tumor or several?" directly from the volume. This matters
because a single connected 3D shape can appear as several disjoint blobs in any
one 2D plane (think of slicing across both tips of a horseshoe), so counting
blobs in a slice tells you nothing about 3D connectivity.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    center_of_mass,
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
)

# scipy connectivity ranks for a 3D structuring element
CONNECTIVITY_NAMES = {
    1: "6-neighbour (faces only, strictest)",
    2: "18-neighbour (faces + edges)",
    3: "26-neighbour (faces + edges + corners, most permissive)",
}

# Anatomical plane -> volume axis, matching the (L-R, A-P, S-I) ordering
# used by BraTS NIfTI volumes and by VolumeVisualizer.
PLANE_AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


@dataclass
class Component:
    """One connected 3D region of the mask."""

    id: int
    n_voxels: int
    volume_fraction: float          # share of all tumor voxels
    centroid: Tuple[float, float, float]
    bbox: Tuple[Tuple[int, int], ...]   # ((lr0, lr1), (ap0, ap1), (si0, si1))

    @property
    def extent(self) -> Tuple[int, int, int]:
        """Bounding-box size in voxels along each axis."""
        return tuple(hi - lo for lo, hi in self.bbox)


@dataclass(repr=False)
class ComponentAnalysis:
    """Result of labelling a binary mask into connected 3D components."""

    labels: np.ndarray              # int array, 0 = background
    connectivity: int
    components: List[Component]     # sorted largest first
    total_voxels: int               # tumor voxels kept after min_voxels filter
    n_dropped: int                  # components removed by min_voxels
    n_dropped_voxels: int

    @property
    def n_components(self) -> int:
        return len(self.components)

    def __repr__(self) -> str:
        # A generated repr would dump the whole label volume, which floods the
        # notebook whenever this object is the value of the last cell line.
        return (
            f"{type(self).__name__}(shape={self.labels.shape}, "
            f"connectivity={self.connectivity}, "
            f"n_components={self.n_components}, "
            f"total_voxels={self.total_voxels})"
        )


def to_3d(array) -> np.ndarray:
    """Squeeze a torch tensor or (C, D, H, W) array down to a 3D numpy array."""
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    elif hasattr(array, "numpy"):
        array = array.numpy()

    array = np.asarray(array)

    # Drop leading singleton axes, e.g. (1, D, H, W) or (1, 1, D, H, W)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]

    if array.ndim == 4:
        array = array[0]   # multi-channel: keep the first channel

    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {array.shape}")

    return array


def analyze_components(
    mask,
    connectivity: int = 3,
    min_voxels: int = 1,
    threshold: float = 0.5,
) -> ComponentAnalysis:
    """
    Label a binary mask into connected 3D components.

    Args:
        mask:         3D/4D array or tensor. Values above `threshold` are tumor.
        connectivity: 1 = 6-neighbour, 2 = 18-neighbour, 3 = 26-neighbour.
        min_voxels:   Components smaller than this are discarded as noise.
        threshold:    Binarisation cutoff for probability masks.
    """
    if connectivity not in CONNECTIVITY_NAMES:
        raise ValueError(f"connectivity must be 1, 2 or 3, got {connectivity}")

    binary = to_3d(mask) > threshold
    structure = generate_binary_structure(3, connectivity)
    labels, n_found = label(binary, structure=structure)

    # bincount index i == label i, so entry 0 is the background
    sizes = np.bincount(labels.ravel(), minlength=n_found + 1)
    boxes = find_objects(labels)

    keep_ids = [i for i in range(1, n_found + 1) if sizes[i] >= min_voxels]
    dropped_ids = [i for i in range(1, n_found + 1) if sizes[i] < min_voxels]
    kept_total = int(sizes[keep_ids].sum()) if keep_ids else 0

    centroids = (
        center_of_mass(binary, labels, keep_ids) if keep_ids else []
    )

    components = [
        Component(
            id=int(comp_id),
            n_voxels=int(sizes[comp_id]),
            volume_fraction=float(sizes[comp_id]) / kept_total if kept_total else 0.0,
            centroid=tuple(round(float(c), 1) for c in centroid),
            bbox=tuple((int(s.start), int(s.stop)) for s in boxes[comp_id - 1]),
        )
        for comp_id, centroid in zip(keep_ids, centroids)
    ]
    components.sort(key=lambda c: c.n_voxels, reverse=True)

    if dropped_ids:
        labels = labels.copy()
        labels[np.isin(labels, dropped_ids)] = 0

    return ComponentAnalysis(
        labels=labels,
        connectivity=connectivity,
        components=components,
        total_voxels=kept_total,
        n_dropped=len(dropped_ids),
        n_dropped_voxels=int(sizes[dropped_ids].sum()) if dropped_ids else 0,
    )


def pairwise_gaps(
    analysis: ComponentAnalysis,
    max_components: int = 8,
) -> Dict[Tuple[int, int], float]:
    """
    Width of the empty space separating each pair of components, in voxels.

    A gap of only 1-2 voxels usually means a thin connecting bridge was lost to
    downsampling rather than the lesion being genuinely multifocal.
    """
    comps = analysis.components[:max_components]
    if len(comps) < 2:
        return {}

    gaps = {}
    for index, comp_a in enumerate(comps):
        # EDT measures distance to the nearest voxel of comp_a, so the closest
        # voxel of another component sits one step beyond the empty space.
        edt = distance_transform_edt(analysis.labels != comp_a.id)
        for comp_b in comps[index + 1:]:
            centre_distance = float(edt[analysis.labels == comp_b.id].min())
            gaps[(comp_a.id, comp_b.id)] = max(centre_distance - 1.0, 0.0)

    return gaps


def diagnose_slice(
    analysis: ComponentAnalysis,
    plane: str,
    index: int,
    connectivity_2d: int = 2,
) -> dict:
    """
    Explain what a single 2D slice shows versus the underlying 3D structure.

    This is the direct test for "the mask looks split in this view": it counts
    the blobs visible in the slice and reports which 3D component each one
    belongs to. Several blobs mapping to one component means the tumor is
    continuous and the slice simply cuts across a curved or lobed shape.
    """
    if plane not in PLANE_AXES:
        raise ValueError(f"plane must be one of {sorted(PLANE_AXES)}, got {plane!r}")

    axis = PLANE_AXES[plane]
    n_slices = analysis.labels.shape[axis]
    if not 0 <= index < n_slices:
        raise IndexError(f"{plane} index {index} out of range (0-{n_slices - 1})")

    slice_labels = np.take(analysis.labels, index, axis=axis)

    blobs_2d, n_blobs = label(
        slice_labels > 0, structure=generate_binary_structure(2, connectivity_2d)
    )

    # Map each 2D blob to the 3D component it came from
    blob_to_component = {}
    for blob_id in range(1, n_blobs + 1):
        ids_3d = np.unique(slice_labels[blobs_2d == blob_id])
        blob_to_component[blob_id] = int(ids_3d[ids_3d > 0][0])

    component_ids = sorted(set(blob_to_component.values()))
    split_components = sorted(
        comp_id
        for comp_id in component_ids
        if list(blob_to_component.values()).count(comp_id) > 1
    )

    return {
        "plane": plane,
        "index": index,
        "n_blobs_2d": n_blobs,
        "component_ids": component_ids,
        "blob_to_component": blob_to_component,
        "split_components": split_components,
    }


def format_report(
    analysis: ComponentAnalysis,
    voxel_volume_mm3: Optional[float] = None,
    max_listed: int = 10,
) -> str:
    """Build a human-readable summary of a ComponentAnalysis."""
    lines = [
        "3D connected-component analysis",
        f"  connectivity : {CONNECTIVITY_NAMES[analysis.connectivity]}",
        f"  volume shape : {analysis.labels.shape} (L-R, A-P, S-I)",
        f"  tumor voxels : {analysis.total_voxels:,}",
        f"  components   : {analysis.n_components}",
    ]

    if analysis.n_dropped:
        lines.append(
            f"  discarded    : {analysis.n_dropped} component(s) below the size "
            f"filter ({analysis.n_dropped_voxels:,} voxels)"
        )

    if not analysis.components:
        lines.append("  mask is empty - nothing to analyse")
        return "\n".join(lines)

    lines.append("")
    for comp in analysis.components[:max_listed]:
        size = f"{comp.n_voxels:,} voxels"
        if voxel_volume_mm3:
            size += f" ({comp.n_voxels * voxel_volume_mm3 / 1000:.2f} cm3)"
        lines.append(
            f"  #{comp.id}: {size}, {comp.volume_fraction:.1%} of tumor, "
            f"extent {comp.extent} at centroid {comp.centroid}"
        )

    if analysis.n_components > max_listed:
        lines.append(f"  ... and {analysis.n_components - max_listed} more")

    lines.append("")
    if analysis.n_components == 1:
        lines.append(
            "  Verdict: one single connected mass. Any view where the mask looks "
            "split is a cross-section through a curved or lobed shape, not a gap "
            "in the segmentation."
        )
    else:
        gaps = pairwise_gaps(analysis)
        if gaps:
            lines.append("  Empty gap between components (voxels):")
            for (id_a, id_b), gap in sorted(gaps.items(), key=lambda kv: kv[1]):
                lines.append(f"    #{id_a} <-> #{id_b}: {gap:.1f}")

            closest = min(gaps.values())
            if closest <= 2.0:
                lines.append(
                    "  Verdict: components are nearly touching, which points to a "
                    "thin bridge lost during downsampling rather than separate "
                    "lesions. Re-check at native resolution."
                )
            else:
                lines.append(
                    "  Verdict: components are well separated, consistent with a "
                    "genuinely multifocal lesion (or, for predictions, spurious "
                    "islands worth filtering)."
                )

    return "\n".join(lines)


def print_component_report(
    mask,
    connectivity: int = 3,
    min_voxels: int = 1,
    voxel_volume_mm3: Optional[float] = None,
) -> ComponentAnalysis:
    """
    Analyse a mask, print the report, and return the analysis.

    Also compares connectivity settings, since a component count that changes
    between strict and permissive neighbourhoods means the pieces touch only at
    edges or corners.
    """
    analysis = analyze_components(mask, connectivity=connectivity, min_voxels=min_voxels)
    print(format_report(analysis, voxel_volume_mm3=voxel_volume_mm3))

    counts = {
        rank: analyze_components(mask, connectivity=rank, min_voxels=min_voxels).n_components
        for rank in (1, 2, 3)
    }
    if len(set(counts.values())) > 1:
        print(
            "\n  Component count by connectivity: "
            + ", ".join(f"{rank}-> {n}" for rank, n in counts.items())
            + "\n  (the count changes with connectivity, so some pieces meet only "
            "at edges or corners)"
        )

    return analysis
