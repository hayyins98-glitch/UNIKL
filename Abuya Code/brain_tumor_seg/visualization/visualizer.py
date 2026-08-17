"""
3D volume visualizer with coronal, sagittal, and axial views.

Use the interactive slider in Jupyter to browse slice-by-slice.
"""
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact


class VolumeVisualizer:
    """
    Display a 3D MRI volume and optional mask in three anatomical views.

    Volume axes follow BraTS NIfTI ordering: (L-R, A-P, S-I).

    Views:
        - Axial    (horizontal, top-down)  — slice along S-I (axis 2)
        - Coronal  (vertical, front)       — slice along A-P (axis 1)
        - Sagittal (vertical, side)        — slice along L-R (axis 0)
    """

    def __init__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        title: str = "MRI Volume",
    ):
        """
        Args:
            image: 3D array (D, H, W) or 4D (C, D, H, W).
                   If 4D, the first channel (t1c) is shown by default.
            mask:  Optional 3D binary mask (D, H, W) or (1, D, H, W).
            title: Title shown above the figure.
        """
        self.image = self._to_3d(image)
        self.mask = self._to_3d(mask) if mask is not None else None
        self.title = title

        # BraTS volumes: axis 0 = L-R, axis 1 = A-P, axis 2 = S-I
        self.n_lr, self.n_ap, self.n_si = self.image.shape

    @staticmethod
    def _to_3d(array: np.ndarray) -> np.ndarray:
        """Convert (C, D, H, W) or (1, D, H, W) to (D, H, W)."""
        if array.ndim == 4:
            return array[0]  # show first channel
        if array.ndim == 3:
            return array
        raise ValueError(f"Expected 3D or 4D array, got shape {array.shape}")

    def _overlay_mask(self, base_slice: np.ndarray, mask_slice: np.ndarray) -> np.ndarray:
        """Return an RGB image with the mask overlaid in red."""
        base = (base_slice - base_slice.min()) / (base_slice.max() - base_slice.min() + 1e-8)
        rgb = np.stack([base, base, base], axis=-1)

        if mask_slice is not None and mask_slice.max() > 0:
            rgb[mask_slice > 0.5, 0] = 1.0   # red channel
            rgb[mask_slice > 0.5, 1] = 0.2
            rgb[mask_slice > 0.5, 2] = 0.2

        return rgb

    def plot_slices(
        self,
        axial_idx: int,
        coronal_idx: int,
        sagittal_idx: int,
        figsize: tuple = (14, 5),
    ) -> None:
        """
        Plot all three views at the given slice indices (static plot).
        """
        axial_idx = np.clip(axial_idx, 0, self.n_si - 1)
        coronal_idx = np.clip(coronal_idx, 0, self.n_ap - 1)
        sagittal_idx = np.clip(sagittal_idx, 0, self.n_lr - 1)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        views = [
            ("Axial",    self.image[:, :, axial_idx],
             self.mask[:, :, axial_idx] if self.mask is not None else None),
            ("Coronal",  self.image[:, coronal_idx, :],
             self.mask[:, coronal_idx, :] if self.mask is not None else None),
            ("Sagittal", self.image[sagittal_idx, :, :],
             self.mask[sagittal_idx, :, :] if self.mask is not None else None),
        ]

        for ax, (name, img_slice, mask_slice) in zip(axes, views):
            display = self._overlay_mask(img_slice, mask_slice)
            ax.imshow(display)
            ax.set_title(f"{name}  (slice {axial_idx if name == 'Axial' else coronal_idx if name == 'Coronal' else sagittal_idx})")
            ax.axis("off")

        fig.suptitle(self.title, fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_interactive(self) -> None:
        """
        Launch interactive sliders in Jupyter to browse each anatomical plane.
        """
        def update(axial_idx: int, coronal_idx: int, sagittal_idx: int):
            plt.close("all")
            self.plot_slices(axial_idx, coronal_idx, sagittal_idx)

        interact(
            update,
            axial_idx=IntSlider(
                min=0,
                max=self.n_si - 1,
                step=1,
                value=self.n_si // 2,
                description="Axial",
                continuous_update=False,
            ),
            coronal_idx=IntSlider(
                min=0,
                max=self.n_ap - 1,
                step=1,
                value=self.n_ap // 2,
                description="Coronal",
                continuous_update=False,
            ),
            sagittal_idx=IntSlider(
                min=0,
                max=self.n_lr - 1,
                step=1,
                value=self.n_lr // 2,
                description="Sagittal",
                continuous_update=False,
            ),
        )


def show_volume_with_slider(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "MRI Volume",
) -> None:
    """
    Convenience function: create a VolumeVisualizer and show the interactive slider.

    Args:
        image: 3D or 4D numpy array (use .numpy() if coming from PyTorch).
        mask:  Optional mask array.
        title: Figure title.
    """
    if hasattr(image, "numpy"):
        image = image.numpy()
    if mask is not None and hasattr(mask, "numpy"):
        mask = mask.numpy()

    visualizer = VolumeVisualizer(image, mask, title)
    visualizer.show_interactive()
