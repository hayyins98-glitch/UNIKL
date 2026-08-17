"""
Two-headed model: tumor segmentation plus patient survival regression.

MultiTaskUNet3D composes an unmodified UNet3D with a SurvivalHead. The
survival head does **not** look at the whole brain: it pools encoder features
only inside the tumor and also sees tumor volume. That is the signal the
clinical question asks for — survival given this mass — rather than a
whole-volume embedding that could latch onto unrelated anatomy.

During training the ground-truth mask is used, so the head learns from the
actual tumor. At inference the predicted mask is used instead. The mask is
always detached, so the survival loss cannot warp the segmentation.

The backbone still lives under the 'segmentation.' prefix, so an existing
best_model.pth can warm-start that half (see load_segmentation_weights).

Input:  (batch, 4, D, H, W)
Output: MultiTaskOutput(seg_logits=(batch, 1, D, H, W), survival=(batch,))

The survival value is on the normalized (z-scored log1p days) scale — convert
it with SurvivalStats.to_days() before reporting.
"""
import inspect
from pathlib import Path
from typing import NamedTuple, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_tumor_seg.models.unet3d import UNet3D

StateDict = Union[str, Path, dict]


class MultiTaskOutput(NamedTuple):
    """Both heads' predictions for one batch."""

    seg_logits: torch.Tensor    # (batch, out_channels, D, H, W)
    survival: torch.Tensor      # (batch,), normalized scale


def split_model_outputs(outputs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Normalize whatever a model returned into (seg_logits, survival).

    Lets the trainer and the evaluator accept a plain UNet3D and a
    MultiTaskUNet3D through the same code path without a class check.

    Args:
        outputs: A logits tensor, or anything exposing .seg_logits/.survival.

    Returns:
        (segmentation logits, survival predictions or None for a single head).
    """
    survival = getattr(outputs, "survival", None)
    seg_logits = getattr(outputs, "seg_logits", outputs)
    return seg_logits, survival


def run_model(
    model: nn.Module,
    images: torch.Tensor,
    tumor_mask: Optional[torch.Tensor] = None,
):
    """
    Forward a batch, passing the tumor mask when the model knows how to use it.

    A plain UNet3D ignores the mask; MultiTaskUNet3D pools survival features
    inside it.
    """
    if tumor_mask is not None and "tumor_mask" in inspect.signature(model.forward).parameters:
        return model(images, tumor_mask=tumor_mask)
    return model(images)


def _resize_mask(mask: torch.Tensor, spatial_size: Tuple[int, ...]) -> torch.Tensor:
    """Match a (batch, 1, ...) mask to a feature map's depth/height/width."""
    if tuple(mask.shape[2:]) == tuple(spatial_size):
        return mask
    return F.interpolate(mask, size=spatial_size, mode="trilinear", align_corners=False)


def masked_average_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Average the feature map only where the tumor mask is on.

    Args:
        features: (batch, channels, D, H, W).
        mask:     (batch, 1, d, h, w) — resized if the spatial size differs.

    Returns:
        (batch, channels). Empty masks contribute a zero vector rather than NaN.
    """
    weights = _resize_mask(mask, features.shape[2:]).clamp(min=0.0)
    denom = weights.sum(dim=(2, 3, 4), keepdim=True)
    empty = denom < 1e-6
    safe_denom = denom.clamp(min=1e-6)
    pooled = (features * weights).sum(dim=(2, 3, 4)) / safe_denom.view(features.size(0), 1)
    return pooled.masked_fill(empty.view(features.size(0), 1), 0.0)


class SurvivalHead(nn.Module):
    """
    Per-patient survival regressor from tumor-local features.

    Masked average pooling keeps only the voxels the tumor occupies, and the
    log tumor volume is concatenated so size is an explicit input rather than
    something the MLP has to recover from the pooled vector.
    """

    def __init__(self, in_channels: int, hidden_features: int = 64, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 1, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )

    def forward(self, features: torch.Tensor, tumor_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features:   Feature map of shape (batch, in_channels, D, H, W).
            tumor_mask: Tumor mask of shape (batch, 1, d, h, w), values in [0, 1].

        Returns:
            Normalized survival prediction of shape (batch,).
        """
        pooled = masked_average_pool(features, tumor_mask)
        # Full-resolution voxel count, not the downsampled bottleneck count, so
        # the volume feature matches the mask the caller actually passed.
        voxel_count = tumor_mask.reshape(tumor_mask.size(0), -1).sum(dim=1, keepdim=True)
        volume = torch.log1p(voxel_count)
        return self.mlp(torch.cat([pooled, volume], dim=1)).squeeze(-1)


class MultiTaskUNet3D(nn.Module):
    """
    UNet3D plus a survival head that reads the tumor, not the whole volume.

    The backbone lives under the 'segmentation.' prefix, so its parameters map
    one-to-one onto a standalone UNet3D state_dict.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_features: int = 16,
        survival_hidden: int = 64,
        survival_dropout: float = 0.3,
        segmentation: Optional[UNet3D] = None,
    ):
        super().__init__()
        self.segmentation = segmentation or UNet3D(in_channels, out_channels, base_features)
        self.survival_head = SurvivalHead(
            self.segmentation.bottleneck_channels,
            hidden_features=survival_hidden,
            dropout=survival_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        tumor_mask: Optional[torch.Tensor] = None,
    ) -> MultiTaskOutput:
        """
        Args:
            x:          MRI volume (batch, 4, D, H, W).
            tumor_mask: Optional (batch, 1, D, H, W) tumor. Pass the
                        ground-truth mask during training. Omit it at
                        inference and the predicted tumor is used instead.
        """
        seg_logits, bottleneck = self.segmentation.forward_features(x)
        if tumor_mask is None:
            tumor_mask = torch.sigmoid(seg_logits)
        # Detach: survival should *read* the tumor, not reshape it to fit the
        # survival target.
        survival = self.survival_head(bottleneck, tumor_mask.detach())
        return MultiTaskOutput(seg_logits=seg_logits, survival=survival)

    def load_segmentation_weights(
        self,
        source: StateDict,
        strict: bool = True,
        map_location: str = "cpu",
    ) -> None:
        """
        Warm-start the segmentation half from a single-head checkpoint.

        Args:
            source:       Path to a .pth file, or an already-loaded state_dict.
                          Keys may be bare UNet3D keys or carry the
                          'segmentation.' prefix of a multi-task checkpoint.
            strict:       Require an exact key match on the backbone.
            map_location: Device to load a checkpoint file onto.

        Raises:
            RuntimeError: If strict is True and the keys do not line up, which
                          usually means a different base_features was used.
        """
        if isinstance(source, (str, Path)):
            state_dict = torch.load(str(source), map_location=map_location)
        else:
            state_dict = source

        # Accept checkpoints saved from either model without the caller caring.
        prefix = "segmentation."
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

        self.segmentation.load_state_dict(state_dict, strict=strict)
