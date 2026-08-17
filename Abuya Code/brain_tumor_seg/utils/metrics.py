"""
Evaluation metrics for both heads.

Segmentation: the Dice coefficient measures overlap between prediction and
ground truth, from 0 (no overlap) to 1 (perfect match).

Survival: the model works on a normalized log1p scale, which is unreadable, so
errors are converted back into days before they are reported.
"""
from typing import Dict, List

import numpy as np
import torch

from brain_tumor_seg.data.survival import SurvivalStats


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute the Dice coefficient between prediction and target.

    Args:
        pred:   Predicted mask (binary or probabilities), any shape.
        target: Ground-truth mask (binary), same shape as pred.
        smooth: Small constant to avoid division by zero.

    Returns:
        Dice score as a Python float.
    """
    pred = (pred > 0.5).float()
    target = target.float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Differentiable Dice loss (1 - Dice coefficient).

    Used during training alongside BCE loss for better segmentation.
    """
    pred = torch.sigmoid(pred) if pred.max() > 1 or pred.min() < 0 else pred

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice


def survival_absolute_errors_days(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
    stats: SurvivalStats,
) -> List[float]:
    """
    Per-sample absolute survival error in days, for the labeled samples only.

    Returned per sample rather than aggregated so a caller can pool a whole
    epoch and take an exact median across it.

    Args:
        pred:      Predicted normalized survival, shape (batch,) or (batch, 1).
        target:    Ground-truth normalized survival, same layout as pred.
        has_label: 1.0 where the sample carries a survival label, else 0.0.
        stats:     Normalization used to build the targets.

    Returns:
        List of |predicted - true| in days; empty if nothing in the batch
        was labeled.
    """
    mask = has_label.detach().reshape(-1) > 0.5
    if not bool(mask.any()):
        return []

    pred_days = stats.to_days(pred.detach().reshape(-1)[mask].cpu().numpy())
    true_days = stats.to_days(target.detach().reshape(-1)[mask].cpu().numpy())

    return [float(value) for value in np.abs(pred_days - true_days)]


def summarize_survival_errors(errors: List[float]) -> Dict[str, float]:
    """
    Aggregate absolute day errors into reportable numbers.

    Median is reported next to the mean because survival is right-skewed: a few
    very long survivors can move the MAE a long way on their own.

    Args:
        errors: Absolute errors in days, typically pooled over an epoch.

    Returns:
        dict with 'mae_days', 'median_ae_days' and 'count'. The two error
        figures are NaN when count is 0, so an unlabeled epoch leaves a gap in
        a plot instead of a misleading zero.
    """
    if not errors:
        return {"mae_days": float("nan"), "median_ae_days": float("nan"), "count": 0}

    array = np.asarray(errors, dtype=np.float64)
    return {
        "mae_days": float(array.mean()),
        "median_ae_days": float(np.median(array)),
        "count": int(array.size),
    }
