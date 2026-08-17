"""
Evaluation metrics for segmentation.

Dice coefficient measures overlap between prediction and ground truth.
Values range from 0 (no overlap) to 1 (perfect match).
"""
import torch


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
