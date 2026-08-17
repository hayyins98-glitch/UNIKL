"""
Loss functions for the survival head.

Survival labels cover well under half the training cases, so the loss is masked
per sample: labeled cases contribute, unlabeled ones contribute exactly zero,
and the segmentation loss still sees the whole batch.
"""
import torch
import torch.nn.functional as F


def masked_survival_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    has_label: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth L1 (Huber) regression loss over the labeled samples of a batch.

    Huber rather than MSE because survival ranges from 10 to over 5000 days;
    even after the log1p transform the tails are heavy enough that a squared
    penalty would let a handful of cases steer the head.

    Args:
        pred:      Predicted normalized survival, shape (batch,) or (batch, 1).
        target:    Ground-truth normalized survival, broadcastable to pred.
        has_label: 1.0 where the sample carries a survival label, else 0.0.
        beta:      Point where Huber switches from squared to linear.

    Returns:
        Scalar loss, averaged over the labeled samples. Exactly 0.0 (finite,
        differentiable) when the batch contains no labeled sample.
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1).to(pred.dtype)
    mask = has_label.reshape(-1).to(pred.dtype)

    per_sample = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")

    # Clamping the denominator instead of branching on it keeps the result
    # attached to the graph, so .backward() works on an all-unlabeled batch
    # (the numerator is zero there, so the gradient is zero rather than NaN).
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)
