"""
Evaluation utilities: performance plots and inference.

Everything here accepts both a segmentation-only UNet3D and the two-headed
MultiTaskUNet3D; survival numbers are reported only when the model has the head
and the data carries labels.
"""
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_tumor_seg.config import PLOT_DIR
from brain_tumor_seg.data.survival import SurvivalStats
from brain_tumor_seg.models.multitask import run_model, split_model_outputs
from brain_tumor_seg.utils.helpers import ensure_dir
from brain_tumor_seg.utils.metrics import (
    dice_coefficient,
    summarize_survival_errors,
    survival_absolute_errors_days,
)


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: Optional[Path] = PLOT_DIR,
    show: bool = True,
) -> None:
    """
    Plot training/validation loss and validation Dice over epochs.

    Two extra panels are added when the history came from a dual-head run;
    a segmentation-only history plots exactly as before.

    Args:
        history:  Dict returned by Trainer.train() with keys
                  'train_loss', 'val_loss', 'val_dice' and optionally
                  'train_surv_loss', 'val_surv_loss', 'val_surv_mae_days'.
        save_dir: Directory to save PNG plots (None to skip saving).
        show:     Whether to display plots inline (True for notebooks).
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    has_survival = "val_surv_mae_days" in history

    n_panels = 4 if has_survival else 2
    fig, axes = plt.subplots(
        n_panels // 2, 2, figsize=(12, 4 * (n_panels // 2)), squeeze=False
    )
    axes = axes.ravel()

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Dice plot
    axes[1].plot(epochs, history["val_dice"], "g-o", label="Val Dice", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice Score")
    axes[1].set_title("Validation Dice Score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    if has_survival:
        axes[2].plot(
            epochs, history["train_surv_loss"], "b-o", label="Train Survival", markersize=4
        )
        axes[2].plot(
            epochs, history["val_surv_loss"], "r-o", label="Val Survival", markersize=4
        )
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Huber Loss (normalized)")
        axes[2].set_title("Survival Loss")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plotted in days rather than on the normalized scale, since that is the
        # only version of this number anyone can judge.
        axes[3].plot(
            epochs, history["val_surv_mae_days"], "m-o", label="Val MAE", markersize=4
        )
        axes[3].set_xlabel("Epoch")
        axes[3].set_ylabel("Mean Absolute Error (days)")
        axes[3].set_title("Validation Survival Error")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        ensure_dir(save_dir)
        fig.savefig(save_dir / "training_history.png", dpi=150)
        print(f"Plot saved to {save_dir / 'training_history.png'}")

    if show:
        plt.show()
    else:
        plt.close(fig)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    survival_stats: Optional[SurvivalStats] = None,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation DataLoader and return average metrics.

    Args:
        model:          UNet3D or MultiTaskUNet3D.
        val_loader:     Loader over cases to score.
        device:         cpu or cuda.
        survival_stats: Normalization used to build the survival targets.
                        Required to report the survival error in days.

    Returns:
        dict with 'loss' and 'dice' (if masks are available in the loader),
        plus 'surv_mae_days', 'surv_median_ae_days' and 'surv_count' when the
        model has a survival head and the loader carries survival labels.
    """
    from brain_tumor_seg.utils.metrics import dice_loss
    import torch.nn as nn

    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    surv_errors: List[float] = []
    sample = val_loader.dataset[0]
    has_masks = "mask" in sample

    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device) if "mask" in batch else None

        logits, survival_pred = split_model_outputs(
            run_model(model, images, tumor_mask=masks)
        )

        if masks is not None:
            loss = bce(logits, masks) + dice_loss(logits, masks)
            probs = torch.sigmoid(logits)
            total_dice += dice_coefficient(probs, masks)
            total_loss += loss.item()
        else:
            total_loss += 0.0

        if survival_pred is not None and "survival" in batch and survival_stats is not None:
            surv_errors.extend(
                survival_absolute_errors_days(
                    survival_pred, batch["survival"], batch["has_survival"], survival_stats
                )
            )

        num_batches += 1

    results = {"loss": total_loss / max(num_batches, 1)}
    if has_masks:
        results["dice"] = total_dice / max(num_batches, 1)

    if surv_errors:
        summary = summarize_survival_errors(surv_errors)
        results["surv_mae_days"] = summary["mae_days"]
        results["surv_median_ae_days"] = summary["median_ae_days"]
        results["surv_count"] = float(summary["count"])

    return results


@torch.no_grad()
def predict_volume(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference on a single volume.

    Args:
        model:     Trained UNet3D or MultiTaskUNet3D.
        image:     Tensor of shape (4, D, H, W) or (1, 4, D, H, W).
        device:    cpu or cuda.
        threshold: Probability threshold for binary mask.

    Returns:
        Binary mask as numpy array (D, H, W).
    """
    model.eval()

    if image.dim() == 4:
        image = image.unsqueeze(0)

    image = image.to(device)
    logits, _ = split_model_outputs(model(image))
    probs = torch.sigmoid(logits)

    mask = (probs > threshold).float()
    return mask.squeeze().cpu().numpy()


@torch.no_grad()
def predict_survival_days(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    survival_stats: SurvivalStats,
    tumor_mask: Optional[torch.Tensor] = None,
) -> float:
    """
    Predict overall survival for a single case, in days.

    Survival is read from the tumor. Pass a ground-truth mask when you have
    one; otherwise the model's predicted tumor is used. Works on official
    Validation cases that have no survival label to compare against.

    Args:
        model:          MultiTaskUNet3D (a segmentation-only model has nothing
                        to predict here).
        image:          Tensor of shape (4, D, H, W) or (1, 4, D, H, W).
        device:         cpu or cuda.
        survival_stats: The normalization the model was trained against.
        tumor_mask:     Optional (1, D, H, W) or (D, H, W) tumor. None means
                        use the predicted tumor from this forward pass.

    Returns:
        Predicted survival in days.

    Raises:
        AttributeError: If the model has no survival head.
    """
    model.eval()

    if image.dim() == 4:
        image = image.unsqueeze(0)
    image = image.to(device)

    mask = None
    if tumor_mask is not None:
        mask = tumor_mask.to(device)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 4:
            mask = mask.unsqueeze(0)

    _, survival = split_model_outputs(run_model(model, image, tumor_mask=mask))
    if survival is None:
        raise AttributeError(
            f"{type(model).__name__} has no survival head; use MultiTaskUNet3D."
        )

    return float(survival_stats.to_days(survival.reshape(-1)[0].cpu().numpy()))
