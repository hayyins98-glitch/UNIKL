"""
Evaluation utilities: performance plots and inference.
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
from brain_tumor_seg.utils.helpers import ensure_dir
from brain_tumor_seg.utils.metrics import dice_coefficient


def plot_training_history(
    history: Dict[str, List[float]],
    save_dir: Optional[Path] = PLOT_DIR,
    show: bool = True,
) -> None:
    """
    Plot training/validation loss and validation Dice over epochs.

    Args:
        history:  Dict returned by Trainer.train() with keys
                  'train_loss', 'val_loss', 'val_dice'.
        save_dir: Directory to save PNG plots (None to skip saving).
        show:     Whether to display plots inline (True for notebooks).
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

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
) -> Dict[str, float]:
    """
    Evaluate the model on a validation DataLoader and return average metrics.

    Returns:
        dict with 'loss' and 'dice' (if masks are available in the loader).
    """
    from brain_tumor_seg.utils.metrics import dice_loss
    import torch.nn as nn

    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_dice = 0.0
    num_batches = 0
    sample = val_loader.dataset[0]
    has_masks = "mask" in sample

    for batch in tqdm(val_loader, desc="Evaluating"):
        images = batch["image"].to(device)

        logits = model(images)

        if "mask" in batch:
            masks = batch["mask"].to(device)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            probs = torch.sigmoid(logits)
            total_dice += dice_coefficient(probs, masks)
            total_loss += loss.item()
        else:
            total_loss += 0.0

        num_batches += 1

    results = {"loss": total_loss / max(num_batches, 1)}
    if has_masks:
        results["dice"] = total_dice / max(num_batches, 1)

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
        model:     Trained UNet3D.
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
    logits = model(image)
    probs = torch.sigmoid(logits)

    mask = (probs > threshold).float()
    return mask.squeeze().cpu().numpy()
