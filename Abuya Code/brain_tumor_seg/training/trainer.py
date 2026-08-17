"""
Training loop for 3D brain tumor segmentation.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_tumor_seg.config import CHECKPOINT_DIR, LEARNING_RATE
from brain_tumor_seg.utils.helpers import ensure_dir, format_duration
from brain_tumor_seg.utils.metrics import dice_coefficient, dice_loss


class Trainer:
    """
    Handles one full training run: train epochs, validate, save checkpoints.

    Usage:
        trainer = Trainer(model, train_loader, val_loader, device)
        history = trainer.train(num_epochs=20)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = LEARNING_RATE,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = ensure_dir(checkpoint_dir)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bce_loss = nn.BCEWithLogitsLoss()

        # History stored here and returned after training
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "epoch_time": [],
        }

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined BCE + Dice loss."""
        bce = self.bce_loss(logits, targets)
        dice = dice_loss(logits, targets)
        return bce + dice

    def _run_epoch(
        self,
        loader: DataLoader,
        training: bool,
        desc: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run one epoch and return average loss (and dice if validating)."""
        self.model.train() if training else self.model.eval()

        total_loss = 0.0
        total_dice = 0.0
        num_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        if desc is None:
            desc = "Train" if training else "Val"

        with context:
            for batch in tqdm(loader, desc=desc, leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                logits = self.model(images)
                loss = self._compute_loss(logits, masks)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()

                if not training:
                    probs = torch.sigmoid(logits)
                    total_dice += dice_coefficient(probs, masks)

                num_batches += 1

        results = {"loss": total_loss / num_batches}
        if not training:
            results["dice"] = total_dice / num_batches

        return results

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Run the full training loop for num_epochs.

        Returns:
            history dict with train_loss, val_loss, val_dice and
            epoch_time (seconds) lists.
        """
        best_dice = 0.0
        run_start = time.perf_counter()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.perf_counter()

            # The epoch label rides along on the progress bar, so the heading can
            # be printed afterwards with the elapsed time next to it.
            label = f"Epoch {epoch}/{num_epochs}"
            train_metrics = self._run_epoch(
                self.train_loader, training=True, desc=f"{label} train"
            )
            val_metrics = self._run_epoch(
                self.val_loader, training=False, desc=f"{label} val"
            )

            epoch_time = time.perf_counter() - epoch_start

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["epoch_time"].append(epoch_time)

            print(f"\n{label}  -  {format_duration(epoch_time)}")
            print("-" * 40)
            print(f"  Train Loss : {train_metrics['loss']:.4f}")
            print(f"  Val Loss   : {val_metrics['loss']:.4f}")
            print(f"  Val Dice   : {val_metrics['dice']:.4f}")

            elapsed = time.perf_counter() - run_start
            summary = f"  Elapsed    : {format_duration(elapsed)}"
            if epoch < num_epochs:
                mean_epoch = elapsed / epoch
                eta = mean_epoch * (num_epochs - epoch)
                summary += f"  (ETA {format_duration(eta)})"
            print(summary)

            # Save best model
            if val_metrics["dice"] > best_dice:
                best_dice = val_metrics["dice"]
                self.save_checkpoint("best_model.pth")
                print(f"  >> Saved best model (Dice={best_dice:.4f})")

        print(f"\nTraining finished in {format_duration(time.perf_counter() - run_start)}")

        return self.history

    def save_checkpoint(self, filename: str) -> None:
        """Save model weights to checkpoint_dir."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename: str) -> None:
        """Load model weights from checkpoint_dir."""
        path = self.checkpoint_dir / filename
        self.model.load_state_dict(torch.load(path, map_location=self.device))
