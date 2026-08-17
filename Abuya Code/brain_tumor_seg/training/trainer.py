"""
Training loop for 3D brain tumor segmentation.

Works with a segmentation-only UNet3D and with the two-headed
MultiTaskUNet3D. The survival series only appear in the history when the model
actually produced survival predictions, so a single-head run behaves and plots
exactly as it did before.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from brain_tumor_seg.config import (
    CHECKPOINT_DIR,
    LEARNING_RATE,
    SURVIVAL_LOSS_WEIGHT,
    SURVIVAL_STATS_FILENAME,
)
from brain_tumor_seg.data.survival import SurvivalStats, load_survival_stats
from brain_tumor_seg.models.multitask import run_model, split_model_outputs
from brain_tumor_seg.utils.helpers import ensure_dir, format_duration
from brain_tumor_seg.utils.losses import masked_survival_loss
from brain_tumor_seg.utils.metrics import (
    dice_coefficient,
    dice_loss,
    summarize_survival_errors,
    survival_absolute_errors_days,
)

# Extra history series recorded only while the survival head is active.
SURVIVAL_HISTORY_KEYS = ("train_surv_loss", "val_surv_loss", "val_surv_mae_days")


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
        survival_loss_weight: float = SURVIVAL_LOSS_WEIGHT,
        survival_stats: Optional[SurvivalStats] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.survival_loss_weight = survival_loss_weight

        # Falls back to whatever create_train_dataloader persisted, so a caller
        # that did not thread the stats through can still report days.
        self.survival_stats = survival_stats or load_survival_stats(
            self.checkpoint_dir / SURVIVAL_STATS_FILENAME
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.bce_loss = nn.BCEWithLogitsLoss()

        # Set on the first batch that yields a survival prediction, so the
        # extra history series stay absent for a segmentation-only run.
        self.survival_active = False

        # History stored here and returned after training
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "epoch_time": [],
        }

    def _activate_survival(self) -> None:
        """Add the survival series to the history the first time the head fires."""
        if self.survival_active:
            return

        self.survival_active = True
        for key in SURVIVAL_HISTORY_KEYS:
            # Backfill so every series stays the same length as the epoch axis
            # even if the head only came online part-way through a run.
            self.history[key] = [float("nan")] * len(self.history["train_loss"])

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Combined BCE + Dice loss."""
        bce = self.bce_loss(logits, targets)
        dice = dice_loss(logits, targets)
        return bce + dice

    def _survival_loss(
        self,
        survival_pred: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Masked survival loss for one batch.

        Returns:
            Scalar loss tensor, or None when either the model has no survival
            head or the loader carries no survival targets.
        """
        if survival_pred is None or "survival" not in batch:
            return None

        targets = batch["survival"].to(self.device)
        has_label = batch["has_survival"].to(self.device)
        return masked_survival_loss(survival_pred, targets, has_label)

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
        total_surv_loss = 0.0
        surv_errors: List[float] = []
        num_batches = 0

        context = torch.enable_grad() if training else torch.no_grad()
        if desc is None:
            desc = "Train" if training else "Val"

        with context:
            for batch in tqdm(loader, desc=desc, leave=False):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Pass the ground-truth tumor so the survival head trains on
                # the actual mass, not a whole-brain average or a half-trained
                # predicted mask.
                logits, survival_pred = split_model_outputs(
                    run_model(self.model, images, tumor_mask=masks)
                )
                loss = self._compute_loss(logits, masks)

                surv_loss = self._survival_loss(survival_pred, batch)
                if surv_loss is not None:
                    self._activate_survival()
                    loss = loss + self.survival_loss_weight * surv_loss
                    total_surv_loss += surv_loss.item()

                    if not training and self.survival_stats is not None:
                        surv_errors.extend(
                            survival_absolute_errors_days(
                                survival_pred,
                                batch["survival"],
                                batch["has_survival"],
                                self.survival_stats,
                            )
                        )

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

        if self.survival_active:
            results["surv_loss"] = total_surv_loss / num_batches
            if not training:
                results.update(summarize_survival_errors(surv_errors))

        return results

    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """
        Run the full training loop for num_epochs.

        Returns:
            history dict with train_loss, val_loss, val_dice and
            epoch_time (seconds) lists, plus train_surv_loss, val_surv_loss
            and val_surv_mae_days when the survival head is active.
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

            if self.survival_active:
                self._record_survival_epoch(train_metrics, val_metrics)

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

    def _record_survival_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Append the survival series for one epoch and print them."""
        mae_days = val_metrics.get("mae_days", float("nan"))

        self.history["train_surv_loss"].append(train_metrics.get("surv_loss", float("nan")))
        self.history["val_surv_loss"].append(val_metrics.get("surv_loss", float("nan")))
        self.history["val_surv_mae_days"].append(mae_days)

        print(f"  Surv Train : {train_metrics.get('surv_loss', float('nan')):.4f}")
        print(f"  Surv Val   : {val_metrics.get('surv_loss', float('nan')):.4f}")

        labeled = int(val_metrics.get("count", 0))
        if labeled:
            median_days = val_metrics.get("median_ae_days", float("nan"))
            print(
                f"  Surv MAE   : {mae_days:,.0f} days "
                f"(median {median_days:,.0f}, {labeled} labeled)"
            )
        else:
            print("  Surv MAE   : n/a (no labeled validation cases)")

    def save_checkpoint(self, filename: str) -> None:
        """Save model weights to checkpoint_dir."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename: str) -> None:
        """
        Load model weights from checkpoint_dir.

        A checkpoint written before the survival head existed carries no
        survival-head weights, so loading it into a MultiTaskUNet3D falls back
        to warm-starting the segmentation half rather than failing outright.
        """
        path = self.checkpoint_dir / filename
        state_dict = torch.load(path, map_location=self.device)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError:
            warm_start = getattr(self.model, "load_segmentation_weights", None)
            if warm_start is None:
                raise
            warm_start(state_dict)
            print(
                f"{filename} has no survival weights (segmentation-only run); "
                "loaded the segmentation half and left the survival head untrained."
            )
