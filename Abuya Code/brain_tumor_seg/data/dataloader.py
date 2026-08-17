"""
DataLoader factory functions.

Creates PyTorch DataLoaders for training, validation, and inference.
"""
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split

from brain_tumor_seg.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    TARGET_SHAPE,
    TRAIN_DIR,
    VAL_DIR,
    VAL_SPLIT,
)
from brain_tumor_seg.data.dataset import BraTSDataset, BraTSInferenceDataset, _list_case_ids
from brain_tumor_seg.utils.helpers import set_seed


def create_train_dataloader(
    train_dir: Path = TRAIN_DIR,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders from the Training folder.

    Because the Validation folder has no ground-truth masks, we hold out
    a fraction of training cases for validation metrics (Dice, loss).

    Returns:
        (train_loader, val_loader)
    """
    set_seed(SEED)

    all_case_ids = _list_case_ids(train_dir)
    full_dataset = BraTSDataset(train_dir, case_ids=all_case_ids, target_shape=target_shape)

    val_size = max(1, int(len(full_dataset) * val_split))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Training cases:   {len(train_dataset)}")
    print(f"Validation cases: {len(val_dataset)}")

    return train_loader, val_loader


def create_val_dataloader(
    val_dir: Path = VAL_DIR,
    batch_size: int = BATCH_SIZE,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> DataLoader:
    """
    Create a DataLoader for the official Validation folder (no masks).

    Use this for running inference on unseen cases.
    """
    dataset = BraTSInferenceDataset(val_dir, target_shape=target_shape)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Inference cases (Validation folder): {len(dataset)}")
    return loader


def create_inference_dataloader(
    data_dir: Path = VAL_DIR,
    batch_size: int = BATCH_SIZE,
    target_shape: Tuple[int, int, int] = TARGET_SHAPE,
) -> DataLoader:
    """Alias for create_val_dataloader — loads cases without masks."""
    return create_val_dataloader(data_dir, batch_size, target_shape)
