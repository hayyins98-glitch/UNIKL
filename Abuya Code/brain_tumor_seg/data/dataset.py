"""
Dataset classes for BraTS-PEDs NIfTI volumes.

Each case folder contains 4 MRI modalities and (for training) a segmentation mask.
"""
from pathlib import Path
from typing import List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from brain_tumor_seg.config import MODALITIES, TARGET_SHAPE


def _load_nifti(path: Path) -> np.ndarray:
    """Load a .nii.gz file and return the voxel data as a numpy array."""
    return nib.load(str(path)).get_fdata().astype(np.float32)


def _normalize(volume: np.ndarray) -> np.ndarray:
    """Scale each volume to zero mean and unit variance (per volume)."""
    mean = volume.mean()
    std = volume.std()
    if std < 1e-8:
        return volume - mean
    return (volume - mean) / std


def _resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize a 3D volume to target_shape using trilinear interpolation."""
    if volume.shape == target_shape:
        return volume

    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    order = 0 if volume.dtype == np.int64 or volume.dtype == np.int32 else 1
    return zoom(volume, zoom_factors, order=order)


def _list_case_ids(data_dir: Path) -> List[str]:
    """Return sorted list of case folder names inside data_dir."""
    return sorted([p.name for p in data_dir.iterdir() if p.is_dir()])


class BraTSDataset(Dataset):
    """
    PyTorch Dataset for BraTS training cases (with segmentation masks).

    Returns a 4-channel MRI volume and a binary whole-tumor mask.
    Whole tumor = any voxel where seg label is 1, 2, or 4.
    """

    def __init__(
        self,
        data_dir: Path,
        case_ids: Optional[List[str]] = None,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    ):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.case_ids = case_ids if case_ids is not None else _list_case_ids(self.data_dir)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index]
        case_dir = self.data_dir / case_id

        # Load all 4 MRI modalities and stack into shape (4, D, H, W)
        channels = []
        for mod in MODALITIES:
            path = case_dir / f"{case_id}-{mod}.nii.gz"
            volume = _normalize(_load_nifti(path))
            volume = _resize_volume(volume, self.target_shape)
            channels.append(volume)

        image = np.stack(channels, axis=0)  # (4, D, H, W)

        # Load segmentation mask -> binary whole-tumor mask
        seg_path = case_dir / f"{case_id}-seg.nii.gz"
        seg = _load_nifti(seg_path)
        seg = _resize_volume(seg, self.target_shape)
        mask = (seg > 0).astype(np.float32)  # (D, H, W)
        mask = np.expand_dims(mask, axis=0)  # (1, D, H, W)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "case_id": case_id,
        }


class BraTSInferenceDataset(Dataset):
    """
    Dataset for Validation folder cases (no segmentation mask).

    Used for inference and visualization on unseen data.
    """

    def __init__(
        self,
        data_dir: Path,
        case_ids: Optional[List[str]] = None,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
    ):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.case_ids = case_ids if case_ids is not None else _list_case_ids(self.data_dir)

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, index: int) -> dict:
        case_id = self.case_ids[index]
        case_dir = self.data_dir / case_id

        channels = []
        for mod in MODALITIES:
            path = case_dir / f"{case_id}-{mod}.nii.gz"
            volume = _normalize(_load_nifti(path))
            volume = _resize_volume(volume, self.target_shape)
            channels.append(volume)

        image = np.stack(channels, axis=0)

        return {
            "image": torch.from_numpy(image),
            "case_id": case_id,
        }
