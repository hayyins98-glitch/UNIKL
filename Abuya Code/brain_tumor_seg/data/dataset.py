"""
Dataset classes for BraTS-PEDs NIfTI volumes.

Each case folder contains 4 MRI modalities and (for training) a segmentation mask.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from brain_tumor_seg.config import MODALITIES, TARGET_SHAPE
from brain_tumor_seg.data.survival import SurvivalStats


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

    Optionally also returns a survival regression target. Survival labels exist
    for less than half the cases, so every sample carries a 'has_survival' flag
    and the unlabeled ones are masked out of the survival loss instead of being
    excluded from the dataset.
    """

    def __init__(
        self,
        data_dir: Path,
        case_ids: Optional[List[str]] = None,
        target_shape: Tuple[int, int, int] = TARGET_SHAPE,
        survival: Optional[Dict[str, float]] = None,
        survival_stats: Optional[SurvivalStats] = None,
    ):
        self.data_dir = Path(data_dir)
        self.target_shape = target_shape
        self.case_ids = case_ids if case_ids is not None else _list_case_ids(self.data_dir)

        self.survival: Optional[Dict[str, float]] = None
        self.survival_stats: Optional[SurvivalStats] = None
        if survival is not None:
            self.attach_survival(survival, survival_stats)

    def attach_survival(
        self,
        survival: Dict[str, float],
        survival_stats: Optional[SurvivalStats] = None,
    ) -> None:
        """
        Add survival targets to an already-constructed dataset.

        Needed because the train/val split wraps this dataset in a Subset, so
        the stats can only be fitted (on the training cases alone) after the
        split has been made.

        Args:
            survival:       {case_id: survival_days}, unlabeled cases omitted.
            survival_stats: Normalization fitted on the training cases. Passing
                            None fits it on `survival`, which is only safe when
                            the mapping already contains training cases only.
        """
        self.survival = dict(survival)
        self.survival_stats = survival_stats or SurvivalStats.from_days(survival.values())

    @property
    def has_survival_targets(self) -> bool:
        """Whether __getitem__ emits the survival keys."""
        return self.survival is not None

    def n_survival_labels(self, case_ids: Optional[List[str]] = None) -> int:
        """Count how many of the given cases (all of them by default) are labeled."""
        if self.survival is None:
            return 0
        ids = self.case_ids if case_ids is None else case_ids
        return sum(1 for case_id in ids if case_id in self.survival)

    def _survival_sample(self, case_id: str) -> Dict[str, torch.Tensor]:
        """Normalized target, label flag and raw days for one case."""
        days = self.survival.get(case_id)
        labeled = days is not None

        # Unlabeled cases still need a finite placeholder target: the flag is
        # what zeroes their contribution to the loss, not the value.
        normalized = float(self.survival_stats.to_normalized(days)) if labeled else 0.0

        return {
            "survival": torch.tensor(normalized, dtype=torch.float32),
            "has_survival": torch.tensor(1.0 if labeled else 0.0, dtype=torch.float32),
            "survival_days": torch.tensor(float(days) if labeled else 0.0, dtype=torch.float32),
        }

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

        sample = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask),
            "case_id": case_id,
        }

        if self.survival is not None:
            sample.update(self._survival_sample(case_id))

        return sample


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
