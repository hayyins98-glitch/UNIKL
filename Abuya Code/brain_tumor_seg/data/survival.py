"""
Survival labels for BraTS-PEDs cases.

The dataset ships a single metadata TSV; only the subject ID and the overall
survival column are read here. Fewer than half the training cases carry a
survival value, so parsing returns a deliberately sparse mapping: callers are
expected to mask the unlabeled cases rather than drop them, which would throw
away most of the segmentation training data.

Survival in days is heavily right-skewed (10 to 5274 days, median around 400),
so the model target is log1p(days) z-scored with SurvivalStats. Reporting goes
back the other way through to_days().
"""
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import numpy as np

from brain_tumor_seg.config import (
    CHECKPOINT_DIR,
    SURVIVAL_DAYS_COLUMN,
    SURVIVAL_ID_COLUMN,
    SURVIVAL_METADATA_PATH,
    SURVIVAL_STATS_FILENAME,
)

# Placeholders seen in the survival column that all mean "no label".
MISSING_MARKERS = {"", "not reported", "na", "n/a", "-", "--", "nan", "none", "unknown"}

# Accepts a scalar or a whole array so the same transform serves the dataset
# (one case at a time) and the metrics (a batch at a time).
Numeric = Union[float, np.ndarray]


def _parse_days(raw: Optional[str]) -> Optional[float]:
    """
    Convert one survival cell to a float.

    Returns:
        Survival in days, or None when the cell is empty, a missing-value
        placeholder, unparseable, or negative.
    """
    if raw is None:
        return None

    text = raw.strip()
    if text.lower() in MISSING_MARKERS:
        return None

    try:
        days = float(text)
    except ValueError:
        return None

    return days if days >= 0.0 else None


def load_survival_days(
    path: Path = SURVIVAL_METADATA_PATH,
    id_column: str = SURVIVAL_ID_COLUMN,
    days_column: str = SURVIVAL_DAYS_COLUMN,
) -> Dict[str, float]:
    """
    Read the metadata TSV and return the usable survival labels.

    Args:
        path:        Tab-separated metadata file.
        id_column:   Column holding the case ID (matches the case folder names).
        days_column: Column holding overall survival in days.

    Returns:
        {case_id: survival_days} containing only rows with a usable value.
        Cases absent from the mapping have no label.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Survival metadata not found at {path}")

    # utf-8-sig: the file starts with a BOM, which would otherwise end up glued
    # to the first column name and break the lookup.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []

        missing = [col for col in (id_column, days_column) if col not in fieldnames]
        if missing:
            raise KeyError(f"Columns {missing} not found in {path.name}; got {fieldnames}")

        survival: Dict[str, float] = {}
        for row in reader:
            case_id = (row.get(id_column) or "").strip()
            days = _parse_days(row.get(days_column))
            if case_id and days is not None:
                survival[case_id] = days

    return survival


def select_survival(case_ids: Iterable[str], survival: Dict[str, float]) -> Dict[str, float]:
    """Restrict a survival mapping to the given case IDs (labeled ones only)."""
    return {case_id: survival[case_id] for case_id in case_ids if case_id in survival}


@dataclass(frozen=True)
class SurvivalStats:
    """
    Normalization constants for the survival target.

    Holds the mean and std of log1p(days) over the cases used to fit them.
    Fit these on the training split only — fitting on the held-out cases would
    leak their label distribution into the model.
    """

    mean: float
    std: float
    count: int = 0

    @classmethod
    def from_days(cls, values: Iterable[float]) -> "SurvivalStats":
        """
        Fit the log1p mean/std over a collection of survival times in days.

        Args:
            values: Survival times in days (unlabeled cases must be excluded).

        Returns:
            Fitted SurvivalStats.
        """
        days = np.asarray(list(values), dtype=np.float64)
        if days.size == 0:
            raise ValueError("Cannot fit SurvivalStats without any labeled cases.")

        log_days = np.log1p(days)
        std = float(log_days.std())

        # One label, or several identical ones, gives std 0 and would turn the
        # z-score into a division by zero. Falling back to 1.0 leaves the target
        # mean-centred, which is still a usable regression target.
        if std < 1e-6:
            std = 1.0

        return cls(mean=float(log_days.mean()), std=std, count=int(days.size))

    def to_normalized(self, days: Numeric) -> Numeric:
        """Map survival in days onto the z-scored log1p scale the model predicts."""
        return (np.log1p(days) - self.mean) / self.std

    def to_days(self, normalized: Numeric) -> Numeric:
        """Invert to_normalized, turning a model prediction back into days."""
        return np.expm1(np.asarray(normalized) * self.std + self.mean)

    def save(self, path: Path) -> Path:
        """Write the stats to JSON so inference can reuse the exact same scale."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "SurvivalStats":
        """Read stats previously written by save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mean=float(data["mean"]),
            std=float(data["std"]),
            count=int(data.get("count", 0)),
        )


def load_survival_stats(path: Optional[Path] = None) -> Optional[SurvivalStats]:
    """
    Read the stats create_train_dataloader persisted next to the checkpoints.

    Args:
        path: Stats JSON to read; defaults to the location the dataloader
              writes to.

    Returns:
        The saved SurvivalStats, or None if no run has written them yet.
    """
    path = Path(path) if path is not None else CHECKPOINT_DIR / SURVIVAL_STATS_FILENAME
    return SurvivalStats.load(path) if path.is_file() else None


def summarize_survival_days(values: Iterable[float]) -> Dict[str, float]:
    """
    Descriptive statistics of a set of survival times, for logging.

    Returns:
        dict with 'count', 'min', 'max', 'mean' and 'median' in days.
    """
    days: List[float] = [float(v) for v in values]
    if not days:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}

    array = np.asarray(days, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
    }
