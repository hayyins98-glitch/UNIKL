"""
Configuration file for brain tumor segmentation.

All paths and training settings are defined here so you can
change them in one place without editing other files.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR_NAME = "PKG - BraTS-PEDs-v1"
DATA_SUBDIR_NAME = "BraTS-PEDs-v1"


def find_data_root(start: Path) -> Path:
    """
    Look for the dataset next to the project, then in each directory above it.

    The project folder gets moved around (downloads, nested copies), so a fixed
    number of .parent hops breaks easily. If nothing is found we return the
    original guess, so the resulting error names the expected location.
    """
    for directory in [start, *start.parents]:
        candidate = directory / DATA_DIR_NAME / DATA_SUBDIR_NAME
        if candidate.is_dir():
            return candidate
    return start / DATA_DIR_NAME / DATA_SUBDIR_NAME


DATA_ROOT = find_data_root(PROJECT_ROOT)
TRAIN_DIR = DATA_ROOT / "Training"
VAL_DIR = DATA_ROOT / "Validation"

# MRI modality file suffixes (4 channels used as model input)
MODALITIES = ["t1c", "t1n", "t2f", "t2w"]

# ---------------------------------------------------------------------------
# Survival metadata
# ---------------------------------------------------------------------------
# The metadata TSV sits next to the case folders. Only the subject ID and the
# overall survival column are used; the other clinical columns are ignored.
SURVIVAL_METADATA_PATH = DATA_ROOT / "BraTS-PEDs_metadata(Survival rate).tsv"
SURVIVAL_ID_COLUMN = "BraTS-SubjectID"
SURVIVAL_DAYS_COLUMN = "Overall survival (days)"

# ---------------------------------------------------------------------------
# Data settings
# ---------------------------------------------------------------------------
# Volumes are resized to this shape for training (depth, height, width)
TARGET_SHAPE = (96, 96, 96)

# Fraction of training cases held out for validation metrics.
# Note: the Validation folder has no segmentation masks, so we split
# the Training folder for computing Dice / loss during training.
VAL_SPLIT = 0.15

# ---------------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------------
BATCH_SIZE = 1          # 3D volumes are memory-heavy; keep batch size small
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 0         # 0 is safest on Windows

# ---------------------------------------------------------------------------
# Survival head settings
# ---------------------------------------------------------------------------
# Turns the second (survival regression) head on. Only ~115 of the 257 training
# cases carry a survival label, so the survival loss is masked per sample
# instead of dropping the unlabeled cases from segmentation training.
PREDICT_SURVIVAL = True

# Segmentation stays the primary task. Survival labels are sparse and noisy, so
# their loss is scaled down to keep it from dominating the BCE + Dice gradient.
SURVIVAL_LOSS_WEIGHT = 0.3

# log1p mean/std of the survival target, saved next to the checkpoints so that
# inference can convert a normalized prediction back into days.
SURVIVAL_STATS_FILENAME = "survival_stats.json"

# Where to save model checkpoints and plots
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOT_DIR = OUTPUT_DIR / "plots"

# Random seed for reproducibility
SEED = 42
