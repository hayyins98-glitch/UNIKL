"""
Brain Tumor Viewer — layout matches the wireframe:

    [Enter image]        [ 2D sagittal ]        [   3D tumor    ]
    [Run segmentation]   [ 2D axial    ]        [               ]
    [Patients record]    [ 2D coronal  ]        [  3D brain view]

Wires the PySide6 shell to `brain_tumor_seg`: the trained 3D U-Net (with
survival head), the Plotly 3D viewer, and the BraTS-PEDs survival metadata.

Run from this folder or the project root:
    python ui_layout.py
"""
import math
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

# Interface/ sits next to brain_tumor_seg/; running this file directly would
# otherwise miss the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import nibabel as nib
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage import zoom, label as ndi_label, center_of_mass as ndi_center_of_mass
import plotly.graph_objects as go

from PySide6.QtCore import Qt, QUrl, QTimer, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QSizePolicy, QSlider, QMessageBox,
    QProgressBar, QComboBox, QStackedLayout,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from brain_tumor_seg.config import CHECKPOINT_DIR, MODALITIES, TARGET_SHAPE
from brain_tumor_seg.data.dataset import _normalize
from brain_tumor_seg.data.survival import load_survival_days, load_survival_stats
from brain_tumor_seg.evaluation import predict_survival_days
from brain_tumor_seg.models import MultiTaskUNet3D
from brain_tumor_seg.models.multitask import run_model, split_model_outputs
from brain_tumor_seg.visualization.survival import format_survival
from brain_tumor_seg.visualization.viewer3d import INTERACTION_CONFIG, show_volume_3d, brain_surface_level

# ---------------------------------------------------------------------------
# Dark theme palette
# ---------------------------------------------------------------------------
# Apple-style palette (macOS/iOS dark mode system colors) — flat, no glow.
# Named after Apple's own system color roles so the mapping is obvious:
# https://developer.apple.com/design/human-interface-guidelines/color
# ---------------------------------------------------------------------------
BG_PAGE = "#1c1c1e"          # systemGray6 — page background
BG_PANEL = "#2c2c2e"         # systemGray5 — card/panel background
BORDER_DIM = "#3a3a3c"       # systemGray4 — subtle dividers only, not card borders
ACCENT_TEAL = "#0a84ff"      # systemBlue — primary accent (2D views, brain, primary button)
ACCENT_TEAL_SOFT = "#409cff"
ACCENT_CORAL = "#ff9f0a"     # systemOrange — tumor-related accent (3D tumor panel)
ACCENT_CORAL_SOFT = "#ffb340"
TEXT_LIGHT = "#f2f2f7"       # label — primary text
TEXT_MUTED = "#8e8e93"       # systemGray — secondary/muted text

# Distinct colors assigned to separate tumor components (a brain can have
# more than one lesion) — largest lesion gets the first color, and the
# same color is used consistently across the 2D heatmap, the report
# panel, and the PDF export so "the blue one" means the same tumor
# everywhere. Cycles if there are more components than colors.
COMPONENT_PALETTE = [
    ("Red", "#ff3b30"),
    ("Blue", "#0a84ff"),
    ("Green", "#30d158"),
    ("Yellow", "#ffd60a"),
    ("Purple", "#bf5af2"),
    ("Cyan", "#64d2ff"),
    ("Orange", "#ff9f0a"),
    ("Pink", "#ff375f"),
]


def _hex_to_rgb01(hex_color: str) -> tuple:
    """'#ff3b30' -> (1.0, 0.23, 0.19), for blending into a matplotlib RGB array."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)

# Single spacing unit used everywhere: gap between columns, gap between
# panels within a column, gap between buttons, and the window's outer
# margin. One number instead of several near-matching ones keeps the
# rhythm consistent across the whole interface.
SPACING = 16

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_PAGE};
}}
QLabel {{
    color: {TEXT_LIGHT};
}}
QMessageBox {{
    background-color: {BG_PANEL};
}}
QMessageBox QLabel {{
    color: {TEXT_LIGHT};
}}
QSlider {{
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: #000000;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT_TEAL};
    width: 14px;
    height: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}
QProgressBar {{
    border: none;
    border-radius: 5px;
    background-color: {BG_PANEL};
    height: 10px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_TEAL};
    border-radius: 5px;
}}
"""


def _panel_style(border_color: Optional[str] = None) -> str:
    """
    Flat Apple-style card: no border by default (contrast comes from the
    background color alone, same as a macOS card on a page). Passing an
    accent color adds a thin 1.5px outline for the 'active/has data' state
    — no drop shadow, no glow, just a quiet color change.
    """
    border = f"1.5px solid {border_color}" if border_color else "none"
    return f"QFrame {{ border: {border}; border-radius: 16px; background-color: {BG_PANEL}; }}"


MAXIMIZE_BTN_STYLE = f"""
QPushButton {{
    border: none;
    border-radius: 6px;
    background-color: transparent;
    color: {TEXT_MUTED};
    font-size: 11px;
    padding: 1px 5px;
}}
QPushButton:hover {{
    color: {ACCENT_TEAL_SOFT};
    background-color: #3a3a3c;
}}
"""

FILE_SELECTOR_STYLE = f"""
QComboBox {{
    border: none;
    border-radius: 14px;
    padding: 8px {SPACING}px;
    background-color: {BG_PANEL};
    color: {TEXT_LIGHT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_LIGHT};
    selection-background-color: {ACCENT_TEAL};
    border: none;
    outline: none;
    padding: 4px;
}}
"""

# ---------------------------------------------------------------------------
# Model configuration — uses the project checkpoint and the same volume size
# the U-Net was trained at (mask is then resized back to native resolution).
# ---------------------------------------------------------------------------
def _resolve_checkpoint() -> Path:
    candidates = (
        CHECKPOINT_DIR / "best_model.pth",
        CHECKPOINT_DIR / "best_model_segmentation_only.pth",
        Path(__file__).resolve().parent / "checkpoints" / "best_model.pth",
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


MODEL_PATH = _resolve_checkpoint()
INFERENCE_SIZE = TARGET_SHAPE
# 3D marching cubes on a native 240^3 volume stalls the UI; downsample first.
RENDER_MAX_DIM = 96
# Every loaded scan gets resampled to this shape before display, regardless
# of its native resolution — different patients/scanners can have different
# native voxel grids, and without this the 2D panels would show
# inconsistent proportions from one case to the next.
STANDARD_DISPLAY_SHAPE = (128, 128, 128)


def _standardize_volume(volume: np.ndarray, affine: np.ndarray, order: int = 1):
    """
    Resamples `volume` to STANDARD_DISPLAY_SHAPE and returns a matching
    diagonal affine reflecting the new (larger or smaller) voxel spacing,
    so downstream physical measurements — report volumes, voxel-cm3 math,
    3D mesh spacing — stay accurate after the resize rather than silently
    assuming 1mm voxels.
    """
    native_spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    old_shape = np.array(volume.shape, dtype=np.float64)
    new_shape = np.array(STANDARD_DISPLAY_SHAPE, dtype=np.float64)
    zoom_factors = new_shape / old_shape

    resampled = zoom(volume, zoom_factors, order=order)
    # zoom()'s output can be off by a voxel from rounding; crop/pad to the
    # exact target shape so every panel gets a truly identical array size.
    if resampled.shape != STANDARD_DISPLAY_SHAPE:
        slices = tuple(slice(0, min(s, t)) for s, t in zip(resampled.shape, STANDARD_DISPLAY_SHAPE))
        cropped = resampled[slices]
        pad_widths = [(0, t - c) for c, t in zip(cropped.shape, STANDARD_DISPLAY_SHAPE)]
        resampled = np.pad(cropped, pad_widths, mode="constant")

    new_spacing = native_spacing * (old_shape / new_shape)
    new_affine = np.eye(4)
    new_affine[0, 0] = new_spacing[0]
    new_affine[1, 1] = new_spacing[1]
    new_affine[2, 2] = new_spacing[2]
    return resampled, new_affine


def _muted_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(
        f"border: none; color: {TEXT_MUTED}; font-size: 11px; font-weight: 500;"
    )
    return label


def _value_label(placeholder: str = "—") -> QLabel:
    label = QLabel(placeholder)
    label.setWordWrap(True)
    label.setStyleSheet(
        f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 500;"
    )
    return label


class PatientRecordPanel(QFrame):
    """Case ID, metadata survival, and (after a run) predicted survival."""

    def __init__(self, min_height: int = 150):
        super().__init__()
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet(_panel_style())
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        title = QLabel("Patients record")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 500;"
        )
        layout.addWidget(title)

        layout.addWidget(_muted_label("Patient ID"))
        self.case_label = _value_label("No case loaded")
        layout.addWidget(self.case_label)

        layout.addWidget(_muted_label("Overall survival"))
        self.survival_label = _value_label("—")
        layout.addWidget(self.survival_label)

        layout.addWidget(_muted_label("Predicted survival"))
        self.predicted_label = _value_label("Run segmentation to predict")
        layout.addWidget(self.predicted_label)

        layout.addWidget(_muted_label("Tumor volume"))
        self.volume_label = _value_label("—")
        layout.addWidget(self.volume_label)
        layout.addStretch()

    def set_case(self, case_id: str, survival_days: Optional[float]) -> None:
        self.case_label.setText(case_id)
        if survival_days is None:
            self.survival_label.setText("No label in metadata")
        else:
            self.survival_label.setText(format_survival(survival_days))
        self.predicted_label.setText("Run segmentation to predict")
        self.volume_label.setText("—")
        self.setStyleSheet(_panel_style(ACCENT_TEAL))

    def set_predictions(
        self,
        predicted_days: Optional[float],
        tumor_cm3: Optional[float],
    ) -> None:
        if predicted_days is None:
            self.predicted_label.setText("Unavailable (no survival head)")
        else:
            self.predicted_label.setText(format_survival(predicted_days))
        if tumor_cm3 is None:
            self.volume_label.setText("—")
        else:
            self.volume_label.setText(f"{tumor_cm3:.2f} cm\u00b3")


class PixelateEffect(QWidget):
    """
    Renders a pixelated/blocky version of a snapshot of the actual slice
    image (not an abstract shape), with the block size animating between
    fine and coarse over time — reads as 'the system is actively
    processing this specific image' rather than a generic overlay.
    """

    def __init__(self, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._tint = QColor(color)
        self._source: Optional[QPixmap] = None
        self._phase = 0.0
        self._interval_ms = 60
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def set_source(self, pixmap: QPixmap):
        """Call this right before start() with a fresh grab() of the
        canvas — the effect pixelates whatever was last actually shown."""
        self._source = pixmap
        self.update()

    def start(self):
        self._timer.start(self._interval_ms)

    def stop(self):
        self._timer.stop()
        self._phase = 0.0

    def _advance(self):
        self._phase = (self._phase + 0.05) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)  # crisp blocks, no smoothing

        w, h = self.width(), self.height()
        if self._source is None or self._source.isNull() or w <= 0 or h <= 0:
            painter.end()
            return

        # Block size oscillates between fine and coarse pixelation —
        # downscale-then-upscale with no smoothing is the classic
        # pixelation trick.
        wave = (math.sin(self._phase) + 1) / 2  # 0..1
        block = max(2, int(4 + wave * 24))
        small_w = max(1, w // block)
        small_h = max(1, h // block)

        scaled_down = self._source.scaled(
            small_w, small_h, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )
        pixelated = scaled_down.scaled(
            w, h, Qt.IgnoreAspectRatio, Qt.FastTransformation
        )

        painter.setOpacity(0.9)
        painter.drawPixmap(0, 0, pixelated)

        # Faint accent tint so the pixelated frame reads as "processing,"
        # not just a low-res copy of the image.
        tint = QColor(self._tint)
        tint.setAlphaF(0.12)
        painter.setOpacity(1.0)
        painter.fillRect(self.rect(), tint)

        painter.end()


class ScanLineSweep(QWidget):
    """
    A glowing horizontal line that sweeps top-to-bottom across the whole
    panel on a loop, with a short fading trail behind it — a literal
    'scanning the image' cue. Unlike the small fixed-size loaders before
    this one, it stretches to fill whatever space it's given.
    """

    def __init__(self, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._base_color = QColor(color)
        self._progress = 0.0  # 0..1, position from top to bottom
        self._interval_ms = 20
        self._speed = 0.012  # progress added per tick
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    def start(self):
        self._timer.start(self._interval_ms)

    def stop(self):
        self._timer.stop()
        self._progress = 0.0

    def _advance(self):
        self._progress = (self._progress + self._speed) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        y = self._progress * h

        # Main line plus a few fainter copies trailing behind it, standing
        # in for a soft glow/motion-blur without needing a real blur effect.
        trail = [(0, 1.0, 3.0), (9, 0.5, 2.2), (18, 0.25, 1.6), (28, 0.1, 1.0)]
        for offset, opacity, pen_width in trail:
            trail_y = y - offset
            if trail_y < 0:
                continue
            color = QColor(self._base_color)
            color.setAlphaF(opacity)
            painter.setPen(QPen(color, pen_width))
            painter.drawLine(QPointF(0, trail_y), QPointF(w, trail_y))

        painter.end()


class ArcSpinner(QWidget):
    """
    Small segmented rotating arc — the secondary 'processing' cue paired
    with the scan line sweep above, meant to sit tucked in a corner
    rather than as the main focal point.
    """

    def __init__(self, diameter: int = 26, color: str = ACCENT_TEAL, parent=None):
        super().__init__(parent)
        self._base_color = QColor(color)
        self._angle = 0.0
        self.setFixedSize(diameter, diameter)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        self._segments = [
            (0, 60, 1.00),
            (75, 40, 0.60),
            (130, 25, 0.30),
        ]

    def start(self):
        self._timer.start(16)

    def stop(self):
        self._timer.stop()

    def _advance(self):
        self._angle = (self._angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(3, 3, -3, -3)
        pen_width = 2.5

        for offset, span, opacity in self._segments:
            color = QColor(self._base_color)
            color.setAlphaF(opacity)
            painter.setPen(QPen(color, pen_width, Qt.SolidLine, Qt.RoundCap))
            start_angle = int((self._angle + offset) * 16)
            span_angle = int(span * 16)
            painter.drawArc(rect, start_angle, span_angle)

        painter.end()


class SlicePanel(QFrame):
    """
    A bordered panel that shows one anatomical plane of a 3D volume as a
    matplotlib image, with a slider underneath to scroll through slices
    along that axis, and an optional tumor mask overlaid in red.

    `axis` is which array axis is sliced to produce this 2D view:
        0 -> sagittal, 1 -> coronal, 2 -> axial
    """

    def __init__(self, title: str, axis: int, min_height: int = 200):
        super().__init__()
        self.axis = axis
        self.volume = None      # grayscale background, 3D numpy array, 0-1 range
        self.mask = None        # optional binary mask, same shape as volume
        self.probs = None       # optional continuous 0-1 probability map, same shape
        self.labels = None      # optional int array, same shape: 0=background, 1..N=component id
        self.color_map = None   # optional {component_id: hex_color}, matching self.labels

        self.setFrameShape(QFrame.Box)
        # Pure black background (not the charcoal BG_PANEL used elsewhere)
        # — MRI slices have their own black background outside the brain,
        # so matching that exactly makes the panel disappear into the
        # image instead of showing as a separate charcoal-colored frame
        # around it.
        self.setStyleSheet(
            "QFrame { border: 1.5px solid #000000; border-radius: 16px; "
            "background-color: #000000; }"
        )
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addStretch(1)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 500;")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.maximize_btn = QPushButton("⛶")
        self.maximize_btn.setFixedSize(22, 20)
        self.maximize_btn.setStyleSheet(MAXIMIZE_BTN_STYLE)
        self.maximize_btn.setToolTip("Maximize")
        header.addWidget(self.maximize_btn)
        layout.addLayout(header)

        self.figure = Figure(figsize=(3, 3))
        self.figure.patch.set_facecolor("#000000")
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet("background-color: #000000;")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#000000")
        self.ax.axis("off")

        # Canvas + loading overlay share the same space via a stacked
        # layout, so the spinner appears directly on top of the slice
        # image instead of needing a separate panel or popup.
        self.canvas_stack = QStackedLayout()
        self.canvas_stack.setStackingMode(QStackedLayout.StackAll)
        canvas_container = QWidget()
        canvas_container.setLayout(self.canvas_stack)
        self.canvas_stack.addWidget(self.canvas)

        self.loading_overlay = QWidget()
        # Lighter than before — the pixelated image itself now carries most
        # of the visual weight, so a heavy flat black backdrop would just
        # compete with it instead of framing it.
        self.loading_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 90);")
        overlay_stack = QStackedLayout(self.loading_overlay)
        overlay_stack.setStackingMode(QStackedLayout.StackAll)

        # Base layer: a pixelated snapshot of whatever was last actually
        # shown in this panel, with block size animating — "the system is
        # processing this specific image," not just a generic dimmed panel.
        self.pixelate_effect = PixelateEffect(color=ACCENT_TEAL)
        overlay_stack.addWidget(self.pixelate_effect)

        # Middle effect: the scan line sweeps top-to-bottom across the
        # (now pixelated) slice image.
        self.scan_line = ScanLineSweep(color=ACCENT_TEAL)
        overlay_stack.addWidget(self.scan_line)

        # Top effect: small arc spinner tucked in the top-right corner,
        # with the caption anchored at the bottom — this layer's own
        # background stays transparent so the layers underneath stay
        # visible everywhere except where these small widgets sit.
        corner_layer = QWidget()
        corner_layer.setStyleSheet("background: transparent;")
        corner_layout = QVBoxLayout(corner_layer)
        corner_layout.setContentsMargins(10, 10, 10, 10)

        spinner_row = QHBoxLayout()
        spinner_row.addStretch(1)
        self.loading_animation = ArcSpinner(diameter=26, color=ACCENT_TEAL_SOFT)
        spinner_row.addWidget(self.loading_animation)
        corner_layout.addLayout(spinner_row)
        corner_layout.addStretch(1)

        self.loading_caption = QLabel("SCANNING")
        self.loading_caption.setAlignment(Qt.AlignCenter)
        self.loading_caption.setStyleSheet(
            f"color: {ACCENT_TEAL_SOFT}; font-size: 10px; font-weight: 600; "
            f"letter-spacing: 2px; background: transparent; border: none;"
        )
        corner_layout.addWidget(self.loading_caption)

        overlay_stack.addWidget(corner_layer)

        self.canvas_stack.addWidget(self.loading_overlay)
        self.loading_overlay.hide()

        layout.addWidget(canvas_container, stretch=1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self._base_title = title

    def set_loading(self, active: bool):
        """Shows/hides the pixelate + scan-line + corner-arc loading
        animation over the slice image — used while Run segmentation is
        processing this panel's data."""
        if active:
            # Grab whatever's currently on screen (the last-shown slice)
            # right now, before any of it changes — that's what gets
            # pixelated for the duration of this loading pass.
            self.pixelate_effect.set_source(self.canvas.grab())
            self.loading_overlay.show()
            self.loading_overlay.raise_()
            self.pixelate_effect.start()
            self.scan_line.start()
            self.loading_animation.start()
        else:
            self.pixelate_effect.stop()
            self.scan_line.stop()
            self.loading_animation.stop()
            self.loading_overlay.hide()

    def set_volume(self, volume: np.ndarray):
        """volume: 3D numpy array (already normalized for display, 0-1 range)."""
        self.volume = volume
        self.mask = None   # new volume invalidates any previous overlay
        self.probs = None
        self.labels = None
        self.color_map = None
        n_slices = volume.shape[self.axis]
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n_slices - 1, 0))
        self.slider.setValue(n_slices // 2)
        self._draw_slice(n_slices // 2)

    def set_mask(self, mask: np.ndarray):
        """
        mask: binary 3D numpy array, same shape as the current volume.
        Renders as a hard on/off red overlay. Superseded by
        set_detection() when a probability map + per-component labels
        are available (see on_run_segmentation) — kept as a fallback API
        for callers that only have a plain binary mask.
        """
        if self.volume is not None and mask.shape != self.volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {self.volume.shape}"
            )
        self.mask = mask
        self.probs = None
        self.labels = None
        self.color_map = None
        self._draw_slice(self.slider.value())

    def set_detection(self, probs: np.ndarray, labels: np.ndarray, color_map: dict):
        """
        probs: continuous 0-1 probability array (the model's raw sigmoid
        output, before thresholding). labels: same-shape int array from
        connected-component labeling (0=background, 1..N=component id —
        a brain can have more than one separate lesion). color_map:
        {component_id: hex_color}, assigning each distinct tumor its own
        color, matching the same colors used in the report/PDF.

        Blend strength still scales with confidence per voxel (see
        _draw_slice), but the color itself now depends on which
        component that voxel belongs to, instead of every tumor
        rendering in the same flat red.
        """
        if self.volume is not None and probs.shape != self.volume.shape:
            raise ValueError(
                f"probability map shape {probs.shape} does not match volume shape {self.volume.shape}"
            )
        self.probs = probs
        self.labels = labels
        self.color_map = color_map
        self.mask = None  # heatmap supersedes the binary overlay for display
        self._draw_slice(self.slider.value())

    def _on_slider_changed(self, index: int):
        self._draw_slice(index)

    def _slice_along_axis(self, array: np.ndarray, index: int) -> np.ndarray:
        if self.axis == 0:
            return array[index, :, :]
        elif self.axis == 1:
            return array[:, index, :]
        else:
            return array[:, :, index]

    def _draw_slice(self, index: int):
        if self.volume is None:
            return
        img_slice = self._slice_along_axis(self.volume, index)
        rgb = np.stack([img_slice, img_slice, img_slice], axis=-1)

        if self.probs is not None:
            # Continuous heatmap: blend strength scales with the model's
            # actual confidence at each voxel, rather than a flat on/off
            # color. Three things make the gradient actually visible
            # instead of a flat wash:
            #   1. A higher floor (0.2) drops low-confidence background
            #      noise entirely, instead of tinting the whole image
            #      faintly and diluting the contrast that matters.
            #   2. A gamma < 1 stretches the remaining 0.2-1.0 range so
            #      mid-confidence differences are visually distinguishable,
            #      not compressed into a narrow band near full opacity.
            #   3. A higher max alpha (0.95) lets fully-confident voxels
            #      read as strong, clearly-visible color.
            prob_slice = self._slice_along_axis(self.probs, index)
            confidence = np.clip(prob_slice, 0.0, 1.0)

            floor = 0.2
            gamma = 0.55
            max_alpha = 0.95

            normalized = np.zeros_like(confidence)
            visible = confidence > floor
            normalized[visible] = (confidence[visible] - floor) / (1.0 - floor)
            alpha_all = np.power(normalized, gamma) * max_alpha

            if self.labels is not None and self.color_map:
                # Color each connected component with its own assigned
                # color, so separate tumors are visually distinguishable
                # and match the same colors used in the report.
                label_slice = self._slice_along_axis(self.labels, index)
                for label_id, hex_color in self.color_map.items():
                    comp_rgb = _hex_to_rgb01(hex_color)
                    comp_region = label_slice == label_id
                    if not comp_region.any():
                        continue
                    a = alpha_all * comp_region
                    for c in range(3):
                        rgb[..., c] = np.where(
                            comp_region, (1 - a) * rgb[..., c] + a * comp_rgb[c], rgb[..., c]
                        )
                # Any visible-confidence voxels that ended up outside every
                # labeled component (shouldn't normally happen, since
                # labels come from the same mask) still show up in the
                # default red rather than silently vanishing.
                labeled_anywhere = label_slice > 0
                leftover = visible & (~labeled_anywhere)
                if leftover.any():
                    a = alpha_all * leftover
                    default_rgb = (1.0, 0.15, 0.15)
                    for c in range(3):
                        rgb[..., c] = np.where(
                            leftover, (1 - a) * rgb[..., c] + a * default_rgb[c], rgb[..., c]
                        )
            else:
                # No per-component labels available — fall back to the
                # single flat red heatmap.
                red = (1.0, 0.15, 0.15)
                for c in range(3):
                    rgb[..., c] = (1 - alpha_all) * rgb[..., c] + alpha_all * red[c]
        elif self.mask is not None:
            mask_slice = self._slice_along_axis(self.mask, index)
            hit = mask_slice > 0.5
            rgb[hit, 0] = 1.0
            rgb[hit, 1] = 0.15
            rgb[hit, 2] = 0.15

        self.ax.clear()
        self.ax.imshow(np.rot90(rgb))
        self.ax.axis("off")
        self.canvas.draw_idle()

        self.title_label.setText(f"{self._base_title}  (slice {index})")

    def set_maximized(self, is_max: bool):
        """Swaps the button glyph/tooltip; actual show/hide of sibling
        panels is handled by MainWindow.toggle_maximize()."""
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")


def _inject_dark_page_style(html_path: str):
    """
    Plotly's write_html doesn't expose the page <body> background directly
    — only the plot's own paper/plot background, which leaves the HTML
    page's default white margin visible around the figure if the plot
    doesn't exactly fill the QWebEngineView. This patches a small <style>
    block into the generated file so the whole page matches the dark
    panel instead.
    """
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return
    style_block = (
        f"<style>html,body{{margin:0;padding:0;background:{BG_PANEL};"
        f"overflow:hidden;}}</style>"
    )
    if "<head>" in content:
        content = content.replace("<head>", f"<head>{style_block}", 1)
        try:
            path.write_text(content, encoding="utf-8")
        except OSError:
            pass


def _inject_middle_click_pan(html_path: str):
    """
    Plotly's 3D scenes support panning natively via right-click-drag, but
    that fights with the browser's own context menu inside a
    QWebEngineView. This injects a small script that instead pans on
    middle-mouse-button drag (holding the scroll wheel down and moving
    the mouse) — left-drag still rotates and the wheel still zooms,
    exactly as before, this only adds the missing pan gesture.

    The math: compute the camera's current right and up vectors from its
    eye/center/up, then translate both eye and center together along
    those vectors by an amount proportional to the mouse movement — a
    true pan that keeps the viewing direction unchanged, rather than a
    rotation.
    """
    try:
        path = Path(html_path)
        content = path.read_text(encoding="utf-8")
    except OSError:
        return

    script_block = """
<script>
(function() {
    function setupMiddleClickPan() {
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd) { setTimeout(setupMiddleClickPan, 100); return; }

        var dragging = false;
        var lastX = 0, lastY = 0;
        var panScale = 0.0022;

        gd.addEventListener('mousedown', function(e) {
            if (e.button === 1) {
                dragging = true;
                lastX = e.clientX;
                lastY = e.clientY;
                e.preventDefault();
            }
        });

        window.addEventListener('mouseup', function(e) {
            if (e.button === 1) { dragging = false; }
        });

        window.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            e.preventDefault();

            var dx = e.clientX - lastX;
            var dy = e.clientY - lastY;
            lastX = e.clientX;
            lastY = e.clientY;

            var layout = gd._fullLayout;
            var scene = layout && layout.scene;
            var camera = scene && scene.camera;
            if (!camera) return;

            var eye = camera.eye;
            var center = camera.center || {x: 0, y: 0, z: 0};
            var up = camera.up || {x: 0, y: 0, z: 1};

            var fx = center.x - eye.x, fy = center.y - eye.y, fz = center.z - eye.z;
            var flen = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
            fx /= flen; fy /= flen; fz /= flen;

            var rx = fy * up.z - fz * up.y;
            var ry = fz * up.x - fx * up.z;
            var rz = fx * up.y - fy * up.x;
            var rlen = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;
            rx /= rlen; ry /= rlen; rz /= rlen;

            var ux = ry * fz - rz * fy;
            var uy = rz * fx - rx * fz;
            var uz = rx * fy - ry * fx;

            var dxp = -dx * panScale;
            var dyp = dy * panScale;

            var moveX = rx * dxp + ux * dyp;
            var moveY = ry * dxp + uy * dyp;
            var moveZ = rz * dxp + uz * dyp;

            Plotly.relayout(gd, {
                'scene.camera.eye': {x: eye.x + moveX, y: eye.y + moveY, z: eye.z + moveZ},
                'scene.camera.center': {x: center.x + moveX, y: center.y + moveY, z: center.z + moveZ}
            });
        });
    }
    setupMiddleClickPan();
})();
</script>
"""
    if "</body>" in content:
        content = content.replace("</body>", f"{script_block}</body>", 1)
    else:
        content += script_block

    try:
        path.write_text(content, encoding="utf-8")
    except OSError:
        pass


class Panel3D(QFrame):
    """
    A bordered panel that shows an interactive 3D Plotly figure (rotate,
    zoom, pan) via an embedded web view. Qt has no native Plotly renderer,
    so the figure is written to a temporary standalone HTML file and loaded
    into a QWebEngineView — the same approach as save_figure_html() in the
    existing viewer3d.py, just pointed at a temp file instead of a
    user-chosen path.
    """

    def __init__(self, title: str, min_height: int = 200, accent: str = ACCENT_TEAL):
        super().__init__()
        self.accent = accent
        self.setFrameShape(QFrame.Box)
        # No border at all, active or not — unlike the 2D panels, the 3D
        # panels stay borderless; the accent color is used elsewhere
        # (e.g. tumor-panel labeling) instead of as a frame outline here.
        self.setStyleSheet(_panel_style())
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)
        header.addStretch(1)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet(f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 500;")
        header.addWidget(self.title_label)
        header.addStretch(1)
        self.maximize_btn = QPushButton("⛶")
        self.maximize_btn.setFixedSize(22, 20)
        self.maximize_btn.setStyleSheet(MAXIMIZE_BTN_STYLE)
        self.maximize_btn.setToolTip("Maximize")
        header.addWidget(self.maximize_btn)
        layout.addLayout(header)

        self.placeholder_label = QLabel("(run segmentation to render)")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet(f"border: none; color: {TEXT_MUTED}; font-size: 12px;")
        layout.addWidget(self.placeholder_label, stretch=1)

        self.web_view = None  # created lazily on first set_figure() call
        self._layout = layout
        self._temp_files = []  # keep references so temp files aren't GC'd/deleted early

    def set_figure(self, fig: go.Figure):
        if self.web_view is None:
            self.placeholder_label.hide()
            self.web_view = QWebEngineView()
            # A transparent-looking background on the QWebEngineView itself
            # so there's no flash of default white before the page loads.
            self.web_view.setStyleSheet(f"background-color: {BG_PANEL}; border: none;")
            self._layout.addWidget(self.web_view, stretch=1)

        # Hide Plotly's icon toolbar (zoom/pan/camera/reset buttons) — the
        # panel is still fully interactive via mouse drag/scroll, just
        # without the visible icon strip.
        config = {**INTERACTION_CONFIG, "displayModeBar": False}

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(
            tmp.name,
            include_plotlyjs=True,
            full_html=True,
            config=config,
            default_width="100%",
            default_height="100%",
        )
        _inject_dark_page_style(tmp.name)
        _inject_middle_click_pan(tmp.name)
        self._temp_files.append(tmp.name)
        self.web_view.load(QUrl.fromLocalFile(tmp.name))

    def set_maximized(self, is_max: bool):
        """Swaps the button glyph/tooltip; actual show/hide of sibling
        panels is handled by MainWindow.toggle_maximize()."""
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")


def _style_ui_figure(fig: go.Figure) -> go.Figure:
    """Strip notebook chrome so the mesh fills the dark Qt panel."""
    fig.update_layout(
        title=None,
        width=None,
        height=None,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_PANEL,
        plot_bgcolor=BG_PANEL,
        scene_bgcolor=BG_PANEL,
        showlegend=False,
        font=dict(color=TEXT_LIGHT),
    )
    return fig


def _prepare_3d(volume: np.ndarray, mask: Optional[np.ndarray], affine: np.ndarray):
    """Downsample for marching cubes and convert the affine into voxel spacing."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).astype(np.float64)
    longest = float(max(volume.shape))
    factor = min(1.0, RENDER_MAX_DIM / longest) if longest else 1.0
    if factor < 0.99:
        new_shape = tuple(max(8, int(round(size * factor))) for size in volume.shape)
        zooms = [new / old for new, old in zip(new_shape, volume.shape)]
        volume = zoom(volume, zooms, order=1)
        if mask is not None:
            mask = (zoom(mask.astype(np.float32), zooms, order=0) > 0.5).astype(np.float32)
        spacing = spacing / factor
    return volume, mask, tuple(float(s) for s in spacing)


def build_tumor_figure(mask: np.ndarray, spacing: tuple) -> go.Figure:
    """Tumor mesh alone, from the shared 3D viewer."""
    fig = show_volume_3d(
        mask,
        mask,
        show_brain=False,
        level=0.5,
        min_voxels=25,
        spacing=spacing,
        title="3D tumor",
        clean=True,
        show_legend=False,
    )
    return _style_ui_figure(fig)


def build_brain_figure(
    volume: np.ndarray,
    mask: Optional[np.ndarray],
    spacing: tuple,
) -> go.Figure:
    """Brain shell, with the tumor inside it when a mask is available."""
    fig = show_volume_3d(
        volume,
        mask,
        show_brain=True,
        min_voxels=25 if mask is not None else 1,
        spacing=spacing,
        title="3D brain",
        clean=True,
        show_legend=False,
    )
    return _style_ui_figure(fig)


class MainWindow(QMainWindow):
    def _size_to_screen(self):
        """
        Sizes and centers the window relative to whatever screen it's
        opened on — 85% of available space, capped so it doesn't become
        awkwardly huge on an ultrawide/4K monitor, with a sane minimum so
        it doesn't get squashed on a small laptop screen either.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 720)
            return

        avail = screen.availableGeometry()
        width = max(min(int(avail.width() * 0.85), 1500), 900)
        height = max(min(int(avail.height() * 0.85), 950), 600)
        self.resize(width, height)

        x = avail.x() + (avail.width() - width) // 2
        y = avail.y() + (avail.height() - height) // 2
        self.move(x, y)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain Tumor Viewer")
        self._size_to_screen()

        # Current case state — shared between "Enter image" and "Run
        # segmentation" so picking a patient folder once is enough for both.
        self.current_folder: Optional[Path] = None
        self._nii_files: list = []
        self._display_volume: Optional[np.ndarray] = None
        self._display_affine: Optional[np.ndarray] = None
        self._model = None
        self._device = None
        self._survival_stats = load_survival_stats()
        try:
            self._survival_table = load_survival_days()
        except (FileNotFoundError, KeyError):
            self._survival_table = {}
        self._current_report: Optional[dict] = None

        central = QWidget()

        # ---------------- Left column: image input + patient record -------
        left = QVBoxLayout()
        left.setSpacing(SPACING)

        self.enter_image_btn = QPushButton("Enter image")
        self.enter_image_btn.setStyleSheet(
            f"QPushButton {{ border: none; border-radius: 14px; "
            f"padding: 10px {SPACING}px; background-color: {BG_PANEL}; color: {ACCENT_TEAL}; font-weight: 500; }}"
            f"QPushButton:hover {{ background-color: #3a3a3c; }}"
            f"QPushButton:pressed {{ background-color: #48484a; }}"
        )
        self.enter_image_btn.clicked.connect(self.on_enter_image)
        left.addWidget(self.enter_image_btn)

        # Lists every .nii/.nii.gz file found in the selected folder;
        # switching the selection re-previews that file in the 2D panels.
        # Hidden until a folder with at least one match has been loaded.
        self.file_selector = QComboBox()
        self.file_selector.setStyleSheet(FILE_SELECTOR_STYLE)
        self.file_selector.setVisible(False)
        self.file_selector.currentIndexChanged.connect(self.on_file_selected)
        left.addWidget(self.file_selector)

        # Primary action gets the one filled accent button in this view
        # (Apple's restraint pattern: exactly one solid-accent button per
        # screen, everything else stays quiet/secondary). The play glyph
        # reinforces "this executes something" vs. "Enter image" which is
        # just a selection/input action.
        self.run_seg_btn = QPushButton("\u25b6  Run segmentation")
        self.run_seg_btn.setStyleSheet(
            f"QPushButton {{ border: none; border-radius: 14px; "
            f"padding: 10px {SPACING}px; background-color: {ACCENT_TEAL}; color: #ffffff; font-weight: 500; }}"
            f"QPushButton:hover {{ background-color: {ACCENT_TEAL_SOFT}; }}"
            f"QPushButton:pressed {{ background-color: #0060c4; }}"
        )
        self.run_seg_btn.clicked.connect(self.on_run_segmentation)
        left.addWidget(self.run_seg_btn)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate — pulses while active
        self.busy_bar.setTextVisible(False)
        self.busy_bar.hide()
        left.addWidget(self.busy_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        left.addWidget(self.status_label)

        self.patients_record_panel = PatientRecordPanel(min_height=150)
        left.addWidget(self.patients_record_panel)

        # Report summary — built only from what this single whole-tumor
        # mask can honestly support, plus the survival prediction the
        # patients-record panel already computed. Left-aligned body text,
        # unlike the centered titles on the 2D/3D panels, since this is
        # read top-to-bottom like a small document rather than glanced at
        # like a chart title.
        self.report_frame = QFrame()
        self.report_frame.setStyleSheet(_panel_style())
        self.report_frame.setMinimumHeight(150)
        report_layout = QVBoxLayout(self.report_frame)
        report_layout.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        report_layout.setSpacing(6)

        report_title = QLabel("Report summary")
        report_title.setStyleSheet(f"border: none; color: {TEXT_LIGHT}; font-size: 13px; font-weight: 600;")
        report_layout.addWidget(report_title)

        self.report_body_label = QLabel("Run segmentation to generate a report.")
        self.report_body_label.setWordWrap(True)
        self.report_body_label.setStyleSheet(f"border: none; color: {TEXT_MUTED}; font-size: 11px;")
        report_layout.addWidget(self.report_body_label)
        report_layout.addStretch()

        left.addWidget(self.report_frame)

        self.export_report_btn = QPushButton("Export PDF report")
        self.export_report_btn.setEnabled(False)
        self.export_report_btn.setStyleSheet(
            f"QPushButton {{ border: none; border-radius: 14px; "
            f"padding: 10px {SPACING}px; background-color: {BG_PANEL}; color: {ACCENT_TEAL}; font-weight: 500; }}"
            f"QPushButton:hover {{ background-color: #3a3a3c; }}"
            f"QPushButton:pressed {{ background-color: #48484a; }}"
            f"QPushButton:disabled {{ color: {TEXT_MUTED}; }}"
        )
        self.export_report_btn.clicked.connect(self.on_export_report)
        left.addWidget(self.export_report_btn)

        left.addStretch()

        self.left_container = QWidget()
        self.left_container.setLayout(left)
        self.left_container.setMinimumWidth(160)
        self.left_container.setMaximumWidth(240)  # keeps this column narrow at any window size

        # ---------------- Center column: 2D views --------------------------
        # min_height is kept modest (not the visual target size) — it's only
        # the floor for the smallest supported window. QSizePolicy.Expanding
        # + the stretch factors below do the actual responsive growing, so
        # these panels fill available space on a normal-sized window without
        # ever needing a scrollbar at the small end either.
        center = QVBoxLayout()
        center.setSpacing(SPACING)
        # axis 0 = L-R (sagittal), axis 1 = A-P (coronal), axis 2 = S-I (axial)
        self.sagittal_panel = SlicePanel("2D sagittal view", axis=0, min_height=140)
        self.axial_panel = SlicePanel("2D axial view", axis=2, min_height=140)
        self.coronal_panel = SlicePanel("2D coronal view", axis=1, min_height=140)
        center.addWidget(self.sagittal_panel, stretch=1)
        center.addWidget(self.axial_panel, stretch=1)
        center.addWidget(self.coronal_panel, stretch=1)

        self.center_container = QWidget()
        self.center_container.setLayout(center)

        # ---------------- Right column: 3D views ---------------------------
        right = QVBoxLayout()
        right.setSpacing(SPACING)
        self.tumor_3d_panel = Panel3D("3D tumor", min_height=150, accent=ACCENT_CORAL)
        self.brain_3d_panel = Panel3D("3D brain view", min_height=200, accent=ACCENT_TEAL)
        # brain view gets more of the extra vertical space than the smaller
        # tumor-only panel above it, matching the original wireframe's
        # proportions (small panel on top, tall panel below)
        right.addWidget(self.tumor_3d_panel, stretch=1)
        right.addWidget(self.brain_3d_panel, stretch=2)

        self.right_container = QWidget()
        self.right_container.setLayout(right)
        self.right_container.setMinimumWidth(240)

        # ---------------- Assemble ------------------------------------------
        # Plain QHBoxLayout with stretch factors — pure responsive design.
        # Every column has QSizePolicy.Expanding, so the layout grows/shrinks
        # proportionally with the window on its own; no draggable splitter
        # and no scroll-area fallback needed, since the minimum heights above
        # were chosen to always fit within _size_to_screen()'s smallest
        # supported window size.
        root = QHBoxLayout(central)
        root.setSpacing(SPACING)
        root.setContentsMargins(SPACING, SPACING, SPACING, SPACING)
        root.addWidget(self.left_container, stretch=0)
        root.addWidget(self.center_container, stretch=3)
        root.addWidget(self.right_container, stretch=2)

        self.setCentralWidget(central)

        # ---------------- Maximize / restore wiring --------------------------
        self.center_panels = [self.sagittal_panel, self.axial_panel, self.coronal_panel]
        self.right_panels = [self.tumor_3d_panel, self.brain_3d_panel]
        self.all_panels = self.center_panels + self.right_panels
        self._maximized_panel = None

        for panel in self.all_panels:
            panel.maximize_btn.clicked.connect(lambda checked=False, p=panel: self.toggle_maximize(p))

    def toggle_maximize(self, panel):
        """Clicking a panel's maximize button hides every other panel (and
        the now-empty side column) so it fills the available space; clicking
        it again restores the normal 3-column grid. A plain QHBoxLayout
        already collapses a hidden widget's space to its neighbors, so no
        manual width juggling is needed here."""
        if self._maximized_panel is panel:
            # restore
            self.left_container.show()
            self.center_container.show()
            self.right_container.show()
            for p in self.all_panels:
                p.show()
            panel.set_maximized(False)
            self._maximized_panel = None
            return

        if self._maximized_panel is not None:
            self._maximized_panel.set_maximized(False)

        self.left_container.hide()
        if panel in self.center_panels:
            self.right_container.hide()
            self.center_container.show()
            for p in self.center_panels:
                p.setVisible(p is panel)
        else:
            self.center_container.hide()
            self.right_container.show()
            for p in self.right_panels:
                p.setVisible(p is panel)

        panel.set_maximized(True)
        self._maximized_panel = panel

    def on_run_segmentation(self):
        # Reuse the folder already loaded via "Enter image" if there is
        # one — only prompt when no case is loaded yet, so the two buttons
        # share a single "current patient" instead of asking twice.
        if self.current_folder is None:
            folder = QFileDialog.getExistingDirectory(
                self, "Select folder with t1, t1ce, t2, flair NIfTI files"
            )
            if not folder:
                return
            if not self._load_folder(Path(folder)):
                return

        folder_path = self.current_folder
        try:
            modality_paths = self._find_modality_files(folder_path)
            display_volume, model_input, affine = self._load_and_stack(modality_paths)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load scans", str(exc))
            return

        try:
            model = self._get_model()
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load model",
                f"{exc}\n\nExpected checkpoint at:\n{MODEL_PATH}"
            )
            return

        self.busy_bar.show()
        self._set_2d_loading(True)
        self.status_label.setText("Running segmentation...")
        QApplication.processEvents()  # let the UI repaint before the model blocks

        try:
            mask, probs, predicted_days = self._run_model(
                model, model_input, target_shape=display_volume.shape
            )
        except Exception as exc:
            self.busy_bar.hide()
            self._set_2d_loading(False)
            QMessageBox.critical(self, "Inference failed", str(exc))
            self.status_label.setText("")
            return

        self.sagittal_panel.set_volume(display_volume)
        self.axial_panel.set_volume(display_volume)
        self.coronal_panel.set_volume(display_volume)

        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        labels, components, color_map = self._label_tumor_components(mask, probs, voxel_vol_cm3)

        self.sagittal_panel.set_detection(probs, labels, color_map)
        self.axial_panel.set_detection(probs, labels, color_map)
        self.coronal_panel.set_detection(probs, labels, color_map)
        self._set_2d_loading(False)

        tumor_cm3 = float(mask.sum()) * voxel_vol_cm3
        self.status_label.setText(f"Tumor volume: {tumor_cm3:.2f} cm\u00b3")
        self.patients_record_panel.set_predictions(predicted_days, tumor_cm3)

        report = self._compute_report(mask, probs, display_volume, affine, predicted_days, components)
        self._update_report_panel(report)

        self._update_3d_views(display_volume, mask, affine)
        self.busy_bar.hide()

    def _label_tumor_components(self, mask: np.ndarray, probs: np.ndarray, voxel_vol_cm3: float):
        """
        Splits the binary mask into separate connected components — a
        brain can have more than one lesion — and assigns each a distinct
        color from COMPONENT_PALETTE, largest volume first, so the same
        color consistently identifies the same tumor across the 2D
        heatmap, the report panel, and the PDF export.

        Returns (labels, components, color_map):
            labels:     int array, same shape as mask (0=background, 1..N=component id)
            components: list of dicts, sorted largest-first, each with
                        label_id, voxels, volume_cm3, confidence, rank,
                        color_name, color_hex
            color_map:  {label_id: hex_color}, for passing straight into
                        SlicePanel.set_detection()
        """
        labels, n_components = ndi_label(mask > 0.5)

        entries = []
        for label_id in range(1, n_components + 1):
            comp_mask = labels == label_id
            voxels = int(comp_mask.sum())
            if voxels == 0:
                continue
            entries.append({
                "label_id": label_id,
                "voxels": voxels,
                "volume_cm3": voxels * voxel_vol_cm3,
                "confidence": float(probs[comp_mask].mean()),
            })

        # Largest first, so "#1" consistently means the biggest lesion
        # rather than an arbitrary scan-order label id.
        entries.sort(key=lambda e: e["volume_cm3"], reverse=True)

        color_map = {}
        for rank, entry in enumerate(entries):
            color_name, color_hex = COMPONENT_PALETTE[rank % len(COMPONENT_PALETTE)]
            entry["rank"] = rank + 1
            entry["color_name"] = color_name
            entry["color_hex"] = color_hex
            color_map[entry["label_id"]] = color_hex

        return labels, entries, color_map

    def _set_2d_loading(self, active: bool):
        """Shows/hides the spinner overlay on all three 2D panels at once."""
        for panel in self.center_panels:
            panel.set_loading(active)

    def _update_3d_views(
        self,
        display_volume: np.ndarray,
        mask: Optional[np.ndarray],
        affine: np.ndarray,
    ):
        """
        Build the two 3D Plotly figures via `show_volume_3d` and load them
        into the right-column panels. A None/empty mask still draws the brain.
        """
        volume_3d, mask_3d, spacing = _prepare_3d(display_volume, mask, affine)

        try:
            has_tumor = mask_3d is not None and float(np.asarray(mask_3d).sum()) > 0
            if has_tumor:
                self.tumor_3d_panel.set_figure(build_tumor_figure(mask_3d, spacing))
            self.brain_3d_panel.set_figure(build_brain_figure(volume_3d, mask_3d, spacing))
        except Exception as exc:
            QMessageBox.warning(self, "3D render failed", str(exc))

    @staticmethod
    def _find_modality_files(folder: Path) -> dict:
        """
        Looks for files whose names contain a modality tag (case-insensitive),
        immediately preceded by a hyphen or underscore and followed by a dot
        or another separator — e.g. "BraTS-PED-00005-000-t1n.nii.gz" (BraTS
        uses hyphens) or "patient_01_t1.nii.gz" (underscore convention).

        Tries both common BraTS naming conventions per modality:
            - classic BraTS:  t1 / t1ce / t2 / flair
            - BraTS-PEDs:     t1n / t1c / t2w / t2f
        (t1n=T1 native, t1c=T1 post-contrast, t2w=T2-weighted, t2f=T2-FLAIR —
        matching the MODALITIES list in your existing config.py)
        """
        tag_options = {
            "t1c": ["t1c", "t1ce"],
            "t1n": ["t1n", "t1"],
            "t2f": ["t2f", "flair"],
            "t2w": ["t2w", "t2"],
        }
        candidates = list(folder.glob("*.nii*"))
        found = {}
        for key in MODALITIES:
            tags = tag_options[key]
            match = None
            for tag in tags:
                pattern = re.compile(rf"[_-]{tag}[_.]", re.IGNORECASE)
                hits = [p for p in candidates if pattern.search(p.name)]
                if hits:
                    match = hits[0]
                    break
            if match is None:
                found_names = "\n  ".join(p.name for p in candidates) or "(no .nii/.nii.gz files found)"
                raise FileNotFoundError(
                    f"No file matching modality '{key}' (tried tags {tags}) in {folder}\n"
                    f"Files present:\n  {found_names}"
                )
            found[key] = match
        return found

    @staticmethod
    def _load_and_stack(paths: dict):
        """
        Loads the 4 modalities, returns:
            display_volume: normalized FLAIR volume for background display (D,H,W)
            model_input:    torch tensor [1, 4, D, H, W], z-score normalized per channel
            affine:         nibabel affine of the FLAIR image, for voxel-size calculation
        Assumes all 4 modalities are already co-registered to the same grid
        (standard for BraTS-style preprocessed data). If yours aren't, resample
        them to a common grid before this step.
        """
        arrays = {}
        affine = None
        for key in MODALITIES:
            img = nib.load(str(paths[key]))
            data = img.get_fdata().astype(np.float32)
            if key == "t2f":
                affine = img.affine
            arrays[key] = data

        shape = arrays["t2f"].shape
        for key, arr in arrays.items():
            if arr.shape != shape:
                raise ValueError(
                    f"Modality '{key}' has shape {arr.shape}, expected {shape} "
                    f"(all 4 modalities must be co-registered to the same grid)"
                )

        # Display volume: normalized FLAIR, 0-1 range
        flair = arrays["t2f"]
        d_min, d_max = float(flair.min()), float(flair.max())
        display_volume = (flair - d_min) / (d_max - d_min) if d_max > d_min else flair

        # Standardize to a fixed shape so every case displays at consistent
        # size/proportions regardless of native resolution. The mask that
        # comes back from the model will be upsampled to this same
        # standardized shape (see on_run_segmentation/_run_model), so it
        # stays pixel-aligned with this display volume automatically.
        display_volume, affine = _standardize_volume(display_volume, affine)

        # Same per-volume z-score the training dataset uses, stacked in the
        # same channel order as config.MODALITIES.
        channels = [_normalize(arrays[key]) for key in MODALITIES]
        stacked = np.stack(channels, axis=0)  # [4, D, H, W]
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()  # [1, 4, D, H, W]
        return display_volume, tensor, affine

    def _get_model(self):
        if getattr(self, "_model", None) is not None:
            return self._model

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(MODEL_PATH, map_location=device)

        model = MultiTaskUNet3D(in_channels=4, out_channels=1)
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            model.load_segmentation_weights(state_dict)

        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        if self._survival_stats is None:
            self._survival_stats = load_survival_stats()
        return model

    def _run_model(self, model, model_input: torch.Tensor, target_shape: tuple):
        """
        Resize the input to INFERENCE_SIZE, run both heads, upsample the
        probability map back to native resolution, and return
        (mask, probs, days). `probs` is the raw sigmoid output (0-1 per
        voxel) — kept rather than discarded after thresholding, so the UI
        can show actual model confidence instead of a hard yes/no mask.
        """
        device = self._device
        x = model_input.to(device)

        resized = torch.nn.functional.interpolate(
            x, size=INFERENCE_SIZE, mode="trilinear", align_corners=False
        )

        with torch.no_grad():
            outputs = run_model(model, resized)
            logits, _ = split_model_outputs(outputs)
            probs = torch.sigmoid(logits)

        probs_native = torch.nn.functional.interpolate(
            probs, size=target_shape, mode="trilinear", align_corners=False
        )
        probs_np = probs_native.squeeze(0).squeeze(0).cpu().numpy()
        mask = (probs_np > 0.5).astype(np.float32)

        predicted_days = None
        if self._survival_stats is not None:
            try:
                predicted_days = predict_survival_days(
                    model,
                    resized.squeeze(0),
                    device,
                    self._survival_stats,
                )
            except AttributeError:
                predicted_days = None

        return mask, probs_np, predicted_days

    @staticmethod
    def _voxel_volume_cm3(affine: np.ndarray) -> float:
        """Voxel volume in cm^3, from the NIfTI affine's voxel dimensions (mm)."""
        voxel_dims_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_volume_mm3 = float(np.prod(voxel_dims_mm))
        return voxel_volume_mm3 / 1000.0

    def _compute_report(
        self,
        mask: np.ndarray,
        probs: np.ndarray,
        display_volume: np.ndarray,
        affine: np.ndarray,
        predicted_days: Optional[float],
        components: list,
    ) -> dict:
        """
        Builds the metrics a single whole-tumor binary mask can honestly
        support. This model predicts one class only, so tumor core and
        edema are NOT included here — those need a retrained multi-class
        (WT/TC/ET) model, since they can't be derived from a single mask
        after the fact. Predicted survival is included since the
        multi-task model already computes it separately.

        `avg_confidence` is the mean of the model's raw sigmoid output
        (probs) across only the voxels included in the mask — it reflects
        how sure the model was about the region it flagged, not an
        independently validated or calibrated probability. `components`
        is the same per-tumor list from _label_tumor_components(), passed
        in rather than recomputed here, since the caller already needs it
        for the 2D heatmap coloring.
        """
        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        total_voxels = int(mask.sum())
        total_cm3 = total_voxels * voxel_vol_cm3
        n_components = len(components)

        if total_voxels > 0:
            centroid = np.array(ndi_center_of_mass(mask))
            shape = np.array(mask.shape)
            rel = (centroid - shape / 2.0) / (shape / 2.0)  # -1..1 per axis

            # Assumes BraTS-style axis ordering (0=L-R, 1=A-P, 2=S-I), same
            # assumption already used for the sagittal/coronal/axial panels.
            # This isn't verified against the file's own orientation, so
            # it's a rough geometric description relative to the volume's
            # center, not a clinically confirmed laterality reading.
            lr = "right" if rel[0] > 0.1 else ("left" if rel[0] < -0.1 else "midline")
            ap = "posterior" if rel[1] > 0.1 else ("anterior" if rel[1] < -0.1 else "central")
            si = "superior" if rel[2] > 0.1 else ("inferior" if rel[2] < -0.1 else "central")
            location = f"{lr}, {ap}, {si} (approximate)"
            avg_confidence = float(probs[mask > 0.5].mean())
        else:
            location = "n/a — no tumor voxels detected"
            avg_confidence = float("nan")

        try:
            brain_level = brain_surface_level(display_volume)
            brain_voxels = int((display_volume > brain_level).sum())
            percent_of_brain = (total_voxels / brain_voxels * 100.0) if brain_voxels > 0 else float("nan")
        except ValueError:
            percent_of_brain = float("nan")

        return {
            "total_cm3": total_cm3,
            "n_components": n_components,
            "location": location,
            "percent_of_brain": percent_of_brain,
            "predicted_days": predicted_days,
            "avg_confidence": avg_confidence,
            "components": components,
        }

    def _update_report_panel(self, report: dict):
        self._current_report = report
        lines = [
            f"Total lesion volume: {report['total_cm3']:.2f} cm&sup3;",
            f"Lesion components: {report['n_components']}",
            f"Approx. location: {report['location']}",
        ]
        if not np.isnan(report["percent_of_brain"]):
            lines.append(f"% of brain volume: {report['percent_of_brain']:.1f}%")

        components = report.get("components") or []
        if components:
            # Per-tumor confidence, listed by the same color used in the
            # 2D heatmap — a colored dot next to each entry, not just a
            # color name, so the report and the image visually match at a
            # glance.
            lines.append("<br><b>Confidence by tumor:</b>")
            for comp in components:
                swatch = f"<span style='color:{comp['color_hex']};'>&#9679;</span>"
                lines.append(
                    f"{swatch} {comp['color_name']} (#{comp['rank']}): "
                    f"{comp['confidence'] * 100:.1f}% conf., {comp['volume_cm3']:.2f} cm&sup3;"
                )
        elif not np.isnan(report.get("avg_confidence", float("nan"))):
            lines.append(f"Avg. model confidence: {report['avg_confidence'] * 100:.1f}%")

        if report.get("predicted_days") is not None:
            lines.append(f"Predicted survival: {format_survival(report['predicted_days'])}")

        self.report_body_label.setTextFormat(Qt.RichText)
        self.report_body_label.setText("<br>".join(lines))
        self.export_report_btn.setEnabled(True)

    def on_export_report(self):
        if not self._current_report:
            return

        default_name = f"{self.current_folder.name if self.current_folder else 'patient'}_report.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Save PDF report", default_name, "PDF files (*.pdf)")
        if not path:
            return

        try:
            self._export_pdf(path, self._current_report)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to export report", str(exc))
            return

        QMessageBox.information(self, "Report exported", f"Saved to:\n{path}")

    def _export_pdf(self, path: str, report: dict):
        """
        Builds a one-page PDF summary using reportlab. Kept as a local
        import since this is the only place in the file that needs it —
        matches the "load heavy dependencies where they're used" pattern
        already used for the MONAI-adjacent model imports elsewhere.
        """
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(path, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=16)
        disclaimer_style = ParagraphStyle(
            "Disclaimer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#8a1c1c")
        )

        disclaimer = (
            "This report is generated by an automated research pipeline and is NOT a "
            "diagnostic or clinical report. It has not been reviewed by a radiologist. "
            "The current model predicts whole-tumor extent only \u2014 tumor core and edema "
            "sub-regions are not available without a retrained multi-class model. Location "
            "is a rough geometric estimate relative to the volume center, not a confirmed "
            "anatomical reading. Predicted survival is a model estimate, not a clinical "
            "prognosis. Model confidence is the average of the model's own raw output "
            "probability within the flagged region \u2014 it has not been independently "
            "calibrated or validated, and should not be read as a statistically precise "
            "likelihood."
        )

        story = [
            Paragraph("Brain Tumor Segmentation \u2014 Report Summary", title_style),
            Spacer(1, 4),
            Paragraph(disclaimer, disclaimer_style),
            Spacer(1, 14),
        ]

        patient_code = self.current_folder.name if self.current_folder else "unknown"
        meta_table = Table(
            [
                ["Patient / case folder", patient_code],
                ["Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")],
            ],
            colWidths=[2 * inch, 4 * inch],
        )
        meta_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 16))

        story.append(Paragraph("Metrics", styles["Heading2"]))
        percent_text = (
            f"{report['percent_of_brain']:.1f}%" if not np.isnan(report["percent_of_brain"]) else "n/a"
        )
        confidence_text = (
            f"{report['avg_confidence'] * 100:.1f}%"
            if not np.isnan(report.get("avg_confidence", float("nan")))
            else "n/a"
        )
        survival_text = (
            format_survival(report["predicted_days"])
            if report.get("predicted_days") is not None
            else "n/a"
        )
        metric_table = Table(
            [
                ["Metric", "Value"],
                ["Total lesion volume", f"{report['total_cm3']:.2f} cm\u00b3"],
                ["Lesion components", str(report["n_components"])],
                ["Approximate location", report["location"]],
                ["% of brain volume", percent_text],
                ["Avg. model confidence", confidence_text],
                ["Predicted survival", survival_text],
            ],
            colWidths=[2.5 * inch, 3 * inch],
        )
        metric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(metric_table)

        components = report.get("components") or []
        if components:
            story.append(Spacer(1, 16))
            story.append(Paragraph("Confidence by Tumor", styles["Heading2"]))
            story.append(Paragraph(
                "Colors distinguish separate lesions from each other for reference "
                "across this report and the app's image views \u2014 they are not a "
                "standardized clinical color scheme.",
                disclaimer_style,
            ))
            story.append(Spacer(1, 6))

            comp_rows = [["#", "Color", "Confidence", "Volume"]]
            comp_style_commands = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
            for row_index, comp in enumerate(components, start=1):
                comp_rows.append([
                    str(comp["rank"]),
                    comp["color_name"],
                    f"{comp['confidence'] * 100:.1f}%",
                    f"{comp['volume_cm3']:.2f} cm\u00b3",
                ])
                # Tint the Color cell with the tumor's actual assigned
                # color, so the table visually matches the app's heatmap.
                comp_style_commands.append(
                    ("BACKGROUND", (1, row_index), (1, row_index), colors.HexColor(comp["color_hex"]))
                )
                comp_style_commands.append(
                    ("TEXTCOLOR", (1, row_index), (1, row_index), colors.white)
                )

            comp_table = Table(comp_rows, colWidths=[0.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
            comp_table.setStyle(TableStyle(comp_style_commands))
            story.append(comp_table)

        doc.build(story)

    def on_enter_image(self):
        folder = QFileDialog.getExistingDirectory(self, "Select patient folder")
        if not folder:
            return
        self._load_folder(Path(folder))

    def _load_folder(self, folder_path: Path) -> bool:
        """
        Scans a patient folder for every .nii/.nii.gz file, populates the
        file selector dropdown with all of them, and previews a sensible
        default (a FLAIR-tagged file if present, else the first one found).
        Shared by "Enter image" and "Run segmentation" so both act on the
        same current case. Returns False (with a dialog shown) if the
        folder has no matching files.
        """
        nii_files = sorted(folder_path.glob("*.nii*"))
        if not nii_files:
            QMessageBox.warning(
                self, "No images found",
                f"No .nii or .nii.gz files found in:\n{folder_path}"
            )
            return False

        self.current_folder = folder_path
        self._nii_files = nii_files
        case_id = folder_path.name
        self.patients_record_panel.set_case(
            case_id, self._survival_table.get(case_id)
        )

        self.file_selector.blockSignals(True)
        self.file_selector.clear()
        self.file_selector.addItems([f.name for f in nii_files])
        self.file_selector.blockSignals(False)
        self.file_selector.setVisible(True)

        # Prefer previewing the FLAIR volume by default — it's the same
        # background volume Run segmentation displays results on, so the
        # preview matches what you'll see after running the model.
        default_index = 0
        for i, f in enumerate(nii_files):
            if re.search(r"[_-](t2f|flair)[_.]", f.name, re.IGNORECASE):
                default_index = i
                break

        self.file_selector.setCurrentIndex(default_index)
        self._load_and_display(nii_files[default_index])
        return True

    def on_file_selected(self, index: int):
        if 0 <= index < len(self._nii_files):
            self._load_and_display(self._nii_files[index])

    def _load_and_display(self, path: Path):
        try:
            volume, affine = self._load_volume(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load volume", str(exc))
            return

        self.sagittal_panel.set_volume(volume)
        self.axial_panel.set_volume(volume)
        self.coronal_panel.set_volume(volume)
        self._display_volume = volume
        self._display_affine = affine
        self._update_3d_views(volume, None, affine)

    @staticmethod
    def _load_volume(path: str):
        """
        Loads a NIfTI file and returns a normalized 3D numpy array ready for
        display (values roughly 0-1, resampled to STANDARD_DISPLAY_SHAPE)
        plus a matching affine for 3D voxel spacing.
        """
        img = nib.load(path)
        data = img.get_fdata()

        if data.ndim == 4:
            # Some NIfTI files carry a trailing singleton or modality dim;
            # take the first volume/channel so we always end up 3D.
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {data.shape}")

        data = data.astype(np.float32)
        d_min, d_max = float(data.min()), float(data.max())
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        return _standardize_volume(data, img.affine)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
