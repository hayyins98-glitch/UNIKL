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
import re
import sys
import tempfile
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
from scipy.ndimage import zoom
import plotly.graph_objects as go

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QSizePolicy, QSlider, QMessageBox,
    QProgressBar, QComboBox,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

from brain_tumor_seg.config import CHECKPOINT_DIR, MODALITIES, TARGET_SHAPE
from brain_tumor_seg.data.dataset import _normalize
from brain_tumor_seg.data.survival import load_survival_days, load_survival_stats
from brain_tumor_seg.evaluation import predict_survival_days
from brain_tumor_seg.models import MultiTaskUNet3D
from brain_tumor_seg.models.multitask import run_model, split_model_outputs
from brain_tumor_seg.visualization.survival import format_survival
from brain_tumor_seg.visualization.viewer3d import INTERACTION_CONFIG, show_volume_3d

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
QSlider::groove:horizontal {{
    height: 4px;
    background: {BORDER_DIM};
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
        self.volume = None   # grayscale background, 3D numpy array, 0-1 range
        self.mask = None     # optional binary mask, same shape as volume

        self.setFrameShape(QFrame.Box)
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

        self.figure = Figure(figsize=(3, 3))
        self.figure.patch.set_facecolor(BG_PANEL)
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setStyleSheet(f"background-color: {BG_PANEL};")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_PANEL)
        self.ax.axis("off")
        layout.addWidget(self.canvas, stretch=1)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider)

        self._base_title = title

    def set_volume(self, volume: np.ndarray):
        """volume: 3D numpy array (already normalized for display, 0-1 range)."""
        self.volume = volume
        self.mask = None  # new volume invalidates any previous mask overlay
        n_slices = volume.shape[self.axis]
        self.slider.setEnabled(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(n_slices - 1, 0))
        self.slider.setValue(n_slices // 2)
        self._draw_slice(n_slices // 2)

        self.setStyleSheet(_panel_style(ACCENT_TEAL))

    def set_mask(self, mask: np.ndarray):
        """mask: binary 3D numpy array, same shape as the current volume."""
        if self.volume is not None and mask.shape != self.volume.shape:
            raise ValueError(
                f"mask shape {mask.shape} does not match volume shape {self.volume.shape}"
            )
        self.mask = mask
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

        if self.mask is not None:
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
            self._layout.addWidget(self.web_view, stretch=1)

        tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        fig.write_html(
            tmp.name,
            include_plotlyjs=True,
            full_html=True,
            config=INTERACTION_CONFIG,
        )
        self._temp_files.append(tmp.name)
        self.web_view.load(QUrl.fromLocalFile(tmp.name))

        self.setStyleSheet(_panel_style(self.accent))

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
        self.status_label.setText("Running segmentation...")
        QApplication.processEvents()  # let the UI repaint before the model blocks

        try:
            mask, predicted_days = self._run_model(
                model, model_input, target_shape=display_volume.shape
            )
        except Exception as exc:
            self.busy_bar.hide()
            QMessageBox.critical(self, "Inference failed", str(exc))
            self.status_label.setText("")
            return

        self.sagittal_panel.set_volume(display_volume)
        self.axial_panel.set_volume(display_volume)
        self.coronal_panel.set_volume(display_volume)
        self.sagittal_panel.set_mask(mask)
        self.axial_panel.set_mask(mask)
        self.coronal_panel.set_mask(mask)

        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        tumor_cm3 = float(mask.sum()) * voxel_vol_cm3
        self.status_label.setText(f"Tumor volume: {tumor_cm3:.2f} cm\u00b3")
        self.patients_record_panel.set_predictions(predicted_days, tumor_cm3)

        self._update_3d_views(display_volume, mask, affine)
        self.busy_bar.hide()

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
        probability map back to native resolution, and return (mask, days).
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
        mask = (probs_native > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()

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

        return mask, predicted_days

    @staticmethod
    def _voxel_volume_cm3(affine: np.ndarray) -> float:
        """Voxel volume in cm^3, from the NIfTI affine's voxel dimensions (mm)."""
        voxel_dims_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_volume_mm3 = float(np.prod(voxel_dims_mm))
        return voxel_volume_mm3 / 1000.0

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
        display (values roughly 0-1) plus the affine for 3D voxel spacing.
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
        return data, img.affine


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
