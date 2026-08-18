"""
Standalone UI layout — matches the wireframe:

    [Enter image]        [ 2D sagittal ]        [   3D tumor    ]
    [Run segmentation]   [ 2D axial    ]        [               ]
    [Patients record]    [ 2D coronal  ]        [  3D brain view]
                                                  [               ]

Status:
  - 2D panels: WIRED — real slices, scroll sliders.
  - Run segmentation: WIRED — loads a 4-modality folder (BraTS or BraTS-PEDs
    naming), runs a 3D U-Net checkpoint, overlays the predicted tumor mask
    in red on the 2D views, and reports tumor volume in cm^3.
  - 3D panels: WIRED — marching-cubes tumor mesh and brain+tumor render,
    built after each segmentation run and shown via embedded Plotly (rotate/
    zoom/pan with the mouse).
  - Patient record: still a placeholder — next step.
  - Maximize: WIRED — hides the other panels/columns in place so the
    maximized panel fills the window. (No popup dialog — avoids a
    QWebEngineView reparenting quirk that broke centering for 3D panels.)

This file is self-contained: run it directly with `python ui_layout.py`.
It has no dependency on database.py / inference_worker.py / report_generator.py
from the earlier scaffold.

MODEL CHECKPOINT: no path to edit in this file anymore — the first time you
run segmentation, a file dialog asks you to pick your trained .pth file,
and the choice is remembered in a settings file in your user profile
(~/.brain_tumor_viewer/settings.json), independent of wherever this
project folder happens to live. If the saved checkpoint file is later
moved or deleted, it will prompt again automatically. The architecture
here (MONAI UNet, in_channels=4, out_channels=1) must match whatever
architecture that checkpoint was trained with, or loading will fail with
a state_dict mismatch error.
"""
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import torch
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, label as ndi_label, center_of_mass as ndi_center_of_mass
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
import plotly.graph_objects as go

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QFileDialog, QSizePolicy, QSlider, QMessageBox,
    QProgressBar, QComboBox
)
from PySide6.QtWebEngineWidgets import QWebEngineView

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
# Model configuration
# ---------------------------------------------------------------------------
# The checkpoint path lives in a small settings file in the user's profile
# (NOT next to this script), so it survives moving/renaming the project
# folder — which was breaking MODEL_PATH every time before. If it's unset
# or the saved file no longer exists, the app prompts once via a file
# dialog and remembers the answer for every future launch.
SETTINGS_DIR = Path.home() / ".brain_tumor_viewer"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"

INFERENCE_SIZE = (128, 128, 128)  # volume is resized to this for the model,
                                   # then the mask is resized back to native size


def _load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_settings(settings: dict):
    SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


class _ModelSelectionCancelled(Exception):
    """Raised when the user closes the checkpoint file dialog without
    picking anything — a cancel, not a real error, so callers should
    return quietly instead of showing an error dialog."""
    pass


class PanelBox(QFrame):
    """A bordered placeholder panel with a centered label — stands in for a
    patient record list or any other content area until that content is
    wired in."""

    def __init__(self, text: str, min_height: int = 150):
        super().__init__()
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet(_panel_style())
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(f"border: none; color: {TEXT_MUTED}; font-size: 13px;")
        layout.addWidget(label)


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
        fig.write_html(tmp.name, include_plotlyjs=True, full_html=True)
        self._temp_files.append(tmp.name)
        self.web_view.load(QUrl.fromLocalFile(tmp.name))

        self.setStyleSheet(_panel_style(self.accent))

    def set_maximized(self, is_max: bool):
        """Swaps the button glyph/tooltip; actual show/hide of sibling
        panels is handled by MainWindow.toggle_maximize()."""
        self.maximize_btn.setText("🗗" if is_max else "⛶")
        self.maximize_btn.setToolTip("Restore" if is_max else "Maximize")


def _mesh_trace(volume: np.ndarray, level: float, step_size: int, color: str,
                 opacity: float, name: str, spacing: tuple) -> Optional[go.Mesh3d]:
    """Marching-cubes isosurface as a Plotly Mesh3d, or None if level is out
    of range or the mesh comes out empty (e.g. an all-zero mask)."""
    if not (volume.min() < level < volume.max()):
        return None
    try:
        verts, faces, _, _ = marching_cubes(volume, level=level, step_size=step_size, spacing=spacing)
    except (ValueError, RuntimeError):
        return None
    if len(faces) == 0:
        return None
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name,
        lighting=dict(ambient=0.55, diffuse=0.8, specular=0.12, roughness=0.85),
    )


def build_tumor_figure(mask: np.ndarray, spacing: tuple) -> go.Figure:
    """Tumor mesh alone, for the small '3D tumor' panel."""
    trace = _mesh_trace(mask.astype(np.float32), 0.5, step_size=1,
                         color="rgb(216, 90, 48)", opacity=1.0, name="tumor", spacing=spacing)
    fig = go.Figure(data=[trace] if trace else [])
    fig.update_layout(
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    )
    return fig


def build_brain_figure(volume: np.ndarray, mask: np.ndarray, spacing: tuple) -> go.Figure:
    """Semi-transparent brain shell with the tumor mesh visible inside it,
    for the larger '3D brain view' panel."""
    smoothed = gaussian_filter(volume, sigma=1.0)
    brain_level = float(threshold_otsu(volume))
    brain = _mesh_trace(smoothed, brain_level, step_size=2,
                         color="rgb(188, 190, 200)", opacity=0.18, name="brain", spacing=spacing)
    tumor = _mesh_trace(mask.astype(np.float32), 0.5, step_size=1,
                         color="rgb(216, 90, 48)", opacity=1.0, name="tumor", spacing=spacing)

    traces = [t for t in (brain, tumor) if t is not None]
    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            camera=dict(eye=dict(x=1.55, y=-1.55, z=0.95), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
    )
    return fig


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

        # Small, quiet control: shows which checkpoint is active and lets
        # you pick a different one without waiting for it to go missing.
        self.checkpoint_label = QLabel(self._checkpoint_display_text())
        self.checkpoint_label.setWordWrap(True)
        self.checkpoint_label.setStyleSheet(
            f"font-size: 10px; color: {TEXT_MUTED}; text-decoration: underline;"
        )
        self.checkpoint_label.setCursor(Qt.PointingHandCursor)
        self.checkpoint_label.mousePressEvent = lambda _event: self.on_change_checkpoint()
        left.addWidget(self.checkpoint_label)

        self.busy_bar = QProgressBar()
        self.busy_bar.setRange(0, 0)  # indeterminate — pulses while active
        self.busy_bar.setTextVisible(False)
        self.busy_bar.hide()
        left.addWidget(self.busy_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"font-size: 11px; color: {TEXT_MUTED};")
        left.addWidget(self.status_label)

        self.patients_record_panel = PanelBox("Patients record", min_height=150)
        left.addWidget(self.patients_record_panel)

        # Report summary — built only from what a single whole-tumor mask
        # can honestly support (see _compute_report). Left-aligned body
        # text, unlike the centered titles on the 2D/3D panels, since this
        # is read top-to-bottom like a small document rather than glanced
        # at like a chart title.
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
        except _ModelSelectionCancelled:
            return  # user closed the checkpoint picker — not an error, just stop quietly
        except Exception as exc:
            QMessageBox.critical(
                self, "Failed to load model",
                f"{exc}\n\nCheckpoint path:\n{getattr(self, '_model_path', '(not set)')}"
            )
            return

        self.busy_bar.show()
        self.status_label.setText("Running segmentation...")
        QApplication.processEvents()  # let the UI repaint before the model blocks

        try:
            mask = self._run_model(model, model_input, target_shape=display_volume.shape)
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

        self.status_label.setText("Segmentation complete.")

        report = self._compute_report(mask, display_volume, affine)
        self._update_report_panel(report)

        self._update_3d_views(display_volume, mask, affine)
        self.busy_bar.hide()

    def _update_3d_views(self, display_volume: np.ndarray, mask: np.ndarray, affine: np.ndarray):
        """
        Builds the two 3D Plotly figures (tumor-only, and brain+tumor) and
        loads them into the right-column panels. Runs synchronously on the
        main thread — marching cubes on a full-resolution volume can take a
        couple seconds, so the UI will briefly pause here. Fine for a single
        case; if this becomes annoying, move it to a background thread like
        the model inference eventually should be too.
        """
        spacing = tuple(np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))

        if mask.sum() == 0:
            self.status_label.setText(self.status_label.text() + "  (no tumor voxels — skipping 3D render)")
            return

        try:
            tumor_fig = build_tumor_figure(mask, spacing)
            self.tumor_3d_panel.set_figure(tumor_fig)

            brain_fig = build_brain_figure(display_volume, mask, spacing)
            self.brain_3d_panel.set_figure(brain_fig)
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
            "t1": ["t1n", "t1"],
            "t1ce": ["t1c", "t1ce"],
            "t2": ["t2w", "t2"],
            "flair": ["t2f", "flair"],
        }
        candidates = list(folder.glob("*.nii*"))
        found = {}
        for key, tags in tag_options.items():
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
        for key in ["t1", "t1ce", "t2", "flair"]:
            img = nib.load(str(paths[key]))
            data = img.get_fdata().astype(np.float32)
            if key == "flair":
                affine = img.affine
            arrays[key] = data

        shape = arrays["flair"].shape
        for key, arr in arrays.items():
            if arr.shape != shape:
                raise ValueError(
                    f"Modality '{key}' has shape {arr.shape}, expected {shape} "
                    f"(all 4 modalities must be co-registered to the same grid)"
                )

        # Display volume: normalized FLAIR, 0-1 range
        flair = arrays["flair"]
        d_min, d_max = float(flair.min()), float(flair.max())
        display_volume = (flair - d_min) / (d_max - d_min) if d_max > d_min else flair

        # Model input: z-score each modality over its non-zero voxels, stack as channels
        channels = []
        for key in ["t1", "t1ce", "t2", "flair"]:
            arr = arrays[key]
            nonzero = arr[arr > 0]
            mean = float(nonzero.mean()) if nonzero.size else 0.0
            std = float(nonzero.std()) if nonzero.size else 1.0
            channels.append((arr - mean) / (std + 1e-8))

        stacked = np.stack(channels, axis=0)  # [4, D, H, W]
        tensor = torch.from_numpy(stacked).unsqueeze(0).float()  # [1, 4, D, H, W]
        return display_volume, tensor, affine

    @staticmethod
    def _checkpoint_display_text() -> str:
        settings = _load_settings()
        saved = settings.get("model_path")
        if saved and Path(saved).exists():
            return f"Model: {Path(saved).name} (change)"
        return "Model: not set (click to choose)"

    def on_change_checkpoint(self):
        """Lets you pick a different checkpoint on demand, rather than only
        being prompted when the saved one goes missing."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained model checkpoint (.pth)", "", "PyTorch checkpoint (*.pth)"
        )
        if not path:
            return
        settings = _load_settings()
        settings["model_path"] = path
        _save_settings(settings)
        self._model = None  # force _get_model() to reload from the new path next run
        self.checkpoint_label.setText(self._checkpoint_display_text())

    def _resolve_model_path(self) -> Path:
        """
        Reads the checkpoint path from the persistent settings file. If it's
        missing or the saved file no longer exists (moved/deleted since last
        run), prompts once via a file dialog and saves the answer so this
        only happens again if the checkpoint actually moves again.
        Raises _ModelSelectionCancelled if the user closes the dialog
        without picking anything — that's a cancel, not an error.
        """
        settings = _load_settings()
        saved = settings.get("model_path")
        if saved and Path(saved).exists():
            return Path(saved)

        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained model checkpoint (.pth)", "", "PyTorch checkpoint (*.pth)"
        )
        if not path:
            raise _ModelSelectionCancelled()

        settings["model_path"] = path
        _save_settings(settings)
        return Path(path)

    def _get_model(self):
        if getattr(self, "_model", None) is not None:
            return self._model

        model_path = self._resolve_model_path()

        from monai.networks.nets import UNet
        from monai.networks.layers import Norm

        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=1,
            channels=(32, 64, 128, 256, 320),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.INSTANCE,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._model_path = model_path
        self.checkpoint_label.setText(self._checkpoint_display_text())
        return model

    def _run_model(self, model, model_input: torch.Tensor, target_shape: tuple) -> np.ndarray:
        """
        Resizes the input to INFERENCE_SIZE, runs the model, upsamples the
        sigmoid probability map back to target_shape with trilinear
        interpolation (smoother than nearest for a probability field), then
        thresholds at 0.5 to get the final binary mask at native resolution.
        """
        device = self._device
        x = model_input.to(device)
        native_shape = x.shape[2:]

        resized = torch.nn.functional.interpolate(
            x, size=INFERENCE_SIZE, mode="trilinear", align_corners=False
        )

        with torch.no_grad():
            logits = model(resized)
            probs = torch.sigmoid(logits)

        probs_native = torch.nn.functional.interpolate(
            probs, size=target_shape, mode="trilinear", align_corners=False
        )
        mask = (probs_native > 0.5).float().squeeze(0).squeeze(0).cpu().numpy()
        return mask

    @staticmethod
    def _voxel_volume_cm3(affine: np.ndarray) -> float:
        """Voxel volume in cm^3, from the NIfTI affine's voxel dimensions (mm)."""
        voxel_dims_mm = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        voxel_volume_mm3 = float(np.prod(voxel_dims_mm))
        return voxel_volume_mm3 / 1000.0

    def _compute_report(self, mask: np.ndarray, display_volume: np.ndarray, affine: np.ndarray) -> dict:
        """
        Builds the metrics that a single whole-tumor binary mask can
        honestly support. This model predicts one class only, so tumor
        core and edema are NOT included here — those need a retrained
        multi-class (WT/TC/ET) model, since they can't be derived from a
        single mask after the fact.
        """
        voxel_vol_cm3 = self._voxel_volume_cm3(affine)
        total_voxels = int(mask.sum())
        total_cm3 = total_voxels * voxel_vol_cm3

        _, n_components = ndi_label(mask > 0.5)

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
        else:
            location = "n/a — no tumor voxels detected"

        try:
            brain_level = float(threshold_otsu(display_volume))
            brain_voxels = int((display_volume > brain_level).sum())
            percent_of_brain = (total_voxels / brain_voxels * 100.0) if brain_voxels > 0 else float("nan")
        except ValueError:
            percent_of_brain = float("nan")

        return {
            "total_cm3": total_cm3,
            "n_components": int(n_components),
            "location": location,
            "percent_of_brain": percent_of_brain,
        }

    def _update_report_panel(self, report: dict):
        self._current_report = report
        lines = [
            f"Total lesion volume: {report['total_cm3']:.2f} cm\u00b3",
            f"Lesion components: {report['n_components']}",
            f"Approx. location: {report['location']}",
        ]
        if not np.isnan(report["percent_of_brain"]):
            lines.append(f"% of brain volume: {report['percent_of_brain']:.1f}%")
        self.report_body_label.setText("\n".join(lines))
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
        already used for the MONAI import in _get_model().
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
            "anatomical reading."
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
        metric_table = Table(
            [
                ["Metric", "Value"],
                ["Total lesion volume", f"{report['total_cm3']:.2f} cm\u00b3"],
                ["Lesion components", str(report["n_components"])],
                ["Approximate location", report["location"]],
                ["% of brain volume", percent_text],
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
            volume = self._load_volume(str(path))
        except Exception as exc:
            QMessageBox.critical(self, "Failed to load volume", str(exc))
            return

        self.sagittal_panel.set_volume(volume)
        self.axial_panel.set_volume(volume)
        self.coronal_panel.set_volume(volume)

    @staticmethod
    def _load_volume(path: str) -> np.ndarray:
        """
        Loads a NIfTI file and returns a normalized 3D numpy array ready for
        display (values roughly 0-1, so matplotlib's default grayscale scale
        looks sensible regardless of the raw MRI intensity range).
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
        return data


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
