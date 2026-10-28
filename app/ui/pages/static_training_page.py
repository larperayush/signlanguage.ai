"""
Static Sign Training & Camera Data Collection Workspace.
Combines live camera preview, landmark coordinate collection controls,
sample review buffer, collection guide, train_model.py hyperparameter training,
progress tracking, and model artifact registration.
"""
import os
import time
from typing import Optional, List
import math
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, QComboBox,
    QScrollArea, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedLayout, QSizePolicy, QApplication, QLineEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, Slot, QEvent

from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager, DeviceInfo
from app.core.interfaces import (
    BackendRegistry, TrainingBackendInterface, StaticCollectionInterface,
    TrainingProgress, SampleRecord
)
from app.core.dataset_registry import DatasetRegistry
from app.core.model_registry import ModelRegistry
from app.core.backend_manager import BackendManager
from app.core.landmark_service import LandmarkService
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import SectionHeader, StatusPill
from app.ui.components.searchable_combo import SearchableCameraComboBox
from app.ui.pages.predict_page import CameraPreviewWidget, CameraSettingsDialog


STATIC_SIGN_OPTIONS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Space", "Delete"
]


class StaticTrainingPage(QWidget):
    """
    Camera-based static sign data collection and model training workstation.
    Reuses the centralized CameraManager for live capture.
    """
    def __init__(self, camera_manager: CameraManager = None, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._training_backend: TrainingBackendInterface = BackendRegistry.get_training_backend()
        self._collection_backend: StaticCollectionInterface = BackendRegistry.get_static_collection_backend()
        self._dataset_registry = DatasetRegistry.get_instance()
        self._model_registry = ModelRegistry.get_instance()
        self._backend_manager = BackendManager.get_instance()
        self._is_collecting = False
        self._session_sample_count = 0

        palette = ThemeManager.get_palette()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        content_widget = QWidget()
        content_widget.setObjectName("contentContainer")
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(24, 20, 24, 28)
        main_layout.setSpacing(20)

        # Header
        header = SectionHeader(
            "Static Training & Collection Workspace",
            "Capture single-frame 3D hand landmark coordinates, curate training samples, and train Random Forest models."
        )
        main_layout.addWidget(header)

        # =====================================================================
        # TOP SPLIT: CAMERA VIEWPORT (LEFT) & DATA COLLECTION CONTROLS (RIGHT)
        # =====================================================================
        top_split = QHBoxLayout()
        top_split.setSpacing(18)

        # ---------------------------------------------------------------------
        # LEFT: CAMERA VIEWPORT & DEVICE CONTROLS
        # ---------------------------------------------------------------------
        cam_col = QVBoxLayout()
        cam_col.setSpacing(10)

        cam_header = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA VIEWPORT")
        cam_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        self.cam_status_pill = StatusPill("Camera Inactive", "neutral")
        cam_header.addWidget(cam_title)
        cam_header.addStretch()
        cam_header.addWidget(self.cam_status_pill)
        cam_col.addLayout(cam_header)

        # Camera Video Box
        self.video_container = QFrame()
        self.video_container.setProperty("class", "cardPanel")
        self.video_container.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.video_bg};
                border: 1px solid {palette.border_default};
                border-radius: 8px;
            }}
        """)
        self.video_container.setMinimumSize(460, 310)
        self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        video_stack = QStackedLayout(self.video_container)
        video_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)

        # Native Preview Widget
        self.video_widget = CameraPreviewWidget()
        self.video_widget.setStyleSheet("background-color: transparent;")
        video_stack.addWidget(self.video_widget)

        # Standby Overlay
        self.standby_overlay = QWidget()
        overlay_layout = QVBoxLayout(self.standby_overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.setSpacing(8)

        standby_icon = QLabel()
        standby_icon.setPixmap(get_pixmap("camera", palette.text_muted, size=40))
        standby_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_title = QLabel("Workspace Camera Standby")
        standby_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #FFFFFF;")
        standby_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_desc = QLabel("Click 'Start Camera' below to initialize video capture for sample recording.")
        standby_desc.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")
        standby_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        overlay_layout.addWidget(standby_icon)
        overlay_layout.addWidget(standby_title)
        overlay_layout.addWidget(standby_desc)
        video_stack.addWidget(self.standby_overlay)

        cam_col.addWidget(self.video_container, 1)

        # Camera Device Bar
        cam_bar = QFrame()
        cam_bar.setProperty("class", "cardPanelElevated")
        cam_bar_layout = QHBoxLayout(cam_bar)
        cam_bar_layout.setContentsMargins(10, 8, 10, 8)
        cam_bar_layout.setSpacing(8)

        # Camera selector
        self.cam_combo = SearchableCameraComboBox()
        self.cam_combo.setMinimumWidth(180)
        if self._camera_manager:
            self.cam_combo.device_chosen.connect(self._camera_manager.select_device)
        cam_bar_layout.addWidget(self.cam_combo)

        # Refresh
        self.btn_refresh_cam = QPushButton()
        self.btn_refresh_cam.setProperty("class", "secondaryBtn")
        self.btn_refresh_cam.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        self.btn_refresh_cam.setToolTip("Refresh camera device list")
        self.btn_refresh_cam.clicked.connect(self._refresh_cameras)
        cam_bar_layout.addWidget(self.btn_refresh_cam)

        # Start / Stop Buttons
        self.btn_start_cam = QPushButton("Start Camera")
        self.btn_start_cam.setProperty("class", "primaryBtn")
        self.btn_start_cam.setIcon(get_icon("play", "#FFFFFF", size=13))
        self.btn_start_cam.clicked.connect(self._start_camera)
        cam_bar_layout.addWidget(self.btn_start_cam)

        self.btn_stop_cam = QPushButton("Stop")
        self.btn_stop_cam.setProperty("class", "secondaryBtn")
        self.btn_stop_cam.setIcon(get_icon("stop", palette.text_secondary, size=13))
        self.btn_stop_cam.setEnabled(False)
        self.btn_stop_cam.clicked.connect(self._stop_camera)
        cam_bar_layout.addWidget(self.btn_stop_cam)

        # Settings
        self.btn_cam_settings = QPushButton("Settings")
        self.btn_cam_settings.setProperty("class", "secondaryBtn")
        self.btn_cam_settings.setIcon(get_icon("sliders", palette.text_secondary, palette.accent, size=13))
        self.btn_cam_settings.clicked.connect(self._open_cam_settings)
        cam_bar_layout.addWidget(self.btn_cam_settings)

        cam_bar_layout.addStretch()

        self.lbl_cam_fps = QLabel("0.0 FPS")
        self.lbl_cam_fps.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_muted};")
        cam_bar_layout.addWidget(self.lbl_cam_fps)

        cam_col.addWidget(cam_bar)
        top_split.addLayout(cam_col, 5)

        # ---------------------------------------------------------------------
        # RIGHT: SAMPLE COLLECTION & REVIEW PANEL
        # ---------------------------------------------------------------------
        collect_panel = QFrame()
        collect_panel.setProperty("class", "cardPanel")
        collect_layout = QVBoxLayout(collect_panel)
        collect_layout.setContentsMargins(18, 16, 18, 16)
        collect_layout.setSpacing(12)

        c_header = QHBoxLayout()
        c_title = QLabel("DATA COLLECTION WORKSPACE")
        c_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        self.collect_status_pill = StatusPill("Backend Disconnected", "neutral")
        c_header.addWidget(c_title)
        c_header.addStretch()
        c_header.addWidget(self.collect_status_pill)
        collect_layout.addLayout(c_header)

        # Target Sign Selector
        sign_row = QHBoxLayout()
        sign_row.setSpacing(10)
        sign_row.addWidget(QLabel("Target Sign Label:"))

        self.combo_sign = QComboBox()
        self.combo_sign.setMinimumWidth(120)
        self.combo_sign.addItems(STATIC_SIGN_OPTIONS)
        sign_row.addWidget(self.combo_sign)
        sign_row.addStretch()

        self.lbl_active_sign = QLabel("Sign: A")
        self.lbl_active_sign.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {palette.accent};")
        self.combo_sign.currentTextChanged.connect(lambda s: self.lbl_active_sign.setText(f"Sign: {s}"))
        sign_row.addWidget(self.lbl_active_sign)

        collect_layout.addLayout(sign_row)

        # Collection Action Buttons
        act_row = QHBoxLayout()
        act_row.setSpacing(8)

        self.btn_start_collect = QPushButton("Start Collection")
        self.btn_start_collect.setProperty("class", "primaryBtn")
        self.btn_start_collect.setIcon(get_icon("play", "#FFFFFF", size=13))
        self.btn_start_collect.clicked.connect(self._start_collection)

        self.btn_capture_sample = QPushButton("Capture Sample (Space)")
        self.btn_capture_sample.setProperty("class", "primaryBtn")
        self.btn_capture_sample.setIcon(get_icon("camera", "#FFFFFF", size=13))
        self.btn_capture_sample.setEnabled(False)
        self.btn_capture_sample.setToolTip("Capture one sample using MediaPipe landmarks (or press Spacebar)")
        self.btn_capture_sample.clicked.connect(self._capture_sample_action)

        lbl_space_badge = QLabel("SPACE")
        lbl_space_badge.setStyleSheet(f"""
            background-color: {palette.bg_card_elevated};
            color: {palette.accent};
            border: 1px solid {palette.accent};
            border-radius: 4px;
            padding: 3px 6px;
            font-size: 10px;
            font-weight: 700;
        """)
        lbl_space_badge.setToolTip("Press Spacebar anywhere in the workspace while collection is active")

        self.btn_stop_collect = QPushButton("Stop Collection")
        self.btn_stop_collect.setProperty("class", "secondaryBtn")
        self.btn_stop_collect.setIcon(get_icon("stop", palette.text_secondary, size=13))
        self.btn_stop_collect.setEnabled(False)
        self.btn_stop_collect.clicked.connect(self._stop_collection)

        act_row.addWidget(self.btn_start_collect)
        act_row.addWidget(self.btn_capture_sample)
        act_row.addWidget(lbl_space_badge)
        act_row.addWidget(self.btn_stop_collect)
        collect_layout.addLayout(act_row)

        # Sample Count Metric & Last Capture Toast
        metric_box = QFrame()
        metric_box.setProperty("class", "cardPanelElevated")
        m_layout = QVBoxLayout(metric_box)
        m_layout.setContentsMargins(12, 8, 12, 8)
        m_layout.setSpacing(4)

        cnt_row = QHBoxLayout()
        cnt_row.addWidget(QLabel("Collected Samples (This Session):"))
        cnt_row.addStretch()
        self.lbl_sample_count = QLabel("0 recorded")
        self.lbl_sample_count.setStyleSheet(f"font-weight: 700; color: {palette.text_primary};")
        cnt_row.addWidget(self.lbl_sample_count)
        m_layout.addLayout(cnt_row)

        self.lbl_last_capture = QLabel("No samples captured this session")
        self.lbl_last_capture.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")
        m_layout.addWidget(self.lbl_last_capture)

        collect_layout.addWidget(metric_box)

        # Review Buffer Table
        review_title = QLabel("Candidate Sample Buffer (Review & Commit):")
        review_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_secondary};")
        collect_layout.addWidget(review_title)

        self.sample_table = QTableWidget(0, 4)
        self.sample_table.setHorizontalHeaderLabels(["ID", "Sign", "Landmarks", "Status"])
        self.sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.sample_table.setMinimumHeight(110)
        self.sample_table.setMaximumHeight(140)
        collect_layout.addWidget(self.sample_table)

        # Accept / Discard Controls
        review_actions = QHBoxLayout()
        review_actions.setSpacing(10)

        self.btn_accept_sample = QPushButton(" Accept Sample")
        self.btn_accept_sample.setProperty("class", "secondaryBtn")
        self.btn_accept_sample.setIcon(get_icon("check", palette.success, size=13))
        self.btn_accept_sample.clicked.connect(self._accept_sample)

        self.btn_discard_sample = QPushButton(" Discard")
        self.btn_discard_sample.setProperty("class", "secondaryBtn")
        self.btn_discard_sample.setIcon(get_icon("trash", palette.danger, size=13))
        self.btn_discard_sample.clicked.connect(self._discard_sample)

        review_actions.addWidget(self.btn_accept_sample)
        review_actions.addWidget(self.btn_discard_sample)
        review_actions.addStretch()
        collect_layout.addLayout(review_actions)

        # Collection Guide
        guide_box = QFrame()
        guide_box.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_card_elevated};
                border: 1px solid {palette.border_subtle};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        g_layout = QVBoxLayout(guide_box)
        g_layout.setContentsMargins(12, 10, 12, 10)
        g_layout.setSpacing(4)

        g_title = QLabel("SAMPLE DIVERSITY COLLECTION GUIDE")
        g_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        g_layout.addWidget(g_title)

        tips = [
            "1. Natural Hand Variations: Vary finger tension and natural hand postures.",
            "2. Distance Variations: Capture samples from 0.4m, 0.7m, and 1.0m from lens.",
            "3. Angle Variations: Tilt hand ±15° in pitch and yaw for spatial invariance.",
            "4. Lighting Variations: Record under ambient daylight and direct desk lamps."
        ]
        for tip in tips:
            lbl_tip = QLabel(tip)
            lbl_tip.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary};")
            lbl_tip.setWordWrap(True)
            g_layout.addWidget(lbl_tip)

        collect_layout.addWidget(guide_box)

        top_split.addWidget(collect_panel, 4)
        main_layout.addLayout(top_split)

        # =====================================================================
        # BOTTOM SECTION: DATASET INSPECTION & MODEL TRAINING CONTROLS
        # =====================================================================
        bottom_frame = QFrame()
        bottom_frame.setProperty("class", "cardPanel")
        b_layout = QVBoxLayout(bottom_frame)
        b_layout.setContentsMargins(20, 18, 20, 18)
        b_layout.setSpacing(16)

        b_header = QHBoxLayout()
        b_title = QLabel("MODEL TRAINING WORKSPACE (train_model.py)")
        b_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        self.training_status_pill = StatusPill("Backend Disconnected", "neutral")
        b_header.addWidget(b_title)
        b_header.addStretch()
        b_header.addWidget(self.training_status_pill)
        b_layout.addLayout(b_header)

        # Active Dataset Inspector Row
        ds_grid = QGridLayout()
        ds_grid.setSpacing(10)

        self.lbl_ds_name = QLabel("Loading active dataset...")
        self.lbl_ds_name.setStyleSheet("font-weight: 600;")
        self.lbl_ds_samples = QLabel("—")
        self.lbl_ds_dims = QLabel("63 coordinates (21 joints × 3D)")

        ds_grid.addWidget(QLabel("Target Training Dataset:"), 0, 0)
        ds_grid.addWidget(self.lbl_ds_name, 0, 1)
        ds_grid.addWidget(QLabel("Recorded Samples:"), 0, 2)
        ds_grid.addWidget(self.lbl_ds_samples, 0, 3)

        b_layout.addLayout(ds_grid)

        # Training Hyperparameters Grid (mirroring train_model.py)
        param_grid = QGridLayout()
        param_grid.setSpacing(12)

        param_grid.addWidget(QLabel("Estimators (n_estimators):"), 0, 0)
        self.spin_estimators = QSpinBox()
        self.spin_estimators.setRange(10, 500)
        self.spin_estimators.setValue(100)
        param_grid.addWidget(self.spin_estimators, 0, 1)

        param_grid.addWidget(QLabel("Test Split Ratio:"), 0, 2)
        self.spin_test_size = QDoubleSpinBox()
        self.spin_test_size.setRange(0.05, 0.50)
        self.spin_test_size.setSingleStep(0.05)
        self.spin_test_size.setValue(0.20)
        param_grid.addWidget(self.spin_test_size, 0, 3)

        param_grid.addWidget(QLabel("Random Seed (random_state):"), 1, 0)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 9999)
        self.spin_seed.setValue(42)
        param_grid.addWidget(self.spin_seed, 1, 1)

        param_grid.addWidget(QLabel("Criterion:"), 1, 2)
        self.combo_criterion = QComboBox()
        self.combo_criterion.addItems(["gini", "entropy", "log_loss"])
        param_grid.addWidget(self.combo_criterion, 1, 3)

        b_layout.addLayout(param_grid)

        # Training Start/Stop Buttons & Progress Bar
        train_ctrl_row = QHBoxLayout()
        train_ctrl_row.setSpacing(12)

        self.btn_start_train = QPushButton("Start Model Training")
        self.btn_start_train.setProperty("class", "primaryBtn")
        self.btn_start_train.setIcon(get_icon("play", "#FFFFFF", size=13))
        self.btn_start_train.clicked.connect(self._start_training)

        self.btn_stop_train = QPushButton("Interrupt Training")
        self.btn_stop_train.setProperty("class", "secondaryBtn")
        self.btn_stop_train.setIcon(get_icon("stop", palette.text_secondary, size=13))
        self.btn_stop_train.setEnabled(False)
        self.btn_stop_train.clicked.connect(self._stop_training)

        train_ctrl_row.addWidget(self.btn_start_train)
        train_ctrl_row.addWidget(self.btn_stop_train)
        train_ctrl_row.addStretch()

        b_layout.addLayout(train_ctrl_row)

        # Progress Bar
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        self.train_progress.setValue(0)
        self.train_progress.setTextVisible(True)
        b_layout.addWidget(self.train_progress)

        # Activity Log Console
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(100)
        self.log_console.setMaximumHeight(140)
        self.log_console.setStyleSheet(f"""
            QTextEdit {{
                background-color: {palette.code_bg};
                font-family: monospace;
                font-size: 11px;
                color: {palette.text_secondary};
                border: 1px solid {palette.border_subtle};
                border-radius: 4px;
            }}
        """)
        self.log_console.append("[System] Static Training Workspace ready.")
        self.log_console.append("[Backend Status] ML backend adapter not linked. Integration contract defined in app/core/interfaces.py.")
        b_layout.addWidget(self.log_console)

        # Register New Model Version Row
        model_reg_row = QHBoxLayout()
        model_reg_row.setSpacing(10)
        model_reg_row.addWidget(QLabel("Register Model Version:"))

        self.txt_model_version = QLabel("asl_classifier_latest.joblib")
        self.txt_model_version.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {palette.text_primary};")
        model_reg_row.addWidget(self.txt_model_version)
        model_reg_row.addStretch()

        self.btn_save_model = QPushButton("Register Model Artifact")
        self.btn_save_model.setProperty("class", "secondaryBtn")
        self.btn_save_model.setIcon(get_icon("box", palette.text_secondary, palette.accent, size=13))
        self.btn_save_model.clicked.connect(self._register_model_version)
        model_reg_row.addWidget(self.btn_save_model)

        b_layout.addLayout(model_reg_row)
        main_layout.addWidget(bottom_frame)

        main_layout.addStretch()
        scroll.setWidget(content_widget)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

        # Connect Backend Signals
        self._training_backend.log_emitted.connect(self._append_log)
        self._training_backend.progress_updated.connect(self._on_training_progress)
        self._training_backend.training_completed.connect(self._on_training_completed)
        self._collection_backend.sample_collected.connect(self._on_sample_collected)
        self._collection_backend.count_updated.connect(self._on_sample_count_updated)

        self._backend_manager.status_changed.connect(self._on_backend_manager_status)
        self._update_backend_state()

        # Connect Camera Manager Signals
        if self._camera_manager:
            self._camera_manager.devices_changed.connect(self._on_devices_updated)
            self._camera_manager.status_changed.connect(self._on_camera_status)
            self._camera_manager.preview_started.connect(self._on_preview_started)
            self._camera_manager.preview_stopped.connect(self._on_preview_stopped)

        self._refresh_dataset_display()
        self._refresh_cameras()

    def _on_backend_manager_status(self, status: str, severity: str):
        self._update_backend_state()

    def _update_backend_state(self):
        new_train = BackendRegistry.get_training_backend()
        if new_train != self._training_backend:
            try:
                self._training_backend.log_emitted.disconnect(self._append_log)
                self._training_backend.progress_updated.disconnect(self._on_training_progress)
                self._training_backend.training_completed.disconnect(self._on_training_completed)
            except Exception:
                pass
            self._training_backend = new_train
            self._training_backend.log_emitted.connect(self._append_log)
            self._training_backend.progress_updated.connect(self._on_training_progress)
            self._training_backend.training_completed.connect(self._on_training_completed)

        new_coll = BackendRegistry.get_static_collection_backend()
        if new_coll != self._collection_backend:
            try:
                self._collection_backend.sample_collected.disconnect(self._on_sample_collected)
                self._collection_backend.count_updated.disconnect(self._on_sample_count_updated)
            except Exception:
                pass
            self._collection_backend = new_coll
            self._collection_backend.sample_collected.connect(self._on_sample_collected)
            self._collection_backend.count_updated.connect(self._on_sample_count_updated)

        if self._collection_backend.is_connected():
            self.collect_status_pill.set_status("Ready to Collect", "active")
        else:
            self.collect_status_pill.set_status("Backend Disconnected", "neutral")

        if self._training_backend.is_connected():
            self.training_status_pill.set_status("Ready to Train", "active")
        else:
            self.training_status_pill.set_status("Backend Disconnected", "neutral")

    # =========================================================================
    # CAMERA CONTROLS
    # =========================================================================
    def _refresh_cameras(self):
        if not self._camera_manager:
            return
        devices = self._camera_manager.refresh_devices()
        curr = self._camera_manager.current_device
        sel_id = curr.id_str if curr else None
        self.cam_combo.populate_devices(devices, sel_id)

    def _on_devices_updated(self, devices: List[DeviceInfo]):
        if not self._camera_manager:
            return
        curr = self._camera_manager.current_device
        sel_id = curr.id_str if curr else None
        self.cam_combo.populate_devices(devices, sel_id)

    def _start_camera(self):
        if not self._camera_manager:
            return
        self._camera_manager.set_video_widget(self.video_widget)
        self._camera_manager.start_preview()

    def _stop_camera(self):
        if not self._camera_manager:
            return
        self._camera_manager.stop_preview()

    def _open_cam_settings(self):
        if self._camera_manager:
            dlg = CameraSettingsDialog(self._camera_manager, self.window())
            dlg.exec()

    def _on_preview_started(self):
        self.standby_overlay.setVisible(False)
        self.btn_start_cam.setEnabled(False)
        self.btn_stop_cam.setEnabled(True)
        self.cam_status_pill.set_status("Preview Active", "active")

    def _on_preview_stopped(self):
        if getattr(self, "_is_collecting", False):
            self._stop_collection()
        self.standby_overlay.setVisible(True)
        self.video_widget.clear_frame()
        self.btn_start_cam.setEnabled(True)
        self.btn_stop_cam.setEnabled(False)
        self.cam_status_pill.set_status("Camera Inactive", "neutral")
        self.lbl_cam_fps.setText("0.0 FPS")

    def _on_camera_status(self, text: str, status_type: str):
        self.cam_status_pill.set_status(text, status_type)
        if self._camera_manager and self._camera_manager.is_previewing:
            self.lbl_cam_fps.setText(f"{self._camera_manager.current_fps} FPS")

    # =========================================================================
    # EVENT FILTER & LIFECYCLE MANAGEMENT
    # =========================================================================
    def eventFilter(self, watched, event):
        """Intercept Spacebar keypress to trigger sample capture without scrolling or repeats."""
        if event.type() == QEvent.Type.KeyPress:
            if event.key() == Qt.Key.Key_Space:
                if not getattr(self, "_is_collecting", False):
                    return super().eventFilter(watched, event)
                # Ignore auto-repeat when held down
                if event.isAutoRepeat():
                    return True
                # Ignore Spacebar when typing in an editable field
                focus_widget = QApplication.focusWidget()
                if focus_widget and isinstance(focus_widget, (QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox)):
                    return super().eventFilter(watched, event)
                # Perform capture and consume event (prevents scrolling)
                self._capture_sample_action()
                return True
        return super().eventFilter(watched, event)

    def hideEvent(self, event):
        if getattr(self, "_is_collecting", False):
            self._stop_collection()
        super().hideEvent(event)

    def closeEvent(self, event):
        if getattr(self, "_is_collecting", False):
            self._stop_collection()
        super().closeEvent(event)

    # =========================================================================
    # DATA COLLECTION ACTIONS
    # =========================================================================
    def _start_collection(self):
        sign = self.combo_sign.currentText()
        active_ds = self._dataset_registry.get_active_static_dataset()
        target_path = active_ds.file_path if active_ds else ""

        success = self._collection_backend.start_collection(sign, target_path)
        if not success:
            self.collect_status_pill.set_status("Backend Disconnected", "neutral")
            msg = QMessageBox(self)
            msg.setWindowTitle("Sample Collection Backend Disconnected")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Sample Collection Backend Disconnected")
            msg.setInformativeText(
                f"Data collection backend contract is defined in 'app/core/interfaces.py' (StaticCollectionInterface).\n\n"
                f"Target Sign: '{sign}'\n\n"
                "To enable live coordinate logging:\n"
                "1. Subclass StaticCollectionInterface.\n"
                "2. Connect collect_static_data.py to extract 21 MediaPipe hand landmarks.\n"
                "3. Wire BackendRegistry.set_static_collection_backend(your_instance).\n\n"
                "No fake samples will be recorded until the ML backend is linked."
            )
            msg.open()
        else:
            self._is_collecting = True
            self.btn_start_collect.setEnabled(False)
            self.btn_capture_sample.setEnabled(True)
            self.btn_stop_collect.setEnabled(True)
            self.collect_status_pill.set_status("Collection Active", "active")

            # Install event filter for Spacebar capture
            app = QApplication.instance()
            if app:
                app.installEventFilter(self)

            # Ensure camera preview is running
            if self._camera_manager and not self._camera_manager.is_previewing:
                self._start_camera()

            # Status banner on live preview
            self.video_widget.set_status_banner("COLLECTION ACTIVE — PRESS SPACE TO CAPTURE")

    def _stop_collection(self):
        self._is_collecting = False
        app = QApplication.instance()
        if app:
            try:
                app.removeEventFilter(self)
            except Exception:
                pass

        self._collection_backend.stop_collection()
        self.btn_start_collect.setEnabled(True)
        self.btn_capture_sample.setEnabled(False)
        self.btn_stop_collect.setEnabled(False)
        self.collect_status_pill.set_status("Idle", "neutral")
        self.video_widget.set_status_banner("")

    def _capture_sample_action(self):
        """Execute manual sample capture using latest valid MediaPipe landmarks."""
        if not getattr(self, "_is_collecting", False):
            return

        if not self._camera_manager or not self._camera_manager.is_previewing:
            return

        # Check latest landmarks from LandmarkService
        service = LandmarkService.get_instance()
        latest = service.latest_landmarks
        if not latest or not latest.get("hands") or latest.get("num_hands", 0) == 0:
            # No hand detected: do not capture
            return

        features = latest.get("feature_vector")
        if not features or len(features) != 63 or any(not math.isfinite(v) for v in features):
            # Invalid or incomplete feature vector: do not capture
            return

        sign = self.combo_sign.currentText().strip().upper()
        if hasattr(self._collection_backend, "capture_sample"):
            rec = self._collection_backend.capture_sample(sign, features)
        else:
            rec = None

        if rec:
            # Short capture flash visual confirmation on the camera viewport
            self.video_widget.trigger_capture_flash()

            # Display confirmation message and time
            now_str = time.strftime("%H:%M:%S")
            confirm_msg = f"Captured '{sign}' at {now_str}"
            self.video_widget.set_last_capture_info(confirm_msg)
            self.lbl_last_capture.setText(f"Last capture: Sign {sign} ({now_str})")

    def _accept_sample(self):
        row = self.sample_table.currentRow()
        if row < 0 and self.sample_table.rowCount() > 0:
            row = 0
        if row >= 0:
            sample_id = self.sample_table.item(row, 0).text()
            self._collection_backend.accept_sample(sample_id)
            self.sample_table.setItem(row, 3, QTableWidgetItem("Accepted"))
        else:
            QMessageBox.information(self, "Buffer Empty", "No candidate samples currently in review buffer.")

    def _discard_sample(self):
        row = self.sample_table.currentRow()
        if row >= 0:
            sample_id = self.sample_table.item(row, 0).text()
            self._collection_backend.discard_sample(sample_id)
            self.sample_table.removeRow(row)

    @Slot(object)
    def _on_sample_collected(self, sample: SampleRecord):
        self._session_sample_count += 1
        self.lbl_sample_count.setText(f"{self._session_sample_count} recorded")

        row = self.sample_table.rowCount()
        self.sample_table.insertRow(row)
        self.sample_table.setItem(row, 0, QTableWidgetItem(sample.id))
        self.sample_table.setItem(row, 1, QTableWidgetItem(sample.sign_label))
        self.sample_table.setItem(row, 2, QTableWidgetItem(f"{sample.landmarks_count} coords"))
        self.sample_table.setItem(row, 3, QTableWidgetItem("Candidate"))
        self.sample_table.scrollToBottom()

    @Slot(int)
    def _on_sample_count_updated(self, count: int):
        pass

    # =========================================================================
    # MODEL TRAINING ACTIONS
    # =========================================================================
    def _start_training(self):
        config = {
            "n_estimators": self.spin_estimators.value(),
            "test_size": self.spin_test_size.value(),
            "random_state": self.spin_seed.value(),
            "criterion": self.combo_criterion.currentText(),
        }

        self._append_log(f"[Action] Starting static training with parameters: {config}")
        success = self._training_backend.start_static_training(config)

        if not success:
            self.training_status_pill.set_status("Backend Disconnected", "neutral")
            msg = QMessageBox(self)
            msg.setWindowTitle("Training Backend Disconnected")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Training Backend Disconnected")
            msg.setInformativeText(
                "Model training backend contract is defined in 'app/core/interfaces.py' (TrainingBackendInterface).\n\n"
                "To execute real training:\n"
                "1. Subclass TrainingBackendInterface.\n"
                "2. Integrate train_model.py routines with scikit-learn.\n"
                "3. Wire BackendRegistry.set_training_backend(your_instance).\n\n"
                "Zero fake progress or fabricated accuracy will be generated."
            )
            msg.open()
        else:
            self.btn_start_train.setEnabled(False)
            self.btn_stop_train.setEnabled(True)
            self.training_status_pill.set_status("Training Active", "active")

    def _stop_training(self):
        self._training_backend.stop_training()
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        self.training_status_pill.set_status("Idle", "neutral")

    def _register_model_version(self):
        QMessageBox.information(
            self,
            "Model Registry",
            "Model version registration contract is defined. When train_model.py completes "
            "and generates an updated .joblib classifier, it will automatically be cataloged "
            "into Model Manager."
        )

    def _refresh_dataset_display(self):
        active = self._dataset_registry.get_active_static_dataset()
        if active:
            self.lbl_ds_name.setText(active.name)
            self.lbl_ds_samples.setText(f"{active.row_count} samples")
        else:
            self.lbl_ds_name.setText("None selected (Register in Dataset Manager)")
            self.lbl_ds_samples.setText("0 samples")

    @Slot(str)
    def _append_log(self, text: str):
        self.log_console.append(text)

    @Slot(object)
    def _on_training_progress(self, progress: TrainingProgress):
        self.train_progress.setValue(int(progress.percent))
        if progress.message:
            self._append_log(f"[{progress.stage.upper()}] {progress.message}")

    @Slot(bool, str)
    def _on_training_completed(self, success: bool, message: str):
        self.btn_start_train.setEnabled(True)
        self.btn_stop_train.setEnabled(False)
        if success:
            self.training_status_pill.set_status("Completed", "active")
            self._append_log(f"[Success] Training finished: {message}")
        else:
            self.training_status_pill.set_status("Failed", "error")
            self._append_log(f"[Error] Training failed: {message}")
