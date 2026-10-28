"""
Dynamic Gesture Training & Camera Sequence Recording Workspace.
Prioritizes sign Z with J as future support.
Features live camera preview, 2D motion trajectory visualizer, sequence recording,
candidate review buffer, temporal architecture principles, and model training.
"""
from typing import Optional, List
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QSpinBox, QDoubleSpinBox, QProgressBar, QTextEdit, QComboBox,
    QScrollArea, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedLayout, QSizePolicy, QButtonGroup
)
from PySide6.QtCore import Qt, QPointF, Slot
from PySide6.QtGui import QPainter, QPen, QColor, QPainterPath, QBrush

from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager, DeviceInfo
from app.core.interfaces import (
    BackendRegistry, TrainingBackendInterface, DynamicRecordingInterface,
    TrainingProgress, DynamicSequenceRecord
)
from app.core.dataset_registry import DatasetRegistry
from app.core.model_registry import ModelRegistry
from app.core.backend_manager import BackendManager
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import SectionHeader, StatusPill
from app.ui.components.searchable_combo import SearchableCameraComboBox
from app.ui.pages.predict_page import CameraPreviewWidget, CameraSettingsDialog


class TrajectoryVisualizerWidget(QFrame):
    """
    2D vector canvas rendering motion trajectories and directional vectors
    for movement-based gestures (Z zigzag stroke and J swoop).
    """
    def __init__(self, gesture_sign: str = "Z", parent=None):
        super().__init__(parent)
        self.gesture_sign = gesture_sign
        self.setMinimumHeight(200)
        self.setProperty("class", "cardPanel")

    def set_gesture(self, gesture: str):
        self.gesture_sign = gesture.upper()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = ThemeManager.get_palette()
        w = self.width()
        h = self.height()

        # Grid background
        grid_pen = QPen(QColor(palette.border_subtle), 1, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        step = 28
        for x in range(0, w, step):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, step):
            painter.drawLine(0, y, w, y)

        accent_color = QColor(palette.accent)
        path_pen = QPen(accent_color, 3.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(path_pen)

        path = QPainterPath()
        keypoints = []

        if self.gesture_sign == "Z":
            # Z Trajectory: Top horizontal stroke (left to right), diagonal stroke down-left, bottom horizontal stroke
            cx = w / 2
            p1 = QPointF(cx - 70, h * 0.22)
            p2 = QPointF(cx + 70, h * 0.22)
            p3 = QPointF(cx - 70, h * 0.78)
            p4 = QPointF(cx + 70, h * 0.78)

            path.moveTo(p1)
            path.lineTo(p2)
            path.lineTo(p3)
            path.lineTo(p4)
            keypoints = [p1, p2, p3, p4]
        else:
            # J Trajectory: Plunge downward, curve hook left, rise upward
            cx = w / 2
            p1 = QPointF(cx + 40, h * 0.18)
            p2 = QPointF(cx + 40, h * 0.62)
            hook_bot = QPointF(cx - 10, h * 0.82)
            p3 = QPointF(cx - 50, h * 0.58)

            path.moveTo(p1)
            path.lineTo(p2)
            path.quadTo(QPointF(cx + 35, h * 0.82), hook_bot)
            path.quadTo(QPointF(cx - 50, h * 0.82), p3)
            keypoints = [p1, p2, hook_bot, p3]

        painter.drawPath(path)

        # Draw Keypoint Nodes
        node_brush = QBrush(QColor(palette.bg_card))
        node_pen = QPen(accent_color, 2)
        painter.setBrush(node_brush)
        painter.setPen(node_pen)

        for idx, pt in enumerate(keypoints):
            painter.drawEllipse(pt, 7, 7)
            # Draw step label
            painter.setPen(QColor(palette.text_primary))
            painter.drawText(int(pt.x() + 10), int(pt.y() + 4), f"T{idx+1}")
            painter.setPen(node_pen)


class DynamicTrainingPage(QWidget):
    """
    Camera-based dynamic sequence recording and temporal model training workstation.
    Prioritizes movement-based gesture Z, keeping J available as future support.
    """
    def __init__(self, camera_manager: CameraManager = None, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._training_backend: TrainingBackendInterface = BackendRegistry.get_training_backend()
        self._recording_backend: DynamicRecordingInterface = BackendRegistry.get_dynamic_recording_backend()
        self._dataset_registry = DatasetRegistry.get_instance()
        self._model_registry = ModelRegistry.get_instance()
        self._selected_gesture = "Z"

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
            "Dynamic Sequence Training Workspace",
            "Record continuous temporal hand landmark sequences across consecutive video frames for movement-based gestures."
        )
        main_layout.addWidget(header)

        # =====================================================================
        # TOP SPLIT: CAMERA & TRAJECTORY VIEWPORT (LEFT) & SEQUENCE CONTROLS (RIGHT)
        # =====================================================================
        top_split = QHBoxLayout()
        top_split.setSpacing(18)

        # ---------------------------------------------------------------------
        # LEFT: LIVE CAMERA VIEWPORT & TRAJECTORY OVERLAY
        # ---------------------------------------------------------------------
        cam_col = QVBoxLayout()
        cam_col.setSpacing(10)

        cam_header = QHBoxLayout()
        cam_title = QLabel("LIVE CAMERA VIEWPORT & TRAJECTORY")
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
        self.video_container.setMinimumSize(460, 280)
        self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        video_stack = QStackedLayout(self.video_container)
        video_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.video_widget = CameraPreviewWidget()
        self.video_widget.setStyleSheet("background-color: transparent;")
        video_stack.addWidget(self.video_widget)

        # Standby Overlay
        self.standby_overlay = QWidget()
        overlay_layout = QVBoxLayout(self.standby_overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.setSpacing(8)

        standby_icon = QLabel()
        standby_icon.setPixmap(get_pixmap("dynamic_training", palette.text_muted, size=40))
        standby_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_title = QLabel("Dynamic Sequence Camera Standby")
        standby_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #FFFFFF;")
        standby_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_desc = QLabel("Click 'Start Camera' below to initialize video capture for temporal gesture recording.")
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

        self.cam_combo = SearchableCameraComboBox()
        self.cam_combo.setMinimumWidth(180)
        if self._camera_manager:
            self.cam_combo.device_chosen.connect(self._camera_manager.select_device)
        cam_bar_layout.addWidget(self.cam_combo)

        self.btn_refresh_cam = QPushButton()
        self.btn_refresh_cam.setProperty("class", "secondaryBtn")
        self.btn_refresh_cam.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        self.btn_refresh_cam.setToolTip("Refresh camera devices")
        self.btn_refresh_cam.clicked.connect(self._refresh_cameras)
        cam_bar_layout.addWidget(self.btn_refresh_cam)

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

        # 2D Motion Trajectory Visualizer
        self.visualizer = TrajectoryVisualizerWidget(gesture_sign="Z")
        cam_col.addWidget(self.visualizer)

        top_split.addLayout(cam_col, 5)

        # ---------------------------------------------------------------------
        # RIGHT: DYNAMIC SEQUENCE RECORDING & REVIEW PANEL
        # ---------------------------------------------------------------------
        rec_panel = QFrame()
        rec_panel.setProperty("class", "cardPanel")
        rec_layout = QVBoxLayout(rec_panel)
        rec_layout.setContentsMargins(18, 16, 18, 16)
        rec_layout.setSpacing(12)

        r_header = QHBoxLayout()
        r_title = QLabel("SEQUENCE RECORDING WORKSPACE")
        r_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        self.rec_status_pill = StatusPill("Backend Disconnected", "neutral")
        r_header.addWidget(r_title)
        r_header.addStretch()
        r_header.addWidget(self.rec_status_pill)
        rec_layout.addLayout(r_header)

        # Target Gesture Selector (Prioritizing Z)
        g_row = QHBoxLayout()
        g_row.setSpacing(8)

        self.gesture_group = QButtonGroup(self)
        self.gesture_group.setExclusive(True)

        self.btn_gesture_z = QPushButton("Z — Active Gesture")
        self.btn_gesture_z.setCheckable(True)
        self.btn_gesture_z.setChecked(True)
        self.btn_gesture_z.setProperty("class", "primaryBtn")

        self.btn_gesture_j = QPushButton("J — Future Support")
        self.btn_gesture_j.setCheckable(True)
        self.btn_gesture_j.setProperty("class", "secondaryBtn")

        self.gesture_group.addButton(self.btn_gesture_z, 0)
        self.gesture_group.addButton(self.btn_gesture_j, 1)
        self.gesture_group.idClicked.connect(self._on_gesture_selected)

        g_row.addWidget(self.btn_gesture_z)
        g_row.addWidget(self.btn_gesture_j)
        rec_layout.addLayout(g_row)

        # Recording Duration Window
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Sequence Window:"))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(15, 60)
        self.spin_window.setValue(30)
        self.spin_window.setSuffix(" frames")
        win_row.addWidget(self.spin_window)
        win_row.addStretch()
        self.lbl_time_est = QLabel("~1.0 second @ 30 FPS")
        self.lbl_time_est.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")
        win_row.addWidget(self.lbl_time_est)
        rec_layout.addLayout(win_row)

        # Record Actions
        rec_act_row = QHBoxLayout()
        rec_act_row.setSpacing(10)

        self.btn_start_rec = QPushButton("Start Recording Sequence")
        self.btn_start_rec.setProperty("class", "primaryBtn")
        self.btn_start_rec.setIcon(get_icon("play", "#FFFFFF", size=13))
        self.btn_start_rec.clicked.connect(self._start_recording)

        self.btn_stop_rec = QPushButton("Stop")
        self.btn_stop_rec.setProperty("class", "secondaryBtn")
        self.btn_stop_rec.setEnabled(False)
        self.btn_stop_rec.clicked.connect(self._stop_recording)

        rec_act_row.addWidget(self.btn_start_rec, 1)
        rec_act_row.addWidget(self.btn_stop_rec)
        rec_layout.addLayout(rec_act_row)

        # Sequence Count Metric
        metric_box = QFrame()
        metric_box.setProperty("class", "cardPanelElevated")
        m_layout = QHBoxLayout(metric_box)
        m_layout.setContentsMargins(12, 8, 12, 8)
        m_layout.addWidget(QLabel("Accepted Sequences:"))
        m_layout.addStretch()
        self.lbl_seq_count = QLabel("0 recorded")
        self.lbl_seq_count.setStyleSheet(f"font-weight: 700; color: {palette.text_primary};")
        m_layout.addWidget(self.lbl_seq_count)
        rec_layout.addWidget(metric_box)

        # Sequence Review Buffer
        rev_title = QLabel("Recorded Sequence Buffer (Review & Commit):")
        rev_title.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_secondary};")
        rec_layout.addWidget(rev_title)

        self.seq_table = QTableWidget(0, 4)
        self.seq_table.setHorizontalHeaderLabels(["ID", "Gesture", "Frames", "Status"])
        self.seq_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.seq_table.setMinimumHeight(100)
        self.seq_table.setMaximumHeight(130)
        rec_layout.addWidget(self.seq_table)

        # Accept / Discard Controls
        review_actions = QHBoxLayout()
        review_actions.setSpacing(10)

        self.btn_accept_seq = QPushButton(" Accept Sequence")
        self.btn_accept_seq.setProperty("class", "secondaryBtn")
        self.btn_accept_seq.setIcon(get_icon("check", palette.success, size=13))
        self.btn_accept_seq.clicked.connect(self._accept_sequence)

        self.btn_discard_seq = QPushButton(" Discard")
        self.btn_discard_seq.setProperty("class", "secondaryBtn")
        self.btn_discard_seq.setIcon(get_icon("trash", palette.danger, size=13))
        self.btn_discard_seq.clicked.connect(self._discard_sequence)

        review_actions.addWidget(self.btn_accept_seq)
        review_actions.addWidget(self.btn_discard_seq)
        review_actions.addStretch()
        rec_layout.addLayout(review_actions)

        # Temporal Principle Callout
        temporal_card = QFrame()
        temporal_card.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_card_elevated};
                border: 1px solid {palette.border_subtle};
                border-radius: 6px;
                padding: 4px;
            }}
        """)
        t_layout = QVBoxLayout(temporal_card)
        t_layout.setContentsMargins(12, 10, 12, 10)
        t_layout.setSpacing(4)

        t_title = QLabel("TEMPORAL SEQUENCE PRINCIPLE")
        t_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.5px;")
        t_desc = QLabel(
            "Movement-based gestures (Z and J) require temporal coordinate sequences across consecutive frames "
            "(T1 → T30). Rather than treating frames as independent static images, dynamic models evaluate motion "
            "trajectories, stroke speed, and spatial curve paths."
        )
        t_desc.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary}; line-height: 1.4;")
        t_desc.setWordWrap(True)

        t_layout.addWidget(t_title)
        t_layout.addWidget(t_desc)
        rec_layout.addWidget(temporal_card)

        top_split.addWidget(rec_panel, 4)
        main_layout.addLayout(top_split)

        # =====================================================================
        # BOTTOM: DYNAMIC TRAINING CONFIGURATION & LOGS
        # =====================================================================
        bottom_frame = QFrame()
        bottom_frame.setProperty("class", "cardPanel")
        b_layout = QVBoxLayout(bottom_frame)
        b_layout.setContentsMargins(20, 18, 20, 18)
        b_layout.setSpacing(16)

        b_header = QHBoxLayout()
        b_title = QLabel("DYNAMIC MODEL TRAINING (LSTM / Temporal Classifier)")
        b_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        self.training_status_pill = StatusPill("Backend Disconnected", "neutral")
        b_header.addWidget(b_title)
        b_header.addStretch()
        b_header.addWidget(self.training_status_pill)
        b_layout.addLayout(b_header)

        # Dataset details
        ds_grid = QGridLayout()
        ds_grid.setSpacing(10)

        self.lbl_ds_name = QLabel("dynamic_gestures.csv (Pending Registration)")
        self.lbl_ds_name.setStyleSheet("font-weight: 600;")
        self.lbl_ds_samples = QLabel("0 sequences")

        ds_grid.addWidget(QLabel("Dynamic Dataset:"), 0, 0)
        ds_grid.addWidget(self.lbl_ds_name, 0, 1)
        ds_grid.addWidget(QLabel("Recorded Sequences:"), 0, 2)
        ds_grid.addWidget(self.lbl_ds_samples, 0, 3)
        b_layout.addLayout(ds_grid)

        # Training Hyperparameters
        param_grid = QGridLayout()
        param_grid.setSpacing(12)

        param_grid.addWidget(QLabel("Temporal Window:"), 0, 0)
        self.spin_seq_len = QSpinBox()
        self.spin_seq_len.setRange(15, 60)
        self.spin_seq_len.setValue(30)
        param_grid.addWidget(self.spin_seq_len, 0, 1)

        param_grid.addWidget(QLabel("Hidden Units:"), 0, 2)
        self.spin_hidden = QSpinBox()
        self.spin_hidden.setRange(16, 256)
        self.spin_hidden.setValue(64)
        param_grid.addWidget(self.spin_hidden, 0, 3)

        param_grid.addWidget(QLabel("Epochs:"), 1, 0)
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(5, 200)
        self.spin_epochs.setValue(30)
        param_grid.addWidget(self.spin_epochs, 1, 1)

        param_grid.addWidget(QLabel("Test Split Ratio:"), 1, 2)
        self.spin_split = QDoubleSpinBox()
        self.spin_split.setRange(0.10, 0.40)
        self.spin_split.setSingleStep(0.05)
        self.spin_split.setValue(0.20)
        param_grid.addWidget(self.spin_split, 1, 3)

        b_layout.addLayout(param_grid)

        # Start/Stop Training Buttons
        train_ctrl_row = QHBoxLayout()
        train_ctrl_row.setSpacing(12)

        self.btn_start_train = QPushButton("Start Sequence Model Training")
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
        self.log_console.append("[System] Dynamic Gesture Training Workspace ready.")
        self.log_console.append("[Gesture Prioritization] Z active; J available as future support.")
        b_layout.addWidget(self.log_console)

        # Register New Model Version Row
        model_reg_row = QHBoxLayout()
        model_reg_row.setSpacing(10)
        model_reg_row.addWidget(QLabel("Register Model Version:"))

        self.txt_model_version = QLabel("dynamic_gesture_z_v1.onnx")
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
        self._recording_backend.sequence_recorded.connect(self._on_sequence_recorded)
        self._recording_backend.count_updated.connect(self._on_sequence_count_updated)

        # Connect Camera Manager Signals
        if self._camera_manager:
            self._camera_manager.devices_changed.connect(self._on_devices_updated)
            self._camera_manager.status_changed.connect(self._on_camera_status)
            self._camera_manager.preview_started.connect(self._on_preview_started)
            self._camera_manager.preview_stopped.connect(self._on_preview_stopped)

        self._refresh_dataset_display()
        self._refresh_cameras()

    # =========================================================================
    # GESTURE SELECTION & CAMERA
    # =========================================================================
    def _on_gesture_selected(self, id_val: int):
        self._selected_gesture = "Z" if id_val == 0 else "J"
        self.visualizer.set_gesture(self._selected_gesture)
        self.txt_model_version.setText(f"dynamic_gesture_{self._selected_gesture.lower()}_v1.onnx")

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
    # SEQUENCE RECORDING ACTIONS
    # =========================================================================
    def _start_recording(self):
        seq_len = self.spin_window.value()
        success = self._recording_backend.start_recording(self._selected_gesture, seq_len)
        if not success:
            self.rec_status_pill.set_status("Backend Disconnected", "neutral")
            msg = QMessageBox(self)
            msg.setWindowTitle("Dynamic Sequence Backend Disconnected")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Dynamic Sequence Backend Disconnected")
            msg.setInformativeText(
                f"Dynamic recording backend contract is defined in 'app/core/interfaces.py' (DynamicRecordingInterface).\n\n"
                f"Target Gesture: '{self._selected_gesture}' ({seq_len} frames)\n\n"
                "To enable consecutive frame recording:\n"
                "1. Subclass DynamicRecordingInterface.\n"
                "2. Implement temporal coordinate buffer across video frames.\n"
                "3. Wire BackendRegistry.set_dynamic_recording_backend(your_instance).\n\n"
                "No fake sequences will be fabricated until the ML backend is linked."
            )
            msg.open()
        else:
            self.btn_start_rec.setEnabled(False)
            self.btn_stop_rec.setEnabled(True)
            self.rec_status_pill.set_status("Recording", "active")

    def _stop_recording(self):
        self._recording_backend.stop_recording()
        self.btn_start_rec.setEnabled(True)
        self.btn_stop_rec.setEnabled(False)
        self.rec_status_pill.set_status("Idle", "neutral")

    def _accept_sequence(self):
        row = self.seq_table.currentRow()
        if row < 0 and self.seq_table.rowCount() > 0:
            row = 0
        if row >= 0:
            seq_id = self.seq_table.item(row, 0).text()
            self._recording_backend.accept_sequence(seq_id)
            self.seq_table.setItem(row, 3, QTableWidgetItem("Accepted"))
        else:
            QMessageBox.information(self, "Buffer Empty", "No recorded sequences currently in review buffer.")

    def _discard_sequence(self):
        row = self.seq_table.currentRow()
        if row >= 0:
            seq_id = self.seq_table.item(row, 0).text()
            self._recording_backend.discard_sequence(seq_id)
            self.seq_table.removeRow(row)

    @Slot(object)
    def _on_sequence_recorded(self, seq: DynamicSequenceRecord):
        row = self.seq_table.rowCount()
        self.seq_table.insertRow(row)
        self.seq_table.setItem(row, 0, QTableWidgetItem(seq.id))
        self.seq_table.setItem(row, 1, QTableWidgetItem(seq.gesture_name))
        self.seq_table.setItem(row, 2, QTableWidgetItem(f"{seq.frame_count} frames ({seq.duration_seconds:.1f}s)"))
        self.seq_table.setItem(row, 3, QTableWidgetItem("Candidate"))

    @Slot(int)
    def _on_sequence_count_updated(self, count: int):
        self.lbl_seq_count.setText(f"{count} recorded")

    # =========================================================================
    # MODEL TRAINING ACTIONS
    # =========================================================================
    def _start_training(self):
        config = {
            "gesture": self._selected_gesture,
            "sequence_length": self.spin_seq_len.value(),
            "hidden_units": self.spin_hidden.value(),
            "epochs": self.spin_epochs.value(),
            "test_split": self.spin_split.value(),
        }

        self._append_log(f"[Action] Starting dynamic sequence training: {config}")
        success = self._training_backend.start_dynamic_training(config)

        if not success:
            self.training_status_pill.set_status("Backend Disconnected", "neutral")
            msg = QMessageBox(self)
            msg.setWindowTitle("Dynamic Training Backend Disconnected")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Dynamic Training Backend Disconnected")
            msg.setInformativeText(
                "Dynamic sequence training backend contract is defined in 'app/core/interfaces.py' (TrainingBackendInterface).\n\n"
                "To execute real temporal sequence training:\n"
                "1. Subclass TrainingBackendInterface.\n"
                "2. Implement start_dynamic_training with PyTorch / TensorFlow / Keras.\n"
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
            "Dynamic model artifact registration contract is defined. When the temporal model finishes training, "
            "it will automatically be cataloged into Model Manager."
        )

    def _refresh_dataset_display(self):
        active = self._dataset_registry.get_active_dynamic_dataset()
        if active:
            self.lbl_ds_name.setText(active.name)
            self.lbl_ds_samples.setText(f"{active.row_count} sequences")
        else:
            self.lbl_ds_name.setText("dynamic_gestures.csv (Pending Registration)")
            self.lbl_ds_samples.setText("0 sequences")

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
            self._append_log(f"[Success] Dynamic training completed: {message}")
        else:
            self.training_status_pill.set_status("Failed", "error")
            self._append_log(f"[Error] Dynamic training failed: {message}")
