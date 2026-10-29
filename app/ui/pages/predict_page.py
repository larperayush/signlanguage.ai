"""
Live Translation Workspace & Open-Source Model Harness.
Greets user with top menu bar for Model Selection, Camera Selection, and Add Model button.
Outputs live video with real-time sign language captions to Virtual Camera (Discord/Zoom/OBS).
"""
import time
from typing import Optional, List, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QComboBox, QTextEdit, QMessageBox, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRect, QSize, QPoint
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QClipboard
from PySide6.QtMultimedia import QVideoFrame

from app.core.camera_manager import CameraManager, DeviceInfo
from app.core.theme import ThemeManager
from app.core.interfaces import BackendRegistry, PredictionResult
from app.core.backend_adapter import qvideoframe_to_rgb_array, PythonMLPredictionAdapter
from app.core.model_registry import ModelRegistry, ModelPackage
from app.core.virtual_cam_service import VirtualCameraService
from app.core.landmark_service import LandmarkService, HAND_CONNECTIONS, FINGERTIP_INDICES, WRIST_INDEX
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill
from app.ui.components.add_model_dialog import AddModelDialog
from app.ui.components.camera_settings_dialog import CameraSettingsDialog


class CameraPreviewWidget(QWidget):
    """
    High-performance native camera preview canvas with
    MediaPipe landmark skeleton overlay.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = QVideoFrame()
        self._is_mirrored = True
        self._landmarks_data: Optional[Dict[str, Any]] = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_mirrored(self, mirrored: bool):
        self._is_mirrored = mirrored
        self.update()

    def set_frame(self, frame: QVideoFrame):
        self._frame = frame
        self.update()

    def set_landmarks(self, data: Optional[Dict[str, Any]]):
        self._landmarks_data = data
        self.update()

    def clear_frame(self):
        self._frame = QVideoFrame()
        self._landmarks_data = None
        self.update()

    @staticmethod
    def _compute_target_rect(widget_rect: QRect, frame_size: QSize) -> QRect:
        if not frame_size.isValid() or frame_size.width() <= 0 or frame_size.height() <= 0:
            return widget_rect
        fw, fh = frame_size.width(), frame_size.height()
        rw, rh = widget_rect.width(), widget_rect.height()
        scale = min(rw / fw, rh / fh)
        tw = int(fw * scale)
        th = int(fh * scale)
        tx = widget_rect.x() + (rw - tw) // 2
        ty = widget_rect.y() + (rh - th) // 2
        return QRect(tx, ty, tw, th)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager.get_palette()

        # Canvas background
        painter.fillRect(self.rect(), QColor(palette.video_bg))

        if not self._frame.isValid():
            return

        target_rect = self._compute_target_rect(self.rect(), self._frame.size())

        # 1. Paint Native Video Frame with Aspect Ratio
        painter.save()
        if self._is_mirrored:
            painter.translate(self.width(), 0)
            painter.scale(-1, 1)
        opts = QVideoFrame.PaintOptions()
        opts.aspectRatioMode = Qt.AspectRatioMode.KeepAspectRatio
        self._frame.paint(painter, self.rect(), opts)
        painter.restore()

        # 2. Draw Landmark Skeleton
        if self._landmarks_data:
            hands = self._landmarks_data.get("hands", [])
            tw, th = target_rect.width(), target_rect.height()
            tx, ty = target_rect.x(), target_rect.y()

            for hand_pts in hands:
                if len(hand_pts) != 21:
                    continue

                coords = []
                for pt in hand_pts:
                    lx = (1.0 - pt.x) if self._is_mirrored else pt.x
                    px = tx + lx * tw
                    py = ty + pt.y * th
                    coords.append((px, py))

                # Draw Bones with warm golden accent
                line_pen = QPen(QColor(palette.accent), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
                painter.setPen(line_pen)

                for start_idx, end_idx in HAND_CONNECTIONS:
                    if start_idx < len(coords) and end_idx < len(coords):
                        p1, p2 = coords[start_idx], coords[end_idx]
                        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

                # Draw Keypoints
                for idx, (px, py) in enumerate(coords):
                    if idx in FINGERTIP_INDICES:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                        painter.setBrush(QBrush(QColor(palette.accent_label)))
                        painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                    elif idx == WRIST_INDEX:
                        painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                        painter.setBrush(QBrush(QColor(palette.success)))
                        painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                    else:
                        painter.setPen(QPen(QColor(22, 21, 19, 180), 1.0))
                        painter.setBrush(QBrush(QColor(palette.accent)))
                        painter.drawEllipse(int(px - 3.5), int(py - 3.5), 7, 7)


class PredictPage(QWidget):
    """
    Main Translation Screen & AI Model Harness.
    Includes Top Control Bar (Model, Camera, Start/Stop),
    Live Preview canvas, Real-time Transcript, and Virtual Camera Streaming.
    """
    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._model_registry = ModelRegistry.get_instance()
        self._virtual_cam = VirtualCameraService.get_instance()

        self._is_translating = False
        self._current_sign = "—"
        self._current_confidence = 0.0
        self._transcript_text = ""
        self._last_sign_time = 0.0

        self._setup_ui()
        self._connect_signals()
        self._populate_models()
        self._populate_cameras()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 16, 20, 20)
        root_layout.setSpacing(14)

        # =====================================================================
        # TOP MENU BAR: Model Selection, Add Model, Camera Selection, Start
        # =====================================================================
        top_bar = QFrame()
        top_bar.setProperty("class", "cardPanelElevated")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(12)

        # 1. Model Selector Group
        model_vbox = QVBoxLayout()
        model_vbox.setSpacing(3)
        lbl_model = QLabel("AI MODEL HARNESS")
        lbl_model.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        
        model_hrow = QHBoxLayout()
        model_hrow.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)

        self.add_model_btn = QPushButton(" + Add Model")
        self.add_model_btn.setProperty("class", "secondaryBtn")
        self.add_model_btn.setIcon(get_icon("box", palette.text_secondary, palette.accent, size=14))
        self.add_model_btn.setToolTip("Import and register a custom trained sign language model folder")
        self.add_model_btn.clicked.connect(self._open_add_model_dialog)

        model_hrow.addWidget(self.model_combo)
        model_hrow.addWidget(self.add_model_btn)
        model_vbox.addWidget(lbl_model)
        model_vbox.addLayout(model_hrow)
        top_layout.addLayout(model_vbox)

        # Vertical Divider
        vsep1 = QFrame()
        vsep1.setFrameShape(QFrame.Shape.VLine)
        vsep1.setStyleSheet(f"border-left: 1px solid {palette.border_subtle};")
        top_layout.addWidget(vsep1)

        # 2. Camera Selector Group
        cam_vbox = QVBoxLayout()
        cam_vbox.setSpacing(3)
        lbl_cam = QLabel("CAMERA DEVICE")
        lbl_cam.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")

        cam_hrow = QHBoxLayout()
        cam_hrow.setSpacing(8)

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(210)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_selected)

        self.refresh_cams_btn = QPushButton()
        self.refresh_cams_btn.setProperty("class", "secondaryBtn")
        self.refresh_cams_btn.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        self.refresh_cams_btn.setToolTip("Scan system for video hardware and virtual cameras")
        self.refresh_cams_btn.clicked.connect(self._populate_cameras)

        cam_hrow.addWidget(self.camera_combo)
        cam_hrow.addWidget(self.refresh_cams_btn)
        cam_vbox.addWidget(lbl_cam)
        cam_vbox.addLayout(cam_hrow)
        top_layout.addLayout(cam_vbox)

        top_layout.addStretch()

        # 3. Virtual Cam Status Pill
        self.vcam_pill = StatusPill("Virtual Cam: Ready", "neutral")
        top_layout.addWidget(self.vcam_pill)

        # 4. Primary Start / Stop Translation Action Button
        self.action_btn = QPushButton(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setMinimumHeight(38)
        self.action_btn.setIcon(get_icon("play", "#1C1917" if not palette.is_dark else "#161513", size=16))
        self.action_btn.clicked.connect(self._toggle_translation)
        top_layout.addWidget(self.action_btn)

        root_layout.addWidget(top_bar)

        # =====================================================================
        # MAIN WORKSPACE: Camera Preview (Left) + Translation Card (Right)
        # =====================================================================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)

        # Left: Native Camera Canvas
        self.video_container = QFrame()
        self.video_container.setProperty("class", "cardPanel")
        self.video_container.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.video_bg};
                border: 1px solid {palette.border_default};
                border-radius: 8px;
            }}
        """)
        self.video_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        video_vbox = QVBoxLayout(self.video_container)
        video_vbox.setContentsMargins(0, 0, 0, 0)

        self.video_widget = CameraPreviewWidget(self.video_container)
        self._camera_manager.set_video_widget(self.video_widget)
        video_vbox.addWidget(self.video_widget)

        body_layout.addWidget(self.video_container, 6)

        # Right: Translation Output & Subtitle Card (Editorial & Tactile)
        readout_panel = QFrame()
        readout_panel.setProperty("class", "cardPanelElevated")
        readout_panel.setFixedWidth(330)
        ro_layout = QVBoxLayout(readout_panel)
        ro_layout.setContentsMargins(18, 18, 18, 18)
        ro_layout.setSpacing(14)

        # Panel Header
        hdr_row = QHBoxLayout()
        lbl_ro_title = QLabel("Translation Engine")
        lbl_ro_title.setStyleSheet("font-size: 14px; font-weight: 700;")
        self.live_indicator = StatusPill("Standby", "neutral")

        hdr_row.addWidget(lbl_ro_title)
        hdr_row.addStretch()
        hdr_row.addWidget(self.live_indicator)
        ro_layout.addLayout(hdr_row)

        # Recognized Sign Display Card
        sign_box = QFrame()
        sign_box.setProperty("class", "cardPanel")
        sb_layout = QVBoxLayout(sign_box)
        sb_layout.setContentsMargins(16, 18, 16, 18)
        sb_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.setSpacing(4)

        lbl_sign_hint = QLabel("RECOGNIZED SIGN")
        lbl_sign_hint.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        lbl_sign_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_sign = QLabel("—")
        self.lbl_sign.setStyleSheet(f"font-size: 54px; font-weight: 800; color: {palette.accent_label}; padding: 2px;")
        self.lbl_sign.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_confidence = QLabel("Confidence: 0%")
        self.lbl_confidence.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary}; font-weight: 500;")
        self.lbl_confidence.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sb_layout.addWidget(lbl_sign_hint)
        sb_layout.addWidget(self.lbl_sign)
        sb_layout.addWidget(self.lbl_confidence)
        ro_layout.addWidget(sign_box)

        # Real-time Transcribed Sentence / Text
        lbl_trans_hint = QLabel("TRANSCRIBED SENTENCE")
        lbl_trans_hint.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        ro_layout.addWidget(lbl_trans_hint)

        self.transcript_box = QTextEdit()
        self.transcript_box.setPlaceholderText("Signs translated in real time will accumulate here...")
        self.transcript_box.setReadOnly(False)
        self.transcript_box.setProperty("class", "consoleText")
        self.transcript_box.setMinimumHeight(130)
        ro_layout.addWidget(self.transcript_box, 1)

        # Text Manipulation Action Buttons
        txt_actions = QHBoxLayout()
        txt_actions.setSpacing(6)

        btn_space = QPushButton("Space")
        btn_space.setProperty("class", "secondaryBtn")
        btn_space.clicked.connect(self._add_space)

        btn_back = QPushButton("⌫ Del")
        btn_back.setProperty("class", "secondaryBtn")
        btn_back.clicked.connect(self._delete_char)

        btn_clear = QPushButton("Clear")
        btn_clear.setProperty("class", "subtleBtn")
        btn_clear.clicked.connect(self._clear_transcript)

        btn_copy = QPushButton("Copy Text")
        btn_copy.setProperty("class", "secondaryBtn")
        btn_copy.setIcon(get_icon("copy", palette.text_secondary, palette.accent, size=13))
        btn_copy.clicked.connect(self._copy_transcript)

        txt_actions.addWidget(btn_space)
        txt_actions.addWidget(btn_back)
        txt_actions.addWidget(btn_clear)
        txt_actions.addStretch()
        txt_actions.addWidget(btn_copy)
        ro_layout.addLayout(txt_actions)

        # Virtual Camera Guide Notice
        vcam_info = QFrame()
        vcam_info.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.accent_subtle};
                border: 1px solid {palette.accent_border};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        vi_layout = QVBoxLayout(vcam_info)
        vi_layout.setContentsMargins(6, 6, 6, 6)
        vi_layout.setSpacing(4)

        vcam_title = QLabel("Virtual Camera Broadcast")
        vcam_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_primary};")
        vcam_desc = QLabel(
            "When translation is started, apps like Discord, Zoom, and Teams can select "
            "'OBS Virtual Camera' to view your live signed video with translation subtitles."
        )
        vcam_desc.setStyleSheet(f"font-size: 10.5px; color: {palette.text_secondary}; line-height: 1.35;")
        vcam_desc.setWordWrap(True)

        vi_layout.addWidget(vcam_title)
        vi_layout.addWidget(vcam_desc)
        ro_layout.addWidget(vcam_info)

        body_layout.addWidget(readout_panel, 4)
        root_layout.addLayout(body_layout, 1)

    def _connect_signals(self):
        # Model Registry events
        self._model_registry.package_added.connect(self._on_package_added)
        self._model_registry.active_package_changed.connect(self._on_active_package_changed)

        # Virtual Camera status
        self._virtual_cam.status_changed.connect(self._on_vcam_status)

        # Camera Manager frames
        self._camera_manager.frame_ready.connect(self._on_frame_ready)

        # Backend predictions
        backend = BackendRegistry.get_prediction_backend()
        if hasattr(backend, "prediction_ready"):
            backend.prediction_ready.connect(self._on_prediction_ready)

    def _populate_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        packages = self._model_registry.get_all_packages()
        active = self._model_registry.get_active_package()

        active_idx = 0
        for i, pkg in enumerate(packages):
            self.model_combo.addItem(f"📦 {pkg.name}", userData=pkg.id)
            if active and pkg.id == active.id:
                active_idx = i

        if packages:
            self.model_combo.setCurrentIndex(active_idx)
        else:
            self.model_combo.addItem("No models found — Click Add Model", userData="")

        self.model_combo.blockSignals(False)

    def _populate_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        devices = self._camera_manager.devices

        for i, dev in enumerate(devices):
            self.camera_combo.addItem(f"📹 {dev.description}", userData=dev.id_str)

        if not devices:
            self.camera_combo.addItem("No camera hardware detected", userData="")

        self.camera_combo.blockSignals(False)

    def _on_model_selected(self, index: int):
        pkg_id = self.model_combo.currentData()
        if not pkg_id:
            return
        pkg = self._model_registry.get_package(pkg_id)
        if pkg:
            self._model_registry.set_active_package(pkg_id)
            backend = BackendRegistry.get_prediction_backend()
            if isinstance(backend, PythonMLPredictionAdapter):
                backend.load_model_package(pkg.task_path, pkg.classifier_path)

    def _on_package_added(self, package: ModelPackage):
        self._populate_models()

    def _on_active_package_changed(self, package: ModelPackage):
        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == package.id:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
                break

    def _open_add_model_dialog(self):
        dlg = AddModelDialog(self)
        dlg.model_imported.connect(self._on_model_imported)
        dlg.exec()

    def _on_model_imported(self, package: ModelPackage):
        self._populate_models()
        self._on_model_selected(self.model_combo.currentIndex())
        QMessageBox.information(
            self,
            "Model Harness Activated",
            f"Successfully imported and activated '{package.name}'.\n"
            f"Classifier: {Path(package.classifier_path).name}\n"
            f"Landmarker: {Path(package.task_path).name}"
        )

    def _on_camera_selected(self, index: int):
        dev_id = self.camera_combo.currentData()
        if dev_id:
            self._camera_manager.select_device(dev_id)

    def _toggle_translation(self):
        if not self._is_translating:
            self._start_translation()
        else:
            self._stop_translation()

    def _start_translation(self):
        palette = ThemeManager.get_palette()
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"

        # Ensure active model is loaded into backend
        active_pkg = self._model_registry.get_active_package()
        if active_pkg:
            backend = BackendRegistry.get_prediction_backend()
            if isinstance(backend, PythonMLPredictionAdapter):
                backend.load_model_package(active_pkg.task_path, active_pkg.classifier_path)

        # 1. Start camera hardware stream if not running
        if not self._camera_manager.is_previewing:
            self._camera_manager.start_preview()

        # 2. Start ML prediction adapter
        backend = BackendRegistry.get_prediction_backend()
        backend.start_prediction("Static", "")

        # 3. Start Virtual Camera stream for Discord / Zoom
        self._virtual_cam.start(width=640, height=480, fps=30)

        self._is_translating = True
        self.action_btn.setText(" Stop Translation")
        self.action_btn.setProperty("class", "dangerBtn")
        self.action_btn.setIcon(get_icon("stop", "#FFFFFF", size=15))
        self.live_indicator.set_status("Translating Live", "active")
        self._apply_button_style()

    def _stop_translation(self):
        palette = ThemeManager.get_palette()
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"

        backend = BackendRegistry.get_prediction_backend()
        backend.stop_prediction()

        self._virtual_cam.stop()

        self._is_translating = False
        self.action_btn.setText(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setIcon(get_icon("play", btn_text_color, size=16))
        self.live_indicator.set_status("Standby", "neutral")
        self._apply_button_style()

    def _apply_button_style(self):
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)

    def _on_frame_ready(self, frame: QVideoFrame):
        """Pass live video frame to Virtual Camera broadcast."""
        if not self._is_translating or not self._virtual_cam.is_active:
            return

        rgb_arr = qvideoframe_to_rgb_array(frame)
        if rgb_arr is not None:
            self._virtual_cam.broadcast_frame(
                rgb_arr,
                sign=self._current_sign,
                confidence=self._current_confidence,
                transcript=self.transcript_box.toPlainText()
            )

    @Slot(object)
    def _on_prediction_ready(self, res: PredictionResult):
        if not self._is_translating:
            return

        sign = res.sign
        conf = res.confidence

        if sign and sign not in ("No Hand", "Uncertain"):
            self._current_sign = sign
            self._current_confidence = conf
            self.lbl_sign.setText(sign)
            self.lbl_confidence.setText(f"Confidence: {int(conf * 100)}%")

            now = time.time()
            if now - self._last_sign_time > 1.2:
                self._transcript_text += sign
                self.transcript_box.setText(self._transcript_text)
                self.transcript_box.moveCursor(self.transcript_box.textCursor().MoveOperation.End)
                self._last_sign_time = now

    def _on_vcam_status(self, is_active: bool, msg: str):
        if is_active:
            self.vcam_pill.set_status(msg, "active")
        else:
            self.vcam_pill.set_status("Virtual Cam: Ready", "neutral")

    def _add_space(self):
        self._transcript_text += " "
        self.transcript_box.setText(self._transcript_text)

    def _delete_char(self):
        if self._transcript_text:
            self._transcript_text = self._transcript_text[:-1]
            self.transcript_box.setText(self._transcript_text)

    def _clear_transcript(self):
        self._transcript_text = ""
        self.transcript_box.clear()
        self.lbl_sign.setText("—")
        self.lbl_confidence.setText("Confidence: 0%")

    def _copy_transcript(self):
        txt = self.transcript_box.toPlainText()
        if txt:
            QApplication.clipboard().setText(txt)
            QMessageBox.information(self, "Copied", "Transcribed sign text copied to clipboard!")

    def on_page_activated(self):
        self._populate_models()
        self._populate_cameras()

    def on_page_deactivated(self):
        if self._is_translating:
            self._stop_translation()
