"""
Predict Page - Primary Application Feature.
Hosts native camera preview via Qt Multimedia (QCamera, QVideoFrame),
recognition mode selector (Static, Dynamic, Combined), device manager controls,
and the right-side prediction panel with explicit backend disconnection handling.
"""
from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QButtonGroup, QTextEdit, QDialog, QComboBox, QCheckBox, QMessageBox,
    QStackedLayout, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRect, QSize, QPoint
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtMultimedia import QVideoFrame

from app.core.camera_manager import CameraManager, DeviceInfo
from app.core.config import (
    MODE_STATIC, MODE_DYNAMIC, MODE_COMBINED,
    SettingsManager
)
from app.core.theme import ThemeManager
from app.core.interfaces import (
    BackendRegistry, PredictionResult, PredictionBackendInterface
)
from app.core.backend_manager import BackendManager, STATUS_CONNECTED
from app.core.landmark_service import LandmarkService, HAND_CONNECTIONS, FINGERTIP_INDICES, WRIST_INDEX
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill
from app.ui.components.searchable_combo import SearchableCameraComboBox


class CameraPreviewWidget(QWidget):
    """
    High-performance native video preview canvas with transparent MediaPipe
    landmark overlay, dual FPS display, status banner, and capture flash.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = QVideoFrame()
        self._is_mirrored = SettingsManager.get_mirror_preview()
        self._landmarks_data: Optional[Dict[str, Any]] = None
        self._status_banner: str = ""
        self._last_capture_info: str = ""
        self._flash_opacity: float = 0.0
        self._flash_timer: Optional[QTimer] = None
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_mirrored(self, mirrored: bool):
        if self._is_mirrored != mirrored:
            self._is_mirrored = mirrored
            self.update()

    def is_mirrored(self) -> bool:
        return self._is_mirrored

    def set_frame(self, frame: QVideoFrame):
        self._frame = frame
        self.update()

    def set_landmarks(self, data: Optional[Dict[str, Any]]):
        self._landmarks_data = data
        self.update()

    def clear_landmarks(self):
        self._landmarks_data = None
        self.update()

    def clear_frame(self):
        self._frame = QVideoFrame()
        self._landmarks_data = None
        self._status_banner = ""
        self._last_capture_info = ""
        self.update()

    def set_status_banner(self, text: str):
        self._status_banner = text
        self.update()

    def set_last_capture_info(self, text: str):
        self._last_capture_info = text
        self.update()

    def trigger_capture_flash(self):
        """Initiate short capture flash animation on camera viewport."""
        self._flash_opacity = 0.55
        if self._flash_timer is None:
            self._flash_timer = QTimer(self)
            self._flash_timer.setInterval(25)
            self._flash_timer.timeout.connect(self._on_flash_step)
        self._flash_timer.start()
        self.update()

    def _on_flash_step(self):
        self._flash_opacity -= 0.10
        if self._flash_opacity <= 0.0:
            self._flash_opacity = 0.0
            if self._flash_timer:
                self._flash_timer.stop()
        self.update()

    @staticmethod
    def _compute_target_rect(widget_rect: QRect, frame_size: QSize) -> QRect:
        if not frame_size.isValid() or frame_size.width() <= 0 or frame_size.height() <= 0:
            return widget_rect
        fw = frame_size.width()
        fh = frame_size.height()
        rw = widget_rect.width()
        rh = widget_rect.height()
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

        # Fill background with dark video backdrop
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

        # 2. Draw Capture Flash Effect
        if self._flash_opacity > 0.0:
            painter.fillRect(target_rect, QColor(255, 255, 255, int(self._flash_opacity * 255)))

        # 3. Draw MediaPipe Hand Landmarks & Bone Connections
        has_hands = False
        num_hands = 0
        cam_fps = 0.0
        inf_fps = 0.0

        if self._landmarks_data:
            cam_fps = self._landmarks_data.get("camera_fps", 0.0)
            inf_fps = self._landmarks_data.get("inference_fps", 0.0)
            hands = self._landmarks_data.get("hands", [])
            num_hands = len(hands)

            if num_hands > 0:
                has_hands = True
                tw = target_rect.width()
                th = target_rect.height()
                tx = target_rect.x()
                ty = target_rect.y()

                for hand_pts in hands:
                    if len(hand_pts) != 21:
                        continue

                    # Calculate screen coordinates for all 21 points
                    coords = []
                    for pt in hand_pts:
                        lx = (1.0 - pt.x) if self._is_mirrored else pt.x
                        px = tx + lx * tw
                        py = ty + pt.y * th
                        coords.append((px, py))

                    # Draw Bone Connections (Cyan / Green high contrast)
                    line_pen = QPen(QColor(0, 240, 255, 210), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                    painter.setPen(line_pen)

                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(coords) and end_idx < len(coords):
                            p1 = coords[start_idx]
                            p2 = coords[end_idx]
                            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

                    # Draw Landmark Points
                    for idx, (px, py) in enumerate(coords):
                        if idx in FINGERTIP_INDICES:
                            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                            painter.setBrush(QBrush(QColor(0, 240, 255)))
                            painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                        elif idx == WRIST_INDEX:
                            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                            painter.setBrush(QBrush(QColor(0, 230, 118)))
                            painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                        else:
                            painter.setPen(QPen(QColor(15, 23, 42, 180), 1.0))
                            painter.setBrush(QBrush(QColor(0, 240, 255, 230)))
                            painter.drawEllipse(int(px - 3.5), int(py - 3.5), 7, 7)

        # 4. HUD Overlays & Badges
        pad = 12
        badge_y = target_rect.y() + pad
        badge_x = target_rect.x() + pad

        # A. Hand Detection Badge (Top Left)
        if has_hands:
            badge_text = f"Hand detected ({num_hands} hand{'s' if num_hands > 1 else ''})"
            dot_color = QColor("#00E676")
            bg_color = QColor(15, 23, 42, 210)
            text_color = QColor("#FFFFFF")
        else:
            badge_text = "No hand detected"
            dot_color = QColor("#EF4444")
            bg_color = QColor(15, 23, 42, 190)
            text_color = QColor("#94A3B8")

        fm = painter.fontMetrics()
        txt_w = fm.horizontalAdvance(badge_text)
        badge_w = txt_w + 32
        badge_h = 24

        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(badge_x, badge_y, badge_w, badge_h, 5, 5)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(badge_x + 9, badge_y + 8, 8, 8)

        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(9)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.drawText(badge_x + 24, badge_y + 16, badge_text)

        # B. Center Hint when no hand detected
        if not has_hands and self._frame.isValid():
            hint_text = "Show your hand to the camera"
            hw = fm.horizontalAdvance(hint_text) + 28
            hh = 32
            hx = target_rect.x() + (target_rect.width() - hw) // 2
            hy = target_rect.y() + (target_rect.height() - hh) // 2

            painter.setPen(QPen(QColor(255, 255, 255, 30), 1))
            painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
            painter.drawRoundedRect(hx, hy, hw, hh, 8, 8)

            painter.setPen(QColor("#CBD5E1"))
            font_hint = painter.font()
            font_hint.setPointSize(10)
            font_hint.setWeight(QFont.Weight.Medium)
            painter.setFont(font_hint)
            painter.drawText(hx + 14, hy + 21, hint_text)

        # C. Dual FPS Display (Bottom Left)
        fps_text = f"Camera: {cam_fps:.1f} FPS  |  Inference: {inf_fps:.1f} FPS"
        fps_w = fm.horizontalAdvance(fps_text) + 20
        fps_h = 22
        fps_x = target_rect.x() + pad
        fps_y = target_rect.y() + target_rect.height() - fps_h - pad

        painter.setPen(QPen(QColor(255, 255, 255, 20), 1))
        painter.setBrush(QBrush(QColor(15, 23, 42, 190)))
        painter.drawRoundedRect(fps_x, fps_y, fps_w, fps_h, 4, 4)

        painter.setPen(QColor("#94A3B8"))
        font_fps = painter.font()
        font_fps.setPointSize(8)
        font_fps.setFamily("monospace")
        painter.setFont(font_fps)
        painter.drawText(fps_x + 10, fps_y + 15, fps_text)

        # D. Status Banner (Top Center)
        if self._status_banner:
            bw = fm.horizontalAdvance(self._status_banner) + 32
            bh = 28
            bx = target_rect.x() + (target_rect.width() - bw) // 2
            by = target_rect.y() + pad

            painter.setPen(QPen(QColor(0, 240, 255, 180), 1.5))
            painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
            painter.drawRoundedRect(bx, by, bw, bh, 6, 6)

            painter.setPen(QColor("#00F0FF"))
            font_sb = painter.font()
            font_sb.setPointSize(9)
            font_sb.setWeight(QFont.Weight.Bold)
            painter.setFont(font_sb)
            painter.drawText(bx + 16, by + 19, self._status_banner)

        # E. Capture Toast / Confirmation (Bottom Right)
        if self._last_capture_info:
            cw = fm.horizontalAdvance(self._last_capture_info) + 24
            ch = 24
            cx = target_rect.x() + target_rect.width() - cw - pad
            cy = target_rect.y() + target_rect.height() - ch - pad

            painter.setPen(QPen(QColor(0, 230, 118, 140), 1))
            painter.setBrush(QBrush(QColor(15, 23, 42, 210)))
            painter.drawRoundedRect(cx, cy, cw, ch, 4, 4)

            painter.setPen(QColor("#00E676"))
            font_cw = painter.font()
            font_cw.setPointSize(8)
            font_cw.setWeight(QFont.Weight.DemiBold)
            painter.setFont(font_cw)
            painter.drawText(cx + 12, cy + 16, f"✓ {self._last_capture_info}")


class CameraSettingsDialog(QDialog):
    """
    Camera configuration modal for genuine resolution format negotiation
    and live preview orientation mirroring.
    """
    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Device Settings")
        self.setMinimumWidth(440)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._camera_manager = camera_manager
        palette = ThemeManager.get_palette()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        layout.setSpacing(16)

        title = QLabel("Video Capture Preferences")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        layout.addWidget(title)

        curr_device = self._camera_manager.current_device
        device_name = curr_device.description if curr_device else "No Camera Selected"

        # Active Device Info Box
        info_frame = QFrame()
        info_frame.setProperty("class", "cardPanel")
        info_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_card};
                border: 1px solid {palette.border_subtle};
                border-radius: 6px;
            }}
        """)
        info_layout = QVBoxLayout(info_frame)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(6)

        dev_title = QLabel(f"Active Device: {device_name}")
        dev_title.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {palette.text_primary};")
        info_layout.addWidget(dev_title)

        act_w, act_h = self._camera_manager.actual_resolution
        if act_w > 0 and act_h > 0:
            status_desc = f"Verified Stream: {act_w} × {act_h} px @ {self._camera_manager.current_fps} FPS"
        elif self._camera_manager.is_previewing:
            status_desc = "Streaming: Live capture active (evaluating frame dimensions...)"
        else:
            status_desc = "Capture Status: Standby (Preview Inactive)"

        self.lbl_effective_status = QLabel(status_desc)
        status_color = palette.accent if self._camera_manager.is_previewing else palette.text_muted
        self.lbl_effective_status.setStyleSheet(f"font-size: 11.5px; color: {status_color};")
        info_layout.addWidget(self.lbl_effective_status)

        layout.addWidget(info_frame)

        # 1. Stream Resolution Section
        res_vbox = QVBoxLayout()
        res_vbox.setSpacing(6)

        res_label = QLabel("Stream Resolution:")
        res_label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {palette.text_secondary};")
        res_vbox.addWidget(res_label)

        self.res_combo = QComboBox()
        self.res_combo.setMinimumHeight(32)

        has_discrete_formats = False
        if curr_device and curr_device.supported_resolutions:
            has_discrete_formats = True
            for r in curr_device.supported_resolutions:
                self.res_combo.addItem(r, userData=r)
        else:
            self.res_combo.addItem("Default Hardware Stream (Driver Managed)", userData="default")
            self.res_combo.setEnabled(False)

        # Pre-select matching format:
        # Priority: active frame resolution > saved preferred resolution > first supported
        target_res = ""
        if act_w > 0 and act_h > 0:
            target_res = f"{act_w}x{act_h}"
        elif self._camera_manager.preferred_resolution:
            target_res = self._camera_manager.preferred_resolution

        if target_res:
            idx = self.res_combo.findText(target_res)
            if idx >= 0:
                self.res_combo.setCurrentIndex(idx)

        res_vbox.addWidget(self.res_combo)

        if not has_discrete_formats:
            res_note = QLabel("Note: This camera driver manages resolution internally and does not expose discrete format negotiation.")
            res_note.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-style: italic;")
            res_note.setWordWrap(True)
            res_vbox.addWidget(res_note)

        layout.addLayout(res_vbox)

        # 2. Mirror Preview Checkbox
        self.mirror_check = QCheckBox("Mirror camera preview horizontally (Interaction View)")
        self.mirror_check.setStyleSheet("font-size: 13px; font-weight: 500;")
        self.mirror_check.setChecked(SettingsManager.get_mirror_preview())
        self.mirror_check.toggled.connect(self._on_mirror_toggled)
        layout.addWidget(self.mirror_check)

        mirror_hint = QLabel("Applies visual mirroring exclusively to the preview canvas. Raw camera frames emitted to the ML recognition engine remain in standardized orientation.")
        mirror_hint.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; line-height: 1.4;")
        mirror_hint.setWordWrap(True)
        layout.addWidget(mirror_hint)

        # Status Alert Banner
        self.alert_box = QFrame()
        self.alert_box.setVisible(False)
        self.alert_layout = QHBoxLayout(self.alert_box)
        self.alert_layout.setContentsMargins(12, 8, 12, 8)
        self.alert_msg = QLabel()
        self.alert_msg.setWordWrap(True)
        self.alert_layout.addWidget(self.alert_msg)
        layout.addWidget(self.alert_box)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        btn_layout.addStretch()

        self.apply_btn = QPushButton("Apply Settings")
        self.apply_btn.setProperty("class", "primaryBtn")
        self.apply_btn.clicked.connect(self._on_apply_clicked)

        self.close_btn = QPushButton("Close")
        self.close_btn.setProperty("class", "secondaryBtn")
        self.close_btn.clicked.connect(self.accept)

        btn_layout.addWidget(self.close_btn)
        btn_layout.addWidget(self.apply_btn)
        layout.addLayout(btn_layout)

        # Hook resolution verification signal
        self._camera_manager.resolution_changed.connect(self._on_resolution_verified)

    def _on_mirror_toggled(self, checked: bool):
        # Visually mirrors immediately on live canvas
        self._camera_manager.set_mirror_preview(checked)

    def _show_alert(self, text: str, is_error: bool = False):
        palette = ThemeManager.get_palette()
        border_c = palette.danger if is_error else palette.success
        bg_c = palette.danger_subtle if is_error else palette.success_subtle

        self.alert_box.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_c};
                border: 1px solid {border_c};
                border-radius: 6px;
            }}
        """)
        self.alert_msg.setText(text)
        self.alert_msg.setStyleSheet(f"font-size: 11.5px; color: {border_c}; font-weight: 500;")
        self.alert_box.setVisible(True)

    def _on_apply_clicked(self):
        chosen_res = self.res_combo.currentText()
        is_mirrored = self.mirror_check.isChecked()

        # 1. Apply mirroring preference
        self._camera_manager.set_mirror_preview(is_mirrored)

        # 2. Apply resolution if discrete format is selected
        if "x" in chosen_res and self.res_combo.isEnabled():
            success, msg, eff_res = self._camera_manager.set_resolution(chosen_res)
            if success:
                self._show_alert(f"Settings applied: {msg}", is_error=False)
                act_w, act_h = self._camera_manager.actual_resolution
                if act_w > 0:
                    self.lbl_effective_status.setText(f"Verified Stream: {act_w} × {act_h} px @ {self._camera_manager.current_fps} FPS")
                else:
                    self.lbl_effective_status.setText(f"Effective Resolution: {eff_res}")
                QTimer.singleShot(600, self.accept)
            else:
                self._show_alert(f"Failed to apply resolution: {msg}", is_error=True)
        else:
            self._show_alert("Hardware default format preserved.", is_error=False)
            QTimer.singleShot(600, self.accept)

    def _on_resolution_verified(self, w: int, h: int):
        self.lbl_effective_status.setText(f"Verified Stream: {w} × {h} px @ {self._camera_manager.current_fps} FPS")


class PredictPage(QWidget):
    """
    Main live inference and camera streaming screen.
    Includes hardware device switcher, preview, and prediction inspector.
    """
    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._backend: PredictionBackendInterface = BackendRegistry.get_prediction_backend()
        self._current_mode = SettingsManager.get_default_mode()
        self._text_accumulator = ""

        palette = ThemeManager.get_palette()

        # Main horizontal split: Camera preview workspace (left) & Prediction panel (right)
        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(20, 16, 20, 20)
        root_layout.setSpacing(18)

        # =====================================================================
        # LEFT AREA: CAMERA WORKSPACE
        # =====================================================================
        cam_workspace = QVBoxLayout()
        cam_workspace.setSpacing(12)

        # Top Bar above Camera: Mode Switcher + Status
        top_ctrl_row = QHBoxLayout()
        top_ctrl_row.setSpacing(10)

        mode_lbl = QLabel("Recognition Mode:")
        mode_lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {palette.text_secondary};")
        top_ctrl_row.addWidget(mode_lbl)

        # Mode Selector Buttons (Segmented)
        self.mode_btn_group = QButtonGroup(self)
        self.mode_btn_group.setExclusive(True)

        self.btn_static = QPushButton("Static Signs")
        self.btn_static.setCheckable(True)
        self.btn_static.setProperty("class", "secondaryBtn")

        self.btn_dynamic = QPushButton("Dynamic Signs (J / Z)")
        self.btn_dynamic.setCheckable(True)
        self.btn_dynamic.setProperty("class", "secondaryBtn")

        self.btn_combined = QPushButton("Combined Recognition")
        self.btn_combined.setCheckable(True)
        self.btn_combined.setProperty("class", "secondaryBtn")

        self.mode_btn_group.addButton(self.btn_static, 0)
        self.mode_btn_group.addButton(self.btn_dynamic, 1)
        self.mode_btn_group.addButton(self.btn_combined, 2)

        if self._current_mode == MODE_DYNAMIC:
            self.btn_dynamic.setChecked(True)
        elif self._current_mode == MODE_COMBINED:
            self.btn_combined.setChecked(True)
        else:
            self.btn_static.setChecked(True)

        self.mode_btn_group.idClicked.connect(self._on_mode_button_clicked)

        top_ctrl_row.addWidget(self.btn_static)
        top_ctrl_row.addWidget(self.btn_dynamic)
        top_ctrl_row.addWidget(self.btn_combined)
        top_ctrl_row.addStretch()

        self.camera_status_pill = StatusPill("Camera Inactive", "neutral")
        top_ctrl_row.addWidget(self.camera_status_pill)

        cam_workspace.addLayout(top_ctrl_row)

        # Center: Camera Video Box with Stacked Inactive Overlay
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

        video_stack_layout = QStackedLayout(self.video_container)
        video_stack_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        # 1. Native High-Performance Preview Canvas
        self.video_widget = CameraPreviewWidget()
        self.video_widget.setStyleSheet("background-color: transparent;")
        self._camera_manager.set_video_widget(self.video_widget)
        video_stack_layout.addWidget(self.video_widget)

        # 2. Inactive / Standby Overlay
        self.standby_overlay = QWidget()
        overlay_layout = QVBoxLayout(self.standby_overlay)
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.setSpacing(10)

        standby_icon = QLabel()
        standby_icon.setPixmap(get_pixmap("camera", palette.text_muted, size=48, stroke_width=1.8))
        standby_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_title = QLabel("Camera Preview Inactive")
        standby_title.setStyleSheet("font-size: 16px; font-weight: 700; color: #FFFFFF;")
        standby_title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        standby_desc = QLabel(
            "Select your camera device below and click 'Start Preview' to initialize video capture.\n"
            "Supports USB webcams, Iriun Webcam, OBS Virtual Camera, and integrated laptop sensors."
        )
        standby_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_muted};")
        standby_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        overlay_layout.addWidget(standby_icon)
        overlay_layout.addWidget(standby_title)
        overlay_layout.addWidget(standby_desc)
        video_stack_layout.addWidget(self.standby_overlay)

        cam_workspace.addWidget(self.video_container, 1)

        # Bottom Camera Controls Row
        cam_bar = QFrame()
        cam_bar.setProperty("class", "cardPanelElevated")
        cam_bar_layout = QHBoxLayout(cam_bar)
        cam_bar_layout.setContentsMargins(12, 10, 12, 10)
        cam_bar_layout.setSpacing(10)

        cam_icon = QLabel()
        cam_icon.setPixmap(get_pixmap("camera", palette.text_secondary, size=18))
        cam_bar_layout.addWidget(cam_icon)

        # Searchable Camera Combobox
        self.camera_combo = SearchableCameraComboBox()
        self.camera_combo.setMinimumWidth(240)
        self.camera_combo.device_chosen.connect(self._camera_manager.select_device)
        cam_bar_layout.addWidget(self.camera_combo)

        # Refresh Cameras Button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setProperty("class", "secondaryBtn")
        self.refresh_btn.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        self.refresh_btn.setToolTip("Scan system for newly connected hardware or virtual cameras")
        self.refresh_btn.clicked.connect(self._refresh_cameras)
        cam_bar_layout.addWidget(self.refresh_btn)

        # Start/Stop Camera Buttons
        self.start_cam_btn = QPushButton("Start Camera")
        self.start_cam_btn.setProperty("class", "primaryBtn")
        self.start_cam_btn.setIcon(get_icon("play", "#FFFFFF", size=14))
        self.start_cam_btn.clicked.connect(self._camera_manager.start_preview)
        cam_bar_layout.addWidget(self.start_cam_btn)

        self.stop_cam_btn = QPushButton("Stop Camera")
        self.stop_cam_btn.setProperty("class", "secondaryBtn")
        self.stop_cam_btn.setIcon(get_icon("stop", palette.text_secondary, size=14))
        self.stop_cam_btn.setEnabled(False)
        self.stop_cam_btn.clicked.connect(self._camera_manager.stop_preview)
        cam_bar_layout.addWidget(self.stop_cam_btn)

        # Camera Settings Button
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.setProperty("class", "secondaryBtn")
        self.settings_btn.setIcon(get_icon("sliders", palette.text_secondary, palette.accent, size=14))
        self.settings_btn.clicked.connect(self._open_camera_settings)
        cam_bar_layout.addWidget(self.settings_btn)

        cam_bar_layout.addStretch()

        # FPS indicator
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_muted};")
        cam_bar_layout.addWidget(self.fps_label)

        cam_workspace.addWidget(cam_bar)

        # Architectural integration footnote
        boundary_note = QLabel(
            "Frame Boundary Note: Live video preview operates natively. "
            "Backend pipelines capture raw frames via CameraManager.frame_ready signal."
        )
        boundary_note.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-style: italic;")
        cam_workspace.addWidget(boundary_note)

        root_layout.addLayout(cam_workspace, 1)

        # =====================================================================
        # RIGHT AREA: PREDICTION PANEL
        # =====================================================================
        pred_panel = QFrame()
        pred_panel.setFixedWidth(340)
        pred_panel.setProperty("class", "cardPanel")
        
        pred_layout = QVBoxLayout(pred_panel)
        pred_layout.setContentsMargins(18, 18, 18, 18)
        pred_layout.setSpacing(14)

        # Header
        panel_header = QHBoxLayout()
        panel_title = QLabel("Prediction Engine")
        panel_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        
        self.mode_badge = QLabel(self._current_mode.upper())
        self.mode_badge.setStyleSheet(f"""
            padding: 2px 6px; border-radius: 4px;
            background-color: {palette.accent_subtle};
            color: {palette.accent}; font-size: 10px; font-weight: 700;
        """)

        panel_header.addWidget(panel_title)
        panel_header.addStretch()
        panel_header.addWidget(self.mode_badge)
        pred_layout.addLayout(panel_header)

        # Prediction Status Pill
        self.pred_status_pill = StatusPill("Backend Disconnected", "neutral")
        pred_layout.addWidget(self.pred_status_pill)

        # Recognized Sign Display Card
        sign_card = QFrame()
        sign_card.setProperty("class", "cardPanelElevated")
        sign_layout = QVBoxLayout(sign_card)
        sign_layout.setContentsMargins(14, 16, 14, 16)
        sign_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sign_layout.setSpacing(4)

        sign_hint = QLabel("RECOGNIZED SIGN")
        sign_hint.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        sign_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sign_display = QLabel("—")
        self.sign_display.setStyleSheet("font-size: 46px; font-weight: 800; padding: 4px;")
        self.sign_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.sign_sub_desc = QLabel("Stream disconnected")
        self.sign_sub_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_muted};")
        self.sign_sub_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)

        sign_layout.addWidget(sign_hint)
        sign_layout.addWidget(self.sign_display)
        sign_layout.addWidget(self.sign_sub_desc)
        pred_layout.addWidget(sign_card)

        # Metadata Row: Confidence & Type
        meta_grid = QHBoxLayout()
        meta_grid.setSpacing(10)

        # Confidence card (only shows data when actually available)
        conf_box = QFrame()
        conf_box.setProperty("class", "cardPanelElevated")
        conf_layout = QVBoxLayout(conf_box)
        conf_layout.setContentsMargins(10, 8, 10, 8)
        conf_lbl = QLabel("CONFIDENCE")
        conf_lbl.setStyleSheet(f"font-size: 9px; font-weight: 700; color: {palette.text_muted};")
        self.conf_val = QLabel("—")
        self.conf_val.setStyleSheet("font-size: 14px; font-weight: 700;")
        conf_layout.addWidget(conf_lbl)
        conf_layout.addWidget(self.conf_val)
        meta_grid.addWidget(conf_box)

        # Type card
        type_box = QFrame()
        type_box.setProperty("class", "cardPanelElevated")
        type_layout = QVBoxLayout(type_box)
        type_layout.setContentsMargins(10, 8, 10, 8)
        type_lbl = QLabel("TYPE")
        type_lbl.setStyleSheet(f"font-size: 9px; font-weight: 700; color: {palette.text_muted};")
        self.type_val = QLabel(self._current_mode)
        self.type_val.setStyleSheet("font-size: 14px; font-weight: 700;")
        type_layout.addWidget(type_lbl)
        type_layout.addWidget(self.type_val)
        meta_grid.addWidget(type_box)

        pred_layout.addLayout(meta_grid)

        # Primary Controls: Start Prediction, Stop Prediction
        ctrl_buttons = QHBoxLayout()
        ctrl_buttons.setSpacing(10)

        self.start_pred_btn = QPushButton("Start Prediction")
        self.start_pred_btn.setProperty("class", "primaryBtn")
        self.start_pred_btn.setIcon(get_icon("play", "#FFFFFF", size=14))
        self.start_pred_btn.clicked.connect(self._start_prediction)

        self.stop_pred_btn = QPushButton("Stop")
        self.stop_pred_btn.setProperty("class", "secondaryBtn")
        self.stop_pred_btn.setIcon(get_icon("stop", palette.text_secondary, size=14))
        self.stop_pred_btn.setEnabled(False)
        self.stop_pred_btn.clicked.connect(self._stop_prediction)

        ctrl_buttons.addWidget(self.start_pred_btn)
        ctrl_buttons.addWidget(self.stop_pred_btn)
        pred_layout.addLayout(ctrl_buttons)

        # Recognized Text Output Section
        text_header = QHBoxLayout()
        text_title = QLabel("Recognized Text Stream")
        text_title.setStyleSheet("font-size: 12px; font-weight: 600;")
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setProperty("class", "subtleBtn")
        self.clear_btn.setIcon(get_icon("trash", palette.text_muted, palette.danger, size=13))
        self.clear_btn.clicked.connect(self._clear_output)

        text_header.addWidget(text_title)
        text_header.addStretch()
        text_header.addWidget(self.clear_btn)
        pred_layout.addLayout(text_header)

        self.output_text_edit = QTextEdit()
        self.output_text_edit.setReadOnly(True)
        self.output_text_edit.setPlaceholderText("Recognized characters and sentences will accumulate here...")
        self.output_text_edit.setFixedHeight(95)
        self.output_text_edit.setProperty("class", "consoleText")
        pred_layout.addWidget(self.output_text_edit)

        # Recent Predictions History
        history_header = QLabel("Recent Predictions")
        history_header.setStyleSheet("font-size: 12px; font-weight: 600;")
        pred_layout.addWidget(history_header)

        self.history_list = QTextEdit()
        self.history_list.setReadOnly(True)
        self.history_list.setProperty("class", "consoleText")
        self.history_list.setPlaceholderText("No prediction history recorded.")
        self.history_list.setFixedHeight(110)
        pred_layout.addWidget(self.history_list)

        pred_layout.addStretch()
        root_layout.addWidget(pred_panel)

        # Hook up camera manager signals
        self._camera_manager.devices_changed.connect(self._on_devices_updated)
        self._camera_manager.status_changed.connect(self._on_camera_status)
        self._camera_manager.preview_started.connect(self._on_preview_started)
        self._camera_manager.preview_stopped.connect(self._on_preview_stopped)

        # Hook up backend interface signals
        self._backend.prediction_result.connect(self._on_prediction_result)
        self._backend.status_changed.connect(self._on_backend_status)
        self._backend.error_occurred.connect(self._on_backend_error)

        # Hook to backend manager updates
        BackendManager.get_instance().status_changed.connect(self._on_backend_manager_status)
        self._update_backend_state()

        # Initial device discovery
        self._refresh_cameras()

    def _on_backend_manager_status(self, status: str, severity: str):
        self._update_backend_state()

    def _update_backend_state(self):
        new_backend = BackendRegistry.get_prediction_backend()
        if new_backend != self._backend:
            try:
                self._backend.prediction_result.disconnect(self._on_prediction_result)
                self._backend.status_changed.disconnect(self._on_backend_status)
                self._backend.error_occurred.disconnect(self._on_backend_error)
            except Exception:
                pass
            self._backend = new_backend
            self._backend.prediction_result.connect(self._on_prediction_result)
            self._backend.status_changed.connect(self._on_backend_status)
            self._backend.error_occurred.connect(self._on_backend_error)

        if self._backend.is_connected():
            if self._backend.is_active():
                self.pred_status_pill.set_status("Inference Running", "active")
            else:
                self.pred_status_pill.set_status("Backend Connected", "active")
                self.sign_sub_desc.setText("Ready for live signs")
        else:
            self.pred_status_pill.set_status("Backend Disconnected", "neutral")
            self.sign_sub_desc.setText("Stream disconnected")

    def _on_mode_button_clicked(self, id_val: int):
        modes = [MODE_STATIC, MODE_DYNAMIC, MODE_COMBINED]
        if 0 <= id_val < len(modes):
            self._current_mode = modes[id_val]
            self.mode_badge.setText(self._current_mode.upper())
            self.type_val.setText(self._current_mode)
            SettingsManager.set_default_mode(self._current_mode)

    def _refresh_cameras(self):
        devices = self._camera_manager.refresh_devices()
        curr = self._camera_manager.current_device
        sel_id = curr.id_str if curr else None
        self.camera_combo.populate_devices(devices, sel_id)

    def _on_devices_updated(self, devices: List[DeviceInfo]):
        curr = self._camera_manager.current_device
        sel_id = curr.id_str if curr else None
        self.camera_combo.populate_devices(devices, sel_id)

    def _toggle_preview(self):
        if self._camera_manager.is_previewing:
            self._camera_manager.stop_preview()
        else:
            self._camera_manager.start_preview()

    def _on_preview_started(self):
        self.standby_overlay.setVisible(False)
        self.start_cam_btn.setEnabled(False)
        self.stop_cam_btn.setEnabled(True)
        self.camera_status_pill.set_status("Preview Active", "active")

    def _on_preview_stopped(self):
        self.standby_overlay.setVisible(True)
        self.video_widget.clear_frame()
        self.start_cam_btn.setEnabled(True)
        self.stop_cam_btn.setEnabled(False)
        self.camera_status_pill.set_status("Camera Inactive", "neutral")
        self.fps_label.setText("0.0 FPS")

    def _on_camera_status(self, text: str, status_type: str):
        self.camera_status_pill.set_status(text, status_type)
        if self._camera_manager.is_previewing:
            self.fps_label.setText(f"{self._camera_manager.current_fps} FPS")

    def _open_camera_settings(self):
        dialog = CameraSettingsDialog(self._camera_manager, self.window())
        dialog.exec()

    # =========================================================================
    # PREDICTION ACTIONS & BACKEND BOUNDARY
    # =========================================================================
    def _start_prediction(self):
        curr_device = self._camera_manager.current_device
        cam_id = curr_device.id_str if curr_device else ""
        
        # Invoke backend interface contract
        success = self._backend.start_prediction(self._current_mode, cam_id)
        if not success:
            # Backend is disconnected: Explain integration contract clearly
            self.pred_status_pill.set_status("Backend Disconnected", "neutral")
            
            msg = QMessageBox(self)
            msg.setWindowTitle("ML Backend Disconnected")
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Sign Language Recognition Backend is Disconnected")
            msg.setInformativeText(
                "The live prediction contract is defined in 'app/core/interfaces.py'.\n\n"
                "To connect real-time inference:\n"
                "1. Subclass PredictionBackendInterface.\n"
                "2. Integrate predict_live.py inference routines.\n"
                "3. Wire BackendRegistry.set_prediction_backend(your_instance).\n\n"
                "Prediction remains safely unavailable until the ML backend is attached."
            )
            msg.open()
        else:
            self.start_pred_btn.setEnabled(False)
            self.stop_pred_btn.setEnabled(True)
            self.pred_status_pill.set_status("Inference Running", "active")

    def _stop_prediction(self):
        self._backend.stop_prediction()
        self.start_pred_btn.setEnabled(True)
        self.stop_pred_btn.setEnabled(False)
        self.pred_status_pill.set_status("Inference Halted", "neutral")

    @Slot(object)
    def _on_prediction_result(self, result: PredictionResult):
        """Received real inference result from bound backend (no fake data)."""
        if not result:
            return
        try:
            self.sign_display.setText(result.sign)
            self.sign_sub_desc.setText(f"Detected: {result.sign}")
            self.conf_val.setText(f"{result.confidence * 100:.1f}%")
            self.type_val.setText(result.mode)

            # Accumulate recognized string
            self._text_accumulator += result.sign
            self.output_text_edit.setPlainText(self._text_accumulator)

            # Append history
            self.history_list.append(f"[{result.mode}] Sign: {result.sign} ({result.confidence * 100:.1f}%)")
        except RuntimeError:
            pass

    @Slot(str)
    def _on_backend_status(self, status_msg: str):
        try:
            self.pred_status_pill.set_status(status_msg, "neutral")
        except RuntimeError:
            pass

    @Slot(str)
    def _on_backend_error(self, err_msg: str):
        try:
            self.pred_status_pill.set_status("Backend Offline", "warning")
        except RuntimeError:
            pass

    def _clear_output(self):
        self._text_accumulator = ""
        self.output_text_edit.clear()
        self.history_list.clear()
        self.sign_display.setText("—")
        self.sign_sub_desc.setText("Output cleared")
        self.conf_val.setText("—")
