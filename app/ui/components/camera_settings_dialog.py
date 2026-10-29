"""
Camera Device Settings Dialog for SignFlow Studio.
Configures resolution format negotiation, preview mirroring, and hardware controls.
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QComboBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer

from app.core.camera_manager import CameraManager
from app.core.theme import ThemeManager
from app.core.config import SettingsManager


class CameraSettingsDialog(QDialog):
    """Camera hardware preferences and stream resolution format switcher."""
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

        mirror_hint = QLabel("Applies visual mirroring exclusively to the preview canvas.")
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

        self._camera_manager.resolution_changed.connect(self._on_resolution_verified)

    def _on_mirror_toggled(self, checked: bool):
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
        self._camera_manager.set_mirror_preview(is_mirrored)

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
