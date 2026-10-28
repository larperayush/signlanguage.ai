"""
Application-wide Error Handling System.
Provides professional error dialogs, technical detail drawers,
and inline alert banners. Strictly avoids ambiguous messages.
"""
import traceback
from typing import Optional, Callable
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QFrame, QWidget
)
from PySide6.QtCore import Qt

from app.core.theme import ThemeManager
from app.resources.icons import get_icon, get_pixmap


class InlineAlertBanner(QFrame):
    """
    Inline alert / error banner for embedding within workspaces.
    severity: 'error' | 'warning' | 'info'
    """
    def __init__(self, title: str, message: str, severity: str = "warning", parent=None):
        super().__init__(parent)
        self.setProperty("class", "cardPanelElevated")

        palette = ThemeManager.get_palette()
        border_colors = {
            "error": palette.danger,
            "warning": palette.warning,
            "info": palette.accent
        }
        bg_colors = {
            "error": palette.danger_subtle,
            "warning": palette.warning_subtle,
            "info": palette.accent_subtle
        }
        icon_names = {
            "error": "alert",
            "warning": "alert",
            "info": "info"
        }

        border_c = border_colors.get(severity, palette.warning)
        bg_c = bg_colors.get(severity, palette.warning_subtle)

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_c};
                border: 1px solid {border_c};
                border-radius: 6px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        icon_lbl = QLabel()
        icon_lbl.setPixmap(get_pixmap(icon_names.get(severity, "info"), border_c, size=20))

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {border_c};")

        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_primary};")
        msg_lbl.setWordWrap(True)

        text_layout.addWidget(title_lbl)
        text_layout.addWidget(msg_lbl)

        layout.addWidget(icon_lbl)
        layout.addLayout(text_layout, 1)


class ErrorDialog(QDialog):
    """
    Professional desktop error dialog with actionable guidance
    and an expandable technical diagnostic drawer.
    """
    def __init__(
        self,
        title: str,
        message: str,
        severity: str = "error",
        technical_details: Optional[str] = None,
        suggested_action: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(480)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        palette = ThemeManager.get_palette()
        self._technical_details = technical_details

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header with icon + title + message
        header_row = QHBoxLayout()
        header_row.setSpacing(14)

        icon_color = palette.danger if severity == "error" else palette.warning
        icon_name = "alert" if severity in ("error", "warning") else "info"

        icon_lbl = QLabel()
        icon_lbl.setPixmap(get_pixmap(icon_name, icon_color, size=32))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)

        text_vbox = QVBoxLayout()
        text_vbox.setSpacing(6)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 700;")

        msg_lbl = QLabel(message)
        msg_lbl.setStyleSheet(f"font-size: 13px; color: {palette.text_secondary}; line-height: 1.4;")
        msg_lbl.setWordWrap(True)

        text_vbox.addWidget(title_lbl)
        text_vbox.addWidget(msg_lbl)

        if suggested_action:
            action_lbl = QLabel(f"Recommended action: {suggested_action}")
            action_lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {palette.accent};")
            action_lbl.setWordWrap(True)
            text_vbox.addWidget(action_lbl)

        header_row.addWidget(icon_lbl)
        header_row.addLayout(text_vbox, 1)
        main_layout.addLayout(header_row)

        # Expandable Technical Details Drawer
        if technical_details:
            self.details_btn = QPushButton("▸ Show Technical Details")
            self.details_btn.setProperty("class", "subtleBtn")
            self.details_btn.setStyleSheet("text-align: left; padding: 4px 0px; font-size: 12px;")
            self.details_btn.clicked.connect(self._toggle_details)
            main_layout.addWidget(self.details_btn)

            self.details_box = QTextEdit()
            self.details_box.setReadOnly(True)
            self.details_box.setProperty("class", "consoleText")
            self.details_box.setPlainText(technical_details)
            self.details_box.setFixedHeight(120)
            self.details_box.setVisible(False)
            main_layout.addWidget(self.details_box)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        dismiss_btn = QPushButton("Dismiss")
        dismiss_btn.setProperty("class", "secondaryBtn")
        dismiss_btn.clicked.connect(self.accept)
        btn_layout.addWidget(dismiss_btn)

        main_layout.addLayout(btn_layout)

    def _toggle_details(self):
        is_visible = not self.details_box.isVisible()
        self.details_box.setVisible(is_visible)
        self.details_btn.setText("▾ Hide Technical Details" if is_visible else "▸ Show Technical Details")
        self.adjustSize()

    # Preset factory helpers for consistent error handling across the application
    @classmethod
    def show_missing_backend(cls, path: str, parent=None):
        dlg = cls(
            title="Backend Folder Not Found",
            message=f"The registered ML backend directory does not exist or has been moved.",
            severity="warning",
            technical_details=f"Target path: {path}\nFilesystem check failed at os.path.exists().",
            suggested_action="Select an active Python ML backend folder using the Backend Import Manager.",
            parent=parent
        )
        return dlg.exec()

    @classmethod
    def show_invalid_csv(cls, path: str, reason: str, parent=None):
        dlg = cls(
            title="Invalid CSV Dataset",
            message=f"Failed to parse or inspect the specified dataset file: {reason}",
            severity="error",
            technical_details=f"File: {path}\nParser Exception: {reason}",
            suggested_action="Ensure the file is a valid UTF-8 CSV containing numeric features or landmark coordinates.",
            parent=parent
        )
        return dlg.exec()

    @classmethod
    def show_empty_dataset(cls, path: str, parent=None):
        dlg = cls(
            title="Dataset Is Empty",
            message="The specified dataset contains no sample rows.",
            severity="warning",
            technical_details=f"File: {path}\nRow count evaluation returned 0.",
            suggested_action="Collect or import training data before initializing the training workspace.",
            parent=parent
        )
        return dlg.exec()

    @classmethod
    def show_unsupported_model(cls, path: str, extension: str, parent=None):
        dlg = cls(
            title="Unsupported Model Format",
            message=f"The file extension '{extension}' is not recognized as a supported ML model artifact.",
            severity="warning",
            technical_details=f"File: {path}\nExtension: {extension}\nSupported: .joblib, .task, .h5, .onnx, .pt, .tflite",
            suggested_action="Select a recognized trained model file.",
            parent=parent
        )
        return dlg.exec()

    @classmethod
    def show_camera_error(cls, device_name: str, error_detail: str, parent=None):
        dlg = cls(
            title="Camera Device Unavailable",
            message=f"Could not open video stream from '{device_name}'. The device may be in use by another application or permissions are restricted.",
            severity="error",
            technical_details=f"Device: {device_name}\nError: {error_detail}",
            suggested_action="Close other camera apps (OBS, Zoom, Windows Camera) or choose an alternate camera.",
            parent=parent
        )
        return dlg.exec()

    @classmethod
    def show_exception(cls, title: str, exc: Exception, parent=None):
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        dlg = cls(
            title=title,
            message=str(exc) or "An unexpected frontend exception occurred.",
            severity="error",
            technical_details=tb,
            suggested_action="Check the technical details drawer or restart the application workspace.",
            parent=parent
        )
        return dlg.exec()
