"""
Compact Top Toolbar Component.
Includes Page Title, Model Selection Dropdown, Connect Apps Modal Trigger,
Backend Status Pill, and Theme Switcher.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QComboBox, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from app.core.config import MODEL_SELECTIONS, SettingsManager
from app.core.theme import ThemeManager
from app.core.backend_manager import BackendManager, STATUS_CONNECTED, STATUS_INTEGRATION_PENDING, STATUS_FILE_VALIDATED, STATUS_CONNECTION_ERROR
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill
from app.ui.components.connect_apps_dialog import ConnectAppsDialog
from app.ui.components.backend_import_dialog import BackendImportDialog


class TopBar(QWidget):
    """Refined application header toolbar."""
    theme_toggled = Signal(bool)          # Emits True if dark mode
    model_changed = Signal(str)           # Emits selected model name
    connect_apps_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBarContainer")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(14)

        # Page Breadcrumb / Title
        self.breadcrumb_root = QLabel("Studio")
        self.breadcrumb_root.setStyleSheet("font-size: 13px; font-weight: 500;")
        
        self.separator = QLabel("/")
        palette = ThemeManager.get_palette()
        self.separator.setStyleSheet(f"font-size: 13px; color: {palette.text_muted};")

        self.page_title = QLabel("Overview")
        self.page_title.setStyleSheet("font-size: 13px; font-weight: 600;")

        layout.addWidget(self.breadcrumb_root)
        layout.addWidget(self.separator)
        layout.addWidget(self.page_title)

        layout.addStretch()

        # Backend Connection Badge & Setup Button
        self._backend_manager = BackendManager.get_instance()
        self.backend_status = StatusPill(f"ML Core: {self._backend_manager.status}", self._get_status_severity(self._backend_manager.status))
        self.backend_status.setToolTip("Click 'Backend Setup' to configure Python ML project path and mappings.")
        layout.addWidget(self.backend_status)

        self.backend_btn = QPushButton(" Backend Setup")
        self.backend_btn.setProperty("class", "secondaryBtn")
        self.backend_btn.setIcon(get_icon("folder", palette.text_secondary, palette.accent, size=14))
        self.backend_btn.setToolTip("Open Backend Import Manager to register or replace external ML folder")
        self.backend_btn.clicked.connect(self._open_backend_manager)
        layout.addWidget(self.backend_btn)

        self._backend_manager.status_changed.connect(self._on_backend_status_changed)

        # Global Model Selector Dropdown
        model_lbl = QLabel("Model:")
        model_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        layout.addWidget(model_lbl)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(210)
        for m in MODEL_SELECTIONS:
            self.model_combo.addItem(m)

        saved_model = SettingsManager.get_global_model()
        idx = self.model_combo.findText(saved_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        layout.addWidget(self.model_combo)

        # Connect Apps + Button
        self.connect_apps_btn = QPushButton(" Connect Apps +")
        self.connect_apps_btn.setProperty("class", "secondaryBtn")
        self.connect_apps_btn.setIcon(get_icon("apps", palette.text_secondary, palette.accent, size=15))
        self.connect_apps_btn.clicked.connect(self._open_connect_apps)
        layout.addWidget(self.connect_apps_btn)

        # Theme Switcher Button
        self.theme_btn = QPushButton()
        self.theme_btn.setProperty("class", "subtleBtn")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setToolTip("Toggle Light/Dark Appearance")
        self.theme_btn.clicked.connect(self._toggle_theme)
        self._update_theme_icon()
        layout.addWidget(self.theme_btn)

    def set_page_title(self, title: str):
        self.page_title.setText(title)

    def _on_model_changed(self, text: str):
        SettingsManager.set_global_model(text)
        self.model_changed.emit(text)

    def _open_connect_apps(self):
        self.connect_apps_clicked.emit()
        dialog = ConnectAppsDialog(self.window())
        dialog.exec()

    def _open_backend_manager(self):
        dialog = BackendImportDialog(self.window())
        dialog.exec()

    def _on_backend_status_changed(self, status: str, severity: str):
        self.backend_status.set_status(f"ML Core: {status}", severity)

    @staticmethod
    def _get_status_severity(status: str) -> str:
        if status == STATUS_CONNECTED:
            return "active"
        elif status in (STATUS_INTEGRATION_PENDING, STATUS_FILE_VALIDATED):
            return "warning"
        elif status == STATUS_CONNECTION_ERROR:
            return "error"
        return "neutral"

    def _toggle_theme(self):
        is_currently_dark = (SettingsManager.get_theme() == "dark")
        new_is_dark = not is_currently_dark
        SettingsManager.set_theme("dark" if new_is_dark else "light")
        self._update_theme_icon()
        self.theme_toggled.emit(new_is_dark)

    def _update_theme_icon(self):
        is_dark = (SettingsManager.get_theme() == "dark")
        palette = ThemeManager.get_palette()
        icon_name = "sun" if is_dark else "moon"
        self.theme_btn.setIcon(get_icon(icon_name, palette.text_secondary, palette.accent, size=18))
