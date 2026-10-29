"""
Compact Top Toolbar Component for SignFlow Studio.
Displays Breadcrumb Page Title on left, and Theme Switcher + Settings Gear Button on right.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt, Signal

from app.core.config import SettingsManager
from app.core.theme import ThemeManager
from app.resources.icons import get_icon


class TopBar(QWidget):
    """Refined minimalist application header toolbar."""
    theme_toggled = Signal(bool)          # Emits True if dark mode
    settings_clicked = Signal()           # Opens Settings & Tools modal

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("topBarContainer")
        self.setFixedHeight(52)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(12)

        # Page Breadcrumb / Title
        self.breadcrumb_root = QLabel("SignFlow AI")
        self.breadcrumb_root.setStyleSheet("font-size: 13px; font-weight: 500;")

        self.separator = QLabel("/")
        palette = ThemeManager.get_palette()
        self.separator.setStyleSheet(f"font-size: 13px; color: {palette.text_muted};")

        self.page_title = QLabel("Live Sign Language Translation")
        self.page_title.setStyleSheet("font-size: 13px; font-weight: 600;")

        layout.addWidget(self.breadcrumb_root)
        layout.addWidget(self.separator)
        layout.addWidget(self.page_title)

        layout.addStretch()

        # 1. Dark Mode / Theme Switcher Button
        self.theme_btn = QPushButton()
        self.theme_btn.setProperty("class", "subtleBtn")
        self.theme_btn.setFixedSize(36, 36)
        self.theme_btn.setToolTip("Toggle Light/Dark Theme")
        self.theme_btn.clicked.connect(self._toggle_theme)
        self._update_theme_icon()
        layout.addWidget(self.theme_btn)

        # 2. Settings Gear Button (right beside the dark mode button)
        self.settings_btn = QPushButton()
        self.settings_btn.setProperty("class", "subtleBtn")
        self.settings_btn.setFixedSize(36, 36)
        self.settings_btn.setToolTip("Settings & Developer Tools")
        self.settings_btn.setIcon(get_icon("settings", palette.text_secondary, palette.accent, size=18))
        self.settings_btn.clicked.connect(self._on_settings_clicked)
        layout.addWidget(self.settings_btn)

    def set_page_title(self, title: str):
        self.page_title.setText(title)

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
        if hasattr(self, "settings_btn"):
            self.settings_btn.setIcon(get_icon("settings", palette.text_secondary, palette.accent, size=18))

    def _on_settings_clicked(self):
        self.settings_clicked.emit()
