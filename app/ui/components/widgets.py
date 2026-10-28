"""
Reusable UI primitives: Status pills, Cards, Badges, and Section headers.
Adheres strictly to the Linear / developer tool design system.
"""
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QPainter, QBrush, QPen

from app.resources.icons import get_icon, get_pixmap
from app.core.theme import ThemeManager


class StatusPill(QFrame):
    """
    Compact status indicator with colored dot and text.
    status_type: 'neutral' | 'active' | 'warning' | 'error' | 'accent'
    """
    def __init__(self, text: str = "Ready", status_type: str = "neutral", parent=None):
        super().__init__(parent)
        self._text = text
        self._type = status_type

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(8, 3, 10, 3)
        self.layout.setSpacing(6)

        # Dot
        self.dot_label = QLabel()
        self.dot_label.setFixedSize(8, 8)

        # Text
        self.text_label = QLabel(text)
        self.text_label.setStyleSheet("font-size: 11px; font-weight: 600;")

        self.layout.addWidget(self.dot_label)
        self.layout.addWidget(self.text_label)

        self._update_style()

    def text(self) -> str:
        return self._text

    def set_status(self, text: str, status_type: str = "neutral"):
        self._text = text
        self._type = status_type
        self.text_label.setText(text)
        self._update_style()

    def _update_style(self):
        palette = ThemeManager.get_palette()
        dot_colors = {
            "active": palette.success,
            "warning": palette.warning,
            "error": palette.danger,
            "accent": palette.accent,
            "neutral": palette.text_muted
        }
        bg_colors = {
            "active": palette.success_subtle,
            "warning": palette.warning_subtle,
            "error": palette.danger_subtle,
            "accent": palette.accent_subtle,
            "neutral": palette.bg_hover
        }
        text_colors = {
            "active": palette.success,
            "warning": palette.warning,
            "error": palette.danger,
            "accent": palette.accent,
            "neutral": palette.text_secondary
        }

        dot_c = dot_colors.get(self._type, palette.text_muted)
        bg_c = bg_colors.get(self._type, palette.bg_hover)
        txt_c = text_colors.get(self._type, palette.text_secondary)

        self.dot_label.setStyleSheet(f"""
            background-color: {dot_c};
            border-radius: 4px;
        """)
        self.text_label.setStyleSheet(f"""
            color: {txt_c};
            font-size: 11px;
            font-weight: 600;
        """)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_c};
                border: 1px solid {dot_c}33;
                border-radius: 12px;
            }}
        """)


class StatCard(QFrame):
    """Refined metrics card displaying key project/hardware statistics."""
    def __init__(self, title: str, value: str, subtitle: str = "", icon_name: str = "info", parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setProperty("class", "cardPanel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(6)

        # Header with title and icon
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet("font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        
        palette = ThemeManager.get_palette()
        self.icon_lbl = QLabel()
        self.icon_lbl.setPixmap(get_pixmap(icon_name, palette.text_muted, size=16))

        top_layout.addWidget(self.title_lbl)
        top_layout.addStretch()
        top_layout.addWidget(self.icon_lbl)

        # Value
        self.val_lbl = QLabel(value)
        self.val_lbl.setStyleSheet("font-size: 20px; font-weight: 700;")

        # Subtitle
        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_muted};")

        layout.addLayout(top_layout)
        layout.addWidget(self.val_lbl)
        if subtitle:
            layout.addWidget(self.sub_lbl)

    def set_value(self, val: str, subtitle: str = ""):
        self.val_lbl.setText(val)
        if subtitle:
            self.sub_lbl.setText(subtitle)


class SectionHeader(QWidget):
    """Consistent page/section header with title and subtitle."""
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 700;")

        layout.addWidget(self.title_label)

        if subtitle:
            self.sub_label = QLabel(subtitle)
            palette = ThemeManager.get_palette()
            self.sub_label.setStyleSheet(f"font-size: 13px; color: {palette.text_secondary};")
            self.sub_label.setWordWrap(True)
            layout.addWidget(self.sub_label)
