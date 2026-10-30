"""
EmbeddedCameraCanvas - High-performance native video canvas widget for SignFlow Studio.
Renders 60 FPS live video frames with MediaPipe skeletons and HUD overlays,
handles aspect-ratio preservation, and captures Space/Backspace/C shortcut keys.
"""
from typing import Optional
from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt, Signal, QRect, QSize
from PySide6.QtGui import QPainter, QImage, QColor, QFont, QPen, QBrush

from app.core.theme import ThemeManager
from app.resources.icons import get_pixmap


class EmbeddedCameraCanvas(QWidget):
    """
    Native video canvas that renders live inference frames directly inside the app.
    Displays standby HUD when idle, and handles keyboard controls for typing sentences.
    """
    key_space_pressed = Signal()
    key_backspace_pressed = Signal()
    key_clear_pressed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_frame: Optional[QImage] = None
        self._is_active: bool = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(640, 420)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def set_active(self, active: bool):
        self._is_active = active
        if not active:
            self._current_frame = None
        self.update()

    def update_frame(self, image: QImage):
        """Receive new rendered frame from the background inference worker."""
        self._current_frame = image
        self._is_active = True
        self.update()

    def clear(self):
        self._current_frame = None
        self._is_active = False
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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        palette = ThemeManager.get_palette()

        # Fill background with dark video backdrop
        bg_color = QColor(palette.video_bg) if hasattr(palette, "video_bg") else QColor("#12141A")
        painter.fillRect(self.rect(), bg_color)

        # 1. Active Video Stream
        if self._is_active and self._current_frame is not None and not self._current_frame.isNull():
            target_rect = self._compute_target_rect(self.rect(), self._current_frame.size())
            painter.drawImage(target_rect, self._current_frame)
            return

        # 2. Standby / Inactive Placeholder
        w = self.width()
        h = self.height()
        cx = w // 2
        cy = h // 2

        # Standby Box
        box_w = min(460, w - 40)
        box_h = 170
        bx = cx - box_w // 2
        by = cy - box_h // 2

        painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
        painter.setBrush(QBrush(QColor(22, 24, 30, 220)))
        painter.drawRoundedRect(bx, by, box_w, box_h, 12, 12)

        # Camera Icon
        icon_pixmap = get_pixmap("camera", palette.text_muted, size=40, stroke_width=1.6)
        if icon_pixmap:
            painter.drawPixmap(cx - 20, by + 22, icon_pixmap)

        # Title
        painter.setPen(QColor("#FFFFFF"))
        font_title = painter.font()
        font_title.setPointSize(12)
        font_title.setWeight(QFont.Weight.Bold)
        painter.setFont(font_title)
        fm = painter.fontMetrics()
        title_text = "Live Translation Ready"
        painter.drawText(cx - fm.horizontalAdvance(title_text) // 2, by + 84, title_text)

        # Description
        painter.setPen(QColor("#94A3B8"))
        font_desc = painter.font()
        font_desc.setPointSize(9)
        font_desc.setWeight(QFont.Weight.Normal)
        painter.setFont(font_desc)
        fm_desc = painter.fontMetrics()
        desc_l1 = "Select your model and click 'Start Translation'"
        desc_l2 = "Video output with green landmarks and HUD will appear here"
        painter.drawText(cx - fm_desc.horizontalAdvance(desc_l1) // 2, by + 108, desc_l1)
        painter.drawText(cx - fm_desc.horizontalAdvance(desc_l2) // 2, by + 126, desc_l2)

        # Focus indicator border if focused
        if self.hasFocus():
            painter.setPen(QPen(QColor(palette.accent), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 6, 6)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.key_space_pressed.emit()
            event.accept()
        elif key == Qt.Key.Key_Backspace:
            self.key_backspace_pressed.emit()
            event.accept()
        elif key == Qt.Key.Key_C:
            self.key_clear_pressed.emit()
            event.accept()
        else:
            super().keyPressEvent(event)
