"""
CameraPreviewWidget - Native video preview canvas for camera previews and data collection.
Provides video frame painting, optional landmark overlays, status banners, and capture flash animations.
"""
from typing import Optional, Dict, Any
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QTimer, QRect, QSize
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtMultimedia import QVideoFrame

from app.core.config import SettingsManager
from app.core.theme import ThemeManager
from app.core.landmark_service import HAND_CONNECTIONS, FINGERTIP_INDICES, WRIST_INDEX


class CameraPreviewWidget(QWidget):
    """
    High-performance native video preview canvas with landmark overlay,
    dual FPS display, status banner, and capture flash.
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

                    # Draw Bone Connections (Green high contrast)
                    line_pen = QPen(QColor(0, 255, 0, 220), 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
                    painter.setPen(line_pen)

                    for start_idx, end_idx in HAND_CONNECTIONS:
                        if start_idx < len(coords) and end_idx < len(coords):
                            p1 = coords[start_idx]
                            p2 = coords[end_idx]
                            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))

                    # Draw Landmark Points (Red dots matching predict_live.py)
                    for idx, (px, py) in enumerate(coords):
                        if idx in FINGERTIP_INDICES:
                            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                            painter.setBrush(QBrush(QColor(255, 0, 0)))
                            painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                        elif idx == WRIST_INDEX:
                            painter.setPen(QPen(QColor("#FFFFFF"), 1.8))
                            painter.setBrush(QBrush(QColor(0, 230, 118)))
                            painter.drawEllipse(int(px - 5), int(py - 5), 10, 10)
                        else:
                            painter.setPen(QPen(QColor(15, 23, 42, 180), 1.0))
                            painter.setBrush(QBrush(QColor(255, 0, 0, 230)))
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

            painter.setPen(QPen(QColor(245, 180, 25, 180), 1.5))
            painter.setBrush(QBrush(QColor(15, 23, 42, 230)))
            painter.drawRoundedRect(bx, by, bw, bh, 6, 6)

            painter.setPen(QColor("#F5B419"))
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
