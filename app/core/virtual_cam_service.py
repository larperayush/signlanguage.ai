"""
Virtual Camera Output Service for SignFlow Studio.
Broadcasts camera frames with real-time sign language translation overlays
to system virtual webcams (e.g. OBS Virtual Camera, Unity Capture) so applications
like Discord, Zoom, Microsoft Teams, and Google Meet can select it as a live video feed.
"""
import time
from typing import Optional, Tuple
import numpy as np
import cv2

from PySide6.QtCore import QObject, Signal, Slot


class VirtualCameraService(QObject):
    """
    Manages broadcasting live video frames with translation HUD
    to a system virtual camera using pyvirtualcam.
    """
    status_changed = Signal(bool, str)   # (is_active, device_info_or_msg)
    error_occurred = Signal(str)

    _instance: Optional["VirtualCameraService"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cam = None
        self._is_active = False
        self._device_name = ""
        self._width = 640
        self._height = 480
        self._fps = 30
        self._show_overlay = True

    @classmethod
    def get_instance(cls) -> "VirtualCameraService":
        if cls._instance is None:
            cls._instance = VirtualCameraService()
        return cls._instance

    @property
    def is_active(self) -> bool:
        return self._is_active

    @property
    def device_name(self) -> str:
        return self._device_name

    def set_overlay_enabled(self, enabled: bool):
        self._show_overlay = enabled

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> Tuple[bool, str]:
        """Initialize and open the virtual camera output."""
        if self._is_active:
            return True, f"Virtual camera already streaming on '{self._device_name}'"

        try:
            import pyvirtualcam
            self._width = max(320, int(width))
            self._height = max(240, int(height))
            self._fps = max(10, min(60, int(fps)))

            self._cam = pyvirtualcam.Camera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                fmt=pyvirtualcam.PixelFormat.RGB
            )
            self._device_name = getattr(self._cam, "device", "Virtual Camera")
            self._is_active = True
            msg = f"Virtual Camera Active ({self._device_name})"
            self.status_changed.emit(True, msg)
            return True, msg
        except ImportError:
            msg = "pyvirtualcam is not installed."
            self._is_active = False
            self.status_changed.emit(False, msg)
            return False, msg
        except Exception as e:
            msg = f"Virtual camera error: {e}"
            self._is_active = False
            self.status_changed.emit(False, msg)
            self.error_occurred.emit(msg)
            return False, msg

    def stop(self):
        """Safely close and release virtual camera device."""
        if self._cam is not None:
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
        self._is_active = False
        self._device_name = ""
        self.status_changed.emit(False, "Virtual Camera Standby")

    def broadcast_frame(
        self,
        rgb_frame: np.ndarray,
        sign: str = "",
        confidence: float = 0.0,
        transcript: str = ""
    ):
        """
        Burn in translation HUD and transmit frame to virtual camera.
        rgb_frame must be an RGB uint8 numpy array.
        """
        if not self._is_active or self._cam is None:
            return

        try:
            h, w = rgb_frame.shape[:2]
            if h != self._height or w != self._width:
                frame_out = cv2.resize(rgb_frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
            else:
                frame_out = rgb_frame.copy()

            if self._show_overlay:
                frame_out = self._draw_translation_overlay(frame_out, sign, confidence, transcript)

            self._cam.send(frame_out)
        except Exception:
            # Drop silently on frame timing blip to prevent lag
            pass

    def _draw_translation_overlay(
        self,
        frame: np.ndarray,
        sign: str,
        confidence: float,
        transcript: str
    ) -> np.ndarray:
        """
        Draw clean, editorial typography HUD on the virtual camera frame
        matching the warm cream & espresso charcoal design.
        """
        h, w = frame.shape[:2]

        # 1. Top-Left Badge: [ SignFlow AI ]
        badge_text = "SignFlow AI"
        cv2.rectangle(frame, (16, 16), (140, 44), (22, 21, 19), -1)  # #161513 deep charcoal
        cv2.rectangle(frame, (16, 16), (140, 44), (61, 56, 51), 1)   # #3D3833 border
        cv2.circle(frame, (30, 30), 4, (25, 180, 245), -1)           # Amber gold dot #F5B419 (in RGB)
        cv2.putText(
            frame, badge_text, (42, 34),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 180, 25), 1, cv2.LINE_AA
        )

        # 2. Bottom Caption & Transcript Bar
        bar_height = 70
        y_start = h - bar_height - 14
        y_end = h - 14
        x_start = 16
        x_end = w - 16

        # Draw semi-transparent dark charcoal card
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), (22, 21, 19), -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (122, 94, 42), 1)  # #7A5E2A border

        # Recognized Sign Display (Large Amber Gold text)
        display_sign = sign if sign and sign not in ("No Hand", "Uncertain") else "—"
        sign_label = f"SIGN: {display_sign}"
        cv2.putText(
            frame, sign_label, (x_start + 18, y_start + 34),
            cv2.FONT_HERSHEY_DUPLEX, 0.82, (25, 180, 245), 2, cv2.LINE_AA  # Warm gold
        )

        if confidence > 0.0 and display_sign != "—":
            conf_str = f"({int(confidence * 100)}%)"
            cv2.putText(
                frame, conf_str, (x_start + 180, y_start + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (197, 208, 214), 1, cv2.LINE_AA
            )

        # Live Transcript Line
        text_disp = transcript[-40:] if transcript else "Translating live sign language..."
        cv2.putText(
            frame, text_disp, (x_start + 18, y_start + 58),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (247, 251, 253), 1, cv2.LINE_AA  # Warm cream white
        )

        return frame
