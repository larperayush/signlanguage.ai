"""
Robust Native Camera Device Manager using Qt Multimedia (QMediaDevices, QCamera).
Supports dynamic device discovery, virtual cameras (OBS, Iriun, DroidCam),
safe camera switching, genuine resolution format negotiation, frame verification,
and live horizontal preview mirroring without altering raw frames emitted to ML pipelines.
"""
from typing import List, Optional, Tuple, Any
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal, QByteArray, QSize
from PySide6.QtMultimedia import (
    QMediaDevices,
    QCameraDevice,
    QCamera,
    QMediaCaptureSession,
    QVideoSink,
    QVideoFrame,
    QCameraFormat,
)

from app.core.config import SettingsManager


@dataclass
class DeviceInfo:
    id_str: str
    description: str
    is_default: bool
    device: QCameraDevice
    supported_resolutions: List[str]


class CameraManager(QObject):
    """
    Centralized controller for native camera capture and device discovery.
    Provides safe lifecycle management, genuine resolution configuration,
    frame dimension verification, and clean frame-tapping boundary.
    """
    devices_changed = Signal(list)         # List[DeviceInfo]
    device_selected = Signal(str)          # description
    preview_started = Signal()
    preview_stopped = Signal()
    status_changed = Signal(str, str)      # (status_text, status_type: info|active|warning|error)
    resolution_changed = Signal(int, int)  # (actual_width, actual_height)
    mirror_changed = Signal(bool)          # is_mirrored
    error_occurred = Signal(str)
    frame_ready = Signal(object)           # Raw un-mirrored QVideoFrame tap for ML backend

    _instance: Optional["CameraManager"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        CameraManager._instance = self
        self._devices: List[DeviceInfo] = []
        self._current_device_info: Optional[DeviceInfo] = None
        self._camera: Optional[QCamera] = None
        self._capture_session: QMediaCaptureSession = QMediaCaptureSession(self)
        self._video_widget: Optional[Any] = None
        self._video_sink: QVideoSink = QVideoSink(self)
        self._is_previewing = False

        # Resolution & stream verification state
        self._actual_resolution: Tuple[int, int] = (0, 0)
        self._current_format: Optional[QCameraFormat] = None
        self._preferred_resolution: str = SettingsManager.get_preferred_resolution()

        # Connect session sink to receive native video frames
        self._capture_session.setVideoSink(self._video_sink)
        self._video_sink.videoFrameChanged.connect(self._on_video_frame)

        # FPS tracking
        self._frame_counter = 0
        self._last_fps_time = time.time()
        self._current_fps = 0.0

        # Hook to hardware change events
        self._media_devices = QMediaDevices(self)
        self._media_devices.videoInputsChanged.connect(self.refresh_devices)

    @classmethod
    def get_instance(cls) -> Optional["CameraManager"]:
        return cls._instance

    @property
    def devices(self) -> List[DeviceInfo]:
        return list(self._devices)

    @property
    def is_previewing(self) -> bool:
        return self._is_previewing

    @property
    def current_device(self) -> Optional[DeviceInfo]:
        return self._current_device_info

    @property
    def current_fps(self) -> float:
        return self._current_fps

    @property
    def actual_resolution(self) -> Tuple[int, int]:
        """Verified dimensions of delivered camera video frames."""
        return self._actual_resolution

    @property
    def current_format(self) -> Optional[QCameraFormat]:
        return self._current_format

    @property
    def preferred_resolution(self) -> str:
        return self._preferred_resolution

    def set_video_widget(self, widget: Any):
        """Bind video preview widget to the camera pipeline."""
        if self._video_widget and self._video_widget != widget:
            if hasattr(self._video_widget, "clear_frame"):
                try:
                    self._video_widget.clear_frame()
                except Exception:
                    pass

        self._video_widget = widget
        if hasattr(widget, "set_mirrored"):
            widget.set_mirrored(SettingsManager.get_mirror_preview())

        # If it's a widget using custom set_frame, session uses our video sink
        if hasattr(widget, "set_frame"):
            self._capture_session.setVideoSink(self._video_sink)
        elif hasattr(widget, "videoSink"):
            # Fallback for standard QVideoWidget
            self._capture_session.setVideoOutput(widget)
            sink = widget.videoSink()
            if sink and sink != self._video_sink:
                self._video_sink = sink
                self._video_sink.videoFrameChanged.connect(self._on_video_frame)

    def set_mirror_preview(self, mirror: bool):
        """
        Immediately toggle horizontal visual mirroring on the live preview.
        Persists to QSettings without modifying the raw frame delivered to ML.
        """
        SettingsManager.set_mirror_preview(mirror)
        if self._video_widget and hasattr(self._video_widget, "set_mirrored"):
            self._video_widget.set_mirrored(mirror)
        self.mirror_changed.emit(mirror)

    def _on_video_frame(self, frame: QVideoFrame):
        """Process incoming video frame, verify dimensions, and deliver stream."""
        if not frame.isValid():
            return

        self._frame_counter += 1
        now = time.time()
        delta = now - self._last_fps_time
        if delta >= 1.0:
            self._current_fps = round(self._frame_counter / delta, 1)
            self._frame_counter = 0
            self._last_fps_time = now
            if self._is_previewing and self._current_device_info:
                w, h = self._actual_resolution
                res_info = f"{w}x{h}" if w > 0 else ""
                self.status_changed.emit(
                    f"Streaming {res_info} ({self._current_fps} FPS)".strip(), "active"
                )

        # Track and verify delivered frame dimensions
        w = frame.width()
        h = frame.height()
        if (w, h) != self._actual_resolution:
            self._actual_resolution = (w, h)
            self.resolution_changed.emit(w, h)

        # Deliver to preview widget for visual rendering (mirrored or normal)
        if self._video_widget and hasattr(self._video_widget, "set_frame"):
            self._video_widget.set_frame(frame)

        # Emit raw un-mirrored frame tap for ML prediction pipeline
        self.frame_ready.emit(frame)

    def refresh_devices(self) -> List[DeviceInfo]:
        """
        Enumerate all connected hardware and virtual video input devices.
        Never hardcodes camera names.
        """
        raw_inputs = QMediaDevices.videoInputs()
        self._devices = []

        for dev in raw_inputs:
            id_bytes = bytes(dev.id())
            id_str = id_bytes.decode("utf-8", errors="replace")
            desc = dev.description() or "Unknown Camera Device"

            # Query supported formats / resolutions
            formats = dev.videoFormats()
            res_set = set()
            for f in formats:
                sz = f.resolution()
                if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                    res_set.add(f"{sz.width()}x{sz.height()}")

            sorted_res = sorted(
                list(res_set),
                key=lambda x: int(x.split("x")[0]) * int(x.split("x")[1]),
                reverse=True
            )

            info = DeviceInfo(
                id_str=id_str,
                description=desc,
                is_default=dev.isDefault(),
                device=dev,
                supported_resolutions=sorted_res
            )
            self._devices.append(info)

        self.devices_changed.emit(self._devices)

        # Check if currently selected device is still present
        if self._current_device_info:
            found = any(d.id_str == self._current_device_info.id_str for d in self._devices)
            if not found:
                self.stop_preview()
                self._current_device_info = None
                self.status_changed.emit("Previous camera disconnected", "warning")

        # Resolve preferred camera from QSettings or fallback to default
        if not self._current_device_info and self._devices:
            preferred_id = SettingsManager.get_preferred_camera_id()
            matched = next((d for d in self._devices if d.id_str == preferred_id), None)
            if not matched:
                matched = next((d for d in self._devices if d.is_default), self._devices[0])
            self.select_device(matched.id_str)
        elif not self._devices:
            self.status_changed.emit("No camera devices detected", "error")

        return self._devices

    def select_device(self, id_str: str) -> bool:
        """Select a camera device by its unique ID string."""
        target = next((d for d in self._devices if d.id_str == id_str), None)
        if not target:
            return False

        if self._current_device_info and self._current_device_info.id_str == target.id_str and self._camera:
            return True

        was_previewing = self._is_previewing
        if was_previewing:
            self.stop_preview()

        self._release_camera()
        self._current_device_info = target
        SettingsManager.set_preferred_camera_id(target.id_str, target.description)
        self.device_selected.emit(target.description)
        self.status_changed.emit(f"Selected: {target.description}", "info")

        if was_previewing:
            self.start_preview()

        return True

    def select_device_by_index(self, index: int) -> bool:
        if 0 <= index < len(self._devices):
            return self.select_device(self._devices[index].id_str)
        return False

    def start_preview(self) -> bool:
        """Safely start the live video preview with configured resolution."""
        if not self._current_device_info:
            self.error_occurred.emit("No video device selected.")
            self.status_changed.emit("No device selected", "error")
            return False

        try:
            if not self._camera:
                self._camera = QCamera(self._current_device_info.device, self)
                self._camera.errorOccurred.connect(self._on_camera_error)
                self._capture_session.setCamera(self._camera)

                # Apply preferred resolution format if supported by this device
                pref_res = self._preferred_resolution or SettingsManager.get_preferred_resolution()
                formats = self._current_device_info.device.videoFormats()
                if pref_res and "x" in pref_res and formats:
                    try:
                        w, h = map(int, pref_res.split("x")[:2])
                        matched = self._find_best_format(formats, w, h)
                        if matched:
                            self._camera.setCameraFormat(matched)
                            self._current_format = matched
                    except Exception:
                        pass

            self._camera.start()
            self._is_previewing = True
            self._last_fps_time = time.time()
            self._frame_counter = 0
            self.preview_started.emit()
            self.status_changed.emit(f"Preview Active ({self._current_device_info.description})", "active")
            return True
        except Exception as e:
            err_msg = f"Failed to start camera preview: {str(e)}"
            self.error_occurred.emit(err_msg)
            self.status_changed.emit("Camera Error", "error")
            self._is_previewing = False
            return False

    def stop_preview(self):
        """Safely stop video preview and release stream."""
        if self._camera and self._camera.isActive():
            self._camera.stop()
        self._is_previewing = False
        self._current_fps = 0.0
        self._actual_resolution = (0, 0)
        if self._video_widget and hasattr(self._video_widget, "clear_frame"):
            self._video_widget.clear_frame()
        self.preview_stopped.emit()
        self.status_changed.emit("Camera Inactive", "info")

    def _release_camera(self):
        """Explicitly tear down camera object to free system hardware locks."""
        if self._camera:
            try:
                if self._camera.isActive():
                    self._camera.stop()
                self._capture_session.setCamera(None)
                self._camera.errorOccurred.disconnect()
                self._camera.deleteLater()
            except Exception:
                pass
            finally:
                self._camera = None
                self._current_format = None
                self._actual_resolution = (0, 0)

    def _on_camera_error(self, error: QCamera.Error, error_string: str):
        err = f"Camera Hardware Alert: {error_string} (code: {error})"
        self.error_occurred.emit(err)
        self.status_changed.emit(f"Hardware Error: {error_string}", "error")
        self._is_previewing = False
        self.preview_stopped.emit()

    def set_resolution(self, res_str: str) -> Tuple[bool, str, str]:
        """
        Safely apply video capture resolution format.
        Returns: (success: bool, message: str, effective_res: str)
        """
        if not self._current_device_info:
            return False, "No camera device is currently selected.", ""

        if not res_str or "x" not in res_str:
            return False, f"Invalid resolution format: '{res_str}'.", ""

        try:
            parts = res_str.split("x")
            w = int(parts[0].strip())
            h = int(parts[1].split()[0].strip())
        except Exception as e:
            return False, f"Could not parse resolution dimensions: {e}", ""

        formats = self._current_device_info.device.videoFormats()
        if not formats:
            return False, "This camera driver does not expose discrete resolution formats (managed by driver).", ""

        matched_format = self._find_best_format(formats, w, h)
        if not matched_format:
            return False, f"Resolution {w}x{h} is not supported by {self._current_device_info.description}.", ""

        previous_format = self._current_format
        prev_res = f"{self._actual_resolution[0]}x{self._actual_resolution[1]}" if self._actual_resolution[0] > 0 else (
            f"{previous_format.resolution().width()}x{previous_format.resolution().height()}" if previous_format else ""
        )

        # If camera is not currently active, save preference and update format state
        if not self._is_previewing or not self._camera:
            self._preferred_resolution = f"{w}x{h}"
            self._current_format = matched_format
            SettingsManager.set_preferred_resolution(f"{w}x{h}")
            return True, f"Configured {w}x{h}. Will be applied when camera starts.", f"{w}x{h}"

        # Active stream reconfiguration
        try:
            was_active = self._camera.isActive()
            if was_active:
                self._camera.stop()

            self._camera.setCameraFormat(matched_format)

            if was_active:
                self._camera.start()

            # Check if camera started successfully
            if not self._camera.isActive():
                # Revert to previous format
                if previous_format:
                    self._camera.setCameraFormat(previous_format)
                    self._camera.start()
                return False, f"Camera driver rejected format {w}x{h}. Reverted to previous format.", prev_res

            self._current_format = matched_format
            self._preferred_resolution = f"{w}x{h}"
            SettingsManager.set_preferred_resolution(f"{w}x{h}")

            # Verify cameraFormat resolution
            cf = self._camera.cameraFormat()
            act_w = cf.resolution().width()
            act_h = cf.resolution().height()
            effective_str = f"{act_w}x{act_h}" if (act_w > 0 and act_h > 0) else f"{w}x{h}"

            self.status_changed.emit(f"Resolution set to {effective_str}", "active")
            return True, f"Successfully applied resolution {effective_str}.", effective_str

        except Exception as e:
            # Revert on unexpected exception
            try:
                if previous_format and self._camera:
                    self._camera.setCameraFormat(previous_format)
                    self._camera.start()
            except Exception:
                pass
            return False, f"Driver error applying resolution {w}x{h}: {str(e)}", prev_res

    def _find_best_format(self, formats: List[QCameraFormat], w: int, h: int) -> Optional[QCameraFormat]:
        """Find the optimal camera format matching the requested resolution."""
        candidates = [f for f in formats if f.resolution().width() == w and f.resolution().height() == h]
        if not candidates:
            return None
        # Prefer higher frame rate (e.g. 30 fps or 60 fps)
        candidates.sort(key=lambda f: f.maxFrameRate(), reverse=True)
        return candidates[0]
