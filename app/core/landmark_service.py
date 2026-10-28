"""
High-Performance MediaPipe Landmark Inference Service for SignFlow Studio.
Decouples camera video rendering from ML landmark detection by executing
MediaPipe HandLandmarker asynchronously in a dedicated worker QThread.

Guarantees:
1. Native 30 FPS video preview on main thread (zero stutter or GUI freezing).
2. Throttled ~18-20 FPS inference rate with strict non-accumulating frame dropping.
3. Separation of camera rendering, landmark detection, and dataset collection.
4. Clean lifecycle management and safe thread shutdown.
"""
import time
import math
import os
from typing import List, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThread, Qt
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoFrame

from app.core.config import ROOT_DIR, HAND_LANDMARKER_TASK
from app.core.camera_manager import CameraManager


# MediaPipe Hand Connections (21 landmarks, 21 connections)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}
WRIST_INDEX = 0


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float


def extract_normalized_features(landmarks) -> List[float]:
    """
    Extract 63 wrist-relative normalized coordinates from 21 MediaPipe hand landmarks.
    Matches the exact feature order, scaling, and normalization in static_signs.csv.
    """
    wrist = landmarks[0]
    features = []
    for point in landmarks:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])
    scale = max(max(abs(v) for v in features), 1e-6)
    return [v / scale for v in features]


def qvideoframe_to_rgb(frame: QVideoFrame) -> Optional[np.ndarray]:
    """Convert QVideoFrame to RGB numpy array efficiently."""
    if not frame.isValid():
        return None
    try:
        qimg = frame.toImage()
        if qimg.isNull():
            return None
        if qimg.format() != QImage.Format.Format_RGB888:
            qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        w = qimg.width()
        h = qimg.height()
        ptr = qimg.bits()
        # Fast buffer reshape with copy to ensure lifetime beyond qimg
        return np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
    except Exception:
        return None


class LandmarkWorker(QObject):
    """Worker object living on a dedicated QThread for MediaPipe inference."""
    detection_finished = Signal(object) # Dict with hands, feature_vector, etc.
    error_occurred = Signal(str)

    def __init__(self, model_task_path: str, parent=None):
        super().__init__(parent)
        self.model_task_path = model_task_path
        self._detector = None
        self._is_initialized = False

        # Rolling FPS calculation
        self._last_fps_time = time.time()
        self._frame_count = 0
        self._inference_fps = 0.0

    def init_detector(self) -> bool:
        if self._is_initialized and self._detector:
            return True

        if not os.path.exists(self.model_task_path):
            self.error_occurred.emit(f"Model task file not found: {self.model_task_path}")
            return False

        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_task_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._detector = HandLandmarker.create_from_options(options)
            self._is_initialized = True
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize MediaPipe HandLandmarker: {e}")
            return False

    @Slot(object, float)
    def process_image(self, rgb_array: np.ndarray, timestamp: float):
        """Perform landmark detection off the GUI thread."""
        if not self._is_initialized:
            if not self.init_detector():
                self.detection_finished.emit(None)
                return

        t_start = time.perf_counter()
        hands_result = []
        feature_vector = None

        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)
            result = self._detector.detect(mp_image)

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                for idx, hand_lms in enumerate(result.hand_landmarks):
                    pts = [LandmarkPoint(p.x, p.y, p.z) for p in hand_lms]
                    hands_result.append(pts)
                    if idx == 0:
                        feature_vector = extract_normalized_features(pts)

            # Update inference FPS
            self._frame_count += 1
            now = time.time()
            dt = now - self._last_fps_time
            if dt >= 0.8:
                self._inference_fps = round(self._frame_count / dt, 1)
                self._frame_count = 0
                self._last_fps_time = now

            inf_ms = round((time.perf_counter() - t_start) * 1000, 1)

            payload = {
                "hands": hands_result,
                "num_hands": len(hands_result),
                "feature_vector": feature_vector,
                "inference_fps": self._inference_fps,
                "inference_ms": inf_ms,
                "timestamp": timestamp
            }
            self.detection_finished.emit(payload)
        except Exception as e:
            self.error_occurred.emit(f"Inference error: {e}")
            self.detection_finished.emit(None)

    def close(self):
        if self._detector:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._is_initialized = False


class LandmarkService(QObject):
    """
    Centralized controller for asynchronous hand landmark inference.
    Throttles inference to 15-20 FPS, drops frames when busy,
    and forwards landmarks directly to active preview canvas.
    """
    request_inference = Signal(object, float) # (rgb_array, timestamp) dispatched to QThread worker
    landmarks_updated = Signal(object) # Payload dict
    error_occurred = Signal(str)

    _instance: Optional["LandmarkService"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        LandmarkService._instance = self

        self._task_path = str(HAND_LANDMARKER_TASK.resolve())
        self._thread: Optional[QThread] = None
        self._worker: Optional[LandmarkWorker] = None

        # Throttling state: default targeting ~18 FPS (55ms interval)
        self._min_interval = 0.055
        self._last_dispatch_time = 0.0
        self._is_worker_busy = False
        self._is_enabled = True

        self._latest_payload: Optional[Dict[str, Any]] = None

        self._setup_worker()

        # Connect to shared CameraManager
        cam_mgr = CameraManager.get_instance()
        if cam_mgr:
            cam_mgr.frame_ready.connect(self._on_camera_frame)
            cam_mgr.preview_stopped.connect(self._on_preview_stopped)

    @classmethod
    def get_instance(cls) -> "LandmarkService":
        if cls._instance is None:
            cls._instance = LandmarkService()
        return cls._instance

    def _setup_worker(self):
        """Spawn the dedicated inference worker thread."""
        self._thread = QThread()
        self._worker = LandmarkWorker(self._task_path)
        self._worker.moveToThread(self._thread)

        self.request_inference.connect(self._worker.process_image, Qt.ConnectionType.QueuedConnection)
        self._worker.detection_finished.connect(self._on_detection_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.error_occurred.connect(self.error_occurred)

        self._thread.start()

    @property
    def latest_landmarks(self) -> Optional[Dict[str, Any]]:
        """Non-reactive reference to latest valid landmark detection output."""
        return self._latest_payload

    def set_enabled(self, enabled: bool):
        self._is_enabled = enabled
        if not enabled:
            self.clear_landmarks()

    def set_target_fps(self, fps: float):
        if fps > 0:
            self._min_interval = 1.0 / fps

    @Slot(object)
    def _on_camera_frame(self, frame: QVideoFrame):
        """Receive frame from CameraManager; throttled & non-blocking."""
        if not self._is_enabled or self._worker is None:
            return

        now = time.time()
        # 1. Throttling check (target 15-20 FPS)
        if (now - self._last_dispatch_time) < self._min_interval:
            return

        # 2. Drop frame if worker is busy processing previous request (Zero queue buildup)
        if self._is_worker_busy:
            return

        # 3. Convert frame to RGB numpy array
        rgb_arr = qvideoframe_to_rgb(frame)
        if rgb_arr is None:
            return

        self._is_worker_busy = True
        self._last_dispatch_time = now

        # 4. Asynchronously invoke worker on QThread via queued signal
        self.request_inference.emit(rgb_arr, now)

    @Slot(object)
    def _on_detection_finished(self, payload: Optional[Dict[str, Any]]):
        """Receive detection results on main GUI thread."""
        self._is_worker_busy = False

        if payload is not None:
            # Attach current camera FPS
            cam_mgr = CameraManager.get_instance()
            cam_fps = cam_mgr.current_fps if cam_mgr else 0.0
            payload["camera_fps"] = cam_fps

            self._latest_payload = payload

            # Automatically deliver landmarks to the currently bound preview widget
            if cam_mgr and cam_mgr._video_widget and hasattr(cam_mgr._video_widget, "set_landmarks"):
                cam_mgr._video_widget.set_landmarks(payload)

            self.landmarks_updated.emit(payload)
        else:
            self._latest_payload = None
            cam_mgr = CameraManager.get_instance()
            if cam_mgr and cam_mgr._video_widget and hasattr(cam_mgr._video_widget, "clear_landmarks"):
                cam_mgr._video_widget.clear_landmarks()

        if payload is not None:
            # Attach current camera FPS
            cam_mgr = CameraManager.get_instance()
            cam_fps = cam_mgr.current_fps if cam_mgr else 0.0
            payload["camera_fps"] = cam_fps

            self._latest_payload = payload

            # Automatically deliver landmarks to the currently bound preview widget
            if cam_mgr and cam_mgr._video_widget and hasattr(cam_mgr._video_widget, "set_landmarks"):
                cam_mgr._video_widget.set_landmarks(payload)

            self.landmarks_updated.emit(payload)

    @Slot()
    def _on_preview_stopped(self):
        """Camera preview halted; clear stale landmarks immediately."""
        self.clear_landmarks()

    def clear_landmarks(self):
        self._latest_payload = None
        cam_mgr = CameraManager.get_instance()
        if cam_mgr and cam_mgr._video_widget and hasattr(cam_mgr._video_widget, "clear_landmarks"):
            cam_mgr._video_widget.clear_landmarks()

    def teardown(self):
        """Clean up thread and MediaPipe resources."""
        if self._worker:
            self._worker.close()
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)
