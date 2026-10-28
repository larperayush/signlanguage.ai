"""
Concrete Python ML Backend Adapters for SignFlow Studio.
Bridges MediaPipe (hand_landmarker.task) and Scikit-Learn (asl_classifier.joblib)
to the Qt frontend via contracts in app/core/interfaces.py.

Key Architectural Guarantees:
1. Reuses feature extraction from predict_live.py and collect_static_data.py.
2. Uses the shared CameraManager (no duplicate cv2.VideoCapture or cv2.imshow).
3. Executes inference and training asynchronously to prevent blocking the Qt UI thread.
4. Loads hand_landmarker.task for landmark detection and asl_classifier.joblib separately.
5. Strictly prevents dynamic sequence training from reusing static scripts.
"""
import os
import time
import math
import uuid
import csv
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot, QThread, QMutex, QMutexLocker
from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QVideoFrame

from app.core.interfaces import (
    PredictionBackendInterface,
    TrainingBackendInterface,
    StaticCollectionInterface,
    DynamicRecordingInterface,
    PredictionResult,
    TrainingProgress,
    SampleRecord,
    DynamicSequenceRecord,
    BackendRegistry
)
from app.core.camera_manager import CameraManager


def extract_normalized_features(landmarks) -> List[float]:
    """
    Extract 63 wrist-relative normalized coordinates from 21 MediaPipe hand landmarks.
    Reuses the exact algorithm from predict_live.py and collect_static_data.py.
    """
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.append(lm.x - wrist.x)
        raw_coords.append(lm.y - wrist.y)
        raw_coords.append(lm.z - wrist.z)

    max_abs = max(abs(val) for val in raw_coords)
    scale = max(max_abs, 1e-6)
    return [val / scale for val in raw_coords]


def qvideoframe_to_rgb_array(frame: QVideoFrame) -> Optional[np.ndarray]:
    """Convert QVideoFrame to an RGB numpy uint8 array without GUI blocking."""
    if not frame.isValid():
        return None
    try:
        qimg = frame.toImage()
        if qimg.isNull():
            return None
        if qimg.format() != QImage.Format.Format_RGB888:
            qimg = qimg.convertToFormat(QImage.Format.Format_RGB888)
        
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        # Create array copy so memory remains valid after qimg goes out of scope
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 3)).copy()
        return arr
    except Exception:
        return None


# =========================================================================
# Prediction Adapter
# =========================================================================

class PredictionWorker(QObject):
    """Worker object for off-thread landmark detection and ML classification."""
    finished = Signal()
    prediction_ready = Signal(object)  # Emits PredictionResult
    status_msg = Signal(str)

    def __init__(self, model_path: str, classifier_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.classifier_path = classifier_path
        self._detector = None
        self._classifier = None
        self._is_initialized = False
        self._last_timestamp_ms = 0
        self._buffer = deque(maxlen=6)
        self._last_emitted_sign = ""
        self._last_emitted_time = 0.0

    def initialize(self) -> bool:
        """Load MediaPipe hand landmarker task and classifier separately."""
        if self._is_initialized:
            return True

        if not os.path.exists(self.model_path):
            self.status_msg.emit(f"MediaPipe task missing: {self.model_path}")
            return False

        if not os.path.exists(self.classifier_path):
            self.status_msg.emit(f"Classifier model missing: {self.classifier_path}")
            return False

        try:
            import mediapipe as mp
            import joblib

            BaseOptions = mp.tasks.BaseOptions
            HandLandmarker = mp.tasks.vision.HandLandmarker
            HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode

            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self._detector = HandLandmarker.create_from_options(options)
            self._classifier = joblib.load(self.classifier_path)
            self._is_initialized = True
            return True
        except Exception as e:
            self.status_msg.emit(f"Initialization error: {e}")
            return False

    def process_rgb_image(self, rgb_arr: np.ndarray, mode: str):
        """Perform landmark detection and classification on an RGB image."""
        if not self._is_initialized or self._detector is None or self._classifier is None:
            return

        try:
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_arr)
            detection_result = self._detector.detect(mp_image)

            predicted_sign = "No Hand"
            confidence = 0.0
            raw_landmarks = None

            if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                lms = detection_result.hand_landmarks[0]
                features = extract_normalized_features(lms)
                raw_landmarks = features

                probs = self._classifier.predict_proba([features])[0]
                max_idx = probs.argmax()
                predicted_sign = str(self._classifier.classes_[max_idx])
                confidence = float(probs[max_idx])

                if confidence < 0.55:
                    predicted_sign = "Uncertain"

            # Apply smoothing buffer
            self._buffer.append(predicted_sign)
            stable_sign = max(set(self._buffer), key=self._buffer.count)

            # Emit prediction if stable sign is valid and debounced
            if stable_sign not in ("No Hand", "Uncertain"):
                now = time.time()
                # Emit if sign changed, or if held for > 1.2 seconds
                if stable_sign != self._last_emitted_sign or (now - self._last_emitted_time) >= 1.2:
                    self._last_emitted_sign = stable_sign
                    self._last_emitted_time = now
                    res = PredictionResult(
                        sign=stable_sign,
                        confidence=confidence,
                        mode=mode,
                        timestamp=now,
                        raw_landmarks=raw_landmarks,
                        metadata={"classes": list(self._classifier.classes_)}
                    )
                    self.prediction_ready.emit(res)
            elif stable_sign == "No Hand":
                if self._last_emitted_sign != "":
                    self._last_emitted_sign = ""
        except Exception as e:
            self.status_msg.emit(f"Inference error: {e}")

    def predict_features(self, features: Optional[List[float]], mode: str = "Static"):
        """Classify sign from precomputed 63-feature normalized landmark vector (runs in < 0.1ms)."""
        if not self._is_initialized or self._classifier is None:
            return

        predicted_sign = "No Hand"
        confidence = 0.0
        raw_landmarks = None

        if features and len(features) == 63 and all(math.isfinite(v) for v in features):
            try:
                raw_landmarks = features
                probs = self._classifier.predict_proba([features])[0]
                max_idx = probs.argmax()
                predicted_sign = str(self._classifier.classes_[max_idx])
                confidence = float(probs[max_idx])

                if confidence < 0.55:
                    predicted_sign = "Uncertain"
            except Exception as e:
                self.status_msg.emit(f"Classifier evaluation error: {e}")
                predicted_sign = "Uncertain"

        # Apply smoothing buffer
        self._buffer.append(predicted_sign)
        stable_sign = max(set(self._buffer), key=self._buffer.count)

        # Emit prediction if stable sign is valid and debounced
        if stable_sign not in ("No Hand", "Uncertain"):
            now = time.time()
            if stable_sign != self._last_emitted_sign or (now - self._last_emitted_time) >= 1.2:
                self._last_emitted_sign = stable_sign
                self._last_emitted_time = now
                res = PredictionResult(
                    sign=stable_sign,
                    confidence=confidence,
                    mode=mode,
                    timestamp=now,
                    raw_landmarks=raw_landmarks,
                    metadata={"classes": list(self._classifier.classes_)}
                )
                self.prediction_ready.emit(res)
        elif stable_sign == "No Hand":
            if self._last_emitted_sign != "":
                self._last_emitted_sign = ""

    def close(self):
        """Release MediaPipe detector resources."""
        if self._detector:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
        self._classifier = None
        self._is_initialized = False


class PythonMLPredictionAdapter(PredictionBackendInterface):
    """
    Connected adapter bridging predict_live.py logic into SignFlow Studio.
    Reuses off-thread landmarks from LandmarkService to eliminate duplicate inference loops.
    """
    def __init__(self, model_task_path: str, classifier_path: str, parent=None):
        super().__init__(parent)
        self.model_task_path = str(Path(model_task_path).resolve())
        self.classifier_path = str(Path(classifier_path).resolve())
        self._is_active = False
        self._landmarks_connected = False
        self._mode = "Static"
        self._worker = PredictionWorker(self.model_task_path, self.classifier_path)
        self._worker.prediction_ready.connect(self._on_prediction_ready)
        self._worker.status_msg.connect(self._on_worker_status)
        self._connected = False
        self._verify_connection()

    def _verify_connection(self):
        """Verify that necessary files exist and dependencies can be loaded."""
        if os.path.exists(self.model_task_path) and os.path.exists(self.classifier_path):
            self._connected = True
        else:
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def is_active(self) -> bool:
        return self._is_active

    def get_status_description(self) -> str:
        if self._is_active:
            return "Inference Running"
        if self._connected:
            return "Connected (RandomForest + MediaPipe)"
        return "Backend Disconnected"

    def start_prediction(self, mode: str, camera_device_id: str) -> bool:
        if not self._connected:
            self.error_occurred.emit("ML model files are missing or unlinked.")
            self.status_changed.emit("Backend Disconnected")
            return False

        if not self._worker.initialize():
            self.error_occurred.emit("Failed to initialize MediaPipe or ASL classifier.")
            self.status_changed.emit("Backend Error")
            return False

        self._mode = mode
        self._is_active = True

        # Connect to LandmarkService asynchronous landmark stream (reusing single MediaPipe pipeline)
        if not self._landmarks_connected:
            from app.core.landmark_service import LandmarkService
            LandmarkService.get_instance().landmarks_updated.connect(self._on_landmarks_received)
            self._landmarks_connected = True

        cam_mgr = CameraManager.get_instance()
        if cam_mgr and not cam_mgr.is_previewing:
            cam_mgr.start_preview()

        self.status_changed.emit("Inference Running")
        return True

    def stop_prediction(self) -> None:
        self._is_active = False
        if self._landmarks_connected:
            from app.core.landmark_service import LandmarkService
            try:
                LandmarkService.get_instance().landmarks_updated.disconnect(self._on_landmarks_received)
            except Exception:
                pass
            self._landmarks_connected = False

        self._worker.close()
        self.status_changed.emit("Inference Halted")

    @Slot(object)
    def _on_landmarks_received(self, payload: Optional[Dict[str, Any]]):
        """Consume precomputed landmarks from LandmarkService; evaluates in < 0.1ms."""
        if not self._is_active:
            return
        features = payload.get("feature_vector") if payload else None
        self._worker.predict_features(features, self._mode)

    def predict_direct(self, rgb_arr: np.ndarray, mode: str = "Static"):
        """Direct inference on an RGB numpy image (for smoke tests / offline calls)."""
        if not self._worker._is_initialized:
            self._worker.initialize()
        self._worker.process_rgb_image(rgb_arr, mode)

    @Slot(object)
    def _on_prediction_ready(self, result: PredictionResult):
        self.prediction_result.emit(result)

    @Slot(str)
    def _on_worker_status(self, msg: str):
        self.error_occurred.emit(msg)


# =========================================================================
# Static Collection Adapter
# =========================================================================

class PythonMLStaticCollectionAdapter(StaticCollectionInterface):
    """
    Connected adapter bridging collect_static_data.py routines.
    Captures 63 wrist-relative normalized landmarks manually and commits to static_signs.csv.
    """
    def __init__(self, model_task_path: str, dataset_path: str, parent=None):
        super().__init__(parent)
        self.model_task_path = str(Path(model_task_path).resolve())
        self.dataset_path = str(Path(dataset_path).resolve())
        self._is_collecting = False
        self._current_sign = "A"
        self._candidate_samples: Dict[str, List[float]] = {}
        self._sample_counts: Dict[str, int] = {}
        self._connected = os.path.exists(self.model_task_path)
        self._load_dataset_counts()

    def _load_dataset_counts(self):
        self._sample_counts = {}
        if os.path.exists(self.dataset_path) and os.path.getsize(self.dataset_path) > 0:
            try:
                with open(self.dataset_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if row and len(row) >= 1:
                            lbl = row[0].strip().upper()
                            self._sample_counts[lbl] = self._sample_counts.get(lbl, 0) + 1
            except Exception:
                pass

    def is_connected(self) -> bool:
        return self._connected

    def is_collecting(self) -> bool:
        return self._is_collecting

    def start_collection(self, sign_label: str, target_dataset_path: str = "") -> bool:
        if not self._connected or not os.path.exists(self.model_task_path):
            self.error_occurred.emit("MediaPipe hand_landmarker.task not found.")
            return False

        if target_dataset_path:
            self.dataset_path = str(Path(target_dataset_path).resolve())

        self._current_sign = sign_label.strip().upper()
        self._is_collecting = True
        self.status_changed.emit("Collecting")
        count = self._sample_counts.get(self._current_sign, 0)
        self.count_updated.emit(count)
        return True

    def stop_collection(self) -> None:
        self._is_collecting = False
        self.status_changed.emit("Idle")

    def capture_sample(self, sign_label: str = "", features: Optional[List[float]] = None) -> Optional[SampleRecord]:
        """
        Capture exactly one sample using the latest valid MediaPipe landmarks.
        Validates feature vector and places sample in candidate buffer for review.
        """
        if not self._is_collecting:
            self.error_occurred.emit("Collection session is not active.")
            return None

        target_sign = (sign_label or self._current_sign).strip().upper()
        if not target_sign:
            target_sign = "A"

        # Obtain features from LandmarkService if not explicitly supplied
        if features is None:
            from app.core.landmark_service import LandmarkService
            service = LandmarkService.get_instance()
            latest = service.latest_landmarks
            if not latest or not latest.get("hands") or latest.get("num_hands", 0) == 0:
                self.error_occurred.emit("No hand detected. Cannot capture sample.")
                return None
            features = latest.get("feature_vector")

        # Validate feature vector: must have exactly 63 finite numbers
        if not features or len(features) != 63 or any(not math.isfinite(v) for v in features):
            self.error_occurred.emit("Invalid or non-finite landmark features.")
            return None

        sample_id = f"smp_{uuid.uuid4().hex[:6]}"
        self._candidate_samples[sample_id] = {
            "features": list(features),
            "label": target_sign
        }

        record = SampleRecord(
            id=sample_id,
            sign_label=target_sign,
            timestamp=time.time(),
            landmarks_count=len(features),
            is_accepted=False,
            metadata={"features": list(features)}
        )
        self.sample_collected.emit(record)
        return record

    def accept_sample(self, sample_id: str) -> bool:
        """Commit accepted sample into target CSV dataset file."""
        if sample_id not in self._candidate_samples:
            return False

        item = self._candidate_samples.pop(sample_id)
        if isinstance(item, dict):
            features = item["features"]
            label = item.get("label", self._current_sign)
        else:
            features = item
            label = self._current_sign

        csv_path = Path(self.dataset_path)

        # Ensure header if new file
        needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
        header = ["label"] + [f"feature_{i}" for i in range(63)]

        try:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if needs_header:
                    writer.writerow(header)
                writer.writerow([label] + features)

            self._sample_counts[label] = self._sample_counts.get(label, 0) + 1
            self.count_updated.emit(self._sample_counts[label])
            return True
        except Exception as e:
            self.error_occurred.emit(f"Failed to save sample: {e}")
            return False

    def discard_sample(self, sample_id: str) -> bool:
        """Discard sample from buffer."""
        if sample_id in self._candidate_samples:
            del self._candidate_samples[sample_id]
            return True
        return False


# =========================================================================
# Training Adapter
# =========================================================================

class TrainingWorkerThread(QThread):
    """Background thread executing scikit-learn model training."""
    progress = Signal(object)      # TrainingProgress
    log = Signal(str)
    completed = Signal(bool, str)

    def __init__(self, dataset_path: str, output_model_path: str, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.dataset_path = dataset_path
        self.output_model_path = output_model_path
        self.config = config
        self._is_interrupted = False

    def interrupt(self):
        self._is_interrupted = True

    def run(self):
        try:
            self.log.emit("[Train Worker] Initializing training environment...")
            self.progress.emit(TrainingProgress(stage="loading", percent=10.0, message="Validating dataset file..."))

            if not os.path.exists(self.dataset_path):
                self.completed.emit(False, f"Dataset file not found: {self.dataset_path}")
                return

            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            import joblib

            labels = []
            features = []

            with open(self.dataset_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    self.completed.emit(False, "Dataset CSV file is empty.")
                    return

                for row_idx, row in enumerate(reader, start=2):
                    if self._is_interrupted:
                        self.completed.emit(False, "Training interrupted by user.")
                        return
                    if not row or len(row) != 64:
                        continue
                    label = row[0].strip().upper()
                    try:
                        f_vals = [float(v) for v in row[1:]]
                        labels.append(label)
                        features.append(f_vals)
                    except ValueError:
                        continue

            X = np.array(features)
            y = np.array(labels)

            if len(X) < 10:
                self.completed.emit(False, f"Insufficient samples: found {len(X)} samples. At least 10 required.")
                return

            unique_labels, counts = np.unique(y, return_counts=True)
            self.log.emit(f"[Dataset] Total samples: {len(X)}, Classes ({len(unique_labels)}): {dict(zip(unique_labels, counts))}")

            if len(unique_labels) < 2:
                self.completed.emit(False, f"Training requires at least 2 distinct sign classes. Found: {list(unique_labels)}")
                return

            self.progress.emit(TrainingProgress(stage="splitting", percent=35.0, message="Splitting train and test sets..."))

            test_size = float(self.config.get("test_size", 0.2))
            random_state = int(self.config.get("random_state", 42))
            n_estimators = int(self.config.get("n_estimators", 100))
            criterion = str(self.config.get("criterion", "gini"))

            # Handle small class counts for stratify
            min_count = min(counts)
            use_stratify = y if min_count >= 2 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=use_stratify
            )

            if self._is_interrupted:
                self.completed.emit(False, "Training interrupted.")
                return

            self.progress.emit(TrainingProgress(
                stage="training", percent=50.0,
                message=f"Fitting RandomForest (n_estimators={n_estimators}, criterion={criterion})..."
            ))
            self.log.emit(f"[Training] Fitting RandomForest on {len(X_train)} samples, testing on {len(X_test)} samples...")

            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                criterion=criterion
            )
            clf.fit(X_train, y_train)

            if self._is_interrupted:
                self.completed.emit(False, "Training interrupted.")
                return

            self.progress.emit(TrainingProgress(stage="evaluating", percent=80.0, message="Evaluating model accuracy..."))
            y_pred = clf.predict(X_test)
            acc = float(accuracy_score(y_test, y_pred))
            self.log.emit(f"[Evaluation] Accuracy: {acc * 100:.2f}%")
            self.log.emit(f"[Evaluation] Classification Summary:\n{classification_report(y_test, y_pred, zero_division=0)}")

            # Save model
            self.progress.emit(TrainingProgress(stage="saving", percent=90.0, message="Saving model artifact..."))
            out_p = Path(self.output_model_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, str(out_p.resolve()))

            self.progress.emit(TrainingProgress(
                stage="saved", percent=100.0,
                metrics={"accuracy": acc},
                message=f"Model successfully saved to {out_p.name}"
            ))
            self.completed.emit(True, f"Model trained with {acc * 100:.2f}% test accuracy -> {out_p.name}")

        except Exception as e:
            self.completed.emit(False, f"Training execution error: {str(e)}")


class PythonMLTrainingAdapter(TrainingBackendInterface):
    """
    Connected adapter bridging train_model.py routines.
    Executes training on a separate worker thread so the UI remains responsive.
    """
    def __init__(self, dataset_path: str, model_dir: str, parent=None):
        super().__init__(parent)
        self.dataset_path = str(Path(dataset_path).resolve())
        self.model_dir = str(Path(model_dir).resolve())
        self._worker_thread: Optional[TrainingWorkerThread] = None
        self._is_training = False
        self._connected = True

    def is_connected(self) -> bool:
        return self._connected

    def is_training(self) -> bool:
        return self._is_training

    def start_static_training(self, config: Dict[str, Any]) -> bool:
        if self._is_training:
            self.log_emitted.emit("[Training] A training run is already in progress.")
            return False

        if not os.path.exists(self.dataset_path):
            self.log_emitted.emit(f"[Error] Dataset file not found at: {self.dataset_path}")
            self.training_completed.emit(False, f"Dataset not found: {self.dataset_path}")
            return False

        output_path = os.path.join(self.model_dir, "asl_classifier.joblib")
        self._is_training = True
        self.status_changed.emit("Training Active")

        self._worker_thread = TrainingWorkerThread(
            dataset_path=self.dataset_path,
            output_model_path=output_path,
            config=config,
            parent=self
        )
        self._worker_thread.progress.connect(self._on_progress)
        self._worker_thread.log.connect(self._on_log)
        self._worker_thread.completed.connect(self._on_completed)
        self._worker_thread.start()
        return True

    def start_dynamic_training(self, config: Dict[str, Any]) -> bool:
        """
        Dynamic sequence training must NOT reuse static scripts.
        Strictly enforces architectural boundary per Requirement 4.
        """
        msg = (
            "Dynamic sequence training (LSTM / Temporal Classifier) is not implemented in the current backend. "
            "Per architecture requirements, static single-frame RandomForest (train_model.py) cannot be reused "
            "for dynamic movement sequences (Z and J)."
        )
        self.log_emitted.emit(f"[Backend Notice] {msg}")
        self.training_completed.emit(False, msg)
        self.status_changed.emit("Dynamic Training Unsupported")
        return False

    def stop_training(self) -> None:
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.interrupt()
            self._worker_thread.wait(3000)
        self._is_training = False
        self.status_changed.emit("Idle")

    @Slot(object)
    def _on_progress(self, prog: TrainingProgress):
        self.progress_updated.emit(prog)

    @Slot(str)
    def _on_log(self, text: str):
        self.log_emitted.emit(text)

    @Slot(bool, str)
    def _on_completed(self, success: bool, message: str):
        self._is_training = False
        self.status_changed.emit("Completed" if success else "Failed")
        self.training_completed.emit(success, message)


# =========================================================================
# Dynamic Recording Adapter (Explicitly Unlinked per Requirement 4)
# =========================================================================

class PythonMLDynamicRecordingAdapter(DynamicRecordingInterface):
    """
    Explicitly refuses to reuse collect_static_data.py for dynamic sequences.
    Preserves requirement that dynamic sequences require multi-frame temporal capturing.
    """
    def start_recording(self, gesture: str, sequence_length: int = 30) -> bool:
        msg = (
            f"Dynamic gesture recording for '{gesture}' is not supported by the static data collection backend. "
            "collect_static_data.py only captures static single-frame landmarks. "
            "A dedicated temporal buffer pipeline is required."
        )
        self.error_occurred.emit(msg)
        self.status_changed.emit("Dynamic Recording Unsupported")
        return False

    def stop_recording(self) -> None:
        self.status_changed.emit("Idle")

    def accept_sequence(self, seq_id: str) -> bool:
        return False

    def discard_sequence(self, seq_id: str) -> bool:
        return False

    def is_recording(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False
