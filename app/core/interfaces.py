"""
Backend Integration Contracts & Disconnected Adapters.

This module defines the clean boundary between the Qt frontend and future
Python ML prediction/training backends (predict_live.py, train_model.py,
collect_static_data.py, and dynamic J/Z sequence recognition).

HOW TO INTEGRATE:
1. Subclass PredictionBackendInterface, TrainingBackendInterface,
   StaticCollectionInterface, and DynamicRecordingInterface.
2. Implement methods and emit signals (prediction_result, sample_collected, etc.).
3. Pass active backend instances via BackendRegistry.
"""
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any, List
from PySide6.QtCore import QObject, Signal


@dataclass
class PredictionResult:
    """Standardized prediction packet received from ML engine."""
    sign: str
    confidence: float
    mode: str
    timestamp: float = field(default_factory=time.time)
    raw_landmarks: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingProgress:
    """Standardized training progress packet received from ML engine."""
    stage: str               # "loading", "training", "evaluating", "saved", "failed"
    current_step: int = 0
    total_steps: int = 100
    percent: float = 0.0     # 0.0 to 100.0
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class SampleRecord:
    """A collected static landmark sample packet."""
    id: str
    sign_label: str
    timestamp: float = field(default_factory=time.time)
    landmarks_count: int = 63  # 21 points x 3D (x, y, z)
    is_accepted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DynamicSequenceRecord:
    """A collected movement gesture sequence packet spanning multiple consecutive frames."""
    id: str
    gesture_name: str         # "Z" | "J"
    frame_count: int          # e.g. 30 frames
    duration_seconds: float   # e.g. 1.2s
    timestamp: float = field(default_factory=time.time)
    temporal_frames: List[List[float]] = field(default_factory=list) # List of landmark vectors
    is_accepted: bool = False


# =========================================================================
# Abstract Contracts
# =========================================================================

class PredictionBackendInterface(QObject):
    """
    Abstract contract for real-time sign language inference backends.
    Connects to predict_live.py and dynamic J/Z sequence recognizers.
    """
    prediction_result = Signal(object)   # Emits PredictionResult
    status_changed = Signal(str)        # Emits human-readable status
    error_occurred = Signal(str)        # Emits error description

    def start_prediction(self, mode: str, camera_device_id: str) -> bool:
        raise NotImplementedError

    def stop_prediction(self) -> None:
        raise NotImplementedError

    def is_active(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False

    def get_status_description(self) -> str:
        return "Backend Disconnected"


class TrainingBackendInterface(QObject):
    """
    Abstract contract for model training workspaces.
    Connects to train_model.py for static signs and future dynamic sequence trainers.
    """
    progress_updated = Signal(object)      # Emits TrainingProgress
    log_emitted = Signal(str)              # Emits log lines to console
    training_completed = Signal(bool, str) # (success: bool, output_model_path_or_err: str)
    status_changed = Signal(str)

    def start_static_training(self, config: Dict[str, Any]) -> bool:
        """
        Execute static sign training with configuration matching train_model.py:
        e.g. {'n_estimators': 100, 'test_size': 0.2, 'random_state': 42}
        """
        raise NotImplementedError

    def start_dynamic_training(self, config: Dict[str, Any]) -> bool:
        """
        Execute dynamic sequence training for movement-based signs J & Z.
        e.g. {'sequence_length': 30, 'batch_size': 32, 'epochs': 50}
        """
        raise NotImplementedError

    def stop_training(self) -> None:
        raise NotImplementedError

    def is_training(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False


class StaticCollectionInterface(QObject):
    """
    Abstract contract for camera-based static landmark data collection.
    Connects to collect_static_data.py routines.
    """
    sample_collected = Signal(object)      # Emits SampleRecord
    count_updated = Signal(int)            # Emits genuine accepted sample count
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def start_collection(self, sign_label: str, target_dataset_path: str) -> bool:
        raise NotImplementedError

    def stop_collection(self) -> None:
        raise NotImplementedError

    def capture_sample(self, sign_label: str = "", features: Optional[List[float]] = None) -> Optional[SampleRecord]:
        """Manually trigger capture of a single sample using current hand landmarks."""
        return None

    def accept_sample(self, sample_id: str) -> bool:
        raise NotImplementedError

    def discard_sample(self, sample_id: str) -> bool:
        raise NotImplementedError

    def is_collecting(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False


class DynamicRecordingInterface(QObject):
    """
    Abstract contract for camera-based dynamic gesture sequence recording.
    Captures temporal hand coordinates across consecutive frames for Z and J.
    """
    sequence_recorded = Signal(object)     # Emits DynamicSequenceRecord
    recording_progress = Signal(float)     # 0.0 to 1.0 progress
    count_updated = Signal(int)            # Emits accepted sequence count
    status_changed = Signal(str)
    error_occurred = Signal(str)

    def start_recording(self, gesture: str, sequence_length: int = 30) -> bool:
        raise NotImplementedError

    def stop_recording(self) -> None:
        raise NotImplementedError

    def accept_sequence(self, seq_id: str) -> bool:
        raise NotImplementedError

    def discard_sequence(self, seq_id: str) -> bool:
        raise NotImplementedError

    def is_recording() -> bool:
        return False

    def is_connected(self) -> bool:
        return False


# =========================================================================
# Disconnected Concrete Adapters (Frontend Safe Defaults)
# =========================================================================

class DisconnectedPredictionBackend(PredictionBackendInterface):
    """Explicitly disconnected adapter demonstrating the contract safely."""
    def start_prediction(self, mode: str, camera_device_id: str) -> bool:
        msg = "Backend Disconnected: Live prediction backend has not been linked."
        self.error_occurred.emit(msg)
        self.status_changed.emit("Backend Disconnected")
        return False

    def stop_prediction(self) -> None:
        self.status_changed.emit("Backend Disconnected")

    def is_active(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False

    def get_status_description(self) -> str:
        return "Backend Not Connected (Contract Ready)"


class DisconnectedTrainingBackend(TrainingBackendInterface):
    """Explicitly disconnected training adapter."""
    def start_static_training(self, config: Dict[str, Any]) -> bool:
        self.log_emitted.emit("[Backend Notice] Static training backend is currently unlinked.")
        self.log_emitted.emit("[Integration] Wire train_model.py by implementing TrainingBackendInterface.")
        self.status_changed.emit("Backend Not Connected")
        return False

    def start_dynamic_training(self, config: Dict[str, Any]) -> bool:
        self.log_emitted.emit("[Backend Notice] Dynamic training backend is currently unlinked.")
        self.log_emitted.emit("[Integration] Wire dynamic J/Z sequence trainer by implementing TrainingBackendInterface.")
        self.status_changed.emit("Backend Not Connected")
        return False

    def stop_training(self) -> None:
        self.status_changed.emit("Backend Not Connected")

    def is_training(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False


class DisconnectedStaticCollectionBackend(StaticCollectionInterface):
    """Explicitly disconnected static landmark collection adapter."""
    def start_collection(self, sign_label: str, target_dataset_path: str) -> bool:
        self.error_occurred.emit(
            "Collection Backend Not Connected: Implement StaticCollectionInterface "
            "to bridge collect_static_data.py coordinate logging."
        )
        self.status_changed.emit("Backend Not Connected")
        return False

    def stop_collection(self) -> None:
        self.status_changed.emit("Idle")

    def accept_sample(self, sample_id: str) -> bool:
        return False

    def discard_sample(self, sample_id: str) -> bool:
        return False

    def is_collecting(self) -> bool:
        return False

    def is_connected(self) -> bool:
        return False


class DisconnectedDynamicRecordingBackend(DynamicRecordingInterface):
    """Explicitly disconnected dynamic sequence recording adapter."""
    def start_recording(self, gesture: str, sequence_length: int = 30) -> bool:
        self.error_occurred.emit(
            "Dynamic Sequence Backend Not Connected: Implement DynamicRecordingInterface "
            "to capture consecutive temporal frame coordinates."
        )
        self.status_changed.emit("Backend Not Connected")
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


class BackendRegistry:
    """Global registry providing access to backend adapters."""
    _prediction_backend: PredictionBackendInterface = None
    _training_backend: TrainingBackendInterface = None
    _static_collection: StaticCollectionInterface = None
    _dynamic_recording: DynamicRecordingInterface = None

    @classmethod
    def get_prediction_backend(cls) -> PredictionBackendInterface:
        if cls._prediction_backend is None:
            cls._prediction_backend = DisconnectedPredictionBackend()
        return cls._prediction_backend

    @classmethod
    def set_prediction_backend(cls, backend: PredictionBackendInterface):
        cls._prediction_backend = backend

    @classmethod
    def get_training_backend(cls) -> TrainingBackendInterface:
        if cls._training_backend is None:
            cls._training_backend = DisconnectedTrainingBackend()
        return cls._training_backend

    @classmethod
    def set_training_backend(cls, backend: TrainingBackendInterface):
        cls._training_backend = backend

    @classmethod
    def get_static_collection_backend(cls) -> StaticCollectionInterface:
        if cls._static_collection is None:
            cls._static_collection = DisconnectedStaticCollectionBackend()
        return cls._static_collection

    @classmethod
    def set_static_collection_backend(cls, backend: StaticCollectionInterface):
        cls._static_collection = backend

    @classmethod
    def get_dynamic_recording_backend(cls) -> DynamicRecordingInterface:
        if cls._dynamic_recording is None:
            cls._dynamic_recording = DisconnectedDynamicRecordingBackend()
        return cls._dynamic_recording

    @classmethod
    def set_dynamic_recording_backend(cls, backend: DynamicRecordingInterface):
        cls._dynamic_recording = backend
