"""
Model Manager Registry & Artifact Registration Service.
Registers existing trained model artifacts (.joblib, .task, .onnx, .h5, etc.)
WITHOUT deserializing or executing them. Pure frontend artifact cataloging.
"""
from pathlib import Path
import json
import uuid
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from PySide6.QtCore import QObject, Signal, QSettings

from app.core.config import ROOT_DIR, MODELS_DIR, ASL_CLASSIFIER_MODEL, HAND_LANDMARKER_TASK, ORGANIZATION_NAME, APP_NAME

MODEL_EXTENSIONS = {
    ".joblib": "Scikit-Learn Classifier (.joblib)",
    ".task": "MediaPipe Vision Task Bundle (.task)",
    ".onnx": "Open Neural Network Exchange (.onnx)",
    ".h5": "TensorFlow / Keras HDF5 (.h5)",
    ".keras": "Keras Native Model (.keras)",
    ".pt": "PyTorch TorchScript / State Dict (.pt)",
    ".pth": "PyTorch Model Checkpoint (.pth)",
    ".tflite": "TensorFlow Lite FlatBuffer (.tflite)",
    ".pkl": "Serialized Model Pickle (.pkl)"
}


@dataclass
class ModelEntry:
    id: str
    name: str
    file_path: str
    file_extension: str
    model_type: str
    file_size_bytes: int
    status: str             # e.g. "Registered (Deserialization Held until Backend Integration)"
    is_accessible: bool
    registered_at: float = field(default_factory=time.time)
    previous_reference: Optional[Dict[str, Any]] = None


class ModelRegistry(QObject):
    """
    Frontend model artifact registry.
    Strictly avoids deserializing or executing imported binaries.
    """
    model_added = Signal(object)
    model_updated = Signal(object)
    model_removed = Signal(str)

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: Dict[str, ModelEntry] = {}
        self._load_from_settings()

        # Seed with existing workspace model files if unindexed
        if not self._models:
            self._seed_existing_models()
        else:
            existing_names = {Path(m.file_path).name for m in self._models.values()}
            for path, display_name in [
                (MODELS_DIR / "asl_neural_net_dense.joblib", "ASL Deep Neural Network (MLP 63-64-32-6)"),
                (MODELS_DIR / "asl_neural_net_compact.joblib", "ASL Compact Neural Network (MLP 63-32-16-6)")
            ]:
                if path.exists() and path.name not in existing_names:
                    ext = path.suffix.lower()
                    entry = ModelEntry(
                        id=str(uuid.uuid4())[:8],
                        name=display_name,
                        file_path=str(path.resolve()),
                        file_extension=ext,
                        model_type=MODEL_EXTENSIONS.get(ext, "Custom Model Artifact"),
                        file_size_bytes=path.stat().st_size,
                        status="Registered (Ready for Backend Binding)",
                        is_accessible=True
                    )
                    self._models[entry.id] = entry
            self._save_to_settings()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    @staticmethod
    def _settings():
        return QSettings(ORGANIZATION_NAME, APP_NAME)

    def _save_to_settings(self):
        s = self._settings()
        serializable = {k: asdict(v) for k, v in self._models.items()}
        s.setValue("model_registry/entries", json.dumps(serializable))

    def _load_from_settings(self):
        s = self._settings()
        raw = s.value("model_registry/entries", "{}", type=str)
        try:
            data = json.loads(raw)
            for k, d in data.items():
                self._models[k] = ModelEntry(**d)
        except Exception:
            self._models = {}

    def _seed_existing_models(self):
        """Register default project models safely."""
        initial_files = [
            (ASL_CLASSIFIER_MODEL, "ASL Static Sign Classifier (Random Forest)"),
            (MODELS_DIR / "asl_neural_net_dense.joblib", "ASL Deep Neural Network (MLP 63-64-32-6)"),
            (MODELS_DIR / "asl_neural_net_compact.joblib", "ASL Compact Neural Network (MLP 63-32-16-6)"),
            (HAND_LANDMARKER_TASK, "MediaPipe Hand Landmarker Task")
        ]
        for path, display_name in initial_files:
            if path.exists():
                ext = path.suffix.lower()
                m_type = MODEL_EXTENSIONS.get(ext, "Custom Model Artifact")
                entry = ModelEntry(
                    id=str(uuid.uuid4())[:8],
                    name=display_name,
                    file_path=str(path.resolve()),
                    file_extension=ext,
                    model_type=m_type,
                    file_size_bytes=path.stat().st_size,
                    status="Registered (Ready for Backend Binding)",
                    is_accessible=True
                )
                self._models[entry.id] = entry
        self._save_to_settings()

    @staticmethod
    def identify_model_type(path: Path) -> str:
        ext = path.suffix.lower()
        return MODEL_EXTENSIONS.get(ext, f"Binary Artifact ({ext})")

    def register_model(self, file_path: str, custom_name: str = "") -> Tuple[bool, Optional[ModelEntry], str]:
        src = Path(file_path).resolve()
        if not src.exists():
            return False, None, f"Model file not found: {file_path}"

        ext = src.suffix.lower()
        if ext not in MODEL_EXTENSIONS:
            return False, None, f"Unsupported model extension: '{ext}'. Supported: {', '.join(MODEL_EXTENSIONS.keys())}"

        entry_id = str(uuid.uuid4())[:8]
        entry = ModelEntry(
            id=entry_id,
            name=custom_name or src.name,
            file_path=str(src),
            file_extension=ext,
            model_type=self.identify_model_type(src),
            file_size_bytes=src.stat().st_size,
            status="Registered (Deserialization Held until Backend Integration)",
            is_accessible=True
        )

        self._models[entry.id] = entry
        self._save_to_settings()
        self.model_added.emit(entry)
        return True, entry, f"Successfully registered '{entry.name}'."

    def replace_model(self, model_id: str, new_file_path: str) -> Tuple[bool, str]:
        old_entry = self._models.get(model_id)
        if not old_entry:
            return False, "Target model not found."

        new_src = Path(new_file_path).resolve()
        if not new_src.exists():
            return False, f"Replacement file not found: {new_file_path}"

        ext = new_src.suffix.lower()
        if ext not in MODEL_EXTENSIONS:
            return False, f"Unsupported model extension '{ext}'."

        # Snapshot for revert
        snapshot = {
            "name": old_entry.name,
            "file_path": old_entry.file_path,
            "file_extension": old_entry.file_extension,
            "model_type": old_entry.model_type,
            "file_size_bytes": old_entry.file_size_bytes,
            "status": old_entry.status,
            "is_accessible": old_entry.is_accessible
        }

        old_entry.name = new_src.name
        old_entry.file_path = str(new_src)
        old_entry.file_extension = ext
        old_entry.model_type = self.identify_model_type(new_src)
        old_entry.file_size_bytes = new_src.stat().st_size
        old_entry.status = "Registered (Deserialization Held until Backend Integration)"
        old_entry.is_accessible = True
        old_entry.previous_reference = snapshot

        self._save_to_settings()
        self.model_updated.emit(old_entry)
        return True, f"Replaced with '{new_src.name}'."

    def revert_replacement(self, model_id: str) -> Tuple[bool, str]:
        entry = self._models.get(model_id)
        if not entry or not entry.previous_reference:
            return False, "No previous reference available to revert."

        prev = entry.previous_reference
        entry.name = prev["name"]
        entry.file_path = prev["file_path"]
        entry.file_extension = prev["file_extension"]
        entry.model_type = prev["model_type"]
        entry.file_size_bytes = prev["file_size_bytes"]
        entry.status = prev["status"]
        entry.is_accessible = prev["is_accessible"]
        entry.previous_reference = None

        self._save_to_settings()
        self.model_updated.emit(entry)
        return True, f"Reverted back to '{entry.name}'."

    def remove_model(self, model_id: str) -> bool:
        if model_id in self._models:
            del self._models[model_id]
            self._save_to_settings()
            self.model_removed.emit(model_id)
            return True
        return False

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def get_all_models(self) -> List[ModelEntry]:
        return list(self._models.values())
