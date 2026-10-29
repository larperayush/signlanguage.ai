"""
Model Manager Registry & Harness Service for SignFlow Studio.
Registers runnable AI model packages (classifier + landmarker task + predict script)
and individual artifacts, enabling fast switching between trained models.
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
class ModelPackage:
    """A runnable model harness package containing classifier, task, and optional predict script."""
    id: str
    name: str
    folder_path: str
    classifier_path: str
    task_path: str
    predict_script: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    is_valid: bool = True


@dataclass
class ModelEntry:
    id: str
    name: str
    file_path: str
    file_extension: str
    model_type: str
    file_size_bytes: int
    status: str
    is_accessible: bool
    registered_at: float = field(default_factory=time.time)
    previous_reference: Optional[Dict[str, Any]] = None


def detect_model_in_folder(folder_path: str) -> Dict[str, Any]:
    """
    Inspect a folder to auto-detect model components:
    - predict_live.py
    - asl_classifier.joblib (or *.joblib classifier)
    - hand_landmarker.task (or *.task landmarker)
    """
    root = Path(folder_path).resolve()
    if not root.exists() or not root.is_dir():
        return {
            "is_valid": False,
            "error_msg": f"Directory not found: {folder_path}"
        }

    # Search candidates
    predict_script = None
    classifier_path = None
    task_path = None

    # Check root for predict_live.py
    candidate_scripts = list(root.glob("predict_live.py")) + list(root.glob("*.py"))
    if candidate_scripts:
        # Prefer predict_live.py if present
        named = [s for s in candidate_scripts if s.name.lower() == "predict_live.py"]
        predict_script = str(named[0]) if named else str(candidate_scripts[0])

    # Search for classifier (.joblib, .pkl)
    candidate_classifiers = (
        list(root.glob("models/*.joblib")) +
        list(root.glob("model/*.joblib")) +
        list(root.glob("*.joblib")) +
        list(root.glob("models/*.pkl")) +
        list(root.glob("*.pkl"))
    )
    if candidate_classifiers:
        # Prefer asl_classifier.joblib if present
        named = [c for c in candidate_classifiers if "classifier" in c.name.lower()]
        classifier_path = str(named[0]) if named else str(candidate_classifiers[0])

    # Search for hand_landmarker.task
    candidate_tasks = (
        list(root.glob("models/*.task")) +
        list(root.glob("model/*.task")) +
        list(root.glob("*.task"))
    )
    if candidate_tasks:
        task_path = str(candidate_tasks[0])
    elif HAND_LANDMARKER_TASK.exists():
        # Fallback to studio's default hand_landmarker.task
        task_path = str(HAND_LANDMARKER_TASK.resolve())

    # Validation
    if not classifier_path:
        return {
            "is_valid": False,
            "error_msg": "No classifier file (*.joblib / *.pkl) found in selected folder or 'models/' subfolder."
        }
    if not task_path:
        return {
            "is_valid": False,
            "error_msg": "No MediaPipe task bundle (*.task) found in selected folder or studio defaults."
        }

    # Extract class labels safely if possible
    classes: List[str] = []
    try:
        import joblib
        clf = joblib.load(classifier_path)
        if hasattr(clf, "classes_"):
            classes = [str(c) for c in clf.classes_]
    except Exception:
        pass

    return {
        "is_valid": True,
        "folder_path": str(root),
        "suggested_name": f"{root.name} ASL Model",
        "predict_script": predict_script,
        "classifier_path": classifier_path,
        "task_path": task_path,
        "classes": classes,
        "error_msg": ""
    }


class ModelRegistry(QObject):
    """
    Application model registry managing runnable model packages and individual artifacts.
    """
    model_added = Signal(object)
    model_updated = Signal(object)
    model_removed = Signal(str)

    package_added = Signal(object)       # Emits ModelPackage
    package_removed = Signal(str)        # package_id
    active_package_changed = Signal(object)  # ModelPackage

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: Dict[str, ModelEntry] = {}
        self._packages: Dict[str, ModelPackage] = {}
        self._active_package_id: str = ""

        self._load_from_settings()
        self._seed_default_packages()

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
        serializable_entries = {k: asdict(v) for k, v in self._models.items()}
        s.setValue("model_registry/entries", json.dumps(serializable_entries))

        serializable_pkgs = {k: asdict(v) for k, v in self._packages.items()}
        s.setValue("model_registry/packages", json.dumps(serializable_pkgs))
        s.setValue("model_registry/active_package_id", self._active_package_id)

    def _load_from_settings(self):
        s = self._settings()
        raw_entries = s.value("model_registry/entries", "{}", type=str)
        try:
            data = json.loads(raw_entries)
            for k, d in data.items():
                self._models[k] = ModelEntry(**d)
        except Exception:
            self._models = {}

        raw_pkgs = s.value("model_registry/packages", "{}", type=str)
        try:
            pkg_data = json.loads(raw_pkgs)
            for k, d in pkg_data.items():
                self._packages[k] = ModelPackage(**d)
        except Exception:
            self._packages = {}

        self._active_package_id = s.value("model_registry/active_package_id", "", type=str)

    def _seed_default_packages(self):
        """Seed packages from local workspace and user's training directory if present."""
        # 1. Local workspace models
        if ASL_CLASSIFIER_MODEL.exists() and HAND_LANDMARKER_TASK.exists():
            default_id = "default_asl_classifier"
            if default_id not in self._packages:
                pkg = ModelPackage(
                    id=default_id,
                    name="Default ASL Classifier (Built-in)",
                    folder_path=str(ROOT_DIR.resolve()),
                    classifier_path=str(ASL_CLASSIFIER_MODEL.resolve()),
                    task_path=str(HAND_LANDMARKER_TASK.resolve()),
                    predict_script=str((ROOT_DIR / "predict_live.py").resolve()) if (ROOT_DIR / "predict_live.py").exists() else None,
                    classes=["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
                )
                self._packages[default_id] = pkg
                if not self._active_package_id:
                    self._active_package_id = default_id

        # 2. User's active training workspace in Downloads
        user_training_dir = Path(r"C:\Users\Rajat\Downloads\SignLanguageAI-20261026T190221Z-1-001\SignLanguageAI")
        if user_training_dir.exists():
            detected = detect_model_in_folder(str(user_training_dir))
            if detected["is_valid"]:
                user_pkg_id = "user_signlanguage_ai"
                if user_pkg_id not in self._packages:
                    user_pkg = ModelPackage(
                        id=user_pkg_id,
                        name="SignLanguageAI Active Model",
                        folder_path=detected["folder_path"],
                        classifier_path=detected["classifier_path"],
                        task_path=detected["task_path"],
                        predict_script=detected["predict_script"],
                        classes=detected["classes"]
                    )
                    self._packages[user_pkg_id] = user_pkg
                    # Prefer user's custom trained model if present!
                    self._active_package_id = user_pkg_id

        self._save_to_settings()

    def register_model_package(self, folder_path: str, custom_name: str = "") -> Tuple[bool, Optional[ModelPackage], str]:
        """
        Validate and register a complete model package from a directory.
        Checks predict_live.py, classifier model, and hand landmarker task.
        """
        detected = detect_model_in_folder(folder_path)
        if not detected["is_valid"]:
            return False, None, detected["error_msg"]

        pkg_id = str(uuid.uuid4())[:8]
        name = custom_name.strip() if custom_name.strip() else detected["suggested_name"]

        package = ModelPackage(
            id=pkg_id,
            name=name,
            folder_path=detected["folder_path"],
            classifier_path=detected["classifier_path"],
            task_path=detected["task_path"],
            predict_script=detected["predict_script"],
            classes=detected["classes"]
        )

        self._packages[pkg_id] = package
        self._active_package_id = pkg_id
        self._save_to_settings()

        self.package_added.emit(package)
        self.active_package_changed.emit(package)
        return True, package, f"Successfully registered '{name}'."

    def get_all_packages(self) -> List[ModelPackage]:
        return list(self._packages.values())

    def get_package(self, package_id: str) -> Optional[ModelPackage]:
        return self._packages.get(package_id)

    def get_active_package(self) -> Optional[ModelPackage]:
        if self._active_package_id in self._packages:
            return self._packages[self._active_package_id]
        if self._packages:
            # Pick first available
            first_id = next(iter(self._packages))
            self._active_package_id = first_id
            return self._packages[first_id]
        return None

    def set_active_package(self, package_id: str) -> bool:
        if package_id in self._packages:
            self._active_package_id = package_id
            self._save_to_settings()
            self.active_package_changed.emit(self._packages[package_id])
            return True
        return False

    def remove_package(self, package_id: str) -> bool:
        if package_id in self._packages:
            del self._packages[package_id]
            if self._active_package_id == package_id:
                self._active_package_id = next(iter(self._packages), "")
            self._save_to_settings()
            self.package_removed.emit(package_id)
            active = self.get_active_package()
            if active:
                self.active_package_changed.emit(active)
            return True
        return False

    # -------------------------------------------------------------------------
    # Backward compatibility for individual ModelEntry cataloging
    # -------------------------------------------------------------------------
    def get_all_models(self) -> List[ModelEntry]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        return self._models.get(model_id)

    def register_model(self, file_path: str, custom_name: str = "") -> Tuple[bool, Optional[ModelEntry], str]:
        src = Path(file_path).resolve()
        if not src.exists():
            return False, None, f"Model file not found: {file_path}"
        ext = src.suffix.lower()
        if ext not in MODEL_EXTENSIONS:
            return False, None, f"Unsupported extension: {ext}"

        entry_id = str(uuid.uuid4())[:8]
        entry = ModelEntry(
            id=entry_id,
            name=custom_name or src.name,
            file_path=str(src),
            file_extension=ext,
            model_type=MODEL_EXTENSIONS.get(ext, "Custom Model"),
            file_size_bytes=src.stat().st_size,
            status="Registered",
            is_accessible=True
        )
        self._models[entry.id] = entry
        self._save_to_settings()
        self.model_added.emit(entry)
        return True, entry, f"Registered {entry.name}"
