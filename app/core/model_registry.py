"""
Model Manager Registry & Open-Source Harness Service for SignFlow Studio.
Acts as a pure harness: no built-in models.
Allows importing:
- User Model Packages (inference only: predict_live.py + models/asl_classifier.joblib + models/hand_landmarker.task)
- Developer Model Packages (full suite: includes collect_static/dynamic, train_model.py, datasets, etc.)
"""
from pathlib import Path
import json
import uuid
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from PySide6.QtCore import QObject, Signal, QSettings

from app.core.config import ORGANIZATION_NAME, APP_NAME

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
    """A runnable model harness package containing classifier, task, and predict script."""
    id: str
    name: str
    folder_path: str
    classifier_path: str
    task_path: str
    predict_script: str
    is_developer: bool = False
    model_type: str = "user"  # "developer" or "user"
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
    Inspect a folder to determine whether it is a User Model or Developer Model.
    
    Structure Required:
    - User Model:
        predict_live.py
        models/asl_classifier.joblib
        models/hand_landmarker.task
        
    - Developer Model adds:
        collect_dynamic_data.py
        collect_static_data.py
        train_model.py
        models (folder)
        old models (folder)
        dynamic_signs.csv
        static_signs.csv
    """
    root = Path(folder_path).resolve()
    if not root.exists() or not root.is_dir():
        return {
            "is_valid": False,
            "error_msg": f"Directory not found: {folder_path}"
        }

    # 1. Check core execution script: predict_live.py
    predict_script = root / "predict_live.py"
    if not predict_script.exists():
        candidates = list(root.glob("*.py"))
        if candidates:
            predict_script = candidates[0]
        else:
            return {
                "is_valid": False,
                "error_msg": "Missing predict_live.py in the selected folder."
            }

    # 2. Check classifier in models/ or root
    models_dir = root / "models"
    classifier_path = None
    if models_dir.exists():
        c_target = models_dir / "asl_classifier.joblib"
        if c_target.exists():
            classifier_path = c_target
        else:
            joblibs = list(models_dir.glob("*.joblib")) + list(models_dir.glob("*.pkl"))
            if joblibs:
                classifier_path = joblibs[0]

    if not classifier_path:
        root_joblibs = list(root.glob("*.joblib")) + list(root.glob("*.pkl"))
        if root_joblibs:
            classifier_path = root_joblibs[0]

    if not classifier_path:
        return {
            "is_valid": False,
            "error_msg": "Missing asl_classifier.joblib (checked models/ and root folder)."
        }

    # 3. Check MediaPipe hand task in models/ or root
    task_path = None
    if models_dir.exists():
        t_target = models_dir / "hand_landmarker.task"
        if t_target.exists():
            task_path = t_target
        else:
            tasks = list(models_dir.glob("*.task"))
            if tasks:
                task_path = tasks[0]

    if not task_path:
        root_tasks = list(root.glob("*.task"))
        if root_tasks:
            task_path = root_tasks[0]

    if not task_path:
        return {
            "is_valid": False,
            "error_msg": "Missing hand_landmarker.task (checked models/ and root folder)."
        }

    # 4. Check Developer Model Suite criteria
    dev_files = {
        "collect_dynamic_data.py": (root / "collect_dynamic_data.py").exists(),
        "collect_static_data.py": (root / "collect_static_data.py").exists(),
        "predict_live.py": (root / "predict_live.py").exists(),
        "train_model.py": (root / "train_model.py").exists(),
        "models": (root / "models").is_dir(),
        "old_models": (root / "old models").is_dir() or (root / "old_models").is_dir(),
        "dynamic_signs.csv": (root / "dynamic_signs.csv").exists() or (root / "dynamic_sign.csv").exists(),
        "static_signs.csv": (root / "static_signs.csv").exists() or (root / "static_sign.csv").exists(),
    }

    # Qualifying as Developer Model requires train_model.py, collection scripts, and dataset CSVs
    is_developer = (
        dev_files["train_model.py"] and
        dev_files["collect_static_data.py"] and
        (dev_files["static_signs.csv"] or dev_files["dynamic_signs.csv"])
    )

    model_type = "developer" if is_developer else "user"
    badge_label = "Developer Suite" if is_developer else "User Model"
    suggested_name = f"{root.name} [{badge_label}]"

    # Extract class labels safely if possible
    classes: List[str] = []
    try:
        import joblib
        clf = joblib.load(str(classifier_path))
        if hasattr(clf, "classes_"):
            classes = [str(c) for c in clf.classes_]
        elif isinstance(clf, dict) and "classes_" in clf:
            classes = [str(c) for c in clf["classes_"]]
    except Exception:
        pass

    return {
        "is_valid": True,
        "is_developer": is_developer,
        "model_type": model_type,
        "folder_path": str(root),
        "suggested_name": suggested_name,
        "predict_script": str(predict_script.resolve()),
        "classifier_path": str(classifier_path.resolve()),
        "task_path": str(task_path.resolve()),
        "classes": classes,
        "dev_files": dev_files,
        "error_msg": ""
    }


class ModelRegistry(QObject):
    """
    Pure Model Harness Registry.
    Contains zero hardcoded built-in models.
    Loads only models explicitly added by the user.
    """
    model_added = Signal(object)
    model_updated = Signal(object)
    model_removed = Signal(str)

    package_added = Signal(object)       # Emits ModelPackage
    package_removed = Signal(str)        # package_id
    active_package_changed = Signal(object)  # Optional[ModelPackage]

    _instance: Optional["ModelRegistry"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._models: Dict[str, ModelEntry] = {}
        self._packages: Dict[str, ModelPackage] = {}
        self._active_package_id: str = ""

        self._load_from_settings()

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
        raw_pkgs = s.value("model_registry/packages", "{}", type=str)
        try:
            pkg_data = json.loads(raw_pkgs)
            self._packages = {}
            for k, d in pkg_data.items():
                if k.startswith("default_") or k == "user_signlanguage_ai":
                    continue
                # Verify that package folder and files still exist on disk
                folder = Path(d.get("folder_path", ""))
                clf = Path(d.get("classifier_path", ""))
                if folder.exists() and clf.exists():
                    self._packages[k] = ModelPackage(**d)
        except Exception:
            self._packages = {}

        self._active_package_id = s.value("model_registry/active_package_id", "", type=str)
        if self._active_package_id not in self._packages:
            self._active_package_id = next(iter(self._packages), "")

    def clear_all_packages(self):
        """Purge any saved packages to ensure completely clean harness state."""
        self._packages.clear()
        self._active_package_id = ""
        self._save_to_settings()
        self.active_package_changed.emit(None)

    def register_model_package(self, folder_path: str, custom_name: str = "") -> Tuple[bool, Optional[ModelPackage], str]:
        """
        Validate and register a complete model package from a directory.
        Identifies whether it is a User Model or Developer Model Suite.
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
            is_developer=detected["is_developer"],
            model_type=detected["model_type"],
            classes=detected["classes"]
        )

        self._packages[pkg_id] = package
        self._active_package_id = pkg_id
        self._save_to_settings()

        self.package_added.emit(package)
        self.active_package_changed.emit(package)
        return True, package, f"Successfully registered '{name}' ({'Developer Suite' if package.is_developer else 'User Model'})."

    def get_all_packages(self) -> List[ModelPackage]:
        return list(self._packages.values())

    def get_packages(self) -> List[ModelPackage]:
        return self.get_all_packages()

    def get_package(self, package_id: str) -> Optional[ModelPackage]:
        return self._packages.get(package_id)

    def get_active_package(self) -> Optional[ModelPackage]:
        if self._active_package_id in self._packages:
            return self._packages[self._active_package_id]
        if self._packages:
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
