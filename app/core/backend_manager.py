"""
Backend Import Manager & Module Mapping Service.
Allows users to register an external Python ML backend directory,
inspects files non-executively, validates project structure and dependencies,
and manages explicit typed mappings and adapter binding to the Qt frontend.
"""
from pathlib import Path
import os
import csv
import json
import importlib
from typing import Dict, List, Optional, Tuple, Any
from PySide6.QtCore import QObject, Signal, QSettings

from app.core.config import ROOT_DIR, ORGANIZATION_NAME, APP_NAME
from app.core.interfaces import (
    BackendRegistry,
    DisconnectedPredictionBackend,
    DisconnectedTrainingBackend,
    DisconnectedStaticCollectionBackend,
    DisconnectedDynamicRecordingBackend
)

# Connection Statuses (Strictly transparent, no fake connected claims)
STATUS_NO_BACKEND = "No Backend Configured"
STATUS_FILE_SELECTED = "Folder Selected (Unvalidated)"
STATUS_FILE_VALIDATED = "Structure Validated (Files Discovered)"
STATUS_INTEGRATION_PENDING = "Integration Pending (Awaiting Adapter)"
STATUS_CONNECTED = "Connected (Adapter Bound)"
STATUS_CONNECTION_ERROR = "Structure Error / Missing Files"

KNOWN_TARGET_FILES = [
    "train_model.py",
    "predict_live.py",
    "collect_static_data.py",
    "hand_detection.py",
    "static_signs.csv",
    "models"
]

REQUIRED_DEPENDENCIES = [
    ("mediapipe", "mediapipe"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy")
]


class BackendManager(QObject):
    """
    Frontend service for registering external Python ML backends and binding adapters.
    Validates files, dependencies, dataset integrity, and model files before connecting.
    """
    backend_changed = Signal(str)          # directory path
    status_changed = Signal(str, str)      # (status_text, severity)
    mappings_changed = Signal(dict)

    _instance: Optional["BackendManager"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._backend_dir: Optional[str] = None
        self._status: str = STATUS_NO_BACKEND
        self._detected_files: List[Dict[str, Any]] = []
        self._module_mappings: Dict[str, str] = {
            "static_training": "",
            "dynamic_training": "",
            "live_prediction": "",
            "static_collection": "",
            "static_dataset": "",
            "dynamic_dataset": "",
            "models_dir": ""
        }
        self._load_from_settings()

        # Seed with workspace root if present and not yet registered
        if not self._backend_dir and (ROOT_DIR / "train_model.py").exists():
            self.register_backend_directory(str(ROOT_DIR.resolve()))
        elif self._backend_dir:
            self.validate_and_connect()

    @classmethod
    def get_instance(cls) -> "BackendManager":
        if cls._instance is None:
            cls._instance = BackendManager()
        return cls._instance

    @staticmethod
    def _settings():
        return QSettings(ORGANIZATION_NAME, APP_NAME)

    def _save_to_settings(self):
        s = self._settings()
        s.setValue("backend_manager/backend_dir", self._backend_dir or "")
        s.setValue("backend_manager/status", self._status)
        s.setValue("backend_manager/mappings", json.dumps(self._module_mappings))

    def _load_from_settings(self):
        s = self._settings()
        self._backend_dir = s.value("backend_manager/backend_dir", "", type=str) or None
        self._status = s.value("backend_manager/status", STATUS_NO_BACKEND, type=str)
        raw_map = s.value("backend_manager/mappings", "{}", type=str)
        try:
            loaded = json.loads(raw_map)
            self._module_mappings.update(loaded)
        except Exception:
            pass
        if self._backend_dir:
            self.scan_directory(self._backend_dir)
            self._clean_and_correct_mappings()
            self._save_to_settings()

    def _clean_and_correct_mappings(self):
        """
        Validate and enforce architectural correctness of module mappings.
        Fixes stale/inverted mappings stored in QSettings:
        - static_training must map to train_model.py (not collect_static_data.py)
        - live_prediction must map to predict_live.py (not train_model.py)
        - static_collection must map to collect_static_data.py
        - dynamic_training must NOT reuse static collection or static training scripts
        - dynamic_dataset must NOT reuse static_signs.csv
        """
        st = self._module_mappings.get("static_training", "")
        lp = self._module_mappings.get("live_prediction", "")
        sc = self._module_mappings.get("static_collection", "")
        dt = self._module_mappings.get("dynamic_training", "")
        dd = self._module_mappings.get("dynamic_dataset", "")

        filenames = {f["name"]: f["path"] for f in self._detected_files}

        # Fix static_training if mapped to collection script
        if "collect_static_data" in st.lower() or not st:
            if "train_model.py" in filenames:
                self._module_mappings["static_training"] = filenames["train_model.py"]

        # Fix live_prediction if mapped to train_model
        if "train_model" in lp.lower() or not lp:
            if "predict_live.py" in filenames:
                self._module_mappings["live_prediction"] = filenames["predict_live.py"]

        # Fix static_collection
        if not sc or "train_model" in sc.lower():
            if "collect_static_data.py" in filenames:
                self._module_mappings["static_collection"] = filenames["collect_static_data.py"]

        # Dynamic sequence training must NOT reuse static scripts
        if "collect_static_data" in dt.lower() or "train_model" in dt.lower():
            self._module_mappings["dynamic_training"] = ""

        # Dynamic dataset must NOT reuse static dataset
        if "static_signs" in dd.lower():
            self._module_mappings["dynamic_dataset"] = ""

        # Ensure static dataset
        if not self._module_mappings.get("static_dataset") and "static_signs.csv" in filenames:
            self._module_mappings["static_dataset"] = filenames["static_signs.csv"]

        # Ensure models dir
        if not self._module_mappings.get("models_dir") and "models" in filenames:
            self._module_mappings["models_dir"] = filenames["models"]

    @property
    def backend_directory(self) -> Optional[str]:
        return self._backend_dir

    @property
    def status(self) -> str:
        return self._status

    @property
    def detected_files(self) -> List[Dict[str, Any]]:
        return self._detected_files

    @property
    def module_mappings(self) -> Dict[str, str]:
        return dict(self._module_mappings)

    def scan_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Inspect directory non-executively.
        Identifies Python scripts, CSVs, and model directories.
        """
        p = Path(dir_path)
        self._detected_files = []
        if not p.exists() or not p.is_dir():
            return []

        try:
            for item in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                # Ignore hidden folders or venvs
                if item.name.startswith(".") or item.name in ("__pycache__", "build", "dist", ".venv"):
                    continue

                is_dir = item.is_dir()
                size_str = "Folder" if is_dir else f"{item.stat().st_size / 1024:.1f} KB"
                ext = "directory" if is_dir else item.suffix.lower()

                is_known = item.name in KNOWN_TARGET_FILES
                file_info = {
                    "name": item.name,
                    "path": str(item.resolve()),
                    "is_directory": is_dir,
                    "size_str": size_str,
                    "extension": ext,
                    "is_known_project_file": is_known,
                    "role_hint": self._suggest_role_hint(item.name, is_dir)
                }
                self._detected_files.append(file_info)
        except Exception:
            pass

        return self._detected_files

    @staticmethod
    def _suggest_role_hint(filename: str, is_dir: bool) -> str:
        fn = filename.lower()
        if fn == "train_model.py":
            return "Static Training Module"
        elif fn == "collect_static_data.py":
            return "Static Data Collection Module"
        elif "dynamic" in fn and fn.endswith(".py"):
            return "Dynamic Training / Sequence Module"
        elif fn == "predict_live.py":
            return "Live Prediction Module"
        elif fn == "hand_detection.py":
            return "Landmark Detection Utility"
        elif fn == "static_signs.csv":
            return "Static Sign Dataset"
        elif is_dir and fn == "models":
            return "Trained Model Directory"
        elif fn.endswith(".py"):
            return "Python Script"
        elif fn.endswith(".csv"):
            return "Dataset File"
        return "File / Directory"

    def register_backend_directory(self, dir_path: str) -> Tuple[bool, str]:
        """Register folder, scan files, auto-populate mappings, and validate/connect."""
        p = Path(dir_path).resolve()
        if not p.exists() or not p.is_dir():
            self._status = STATUS_CONNECTION_ERROR
            self.status_changed.emit(self._status, "error")
            return False, f"Directory does not exist: {dir_path}"

        self._backend_dir = str(p)
        self.scan_directory(self._backend_dir)
        self._clean_and_correct_mappings()

        # Auto-match known project files if not yet mapped
        filenames = {f["name"]: f["path"] for f in self._detected_files}
        if "train_model.py" in filenames and not self._module_mappings.get("static_training"):
            self._module_mappings["static_training"] = filenames["train_model.py"]
        if "predict_live.py" in filenames and not self._module_mappings.get("live_prediction"):
            self._module_mappings["live_prediction"] = filenames["predict_live.py"]
        if "collect_static_data.py" in filenames and not self._module_mappings.get("static_collection"):
            self._module_mappings["static_collection"] = filenames["collect_static_data.py"]
        if "static_signs.csv" in filenames and not self._module_mappings.get("static_dataset"):
            self._module_mappings["static_dataset"] = filenames["static_signs.csv"]
        if "models" in filenames and not self._module_mappings.get("models_dir"):
            self._module_mappings["models_dir"] = filenames["models"]

        # Ensure dynamic_training is not improperly bound to static scripts
        if self._module_mappings.get("dynamic_training") in (filenames.get("collect_static_data.py"), filenames.get("train_model.py")):
            self._module_mappings["dynamic_training"] = ""

        # Validate structure, dependencies, models and bind adapters
        success, msg, _ = self.validate_and_connect()
        self._save_to_settings()
        self.backend_changed.emit(self._backend_dir)
        return success, msg

    def validate_structure(self) -> Tuple[bool, str, Dict[str, bool]]:
        """Validate presence of key project files without executing them."""
        if not self._backend_dir:
            self._status = STATUS_NO_BACKEND
            self.status_changed.emit(self._status, "neutral")
            return False, "No backend folder configured.", {}

        p = Path(self._backend_dir)
        if not p.exists():
            self._status = STATUS_CONNECTION_ERROR
            self.status_changed.emit(self._status, "error")
            return False, "Configured backend folder no longer exists on disk.", {}

        self.scan_directory(self._backend_dir)
        names = {f["name"] for f in self._detected_files}

        results = {
            "train_model.py": "train_model.py" in names,
            "predict_live.py": "predict_live.py" in names,
            "collect_static_data.py": "collect_static_data.py" in names,
            "hand_detection.py": "hand_detection.py" in names,
            "static_signs.csv": "static_signs.csv" in names,
            "models": "models" in names
        }

        return True, "Structure scanned.", results

    def _check_dependencies(self) -> Tuple[bool, List[str]]:
        """Verify that necessary Python ML libraries are importable."""
        missing = []
        for mod_name, pip_name in REQUIRED_DEPENDENCIES:
            try:
                importlib.import_module(mod_name)
            except ImportError:
                missing.append(pip_name)
        return len(missing) == 0, missing

    def _resolve_mapped_path(self, key: str, fallback_filename: str) -> Optional[Path]:
        """Resolve a module mapping key relative to backend_dir."""
        val = self._module_mappings.get(key, "").strip()
        if val:
            p = Path(val)
            if not p.is_absolute() and self._backend_dir:
                p = Path(self._backend_dir) / p
            if p.exists():
                return p

        # Fallback to backend_dir / fallback_filename
        if self._backend_dir:
            fb = Path(self._backend_dir) / fallback_filename
            if fb.exists():
                return fb
        return None

    def validate_and_connect(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Deep validation of files, dependencies, dataset integrity, and models.
        Binds adapters into BackendRegistry only if all checks succeed.
        """
        if not self._backend_dir:
            self._status = STATUS_NO_BACKEND
            self.status_changed.emit(self._status, "neutral")
            self._restore_disconnected_backends()
            return False, "No backend directory configured.", {}

        bdir = Path(self._backend_dir)
        if not bdir.exists():
            self._status = STATUS_CONNECTION_ERROR
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, f"Backend directory does not exist: {self._backend_dir}", {}

        # 1. Dependency validation
        deps_ok, missing_deps = self._check_dependencies()
        if not deps_ok:
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Missing Python ML dependencies: {', '.join(missing_deps)}. Please install them in the Python environment."
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"missing_dependencies": missing_deps}

        # 2. Key files resolution
        predict_script = self._resolve_mapped_path("live_prediction", "predict_live.py")
        train_script = self._resolve_mapped_path("static_training", "train_model.py")
        collect_script = self._resolve_mapped_path("static_collection", "collect_static_data.py")
        models_dir = self._resolve_mapped_path("models_dir", "models")
        dataset_file = self._resolve_mapped_path("static_dataset", "static_signs.csv")

        # Check models directory and files
        if not models_dir or not models_dir.is_dir():
            self._status = STATUS_CONNECTION_ERROR
            msg = "Missing models storage directory ('models')."
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"models_dir": False}

        hand_task_file = models_dir / "hand_landmarker.task"
        if not hand_task_file.exists() or hand_task_file.stat().st_size < 1000:
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Missing or corrupted MediaPipe task bundle at: {hand_task_file}"
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"hand_landmarker.task": False}

        classifier_file = models_dir / "asl_classifier.joblib"
        if not classifier_file.exists():
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Trained classifier model missing at: {classifier_file}. Please run model training first."
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"asl_classifier.joblib": False}

        # 3. Dataset validation
        if not dataset_file or not dataset_file.exists():
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Dataset file missing at: {bdir / 'static_signs.csv'}."
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"dataset": False}

        try:
            with open(dataset_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) != 64:
                    self._status = STATUS_CONNECTION_ERROR
                    msg = f"Invalid dataset format in {dataset_file.name}: expected 64 columns, got {len(header) if header else 0}."
                    self.status_changed.emit(self._status, "error")
                    self._restore_disconnected_backends()
                    return False, msg, {"dataset_valid": False}
        except Exception as e:
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Could not read dataset {dataset_file.name}: {e}"
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"dataset_read_error": str(e)}

        # 4. Instantiate and bind concrete adapters
        try:
            from app.core.backend_adapter import (
                PythonMLPredictionAdapter,
                PythonMLTrainingAdapter,
                PythonMLStaticCollectionAdapter,
                PythonMLDynamicRecordingAdapter
            )

            pred_adapter = PythonMLPredictionAdapter(
                model_task_path=str(hand_task_file),
                classifier_path=str(classifier_file)
            )
            train_adapter = PythonMLTrainingAdapter(
                dataset_path=str(dataset_file),
                model_dir=str(models_dir)
            )
            coll_adapter = PythonMLStaticCollectionAdapter(
                model_task_path=str(hand_task_file),
                dataset_path=str(dataset_file)
            )
            dynamic_adapter = PythonMLDynamicRecordingAdapter()

            BackendRegistry.set_prediction_backend(pred_adapter)
            BackendRegistry.set_training_backend(train_adapter)
            BackendRegistry.set_static_collection_backend(coll_adapter)
            BackendRegistry.set_dynamic_recording_backend(dynamic_adapter)

            self._status = STATUS_CONNECTED
            severity = "active"
            msg = f"ML Backend Connected: All modules bound ({models_dir.name}, {dataset_file.name})."
            self.status_changed.emit(self._status, severity)
            self._save_to_settings()
            return True, msg, {
                "prediction": True,
                "training": True,
                "collection": True,
                "models": str(models_dir),
                "dataset": str(dataset_file)
            }
        except Exception as e:
            self._status = STATUS_CONNECTION_ERROR
            msg = f"Failed to instantiate backend adapters: {e}"
            self.status_changed.emit(self._status, "error")
            self._restore_disconnected_backends()
            return False, msg, {"exception": str(e)}

    def _restore_disconnected_backends(self):
        """Restore safe disconnected backend contracts."""
        BackendRegistry.set_prediction_backend(DisconnectedPredictionBackend())
        BackendRegistry.set_training_backend(DisconnectedTrainingBackend())
        BackendRegistry.set_static_collection_backend(DisconnectedStaticCollectionBackend())
        BackendRegistry.set_dynamic_recording_backend(DisconnectedDynamicRecordingBackend())

    def set_mapping(self, module_key: str, file_path: str):
        if module_key in self._module_mappings:
            self._module_mappings[module_key] = file_path
            self._save_to_settings()
            self.mappings_changed.emit(self._module_mappings)
            # Revalidate with updated mapping
            if self._backend_dir:
                self.validate_and_connect()

    def remove_backend(self):
        self._backend_dir = None
        self._status = STATUS_NO_BACKEND
        self._detected_files = []
        for k in self._module_mappings:
            self._module_mappings[k] = ""
        self._restore_disconnected_backends()
        self._save_to_settings()
        self.backend_changed.emit("")
        self.status_changed.emit(self._status, "neutral")
        self.mappings_changed.emit(self._module_mappings)
