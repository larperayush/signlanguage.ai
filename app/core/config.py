"""
Application configuration, paths, and persistent settings management.
"""
from pathlib import Path
from PySide6.QtCore import QSettings

APP_NAME = "SignFlow Studio"
APP_DISPLAY_NAME = "Sign Language Recognition Studio"
APP_VERSION = "1.0.0"
ORGANIZATION_NAME = "SignFlowAI"
ORGANIZATION_DOMAIN = "signflow.local"

# Root workspace directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
RESOURCES_DIR = APP_DIR / "resources"
MODELS_DIR = ROOT_DIR / "models"
STATIC_SIGNS_CSV = ROOT_DIR / "static_signs.csv"
ASL_CLASSIFIER_MODEL = MODELS_DIR / "asl_classifier.joblib"
HAND_LANDMARKER_TASK = MODELS_DIR / "hand_landmarker.task"

# Recognition Modes
MODE_STATIC = "Static"
MODE_DYNAMIC = "Dynamic"
MODE_COMBINED = "Combined"
RECOGNITION_MODES = [MODE_STATIC, MODE_DYNAMIC, MODE_COMBINED]

# Model Selections
MODEL_SELECTIONS = [
    "Static Model (asl_classifier.joblib)",
    "Dynamic Model (Movement J & Z)",
    "Combined Recognition (Static + Dynamic Router)"
]

# Settings Keys
KEY_THEME = "ui/theme"
KEY_PREFERRED_CAMERA = "camera/preferred_id"
KEY_PREFERRED_CAMERA_NAME = "camera/preferred_name"
KEY_PREFERRED_RESOLUTION = "camera/preferred_resolution"
KEY_MIRROR_PREVIEW = "camera/mirror_preview"
KEY_DEFAULT_MODE = "recognition/default_mode"
KEY_CONFIDENCE_THRESHOLD = "recognition/confidence_threshold"
KEY_GLOBAL_MODEL = "recognition/global_model"

class SettingsManager:
    """Type-safe helper for QSettings persistence."""
    
    @staticmethod
    def _settings():
        return QSettings(ORGANIZATION_NAME, APP_NAME)

    @classmethod
    def get_theme(cls) -> str:
        return cls._settings().value(KEY_THEME, "dark", type=str)

    @classmethod
    def set_theme(cls, theme: str):
        cls._settings().setValue(KEY_THEME, theme)

    @classmethod
    def get_preferred_camera_id(cls) -> str:
        return cls._settings().value(KEY_PREFERRED_CAMERA, "", type=str)

    @classmethod
    def set_preferred_camera_id(cls, cam_id: str, cam_name: str = ""):
        cls._settings().setValue(KEY_PREFERRED_CAMERA, cam_id)
        if cam_name:
            cls._settings().setValue(KEY_PREFERRED_CAMERA_NAME, cam_name)

    @classmethod
    def get_preferred_camera_name(cls) -> str:
        return cls._settings().value(KEY_PREFERRED_CAMERA_NAME, "", type=str)

    @classmethod
    def get_preferred_resolution(cls) -> str:
        return cls._settings().value(KEY_PREFERRED_RESOLUTION, "", type=str)

    @classmethod
    def set_preferred_resolution(cls, resolution: str):
        cls._settings().setValue(KEY_PREFERRED_RESOLUTION, resolution)

    @classmethod
    def get_mirror_preview(cls) -> bool:
        return cls._settings().value(KEY_MIRROR_PREVIEW, True, type=bool)

    @classmethod
    def set_mirror_preview(cls, mirror: bool):
        cls._settings().setValue(KEY_MIRROR_PREVIEW, mirror)

    @classmethod
    def get_default_mode(cls) -> str:
        mode = cls._settings().value(KEY_DEFAULT_MODE, MODE_STATIC, type=str)
        return mode if mode in RECOGNITION_MODES else MODE_STATIC

    @classmethod
    def set_default_mode(cls, mode: str):
        if mode in RECOGNITION_MODES:
            cls._settings().setValue(KEY_DEFAULT_MODE, mode)

    @classmethod
    def get_global_model(cls) -> str:
        val = cls._settings().value(KEY_GLOBAL_MODEL, MODEL_SELECTIONS[0], type=str)
        return val if val in MODEL_SELECTIONS else MODEL_SELECTIONS[0]

    @classmethod
    def set_global_model(cls, model_name: str):
        if model_name in MODEL_SELECTIONS:
            cls._settings().setValue(KEY_GLOBAL_MODEL, model_name)

    @classmethod
    def get_confidence_threshold(cls) -> float:
        return cls._settings().value(KEY_CONFIDENCE_THRESHOLD, 0.70, type=float)

    @classmethod
    def set_confidence_threshold(cls, threshold: float):
        cls._settings().setValue(KEY_CONFIDENCE_THRESHOLD, threshold)
