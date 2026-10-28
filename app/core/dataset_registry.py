"""
Dataset Management Registry & Validation Service.
Manages imported CSV datasets (both managed copies in data/ and external references).
Maintains replacement history for reversibility and provides CSV structural validation.
"""
from pathlib import Path
import os
import shutil
import csv
import json
import uuid
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

from PySide6.QtCore import QObject, Signal, QSettings

from app.core.config import ROOT_DIR, STATIC_SIGNS_CSV, ORGANIZATION_NAME, APP_NAME

MANAGED_DATA_DIR = ROOT_DIR / "data"


@dataclass
class DatasetEntry:
    id: str
    name: str
    file_path: str
    original_location: str
    import_mode: str          # "managed_copy" | "external_reference"
    file_size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]
    first_10_rows: List[List[str]]
    validation_status: str
    is_valid: bool
    registered_at: float = field(default_factory=time.time)
    previous_reference: Optional[Dict[str, Any]] = None  # To revert replacement


class DatasetRegistry(QObject):
    """
    Frontend service for managing CSV datasets.
    Does not invent model-specific dataset formats.
    """
    dataset_added = Signal(object)
    dataset_updated = Signal(object)
    dataset_removed = Signal(str)
    active_static_changed = Signal(object)
    active_dynamic_changed = Signal(object)

    _instance: Optional["DatasetRegistry"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        MANAGED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._datasets: Dict[str, DatasetEntry] = {}
        self._active_static_id: Optional[str] = None
        self._active_dynamic_id: Optional[str] = None
        self._load_from_settings()

        # Seed with static_signs.csv if available and not yet indexed
        if not self._datasets and STATIC_SIGNS_CSV.exists():
            self._register_initial_dataset(STATIC_SIGNS_CSV)

    @classmethod
    def get_instance(cls) -> "DatasetRegistry":
        if cls._instance is None:
            cls._instance = DatasetRegistry()
        return cls._instance

    @staticmethod
    def _settings():
        return QSettings(ORGANIZATION_NAME, APP_NAME)

    def _save_to_settings(self):
        s = self._settings()
        serializable = {k: asdict(v) for k, v in self._datasets.items()}
        s.setValue("dataset_registry/entries", json.dumps(serializable))
        s.setValue("dataset_registry/active_static", self._active_static_id or "")
        s.setValue("dataset_registry/active_dynamic", self._active_dynamic_id or "")

    def _load_from_settings(self):
        s = self._settings()
        raw = s.value("dataset_registry/entries", "{}", type=str)
        try:
            data = json.loads(raw)
            for k, d in data.items():
                self._datasets[k] = DatasetEntry(**d)
        except Exception:
            self._datasets = {}

        self._active_static_id = s.value("dataset_registry/active_static", "", type=str) or None
        self._active_dynamic_id = s.value("dataset_registry/active_dynamic", "", type=str) or None

    def _register_initial_dataset(self, path: Path):
        meta = self.inspect_csv(path)
        entry = DatasetEntry(
            id="default-static-signs",
            name="Default Static Signs Dataset",
            file_path=str(path.resolve()),
            original_location=str(path.resolve()),
            import_mode="external_reference",
            file_size_bytes=meta["file_size_bytes"],
            row_count=meta["row_count"],
            column_count=meta["column_count"],
            columns=meta["columns"],
            first_10_rows=meta["first_10_rows"],
            validation_status=meta["status"],
            is_valid=meta["is_valid"]
        )
        self._datasets[entry.id] = entry
        self._active_static_id = entry.id
        self._save_to_settings()

    @staticmethod
    def inspect_csv(path: Path) -> Dict[str, Any]:
        """Validate CSV readability and extract sample rows safely."""
        result = {
            "file_size_bytes": 0,
            "row_count": 0,
            "column_count": 0,
            "columns": [],
            "first_10_rows": [],
            "status": "Unvalidated",
            "is_valid": False,
            "error_msg": ""
        }

        if not path.exists():
            result["status"] = "File Not Found"
            result["error_msg"] = f"Path does not exist: {path}"
            return result

        result["file_size_bytes"] = path.stat().st_size
        if result["file_size_bytes"] == 0:
            result["status"] = "Empty File"
            result["error_msg"] = "Dataset file is 0 bytes."
            return result

        try:
            with open(path, mode="r", newline="", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    result["status"] = "Empty CSV"
                    result["error_msg"] = "No header row detected."
                    return result

                result["columns"] = [col.strip() for col in header]
                result["column_count"] = len(header)

                rows_sample = []
                count = 0
                for row in reader:
                    if row:
                        count += 1
                        if len(rows_sample) < 10:
                            rows_sample.append(row[:20])  # Cap preview columns for memory

                result["row_count"] = count
                result["first_10_rows"] = rows_sample

                if count == 0:
                    result["status"] = "Header Only (No Data Rows)"
                    result["error_msg"] = "CSV contains column headers but zero data rows."
                    result["is_valid"] = False
                else:
                    result["status"] = "Valid CSV Structure"
                    result["is_valid"] = True

        except Exception as e:
            result["status"] = "Read Error"
            result["error_msg"] = str(e)
            result["is_valid"] = False

        return result

    def import_dataset(self, file_path: str, mode: str = "managed_copy") -> Tuple[bool, Optional[DatasetEntry], str]:
        """
        Import CSV dataset via either 'managed_copy' or 'external_reference'.
        Returns (success: bool, entry: DatasetEntry, message: str).
        """
        src = Path(file_path).resolve()
        if not src.exists():
            return False, None, f"Source file does not exist: {file_path}"

        meta = self.inspect_csv(src)
        if not meta["is_valid"]:
            return False, None, f"Dataset validation failed: {meta['status']} ({meta['error_msg']})"

        entry_id = str(uuid.uuid4())[:8]
        dest_path = str(src)

        if mode == "managed_copy":
            # Copy into data/ directory without touching original
            dest = MANAGED_DATA_DIR / f"{entry_id}_{src.name}"
            try:
                shutil.copy2(src, dest)
                dest_path = str(dest.resolve())
            except Exception as e:
                return False, None, f"Failed to copy file to managed storage: {e}"

        entry = DatasetEntry(
            id=entry_id,
            name=src.name,
            file_path=dest_path,
            original_location=str(src),
            import_mode=mode,
            file_size_bytes=meta["file_size_bytes"],
            row_count=meta["row_count"],
            column_count=meta["column_count"],
            columns=meta["columns"],
            first_10_rows=meta["first_10_rows"],
            validation_status=meta["status"],
            is_valid=meta["is_valid"]
        )

        self._datasets[entry.id] = entry
        if not self._active_static_id:
            self._active_static_id = entry.id
        self._save_to_settings()
        self.dataset_added.emit(entry)
        return True, entry, f"Successfully imported '{src.name}' ({meta['row_count']:,} rows)."

    def replace_dataset(self, dataset_id: str, new_file_path: str, mode: str = "managed_copy") -> Tuple[bool, str]:
        """
        Replace an existing dataset reference while recording previous metadata
        so the replacement can be cleanly reverted.
        """
        old_entry = self._datasets.get(dataset_id)
        if not old_entry:
            return False, "Target dataset not found."

        new_src = Path(new_file_path).resolve()
        if not new_src.exists():
            return False, f"Replacement file not found: {new_file_path}"

        meta = self.inspect_csv(new_src)
        if not meta["is_valid"]:
            return False, f"Replacement dataset invalid: {meta['status']}"

        # Preserve snapshot of old entry for undo/revert
        previous_snapshot = {
            "file_path": old_entry.file_path,
            "original_location": old_entry.original_location,
            "name": old_entry.name,
            "file_size_bytes": old_entry.file_size_bytes,
            "row_count": old_entry.row_count,
            "column_count": old_entry.column_count,
            "columns": old_entry.columns,
            "first_10_rows": old_entry.first_10_rows,
            "validation_status": old_entry.validation_status,
            "import_mode": old_entry.import_mode,
        }

        dest_path = str(new_src)
        if mode == "managed_copy":
            dest = MANAGED_DATA_DIR / f"{old_entry.id}_{new_src.name}"
            try:
                shutil.copy2(new_src, dest)
                dest_path = str(dest.resolve())
            except Exception as e:
                return False, f"Failed copying replacement file: {e}"

        # Update entry
        old_entry.name = new_src.name
        old_entry.file_path = dest_path
        old_entry.original_location = str(new_src)
        old_entry.import_mode = mode
        old_entry.file_size_bytes = meta["file_size_bytes"]
        old_entry.row_count = meta["row_count"]
        old_entry.column_count = meta["column_count"]
        old_entry.columns = meta["columns"]
        old_entry.first_10_rows = meta["first_10_rows"]
        old_entry.validation_status = meta["status"]
        old_entry.is_valid = meta["is_valid"]
        old_entry.previous_reference = previous_snapshot

        self._save_to_settings()
        self.dataset_updated.emit(old_entry)
        return True, f"Replaced with '{new_src.name}'. Previous reference saved for rollback."

    def revert_replacement(self, dataset_id: str) -> Tuple[bool, str]:
        """Undo a replacement by restoring the previous dataset reference snapshot."""
        entry = self._datasets.get(dataset_id)
        if not entry or not entry.previous_reference:
            return False, "No previous reference snapshot available to revert."

        prev = entry.previous_reference
        entry.name = prev["name"]
        entry.file_path = prev["file_path"]
        entry.original_location = prev["original_location"]
        entry.import_mode = prev["import_mode"]
        entry.file_size_bytes = prev["file_size_bytes"]
        entry.row_count = prev["row_count"]
        entry.column_count = prev["column_count"]
        entry.columns = prev["columns"]
        entry.first_10_rows = prev["first_10_rows"]
        entry.validation_status = prev["validation_status"]
        entry.previous_reference = None  # Consumed revert

        self._save_to_settings()
        self.dataset_updated.emit(entry)
        return True, f"Reverted back to '{entry.name}'."

    def remove_dataset(self, dataset_id: str) -> bool:
        if dataset_id in self._datasets:
            del self._datasets[dataset_id]
            if self._active_static_id == dataset_id:
                self._active_static_id = next(iter(self._datasets), None)
            if self._active_dynamic_id == dataset_id:
                self._active_dynamic_id = None
            self._save_to_settings()
            self.dataset_removed.emit(dataset_id)
            return True
        return False

    def get_dataset(self, dataset_id: str) -> Optional[DatasetEntry]:
        return self._datasets.get(dataset_id)

    def get_all_datasets(self) -> List[DatasetEntry]:
        return list(self._datasets.values())

    def set_active_static_dataset(self, dataset_id: str):
        if dataset_id in self._datasets or dataset_id is None:
            self._active_static_id = dataset_id
            self._save_to_settings()
            self.active_static_changed.emit(self.get_active_static_dataset())

    def get_active_static_dataset(self) -> Optional[DatasetEntry]:
        if self._active_static_id and self._active_static_id in self._datasets:
            return self._datasets[self._active_static_id]
        return None

    def set_active_dynamic_dataset(self, dataset_id: str):
        if dataset_id in self._datasets or dataset_id is None:
            self._active_dynamic_id = dataset_id
            self._save_to_settings()
            self.active_dynamic_changed.emit(self.get_active_dynamic_dataset())

    def get_active_dynamic_dataset(self) -> Optional[DatasetEntry]:
        if self._active_dynamic_id and self._active_dynamic_id in self._datasets:
            return self._datasets[self._active_dynamic_id]
        return None
