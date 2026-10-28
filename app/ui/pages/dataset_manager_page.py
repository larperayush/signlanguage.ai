"""
Dataset Manager Page.
Provides full local dataset import, preview, replacement with rollback,
and active dataset assignment for static and dynamic workflows.
"""
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QListWidget,
    QListWidgetItem, QFileDialog, QMessageBox, QDialog, QRadioButton,
    QButtonGroup, QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, Slot

from app.core.theme import ThemeManager
from app.core.dataset_registry import DatasetRegistry, DatasetEntry
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import SectionHeader, StatusPill, StatCard
from app.ui.components.error_dialog import ErrorDialog


class ImportDatasetDialog(QDialog):
    """Dialog allowing user to choose between managed copy vs external reference."""
    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Dataset Options")
        self.setMinimumWidth(440)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        palette = ThemeManager.get_palette()
        self.file_path = file_path

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(16)

        title = QLabel("Choose Import Storage Mode")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(title)

        file_lbl = QLabel(f"Selected: {Path(file_path).name}")
        file_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        layout.addWidget(file_lbl)

        # Radio options
        self.btn_group = QButtonGroup(self)

        self.radio_copy = QRadioButton("Import Copy (Recommended)")
        self.radio_copy.setChecked(True)
        self.radio_ref = QRadioButton("Use Existing File")

        self.btn_group.addButton(self.radio_copy, 0)
        self.btn_group.addButton(self.radio_ref, 1)

        copy_desc = QLabel("Copies the CSV into the application's managed 'data/' directory. The original file remains untouched.")
        copy_desc.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; margin-left: 20px;")
        copy_desc.setWordWrap(True)

        ref_desc = QLabel("Maintains a linked reference to the CSV at its current location on disk without copying.")
        ref_desc.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; margin-left: 20px;")
        ref_desc.setWordWrap(True)

        layout.addWidget(self.radio_copy)
        layout.addWidget(copy_desc)
        layout.addWidget(self.radio_ref)
        layout.addWidget(ref_desc)

        # Buttons
        btns = QHBoxLayout()
        btns.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setProperty("class", "secondaryBtn")
        cancel_btn.clicked.connect(self.reject)

        import_btn = QPushButton("Import Dataset")
        import_btn.setProperty("class", "primaryBtn")
        import_btn.clicked.connect(self.accept)

        btns.addWidget(cancel_btn)
        btns.addWidget(import_btn)
        layout.addLayout(btns)

    def get_selected_mode(self) -> str:
        return "managed_copy" if self.radio_copy.isChecked() else "external_reference"


class DatasetManagerPage(QWidget):
    """Complete local dataset management interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry = DatasetRegistry.get_instance()
        self._selected_dataset_id: Optional[str] = None

        palette = ThemeManager.get_palette()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header = SectionHeader(
            "Local Dataset Manager",
            "Import, preview, replace, and configure CSV training datasets for static and dynamic recognition."
        )
        main_layout.addWidget(header)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.import_btn = QPushButton(" Import CSV Dataset...")
        self.import_btn.setProperty("class", "primaryBtn")
        self.import_btn.setIcon(get_icon("plus", "#FFFFFF", size=15))
        self.import_btn.clicked.connect(self._on_import_click)
        toolbar.addWidget(self.import_btn)

        self.refresh_btn = QPushButton(" Refresh")
        self.refresh_btn.setProperty("class", "secondaryBtn")
        self.refresh_btn.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        self.refresh_btn.clicked.connect(self._refresh_list)
        toolbar.addWidget(self.refresh_btn)

        toolbar.addStretch()

        self.active_static_pill = StatusPill("Active Static: None", "neutral")
        self.active_dynamic_pill = StatusPill("Active Dynamic: None", "neutral")
        toolbar.addWidget(self.active_static_pill)
        toolbar.addWidget(self.active_dynamic_pill)

        main_layout.addLayout(toolbar)

        # Main Splitter (Left: Dataset List, Right: Details & Preview)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Pane: Datasets List
        left_frame = QFrame()
        left_frame.setProperty("class", "cardPanel")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        list_title = QLabel("REGISTERED DATASETS")
        list_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        left_layout.addWidget(list_title)

        self.dataset_list = QListWidget()
        self.dataset_list.setStyleSheet(f"""
            QListWidget {{
                background-color: transparent;
                border: none;
            }}
            QListWidget::item {{
                padding: 10px 8px;
                border-bottom: 1px solid {palette.border_subtle};
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {palette.accent_subtle};
                color: {palette.accent};
            }}
        """)
        self.dataset_list.currentItemChanged.connect(self._on_item_selected)
        left_layout.addWidget(self.dataset_list)
        splitter.addWidget(left_frame)

        # Right Pane: Inspector & Data Preview
        right_frame = QFrame()
        right_frame.setProperty("class", "cardPanel")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(14)

        # Details Header
        self.details_title = QLabel("Select a dataset to inspect details")
        self.details_title.setStyleSheet("font-size: 15px; font-weight: 700;")
        right_layout.addWidget(self.details_title)

        # Metadata Grid
        self.meta_grid = QGridLayout()
        self.meta_grid.setSpacing(8)

        self.lbl_loc = QLabel("—")
        self.lbl_size = QLabel("—")
        self.lbl_rows = QLabel("—")
        self.lbl_cols = QLabel("—")
        self.lbl_mode = QLabel("—")
        self.lbl_status = QLabel("—")

        self.meta_grid.addWidget(QLabel("Original Location:"), 0, 0)
        self.meta_grid.addWidget(self.lbl_loc, 0, 1)
        self.meta_grid.addWidget(QLabel("Storage Mode:"), 0, 2)
        self.meta_grid.addWidget(self.lbl_mode, 0, 3)

        self.meta_grid.addWidget(QLabel("File Size:"), 1, 0)
        self.meta_grid.addWidget(self.lbl_size, 1, 1)
        self.meta_grid.addWidget(QLabel("Row Count:"), 1, 2)
        self.meta_grid.addWidget(self.lbl_rows, 1, 3)

        self.meta_grid.addWidget(QLabel("Columns:"), 2, 0)
        self.meta_grid.addWidget(self.lbl_cols, 2, 1)
        self.meta_grid.addWidget(QLabel("Validation Status:"), 2, 2)
        self.meta_grid.addWidget(self.lbl_status, 2, 3)

        right_layout.addLayout(self.meta_grid)

        # Actions Bar for Selected Dataset
        action_bar = QHBoxLayout()
        action_bar.setSpacing(8)

        self.btn_set_static = QPushButton("Set Active Static")
        self.btn_set_static.setProperty("class", "secondaryBtn")
        self.btn_set_static.clicked.connect(self._set_active_static)

        self.btn_set_dynamic = QPushButton("Set Active Dynamic")
        self.btn_set_dynamic.setProperty("class", "secondaryBtn")
        self.btn_set_dynamic.clicked.connect(self._set_active_dynamic)

        self.btn_replace = QPushButton("Replace Dataset...")
        self.btn_replace.setProperty("class", "secondaryBtn")
        self.btn_replace.clicked.connect(self._replace_dataset)

        self.btn_revert = QPushButton("Revert Replacement")
        self.btn_revert.setProperty("class", "secondaryBtn")
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._revert_dataset)

        self.btn_remove = QPushButton("Remove Reference")
        self.btn_remove.setProperty("class", "dangerBtn")
        self.btn_remove.clicked.connect(self._remove_dataset)

        action_bar.addWidget(self.btn_set_static)
        action_bar.addWidget(self.btn_set_dynamic)
        action_bar.addWidget(self.btn_replace)
        action_bar.addWidget(self.btn_revert)
        action_bar.addStretch()
        action_bar.addWidget(self.btn_remove)
        right_layout.addLayout(action_bar)

        # First 10 Rows Preview Table
        preview_label = QLabel("DATASET PREVIEW (FIRST 10 ROWS)")
        preview_label.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        right_layout.addWidget(preview_label)

        self.preview_table = QTableWidget(0, 0)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.preview_table, 1)

        splitter.addWidget(right_frame)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, 1)

        # Wire registry signals
        self._registry.dataset_added.connect(self._on_registry_changed)
        self._registry.dataset_updated.connect(self._on_registry_changed)
        self._registry.dataset_removed.connect(self._on_registry_changed)
        self._registry.active_static_changed.connect(self._update_active_badges)
        self._registry.active_dynamic_changed.connect(self._update_active_badges)

        self._refresh_list()
        self._update_active_badges()

    def _on_import_click(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV Dataset to Import",
            "",
            "CSV Dataset Files (*.csv)"
        )
        if file_path:
            dialog = ImportDatasetDialog(file_path, self)
            if dialog.exec():
                mode = dialog.get_selected_mode()
                success, entry, msg = self._registry.import_dataset(file_path, mode=mode)
                if success:
                    self._refresh_list(select_id=entry.id)
                    QMessageBox.information(self, "Dataset Imported", msg)
                else:
                    ErrorDialog.show_invalid_csv(file_path, msg, parent=self)

    def _refresh_list(self, select_id: Optional[str] = None):
        self.dataset_list.clear()
        datasets = self._registry.get_all_datasets()

        target_row = 0
        for row, d in enumerate(datasets):
            mode_tag = "[Managed Copy]" if d.import_mode == "managed_copy" else "[Linked Ref]"
            item_text = f"{d.name}\n{d.row_count:,} rows  •  {d.file_size_bytes / 1024:.1f} KB  •  {mode_tag}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, d.id)
            self.dataset_list.addItem(item)

            if select_id and d.id == select_id:
                target_row = row
            elif self._selected_dataset_id and d.id == self._selected_dataset_id:
                target_row = row

        if datasets:
            self.dataset_list.setCurrentRow(target_row)
        else:
            self._clear_details()

    def _on_item_selected(self, current: QListWidgetItem, previous=None):
        if not current:
            self._clear_details()
            return

        ds_id = current.data(Qt.ItemDataRole.UserRole)
        self._selected_dataset_id = ds_id
        entry = self._registry.get_dataset(ds_id)
        if entry:
            self._display_details(entry)

    def _display_details(self, entry: DatasetEntry):
        palette = ThemeManager.get_palette()
        self.details_title.setText(entry.name)
        self.lbl_loc.setText(entry.original_location)
        self.lbl_mode.setText("Managed Copy (in data/)" if entry.import_mode == "managed_copy" else "External Reference (Linked)")
        self.lbl_size.setText(f"{entry.file_size_bytes / 1024:.1f} KB ({entry.file_size_bytes:,} bytes)")
        self.lbl_rows.setText(f"{entry.row_count:,} rows")
        self.lbl_cols.setText(f"{entry.column_count} columns")

        status_color = palette.success if entry.is_valid else palette.danger
        self.lbl_status.setText(entry.validation_status)
        self.lbl_status.setStyleSheet(f"color: {status_color}; font-weight: 600;")

        # Enable Revert if replacement snapshot exists
        self.btn_revert.setEnabled(entry.previous_reference is not None)

        # Populate Preview Table
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(entry.columns))
        self.preview_table.setHorizontalHeaderLabels(entry.columns)
        self.preview_table.setRowCount(len(entry.first_10_rows))

        for r_idx, row in enumerate(entry.first_10_rows):
            for c_idx, val in enumerate(row):
                if c_idx < len(entry.columns):
                    self.preview_table.setItem(r_idx, c_idx, QTableWidgetItem(str(val)))

    def _clear_details(self):
        self.details_title.setText("No dataset selected")
        self.lbl_loc.setText("—")
        self.lbl_mode.setText("—")
        self.lbl_size.setText("—")
        self.lbl_rows.setText("—")
        self.lbl_cols.setText("—")
        self.lbl_status.setText("—")
        self.btn_revert.setEnabled(False)
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)

    def _set_active_static(self):
        if self._selected_dataset_id:
            self._registry.set_active_static_dataset(self._selected_dataset_id)
            self._update_active_badges()

    def _set_active_dynamic(self):
        if self._selected_dataset_id:
            self._registry.set_active_dynamic_dataset(self._selected_dataset_id)
            self._update_active_badges()

    def _update_active_badges(self, *args):
        st = self._registry.get_active_static_dataset()
        dy = self._registry.get_active_dynamic_dataset()

        self.active_static_pill.set_status(
            f"Active Static: {st.name if st else 'None'}",
            "active" if st else "neutral"
        )
        self.active_dynamic_pill.set_status(
            f"Active Dynamic: {dy.name if dy else 'None'}",
            "warning" if dy else "neutral"
        )

    def _replace_dataset(self):
        if not self._selected_dataset_id:
            return

        entry = self._registry.get_dataset(self._selected_dataset_id)
        if not entry:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Dataset Replacement",
            f"Are you sure you want to replace '{entry.name}'?\n\n"
            "The previous reference will be recorded so you can reverse this replacement if needed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Replacement CSV Dataset", "", "CSV Dataset Files (*.csv)"
        )
        if file_path:
            dialog = ImportDatasetDialog(file_path, self)
            if dialog.exec():
                mode = dialog.get_selected_mode()
                success, msg = self._registry.replace_dataset(self._selected_dataset_id, file_path, mode=mode)
                if success:
                    QMessageBox.information(self, "Dataset Replaced", msg)
                    self._refresh_list(self._selected_dataset_id)
                else:
                    ErrorDialog.show_invalid_csv(file_path, msg, parent=self)

    def _revert_dataset(self):
        if not self._selected_dataset_id:
            return

        success, msg = self._registry.revert_replacement(self._selected_dataset_id)
        if success:
            QMessageBox.information(self, "Replacement Reverted", msg)
            self._refresh_list(self._selected_dataset_id)
        else:
            QMessageBox.warning(self, "Cannot Revert", msg)

    def _remove_dataset(self):
        if not self._selected_dataset_id:
            return

        entry = self._registry.get_dataset(self._selected_dataset_id)
        if not entry:
            return

        confirm = QMessageBox.question(
            self,
            "Remove Dataset Reference",
            f"Remove reference to '{entry.name}' from the manager?\n(Underlying file will not be deleted)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._registry.remove_dataset(self._selected_dataset_id)
            self._refresh_list()

    def _on_registry_changed(self, *args):
        self._refresh_list(self._selected_dataset_id)
        self._update_active_badges()
