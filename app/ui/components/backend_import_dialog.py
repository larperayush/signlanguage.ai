"""
Backend Import Manager Modal Dialog.
Allows users to select an existing Python ML backend folder,
view detected files, validate project structure, and map modules to frontend screens.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QFileDialog, QMessageBox, QScrollArea, QWidget
)
from PySide6.QtCore import Qt

from app.core.theme import ThemeManager
from app.core.backend_manager import (
    BackendManager, STATUS_NO_BACKEND, STATUS_FILE_SELECTED,
    STATUS_FILE_VALIDATED, STATUS_INTEGRATION_PENDING, STATUS_CONNECTED,
    STATUS_CONNECTION_ERROR
)
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill, SectionHeader


class BackendImportDialog(QDialog):
    """
    Dedicated manager for registering and mapping external Python ML folders.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Backend Import & Module Mapping Manager")
        self.setMinimumSize(820, 640)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._manager = BackendManager.get_instance()
        palette = ThemeManager.get_palette()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        content = QWidget()
        content.setObjectName("contentContainer")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(18)

        # Header
        header = SectionHeader(
            "Python ML Backend Import Manager",
            "Register an existing machine learning project directory and map Python modules to frontend screens."
        )
        layout.addWidget(header)

        # Directory Selection Card
        dir_frame = QFrame()
        dir_frame.setProperty("class", "cardPanel")
        df_layout = QVBoxLayout(dir_frame)
        df_layout.setContentsMargins(18, 16, 18, 16)
        df_layout.setSpacing(12)

        dir_top = QHBoxLayout()
        dir_title = QLabel("REGISTERED BACKEND DIRECTORY")
        dir_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        
        self.status_pill = StatusPill(self._manager.status, self._get_status_severity(self._manager.status))
        dir_top.addWidget(dir_title)
        dir_top.addStretch()
        dir_top.addWidget(self.status_pill)
        df_layout.addLayout(dir_top)

        # Path display & buttons
        path_row = QHBoxLayout()
        path_row.setSpacing(10)

        self.path_label = QLabel(self._manager.backend_directory or "No directory selected")
        self.path_label.setStyleSheet(f"""
            background-color: {palette.bg_input};
            border: 1px solid {palette.border_default};
            border-radius: 6px;
            padding: 8px 12px;
            font-family: monospace;
            font-size: 12px;
        """)
        path_row.addWidget(self.path_label, 1)

        self.browse_btn = QPushButton(" Select Folder...")
        self.browse_btn.setProperty("class", "primaryBtn")
        self.browse_btn.setIcon(get_icon("folder", "#FFFFFF", size=15))
        self.browse_btn.clicked.connect(self._browse_folder)
        path_row.addWidget(self.browse_btn)

        self.validate_btn = QPushButton(" Validate Structure")
        self.validate_btn.setProperty("class", "secondaryBtn")
        self.validate_btn.setIcon(get_icon("check", palette.text_secondary, palette.accent, size=15))
        self.validate_btn.clicked.connect(self._validate_backend)
        path_row.addWidget(self.validate_btn)

        self.remove_btn = QPushButton(" Disconnect")
        self.remove_btn.setProperty("class", "secondaryBtn")
        self.remove_btn.setIcon(get_icon("trash", palette.text_secondary, palette.danger, size=14))
        self.remove_btn.clicked.connect(self._remove_backend)
        path_row.addWidget(self.remove_btn)

        df_layout.addLayout(path_row)
        layout.addWidget(dir_frame)

        # Detected Files Inspection Table
        table_header = QLabel("DETECTED PROJECT FILES & ARTIFACTS")
        table_header.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(table_header)

        self.files_table = QTableWidget(0, 4)
        self.files_table.setHorizontalHeaderLabels(["Item Name", "Type / Extension", "Size", "Identified Project Role"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.files_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.files_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.files_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.files_table.setFixedHeight(170)
        self.files_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.files_table)

        # Module Mapping Section
        map_frame = QFrame()
        map_frame.setProperty("class", "cardPanelElevated")
        mf_layout = QVBoxLayout(map_frame)
        mf_layout.setContentsMargins(18, 16, 18, 16)
        mf_layout.setSpacing(14)

        map_title = QLabel("FRONTEND MODULE TO BACKEND SCRIPT MAPPINGS")
        map_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        mf_layout.addWidget(map_title)

        grid = QGridLayout()
        grid.setSpacing(10)

        # 1. Static Training
        grid.addWidget(QLabel("Static Training Pipeline:"), 0, 0)
        self.combo_static_train = QComboBox()
        grid.addWidget(self.combo_static_train, 0, 1)

        # 2. Static Collection
        grid.addWidget(QLabel("Static Data Collection:"), 1, 0)
        self.combo_static_collect = QComboBox()
        grid.addWidget(self.combo_static_collect, 1, 1)

        # 3. Dynamic Training
        grid.addWidget(QLabel("Dynamic Sequence Training:"), 2, 0)
        self.combo_dynamic_train = QComboBox()
        grid.addWidget(self.combo_dynamic_train, 2, 1)

        # 4. Live Prediction
        grid.addWidget(QLabel("Live Prediction Pipeline:"), 3, 0)
        self.combo_predict = QComboBox()
        grid.addWidget(self.combo_predict, 3, 1)

        # 5. Static Dataset
        grid.addWidget(QLabel("Static Dataset Reference:"), 4, 0)
        self.combo_static_ds = QComboBox()
        grid.addWidget(self.combo_static_ds, 4, 1)

        # 6. Dynamic Dataset
        grid.addWidget(QLabel("Dynamic Dataset Reference:"), 5, 0)
        self.combo_dynamic_ds = QComboBox()
        grid.addWidget(self.combo_dynamic_ds, 5, 1)

        # 7. Models Directory
        grid.addWidget(QLabel("Models Storage Directory:"), 6, 0)
        self.combo_models_dir = QComboBox()
        grid.addWidget(self.combo_models_dir, 6, 1)

        mf_layout.addLayout(grid)
        layout.addWidget(map_frame)

        # Architectural Guard Note
        note = QLabel(
            "Architectural Boundary Note: Registering a Python file maps the integration reference only. "
            "Arbitrary Python code is NEVER automatically executed. Actual training and prediction routines "
            "will be attached via the typed adapter contracts in 'app/core/interfaces.py'."
        )
        note.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-style: italic;")
        note.setWordWrap(True)
        layout.addWidget(note)

        # Bottom Action Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.addStretch()

        save_btn = QPushButton("Save Mappings & Close")
        save_btn.setProperty("class", "primaryBtn")
        save_btn.clicked.connect(self._save_mappings_and_close)
        bottom_bar.addWidget(save_btn)

        layout.addLayout(bottom_bar)

        scroll.setWidget(content)
        main_dlg_layout = QVBoxLayout(self)
        main_dlg_layout.setContentsMargins(0, 0, 0, 0)
        main_dlg_layout.addWidget(scroll)

        # Populate initial UI
        self._refresh_ui()

    @staticmethod
    def _get_status_severity(status: str) -> str:
        if status == STATUS_CONNECTED:
            return "active"
        elif status in (STATUS_INTEGRATION_PENDING, STATUS_FILE_VALIDATED):
            return "warning"
        elif status == STATUS_CONNECTION_ERROR:
            return "error"
        return "neutral"

    def _refresh_ui(self):
        curr_dir = self._manager.backend_directory or "No directory selected"
        self.path_label.setText(curr_dir)
        self.status_pill.set_status(self._manager.status, self._get_status_severity(self._manager.status))

        # Populate files table
        detected = self._manager.detected_files
        self.files_table.setRowCount(len(detected))
        for row, item in enumerate(detected):
            self.files_table.setItem(row, 0, QTableWidgetItem(item["name"]))
            self.files_table.setItem(row, 1, QTableWidgetItem(item["extension"]))
            self.files_table.setItem(row, 2, QTableWidgetItem(item["size_str"]))
            self.files_table.setItem(row, 3, QTableWidgetItem(item["role_hint"]))

        # Populate module mapping dropdowns
        py_files = [""] + [f["name"] for f in detected if f["extension"] == ".py"]
        csv_files = [""] + [f["name"] for f in detected if f["extension"] == ".csv"]
        dirs = [""] + [f["name"] for f in detected if f["is_directory"]]

        self._populate_combo(self.combo_static_train, py_files, self._manager.module_mappings.get("static_training", ""))
        self._populate_combo(self.combo_static_collect, py_files, self._manager.module_mappings.get("static_collection", ""))
        self._populate_combo(self.combo_dynamic_train, py_files, self._manager.module_mappings.get("dynamic_training", ""))
        self._populate_combo(self.combo_predict, py_files, self._manager.module_mappings.get("live_prediction", ""))
        self._populate_combo(self.combo_static_ds, csv_files, self._manager.module_mappings.get("static_dataset", ""))
        self._populate_combo(self.combo_dynamic_ds, csv_files, self._manager.module_mappings.get("dynamic_dataset", ""))
        self._populate_combo(self.combo_models_dir, dirs, self._manager.module_mappings.get("models_dir", ""))

    def _populate_combo(self, combo: QComboBox, items: list, selected_val: str):
        combo.clear()
        sel_idx = 0
        for idx, item in enumerate(items):
            combo.addItem(item or "— Not Mapped —", userData=item)
            if selected_val and (selected_val.endswith(item) or selected_val == item):
                sel_idx = idx
        combo.setCurrentIndex(sel_idx)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Existing Python ML Backend Directory")
        if folder:
            success, msg = self._manager.register_backend_directory(folder)
            self._refresh_ui()
            if success:
                QMessageBox.information(self, "Backend Directory Registered", msg)
            else:
                QMessageBox.warning(self, "Backend Validation Warning", msg)

    def _validate_backend(self):
        success, msg, results = self._manager.validate_and_connect()
        self._refresh_ui()
        if success:
            QMessageBox.information(self, "Backend Structure Validation", msg)
        else:
            QMessageBox.warning(self, "Backend Structure Validation", msg)

    def _remove_backend(self):
        confirm = QMessageBox.question(
            self,
            "Disconnect Backend",
            "Are you sure you want to remove the backend directory reference?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._manager.remove_backend()
            self._refresh_ui()

    def _save_mappings_and_close(self):
        def _get_val(combo):
            data = combo.currentData()
            return str(data) if data else ""

        self._manager.set_mapping("static_training", _get_val(self.combo_static_train))
        self._manager.set_mapping("static_collection", _get_val(self.combo_static_collect))
        self._manager.set_mapping("dynamic_training", _get_val(self.combo_dynamic_train))
        self._manager.set_mapping("live_prediction", _get_val(self.combo_predict))
        self._manager.set_mapping("static_dataset", _get_val(self.combo_static_ds))
        self._manager.set_mapping("dynamic_dataset", _get_val(self.combo_dynamic_ds))
        self._manager.set_mapping("models_dir", _get_val(self.combo_models_dir))
        self._manager.validate_and_connect()
        self.accept()
