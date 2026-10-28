"""
Model Manager Page.
Dedicated interface for cataloging and managing trained ML model artifacts
(.joblib, .task, .onnx, .h5, .pt) without deserializing them prematurely.
"""
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QScrollArea
)
from PySide6.QtCore import Qt, Slot

from app.core.theme import ThemeManager
from app.core.model_registry import ModelRegistry, ModelEntry, MODEL_EXTENSIONS
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import SectionHeader, StatusPill, StatCard
from app.ui.components.error_dialog import ErrorDialog


class ModelManagerPage(QWidget):
    """Catalog of trained sign language recognition model artifacts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._registry = ModelRegistry.get_instance()
        self._selected_model_id: Optional[str] = None

        palette = ThemeManager.get_palette()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(24, 20, 24, 20)
        main_layout.setSpacing(16)

        # Header
        header = SectionHeader(
            "Model Artifact Manager",
            "Catalog, inspect, and replace trained machine learning model files and vision task bundles."
        )
        main_layout.addWidget(header)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.import_btn = QPushButton(" Import Model Artifact...")
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

        self.count_pill = StatusPill("0 Models Registered", "neutral")
        toolbar.addWidget(self.count_pill)
        main_layout.addLayout(toolbar)

        # Splitter (Left: Model List, Right: Details)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Pane: Model Artifacts List
        left_frame = QFrame()
        left_frame.setProperty("class", "cardPanel")
        left_layout = QVBoxLayout(left_frame)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(8)

        list_title = QLabel("REGISTERED MODEL ARTIFACTS")
        list_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        left_layout.addWidget(list_title)

        self.model_list = QListWidget()
        self.model_list.setStyleSheet(f"""
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
        self.model_list.currentItemChanged.connect(self._on_item_selected)
        left_layout.addWidget(self.model_list)
        splitter.addWidget(left_frame)

        # Right Pane: Detailed Inspector
        right_frame = QFrame()
        right_frame.setProperty("class", "cardPanel")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(18, 16, 18, 16)
        right_layout.setSpacing(16)

        self.details_title = QLabel("Select a model to view metadata")
        self.details_title.setStyleSheet("font-size: 16px; font-weight: 700;")
        right_layout.addWidget(self.details_title)

        # Metadata Grid
        self.meta_grid = QGridLayout()
        self.meta_grid.setSpacing(10)

        self.lbl_type = QLabel("—")
        self.lbl_path = QLabel("—")
        self.lbl_size = QLabel("—")
        self.lbl_status = QLabel("—")

        self.meta_grid.addWidget(QLabel("Model Architecture:"), 0, 0)
        self.meta_grid.addWidget(self.lbl_type, 0, 1)

        self.meta_grid.addWidget(QLabel("Artifact Location:"), 1, 0)
        self.meta_grid.addWidget(self.lbl_path, 1, 1)

        self.meta_grid.addWidget(QLabel("File Size:"), 2, 0)
        self.meta_grid.addWidget(self.lbl_size, 2, 1)

        self.meta_grid.addWidget(QLabel("Integration Status:"), 3, 0)
        self.meta_grid.addWidget(self.lbl_status, 3, 1)

        right_layout.addLayout(self.meta_grid)

        # Security & Boundary Banner
        boundary_card = QFrame()
        boundary_card.setProperty("class", "cardPanelElevated")
        bc_layout = QVBoxLayout(boundary_card)
        bc_layout.setContentsMargins(14, 12, 14, 12)
        bc_layout.setSpacing(4)

        bc_title = QLabel("SAFE ARTIFACT REGISTRATION POLICY")
        bc_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.5px;")

        bc_desc = QLabel(
            "To prevent arbitrary code execution vulnerabilities, model binaries (.joblib, .pkl) "
            "are cataloged purely as verified filesystem artifacts and are NEVER deserialized by the frontend. "
            "The backend engineer handles secure model instantiation inside the prediction pipeline."
        )
        bc_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        bc_desc.setWordWrap(True)

        bc_layout.addWidget(bc_title)
        bc_layout.addWidget(bc_desc)
        right_layout.addWidget(boundary_card)

        # Actions
        action_bar = QHBoxLayout()
        action_bar.setSpacing(10)

        self.btn_replace = QPushButton("Replace Model File...")
        self.btn_replace.setProperty("class", "secondaryBtn")
        self.btn_replace.clicked.connect(self._replace_model)

        self.btn_revert = QPushButton("Revert Replacement")
        self.btn_revert.setProperty("class", "secondaryBtn")
        self.btn_revert.setEnabled(False)
        self.btn_revert.clicked.connect(self._revert_model)

        self.btn_verify = QPushButton("Verify Disk Presence")
        self.btn_verify.setProperty("class", "secondaryBtn")
        self.btn_verify.clicked.connect(self._verify_file)

        self.btn_remove = QPushButton("Remove Reference")
        self.btn_remove.setProperty("class", "dangerBtn")
        self.btn_remove.clicked.connect(self._remove_model)

        action_bar.addWidget(self.btn_replace)
        action_bar.addWidget(self.btn_revert)
        action_bar.addWidget(self.btn_verify)
        action_bar.addStretch()
        action_bar.addWidget(self.btn_remove)
        right_layout.addLayout(action_bar)
        right_layout.addStretch()

        splitter.addWidget(right_frame)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, 1)

        # Wire signals
        self._registry.model_added.connect(self._refresh_list)
        self._registry.model_updated.connect(self._refresh_list)
        self._registry.model_removed.connect(self._refresh_list)

        self._refresh_list()

    def _on_import_click(self):
        filter_str = "ML Model Artifacts (*.joblib *.task *.onnx *.h5 *.keras *.pt *.pth *.tflite);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model File", "", filter_str)
        if file_path:
            success, entry, msg = self._registry.register_model(file_path)
            if success:
                self._refresh_list(select_id=entry.id)
                QMessageBox.information(self, "Model Registered", msg)
            else:
                ext = Path(file_path).suffix.lower()
                ErrorDialog.show_unsupported_model(file_path, ext, parent=self)

    def _refresh_list(self, select_id: Optional[str] = None):
        self.model_list.clear()
        models = self._registry.get_all_models()
        self.count_pill.set_status(f"{len(models)} Models Registered", "active" if models else "neutral")

        target_row = 0
        for row, m in enumerate(models):
            item_text = f"{m.name}\n{m.model_type}  •  {m.file_size_bytes / 1024:.1f} KB"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, m.id)
            self.model_list.addItem(item)

            if select_id and m.id == select_id:
                target_row = row
            elif self._selected_model_id and m.id == self._selected_model_id:
                target_row = row

        if models:
            self.model_list.setCurrentRow(target_row)
        else:
            self._clear_details()

    def _on_item_selected(self, current: QListWidgetItem, previous=None):
        if not current:
            self._clear_details()
            return
        m_id = current.data(Qt.ItemDataRole.UserRole)
        self._selected_model_id = m_id
        entry = self._registry.get_model(m_id)
        if entry:
            self._display_details(entry)

    def _display_details(self, entry: ModelEntry):
        palette = ThemeManager.get_palette()
        self.details_title.setText(entry.name)
        self.lbl_type.setText(entry.model_type)
        self.lbl_path.setText(entry.file_path)
        self.lbl_size.setText(f"{entry.file_size_bytes / 1024:.1f} KB ({entry.file_size_bytes:,} bytes)")
        
        self.lbl_status.setText(entry.status)
        self.lbl_status.setStyleSheet(f"color: {palette.warning}; font-weight: 600;")
        self.btn_revert.setEnabled(entry.previous_reference is not None)

    def _clear_details(self):
        self.details_title.setText("No model selected")
        self.lbl_type.setText("—")
        self.lbl_path.setText("—")
        self.lbl_size.setText("—")
        self.lbl_status.setText("—")
        self.btn_revert.setEnabled(False)

    def _replace_model(self):
        if not self._selected_model_id:
            return
        entry = self._registry.get_model(self._selected_model_id)
        if not entry:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Model Replacement",
            f"Are you sure you want to replace '{entry.name}'?\nPrevious reference will be preserved for rollback.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        filter_str = "ML Model Artifacts (*.joblib *.task *.onnx *.h5 *.keras *.pt *.pth *.tflite);;All Files (*.*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Replacement Model File", "", filter_str)
        if file_path:
            success, msg = self._registry.replace_model(self._selected_model_id, file_path)
            if success:
                QMessageBox.information(self, "Model Replaced", msg)
                self._refresh_list(self._selected_model_id)
            else:
                QMessageBox.warning(self, "Replacement Failed", msg)

    def _revert_model(self):
        if not self._selected_model_id:
            return
        success, msg = self._registry.revert_replacement(self._selected_model_id)
        if success:
            QMessageBox.information(self, "Replacement Reverted", msg)
            self._refresh_list(self._selected_model_id)
        else:
            QMessageBox.warning(self, "Cannot Revert", msg)

    def _verify_file(self):
        if not self._selected_model_id:
            return
        entry = self._registry.get_model(self._selected_model_id)
        if not entry:
            return
        exists = Path(entry.file_path).exists()
        if exists:
            QMessageBox.information(
                self, "File Verification",
                f"Model file verified on disk:\n\n{entry.file_path}\nSize: {entry.file_size_bytes / 1024:.1f} KB"
            )
        else:
            QMessageBox.warning(
                self, "File Missing",
                f"Warning: Model artifact was not found at its registered path:\n{entry.file_path}"
            )

    def _remove_model(self):
        if not self._selected_model_id:
            return
        entry = self._registry.get_model(self._selected_model_id)
        if not entry:
            return
        confirm = QMessageBox.question(
            self,
            "Remove Model Reference",
            f"Remove reference to '{entry.name}' from the manager?\n(Underlying file will not be deleted)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._registry.remove_model(self._selected_model_id)
            self._refresh_list()
