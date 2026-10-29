"""
Smart Model Importer Dialog for SignFlow Studio.
Allows users to select an external model folder, inspects and verifies:
- User Model: predict_live.py + models/asl_classifier.joblib + models/hand_landmarker.task
- Developer Model: adds train_model.py, collect_static/dynamic, datasets, and old models.
Prompts for custom name and registers it into the ModelRegistry harness.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QFileDialog, QFrame, QMessageBox
)
from PySide6.QtCore import Qt, Signal

from app.core.theme import ThemeManager
from app.core.model_registry import ModelRegistry, detect_model_in_folder
from app.resources.icons import get_icon


class AddModelDialog(QDialog):
    """Editorial dialog for discovering, validating, and adding custom AI models."""
    model_imported = Signal(object)  # Emits ModelPackage

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add AI Model Package")
        self.setMinimumWidth(580)
        self.setModal(True)

        self._detected_data = None
        self._setup_ui()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header Title & Editorial Note
        header_vbox = QVBoxLayout()
        header_vbox.setSpacing(4)
        title = QLabel("Import Sign Language Model Package")
        title.setStyleSheet(f"font-size: 18px; font-weight: 700; color: {palette.text_primary};")
        desc = QLabel(
            "Point to any model folder. SignFlow inspects the folder to detect if it is a "
            "User Model (inference only) or a Developer Model Suite (with training scripts and datasets)."
        )
        desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.4;")
        desc.setWordWrap(True)
        header_vbox.addWidget(title)
        header_vbox.addWidget(desc)
        layout.addLayout(header_vbox)

        # Folder Picker Card
        folder_card = QFrame()
        folder_card.setProperty("class", "cardPanelElevated")
        fc_layout = QVBoxLayout(folder_card)
        fc_layout.setContentsMargins(16, 14, 16, 14)
        fc_layout.setSpacing(10)

        f_label = QLabel("MODEL DIRECTORY PATH")
        f_label.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        fc_layout.addWidget(f_label)

        f_input_row = QHBoxLayout()
        f_input_row.setSpacing(10)

        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder (e.g. C:/Users/.../SignLanguageAI)")
        self.folder_input.textChanged.connect(self._on_path_changed)
        f_input_row.addWidget(self.folder_input, 1)

        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.setProperty("class", "secondaryBtn")
        self.browse_btn.setIcon(get_icon("folder", palette.text_secondary, palette.accent, size=15))
        self.browse_btn.clicked.connect(self._browse_folder)
        f_input_row.addWidget(self.browse_btn)

        fc_layout.addLayout(f_input_row)
        layout.addWidget(folder_card)

        # Verification Status Panel
        self.status_panel = QFrame()
        self.status_panel.setProperty("class", "cardPanel")
        self.status_panel.setVisible(False)
        sp_layout = QVBoxLayout(self.status_panel)
        sp_layout.setContentsMargins(16, 14, 16, 14)
        sp_layout.setSpacing(6)

        self.status_title = QLabel("Model Verification")
        self.status_title.setStyleSheet("font-size: 12.5px; font-weight: 700;")
        sp_layout.addWidget(self.status_title)

        self.lbl_type_badge = QLabel("—")
        self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 600; color: {palette.accent_label};")
        sp_layout.addWidget(self.lbl_type_badge)

        self.lbl_predict = QLabel("—")
        self.lbl_classifier = QLabel("—")
        self.lbl_task = QLabel("—")
        self.lbl_dev_suite = QLabel("—")

        for lbl in (self.lbl_predict, self.lbl_classifier, self.lbl_task, self.lbl_dev_suite):
            lbl.setStyleSheet(f"font-size: 11.5px; color: {palette.text_secondary};")
            sp_layout.addWidget(lbl)

        layout.addWidget(self.status_panel)

        # Model Custom Name Card
        name_card = QFrame()
        name_card.setProperty("class", "cardPanel")
        nc_layout = QVBoxLayout(name_card)
        nc_layout.setContentsMargins(16, 14, 16, 14)
        nc_layout.setSpacing(8)

        name_lbl = QLabel("DISPLAY NAME FOR THIS MODEL")
        name_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        nc_layout.addWidget(name_lbl)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("e.g. My Custom ASL Model")
        nc_layout.addWidget(self.name_input)

        layout.addWidget(name_card)

        # Action Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setProperty("class", "secondaryBtn")
        self.cancel_btn.clicked.connect(self.reject)

        self.import_btn = QPushButton("Import & Register Model")
        self.import_btn.setProperty("class", "primaryBtn")
        self.import_btn.setEnabled(False)
        self.import_btn.clicked.connect(self._on_import_clicked)

        btn_row.addWidget(self.cancel_btn)
        btn_row.addWidget(self.import_btn)
        layout.addLayout(btn_row)

    def _browse_folder(self):
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select AI Model Directory",
            str(Path.home() / "Downloads")
        )
        if chosen:
            self.folder_input.setText(chosen)

    def _on_path_changed(self, path_text: str):
        path = path_text.strip()
        if not path or not Path(path).exists() or not Path(path).is_dir():
            self.status_panel.setVisible(False)
            self.import_btn.setEnabled(False)
            self._detected_data = None
            return

        res = detect_model_in_folder(path)
        self._detected_data = res
        self.status_panel.setVisible(True)

        palette = ThemeManager.get_palette()
        if res["is_valid"]:
            self.status_title.setText("✓ Verified Model Package")
            self.status_title.setStyleSheet(f"font-size: 12.5px; font-weight: 700; color: {palette.success};")

            is_dev = res.get("is_developer", False)
            if is_dev:
                self.lbl_type_badge.setText("PACKAGE TYPE: DEVELOPER MODEL SUITE (TRAINING + INFERENCE)")
                self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 700; color: {palette.accent_label};")
                self.lbl_dev_suite.setText("• Developer Suite: train_model.py, collect scripts, and datasets verified!")
            else:
                self.lbl_type_badge.setText("PACKAGE TYPE: USER MODEL (INFERENCE ONLY)")
                self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; font-weight: 700; color: {palette.text_secondary};")
                self.lbl_dev_suite.setText("• Standalone User Model (no training files detected)")

            p_txt = Path(res['predict_script']).name
            self.lbl_predict.setText(f"• Predict Script: {p_txt} (✓ Found)")
            self.lbl_classifier.setText(f"• Classifier: {Path(res['classifier_path']).name} (✓ Found)")
            self.lbl_task.setText(f"• Landmarker: {Path(res['task_path']).name} (✓ Found)")

            if not self.name_input.text():
                self.name_input.setText(res["suggested_name"])

            self.import_btn.setEnabled(True)
        else:
            self.status_title.setText("✗ Incomplete Model Directory")
            self.status_title.setStyleSheet(f"font-size: 12.5px; font-weight: 700; color: {palette.danger};")
            self.lbl_type_badge.setText(f"Error: {res['error_msg']}")
            self.lbl_type_badge.setStyleSheet(f"font-size: 11.5px; color: {palette.danger};")
            self.lbl_predict.setText("Required structure:")
            self.lbl_classifier.setText("  - predict_live.py (in root)")
            self.lbl_task.setText("  - models/asl_classifier.joblib & models/hand_landmarker.task")
            self.lbl_dev_suite.setText("")
            self.import_btn.setEnabled(False)

    def _on_import_clicked(self):
        if not self._detected_data or not self._detected_data["is_valid"]:
            return

        reg = ModelRegistry.get_instance()
        custom_name = self.name_input.text().strip() or self._detected_data["suggested_name"]
        success, pkg, msg = reg.register_model_package(
            folder_path=self._detected_data["folder_path"],
            custom_name=custom_name
        )

        if success and pkg:
            self.model_imported.emit(pkg)
            self.accept()
        else:
            QMessageBox.warning(self, "Registration Error", msg)
