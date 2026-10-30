"""
Live Translation Workspace & Open-Source AI Model Harness.
Acts as a pure harness:
- If no models are added, displays "No models detected" and prompts user to click "+ Add Model".
- Distinguishes between User Model (inference only) and Developer Model (training suite).
- On "Start Translation", directly launches the native 60 FPS predict_live.py window
  with standard green landmark connections (0, 255, 0), red dots, and native HUD.
"""
import os
import sys
import subprocess
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QComboBox, QMessageBox, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager
from app.core.model_registry import ModelRegistry, ModelPackage
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill
from app.ui.components.add_model_dialog import AddModelDialog
from app.ui.components.camera_settings_dialog import CameraSettingsDialog
from app.ui.components.camera_preview_widget import CameraPreviewWidget

__all__ = ["PredictPage", "CameraPreviewWidget", "CameraSettingsDialog"]


class PredictPage(QWidget):
    """
    Main Live Translation Screen and Model Harness.
    Hosts Model Selector, Add Model action, and Native Inference Runner.
    """
    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._model_registry = ModelRegistry.get_instance()

        self._proc: Optional[subprocess.Popen] = None
        self._is_translating = False

        # Process monitor timer
        self._proc_timer = QTimer(self)
        self._proc_timer.timeout.connect(self._check_proc_status)

        self._setup_ui()
        self._connect_signals()
        self._populate_models()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(24, 18, 24, 24)
        root_layout.setSpacing(18)

        # =====================================================================
        # TOP MENU BAR: Model Selection, Add Model, Start Translation
        # =====================================================================
        top_bar = QFrame()
        top_bar.setProperty("class", "cardPanelElevated")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 12, 16, 12)
        top_layout.setSpacing(14)

        # 1. Model Selector Group
        model_vbox = QVBoxLayout()
        model_vbox.setSpacing(3)
        lbl_model = QLabel("AI MODEL HARNESS")
        lbl_model.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")

        model_hrow = QHBoxLayout()
        model_hrow.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(260)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)

        self.add_model_btn = QPushButton(" + Add Model")
        self.add_model_btn.setProperty("class", "secondaryBtn")
        self.add_model_btn.setIcon(get_icon("box", palette.text_secondary, palette.accent, size=15))
        self.add_model_btn.setToolTip("Import any trained sign language model folder (User or Developer model)")
        self.add_model_btn.clicked.connect(self._open_add_model_dialog)

        model_hrow.addWidget(self.model_combo)
        model_hrow.addWidget(self.add_model_btn)
        model_vbox.addWidget(lbl_model)
        model_vbox.addLayout(model_hrow)
        top_layout.addLayout(model_vbox)

        top_layout.addStretch()

        # 2. Engine Status Pill
        self.status_pill = StatusPill("No Model Loaded", "neutral")
        top_layout.addWidget(self.status_pill)

        # 3. Primary Start / Stop Translation Action Button
        palette = ThemeManager.get_palette()
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"

        self.action_btn = QPushButton(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setMinimumHeight(40)
        self.action_btn.setIcon(get_icon("play", btn_text_color, size=16))
        self.action_btn.clicked.connect(self._toggle_translation)
        top_layout.addWidget(self.action_btn)

        root_layout.addWidget(top_bar)

        # =====================================================================
        # MAIN WORKSPACE: Model Details Card / Empty State View
        # =====================================================================
        self.main_content_frame = QFrame()
        self.main_content_frame.setProperty("class", "cardPanel")
        self.main_content_layout = QVBoxLayout(self.main_content_frame)
        self.main_content_layout.setContentsMargins(28, 28, 28, 28)
        self.main_content_layout.setSpacing(18)

        # A. Empty State Widget (Shown when NO models exist)
        self.empty_state_box = QWidget()
        es_layout = QVBoxLayout(self.empty_state_box)
        es_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        es_layout.setSpacing(14)

        icon_empty = QLabel()
        icon_empty.setPixmap(get_pixmap("box", palette.text_muted, size=56, stroke_width=1.5))
        icon_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_empty = QLabel("No Models Detected")
        title_empty.setStyleSheet(f"font-size: 20px; font-weight: 800; color: {palette.text_primary};")
        title_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        desc_empty = QLabel(
            "SignFlow is an open-source model harness with zero bundled models.\n"
            "Click '+ Add Model' in the top bar to connect your model directory\n"
            "(e.g. C:/Users/.../SignLanguageAI)."
        )
        desc_empty.setStyleSheet(f"font-size: 13px; color: {palette.text_secondary}; line-height: 1.5;")
        desc_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)

        btn_add_empty = QPushButton(" + Add Your First Model")
        btn_add_empty.setProperty("class", "primaryBtn")
        btn_add_empty.setMinimumHeight(38)
        btn_add_empty.setIcon(get_icon("box", btn_text_color, size=16))
        btn_add_empty.clicked.connect(self._open_add_model_dialog)

        es_layout.addWidget(icon_empty)
        es_layout.addWidget(title_empty)
        es_layout.addWidget(desc_empty)
        es_layout.addWidget(btn_add_empty, alignment=Qt.AlignmentFlag.AlignCenter)
        self.main_content_layout.addWidget(self.empty_state_box)

        # B. Loaded Model Details Card (Shown when a model IS loaded)
        self.loaded_card = QWidget()
        lc_layout = QVBoxLayout(self.loaded_card)
        lc_layout.setSpacing(16)

        # Header Row
        lc_header = QHBoxLayout()
        self.lbl_active_name = QLabel("Active Model")
        self.lbl_active_name.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {palette.text_primary};")

        self.model_type_badge = QLabel("USER MODEL")
        self.model_type_badge.setStyleSheet(f"""
            padding: 3px 10px; border-radius: 12px;
            background-color: {palette.accent_subtle};
            color: {palette.accent_label}; font-size: 11px; font-weight: 700;
            border: 1px solid {palette.accent_border};
        """)

        lc_header.addWidget(self.lbl_active_name)
        lc_header.addWidget(self.model_type_badge)
        lc_header.addStretch()
        lc_layout.addLayout(lc_header)

        # Components Checklist Card
        components_box = QFrame()
        components_box.setProperty("class", "cardPanelElevated")
        cb_layout = QVBoxLayout(components_box)
        cb_layout.setContentsMargins(18, 16, 18, 16)
        cb_layout.setSpacing(8)

        lbl_comp_title = QLabel("VERIFIED HARNESS COMPONENTS")
        lbl_comp_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")
        cb_layout.addWidget(lbl_comp_title)

        self.lbl_path_script = QLabel("—")
        self.lbl_path_clf = QLabel("—")
        self.lbl_path_task = QLabel("—")
        self.lbl_path_folder = QLabel("—")

        for lbl in (self.lbl_path_script, self.lbl_path_clf, self.lbl_path_task, self.lbl_path_folder):
            lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; font-family: monospace;")
            cb_layout.addWidget(lbl)

        lc_layout.addWidget(components_box)

        # Live Translation Guide & Keyboard Controls
        guide_box = QFrame()
        guide_box.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_sidebar};
                border: 1px solid {palette.border_default};
                border-radius: 8px;
                padding: 16px;
            }}
        """)
        gb_layout = QVBoxLayout(guide_box)
        gb_layout.setSpacing(10)

        gb_title = QLabel("Native High-Performance Recognition Engine")
        gb_title.setStyleSheet(f"font-size: 13.5px; font-weight: 700; color: {palette.text_primary};")
        gb_desc = QLabel(
            "Clicking 'Start Translation' runs predict_live.py natively with full 60 FPS performance, "
            "standard green landmark connections, red keypoint dots, dynamic gesture tracking (J & Z), "
            "and live sentence formation. No extra overhead or performance lag."
        )
        gb_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.45;")
        gb_desc.setWordWrap(True)

        gb_shortcuts = QLabel(
            "Native Window Controls:  [SPACE] Add Letter to Sentence  |  [BACKSPACE] Delete  |  "
            "[C] Clear  |  [R] Toggle 720p/1080p  |  [TAB] Switch Camera  |  [Q] Exit Window"
        )
        gb_shortcuts.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.accent_label};")

        gb_layout.addWidget(gb_title)
        gb_layout.addWidget(gb_desc)
        gb_layout.addWidget(gb_shortcuts)
        lc_layout.addWidget(guide_box)

        self.main_content_layout.addWidget(self.loaded_card)
        root_layout.addWidget(self.main_content_frame, 1)

    def _connect_signals(self):
        self._model_registry.package_added.connect(self._on_package_added)
        self._model_registry.active_package_changed.connect(self._on_active_package_changed)

    def _populate_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        packages = self._model_registry.get_all_packages()
        active = self._model_registry.get_active_package()

        if not packages:
            self.model_combo.addItem("No models detected — Click + Add Model", userData="")
            self.action_btn.setEnabled(False)
            self.status_pill.set_status("No Models Detected", "neutral")
            self.empty_state_box.setVisible(True)
            self.loaded_card.setVisible(False)
            self.model_combo.blockSignals(False)
            return

        active_idx = 0
        for i, pkg in enumerate(packages):
            tag = " [Dev Suite]" if pkg.is_developer else " [User Model]"
            self.model_combo.addItem(f"📦 {pkg.name}{tag}", userData=pkg.id)
            if active and pkg.id == active.id:
                active_idx = i

        self.model_combo.setCurrentIndex(active_idx)
        self.action_btn.setEnabled(True)
        self.empty_state_box.setVisible(False)
        self.loaded_card.setVisible(True)
        self._update_loaded_model_view(active or packages[0])
        self.model_combo.blockSignals(False)

    def _update_loaded_model_view(self, pkg: ModelPackage):
        self.lbl_active_name.setText(pkg.name)
        if pkg.is_developer:
            self.model_type_badge.setText("DEVELOPER MODEL SUITE (TRAINING + INFERENCE)")
        else:
            self.model_type_badge.setText("USER MODEL (INFERENCE ONLY)")

        self.lbl_path_script.setText(f"• Predict Script: {Path(pkg.predict_script).name} (Verified)")
        self.lbl_path_clf.setText(f"• Classifier:     {Path(pkg.classifier_path).name} (Verified)")
        self.lbl_path_task.setText(f"• Landmarker:     {Path(pkg.task_path).name} (Verified)")
        self.lbl_path_folder.setText(f"• Package Folder: {pkg.folder_path}")

        if not self._is_translating:
            self.status_pill.set_status("Native Engine Ready", "active")

    def get_selected_package(self) -> Optional[ModelPackage]:
        pkg_id = self.model_combo.currentData()
        if not pkg_id:
            return None
        return self._model_registry.get_package(pkg_id)

    def _on_model_selected(self, index: int):
        pkg_id = self.model_combo.currentData()
        if not pkg_id:
            return
        pkg = self._model_registry.get_package(pkg_id)
        if pkg:
            self._model_registry.set_active_package(pkg_id)
            self._update_loaded_model_view(pkg)

    def _on_package_added(self, package: ModelPackage):
        self._populate_models()

    def _on_active_package_changed(self, package: Optional[ModelPackage]):
        if package is None:
            self._populate_models()
            return

        for i in range(self.model_combo.count()):
            if self.model_combo.itemData(i) == package.id:
                self.model_combo.blockSignals(True)
                self.model_combo.setCurrentIndex(i)
                self.model_combo.blockSignals(False)
                break
        self._update_loaded_model_view(package)

    def _open_add_model_dialog(self):
        dlg = AddModelDialog(self)
        dlg.model_imported.connect(self._on_model_imported)
        dlg.exec()

    def _on_model_imported(self, package: ModelPackage):
        self._populate_models()
        suite_desc = "Developer Model Suite (training tools unlocked)" if package.is_developer else "User Model"
        QMessageBox.information(
            self,
            "Model Harness Activated",
            f"Successfully imported and activated:\n\n"
            f"Name: {package.name}\n"
            f"Type: {suite_desc}\n"
            f"Folder: {package.folder_path}"
        )

    def _toggle_translation(self):
        if not self._is_translating:
            self._start_translation()
        else:
            self._stop_translation()

    def _start_translation(self):
        active_pkg = self._model_registry.get_active_package()
        if not active_pkg:
            QMessageBox.warning(self, "No Model Selected", "Please import a model folder first using '+ Add Model'.")
            return

        predict_path = active_pkg.predict_script
        if not os.path.exists(predict_path):
            QMessageBox.critical(self, "Script Missing", f"Could not find predict_live.py at:\n{predict_path}")
            return

        try:
            # Stop any camera preview inside Qt to guarantee exclusive, zero-lag camera access
            if self._camera_manager and self._camera_manager.is_previewing:
                self._camera_manager.stop_preview()

            # Launch native predict_live.py process directly in its own directory
            # Full 60 FPS, green landmark skeleton (0, 255, 0), red dots, native HUD, zero lag
            self._proc = subprocess.Popen(
                [sys.executable, predict_path],
                cwd=active_pkg.folder_path
            )
            self._is_translating = True
            self.action_btn.setText(" Stop Translation")
            self.action_btn.setProperty("class", "dangerBtn")
            self.action_btn.setIcon(get_icon("stop", "#FFFFFF", size=15))
            self.status_pill.set_status(f"Live Window Running (PID {self._proc.pid})", "active")
            self._apply_button_style()

            # Poll for process termination
            self._proc_timer.start(500)
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"Failed to launch native predict_live.py:\n{str(e)}")

    def _stop_translation(self):
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None

        self._proc_timer.stop()
        self._is_translating = False

        palette = ThemeManager.get_palette()
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"

        self.action_btn.setText(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setIcon(get_icon("play", btn_text_color, size=16))
        self.status_pill.set_status("Native Engine Ready", "active" if self._model_registry.get_all_packages() else "neutral")
        self._apply_button_style()

    def _check_proc_status(self):
        if self._proc is not None:
            ret = self._proc.poll()
            if ret is not None:
                # Native window was closed (e.g. user pressed 'Q')
                self._stop_translation()

    def _apply_button_style(self):
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)

    def on_page_activated(self):
        self._populate_models()

    def on_page_deactivated(self):
        pass
