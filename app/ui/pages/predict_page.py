"""
Live Translation Workspace & Open-Source AI Model Harness for SignFlow Studio.
Directly embeds the native 60 FPS OpenCV + MediaPipe + Scikit-Learn prediction pipeline
from predict_live.py inside the application window.

Features:
- Pure Model Harness: prompts user to add models if none exist
- Embedded Camera Canvas: displays live camera output with green skeleton bones (0, 255, 0),
  red landmark dots (0, 0, 255), dynamic J/Z gesture trails, and sleek glassmorphic HUD
- Threaded background worker (NativeInferenceWorker): zero GUI lag or dropped frames
- Interactive Sentence Accumulator with Space, Backspace, Clear, and Copy controls
- Direct integration with OBS Virtual Camera (Discord / Zoom / Google Meet)
"""
import os
import sys
from typing import Optional, List
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QComboBox, QMessageBox, QSizePolicy, QApplication, QStackedLayout
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QClipboard

from app.core.config import MODELS_DIR
from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager
from app.core.model_registry import ModelRegistry, ModelPackage
from app.core.native_inference_engine import NativeInferenceWorker
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.add_model_dialog import AddModelDialog
from app.ui.components.camera_settings_dialog import CameraSettingsDialog
from app.ui.components.camera_preview_widget import CameraPreviewWidget
from app.ui.components.embedded_camera_canvas import EmbeddedCameraCanvas

__all__ = ["PredictPage", "CameraPreviewWidget", "CameraSettingsDialog"]


def get_available_cameras() -> List[str]:
    """Discover available physical and virtual webcams on Windows."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        devices = FilterGraph().get_input_devices()
        if devices:
            return devices
    except Exception:
        pass
    return ["Camera Index 0", "Camera Index 1"]


class PredictPage(QWidget):
    """
    Main Live Translation Screen and Open-Source Model Harness.
    Hosts Model Selector, Embedded Real-Time Camera Canvas, and Interactive Sentence Builder.
    """
    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._model_registry = ModelRegistry.get_instance()

        self._worker: Optional[NativeInferenceWorker] = None
        self._is_translating = False
        self._current_sentence = ""

        self._setup_ui()
        self._connect_signals()
        self._populate_cameras()
        self._populate_models()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 14, 20, 18)
        root_layout.setSpacing(12)

        # =====================================================================
        # 1. TOP MENU BAR: Model Harness Selector, Camera Picker, Translation Action
        # =====================================================================
        top_bar = QFrame()
        top_bar.setProperty("class", "cardPanelElevated")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(12)

        # Model Selection Group
        model_vbox = QVBoxLayout()
        model_vbox.setSpacing(2)
        lbl_model = QLabel("AI MODEL HARNESS")
        lbl_model.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")

        model_hrow = QHBoxLayout()
        model_hrow.setSpacing(8)

        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(240)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)

        self.add_model_btn = QPushButton(" + Add Model")
        self.add_model_btn.setProperty("class", "secondaryBtn")
        self.add_model_btn.setIcon(get_icon("box", palette.text_secondary, palette.accent, size=14))
        self.add_model_btn.setToolTip("Import or download trained sign language models")
        self.add_model_btn.clicked.connect(lambda: self._open_add_model_dialog(initial_tab=0))

        self.delete_model_btn = QPushButton(" 🗑 Delete")
        self.delete_model_btn.setProperty("class", "dangerBtn")
        self.delete_model_btn.setToolTip("Delete or unregister the currently selected model")
        self.delete_model_btn.clicked.connect(self._delete_selected_model)

        model_hrow.addWidget(self.model_combo)
        model_hrow.addWidget(self.add_model_btn)
        model_hrow.addWidget(self.delete_model_btn)
        model_vbox.addWidget(lbl_model)
        model_vbox.addLayout(model_hrow)
        top_layout.addLayout(model_vbox)

        # Camera Device Selector Group
        cam_vbox = QVBoxLayout()
        cam_vbox.setSpacing(2)
        lbl_cam = QLabel("CAMERA DEVICE")
        lbl_cam.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")

        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(200)
        cam_vbox.addWidget(lbl_cam)
        cam_vbox.addWidget(self.camera_combo)
        top_layout.addLayout(cam_vbox)

        top_layout.addStretch()

        # Primary Action Button: Start / Stop Translation
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"
        self.action_btn = QPushButton(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setMinimumHeight(38)
        self.action_btn.setIcon(get_icon("play", btn_text_color, size=15))
        self.action_btn.clicked.connect(self._toggle_translation)
        top_layout.addWidget(self.action_btn)

        root_layout.addWidget(top_bar)

        # =====================================================================
        # 2. CENTRAL WORKSPACE: Stack containing Empty State OR Embedded Canvas
        # =====================================================================
        self.central_stack = QFrame()
        self.central_stack.setProperty("class", "cardPanel")
        self.central_stack_layout = QStackedLayout(self.central_stack)

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
            "(e.g. models/SignFlow-model-v1-USER or your custom training folder)."
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

        # B. Live Translation Container (Shown when a model is selected)
        self.live_container = QWidget()
        lc_layout = QVBoxLayout(self.live_container)
        lc_layout.setContentsMargins(0, 0, 0, 0)
        lc_layout.setSpacing(10)

        # Embedded Camera Video Canvas
        self.camera_canvas = EmbeddedCameraCanvas(self)
        self.camera_canvas.key_space_pressed.connect(self._on_space_pressed)
        self.camera_canvas.key_backspace_pressed.connect(self._on_backspace_pressed)
        self.camera_canvas.key_clear_pressed.connect(self._on_clear_pressed)
        lc_layout.addWidget(self.camera_canvas, 1)

        # Interactive Sentence Bar & Shortcuts Card
        self.sentence_card = QFrame()
        self.sentence_card.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_sidebar};
                border: 1px solid {palette.border_default};
                border-radius: 8px;
                padding: 10px 14px;
            }}
        """)
        sc_layout = QHBoxLayout(self.sentence_card)
        sc_layout.setContentsMargins(6, 4, 6, 4)
        sc_layout.setSpacing(12)

        lbl_sent_tag = QLabel("ACCUMULATED SENTENCE:")
        lbl_sent_tag.setStyleSheet(f"font-size: 10px; font-weight: 800; color: {palette.text_muted}; letter-spacing: 0.6px;")

        self.sentence_display = QLabel("—")
        self.sentence_display.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {palette.accent_label};")
        self.sentence_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        # Interactive action buttons
        self.btn_add_letter = QPushButton(" ␣ Space / Commit")
        self.btn_add_letter.setProperty("class", "secondaryBtn")
        self.btn_add_letter.clicked.connect(self._on_space_pressed)

        self.btn_backspace = QPushButton(" ⌫ Delete")
        self.btn_backspace.setProperty("class", "secondaryBtn")
        self.btn_backspace.clicked.connect(self._on_backspace_pressed)

        self.btn_clear = QPushButton(" ✕ Clear")
        self.btn_clear.setProperty("class", "secondaryBtn")
        self.btn_clear.clicked.connect(self._on_clear_pressed)

        self.btn_copy = QPushButton(" 📋 Copy")
        self.btn_copy.setProperty("class", "secondaryBtn")
        self.btn_copy.clicked.connect(self._copy_sentence_to_clipboard)

        sc_vbox = QVBoxLayout()
        sc_vbox.setSpacing(2)
        sc_vbox.addWidget(lbl_sent_tag)
        sc_vbox.addWidget(self.sentence_display)

        sc_layout.addLayout(sc_vbox, 1)
        sc_layout.addWidget(self.btn_add_letter)
        sc_layout.addWidget(self.btn_backspace)
        sc_layout.addWidget(self.btn_clear)
        sc_layout.addWidget(self.btn_copy)

        lc_layout.addWidget(self.sentence_card)

        self.central_stack_layout.addWidget(self.empty_state_box) # Index 0
        self.central_stack_layout.addWidget(self.live_container)   # Index 1

        root_layout.addWidget(self.central_stack, 1)

    def _connect_signals(self):
        self._model_registry.package_added.connect(self._on_package_added)
        self._model_registry.active_package_changed.connect(self._on_active_package_changed)

    def _populate_cameras(self):
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        cameras = get_available_cameras()
        for idx, name in enumerate(cameras):
            self.camera_combo.addItem(f"📷 [{idx}] {name}", userData=idx)
        self.camera_combo.blockSignals(False)

    def _populate_models(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        packages = self._model_registry.get_all_packages()
        active = self._model_registry.get_active_package()

        if not packages:
            self.model_combo.addItem("No models detected — Click + Add Model", userData="")
            self.action_btn.setEnabled(False)
            self.delete_model_btn.setEnabled(False)
            self.central_stack_layout.setCurrentWidget(self.empty_state_box)
            self.model_combo.blockSignals(False)
            return

        active_idx = 0
        for i, pkg in enumerate(packages):
            tag = " [Dev Suite]" if pkg.is_developer else " [User Model]"
            self.model_combo.addItem(f"📦 {pkg.name}{tag}", userData=pkg.id)
            if active and pkg.id == active.id:
                active_idx = i

        # Provide a quick jump to the Installed Models manager
        self.model_combo.addItem("⚙ Manage / Delete Models...", userData="__manage__")

        self.model_combo.setCurrentIndex(active_idx)
        self.action_btn.setEnabled(True)
        self.delete_model_btn.setEnabled(True)
        self.central_stack_layout.setCurrentWidget(self.live_container)
        self.model_combo.blockSignals(False)

    def get_selected_package(self) -> Optional[ModelPackage]:
        pkg_id = self.model_combo.currentData()
        if not pkg_id or pkg_id == "__manage__":
            return None
        return self._model_registry.get_package(pkg_id)

    def _on_model_selected(self, index: int):
        pkg_id = self.model_combo.currentData()
        if not pkg_id:
            return

        if pkg_id == "__manage__":
            # Revert dropdown to current active package
            active = self._model_registry.get_active_package()
            if active:
                for i in range(self.model_combo.count()):
                    if self.model_combo.itemData(i) == active.id:
                        self.model_combo.blockSignals(True)
                        self.model_combo.setCurrentIndex(i)
                        self.model_combo.blockSignals(False)
                        break
            self._open_add_model_dialog(initial_tab=2)
            return

        pkg = self._model_registry.get_package(pkg_id)
        if pkg:
            self._model_registry.set_active_package(pkg_id)
            # If translating, restart engine with the new model
            if self._is_translating:
                self._stop_translation()
                self._start_translation()

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

        self.central_stack_layout.setCurrentWidget(self.live_container)

    def _open_add_model_dialog(self, initial_tab: int = 0):
        dlg = AddModelDialog(self, initial_tab=initial_tab)
        dlg.model_imported.connect(self._on_model_imported)
        dlg.exec()
        self._populate_models()

    def _delete_selected_model(self):
        pkg = self.get_selected_package()
        if not pkg:
            QMessageBox.information(self, "No Model Selected", "Please select a valid model from the dropdown to delete.")
            return

        if self._is_translating:
            self._stop_translation()

        folder = Path(pkg.folder_path)
        is_under_models = (MODELS_DIR.resolve() in folder.resolve().parents) or (folder.resolve() == MODELS_DIR.resolve()) or (folder.parent.resolve() == MODELS_DIR.resolve())

        prompt = (
            f"Are you sure you want to delete model '{pkg.name}'?\n\n"
            f"Folder: {pkg.folder_path}\n\n"
            + ("This will permanently delete the model directory and unregister it."
               if is_under_models else "This will unregister the model from SignFlow.")
        )
        reply = QMessageBox.question(
            self,
            "Delete Model Package",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )
        if reply == QMessageBox.StandardButton.Yes:
            success, msg = self._model_registry.delete_package(pkg.id, delete_disk_files=is_under_models)
            self._populate_models()
            QMessageBox.information(self, "Model Deleted", msg)

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

        if not os.path.exists(active_pkg.classifier_path):
            QMessageBox.critical(self, "Classifier Missing", f"Could not find classifier at:\n{active_pkg.classifier_path}")
            return

        if not os.path.exists(active_pkg.task_path):
            QMessageBox.critical(self, "MediaPipe Task Missing", f"Could not find hand_landmarker.task at:\n{active_pkg.task_path}")
            return

        # Ensure internal camera manager preview is stopped to prevent exclusive camera lock
        if self._camera_manager and self._camera_manager.is_previewing:
            self._camera_manager.stop_preview()

        cam_idx = self.camera_combo.currentData()
        if cam_idx is None:
            cam_idx = 0

        # Instantiate background native inference worker
        self._worker = NativeInferenceWorker(
            classifier_path=active_pkg.classifier_path,
            task_path=active_pkg.task_path,
            camera_index=cam_idx,
            parent=self
        )

        self._worker.frame_ready.connect(self.camera_canvas.update_frame)
        self._worker.sentence_changed.connect(self._on_sentence_updated)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.engine_stopped.connect(self._on_worker_stopped)

        self._worker.start()
        self._is_translating = True

        palette = ThemeManager.get_palette()
        self.action_btn.setText(" Stop Translation")
        self.action_btn.setProperty("class", "dangerBtn")
        self.action_btn.setIcon(get_icon("stop", "#FFFFFF", size=15))
        self._apply_button_style()

        # Canvas grabs focus so SPACE and BACKSPACE work immediately
        self.camera_canvas.setFocus()

    def _stop_translation(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker = None

        self._is_translating = False
        self.camera_canvas.clear()

        palette = ThemeManager.get_palette()
        btn_text_color = "#1C1917" if not palette.is_dark else "#161513"

        self.action_btn.setText(" Start Translation")
        self.action_btn.setProperty("class", "primaryBtn")
        self.action_btn.setIcon(get_icon("play", btn_text_color, size=15))
        self._apply_button_style()

    @Slot(str)
    def _on_sentence_updated(self, text: str):
        self._current_sentence = text
        self.sentence_display.setText(text if text else "—")

    @Slot(str)
    def _on_worker_error(self, err_msg: str):
        QMessageBox.critical(self, "Inference Engine Error", err_msg)
        self._stop_translation()

    @Slot()
    def _on_worker_stopped(self):
        if self._is_translating:
            self._stop_translation()

    def _on_space_pressed(self):
        if self._worker is not None:
            self._worker.commit_current_letter()
            # If current stable prediction was committed, also add a space if user pressed space again
            # worker handles appending stable letter or space

    def _on_backspace_pressed(self):
        if self._worker is not None:
            self._worker.backspace()

    def _on_clear_pressed(self):
        if self._worker is not None:
            self._worker.clear_sentence()
        else:
            self.sentence_display.setText("—")

    def _copy_sentence_to_clipboard(self):
        clipboard = QApplication.clipboard()
        text = self.sentence_display.text()
        if text and text != "—":
            clipboard.setText(text)
            orig_text = self.btn_copy.text()
            self.btn_copy.setText(" ✓ Copied!")
            QTimer.singleShot(1500, lambda: self.btn_copy.setText(orig_text))

    def _apply_button_style(self):
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)
        self.action_btn.update()

    def on_page_deactivated(self):
        """Called by MainWindow when navigating away from this page."""
        if self._is_translating:
            self._stop_translation()
