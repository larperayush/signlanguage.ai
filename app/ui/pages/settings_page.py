"""
Settings Page.
Functional configuration page managing appearance (Dark, Light, System),
preferred camera hardware, default recognition mode, storage directory verification,
and system diagnostics.
"""
import os
import sys
import platform
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QRadioButton, QButtonGroup, QComboBox, QCheckBox,
    QScrollArea, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from app.core.config import (
    SettingsManager, MODE_STATIC, MODE_DYNAMIC, MODE_COMBINED,
    ROOT_DIR, MODELS_DIR
)
from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager
from app.core.backend_manager import BackendManager
from app.core.dataset_registry import DatasetRegistry, MANAGED_DATA_DIR
from app.core.model_registry import ModelRegistry
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import SectionHeader, StatusPill
from app.ui.components.backend_import_dialog import BackendImportDialog


class StorageDirRow(QFrame):
    """Row displaying a storage path, status pill, and Open/Configure action."""
    def __init__(self, title: str, path_str: str, on_open_click=None, custom_btn_text: str = "Open Folder", parent=None):
        super().__init__(parent)
        palette = ThemeManager.get_palette()

        self.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_card};
                border: 1px solid {palette.border_subtle};
                border-radius: 6px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        info_vbox = QVBoxLayout()
        info_vbox.setSpacing(3)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-size: 13px; font-weight: 600; color: {palette.text_primary};")

        self.path_lbl = QLabel(path_str)
        self.path_lbl.setStyleSheet(f"font-size: 11px; font-family: monospace; color: {palette.text_secondary};")
        self.path_lbl.setWordWrap(True)

        info_vbox.addWidget(title_lbl)
        info_vbox.addWidget(self.path_lbl)

        layout.addLayout(info_vbox, 1)

        # Verification status pill
        exists = os.path.exists(path_str) if path_str and path_str != "None Registered" else False
        status_text = "Verified on Disk" if exists else ("Unlinked" if path_str == "None Registered" else "Missing on Disk")
        severity = "active" if exists else ("neutral" if path_str == "None Registered" else "error")
        self.pill = StatusPill(status_text, severity)
        layout.addWidget(self.pill)

        if on_open_click:
            btn = QPushButton(f" {custom_btn_text}")
            btn.setProperty("class", "secondaryBtn")
            btn.setIcon(get_icon("folder_open" if "Open" in custom_btn_text else "settings", palette.text_secondary, palette.accent, size=14))
            btn.clicked.connect(on_open_click)
            layout.addWidget(btn)


class SettingsPage(QWidget):
    """
    Compact, fully functional Settings page with persistent QSettings.
    """
    theme_changed = Signal(bool)  # is_dark

    def __init__(self, camera_manager: CameraManager, parent=None):
        super().__init__(parent)
        self._camera_manager = camera_manager
        self._backend_manager = BackendManager.get_instance()
        self._dataset_registry = DatasetRegistry.get_instance()
        self._model_registry = ModelRegistry.get_instance()

        palette = ThemeManager.get_palette()

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        content_widget = QWidget()
        content_widget.setObjectName("contentContainer")
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(32, 24, 32, 32)
        main_layout.setSpacing(22)

        # Header
        header = SectionHeader(
            "Application Settings",
            "Configure appearance themes, preferred camera hardware, default recognition modes, verified local storage, and system diagnostics."
        )
        main_layout.addWidget(header)

        # =====================================================================
        # 1. APPEARANCE
        # =====================================================================
        theme_card = QFrame()
        theme_card.setProperty("class", "cardPanel")
        th_layout = QVBoxLayout(theme_card)
        th_layout.setContentsMargins(20, 16, 20, 16)
        th_layout.setSpacing(12)

        th_title = QLabel("1. APPEARANCE")
        th_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        th_layout.addWidget(th_title)

        th_desc = QLabel("Select your preferred visual environment. Themes are applied instantly across all application screens.")
        th_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        th_layout.addWidget(th_desc)

        th_group = QHBoxLayout()
        th_group.setSpacing(24)

        self.btn_group_theme = QButtonGroup(self)
        self.radio_dark = QRadioButton("Dark Mode (Recommended)")
        self.radio_light = QRadioButton("Light Mode")
        self.radio_system = QRadioButton("System Default")

        self.btn_group_theme.addButton(self.radio_dark, 0)
        self.btn_group_theme.addButton(self.radio_light, 1)
        self.btn_group_theme.addButton(self.radio_system, 2)

        current_theme = SettingsManager.get_theme()
        if current_theme == "light":
            self.radio_light.setChecked(True)
        else:
            self.radio_dark.setChecked(True)

        self.btn_group_theme.idClicked.connect(self._on_theme_selected)

        th_group.addWidget(self.radio_dark)
        th_group.addWidget(self.radio_light)
        th_group.addWidget(self.radio_system)
        th_group.addStretch()

        th_layout.addLayout(th_group)
        main_layout.addWidget(theme_card)

        # =====================================================================
        # 2. PREFERRED CAMERA & HARDWARE
        # =====================================================================
        cam_card = QFrame()
        cam_card.setProperty("class", "cardPanel")
        c_layout = QVBoxLayout(cam_card)
        c_layout.setContentsMargins(20, 16, 20, 16)
        c_layout.setSpacing(12)

        c_title = QLabel("2. PREFERRED CAMERA DEVICE")
        c_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        c_layout.addWidget(c_title)

        c_desc = QLabel("Select your default video capture input for live sign recognition. Hardware selection is persisted across sessions.")
        c_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        c_layout.addWidget(c_desc)

        cam_select_row = QHBoxLayout()
        cam_select_row.setSpacing(10)

        cam_select_row.addWidget(QLabel("Camera Device:"))
        self.camera_pref_combo = QComboBox()
        self.camera_pref_combo.setMinimumWidth(280)
        self.camera_pref_combo.currentIndexChanged.connect(self._on_pref_camera_changed)
        cam_select_row.addWidget(self.camera_pref_combo, 1)

        btn_refresh_cams = QPushButton(" Refresh Devices")
        btn_refresh_cams.setProperty("class", "secondaryBtn")
        btn_refresh_cams.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=14))
        btn_refresh_cams.clicked.connect(self._refresh_cameras)
        cam_select_row.addWidget(btn_refresh_cams)

        c_layout.addLayout(cam_select_row)

        # Warning / Status Banner for camera availability
        self.cam_status_banner = QFrame()
        self.cam_status_banner.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.warning_subtle};
                border: 1px solid {palette.warning};
                border-radius: 6px;
                padding: 6px;
            }}
        """)
        csb_layout = QHBoxLayout(self.cam_status_banner)
        csb_layout.setContentsMargins(10, 6, 10, 6)
        self.cam_status_icon = QLabel()
        self.cam_status_icon.setPixmap(get_pixmap("alert", palette.warning, size=16))
        self.cam_status_msg = QLabel("Checking camera status...")
        self.cam_status_msg.setStyleSheet(f"font-size: 11px; color: {palette.warning};")
        csb_layout.addWidget(self.cam_status_icon)
        csb_layout.addWidget(self.cam_status_msg, 1)
        c_layout.addWidget(self.cam_status_banner)

        # Mirror Preview Checkbox
        self.mirror_pref = QCheckBox("Mirror camera video horizontally (recommended for webcam signing)")
        self.mirror_pref.setChecked(SettingsManager.get_mirror_preview())
        self.mirror_pref.toggled.connect(self._camera_manager.set_mirror_preview)
        c_layout.addWidget(self.mirror_pref)

        self._populate_camera_prefs()
        main_layout.addWidget(cam_card)

        # =====================================================================
        # 3. DEFAULT RECOGNITION MODE
        # =====================================================================
        mode_card = QFrame()
        mode_card.setProperty("class", "cardPanel")
        m_layout = QVBoxLayout(mode_card)
        m_layout.setContentsMargins(20, 16, 20, 16)
        m_layout.setSpacing(12)

        m_title = QLabel("3. DEFAULT RECOGNITION MODE")
        m_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        m_layout.addWidget(m_title)

        m_desc = QLabel("Configure which recognition mode is initially activated upon opening the Predict Live workspace.")
        m_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        m_layout.addWidget(m_desc)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_row.addWidget(QLabel("Default Mode:"))

        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(280)
        self.mode_combo.addItem(f"{MODE_STATIC} — Single-frame posture landmarks", userData=MODE_STATIC)
        self.mode_combo.addItem(f"{MODE_DYNAMIC} — Multi-frame temporal sequence trajectories", userData=MODE_DYNAMIC)
        self.mode_combo.addItem(f"{MODE_COMBINED} — Integrated static & dynamic pipeline", userData=MODE_COMBINED)

        curr_mode = SettingsManager.get_default_mode()
        for i in range(self.mode_combo.count()):
            if self.mode_combo.itemData(i) == curr_mode:
                self.mode_combo.setCurrentIndex(i)
                break

        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        mode_row.addStretch()

        m_layout.addLayout(mode_row)
        main_layout.addWidget(mode_card)

        # =====================================================================
        # 4. STORAGE DIRECTORIES
        # =====================================================================
        storage_card = QFrame()
        storage_card.setProperty("class", "cardPanel")
        s_layout = QVBoxLayout(storage_card)
        s_layout.setContentsMargins(20, 16, 20, 16)
        s_layout.setSpacing(12)

        s_title = QLabel("4. STORAGE DIRECTORIES & BACKEND LOCATION")
        s_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        s_layout.addWidget(s_title)

        s_desc = QLabel("Verified local directories used for managed datasets, model artifacts, and external ML backend links.")
        s_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        s_layout.addWidget(s_desc)

        # Row 1: Managed Datasets
        self.row_data = StorageDirRow(
            title="Managed Datasets Directory",
            path_str=str(MANAGED_DATA_DIR.resolve()),
            on_open_click=lambda: self._open_directory(MANAGED_DATA_DIR),
            custom_btn_text="Open Folder"
        )
        s_layout.addWidget(self.row_data)

        # Row 2: Model Storage
        self.row_models = StorageDirRow(
            title="Model Artifacts Directory",
            path_str=str(MODELS_DIR.resolve()),
            on_open_click=lambda: self._open_directory(MODELS_DIR),
            custom_btn_text="Open Folder"
        )
        s_layout.addWidget(self.row_models)

        # Row 3: Registered Backend Folder
        be_dir = self._backend_manager.backend_directory or "None Registered"
        self.row_backend = StorageDirRow(
            title="Registered Backend Reference",
            path_str=be_dir,
            on_open_click=self._open_backend_config,
            custom_btn_text="Configure..."
        )
        s_layout.addWidget(self.row_backend)

        main_layout.addWidget(storage_card)

        # =====================================================================
        # 5. APPLICATION INFORMATION & DIAGNOSTICS
        # =====================================================================
        diag_card = QFrame()
        diag_card.setProperty("class", "cardPanelElevated")
        d_layout = QVBoxLayout(diag_card)
        d_layout.setContentsMargins(20, 18, 20, 18)
        d_layout.setSpacing(14)

        d_title = QLabel("5. APPLICATION INFORMATION & SYSTEM DIAGNOSTICS")
        d_title.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.accent}; letter-spacing: 0.6px;")
        d_layout.addWidget(d_title)

        diag_grid = QGridLayout()
        diag_grid.setHorizontalSpacing(16)
        diag_grid.setVerticalSpacing(8)

        diag_info = [
            ("Application Version:", "SignFlow Studio v1.0.0 (School Exhibition Release)"),
            ("Exhibition Affiliation:", "Class XI • Yojshakti Model School, Rohini Sector 3"),
            ("UI Framework:", "PySide6 (Qt for Python 6.12.0)"),
            ("Python Runtime:", f"{platform.python_version()} ({Path(sys.executable).name})"),
            ("Host OS:", f"{platform.system()} {platform.release()} ({platform.machine()})"),
            ("Camera Subsystem:", "Qt Multimedia (QMediaDevices & FFmpeg engine)"),
            ("Active Dataset:", self._get_active_dataset_summary()),
            ("Cataloged Models:", f"{len(self._model_registry.get_all_models())} verified model artifact(s)"),
            ("ML Adapter Status:", self._backend_manager.status)
        ]

        for idx, (label_txt, val_txt) in enumerate(diag_info):
            lbl_key = QLabel(label_txt)
            lbl_key.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {palette.text_secondary};")
            lbl_val = QLabel(val_txt)
            lbl_val.setStyleSheet(f"font-size: 12px; color: {palette.text_primary};")
            diag_grid.addWidget(lbl_key, idx, 0)
            diag_grid.addWidget(lbl_val, idx, 1)

        d_layout.addLayout(diag_grid)

        # Reset button
        btn_row = QHBoxLayout()
        btn_reset = QPushButton(" Reset Preferences to Defaults")
        btn_reset.setProperty("class", "secondaryBtn")
        btn_reset.setIcon(get_icon("trash", palette.text_secondary, palette.danger, size=14))
        btn_reset.clicked.connect(self._reset_defaults)

        btn_row.addStretch()
        btn_row.addWidget(btn_reset)
        d_layout.addLayout(btn_row)

        main_layout.addWidget(diag_card)

        # Bottom stretch
        main_layout.addStretch()

        scroll.setWidget(content_widget)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def _get_active_dataset_summary(self) -> str:
        active = self._dataset_registry.get_active_static_dataset()
        if active:
            return f"{active.name} ({active.row_count} rows, {active.column_count} columns)"
        return "None selected"

    def _refresh_cameras(self):
        self._camera_manager.refresh_devices()
        self._populate_camera_prefs()

    def _populate_camera_prefs(self):
        palette = ThemeManager.get_palette()
        devices = self._camera_manager.devices
        saved_id = SettingsManager.get_preferred_camera_id()
        saved_name = SettingsManager.get_preferred_camera_name()

        self.camera_pref_combo.blockSignals(True)
        self.camera_pref_combo.clear()

        if not devices:
            self.camera_pref_combo.addItem("No video capture devices detected", userData="")
            self.cam_status_banner.setVisible(True)
            self.cam_status_banner.setStyleSheet(f"background-color: {palette.danger_subtle}; border: 1px solid {palette.danger}; border-radius: 6px;")
            self.cam_status_icon.setPixmap(get_pixmap("alert", palette.danger, size=16))
            self.cam_status_msg.setText("No cameras detected. Please connect a physical USB webcam or virtual camera.")
            self.cam_status_msg.setStyleSheet(f"font-size: 11px; color: {palette.danger};")
            self.camera_pref_combo.blockSignals(False)
            return

        sel_idx = 0
        device_found = False
        for idx, dev in enumerate(devices):
            self.camera_pref_combo.addItem(dev.description, userData=dev.id_str)
            if dev.id_str == saved_id:
                sel_idx = idx
                device_found = True

        # If user had a saved camera that is currently disconnected
        if saved_id and not device_found:
            disp_name = saved_name or "Previous Camera"
            self.camera_pref_combo.addItem(f"{disp_name} (Disconnected)", userData=saved_id)
            sel_idx = self.camera_pref_combo.count() - 1
            self.cam_status_banner.setVisible(True)
            self.cam_status_banner.setStyleSheet(f"background-color: {palette.warning_subtle}; border: 1px solid {palette.warning}; border-radius: 6px;")
            self.cam_status_icon.setPixmap(get_pixmap("alert", palette.warning, size=16))
            self.cam_status_msg.setText(f"Preferred device '{disp_name}' is currently disconnected. Please reconnect or switch.")
            self.cam_status_msg.setStyleSheet(f"font-size: 11px; color: {palette.warning};")
        else:
            self.cam_status_banner.setVisible(True)
            self.cam_status_banner.setStyleSheet(f"background-color: {palette.success_subtle}; border: 1px solid {palette.success}; border-radius: 6px;")
            self.cam_status_icon.setPixmap(get_pixmap("check", palette.success, size=16))
            self.cam_status_msg.setText(f"Camera verified and ready for live recognition.")
            self.cam_status_msg.setStyleSheet(f"font-size: 11px; color: {palette.success};")

        self.camera_pref_combo.setCurrentIndex(sel_idx)
        self.camera_pref_combo.blockSignals(False)

    def _on_pref_camera_changed(self, index: int):
        if index >= 0:
            dev_id = self.camera_pref_combo.itemData(index)
            name = self.camera_pref_combo.currentText()
            if dev_id:
                SettingsManager.set_preferred_camera_id(dev_id, name)
                self._camera_manager.select_device(dev_id)

    def _on_mode_changed(self, index: int):
        if index >= 0:
            mode = self.mode_combo.itemData(index)
            if mode:
                SettingsManager.set_default_mode(mode)

    def _on_theme_selected(self, id_val: int):
        # 0: Dark, 1: Light, 2: System (default to dark)
        is_dark = (id_val != 1)
        theme_str = "light" if id_val == 1 else "dark"
        SettingsManager.set_theme(theme_str)
        self.theme_changed.emit(is_dark)

    def _open_directory(self, folder_path: Path):
        path_str = str(folder_path.resolve())
        if not os.path.exists(path_str):
            QMessageBox.warning(
                self,
                "Directory Unavailable",
                f"The target directory does not exist on disk:\n\n{path_str}\n\nPlease verify the storage path."
            )
            return

        try:
            os.startfile(path_str)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Failed to Open Directory",
                f"Could not open folder in Windows Explorer:\n{str(e)}"
            )

    def _open_backend_config(self):
        dlg = BackendImportDialog(self.window())
        dlg.exec()
        be_dir = self._backend_manager.backend_directory or "None Registered"
        self.row_backend.path_lbl.setText(be_dir)
        exists = os.path.exists(be_dir) if be_dir != "None Registered" else False
        self.row_backend.pill.set_status(
            "Verified on Disk" if exists else ("Unlinked" if be_dir == "None Registered" else "Missing on Disk"),
            "active" if exists else ("neutral" if be_dir == "None Registered" else "error")
        )

    def _reset_defaults(self):
        res = QMessageBox.question(
            self,
            "Reset Preferences",
            "Are you sure you want to restore default application settings?\n\nThis will reset your theme, camera preference, and recognition mode defaults.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if res == QMessageBox.StandardButton.Yes:
            SettingsManager.set_theme("dark")
            SettingsManager.set_default_mode(MODE_STATIC)
            SettingsManager.set_mirror_preview(True)
            self.radio_dark.setChecked(True)
            self.theme_changed.emit(True)
            self._populate_camera_prefs()
            self.mode_combo.setCurrentIndex(0)
            QMessageBox.information(self, "Settings Reset", "Application preferences have been restored to defaults.")
