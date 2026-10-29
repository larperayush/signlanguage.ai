"""
Persistent Left Sidebar Navigation Component for SignFlow Studio.
Displays Live Translation.
Developer tools (Static Training, Dynamic Training, Dataset Manager, etc.)
only appear when a Developer Model Package is loaded.
"""
from typing import Dict, List
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup
)
from PySide6.QtCore import Signal

from app.resources.icons import get_icon, get_pixmap
from app.core.theme import ThemeManager


class Sidebar(QWidget):
    """
    Persistent application sidebar.
    Emits navigation_requested(page_index: int).
    """
    navigation_requested = Signal(int)

    PAGE_TRANSLATE = 0
    PAGE_SETTINGS = 1
    PAGE_STATIC_TRAINING = 2
    PAGE_DYNAMIC_TRAINING = 3
    PAGE_DATASET_MANAGER = 4
    PAGE_MODEL_MANAGER = 5
    PAGE_NN_VISUALIZER = 6
    PAGE_ABOUT = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarContainer")
        self.setFixedWidth(240)

        self._developer_mode = False
        self._buttons: Dict[int, QPushButton] = {}
        self._dev_widgets: List[QWidget] = []

        self._setup_ui()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 20, 14, 20)
        main_layout.setSpacing(16)

        # Brand Header (Minimalist & Editorial)
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(10)

        self.logo_lbl = QLabel()
        self.logo_lbl.setPixmap(get_pixmap("predict", palette.accent, size=24, stroke_width=2.5))

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(1)

        app_name = QLabel("SignFlow AI")
        app_name.setStyleSheet(f"font-size: 15px; font-weight: 800; color: {palette.text_primary}; letter-spacing: -0.3px;")

        app_edition = QLabel("Open-Source Model Harness")
        app_edition.setStyleSheet(f"font-size: 10.5px; color: {palette.text_muted};")

        title_vbox.addWidget(app_name)
        title_vbox.addWidget(app_edition)

        brand_layout.addWidget(self.logo_lbl)
        brand_layout.addLayout(title_vbox)
        brand_layout.addStretch()

        main_layout.addLayout(brand_layout)

        # Hairline divider
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"border-top: 1px solid {palette.border_subtle};")
        main_layout.addWidget(sep)

        # Nav items container
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.nav_vbox = QVBoxLayout()
        self.nav_vbox.setSpacing(6)

        # 1. Main Navigation Item (Exclusively Live Translation)
        btn_translate = self._create_nav_btn(self.PAGE_TRANSLATE, "Live Translation", "predict", True)
        self.nav_vbox.addWidget(btn_translate)

        # 2. Developer Tools Section Header & Items
        # (Only visible when a Developer Model Package is loaded)
        self.dev_header = QLabel("DEVELOPER SUITE")
        self.dev_header.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px; padding: 14px 6px 4px 6px;")
        self.nav_vbox.addWidget(self.dev_header)
        self._dev_widgets.append(self.dev_header)

        dev_items = [
            (self.PAGE_STATIC_TRAINING, "Static Training", "static_training", False),
            (self.PAGE_DYNAMIC_TRAINING, "Dynamic Training", "dynamic_training", False),
            (self.PAGE_DATASET_MANAGER, "Dataset Manager", "database", False),
            (self.PAGE_MODEL_MANAGER, "Model Artifacts", "box", False),
            (self.PAGE_NN_VISUALIZER, "Neural Network Lab", "layers", False),
            (self.PAGE_ABOUT, "About the Project", "overview", False),
        ]

        for page_idx, label, icon_name, is_prominent in dev_items:
            btn = self._create_nav_btn(page_idx, label, icon_name, is_prominent)
            self.nav_vbox.addWidget(btn)
            self._dev_widgets.append(btn)

        main_layout.addLayout(self.nav_vbox)
        main_layout.addStretch()

        # Bottom Hardware / Status Pills
        bottom_box = QFrame()
        bottom_box.setProperty("class", "cardPanelElevated")
        b_layout = QVBoxLayout(bottom_box)
        b_layout.setContentsMargins(12, 10, 12, 10)
        b_layout.setSpacing(6)

        self.cam_status_lbl = QLabel("Camera: Standby")
        self.cam_status_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-weight: 500;")

        self.vcam_status_lbl = QLabel("Virtual Cam: Ready")
        self.vcam_status_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-weight: 500;")

        b_layout.addWidget(self.cam_status_lbl)
        b_layout.addWidget(self.vcam_status_lbl)
        main_layout.addWidget(bottom_box)

        # By default developer mode is False
        self.set_developer_mode(False)

    def _create_nav_btn(self, page_idx: int, label: str, icon_name: str, is_prominent: bool) -> QPushButton:
        palette = ThemeManager.get_palette()
        btn = QPushButton(f"  {label}")
        btn.setCheckable(True)
        btn.setIcon(get_icon(icon_name, palette.text_secondary, palette.accent, size=17))

        if is_prominent:
            btn.setProperty("class", "navBtnPredict")
        else:
            btn.setProperty("class", "navBtn")

        btn.clicked.connect(lambda checked, idx=page_idx: self._on_btn_clicked(idx))
        self.btn_group.addButton(btn, page_idx)
        self._buttons[page_idx] = btn
        return btn

    def set_developer_mode(self, enabled: bool):
        """Show or hide developer suite tools."""
        self._developer_mode = enabled
        for w in self._dev_widgets:
            w.setVisible(enabled)

    def is_developer_mode(self) -> bool:
        return self._developer_mode

    def set_current_page(self, page_idx: int):
        btn = self._buttons.get(page_idx)
        if btn:
            btn.setChecked(True)

    def _on_btn_clicked(self, page_idx: int):
        self.navigation_requested.emit(page_idx)

    def update_camera_status(self, text: str, is_active: bool):
        palette = ThemeManager.get_palette()
        color = palette.accent_label if is_active else palette.text_muted
        self.cam_status_lbl.setText(f"Camera: {text}")
        self.cam_status_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: 500;")

    def update_virtual_cam_status(self, text: str, is_active: bool):
        palette = ThemeManager.get_palette()
        color = palette.accent_label if is_active else palette.text_muted
        self.vcam_status_lbl.setText(text)
        self.vcam_status_lbl.setStyleSheet(f"font-size: 11px; color: {color}; font-weight: 500;")

    def update_backend_status(self, text: str, is_connected: bool):
        pass
