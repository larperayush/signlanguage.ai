"""
Persistent Left Sidebar Navigation Component.
Features distinctive branding, Linear-inspired navigation items,
and prominent Predict action item.
"""
from typing import Dict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt, Signal

from app.resources.icons import get_icon, get_pixmap
from app.core.theme import ThemeManager


class Sidebar(QWidget):
    """
    Persistent application sidebar.
    Emits navigation_requested(page_index: int).
    """
    navigation_requested = Signal(int)

    PAGE_OVERVIEW = 0
    PAGE_ABOUT = 0
    PAGE_PREDICT = 1
    PAGE_STATIC_TRAINING = 2
    PAGE_DYNAMIC_TRAINING = 3
    PAGE_DATASET_MANAGER = 4
    PAGE_MODEL_MANAGER = 5
    PAGE_SETTINGS = 6
    PAGE_NN_VISUALIZER = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebarContainer")
        self.setFixedWidth(240)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 18, 14, 18)
        main_layout.setSpacing(16)

        # Brand Header
        brand_layout = QHBoxLayout()
        brand_layout.setSpacing(10)

        palette = ThemeManager.get_palette()
        self.logo_lbl = QLabel()
        self.logo_lbl.setPixmap(get_pixmap("predict", palette.accent, size=22, stroke_width=2.5))

        title_vbox = QVBoxLayout()
        title_vbox.setSpacing(0)
        
        app_name = QLabel("SignFlow Studio")
        app_name.setStyleSheet("font-size: 14px; font-weight: 700; letter-spacing: -0.2px;")

        app_edition = QLabel("Computer Vision ASL")
        app_edition.setStyleSheet(f"font-size: 11px; color: {palette.text_muted};")

        title_vbox.addWidget(app_name)
        title_vbox.addWidget(app_edition)

        brand_layout.addWidget(self.logo_lbl)
        brand_layout.addLayout(title_vbox)
        brand_layout.addStretch()

        main_layout.addLayout(brand_layout)

        # Nav items
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        nav_vbox = QVBoxLayout()
        nav_vbox.setSpacing(4)

        nav_items = [
            (self.PAGE_ABOUT, "About the Project", "overview", False),
            (self.PAGE_PREDICT, "Predict Live", "predict", True), # High prominence
            (self.PAGE_STATIC_TRAINING, "Static Training", "static_training", False),
            (self.PAGE_DYNAMIC_TRAINING, "Dynamic Training", "dynamic_training", False),
            (self.PAGE_DATASET_MANAGER, "Dataset Manager", "database", False),
            (self.PAGE_MODEL_MANAGER, "Model Manager", "box", False),
            (self.PAGE_NN_VISUALIZER, "Neural Network Lab", "layers", False),
            (self.PAGE_SETTINGS, "Settings", "settings", False),
        ]

        self.buttons: Dict[int, QPushButton] = {}

        for page_idx, label, icon_name, is_prominent in nav_items:
            btn = QPushButton(f"  {label}")
            btn.setCheckable(True)
            btn.setIcon(get_icon(icon_name, palette.text_secondary, palette.accent, size=18))

            if is_prominent:
                btn.setProperty("class", "navBtnPredict")
                # Extra visual marker
                btn.setText(f"  {label}  ●")
            else:
                btn.setProperty("class", "navBtn")

            btn.clicked.connect(lambda checked, idx=page_idx: self._on_btn_clicked(idx))
            self.btn_group.addButton(btn, page_idx)
            nav_vbox.addWidget(btn)
            self.buttons[page_idx] = btn

        # Default select Overview
        self.buttons[self.PAGE_OVERVIEW].setChecked(True)

        main_layout.addLayout(nav_vbox)
        main_layout.addStretch()

        # Bottom Hardware / System Status Panel
        bottom_panel = QFrame()
        bottom_panel.setProperty("class", "cardPanelElevated")
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(12, 10, 12, 10)
        bottom_layout.setSpacing(6)

        hw_header = QHBoxLayout()
        hw_title = QLabel("HARDWARE STATUS")
        hw_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        hw_header.addWidget(hw_title)
        hw_header.addStretch()

        self.camera_status_lbl = QLabel("Camera: Standby")
        self.camera_status_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary};")

        self.backend_status_lbl = QLabel("ML Adapter: Unlinked")
        self.backend_status_lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-style: italic;")

        bottom_layout.addLayout(hw_header)
        bottom_layout.addWidget(self.camera_status_lbl)
        bottom_layout.addWidget(self.backend_status_lbl)

        main_layout.addWidget(bottom_panel)

    def _on_btn_clicked(self, page_index: int):
        self.navigation_requested.emit(page_index)

    def set_current_page(self, page_index: int):
        btn = self.buttons.get(page_index)
        if btn:
            btn.setChecked(True)

    def update_camera_status(self, text: str, is_active: bool = False):
        palette = ThemeManager.get_palette()
        color = palette.success if is_active else palette.text_secondary
        self.camera_status_lbl.setText(f"Camera: {text}")
        self.camera_status_lbl.setStyleSheet(f"font-size: 11px; color: {color};")

    def update_backend_status(self, text: str, is_connected: bool = False):
        palette = ThemeManager.get_palette()
        color = palette.success if is_connected else palette.text_muted
        self.backend_status_lbl.setText(f"ML Adapter: {text}")
        self.backend_status_lbl.setStyleSheet(f"font-size: 11px; color: {color};")
