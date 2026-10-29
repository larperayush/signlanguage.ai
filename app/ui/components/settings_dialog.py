"""
Settings and Developer Tools Modal Dialog for SignFlow Studio.
Organized per design specifications:
1. Connect Apps (first item, moved from top bar)
2. Backend Setup (external ML project location & mappings)
3. Development Tools Bar with "Self Train" slider toggle to unlock developer workspaces.
"""
from typing import Optional
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame,
    QScrollArea, QWidget, QRadioButton, QButtonGroup, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QPainter, QColor, QBrush, QPen

from app.core.theme import ThemeManager
from app.core.config import SettingsManager
from app.core.backend_manager import BackendManager, STATUS_CONNECTED
from app.resources.icons import get_icon, get_pixmap
from app.ui.components.widgets import StatusPill
from app.ui.components.connect_apps_dialog import ConnectAppsDialog
from app.ui.components.backend_import_dialog import BackendImportDialog


class ToggleSwitch(QWidget):
    """
    Minimalist, tactile pill slider switch matching the warm cream / espresso theme.
    """
    toggled = Signal(bool)

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setFixedSize(48, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        if self._checked != checked:
            self._checked = checked
            self.update()
            self.toggled.emit(self._checked)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setChecked(not self._checked)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager.get_palette()

        # Track background
        if self._checked:
            bg_color = QColor(palette.accent)
            border_color = QColor(palette.accent_pressed)
        else:
            bg_color = QColor(palette.bg_hover)
            border_color = QColor(palette.border_default)

        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(1, 1, 46, 24, 12, 12)

        # Thumb
        thumb_x = 25 if self._checked else 3
        thumb_color = QColor("#1C1917" if (self._checked and not palette.is_dark) else "#FFFFFF")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(thumb_color))
        painter.drawEllipse(thumb_x, 3, 20, 20)


class SettingsDialog(QDialog):
    """
    Unified Settings modal providing:
    - Connect Apps trigger
    - Backend Setup configuration
    - Development Tools bar with Self Train slider
    """
    developer_mode_changed = Signal(bool)
    navigate_requested = Signal(int)
    theme_toggled = Signal(bool)

    def __init__(self, is_developer_mode: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings & Developer Tools")
        self.setMinimumWidth(600)
        self.setModal(True)
        self._is_dev_mode = is_developer_mode
        self._backend_manager = BackendManager.get_instance()

        self._setup_ui()

    def _setup_ui(self):
        palette = ThemeManager.get_palette()
        root_vbox = QVBoxLayout(self)
        root_vbox.setContentsMargins(0, 0, 0, 0)
        root_vbox.setSpacing(0)

        # Scroll Area for clean scrolling on smaller screens
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("contentScrollArea")

        content_widget = QWidget()
        content_widget.setObjectName("contentContainer")
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        # Header Title
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        title_lbl = QLabel("Application Settings & Tools")
        title_lbl.setStyleSheet(f"font-size: 19px; font-weight: 800; color: {palette.text_primary};")
        desc_lbl = QLabel("Manage live streaming app connections, ML backend linkages, and developer training tools.")
        desc_lbl.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary};")
        title_box.addWidget(title_lbl)
        title_box.addWidget(desc_lbl)
        layout.addLayout(title_box)

        # =====================================================================
        # 1. CONNECT APPS (FIRST ITEM)
        # =====================================================================
        apps_card = QFrame()
        apps_card.setProperty("class", "cardPanelElevated")
        ac_layout = QVBoxLayout(apps_card)
        ac_layout.setContentsMargins(18, 16, 18, 16)
        ac_layout.setSpacing(10)

        ac_header = QHBoxLayout()
        ac_title = QLabel("CONNECT APPLICATIONS")
        ac_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")
        ac_header.addWidget(ac_title)
        ac_header.addStretch()

        self.connect_apps_btn = QPushButton(" Connect Apps +")
        self.connect_apps_btn.setProperty("class", "primaryBtn")
        self.connect_apps_btn.setIcon(get_icon("apps", "#1C1917" if not palette.is_dark else "#161513", size=15))
        self.connect_apps_btn.clicked.connect(self._open_connect_apps)
        ac_header.addWidget(self.connect_apps_btn)

        ac_layout.addLayout(ac_header)

        ac_desc = QLabel(
            "Connect SignFlow live translation to video meeting platforms and streaming applications "
            "like Discord, Zoom, OBS Studio, and Microsoft Teams via Virtual Camera broadcasting."
        )
        ac_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.4;")
        ac_desc.setWordWrap(True)
        ac_layout.addWidget(ac_desc)

        layout.addWidget(apps_card)

        # =====================================================================
        # 2. BACKEND SETUP (BELOW CONNECT APPS)
        # =====================================================================
        backend_card = QFrame()
        backend_card.setProperty("class", "cardPanel")
        be_layout = QVBoxLayout(backend_card)
        be_layout.setContentsMargins(18, 16, 18, 16)
        be_layout.setSpacing(10)

        be_header = QHBoxLayout()
        be_title = QLabel("BACKEND SETUP")
        be_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.6px;")

        is_conn = (self._backend_manager.status == STATUS_CONNECTED)
        self.be_pill = StatusPill("Connected" if is_conn else "Unlinked", "active" if is_conn else "neutral")

        be_header.addWidget(be_title)
        be_header.addStretch()
        be_header.addWidget(self.be_pill)
        be_layout.addLayout(be_header)

        be_desc = QLabel(
            "Configure your external Python machine learning project directory, verify detected scripts "
            "(predict_live.py, train_model.py), and inspect module mappings."
        )
        be_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.4;")
        be_desc.setWordWrap(True)
        be_layout.addWidget(be_desc)

        be_action_row = QHBoxLayout()
        be_path = self._backend_manager.backend_directory or "No external directory registered"
        self.lbl_be_path = QLabel(f"Path: {Path(be_path).name if be_path != 'No external directory registered' else be_path}")
        self.lbl_be_path.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-family: monospace;")
        be_action_row.addWidget(self.lbl_be_path, 1)

        self.btn_backend_setup = QPushButton(" Backend Setup...")
        self.btn_backend_setup.setProperty("class", "secondaryBtn")
        self.btn_backend_setup.setIcon(get_icon("folder", palette.text_secondary, palette.accent, size=14))
        self.btn_backend_setup.clicked.connect(self._open_backend_setup)
        be_action_row.addWidget(self.btn_backend_setup)

        be_layout.addLayout(be_action_row)
        layout.addWidget(backend_card)

        # =====================================================================
        # 3. DEVELOPMENT TOOLS BAR & SELF TRAIN SLIDER
        # =====================================================================
        # Visual divider bar titled DEVELOPMENT TOOLS
        dev_bar = QFrame()
        dev_bar.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.bg_sidebar};
                border-top: 1px solid {palette.border_default};
                border-bottom: 1px solid {palette.border_default};
                border-radius: 4px;
            }}
        """)
        dev_bar_layout = QHBoxLayout(dev_bar)
        dev_bar_layout.setContentsMargins(14, 8, 14, 8)
        lbl_bar = QLabel("DEVELOPMENT TOOLS")
        lbl_bar.setStyleSheet(f"font-size: 10px; font-weight: 800; color: {palette.accent_label}; letter-spacing: 0.8px;")
        dev_bar_layout.addWidget(lbl_bar)
        dev_bar_layout.addStretch()
        layout.addWidget(dev_bar)

        # Self Train Slider Card
        train_card = QFrame()
        train_card.setProperty("class", "cardPanel")
        tc_layout = QVBoxLayout(train_card)
        tc_layout.setContentsMargins(18, 16, 18, 16)
        tc_layout.setSpacing(12)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(14)

        slider_info = QVBoxLayout()
        slider_info.setSpacing(3)
        lbl_train_title = QLabel("Self Train Mode")
        lbl_train_title.setStyleSheet(f"font-size: 13.5px; font-weight: 700; color: {palette.text_primary};")
        lbl_train_desc = QLabel(
            "Enable to unlock dataset collection, static sign classifier training, "
            "dynamic sequence trajectories (J / Z), and the 3D neural network visualizer."
        )
        lbl_train_desc.setStyleSheet(f"font-size: 12px; color: {palette.text_secondary}; line-height: 1.35;")
        lbl_train_desc.setWordWrap(True)
        slider_info.addWidget(lbl_train_title)
        slider_info.addWidget(lbl_train_desc)

        # The Self Train Toggle Switch Slider!
        self.slider_self_train = ToggleSwitch(checked=self._is_dev_mode)
        self.slider_self_train.toggled.connect(self._on_self_train_toggled)

        slider_row.addLayout(slider_info, 1)
        slider_row.addWidget(self.slider_self_train)
        tc_layout.addLayout(slider_row)

        # Developer Workspaces Container (revealed when slider is ON)
        self.dev_tools_box = QFrame()
        self.dev_tools_box.setStyleSheet(f"""
            QFrame {{
                background-color: {palette.accent_subtle};
                border: 1px solid {palette.accent_border};
                border-radius: 6px;
                padding: 10px;
            }}
        """)
        dt_layout = QVBoxLayout(self.dev_tools_box)
        dt_layout.setContentsMargins(8, 8, 8, 8)
        dt_layout.setSpacing(8)

        lbl_dt_status = QLabel("✓ Developer Tools Unlocked — Accessible directly or in sidebar navigation:")
        lbl_dt_status.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {palette.text_primary};")
        dt_layout.addWidget(lbl_dt_status)

        # Quick action jump buttons
        btn_grid = QHBoxLayout()
        btn_grid.setSpacing(8)

        btn_static = QPushButton("Static Training")
        btn_static.setProperty("class", "secondaryBtn")
        btn_static.clicked.connect(lambda: self._jump_to_page(2))

        btn_dynamic = QPushButton("Dynamic Training")
        btn_dynamic.setProperty("class", "secondaryBtn")
        btn_dynamic.clicked.connect(lambda: self._jump_to_page(3))

        btn_dataset = QPushButton("Dataset Manager")
        btn_dataset.setProperty("class", "secondaryBtn")
        btn_dataset.clicked.connect(lambda: self._jump_to_page(4))

        btn_nn = QPushButton("Neural Network Lab")
        btn_nn.setProperty("class", "secondaryBtn")
        btn_nn.clicked.connect(lambda: self._jump_to_page(6))

        btn_grid.addWidget(btn_static)
        btn_grid.addWidget(btn_dynamic)
        btn_grid.addWidget(btn_dataset)
        btn_grid.addWidget(btn_nn)
        dt_layout.addLayout(btn_grid)

        self.dev_tools_box.setVisible(self._is_dev_mode)
        tc_layout.addWidget(self.dev_tools_box)

        layout.addWidget(train_card)

        # Close / Done Button
        bot_row = QHBoxLayout()
        bot_row.addStretch()
        close_btn = QPushButton("Done")
        close_btn.setProperty("class", "primaryBtn")
        close_btn.clicked.connect(self.accept)
        bot_row.addWidget(close_btn)
        layout.addLayout(bot_row)

        scroll.setWidget(content_widget)
        root_vbox.addWidget(scroll)

    def _open_connect_apps(self):
        dlg = ConnectAppsDialog(self)
        dlg.exec()

    def _open_backend_setup(self):
        dlg = BackendImportDialog(self)
        dlg.exec()
        is_conn = (self._backend_manager.status == STATUS_CONNECTED)
        self.be_pill.set_status("Connected" if is_conn else "Unlinked", "active" if is_conn else "neutral")
        be_path = self._backend_manager.backend_directory or "No external directory registered"
        self.lbl_be_path.setText(f"Path: {Path(be_path).name if be_path != 'No external directory registered' else be_path}")

    def _on_self_train_toggled(self, checked: bool):
        self._is_dev_mode = checked
        self.dev_tools_box.setVisible(checked)
        self.developer_mode_changed.emit(checked)

    def _jump_to_page(self, page_index: int):
        self.accept()
        self.navigate_requested.emit(page_index)
