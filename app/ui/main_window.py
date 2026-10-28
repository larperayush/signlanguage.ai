"""
Main Application Window.
Integrates Sidebar, TopBar, and Stacked Page Workspaces.
Provides native window resizing, minimize/maximize/close, and safe cleanup.
"""
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QApplication
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from app.core.config import APP_DISPLAY_NAME, RESOURCES_DIR, SettingsManager
from app.core.theme import ThemeManager
from app.core.camera_manager import CameraManager
from app.core.landmark_service import LandmarkService
from app.core.backend_manager import BackendManager, STATUS_CONNECTED

from app.ui.components.sidebar import Sidebar
from app.ui.components.top_bar import TopBar
from app.ui.pages.overview_page import OverviewPage
from app.ui.pages.predict_page import PredictPage
from app.ui.pages.static_training_page import StaticTrainingPage
from app.ui.pages.dynamic_training_page import DynamicTrainingPage
from app.ui.pages.dataset_manager_page import DatasetManagerPage
from app.ui.pages.model_manager_page import ModelManagerPage
from app.ui.pages.settings_page import SettingsPage
from app.ui.pages.neural_network_page import NeuralNetworkPage


class MainWindow(QMainWindow):
    """Primary application shell window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(1040, 680)
        self.resize(1260, 820)

        # Set Window Icon
        icon_path = RESOURCES_DIR / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Initialize shared Camera Manager & Asynchronous Landmark Service
        self.camera_manager = CameraManager(self)
        self.landmark_service = LandmarkService.get_instance()

        # Central Widget & Shell Layout
        central_widget = QWidget(self)
        central_widget.setObjectName("centralWidget")
        self.setCentralWidget(central_widget)

        shell_layout = QHBoxLayout(central_widget)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        # 1. Left Sidebar
        self.sidebar = Sidebar(self)
        shell_layout.addWidget(self.sidebar)

        # 2. Main Right Column (TopBar + Content Stack)
        content_column = QWidget()
        content_vbox = QVBoxLayout(content_column)
        content_vbox.setContentsMargins(0, 0, 0, 0)
        content_vbox.setSpacing(0)

        # Top Bar
        self.top_bar = TopBar(self)
        content_vbox.addWidget(self.top_bar)

        # Content Pages Stack
        self.stack = QStackedWidget()
        self.stack.setObjectName("pageStack")

        # Page instances
        self.page_overview = OverviewPage(camera_manager=self.camera_manager, parent=self)
        self.page_predict = PredictPage(camera_manager=self.camera_manager, parent=self)
        self.page_static = StaticTrainingPage(camera_manager=self.camera_manager, parent=self)
        self.page_dynamic = DynamicTrainingPage(camera_manager=self.camera_manager, parent=self)
        self.page_dataset_manager = DatasetManagerPage(parent=self)
        self.page_model_manager = ModelManagerPage(parent=self)
        self.page_settings = SettingsPage(camera_manager=self.camera_manager, parent=self)
        self.page_nn_visualizer = NeuralNetworkPage(parent=self)

        self.stack.addWidget(self.page_overview)         # Index 0
        self.stack.addWidget(self.page_predict)          # Index 1
        self.stack.addWidget(self.page_static)           # Index 2
        self.stack.addWidget(self.page_dynamic)          # Index 3
        self.stack.addWidget(self.page_dataset_manager)  # Index 4
        self.stack.addWidget(self.page_model_manager)    # Index 5
        self.stack.addWidget(self.page_settings)         # Index 6
        self.stack.addWidget(self.page_nn_visualizer)    # Index 7

        content_vbox.addWidget(self.stack)
        shell_layout.addWidget(content_column, 1)

        # Connect Navigation Signals
        self.sidebar.navigation_requested.connect(self._navigate_to_page)
        self.page_overview.navigate_to.connect(self._navigate_to_page)

        # Connect Theme Signals
        self.top_bar.theme_toggled.connect(self._set_theme)
        self.page_settings.theme_changed.connect(self._set_theme)

        # Connect Camera Manager Status to Sidebar & Topbar
        self.camera_manager.status_changed.connect(self._on_camera_status)

        # Connect Backend Manager Status to Sidebar
        self.backend_manager = BackendManager.get_instance()
        self.backend_manager.status_changed.connect(self._on_backend_status)
        self._on_backend_status(self.backend_manager.status, "active" if self.backend_manager.status == STATUS_CONNECTED else "neutral")

        # Initial Page
        self._navigate_to_page(0)

    def _navigate_to_page(self, page_index: int):
        prev_index = self.stack.currentIndex()
        if prev_index != page_index and prev_index >= 0:
            prev_page = self.stack.widget(prev_index)
            if hasattr(prev_page, "on_page_deactivated"):
                prev_page.on_page_deactivated()

        self.stack.setCurrentIndex(page_index)
        self.sidebar.set_current_page(page_index)

        current_page = self.stack.widget(page_index)
        if hasattr(current_page, "on_page_activated"):
            current_page.on_page_activated()

        # Safely transfer camera preview ownership to active workspace
        if page_index == 1 and hasattr(self.page_predict, "video_widget"):
            self.camera_manager.set_video_widget(self.page_predict.video_widget)
        elif page_index == 2 and hasattr(self.page_static, "video_widget"):
            self.camera_manager.set_video_widget(self.page_static.video_widget)
        elif page_index == 3 and hasattr(self.page_dynamic, "video_widget"):
            self.camera_manager.set_video_widget(self.page_dynamic.video_widget)

        titles = {
            0: "About the Project",
            1: "Live ASL Prediction",
            2: "Static Classifier Training",
            3: "Dynamic Sequence Training",
            4: "Local Dataset Manager",
            5: "Model Artifact Manager",
            6: "Preferences & Settings",
            7: "Neural Network Visualizer"
        }
        self.top_bar.set_page_title(titles.get(page_index, "Studio"))

    def _set_theme(self, is_dark: bool):
        app = QApplication.instance()
        if app:
            ThemeManager.apply_theme(app, is_dark=is_dark)

    def _on_camera_status(self, text: str, status_type: str):
        is_active = (status_type == "active")
        self.sidebar.update_camera_status(text, is_active=is_active)

    def _on_backend_status(self, status: str, severity: str):
        is_connected = (status == STATUS_CONNECTED)
        display_txt = "Connected" if is_connected else ("Pending" if "Pending" in status else "Unlinked")
        self.sidebar.update_backend_status(display_txt, is_connected=is_connected)

    def closeEvent(self, event):
        """Safely release camera hardware and background services when closing."""
        if self.camera_manager:
            self.camera_manager.stop_preview()
            self.camera_manager._release_camera()
        if hasattr(self, "landmark_service") and self.landmark_service:
            self.landmark_service.teardown()
        if hasattr(self, "page_nn_visualizer") and self.page_nn_visualizer:
            self.page_nn_visualizer.closeEvent(event)
        event.accept()
