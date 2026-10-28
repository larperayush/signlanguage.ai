"""
Application Entry Point and Lifecycle Manager.
Initializes High-DPI support, Windows AppUserModelID, Theme, and MainWindow.
"""
import sys
import ctypes
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from app.core.config import (
    APP_NAME, APP_DISPLAY_NAME, ORGANIZATION_NAME, ORGANIZATION_DOMAIN,
    SettingsManager, RESOURCES_DIR
)
from app.core.theme import ThemeManager
from app.ui.main_window import MainWindow


def configure_windows_app_id():
    """Ensure Windows taskbar pins and identifies the app with its custom icon."""
    if sys.platform == "win32":
        try:
            app_id = f"{ORGANIZATION_NAME}.{APP_NAME}.v1"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass


def main():
    # Configure Windows OS integration
    configure_windows_app_id()

    # Enable High-DPI attributes
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setOrganizationDomain(ORGANIZATION_DOMAIN)

    # Set Application Icon
    icon_path = RESOURCES_DIR / "app_icon.ico"
    if not icon_path.exists():
        icon_path = RESOURCES_DIR / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Apply Persistent Theme
    is_dark = (SettingsManager.get_theme() != "light")
    ThemeManager.apply_theme(app, is_dark=is_dark)

    # Instantiate and display Main Window
    window = MainWindow()
    window.show()

    # Enter Qt Event Loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
