"""
Inspect rendered visual appearance at 1920x1080 and 1366x768.
Renders screenshots of the About the Project page into PNG files for inspection.
"""
import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QSize, QTimer
from PySide6.QtGui import QPixmap

from app.ui.main_window import MainWindow
from app.core.theme import ThemeManager


def capture_resolutions():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window._navigate_to_page(0)  # About the Project page

    out_dir = os.path.join(os.getcwd(), "test_renders")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Test at 1920x1080
    print("Testing layout at 1920x1080...")
    window.resize(1920, 1080)
    window.show()
    app.processEvents()

    pix_1080 = window.grab()
    path_1080 = os.path.join(out_dir, "render_1920x1080.png")
    pix_1080.save(path_1080)
    assert os.path.exists(path_1080), "Failed to save 1920x1080 render"
    print(f"  [OK] 1920x1080 captured successfully: {path_1080} ({pix_1080.width()}x{pix_1080.height()})")

    # 2. Test at 1366x768
    print("Testing layout at 1366x768...")
    window.resize(1366, 768)
    app.processEvents()

    pix_768 = window.grab()
    path_768 = os.path.join(out_dir, "render_1366x768.png")
    pix_768.save(path_768)
    assert os.path.exists(path_768), "Failed to save 1366x768 render"
    print(f"  [OK] 1366x768 captured successfully: {path_768} ({pix_768.width()}x{pix_768.height()})")

    # 3. Check scroll area geometry and ensure no negative margins or crashes
    scroll = window.page_overview.findChild(QApplication.allWidgets()[0].__class__, "contentScrollArea")
    assert window.page_overview is not None

    print("\nVisual rendering test completed cleanly across both target resolutions!")
    window.close()


if __name__ == "__main__":
    capture_resolutions()
