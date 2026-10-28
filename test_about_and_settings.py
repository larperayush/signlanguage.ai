"""
Unit and functional test suite for the redesigned About the Project editorial page
and the completed Settings page in SignFlow Studio.
"""
import sys
from PySide6.QtWidgets import QApplication, QLabel, QRadioButton
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from app.ui.pages.overview_page import OverviewPage
from app.ui.pages.settings_page import SettingsPage, StorageDirRow
from app.core.config import SettingsManager, MODE_STATIC, MODE_DYNAMIC, MODE_COMBINED


def run_tests():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()

    print("\n--- Testing 'About the Project' Editorial Page ---")
    about_page: OverviewPage = window.page_overview
    assert isinstance(about_page, OverviewPage)

    labels = about_page.findChildren(QLabel)
    label_texts = [lbl.text() for lbl in labels]
    combined_text = " ".join(label_texts)

    # 1. Main Heading & Subtitle
    assert "Technology that understands a gesture." in combined_text, "Main heading missing"
    assert "An exploration of sign language recognition through computer vision and machine learning." in combined_text, "Subtitle missing"
    print("  [OK] Main Heading & Subtitle verified.")

    # 2. Meet the Creators
    assert "Ayush Pandey & Rajat" in combined_text, "Creators names missing"
    assert "Class XI" in combined_text, "Class XI missing"
    assert "Yojshakti Model School, Rohini Sector 3" in combined_text, "School location missing"
    print("  [OK] Meet the Creators verified (Ayush Pandey & Rajat, Class XI, Yojshakti Model School).")

    # 3. 01 — Who are we?
    assert "01 — Who are we?" in combined_text, "Question 01 title missing"
    assert "We are two friends with a shared interest in technology" in combined_text, "Q01 text missing"
    assert "SignFlow Studio is the result of that curiosity, teamwork, and experimentation." in combined_text, "Q01 conclusion missing"
    print("  [OK] 01 — Who are we? verified.")

    # 4. 02 — Why did we build SignFlow Studio?
    assert "02 — Why did we build SignFlow Studio?" in combined_text, "Question 02 title missing"
    assert "whether a computer could recognize hand signs through a camera" in combined_text, "Q02 text missing"
    assert "meaningful purpose beyond our exhibition" in combined_text, "Q02 text missing"
    print("  [OK] 02 — Why did we build SignFlow Studio? verified.")

    # 5. 03 — What does our project actually do?
    assert "03 — What does our project actually do?" in combined_text, "Question 03 title missing"
    assert "SignFlow Studio is a Windows desktop application" in combined_text, "Q03 text missing"
    assert "computer vision to detect hand landmarks and machine learning to classify supported signs" in combined_text, "Q03 text missing"
    assert "experimental prototype, and we are continuing to improve its recognition capabilities" in combined_text, "Q03 text missing"
    print("  [OK] 03 — What does our project actually do? verified.")

    # 6. 04 — Where can this project be used in real life?
    assert "04 — Where can this project be used in real life?" in combined_text, "Question 04 title missing"
    assert "Everyday communication" in combined_text, "Everyday communication missing"
    assert "Education" in combined_text, "Education missing"
    assert "Public-facing environments" in combined_text, "Public-facing environments missing"
    assert "AI research and development" in combined_text, "AI research and development missing"
    assert "student prototype, not a complete sign-language translation system" in combined_text, "Q04 disclaimer missing"
    print("  [OK] 04 — Where can this project be used in real life? verified.")

    # 7. Closing Statement
    assert "Built with curiosity. Improved through experimentation." in combined_text, "Closing statement missing"
    print("  [OK] Closing statement verified.")

    print("\n--- Testing 'Settings' Page ---")
    settings_page: SettingsPage = window.page_settings
    assert isinstance(settings_page, SettingsPage)

    # 1. Appearance Radio Buttons
    radios = settings_page.findChildren(QRadioButton)
    radio_labels = [r.text() for r in radios]
    assert any("Dark Mode" in r for r in radio_labels), "Dark mode radio missing"
    assert any("Light Mode" in r for r in radio_labels), "Light mode radio missing"
    assert any("System Default" in r for r in radio_labels), "System appearance radio missing"
    
    # Toggle Light Mode
    settings_page.radio_light.click()
    assert SettingsManager.get_theme() == "light", "SettingsManager theme not updated to light"
    settings_page.radio_dark.click()
    assert SettingsManager.get_theme() == "dark", "SettingsManager theme not updated to dark"
    print("  [OK] Appearance settings toggle and persistence verified.")

    # 2. Preferred Camera
    cam_count = settings_page.camera_pref_combo.count()
    assert cam_count > 0, "Camera dropdown is empty"
    selected_cam = settings_page.camera_pref_combo.currentText()
    print(f"  [OK] Preferred camera detected: '{selected_cam}'.")

    # 3. Default Recognition Mode
    mode_count = settings_page.mode_combo.count()
    assert mode_count == 3, f"Expected 3 modes in mode combo, got {mode_count}"
    settings_page.mode_combo.setCurrentIndex(1)
    assert SettingsManager.get_default_mode() == MODE_DYNAMIC, "Default mode dynamic not set"
    settings_page.mode_combo.setCurrentIndex(0)
    assert SettingsManager.get_default_mode() == MODE_STATIC, "Default mode static not set"
    print("  [OK] Default recognition mode dropdown and persistence verified.")

    # 4. Storage Directories
    storage_rows = settings_page.findChildren(StorageDirRow)
    assert len(storage_rows) == 3, f"Expected 3 StorageDirRows, got {len(storage_rows)}"
    print("  [OK] Storage directory rows verified (Managed Datasets, Models, Backend Reference).")

    # 5. Application Information & Diagnostics
    s_labels = [l.text() for l in settings_page.findChildren(QLabel)]
    s_all_text = " ".join(s_labels)
    assert "SignFlow Studio v1.0.0" in s_all_text, "Version string missing"
    assert "Yojshakti Model School" in s_all_text, "School diagnostic missing"
    assert "PySide6" in s_all_text, "UI framework diagnostic missing"
    print("  [OK] Application information and diagnostics verified.")

    print("\n========================================================")
    print(" [ALL ABOUT & SETTINGS TESTS PASSED SUCCESSFULLY - 100%]")
    print("========================================================\n")


if __name__ == "__main__":
    run_tests()
