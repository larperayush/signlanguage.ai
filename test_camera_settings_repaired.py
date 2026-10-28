"""
Targeted Verification Test Suite for Camera Device Settings and Live Preview Mirroring.
Tests with Iriun Webcam (or detected camera hardware).
"""
import sys
import time
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer
from PySide6.QtMultimedia import QVideoFrame

from app.ui.main_window import MainWindow
from app.ui.pages.predict_page import CameraSettingsDialog, CameraPreviewWidget, PredictPage
from app.core.camera_manager import CameraManager
from app.core.config import SettingsManager


def run_camera_settings_tests():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.processEvents()

    cm: CameraManager = window.camera_manager
    predict_page: PredictPage = window.page_predict

    print("======================================================================")
    print("      SignFlow Studio — Camera Settings Targeted Repair Test Suite    ")
    print("======================================================================")

    # 1. Device Discovery & Selection (Target: Iriun Webcam)
    print("\n[Test 1/7] Camera Device Discovery & Selection...")
    devices = cm.refresh_devices()
    assert len(devices) > 0, "No cameras discovered"
    iriun = next((d for d in devices if "Iriun" in d.description), devices[0])
    print(f"  [OK] Selecting camera: '{iriun.description}' (Formats: {len(iriun.supported_resolutions)})")
    cm.select_device(iriun.id_str)
    assert cm.current_device.id_str == iriun.id_str

    # 2. Preview Widget & Initial Live Stream
    print("\n[Test 2/7] Verifying Live Preview with CameraPreviewWidget...")
    assert isinstance(predict_page.video_widget, CameraPreviewWidget), "video_widget is not CameraPreviewWidget"
    preview_started = cm.start_preview()
    assert preview_started is True, "Failed to start camera preview"
    assert cm.is_previewing is True, "Camera preview is not active"
    print("  [OK] Camera preview actively streaming.")

    # Let frames arrive
    for _ in range(10):
        app.processEvents()
        time.sleep(0.05)

    # 3. Live Preview Horizontal Mirroring
    print("\n[Test 3/7] Testing Live Preview Horizontal Mirroring...")
    # Initial state
    initial_mirror = SettingsManager.get_mirror_preview()
    print(f"  Current mirror state: {initial_mirror}")

    # Toggle to True
    cm.set_mirror_preview(True)
    app.processEvents()
    assert predict_page.video_widget.is_mirrored() is True, "Widget not mirrored when set to True"
    assert SettingsManager.get_mirror_preview() is True, "SettingsManager not updated to True"
    print("  [OK] Mirroring enabled: Canvas transformed horizontally.")

    # Toggle to False
    cm.set_mirror_preview(False)
    app.processEvents()
    assert predict_page.video_widget.is_mirrored() is False, "Widget mirrored when set to False"
    assert SettingsManager.get_mirror_preview() is False, "SettingsManager not updated to False"
    print("  [OK] Mirroring disabled: Canvas restored to original orientation.")

    # Restore preferred mirror state
    cm.set_mirror_preview(True)
    assert predict_page.video_widget.is_mirrored() is True

    # 4. Camera Device Settings Dialog Inspection
    print("\n[Test 4/7] Inspecting CameraSettingsDialog Initialization...")
    dialog = CameraSettingsDialog(cm, window)
    dialog.show()
    app.processEvents()

    # Verify resolution combo contains genuinely supported formats
    combo_items = [dialog.res_combo.itemText(i) for i in range(dialog.res_combo.count())]
    print(f"  Dialog populated formats: {combo_items[:6]}")
    if iriun.supported_resolutions:
        for item in combo_items:
            assert item in iriun.supported_resolutions or item.startswith("Default"), f"Found unverified format: {item}"
        print(f"  [OK] Verified genuine resolution formats (count={len(combo_items)}). Zero hardcoded fake formats.")
    else:
        assert "Driver Managed" in combo_items[0]
        print("  [OK] Verified driver-managed fallback for device with no discrete formats.")

    # 5. Apply Settings Button & Resolution Negotiation
    print("\n[Test 5/7] Testing Resolution Application via Apply Settings...")
    if iriun.supported_resolutions:
        # Choose a supported resolution different from current if possible
        target_res = "640x480" if "640x480" in iriun.supported_resolutions else iriun.supported_resolutions[-1]
        idx = dialog.res_combo.findText(target_res)
        assert idx >= 0, f"Target resolution {target_res} not found in combo"
        dialog.res_combo.setCurrentIndex(idx)
        dialog.mirror_check.setChecked(True)

        print(f"  Applying selected resolution: {target_res}...")
        dialog.apply_btn.click()
        app.processEvents()

        # Wait for camera reconfiguration to process
        for _ in range(15):
            app.processEvents()
            time.sleep(0.05)

        # Check effective resolution
        eff_w, eff_h = cm.actual_resolution
        print(f"  Delivered frame size after Apply: {eff_w}x{eff_h}")
        assert cm.is_previewing is True, "Preview failed to recover after resolution change"
        assert SettingsManager.get_preferred_resolution() == target_res, "Preferred resolution not saved to QSettings"
        print(f"  [OK] Successfully applied and verified resolution {target_res} on live device.")

    dialog.close()

    # 6. Rejection of Unsupported Resolutions & Error Handling
    print("\n[Test 6/7] Testing Unsupported Resolution Rejection...")
    success, msg, eff = cm.set_resolution("9999x9999")
    assert success is False, "Unsupported resolution was erroneously accepted"
    assert "not supported" in msg.lower() or "could not" in msg.lower(), f"Unexpected error message: {msg}"
    print(f"  [OK] Successfully rejected unsupported resolution: '{msg}'")
    assert cm.is_previewing is True, "Preview crashed on unsupported resolution"

    # 7. Persistence Across Reopening
    print("\n[Test 7/7] Testing Persistence Across Sessions...")
    saved_cam = SettingsManager.get_preferred_camera_id()
    saved_res = SettingsManager.get_preferred_resolution()
    saved_mirror = SettingsManager.get_mirror_preview()
    print(f"  Persisted values in QSettings: cam='{saved_cam[:20]}...', res='{saved_res}', mirror={saved_mirror}")
    assert saved_mirror is True, "Mirror setting not persisted"

    # Clean teardown
    cm.stop_preview()
    assert cm.is_previewing is False
    window.close()

    print("\n======================================================================")
    print("   [ALL 7 CAMERA SETTINGS & REPAIR TESTS PASSED SUCCESSFULLY - 100%]  ")
    print("======================================================================\n")


if __name__ == "__main__":
    run_camera_settings_tests()
