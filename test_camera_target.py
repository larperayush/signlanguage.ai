"""
Test script validating resolution switching and mirroring on live camera hardware.
"""
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtCore import QTimer

from app.core.camera_manager import CameraManager
from app.core.config import SettingsManager


def test_camera_hardware():
    app = QApplication(sys.argv)
    cm = CameraManager()
    devices = cm.refresh_devices()
    print("Detected devices:", [d.description for d in devices])

    iriun = next((d for d in devices if "Iriun" in d.description), None)
    if not iriun:
        print("Iriun not detected. Using first device:", devices[0].description if devices else "None")
        target = devices[0] if devices else None
    else:
        target = iriun

    if not target:
        print("No camera available to test.")
        return

    print(f"Testing on target: {target.description}")
    print(f"Supported resolutions ({len(target.supported_resolutions)}): {target.supported_resolutions}")

    cm.select_device(target.id_str)
    assert cm.start_preview(), "Failed to start preview"

    # Test initial resolution
    print("Testing initial preview...")
    # Change resolution to 640x480
    if "640x480" in target.supported_resolutions:
        print("Applying 640x480 resolution...")
        success, msg, eff = cm.set_resolution("640x480")
        print(f"set_resolution result: success={success}, msg='{msg}', effective='{eff}'")
        assert success, f"set_resolution failed: {msg}"
    
    # Test mirror toggle
    print("Testing mirror preview toggle...")
    cm.set_mirror_preview(True)
    assert SettingsManager.get_mirror_preview() is True
    cm.set_mirror_preview(False)
    assert SettingsManager.get_mirror_preview() is False
    cm.set_mirror_preview(True)

    cm.stop_preview()
    print("[ALL HARDWARE CAMERA TESTS PASSED]")


if __name__ == "__main__":
    test_camera_hardware()
