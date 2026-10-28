"""
Targeted Verification Test Suite for Camera-Based Static and Dynamic Training Workspaces.
Verifies shared camera architecture, live camera previews, sample collection controls,
gesture sequence recording, and explicit backend disconnect handling.
"""
import sys
import time
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from app.ui.pages.predict_page import PredictPage
from app.ui.pages.static_training_page import StaticTrainingPage
from app.ui.pages.dynamic_training_page import DynamicTrainingPage, TrajectoryVisualizerWidget
from app.core.camera_manager import CameraManager, DeviceInfo
from app.core.interfaces import (
    BackendRegistry,
    DisconnectedStaticCollectionBackend,
    DisconnectedDynamicRecordingBackend
)


def run_training_workspace_tests():
    app = QApplication.instance() or QApplication(sys.argv)

    # Safeguard against any blocking modal Qt dialogs in test runs
    patcher_info = patch.object(QMessageBox, "information", return_value=QMessageBox.StandardButton.Ok)
    patcher_warn = patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Ok)
    patcher_crit = patch.object(QMessageBox, "critical", return_value=QMessageBox.StandardButton.Ok)
    patcher_exec = patch.object(QMessageBox, "exec", return_value=QMessageBox.StandardButton.Ok)
    patcher_info.start()
    patcher_warn.start()
    patcher_crit.start()
    patcher_exec.start()

    window = MainWindow()
    window.show()
    app.processEvents()

    cm: CameraManager = window.camera_manager
    p_predict: PredictPage = window.page_predict
    p_static: StaticTrainingPage = window.page_static
    p_dynamic: DynamicTrainingPage = window.page_dynamic

    print("======================================================================")
    print("  SignFlow Studio — Camera-Based Training Workspaces Verification     ")
    print("======================================================================")

    try:
        # 1. Shared Camera Architecture & Initial Device Setup
        print("\n[Test 1/8] Verifying Shared Camera Architecture...")
        devices = cm.refresh_devices()
        if not devices:
            # Mock camera dependency when no physical hardware is present
            mock_dev = DeviceInfo("mock_cam_id", "Mock Virtual Camera", is_default=True, supported_resolutions=["1280x720", "640x480"])
            cm._devices = [mock_dev]
            devices = cm._devices

        assert len(devices) > 0, "No cameras discovered"
        target_cam = next((d for d in devices if "Iriun" in d.description), devices[0])
        cm.select_device(target_cam.id_str)
        print(f"  [OK] Centralized CameraManager active with device: '{target_cam.description}'")

        # 2. Camera Ownership on Static Training Workspace (Page 2)
        print("\n[Test 2/8] Testing Live Camera Preview on Static Training Page...")
        window._navigate_to_page(2)
        app.processEvents()
        assert window.stack.currentIndex() == 2
        assert cm._video_widget == p_static.video_widget, "Camera ownership not transferred to Static page"

        # Start preview from Static page
        assert p_static.btn_start_cam.isEnabled() is True
        p_static.btn_start_cam.click()
        app.processEvents()
        for _ in range(8):
            app.processEvents()
            time.sleep(0.05)

        assert cm.is_previewing is True, "Preview failed to start on Static page"
        assert p_static.standby_overlay.isVisible() is False, "Standby overlay still visible on Static page"
        assert p_static.btn_stop_cam.isEnabled() is True
        print("  [OK] Live camera preview actively rendering on Static Training workspace.")

        # 3. Static Collection Controls & Sample Buffer
        print("\n[Test 3/8] Testing Static Sample Collection & Review Controls...")
        # Change target sign
        p_static.combo_sign.setCurrentText("B")
        assert p_static.lbl_active_sign.text() == "Sign: B"

        # Sample count starts at genuine 0
        assert "0" in p_static.lbl_sample_count.text()

        # Test expected disconnected state when ML adapter is unavailable
        saved_coll_backend = BackendRegistry.get_static_collection_backend()
        try:
            BackendRegistry.set_static_collection_backend(DisconnectedStaticCollectionBackend())
            p_static._update_backend_state()
            assert p_static.collect_status_pill.text().endswith("Disconnected"), (
                "Must not report disconnected backend as connected!"
            )
            p_static._start_collection()
            app.processEvents()
            assert p_static.collect_status_pill.text().endswith("Disconnected"), (
                "Backend status must remain Disconnected when ML adapter is unavailable."
            )
        finally:
            BackendRegistry.set_static_collection_backend(saved_coll_backend)
            p_static._update_backend_state()

        # Trigger collection with active backend -> cleanly handles state
        p_static._start_collection()
        app.processEvents()
        assert p_static.collect_status_pill.text().endswith("Disconnected") or "Active" in p_static.collect_status_pill.text()
        if getattr(p_static, "_is_collecting", False):
            p_static._stop_collection()
        app.processEvents()
        print("  [OK] Sample collection safely handles backend state.")

        # Check collection guide
        guide_labels = [l.text() for l in p_static.findChildren(QLabel)]
        all_guide_txt = " ".join(guide_labels)
        assert "Natural Hand Variations" in all_guide_txt, "Missing Natural Hand Variations tip"
        assert "Distance Variations" in all_guide_txt, "Missing Distance Variations tip"
        assert "Angle Variations" in all_guide_txt, "Missing Angle Variations tip"
        assert "Lighting Variations" in all_guide_txt, "Missing Lighting Variations tip"
        print("  [OK] Collection Diversity Guide verified across all 4 categories.")

        # Hyperparameters
        assert p_static.spin_estimators.value() == 100
        assert abs(p_static.spin_test_size.value() - 0.20) < 0.01
        assert p_static.spin_seed.value() == 42
        print("  [OK] train_model.py hyperparameters verified.")

        # 4. Safe Ownership Transfer to Dynamic Training Workspace (Page 3)
        print("\n[Test 4/8] Transferring Live Camera Preview to Dynamic Training Page...")
        window._navigate_to_page(3)
        app.processEvents()
        assert window.stack.currentIndex() == 3
        assert cm._video_widget == p_dynamic.video_widget, "Camera ownership not transferred to Dynamic page"
        assert cm.is_previewing is True, "Camera stream dropped during workspace transfer"
        assert p_dynamic.standby_overlay.isVisible() is False, "Standby overlay still visible on Dynamic page"
        print("  [OK] Camera preview transferred live and seamlessly to Dynamic Training workspace.")

        # 5. Dynamic Gesture Selection (Z Prioritized, J Supported)
        print("\n[Test 5/8] Testing Dynamic Gesture Selector (Z vs J)...")
        assert p_dynamic._selected_gesture == "Z"
        assert p_dynamic.visualizer.gesture_sign == "Z"

        # Switch to J
        p_dynamic.btn_gesture_j.click()
        app.processEvents()
        assert p_dynamic._selected_gesture == "J"
        assert p_dynamic.visualizer.gesture_sign == "J"

        # Switch back to Z (Priority)
        p_dynamic.btn_gesture_z.click()
        app.processEvents()
        assert p_dynamic._selected_gesture == "Z"
        assert p_dynamic.visualizer.gesture_sign == "Z"
        print("  [OK] Gesture selector correctly prioritizes Z with J as future support.")

        # 6. Dynamic Sequence Recording Controls
        print("\n[Test 6/8] Testing Dynamic Sequence Recording Controls...")
        assert p_dynamic.spin_window.value() == 30
        assert "0" in p_dynamic.lbl_seq_count.text()

        # Trigger sequence recording without backend -> cleanly handles disconnection
        p_dynamic._start_recording()
        app.processEvents()
        assert p_dynamic.rec_status_pill.text().endswith("Disconnected")
        print("  [OK] Sequence recording safely handles disconnected backend.")

        # Temporal Sequence Principle
        dyn_labels = [l.text() for l in p_dynamic.findChildren(QLabel)]
        dyn_all_txt = " ".join(dyn_labels)
        assert "TEMPORAL SEQUENCE PRINCIPLE" in dyn_all_txt, "Missing Temporal Sequence Principle heading"
        assert "consecutive frames" in dyn_all_txt.lower(), "Missing temporal frame explanation"
        print("  [OK] Temporal Sequence Principle callout verified.")

        # 7. Safe Ownership Transfer Back to Predict Live (Page 1)
        print("\n[Test 7/8] Transferring Camera Ownership to Predict Live...")
        window._navigate_to_page(1)
        app.processEvents()
        assert window.stack.currentIndex() == 1
        assert cm._video_widget == p_predict.video_widget, "Camera ownership not transferred to Predict Live"
        assert cm.is_previewing is True, "Preview dropped during return to Predict Live"
        print("  [OK] Preview transferred back to Predict Live workspace without restarting camera.")

        # 8. Clean Camera Teardown
        print("\n[Test 8/8] Teardown and Resource Cleanup...")
        cm.stop_preview()
        cm._release_camera()
        assert cm.is_previewing is False
        assert p_predict.standby_overlay.isVisible() is True
        window.close()
        app.processEvents()
        from app.core.landmark_service import LandmarkService
        LandmarkService.get_instance().teardown()
        app.processEvents()
        print("  [OK] Centralized camera safely released.")

        print("\n======================================================================")
        print("   [ALL 8 TRAINING WORKSPACE TESTS PASSED SUCCESSFULLY - 100%]        ")
        print("======================================================================\n")

    finally:
        patcher_info.stop()
        patcher_warn.stop()
        patcher_crit.stop()
        patcher_exec.stop()


if __name__ == "__main__":
    run_training_workspace_tests()
    sys.exit(0)

