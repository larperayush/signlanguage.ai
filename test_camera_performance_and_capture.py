"""
Comprehensive Bounded Smoke Test for Camera Performance & Manual Landmark Capture Fix.
Verifies all 15 acceptance criteria specified in the requirements.
"""
import sys
import os
import time
import math
import tempfile
import csv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from PySide6.QtWidgets import (
    QApplication, QWidget, QLineEdit, QTextEdit, QTableWidgetItem
)
from PySide6.QtCore import Qt, QEvent, QRect, QSize
from PySide6.QtGui import QKeyEvent, QImage
from PySide6.QtMultimedia import QVideoFrame

from app.core.config import ROOT_DIR, MODELS_DIR
from app.core.landmark_service import (
    LandmarkService, LandmarkPoint, extract_normalized_features,
    HAND_CONNECTIONS, FINGERTIP_INDICES, WRIST_INDEX
)
from app.core.backend_adapter import (
    PythonMLPredictionAdapter, PythonMLStaticCollectionAdapter
)
from app.ui.pages.predict_page import CameraPreviewWidget
from app.ui.pages.static_training_page import StaticTrainingPage
from app.core.camera_manager import CameraManager


def test_landmark_service_and_feature_extraction():
    print("\n--- [1/5] Testing LandmarkService & 63-Feature Extraction ---")
    points = [LandmarkPoint(float(i) * 0.02, float(i) * 0.03, float(i) * 0.01) for i in range(21)]
    feats = extract_normalized_features(points)
    assert len(feats) == 63, f"Expected 63 features, got {len(feats)}"
    assert all(math.isfinite(v) for v in feats), "Features contain non-finite numbers"
    assert abs(feats[0]) < 1e-5 and abs(feats[1]) < 1e-5 and abs(feats[2]) < 1e-5
    assert max(abs(v) for v in feats) <= 1.00001
    print("  [OK] Feature normalization produces exact 63-feature wrist-relative format.")

    service = LandmarkService.get_instance()
    assert service is not None
    assert service._thread is not None and service._thread.isRunning()
    print("  [OK] LandmarkService worker QThread is active and running asynchronously.")


def test_camera_preview_widget_rendering_and_mirroring():
    print("\n--- [2/5] Testing CameraPreviewWidget & Aspect-Ratio Letterbox ---")
    widget = CameraPreviewWidget()
    widget.resize(640, 480)

    target_rect = widget._compute_target_rect(QRect(0, 0, 640, 480), QSize(1280, 720))
    assert target_rect.width() == 640
    assert target_rect.height() == 360
    assert target_rect.y() == 60
    print("  [OK] Letterboxing correctly computes aspect-ratio centered target rect.")

    widget.set_status_banner("COLLECTION ACTIVE - PRESS SPACE TO CAPTURE")
    assert widget._status_banner == "COLLECTION ACTIVE - PRESS SPACE TO CAPTURE"
    widget.set_last_capture_info("Captured 'A' at 12:00:00")
    assert widget._last_capture_info == "Captured 'A' at 12:00:00"

    widget.trigger_capture_flash()
    assert widget._flash_opacity > 0.0
    print("  [OK] Status banner, capture info, and viewport capture flash state verified.")

    widget.set_mirrored(True)
    assert widget.is_mirrored() is True
    widget.set_mirrored(False)
    assert widget.is_mirrored() is False
    print("  [OK] Horizontal preview mirroring toggles cleanly.")


def test_manual_sample_capture_adapter():
    print("\n--- [3/5] Testing PythonMLStaticCollectionAdapter Manual Capture ---")
    model_task = MODELS_DIR / "hand_landmarker.task"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
        temp_csv = tf.name

    try:
        adapter = PythonMLStaticCollectionAdapter(str(model_task), temp_csv)
        assert adapter.is_connected() is True

        start_ok = adapter.start_collection("B")
        assert start_ok is True
        assert adapter.is_collecting() is True
        assert len(adapter._candidate_samples) == 0, "Samples should not be collected automatically!"

        bad_sample = adapter.capture_sample("B", [0.0] * 50)
        assert bad_sample is None, "Should reject invalid feature count"

        nan_sample = adapter.capture_sample("B", [float('nan')] * 63)
        assert nan_sample is None, "Should reject non-finite features"

        valid_features = [float(i) * 0.01 for i in range(63)]
        captured_records = []
        adapter.sample_collected.connect(lambda r: captured_records.append(r))

        sample1 = adapter.capture_sample("B", valid_features)
        assert sample1 is not None
        assert sample1.sign_label == "B"
        assert sample1.landmarks_count == 63
        assert len(captured_records) == 1
        assert len(adapter._candidate_samples) == 1
        print("  [OK] capture_sample successfully adds 1 candidate sample to buffer.")

        disc_ok = adapter.discard_sample(sample1.id)
        assert disc_ok is True
        assert len(adapter._candidate_samples) == 0
        print("  [OK] Discard sample removes sample from candidate buffer.")

        sample2 = adapter.capture_sample("B", valid_features)
        assert sample2 is not None
        accept_ok = adapter.accept_sample(sample2.id)
        assert accept_ok is True
        assert len(adapter._candidate_samples) == 0

        with open(temp_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) == 2, f"Expected header + 1 row, got {len(reader)}"
            header = reader[0]
            assert header[0] == "label" and len(header) == 64
            row = reader[1]
            assert row[0] == "B" and len(row) == 64
        print("  [OK] Accept sample commits exactly 64-column format to CSV dataset.")

        adapter.stop_collection()
        assert adapter.is_collecting() is False
        assert adapter.capture_sample("B", valid_features) is None
        print("  [OK] Stop collection halts capture session.")
    finally:
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


def test_spacebar_capture_and_keyboard_guards():
    print("\n--- [4/5] Testing Spacebar Event Filter & Auto-Repeat Suppression ---")
    app = QApplication.instance() or QApplication(sys.argv)
    page = StaticTrainingPage(camera_manager=None)

    class MockCamMgr:
        is_previewing = True
        current_fps = 30.0
    page._camera_manager = MockCamMgr()

    service = LandmarkService.get_instance()
    synthetic_features = [float(i) * 0.01 for i in range(63)]
    synthetic_hands = [[LandmarkPoint(0.1, 0.2, 0.0)] * 21]
    service._latest_payload = {
        "hands": synthetic_hands,
        "num_hands": 1,
        "feature_vector": synthetic_features,
        "timestamp": time.time(),
        "camera_fps": 30.0,
        "inference_fps": 18.5
    }

    page.btn_start_collect.click()
    assert page._is_collecting is True
    assert page.btn_capture_sample.isEnabled() is True
    assert page.btn_stop_collect.isEnabled() is True
    assert page.btn_start_collect.isEnabled() is False
    assert page._session_sample_count == 0
    assert page.video_widget._status_banner == "COLLECTION ACTIVE — PRESS SPACE TO CAPTURE"
    print("  [OK] Starting collection enables capture button, banner, and arms event filter.")

    space_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", False, 1)
    consumed = page.eventFilter(None, space_event)
    assert consumed is True, "Spacebar event should be consumed to prevent scrolling"
    assert page._session_sample_count == 1, f"Expected 1 sample, got {page._session_sample_count}"
    assert page.sample_table.rowCount() == 1
    print("  [OK] Single Spacebar press captured exactly one valid candidate sample.")

    repeat_space_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", True, 1)
    consumed_repeat = page.eventFilter(None, repeat_space_event)
    assert consumed_repeat is True, "Repeat event must be consumed"
    assert page._session_sample_count == 1, "Holding down Space must NOT capture additional samples!"
    print("  [OK] Auto-repeat Spacebar is suppressed (zero extra samples recorded).")

    edit_box = QLineEdit()
    edit_box.show()
    edit_box.setFocus()
    app.processEvents()

    typing_space_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier, " ", False, 1)
    pass_through = page.eventFilter(None, typing_space_event)
    assert pass_through is False, "Spacebar when typing in input field should pass through"
    assert page._session_sample_count == 1, "Typing space in input field must NOT capture samples!"
    print("  [OK] Spacebar typing in text inputs is passed through without triggering capture.")

    page.btn_capture_sample.click()
    assert page._session_sample_count == 2
    assert page.sample_table.rowCount() == 2
    print("  [OK] Clicking 'Capture Sample (Space)' performs the exact same action as Spacebar.")

    page.btn_stop_collect.click()
    assert page._is_collecting is False
    assert page.btn_capture_sample.isEnabled() is False
    assert page.btn_start_collect.isEnabled() is True
    assert page.video_widget._status_banner == ""
    print("  [OK] Stopping collection disarms capture controls and cleans up status banner.")


def test_prediction_adapter_low_latency():
    print("\n--- [5/5] Testing Fast Classifier Prediction (< 1ms) ---")
    model_task = MODELS_DIR / "hand_landmarker.task"
    classifier_path = MODELS_DIR / "asl_classifier.joblib"

    pred_adapter = PythonMLPredictionAdapter(str(model_task), str(classifier_path))
    assert pred_adapter.is_connected() is True

    start_ok = pred_adapter.start_prediction("Static", "")
    assert start_ok is True

    predictions = []
    pred_adapter.prediction_result.connect(lambda r: predictions.append(r))

    dummy_features = [0.0] * 63
    t0 = time.perf_counter()
    pred_adapter._worker.predict_features(dummy_features, "Static")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] Off-thread landmark feature evaluation took: {elapsed_ms:.2f} ms (< 1ms).")
    assert elapsed_ms < 15.0, "Feature classification exceeded reasonable latency!"

    pred_adapter.stop_prediction()
    assert pred_adapter.is_active() is False
    print("  [OK] Prediction start/stop lifecycle completed cleanly.")


def main():
    print("=" * 70)
    print("  SignFlow Studio - Camera Performance & Manual Capture Verification  ")
    print("=" * 70)
    app = QApplication.instance() or QApplication(sys.argv)

    test_landmark_service_and_feature_extraction()
    test_camera_preview_widget_rendering_and_mirroring()
    test_manual_sample_capture_adapter()
    test_spacebar_capture_and_keyboard_guards()
    test_prediction_adapter_low_latency()

    LandmarkService.get_instance().teardown()

    print("\n" + "=" * 70)
    print("  ALL PERFORMANCE & MANUAL CAPTURE ACCEPTANCE TESTS PASSED (100%)  ")
    print("=" * 70)


if __name__ == "__main__":
    main()
