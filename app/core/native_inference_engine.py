"""
Native Inference & HUD Rendering Engine for SignFlow Studio.
Directly embeds the 60 FPS OpenCV + MediaPipe + Scikit-Learn prediction pipeline
from predict_live.py inside a high-performance Qt QThread worker.

Features:
- Threaded non-blocking webcam capture (DirectShow / MJPG 60 FPS)
- MediaPipe HandLandmarker with 640px subsampling for ultra-low latency
- Real-time skeleton drawing: Green bone lines (0, 255, 0) and Red dots (0, 0, 255)
- Dynamic trajectory recognition: Yellow active stroke trails, Purple locked trails (J & Z)
- Sleek glassmorphic HUD: Sentence Bar, Predicted Sign, Confidence %, Camera FPS
- Direct QImage emission for seamless Qt widget embedding
- Automatic Virtual Camera broadcasting (Discord, Zoom, Google Meet)
"""
import os
import sys
import time
import math
import cv2
import joblib
import numpy as np
import mediapipe as mp

from PySide6.QtCore import QThread, Signal, Slot, QMutex, QMutexLocker
from PySide6.QtGui import QImage

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

RESAMPLE_POINTS = 16
MIN_STEP_DIST = 0.006

STATE_WAIT_STEADY = 0
STATE_ARMED = 1
STATE_DRAWING = 2


def extract_features(landmarks):
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = max(max(abs(val) for val in raw_coords), 1e-6)
    return [val / scale for val in raw_coords]


def path_arc_length(points):
    if len(points) < 2:
        return 0.0
    return sum(
        math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        for i in range(1, len(points))
    )


def resample_and_normalize_path(points, target_count=RESAMPLE_POINTS):
    if not points:
        return [0.0] * (target_count * 2)

    dists = [0.0]
    for i in range(1, len(points)):
        d = math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
        dists.append(dists[-1] + d)

    total_dist = dists[-1]
    if total_dist < 1e-6:
        return [0.0] * (target_count * 2)

    step = total_dist / (target_count - 1)
    resampled = [points[0]]

    for i in range(1, target_count - 1):
        target = i * step
        idx = 0
        while idx < len(dists) - 2 and dists[idx + 1] < target:
            idx += 1
        seg_len = dists[idx + 1] - dists[idx]
        if seg_len < 1e-6:
            rx, ry = points[idx]
        else:
            ratio = (target - dists[idx]) / seg_len
            rx = points[idx][0] + ratio * (points[idx + 1][0] - points[idx][0])
            ry = points[idx][1] + ratio * (points[idx + 1][1] - points[idx][1])
        resampled.append((rx, ry))

    resampled.append(points[-1])

    ox, oy = resampled[0]
    rel_coords = []
    for rx, ry in resampled:
        rel_coords.extend([rx - ox, ry - oy])

    max_abs = max(max(abs(v) for v in rel_coords), 1e-6)
    return [v / max_abs for v in rel_coords]


def build_dynamic_feature_vector(index_path, pinky_path, pose_list):
    norm_index = resample_and_normalize_path(index_path, RESAMPLE_POINTS)
    norm_pinky = resample_and_normalize_path(pinky_path, RESAMPLE_POINTS)
    mean_pose = np.mean(np.array(pose_list, dtype=np.float64), axis=0)
    max_pose = max(np.max(np.abs(mean_pose)), 1e-6)
    norm_pose = (mean_pose / max_pose).tolist()
    return norm_index + norm_pinky + norm_pose


class NativeInferenceWorker(QThread):
    """
    Dedicated QThread that runs the exact predict_live.py computer-vision loop
    and emits live annotated video frames and prediction metadata to Qt.
    """
    frame_ready = Signal(QImage)
    prediction_changed = Signal(str, float)     # (predicted_sign, confidence_pct)
    sentence_changed = Signal(str)              # accumulated sentence text
    fps_changed = Signal(float)                 # current fps
    error_occurred = Signal(str)
    engine_started = Signal()
    engine_stopped = Signal()

    def __init__(self, classifier_path: str, task_path: str, camera_index: int = 0, parent=None):
        super().__init__(parent)
        self.classifier_path = classifier_path
        self.task_path = task_path
        self.camera_index = camera_index

        self._is_running = False
        self._mutex = QMutex()
        self._current_sentence = ""
        self._latest_prediction = "No Hand"

    def run(self):
        self._is_running = True
        self.engine_started.emit()

        # 1. Load ML Classifier
        if not os.path.exists(self.classifier_path):
            self.error_occurred.emit(f"Classifier model missing:\n{self.classifier_path}")
            self._is_running = False
            self.engine_stopped.emit()
            return

        if not os.path.exists(self.task_path):
            self.error_occurred.emit(f"MediaPipe task missing:\n{self.task_path}")
            self._is_running = False
            self.engine_stopped.emit()
            return

        try:
            loaded = joblib.load(self.classifier_path)
            if isinstance(loaded, dict) and loaded.get("type") == "unified_asl_model":
                static_clf = loaded["static_model"]
                dynamic_clf = loaded.get("dynamic_model", None)
            else:
                static_clf = loaded
                dynamic_clf = None
        except Exception as e:
            self.error_occurred.emit(f"Failed to load classifier: {str(e)}")
            self._is_running = False
            self.engine_stopped.emit()
            return

        # 2. Initialize MediaPipe Detector in Video Mode
        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=self.task_path),
                running_mode=VisionRunningMode.VIDEO,
                num_hands=1
            )
            detector = HandLandmarker.create_from_options(options)
        except Exception as e:
            self.error_occurred.emit(f"Failed to initialize MediaPipe HandLandmarker: {str(e)}")
            self._is_running = False
            self.engine_stopped.emit()
            return

        # 3. Open Camera with DirectShow
        camera = None
        backends = [("DirectShow", cv2.CAP_DSHOW), ("Media Foundation", cv2.CAP_MSMF), ("Default", cv2.CAP_ANY)]
        for name, backend in backends:
            cam = cv2.VideoCapture(self.camera_index, backend)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                cam.set(cv2.CAP_PROP_FPS, 60)
                ret, test_frame = cam.read()
                if ret and test_frame is not None:
                    camera = cam
                    break
                cam.release()

        if camera is None:
            self.error_occurred.emit(f"Could not open camera index {self.camera_index}.")
            detector.close()
            self._is_running = False
            self.engine_stopped.emit()
            return

        # Inference loop state
        from collections import deque
        prediction_buffer = deque(maxlen=8)
        last_timestamp_ms = -1
        prev_frame_time = time.time()
        fps_smooth = 0.0

        pen_state = STATE_WAIT_STEADY
        prev_idx_pt = None
        prev_pnk_pt = None
        steady_frames = 0
        stroke_pause_frames = 0

        live_index_path = []
        live_pinky_path = []
        live_poses = []

        dynamic_hold_label = None
        dynamic_hold_conf = 0.0
        dynamic_hold_until = 0.0
        locked_trail_px = []

        def reset_motion():
            nonlocal pen_state, prev_idx_pt, prev_pnk_pt, steady_frames, stroke_pause_frames
            pen_state = STATE_WAIT_STEADY
            prev_idx_pt = None
            prev_pnk_pt = None
            steady_frames = 0
            stroke_pause_frames = 0
            live_index_path.clear()
            live_pinky_path.clear()
            live_poses.clear()

        # Try to import VirtualCameraService for broadcasting
        try:
            from app.core.virtual_cam_service import VirtualCameraService
            vcam = VirtualCameraService.get_instance()
        except Exception:
            vcam = None

        # 4. Main Inference & Rendering Loop
        while self._is_running:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            cur_time = time.time()
            dt = cur_time - prev_frame_time
            prev_frame_time = cur_time
            if dt > 0:
                inst_fps = 1.0 / dt
                fps_smooth = inst_fps if fps_smooth == 0.0 else (fps_smooth * 0.9 + inst_fps * 0.1)

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Fast 640px processing subsampling for MediaPipe
            target_proc_w = 640
            if w > target_proc_w:
                proc_h = int(target_proc_w * (h / w))
                proc_frame = cv2.resize(frame, (target_proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
                rgb_proc = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_proc = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_proc)
            timestamp_ms = int(time.time() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)

            predicted_label = "No Hand"
            confidence_pct = 0.0
            now = time.time()

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                # Draw standard Green Bones (0, 255, 0)
                for start, end in CONNECTIONS:
                    cv2.line(frame, (int(landmarks[start].x * w), int(landmarks[start].y * h)),
                                   (int(landmarks[end].x * w), int(landmarks[end].y * h)), (0, 255, 0), 2)
                # Draw standard Red Landmark Dots (0, 0, 255)
                for pt in landmarks:
                    cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, (0, 0, 255), -1)

                features = extract_features(landmarks)

                # Static Prediction
                try:
                    probs = static_clf.predict_proba([features])[0]
                    max_idx = probs.argmax()
                    static_conf = probs[max_idx] * 100.0
                    static_pred = static_clf.classes_[max_idx] if static_conf >= 40.0 else "Uncertain"
                except Exception:
                    static_pred = "Uncertain"
                    static_conf = 0.0

                idx_pt = (landmarks[8].x, landmarks[8].y)
                pnk_pt = (landmarks[20].x, landmarks[20].y)

                # Dynamic Gestures: Hold Z/J or Smart Pen Tracking
                if now < dynamic_hold_until and dynamic_hold_label is not None:
                    predicted_label = dynamic_hold_label
                    confidence_pct = dynamic_hold_conf
                else:
                    is_valid_base = (static_pred in ["D", "I"])
                    if not is_valid_base:
                        reset_motion()
                        predicted_label = static_pred
                        confidence_pct = static_conf
                    else:
                        if pen_state == STATE_WAIT_STEADY:
                            if prev_idx_pt is not None:
                                move = max(
                                    math.hypot(idx_pt[0] - prev_idx_pt[0], idx_pt[1] - prev_idx_pt[1]),
                                    math.hypot(pnk_pt[0] - prev_pnk_pt[0], pnk_pt[1] - prev_pnk_pt[1])
                                )
                                if move < MIN_STEP_DIST:
                                    steady_frames += 1
                                else:
                                    steady_frames = 0
                                if steady_frames >= 3:
                                    pen_state = STATE_ARMED
                                    live_index_path = [idx_pt]
                                    live_pinky_path = [pnk_pt]
                                    live_poses = [features]
                            prev_idx_pt = idx_pt
                            prev_pnk_pt = pnk_pt
                            predicted_label = static_pred
                            confidence_pct = static_conf

                        elif pen_state == STATE_ARMED:
                            step = max(
                                math.hypot(idx_pt[0] - live_index_path[-1][0], idx_pt[1] - live_index_path[-1][1]),
                                math.hypot(pnk_pt[0] - live_pinky_path[-1][0], pnk_pt[1] - live_pinky_path[-1][1])
                            )
                            if step >= MIN_STEP_DIST:
                                pen_state = STATE_DRAWING
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)
                            predicted_label = static_pred
                            confidence_pct = static_conf

                        elif pen_state == STATE_DRAWING:
                            step = max(
                                math.hypot(idx_pt[0] - live_index_path[-1][0], idx_pt[1] - live_index_path[-1][1]),
                                math.hypot(pnk_pt[0] - live_pinky_path[-1][0], pnk_pt[1] - live_pinky_path[-1][1])
                            )
                            if step >= 0.003:
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)
                                stroke_pause_frames = 0
                            else:
                                stroke_pause_frames += 1
                                live_index_path.append(idx_pt)
                                live_pinky_path.append(pnk_pt)
                                live_poses.append(features)

                            arc_len = max(path_arc_length(live_index_path), path_arc_length(live_pinky_path))
                            if dynamic_clf is not None and arc_len >= 0.18 and len(live_index_path) >= 12 and stroke_pause_frames >= 4:
                                try:
                                    dyn_vec = build_dynamic_feature_vector(live_index_path, live_pinky_path, live_poses)
                                    d_probs = dynamic_clf.predict_proba([dyn_vec])[0]
                                    d_idx = d_probs.argmax()
                                    d_label = dynamic_clf.classes_[d_idx]
                                    d_conf = d_probs[d_idx] * 100.0

                                    if d_label in ["J", "Z"] and d_conf >= 40.0:
                                        dynamic_hold_label = d_label
                                        dynamic_hold_conf = d_conf
                                        dynamic_hold_until = now + 2.0
                                        active_path = live_pinky_path if d_label == "J" else live_index_path
                                        locked_trail_px = [(int(px * w), int(py * h)) for px, py in active_path]
                                        prediction_buffer.clear()
                                        reset_motion()
                                    else:
                                        reset_motion()
                                except Exception:
                                    reset_motion()

                            predicted_label = static_pred
                            confidence_pct = static_conf

                # Draw Yellow Active Trail or Purple Locked Trail
                if now < dynamic_hold_until and locked_trail_px:
                    for i in range(1, len(locked_trail_px)):
                        cv2.line(frame, locked_trail_px[i - 1], locked_trail_px[i], (255, 0, 255), 4)
                elif pen_state == STATE_DRAWING:
                    trail_pts = [(int(px * w), int(py * h)) for px, py in live_index_path]
                    for i in range(1, len(trail_pts)):
                        cv2.line(frame, trail_pts[i - 1], trail_pts[i], (0, 255, 255), 3)

            else:
                reset_motion()

            if now < dynamic_hold_until and dynamic_hold_label is not None:
                stable_prediction = dynamic_hold_label
                rem_sec = dynamic_hold_until - now
                pred_display = f"{stable_prediction} (Locked {rem_sec:.1f}s)"
            else:
                prediction_buffer.append(predicted_label)
                stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count) if prediction_buffer else "No Hand"
                pred_display = stable_prediction

            self._latest_prediction = stable_prediction

            # --- SLEEK BOTTOM GLASSMORPHIC HUD ---
            margin_x = 20
            hud_h = 70
            hud_y2 = h - 15
            hud_y1 = hud_y2 - hud_h
            sent_h = 42
            sent_y2 = hud_y1 - 10
            sent_y1 = sent_y2 - sent_h

            if sent_y1 > 0:
                roi = frame[sent_y1:hud_y2, margin_x:w - margin_x]
                roi_overlay = roi.copy()
                cv2.rectangle(roi_overlay, (0, 0), (w - 2 * margin_x, sent_h), (18, 20, 26), -1)
                cv2.rectangle(roi_overlay, (0, hud_y1 - sent_y1), (w - 2 * margin_x, hud_y2 - sent_y1), (18, 20, 26), -1)
                cv2.addWeighted(roi_overlay, 0.78, roi, 0.22, 0, roi)
                frame[sent_y1:hud_y2, margin_x:w - margin_x] = roi

                cv2.rectangle(frame, (margin_x, sent_y1), (w - margin_x, sent_y2), (55, 62, 78), 1)
                cv2.rectangle(frame, (margin_x, hud_y1), (w - margin_x, hud_y2), (55, 62, 78), 1)

                tag_scale = 0.44 if w >= 1600 else 0.38
                val_scale = 0.78 if w >= 1600 else 0.65
                sent_scale = 0.65 if w >= 1600 else 0.52

                # 1. Sentence Bar
                sent_text_y = sent_y1 + int(sent_h * 0.68)
                with QMutexLocker(self._mutex):
                    sent_str = self._current_sentence

                if sent_str:
                    cv2.putText(frame, "SENTENCE:", (margin_x + 18, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (140, 150, 165), 1)
                    (tw, _), _ = cv2.getTextSize("SENTENCE: ", cv2.FONT_HERSHEY_SIMPLEX, sent_scale, 1)
                    cv2.putText(frame, sent_str, (margin_x + 18 + tw, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (255, 255, 255), 2)
                else:
                    cv2.putText(frame, "SENTENCE: [Press SPACE to type letter, BACKSPACE to delete, C to clear]",
                                (margin_x + 18, sent_text_y), cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (110, 115, 130), 1)

                # 2. Four Columns in HUD: [LETTER, CONFIDENCE, RESOLUTION, FPS]
                card_w = (w - 2 * margin_x)
                c1_x = margin_x + int(card_w * 0.03)
                c2_x = margin_x + int(card_w * 0.32)
                c3_x = margin_x + int(card_w * 0.60)
                c4_x = margin_x + int(card_w * 0.83)

                tag_y = hud_y1 + 24
                val_y = hud_y1 + 54

                # Col 1: Letter
                cv2.putText(frame, "PREDICTED LETTER", (c1_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
                letter_color = (0, 245, 255) if stable_prediction not in ["No Hand", "Uncertain"] else (130, 135, 145)
                cv2.putText(frame, str(pred_display), (c1_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, letter_color, 2)

                # Col 2: Confidence
                cv2.putText(frame, "CONFIDENCE", (c2_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
                if stable_prediction in ["No Hand", "Uncertain"]:
                    conf_str = "---"
                    conf_color = (130, 135, 145)
                else:
                    conf_str = f"{confidence_pct:.1f}%"
                    conf_color = (100, 255, 140) if confidence_pct >= 70.0 else (80, 210, 255) if confidence_pct >= 40.0 else (130, 135, 145)
                cv2.putText(frame, conf_str, (c2_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, conf_color, 2)

                # Col 3: Resolution
                cv2.putText(frame, "RESOLUTION", (c3_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
                cv2.putText(frame, f"{w}x{h}", (c3_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, (220, 225, 235), 2)

                # Col 4: FPS
                cv2.putText(frame, "PERFORMANCE", (c4_x, tag_y), cv2.FONT_HERSHEY_SIMPLEX, tag_scale, (140, 150, 165), 1)
                fps_color = (100, 255, 140) if fps_smooth >= 28.0 else (80, 210, 255)
                cv2.putText(frame, f"{fps_smooth:.1f} FPS", (c4_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, fps_color, 2)

            # 5. Broadcast to Virtual Camera if active
            rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if vcam and getattr(vcam, "is_broadcasting", False):
                try:
                    vcam.broadcast_frame(rgb_full)
                except Exception:
                    pass

            # 6. Convert to QImage and emit to Qt GUI
            qimg = QImage(rgb_full.data, w, h, w * 3, QImage.Format.Format_RGB888).copy()
            self.frame_ready.emit(qimg)
            self.prediction_changed.emit(stable_prediction, confidence_pct)
            self.fps_changed.emit(fps_smooth)

        # Cleanup
        try:
            camera.release()
            detector.close()
        except Exception:
            pass

        self._is_running = False
        self.engine_stopped.emit()

    # --- Interactive Controls ---
    def commit_current_letter(self):
        """Append the current stable prediction to the sentence."""
        with QMutexLocker(self._mutex):
            cand = self._latest_prediction
            if cand and cand not in ["No Hand", "Uncertain"]:
                # If prediction is like "Z (Locked 1.8s)", extract pure letter
                letter = cand.split()[0].strip()
                self._current_sentence += letter
                self.sentence_changed.emit(self._current_sentence)

    def add_space(self):
        """Append a space to the sentence."""
        with QMutexLocker(self._mutex):
            if self._current_sentence and not self._current_sentence.endswith(" "):
                self._current_sentence += " "
                self.sentence_changed.emit(self._current_sentence)

    def backspace(self):
        """Delete the last character from the sentence."""
        with QMutexLocker(self._mutex):
            if self._current_sentence:
                self._current_sentence = self._current_sentence[:-1]
                self.sentence_changed.emit(self._current_sentence)

    def clear_sentence(self):
        """Clear the accumulated sentence."""
        with QMutexLocker(self._mutex):
            self._current_sentence = ""
            self.sentence_changed.emit(self._current_sentence)

    def get_sentence(self) -> str:
        with QMutexLocker(self._mutex):
            return self._current_sentence

    def stop(self):
        """Signal thread to stop and wait for termination."""
        self._is_running = False
        self.wait(1500)
