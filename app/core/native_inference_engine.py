"""
SignFlow Studio - Native Inference Engine
Real-time MediaPipe Hand Landmarks & 26-Class ASL Gesture Classification.
Ports full predict_live.py geometric disambiguation, trajectory tracking,
and real-time glassmorphic HUD overlay to high-performance PySide6 QThread.
"""

import os
import sys
import time
import math
from collections import deque
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


def extract_features(landmarks):
    """
    Extract wrist-relative, scale-normalized 3D coordinates (63 features).
    Exactly matches the model's training pipeline.
    """
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = max(max(abs(val) for val in raw_coords), 1e-6)
    return [val / scale for val in raw_coords]


def disambiguate_r_u_v(landmarks):
    """
    Geometrically separates R, U, and V using physical ASL invariants:
    - R: Index and middle fingers crossed (inversion of finger axis or ratio <= 0.65).
    - V: Index and middle fingers spread apart (ratio >= 1.25 or diverging angle cos < 0.96).
    - U: Index and middle fingers touching and parallel side-by-side (ratio < 1.25 and cos >= 0.95).
    """
    dx_tip = landmarks[8].x - landmarks[12].x
    dy_tip = landmarks[8].y - landmarks[12].y
    dz_tip = landmarks[8].z - landmarks[12].z
    d_tips = math.sqrt(dx_tip * dx_tip + dy_tip * dy_tip + dz_tip * dz_tip)

    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    ratio = d_tips / d_mcp
    cross_prod = dx_tip * dx_mcp

    idx_vx = landmarks[8].x - landmarks[6].x
    idx_vy = landmarks[8].y - landmarks[6].y
    idx_vz = landmarks[8].z - landmarks[6].z
    mag_idx = max(math.sqrt(idx_vx * idx_vx + idx_vy * idx_vy + idx_vz * idx_vz), 1e-6)

    mid_vx = landmarks[12].x - landmarks[10].x
    mid_vy = landmarks[12].y - landmarks[10].y
    mid_vz = landmarks[12].z - landmarks[10].z
    mag_mid = max(math.sqrt(mid_vx * mid_vx + mid_vy * mid_vy + mid_vz * mid_vz), 1e-6)

    cos_spread = (idx_vx * mid_vx + idx_vy * mid_vy + idx_vz * mid_vz) / (mag_idx * mag_mid)

    if cross_prod < 0 or ratio <= 0.65:
        return "R"
    elif ratio >= 1.25 or (cos_spread < 0.96 and ratio >= 1.05):
        return "V"
    else:
        return "U"


def disambiguate_e_o(landmarks):
    """
    Separates E and O:
    - E: Fingertips are curled tightly against knuckle/palm (idx_ext < 2.30).
    - O: Fingers arch forward forming an open circular loop (idx_ext >= 2.30).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx = landmarks[8].x - landmarks[5].x
    dy = landmarks[8].y - landmarks[5].y
    dz = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx * dx + dy * dy + dz * dz) / d_mcp

    return "E" if idx_ext < 2.30 else "O"


def disambiguate_g_h(landmarks):
    """
    Separates G and H:
    - G: Only index finger extended horizontally; middle finger curled (mid_ext/idx_ext < 0.82).
    - H: Both index and middle fingers extended parallel together (mid_ext/idx_ext >= 0.82, d_tips < 1.45, mid_ext >= 1.50).
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx_idx = landmarks[8].x - landmarks[5].x
    dy_idx = landmarks[8].y - landmarks[5].y
    dz_idx = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx_idx * dx_idx + dy_idx * dy_idx + dz_idx * dz_idx) / d_mcp

    dx_mid = landmarks[12].x - landmarks[9].x
    dy_mid = landmarks[12].y - landmarks[9].y
    dz_mid = landmarks[12].z - landmarks[9].z
    mid_ext = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid) / d_mcp

    dx_tips = landmarks[8].x - landmarks[12].x
    dy_tips = landmarks[8].y - landmarks[12].y
    dz_tips = landmarks[8].z - landmarks[12].z
    d_tips = math.sqrt(dx_tips * dx_tips + dy_tips * dy_tips + dz_tips * dz_tips) / d_mcp

    if (mid_ext / max(idx_ext, 1e-6) >= 0.82) and (d_tips < 1.45) and (mid_ext >= 1.50):
        return "H"
    return "G"


def disambiguate_d_l(landmarks):
    """
    Separates D and L:
    - Both have index finger extended straight up (idx_ext >= 1.45, index tip highest).
    - L: Thumb extended outward laterally (90 degrees), away from index knuckle and curled fingers.
    - D: Thumb curled inward touching or closing loop against middle/ring fingertip.
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx_tm = landmarks[4].x - landmarks[12].x
    dy_tm = landmarks[4].y - landmarks[12].y
    dz_tm = landmarks[4].z - landmarks[12].z
    thumb_mid = math.sqrt(dx_tm * dx_tm + dy_tm * dy_tm + dz_tm * dz_tm) / d_mcp

    dx_tr = landmarks[4].x - landmarks[16].x
    dy_tr = landmarks[4].y - landmarks[16].y
    dz_tr = landmarks[4].z - landmarks[16].z
    thumb_ring = math.sqrt(dx_tr * dx_tr + dy_tr * dy_tr + dz_tr * dz_tr) / d_mcp

    dx_thumb = abs(landmarks[4].x - landmarks[5].x) / d_mcp

    if dx_thumb >= 1.35 and thumb_mid >= 1.35 and thumb_ring >= 1.25:
        return "L", dx_thumb
    return "D", min(thumb_mid, thumb_ring)


def disambiguate_x_z(landmarks):
    """
    Separates X and Z:
    - In X: Closed fist with index finger hooked/bent like a pirate hook.
    - In Z: Index finger extended straight out pointing.
    """
    dx_mcp = landmarks[5].x - landmarks[9].x
    dy_mcp = landmarks[5].y - landmarks[9].y
    dz_mcp = landmarks[5].z - landmarks[9].z
    d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

    dx_idx = landmarks[8].x - landmarks[5].x
    dy_idx = landmarks[8].y - landmarks[5].y
    dz_idx = landmarks[8].z - landmarks[5].z
    idx_ext = math.sqrt(dx_idx * dx_idx + dy_idx * dy_idx + dz_idx * dz_idx) / d_mcp

    u_x = landmarks[6].x - landmarks[5].x
    u_y = landmarks[6].y - landmarks[5].y
    u_z = landmarks[6].z - landmarks[5].z
    mag_u = max(math.sqrt(u_x * u_x + u_y * u_y + u_z * u_z), 1e-6)

    v_x = landmarks[8].x - landmarks[6].x
    v_y = landmarks[8].y - landmarks[6].y
    v_z = landmarks[8].z - landmarks[6].z
    mag_v = max(math.sqrt(v_x * v_x + v_y * v_y + v_z * v_z), 1e-6)

    cos_hook = (u_x * v_x + u_y * v_y + u_z * v_z) / (mag_u * mag_v)

    if cos_hook < 0.65 or idx_ext < 1.45:
        return "X", cos_hook
    else:
        return "Z", cos_hook


def detect_silent_z_stroke(index_history):
    """
    Robust ASL 'Z' dynamic trajectory recognition:
    - Analyzes index fingertip (Landmark 8) over recent history.
    - Requires horizontal stroke -> diagonal down stroke -> horizontal stroke.
    - Rejects stationary hand tremors and non-Z motions.
    """
    if len(index_history) < 8:
        return False

    pts = [(pt[1], pt[2]) for pt in index_history]
    ts = [pt[0] for pt in index_history]

    dt = ts[-1] - ts[0]
    if dt < 0.15 or dt > 1.6:
        return False

    arc_length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]) for i in range(1, len(pts)))
    if arc_length < 0.07:
        return False

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    dx_span = max(xs) - min(xs)
    dy_span = max(ys) - min(ys)

    if dx_span < 0.035 or dy_span < 0.035:
        return False

    n = len(pts)
    s_xs = [(xs[max(0, i - 1)] + xs[i] + xs[min(n - 1, i + 1)]) / 3.0 for i in range(n)]

    best_zigzag = False
    for s1 in [int(n * 0.28), int(n * 0.33), int(n * 0.38)]:
        for s2 in [int(n * 0.62), int(n * 0.67), int(n * 0.72)]:
            if s1 < s2:
                d1 = s_xs[s1] - s_xs[0]
                d2 = s_xs[s2] - s_xs[s1]
                d3 = s_xs[-1] - s_xs[s2]
                if (d1 * d2 < -0.0002) and (d2 * d3 < -0.0002) and (abs(d1) >= 0.02 and abs(d2) >= 0.02 and abs(d3) >= 0.02):
                    best_zigzag = True
                    break
        if best_zigzag:
            break

    downward = (max(ys[n // 2:]) - min(ys[:n // 2])) >= 0.025
    return best_zigzag and downward


def detect_silent_j_hook(pinky_history):
    """
    Robust ASL 'J' trajectory recognition:
    - Analyzes the pinky tip (Landmark 20) over recent history.
    - Requires a distinct downward plunge followed by a sideways and upward hook.
    - Stationary jitter, straight downward drops, and waving are rejected.
    """
    if len(pinky_history) < 8:
        return False

    pts = [(pt[1], pt[2]) for pt in pinky_history]
    ts = [pt[0] for pt in pinky_history]

    dt_total = ts[-1] - ts[0]
    if dt_total < 0.12:
        return False

    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]

    trough_idx = max(range(len(ys)), key=lambda i: ys[i])

    if trough_idx >= len(ys) - 1 or trough_idx < 3:
        return False

    y_trough = ys[trough_idx]
    y_start = min(ys[:trough_idx])
    downward_stroke = (y_trough - y_start) >= 0.045

    y_after = ys[-1]
    hook_up = (y_trough - y_after) >= 0.012

    x_after = xs[-1]
    x_trough = xs[trough_idx]
    hook_sideways = abs(x_after - x_trough) >= 0.016 or (max(xs[trough_idx:]) - min(xs[trough_idx:])) >= 0.016

    start_idx = ys.index(y_start)
    stroke_pts = pts[start_idx:]
    arc_length = sum(math.hypot(stroke_pts[i][0] - stroke_pts[i - 1][0], stroke_pts[i][1] - stroke_pts[i - 1][1]) for i in range(1, len(stroke_pts)))

    return downward_stroke and hook_up and hook_sideways and (arc_length >= 0.055)


class NativeInferenceWorker(QThread):
    """
    Dedicated QThread that runs the high-performance computer-vision loop
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
            if isinstance(loaded, dict):
                static_clf = loaded.get("static_model", loaded)
                classes = list(loaded.get("classes", getattr(static_clf, "classes_", [])))
            else:
                static_clf = loaded
                classes = list(getattr(static_clf, "classes_", []))
            static_clf.n_jobs = 1
        except Exception as e:
            self.error_occurred.emit(f"Failed to load classifier: {str(e)}")
            self._is_running = False
            self.engine_stopped.emit()
            return

        # Pre-resolve class indices for ultra-fast frame lookup
        r_idx = classes.index('R') if 'R' in classes else None
        u_idx = classes.index('U') if 'U' in classes else None
        v_idx = classes.index('V') if 'V' in classes else None
        e_idx = classes.index('E') if 'E' in classes else None
        o_idx = classes.index('O') if 'O' in classes else None
        g_idx = classes.index('G') if 'G' in classes else None
        h_idx = classes.index('H') if 'H' in classes else None
        d_idx = classes.index('D') if 'D' in classes else None
        l_idx = classes.index('L') if 'L' in classes else None
        x_idx = classes.index('X') if 'X' in classes else None
        z_idx = classes.index('Z') if 'Z' in classes else None
        j_idx = classes.index('J') if 'J' in classes else None
        i_idx = classes.index('I') if 'I' in classes else None
        k_idx = classes.index('K') if 'K' in classes else None

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

        # Safe virtual camera reference
        try:
            from app.core.virtual_cam_service import VirtualCameraService
            vcam = VirtualCameraService.get_instance()
        except Exception:
            vcam = None

        # Inference loop state
        prediction_buffer = deque(maxlen=6)
        pinky_history = deque(maxlen=24)
        index_history = deque(maxlen=24)

        z_hold_until = 0.0
        z_cooldown_until = 0.0
        j_hold_until = 0.0
        j_cooldown_until = 0.0

        last_timestamp_ms = -1
        prev_frame_time = time.time()
        fps_smooth = 0.0
        smooth_conf = 0.0

        # 4. Main Inference & Rendering Loop
        while self._is_running:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            curr_frame_time = time.time()
            dt = curr_frame_time - prev_frame_time
            prev_frame_time = curr_frame_time
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

                # Anatomical scale reference: palm width (Knuckle 5 to Knuckle 9)
                dx_mcp = landmarks[5].x - landmarks[9].x
                dy_mcp = landmarks[5].y - landmarks[9].y
                dz_mcp = landmarks[5].z - landmarks[9].z
                d_mcp = max(math.sqrt(dx_mcp * dx_mcp + dy_mcp * dy_mcp + dz_mcp * dz_mcp), 1e-6)

                # Index extension (Knuckle 5 to Tip 8)
                dx = landmarks[8].x - landmarks[5].x
                dy = landmarks[8].y - landmarks[5].y
                dz = landmarks[8].z - landmarks[5].z
                idx_ext = math.sqrt(dx * dx + dy * dy + dz * dz) / d_mcp

                # Middle extension (Knuckle 9 to Tip 12)
                dx_mid = landmarks[12].x - landmarks[9].x
                dy_mid = landmarks[12].y - landmarks[9].y
                dz_mid = landmarks[12].z - landmarks[9].z
                mid_ext = math.sqrt(dx_mid * dx_mid + dy_mid * dy_mid + dz_mid * dz_mid) / d_mcp

                # Pinky extension (Knuckle 17 to Tip 20)
                dx_p = landmarks[20].x - landmarks[17].x
                dy_p = landmarks[20].y - landmarks[17].y
                dz_p = landmarks[20].z - landmarks[17].z
                pinky_ext = math.sqrt(dx_p * dx_p + dy_p * dy_p + dz_p * dz_p) / d_mcp

                # Thumb spread from index knuckle
                dx_t = landmarks[4].x - landmarks[5].x
                dy_t = landmarks[4].y - landmarks[5].y
                dz_t = landmarks[4].z - landmarks[5].z
                thumb_idx = math.sqrt(dx_t * dx_t + dy_t * dy_t + dz_t * dz_t) / d_mcp

                # Extract features for classifier
                features = extract_features(landmarks)

                try:
                    probs = static_clf.predict_proba([features])[0]
                except Exception:
                    probs = np.zeros(len(classes), dtype=np.float32)

                # Group probabilities
                ruv_group_prob = 0.0
                if r_idx is not None and u_idx is not None and v_idx is not None:
                    ruv_group_prob = probs[r_idx] + probs[u_idx] + probs[v_idx]

                eo_group_prob = 0.0
                if e_idx is not None and o_idx is not None:
                    eo_group_prob = probs[e_idx] + probs[o_idx]

                gh_group_prob = 0.0
                if g_idx is not None and h_idx is not None:
                    gh_group_prob = probs[g_idx] + probs[h_idx]

                dl_group_prob = 0.0
                if d_idx is not None and l_idx is not None:
                    dl_group_prob = probs[d_idx] + probs[l_idx]

                xz_group_prob = 0.0
                if x_idx is not None and z_idx is not None:
                    xz_group_prob = probs[x_idx] + probs[z_idx]

                # Check thumb geometry
                lm4 = landmarks[4]
                lm5 = landmarks[5]
                lm9 = landmarks[9]
                thumb_up = (lm4.y < lm5.y or lm4.y < lm9.y)

                is_index_extended = (idx_ext >= 1.45) and (landmarks[8].y < landmarks[5].y)
                is_other_fingers_down = (landmarks[8].y < landmarks[12].y - 0.03) and (landmarks[8].y < landmarks[16].y - 0.03) and (landmarks[8].y < landmarks[20].y - 0.03)
                is_single_index_up = is_index_extended and is_other_fingers_down and (idx_ext > pinky_ext + 0.25)

                if ruv_group_prob >= 0.40 and not (thumb_up and (k_idx is not None and probs[k_idx] > 0.35)):
                    raw_pred = disambiguate_r_u_v(landmarks)
                    conf = ruv_group_prob * 100.0
                elif eo_group_prob >= 0.45:
                    raw_pred = disambiguate_e_o(landmarks)
                    conf = eo_group_prob * 100.0
                elif gh_group_prob >= 0.45 and idx_ext >= 1.55:
                    raw_pred = disambiguate_g_h(landmarks)
                    conf = gh_group_prob * 100.0
                elif is_single_index_up:
                    raw_pred, geom_val = disambiguate_d_l(landmarks)
                    if raw_pred == "L":
                        spread_score = min(1.0, max(0.0, (geom_val - 1.25) / 0.8))
                        base_p = max(dl_group_prob, probs[l_idx] if l_idx is not None else 0.5)
                        conf = min(98.5, max(82.0, base_p * 35.0 + spread_score * 30.0 + 40.0))
                    else:
                        touch_score = min(1.0, max(0.0, (1.45 - geom_val) / 0.8))
                        base_p = max(dl_group_prob, probs[d_idx] if d_idx is not None else 0.5)
                        conf = min(98.5, max(82.0, base_p * 35.0 + touch_score * 30.0 + 40.0))
                elif not is_single_index_up and (xz_group_prob >= 0.25 or (probs.argmax() in [x_idx, z_idx]) or (x_idx is not None and probs[x_idx] > 0.15) or (z_idx is not None and probs[z_idx] > 0.15)):
                    raw_pred, geom_val = disambiguate_x_z(landmarks)
                    if raw_pred == "X":
                        hook_score = min(1.0, max(0.0, (0.70 - geom_val) / 1.2))
                        conf = min(98.5, max(75.0, xz_group_prob * 80.0 + hook_score * 18.0))
                    else:
                        ext_score = min(1.0, max(0.0, (geom_val - 0.50) / 0.5))
                        conf = min(98.5, max(75.0, xz_group_prob * 80.0 + ext_score * 18.0))
                else:
                    max_idx = probs.argmax()
                    conf = probs[max_idx] * 100.0
                    raw_pred = classes[max_idx] if conf >= 30.0 else "Uncertain"

                # Dynamic 'J' vs Static 'I'
                is_pinky_extended = (pinky_ext >= 1.45) and (landmarks[20].y < landmarks[18].y) and (pinky_ext > idx_ext + 0.20) and (pinky_ext > mid_ext + 0.20)
                is_thumb_out = (thumb_idx >= 1.40)
                is_pure_i_or_j = is_pinky_extended and (not is_thumb_out)

                if is_pure_i_or_j or len(pinky_history) > 0:
                    pinky_history.append((now, landmarks[20].x, landmarks[20].y))
                    if detect_silent_j_hook(pinky_history):
                        j_hold_until = now + 1.5
                        j_cooldown_until = j_hold_until + 0.7
                        pinky_history.clear()
                        prediction_buffer.clear()
                        prediction_buffer.append("J")
                else:
                    pinky_history.clear()

                # Dynamic 'Z'
                is_pointing_or_drawing = (idx_ext >= 1.20) and not is_pure_i_or_j
                if is_pointing_or_drawing or len(index_history) > 0:
                    index_history.append((now, landmarks[8].x, landmarks[8].y))
                    if detect_silent_z_stroke(index_history):
                        z_hold_until = now + 1.5
                        z_cooldown_until = z_hold_until + 0.6
                        index_history.clear()
                        prediction_buffer.clear()
                        prediction_buffer.append("Z")
                else:
                    index_history.clear()

                if now < z_hold_until:
                    predicted_label = "Z"
                    z_prob = (probs[z_idx] + probs[x_idx]) * 100.0 if (z_idx is not None and x_idx is not None) else conf
                    active_conf = min(98.2, max(85.0, 84.0 + 12.0 * min(idx_ext / 2.0, 1.0) + (z_prob * 0.04)))
                elif now < z_cooldown_until and (predicted_label == "Z" or raw_pred in ["X", "Uncertain"]):
                    predicted_label = "Z"
                    active_conf = 88.0 + 5.0 * min(idx_ext / 2.0, 1.0)
                elif now < j_hold_until:
                    predicted_label = "J"
                    j_conf = (probs[j_idx] + probs[i_idx]) * 100.0 if (j_idx is not None and i_idx is not None) else conf
                    active_conf = min(98.2, max(85.0, 84.0 + 12.0 * min(pinky_ext / 2.0, 1.0) + (j_conf * 0.04)))
                elif now < j_cooldown_until and (predicted_label == "J" or raw_pred in ["G", "H", "Uncertain"] or idx_ext < 1.55):
                    predicted_label = "J"
                    active_conf = 88.0 + 4.0 * min(pinky_ext / 2.0, 1.0)
                elif is_pure_i_or_j:
                    predicted_label = "I"
                    i_conf = (probs[i_idx] + probs[j_idx]) * 100.0 if (i_idx is not None and j_idx is not None) else conf
                    active_conf = min(98.5, max(82.0, i_conf * 0.85 + 14.0 * min(pinky_ext / 2.0, 1.0)))
                else:
                    predicted_label = raw_pred
                    active_conf = conf

                smooth_conf = smooth_conf * 0.75 + active_conf * 0.25

                if predicted_label not in ["Uncertain", "No Hand"]:
                    prediction_buffer.append(predicted_label)
                    stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                else:
                    stable_prediction = predicted_label
                confidence_pct = smooth_conf

            else:
                prediction_buffer.clear()
                pinky_history.clear()
                index_history.clear()
                stable_prediction = "No Hand"
                confidence_pct = 0.0

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
                cv2.putText(frame, str(stable_prediction), (c1_x, val_y), cv2.FONT_HERSHEY_SIMPLEX, val_scale, letter_color, 2)

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
