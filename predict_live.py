import os
import time
import math
from collections import deque
import cv2
import joblib
import mediapipe as mp

# MediaPipe Task API setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "models", "asl_classifier.joblib")
WINDOW_NAME = "Live ASL Recognition"

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def extract_features(landmarks):
    # Convert 21 hand landmarks into 63 wrist-relative normalized features
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.append(lm.x - wrist.x)
        raw_coords.append(lm.y - wrist.y)
        raw_coords.append(lm.z - wrist.z)

    max_abs = max(abs(val) for val in raw_coords)
    scale = max(max_abs, 1e-6)
    return [val / scale for val in raw_coords]


def open_camera():
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]
    camera = None
    chosen_backend_name = "None"

    for name, backend in backends:
        cam = cv2.VideoCapture(0, backend)
        if cam.isOpened():
            # FIXED: Set Fourcc properly with property ID and value
            fourcc_val = cv2.VideoWriter_fourcc(*"MJPG")
            cam.set(cv2.CAP_PROP_FOURCC, fourcc_val)
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cam.set(cv2.CAP_PROP_FPS, 30)

            ret, test_frame = cam.read()
            if ret and test_frame is not None:
                camera = cam
                chosen_backend_name = name
                break
            else:
                cam.release()

    if camera is None:
        raise RuntimeError("Could not open webcam using any supported backend.")

    return camera, chosen_backend_name


def draw_landmarks(image, landmarks, width, height):
    points = []
    for lm in landmarks:
        px = int(lm.x * width)
        py = int(lm.y * height)
        points.append((px, py))

    for start_idx, end_idx in CONNECTIONS:
        cv2.line(image, points[start_idx], points[end_idx], (0, 255, 0), 2)

    for px, py in points:
        cv2.circle(image, (px, py), 4, (0, 0, 255), -1)


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MediaPipe model missing at: {MODEL_PATH}")
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Trained classifier missing at: {CLASSIFIER_PATH}\nRun train_model.py first!")

    print("Loading trained ASL classifier...")
    classifier = joblib.load(CLASSIFIER_PATH)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    detector = HandLandmarker.create_from_options(options)
    camera, backend_name = open_camera()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    last_timestamp_ms = 0
    prediction_buffer = deque(maxlen=10)  # Smoothing window to prevent flickering
    current_word = ""

    print("\n--- Live ASL Recognition Started ---")
    print("SPACEBAR : Add current stable letter to sentence")
    print("BACKSPACE: Delete last letter")
    print("C        : Clear sentence")
    print("Q        : Quit\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            result = detector.detect_for_video(mp_image, current_timestamp_ms)

            predicted_label = "No Hand"
            confidence_pct = 0.0

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                landmarks = result.hand_landmarks[0]
                draw_landmarks(frame, landmarks, w, h)

                features = extract_features(landmarks)
                X_input = [features]

                # Predict using the trained model
                probs = classifier.predict_proba(X_input)[0]
                max_prob_idx = probs.argmax()
                predicted_label = classifier.classes_[max_prob_idx]
                confidence_pct = probs[max_prob_idx] * 100

                # Only accept prediction if confidence is reasonable (>60%)
                if confidence_pct < 60.0:
                    predicted_label = "Uncertain"

            # Apply smoothing buffer
            prediction_buffer.append(predicted_label)
            if len(prediction_buffer) > 0:
                stable_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
            else:
                stable_prediction = "No Hand"

            # UI Display Box (Top Left)
            cv2.rectangle(frame, (10, 10), (450, 140), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, 10), (450, 140), (100, 100, 100), 1)

            cv2.putText(frame, f"Prediction: {stable_prediction}", (25, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {confidence_pct:.1f}%", (25, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Backend: {backend_name} ({w}x{h})", (25, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 255, 150), 1)

            # Word Builder Box (Bottom)
            cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (20, 20, 20), -1)
            cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (100, 100, 100), 1)
            cv2.putText(frame, f"Sentence: {current_word}", (25, h - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == 32:  # SPACEBAR adds stable letter
                if stable_prediction not in ["No Hand", "Uncertain"]:
                    current_word += stable_prediction
            elif key == 8 or key == 127:  # BACKSPACE deletes last letter
                current_word = current_word[:-1]
            elif key == ord('c') or key == ord('C'):  # Clear sentence
                current_word = ""

    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()