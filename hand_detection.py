import cv2
import mediapipe as mp
import os
import time

# Locate the pretrained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

# Open Iriun Webcam. Try Windows camera backends in turn.
camera = cv2.VideoCapture(8, cv2.CAP_DSHOW)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0, cv2.CAP_MSMF)

if not camera.isOpened():
    camera.release()
    camera = cv2.VideoCapture(0)

if not camera.isOpened():
    camera.release()
    raise RuntimeError("Could not open Iriun Webcam. Check Iriun and camera permissions.")

# Request MJPEG first; some virtual cameras expose higher resolutions with it.
camera.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
camera.set(cv2.CAP_PROP_FPS, 30)

print("Camera backend:", camera.getBackendName())
print("Camera FPS reported:", camera.get(cv2.CAP_PROP_FPS))
print(
    "Requested resolution: 1920x1080 | Reported capture resolution:",
    int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
    "x",
    int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
print("The resolution shown on the video overlay comes from the actual frames.")

fps = 0.0
previous_time = time.perf_counter()
inference_ms = 0.0
last_timestamp_ms = 0

try:
    with HandLandmarker.create_from_options(options) as detector:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read from webcam.")
                break

            frame = cv2.flip(frame, 1)

            # Actual dimensions of the frame received from Iriun
            height, width = frame.shape[:2]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # MediaPipe VIDEO timestamps must increase on every frame.
            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            inference_start = time.perf_counter()
            result = detector.detect_for_video(mp_image, timestamp_ms)
            inference_ms = (time.perf_counter() - inference_start) * 1000

            # Draw landmarks and connections
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    points = []

                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                    for connection in connections:
                        start = connection.start
                        end = connection.end
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (255, 0, 0),
                            2
                        )

                status = "Hand detected"
                status_color = (0, 255, 0)
            else:
                status = "No hand detected"
                status_color = (0, 0, 255)

            # Calculate overall loop FPS
            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                if fps == 0:
                    fps = instant_fps
                else:
                    fps = 0.85 * fps + 0.15 * instant_fps

            previous_time = current_time

            # Display diagnostics
            cv2.putText(
                frame, f"Loop FPS: {fps:.1f}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"Inference: {inference_ms:.1f} ms",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 0), 2
            )
            cv2.putText(
                frame, f"Resolution: {width}x{height}",
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2
            )
            cv2.putText(
                frame, status,
                (15, 125), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, status_color, 2
            )

            cv2.imshow("MediaPipe Hand Detection - Diagnostics", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    camera.release()
    cv2.destroyAllWindows()
