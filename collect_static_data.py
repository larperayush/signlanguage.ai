import csv
import os
import sys
import subprocess
import time
import math
import tkinter as tk
from tkinter import ttk

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WINDOW_NAME = "Static ASL Data Collection"
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "hand_landmarker.task")
DATA_FILE = os.path.join(SCRIPT_DIR, "static_signs.csv")

LETTERS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y"
]

TARGET_SAMPLES = 100
HEADER = ["label"] + [f"feature_{i}" for i in range(63)]

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def get_camera_names():
    """Fetch real camera names from Windows DirectShow in exact index order."""
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        print("Installing 'pygrabber' to detect real camera names...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pygrabber", "-q"])
            from pygrabber.dshow_graph import FilterGraph
        except Exception as e:
            print(f"Could not auto-install pygrabber ({e}). Falling back to generic names.")
            return [f"Camera Index {i}" for i in range(10)]

    try:
        graph = FilterGraph()
        devices = graph.get_input_devices()
        if devices:
            return devices
    except Exception as e:
        print(f"Warning reading DirectShow camera names: {e}")

    return [f"Camera Index {i}" for i in range(10)]


def select_camera_popup(camera_names, current_index=0):
    """Open a popup window letting the user pick a camera by its real name."""
    selected = {"index": None}

    root = tk.Tk()
    root.title("Select Camera")
    root.geometry("420x340")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # Center the popup on screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = (sw // 2) - (420 // 2)
    y = (sh // 2) - (340 // 2)
    root.geometry(f"420x340+{x}+{y}")

    lbl = tk.Label(root, text="Select Camera Device:", font=("Arial", 12, "bold"))
    lbl.pack(pady=(15, 5))

    sub_lbl = tk.Label(root, text="Double-click a camera or click 'Use Selected Camera'", font=("Arial", 9), fg="#555")
    sub_lbl.pack(pady=(0, 10))

    frame = tk.Frame(root)
    frame.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL)
    listbox = tk.Listbox(
        frame,
        font=("Arial", 11),
        yscrollcommand=scrollbar.set,
        selectmode=tk.SINGLE,
        activestyle="none"
    )
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    for idx, name in enumerate(camera_names):
        listbox.insert(tk.END, f"[{idx}]  {name}")

    if 0 <= current_index < len(camera_names):
        listbox.selection_set(current_index)
        listbox.see(current_index)
    elif len(camera_names) > 0:
        listbox.selection_set(0)

    def confirm(event=None):
        sel = listbox.curselection()
        if sel:
            selected["index"] = int(sel[0])
            root.destroy()

    listbox.bind("<Double-Button-1>", confirm)
    listbox.bind("<Return>", confirm)

    btn = tk.Button(
        root,
        text="Use Selected Camera",
        font=("Arial", 11, "bold"),
        bg="#4CAF50",
        fg="white",
        padx=15,
        pady=6,
        command=confirm
    )
    btn.pack(pady=15)

    root.mainloop()
    return selected["index"]


def get_total_samples():
    return sum(sample_counts.values())


def get_features(landmarks):
    # Convert 21 hand landmarks into 63 wrist-relative normalized features.
    wrist = landmarks[0]
    features = []

    for point in landmarks:
        features.extend([
            point.x - wrist.x,
            point.y - wrist.y,
            point.z - wrist.z
        ])

    scale = max(max(abs(value) for value in features), 1e-6)
    return [value / scale for value in features]


def save_sample(label, landmarks):
    # Append one labeled hand-landmark sample to the CSV.
    if landmarks is None:
        print("No hand detected. Sample not saved.")
        return

    if sample_counts[label] >= TARGET_SAMPLES:
        print(f"{label} already has {TARGET_SAMPLES} samples.")
        return

    features = get_features(landmarks)

    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow([label] + features)

    sample_counts[label] += 1

    print(
        f"Saved {label}: {sample_counts[label]}/{TARGET_SAMPLES} | "
        f"Total dataset: {get_total_samples()}"
    )

    if sample_counts[label] == TARGET_SAMPLES:
        print(f"Target reached for {label}!")


def delete_last_sample(label):
    """Remove the most recent row for the current label from static_signs.csv."""
    if sample_counts[label] <= 0 or not os.path.exists(DATA_FILE):
        print(f"No samples of {label} to delete.")
        return

    with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if len(rows) <= 1:
        return

    header = rows[0]
    data_rows = rows[1:]

    deleted = False
    for i in range(len(data_rows) - 1, -1, -1):
        if data_rows[i] and data_rows[i][0].strip().upper() == label:
            data_rows.pop(i)
            deleted = True
            break

    if deleted:
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data_rows)

        sample_counts[label] = max(0, sample_counts[label] - 1)
        print(
            f"Deleted last {label} sample -> Now {sample_counts[label]}/{TARGET_SAMPLES} | "
            f"Total dataset: {get_total_samples()}"
        )


def open_camera(camera_index=0, width=1920, height=1080, fps=30):
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]

    for backend_name, backend in backends:
        if backend == cv2.CAP_ANY:
            camera = cv2.VideoCapture(camera_index)
        else:
            camera = cv2.VideoCapture(camera_index, backend)

        if not camera.isOpened():
            camera.release()
            continue

        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, fps)

        success, test_frame = camera.read()
        if success and test_frame is not None and test_frame.size > 0:
            actual_h, actual_w = test_frame.shape[:2]
            reported_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            reported_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_fps = camera.get(cv2.CAP_PROP_FPS)

            print(f"Camera [{camera_index}] opened with backend: {backend_name}")
            print(f"Reported capture mode: {reported_w}x{reported_h} @ {reported_fps:.1f} FPS")
            print(f"Actual first frame resolution: {actual_w}x{actual_h}")
            return camera, backend_name

        camera.release()

    raise RuntimeError(f"Could not open camera index {camera_index}. Check camera permissions or try another camera.")


def main():
    global sample_counts

    # Check that the MediaPipe model exists.
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Make sure models/hand_landmarker.task exists."
        )

    # Load existing dataset counts without deleting existing samples.
    sample_counts = {letter: 0 for letter in LETTERS}

    if os.path.exists(DATA_FILE) and os.path.getsize(DATA_FILE) > 0:
        with open(DATA_FILE, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            existing_header = next(reader, None)

            if existing_header != HEADER:
                raise ValueError(
                    f"Unexpected CSV format in {DATA_FILE}. "
                    "Existing data has not been modified."
                )

            for line_number, row in enumerate(reader, start=2):
                if len(row) != 64:
                    print(f"Warning: Skipping malformed row {line_number}.")
                    continue

                label = row[0]

                if label not in sample_counts:
                    print(f"Warning: Unknown label {label!r} on row {line_number}.")
                    continue

                try:
                    features = [float(value) for value in row[1:]]
                except ValueError:
                    print(f"Warning: Invalid numeric data on row {line_number}.")
                    continue

                if not all(math.isfinite(value) for value in features):
                    print(f"Warning: Non-finite feature on row {line_number}.")
                    continue

                sample_counts[label] += 1
    else:
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(HEADER)

    # Detect real camera names and show selector popup at startup.
    camera_names = get_camera_names()
    print("\nDetected Cameras:")
    for idx, name in enumerate(camera_names):
        print(f"  [{idx}] {name}")

    chosen_index = select_camera_popup(camera_names, current_index=0)
    if chosen_index is None:
        print("No camera selected. Exiting.")
        return

    current_cam_index = chosen_index
    current_cam_name = camera_names[current_cam_index] if current_cam_index < len(camera_names) else f"Camera {current_cam_index}"

    # Start MediaPipe.
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    try:
        camera, backend_name = open_camera(camera_index=current_cam_index, width=1920, height=1080, fps=30)
    except Exception:
        detector.close()
        raise

    # Normal window: no forced fullscreen scaling.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    current_label = "A"
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    print("\n==========================================")
    print("      STATIC ASL DATA COLLECTION")
    print("==========================================")
    print("Letters: A-I and K-Y (J and Z excluded)")
    print(f"Maximum samples per letter: {TARGET_SAMPLES}")
    print()
    print("A-I, K-Y  : Select a letter")
    print("SPACE     : Save one sample")
    print("BACKSPACE : Delete last sample of current letter")
    print("TAB       : Switch Camera (opens camera popup)")
    print("Q         : Quit")
    print("Click the webcam window before using keys.")
    print("==========================================\n")

    try:
        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Could not read webcam frame. Press TAB to switch camera or Q to quit.")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int((time.perf_counter() - start_time) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = detector.detect_for_video(mp_image, timestamp_ms)
            landmarks = None

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

                for start, end in CONNECTIONS:
                    p1 = landmarks[start]
                    p2 = landmarks[end]

                    x1 = int(p1.x * frame_width)
                    y1 = int(p1.y * frame_height)
                    x2 = int(p2.x * frame_width)
                    y2 = int(p2.y * frame_height)

                    cv2.line(
                        frame, (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )

                for point in landmarks:
                    x = int(point.x * frame_width)
                    y = int(point.y * frame_height)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            count = sample_counts[current_label]

            cv2.putText(
                frame, f"Selected: {current_label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2
            )
            cv2.putText(
                frame, f"Samples: {count}/{TARGET_SAMPLES}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"Total: {get_total_samples()}/{len(LETTERS) * TARGET_SAMPLES}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )
            cv2.putText(
                frame,
                f"Cam [{current_cam_index}]: {current_cam_name} ({frame_width}x{frame_height})",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 255, 200), 2
            )
            cv2.putText(
                frame, "A-Y: Select | SPACE: Save | BACKSPACE: Undo | TAB: Switch Cam | Q: Quit",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1
            )

            if landmarks is None:
                status = "NO HAND DETECTED"
                status_color = (0, 0, 255)
            else:
                status = "HAND DETECTED"
                status_color = (0, 255, 0)

            cv2.putText(
                frame, status,
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, status_color, 2
            )

            if count >= TARGET_SAMPLES:
                cv2.putText(
                    frame, "TARGET REACHED",
                    (10, 210), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                break
            elif key == 32:  # SPACE
                save_sample(current_label, landmarks)
            elif key == 8:   # BACKSPACE
                delete_last_sample(current_label)
            elif key == 9:   # TAB -> Switch camera
                camera_names = get_camera_names()
                new_index = select_camera_popup(camera_names, current_index=current_cam_index)
                if new_index is not None and new_index != current_cam_index:
                    print(f"Switching to camera [{new_index}]...")
                    camera.release()
                    try:
                        camera, backend_name = open_camera(camera_index=new_index, width=1920, height=1080, fps=30)
                        current_cam_index = new_index
                        current_cam_name = camera_names[current_cam_index] if current_cam_index < len(camera_names) else f"Camera {current_cam_index}"
                    except Exception as e:
                        print(f"Failed to open camera [{new_index}]: {e}. Re-opening previous camera...")
                        camera, backend_name = open_camera(camera_index=current_cam_index, width=1920, height=1080, fps=30)
            elif key != 255:
                try:
                    pressed_key = chr(key).upper()
                except ValueError:
                    continue

                if pressed_key in LETTERS:
                    current_label = pressed_key
                    print(
                        f"Selected {current_label}: "
                        f"{sample_counts[current_label]}/{TARGET_SAMPLES}"
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()
        detector.close()

        print("\n==========================================")
        print("           COLLECTION SUMMARY")
        print("==========================================")

        for letter in LETTERS:
            print(f"{letter}: {sample_counts[letter]}/{TARGET_SAMPLES}")

        print(f"\nTotal samples: {get_total_samples()}")
        print(f"Dataset: {DATA_FILE}")
        print("==========================================")


sample_counts = {letter: 0 for letter in LETTERS}
if __name__ == "__main__":
    main()