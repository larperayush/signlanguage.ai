import os
import csv
import time
import math
import cv2
import mediapipe as mp
from flask import Flask, render_template_string, Response, request, jsonify

app = Flask(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# Global state variables
current_camera_index = 0
camera = None
detector = None
last_timestamp_ms = 0
current_letter = "A"
latest_landmarks = None

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def init_detector():
    global detector
    if detector is not None:
        try:
            detector.close()
        except Exception:
            pass
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )
    detector = HandLandmarker.create_from_options(options)


def open_camera(index=0):
    global camera
    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    
    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]
    
    for name, backend in backends:
        try:
            cam = cv2.VideoCapture(index, backend)
            if cam.isOpened():
                fourcc_val = cv2.VideoWriter_fourcc(*"MJPG")
                cam.set(cv2.CAP_PROP_FOURCC, fourcc_val)
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, frame = cam.read()
                if ret and frame is not None:
                    camera = cam
                    print(f"[Camera] Opened index {index} using backend {name}")
                    return True
            cam.release()
        except Exception as e:
            print(f"[Camera Warning] Failed backend {name} on index {index}: {e}")
            
    return False


def ensure_csv():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def get_sample_counts():
    ensure_csv()
    counts = {letter: 0 for letter in LETTERS}
    try:
        with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header != HEADER:
                return counts
            for row in reader:
                if row and len(row) > 0:
                    lbl = row[0].strip().upper()
                    if lbl in counts:
                        counts[lbl] += 1
    except Exception as e:
        print(f"[CSV Read Error]: {e}")
    return counts


def extract_features(landmarks):
    wrist = landmarks[0]
    raw_coords = []
    for lm in landmarks:
        raw_coords.append(lm.x - wrist.x)
        raw_coords.append(lm.y - wrist.y)
        raw_coords.append(lm.z - wrist.z)
    max_abs = max(abs(val) for val in raw_coords)
    scale = max(max_abs, 1e-6)
    return [val / scale for val in raw_coords]


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ASL Remote Data Collection</title>
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
        .container { max-width: 900px; margin: auto; background: #1e1e1e; padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        .video-box { margin: 20px 0; }
        img { width: 100%; max-width: 800px; border: 3px solid #333; border-radius: 8px; }
        .controls { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin: 20px 0; align-items: center; }
        button, select { padding: 12px 20px; font-size: 16px; font-weight: bold; border: none; border-radius: 5px; cursor: pointer; background: #333; color: white; transition: 0.2s; }
        button:hover, select:hover { background: #555; }
        select { background: #2a2a2a; border: 2px solid #555; }
        button.save-btn { background: #4CAF50; font-size: 20px; padding: 15px 30px; }
        button.save-btn:hover { background: #45a049; }
        button.del-btn { background: #f44336; }
        button.del-btn:hover { background: #d32f2f; }
        .stats { font-size: 18px; margin: 15px 0; background: #2a2a2a; padding: 10px; border-radius: 5px; }
        .grid { display: grid; grid-template-columns: repeat(6, 1fr); gap: 8px; margin-top: 15px; }
        .letter-btn { padding: 10px; font-size: 16px; background: #444; }
        .letter-btn.selected { background: #ff9800; color: black; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASL Remote Data Collection</h1>
        <div class="stats">
            Selected Letter: <span id="lbl-display" style="color: #ff9800; font-weight: bold;">A</span> | 
            Count: <span id="count-display">0</span> / 100 | 
            Hand Status: <span id="status-display" style="color: red;">Checking...</span>
        </div>

        <div class="controls">
            <label for="camera-select">Camera Index:</label>
            <select id="camera-select" onchange="onCameraSelected()">
                <script>
                    for(let i=0; i<10; i++) {
                        document.write(`<option value="${i}">Camera Index ${i}</option>`);
                    }
                </script>
            </select>
        </div>

        <div class="video-box">
            <img src="{{ url_for('video_feed') }}" alt="Webcam Stream">
        </div>

        <div class="controls">
            <button class="save-btn" onclick="saveSample()">SAVE SAMPLE (SPACE)</button>
            <button class="del-btn" onclick="deleteLastSample()">DELETE LAST (BACKSPACE)</button>
        </div>

        <h3>Select Letter</h3>
        <div class="grid" id="letter-grid"></div>
    </div>

<script>
    const letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y'];
    let currentLetter = 'A';

    function onCameraSelected() {
        const select = document.getElementById('camera-select');
        const index = select.value;
        fetch('/change_camera', { 
            method: 'POST', 
            headers: {'Content-Type': 'application/json'}, 
            body: JSON.stringify({index: index}) 
        })
        .then(res => res.json())
        .then(data => console.log("Switched to camera index " + index));
    }

    function buildGrid() {
        const grid = document.getElementById('letter-grid');
        grid.innerHTML = '';
        letters.forEach(l => {
            const btn = document.createElement('button');
            btn.className = 'letter-btn' + (l === currentLetter ? ' selected' : '');
            btn.id = 'btn-' + l;
            btn.innerText = l;
            btn.onclick = () => setLetter(l);
            grid.appendChild(btn);
        });
    }

    function setLetter(l) {
        currentLetter = l;
        fetch('/set_letter', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({letter: l}) })
        .then(res => res.json())
        .then(data => updateUI(data));
    }

    function saveSample() {
        fetch('/save_sample', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if(data.success) {
                updateUI(data);
            } else {
                alert(data.message);
            }
        });
    }

    function deleteLastSample() {
        fetch('/delete_last', { method: 'POST' })
        .then(res => res.json())
        .then(data => updateUI(data));
    }

    function updateUI(data) {
        document.getElementById('lbl-display').innerText = data.current_letter;
        document.getElementById('count-display').innerText = data.current_count;
        const statusEl = document.getElementById('status-display');
        statusEl.innerText = data.hand_detected ? "HAND DETECTED" : "NO HAND";
        statusEl.style.color = data.hand_detected ? "#4CAF50" : "red";

        letters.forEach(l => {
            const btn = document.getElementById('btn-' + l);
            if(btn) {
                if(l === data.current_letter) {
                    btn.classList.add('selected');
                } else {
                    btn.classList.remove('selected');
                }
            }
        });
    }

    document.addEventListener('keydown', (e) => {
        if(e.code === 'Space') {
            e.preventDefault();
            saveSample();
        } else if(e.code === 'Backspace') {
            e.preventDefault();
            deleteLastSample();
        } else {
            const key = e.key.toUpperCase();
            if(letters.includes(key)) {
                setLetter(key);
            }
        }
    });

    setInterval(() => {
        fetch('/status').then(res => res.json()).then(data => updateUI(data));
    }, 1000);

    buildGrid();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


def generate_frames():
    global last_timestamp_ms, latest_landmarks
    while True:
        try:
            if camera is None or not camera.isOpened():
                time.sleep(0.5)
                continue
            
            success, frame = camera.read()
            if not success or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            current_timestamp_ms = int(time.time() * 1000)
            if current_timestamp_ms <= last_timestamp_ms:
                current_timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = current_timestamp_ms

            result = detector.detect_for_video(mp_image, current_timestamp_ms)
            latest_landmarks = None

            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                latest_landmarks = result.hand_landmarks[0]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in latest_landmarks]
                for start, end in CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
                for pt in points:
                    cv2.circle(frame, pt, 4, (0, 0, 255), -1)

            counts = get_sample_counts()
            current_count = counts.get(current_letter, 0)
            cv2.putText(frame, f"Letter: {current_letter} ({current_count}/100)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            status_txt = "HAND OK" if latest_landmarks else "NO HAND"
            status_col = (0, 255, 0) if latest_landmarks else (0, 0, 255)
            cv2.putText(frame, status_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"[Frame Generation Error]: {e}")
            time.sleep(0.5)


@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/status")
def status():
    counts = get_sample_counts()
    return jsonify({
        "current_letter": current_letter,
        "current_count": counts.get(current_letter, 0),
        "hand_detected": latest_landmarks is not None
    })


@app.route("/set_letter", methods=["POST"])
def set_letter_route():
    global current_letter
    data = request.get_json()
    if data and "letter" in data:
        current_letter = data["letter"].upper()
    return jsonify({
        "current_letter": current_letter,
        "current_count": get_sample_counts().get(current_letter, 0),
        "hand_detected": latest_landmarks is not None
    })


@app.route("/save_sample", methods=["POST"])
def save_sample_route():
    if latest_landmarks is None:
        return jsonify({"success": False, "message": "No hand detected! Sample not saved."})
    
    features = extract_features(latest_landmarks)
    ensure_csv()
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([current_letter] + features)
    
    counts = get_sample_counts()
    return jsonify({
        "success": True,
        "current_letter": current_letter,
        "current_count": counts.get(current_letter, 0),
        "hand_detected": True
    })


@app.route("/delete_last", methods=["POST"])
def delete_last_route():
    ensure_csv()
    rows = []
    try:
        with open(DATA_FILE, "r", newline="", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            if reader:
                header = reader[0]
                data_rows = reader[1:]
                for i in range(len(data_rows) - 1, -1, -1):
                    if data_rows[i][0].strip().upper() == current_letter:
                        data_rows.pop(i)
                        break
                rows = [header] + data_rows

        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except Exception as e:
        print(f"[Delete Error]: {e}")

    counts = get_sample_counts()
    return jsonify({
        "current_letter": current_letter,
        "current_count": counts.get(current_letter, 0),
        "hand_detected": latest_landmarks is not None
    })


@app.route("/change_camera", methods=["POST"])
def change_camera_route():
    global current_camera_index
    data = request.get_json()
    if data and "index" in data:
        new_index = int(data["index"])
        current_camera_index = new_index
        open_camera(current_camera_index)
    return jsonify({"success": True})


if __name__ == "__main__":
    ensure_csv()
    init_detector()
    if not open_camera(0):
        print("[Error] Could not open default camera!")
    
    print("\n==============================================")
    print(" ASL REMOTE DATA COLLECTION WEBSITE HOSTED")
    print("==============================================")
    print(" Local URL: http://127.0.0.1:3478")
    print(" Tailscale URL: http://<YOUR_TAILSCALE_IP>:3478")
    print(" Give your Tailscale URL to your friend!")
    print("==============================================\n")
    
    app.use_reloader = False
    app.run(host="0.0.0.0", port=3478, threaded=True)