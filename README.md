# SignFlow AI — Open-Source Sign Language Model Harness & Desktop Studio

A tactile, editorial desktop application and open-source harness for real-time American Sign Language (ASL) recognition, virtual camera video broadcasting, and machine learning model lifecycle management.

Built with **Python 3**, **PySide6 (Qt for Python)**, **OpenCV**, and **MediaPipe**.

---

## ✦ Key Highlights

- **Open-Source Model Harness**:
  - Acts as an agnostic desktop harness for AI models trained by developers and researchers.
  - Model weights and datasets are decoupled from the application repository.
  - **`+ Add Model` Smart Importer**: Point the app to any model directory (e.g., your training repository), and SignFlow automatically verifies the presence of `predict_live.py`, `asl_classifier.joblib`, and `hand_landmarker.task`, lets you name the model, and adds it to your active model switcher.
  - Switch between multiple models on-the-fly.

- **Virtual Camera Output (Discord, Zoom, Google Meet, Teams)**:
  - Broadcasts live webcam video with real-time sign language captions burned directly onto the video feed via `pyvirtualcam` (OBS Virtual Camera).
  - Anyone in your Discord voice channel, Zoom call, or meeting can see your live sign language translated into subtitles right on your video stream.

- **Tactile Editorial Aesthetic**:
  - **Light Mode**: Warm Cream & Alabaster (`#FAF7F0`), Soft Butter Yellow (`#FEF6D8` bordered in `#F3D580`), Warm Oat (`#F3EEE4`), Deep Espresso Charcoal typography (`#1C1917`), and Warm Gold accents (`#F5B419`).
  - **Dark Mode**: Deep Warm Charcoal canvas (`#161513`), Warm Dark Slate cards (`#262320`), Espresso surface (`#1E1C19`), and Amber Gold accents (`#F5B419` / `#FCD34D`).
  - Avoids harsh pure whites, cold corporate grays, and generic neon purple/blue accents.

- **Streamlined Workflow**:
  - **Main Landing**: Greeted with a top menu bar to select your active model, pick your camera device, view a live preview, and start translation with one click.
  - **Self Train & Developer Studio**: Training tools, dynamic gesture trajectory recorders, dataset managers, and neural network visualizers are cleanly organized under the **Self Train** section in Settings to keep the primary translation interface focused and minimal.

---

## ✦ Project Architecture

```
signlanguageapp/
├── run_app.py                      # Development launcher script
├── run.bat                         # Double-click Windows launcher (virtualenv-aware)
├── build_exe.py                    # Automated PyInstaller build script
├── SignLanguageStudio.spec         # PyInstaller build spec
├── requirements.txt                # Core runtime dependencies
├── README.md                       # Documentation
│
├── app/
│   ├── main.py                     # Entry point & QApplication lifecycle
│   ├── core/
│   │   ├── config.py               # SettingsManager, constants, QSettings persistence
│   │   ├── theme.py                # Warm Cream & Espresso Charcoal theme palette & QSS
│   │   ├── camera_manager.py       # QMediaDevices, QCamera, QVideoSink frame tap
│   │   ├── virtual_cam_service.py  # pyvirtualcam OBS Virtual Camera broadcast service
│   │   ├── model_registry.py       # Model harness package registry & smart folder scanner
│   │   ├── interfaces.py           # Backend contracts (Prediction & Training)
│   │   ├── backend_adapter.py      # MediaPipe & Scikit-learn inference adapter
│   │   ├── dataset_registry.py     # CSV dataset manager & versioning
│   │   └── neural_network_service.py # Neural network weights and layer activations
│   ├── ui/
│   │   ├── main_window.py          # Application shell, sidebar, topbar, page stack
│   │   ├── components/
│   │   │   ├── sidebar.py          # Minimalist sidebar with Self Train developer toggle
│   │   │   ├── top_bar.py          # Header toolbar with breadcrumb & theme toggle
│   │   │   ├── add_model_dialog.py # Smart folder inspector for importing external models
│   │   │   ├── camera_settings_dialog.py # Resolution & mirroring switcher
│   │   │   ├── searchable_combo.py # Searchable camera device dropdown
│   │   │   └── widgets.py          # Status pills, stat cards, section headers
│   │   └── pages/
│   │       ├── predict_page.py     # Live translation workspace & virtual camera output
│   │       ├── settings_page.py    # Preferences, virtual cam settings, and Self Train launcher
│   │       ├── static_training_page.py  # Static classifier training workspace
│   │       ├── dynamic_training_page.py # Dynamic sequence gesture workspace (J & Z)
│   │       ├── dataset_manager_page.py  # Dataset management workspace
│   │       ├── model_manager_page.py    # Model artifact catalog
│   │       ├── neural_network_page.py   # Neural network visualizer
│   │       └── overview_page.py    # Project overview & quick actions
│   └── resources/
│       ├── app_icon.ico            # Windows icon
│       ├── app_icon.png            # High-res 512x512 application icon
│       └── icons.py                # High-DPI theme-aware SVG vector icons
```

---

## ✦ Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/larperayush/signlanguage.ai.git
cd signlanguage.ai
pip install -r requirements.txt
```

### 2. Running the Application
Double-click `run.bat` or run via terminal:
```bash
python run_app.py
```

### 3. Adding Your Model
1. Click **+ Add Model** in the top menu bar.
2. Browse to your model folder (e.g. your training repository containing `asl_classifier.joblib`, `hand_landmarker.task`, and `predict_live.py`).
3. Enter a custom name and click **Import & Register Model**.
4. Select your camera, hit **Start Translation**, and choose **OBS Virtual Camera** in Discord, Zoom, or Teams!
