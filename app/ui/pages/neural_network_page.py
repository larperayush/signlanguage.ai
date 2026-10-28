"""
Neural Network Visualizer Page - Primary Dedicated Workspace.
Integrates cinematic 3D WebGL visualization, engineering-grade Technical Inspector,
native 2D canvas fallback, real-time forward pass propagation, and live camera integration.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QButtonGroup, QSplitter, QStackedWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Slot, QTimer

from app.core.theme import ThemeManager
from app.core.neural_network_service import (
    NeuralNetworkService, ModelArchitecture, ForwardPassResult
)
from app.core.landmark_service import LandmarkService
from app.resources.icons import get_icon
from app.ui.components.widgets import StatusPill, SectionHeader
from app.ui.components.neural_network_3d_widget import NeuralNetwork3DWidget
from app.ui.components.neural_network_2d_widget import NeuralNetwork2DWidget
from app.ui.components.neural_network_inspector import TechnicalInspector


class NeuralNetworkPage(QWidget):
    """
    Dedicated workspace for interactive Neural Network exploration and inspection.
    """
    MODE_3D = 0
    MODE_INSPECTOR = 1
    MODE_SPLIT = 2
    MODE_2D = 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = NeuralNetworkService.get_instance()
        self._landmark_service = LandmarkService.get_instance()
        self._current_mode = self.MODE_3D
        self._selected_layer = 0
        self._selected_neuron = (0, 0)
        self._is_active = False

        self._build_ui()
        self._connect_signals()

        # Initialize with current model
        if self._service.current_architecture:
            self._on_model_loaded(self._service.current_architecture)

    def _build_ui(self):
        palette = ThemeManager.get_palette()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)

        # 1. Header Toolbar
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        # Title & Subtitle
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        lbl_title = QLabel("Neural Network Lab")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: -0.3px;")
        lbl_subtitle = QLabel("Interactive 3D Deep Architecture & Real-Time Tensor Activation Inspector")
        lbl_subtitle.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary};")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        top_bar.addLayout(title_box)

        top_bar.addStretch()

        # Model Selector Dropdown
        model_box = QHBoxLayout()
        model_box.setSpacing(6)
        lbl_model = QLabel("Model:")
        lbl_model.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_muted};")
        self.combo_models = QComboBox()
        self.combo_models.setMinimumWidth(240)
        self.combo_models.currentIndexChanged.connect(self._on_model_combo_changed)
        model_box.addWidget(lbl_model)
        model_box.addWidget(self.combo_models)
        top_bar.addLayout(model_box)

        # Status Pill
        self.status_pill = StatusPill("Neural Core Active", "active")
        top_bar.addWidget(self.status_pill)

        main_layout.addLayout(top_bar)

        # 2. Secondary Control Bar (View Mode Toggle & Forward Pass Triggers)
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(10)

        # Mode Selector Buttons
        self.btn_group_mode = QButtonGroup(self)
        self.btn_group_mode.setExclusive(True)

        self.btn_mode_3d = QPushButton(" 3D View")
        self.btn_mode_3d.setCheckable(True)
        self.btn_mode_3d.setChecked(True)
        self.btn_mode_3d.setProperty("class", "toggleBtn")
        self.btn_mode_3d.setIcon(get_icon("layers", palette.text_secondary, palette.accent, size=14))
        self.btn_group_mode.addButton(self.btn_mode_3d, self.MODE_3D)
        ctrl_bar.addWidget(self.btn_mode_3d)

        self.btn_mode_split = QPushButton(" Split View")
        self.btn_mode_split.setCheckable(True)
        self.btn_mode_split.setProperty("class", "toggleBtn")
        self.btn_mode_split.setIcon(get_icon("overview", palette.text_secondary, palette.accent, size=14))
        self.btn_group_mode.addButton(self.btn_mode_split, self.MODE_SPLIT)
        ctrl_bar.addWidget(self.btn_mode_split)

        self.btn_mode_inspector = QPushButton(" Technical Inspector")
        self.btn_mode_inspector.setCheckable(True)
        self.btn_mode_inspector.setProperty("class", "toggleBtn")
        self.btn_mode_inspector.setIcon(get_icon("cpu", palette.text_secondary, palette.accent, size=14))
        self.btn_group_mode.addButton(self.btn_mode_inspector, self.MODE_INSPECTOR)
        ctrl_bar.addWidget(self.btn_mode_inspector)

        self.btn_mode_2d = QPushButton(" 2D Schematic")
        self.btn_mode_2d.setCheckable(True)
        self.btn_mode_2d.setProperty("class", "toggleBtn")
        self.btn_mode_2d.setIcon(get_icon("sliders", palette.text_secondary, palette.accent, size=14))
        self.btn_group_mode.addButton(self.btn_mode_2d, self.MODE_2D)
        ctrl_bar.addWidget(self.btn_mode_2d)

        self.btn_group_mode.idClicked.connect(self._set_view_mode)

        ctrl_bar.addSpacing(16)

        # Input Data Source Selector
        lbl_inp = QLabel("Input Data:")
        lbl_inp.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {palette.text_muted};")
        ctrl_bar.addWidget(lbl_inp)

        self.combo_inputs = QComboBox()
        self.combo_inputs.addItem("Dataset Sample: Sign B", "dataset_B")
        self.combo_inputs.addItem("Dataset Sample: Sign A", "dataset_A")
        self.combo_inputs.addItem("Dataset Sample: Sign C", "dataset_C")
        self.combo_inputs.addItem("Dataset Sample: Sign D", "dataset_D")
        self.combo_inputs.addItem("Dataset Sample: Sign E", "dataset_E")
        self.combo_inputs.addItem("Dataset Sample: Sign F", "dataset_F")
        self.combo_inputs.addItem("Live Camera Hand (MediaPipe)", "camera")
        ctrl_bar.addWidget(self.combo_inputs)

        # Run Forward Pass Button
        self.btn_run_pass = QPushButton(" Run Forward Pass")
        self.btn_run_pass.setProperty("class", "primaryBtn")
        self.btn_run_pass.setIcon(get_icon("play", "#FFFFFF", size=13))
        self.btn_run_pass.clicked.connect(self._on_run_forward_pass_clicked)
        ctrl_bar.addWidget(self.btn_run_pass)

        # Replay Animation
        self.btn_replay = QPushButton(" Replay")
        self.btn_replay.setProperty("class", "secondaryBtn")
        self.btn_replay.setIcon(get_icon("refresh", palette.text_secondary, palette.accent, size=13))
        self.btn_replay.clicked.connect(self._on_replay_clicked)
        ctrl_bar.addWidget(self.btn_replay)

        ctrl_bar.addStretch()

        # 3D Viewport Controls
        self.btn_fit = QPushButton(" Fit Network")
        self.btn_fit.setProperty("class", "secondaryBtn")
        self.btn_fit.clicked.connect(lambda: self.widget_3d.fit_network())
        ctrl_bar.addWidget(self.btn_fit)

        self.btn_reset_cam = QPushButton(" Reset View")
        self.btn_reset_cam.setProperty("class", "secondaryBtn")
        self.btn_reset_cam.clicked.connect(lambda: self.widget_3d.reset_camera())
        ctrl_bar.addWidget(self.btn_reset_cam)

        main_layout.addLayout(ctrl_bar)

        # 3. Main Workspace Area (Stacked / Split Views)
        self.main_stack = QStackedWidget()

        # 3A. Dedicated 3D View
        self.widget_3d = NeuralNetwork3DWidget(self)
        self.main_stack.addWidget(self.widget_3d) # Index 0

        # 3B. Dedicated Technical Inspector
        self.widget_inspector = TechnicalInspector(self)
        self.main_stack.addWidget(self.widget_inspector) # Index 1

        # 3C. Split View (Splitter: Left 3D View, Right Inspector)
        self.split_view = QSplitter(Qt.Orientation.Horizontal)
        self.split_widget_3d = NeuralNetwork3DWidget(self)
        self.split_inspector = TechnicalInspector(self)
        self.split_view.addWidget(self.split_widget_3d)
        self.split_view.addWidget(self.split_inspector)
        self.split_view.setStretchFactor(0, 3)
        self.split_view.setStretchFactor(1, 2)
        self.main_stack.addWidget(self.split_view) # Index 2

        # 3D. 2D Canvas Schematic Fallback
        self.widget_2d = NeuralNetwork2DWidget(self)
        self.main_stack.addWidget(self.widget_2d) # Index 3

        main_layout.addWidget(self.main_stack, 1)

        # 4. Bottom HUD Inference Banner
        self.hud_frame = QFrame()
        self.hud_frame.setProperty("class", "cardPanelElevated")
        hud_layout = QHBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(16, 8, 16, 8)
        hud_layout.setSpacing(14)

        self.lbl_hud_sign = QLabel("LATEST INFERENCE: None")
        self.lbl_hud_sign.setStyleSheet("font-size: 12px; font-weight: 700; color: #10B981;")
        hud_layout.addWidget(self.lbl_hud_sign)

        self.lbl_hud_details = QLabel("Select an input source and press 'Run Forward Pass'")
        self.lbl_hud_details.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary};")
        hud_layout.addWidget(self.lbl_hud_details)

        hud_layout.addStretch()

        self.lbl_hud_progression = QLabel("Input (63) ➔ Hidden 1 ➔ Hidden 2 ➔ Output (6)")
        self.lbl_hud_progression.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-family: monospace;")
        hud_layout.addWidget(self.lbl_hud_progression)

        main_layout.addWidget(self.hud_frame)

        self._refresh_models_list()

    def _connect_signals(self):
        # Service Signals
        self._service.model_loaded.connect(self._on_model_loaded)
        self._service.forward_pass_completed.connect(self._on_forward_pass_completed)

        # 3D Interaction Signals -> Synchronize Inspector
        self.widget_3d.layer_selected.connect(self._on_layer_selected_3d)
        self.widget_3d.neuron_selected.connect(self._on_neuron_selected_3d)
        self.split_widget_3d.layer_selected.connect(self._on_layer_selected_3d)
        self.split_widget_3d.neuron_selected.connect(self._on_neuron_selected_3d)

        # 2D Interaction Signals
        self.widget_2d.layer_selected.connect(self._on_layer_selected_3d)
        self.widget_2d.neuron_selected.connect(self._on_neuron_selected_3d)

        # Inspector Selection Signals -> Synchronize 3D / 2D
        self.widget_inspector.layer_selected.connect(self._on_inspector_layer_selected)
        self.widget_inspector.neuron_selected.connect(self._on_inspector_neuron_selected)
        self.split_inspector.layer_selected.connect(self._on_inspector_layer_selected)
        self.split_inspector.neuron_selected.connect(self._on_inspector_neuron_selected)

    def _refresh_models_list(self):
        self.combo_models.blockSignals(True)
        self.combo_models.clear()
        models = self._service.list_compatible_models()
        for m in models:
            self.combo_models.addItem(m["name"], m["file_path"])
        self.combo_models.blockSignals(False)

    def _on_model_combo_changed(self, idx: int):
        if idx >= 0:
            file_path = self.combo_models.currentData()
            if file_path:
                self._service.load_model(file_path)

    def on_page_activated(self):
        """Called when user navigates to Neural Network Lab page."""
        self._is_active = True

        # Activate according to current mode
        if self._current_mode == self.MODE_3D:
            self.widget_3d.activate()
            QTimer.singleShot(100, self.widget_3d.fit_network)
        elif self._current_mode == self.MODE_SPLIT:
            self.split_widget_3d.activate()
            QTimer.singleShot(100, self.split_widget_3d.fit_network)
        elif self._current_mode == self.MODE_2D:
            self.widget_2d.update()

        # Sync architecture if model loaded while inactive
        if self._service.current_architecture:
            self._on_model_loaded(self._service.current_architecture)

    def on_page_deactivated(self):
        """Immediately halts all 3D rendering loops, polling timers, and pulse animations."""
        self._is_active = False
        self.widget_3d.deactivate()
        self.split_widget_3d.deactivate()
        self.widget_2d.stop_animation()

    def showEvent(self, event):
        super().showEvent(event)
        self.on_page_activated()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.on_page_deactivated()

    @Slot(int)
    def _set_view_mode(self, mode_idx: int):
        prev_mode = self._current_mode
        self._current_mode = mode_idx
        self.main_stack.setCurrentIndex(mode_idx)

        if not self._is_active:
            return

        # Deactivate unused 3D widgets to save GPU/WebGL resources
        if prev_mode == self.MODE_3D and mode_idx != self.MODE_3D:
            self.widget_3d.deactivate()
        elif prev_mode == self.MODE_SPLIT and mode_idx != self.MODE_SPLIT:
            self.split_widget_3d.deactivate()
        elif prev_mode == self.MODE_2D and mode_idx != self.MODE_2D:
            self.widget_2d.stop_animation()

        # Activate selected view on demand
        if mode_idx == self.MODE_3D:
            self.widget_3d.activate()
            QTimer.singleShot(100, self.widget_3d.fit_network)
        elif mode_idx == self.MODE_SPLIT:
            self.split_widget_3d.activate()
            QTimer.singleShot(100, self.split_widget_3d.fit_network)
        elif mode_idx == self.MODE_2D:
            self.widget_2d.update()

    @Slot(object)
    def _on_model_loaded(self, arch: ModelArchitecture):
        self.status_pill.set_status(f"Active: {arch.total_parameters:,} Params", "active")
        self.lbl_hud_progression.setText(" ➔ ".join([f"{l.name.split()[0]} ({l.units})" for l in arch.layers]))

        # Update all active viewports
        self.widget_3d.load_model(arch)
        self.split_widget_3d.load_model(arch)
        self.widget_2d.set_model_architecture(arch)
        self.widget_inspector.set_model_architecture(arch)
        self.split_inspector.set_model_architecture(arch)

    @Slot()
    def _on_run_forward_pass_clicked(self):
        source_key = self.combo_inputs.currentData()

        if source_key == "camera":
            # Extract live hand landmarks from LandmarkService
            latest = self._landmark_service.latest_landmarks
            if not latest or not latest.get("feature_vector") or len(latest.get("feature_vector")) != 63:
                self.lbl_hud_details.setText("No active hand detected in camera stream. Switch input or show hand.")
                return
            features = latest["feature_vector"]
            source_desc = "Live Camera Hand"
        else:
            # Load genuine dataset sample for specified sign
            sign_char = source_key.split("_")[-1]
            sample_data = self._service.load_dataset_sample(sign_char)
            if not sample_data:
                self.lbl_hud_details.setText(f"Could not load dataset sample for sign {sign_char}.")
                return
            features, sign_char = sample_data
            source_desc = f"Dataset Sample (Sign {sign_char})"

        # Execute authoritative mathematical forward pass
        result = self._service.execute_forward_pass(features, source_desc=source_desc)
        if result:
            self._on_forward_pass_completed(result)

    @Slot()
    def _on_replay_clicked(self):
        last = self._service.last_forward_pass
        if last:
            self._on_forward_pass_completed(last)

    @Slot(object)
    def _on_forward_pass_completed(self, result: ForwardPassResult):
        # Update HUD Banner
        self.lbl_hud_sign.setText(f"PREDICTED SIGN: {result.predicted_class} ({result.confidence * 100:.1f}%)")
        self.lbl_hud_details.setText(f"Latency: {result.latency_ms:.2f}ms • Source: {result.source} • ID: {result.inference_id}")

        # Trigger animated forward pass across active viewports
        self.widget_3d.animate_forward_pass(result)
        self.split_widget_3d.animate_forward_pass(result)
        self.widget_2d.set_forward_pass_result(result)
        self.widget_inspector.set_forward_pass_result(result)
        self.split_inspector.set_forward_pass_result(result)

    # =========================================================================
    # SYNCHRONIZATION: 3D <-> INSPECTOR
    # =========================================================================
    @Slot(int)
    def _on_layer_selected_3d(self, layer_idx: int):
        self._selected_layer = layer_idx
        self.widget_inspector.select_layer(layer_idx)
        self.split_inspector.select_layer(layer_idx)
        self.widget_2d.select_layer(layer_idx)

    @Slot(int, int)
    def _on_neuron_selected_3d(self, layer_idx: int, neuron_idx: int):
        self._selected_layer = layer_idx
        self._selected_neuron = (layer_idx, neuron_idx)
        self.widget_inspector.select_layer(layer_idx)
        self.split_inspector.select_layer(layer_idx)
        self.widget_2d.select_neuron(layer_idx, neuron_idx)

    @Slot(int)
    def _on_inspector_layer_selected(self, layer_idx: int):
        self._selected_layer = layer_idx
        self.widget_3d.select_layer(layer_idx)
        self.split_widget_3d.select_layer(layer_idx)
        self.widget_2d.select_layer(layer_idx)

    @Slot(int, int)
    def _on_inspector_neuron_selected(self, layer_idx: int, neuron_idx: int):
        self._selected_layer = layer_idx
        self._selected_neuron = (layer_idx, neuron_idx)
        self.widget_3d.select_neuron(layer_idx, neuron_idx)
        self.split_widget_3d.select_neuron(layer_idx, neuron_idx)
        self.widget_2d.select_neuron(layer_idx, neuron_idx)

    def closeEvent(self, event):
        self.on_page_deactivated()
        self.widget_3d.cleanup()
        self.split_widget_3d.cleanup()
        super().closeEvent(event)
