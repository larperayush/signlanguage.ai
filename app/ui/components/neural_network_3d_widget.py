"""
Neural Network 3D Viewport Component.
Embeds high-performance hardware-accelerated WebGL Three.js canvas in QWebEngineView.
Provides interactive 3D neuron inspection, connection raycasting, OrbitControls,
and synchronized two-way event communication with SignFlow Studio Qt framework.

Strictly on-demand: does not start WebGL contexts, animation loops, or polling timers
until the visualizer workspace is activated by the user. Immediately suspends all
rendering and background activity when navigating away.
"""
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import asdict

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

from app.core.neural_network_service import ModelArchitecture, ForwardPassResult
from app.core.config import ROOT_DIR


class NeuralNetwork3DWidget(QWidget):
    """
    Hardware-accelerated 3D Neural Network Viewport.
    Bridges Python Qt UI with the embedded Three.js WebGL visualization.
    Operates strictly on-demand with zero background rendering overhead.
    """
    neuron_selected = Signal(int, int)        # (layer_idx, neuron_idx)
    layer_selected = Signal(int)              # (layer_idx)
    scene_ready = Signal()
    forward_pass_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_initialized = False
        self._is_loaded = False
        self._is_page_active = False
        self.web_view: Optional[QWebEngineView] = None
        self._pending_model_arch: Optional[Dict[str, Any]] = None
        self._pending_forward_pass: Optional[Dict[str, Any]] = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Event Polling Timer for Javascript -> Python Bridge
        # Strictly stopped by default; only runs while the visualizer is actively open.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._poll_javascript_events)

    def ensure_initialized(self):
        """
        Lazily initialize QWebEngineView and load Three.js scene only when workspace is opened.
        Guarantees zero background GPU/WebGL initialization when user is on other pages.
        """
        if self._is_initialized:
            return

        self._is_initialized = True
        self.web_view = QWebEngineView(self)
        self._layout.addWidget(self.web_view)

        # Connect Load Signal
        self.web_view.loadFinished.connect(self._on_web_loaded)

        # Load Template HTML via file:// URL
        template_path = ROOT_DIR / "app" / "resources" / "templates" / "neural_network_3d.html"
        if template_path.exists():
            self.web_view.load(QUrl.fromLocalFile(str(template_path.resolve())))
        else:
            fallback = QLabel("Neural Network 3D Template not found.")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setStyleSheet("color: #EF4444; font-size: 13px; font-weight: 600;")
            self._layout.addWidget(fallback)

    def activate(self):
        """Resume viewport, start event polling, and re-enable on-demand rendering."""
        self._is_page_active = True
        self.ensure_initialized()

        if self._is_loaded and self.web_view:
            if not self._poll_timer.isActive():
                self._poll_timer.start()
            self.web_view.page().runJavaScript("if (window.setPageActive) { window.setPageActive(true); }")

    def deactivate(self):
        """Immediately stop all WebGL animation frames, JS rendering loops, and Qt polling timers."""
        self._is_page_active = False
        if self._poll_timer.isActive():
            self._poll_timer.stop()

        if self._is_initialized and self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.setPageActive) { window.setPageActive(false); }")

    def pause_rendering(self):
        self.deactivate()

    def resume_rendering(self):
        self.activate()

    def _on_web_loaded(self, ok: bool):
        self._is_loaded = ok
        if ok and self.web_view:
            if self._is_page_active:
                if not self._poll_timer.isActive():
                    self._poll_timer.start()
                self.web_view.page().runJavaScript("if (window.setPageActive) { window.setPageActive(true); }")
            else:
                self.web_view.page().runJavaScript("if (window.setPageActive) { window.setPageActive(false); }")

            self.scene_ready.emit()

            if self._pending_model_arch:
                self.load_model(self._pending_model_arch)
                self._pending_model_arch = None

            if self._pending_forward_pass:
                self.animate_forward_pass(self._pending_forward_pass)
                self._pending_forward_pass = None

    def _poll_javascript_events(self):
        """Poll queued interaction events from Three.js scene only when active."""
        if not self._is_page_active or not self._is_loaded or not self.web_view:
            return

        def handle_events(events):
            if not events or not isinstance(events, list):
                return
            for ev in events:
                evt_type = ev.get("event")
                data = ev.get("data", {})
                if evt_type == "neuron_selected":
                    l = data.get("layer_index", 0)
                    n = data.get("neuron_index", 0)
                    self.neuron_selected.emit(l, n)
                elif evt_type == "layer_selected":
                    l = data.get("layer_index", 0)
                    self.layer_selected.emit(l)
                elif evt_type == "forward_pass_finished":
                    self.forward_pass_finished.emit()

        self.web_view.page().runJavaScript("window.popBridgeEvents ? window.popBridgeEvents() : []", handle_events)

    # =========================================================================
    # PUBLIC PYTHON -> JAVASCRIPT CONTROLS
    # =========================================================================
    def load_model(self, architecture: ModelArchitecture):
        """Send authoritative model architecture parameters to 3D scene."""
        if isinstance(architecture, ModelArchitecture):
            raw = asdict(architecture)
        else:
            raw = architecture

        if not self._is_loaded or not self.web_view:
            self._pending_model_arch = raw
            return

        json_str = json.dumps(raw)
        js_cmd = f"if (window.loadModel) {{ window.loadModel({json_str}); }}"
        self.web_view.page().runJavaScript(js_cmd)

    def animate_forward_pass(self, forward_result: ForwardPassResult):
        """Trigger animated forward pass across 3D network with real activations."""
        if isinstance(forward_result, ForwardPassResult):
            raw = asdict(forward_result)
        else:
            raw = forward_result

        if not self._is_loaded or not self.web_view:
            self._pending_forward_pass = raw
            return

        json_str = json.dumps(raw)
        js_cmd = f"if (window.runForwardPass) {{ window.runForwardPass({json_str}); }}"
        self.web_view.page().runJavaScript(js_cmd)

    def select_layer(self, layer_index: int):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript(f"if (window.selectLayer) {{ window.selectLayer({layer_index}); }}")

    def select_neuron(self, layer_index: int, neuron_index: int):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript(f"if (window.selectNeuron) {{ window.selectNeuron({layer_index}, {neuron_index}); }}")

    def reset_camera(self):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.resetCamera) { window.resetCamera(); }")

    def fit_network(self):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.fitNetwork) { window.fitNetwork(); }")

    def set_speed(self, speed: float):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript(f"if (window.setAnimationSpeed) {{ window.setAnimationSpeed({speed}); }}")

    def pause(self):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.pauseAnimation) { window.pauseAnimation(); }")

    def resume(self):
        if self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.resumeAnimation) { window.resumeAnimation(); }")

    def check_loop_running(self, callback: Callable[[bool], None]):
        """Diagnostic probe for verifying zero background rendering in tests."""
        if not self._is_initialized or not self._is_loaded or not self.web_view:
            callback(False)
            return
        self.web_view.page().runJavaScript(
            "window.isLoopRunning ? window.isLoopRunning() : false",
            callback
        )

    def cleanup(self):
        """Safely stop polling and release resources."""
        self.deactivate()
        if self._is_initialized and self._is_loaded and self.web_view:
            self.web_view.page().runJavaScript("if (window.cleanupScene) { window.cleanupScene(); }")
            self.web_view.stop()
