"""
Neural Network Native 2D Architecture Canvas.
High-performance antialiased QPainter rendering providing a clean 2D schematic diagram
and serving as an authoritative fallback when WebGL 3D acceleration is unavailable.
"""
from typing import Optional, List, Dict, Any, Tuple
from PySide6.QtWidgets import QWidget, QToolTip
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont, QPainterPath

from app.core.theme import ThemeManager
from app.core.neural_network_service import ModelArchitecture, ForwardPassResult


class NeuralNetwork2DWidget(QWidget):
    """
    Native 2D Schematic Neural Network Viewport.
    Draws layers, connection edges, active signal pulses, and node activations.
    """
    neuron_selected = Signal(int, int)        # (layer_idx, neuron_idx)
    layer_selected = Signal(int)              # (layer_idx)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(400, 300)

        self._architecture: Optional[ModelArchitecture] = None
        self._forward_result: Optional[ForwardPassResult] = None
        self._selected_layer: Optional[int] = None
        self._selected_neuron: Optional[Tuple[int, int]] = None
        self._hovered_neuron: Optional[Tuple[int, int]] = None

        # Node coordinates map: node_positions[layer_idx][neuron_idx] = QPointF
        self._node_positions: List[List[QPointF]] = []

        # Forward Pass Animation state
        self._is_animating = False
        self._anim_step = 0
        self._anim_progress = 0.0
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(30)
        self._anim_timer.timeout.connect(self._on_anim_tick)

    def set_model_architecture(self, arch: ModelArchitecture):
        self._architecture = arch
        self._selected_layer = None
        self._selected_neuron = None
        self._hovered_neuron = None
        self._forward_result = None
        self._compute_layout()
        self.update()

    def set_forward_pass_result(self, result: ForwardPassResult):
        self._forward_result = result
        self._anim_step = 0
        self._anim_progress = 0.0
        self._is_animating = True
        self._anim_timer.start()
        self.update()

    def select_layer(self, layer_idx: int):
        self._selected_layer = layer_idx
        self.update()

    def select_neuron(self, layer_idx: int, neuron_idx: int):
        self._selected_neuron = (layer_idx, neuron_idx)
        self._selected_layer = layer_idx
        self.update()

    def stop_animation(self):
        """Immediately stop 2D pulse animation timer."""
        self._is_animating = False
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    def hideEvent(self, event):
        self.stop_animation()
        super().hideEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._compute_layout()

    def _compute_layout(self):
        """Calculate screen positions for every neuron in the network."""
        self._node_positions = []
        if not self._architecture or not self._architecture.layers:
            return

        w = self.width()
        h = self.height()
        layers = self._architecture.layers
        num_layers = len(layers)

        margin_x = 60
        margin_y = 60
        avail_w = max(w - margin_x * 2, 100)
        avail_h = max(h - margin_y * 2, 100)

        x_spacing = avail_w / max(num_layers - 1, 1)

        for l_idx, layer in enumerate(layers):
            x = margin_x + l_idx * x_spacing
            count = layer.units
            layer_pts = []

            # Bound display nodes for huge layers (e.g. max 32 representative nodes drawn)
            display_count = min(count, 32)
            y_spacing = avail_h / max(display_count - 1, 1)
            start_y = margin_y

            for n_idx in range(display_count):
                y = start_y + n_idx * y_spacing
                layer_pts.append(QPointF(x, y))

            self._node_positions.append(layer_pts)

    def _on_anim_tick(self):
        if not self._is_animating:
            return

        self._anim_progress += 0.08
        if self._anim_progress >= 1.0:
            self._anim_progress = 0.0
            self._anim_step += 1
            if not self._architecture or self._anim_step >= len(self._architecture.layers):
                self._is_animating = False
                self._anim_timer.stop()

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        palette = ThemeManager.get_palette()
        painter.fillRect(self.rect(), QColor(palette.bg_window))

        if not self._architecture or not self._node_positions:
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Neural Network Architecture Loaded")
            painter.end()
            return

        # 1. Draw Layer Background Columns & Headers
        self._draw_layer_columns(painter, palette)

        # 2. Draw Connection Edges
        self._draw_connections(painter, palette)

        # 3. Draw Neurons & Activation Rings
        self._draw_neurons(painter, palette)
        painter.end()

    def _draw_layer_columns(self, painter: QPainter, palette: Any):
        layers = self._architecture.layers
        h = self.height()

        for l_idx, layer in enumerate(layers):
            if l_idx >= len(self._node_positions) or not self._node_positions[l_idx]:
                continue

            x = self._node_positions[l_idx][0].x()
            is_selected = (self._selected_layer == l_idx)

            # Column highlight
            if is_selected:
                painter.fillRect(QRectF(x - 30, 20, 60, h - 40), QColor(94, 106, 210, 25))

            # Header Text
            painter.setPen(QColor(palette.accent if is_selected else palette.text_secondary))
            font = QFont()
            font.setPointSize(9)
            font.setBold(True)
            painter.setFont(font)

            name = "INPUT" if l_idx == 0 else ("OUTPUT" if l_idx == len(layers) - 1 else f"HIDDEN {l_idx}")
            painter.drawText(QRectF(x - 60, 16, 120, 20), Qt.AlignmentFlag.AlignCenter, name)

            font.setPointSize(8)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(QRectF(x - 60, 34, 120, 16), Qt.AlignmentFlag.AlignCenter, f"{layer.units}u • {layer.activation}")

    def _draw_connections(self, painter: QPainter, palette: Any):
        for l_idx in range(len(self._node_positions) - 1):
            nodes_a = self._node_positions[l_idx]
            nodes_b = self._node_positions[l_idx + 1]

            pen = QPen(QColor(44, 51, 69, 70))
            pen.setWidthF(0.8)
            painter.setPen(pen)

            # Sample connections cleanly
            step_a = max(1, len(nodes_a) // 8)
            step_b = max(1, len(nodes_b) // 8)

            for i in range(0, len(nodes_a), step_a):
                p1 = nodes_a[i]
                for j in range(0, len(nodes_b), step_b):
                    p2 = nodes_b[j]

                    path = QPainterPath(p1)
                    dx = (p2.x() - p1.x()) * 0.5
                    path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
                    painter.drawPath(path)

            # Animated Signal Pulses traveling along active connections
            if self._is_animating and self._anim_step == l_idx:
                prog = self._anim_progress
                pulse_pen = QPen(QColor(56, 189, 248, int(220 * (1.0 - abs(prog - 0.5) * 1.5))))
                pulse_pen.setWidthF(2.0)
                painter.setPen(pulse_pen)

                for i in range(0, len(nodes_a), max(1, len(nodes_a) // 4)):
                    p1 = nodes_a[i]
                    for j in range(0, len(nodes_b), max(1, len(nodes_b) // 4)):
                        p2 = nodes_b[j]
                        px = p1.x() + (p2.x() - p1.x()) * prog
                        py = p1.y() + (p2.y() - p1.y()) * prog
                        painter.drawEllipse(QPointF(px, py), 2.5, 2.5)

    def _draw_neurons(self, painter: QPainter, palette: Any):
        activations = self._forward_result.layer_activations if self._forward_result else None
        is_output_step = (self._forward_result and (not self._is_animating or self._anim_step >= len(self._node_positions) - 1))

        for l_idx, nodes in enumerate(self._node_positions):
            layer_acts = activations[l_idx] if activations and l_idx < len(activations) else None
            max_act = max(layer_acts) if layer_acts and any(layer_acts) else 1.0
            is_last = (l_idx == len(self._node_positions) - 1)
            pred_class_idx = -1
            if is_last and layer_acts:
                pred_class_idx = layer_acts.index(max(layer_acts))

            for n_idx, pt in enumerate(nodes):
                is_selected = (self._selected_neuron == (l_idx, n_idx))
                is_hovered = (self._hovered_neuron == (l_idx, n_idx))

                # Real activation calculation
                act_val = layer_acts[n_idx] if layer_acts and n_idx < len(layer_acts) else 0.0
                norm_act = min(max(act_val / max_act, 0.0), 1.0) if max_act > 0 else 0.0

                radius = 5.0

                if is_last and n_idx == pred_class_idx and is_output_step:
                    # Winning predicted class node
                    fill_col = QColor(16, 185, 129)
                    edge_col = QColor(16, 185, 129, 200)
                    radius = 8.5
                    # Halo
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor(16, 185, 129, 60))
                    painter.drawEllipse(pt, 14, 14)
                elif is_selected:
                    fill_col = QColor(168, 85, 247)
                    edge_col = QColor(216, 180, 254)
                    radius = 7.0
                elif is_hovered:
                    fill_col = QColor(56, 189, 248)
                    edge_col = QColor(255, 255, 255)
                    radius = 7.0
                elif layer_acts and norm_act > 0.05:
                    # Electric cyan scaled by real activation
                    r_col = int(30 + norm_act * 26)
                    g_col = int(50 + norm_act * 139)
                    b_col = int(80 + norm_act * 168)
                    fill_col = QColor(r_col, g_col, b_col)
                    edge_col = QColor(56, 189, 248, int(150 + norm_act * 105))
                    radius = 5.0 + norm_act * 2.5
                else:
                    fill_col = QColor(palette.bg_hover)
                    edge_col = QColor(palette.border_default)

                painter.setPen(QPen(edge_col, 1.2))
                painter.setBrush(QBrush(fill_col))
                painter.drawEllipse(pt, radius, radius)

                # Output labels (A, B, C, D, E, F)
                if is_last and self._architecture.classes and n_idx < len(self._architecture.classes):
                    c_name = self._architecture.classes[n_idx]
                    painter.setPen(QColor(16, 185, 129) if n_idx == pred_class_idx else QColor(palette.text_secondary))
                    font = QFont()
                    font.setPointSize(8)
                    font.setBold(n_idx == pred_class_idx)
                    painter.setFont(font)
                    painter.drawText(QRectF(pt.x() + 12, pt.y() - 10, 40, 20), Qt.AlignmentFlag.AlignVCenter, c_name)

    def mouseMoveEvent(self, event):
        pos = event.position()
        found = None

        for l_idx, nodes in enumerate(self._node_positions):
            for n_idx, pt in enumerate(nodes):
                dist_sq = (pt.x() - pos.x())**2 + (pt.y() - pos.y())**2
                if dist_sq < 80:
                    found = (l_idx, n_idx)
                    break
            if found:
                break

        if found != self._hovered_neuron:
            self._hovered_neuron = found
            if found:
                l, n = found
                act_str = "0.0000"
                if self._forward_result and l < len(self._forward_result.layer_activations):
                    acts = self._forward_result.layer_activations[l]
                    if n < len(acts):
                        act_str = f"{acts[n]:.4f}"
                QToolTip.showText(event.globalPosition().toPoint(), f"Layer {l} : Neuron #{n}\nActivation: {act_str}", self)
            else:
                QToolTip.hideText()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._hovered_neuron:
            l, n = self._hovered_neuron
            self.select_neuron(l, n)
            self.neuron_selected.emit(l, n)
            self.layer_selected.emit(l)
        super().mousePressEvent(event)
