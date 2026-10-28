"""
Technical Inspector for Neural Network Visualizer.
Provides deep engineering-focused inspection of model parameters, layer dimensions,
weight matrices, histograms, intermediate activation distributions, and output logits.
"""
from typing import Optional, Dict, Any, List, Tuple
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QTabWidget, QComboBox, QScrollArea, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QToolTip
)
from PySide6.QtCore import Qt, Signal, Slot, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from app.core.theme import ThemeManager
from app.core.neural_network_service import (
    NeuralNetworkService, ModelArchitecture, LayerMetadata, ForwardPassResult
)
from app.resources.icons import get_icon


# =============================================================================
# CUSTOM VISUALIZATION SUB-WIDGETS (Heatmap, Histogram, Activation Bars)
# =============================================================================

class WeightHeatmapWidget(QWidget):
    """Interactive 2D Matrix Heatmap for layer weights."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self._matrix: List[List[float]] = []
        self._min_val = -1.0
        self._max_val = 1.0
        self._hovered_cell: Optional[Tuple[int, int]] = None

    def set_data(self, summary: Dict[str, Any]):
        self._matrix = summary.get("matrix", [])
        self._min_val = summary.get("min", -1.0)
        self._max_val = summary.get("max", 1.0)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager.get_palette()

        painter.fillRect(self.rect(), QColor(palette.bg_card_elevated))

        if not self._matrix or not self._matrix[0]:
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Weight Matrix Available")
            return

        rows = len(self._matrix)
        cols = len(self._matrix[0])

        w = self.width() - 20
        h = self.height() - 20
        cell_w = w / max(cols, 1)
        cell_h = h / max(rows, 1)

        max_abs = max(abs(self._min_val), abs(self._max_val), 0.0001)

        for r in range(rows):
            for c in range(cols):
                val = self._matrix[r][c]
                norm = val / max_abs  # -1.0 to 1.0

                if norm >= 0:
                    # Positive weight: Electric Cyan
                    color = QColor(
                        int(30 + norm * 26),
                        int(41 + norm * 148),
                        int(59 + norm * 189)
                    )
                else:
                    # Negative weight: Deep Indigo / Violet
                    abs_norm = abs(norm)
                    color = QColor(
                        int(30 + abs_norm * 99),
                        int(41 + abs_norm * 61),
                        int(59 + abs_norm * 182)
                    )

                rect = QRectF(10 + c * cell_w, 10 + r * cell_h, max(cell_w, 1.0), max(cell_h, 1.0))
                painter.fillRect(rect, color)
        painter.end()

    def mouseMoveEvent(self, event):
        if not self._matrix or not self._matrix[0]:
            return
        rows = len(self._matrix)
        cols = len(self._matrix[0])
        w = self.width() - 20
        h = self.height() - 20
        cell_w = w / max(cols, 1)
        cell_h = h / max(rows, 1)

        pos = event.position()
        c = int((pos.x() - 10) / cell_w)
        r = int((pos.y() - 10) / cell_h)

        if 0 <= r < rows and 0 <= c < cols:
            val = self._matrix[r][c]
            QToolTip.showText(event.globalPosition().toPoint(), f"W[{r}, {c}] = {val:+.5f}", self)
        else:
            QToolTip.hideText()


class WeightHistogramWidget(QWidget):
    """Weight distribution histogram bar chart."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self._counts: List[int] = []
        self._edges: List[float] = []

    def set_data(self, hist_data: Dict[str, Any]):
        self._counts = hist_data.get("counts", [])
        self._edges = hist_data.get("bin_edges", [])
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager.get_palette()

        painter.fillRect(self.rect(), QColor(palette.bg_card_elevated))

        if not self._counts:
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Histogram Data")
            painter.end()
            return

        w = self.width() - 24
        h = self.height() - 24
        n_bins = len(self._counts)
        bar_w = w / max(n_bins, 1)
        max_c = max(self._counts) or 1

        for i, count in enumerate(self._counts):
            bar_h = (count / max_c) * (h - 10)
            x = 12 + i * bar_w
            y = self.height() - 12 - bar_h

            rect = QRectF(x, y, max(bar_w - 1.5, 1.0), bar_h)
            painter.fillRect(rect, QColor(94, 106, 210, 200))

        # Baseline
        painter.setPen(QPen(QColor(palette.border_default), 1))
        painter.drawLine(12, self.height() - 12, self.width() - 12, self.height() - 12)
        painter.end()


class ClassProbabilityBarsWidget(QWidget):
    """Visualizer for output class probabilities and predicted signs."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self._probabilities: Dict[str, float] = {}
        self._logits: Dict[str, float] = {}
        self._predicted_class = ""

    def set_data(self, probs: Dict[str, float], logits: Dict[str, float], pred_class: str):
        self._probabilities = probs
        self._logits = logits
        self._predicted_class = pred_class
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = ThemeManager.get_palette()

        painter.fillRect(self.rect(), QColor(palette.bg_card_elevated))

        if not self._probabilities:
            painter.setPen(QColor(palette.text_muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Awaiting Forward Pass Inference...")
            painter.end()
            return

        items = list(self._probabilities.items())
        count = len(items)
        row_h = (self.height() - 16) / max(count, 1)

        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)

        for i, (cls_name, prob) in enumerate(items):
            y = 10 + i * row_h
            is_win = (cls_name == self._predicted_class)

            # Class Label
            painter.setPen(QColor(palette.success if is_win else palette.text_primary))
            font.setBold(is_win)
            painter.setFont(font)
            painter.drawText(QRectF(14, y, 40, row_h), Qt.AlignmentFlag.AlignVCenter, f"Sign {cls_name}")

            # Progress Bar Background
            bar_x = 70
            bar_w = self.width() - bar_x - 90
            bar_h = 10
            bar_y = y + (row_h - bar_h) / 2
            painter.fillRect(QRectF(bar_x, bar_y, bar_w, bar_h), QColor(palette.bg_hover))

            # Filled Bar
            fill_w = bar_w * min(max(prob, 0.0), 1.0)
            fill_col = QColor(palette.success if is_win else palette.accent)
            painter.fillRect(QRectF(bar_x, bar_y, fill_w, bar_h), fill_col)

            # Percentage Text
            painter.setPen(QColor(palette.text_secondary))
            font.setBold(False)
            painter.setFont(font)
            pct_str = f"{prob * 100:.1f}%"
            painter.drawText(QRectF(self.width() - 80, y, 70, row_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, pct_str)
        painter.end()


# =============================================================================
# MAIN TECHNICAL INSPECTOR
# =============================================================================

class TechnicalInspector(QWidget):
    """
    Comprehensive multi-panel inspector displaying authoritative architecture,
    weights, intermediate activations, and logits.
    """
    layer_selected = Signal(int)
    neuron_selected = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._service = NeuralNetworkService.get_instance()
        self._architecture: Optional[ModelArchitecture] = None
        self._forward_result: Optional[ForwardPassResult] = None
        self._selected_layer_idx = 0

        self._build_ui()

    def _build_ui(self):
        palette = ThemeManager.get_palette()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Tabs for Modular Inspection
        self.tabs = QTabWidget()
        self.tabs.setProperty("class", "modernTabs")

        # Tab 1: Model Overview
        self.tab_overview = self._create_overview_tab()
        self.tabs.addTab(self.tab_overview, "Overview")

        # Tab 2: Layer & Weights Inspector
        self.tab_layers = self._create_layers_tab()
        self.tabs.addTab(self.tab_layers, "Layers & Tensors")

        # Tab 3: Activation Inspector
        self.tab_activations = self._create_activations_tab()
        self.tabs.addTab(self.tab_activations, "Activations")

        # Tab 4: Output Logits & Predictions
        self.tab_outputs = self._create_outputs_tab()
        self.tabs.addTab(self.tab_outputs, "Output Logits")

        main_layout.addWidget(self.tabs)

    # =========================================================================
    # TAB 1: MODEL OVERVIEW
    # =========================================================================
    def _create_overview_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Overview Grid
        grid_frame = QFrame()
        grid_frame.setProperty("class", "cardPanelElevated")
        grid = QGridLayout(grid_frame)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)

        def add_stat_row(r, label, val_widget):
            lbl = QLabel(label)
            lbl.setStyleSheet(f"font-size: 11px; color: {palette.text_muted}; font-weight: 600;")
            grid.addWidget(lbl, r, 0)
            grid.addWidget(val_widget, r, 1)

        self.lbl_ov_name = QLabel("None")
        self.lbl_ov_name.setStyleSheet("font-weight: 700; font-size: 12px;")
        add_stat_row(0, "Model Name:", self.lbl_ov_name)

        self.lbl_ov_family = QLabel("Feedforward MLP")
        add_stat_row(1, "Architecture:", self.lbl_ov_family)

        self.lbl_ov_framework = QLabel("Scikit-Learn")
        add_stat_row(2, "Framework:", self.lbl_ov_framework)

        self.lbl_ov_layers = QLabel("0 Layers")
        add_stat_row(3, "Total Layers:", self.lbl_ov_layers)

        self.lbl_ov_params = QLabel("0")
        self.lbl_ov_params.setStyleSheet(f"font-family: monospace; font-weight: 700; color: {palette.accent};")
        add_stat_row(4, "Trainable Params:", self.lbl_ov_params)

        self.lbl_ov_dims = QLabel("63 In → 6 Out")
        self.lbl_ov_dims.setStyleSheet("font-family: monospace;")
        add_stat_row(5, "Input / Output:", self.lbl_ov_dims)

        layout.addWidget(grid_frame)

        # Architecture Layers Table
        lbl_tbl = QLabel("ARCHITECTURE SUMMARY")
        lbl_tbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_tbl)

        self.arch_table = QTableWidget()
        self.arch_table.setColumnCount(5)
        self.arch_table.setHorizontalHeaderLabels(["Index", "Layer Type", "Units", "Activation", "Parameters"])
        self.arch_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.arch_table.verticalHeader().setVisible(False)
        self.arch_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.arch_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.arch_table.itemSelectionChanged.connect(self._on_arch_table_selected)
        layout.addWidget(self.arch_table)

        return w

    # =========================================================================
    # TAB 2: LAYERS & TENSORS INSPECTOR
    # =========================================================================
    def _create_layers_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Layer Selection Toolbar
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Select Layer:"))
        self.combo_layer = QComboBox()
        self.combo_layer.currentIndexChanged.connect(self._on_layer_combo_changed)
        sel_row.addWidget(self.combo_layer, 1)
        layout.addLayout(sel_row)

        # Layer Parameter Specs Grid
        specs_frame = QFrame()
        specs_frame.setProperty("class", "cardPanelElevated")
        specs_grid = QGridLayout(specs_frame)
        specs_grid.setContentsMargins(12, 10, 12, 10)
        specs_grid.setHorizontalSpacing(16)
        specs_grid.setVerticalSpacing(6)

        def add_spec(r, c, title, val_lbl):
            lbl = QLabel(title)
            lbl.setStyleSheet(f"font-size: 10px; color: {palette.text_muted}; font-weight: 600;")
            specs_grid.addWidget(lbl, r * 2, c)
            specs_grid.addWidget(val_lbl, r * 2 + 1, c)

        self.lbl_ly_units = QLabel("0")
        self.lbl_ly_units.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: 700;")
        add_spec(0, 0, "UNITS", self.lbl_ly_units)

        self.lbl_ly_act = QLabel("None")
        self.lbl_ly_act.setStyleSheet(f"font-family: monospace; font-size: 12px; color: {palette.accent}; font-weight: 700;")
        add_spec(0, 1, "ACTIVATION", self.lbl_ly_act)

        self.lbl_ly_params = QLabel("0")
        self.lbl_ly_params.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: 700;")
        add_spec(0, 2, "PARAMETERS", self.lbl_ly_params)

        self.lbl_ly_w_shape = QLabel("(0, 0)")
        self.lbl_ly_w_shape.setStyleSheet("font-family: monospace; font-size: 11px;")
        add_spec(1, 0, "WEIGHT TENSOR", self.lbl_ly_w_shape)

        self.lbl_ly_b_shape = QLabel("(0,)")
        self.lbl_ly_b_shape.setStyleSheet("font-family: monospace; font-size: 11px;")
        add_spec(1, 1, "BIAS TENSOR", self.lbl_ly_b_shape)

        self.lbl_ly_w_stats = QLabel("μ=0.0, σ=0.0")
        self.lbl_ly_w_stats.setStyleSheet("font-family: monospace; font-size: 11px; color: #94A3B8;")
        add_spec(1, 2, "STATS (μ, σ)", self.lbl_ly_w_stats)

        layout.addWidget(specs_frame)

        # Weight Heatmap
        lbl_hm = QLabel("WEIGHT TENSOR MATRIX HEATMAP (W)")
        lbl_hm.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_hm)

        self.heatmap_widget = WeightHeatmapWidget()
        layout.addWidget(self.heatmap_widget)

        # Weight Distribution Histogram
        lbl_hist = QLabel("WEIGHT VALUE DISTRIBUTION (20 BINS)")
        lbl_hist.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_hist)

        self.histogram_widget = WeightHistogramWidget()
        layout.addWidget(self.histogram_widget)

        layout.addStretch()
        scroll.setWidget(w)
        return scroll

    # =========================================================================
    # TAB 3: ACTIVATIONS INSPECTOR
    # =========================================================================
    def _create_activations_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Header Stats
        self.lbl_act_summary = QLabel("Awaiting live or dataset forward pass...")
        self.lbl_act_summary.setStyleSheet(f"color: {palette.text_secondary}; font-size: 11px;")
        layout.addWidget(self.lbl_act_summary)

        # Top Neurons Table
        lbl_top = QLabel("LAYER ACTIVATIONS BREAKDOWN")
        lbl_top.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_top)

        self.act_table = QTableWidget()
        self.act_table.setColumnCount(4)
        self.act_table.setHorizontalHeaderLabels(["Neuron", "Activation", "Relative", "Bias"])
        self.act_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.act_table.verticalHeader().setVisible(False)
        self.act_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.act_table.cellClicked.connect(self._on_neuron_table_clicked)
        layout.addWidget(self.act_table)

        return w

    # =========================================================================
    # TAB 4: OUTPUT LOGITS & PREDICTIONS
    # =========================================================================
    def _create_outputs_tab(self) -> QWidget:
        palette = ThemeManager.get_palette()
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        # Prediction Banner Card
        self.banner_frame = QFrame()
        self.banner_frame.setProperty("class", "cardPanelElevated")
        banner_layout = QHBoxLayout(self.banner_frame)
        banner_layout.setContentsMargins(16, 12, 16, 12)

        v_left = QVBoxLayout()
        v_left.setSpacing(2)
        lbl_pred_title = QLabel("PREDICTED SIGN")
        lbl_pred_title.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted};")
        self.lbl_winner_sign = QLabel("None")
        self.lbl_winner_sign.setStyleSheet("font-size: 26px; font-weight: 800; color: #10B981;")
        v_left.addWidget(lbl_pred_title)
        v_left.addWidget(self.lbl_winner_sign)
        banner_layout.addLayout(v_left)

        banner_layout.addStretch()

        v_right = QVBoxLayout()
        v_right.setSpacing(2)
        self.lbl_winner_conf = QLabel("Confidence: 0.0%")
        self.lbl_winner_conf.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {palette.text_primary};")
        self.lbl_winner_latency = QLabel("Latency: 0.0ms")
        self.lbl_winner_latency.setStyleSheet(f"font-size: 11px; color: {palette.text_secondary}; font-family: monospace;")
        v_right.addWidget(self.lbl_winner_conf)
        v_right.addWidget(self.lbl_winner_latency)
        banner_layout.addLayout(v_right)

        layout.addWidget(self.banner_frame)

        # Probability Distribution Bar Chart
        lbl_bars = QLabel("OUTPUT SOFTMAX PROBABILITIES")
        lbl_bars.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_bars)

        self.prob_bars_widget = ClassProbabilityBarsWidget()
        layout.addWidget(self.prob_bars_widget)

        # Logits Comparison Table
        lbl_logits = QLabel("RAW PRE-SOFTMAX LOGITS COMPARISON")
        lbl_logits.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {palette.text_muted}; letter-spacing: 0.5px;")
        layout.addWidget(lbl_logits)

        self.logits_table = QTableWidget()
        self.logits_table.setColumnCount(3)
        self.logits_table.setHorizontalHeaderLabels(["Class Sign", "Raw Logit (z)", "Probability (Softmax)"])
        self.logits_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.logits_table.verticalHeader().setVisible(False)
        self.logits_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.logits_table)

        return w

    # =========================================================================
    # SYNCHRONIZATION SLOTS
    # =========================================================================
    @Slot(object)
    def set_model_architecture(self, arch: ModelArchitecture):
        self._architecture = arch
        self._selected_layer_idx = 0

        # Update Overview Labels
        self.lbl_ov_name.setText(arch.model_name)
        self.lbl_ov_family.setText(arch.family)
        self.lbl_ov_framework.setText(arch.framework)
        self.lbl_ov_layers.setText(f"{arch.total_layers} Layers ({arch.input_dim} In → {arch.output_dim} Out)")
        self.lbl_ov_params.setText(f"{arch.total_parameters:,} params")
        self.lbl_ov_dims.setText(f"{arch.input_dim} In → {arch.output_dim} Out ({', '.join(arch.classes)})")

        # Update Architecture Table
        self.arch_table.setRowCount(0)
        for i, layer in enumerate(arch.layers):
            self.arch_table.insertRow(i)
            self.arch_table.setItem(i, 0, QTableWidgetItem(f"L{layer.index}"))
            self.arch_table.setItem(i, 1, QTableWidgetItem(layer.layer_type))
            self.arch_table.setItem(i, 2, QTableWidgetItem(f"{layer.units} units"))
            self.arch_table.setItem(i, 3, QTableWidgetItem(layer.activation))
            self.arch_table.setItem(i, 4, QTableWidgetItem(f"{layer.trainable_params:,}"))

        # Update Layer Selector Combo
        self.combo_layer.blockSignals(True)
        self.combo_layer.clear()
        for layer in arch.layers:
            self.combo_layer.addItem(f"Layer {layer.index}: {layer.name} ({layer.units} units)")
        self.combo_layer.blockSignals(False)

        # Refresh Layer Specs
        self._update_layer_specs(0)

    @Slot(object)
    def set_forward_pass_result(self, result: ForwardPassResult):
        self._forward_result = result

        # Update Winner Banner
        self.lbl_winner_sign.setText(f"Sign {result.predicted_class}")
        self.lbl_winner_conf.setText(f"Confidence: {result.confidence * 100:.2f}%")
        self.lbl_winner_latency.setText(f"Latency: {result.latency_ms:.2f}ms • {result.source}")

        # Update Probability Bars Widget
        self.prob_bars_widget.set_data(result.probabilities, result.logits, result.predicted_class)

        # Update Logits Table
        self.logits_table.setRowCount(0)
        classes = self._architecture.classes if self._architecture else list(result.probabilities.keys())
        for i, c in enumerate(classes):
            self.logits_table.insertRow(i)
            logit_val = result.logits.get(c, 0.0)
            prob_val = result.probabilities.get(c, 0.0)

            self.logits_table.setItem(i, 0, QTableWidgetItem(f"Sign {c}"))
            self.logits_table.setItem(i, 1, QTableWidgetItem(f"{logit_val:+.4f}"))
            self.logits_table.setItem(i, 2, QTableWidgetItem(f"{prob_val * 100:.2f}%"))

        # Refresh Current Layer Activations
        self._update_activations_display(self._selected_layer_idx)

    def select_layer(self, layer_idx: int):
        if not self._architecture or layer_idx < 0 or layer_idx >= len(self._architecture.layers):
            return
        self._selected_layer_idx = layer_idx
        self.combo_layer.blockSignals(True)
        self.combo_layer.setCurrentIndex(layer_idx)
        self.combo_layer.blockSignals(False)
        self._update_layer_specs(layer_idx)
        self._update_activations_display(layer_idx)
        self.layer_selected.emit(layer_idx)

    def _update_layer_specs(self, layer_idx: int):
        if not self._architecture or layer_idx < 0 or layer_idx >= len(self._architecture.layers):
            return

        layer = self._architecture.layers[layer_idx]
        self.lbl_ly_units.setText(f"{layer.units}")
        self.lbl_ly_act.setText(layer.activation)
        self.lbl_ly_params.setText(f"{layer.trainable_params:,}")
        self.lbl_ly_w_shape.setText(f"{layer.weights_shape}")
        self.lbl_ly_b_shape.setText(f"{layer.bias_shape}")
        self.lbl_ly_w_stats.setText(f"μ={layer.weights_mean:+.3f}, σ={layer.weights_std:.3f}")

        # Update Heatmap & Histogram
        if layer_idx > 0:
            summary = self._service.get_layer_weight_matrix_summary(layer_idx)
            self.heatmap_widget.set_data(summary)

            hist_data = self._service.get_layer_weight_histogram(layer_idx)
            self.histogram_widget.set_data(hist_data)
        else:
            self.heatmap_widget.set_data({})
            self.histogram_widget.set_data({})

    def _update_activations_display(self, layer_idx: int):
        self.act_table.setRowCount(0)
        if not self._forward_result or layer_idx >= len(self._forward_result.layer_activations):
            self.lbl_act_summary.setText("Awaiting live or dataset forward pass...")
            return

        acts = self._forward_result.layer_activations[layer_idx]
        stats = self._forward_result.layer_statistics[layer_idx]
        self.lbl_act_summary.setText(
            f"Layer {layer_idx} Activations: Min={stats['min']:.4f}, Max={stats['max']:.4f}, "
            f"Mean={stats['mean']:.4f}, Active Neurons={stats['active_count']}/{len(acts)}"
        )

        max_act = max(acts) if acts and any(acts) else 1.0
        biases = self._service.get_layer_biases(layer_idx)

        # Show table of activations
        limit = min(len(acts), 64)
        for n_idx in range(limit):
            val = acts[n_idx]
            rel = (val / max_act * 100.0) if max_act > 0 else 0.0
            bias_val = float(biases[n_idx]) if biases is not None and n_idx < len(biases) else 0.0

            self.act_table.insertRow(n_idx)
            self.act_table.setItem(n_idx, 0, QTableWidgetItem(f"Neuron #{n_idx}"))
            self.act_table.setItem(n_idx, 1, QTableWidgetItem(f"{val:.4f}"))
            self.act_table.setItem(n_idx, 2, QTableWidgetItem(f"{rel:.1f}%"))
            self.act_table.setItem(n_idx, 3, QTableWidgetItem(f"{bias_val:+.4f}"))

    def _on_arch_table_selected(self):
        row = self.arch_table.currentRow()
        if row >= 0:
            self.select_layer(row)
            self.layer_selected.emit(row)

    def _on_layer_combo_changed(self, idx: int):
        if idx >= 0:
            self.select_layer(idx)
            self.layer_selected.emit(idx)

    def _on_neuron_table_clicked(self, row: int, col: int):
        self.neuron_selected.emit(self._selected_layer_idx, row)
