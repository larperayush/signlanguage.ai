"""
Neural Network Core Service & Model Inspection Engine.
Provides authoritative architecture inspection, weight tensor extraction,
exact forward-pass intermediate activation computation, and real-time inference
for SignFlow Studio's Neural Network Visualizer.
"""
import os
import time
import math
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal, Slot
from app.core.config import MODELS_DIR, ROOT_DIR


@dataclass
class LayerMetadata:
    index: int
    name: str
    layer_type: str                   # "Input", "Dense", "Output"
    units: int
    input_dim: int
    output_dim: int
    activation: str                   # "None", "relu", "softmax", "tanh", etc.
    trainable_params: int
    weights_shape: Tuple[int, ...]
    bias_shape: Tuple[int, ...]
    weights_min: float = 0.0
    weights_max: float = 0.0
    weights_mean: float = 0.0
    weights_std: float = 0.0
    bias_min: float = 0.0
    bias_max: float = 0.0
    bias_mean: float = 0.0
    bias_std: float = 0.0


@dataclass
class ModelArchitecture:
    model_id: str
    model_name: str
    file_path: str
    file_size_bytes: int
    family: str                       # e.g. "Multi-Layer Perceptron (Feedforward)"
    framework: str                    # "Scikit-Learn MLPClassifier"
    input_dim: int
    output_dim: int
    classes: List[str]
    total_layers: int
    total_parameters: int
    total_trainable_parameters: int
    layers: List[LayerMetadata]
    status: str = "Loaded"


@dataclass
class ForwardPassResult:
    inference_id: str
    timestamp: float
    input_features: List[float]
    layer_activations: List[List[float]]      # One list of activations per layer
    layer_statistics: List[Dict[str, float]]  # Stats (min, max, mean, std, active_count)
    logits: Dict[str, float]                  # Raw pre-softmax scores per class
    probabilities: Dict[str, float]           # Normalized probability distribution
    predicted_class: str
    confidence: float
    latency_ms: float
    source: str                               # "Camera Live", "Dataset Sample: A", etc.


class NeuralNetworkService(QObject):
    """
    Centralized controller for neural network model inspection and inference.
    Inspects real stored weights, runs mathematically exact forward passes,
    and streams activation data to the 3D scene and technical inspector.
    """
    model_loaded = Signal(object)             # Emits ModelArchitecture
    forward_pass_completed = Signal(object)   # Emits ForwardPassResult
    error_occurred = Signal(str)

    _instance: Optional["NeuralNetworkService"] = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_model: Any = None
        self._current_architecture: Optional[ModelArchitecture] = None
        self._current_model_path: Optional[Path] = None
        self._last_forward_pass: Optional[ForwardPassResult] = None
        self._inference_counter = 0

        # Scan and verify models directory
        self._ensure_default_models()

    @classmethod
    def get_instance(cls) -> "NeuralNetworkService":
        if cls._instance is None:
            cls._instance = NeuralNetworkService()
        return cls._instance

    # =========================================================================
    # MODEL DISCOVERY & INITIALIZATION
    # =========================================================================
    def list_compatible_models(self) -> List[Dict[str, str]]:
        """
        Enumerate all compatible genuine neural network model artifacts in models/.
        Strictly filters for neural network classifiers (skips Random Forests).
        """
        compatible = []
        if not MODELS_DIR.exists():
            return compatible

        for file_path in MODELS_DIR.glob("*.joblib"):
            try:
                # Inspect without full overhead
                obj = joblib.load(file_path)
                from sklearn.neural_network import MLPClassifier
                if isinstance(obj, MLPClassifier):
                    layer_dims = [obj.n_features_in_] + list(obj.hidden_layer_sizes) + [obj.n_outputs_]
                    arch_str = "-".join(map(str, layer_dims))
                    display_name = f"{file_path.stem.replace('_', ' ').title()} (MLP {arch_str})"
                    compatible.append({
                        "id": file_path.stem,
                        "name": display_name,
                        "file_path": str(file_path.resolve()),
                        "architecture_str": arch_str,
                        "parameters": sum(w.size + b.size for w, b in zip(obj.coefs_, obj.intercepts_))
                    })
            except Exception:
                continue

        return compatible

    def _ensure_default_models(self):
        """Ensure genuine feedforward neural network models exist in models/."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        dense_path = MODELS_DIR / "asl_neural_net_dense.joblib"
        compact_path = MODELS_DIR / "asl_neural_net_compact.joblib"
        dataset_path = ROOT_DIR / "static_signs.csv"

        if (not dense_path.exists() or not compact_path.exists()) and dataset_path.exists():
            self._train_default_models(dataset_path)

        # Load first available model by default
        available = self.list_compatible_models()
        if available:
            self.load_model(available[0]["file_path"])

    def _train_default_models(self, dataset_path: Path):
        """Train genuine feedforward neural network models on static_signs.csv."""
        import csv
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import train_test_split

        try:
            with open(dataset_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                rows = [row for row in reader if len(row) == 64]

            if len(rows) < 10:
                return

            X = np.array([[float(v) for v in row[1:]] for row in rows])
            y = np.array([row[0].strip().upper() for row in rows])

            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            dense_path = MODELS_DIR / "asl_neural_net_dense.joblib"
            if not dense_path.exists():
                mlp_dense = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=42)
                mlp_dense.fit(X_train, y_train)
                joblib.dump(mlp_dense, dense_path)

            compact_path = MODELS_DIR / "asl_neural_net_compact.joblib"
            if not compact_path.exists():
                mlp_compact = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=300, random_state=42)
                mlp_compact.fit(X_train, y_train)
                joblib.dump(mlp_compact, compact_path)

        except Exception as e:
            self.error_occurred.emit(f"Failed to train default neural networks: {e}")

    def load_model(self, file_path_str: str) -> bool:
        """
        Load a real neural network model, parse its architecture, and emit metadata.
        """
        p = Path(file_path_str).resolve()
        if not p.exists():
            self.error_occurred.emit(f"Model file does not exist: {p}")
            return False

        try:
            model = joblib.load(p)
            from sklearn.neural_network import MLPClassifier
            if not isinstance(model, MLPClassifier):
                self.error_occurred.emit(f"Model '{p.name}' is not an MLPClassifier. Neural Network Visualizer requires a neural network architecture.")
                return False

            self._current_model = model
            self._current_model_path = p
            self._current_architecture = self._extract_architecture(model, p)
            self._last_forward_pass = None

            self.model_loaded.emit(self._current_architecture)
            return True

        except Exception as e:
            err = f"Failed to load neural network model: {e}"
            self.error_occurred.emit(err)
            return False

    def _extract_architecture(self, model: Any, file_path: Path) -> ModelArchitecture:
        """Extract authoritative technical parameters directly from model tensors."""
        coefs = model.coefs_
        intercepts = model.intercepts_
        classes = [str(c) for c in model.classes_]
        n_features = model.n_features_in_
        n_outputs = model.n_outputs_
        hidden_sizes = list(model.hidden_layer_sizes)

        layers: List[LayerMetadata] = []

        # 1. Input Layer
        layers.append(LayerMetadata(
            index=0,
            name="Input Layer (3D Landmarks)",
            layer_type="Input",
            units=n_features,
            input_dim=n_features,
            output_dim=n_features,
            activation="None",
            trainable_params=0,
            weights_shape=(n_features, n_features),
            bias_shape=(0,)
        ))

        total_params = 0

        # 2. Hidden Layers & Output Layer
        for i, (w, b) in enumerate(zip(coefs, intercepts)):
            is_output = (i == len(coefs) - 1)
            layer_idx = i + 1
            units = w.shape[1]
            act = model.out_activation_ if is_output else model.activation
            layer_type = "Output (Softmax)" if is_output else f"Dense {layer_idx}"
            layer_name = f"Output Layer ({units} Signs)" if is_output else f"Hidden Layer {layer_idx} (Dense)"

            params = w.size + b.size
            total_params += params

            w_flat = w.flatten()
            b_flat = b.flatten()

            layers.append(LayerMetadata(
                index=layer_idx,
                name=layer_name,
                layer_type=layer_type,
                units=units,
                input_dim=w.shape[0],
                output_dim=units,
                activation=act,
                trainable_params=params,
                weights_shape=w.shape,
                bias_shape=b.shape,
                weights_min=float(np.min(w_flat)),
                weights_max=float(np.max(w_flat)),
                weights_mean=float(np.mean(w_flat)),
                weights_std=float(np.std(w_flat)),
                bias_min=float(np.min(b_flat)),
                bias_max=float(np.max(b_flat)),
                bias_mean=float(np.mean(b_flat)),
                bias_std=float(np.std(b_flat))
            ))

        arch = ModelArchitecture(
            model_id=file_path.stem,
            model_name=file_path.stem.replace("_", " ").title(),
            file_path=str(file_path),
            file_size_bytes=file_path.stat().st_size,
            family="Multi-Layer Perceptron (Feedforward)",
            framework="Scikit-Learn MLPClassifier",
            input_dim=n_features,
            output_dim=n_outputs,
            classes=classes,
            total_layers=len(layers),
            total_parameters=total_params,
            total_trainable_parameters=total_params,
            layers=layers,
            status="Loaded"
        )
        return arch

    @property
    def current_architecture(self) -> Optional[ModelArchitecture]:
        return self._current_architecture

    @property
    def last_forward_pass(self) -> Optional[ForwardPassResult]:
        return self._last_forward_pass

    # =========================================================================
    # AUTHORITATIVE WEIGHT INSPECTION API
    # =========================================================================
    def get_layer_weights(self, layer_index: int) -> Optional[np.ndarray]:
        """Retrieve real stored weight tensor W for the specified connection layer."""
        if not self._current_model or layer_index <= 0 or layer_index > len(self._current_model.coefs_):
            return None
        return self._current_model.coefs_[layer_index - 1]

    def get_layer_biases(self, layer_index: int) -> Optional[np.ndarray]:
        """Retrieve real stored bias vector b for the specified layer."""
        if not self._current_model or layer_index <= 0 or layer_index > len(self._current_model.intercepts_):
            return None
        return self._current_model.intercepts_[layer_index - 1]

    def get_layer_weight_histogram(self, layer_index: int, bins: int = 20) -> Dict[str, Any]:
        """Compute exact weight distribution histogram for the layer."""
        w = self.get_layer_weights(layer_index)
        if w is None:
            return {"counts": [], "bin_edges": [], "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        flat = w.flatten()
        counts, edges = np.histogram(flat, bins=bins)
        return {
            "counts": [int(c) for c in counts],
            "bin_edges": [float(e) for e in edges],
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "total_weights": len(flat)
        }

    def get_layer_weight_matrix_summary(self, layer_index: int, max_rows: int = 64, max_cols: int = 64) -> Dict[str, Any]:
        """
        Return bounded weight matrix for heatmap visualization.
        For large tensors, bounds dimensions to avoid unnecessary overhead.
        """
        w = self.get_layer_weights(layer_index)
        if w is None:
            return {"matrix": [], "rows": 0, "cols": 0, "is_truncated": False}

        rows, cols = w.shape
        w_sub = w[:max_rows, :max_cols]
        return {
            "matrix": w_sub.tolist(),
            "rows": rows,
            "cols": cols,
            "sub_rows": w_sub.shape[0],
            "sub_cols": w_sub.shape[1],
            "is_truncated": (rows > max_rows or cols > max_cols),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "mean": float(np.mean(w)),
            "std": float(np.std(w))
        }

    # =========================================================================
    # REAL FORWARD PASS & ACTIVATION EXTRACTION
    # =========================================================================
    def execute_forward_pass(self, input_features: List[float], source_desc: str = "Dataset Sample") -> Optional[ForwardPassResult]:
        """
        Execute an authoritative mathematical forward pass through the loaded neural network.
        Calculates exact per-layer intermediate activations:
          z_1 = x W_0 + b_0,  a_1 = ReLU(z_1)
          z_2 = a_1 W_1 + b_1, a_2 = ReLU(z_2)
          ...
          z_L = a_{L-1} W_{L-1} + b_{L-1} (Logits)
          a_L = Softmax(z_L) (Probabilities)
        Never fabricates activations or randomizes outputs.
        """
        if not self._current_model or not self._current_architecture:
            self.error_occurred.emit("No neural network model is loaded.")
            return None

        expected_dim = self._current_architecture.input_dim
        if len(input_features) != expected_dim:
            self.error_occurred.emit(f"Input dimension mismatch: expected {expected_dim}, got {len(input_features)}.")
            return None

        t0 = time.perf_counter()
        self._inference_counter += 1
        inf_id = f"inf_{self._inference_counter:04d}_{int(time.time())}"

        x = np.array(input_features, dtype=np.float64).reshape(1, -1)

        activations: List[List[float]] = []
        statistics: List[Dict[str, float]] = []

        # Layer 0 (Input)
        a_curr = x
        raw_input = input_features[:]
        activations.append(raw_input)
        statistics.append({
            "min": float(np.min(a_curr)),
            "max": float(np.max(a_curr)),
            "mean": float(np.mean(a_curr)),
            "std": float(np.std(a_curr)),
            "active_count": int(np.count_nonzero(a_curr != 0))
        })

        coefs = self._current_model.coefs_
        intercepts = self._current_model.intercepts_
        act_func = self._current_model.activation

        raw_logits: Optional[np.ndarray] = None

        # Propagate through hidden and output layers
        for i, (w, b) in enumerate(zip(coefs, intercepts)):
            z = np.dot(a_curr, w) + b
            is_last = (i == len(coefs) - 1)

            if not is_last:
                # Hidden layer activation
                if act_func == "relu":
                    a_curr = np.maximum(0, z)
                elif act_func == "tanh":
                    a_curr = np.tanh(z)
                elif act_func == "logistic":
                    a_curr = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
                else:
                    a_curr = z
            else:
                # Output layer: Store exact logits
                raw_logits = z.flatten()
                # Stable Softmax for probability distribution
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                a_curr = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            act_flat = a_curr.flatten().tolist()
            activations.append(act_flat)

            act_arr = np.array(act_flat)
            statistics.append({
                "min": float(np.min(act_arr)),
                "max": float(np.max(act_arr)),
                "mean": float(np.mean(act_arr)),
                "std": float(np.std(act_arr)),
                "active_count": int(np.count_nonzero(act_arr > 0.001))
            })

        latency = (time.perf_counter() - t0) * 1000.0

        # Classes and prediction
        classes = self._current_architecture.classes
        output_probs = activations[-1]
        probs_dict = {c: float(p) for c, p in zip(classes, output_probs)}

        logits_dict = {}
        if raw_logits is not None:
            logits_dict = {c: float(l) for c, l in zip(classes, raw_logits)}

        best_idx = int(np.argmax(output_probs))
        pred_class = classes[best_idx]
        confidence = float(output_probs[best_idx])

        result = ForwardPassResult(
            inference_id=inf_id,
            timestamp=time.time(),
            input_features=raw_input,
            layer_activations=activations,
            layer_statistics=statistics,
            logits=logits_dict,
            probabilities=probs_dict,
            predicted_class=pred_class,
            confidence=confidence,
            latency_ms=latency,
            source=source_desc
        )

        self._last_forward_pass = result
        self.forward_pass_completed.emit(result)
        return result

    # =========================================================================
    # DATASET SAMPLES REPOSITORY
    # =========================================================================
    def get_dataset_sample_signs(self) -> List[str]:
        """Available signs in static_signs.csv."""
        return ["A", "B", "C", "D", "E", "F"]

    def load_dataset_sample(self, target_sign: str = "A") -> Optional[Tuple[List[float], str]]:
        """
        Load an authentic 63-feature coordinate vector from static_signs.csv
        corresponding to the specified sign.
        """
        csv_path = ROOT_DIR / "static_signs.csv"
        if not csv_path.exists():
            return None

        import csv
        target_sign = target_sign.strip().upper()

        try:
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) == 64 and row[0].strip().upper() == target_sign:
                        features = [float(v) for v in row[1:]]
                        return features, target_sign

            # Fallback to first row
            with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                first_row = next(reader, None)
                if first_row and len(first_row) == 64:
                    return [float(v) for v in first_row[1:]], first_row[0].strip().upper()

        except Exception as e:
            self.error_occurred.emit(f"Failed to load dataset sample for sign {target_sign}: {e}")

        return None
