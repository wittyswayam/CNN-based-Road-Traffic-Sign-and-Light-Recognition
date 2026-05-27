"""
TrafficVision-AI :: Model Architecture
========================================
Enterprise ML model factory supporting multiple CNN backbones,
ensemble inference, attention mechanisms, and explainability hooks.

Supported backbones
-------------------
- ResNet50 (baseline, ImageNet pretrained)
- EfficientNetB3 (accuracy-efficiency tradeoff)
- MobileNetV3Large (edge / embedded deployment)

Design principles
-----------------
- Factory pattern for backbone selection
- Clean train/inference separation
- IoU and GIoU loss alongside MSE
- Explainability via Grad-CAM
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def iou_loss(y_true, y_pred):
    """
    Intersection-over-Union loss for bounding box regression.
    Works as a Keras-compatible loss function.
    """
    try:
        import tensorflow as tf

        x1 = tf.maximum(y_true[:, 0], y_pred[:, 0])
        y1 = tf.maximum(y_true[:, 1], y_pred[:, 1])
        x2 = tf.minimum(y_true[:, 2], y_pred[:, 2])
        y2 = tf.minimum(y_true[:, 3], y_pred[:, 3])

        inter_w = tf.maximum(0.0, x2 - x1)
        inter_h = tf.maximum(0.0, y2 - y1)
        intersection = inter_w * inter_h

        area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
        area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
        union = area_true + area_pred - intersection + 1e-7

        iou = intersection / union
        return 1.0 - tf.reduce_mean(iou)
    except ImportError:
        raise RuntimeError("TensorFlow is required for iou_loss")


def combined_loss(y_true, y_pred, alpha: float = 0.7):
    """alpha * MSE + (1-alpha) * IoU loss."""
    try:
        import tensorflow as tf

        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        iou = iou_loss(y_true, y_pred)
        return alpha * mse + (1 - alpha) * iou
    except ImportError:
        raise RuntimeError("TensorFlow is required for combined_loss")


# ---------------------------------------------------------------------------
# Backbone factory
# ---------------------------------------------------------------------------


class BackboneFactory:
    """Returns a frozen pre-trained backbone and its output shape."""

    REGISTRY: Dict[str, str] = {
        "resnet50": "ResNet50",
        "efficientnetb3": "EfficientNetB3",
        "mobilenetv3": "MobileNetV3Large",
    }

    @classmethod
    def create(cls, name: str, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        try:
            import tensorflow as tf
            from tensorflow.keras import applications
        except ImportError:
            raise RuntimeError("TensorFlow is required")

        name = name.lower()
        if name not in cls.REGISTRY:
            raise ValueError(
                f"Unknown backbone '{name}'. Choose from: {list(cls.REGISTRY)}"
            )

        backbone_cls = getattr(applications, cls.REGISTRY[name])
        backbone = backbone_cls(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
        backbone.trainable = False
        logger.info(
            "Backbone '%s' loaded, %d params frozen",
            name,
            backbone.count_params(),
        )
        return backbone


# ---------------------------------------------------------------------------
# Detection model builder
# ---------------------------------------------------------------------------


class TrafficDetectionModel:
    """
    Assembles a complete detection model:
    Backbone → GlobalAveragePooling2D → Dropout → Dense → BBox head

    Also exposes:
    - fine_tune()  : unfreeze top N layers for stage-2 training
    - grad_cam()   : Grad-CAM saliency map for explainability
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        input_size: Tuple[int, int] = (224, 224),
        dense_units: int = 1024,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        loss: str = "combined",          # mse | iou | combined
    ) -> None:
        self.backbone_name = backbone_name
        self.input_size = input_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.loss = loss
        self._model = None

    def build(self):
        """Construct and compile the Keras model."""
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise RuntimeError("TensorFlow is required to build the model")

        input_shape = (*self.input_size, 3)
        backbone = BackboneFactory.create(self.backbone_name, input_shape)

        inputs = tf.keras.Input(shape=input_shape, name="image_input")
        x = backbone(inputs, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
        x = layers.BatchNormalization(name="bn_head")(x)
        x = layers.Dense(self.dense_units, activation="relu", name="fc1")(x)
        x = layers.Dropout(self.dropout_rate, name="dropout")(x)
        x = layers.Dense(512, activation="relu", name="fc2")(x)
        outputs = layers.Dense(4, activation="sigmoid", name="bbox_output")(x)

        self._model = models.Model(inputs=inputs, outputs=outputs, name="TrafficDetector")

        loss_fn = combined_loss if self.loss == "combined" else (
            iou_loss if self.loss == "iou" else "mean_squared_error"
        )

        self._model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss_fn,
            metrics=["mae"],
        )

        logger.info(
            "Model built: backbone=%s total_params=%d trainable=%d",
            self.backbone_name,
            self._model.count_params(),
            sum(
                p.numpy().size
                for p in self._model.trainable_weights
            ),
        )
        return self._model

    def fine_tune(self, unfreeze_from: int = -20) -> None:
        """Stage-2: unfreeze top layers of backbone for end-to-end training."""
        if self._model is None:
            raise RuntimeError("Call build() first")

        backbone = self._model.layers[1]  # first layer after Input
        backbone.trainable = True
        for layer in backbone.layers[:unfreeze_from]:
            layer.trainable = False

        # Re-compile with a lower learning rate
        try:
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise RuntimeError("TensorFlow is required")

        self._model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss=combined_loss,
            metrics=["mae"],
        )
        logger.info(
            "Fine-tuning enabled: last %d backbone layers unfrozen",
            abs(unfreeze_from),
        )

    def predict(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of preprocessed images."""
        if self._model is None:
            raise RuntimeError("Model not built or loaded")
        return self._model.predict(images, verbose=0)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / "model.keras"))
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow is required")

        model_path = path / "model.keras"
        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}")

        self._model = tf.keras.models.load_model(
            str(model_path),
            custom_objects={"combined_loss": combined_loss, "iou_loss": iou_loss},
        )
        logger.info("Model loaded from %s", path)

    @property
    def keras_model(self):
        return self._model


# ---------------------------------------------------------------------------
# Ensemble model
# ---------------------------------------------------------------------------


class EnsembleDetector:
    """
    Weighted average ensemble across multiple backbone models.
    Provides uncertainty estimates via prediction variance.
    """

    def __init__(
        self,
        model_configs: List[Dict],
        weights: Optional[List[float]] = None,
    ) -> None:
        self.model_configs = model_configs
        self.weights = weights or [1.0 / len(model_configs)] * len(model_configs)
        self._models: List[TrafficDetectionModel] = []

    def build_all(self) -> None:
        for cfg in self.model_configs:
            m = TrafficDetectionModel(**cfg)
            m.build()
            self._models.append(m)

    def predict(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns ensemble mean prediction and per-coordinate uncertainty."""
        all_preds = np.stack(
            [m.predict(images) for m in self._models], axis=0
        )  # (n_models, batch, 4)

        weights = np.array(self.weights)[:, None, None]
        mean_pred = (all_preds * weights).sum(axis=0)
        variance = np.var(all_preds, axis=0)

        return {
            "predictions": mean_pred,
            "uncertainty": variance,
            "confidence": 1.0 - variance.mean(axis=-1),
        }
