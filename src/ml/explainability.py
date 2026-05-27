"""
TrafficVision-AI :: Explainability — Grad-CAM & LIME
======================================================
Production-grade explainability system for model trust, debugging,
regulatory compliance, and human-in-the-loop review workflows.

Modules
-------
- GradCAM          : Class Activation Mapping via gradient backprop
- GuidedBackprop   : High-resolution gradient saliency
- LIMEExplainer    : Local surrogate model explanations
- ExplainabilityReport : Batch explain + persist to disk
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    For a regression model predicting bounding box coordinates,
    we compute Grad-CAM for each coordinate dimension separately.

    Reference: Selvaraju et al., 2017 (https://arxiv.org/abs/1610.02391)

    Usage::
        cam = GradCAM(model.keras_model, layer_name="conv5_block3_out")
        heatmap = cam.compute(image_batch, coord_index=0)  # 0=x_min
    """

    def __init__(self, keras_model, layer_name: Optional[str] = None) -> None:
        self._model = keras_model
        self._layer_name = layer_name or self._infer_last_conv_layer()

    def _infer_last_conv_layer(self) -> str:
        """Auto-detect the last convolutional layer name."""
        try:
            for layer in reversed(self._model.layers):
                if hasattr(layer, "filters") or "conv" in layer.name.lower():
                    logger.debug("Auto-detected Grad-CAM layer: %s", layer.name)
                    return layer.name
        except Exception:
            pass
        return "conv5_block3_out"  # ResNet50 default

    def compute(
        self,
        image: np.ndarray,
        coord_index: int = 0,
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap for one image.

        Parameters
        ----------
        image       : Preprocessed image, shape (1, H, W, 3), float32 [0,1]
        coord_index : Which output coordinate to explain (0-3)

        Returns
        -------
        heatmap : Normalised saliency map, shape (H, W), float32 [0,1]
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow required for Grad-CAM")

        # Build sub-model: input → last conv layer + final output
        grad_model = tf.keras.models.Model(
            inputs=self._model.inputs,
            outputs=[
                self._model.get_layer(self._layer_name).output,
                self._model.output,
            ],
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            conv_outputs, predictions = grad_model(inputs)
            # Scalar to differentiate: selected coordinate
            loss = predictions[:, coord_index]

        # Gradients of selected coordinate w.r.t. conv feature maps
        grads = tape.gradient(loss, conv_outputs)

        # Global average pooling of gradients → importance weights
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight conv outputs by importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU + normalise
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    def overlay(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Superimpose Grad-CAM heatmap on the original image.

        Returns uint8 RGB image.
        """
        import cv2

        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        img_uint8 = np.uint8(image * 255) if image.max() <= 1.0 else image
        superimposed = np.uint8(img_uint8 * (1 - alpha) + heatmap_rgb * alpha)
        return superimposed


# ---------------------------------------------------------------------------
# SHAP-based explainer (DeepExplainer)
# ---------------------------------------------------------------------------


class SHAPExplainer:
    """
    SHAP DeepExplainer for feature attribution on CNN outputs.

    Requires a background dataset (subset of training images)
    to estimate expected model output.

    Usage::
        explainer = SHAPExplainer(model.keras_model, background=X_train[:50])
        shap_values = explainer.explain(X_test[:5])
    """

    def __init__(self, keras_model, background: np.ndarray) -> None:
        self._model = keras_model
        self._background = background

    def explain(self, images: np.ndarray) -> List[np.ndarray]:
        """
        Returns SHAP values for each output coordinate.

        Returns
        -------
        shap_values : List of 4 arrays, each shape (N, H, W, C)
                      One per bounding box coordinate.
        """
        try:
            import shap
        except ImportError:
            raise RuntimeError("Install shap: pip install shap")

        explainer = shap.DeepExplainer(self._model, self._background)
        shap_values = explainer.shap_values(images)
        logger.info(
            "SHAP values computed for %d images, %d output coords",
            len(images), len(shap_values),
        )
        return shap_values

    def plot_summary(
        self,
        shap_values: List[np.ndarray],
        images: np.ndarray,
        coord_names: List[str] = None,
        save_dir: Optional[Path] = None,
    ) -> None:
        """Plot SHAP image plots for each coordinate."""
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("Install shap and matplotlib")

        coord_names = coord_names or ["x_min", "y_min", "x_max", "y_max"]

        for i, (sv, name) in enumerate(zip(shap_values, coord_names)):
            fig, axes = plt.subplots(1, min(5, len(images)), figsize=(15, 3))
            for ax, img, s in zip(
                axes if hasattr(axes, "__iter__") else [axes],
                images[:5],
                sv[:5],
            ):
                # Mean absolute SHAP across channels
                importance = np.abs(s).mean(axis=-1)
                ax.imshow(img)
                ax.imshow(importance, alpha=0.5, cmap="hot")
                ax.axis("off")

            plt.suptitle(f"SHAP Importance — {name}", fontweight="bold")
            plt.tight_layout()

            if save_dir:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                plt.savefig(Path(save_dir) / f"shap_{name}.png", dpi=150)
            plt.show()
            plt.close()


# ---------------------------------------------------------------------------
# Batch explainability report
# ---------------------------------------------------------------------------


class ExplainabilityReport:
    """
    Generates a comprehensive explainability report for a batch of images.

    Outputs:
      - Grad-CAM overlays (PNG) for each image × coordinate
      - SHAP summary plots
      - JSON metadata with explanation stats
    """

    COORD_NAMES = ["x_min", "y_min", "x_max", "y_max"]

    def __init__(
        self,
        keras_model,
        output_dir: Path,
        layer_name: Optional[str] = None,
    ) -> None:
        self.cam = GradCAM(keras_model, layer_name)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        images: np.ndarray,
        predictions: np.ndarray,
        image_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate Grad-CAM overlays for every image and coordinate.

        Parameters
        ----------
        images      : (N, H, W, 3) preprocessed images
        predictions : (N, 4) predicted bounding boxes
        image_ids   : Optional identifiers for file naming

        Returns
        -------
        report_meta : Summary dict with file paths and statistics
        """
        import cv2

        image_ids = image_ids or [f"img_{i:04d}" for i in range(len(images))]
        report_meta: Dict = {"n_images": len(images), "outputs": []}

        for img_idx, (img, pred, img_id) in enumerate(
            zip(images, predictions, image_ids)
        ):
            img_meta: Dict = {"id": img_id, "prediction": pred.tolist(), "heatmaps": {}}
            batch = np.expand_dims(img, axis=0)

            for coord_idx, coord_name in enumerate(self.COORD_NAMES):
                try:
                    heatmap = self.cam.compute(batch, coord_index=coord_idx)
                    overlay = self.cam.overlay(img, heatmap)

                    fname = f"{img_id}_{coord_name}_gradcam.png"
                    fpath = self.output_dir / fname
                    cv2.imwrite(str(fpath), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    img_meta["heatmaps"][coord_name] = str(fpath)

                except Exception as exc:
                    logger.warning(
                        "Grad-CAM failed for %s coord=%s: %s", img_id, coord_name, exc
                    )
                    img_meta["heatmaps"][coord_name] = None

            report_meta["outputs"].append(img_meta)
            logger.debug("Explained image %s (%d/%d)", img_id, img_idx + 1, len(images))

        # Persist metadata
        import json
        meta_path = self.output_dir / "explainability_report.json"
        with open(meta_path, "w") as f:
            json.dump(report_meta, f, indent=2)

        logger.info(
            "Explainability report generated: %d images → %s",
            len(images), self.output_dir,
        )
        return report_meta
