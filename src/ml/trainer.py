"""
TrafficVision-AI :: Training Orchestrator
==========================================
Full ML lifecycle management:
  Phase 1 – Frozen backbone training (feature extraction)
  Phase 2 – Fine-tuning (end-to-end, low LR)
  MLflow experiment tracking
  Early stopping, LR scheduling, checkpoint management
  Post-training evaluation & metric persistence
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingMetrics:
    run_id: str
    backbone: str
    epochs_phase1: int
    epochs_phase2: int
    best_val_loss: float
    best_val_mae: float
    test_loss: float
    test_mae: float
    mean_iou: float
    training_duration_seconds: float
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class InferenceMetrics:
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_fps: float
    mean_iou: float


# ---------------------------------------------------------------------------
# IoU metric helpers
# ---------------------------------------------------------------------------


def compute_iou(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vectorised IoU across a batch. Returns per-sample IoU array."""
    x1 = np.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = np.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = np.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = np.minimum(y_true[:, 3], y_pred[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_true = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_pred = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union = area_true + area_pred - intersection + 1e-7

    return intersection / union


def compute_precision_at_iou(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5
) -> float:
    ious = compute_iou(y_true, y_pred)
    return float((ious >= threshold).mean())


# ---------------------------------------------------------------------------
# Training orchestrator
# ---------------------------------------------------------------------------


class TrainingOrchestrator:
    """
    Manages the two-phase training strategy with callbacks,
    MLflow tracking, and artifact persistence.
    """

    def __init__(
        self,
        model,               # TrafficDetectionModel instance
        output_dir: Path,
        experiment_name: str = "traffic-detection",
        mlflow_uri: str = "http://localhost:5000",
        use_mlflow: bool = False,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.mlflow_uri = mlflow_uri
        self.use_mlflow = use_mlflow
        self._run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ------------------------------------------------------------------
    # Keras callbacks
    # ------------------------------------------------------------------

    def _build_callbacks(self, phase: int):
        try:
            from tensorflow.keras.callbacks import (
                EarlyStopping,
                ModelCheckpoint,
                ReduceLROnPlateau,
                TensorBoard,
                CSVLogger,
            )
        except ImportError:
            return []

        checkpoint_path = str(
            self.output_dir / f"checkpoints/phase{phase}" / "best_model.keras"
        )
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

        return [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
            TensorBoard(
                log_dir=str(self.output_dir / f"logs/phase{phase}"),
                histogram_freq=1,
            ),
            CSVLogger(str(self.output_dir / f"logs/phase{phase}/training.csv")),
        ]

    # ------------------------------------------------------------------
    # Training phases
    # ------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs_phase1: int = 30,
        epochs_phase2: int = 20,
        batch_size: int = 32,
    ) -> TrainingMetrics:
        """Execute two-phase training and return consolidated metrics."""
        start_time = time.perf_counter()

        logger.info("=== Phase 1: Feature Extraction (frozen backbone) ===")
        keras_model = self.model.build()

        history1 = keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_phase1,
            batch_size=batch_size,
            callbacks=self._build_callbacks(phase=1),
            verbose=1,
        )

        best_val_loss_p1 = min(history1.history.get("val_loss", [float("inf")]))
        logger.info("Phase 1 complete. Best val_loss=%.4f", best_val_loss_p1)

        logger.info("=== Phase 2: Fine-tuning (unfrozen top backbone layers) ===")
        self.model.fine_tune(unfreeze_from=-20)

        history2 = keras_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs_phase2,
            batch_size=batch_size // 2,           # smaller batch for stability
            callbacks=self._build_callbacks(phase=2),
            verbose=1,
        )

        best_val_loss = min(
            min(history2.history.get("val_loss", [float("inf")])),
            best_val_loss_p1,
        )
        best_val_mae = min(history2.history.get("val_mae", [float("inf")]))

        # ----- Evaluation -----
        test_loss, test_mae = keras_model.evaluate(X_test, y_test, verbose=0)
        y_pred = self.model.predict(X_test)
        mean_iou = float(compute_iou(y_test, y_pred).mean())

        elapsed = time.perf_counter() - start_time

        metrics = TrainingMetrics(
            run_id=self._run_id,
            backbone=self.model.backbone_name,
            epochs_phase1=len(history1.history.get("loss", [])),
            epochs_phase2=len(history2.history.get("loss", [])),
            best_val_loss=float(best_val_loss),
            best_val_mae=float(best_val_mae),
            test_loss=float(test_loss),
            test_mae=float(test_mae),
            mean_iou=mean_iou,
            training_duration_seconds=elapsed,
        )

        self._persist_metrics(metrics)
        self._persist_model()

        if self.use_mlflow:
            self._log_to_mlflow(metrics)

        logger.info("Training complete. Mean IoU=%.4f", mean_iou)
        return metrics

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_metrics(self, metrics: TrainingMetrics) -> None:
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)
        logger.info("Metrics saved to %s", path)

    def _persist_model(self) -> None:
        self.model.save(self.output_dir / "final_model")

    def _log_to_mlflow(self, metrics: TrainingMetrics) -> None:
        try:
            import mlflow

            mlflow.set_tracking_uri(self.mlflow_uri)
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run(run_name=self._run_id):
                mlflow.log_params({
                    "backbone": metrics.backbone,
                    "epochs_phase1": metrics.epochs_phase1,
                    "epochs_phase2": metrics.epochs_phase2,
                })
                mlflow.log_metrics({
                    "best_val_loss": metrics.best_val_loss,
                    "test_loss": metrics.test_loss,
                    "test_mae": metrics.test_mae,
                    "mean_iou": metrics.mean_iou,
                })
                mlflow.log_artifact(str(self.output_dir / "metrics.json"))
                mlflow.keras.log_model(
                    self.model.keras_model, "traffic_detector"
                )
            logger.info("Run logged to MLflow: %s", self._run_id)
        except ImportError:
            logger.warning("mlflow not installed; skipping experiment tracking")
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)
