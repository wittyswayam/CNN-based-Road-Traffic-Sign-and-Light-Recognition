"""
TrafficVision-AI :: Hyperparameter Optimization
=================================================
Bayesian hyperparameter search using Optuna with MLflow integration.

Search space
------------
- backbone            : resnet50 | efficientnetb3 | mobilenetv3
- learning_rate       : log-uniform [1e-5, 1e-3]
- batch_size          : 16 | 32 | 64
- dense_units         : 512 | 1024 | 2048
- dropout_rate        : uniform [0.1, 0.5]
- loss_alpha          : uniform [0.5, 0.9]   (weight of MSE vs IoU)
- unfreeze_from       : -10 | -20 | -40       (fine-tune layers)

Strategy
--------
  Trial 1–N  → Optuna TPE sampler selects hyperparams
  Each trial  → Phase-1 training for max 15 epochs (pruned if not improving)
  Best trial  → Full 2-phase training stored in MLflow
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------


SEARCH_SPACE: Dict = {
    "backbone": ["resnet50", "efficientnetb3", "mobilenetv3"],
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-3},
    "batch_size": [16, 32, 64],
    "dense_units": [512, 1024, 2048],
    "dropout_rate": {"type": "uniform", "low": 0.1, "high": 0.5},
    "loss_alpha": {"type": "uniform", "low": 0.5, "high": 0.9},
    "unfreeze_from": [-10, -20, -40],
}


@dataclass
class TrialResult:
    trial_number: int
    params: Dict
    val_loss: float
    mean_iou: float
    duration_seconds: float
    pruned: bool = False


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------


def build_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 15,
    mlflow_uri: Optional[str] = None,
):
    """
    Returns an Optuna objective function closed over the training data.

    Usage::
        import optuna
        study = optuna.create_study(direction="minimize")
        study.optimize(build_objective(X_tr, y_tr, X_val, y_val), n_trials=30)
    """
    def objective(trial) -> float:
        try:
            import tensorflow as tf
            from src.ml.model import TrafficDetectionModel
            from src.ml.trainer import compute_iou

            # Sample hyperparameters
            backbone = trial.suggest_categorical(
                "backbone", SEARCH_SPACE["backbone"]
            )
            lr = trial.suggest_float(
                "learning_rate",
                SEARCH_SPACE["learning_rate"]["low"],
                SEARCH_SPACE["learning_rate"]["high"],
                log=True,
            )
            batch_size = trial.suggest_categorical(
                "batch_size", SEARCH_SPACE["batch_size"]
            )
            dense_units = trial.suggest_categorical(
                "dense_units", SEARCH_SPACE["dense_units"]
            )
            dropout_rate = trial.suggest_float(
                "dropout_rate",
                SEARCH_SPACE["dropout_rate"]["low"],
                SEARCH_SPACE["dropout_rate"]["high"],
            )

            # Build model
            model = TrafficDetectionModel(
                backbone_name=backbone,
                learning_rate=lr,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
            )
            keras_model = model.build()

            best_val_loss = float("inf")

            for epoch in range(max_epochs):
                history = keras_model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=1,
                    batch_size=batch_size,
                    verbose=0,
                )
                val_loss = history.history["val_loss"][-1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # Optuna pruning — stop unpromising trials early
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logger.info(
                        "Trial %d pruned at epoch %d (val_loss=%.4f)",
                        trial.number, epoch, val_loss,
                    )
                    raise __import__("optuna").exceptions.TrialPruned()

            # IoU score for this trial
            y_pred = model.predict(X_val)
            mean_iou = float(compute_iou(y_val, y_pred).mean())

            logger.info(
                "Trial %d complete: backbone=%s lr=%.2e val_loss=%.4f iou=%.4f",
                trial.number, backbone, lr, best_val_loss, mean_iou,
            )

            # Log to MLflow if configured
            if mlflow_uri:
                _log_trial_to_mlflow(trial, best_val_loss, mean_iou, mlflow_uri)

            return best_val_loss

        except Exception as exc:
            if "TrialPruned" in type(exc).__name__:
                raise
            logger.error("Trial %d failed: %s", trial.number, exc, exc_info=True)
            return float("inf")

    return objective


def _log_trial_to_mlflow(trial, val_loss: float, iou: float, uri: str) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("trafficvision-hpo")
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(trial.params)
            mlflow.log_metrics({"val_loss": val_loss, "mean_iou": iou})
    except Exception as exc:
        logger.warning("MLflow logging failed for trial %d: %s", trial.number, exc)


# ---------------------------------------------------------------------------
# HPO runner
# ---------------------------------------------------------------------------


class HPORunner:
    """
    Manages full hyperparameter optimization lifecycle.

    Example::
        runner = HPORunner(n_trials=30, output_dir=Path("models/hpo"))
        best = runner.run(X_train, y_train, X_val, y_val)
        print(best)
    """

    def __init__(
        self,
        n_trials: int = 30,
        max_epochs_per_trial: int = 15,
        output_dir: Path = Path("models/hpo"),
        mlflow_uri: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.max_epochs_per_trial = max_epochs_per_trial
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_uri = mlflow_uri
        self.seed = seed

    def run(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise RuntimeError("Install optuna: pip install optuna")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            study_name="trafficvision-hpo",
            direction="minimize",
            sampler=sampler,
            pruner=pruner,
        )

        objective = build_objective(
            X_train, y_train, X_val, y_val,
            max_epochs=self.max_epochs_per_trial,
            mlflow_uri=self.mlflow_uri,
        )

        logger.info("Starting HPO: %d trials, %d epochs each", self.n_trials, self.max_epochs_per_trial)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        best = {
            "best_trial": study.best_trial.number,
            "best_val_loss": study.best_value,
            "best_params": study.best_params,
            "n_trials_completed": len(study.trials),
            "n_trials_pruned": sum(
                1 for t in study.trials
                if t.state.name == "PRUNED"
            ),
        }

        # Persist results
        results_path = self.output_dir / "hpo_results.json"
        with open(results_path, "w") as f:
            json.dump(best, f, indent=2)

        logger.info("HPO complete. Best params: %s", best["best_params"])
        logger.info("Results saved to %s", results_path)

        return best
