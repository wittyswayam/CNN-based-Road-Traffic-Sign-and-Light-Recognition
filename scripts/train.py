#!/usr/bin/env python3
"""
TrafficVision-AI :: Training CLI
==================================
Entry point for the full two-phase training pipeline.

Usage:
    python scripts/train.py --backbone efficientnetb3 --epochs-phase1 30
    python -m scripts.train --help
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import get_config
from src.core.logging import configure_logging
from src.data.preprocessing import DatasetBuilder
from src.ml.model import TrafficDetectionModel
from src.ml.trainer import TrainingOrchestrator
from sklearn.model_selection import train_test_split
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a TrafficVision-AI detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--backbone", default="resnet50",
                   choices=["resnet50", "efficientnetb3", "mobilenetv3"],
                   help="CNN backbone architecture")
    p.add_argument("--image-dir", default="data/raw/train/images",
                   help="Training images directory")
    p.add_argument("--label-dir", default="data/raw/train/labels",
                   help="YOLO label files directory")
    p.add_argument("--output-dir", default="models/registry/latest",
                   help="Where to save trained model")
    p.add_argument("--epochs-phase1", type=int, default=30)
    p.add_argument("--epochs-phase2", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--augment", action="store_true", default=True,
                   help="Enable training augmentation")
    p.add_argument("--mlflow", action="store_true",
                   help="Enable MLflow experiment tracking")
    p.add_argument("--mlflow-uri", default="http://localhost:5000")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    log = logging.getLogger("trafficvision.train")

    cfg = get_config()
    log.info("Starting training pipeline", extra={
        "backbone": args.backbone,
        "epochs_p1": args.epochs_phase1,
        "epochs_p2": args.epochs_phase2,
    })

    # ── Data loading ────────────────────────────────────────────────────────
    log.info("Building dataset from %s", args.image_dir)
    builder = DatasetBuilder(
        image_dir=Path(args.image_dir),
        label_dir=Path(args.label_dir),
        augment=args.augment,
    )
    X, y = builder.build()
    log.info("Dataset: %d samples, shape=%s", len(X), X.shape)

    # ── Train/Val/Test split ─────────────────────────────────────────────────
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.176, random_state=42   # 0.176 ≈ 15% of total
    )
    log.info(
        "Split: train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    model = TrafficDetectionModel(
        backbone_name=args.backbone,
        learning_rate=args.lr,
    )

    # ── Orchestrator ────────────────────────────────────────────────────────
    orchestrator = TrainingOrchestrator(
        model=model,
        output_dir=Path(args.output_dir),
        use_mlflow=args.mlflow,
        mlflow_uri=args.mlflow_uri,
    )

    metrics = orchestrator.train(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
        batch_size=args.batch_size,
    )

    log.info("Training complete!")
    log.info("  Test Loss : %.4f", metrics.test_loss)
    log.info("  Test MAE  : %.4f", metrics.test_mae)
    log.info("  Mean IoU  : %.4f", metrics.mean_iou)
    log.info("  Duration  : %.0fs", metrics.training_duration_seconds)
    log.info("  Model saved to: %s", args.output_dir)

    if metrics.mean_iou >= 0.70:
        log.info("✅ Model meets production IoU threshold (>=0.70)")
        sys.exit(0)
    else:
        log.warning("⚠️  Model below production IoU threshold: %.4f < 0.70", metrics.mean_iou)
        sys.exit(1)


if __name__ == "__main__":
    main()
