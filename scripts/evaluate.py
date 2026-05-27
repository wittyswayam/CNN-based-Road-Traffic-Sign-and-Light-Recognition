"""
TrafficVision-AI :: Evaluation & Benchmark CLI
===============================================
Runs the full evaluation framework against a trained model.

Usage::
    python scripts/evaluate.py --model-path models/registry/latest
    python scripts/evaluate.py --model-path models/registry/v2.0.0 --benchmark-runs 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging import configure_logging
from src.ml.evaluation import ModelEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TrafficVision-AI model")
    parser.add_argument("--model-path", required=True, help="Path to saved model directory")
    parser.add_argument("--test-data", default="data/processed", help="Processed test data directory")
    parser.add_argument("--output-dir", default="models/evaluation", help="Evaluation output directory")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Latency benchmark iterations")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    log = logging.getLogger("trafficvision.evaluate")

    import numpy as np

    log.info("Loading model from %s", args.model_path)

    try:
        from src.ml.model import TrafficDetectionModel
        model = TrafficDetectionModel(backbone_name=args.backbone)
        model.load(Path(args.model_path))
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        sys.exit(1)

    # Load test data
    test_dir = Path(args.test_data)
    img_files = sorted(test_dir.glob("images_*.npy"))
    box_files = sorted(test_dir.glob("bboxes_*.npy"))

    if not img_files:
        log.warning("No processed shards found — generating synthetic test data")
        np.random.seed(99)
        X_test = np.random.rand(200, 224, 224, 3).astype(np.float32)
        y_test = np.random.rand(200, 4).astype(np.float32)
        y_test[:, 2] = np.maximum(y_test[:, 0] + 0.1, y_test[:, 2])
        y_test[:, 3] = np.maximum(y_test[:, 1] + 0.1, y_test[:, 3])
    else:
        X_test = np.concatenate([np.load(f) for f in img_files])
        y_test = np.concatenate([np.load(f) for f in box_files])
        log.info("Loaded %d test samples from %s", len(X_test), test_dir)

    # Run evaluation
    evaluator = ModelEvaluator(
        model=model,
        version="2.0.0",
        backbone=args.backbone,
        output_dir=Path(args.output_dir),
    )

    report = evaluator.evaluate(
        X_test, y_test, benchmark_runs=args.benchmark_runs
    )

    # Print summary
    print("\n" + "=" * 55)
    print("  TrafficVision-AI :: Evaluation Results")
    print("=" * 55)
    print(f"  Grade          : {report.grade}")
    print(f"  Overall Score  : {report.overall_score:.4f}")
    print(f"  Mean IoU       : {report.iou.mean_iou:.4f}")
    print(f"  Precision@0.50 : {report.iou.precision_at_50:.3f}")
    print(f"  Precision@0.75 : {report.iou.precision_at_75:.3f}")
    print(f"  p50 Latency    : {report.latency.p50_ms:.1f} ms")
    print(f"  p95 Latency    : {report.latency.p95_ms:.1f} ms")
    print(f"  Throughput     : {report.latency.throughput_fps:.1f} FPS")

    if report.notes:
        print("\n  Notes:")
        for note in report.notes:
            print(f"    {note}")
    print("=" * 55)

    # Exit code based on grade
    if report.grade in ("A", "B"):
        log.info("Evaluation PASSED — grade %s", report.grade)
        sys.exit(0)
    else:
        log.warning("Evaluation WARNING — grade %s (below production threshold)", report.grade)
        sys.exit(1)


if __name__ == "__main__":
    main()
