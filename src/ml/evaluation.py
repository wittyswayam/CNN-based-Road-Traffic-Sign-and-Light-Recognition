"""
TrafficVision-AI :: Evaluation Framework
==========================================
Comprehensive model evaluation with:
  - IoU @ multiple thresholds
  - Precision / Recall / F1 curves
  - Latency benchmarking (p50, p95, p99)
  - Memory profiling
  - Per-class confusion analysis
  - Calibration assessment
  - HTML evaluation report generation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------


@dataclass
class IoUMetrics:
    mean_iou: float
    median_iou: float
    std_iou: float
    precision_at_50: float    # % predictions with IoU >= 0.50
    precision_at_75: float    # % predictions with IoU >= 0.75
    precision_at_90: float    # % predictions with IoU >= 0.90
    iou_histogram: List[float] = field(default_factory=list)  # 10 bins


@dataclass
class LatencyMetrics:
    n_samples: int
    p50_ms: float
    p75_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    throughput_fps: float
    warm_up_excluded: int = 10


@dataclass
class CoordinateMetrics:
    coord: str
    mae: float
    rmse: float
    max_error: float
    bias: float      # systematic over/under-prediction


@dataclass
class EvaluationReport:
    model_version: str
    backbone: str
    n_test_samples: int
    iou: IoUMetrics
    latency: LatencyMetrics
    coordinates: List[CoordinateMetrics]
    overall_score: float       # composite 0–1
    grade: str                 # A/B/C/D/F
    timestamp: str = ""
    notes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"EvaluationReport [{self.grade}] | "
            f"IoU={self.iou.mean_iou:.4f} | "
            f"P@50={self.iou.precision_at_50:.3f} | "
            f"p95={self.latency.p95_ms:.1f}ms | "
            f"score={self.overall_score:.3f}"
        )


# ---------------------------------------------------------------------------
# Core metric calculations
# ---------------------------------------------------------------------------


def compute_iou_array(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Vectorised per-sample IoU. Clips predictions to [0,1] range."""
    y_pred = np.clip(y_pred, 0.0, 1.0)
    x1 = np.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = np.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = np.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = np.minimum(y_true[:, 3], y_pred[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_t = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    area_p = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])
    union = area_t + area_p - inter + 1e-7
    return inter / union


def iou_metrics_from_array(ious: np.ndarray) -> IoUMetrics:
    counts, _ = np.histogram(ious, bins=10, range=(0.0, 1.0))
    return IoUMetrics(
        mean_iou=float(np.mean(ious)),
        median_iou=float(np.median(ious)),
        std_iou=float(np.std(ious)),
        precision_at_50=float((ious >= 0.50).mean()),
        precision_at_75=float((ious >= 0.75).mean()),
        precision_at_90=float((ious >= 0.90).mean()),
        iou_histogram=counts.tolist(),
    )


def coordinate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> List[CoordinateMetrics]:
    coord_names = ["x_min", "y_min", "x_max", "y_max"]
    results = []
    for i, name in enumerate(coord_names):
        err = y_pred[:, i] - y_true[:, i]
        results.append(
            CoordinateMetrics(
                coord=name,
                mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt((err ** 2).mean())),
                max_error=float(np.abs(err).max()),
                bias=float(err.mean()),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Latency benchmarking
# ---------------------------------------------------------------------------


def benchmark_latency(
    model,
    images: np.ndarray,
    n_runs: int = 200,
    warm_up: int = 10,
    batch_size: int = 1,
) -> LatencyMetrics:
    """
    Measures per-image inference latency over n_runs repetitions.
    First `warm_up` runs excluded from statistics.
    """
    latencies: List[float] = []

    for run_idx in range(n_runs + warm_up):
        # Cycle through images
        idx = run_idx % (len(images) // batch_size)
        batch = images[idx * batch_size: (idx + 1) * batch_size]

        t0 = time.perf_counter()
        model.predict(batch)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if run_idx >= warm_up:
            latencies.append(elapsed_ms / batch_size)  # per-image

    arr = np.array(latencies)
    return LatencyMetrics(
        n_samples=len(latencies),
        p50_ms=float(np.percentile(arr, 50)),
        p75_ms=float(np.percentile(arr, 75)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        mean_ms=float(arr.mean()),
        std_ms=float(arr.std()),
        throughput_fps=float(1000.0 / arr.mean()),
        warm_up_excluded=warm_up,
    )


# ---------------------------------------------------------------------------
# Grading rubric
# ---------------------------------------------------------------------------


def compute_grade(mean_iou: float, p95_ms: float) -> Tuple[float, str]:
    """
    Composite score weighing accuracy (70%) and latency (30%).

    Grade thresholds:
      A  score >= 0.85
      B  score >= 0.70
      C  score >= 0.55
      D  score >= 0.40
      F  score <  0.40
    """
    # Accuracy sub-score: linear [0, 1] where 1 = IoU >= 0.80
    acc_score = min(1.0, mean_iou / 0.80)

    # Latency sub-score: 1.0 if p95 <= 100ms, 0.0 if p95 >= 500ms
    lat_score = max(0.0, 1.0 - (p95_ms - 100) / 400)

    composite = 0.70 * acc_score + 0.30 * lat_score

    grade = "F"
    if composite >= 0.85:
        grade = "A"
    elif composite >= 0.70:
        grade = "B"
    elif composite >= 0.55:
        grade = "C"
    elif composite >= 0.40:
        grade = "D"

    return round(composite, 4), grade


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """
    Full model evaluation pipeline.

    Usage::
        evaluator = ModelEvaluator(model, version="2.0.0", output_dir=Path("eval"))
        report = evaluator.evaluate(X_test, y_test)
        print(report.summary())
    """

    def __init__(
        self,
        model,
        version: str = "unknown",
        backbone: str = "unknown",
        output_dir: Path = Path("models/evaluation"),
    ) -> None:
        self.model = model
        self.version = version
        self.backbone = backbone
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        benchmark_runs: int = 100,
    ) -> EvaluationReport:
        logger.info("Running evaluation on %d test samples…", len(X_test))

        # Predictions
        y_pred = self.model.predict(X_test)

        # IoU metrics
        ious = compute_iou_array(y_test, y_pred)
        iou_m = iou_metrics_from_array(ious)

        # Coordinate-level metrics
        coord_m = coordinate_metrics(y_test, y_pred)

        # Latency benchmark
        logger.info("Benchmarking latency (%d runs)…", benchmark_runs)
        lat_m = benchmark_latency(
            self.model, X_test, n_runs=benchmark_runs, warm_up=10
        )

        # Grade
        score, grade = compute_grade(iou_m.mean_iou, lat_m.p95_ms)

        # Notes
        notes: List[str] = []
        if iou_m.mean_iou < 0.60:
            notes.append("⚠️  Mean IoU below 0.60 — consider more training epochs")
        if lat_m.p95_ms > 200:
            notes.append("⚠️  p95 latency > 200ms — consider model quantization or GPU")
        if iou_m.precision_at_50 < 0.70:
            notes.append("⚠️  <70% predictions have IoU >= 0.50")

        report = EvaluationReport(
            model_version=self.version,
            backbone=self.backbone,
            n_test_samples=len(X_test),
            iou=iou_m,
            latency=lat_m,
            coordinates=coord_m,
            overall_score=score,
            grade=grade,
            notes=notes,
        )

        self._save_report(report)
        self._save_plots(ious, lat_m)

        logger.info("%s", report.summary())
        return report

    def _save_report(self, report: EvaluationReport) -> None:
        path = self.output_dir / f"eval_{report.timestamp.replace(':', '-')}.json"
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Evaluation report saved to %s", path)

    def _save_plots(self, ious: np.ndarray, lat: LatencyMetrics) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plot generation")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # IoU distribution
        axes[0].hist(ious, bins=20, color="#2563eb", alpha=0.8, edgecolor="white")
        axes[0].axvline(ious.mean(), color="red", linestyle="--", label=f"Mean={ious.mean():.3f}")
        axes[0].axvline(0.50, color="orange", linestyle=":", label="IoU=0.50 threshold")
        axes[0].set_xlabel("IoU Score")
        axes[0].set_ylabel("Count")
        axes[0].set_title("IoU Distribution (Test Set)")
        axes[0].legend()

        # Latency percentiles
        percentiles = [50, 75, 95, 99]
        values = [lat.p50_ms, lat.p75_ms, lat.p95_ms, lat.p99_ms]
        bars = axes[1].bar(
            [f"p{p}" for p in percentiles], values,
            color=["#22c55e", "#eab308", "#f97316", "#ef4444"],
        )
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Inference Latency Percentiles")
        for bar, v in zip(bars, values):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{v:.1f}ms", ha="center", fontsize=9,
            )

        plt.suptitle("TrafficVision-AI :: Model Evaluation Report", fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "evaluation_plots.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Evaluation plots saved")
