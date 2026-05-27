"""
TrafficVision-AI :: Model Monitoring & Drift Detection
=======================================================
Production monitoring pipeline:
  - Feature drift detection (KL divergence, PSI)
  - Prediction drift monitoring
  - Performance degradation alerts
  - Prometheus metrics exposition
  - Sliding window statistics
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Statistical drift metrics
# ---------------------------------------------------------------------------


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """
    PSI < 0.1   : no significant change
    PSI 0.1–0.2 : moderate change, monitor closely
    PSI > 0.2   : significant drift, retrain recommended
    """
    breakpoints = np.linspace(0, 1, bins + 1)

    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    # Avoid log(0)
    expected_pct = np.where(expected_pct == 0, 1e-6, expected_pct)
    actual_pct = np.where(actual_pct == 0, 1e-6, actual_pct)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """KL divergence D(P || Q) between two empirical distributions."""
    breakpoints = np.linspace(
        min(p.min(), q.min()), max(p.max(), q.max()), bins + 1
    )
    p_hist = np.histogram(p, bins=breakpoints)[0].astype(float)
    q_hist = np.histogram(q, bins=breakpoints)[0].astype(float)

    p_hist = p_hist / (p_hist.sum() + 1e-9) + 1e-9
    q_hist = q_hist / (q_hist.sum() + 1e-9) + 1e-9

    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


# ---------------------------------------------------------------------------
# Drift alert thresholds
# ---------------------------------------------------------------------------


@dataclass
class DriftThresholds:
    psi_warning: float = 0.10
    psi_critical: float = 0.20
    kl_warning: float = 0.05
    kl_critical: float = 0.15
    latency_p95_ms_warning: float = 200.0
    latency_p95_ms_critical: float = 500.0
    error_rate_warning: float = 0.02
    error_rate_critical: float = 0.05


# ---------------------------------------------------------------------------
# Monitoring window
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    timestamp: float
    bbox_pred: List[float]   # [x_min, y_min, x_max, y_max]
    latency_ms: float
    error: bool = False


@dataclass
class DriftReport:
    window_size: int
    psi_scores: Dict[str, float]        # per coordinate
    kl_scores: Dict[str, float]
    mean_latency_ms: float
    p95_latency_ms: float
    error_rate: float
    alerts: List[str]
    timestamp: str = ""
    overall_drift_level: str = "none"   # none | warning | critical

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Monitor class
# ---------------------------------------------------------------------------


class ModelMonitor:
    """
    Collects inference telemetry and computes drift metrics against
    a reference (baseline) distribution captured at training time.
    """

    COORD_NAMES = ["x_min", "y_min", "x_max", "y_max"]

    def __init__(
        self,
        reference_predictions: Optional[np.ndarray] = None,
        window_size: int = 1000,
        thresholds: Optional[DriftThresholds] = None,
        output_dir: Optional[Path] = None,
    ) -> None:
        self.window_size = window_size
        self.thresholds = thresholds or DriftThresholds()
        self.output_dir = output_dir or Path("monitoring/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reference distribution (from test set predictions at training time)
        self._reference: Optional[np.ndarray] = reference_predictions

        # Sliding window of recent predictions
        self._window: Deque[PredictionRecord] = deque(maxlen=window_size)

    def set_reference(self, predictions: np.ndarray) -> None:
        """Register baseline prediction distribution."""
        self._reference = predictions
        logger.info(
            "Reference distribution set: %d samples, shape=%s",
            len(predictions), predictions.shape,
        )

    def record(
        self,
        bbox_pred: np.ndarray,
        latency_ms: float,
        error: bool = False,
    ) -> None:
        """Add a single inference result to the monitoring window."""
        self._window.append(
            PredictionRecord(
                timestamp=time.time(),
                bbox_pred=bbox_pred.tolist(),
                latency_ms=latency_ms,
                error=error,
            )
        )

    def compute_drift(self) -> Optional[DriftReport]:
        """Compute drift metrics across the current window vs reference."""
        if len(self._window) < 50:
            logger.debug("Window too small for drift computation (%d)", len(self._window))
            return None

        if self._reference is None:
            logger.warning("No reference distribution set; skipping drift check")
            return None

        current_preds = np.array([r.bbox_pred for r in self._window])
        latencies = np.array([r.latency_ms for r in self._window])
        errors = [r.error for r in self._window]

        psi_scores: Dict[str, float] = {}
        kl_scores: Dict[str, float] = {}
        alerts: List[str] = []

        for i, coord in enumerate(self.COORD_NAMES):
            ref_col = self._reference[:, i]
            cur_col = current_preds[:, i]

            psi = population_stability_index(ref_col, cur_col)
            kl = kl_divergence(ref_col, cur_col)
            psi_scores[coord] = round(psi, 4)
            kl_scores[coord] = round(kl, 4)

            if psi >= self.thresholds.psi_critical:
                alerts.append(
                    f"CRITICAL: PSI({coord})={psi:.3f} exceeds threshold "
                    f"{self.thresholds.psi_critical}"
                )
            elif psi >= self.thresholds.psi_warning:
                alerts.append(
                    f"WARNING: PSI({coord})={psi:.3f} approaching threshold"
                )

        p95_latency = float(np.percentile(latencies, 95))
        mean_latency = float(np.mean(latencies))
        error_rate = sum(errors) / len(errors)

        if p95_latency >= self.thresholds.latency_p95_ms_critical:
            alerts.append(f"CRITICAL: p95 latency {p95_latency:.0f}ms")
        elif p95_latency >= self.thresholds.latency_p95_ms_warning:
            alerts.append(f"WARNING: p95 latency {p95_latency:.0f}ms")

        if error_rate >= self.thresholds.error_rate_critical:
            alerts.append(f"CRITICAL: error rate {error_rate:.1%}")
        elif error_rate >= self.thresholds.error_rate_warning:
            alerts.append(f"WARNING: error rate {error_rate:.1%}")

        drift_level = "none"
        if any("CRITICAL" in a for a in alerts):
            drift_level = "critical"
        elif any("WARNING" in a for a in alerts):
            drift_level = "warning"

        report = DriftReport(
            window_size=len(self._window),
            psi_scores=psi_scores,
            kl_scores=kl_scores,
            mean_latency_ms=round(mean_latency, 2),
            p95_latency_ms=round(p95_latency, 2),
            error_rate=round(error_rate, 4),
            alerts=alerts,
            overall_drift_level=drift_level,
        )

        self._persist_report(report)

        if alerts:
            logger.warning("Drift alerts: %s", alerts)

        return report

    def _persist_report(self, report: DriftReport) -> None:
        fname = f"drift_{report.timestamp.replace(':', '-')}.json"
        path = self.output_dir / fname
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)

    def summary(self) -> Dict:
        return {
            "window_fill": len(self._window),
            "window_capacity": self.window_size,
            "has_reference": self._reference is not None,
        }
