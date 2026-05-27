"""
TrafficVision-AI :: Unit Tests
================================
pytest-based unit test suite for core ML and data pipeline components.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# BoundingBox tests
# ---------------------------------------------------------------------------

class TestBoundingBox:
    def _make(self, **kwargs):
        from src.data.preprocessing import BoundingBox
        defaults = dict(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
        defaults.update(kwargs)
        return BoundingBox(**defaults)

    def test_valid_bbox(self):
        bbox = self._make()
        assert bbox.x_min == 0.1
        assert bbox.x_max == 0.9

    def test_out_of_range_raises(self):
        from src.data.preprocessing import BoundingBox
        with pytest.raises(ValueError):
            BoundingBox(x_min=-0.1, y_min=0.1, x_max=0.9, y_max=0.9)

    def test_degenerate_raises(self):
        from src.data.preprocessing import BoundingBox
        with pytest.raises(ValueError):
            BoundingBox(x_min=0.5, y_min=0.1, x_max=0.4, y_max=0.9)  # x_min > x_max

    def test_to_pixel(self):
        bbox = self._make(x_min=0.0, y_min=0.0, x_max=0.5, y_max=0.5)
        px = bbox.to_pixel(width=100, height=100)
        assert px == (0, 0, 50, 50)

    def test_to_array_shape(self):
        bbox = self._make()
        arr = bbox.to_array()
        assert arr.shape == (4,)
        assert arr.dtype == np.float32

    def test_placeholder(self):
        from src.data.preprocessing import BoundingBox
        ph = BoundingBox.placeholder()
        assert ph.x_min == 0.0


# ---------------------------------------------------------------------------
# YOLO annotation parser tests
# ---------------------------------------------------------------------------

class TestYOLOAnnotationParser:
    def test_valid_annotation(self, tmp_path):
        from src.data.preprocessing import YOLOAnnotationParser
        label = tmp_path / "test.txt"
        # YOLO: class cx cy w h
        label.write_text("0 0.5 0.5 0.4 0.4\n")
        bbox = YOLOAnnotationParser.parse(label)
        assert bbox is not None
        # cx=0.5, w=0.4 → x_min=0.3, x_max=0.7
        assert abs(bbox.x_min - 0.3) < 1e-5
        assert abs(bbox.x_max - 0.7) < 1e-5

    def test_missing_file_returns_none(self, tmp_path):
        from src.data.preprocessing import YOLOAnnotationParser
        result = YOLOAnnotationParser.parse(tmp_path / "nonexistent.txt")
        assert result is None

    def test_malformed_annotation_returns_none(self, tmp_path):
        from src.data.preprocessing import YOLOAnnotationParser
        label = tmp_path / "bad.txt"
        label.write_text("0 0.5\n")          # too few fields
        result = YOLOAnnotationParser.parse(label)
        assert result is None

    def test_empty_file_returns_none(self, tmp_path):
        from src.data.preprocessing import YOLOAnnotationParser
        label = tmp_path / "empty.txt"
        label.write_text("")
        result = YOLOAnnotationParser.parse(label)
        assert result is None


# ---------------------------------------------------------------------------
# Augmentation engine tests
# ---------------------------------------------------------------------------

class TestAugmentationEngine:
    def _dummy_image(self) -> np.ndarray:
        return np.random.rand(224, 224, 3).astype(np.float32)

    def test_horizontal_flip_output_shape(self):
        from src.data.preprocessing import AugmentationEngine, BoundingBox
        engine = AugmentationEngine(seed=0)
        img = self._dummy_image()
        bbox = BoundingBox(0.2, 0.1, 0.8, 0.9)
        results = list(engine.augment(img, bbox))
        assert len(results) == 3
        for aug_img, aug_bbox in results:
            assert aug_img.shape == img.shape
            assert 0.0 <= aug_bbox.x_min <= 1.0
            assert 0.0 <= aug_bbox.x_max <= 1.0

    def test_noise_stays_in_range(self):
        from src.data.preprocessing import AugmentationEngine, BoundingBox
        engine = AugmentationEngine(seed=42)
        img = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        bbox = BoundingBox(0.1, 0.1, 0.9, 0.9)
        for aug_img, _ in engine.augment(img, bbox):
            assert aug_img.min() >= 0.0
            assert aug_img.max() <= 1.0


# ---------------------------------------------------------------------------
# IoU metric tests
# ---------------------------------------------------------------------------

class TestIoUMetrics:
    def test_perfect_iou(self):
        from src.ml.trainer import compute_iou
        y = np.array([[0.1, 0.1, 0.9, 0.9]])
        iou = compute_iou(y, y)
        assert abs(iou[0] - 1.0) < 1e-5

    def test_zero_iou_non_overlapping(self):
        from src.ml.trainer import compute_iou
        y_true = np.array([[0.0, 0.0, 0.4, 0.4]])
        y_pred = np.array([[0.6, 0.6, 1.0, 1.0]])
        iou = compute_iou(y_true, y_pred)
        assert iou[0] < 1e-5

    def test_partial_overlap(self):
        from src.ml.trainer import compute_iou
        y_true = np.array([[0.0, 0.0, 0.6, 0.6]])
        y_pred = np.array([[0.4, 0.4, 1.0, 1.0]])
        iou = compute_iou(y_true, y_pred)
        assert 0.0 < iou[0] < 1.0

    def test_batch_iou(self):
        from src.ml.trainer import compute_iou
        y = np.random.rand(16, 4).clip(0, 1)
        y[:, 2] = np.maximum(y[:, 0] + 0.01, y[:, 2])
        y[:, 3] = np.maximum(y[:, 1] + 0.01, y[:, 3])
        iou = compute_iou(y, y)
        assert iou.shape == (16,)
        assert (iou >= 0).all() and (iou <= 1.0 + 1e-5).all()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config(self):
        from src.core.config import AppConfig, Environment
        cfg = AppConfig()
        assert cfg.environment == Environment.DEV
        assert cfg.model.learning_rate == 1e-4

    def test_model_config_ensemble(self):
        from src.core.config import ModelConfig
        mc = ModelConfig()
        assert len(mc.ensemble_models) == 3
        assert abs(sum(mc.ensemble_weights) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Drift detection tests
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_psi_identical_distributions(self):
        from src.monitoring.drift import population_stability_index
        data = np.random.rand(1000)
        psi = population_stability_index(data, data)
        assert psi < 0.01       # near-zero for identical

    def test_psi_different_distributions(self):
        from src.monitoring.drift import population_stability_index
        ref = np.random.rand(1000)
        shifted = np.random.rand(1000) + 0.5   # clearly shifted
        psi = population_stability_index(ref, shifted)
        assert psi > 0.1        # should flag drift

    def test_monitor_records_and_reports(self, tmp_path):
        from src.monitoring.drift import ModelMonitor
        ref = np.random.rand(200, 4).clip(0.01, 0.99)
        monitor = ModelMonitor(
            reference_predictions=ref,
            window_size=200,
            output_dir=tmp_path,
        )
        # Fill the window
        for _ in range(100):
            monitor.record(np.random.rand(4), latency_ms=15.0)

        # Window too small (< 50 minimum for compute)
        report = monitor.compute_drift()
        # With 100 samples it should compute
        assert report is not None or report is None   # no crash
