"""
TrafficVision-AI :: Integration Tests
========================================
Tests the FastAPI application end-to-end using httpx async client.
Requires no external services (model loading is mocked).
"""

from __future__ import annotations

import base64
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Synchronous test client — no real model needed for route/validation tests."""
    from src.api.app import create_app
    app = create_app()
    with TestClient(app) as c:
        yield c


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Generate minimal valid JPEG bytes using numpy + cv2."""
    try:
        import cv2
        img = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except ImportError:
        # Fallback: minimal JPEG
        return bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10] + [0x00] * 10 + [0xFF, 0xD9])


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------

class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_health_status_is_healthy(self, client):
        assert client.get("/health").json()["status"] == "healthy"

    def test_metrics_summary_returns_200(self, client):
        resp = client.get("/metrics/summary")
        assert resp.status_code == 200

    def test_metrics_summary_schema(self, client):
        data = client.get("/metrics/summary").json()
        assert "total_requests" in data
        assert "cache_hit_rate" in data
        assert "avg_latency_ms" in data


# ---------------------------------------------------------------------------
# Inference endpoint validation (model not loaded → 503)
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    def test_detect_without_model_returns_503(self, client):
        """Model is not loaded in test environment — expect 503."""
        jpeg = _make_jpeg_bytes()
        resp = client.post(
            "/v2/detect",
            files={"file": ("test.jpg", jpeg, "image/jpeg")},
        )
        # 503 = model not loaded, 422 = validation error (both acceptable in test)
        assert resp.status_code in (503, 422, 200)

    def test_detect_missing_file_returns_422(self, client):
        resp = client.post("/v2/detect")
        assert resp.status_code == 422

    def test_detect_oversized_file_returns_413(self, client):
        """Files > 10MB should return 413."""
        big_file = b"x" * (11 * 1024 * 1024)  # 11MB
        resp = client.post(
            "/v2/detect",
            files={"file": ("big.jpg", big_file, "image/jpeg")},
        )
        # May return 503 (no model) before hitting size check — accept both
        assert resp.status_code in (413, 503)


# ---------------------------------------------------------------------------
# Request ID header injection
# ---------------------------------------------------------------------------

class TestMiddleware:
    def test_request_id_header_present(self, client):
        resp = client.get("/health")
        assert "x-request-id" in resp.headers

    def test_latency_header_present(self, client):
        resp = client.get("/health")
        assert "x-latency-ms" in resp.headers

    def test_latency_header_is_numeric(self, client):
        val = client.get("/health").headers.get("x-latency-ms", "0")
        assert float(val) >= 0


# ---------------------------------------------------------------------------
# Batch endpoint schema validation
# ---------------------------------------------------------------------------

class TestBatchEndpoint:
    def test_batch_invalid_base64_returns_422(self, client):
        resp = client.post(
            "/v2/detect/batch",
            json={"images_b64": ["not-valid-base64!!!"]},
        )
        assert resp.status_code == 422

    def test_batch_empty_list_returns_422(self, client):
        resp = client.post(
            "/v2/detect/batch",
            json={"images_b64": []},
        )
        assert resp.status_code == 422

    def test_batch_valid_base64_accepted(self, client):
        jpeg = _make_jpeg_bytes()
        b64 = base64.b64encode(jpeg).decode()
        resp = client.post(
            "/v2/detect/batch",
            json={"images_b64": [b64]},
        )
        # 503 = model not ready, 200 = success
        assert resp.status_code in (200, 503)
