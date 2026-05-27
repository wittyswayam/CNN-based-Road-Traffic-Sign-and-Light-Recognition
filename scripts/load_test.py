"""
TrafficVision-AI :: Load Testing
===================================
Stress-test the inference API under realistic traffic patterns.

Modes
-----
1. Sequential   – baseline single-threaded latency
2. Concurrent   – ThreadPoolExecutor ramp-up
3. Locust file  – production-grade distributed load testing

Usage::
    # Sequential baseline
    python scripts/load_test.py --mode sequential --n-requests 500

    # Concurrent ramp
    python scripts/load_test.py --mode concurrent --max-workers 50 --n-requests 2000

    # Locust web UI (requires: pip install locust)
    locust -f scripts/load_test.py --host http://localhost:8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Synthetic image generator (avoids needing real images for load testing)
# ---------------------------------------------------------------------------


def generate_synthetic_jpeg(width: int = 224, height: int = 224) -> bytes:
    """Generate a random JPEG image as bytes (PIL-free where possible)."""
    try:
        from PIL import Image
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except ImportError:
        # Minimal JPEG header fallback (valid enough for most decoders)
        import struct
        # Return a pre-encoded minimal JPEG (1×1 white pixel)
        return bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
            0x00, 0x43, 0x00, *([0x08] * 64), 0xFF, 0xC0, 0x00, 0x0B, 0x08,
            0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00,
            0x1F, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03,
            0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xFF, 0xDA, 0x00,
            0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xF5, 0x7B, 0xFF, 0xD9,
        ])


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RequestResult:
    status_code: int
    latency_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestReport:
    mode: str
    n_requests: int
    n_success: int
    n_failed: int
    duration_seconds: float
    rps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_min_ms: float
    latency_max_ms: float
    error_rate: float
    target_url: str


# ---------------------------------------------------------------------------
# HTTP client (requests-based)
# ---------------------------------------------------------------------------


def _send_request(url: str, image_bytes: bytes) -> RequestResult:
    try:
        import requests
        t0 = time.perf_counter()
        resp = requests.post(
            f"{url}/v2/detect",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            status_code=resp.status_code,
            latency_ms=latency_ms,
            success=resp.status_code == 200,
        )
    except Exception as exc:
        return RequestResult(
            status_code=0,
            latency_ms=0.0,
            success=False,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Sequential load test
# ---------------------------------------------------------------------------


def run_sequential(url: str, n_requests: int) -> LoadTestReport:
    logger.info("Sequential load test: %d requests → %s", n_requests, url)
    results: List[RequestResult] = []
    start = time.perf_counter()

    for i in range(n_requests):
        img = generate_synthetic_jpeg()
        result = _send_request(url, img)
        results.append(result)
        if (i + 1) % 50 == 0:
            logger.info("  %d/%d completed", i + 1, n_requests)

    return _build_report("sequential", url, results, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Concurrent load test
# ---------------------------------------------------------------------------


def run_concurrent(url: str, n_requests: int, max_workers: int = 20) -> LoadTestReport:
    logger.info(
        "Concurrent load test: %d requests, %d workers → %s",
        n_requests, max_workers, url,
    )
    results: List[RequestResult] = []
    images = [generate_synthetic_jpeg() for _ in range(min(100, n_requests))]
    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_send_request, url, random.choice(images))
            for _ in range(n_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())

    return _build_report("concurrent", url, results, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def _build_report(
    mode: str, url: str, results: List[RequestResult], duration: float
) -> LoadTestReport:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    latencies = [r.latency_ms for r in successful]

    if not latencies:
        latencies = [0.0]

    report = LoadTestReport(
        mode=mode,
        n_requests=len(results),
        n_success=len(successful),
        n_failed=len(failed),
        duration_seconds=round(duration, 2),
        rps=round(len(results) / duration, 1),
        latency_p50_ms=round(statistics.median(latencies), 2),
        latency_p95_ms=round(
            sorted(latencies)[int(len(latencies) * 0.95)], 2
        ),
        latency_p99_ms=round(
            sorted(latencies)[int(len(latencies) * 0.99)], 2
        ),
        latency_mean_ms=round(statistics.mean(latencies), 2),
        latency_min_ms=round(min(latencies), 2),
        latency_max_ms=round(max(latencies), 2),
        error_rate=round(len(failed) / len(results), 4),
        target_url=url,
    )
    return report


def print_report(report: LoadTestReport) -> None:
    print("\n" + "=" * 60)
    print("  TrafficVision-AI :: Load Test Report")
    print("=" * 60)
    print(f"  Mode             : {report.mode}")
    print(f"  Target           : {report.target_url}")
    print(f"  Total Requests   : {report.n_requests}")
    print(f"  Successful       : {report.n_success}")
    print(f"  Failed           : {report.n_failed}")
    print(f"  Error Rate       : {report.error_rate:.1%}")
    print(f"  Duration         : {report.duration_seconds:.1f}s")
    print(f"  Throughput       : {report.rps:.1f} req/s")
    print(f"  Latency p50      : {report.latency_p50_ms:.1f} ms")
    print(f"  Latency p95      : {report.latency_p95_ms:.1f} ms")
    print(f"  Latency p99      : {report.latency_p99_ms:.1f} ms")
    print(f"  Latency mean     : {report.latency_mean_ms:.1f} ms")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Locust file (importable by `locust -f scripts/load_test.py`)
# ---------------------------------------------------------------------------

try:
    from locust import HttpUser, task, between, events

    class TrafficVisionUser(HttpUser):
        """Locust user simulating realistic traffic patterns."""

        wait_time = between(0.1, 1.0)     # 1–10 req/s per user

        def on_start(self) -> None:
            self._images = [generate_synthetic_jpeg() for _ in range(20)]

        @task(10)
        def detect_single(self) -> None:
            img = random.choice(self._images)
            with self.client.post(
                "/v2/detect",
                files={"file": ("image.jpg", img, "image/jpeg")},
                catch_response=True,
                name="POST /v2/detect",
            ) as response:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"HTTP {response.status_code}")

        @task(2)
        def health_check(self) -> None:
            self.client.get("/health", name="GET /health")

        @task(1)
        def metrics_check(self) -> None:
            self.client.get("/metrics/summary", name="GET /metrics/summary")

except ImportError:
    pass  # locust not installed — CLI mode only


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TrafficVision-AI Load Test")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["sequential", "concurrent"], default="sequential")
    parser.add_argument("--n-requests", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--output", default=None, help="Save JSON report to file")
    args = parser.parse_args()

    if args.mode == "sequential":
        report = run_sequential(args.url, args.n_requests)
    else:
        report = run_concurrent(args.url, args.n_requests, args.max_workers)

    print_report(report)

    if args.output:
        Path(args.output).write_text(json.dumps(asdict(report), indent=2))
        logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
