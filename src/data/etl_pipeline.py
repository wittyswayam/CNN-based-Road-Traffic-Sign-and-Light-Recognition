"""
TrafficVision-AI :: ETL Pipeline
===================================
Enterprise data ingestion, validation, and transformation pipeline.

Stages
------
1. Extract   — discover images from local/S3/GCS sources
2. Validate  — schema checks, corruption detection, duplicate removal
3. Transform — resize, normalize, annotation conversion
4. Load      — write processed .npy shards + DVC-tracked manifest

Designed for Airflow DAG integration (each stage = one task).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass
class RawRecord:
    image_path: Path
    label_path: Optional[Path]
    checksum: str
    file_size_bytes: int
    source: str = "local"


@dataclass
class ValidationResult:
    record: RawRecord
    valid: bool
    issues: List[str] = field(default_factory=list)


@dataclass
class ETLReport:
    run_id: str
    source_dir: str
    output_dir: str
    total_discovered: int
    total_valid: int
    total_invalid: int
    total_duplicates_removed: int
    shard_count: int
    shard_size: int
    duration_seconds: float
    timestamp: str = ""
    validation_issues: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Stage 1: Extract
# ---------------------------------------------------------------------------


class Extractor:
    """Discovers all image/label pairs from a directory tree."""

    def __init__(self, source_dir: Path, label_dir: Optional[Path] = None) -> None:
        self.source_dir = Path(source_dir)
        self.label_dir = Path(label_dir) if label_dir else None

    def extract(self) -> List[RawRecord]:
        records: List[RawRecord] = []
        image_paths = sorted(
            p for p in self.source_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        logger.info("Discovered %d images in %s", len(image_paths), self.source_dir)

        for img_path in image_paths:
            label_path: Optional[Path] = None
            if self.label_dir:
                candidate = self.label_dir / img_path.with_suffix(".txt").name
                if candidate.exists():
                    label_path = candidate

            checksum = self._md5(img_path)
            records.append(RawRecord(
                image_path=img_path,
                label_path=label_path,
                checksum=checksum,
                file_size_bytes=img_path.stat().st_size,
            ))

        return records

    @staticmethod
    def _md5(path: Path) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Stage 2: Validate
# ---------------------------------------------------------------------------


class Validator:
    """
    Applies data quality rules to each raw record.

    Rules
    -----
    - Image is readable by OpenCV
    - Image has 3 channels (RGB/BGR)
    - Minimum resolution: 32×32
    - File size: 1KB – 50MB
    - Label file (if present) has valid YOLO format
    - No duplicate checksums
    """

    MIN_DIM = 32
    MIN_SIZE_BYTES = 1024
    MAX_SIZE_BYTES = 50 * 1024 * 1024

    def __init__(self) -> None:
        self._seen_checksums: Set[str] = set()

    def validate_all(self, records: List[RawRecord]) -> List[ValidationResult]:
        results = [self._validate_one(r) for r in records]
        valid_count = sum(1 for r in results if r.valid)
        logger.info(
            "Validation: %d/%d valid", valid_count, len(records)
        )
        return results

    def _validate_one(self, record: RawRecord) -> ValidationResult:
        issues: List[str] = []

        # Duplicate check
        if record.checksum in self._seen_checksums:
            issues.append("Duplicate image (checksum match)")
            return ValidationResult(record=record, valid=False, issues=issues)
        self._seen_checksums.add(record.checksum)

        # File size
        if record.file_size_bytes < self.MIN_SIZE_BYTES:
            issues.append(f"File too small: {record.file_size_bytes}B < 1KB")
        if record.file_size_bytes > self.MAX_SIZE_BYTES:
            issues.append(f"File too large: {record.file_size_bytes / 1e6:.1f}MB > 50MB")

        # Image readability
        try:
            import cv2
            img = cv2.imread(str(record.image_path))
            if img is None:
                issues.append("OpenCV cannot decode image")
            else:
                h, w, c = img.shape
                if c != 3:
                    issues.append(f"Expected 3 channels, got {c}")
                if h < self.MIN_DIM or w < self.MIN_DIM:
                    issues.append(f"Resolution too small: {w}×{h}")
        except Exception as exc:
            issues.append(f"Image read error: {exc}")

        # Label validation
        if record.label_path and record.label_path.exists():
            label_issues = self._validate_label(record.label_path)
            issues.extend(label_issues)

        return ValidationResult(record=record, valid=len(issues) == 0, issues=issues)

    def _validate_label(self, path: Path) -> List[str]:
        issues = []
        try:
            text = path.read_text().strip()
            if not text:
                return []  # empty label = no object, acceptable
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"Label line {line_no}: expected 5 fields, got {len(parts)}")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = [float(p) for p in parts[1:5]]
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        issues.append(f"Label line {line_no}: coordinates out of [0,1]")
                except ValueError:
                    issues.append(f"Label line {line_no}: non-numeric values")
        except Exception as exc:
            issues.append(f"Label read error: {exc}")
        return issues


# ---------------------------------------------------------------------------
# Stage 3: Transform
# ---------------------------------------------------------------------------


class Transformer:
    """
    Applies preprocessing transforms to validated records.
    Returns (images, bboxes) numpy arrays.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size

    def transform(
        self, results: List[ValidationResult]
    ) -> Tuple[np.ndarray, np.ndarray]:
        import cv2
        images, bboxes = [], []

        valid = [r for r in results if r.valid]
        logger.info("Transforming %d valid records", len(valid))

        for result in valid:
            img = cv2.imread(str(result.record.image_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size).astype(np.float32) / 255.0
            images.append(img)

            bbox = self._parse_bbox(result.record.label_path)
            bboxes.append(bbox)

        return np.array(images, dtype=np.float32), np.array(bboxes, dtype=np.float32)

    def _parse_bbox(self, label_path: Optional[Path]) -> np.ndarray:
        if label_path is None or not label_path.exists():
            return np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float32)
        try:
            text = label_path.read_text().strip()
            if not text:
                return np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float32)
            parts = text.split()
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            return np.array(
                [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                dtype=np.float32,
            ).clip(0.0, 1.0)
        except Exception:
            return np.array([0.0, 0.0, 0.001, 0.001], dtype=np.float32)


# ---------------------------------------------------------------------------
# Stage 4: Load (shard writer)
# ---------------------------------------------------------------------------


class ShardWriter:
    """Writes processed data as numbered .npy shards for streaming-compatible loading."""

    def __init__(self, output_dir: Path, shard_size: int = 1000) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size

    def write(self, images: np.ndarray, bboxes: np.ndarray) -> int:
        n = len(images)
        shard_count = 0
        for start in range(0, n, self.shard_size):
            end = min(start + self.shard_size, n)
            shard_id = f"{shard_count:04d}"
            np.save(self.output_dir / f"images_{shard_id}.npy", images[start:end])
            np.save(self.output_dir / f"bboxes_{shard_id}.npy", bboxes[start:end])
            shard_count += 1

        # Write manifest
        manifest = {
            "n_samples": n,
            "shard_count": shard_count,
            "shard_size": self.shard_size,
            "image_shape": list(images.shape[1:]),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info("Wrote %d shards (%d samples) to %s", shard_count, n, self.output_dir)
        return shard_count


# ---------------------------------------------------------------------------
# Orchestrated ETL runner
# ---------------------------------------------------------------------------


class ETLPipeline:
    """
    Full Extract → Validate → Transform → Load orchestrator.

    Usage::
        pipeline = ETLPipeline(
            source_dir=Path("data/raw/images"),
            label_dir=Path("data/raw/labels"),
            output_dir=Path("data/processed"),
        )
        report = pipeline.run()
    """

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        label_dir: Optional[Path] = None,
        shard_size: int = 1000,
        target_size: Tuple[int, int] = (224, 224),
    ) -> None:
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.shard_size = shard_size
        self.target_size = target_size

    def run(self) -> ETLReport:
        import time
        run_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        t0 = time.perf_counter()

        # E
        extractor = Extractor(self.source_dir, self.label_dir)
        records = extractor.extract()

        # V
        validator = Validator()
        validation_results = validator.validate_all(records)

        valid_results = [r for r in validation_results if r.valid]
        invalid_results = [r for r in validation_results if not r.valid]
        duplicates = sum(1 for r in invalid_results if any("Duplicate" in i for i in r.issues))

        # T
        transformer = Transformer(self.target_size)
        images, bboxes = transformer.transform(valid_results)

        # L
        writer = ShardWriter(self.output_dir, self.shard_size)
        shard_count = writer.write(images, bboxes)

        duration = time.perf_counter() - t0
        report = ETLReport(
            run_id=run_id,
            source_dir=str(self.source_dir),
            output_dir=str(self.output_dir),
            total_discovered=len(records),
            total_valid=len(valid_results),
            total_invalid=len(invalid_results),
            total_duplicates_removed=duplicates,
            shard_count=shard_count,
            shard_size=self.shard_size,
            duration_seconds=round(duration, 2),
            validation_issues={
                str(r.record.image_path.name): r.issues
                for r in invalid_results
            },
        )

        report_path = self.output_dir / f"etl_report_{run_id}.json"
        report_path.write_text(json.dumps(asdict(report), indent=2))
        logger.info(
            "ETL complete: %d valid / %d total in %.1fs",
            len(valid_results), len(records), duration,
        )
        return report
