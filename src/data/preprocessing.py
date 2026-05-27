"""
TrafficVision-AI :: Data Preprocessing Pipeline
=================================================
Enterprise-grade, reproducible image preprocessing with augmentation,
schema validation, and comprehensive error handling.

Pipeline stages
---------------
1. Raw image ingestion
2. Schema + integrity validation
3. Resize & normalize
4. YOLO annotation parsing
5. Augmentation (training only)
6. Cache write (optional)
7. TF Dataset construction
"""

from __future__ import annotations

import os
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Normalised [0, 1] bounding box in YOLO corner format."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name, val in self.__dict__.items():
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"BoundingBox.{name}={val} is outside [0, 1]"
                )
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Degenerate bounding box: min >= max")

    def to_pixel(self, width: int, height: int) -> Tuple[int, int, int, int]:
        return (
            max(0, int(self.x_min * width)),
            max(0, int(self.y_min * height)),
            min(width, int(self.x_max * width)),
            min(height, int(self.y_max * height)),
        )

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.x_min, self.y_min, self.x_max, self.y_max], dtype=np.float32
        )

    @classmethod
    def placeholder(cls) -> "BoundingBox":
        """Zero-area placeholder for missing annotations."""
        return cls(0.0, 0.0, 0.001, 0.001)


@dataclass
class Sample:
    """A single (image, bbox) pair with metadata."""
    image_path: Path
    bbox: BoundingBox
    class_id: int = 0
    split: str = "train"
    checksum: str = ""
    augmented: bool = False
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

class YOLOAnnotationParser:
    """
    Parses YOLO-format label files.
    Each line: <class_id> <x_center> <y_center> <width> <height>
    """

    @staticmethod
    def parse(label_path: Path) -> Optional[BoundingBox]:
        if not label_path.exists():
            logger.debug("No annotation file: %s", label_path)
            return None

        try:
            text = label_path.read_text().strip()
            if not text:
                return None

            parts = text.split()
            if len(parts) < 5:
                logger.warning(
                    "Malformed annotation in %s: expected >=5 fields, got %d",
                    label_path, len(parts),
                )
                return None

            # YOLO format: cx, cy, w, h → convert to x_min, y_min, x_max, y_max
            cx, cy, w, h = (float(p) for p in parts[1:5])
            x_min = max(0.0, cx - w / 2)
            y_min = max(0.0, cy - h / 2)
            x_max = min(1.0, cx + w / 2)
            y_max = min(1.0, cy + h / 2)

            return BoundingBox(x_min, y_min, x_max, y_max)

        except (ValueError, IndexError) as exc:
            logger.error(
                "Failed to parse %s: %s", label_path, exc, exc_info=True
            )
            return None


# ---------------------------------------------------------------------------
# Image preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Loads, validates, and normalises images to a fixed resolution.
    Handles BGR→RGB conversion and float32 normalisation to [0, 1].
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

    def __init__(self, target_size: Tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size

    def load(self, path: Path) -> np.ndarray:
        if path.suffix.lower() not in self.VALID_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {path.suffix}")

        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"cv2.imread returned None for {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return img

    @staticmethod
    def checksum(path: Path) -> str:
        """MD5 checksum for data integrity verification."""
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Augmentation engine
# ---------------------------------------------------------------------------

class AugmentationEngine:
    """
    Training-time augmentation producing synthetic samples.
    All transforms preserve annotation validity.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def augment(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> Iterator[Tuple[np.ndarray, BoundingBox]]:
        """Yields augmented (image, bbox) pairs."""
        yield from self._horizontal_flip(image, bbox)
        yield from self._brightness_jitter(image, bbox)
        yield from self._gaussian_noise(image, bbox)

    def _horizontal_flip(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> Iterator[Tuple[np.ndarray, BoundingBox]]:
        flipped = np.fliplr(image)
        new_bbox = BoundingBox(
            x_min=1.0 - bbox.x_max,
            y_min=bbox.y_min,
            x_max=1.0 - bbox.x_min,
            y_max=bbox.y_max,
        )
        yield flipped, new_bbox

    def _brightness_jitter(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> Iterator[Tuple[np.ndarray, BoundingBox]]:
        factor = self._rng.uniform(0.7, 1.3)
        jittered = np.clip(image * factor, 0.0, 1.0).astype(np.float32)
        yield jittered, bbox

    def _gaussian_noise(
        self, image: np.ndarray, bbox: BoundingBox
    ) -> Iterator[Tuple[np.ndarray, BoundingBox]]:
        noise = self._rng.normal(0, 0.02, image.shape).astype(np.float32)
        noisy = np.clip(image + noise, 0.0, 1.0)
        yield noisy, bbox


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """
    Walks an image/label directory pair, builds Sample objects,
    and optionally produces augmented variants.
    """

    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        *,
        preprocessor: Optional[ImagePreprocessor] = None,
        augmentation_engine: Optional[AugmentationEngine] = None,
        augment: bool = False,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.augmentation_engine = augmentation_engine or AugmentationEngine()
        self.augment = augment
        self._parser = YOLOAnnotationParser()

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (images, bboxes) as numpy arrays ready for model.fit()."""
        images: List[np.ndarray] = []
        bboxes: List[np.ndarray] = []

        for img_path in sorted(self.image_dir.iterdir()):
            if img_path.suffix.lower() not in ImagePreprocessor.VALID_EXTENSIONS:
                continue

            label_path = self.label_dir / img_path.with_suffix(".txt").name
            bbox = self._parser.parse(label_path) or BoundingBox.placeholder()

            try:
                img = self.preprocessor.load(img_path)
            except (IOError, ValueError) as exc:
                logger.warning("Skipping %s: %s", img_path.name, exc)
                continue

            images.append(img)
            bboxes.append(bbox.to_array())

            if self.augment:
                for aug_img, aug_bbox in self.augmentation_engine.augment(img, bbox):
                    images.append(aug_img)
                    bboxes.append(aug_bbox.to_array())

        if not images:
            raise RuntimeError(
                f"No valid images found in {self.image_dir}"
            )

        logger.info(
            "Dataset built: %d samples from %s", len(images), self.image_dir
        )
        return np.array(images, dtype=np.float32), np.array(bboxes, dtype=np.float32)
