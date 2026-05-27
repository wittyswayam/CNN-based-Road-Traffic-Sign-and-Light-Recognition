"""
TrafficVision-AI :: Model Registry
=====================================
Versioned model artifact management with promotion workflow.

Stages: experimental → staging → production → archived

Registry layout::
    models/registry/
    ├── registry.json           ← global manifest
    ├── v1.0.0/
    │   ├── model.keras
    │   ├── metadata.json
    │   └── eval_report.json
    ├── v2.0.0/
    │   ├── model.keras
    │   ├── metadata.json
    │   └── eval_report.json
    └── latest -> v2.0.0/      ← symlink (or JSON pointer on Windows)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelStage(str, Enum):
    EXPERIMENTAL = "experimental"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class ModelRegistry:
    """
    Local-filesystem model registry with promotion gates.

    For cloud deployments, swap _save/_load to S3/GCS via boto3/google-cloud-storage.

    Usage::
        registry = ModelRegistry(Path("models/registry"))
        registry.register(model, version="2.1.0", metrics=eval_report.to_dict())
        registry.promote("2.1.0", ModelStage.STAGING)
        registry.promote("2.1.0", ModelStage.PRODUCTION, min_iou=0.72)
        latest = registry.load_production()
    """

    MANIFEST_FILE = "registry.json"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.base_dir / self.MANIFEST_FILE
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> Dict:
        if self._manifest_path.exists():
            return json.loads(self._manifest_path.read_text())
        return {"versions": {}, "production": None, "staging": None}

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        model,
        version: str,
        metrics: Optional[Dict] = None,
        backbone: str = "unknown",
        notes: str = "",
    ) -> Path:
        """
        Save model artifact and metadata to registry.
        Returns path to the versioned directory.
        """
        version_dir = self.base_dir / version
        if version_dir.exists():
            logger.warning("Version %s already exists — overwriting", version)
            shutil.rmtree(version_dir)
        version_dir.mkdir(parents=True)

        # Save model
        model.save(version_dir)

        # Save metadata
        metadata = {
            "version": version,
            "backbone": backbone,
            "stage": ModelStage.EXPERIMENTAL.value,
            "registered_at": datetime.now(tz=timezone.utc).isoformat(),
            "metrics": metrics or {},
            "notes": notes,
        }
        (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        # Update manifest
        self._manifest["versions"][version] = {
            "stage": ModelStage.EXPERIMENTAL.value,
            "path": str(version_dir),
            "backbone": backbone,
            "registered_at": metadata["registered_at"],
            "mean_iou": (metrics or {}).get("iou", {}).get("mean_iou", 0.0),
        }
        self._save_manifest()

        logger.info("Registered model version=%s at %s", version, version_dir)
        return version_dir

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote(
        self,
        version: str,
        target_stage: ModelStage,
        min_iou: float = 0.0,
        max_p95_ms: float = float("inf"),
    ) -> None:
        """
        Promote a version to a higher stage with optional quality gates.

        Raises ValueError if quality gates not met.
        """
        if version not in self._manifest["versions"]:
            raise KeyError(f"Version '{version}' not found in registry")

        entry = self._manifest["versions"][version]
        mean_iou = entry.get("mean_iou", 0.0)

        # Quality gates
        if mean_iou < min_iou:
            raise ValueError(
                f"Promotion blocked: mean_iou={mean_iou:.4f} < required {min_iou}"
            )

        # Archive current production if promoting to production
        if target_stage == ModelStage.PRODUCTION:
            current_prod = self._manifest.get("production")
            if current_prod and current_prod != version:
                self._set_stage(current_prod, ModelStage.ARCHIVED)
                logger.info("Archived previous production version: %s", current_prod)
            self._manifest["production"] = version
            # Update latest symlink / pointer
            self._update_latest_pointer(version)

        elif target_stage == ModelStage.STAGING:
            self._manifest["staging"] = version

        self._set_stage(version, target_stage)
        self._save_manifest()

        logger.info(
            "Promoted version=%s to stage=%s (iou=%.4f)",
            version, target_stage.value, mean_iou,
        )

    def _set_stage(self, version: str, stage: ModelStage) -> None:
        self._manifest["versions"][version]["stage"] = stage.value
        version_dir = Path(self._manifest["versions"][version]["path"])
        meta_path = version_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["stage"] = stage.value
            meta_path.write_text(json.dumps(meta, indent=2))

    def _update_latest_pointer(self, version: str) -> None:
        pointer = self.base_dir / "latest.json"
        pointer.write_text(json.dumps({"version": version}, indent=2))

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_production(self, model_cls):
        """Load and return the current production model."""
        prod_version = self._manifest.get("production")
        if not prod_version:
            raise RuntimeError("No production model registered")
        return self.load_version(prod_version, model_cls)

    def load_version(self, version: str, model_cls):
        if version not in self._manifest["versions"]:
            raise KeyError(f"Version '{version}' not in registry")
        version_dir = Path(self._manifest["versions"][version]["path"])
        model = model_cls()
        model.load(version_dir)
        return model

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_versions(self) -> List[Dict]:
        return [
            {"version": v, **info}
            for v, info in self._manifest["versions"].items()
        ]

    def current_production(self) -> Optional[str]:
        return self._manifest.get("production")

    def __repr__(self) -> str:
        versions = list(self._manifest["versions"].keys())
        prod = self._manifest.get("production", "none")
        return f"ModelRegistry(versions={versions}, production={prod})"
