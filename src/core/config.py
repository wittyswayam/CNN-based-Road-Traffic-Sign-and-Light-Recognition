"""
TrafficVision-AI :: Core Configuration Management
==================================================
Centralised, environment-aware, validated configuration system.
Supports dev / staging / production profiles with secret injection.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Environment enum
# ---------------------------------------------------------------------------

class Environment(str, Enum):
    DEV = "development"
    STAGING = "staging"
    PRODUCTION = "production"


# ---------------------------------------------------------------------------
# Sub-configurations (dataclasses for type-safety + IDE support)
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    backbone: str = "resnet50"
    input_size: tuple = (224, 224)
    num_classes: int = 4          # bbox regression outputs
    pretrained_weights: str = "imagenet"
    freeze_backbone: bool = True
    dense_units: int = 1024
    dropout_rate: float = 0.3
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    reduce_lr_factor: float = 0.5
    model_registry_path: str = "models/registry"
    artifacts_path: str = "models/artifacts"

    # Multi-model ensemble
    ensemble_models: List[str] = field(
        default_factory=lambda: ["resnet50", "efficientnetb3", "mobilenetv3"]
    )
    ensemble_weights: List[float] = field(
        default_factory=lambda: [0.5, 0.3, 0.2]
    )


@dataclass
class DataConfig:
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp"]
    )
    augmentation_enabled: bool = True
    augmentation_factor: int = 3      # synthetic samples per real sample
    cache_preprocessed: bool = True
    num_workers: int = 4


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    max_upload_size_mb: int = 10
    request_timeout_seconds: int = 30
    rate_limit_per_minute: int = 60
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_version: str = "v2"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class MonitoringConfig:
    prometheus_port: int = 9090
    grafana_port: int = 3000
    log_level: str = "INFO"
    log_format: str = "json"
    metrics_enabled: bool = True
    tracing_enabled: bool = True
    drift_detection_enabled: bool = True
    drift_window_size: int = 1000
    alert_webhook_url: Optional[str] = None


@dataclass
class CacheConfig:
    backend: str = "redis"          # redis | memcached | in-memory
    host: str = "localhost"
    port: int = 6379
    ttl_seconds: int = 3600
    max_connections: int = 20
    enabled: bool = True


@dataclass
class StorageConfig:
    backend: str = "local"          # local | s3 | gcs | azure
    bucket_name: str = "trafficvision-models"
    region: str = "eu-central-1"
    local_base_path: str = "/app/storage"


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class AppConfig:
    environment: Environment = Environment.DEV
    project_name: str = "TrafficVision-AI"
    version: str = "2.0.0"
    debug: bool = False
    secret_key: str = ""           # injected from env / secret manager

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Build config from environment variables (12-factor app style)."""
        cfg = cls()
        env_str = os.getenv("APP_ENV", "development")
        cfg.environment = Environment(env_str)
        cfg.debug = cfg.environment == Environment.DEV
        cfg.secret_key = os.getenv("SECRET_KEY", "change-me-in-production")

        # Model overrides
        cfg.model.learning_rate = float(
            os.getenv("MODEL_LR", cfg.model.learning_rate)
        )
        cfg.model.batch_size = int(
            os.getenv("MODEL_BATCH_SIZE", cfg.model.batch_size)
        )

        # API overrides
        cfg.api.port = int(os.getenv("API_PORT", cfg.api.port))
        cfg.api.workers = int(os.getenv("API_WORKERS", cfg.api.workers))

        # Cache overrides
        cfg.cache.host = os.getenv("REDIS_HOST", cfg.cache.host)
        cfg.cache.port = int(os.getenv("REDIS_PORT", cfg.cache.port))

        # Storage overrides
        cfg.storage.backend = os.getenv("STORAGE_BACKEND", cfg.storage.backend)
        cfg.storage.bucket_name = os.getenv(
            "STORAGE_BUCKET", cfg.storage.bucket_name
        )

        return cfg

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


# Singleton accessor
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config
