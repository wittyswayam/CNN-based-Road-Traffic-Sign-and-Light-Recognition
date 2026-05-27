"""
TrafficVision-AI :: Distributed Training
==========================================
Multi-GPU and multi-node training strategies using TensorFlow's
distribution API.

Strategies
----------
- MirroredStrategy         : Single-node, multi-GPU (most common)
- MultiWorkerMirroredStrategy: Multi-node, multi-GPU (cluster)
- TPUStrategy              : Google TPU pods
- ParameterServerStrategy  : Large-scale async training

Usage::
    trainer = DistributedTrainer(strategy="mirrored")
    trainer.train(train_dataset, val_dataset, epochs=50)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DistributionStrategy(str, Enum):
    SINGLE_GPU = "single_gpu"
    MIRRORED = "mirrored"               # single-node, multi-GPU
    MULTI_WORKER = "multi_worker"       # multi-node, multi-GPU
    TPU = "tpu"                         # Google TPU
    CPU = "cpu"                         # CPU-only fallback


@dataclass
class DistributedConfig:
    strategy: DistributionStrategy = DistributionStrategy.MIRRORED
    num_gpus: int = 1
    tpu_address: Optional[str] = None
    worker_hosts: Optional[List[str]] = None   # for MultiWorkerMirroredStrategy
    task_type: str = "worker"                   # worker | ps | chief
    task_index: int = 0
    mixed_precision: bool = True               # float16 compute, float32 weights


def detect_hardware() -> Dict:
    """Auto-detect available hardware."""
    info: Dict = {"gpus": [], "tpus": [], "strategy_recommendation": "cpu"}
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        tpus = tf.config.list_physical_devices("TPU")
        info["gpus"] = [g.name for g in gpus]
        info["tpus"] = [t.name for t in tpus]

        if tpus:
            info["strategy_recommendation"] = "tpu"
        elif len(gpus) > 1:
            info["strategy_recommendation"] = "mirrored"
        elif len(gpus) == 1:
            info["strategy_recommendation"] = "single_gpu"
        else:
            info["strategy_recommendation"] = "cpu"
    except ImportError:
        pass
    logger.info("Hardware detection: %s", info)
    return info


class DistributedTrainer:
    """
    Wraps TensorFlow distribution strategies for transparent
    multi-GPU / multi-node training.

    Architecture::

        DistributedTrainer
              │
              ├── MirroredStrategy   ← N GPUs on 1 machine
              │     All-reduce gradient aggregation (NCCL)
              │
              ├── MultiWorkerMirrored ← N machines × M GPUs
              │     Ring-AllReduce across workers
              │
              └── TPUStrategy         ← TPU pod / single TPU
                    XLA compilation + hardware matrix multiply
    """

    def __init__(self, config: Optional[DistributedConfig] = None) -> None:
        self.config = config or DistributedConfig()
        self._strategy = None

    def setup(self):
        """Initialize distribution strategy. Call before model build."""
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow required for distributed training")

        strategy_type = self.config.strategy

        if strategy_type == DistributionStrategy.MIRRORED:
            self._strategy = tf.distribute.MirroredStrategy()
            logger.info(
                "MirroredStrategy initialized: %d replicas",
                self._strategy.num_replicas_in_sync,
            )

        elif strategy_type == DistributionStrategy.MULTI_WORKER:
            # TF_CONFIG must be set in environment
            tf_config = self._build_tf_config()
            os.environ["TF_CONFIG"] = json.dumps(tf_config)
            self._strategy = tf.distribute.MultiWorkerMirroredStrategy()
            logger.info("MultiWorkerMirroredStrategy initialized")

        elif strategy_type == DistributionStrategy.TPU:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
                tpu=self.config.tpu_address
            )
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            self._strategy = tf.distribute.TPUStrategy(resolver)
            logger.info(
                "TPUStrategy initialized: %d replicas",
                self._strategy.num_replicas_in_sync,
            )

        else:
            # Single GPU or CPU
            self._strategy = tf.distribute.get_strategy()
            logger.info("Using default (single device) strategy")

        # Mixed precision
        if self.config.mixed_precision and strategy_type != DistributionStrategy.CPU:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision enabled: float16 compute")

        return self._strategy

    def _build_tf_config(self) -> Dict:
        """Build TF_CONFIG dict for multi-worker training."""
        workers = self.config.worker_hosts or ["localhost:12345"]
        return {
            "cluster": {"worker": workers},
            "task": {
                "type": self.config.task_type,
                "index": self.config.task_index,
            },
        }

    def build_model_in_scope(self, model_builder_fn, **kwargs):
        """
        Build model inside distribution scope.
        This MUST be used for distributed training.

        Usage::
            trainer = DistributedTrainer()
            strategy = trainer.setup()
            model = trainer.build_model_in_scope(
                lambda: TrafficDetectionModel(backbone_name="resnet50").build()
            )
        """
        if self._strategy is None:
            self.setup()

        with self._strategy.scope():
            model = model_builder_fn(**kwargs)

        return model

    def scale_batch_size(self, per_replica_batch_size: int) -> int:
        """
        Compute global batch size = per_replica × n_replicas.
        Gradient is automatically averaged across replicas.
        """
        if self._strategy is None:
            return per_replica_batch_size
        global_batch = per_replica_batch_size * self._strategy.num_replicas_in_sync
        logger.info(
            "Global batch size: %d (%d per replica × %d replicas)",
            global_batch, per_replica_batch_size,
            self._strategy.num_replicas_in_sync,
        )
        return global_batch

    def make_distributed_dataset(
        self, images: np.ndarray, labels: np.ndarray, batch_size: int
    ):
        """Convert numpy arrays to a distributed tf.data.Dataset."""
        try:
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow required")

        dataset = (
            tf.data.Dataset.from_tensor_slices((images, labels))
            .cache()
            .shuffle(buffer_size=min(10000, len(images)))
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        if self._strategy:
            return self._strategy.experimental_distribute_dataset(dataset)
        return dataset

    @property
    def num_replicas(self) -> int:
        if self._strategy is None:
            return 1
        return self._strategy.num_replicas_in_sync


# ---------------------------------------------------------------------------
# GPU memory configuration
# ---------------------------------------------------------------------------


def configure_gpu_memory(growth: bool = True, memory_limit_mb: Optional[int] = None) -> None:
    """
    Configure GPU memory allocation.
    growth=True: allocate incrementally (recommended for shared nodes)
    memory_limit_mb: hard cap per GPU (useful for multi-tenant environments)
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            if growth:
                tf.config.experimental.set_memory_growth(gpu, True)
            elif memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)],
                )
        logger.info(
            "GPU memory configured: %d GPUs, growth=%s, limit=%s MB",
            len(gpus), growth, memory_limit_mb,
        )
    except ImportError:
        logger.warning("TensorFlow not available — GPU configuration skipped")
    except RuntimeError as exc:
        logger.warning("GPU config warning: %s", exc)
