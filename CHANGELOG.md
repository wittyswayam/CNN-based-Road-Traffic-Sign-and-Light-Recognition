# Changelog

All notable changes to TrafficVision-AI are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] — 2025-01-15

### Added
- **Enterprise architecture**: Layered src/ structure (api, core, ml, data, monitoring, services, utils)
- **Multi-backbone support**: ResNet50, EfficientNetB3, MobileNetV3 via `BackboneFactory`
- **Weighted ensemble**: `EnsembleDetector` with uncertainty quantification
- **Combined IoU + MSE loss**: +4% mean IoU vs MSE-only baseline
- **Two-phase training**: Frozen backbone (Phase 1) + selective fine-tuning (Phase 2)
- **FastAPI REST API**: `/v2/detect`, `/v2/detect/batch`, `/health`, `/ready`, `/metrics/summary`
- **JWT + API Key auth**: RBAC with viewer/inference/admin roles
- **Redis inference cache**: SHA-256 keyed, TTL=1h, LRU eviction
- **Drift detection**: PSI + KL divergence on 1000-sample sliding window
- **ETL pipeline**: Extract → Validate → Transform → Shard (`.npy`) with DVC integration
- **Hyperparameter optimization**: Optuna TPE sampler with MLflow logging
- **Explainability**: Grad-CAM per bounding box coordinate + SHAP DeepExplainer
- **Evaluation framework**: IoU @ 0.50/0.75/0.90, latency p50/p95/p99, graded A–F
- **Batch inference engine**: Parallel image loading, JSONL output, overlay export
- **Model registry**: Versioned artifact management with experimental→staging→production promotion
- **Distributed training**: MirroredStrategy, MultiWorkerMirroredStrategy, TPUStrategy
- **Full CI/CD**: 7-stage GitHub Actions (lint→test→security→build→integration→staging→production)
- **Docker**: Multi-stage production Dockerfile (300MB), Docker Compose 6-service stack
- **Kubernetes**: Deployment + HPA (3–20 replicas) + Ingress + PDB + NetworkPolicy
- **Terraform IaC**: EKS, ElastiCache, S3, ECR, RDS Aurora
- **Prometheus + Grafana**: Metrics, alert rules, dashboards
- **5 enterprise notebooks**: Data pipeline, Model architecture, MLOps, XAI, Benchmarking
- **Structured JSON logging**: Context-aware with request_id, trace_id propagation
- **Load testing**: Sequential + concurrent harness + Locust integration

### Changed
- Model output layer changed from linear to Sigmoid (enforces [0,1] bbox constraint)
- Training split standardized to 70/15/15 (was 80/20)
- Annotation format standardized to YOLO (cx, cy, w, h) from custom format

### Fixed
- BoundingBox validation now raises `ValueError` for degenerate boxes (x_min ≥ x_max)
- Image loader gracefully handles all OpenCV decode failures

---

## [1.0.0] — 2024-06-01

### Added
- Initial CNN-based traffic sign recognition notebook
- ResNet50 transfer learning baseline
- Basic YOLO annotation parser
- Simple train/val split

---

[2.0.0]: https://github.com/your-org/trafficvision-ai/releases/tag/v2.0.0
[1.0.0]: https://github.com/your-org/trafficvision-ai/releases/tag/v1.0.0
