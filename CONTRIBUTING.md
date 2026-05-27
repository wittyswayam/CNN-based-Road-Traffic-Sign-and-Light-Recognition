# Contributing to TrafficVision-AI

Thank you for your interest in contributing! This guide covers the development workflow, code standards, and PR process.

## Development Setup

```bash
git clone https://github.com/your-org/trafficvision-ai.git
cd trafficvision-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pre-commit install
```

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code, protected |
| `develop` | Integration branch, auto-deploys to staging |
| `feature/*` | New features, branch from `develop` |
| `fix/*` | Bug fixes |
| `chore/*` | Maintenance (deps, CI, docs) |

## Commit Convention (Conventional Commits)

```
feat(ml): add EfficientNetV2 backbone support
fix(api): handle corrupt JPEG gracefully in detect endpoint
docs(readme): update benchmarking table with A10G results
chore(deps): bump tensorflow to 2.16.0
test(unit): add BoundingBox degenerate edge cases
```

## Pull Request Checklist

- [ ] Tests pass: `pytest tests/ --cov=src --cov-fail-under=80`
- [ ] No linting errors: `ruff check src/ tests/`
- [ ] Type checks pass: `mypy src/ --ignore-missing-imports`
- [ ] Formatted: `black src/ tests/`
- [ ] Security scan: `bandit -r src/ -ll`
- [ ] Docstrings on all public functions/classes
- [ ] README updated if public API changed
- [ ] Changelog entry added

## Code Standards

- **Line length**: 100 characters
- **Type hints**: Required on all public function signatures
- **Docstrings**: Google style, required on all public classes and functions
- **Test coverage**: Maintain ≥ 80% overall
- **Import order**: stdlib → third-party → local (enforced by ruff)

## Adding a New Backbone

1. Add entry to `BackboneFactory.REGISTRY` in `src/ml/model.py`
2. Add to `ModelConfig.ensemble_models` in `src/core/config.py`
3. Add benchmark row to `notebooks/05_benchmarking_and_optimization.ipynb`
4. Add to `configs/train_config.yaml` comment block
5. Update README benchmarking table

## Running the Full Test Suite

```bash
# Unit tests only (fast, < 30s)
pytest tests/unit/ -v

# Integration tests (requires no external services)
pytest tests/integration/ -v

# Full suite with coverage
pytest tests/ --cov=src --cov-report=html --cov-fail-under=80

# Load test (requires running API)
python scripts/load_test.py --mode concurrent --n-requests 500 --max-workers 20
```

## Reporting Issues

Use GitHub Issues with the appropriate label:
- `bug` — unexpected behaviour
- `enhancement` — new feature request
- `documentation` — docs improvement
- `performance` — latency/throughput regression
- `ml` — model accuracy issue

Include: Python version, OS, reproduction steps, expected vs actual behaviour.
