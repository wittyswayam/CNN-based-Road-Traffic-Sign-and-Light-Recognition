# ==============================================================================
# TrafficVision-AI :: Production Dockerfile
# ==============================================================================
# Multi-stage build:
#   Stage 1 (builder)  – install Python deps in a virtualenv
#   Stage 2 (runtime)  – lean runtime image (~300MB vs ~1.5GB naive)
#
# Build:
#   docker build -t trafficvision-ai:latest .
#
# Run:
#   docker run -p 8000:8000 \
#     -e APP_ENV=production \
#     -e SECRET_KEY=<secret> \
#     trafficvision-ai:latest
# ==============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps for OpenCV + compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create isolated venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first (layer cache)
COPY requirements.txt requirements-prod.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="TrafficVision-AI Engineering <eng@trafficvision.ai>" \
      version="2.0.0" \
      description="Enterprise Road Traffic Detection API"

WORKDIR /app

# Minimal runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Non-root user (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create writable directories for models and logs
RUN mkdir -p /app/models /app/logs /app/monitoring && \
    chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment defaults (override at runtime)
ENV APP_ENV=production \
    API_PORT=8000 \
    API_WORKERS=4 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Entrypoint
CMD ["sh", "-c", \
     "uvicorn src.api.app:app \
      --host 0.0.0.0 \
      --port ${API_PORT} \
      --workers ${API_WORKERS} \
      --log-config configs/uvicorn_log.json \
      --access-log"]
