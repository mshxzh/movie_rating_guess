# Movie Rating Prediction API - Production Dockerfile
# Multi-stage build for smaller image size

# Stage 1: Build dependencies
FROM python:3.11.9-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install system dependencies required for LightGBM and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files (this layer is cached if dependencies don't change)
COPY pyproject.toml uv.lock ./

# Install dependencies (including build tools)
RUN uv sync --frozen --no-dev --no-install-project --no-install-workspace && \
    # Clean up Python cache during build
    find /app/.venv -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete && \
    find /app/.venv -type f -name "*.pyo" -delete

# Stage 2: Runtime (minimal image)
FROM python:3.11.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

# Install only runtime system dependencies (no build tools)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code and model in one layer (better caching)
COPY --chown=appuser:appuser src/api.py src/predict.py src/schemas.py src/config.py src/transformers.py ./src/
COPY --chown=appuser:appuser models/production_pipeline_v1.joblib ./models/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import json,sys,urllib.request; r=urllib.request.urlopen('http://localhost:8000/health', timeout=5); d=json.load(r); sys.exit(0 if d.get('status') in ('healthy','loading') else 1)"

# Run the API server (venv is already in PATH)
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]