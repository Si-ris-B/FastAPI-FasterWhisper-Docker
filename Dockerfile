# ============================================================
# Stage 1: Build wheels in a slim Python image
# ============================================================
FROM python:3.11-slim AS python-builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /build

RUN pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt


# ============================================================
# Stage 2: Production image — CUDA 12.x + cuDNN 9 (required for turbo)
# ============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS production

LABEL org.opencontainers.image.title="FastAPI Faster-Whisper STT Service"
LABEL org.opencontainers.image.description="Production STT service with turbo, batched, and live transcription"
LABEL org.opencontainers.image.version="4.0.0"

# Python env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONPATH=/app

# Application defaults (overridden by docker-compose env_file)
ENV LOG_LEVEL="INFO" \
    CLEANUP_AUDIO="True" \
    HOST="0.0.0.0" \
    PORT="8001"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# --- Dynamic UID/GID (avoids volume permission issues) ---
ARG UID=1001
ARG GID=1001

RUN addgroup --system --gid ${GID} appgroup && \
    adduser  --system --uid ${UID} --ingroup appgroup --no-create-home appuser

# Install Python dependencies from wheels
COPY --from=python-builder /build/wheels /wheels
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --no-index --find-links=/wheels -r /tmp/requirements.txt && \
    pip3 install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12 && \
    rm -rf /wheels /tmp/requirements.txt

# Copy application code
COPY --chown=appuser:appgroup src /app/src
COPY --chown=appuser:appgroup scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Persistent data directories (populated via volume mounts)
RUN mkdir -p /stt_app_data/audio_inbox /stt_app_data/model_cache && \
    chown -R appuser:appgroup /stt_app_data

USER appuser

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/status || exit 1

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []