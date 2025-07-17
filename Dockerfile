# Stage 1: Python base (for building wheels, doesn't need CUDA)
FROM python:3.10-slim AS python-builder-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app_build
RUN pip install --upgrade pip wheel setuptools
COPY requirements.txt .
# Build wheels for all dependencies
RUN pip wheel --no-cache-dir --wheel-dir /app_build/wheels -r requirements.txt


# Stage 2: Final application image using NVIDIA CUDA base
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS production

# Set Python-related ENV VARS again for this stage
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Application-specific ENV VARS (defaults, can be overridden)
ENV LOG_LEVEL="INFO" \
    CLEANUP_AUDIO="True" \
    HOST="0.0.0.0" \
    PORT="8001"

# Add python path for src layout
ENV PYTHONPATH=/app

# Install Python3, pip, and essential tools into the NVIDIA image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user to run the application
RUN addgroup --system --gid 1001 appgroup \
    && adduser --system --uid 1001 --ingroup appgroup appuser

# Install dependencies from wheels
COPY --from=python-builder-base /app_build/wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt && \
    pip3 install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12 && \
    rm -rf /wheels /app/requirements.txt

# Copy application code and scripts
COPY --chown=appuser:appgroup src /app/src
COPY --chown=appuser:appgroup scripts/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Create the FIXED internal directories that will be used for volume mounts.
RUN mkdir -p /stt_app_data/audio_inbox && chown -R appuser:appgroup /stt_app_data/audio_inbox \
    && mkdir -p /stt_app_data/model_cache && chown -R appuser:appgroup /stt_app_data/model_cache

# Switch to the non-root user
USER appuser

EXPOSE ${PORT}
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
