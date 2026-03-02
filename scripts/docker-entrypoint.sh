#!/bin/sh
set -e

APP_HOST="${HOST:-0.0.0.0}"
APP_PORT="${PORT:-8001}"
LOG_LEVEL_INPUT="${LOG_LEVEL:-INFO}"
LOG_LEVEL_LC=$(echo "${LOG_LEVEL_INPUT}" | tr '[:upper:]' '[:lower:]')

# Dynamically build LD_LIBRARY_PATH from pip-installed NVIDIA libs
PYTHON_EXEC="python"
SITE_PACKAGES_LD=$(
  ${PYTHON_EXEC} -c '
import os
try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
    paths = [
        os.path.dirname(nvidia.cublas.lib.__file__),
        os.path.dirname(nvidia.cudnn.lib.__file__),
    ]
    print(":".join(paths))
except ImportError:
    pass
' 2>/dev/null || echo ""
)

if [ -n "${SITE_PACKAGES_LD}" ]; then
  if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${SITE_PACKAGES_LD}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${SITE_PACKAGES_LD}"
  fi
  echo "[entrypoint] NVIDIA libs found: ${SITE_PACKAGES_LD}"
fi

echo "[entrypoint] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[entrypoint] FastAPI Faster-Whisper STT Service v4.0"
echo "[entrypoint] Host:       ${APP_HOST}:${APP_PORT}"
echo "[entrypoint] Log level:  ${LOG_LEVEL_INPUT}"
echo "[entrypoint] LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "[entrypoint] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exec uvicorn src.stt_service.main:app \
    --host "${APP_HOST}" \
    --port "${APP_PORT}" \
    --workers 1 \
    --log-level "${LOG_LEVEL_LC}" \
    --loop uvloop \
    --http httptools