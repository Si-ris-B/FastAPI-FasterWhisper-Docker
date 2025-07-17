#!/bin/sh
set -e

# Read environment variables with defaults
APP_HOST="${HOST:-0.0.0.0}"
APP_PORT="${PORT:-8001}"
APP_LOG_LEVEL_INPUT="${LOG_LEVEL:-INFO}"
APP_LOG_LEVEL_LOWERCASE=$(echo "${APP_LOG_LEVEL_INPUT}" | tr '[:upper:]' '[:lower:]')

# Dynamically construct LD_LIBRARY_PATH to include pip-installed NVIDIA libs
# This makes the container more portable.
PYTHON_EXEC="python"
SITE_PACKAGES_LD_PATH=$(${PYTHON_EXEC} -c 'import os, sys; sys.path = [p for p in sys.path if p]; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))' 2>/dev/null || echo "")

if [ -n "${SITE_PACKAGES_LD_PATH}" ]; then
  if [ -n "${LD_LIBRARY_PATH}" ]; then
    export LD_LIBRARY_PATH="${SITE_PACKAGES_LD_PATH}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${SITE_PACKAGES_LD_PATH}"
  fi
  echo "[entrypoint] Updated LD_LIBRARY_PATH to include pip NVIDIA libs"
fi

echo "[entrypoint] Final LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "[entrypoint] Starting Uvicorn..."
echo "[entrypoint]   - Host: ${APP_HOST}"
echo "[entrypoint]   - Port: ${APP_PORT}"
echo "[entrypoint]   - Log Level: ${APP_LOG_LEVEL_LOWERCASE}"

# Execute the main process. Note the module path for the app.
exec uvicorn src.stt_service.main:app \
    --host "${APP_HOST}" \
    --port "${APP_PORT}" \
    --workers 1 \
    --log-level "${APP_LOG_LEVEL_LOWERCASE}"
