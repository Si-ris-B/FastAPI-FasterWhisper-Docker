from __future__ import annotations

import json
import time
from typing import Dict, Optional

from fastapi import (
    FastAPI, Body, HTTPException, Query, status,
    WebSocket, WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, JSONResponse

from .core.config import settings, AVAILABLE_MODELS
from .core.logging_config import setup_logging, log_websockets
from .models.transcription import (
    ModelConfigParams,
    TranscribeRequest,
    BatchedTranscribeRequest,
    DetectLanguageRequest,
    LiveTranscriptionParams,
    ServiceStatus,
)
from .services.whisper_manager import whisper_manager

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FastAPI Faster-Whisper STT Service",
    description=(
        "Production-grade Speech-to-Text service powered by faster-whisper. "
        "Supports whisper-turbo, batched inference, real-time WebSocket transcription, "
        "and live log streaming."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)
logger = setup_logging(settings.LOG_LEVEL, settings.APP_LOGGER_NAME)

# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info(f"━━━ STT Service Starting (v{app.version}) ━━━")
    try:
        settings.SHARED_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
        settings.MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Audio inbox:   {settings.SHARED_AUDIO_PATH.resolve()}")
        logger.info(f"Model cache:   {settings.MODEL_CACHE_PATH.resolve()}")
        logger.info(f"Log level:     {settings.LOG_LEVEL}")
        logger.info(f"Cleanup audio: {settings.CLEANUP_AUDIO}")
    except Exception:
        logger.error("CRITICAL: Failed to create service directories.", exc_info=True)
    logger.info("Service initialized — IDLE (no model loaded)")


@app.on_event("shutdown")
async def shutdown_event():
    logger.warning("━━━ STT Service Shutting Down ━━━")
    await whisper_manager.unload_model()
    logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/status",
    summary="Service status",
    response_model=ServiceStatus,
    tags=["Management"],
)
async def get_status():
    """Returns current service state, loaded model config, and uptime."""
    return whisper_manager.status()


@app.get(
    "/models",
    summary="List available models",
    tags=["Management"],
)
async def list_models():
    """Returns all supported model identifiers, grouped by family."""
    return {
        "turbo": ["turbo", "large-v3-turbo"],
        "large": ["large-v1", "large-v2", "large-v3"],
        "distil": ["distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"],
        "standard": ["tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en"],
        "note": "You can also pass a HuggingFace repo ID or local CTranslate2 model path.",
    }


@app.post(
    "/load_model",
    summary="Load a Whisper model",
    status_code=status.HTTP_200_OK,
    tags=["Management"],
)
async def api_load_model(config: ModelConfigParams = Body(...)):
    """
    Load a faster-whisper model into memory.

    **Turbo models** (`turbo` / `large-v3-turbo`) offer ~8x faster inference than
    `large-v3` at comparable accuracy — recommended as the default choice.

    The batched inference pipeline is automatically prepared alongside the model.
    """
    try:
        await whisper_manager.load_model(config.model_dump())
        return {
            "status": "success",
            "message": f"Model '{config.model_size_or_path}' loaded.",
            "config": whisper_manager.config,
            "model_type": whisper_manager.model_type,
        }
    except Exception as exc:
        logger.error("API /load_model failed.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@app.post(
    "/unload_model",
    summary="Unload the active model",
    status_code=status.HTTP_200_OK,
    tags=["Management"],
)
async def api_unload_model():
    """Free GPU/CPU memory by unloading the current model."""
    await whisper_manager.unload_model()
    return {"status": "success", "message": "Model unloaded and resources released."}


# ---------------------------------------------------------------------------
# Transcription endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/transcribe",
    summary="Stream-transcribe a file (standard mode)",
    tags=["Transcription"],
)
async def api_transcribe(request: TranscribeRequest = Body(...)):
    """
    Transcribe an audio file using the standard faster-whisper inference.

    **Response**: NDJSON stream — one JSON object per line:
    - `{"type": "info", ...}` — audio metadata, detected language
    - `{"type": "segment", "data": {...}}` — transcription segment
    - `{"type": "final", ...}` — completion stats (RTF, segment count)
    - `{"type": "error", ...}` — error details

    Place the audio file in your `STT_AUDIO_INBOX_PATH` and reference it by filename.
    """
    if not whisper_manager.model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model loaded. POST to /load_model first.",
        )
    stream = whisper_manager.transcribe_stream(request.file_path, request.params)
    return StreamingResponse(stream, media_type="application/x-ndjson")


@app.post(
    "/transcribe/batched",
    summary="Stream-transcribe a file (batched pipeline — high throughput)",
    tags=["Transcription"],
)
async def api_transcribe_batched(request: BatchedTranscribeRequest = Body(...)):
    """
    Transcribe using `BatchedInferencePipeline` for maximum GPU throughput.

    Processes the audio in parallel chunks, significantly faster than standard mode
    for long files. Ideal for batch processing queues.

    **Response format**: Same NDJSON stream as `/transcribe` with an additional
    `"mode": "batched"` field.
    """
    if not whisper_manager.batched_pipeline:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model loaded. POST to /load_model first.",
        )
    stream = whisper_manager.transcribe_batched_stream(request.file_path, request.params)
    return StreamingResponse(stream, media_type="application/x-ndjson")


@app.post(
    "/detect_language",
    summary="Detect the language of an audio file",
    tags=["Transcription"],
)
async def api_detect_language(request: DetectLanguageRequest = Body(...)):
    """
    Run fast language detection on an audio file without full transcription.
    Returns the top detected language and probability scores for all candidates.
    """
    if not whisper_manager.model:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model loaded. POST to /load_model first.",
        )
    try:
        result = await whisper_manager.detect_language(request.file_path, request)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        logger.error("Language detection failed.", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        )


# ---------------------------------------------------------------------------
# WebSocket endpoints
# ---------------------------------------------------------------------------

@app.websocket("/ws/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    """
    **Real-time log streaming**.

    Connect with any WebSocket client to receive a live structured JSON stream
    of all application logs.

    Each message is a JSON object:
    ```json
    {
      "type": "log",
      "timestamp": "2025-01-01T12:00:00.000Z",
      "level": "INFO",
      "logger": "stt_service.whisper_manager",
      "message": "Model loaded in 2.3s",
      "module": "whisper_manager",
      "lineno": 82
    }
    ```

    **Filter by level**: send a text message `{"min_level": "WARNING"}` to
    filter to WARNING and above.
    """
    await websocket.accept()
    client = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Log WebSocket connected: {client}")
    log_websockets.add(websocket)

    try:
        # Send connection acknowledgement
        await websocket.send_text(json.dumps({
            "type": "connected",
            "message": "Real-time log stream active.",
            "service_version": settings.APP_VERSION,
        }))
        while True:
            # Keep connection alive; handle optional control messages
            try:
                data = await websocket.receive_text()
                ctrl = json.loads(data)
                if ctrl.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except (json.JSONDecodeError, KeyError):
                pass
    except WebSocketDisconnect:
        logger.info(f"Log WebSocket disconnected: {client}")
    finally:
        log_websockets.discard(websocket)


@app.websocket("/ws/live-transcribe")
async def websocket_live_transcribe(websocket: WebSocket):
    """
    **Real-time live transcription via WebSocket**.

    Stream raw 16 kHz mono float32 PCM audio from a microphone or audio source
    and receive rolling transcription results.

    **Handshake**: Send initial config as JSON text frame before streaming audio:
    ```json
    {
      "type": "config",
      "language": "en",
      "task": "transcribe",
      "beam_size": 1,
      "vad_filter": true,
      "min_chunk_duration_s": 1.0,
      "max_buffer_duration_s": 10.0
    }
    ```

    Then stream binary frames containing `float32` PCM samples at 16 kHz.

    Send `{"type": "stop"}` to flush the buffer and end the session.
    """
    await websocket.accept()
    client = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Live transcription WebSocket connected: {client}")

    if not whisper_manager.model:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "No model loaded. POST to /load_model first.",
        }))
        await websocket.close(code=1011)
        return

    # Receive optional config frame (with timeout)
    params = LiveTranscriptionParams()
    try:
        import asyncio
        msg = await asyncio.wait_for(websocket.receive(), timeout=5.0)
        if "text" in msg:
            ctrl = json.loads(msg["text"])
            if ctrl.get("type") == "config":
                params = LiveTranscriptionParams(**{
                    k: v for k, v in ctrl.items() if k != "type"
                })
                logger.info(f"Live session configured: {params.model_dump()}")
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception) as exc:
        logger.debug(f"No config received ({exc}), using defaults.")

    try:
        await whisper_manager.live_transcription_session(websocket, params)
    except WebSocketDisconnect:
        logger.info(f"Live transcription client disconnected: {client}")
    except Exception as exc:
        logger.error(f"Live transcription error: {exc}", exc_info=True)
    finally:
        logger.info(f"Live transcription session closed: {client}")