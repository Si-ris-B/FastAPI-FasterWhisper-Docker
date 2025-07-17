import time
from typing import Dict
from fastapi import (FastAPI, Body, HTTPException, status, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import StreamingResponse

from .core.config import settings
from .core.logging_config import setup_logging, log_websockets
from .models.transcription import ModelConfigParams, TranscribeRequest, BatchedTranscribeRequest, DetectLanguageRequest
from .services.whisper_manager import whisper_manager

app = FastAPI(
    title="On-Demand STT Service",
    description="Dynamically load/unload Whisper models via API.",
    version="3.1.0"
)
logger = setup_logging(settings.LOG_LEVEL, settings.APP_LOGGER_NAME)

@app.on_event("startup")
async def startup_event():
    logger.info(f"---- STT Service Starting (v{app.version}) ----")
    try:
        settings.SHARED_AUDIO_PATH.mkdir(parents=True, exist_ok=True)
        settings.MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"Shared audio path: {settings.SHARED_AUDIO_PATH.resolve()}")
        logger.info(f"Model cache path: {settings.MODEL_CACHE_PATH.resolve()}")
    except Exception:
        logger.error("CRITICAL: Failed to create service directories.", exc_info=True)
    logger.info("Service initialized in IDLE state.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.warning("---- STT Service Shutting Down ----")
    await whisper_manager.unload_model()
    logger.info("Service shutdown complete.")

@app.get("/status", summary="Get current service status", response_model=Dict)
async def get_stt_status_api():
    async with whisper_manager.lock:
        return {
            "service_status": "model_loaded" if whisper_manager.model else "idle_no_model",
            "loaded_model_config": whisper_manager.config,
        }

@app.post("/load_model", summary="Load a specific Whisper model", status_code=status.HTTP_200_OK)
async def api_load_model(config: ModelConfigParams = Body(...)):
    try:
        await whisper_manager.load_model(config.dict())
        return {"status": "success", "message": "Model loaded successfully.", "config": whisper_manager.config}
    except Exception as e:
        logger.error("API: Model loading failed.", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/unload_model", summary="Unload the currently active model", status_code=status.HTTP_200_OK)
async def api_unload_model():
    await whisper_manager.unload_model()
    return {"status": "success", "message": "Model unloaded."}

@app.post("/transcribe", summary="Transcribe audio using a streaming response")
async def api_transcribe(request_data: TranscribeRequest = Body(...)):
    if not whisper_manager.model:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No model loaded.")
    stream = whisper_manager.transcribe_stream(request_data.file_path, request_data.params)
    return StreamingResponse(stream, media_type="application/x-ndjson")

@app.websocket("/ws/logs")
async def websocket_log_endpoint(websocket: WebSocket):
    await websocket.accept()
    log_websockets.add(websocket)
    client_info = f"{websocket.client.host}:{websocket.client.port}"
    logger.info(f"Log WebSocket client connected: {client_info}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info(f"Log WebSocket client disconnected: {client_info}")
    finally:
        if websocket in log_websockets:
            log_websockets.remove(websocket)
