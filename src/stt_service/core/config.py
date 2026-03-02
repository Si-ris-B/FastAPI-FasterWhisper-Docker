from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Application Behavior ---
    LOG_LEVEL: str = "INFO"
    CLEANUP_AUDIO: bool = True
    APP_LOGGER_NAME: str = "stt_service"
    APP_VERSION: str = "4.0.0"

    # --- Server Configuration ---
    APP_PORT: int = 8001
    APP_HOST: str = "0.0.0.0"

    # --- Fixed Internal Paths ---
    SHARED_AUDIO_PATH: Path = Path("/stt_app_data/audio_inbox")
    MODEL_CACHE_PATH: Path = Path("/stt_app_data/model_cache")

    # --- Live Transcription ---
    LIVE_SAMPLE_RATE: int = 16000
    LIVE_CHUNK_DURATION_MS: int = 500  # ms of audio per WebSocket chunk

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()


# All available Whisper models including turbo variants
AVAILABLE_MODELS = [
    # Standard models
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    # Turbo models (faster-whisper >= 1.0.0)
    "turbo",           # alias for large-v3-turbo
    "large-v3-turbo",  # 809M params, ~8x faster than large-v3
    "distil-large-v2",
    "distil-large-v3",
    "distil-medium.en",
    "distil-small.en",
]