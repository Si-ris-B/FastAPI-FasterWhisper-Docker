from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # --- Application Behavior ---
    LOG_LEVEL: str = "INFO"
    CLEANUP_AUDIO: bool = True
    APP_LOGGER_NAME: str = "stt_service"

    # --- Server Configuration ---
    # These are read from environment variables set by docker-compose or docker run
    APP_PORT: int = 8001
    APP_HOST: str = "0.0.0.0"

    # --- Fixed Internal Paths ---
    # These paths are fixed inside the container.
    # The host paths are mapped to these via volumes in docker-compose.
    SHARED_AUDIO_PATH: Path = Path("/stt_app_data/audio_inbox")
    MODEL_CACHE_PATH: Path = Path("/stt_app_data/model_cache")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Create a single, importable settings instance
settings = Settings()
