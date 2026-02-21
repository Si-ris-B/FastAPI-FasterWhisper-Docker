import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Set, Optional

from fastapi import WebSocket

# Global set of connected log WebSocket clients
log_websockets: Set[WebSocket] = set()


class StructuredFormatter(logging.Formatter):
    """Produces structured JSON log records for machine consumption."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "lineno": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


class HumanFormatter(logging.Formatter):
    """Human-readable log format for console output."""
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        base = f"[{ts}] {color}[{record.levelname:<8}]{self.RESET} [{record.name}] {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class WebSocketLogHandler(logging.Handler):
    """Broadcasts structured log entries to all connected WebSocket clients."""

    def __init__(self, level: int = logging.NOTSET, min_level: int = logging.DEBUG):
        super().__init__(level=level)
        self._min_level = min_level
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _get_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        try:
            loop = asyncio.get_running_loop()
            self._loop = loop
            return loop
        except RuntimeError:
            return self._loop

    def emit(self, record: logging.LogRecord):
        if record.levelno < self._min_level:
            return
        try:
            log_entry = {
                "type": "log",
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "lineno": record.lineno,
            }
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)

            log_json = json.dumps(log_entry)
            loop = self._get_loop()
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(_broadcast_log(log_json), loop)
        except Exception:
            self.handleError(record)


async def _broadcast_log(message: str):
    """Send a log message to all connected WebSocket clients."""
        if not log_websockets:
            return
    disconnected = set()
    tasks = {client: client.send_text(message) for client in list(log_websockets)}
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for client, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
            disconnected.add(client)
    for client in disconnected:
        log_websockets.discard(client)


def setup_logging(log_level_str: str, logger_name: str) -> logging.Logger:
    """Configure application logging with console + WebSocket handlers."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # --- Console handler (human-readable colored output) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(HumanFormatter())
    console_handler.setLevel(log_level)

    # --- WebSocket handler (structured JSON) ---
    ws_handler = WebSocketLogHandler(min_level=logging.DEBUG)
    ws_handler.setLevel(logging.DEBUG)

    # --- App logger ---
    app_logger = logging.getLogger(logger_name)
    if not app_logger.handlers:
        app_logger.setLevel(logging.DEBUG)  # Handler levels control actual output
        app_logger.addHandler(console_handler)
        app_logger.addHandler(ws_handler)
        app_logger.propagate = False

    # Propagate faster_whisper logs into our handler chain
    fw_logger = logging.getLogger("faster_whisper")
    if not fw_logger.handlers:
        fw_logger.setLevel(log_level)
        fw_logger.addHandler(console_handler)
        fw_logger.addHandler(ws_handler)
        fw_logger.propagate = False

    # Suppress noisy uvicorn access logs at debug
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    app_logger.info(f"Logging initialized — level={log_level_str.upper()}")
    return app_logger

def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the main app namespace."""
    return logging.getLogger(f"stt_service.{name}")
