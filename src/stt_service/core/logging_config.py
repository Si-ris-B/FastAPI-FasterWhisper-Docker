import logging
import asyncio
import json
import sys
from typing import Set
from fastapi import WebSocket

# Global set of connected clients
log_websockets: Set[WebSocket] = set()

class WebSocketLogHandler(logging.Handler):
    """A logging handler that sends log records to connected WebSockets."""
    def __init__(self, level=logging.NOTSET):
        super().__init__(level=level)
        self.loop: asyncio.AbstractEventLoop | None = None

    def get_loop(self) -> asyncio.AbstractEventLoop:
        if self.loop and not self.loop.is_closed():
            return self.loop
        try:
            self.loop = asyncio.get_running_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        return self.loop

    def emit(self, record: logging.LogRecord):
        """Format and queue the log record for broadcasting."""
        try:
            log_entry = self.format(record)
            log_data = {
                "type": "log",
                "level": record.levelname,
                "message": log_entry,
                "logger_name": record.name,
                "timestamp": record.created,
            }
            log_json = json.dumps(log_data)
            loop = self.get_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(self.broadcast(log_json), loop)
        except Exception:
            self.handleError(record)

    async def broadcast(self, message: str):
        """Asynchronously send message to all connected websockets."""
        if not log_websockets:
            return

        disconnected_clients = set()
        tasks = [client.send_text(message) for client in log_websockets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for client, result in zip(list(log_websockets), results):
            if isinstance(result, Exception):
                print(f"Error broadcasting log to {client.client}: {result}", file=sys.stderr)
                disconnected_clients.add(client)

        for client in disconnected_clients:
            if client in log_websockets:
                log_websockets.remove(client)

def setup_logging(log_level_str: str, logger_name: str) -> logging.Logger:
    """Configures logging for the application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)

    websocket_handler = WebSocketLogHandler()
    websocket_handler.setFormatter(log_formatter)

    app_logger = logging.getLogger(logger_name)
    if not app_logger.handlers:
        app_logger.setLevel(log_level)
        app_logger.addHandler(stream_handler)
        app_logger.addHandler(websocket_handler)
        app_logger.propagate = False
        app_logger.info("Application logger configured.")

    logging.getLogger("faster_whisper").setLevel(log_level)
    return app_logger

def get_logger(name: str) -> logging.Logger:
    """Gets a logger instance that is a child of the main app logger."""
    return logging.getLogger(f"stt_service.{name}")
