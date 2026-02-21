from __future__ import annotations

import asyncio
import gc
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, AsyncGenerator

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)

# Models that use the turbo architecture (large-v3-turbo distillation)
TURBO_MODEL_NAMES = {"turbo", "large-v3-turbo"}
DISTIL_MODEL_NAMES = {"distil-large-v2", "distil-large-v3", "distil-medium.en", "distil-small.en"}


def _classify_model(name: str) -> str:
    lower = name.lower()
    if lower in TURBO_MODEL_NAMES:
        return "turbo"
    if any(lower.startswith(d) for d in DISTIL_MODEL_NAMES):
        return "distil"
    if "large" in lower:
        return "large"
    return "standard"


class WhisperManager:
    """
    Manages faster-whisper model lifecycle and all transcription modes:
      - Standard streaming transcription
      - Batched pipeline (max GPU throughput)
      - Live real-time transcription (WebSocket audio chunks)
    """

    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.batched_pipeline: Optional[BatchedInferencePipeline] = None
        self.config: Optional[Dict] = None
        self.model_type: str = "unknown"
        self.lock = asyncio.Lock()
        self._start_time: float = time.time()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    async def load_model(self, model_config: Dict) -> None:
        async with self.lock:
            if self.config == model_config and self.model is not None:
                logger.info(
                    f"Model '{model_config['model_size_or_path']}' is already loaded — skipping reload."
                )
                return

            if self.model:
                await self._unload_model_internal()

            name = model_config["model_size_or_path"]
            self.model_type = _classify_model(name)
            logger.info(
                f"Loading model '{name}' [type={self.model_type}] "
                f"device={model_config['device']} compute={model_config['compute_type']}"
            )
            t0 = time.perf_counter()
            try:
                settings.MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
                self.model = WhisperModel(
                    name,
                    device=model_config["device"],
                    device_index=model_config["device_index"],
                    compute_type=model_config["compute_type"],
                    cpu_threads=model_config["cpu_threads"],
                    num_workers=model_config["num_workers"],
                    download_root=str(settings.MODEL_CACHE_PATH),
                )
                # Also prepare batched pipeline (wraps the same model — no extra VRAM)
                self.batched_pipeline = BatchedInferencePipeline(self.model)
                self.config = model_config
                elapsed = time.perf_counter() - t0
                logger.info(
                    f"✅ Model '{name}' loaded in {elapsed:.2f}s "
                    f"(batched pipeline ready)"
                )
            except Exception as exc:
                logger.error(f"❌ Failed to load model '{name}': {exc}", exc_info=True)
                await self._unload_model_internal()
                raise

    async def unload_model(self) -> None:
        async with self.lock:
            await self._unload_model_internal()

    async def _unload_model_internal(self) -> None:
        if not self.model:
            return
        name = self.config.get("model_size_or_path", "?") if self.config else "?"
        logger.warning(f"Unloading model: {name}")
        del self.batched_pipeline
        del self.model
        gc.collect()
        self.model = None
        self.batched_pipeline = None
        self.config = None
        self.model_type = "unknown"
        logger.info("Model unloaded — resources released.")

    def status(self) -> Dict:
        return {
            "service_status": "model_loaded" if self.model else "idle_no_model",
            "loaded_model_config": self.config,
            "model_type": self.model_type,
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def sanitize_path(self, client_path: str) -> Path:
        if not client_path:
            raise ValueError("file_path cannot be empty.")
        normalized = os.path.normpath(client_path).lstrip("/\\")
        if ".." in normalized.split(os.sep):
            raise ValueError("Path traversal ('..') is forbidden.")
        base = settings.SHARED_AUDIO_PATH.resolve()
        full = (base / normalized).resolve()
        if not str(full).startswith(str(base)):
            raise ValueError("Path resolves outside the designated audio inbox.")
        return full

    def _cleanup_if_needed(self, path: Path) -> None:
        if settings.CLEANUP_AUDIO and path.is_file():
            logger.info(f"Cleaning up audio file: {path.name}")
            path.unlink()

    # ------------------------------------------------------------------
    # Transcription argument preparation
    # ------------------------------------------------------------------

    def _build_transcribe_args(self, params, exclude_keys=None) -> Dict:
        exclude = set(exclude_keys or [])
        exclude.update({"file_path"})
        args = params.model_dump(exclude_unset=True, exclude=exclude)
        # Expand VAD params sub-model
        if params.vad_parameters:
            args["vad_parameters"] = params.vad_parameters.model_dump(exclude_unset=True)
        # Resolve suppress_tokens=-1 to actual non-speech token list
        if -1 in args.get("suppress_tokens", []):
            args = self._resolve_suppress_tokens(args, params)
        return args
        
    def _resolve_suppress_tokens(self, args: Dict, params) -> Dict:
            try:
                from faster_whisper.tokenizer import Tokenizer
            tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=getattr(params, "task", "transcribe"),
                language=getattr(params, "language", None),
            )
                non_speech = list(tokenizer.non_speech_tokens)
            user_tokens = [t for t in args["suppress_tokens"] if t >= 0]
            args["suppress_tokens"] = sorted(set(user_tokens + non_speech))
        except Exception as exc:
            logger.error(f"Error resolving suppress_tokens: {exc}", exc_info=True)
        return args

    # ------------------------------------------------------------------
    # Mode 1: Standard streaming transcription
    # ------------------------------------------------------------------

    async def transcribe_stream(
        self,
        file_path_str: str,
        params,
    ) -> AsyncGenerator[str, None]:
        """
        Stream NDJSON lines: info → segments → final | error
        Yields one JSON line per event so clients can parse incrementally.
        """
        request_id = f"txn-{time.time_ns()}"
        full_path: Optional[Path] = None
        t0 = time.perf_counter()

        try:
            full_path = self.sanitize_path(file_path_str)
            if not full_path.is_file():
                raise FileNotFoundError(f"Audio file not found in inbox: {full_path.name}")

            logger.info(f"[{request_id}] START standard transcription → {full_path.name}")

            # Decode audio outside the model lock (I/O bound)
            logger.debug(f"[{request_id}] Decoding audio...")
            audio_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: decode_audio(str(full_path), sampling_rate=16000)
            )
            duration_s = len(audio_input) / 16000
            logger.info(f"[{request_id}] Audio decoded — duration={duration_s:.1f}s")

            async with self.lock:
                if not self.model:
                    raise RuntimeError("No model loaded. Call /load_model first.")
                transcribe_args = self._build_transcribe_args(params)
                logger.debug(f"[{request_id}] Transcription args: {transcribe_args}")
                segments_iter, info_obj = self.model.transcribe(audio=audio_input, **transcribe_args)

            # Emit audio info header
            info_payload = {
                "type": "info",
                "request_id": request_id,
                "data": {
                "language": info_obj.language,
                    "language_probability": round(info_obj.language_probability, 4),
                    "duration": round(info_obj.duration, 3),
                    "duration_after_vad": round(info_obj.duration_after_vad, 3),
                    "all_language_probs": dict(
                        sorted(
                            (info_obj.all_language_probs or {}).items(),
                            key=lambda x: -x[1]
                        )[:10]  # top-10 language candidates
                    ),
                },
            }
            yield json.dumps(info_payload) + "\n"
            logger.info(
                f"[{request_id}] Detected language='{info_obj.language}' "
                f"(prob={info_obj.language_probability:.2%})"
            )

            # Stream segments
            seg_count = 0
            for segment in segments_iter:
                seg_count += 1
                seg_data = {
                    "id": segment.id,
                    "seek": segment.seek,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text,
                    "tokens": segment.tokens,
                    "avg_logprob": round(segment.avg_logprob, 4),
                    "compression_ratio": round(segment.compression_ratio, 4),
                    "no_speech_prob": round(segment.no_speech_prob, 4),
                    "temperature": segment.temperature,
                }
                if segment.words:
                    seg_data["words"] = [
                        {
                            "word": w.word,
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "probability": round(w.probability, 4),
                        }
                        for w in segment.words
                    ]
                logger.debug(
                    f"[{request_id}] Segment {seg_count}: "
                    f"[{segment.start:.2f}→{segment.end:.2f}] {segment.text[:60]}"
                )
                yield json.dumps({"type": "segment", "data": seg_data}) + "\n"

            elapsed = time.perf_counter() - t0
            rtf = elapsed / max(duration_s, 0.001)
            logger.info(
                f"[{request_id}] ✅ Done — {seg_count} segments in {elapsed:.2f}s "
                f"(RTF={rtf:.3f})"
            )
            yield json.dumps({
                "type": "final",
                "request_id": request_id,
                "segment_count": seg_count,
                "elapsed_seconds": round(elapsed, 3),
                "real_time_factor": round(rtf, 4),
                "message": "Transcription complete.",
            }) + "\n"

        except Exception as exc:
            logger.error(f"[{request_id}] ❌ Stream error: {exc}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "request_id": request_id,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }) + "\n"
        finally:
            if full_path:
                self._cleanup_if_needed(full_path)

    # ------------------------------------------------------------------
    # Mode 2: Batched inference (high throughput)
    # ------------------------------------------------------------------

    async def transcribe_batched_stream(
        self,
        file_path_str: str,
        params,
    ) -> AsyncGenerator[str, None]:
        """
        Uses BatchedInferencePipeline for maximum GPU throughput.
        Best for large audio files or high-volume processing queues.
        Streams NDJSON same as standard transcription.
        """
        request_id = f"batch-{time.time_ns()}"
        full_path: Optional[Path] = None
        t0 = time.perf_counter()

        try:
            full_path = self.sanitize_path(file_path_str)
            if not full_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {full_path.name}")

            logger.info(f"[{request_id}] START batched transcription → {full_path.name}")

            audio_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: decode_audio(str(full_path), sampling_rate=16000)
            )
            duration_s = len(audio_input) / 16000
            logger.info(f"[{request_id}] Audio decoded — duration={duration_s:.1f}s")

            # Extract batch-specific args
            batch_size = params.batch_size
            chunk_length = params.chunk_length
            transcribe_args = self._build_transcribe_args(
                params, exclude_keys=["batch_size", "chunk_length"]
            )

            logger.debug(
                f"[{request_id}] Batched args: batch_size={batch_size} "
                f"chunk_length={chunk_length} args={transcribe_args}"
            )

            async with self.lock:
                if not self.batched_pipeline:
                    raise RuntimeError("No batched pipeline available. Load a model first.")

                kwargs = dict(audio=audio_input, **transcribe_args)
                if chunk_length is not None:
                    kwargs["chunk_length"] = chunk_length

                segments_iter, info_obj = self.batched_pipeline.transcribe(
                    batch_size=batch_size, **kwargs
                )

            yield json.dumps({
                "type": "info",
                "request_id": request_id,
                "mode": "batched",
                "data": {
                    "language": info_obj.language,
                    "language_probability": round(info_obj.language_probability, 4),
                    "duration": round(info_obj.duration, 3),
                    "duration_after_vad": round(info_obj.duration_after_vad, 3),
                    "batch_size": batch_size,
                    "chunk_length": chunk_length,
                },
            }) + "\n"

            seg_count = 0
            for segment in segments_iter:
                seg_count += 1
                seg_data = {
                    "id": seg_count,
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text,
                }
                if hasattr(segment, "words") and segment.words:
                    seg_data["words"] = [
                        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                        for w in segment.words
                    ]
                logger.debug(
                    f"[{request_id}] Batched seg {seg_count}: "
                    f"[{segment.start:.2f}→{segment.end:.2f}] {segment.text[:60]}"
                )
                yield json.dumps({"type": "segment", "data": seg_data}) + "\n"

            elapsed = time.perf_counter() - t0
            rtf = elapsed / max(duration_s, 0.001)
            logger.info(
                f"[{request_id}] ✅ Batched done — {seg_count} segments "
                f"in {elapsed:.2f}s (RTF={rtf:.3f})"
            )
            yield json.dumps({
                "type": "final",
                "request_id": request_id,
                "mode": "batched",
                "segment_count": seg_count,
                "elapsed_seconds": round(elapsed, 3),
                "real_time_factor": round(rtf, 4),
            }) + "\n"

        except Exception as exc:
            logger.error(f"[{request_id}] ❌ Batched stream error: {exc}", exc_info=True)
            yield json.dumps({
                "type": "error",
                "request_id": request_id,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }) + "\n"
        finally:
            if full_path:
                self._cleanup_if_needed(full_path)

    # ------------------------------------------------------------------
    # Mode 3: Language detection only
    # ------------------------------------------------------------------

    async def detect_language(self, file_path_str: str, params) -> Dict:
        full_path = self.sanitize_path(file_path_str)
        if not full_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {full_path.name}")

        audio_input = await asyncio.get_event_loop().run_in_executor(
            None, lambda: decode_audio(str(full_path), sampling_rate=16000)
        )
        # Sample first N segments worth of audio for detection
        sample_audio = audio_input[:16000 * 30]  # cap at 30s

        async with self.lock:
            if not self.model:
                raise RuntimeError("No model loaded.")
            _, info = self.model.transcribe(
                audio=sample_audio,
                language=None,
                language_detection_segments=params.language_detection_segments,
                language_detection_threshold=params.language_detection_threshold,
                beam_size=1,
                without_timestamps=True,
                max_new_tokens=1,
            )

        result = {
            "detected_language": info.language,
            "language_probability": round(info.language_probability, 4),
            "all_language_probs": dict(
                sorted(
                    (info.all_language_probs or {}).items(),
                    key=lambda x: -x[1]
                )[:20]
            ),
        }
        logger.info(
            f"Language detection: '{info.language}' "
            f"({info.language_probability:.2%}) for {full_path.name}"
        )
        return result

    # ------------------------------------------------------------------
    # Mode 4: Live / real-time WebSocket transcription
    # ------------------------------------------------------------------

    async def live_transcription_session(
        self,
        websocket,
        params,
    ) -> None:
        """
        Handle a live transcription WebSocket session.

        Protocol (client → server):
          - Binary frames: raw 16kHz mono float32 PCM audio bytes
          - Text frame: JSON {"type": "config", ...} — optional reconfigure
          - Text frame: JSON {"type": "stop"} — graceful stop

        Protocol (server → client):
          - {"type": "partial", "text": "...", "buffer_duration": 1.2}
          - {"type": "final", "text": "...", "segments": [...]}
          - {"type": "error", "message": "..."}
          - {"type": "status", "message": "..."}
        """
        session_id = f"live-{time.time_ns()}"
        logger.info(f"[{session_id}] Live transcription session started")

        audio_buffer = np.array([], dtype=np.float32)
        min_samples = int(params.min_chunk_duration_s * 16000)
        max_samples = int(params.max_buffer_duration_s * 16000)

        await websocket.send_text(json.dumps({
            "type": "status",
            "session_id": session_id,
            "message": "Session started. Send 16kHz mono float32 PCM audio frames.",
            "params": {
                "min_chunk_duration_s": params.min_chunk_duration_s,
                "max_buffer_duration_s": params.max_buffer_duration_s,
                "vad_filter": params.vad_filter,
            }
        }))

        try:
            from fastapi import WebSocketDisconnect
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[{session_id}] No audio received for 30s — closing.")
                    break

                # Handle binary audio frames
                if "bytes" in message and message["bytes"]:
                    chunk_bytes = message["bytes"]
                    chunk_audio = np.frombuffer(chunk_bytes, dtype=np.float32)
                    audio_buffer = np.concatenate([audio_buffer, chunk_audio])

                    buffer_duration = len(audio_buffer) / 16000
                    logger.debug(
                        f"[{session_id}] Buffer: {buffer_duration:.2f}s "
                        f"({len(audio_buffer)} samples)"
                    )

                    # Transcribe when buffer is large enough OR maxed out
                    should_transcribe = (
                        len(audio_buffer) >= min_samples
                        or len(audio_buffer) >= max_samples
                    )
                    if should_transcribe:
                        result = await self._run_live_transcription(
                            session_id, audio_buffer, params
                        )
                        audio_buffer = np.array([], dtype=np.float32)  # flush buffer

                        if result["text"].strip():
                            await websocket.send_text(json.dumps({
                                "type": "partial",
                                "session_id": session_id,
                                "text": result["text"],
                                "segments": result["segments"],
                                "buffer_duration": round(buffer_duration, 2),
                            }))
                    else:
                        # Acknowledge receipt
                        await websocket.send_text(json.dumps({
                            "type": "buffering",
                            "session_id": session_id,
                            "buffer_duration": round(buffer_duration, 2),
                            "min_needed": params.min_chunk_duration_s,
                        }))

                # Handle control messages
                elif "text" in message and message["text"]:
                    try:
                        ctrl = json.loads(message["text"])
                    except json.JSONDecodeError:
                        continue

                    if ctrl.get("type") == "stop":
                        logger.info(f"[{session_id}] Stop signal received.")
                        # Flush remaining buffer
                        if len(audio_buffer) > 0:
                            result = await self._run_live_transcription(
                                session_id, audio_buffer, params
                            )
                            await websocket.send_text(json.dumps({
                                "type": "final",
                                "session_id": session_id,
                                "text": result["text"],
                                "segments": result["segments"],
                            }))
                        break

        except Exception as exc:
            logger.error(f"[{session_id}] Live session error: {exc}", exc_info=True)
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "session_id": session_id,
                    "message": str(exc),
                }))
            except Exception:
                pass
        finally:
            logger.info(f"[{session_id}] Live transcription session ended.")

    async def _run_live_transcription(
        self, session_id: str, audio: np.ndarray, params
    ) -> Dict:
        """Run transcription on a buffered audio chunk."""
        args = {
            "language": params.language,
            "task": params.task,
            "beam_size": params.beam_size,
            "word_timestamps": params.word_timestamps,
            "vad_filter": params.vad_filter,
            "without_timestamps": params.without_timestamps,
            "suppress_blank": params.suppress_blank,
        }
        if params.vad_parameters:
            args["vad_parameters"] = params.vad_parameters.model_dump(exclude_unset=True)
        if params.hotwords:
            args["hotwords"] = params.hotwords

        async with self.lock:
            if not self.model:
                raise RuntimeError("No model loaded.")
            segments_iter, info = self.model.transcribe(audio=audio, **args)
            segments = list(segments_iter)

        text = " ".join(s.text for s in segments).strip()
        seg_list = [
            {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text}
            for s in segments
        ]
        logger.debug(f"[{session_id}] Live chunk → '{text[:80]}'")
        return {"text": text, "segments": seg_list, "language": info.language}


# Singleton
whisper_manager = WhisperManager()
