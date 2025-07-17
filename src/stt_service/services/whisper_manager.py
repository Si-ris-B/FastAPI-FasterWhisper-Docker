import asyncio
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Optional, Dict

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger(__name__)

class WhisperManager:
    def __init__(self):
        self.model: Optional[WhisperModel] = None
        self.config: Optional[Dict] = None
        self.lock = asyncio.Lock()

    async def load_model(self, model_config: Dict):
        async with self.lock:
            if self.config == model_config:
                logger.info(f"Model '{model_config.get('model_size_or_path')}' is already loaded.")
                return

            if self.model:
                await self._unload_model_internal()

            logger.info(f"Attempting to load model with config: {model_config}")
            start_time = time.time()
            try:
                settings.MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
                self.model = WhisperModel(
                    model_config['model_size_or_path'],
                    device=model_config['device'],
                    device_index=model_config['device_index'],
                    compute_type=model_config['compute_type'],
                    cpu_threads=model_config['cpu_threads'],
                    num_workers=model_config['num_workers'],
                    download_root=str(settings.MODEL_CACHE_PATH)
                )
                self.config = model_config
                load_time = time.time() - start_time
                logger.info(f"Successfully loaded model '{model_config['model_size_or_path']}' in {load_time:.2f}s.")
            except Exception as e:
                logger.error(f"FATAL: Failed to load model: {e}", exc_info=True)
                await self._unload_model_internal()
                raise

    async def unload_model(self):
        async with self.lock:
            await self._unload_model_internal()

    async def _unload_model_internal(self):
        if not self.model:
            return
        logger.warning(f"Unloading model: {self.config.get('model_size_or_path', '?')}")
        del self.model
        gc.collect()
        self.model = None
        self.config = None
        logger.info("Model unloaded and resources released.")

    def sanitize_path(self, client_relative_path: str) -> Path:
        if not client_relative_path:
            raise ValueError("File path cannot be empty.")
        normalized_path = os.path.normpath(client_relative_path).lstrip('/\\')
        if ".." in normalized_path.split(os.sep):
            raise ValueError("Path traversal ('..') is forbidden.")
        base_path = settings.SHARED_AUDIO_PATH.resolve()
        full_path = (base_path / normalized_path).resolve()
        if not str(full_path).startswith(str(base_path)):
             raise ValueError("Path resolves outside designated shared audio directory.")
        return full_path

    def cleanup_audio_source(self, source_path: Path):
        if settings.CLEANUP_AUDIO and source_path.is_file():
            logger.info(f"Cleaning up source audio: {source_path}")
            source_path.unlink()

    def _prepare_transcription_args(self, params):
        args = params.dict(exclude_unset=True)
        if params.vad_parameters:
            args['vad_parameters'] = params.vad_parameters.dict(exclude_unset=True)
        
        if -1 in args.get('suppress_tokens', []):
            try:
                from faster_whisper.tokenizer import Tokenizer
                tokenizer = Tokenizer(self.model.hf_tokenizer, self.model.model.is_multilingual, task=params.task, language=params.language)
                non_speech = list(tokenizer.non_speech_tokens)
                user_set = [t for t in args['suppress_tokens'] if t >= 0]
                args['suppress_tokens'] = sorted(list(set(user_set + non_speech)))
            except Exception as e:
                logger.error(f"Error processing suppress_tokens: {e}", exc_info=True)
                raise ValueError("Internal error processing suppress_tokens.")
        return args

    async def transcribe_stream(self, file_path_str: str, params):
        request_id = f"txn-{time.time_ns()}"
        full_path_obj = None
        try:
            full_path_obj = self.sanitize_path(file_path_str)
            if not full_path_obj.is_file():
                raise FileNotFoundError(f"Audio file not found: {full_path_obj}")

            logger.info(f"[{request_id}] Decoding audio from: {full_path_obj}")
            audio_input = decode_audio(str(full_path_obj), sampling_rate=16000)

            async with self.lock:
                if not self.model:
                    raise RuntimeError("Cannot transcribe, no model is loaded.")
                transcribe_args = self._prepare_transcription_args(params)
                segments_iter, info_obj = self.model.transcribe(audio=audio_input, **transcribe_args)

            info_data = info_obj.__dict__
            yield json.dumps({"type": "info", "data": info_data}) + "\n"

            for segment in segments_iter:
                seg_data = segment.__dict__
                if segment.words:
                    seg_data["words"] = [w.__dict__ for w in segment.words]
                yield json.dumps({"type": "segment", "data": seg_data}) + "\n"

            yield json.dumps({"type": "final", "message": "Transcription complete."}) + "\n"

        except Exception as e:
            logger.error(f"[{request_id}] Stream error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "message": traceback.format_exc()}) + "\n"
        finally:
            if full_path_obj:
                self.cleanup_audio_source(full_path_obj)

whisper_manager = WhisperManager()
