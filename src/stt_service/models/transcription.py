from __future__ import annotations

from typing import Optional, List, Union, Dict, Iterable, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

class ModelConfigParams(BaseModel):
    model_size_or_path: str = Field(
        ...,
        description=(
            "Model identifier. Standard: 'tiny', 'base', 'small', 'medium', 'large-v1/v2/v3'. "
            "Turbo (recommended): 'turbo' or 'large-v3-turbo'. "
            "Distil: 'distil-large-v2', 'distil-large-v3', 'distil-medium.en', 'distil-small.en'. "
            "Also accepts a HuggingFace repo ID or local path to a CTranslate2 model."
        ),
        examples=["turbo", "large-v3-turbo", "large-v3", "base.en"],
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        "auto",
        description="Inference device. 'auto' selects GPU if available, else CPU.",
    )
    compute_type: str = Field(
        "default",
        description=(
            "Quantization. GPU options: 'float16', 'bfloat16', 'int8_float16', 'int8'. "
            "CPU options: 'int8', 'int8_float32', 'float32'. "
            "'default' picks the best for the selected device."
        ),
        examples=["float16", "int8_float16", "int8", "default"],
    )
    device_index: Union[int, List[int]] = Field(
        0,
        description="GPU device index or list of indices for multi-GPU inference.",
    )
    cpu_threads: int = Field(
        0,
        description="CPU threads (0 = auto-detect optimal).",
    )
    num_workers: int = Field(
        1,
        description="Number of parallel workers for the model pipeline.",
    )


# ---------------------------------------------------------------------------
# VAD Parameters
# ---------------------------------------------------------------------------

class VADParams(BaseModel):
    threshold: float = Field(0.5, description="Speech probability threshold [0.0–1.0].")
    min_speech_duration_ms: int = Field(250, description="Min speech chunk duration (ms).")
    max_speech_duration_s: float = Field(
        float("inf"), description="Max speech chunk duration (s). Inf = no limit."
    )
    min_silence_duration_ms: int = Field(
        2000, description="Min silence gap to split segments (ms)."
    )
    window_size_samples: int = Field(
        1024, description="VAD window size in samples (512 / 1024 / 1536)."
    )
    speech_pad_ms: int = Field(400, description="Padding added around speech regions (ms).")


class BatchedVADParams(VADParams):
    """VAD defaults tuned for batched inference pipeline."""
    min_silence_duration_ms: int = Field(160, description="Shorter silence for batch chunking.")
    max_speech_duration_s: float = Field(30.0, description="Hard cap per batch chunk.")


# ---------------------------------------------------------------------------
# Standard (Streaming) Transcription Parameters
# ---------------------------------------------------------------------------

class StandardTranscriptionParams(BaseModel):
    language: Optional[str] = Field(
        None,
        description="BCP-47 language code (e.g. 'en', 'de', 'ja'). None = auto-detect.",
    )
    task: Literal["transcribe", "translate"] = Field(
        "transcribe",
        description="'transcribe' keeps original language. 'translate' outputs English.",
    )
    beam_size: int = Field(5, description="Beam search width. Higher = more accurate but slower.")
    best_of: int = Field(5, description="Number of candidates when sampling.")
    patience: float = Field(1.0, description="Beam search patience multiplier.")
    length_penalty: float = Field(1.0, description="Penalize short/long outputs.")
    repetition_penalty: float = Field(1.0, description="Penalize repeated tokens (>1 = less repetition).")
    no_repeat_ngram_size: int = Field(0, description="Block n-gram repetition (0 = disabled).")
    temperature: Union[float, List[float]] = Field(
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        description="Sampling temperature(s). List = fallback schedule.",
    )
    compression_ratio_threshold: Optional[float] = Field(
        2.4, description="Discard if output compression ratio exceeds this."
    )
    log_prob_threshold: Optional[float] = Field(
        -1.0, description="Retry with higher temperature if avg log-prob below this."
    )
    no_speech_threshold: Optional[float] = Field(
        0.6, description="Mark segment as no-speech if probability exceeds this."
    )
    condition_on_previous_text: bool = Field(
        True, description="Feed previous output as prompt for next segment."
    )
    prompt_reset_on_temperature: float = Field(
        0.5, description="Reset prompt conditioning when temperature exceeds this."
    )
    initial_prompt: Optional[Union[str, Iterable[int]]] = Field(
        None, description="Initial prompt string or token IDs to guide transcription style."
    )
    prefix: Optional[str] = Field(None, description="Forced prefix for first segment.")
    suppress_blank: bool = Field(True, description="Suppress blank outputs at start.")
    suppress_tokens: Optional[List[int]] = Field(
        default=[-1],
        description="Token IDs to suppress. -1 = all non-speech tokens.",
    )
    without_timestamps: bool = Field(False, description="Strip timestamps from output.")
    max_initial_timestamp: float = Field(1.0, description="Max timestamp for first token.")
    word_timestamps: bool = Field(False, description="Enable word-level timestamps.")
    prepend_punctuations: str = Field(
        """\"'"¿([{-""", description="Punctuation attached to preceding word."
    )
    append_punctuations: str = Field(
        """\"'.。,，!！?？:：")]}、""", description="Punctuation attached to following word."
    )
    vad_filter: bool = Field(False, description="Enable Silero VAD pre-filter.")
    vad_parameters: Optional[VADParams] = Field(None, description="VAD tuning parameters.")
    max_new_tokens: Optional[int] = Field(None, description="Max tokens per segment.")
    clip_timestamps: str = Field("0", description="Clip audio at given timestamps.")
    hallucination_silence_threshold: Optional[float] = Field(
        None, description="Skip hallucinations in silent regions (seconds)."
    )
    hotwords: Optional[str] = Field(
        None, description="Comma-separated hotwords to boost recognition probability."
    )
    language_detection_threshold: Optional[float] = Field(
        0.5, description="Minimum language detection confidence."
    )
    language_detection_segments: int = Field(
        1, description="Number of segments to use for language detection."
    )


# ---------------------------------------------------------------------------
# Batched Pipeline Transcription Parameters
# ---------------------------------------------------------------------------

class BatchedTranscriptionParams(BaseModel):
    """
    Parameters for the BatchedInferencePipeline (faster-whisper >= 1.0).
    Uses chunked batched inference for maximum GPU throughput.
    Ideal for large files / high-volume workloads.
    """
    language: Optional[str] = Field(None, description="Language code. None = auto-detect.")
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = Field(
        1, description="Beam width. Usually 1 for batched pipeline (speed/accuracy tradeoff)."
    )
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    temperature: Union[float, List[float]] = Field(default=[0.0])
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = Field(default=[-1])
    without_timestamps: bool = Field(
        True, description="Timestamps are omitted by default in batch mode."
    )
    word_timestamps: bool = False
    prepend_punctuations: str = """\"'"¿([{-"""
    append_punctuations: str = """\"'.。,，!！?？:：")]}、"""
    vad_filter: bool = Field(True, description="VAD is strongly recommended for batch mode.")
    vad_parameters: Optional[BatchedVADParams] = None
    max_new_tokens: Optional[int] = None
    chunk_length: Optional[int] = Field(
        None, description="Audio chunk length in seconds. None = model default (30s)."
    )
    batch_size: int = Field(
        8, description="Number of chunks processed in parallel. Higher = more VRAM + speed."
    )
    hotwords: Optional[str] = None
    language_detection_threshold: Optional[float] = 0.5
    language_detection_segments: int = 1


# ---------------------------------------------------------------------------
# Live / Real-time Transcription Parameters (WebSocket)
# ---------------------------------------------------------------------------

class LiveTranscriptionParams(BaseModel):
    """
    Parameters for the /ws/live-transcribe WebSocket endpoint.
    Client streams raw 16kHz PCM audio chunks; server responds with rolling transcription.
    """
    language: Optional[str] = Field(None, description="Language code. None = auto-detect.")
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = Field(
        1, description="Lower beam size recommended for latency."
    )
    word_timestamps: bool = False
    vad_filter: bool = Field(
        True, description="Highly recommended to suppress silence artifacts."
    )
    vad_parameters: Optional[VADParams] = None
    without_timestamps: bool = True
    suppress_blank: bool = True
    hotwords: Optional[str] = None
    # Buffer settings
    min_chunk_duration_s: float = Field(
        1.0, description="Minimum seconds of audio buffered before transcription attempt."
    )
    max_buffer_duration_s: float = Field(
        30.0, description="Maximum buffer length before forced transcription flush."
    )


# ---------------------------------------------------------------------------
# Request Wrappers
# ---------------------------------------------------------------------------

class TranscribeRequest(BaseModel):
    file_path: str = Field(
        ..., description="Filename (or relative path) within the shared audio inbox."
    )
    params: StandardTranscriptionParams = Field(
        default_factory=StandardTranscriptionParams
    )


class BatchedTranscribeRequest(BaseModel):
    file_path: str = Field(
        ..., description="Filename (or relative path) within the shared audio inbox."
    )
    params: BatchedTranscriptionParams = Field(
        default_factory=BatchedTranscriptionParams
    )


class DetectLanguageRequest(BaseModel):
    file_path: str = Field(
        ..., description="Filename (or relative path) within the shared audio inbox."
    )
    language_detection_segments: int = Field(
        3, description="Number of audio segments to sample for language detection."
    )
    language_detection_threshold: float = Field(
        0.5, description="Minimum confidence threshold for language detection."
    )


# ---------------------------------------------------------------------------
# Response Schemas (for documentation)
# ---------------------------------------------------------------------------

class ServiceStatus(BaseModel):
    service_status: str
    loaded_model_config: Optional[Dict] = None
    model_type: Optional[str] = None  # "standard" | "turbo" | "distil"
    uptime_seconds: Optional[float] = None