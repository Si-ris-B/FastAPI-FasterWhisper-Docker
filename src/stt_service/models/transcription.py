from pydantic import BaseModel, Field
from typing import Optional, List, Union, Iterable, Dict

# This file contains the complete, unabridged Pydantic models based on the original code.

class ModelConfigParams(BaseModel):
    model_size_or_path: str = Field(..., description="Model name (e.g., 'base.en'), HuggingFace ID, or path to a converted model directory.")
    device: str = Field("auto", description="Device: 'cpu', 'cuda', 'auto'.")
    compute_type: str = Field("default", description="Compute type: e.g., 'int8', 'float16', 'default'.")
    device_index: Union[int, List[int]] = Field(0, description="GPU device index/indices.")
    cpu_threads: int = Field(0, description="Number of CPU threads (0 for auto).")
    num_workers: int = Field(1, description="Number of workers for WhisperModel.")

class VADParams(BaseModel):
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    max_speech_duration_s: Optional[float] = float('inf')
    min_silence_duration_ms: Optional[int] = 2000
    window_size_samples: Optional[int] = 1024
    speech_pad_ms: Optional[int] = 400

class BatchedVADParams(VADParams):
    min_silence_duration_ms: Optional[int] = 160
    max_speech_duration_s: Optional[float] = None

class StandardTranscriptionParams(BaseModel):
    language: Optional[str] = Field(None, description="Language code (e.g., 'en'). None for auto-detect.")
    task: str = Field("transcribe", description="'transcribe' or 'translate'")
    beam_size: int = 5
    best_of: int = 5
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    temperature: Union[float, List[float]] = Field(default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    compression_ratio_threshold: Optional[float] = 2.4
    log_prob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = Field(default=[-1], description="List of token IDs. -1 for non-speech.")
    without_timestamps: bool = False
    max_initial_timestamp: float = 1.0
    word_timestamps: bool = Field(False, description="Enable word timestamps.")
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    vad_filter: bool = Field(False, description="Enable VAD filter.")
    vad_parameters: Optional[VADParams] = None
    max_new_tokens: Optional[int] = None
    clip_timestamps: str = "0"
    hallucination_silence_threshold: Optional[float] = None
    hotwords: Optional[str] = None
    language_detection_threshold: Optional[float] = 0.5
    language_detection_segments: int = 1

class BatchedTranscriptionParams(BaseModel):
    language: Optional[str] = None
    task: str = "transcribe"
    beam_size: int = Field(1, description="Usually 1 for batched pipeline.")
    patience: float = 1.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    temperature: Union[float, List[float]] = Field(default=[0.0])
    initial_prompt: Optional[Union[str, Iterable[int]]] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[List[int]] = Field(default=[-1])
    without_timestamps: bool = True
    word_timestamps: bool = False
    prepend_punctuations: str = "\"'“¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、"
    vad_filter: bool = True
    vad_parameters: Optional[BatchedVADParams] = None
    max_new_tokens: Optional[int] = None
    chunk_length: Optional[int] = Field(None, description="Audio chunk length for pipeline.")
    clip_timestamps: Optional[Union[str, List[Dict[str, float]]]] = None
    batch_size: int = Field(8, description="Inference batch size for pipeline.")
    hotwords: Optional[str] = None
    language_detection_threshold: Optional[float] = 0.5
    language_detection_segments: int = 1

class TranscribeRequest(BaseModel):
    params: StandardTranscriptionParams
    file_path: str = Field(..., description="Filename relative to the service's shared audio path.")

class BatchedTranscribeRequest(BaseModel):
    params: BatchedTranscriptionParams
    file_path: str = Field(..., description="Filename relative to the service's shared audio path.")

class DetectLanguageRequest(BaseModel):
    vad_filter: bool = False
    vad_parameters: Optional[VADParams] = None
    language_detection_segments: int = 1
    language_detection_threshold: float = 0.5
    file_path: str = Field(..., description="Filename relative to the service's shared audio path.")
