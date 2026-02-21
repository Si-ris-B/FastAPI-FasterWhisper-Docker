# FastAPI Faster-Whisper STT Service 🚀

A production-grade, containerized Speech-to-Text service powered by [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) and FastAPI. Supports all Whisper model variants including **whisper-turbo** (`large-v3-turbo`), batched high-throughput inference, real-time WebSocket transcription from live microphone input, and structured live log streaming.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🚀 **Whisper Turbo** | `large-v3-turbo` — ~8× faster than `large-v3` at comparable accuracy |
| ⚡ **Batched Pipeline** | `BatchedInferencePipeline` for maximum GPU throughput on long files |
| 🎤 **Live Transcription** | Real-time microphone → WebSocket → rolling transcription |
| 🌊 **Streaming Responses** | NDJSON segment-by-segment streaming as inference runs |
| 📡 **Live Log Stream** | WebSocket `/ws/logs` for real-time structured JSON log monitoring |
| 🔌 **On-Demand Loading** | Load/unload any model at runtime without restarting the service |
| 🌐 **Language Detection** | Fast language identification endpoint (no full transcription needed) |
| 🔐 **Path Safety** | Strict path traversal protection on all file references |
| 🐳 **Docker + GPU** | Multi-stage build, CUDA 12.4 + cuDNN 9, dynamic UID mapping |
| ⚙️ **Fully Configurable** | All settings via `.env` — no code changes needed |

---

## 📋 Table of Contents

- [Supported Models](#-supported-models)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Usage Examples](#-usage-examples)
  - [Load a Model](#1-load-a-model)
  - [Standard Streaming Transcription](#2-standard-streaming-transcription)
  - [Batched Transcription](#3-batched-transcription)
  - [Language Detection](#4-language-detection)
  - [Real-Time Log Streaming](#5-real-time-log-streaming-websocket)
  - [Live Microphone Transcription](#6-live-microphone-transcription-websocket)
- [NDJSON Response Format](#-ndjson-response-format)
- [Performance Guide](#-performance-guide)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Supported Models

### Recommended — Turbo Family
| Model | Params | Speed vs large-v3 | VRAM |
|---|---|---|---|
| `turbo` / `large-v3-turbo` | 809M | **~8× faster** | ~6 GB |

### Standard Models
| Model | Params | Notes |
|---|---|---|
| `tiny` / `tiny.en` | 39M | Fastest, lowest accuracy |
| `base` / `base.en` | 74M | Good for English-only tasks |
| `small` / `small.en` | 244M | Good balance on CPU |
| `medium` / `medium.en` | 769M | Good multilingual |
| `large-v1/v2/v3` | 1.55B | Highest accuracy |

### Distil Models (English-optimized)
| Model | Speed |
|---|---|
| `distil-large-v2` | ~6× faster than large-v2 |
| `distil-large-v3` | ~6× faster than large-v3 |
| `distil-medium.en` | ~5× faster than medium |
| `distil-small.en` | ~5× faster than small |

> You can also pass a **HuggingFace repo ID** or an **absolute path to a local CTranslate2 model** as `model_size_or_path`.

---

## 🏗️ Architecture

```
[ Your Machine / Client ]
        │
        │  HTTP REST (transcribe, load_model, detect_language)
        │  WebSocket (/ws/logs, /ws/live-transcribe)
        ▼
┌──────────────────────────────────────┐
│         FastAPI / Uvicorn            │
│  (src/stt_service/main.py)           │
│                                      │
│  ┌─────────────────────────────┐     │
│  │       WhisperManager        │     │
│  │  - Standard transcription   │     │
│  │  - Batched pipeline         │     │
│  │  - Language detection       │     │
│  │  - Live session handler     │     │
│  └──────────┬──────────────────┘     │
│             │                        │
│  ┌──────────▼──────────────────┐     │
│  │  faster-whisper + CTranslate│     │
│  │  WhisperModel / Batched     │     │
│  └──────────────────────────── ┘     │
└──────────────────────────────────────┘
        │                  │
        ▼                  ▼
  /stt_app_data/     /stt_app_data/
  audio_inbox/       model_cache/
  (Volume mount)     (Volume mount)
```

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-started/) + [Docker Compose](https://docs.docker.com/compose/install/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for GPU)
- GPU with CUDA 12.x support and ≥ 6 GB VRAM (for `turbo`)

### 1. Clone the Repository

```bash
git clone https://github.com/Si-ris-B/FastAPI-FasterWhisper-Docker.git
cd FastAPI-FasterWhisper-Docker
```

### 2. Configure the Environment

```bash
cp .env.example .env
```

Edit `.env` and set your host paths:

```ini
STT_AUDIO_INBOX_PATH=/home/youruser/stt_audio
STT_MODEL_CACHE_PATH=/home/youruser/stt_models
STT_SERVICE_PORT_HOST=8088
LOG_LEVEL=INFO
CLEANUP_AUDIO=True
UID=1000   # run: id -u
GID=1000   # run: id -g
```

Create the directories:

```bash
mkdir -p /home/youruser/stt_audio
mkdir -p /home/youruser/stt_models
```

### 3. Build and Start

```bash
docker compose up -d --build
```

Check that it started correctly:

```bash
docker compose logs -f
curl http://localhost:8088/status
```

The API is now available at **`http://localhost:8088`**.  
Interactive docs: **`http://localhost:8088/docs`**

---

## ⚙️ Configuration

All configuration is done via environment variables (set in `.env`).

| Variable | Default | Description |
|---|---|---|
| `STT_AUDIO_INBOX_PATH` | *(required)* | Host directory for audio files |
| `STT_MODEL_CACHE_PATH` | *(required)* | Host directory for downloaded models |
| `STT_SERVICE_PORT_HOST` | `8088` | Port exposed on host |
| `STT_SERVICE_PORT_CONTAINER` | `8001` | Port inside container |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `CLEANUP_AUDIO` | `True` | Delete audio file after transcription |
| `UID` / `GID` | `1000` | Match your host user for volume permissions |

---

## 📖 API Reference

All endpoints are also documented at `/docs` (Swagger UI) and `/redoc`.

### Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Service state, loaded model info, uptime |
| `GET` | `/models` | List all supported model identifiers |
| `POST` | `/load_model` | Load a model into memory |
| `POST` | `/unload_model` | Free GPU/CPU memory |

### Transcription

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/transcribe` | Stream-transcribe (standard mode) |
| `POST` | `/transcribe/batched` | Stream-transcribe (batched pipeline) |
| `POST` | `/detect_language` | Language identification only |

### WebSockets

| Endpoint | Description |
|---|---|
| `WS /ws/logs` | Real-time structured JSON log stream |
| `WS /ws/live-transcribe` | Live audio → rolling transcription |

---

## 💻 Usage Examples

Install the client dependencies:

```bash
pip install requests websockets sounddevice numpy
```

### 1. Load a Model

```python
import requests

BASE_URL = "http://localhost:8088"

# Load turbo (recommended — best speed/accuracy tradeoff)
r = requests.post(f"{BASE_URL}/load_model", json={
    "model_size_or_path": "turbo",   # or "large-v3-turbo", "large-v3", "base.en", etc.
    "device": "cuda",
    "compute_type": "float16",       # GPU: float16 | int8_float16 | int8
    "cpu_threads": 0,
    "num_workers": 1,
})
print(r.json())
# {"status": "success", "model_type": "turbo", ...}
```

**CPU-only setup:**
```python
r = requests.post(f"{BASE_URL}/load_model", json={
    "model_size_or_path": "small",
    "device": "cpu",
    "compute_type": "int8",
    "cpu_threads": 8,
})
```

### 2. Standard Streaming Transcription

Place your audio file in `STT_AUDIO_INBOX_PATH`, then reference it by filename:

```python
import json, requests

payload = {
    "file_path": "interview.mp3",          # relative to audio inbox
    "params": {
        "language": "en",                  # None for auto-detect
        "task": "transcribe",              # or "translate" to English
        "word_timestamps": True,           # enable word-level timing
        "vad_filter": True,                # skip silence
        "beam_size": 5,
        "vad_parameters": {
            "threshold": 0.5,
            "min_silence_duration_ms": 500,
        }
    }
}

with requests.post(f"{BASE_URL}/transcribe", json=payload, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        event = json.loads(line)

        if event["type"] == "info":
            d = event["data"]
            print(f"Language: {d['language']} ({d['language_probability']:.1%})")
            print(f"Duration: {d['duration']:.1f}s")

        elif event["type"] == "segment":
            seg = event["data"]
            print(f"[{seg['start']:.2f}→{seg['end']:.2f}] {seg['text']}")
            if seg.get("words"):
                for w in seg["words"]:
                    print(f"  '{w['word']}' {w['start']:.2f}s p={w['probability']:.2%}")

        elif event["type"] == "final":
            print(f"\nDone: {event['segment_count']} segments in {event['elapsed_seconds']:.2f}s "
                  f"(RTF={event['real_time_factor']:.3f})")

        elif event["type"] == "error":
            print(f"Error: {event['message']}")
```

### 3. Batched Transcription

Best for large files or high-volume queues — processes chunks in parallel for maximum GPU utilization:

```python
payload = {
    "file_path": "long_lecture.wav",
    "params": {
        "language": None,          # auto-detect
        "task": "transcribe",
        "batch_size": 16,          # increase for more VRAM / higher throughput
        "chunk_length": 30,        # seconds per chunk (None = model default)
        "vad_filter": True,        # strongly recommended for batch mode
        "word_timestamps": False,  # not available in all batched configs
        "beam_size": 1,            # lower for speed in batch mode
    }
}

with requests.post(f"{BASE_URL}/transcribe/batched", json=payload, stream=True) as r:
    r.raise_for_status()
    for line in r.iter_lines():
        if line:
            event = json.loads(line)
            if event["type"] == "segment":
                seg = event["data"]
                print(f"[{seg['start']:.2f}→{seg['end']:.2f}] {seg['text']}")
            elif event["type"] == "final":
                print(f"Done in {event['elapsed_seconds']:.2f}s (RTF={event['real_time_factor']:.3f})")
```

### 4. Language Detection

Identify the language without running a full transcription:

```python
r = requests.post(f"{BASE_URL}/detect_language", json={
    "file_path": "unknown_language.mp3",
    "language_detection_segments": 3,
    "language_detection_threshold": 0.5,
})
result = r.json()
print(f"Language: {result['detected_language']} ({result['language_probability']:.1%})")
print("Top candidates:", list(result['all_language_probs'].items())[:5])
```

### 5. Real-Time Log Streaming (WebSocket)

Connect and receive all application logs in structured JSON format:

```python
import asyncio, json, websockets

async def stream_logs():
    async with websockets.connect("ws://localhost:8088/ws/logs") as ws:
        async for msg in ws:
            log = json.loads(msg)
            if log["type"] == "log":
                print(f"[{log['timestamp'][11:23]}] [{log['level']:<8}] "
                      f"[{log['logger']}] {log['message']}")

asyncio.run(stream_logs())
```

Each log message structure:
```json
{
  "type": "log",
  "timestamp": "2025-01-01T12:00:00.123Z",
  "level": "INFO",
  "logger": "stt_service.whisper_manager",
  "message": "✅ Model 'turbo' loaded in 2.31s",
  "module": "whisper_manager",
  "lineno": 89
}
```

You can also **ping** the connection to keep it alive:
```python
await ws.send(json.dumps({"type": "ping"}))
# → {"type": "pong"}
```

### 6. Live Microphone Transcription (WebSocket)

Stream raw audio from your microphone and receive rolling transcription:

```python
import asyncio, json, numpy as np, sounddevice as sd, websockets

SAMPLE_RATE = 16000
CHUNK_MS = 500

async def live_transcribe():
    audio_queue = asyncio.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put_nowait(indata[:, 0].astype(np.float32).tobytes())

    async with websockets.connect("ws://localhost:8088/ws/live-transcribe") as ws:
        # 1. Send configuration
        await ws.send(json.dumps({
            "type": "config",
            "language": "en",
            "beam_size": 1,
            "vad_filter": True,
            "min_chunk_duration_s": 1.0,
            "max_buffer_duration_s": 10.0,
        }))

        # Wait for session acknowledgement
        ack = json.loads(await ws.recv())
        print(f"Session: {ack['message']}")

        async def send_audio():
            # Stream for 30 seconds then stop
            for _ in range(60):   # 60 × 500ms = 30s
                chunk = await asyncio.wait_for(audio_queue.get(), timeout=2.0)
                await ws.send(chunk)
                await asyncio.sleep(0)
            await ws.send(json.dumps({"type": "stop"}))

        async def receive_results():
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "partial":
                    print(f"\r🗣  {event['text']}", end="", flush=True)
                elif event["type"] == "final":
                    print(f"\n✅ {event['text']}")
                    break
                elif event["type"] == "error":
                    print(f"\n❌ {event['message']}")
                    break

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="float32",
                            blocksize=int(SAMPLE_RATE * CHUNK_MS / 1000),
                            callback=callback):
            print("🎤 Speaking... (30s)")
            await asyncio.gather(send_audio(), receive_results())

asyncio.run(live_transcribe())
```

**Live Transcription Protocol:**

| Direction | Frame | Description |
|---|---|---|
| Client → Server | Binary | Raw 16 kHz mono float32 PCM audio |
| Client → Server | `{"type": "config", ...}` | Session parameters (send before audio) |
| Client → Server | `{"type": "stop"}` | Flush buffer and end session |
| Client → Server | `{"type": "ping"}` | Keepalive |
| Server → Client | `{"type": "status", ...}` | Session start confirmation |
| Server → Client | `{"type": "buffering", ...}` | Buffer fill progress |
| Server → Client | `{"type": "partial", "text": "...", "segments": [...]}` | Rolling result |
| Server → Client | `{"type": "final", "text": "...", "segments": [...]}` | Flush result |
| Server → Client | `{"type": "error", "message": "..."}` | Error |

---

## 📦 NDJSON Response Format

All transcription endpoints stream **Newline Delimited JSON** (one JSON object per line).

### `info` event
```json
{
  "type": "info",
  "request_id": "txn-1234567890",
  "data": {
    "language": "en",
    "language_probability": 0.9987,
    "duration": 142.5,
    "duration_after_vad": 98.3,
    "all_language_probs": {"en": 0.9987, "de": 0.0008, ...}
  }
}
```

### `segment` event
```json
{
  "type": "segment",
  "data": {
    "id": 1,
    "seek": 0,
    "start": 0.0,
    "end": 4.82,
    "text": " Hello, this is a test transcription.",
    "tokens": [50364, 2425, 11, 341, 307, 257, 1500, 8297, 13],
    "avg_logprob": -0.2341,
    "compression_ratio": 1.47,
    "no_speech_prob": 0.0023,
    "temperature": 0.0,
    "words": [
      {"word": " Hello", "start": 0.0, "end": 0.42, "probability": 0.998},
      ...
    ]
  }
}
```

### `final` event
```json
{
  "type": "final",
  "request_id": "txn-1234567890",
  "segment_count": 47,
  "elapsed_seconds": 8.34,
  "real_time_factor": 0.058,
  "message": "Transcription complete."
}
```

### `error` event
```json
{
  "type": "error",
  "request_id": "txn-1234567890",
  "message": "Audio file not found in inbox: interview.mp3",
  "traceback": "Traceback (most recent call last):..."
}
```

---

## 🏎️ Performance Guide

### Model Selection

| Use Case | Recommended Model | Compute Type |
|---|---|---|
| General purpose (GPU) | `turbo` | `float16` |
| Maximum accuracy | `large-v3` | `float16` |
| CPU deployment | `small` or `distil-small.en` | `int8` |
| Realtime / live | `turbo` or `base` | `float16` |
| High-volume batch | `turbo` + batched endpoint | `float16` |

### Compute Type Selection

| Compute Type | Hardware | Notes |
|---|---|---|
| `float16` | GPU (CUDA) | Best accuracy on GPU |
| `bfloat16` | Ampere+ GPU | More numerically stable |
| `int8_float16` | GPU | Faster with minimal accuracy loss |
| `int8` | CPU or GPU | Maximum speed, lowest memory |
| `float32` | CPU | Most accurate on CPU |

### Batched vs Standard

Use the **batched endpoint** (`/transcribe/batched`) when:
- Processing files > 5 minutes
- Running a high-volume transcription queue
- You can trade slightly higher latency-to-first-token for higher total throughput

Use **standard streaming** (`/transcribe`) when:
- You need segments to arrive as soon as they're ready
- Building a real-time UI that displays words as they appear
- Processing short clips (< 2 minutes)

### VAD Filter

Always enable `vad_filter: true` for:
- Audio with significant silence or background noise
- Meeting recordings
- Anything fed from a microphone
- Batched pipeline (strongly recommended)

### Batch Size Tuning (batched endpoint)

| VRAM Available | Recommended `batch_size` |
|---|---|
| 6 GB | 4–8 |
| 12 GB | 8–16 |
| 24 GB | 16–32 |

---

## 🛠️ Development

### Running Locally (without Docker)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set env vars
export SHARED_AUDIO_PATH=./dev_audio
export MODEL_CACHE_PATH=./dev_models
mkdir -p dev_audio dev_models

# Start the server
uvicorn src.stt_service.main:app --host 0.0.0.0 --port 8001 --reload
```

### Running the Demo Client

```bash
# Copy a test audio file to your inbox
cp /path/to/audio.wav /home/youruser/stt_audio/

# Run the demo
python examples/client_demo.py audio.wav
```

---

## 🔧 Troubleshooting

### GPU not detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Permission denied on volume mounts

Set `UID` and `GID` in your `.env` to match your host user:

```bash
echo "UID=$(id -u)"
echo "GID=$(id -g)"
```

### Model download fails inside container

The container needs internet access to download models. If your environment is airgapped:

1. Download the model manually on a machine with internet access
2. Mount the model directory to `STT_MODEL_CACHE_PATH`

The model will be found in the cache and not re-downloaded.

### CUDA out of memory

- Switch to a smaller model (`small`, `base`)
- Use `compute_type: "int8"` or `"int8_float16"`
- Reduce `batch_size` in batched endpoint
- Call `/unload_model` between jobs if running multiple

### `No module named 'sounddevice'` (live transcription)

The `sounddevice` package is optional and only needed on the **client** machine (not inside Docker):

```bash
pip install sounddevice
```

On headless servers without audio hardware, use file-based transcription instead.

---

## 📄 License

MIT License. See `LICENSE` for details.