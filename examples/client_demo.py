"""
FastAPI Faster-Whisper STT Service — Python Client Examples
===========================================================
Demonstrates all service features:
  - Standard streaming transcription
  - Batched pipeline transcription
  - Language detection
  - Live microphone transcription (WebSocket)
  - Real-time log streaming (WebSocket)

Install: pip install requests websockets sounddevice numpy
"""

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Optional

import requests

BASE_URL = "http://localhost:8088"
WS_URL  = "ws://localhost:8088"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_segment(seg: dict):
    start = seg.get("start", 0)
    end   = seg.get("end", 0)
    text  = seg.get("text", "")
    print(f"  [{start:.2f}s → {end:.2f}s]  {text}")


def print_status(label: str):
    r = requests.get(f"{BASE_URL}/status")
    data = r.json()
    print(f"\n[{label}] Service status: {data['service_status']}")
    if data.get("loaded_model_config"):
        print(f"  Model: {data['loaded_model_config']['model_size_or_path']}")
        print(f"  Type:  {data.get('model_type')}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load a model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str = "turbo", device: str = "cuda", compute_type: str = "float16"):
    """Load a model. Recommended: 'turbo' with float16 on GPU."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print('='*60)

    config = {
        "model_size_or_path": model_name,
        "device": device,
        "compute_type": compute_type,
        "cpu_threads": 0,
        "num_workers": 1,
    }
    r = requests.post(f"{BASE_URL}/load_model", json=config, timeout=300)
    r.raise_for_status()
    data = r.json()
    print(f"✅ {data['message']} (type={data['model_type']})")
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Standard streaming transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_stream(audio_filename: str, language: Optional[str] = None,
                      word_timestamps: bool = False, vad_filter: bool = True):
    """Stream-transcribe a file placed in the audio inbox."""
    print(f"\n{'='*60}")
    print(f"Standard streaming transcription: {audio_filename}")
    print('='*60)

    payload = {
        "file_path": audio_filename,
        "params": {
            "language": language,
            "task": "transcribe",
            "word_timestamps": word_timestamps,
            "vad_filter": vad_filter,
            "beam_size": 5,
        }
    }

    full_text = []
    with requests.post(f"{BASE_URL}/transcribe", json=payload, stream=True) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            event = json.loads(raw_line)
            etype = event.get("type")

            if etype == "info":
                d = event["data"]
                print(f"\n  🌐 Language: {d['language']} ({d['language_probability']:.1%})")
                print(f"  ⏱  Duration: {d['duration']:.1f}s (after VAD: {d['duration_after_vad']:.1f}s)")

            elif etype == "segment":
                seg = event["data"]
                full_text.append(seg["text"])
                print_segment(seg)
                if word_timestamps and seg.get("words"):
                    for w in seg["words"]:
                        print(f"    '{w['word']}' [{w['start']:.2f}→{w['end']:.2f}] {w['probability']:.2%}")

            elif etype == "final":
                print(f"\n  ✅ Done — {event['segment_count']} segments in {event['elapsed_seconds']:.2f}s "
                      f"(RTF={event['real_time_factor']:.3f})")

            elif etype == "error":
                print(f"\n  ❌ Error: {event['message']}")

    return " ".join(full_text)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Batched pipeline transcription (high throughput)
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_batched(audio_filename: str, batch_size: int = 8,
                       language: Optional[str] = None):
    """Use batched inference pipeline — best for large files."""
    print(f"\n{'='*60}")
    print(f"Batched transcription (batch_size={batch_size}): {audio_filename}")
    print('='*60)

    payload = {
        "file_path": audio_filename,
        "params": {
            "language": language,
            "task": "transcribe",
            "batch_size": batch_size,
            "vad_filter": True,
            "beam_size": 1,
        }
    }

    full_text = []
    with requests.post(f"{BASE_URL}/transcribe/batched", json=payload, stream=True) as r:
        r.raise_for_status()
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            event = json.loads(raw_line)
            etype = event.get("type")

            if etype == "info":
                d = event["data"]
                print(f"\n  🌐 Language: {d['language']} | Duration: {d['duration']:.1f}s")
                print(f"  ⚡ Batched: size={d['batch_size']} chunk={d['chunk_length']}s")

            elif etype == "segment":
                seg = event["data"]
                full_text.append(seg["text"])
                print_segment(seg)

            elif etype == "final":
                print(f"\n  ✅ Batched done — {event['segment_count']} segments "
                      f"in {event['elapsed_seconds']:.2f}s (RTF={event['real_time_factor']:.3f})")

            elif etype == "error":
                print(f"\n  ❌ Error: {event['message']}")

    return " ".join(full_text)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Language detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_language(audio_filename: str):
    """Fast language identification without full transcription."""
    print(f"\n{'='*60}")
    print(f"Language detection: {audio_filename}")
    print('='*60)

    payload = {
        "file_path": audio_filename,
        "language_detection_segments": 3,
        "language_detection_threshold": 0.5,
    }
    r = requests.post(f"{BASE_URL}/detect_language", json=payload)
    r.raise_for_status()
    result = r.json()
    print(f"  🌐 Detected: {result['detected_language']} ({result['language_probability']:.1%})")
    print("  Top candidates:")
    for lang, prob in list(result["all_language_probs"].items())[:5]:
        bar = "█" * int(prob * 30)
        print(f"    {lang:5s}  {prob:.1%}  {bar}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. Real-time log streaming
# ─────────────────────────────────────────────────────────────────────────────

async def stream_logs(duration_seconds: float = 10.0):
    """Connect to /ws/logs and print live structured logs."""
    import websockets

    print(f"\n{'='*60}")
    print(f"Streaming logs for {duration_seconds}s...")
    print('='*60)

    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    async with websockets.connect(f"{WS_URL}/ws/logs") as ws:
        deadline = asyncio.get_event_loop().time() + duration_seconds
        while asyncio.get_event_loop().time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                log = json.loads(raw)
                if log.get("type") == "log":
                    color = LEVEL_COLORS.get(log["level"], "")
                    ts = log["timestamp"][11:23]  # HH:MM:SS.mmm
                    print(
                        f"{color}[{ts}] [{log['level']:<8}]{RESET} "
                        f"[{log['logger']}] {log['message']}"
                    )
                    if "exception" in log:
                        print(f"  {log['exception']}")
            except asyncio.TimeoutError:
                continue


# ─────────────────────────────────────────────────────────────────────────────
# 6. Live microphone transcription
# ─────────────────────────────────────────────────────────────────────────────

async def live_microphone_transcription(
    language: Optional[str] = None,
    duration_seconds: float = 30.0,
    chunk_ms: int = 500,
):
    """
    Capture microphone audio and stream to /ws/live-transcribe.
    Requires: pip install sounddevice numpy websockets
    """
    try:
        import sounddevice as sd
        import numpy as np
        import websockets
    except ImportError:
        print("Install sounddevice: pip install sounddevice numpy websockets")
        return

    SAMPLE_RATE = 16000
    CHUNK_SAMPLES = int(SAMPLE_RATE * chunk_ms / 1000)

    print(f"\n{'='*60}")
    print(f"🎤 Live microphone transcription ({duration_seconds}s)")
    print(f"   Language: {language or 'auto'} | Chunk: {chunk_ms}ms")
    print('='*60)
    print("Speak now... (Ctrl+C to stop)\n")

    audio_queue: asyncio.Queue = asyncio.Queue()

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio chunk."""
        if status:
            print(f"  [Audio] {status}", file=sys.stderr)
        mono = indata[:, 0].astype(np.float32)
        audio_queue.put_nowait(mono.tobytes())

    async with websockets.connect(f"{WS_URL}/ws/live-transcribe") as ws:
        # Send config
        config = {
            "type": "config",
            "language": language,
            "task": "transcribe",
            "beam_size": 1,
            "vad_filter": True,
            "min_chunk_duration_s": 1.0,
            "max_buffer_duration_s": 15.0,
        }
        await ws.send(json.dumps(config))

        # Start microphone
        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
            callback=audio_callback,
        )

        start_time = asyncio.get_event_loop().time()

        async def send_audio():
            while asyncio.get_event_loop().time() - start_time < duration_seconds:
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    await ws.send(chunk)
                except asyncio.TimeoutError:
                    continue
            await ws.send(json.dumps({"type": "stop"}))

        async def receive_results():
            async for raw_msg in ws:
                event = json.loads(raw_msg)
                etype = event.get("type")
                if etype == "status":
                    print(f"  ℹ {event['message']}")
                elif etype == "partial":
                    print(f"\r  🗣 {event['text']}", end="", flush=True)
                elif etype == "final":
                    print(f"\n  ✅ Final: {event['text']}")
                    break
                elif etype == "buffering":
                    print(f"\r  ⏳ Buffering {event['buffer_duration']:.1f}s/{event['min_needed']}s...",
                          end="", flush=True)
                elif etype == "error":
                    print(f"\n  ❌ {event['message']}")
                    break

        with stream:
            await asyncio.gather(send_audio(), receive_results())


# ─────────────────────────────────────────────────────────────────────────────
# Demo runner
# ─────────────────────────────────────────────────────────────────────────────

def run_full_demo(audio_file: str = "sample.wav"):
    """Run a complete demo of all features."""
    print_status("before")

    # 1. Load turbo model
    load_model("turbo", device="cuda", compute_type="float16")

    print_status("after load")

    # 2. Detect language
    detect_language(audio_file)

    # 3. Standard streaming
    transcribe_stream(audio_file, word_timestamps=True, vad_filter=True)

    # 4. Batched pipeline
    transcribe_batched(audio_file, batch_size=8)

    # 5. Stream logs for 5s while doing another transcription
    print("\nStreaming logs + transcription in parallel...")
    log_task = threading.Thread(
        target=lambda: asyncio.run(stream_logs(5.0)), daemon=True
    )
    log_task.start()
    transcribe_stream(audio_file)
    log_task.join()

    # 6. Unload
    r = requests.post(f"{BASE_URL}/unload_model")
    print(f"\nUnloaded: {r.json()['message']}")

    print_status("after unload")


if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
    run_full_demo(audio)
