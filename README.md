# FastAPI-FasterWhisper-Docker 🚀

A robust, containerized, and on-demand Speech-to-Text (STT) service powered by `faster-whisper` and `FastAPI`. This project provides a high-performance REST API that can dynamically load and unload different Whisper models, process audio files, and deliver transcriptions via a streaming response. It's designed for efficiency, scalability, and ease of use.

## ✨ Key Features

*   **🚀 On-Demand Model Loading**: Load any `faster-whisper` model (`tiny`, `base`, `large-v2`, etc.) via a simple API call. The service remains lightweight until a model is requested.
*   **🔌 Dynamic Unloading**: Free up GPU/CPU resources by unloading the model when it's no longer needed.
*   **⚡ High-Performance Inference**: Leverages `faster-whisper` and CTranslate2 for significant speed improvements and reduced memory usage.
*   **🌊 Streaming Transcriptions**: The `/transcribe` endpoint uses `StreamingResponse` to send back transcription segments as they are generated, perfect for long audio files and responsive UIs.
*   **📡 Real-time Logging via WebSockets**: Connect to a WebSocket endpoint (`/ws/logs`) to get a live stream of the service's logs.
*   **🐳 Fully Containerized**: Ships with a multi-stage `Dockerfile` and `docker-compose.yml` for easy, reproducible deployment on both CPU and NVIDIA GPU setups.
*   **🔐 Secure & Robust**: Implements path traversal protection and handles shared data via Docker volumes, ensuring the service only accesses designated directories.
*   **⚙️ Highly Configurable**: All key settings are managed via environment variables for easy deployment and customization.

## 🏗️ Architectural Overview

The service operates on a shared volume principle. Your client application places audio files into a host directory, which is mounted into the container. The API then references these files by their name within the container's designated path.

```
[ Your Machine (Host) ]                  |    [ Docker Container ]
                                         |
+----------------------+                 |    +-------------------------+
| Your Client/Script   | --(HTTP API)--> |    |  FastAPI / Uvicorn      |
| (e.g., Python `requests`) |            |    |   (src/stt_service)     |
+----------------------+                 |    +-------------------------+
                                         |               |
     |                                   |               | (Loads model into GPU/RAM)
     | (Places file in)                  |               |
     v                                   |               v
+----------------------------+  <-- (Volume Mount) -->  +-------------------------+
| /path/on/your/host/audio   |           |            | /stt_app_data/audio_inbox |
+----------------------------+           |            +-------------------------+
                                         |
+-----------------------------+ <-- (Volume Mount) -->  +-------------------------+
| /path/on/your/host/models  |           |            | /stt_app_data/model_cache |
+-----------------------------+          |            +-------------------------+
```

## 🏁 Getting Started

### Prerequisites

*   [Git](https://git-scm.com/)
*   [Docker](https://www.docker.com/get-started/)
*   [Docker Compose](https://docs.docker.com/compose/install/)
*   (For GPU support) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### 1. Clone the Repository

```bash
git clone https://github.com/Si-ris-B/FastAPI-FasterWhisper-Docker.git
cd FastAPI-FasterWhisper-Docker
```

### 2. Configure the Environment

The service is configured using environment variables. This is the **most important step** to adapt the service to your machine.

**A. Create a `.env` file from the example:**
```bash
cp .env.example .env
```

**B. Edit the `.env` file:**
Open the newly created `.env` file and set the paths to directories on **your host machine**.

```ini
# .env

# --- Required: Paths on Your Host Machine ---
# Provide the ABSOLUTE path to the directories you want to use.
# The service will read audio from the inbox and store models in the cache.
STT_AUDIO_INBOX_PATH=/path/on/your/computer/for_audio_files
STT_MODEL_CACHE_PATH=/path/on/your/computer/for_model_storage

# --- Optional: Network Configuration ---
# Change the port the service is exposed on (e.g., if 8088 is already in use).
STT_SERVICE_PORT_HOST=8088

# ... other settings ...
```
> **IMPORTANT**: Before running `docker-compose up`, you must create the directories you specified for `STT_AUDIO_INBOX_PATH` and `STT_MODEL_CACHE_PATH` on your host machine. For example:
> `mkdir -p /path/on/your/computer/for_audio_files`
> `mkdir -p /path/on/your/computer/for_model_storage`

### 3. Launch the STT Service

With Docker running, launch the service using Docker Compose.

```bash
docker-compose up -d --build
```
This will build the Docker image and start the service in the background. The first time you load a model, it will be downloaded to your `STT_MODEL_CACHE_PATH`.

To check the logs:
```bash
docker-compose logs -f
```

The API will be available at `http://localhost:8088` (or the custom port you set).

## 📖 API Usage Examples

Here is how you can interact with the service using Python's `requests` library.

### Step 1: Load a Model

Send a request to the `/load_model` endpoint. The service will download the model to your `STT_MODEL_CACHE_PATH` if it's not already there.

```python
import requests
import json

BASE_URL = "http://localhost:8088"

model_config = {
    "model_size_or_path": "large-v2",
    "device": "cuda",
    "compute_type": "float16",
    "cpu_threads": 4,
    "num_workers": 1
}

print("Attempting to load model...")
response = requests.post(f"{BASE_URL}/load_model", json=model_config)

if response.status_code == 200:
    print("Model loaded successfully:", response.json())
else:
    print(f"Error loading model: {response.status_code} - {response.text}")
```

### Step 2: Transcribe an Audio File

1.  **Place your audio file** (e.g., `my_test_audio.mp3`) into the host directory you defined as `STT_AUDIO_INBOX_PATH`.
2.  Call the `/transcribe` endpoint, referencing the file **by its name only**.

```python
# Assumes the model is already loaded from Step 1

# This is the filename inside the container's inbox
file_path_in_container = "my_test_audio.mp3"

transcription_params = {
    "language": "en",
    "task": "transcribe",
    "word_timestamps": True,
}

payload = {
    "file_path": file_path_in_container,
    "params": transcription_params
}

print(f"\nRequesting transcription for {file_path_in_container}...")
try:
    response = requests.post(f"{BASE_URL}/transcribe", json=payload, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if data.get("type") == "segment":
                segment = data["data"]
                print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

### Step 3: Unload the Model

To free up system resources (especially GPU memory), you can unload the model.

```python
print("\nUnloading the model...")
response = requests.post(f"{BASE_URL}/unload_model")
print(response.json())
```

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
