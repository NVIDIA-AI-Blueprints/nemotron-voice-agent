# Deploying Voice Agent on Jetson Thor

This guide covers deploying the Nemotron Voice Agent on Jetson Thor using Docker Compose.

---

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.0** using [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed)
- [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) installed and configured
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/)
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models
- Network connectivity

---

## Project Structure

The configuration files for this deployment are the following:

```
./
├── docker-compose.jetson.yml   # Jetson-specific deployment
└── config
    └── env.jetson.example      # Template for .env
```

| File | Purpose |
|------|---------|
| [docker-compose.jetson.yml](../docker-compose.jetson.yml) | Jetson-specific Docker Compose with vLLM |
| [env.jetson.example](../config/env.jetson.example) | Environment template for Jetson deployment |

> **Note:** This deployment uses vLLM for LLM inference instead of NVIDIA NIM. LLM NIM microservices for Jetson Thor are not yet available, so this guide uses vLLM as a flexible alternative to load Hugging Face models directly.

---

## Models Used by Default

All models are deployed on the local Jetson device. Default models used:

| Component | Default model / identifier |
|-----------|----------------------------|
| **ASR**   | `parakeet-1.1b-en-US-asr-streaming` |
| **TTS**   | `magpie_tts_ensemble-Magpie-Multilingual` |
| **LLM**   | `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` |

---

## Deployment Steps

1. Clone the repository and navigate to the root directory.

    ```bash
    git clone git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    git submodule update --init
    ```

2. Configure the environment. Copy the example environment file [env.jetson.example](../config/env.jetson.example) to the root directory:

    ```bash
    cp config/env.jetson.example .env
    ```

3. Set your API keys as environment variables:

    ```bash
    # Required
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    export HF_TOKEN=<your-huggingface-token>
    ```

    **Jetson-specific defaults** (differ from main deployment)
    - `ENABLE_SPECULATIVE_SPEECH=false` — Disabled for resource optimization.
    - `WORKERS=1` — Single worker to reduce memory usage.

4. Deploy Nemotron Speech ASR and TTS models.

    a. Ensure you meet the [prerequisites](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#prerequisites) before proceeding.

    b. Configure NGC CLI with your API key:

    ```bash
    ngc config set
    ```

    c. Download the Nemotron Speech ASR and TTS Quick Start scripts:

    ```bash
    ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0
    cd riva_quickstart_arm64_v2.24.0
    ```

    d. [Optional] Enable Silero VAD for improved ASR performance:

    > **Tip:** Enabling Silero VAD can help improve End-of-Utterance (EOU) detection and performance in noisy environments.

    1. Edit the `config.sh` in Quick Start directory `riva_quickstart_arm64_v2.24.0`:

        ```bash
        # Use Silero Diarizer as accessory model for ASR
        asr_accessory_model=("silero_diarizer")
        ```

    2. Update your `.env` file with the ASR model name:

        ```bash
        ASR_MODEL_NAME=parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer
        ```

    e. Deploy Nemotron Speech ASR and TTS models:

    ```bash
    bash riva_init.sh
    bash riva_start.sh
    ```

    > **Note:** Initialization may take 30-60 minutes on first run.

5. Start LLM Service and Voice Agent Application. Start services from the root directory:

    ```bash
    docker compose -f docker-compose.jetson.yml up -d
    ```

    > **Note:** Deployment may take 15-20 minutes on first run.

6. Access the application at `http://<jetson-ip>:8081` on your browser.

    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.

   > **Note:** To enable microphone access in Chrome, go to `chrome://flags/`, enable "Insecure origins treated as secure", add `http://<jetson-ip>:8081` to the list, and restart Chrome. If you need to access the application from remote locations or deploy on cloud platforms, configure a TURN server. Refer to [Optional: Deploy TURN Server for Remote Access](01-getting-started.md#optional-deploy-turn-server-for-remote-access).

---

## Switching LLM Models

The Jetson deployment uses vLLM to load HuggingFace models. Update these variables in your `.env` file:

| Variable | Description |
|----------|-------------|
| `NVIDIA_LLM_MODEL` | HuggingFace model identifier |
| `GPU_MEMORY_UTILIZATION` | GPU memory fraction (0.0-1.0). Adjust based on model size. |
| `SYSTEM_PROMPT_SELECTOR` | Prompt path from [config/prompt.yaml](../config/prompt.yaml) |

### Available Models

| Model | Size | `GPU_MEMORY_UTILIZATION` |
|-------|------|--------------------------|
| `RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16` | 8B (4-bit) | 0.15 |
| `nvidia/Nemotron-Mini-4B-Instruct` | 4B | 0.10 |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2-FP8` | 9B (FP8) | 0.20 |
| `Qwen/Qwen3-4B-Instruct-2507` | 4B | 0.10 |

### Example: Switch to Nemotron-Mini-4B

1. Edit your `.env` file:

    ```bash
    NVIDIA_LLM_MODEL=nvidia/Nemotron-Mini-4B-Instruct
    GPU_MEMORY_UTILIZATION=0.10
    SYSTEM_PROMPT_SELECTOR=llama/flowershop
    ```

2. Restart the services:

    ```bash
    docker compose -f docker-compose.jetson.yml down
    docker compose -f docker-compose.jetson.yml up -d
    ```

3. Verify the model is loading:

    ```bash
    docker compose -f docker-compose.jetson.yml logs -f llm-nvidia-jetson
    ```

> **Note:** The first model download may take several minutes depending on model size and network speed.

---

## Common Commands

```bash
# View logs
docker compose -f docker-compose.jetson.yml logs -f python-app

# Stop all services
docker compose -f docker-compose.jetson.yml down

# Rebuild after code changes
docker compose -f docker-compose.jetson.yml up --build -d python-app
```
