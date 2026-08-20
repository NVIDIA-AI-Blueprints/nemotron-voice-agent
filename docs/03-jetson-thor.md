# Deploying Voice Agent on Jetson Thor

This guide covers deploying the Nemotron Voice Agent on Jetson Thor using Docker Compose.

Thor deploys with the `*/single-gpu` recipes. They run the NeMo-Speech.cpp speech
stack (ASR and TTS from local GGUF weights) next to vLLM on the same GPU, which is
the supported configuration on Thor.

---

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.0** using [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed). Orin-class Jetsons are not supported.
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/)
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models
- The [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (`hf`) for downloading the speech GGUFs

---

## Deployment Steps

1. Clone the repository and configure the environment:

    ```bash
    git clone git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    cp .env.example .env
    ```

2. Set your API keys in the `.env` file:

    ```bash
    # Required
    NVIDIA_API_KEY=<your-nvidia-api-key>
    HF_TOKEN=<your-huggingface-token>
    ```

    Export the NVIDIA API key in your shell and log in to the NVIDIA NGC Docker Registry before pulling the NGC images:

    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    ```

3. Download the NeMo-Speech.cpp model weights. **One-time per machine.** The script
   fetches the ASR, Magpie TTS, and NanoCodec GGUFs into `models/nemo-speech`:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

    Run as your user, not `sudo`. The script reads `HF_TOKEN` from `.env`.

    > To keep the weights outside the repo so they survive re-clones and worktrees, set
    > `NEMO_SPEECH_MODEL_LOC` in `.env` to an absolute path, or pass that path as
    > the script's first argument.

4. Check available memory before starting. GPU, OS, containers, and page cache
   all share Thor's unified memory, so treat `MemAvailable` as the ceiling:

    ```bash
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader
    awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
    ```

    The Lightning service recognizes Jetson Thor from the platform product name
    and selects the Jetson Thor NVFP4 recipe automatically. No vLLM setting is
    required in `.env`.

5. Start the full stack via Docker Compose. This brings up the LLM (vLLM), the
   NeMo-Speech.cpp sidecar, and the Pipecat pipeline together. Choose the profile for
   your example:

    ```bash
    # Generic Cascaded: NeMo-Speech.cpp ASR + TTS + Nemotron 3.5 Lightning vLLM
    docker compose --profile generic-assistant/single-gpu up -d

    # Multilingual Cascaded: multilingual ASR + Magpie TTS + Nemotron 3.5 Lightning vLLM
    docker compose --profile multilingual-assistant/single-gpu up -d

    # Nemotron Omni: local Omni vLLM + NeMo-Speech.cpp TTS only (Omni does its own ASR)
    docker compose --profile omni-assistant/single-gpu up -d

    # Frontend/Backend Agent: NeMo-Speech.cpp + Lightning vLLM + booking-server
    docker compose --profile frontend-backend-agent/single-gpu up -d
    ```

    > **Note:** First-run deployment can take 30–60 minutes. On local recipes, the **first voice interaction** may also lag while GPU sidecars warm up. Later turns are much faster.

6. Access the application at `https://<machine-ip>:7860` (HTTPS by default, which browser microphone and WebRTC require).

    > **Note:** `PIPELINE_TLS=false` serves plain HTTP for headless/API testing only. For plain-HTTP browser testing, see [plain-HTTP deployment and usage](06-troubleshooting.md#browser-access).

    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.

7. **Manage and tear down.** Use the same profile you started with (`<example>` = `generic-assistant`, `multilingual-assistant`, `omni-assistant`, or `frontend-backend-agent`).

    ```bash
    # View logs for the whole profile
    docker compose --profile <example>/single-gpu logs -f

    # Rebuild after code changes
    docker compose --profile <example>/single-gpu up --build -d

    # Stop all services
    docker compose --profile <example>/single-gpu down
    ```

    If hitting startup or runtime issues, see [Troubleshooting](06-troubleshooting.md#single-gpu), which covers low-memory and vLLM engine-core failures, and more.
