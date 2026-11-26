# Deploying Voice Agent on Jetson Thor

Deploy a Pipecat Single-GPU recipe on Jetson Thor. The cascaded examples run separate NeMo-Speech.cpp ASR and TTS sidecars alongside vLLM. Omni Assistant uses NeMo-Speech.cpp only for TTS. The [Generic Assistant on LiveKit](../src/nemotron_voice_agent/livekit/examples/generic_assistant/README.md) has a separate UI and is not validated on Thor.

---

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.0** using [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed). Orin-class Jetsons are not supported.
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/) 2.24.4 or newer
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models
- Python 3 and either the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) (`hf`) or `uvx` for downloading the speech weights. For the Generic Assistant, install the [voice activity detection (VAD) model converter prerequisites](01-getting-started.md#prerequisites).

---

## Deployment Steps

Run these steps for one Pipecat recipe.

1. Clone the repository.

    ```bash
    git clone https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    ```

2. Change to the repository root.

    ```bash
    cd nemotron-voice-agent
    ```

3. Create `.env` only if it does not exist.

    ```bash
    test -f .env || cp .env.example .env
    ```

4. Set `HF_TOKEN` in `.env` for Hugging Face model downloads. The local Single-GPU recipes do not require `NVIDIA_API_KEY` or `docker login nvcr.io`.

    ```bash
    # Required for */single-gpu
    HF_TOKEN=<your-huggingface-token>
    ```

    `omni-assistant-subagents/single-gpu` is **not supported** on Jetson Thor. Use cloud `omni-assistant-subagents` for that example, which runs on NVIDIA cloud endpoints and needs `NVIDIA_API_KEY` (not `HF_TOKEN`). The no-`NVIDIA_API_KEY` rule above applies only to the `*/single-gpu` path.

5. Download the NeMo-Speech.cpp model weights from your regular user account without `sudo`. Rerun this step after an upgrade. The script saves the ASR, Magpie TTS, and NanoCodec weights and Magpie TTS text-normalization grammars in `models/nemo-speech` by default.

    To use a different directory, set `NEMO_SPEECH_MODEL_LOC` in `.env` to an absolute path. If you previously exported that variable in your shell, run `unset NEMO_SPEECH_MODEL_LOC` in the terminal you use for the download and Compose. Both then use the directory from `.env`.

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

    The script reads `HF_TOKEN` from `.env`. For `generic-assistant/single-gpu`, resolve any VAD build warnings before startup.

6. Check available memory before starting. GPU, OS, containers, and page cache
   all share Thor's unified memory, so treat `MemAvailable` as the ceiling:

    ```bash
    nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader
    awk '/MemTotal|MemAvailable/ {print}' /proc/meminfo
    ```

    The Lightning service recognizes Jetson Thor from the platform product name
    and selects the Jetson Thor NVFP4 recipe automatically. No vLLM setting is
    required in `.env`.

7. Start one recipe. Replace `<example>` with `generic-assistant`, `multilingual-assistant`, `omni-assistant`, or `frontend-backend-live`.

    ```bash
    docker compose --profile <example>/single-gpu up -d
    ```

    The first start can take time to download and load the models. For troubleshooting, inspect logs with `docker compose --profile <example>/single-gpu logs --tail 200` and refer to [Single-GPU Troubleshooting](06-troubleshooting.md#single-gpu).

8. Open `https://<machine-ip>:7860/` in your browser. Keep TLS enabled for microphone access. For certificate warnings, refer to [Browser Access](06-troubleshooting.md#browser-access).
9. Select **Connect**.
10. Allow microphone access if prompted.
11. Ask a question after the greeting. Expect a spoken reply.
12. Select **Disconnect** to end the session.
13. Stop the deployment using the same profile:

    ```bash
    docker compose --profile <example>/single-gpu down
    ```

For an application client rather than the browser UI, refer to [Use the Live Gateway](how-to/use-live-gateway.md) or [Use the Realtime Gateway](how-to/use-realtime-gateway.md).
