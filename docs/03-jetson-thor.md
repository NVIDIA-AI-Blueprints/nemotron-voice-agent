# Deploying Voice Agent on Jetson Thor

This guide covers deploying the Nemotron Voice Agent on Jetson Thor using Docker Compose.

---

## Prerequisites

- **Jetson Thor** flashed with **JetPack 7.0** using [NVIDIA SDK Manager](https://developer.nvidia.com/sdk-manager) (with CUDA, CUDA-X, TensorRT, and NVIDIA Container Runtime components installed). Orin-class Jetsons are not supported.
- [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli) installed and configured
- [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and [Docker Compose](https://docs.docker.com/compose/install/linux/)
- [HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) for downloading LLM models

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

3. Build the Nemotron Speech (Riva) model repository. **One-time per machine.**

    a. Ensure you meet the [prerequisites](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html#prerequisites) before proceeding.

    b. Configure NGC CLI with your API key:

    ```bash
    ngc config set
    ```

    c. Download the Riva Speech Skills v2.26.0 Quick Start bundle for L4T (JetPack 7.0) **next to the repo** (not inside it), so the ~30–50 GB model repo survives re-clones and worktrees:

    ```bash
    cd ..
    ngc registry resource download-version "nvidia/riva/riva_quickstart_arm64:2.26.0"
    cd riva_quickstart_arm64_v2.26.0
    ```

    d. **Configure the Riva deployment.** Edit `config.sh` in the quickstart directory to choose which services and models `riva_init.sh` builds into `model_repository/`:

    | Setting | Parameter in `config.sh` | Default |
    |---|---|---|
    | Enable ASR service | `service_enabled_asr` | `true` |
    | Enable TTS service | `service_enabled_tts` | `true` |
    | ASR acoustic model to fetch from NGC | `asr_acoustic_model` | English Parakeet (default) |
    | ASR language | `asr_language_code` | `en-US` |
    | TTS language | `tts_language_code` | `en-US` |

    - **Omni examples** (`omni-assistant/jetson-thor`): set `service_enabled_asr=false`. Omni LLM doesn't need ASR separately, so Riva only needs to serve TTS.
    - **Multilingual deployments**: multilingual Jetson profiles are not included in this blueprint. If you build a custom profile, switch `asr_acoustic_model` to the multilingual ASR model and set `asr_language_code` (and `tts_language_code`) to your target locales.

    > Exact model identifiers and the full option list live in the downloaded `config.sh`.

    e. Run only `riva_init.sh`. It downloads the configured ASR/TTS models and compiles the TRT engines into `model_repository/`. **Do not run `riva_start.sh`**: the `nemotron-speech` compose service in step 5 will serve the models itself.

    ```bash
    bash riva_init.sh
    cd ../nemotron-voice-agent
    ```

    > **Note:** Initialization may take 30–60 minutes on first run.
    >
    > If your repository clone uses a different directory name, return to that clone directory instead of `../nemotron-voice-agent`.

    > If the quickstart lives somewhere other than the repo's sibling directory, set `RIVA_MODEL_LOC` in `.env` to the absolute path of `model_repository/`.

4. **(Optional) Production tuning: CUDA MPS + CPU pinning.** On Thor, vLLM and Riva share a single GPU and the memory bus. Left unmanaged they contend for GPU SMs and CPU cores, which shows up as audible glitches in the bot's speech. CUDA MPS partitions the GPU's compute between the two services, and CPU pinning gives each its own cores. Both are recommended on Thor. Set the split in `.env`, then start the MPS daemon **before** bringing up the stack (step 5):

    ```bash
    # .env
    VLLM_MPS_THREAD_PCT=50
    RIVA_MPS_THREAD_PCT=50
    VLLM_CPUSET=0-3
    RIVA_CPUSET=4-7
    PIPECAT_CPUSET=8-11
    ```

    ```bash
    sudo bash scripts/start-mps.sh
    ```

5. Start the full stack via Docker Compose. This brings up the LLM (vLLM), Riva speech, and the Pipecat pipeline together. Choose the profile for your example:

    ```bash
    # Generic Cascaded — Riva ASR + TTS + vLLM LLM
    docker compose --profile generic-assistant/jetson-thor up -d

    # Nemotron Omni — local Omni vLLM + Riva TTS only (Omni does its own ASR; set service_enabled_asr=false in step 3d)
    docker compose --profile omni-assistant/jetson-thor up -d
    ```

    > **Note:** First-run deployment can take 30–60 minutes. On local recipes, the **first voice interaction** may also lag while GPU sidecars warm up. Later turns are much faster.

6. Access the application at `https://<machine-ip>:7860` (HTTPS by default, which browser microphone and WebRTC require).

    > **Note:** `PIPELINE_TLS=false` serves plain HTTP for headless/API testing only. For plain-HTTP browser testing, see [plain-HTTP deployment and usage](06-troubleshooting.md#browser-access).

    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.

7. **Manage and tear down.** Use the same profile you started with (`<example>` = `generic-assistant` or `omni-assistant`). If you enabled CUDA MPS in step 4, stop the daemon when tearing down.

    ```bash
    # View logs for the whole profile
    docker compose --profile <example>/jetson-thor logs -f

    # Rebuild after code changes
    docker compose --profile <example>/jetson-thor up --build -d

    # Stop all services
    docker compose --profile <example>/jetson-thor down

    # Stop CUDA MPS (only if you ran scripts/start-mps.sh)
    sudo bash scripts/stop-mps.sh
    ```

    If hitting startup or runtime issues, see [Troubleshooting](06-troubleshooting.md#jetson-thor), which covers Jetson low-memory and vLLM engine-core failures, and more.
