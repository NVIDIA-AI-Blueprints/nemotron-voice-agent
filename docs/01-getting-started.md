# Getting Started

This guide walks you through the cloud-only, server, and single-GPU deployment options for the Nemotron Voice Agent. The single-GPU recipes cover supported workstation GPUs, DGX Spark, and Jetson Thor.

## Prerequisites

Before you begin, ensure you have the following:

- Access to NVIDIA NGC with valid credentials. Refer to the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).
- Docker Compose v2.20 or newer (Check using `docker compose version`).
- NVIDIA API key. Required for accessing NIM ASR, TTS, and LLM models and Docker images. Get yours at [build.nvidia.com](https://build.nvidia.com/).

For cloud-only profiles, Docker and Docker Compose are sufficient. For local GPU profiles, install Docker with NVIDIA GPU support and verify `nvidia-smi` works inside containers. Refer to the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Docker based Deployment

Each example ships as Docker Compose **profiles**. Pick exactly one per deployment. The bare **`<example>`** profile runs cloud-only (no local GPU, using NVIDIA cloud API endpoints). **`<example>/server`** is the scaling-oriented NIM stack. **`<example>/single-gpu`** is the universal one-GPU path. Supported hardware varies by example and is listed below. `docker compose up` with no profile is intentionally a no-op so the deployment is always explicit.

> **Note:** For example-specific architecture, configuration, and tunables, see each example's README (linked in the table below).

| Example | Description | Supported profiles |
|---------|-------------|--------------------|
| [`generic-assistant`](../src/examples/generic/README.md) | Baseline English-only cascaded pipeline (Nemotron ASR + LLM + Magpie TTS) | `generic-assistant`, `generic-assistant/server`, `generic-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor) |
| [`multilingual-assistant`](../src/examples/multilingual/README.md) | Multilingual cascaded pipeline with a fixed language per session | `multilingual-assistant`, `multilingual-assistant/server`, `multilingual-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor) |
| [`omni-assistant`](../src/examples/omni_assistant/README.md) | Nemotron Omni model (ASR + LLM) + Magpie TTS cascaded pipeline | `omni-assistant`, `omni-assistant/server`, `omni-assistant/single-gpu` (workstation, DGX Spark, Jetson Thor) |
| [`omni-assistant-subagents`](../src/examples/omni_assistant_subagents/README.md) | Multi-agent Omni with media + live-webcam understanding | `omni-assistant-subagents`, `omni-assistant-subagents/server`, `omni-assistant-subagents/single-gpu` (workstation, DGX Spark) |
| [`frontend-backend-agent`](../src/examples/frontend_backend_agent/README.md) | Frontend LLM with a stateful backend agent (airline-booking reference) | `frontend-backend-agent`, `frontend-backend-agent/server`, `frontend-backend-agent/single-gpu` (workstation, DGX Spark, Jetson Thor) |
| [`generic-frontend-backend-agent`](../src/examples/frontend_backend_agent/README.md) | Shared Talker/Thinker pipeline with grounded generic tools | Set `EXAMPLE_SELECTION=generic-frontend-backend-agent` with a Frontend/Backend Agent profile |

> Observability overlays `tracing` (Phoenix OTel) and Coturn Server `turn` can be added to any profile.

---

### Deployment Steps

1. Clone the repository and navigate to the root directory.

    ```bash
    git clone git@github.com:NVIDIA-AI-Blueprints/nemotron-voice-agent.git
    cd nemotron-voice-agent
    ```

2. Configure the environment. Copy the example environment file [.env.example](../.env.example) to the root directory, then set `NVIDIA_API_KEY` in `.env`. Docker Compose passes `.env` values into the app and model sidecars, so exporting the key in your shell is not enough for runtime.

    ```bash
    cp .env.example .env
    # Edit .env and replace the placeholder with your key:
    # NVIDIA_API_KEY=<your-nvidia-api-key>
    ```
    > **Local vLLM recipes:** Set `HF_TOKEN` in `.env` for the LLM model download from Hugging Face.

3. Export the same NVIDIA API key in your shell for Docker registry login:

    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

4. Log in to the NVIDIA NGC Docker Registry.

    ```bash
    printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    ```

5. Deploy the example profile of your choice.

    **5.1 Cloud only** (no local GPU):

    ```bash
    docker compose --profile generic-assistant up -d            # Generic Cascaded
    docker compose --profile multilingual-assistant up -d       # Multilingual Cascaded
    docker compose --profile omni-assistant up -d               # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents up -d     # Nemotron Omni Assistant Subagents
    docker compose --profile frontend-backend-agent up -d       # Frontend/Backend Agent Airline Assistant
    ```

    **5.2 Server** (scaling-oriented NIM stack):

    ```bash
    docker compose --profile generic-assistant/server up -d         # Generic Cascaded
    docker compose --profile multilingual-assistant/server up -d    # Multilingual Cascaded
    docker compose --profile omni-assistant/server up -d            # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/server up -d  # Nemotron Omni Assistant Subagents
    docker compose --profile frontend-backend-agent/server up -d    # Frontend/Backend Agent Airline Assistant
    ```

    **5.3 Single GPU** (one supported GPU). This is the universal one-GPU deployment path. Hardware support varies by example as listed above. Cascaded recipes run NeMo-Speech.cpp next to Nemotron 3.5 Lightning. Omni recipes retain the multimodal Omni model and use NeMo-Speech.cpp for TTS. The Lightning container selects NVFP4 or FP8 from the supported platform and GPU compute capability. DGX Spark enables DSpark speculative decoding and Blackwell workstations enable DFlash automatically. Follow the [Jetson Thor guide](03-jetson-thor.md) when applicable.

    Download the NeMo-Speech.cpp weights **once, as your user** (do not use `sudo`). The script reads `HF_TOKEN` from `.env` and creates `models/nemo-speech`:

    ```bash
    bash scripts/download-nemo-speech-models.sh
    ```

    Then start the stack:

    ```bash
    docker compose --profile generic-assistant/single-gpu up -d          # Generic Cascaded
    docker compose --profile multilingual-assistant/single-gpu up -d     # Multilingual Cascaded
    docker compose --profile omni-assistant/single-gpu up -d             # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/single-gpu up -d   # Omni Assistant Subagents (workstation / DGX Spark)
    docker compose --profile frontend-backend-agent/single-gpu up -d     # Frontend/Backend Agent
    ```


    To verify all services are healthy, run `docker compose ps`.

    > **Note:** Each Docker Compose profile pins `EXAMPLE_SELECTION=<example>`, so the container runs that single example. Set `EXAMPLE_SELECTION=all` to expose every example in the UI selector instead.
    >
    > **Note:** First-run deployment can take 30–60 minutes. On local recipes, the **first voice interaction** may also lag while GPU sidecars warm up. Later turns are much faster.

6. Access the application at `https://<machine-ip>:7860` (HTTPS by default, which browser microphone and WebRTC require).

    > **Note:** `PIPELINE_TLS=false` serves plain HTTP for headless/API testing only. For plain-HTTP browser testing, see [plain-HTTP deployment and usage](06-troubleshooting.md#browser-access).
    >
    > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.
    >
    > **Note:** If connecting from a different network (NAT, cloud, restrictive firewall), see [Enable a TURN Server for Remote Access](how-to/enable-turn-server.md).

---

## Optional: Local Development (without Docker)

For development and debugging, you can run the server directly:

1. Install [uv](https://docs.astral.sh/uv/) and Node.js 20+.

2. Install dependencies and build the client:

    ```bash
    uv sync --group dev
    cd client && npm install && npm run build && cd ..
    ```

3. Install local commit hooks:

    ```bash
    uv run --project . --group dev pre-commit install
    ```

    The hooks run formatting and linting checks on staged files during `git commit`.

4. Configure the environment:

    ```bash
    cp .env.example .env
    # Edit .env and set NVIDIA_API_KEY
    ```

5. Start the server:

    ```bash
    uv run python src/server.py --host 0.0.0.0 --port 7860
    ```

    To serve plain HTTP instead of HTTPS, set `PIPELINE_TLS=false` in `.env` or prefix the command:

    ```bash
    PIPELINE_TLS=false uv run python src/server.py --host 0.0.0.0 --port 7860
    ```

    Host-native runs read [`examples_registry.yaml`](../examples_registry.yaml) at the repository root. Edit the `selection` field to choose what the UI exposes, then start the server normally. The server has no example-selection CLI flag. Pipeline options such as `--prompt-file` remain available.

    By default a host-native server uses the cloud (NVCF) service endpoints. To run against **local on-prem services**, start the matching Compose sidecars first. The catalog merges `services.local.yaml` and exposes only endpoints that are reachable, so NIM (`/server`) or NeMo-Speech.cpp (`/single-gpu`) entries appear automatically.

    | `selection` in `examples_registry.yaml` | UI behavior |
    |-----------------------------------------|-------------|
    | `all` | Show every registered example (default) |
    | `generic-assistant` | Lock to Generic Assistant |
    | `multilingual-assistant` | Lock to Multilingual Assistant |
    | `omni-assistant` | Lock to Nemotron Omni Assistant |
    | `omni-assistant-subagents` | Lock to Nemotron Omni Assistant Subagents |
    | `frontend-backend-agent` | Lock to Frontend Backend Agent |

    > **Note:** Docker Compose deployments pin `EXAMPLE_SELECTION=<example>` to a single example. You can set `EXAMPLE_SELECTION=all` to expose every example in the UI selector instead.

6. Access the application locally at `https://localhost:7860`, or from another machine at
   `https://<machine-ip>:7860` (replace `<machine-ip>` with the host IP).

   > **Tip:** For the best experience, we recommend using a headset (preferably wired) instead of your laptop's built-in microphone.
