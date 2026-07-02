# Getting Started

This guide walks you through different deployment options for the Nemotron Voice Agent on your system, from a cloud-only quick start to local GPU, DGX Spark, and Jetson Thor deployments.

## Prerequisites

Before you begin, ensure you have the following:

- Access to NVIDIA NGC with valid credentials. Refer to the [NGC Getting Started Guide](https://docs.nvidia.com/ngc/ngc-overview/index.html#registering-activating-ngc-account).
- Docker Compose v2.20 or newer (Check using `docker compose version`).
- NVIDIA API key. Required for accessing NIM ASR, TTS, and LLM models and Docker images. Get yours at [build.nvidia.com](https://build.nvidia.com/).

For cloud-only profiles, Docker and Docker Compose are sufficient. For local GPU profiles, install Docker with NVIDIA GPU support and verify `nvidia-smi` works inside containers. Refer to the [NVIDIA Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

## Docker based Deployment

Each example ships as Docker Compose **profiles**. Pick exactly one per deployment. The bare **`<example>`** profile runs cloud-only (no local GPU, using NVIDIA cloud API endpoints), while **`<example>/workstation`**, **`<example>/dgx-spark`**, and **`<example>/jetson-thor`** deploy models locally. Pick the profile that matches the example you want to run. `docker compose up` with no profile is intentionally a no-op so the deployment is always explicit.

> **Note:** For example-specific architecture, configuration, and tunables, see each example's README (linked in the table below).

| Example | Description | Supported profiles |
|---------|-------------|--------------------|
| [`generic-assistant`](../src/examples/generic/README.md) | Baseline English-only cascaded pipeline (Nemotron ASR + LLM + Magpie TTS) | `generic-assistant`, `generic-assistant/workstation`, `generic-assistant/dgx-spark`, `generic-assistant/jetson-thor` |
| [`multilingual-assistant`](../src/examples/multilingual/README.md) | Multilingual cascaded pipeline with automatic language/voice switching | `multilingual-assistant`, `multilingual-assistant/workstation`, `multilingual-assistant/dgx-spark` |
| [`omni-assistant`](../src/examples/omni_assistant/README.md) | Nemotron Omni model (ASR + LLM) + Magpie TTS cascaded pipeline | `omni-assistant`, `omni-assistant/workstation`, `omni-assistant/dgx-spark`, `omni-assistant/jetson-thor` |
| [`omni-assistant-subagents`](../src/examples/omni_assistant_subagents/README.md) | Multi-agent Omni with media + live-webcam understanding | `omni-assistant-subagents`, `omni-assistant-subagents/workstation`, `omni-assistant-subagents/dgx-spark` |
| [`frontend-backend-agent`](../src/examples/frontend_backend_agent/README.md) | Frontend LLM with a stateful backend agent (airline-booking reference) | `frontend-backend-agent`, `frontend-backend-agent/workstation` |

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
    > **Optional (DGX Spark / Jetson Thor):** Set `HF_TOKEN` in `.env` for the LLM model download from huggingface.

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

    **5.2 Workstation** (local workstation / server GPUs):

    ```bash
    docker compose --profile generic-assistant/workstation up -d         # Generic Cascaded
    docker compose --profile multilingual-assistant/workstation up -d    # Multilingual Cascaded
    docker compose --profile omni-assistant/workstation up -d            # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/workstation up -d  # Nemotron Omni Assistant Subagents
    docker compose --profile frontend-backend-agent/workstation up -d    # Frontend/Backend Agent Airline Assistant
    ```

    **5.3 DGX Spark** (Blackwell, 128 GB unified memory):

    ```bash
    docker compose --profile generic-assistant/dgx-spark up -d           # Generic Cascaded
    docker compose --profile multilingual-assistant/dgx-spark up -d      # Multilingual Cascaded
    docker compose --profile omni-assistant/dgx-spark up -d              # Nemotron Omni Assistant
    docker compose --profile omni-assistant-subagents/dgx-spark up -d    # Nemotron Omni Assistant Subagents
    ```

    **5.4 Jetson Thor** (edge support). Follow the [Jetson Thor guide](03-jetson-thor.md) for the one-time Riva model build first:

    ```bash
    docker compose --profile generic-assistant/jetson-thor up -d         # Generic Cascaded
    docker compose --profile omni-assistant/jetson-thor up -d            # Nemotron Omni Assistant
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

    Host-native runs read [`examples_registry.yaml`](../examples_registry.yaml) at the repository root. Edit the `selection` field to choose what the UI exposes, then start the server normally. The server has no example/pipeline CLI flags.

    By default a host-native server uses the cloud (NVCF) service endpoints. To run against **local on-prem services** instead, set `PLATFORM` in `.env` to the hardware whose local catalog you want: `workstation`, `dgxspark`, or `jetsonthor`. This selects the matching section of the example's `services.local.yaml`, so the server connects to the local ASR, LLM, and TTS sidecars for that hardware instead of the cloud endpoints. Docker Compose recipe profiles set `PLATFORM` automatically (for example, `<example>/workstation` sets `PLATFORM=workstation`), so you only set it by hand for host-native runs.

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
