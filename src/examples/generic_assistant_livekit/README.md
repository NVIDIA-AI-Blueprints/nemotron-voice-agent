# Generic Assistant on LiveKit

This example is a cascaded voice assistant built on [LiveKit Agents](https://docs.livekit.io/agents/) instead of Pipecat. The agent worker keeps automatic speech recognition (ASR), large language model (LLM), tools, and text-to-speech (TTS) as separate services. It uses Nemotron ASR through the LiveKit NVIDIA plugin, a Nemotron LLM through the OpenAI-compatible API, and Magpie TTS through the same plugin. A small web UI joins a LiveKit room, shows live transcripts, and shows each tool call.

The example is self-contained. It has its own Python environment, image, and Compose services. It does not use the Pipecat server, the React client, or [`examples_registry.yaml`](../../../examples_registry.yaml). It reuses the model recipes in [`docker/`](../../../docker) and the catalog in [`services.yaml`](../../../services.yaml), and nothing outside this folder refers to it. To move the example to another repository, replace the `extends` entries in `docker-compose.yaml` with copies of those service definitions.

![Architecture diagram](images/architecture.svg)

## Default Models

The agent reads the first entry of each slot that the `generic-assistant` example offers in `services.yaml`. Select a different catalog entry with `LLM_SERVICE`, `ASR_SERVICE`, or `TTS_SERVICE` in `.env`. The following table lists the default entries for each profile:

| Profile | ASR | LLM | TTS |
| --- | --- | --- | --- |
| Cloud | Nemotron ASR Streaming English | Nemotron 3.5 Lightning 30B A3B | Magpie TTS Multilingual |
| Server | Nemotron ASR Streaming English (NIM) | Nemotron 3.5 Lightning 30B A3B (NIM) | Magpie TTS Multilingual (NIM) |
| Single GPU | Nemotron Speech Streaming (NeMo-Speech.cpp) | Nemotron 3.5 Lightning 30B A3B (vLLM) | Magpie TTS (NeMo-Speech.cpp) |

## Running the Example

This example runs with the Cloud, Server (NVIDIA NIM), and Single GPU profiles. The Single GPU profile covers workstations, DGX Spark, and Jetson Thor. Refer to the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware detail. Run all commands from the repository root (`nemotron-voice-agent/`).

1. Clone the repository and change to its root. Skip this step if you already have a clone:

   ```bash
   git clone https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent.git
   cd nemotron-voice-agent
   ```

2. Keep your existing `.env` file. If you do not have one, copy the example template. If you already have one for another example, add the entries from [`.env.example`](.env.example) to it:

   ```bash
   test -f .env || cp src/examples/generic_assistant_livekit/.env.example .env
   ```

3. In `.env`, set `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`. Generate a secret with `openssl rand -hex 24`. Set `NVIDIA_API_KEY` for the Cloud and Server profiles. For the Single GPU profile, also set `HF_TOKEN` and leave `NVIDIA_API_KEY` empty.

4. For the Server profile only, log in to the NVIDIA NGC container registry:

   ```bash
   set -a; . ./.env; set +a
   printf '%s' "$NVIDIA_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
   ```

5. Deploy the profile that matches your hardware. The following commands start each recipe:

   The `--env-file .env` option makes Compose read the root `.env` for variable interpolation. Without it, Compose looks for a `.env` file next to the Compose file.

   ```bash
   docker compose --env-file .env -f src/examples/generic_assistant_livekit/docker-compose.yaml \
     --profile generic-assistant-livekit up -d            # Cloud, no local GPU
   docker compose --env-file .env -f src/examples/generic_assistant_livekit/docker-compose.yaml \
     --profile generic-assistant-livekit/server up -d     # Server with NIM

   # One GPU, including DGX Spark and Jetson Thor. Download the speech weights as your user:
   bash scripts/download-nemo-speech-models.sh
   docker compose --env-file .env -f src/examples/generic_assistant_livekit/docker-compose.yaml \
     --profile generic-assistant-livekit/single-gpu up -d
   ```

   The following table lists the services that each recipe profile starts:

   | Recipe profile | App services | Sidecars |
   | --- | --- | --- |
   | `generic-assistant-livekit` | `livekit-server`, `livekit-agent`, `livekit-ui` | none (cloud NVCF) |
   | `generic-assistant-livekit/server` | `livekit-server`, `livekit-agent-server`, `livekit-ui` | `livekit-nvidia-llm`, `livekit-nemotron-asr`, `livekit-magpie-tts` |
   | `generic-assistant-livekit/single-gpu` | `livekit-server`, `livekit-agent-single-gpu`, `livekit-ui` | `livekit-nvidia-llm-vllm`, `livekit-nemo-speech` |

   The sidecars extend the shared service definitions in [`docker/`](../../../docker), so they use the same images and settings as the `generic-assistant` recipes. Each sidecar has a network alias that matches the hostname in `services.yaml`. Wait until they are healthy before you connect, because the first NIM or vLLM start downloads and loads models.

6. Open the UI at `http://localhost:3000/`, select **Connect**, and speak. Try "What is the weather in Paris?". The transcript panel shows each tool call and result.

7. Tear down with the same profile that you started with:

   ```bash
   docker compose --env-file .env -f src/examples/generic_assistant_livekit/docker-compose.yaml \
     --profile generic-assistant-livekit/server down   # or generic-assistant-livekit, or /single-gpu
   ```

Do not run this example and a Pipecat recipe at the same time. Their sidecars publish the same host ports.

## Remote Browser Access

This example does not include a Traversal Using Relays around NAT (TURN) server. The UI is published on `127.0.0.1` only, and LiveKit advertises `LIVEKIT_NODE_IP` (default `127.0.0.1`), so the browser must run on the same machine as Docker. The LiveKit server publishes ports 7880, 7881 (TCP), and 7882 (UDP). Browsers also allow microphone access only on `localhost` or HTTPS.

For a browser on another machine, set `LIVEKIT_NODE_IP` to the host address, serve the UI over HTTPS, and add a TURN server. The repository provides one in [`docker/docker-compose.coturn.yaml`](../../../docker/docker-compose.coturn.yaml). Refer to [Enable the TURN Server](../../../docs/how-to/enable-turn-server.md). The UI does not pass TURN credentials to LiveKit, so you must extend `ui/app.js` to use relay candidates.

## Customization

| Path | Role |
| --- | --- |
| `agent.py` | LiveKit agent worker: builds ASR, LLM, and TTS clients, registers tools, and publishes UI events |
| `config.py` | Resolves endpoints from `services.yaml` for the `cloud`, `server`, and `single-gpu` recipes |
| `prompts.yaml` | Prompt and greeting for the assistant |
| `tools.py` | Plain async functions behind the weather and date tools. No LiveKit imports |
| `tool_wrapper.py` | Wraps tool functions as LiveKit tools that report calls to the UI |
| `ui_server.py` | Serves `ui/` and mints a LiveKit join token at `/token` |
| `ui/` | Web UI |
| `docker-compose.yaml`, `Dockerfile` | LiveKit server, agent, UI, and model sidecar services, and the image the app services share |
| `.env.example` | Environment template for this example |
| `pyproject.toml`, `uv.lock` | Example-only dependencies |

To add a tool, write a function with type hints and a docstring `Args:` section in `agent.py`, add it to `TOOLS`, and mention it in `prompts.yaml`. Put the underlying logic in `tools.py`. To change models, refer to [Configure Services](../../../docs/how-to/configure-services.md). The agent applies the entry's `extra_params` and turns reasoning off, as the `settings.yaml` default does. It does not read the other editable settings.

## Dependencies

This example keeps its dependencies separate from the Pipecat environment. The root `pyproject.toml` forces `nvidia-riva-client>=2.27.0,<3`, and the example locks its own set (currently 2.28.0 in `uv.lock`). Do not merge the two environments.

> **Temporary pin:** the NVIDIA STT and TTS changes from [livekit/agents#6706](https://github.com/livekit/agents/pull/6706) merged on November 12, but no `livekit-plugins-nvidia` release contains them yet (1.8.11 predates the merge). `pyproject.toml` pins `livekit-plugins-nvidia`, `livekit-agents`, and `livekit-plugins-silero` to the merge commit `b7c9e2d0` of `livekit/agents`. When the next release ships, replace the pin with `livekit-plugins-nvidia>=1.8.12` and remove the three `[tool.uv.sources]` entries.

To run the unit tests and install the example environment without Docker, run the following commands:

```bash
uv run pytest tests/unit/test_livekit_example_tools.py tests/unit/test_livekit_example_config.py -v
uv sync --project src/examples/generic_assistant_livekit
```

## Credits

The web UI layout and styling are adapted from [aws-samples/sample-voice-agent-on-aws](https://github.com/aws-samples/sample-voice-agent-on-aws/tree/main/samples/cascading/livekit-transcribe-polly), which is licensed under MIT-0. The license text is in [THIRD_PARTY_NOTICES.txt](../../../THIRD_PARTY_NOTICES.txt). The browser loads [livekit-client](https://github.com/livekit/client-sdk-js) 2.15.3 (Apache-2.0) from the jsDelivr CDN.
