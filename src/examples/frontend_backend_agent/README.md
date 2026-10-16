# Frontend/Backend Agent Example

The Frontend/Backend Agent is the repository's frontend/backend voice agent, built on the shared Pipecat speech stack. A fast frontend large language model (LLM) talks to the user and decides whether to answer or delegate. A reasoning backend LLM does the delegated work with tools.

The agent serves two kinds of sessions:

- Browser UI sessions over WebRTC or the UI's WebSocket transport, with RTVI, like the `generic-assistant` example. The server runs five built-in tools.
- OpenAI Realtime sessions on `/v1/realtime` with the model `nvidia/nemotron-realtime-frontend-backend`. The Realtime client defines the tools and runs them.

This example adds the following behaviors to the cascaded voice loop:

- A frontend barge-in verdict decides whether speech during backend work continues the running task or starts a new one.
- The backend receives the full conversation history with each delegated request.
- Identifier normalization writes spoken identifiers, such as user IDs, in their written form before the agent sees them.
- Caller speech while client tool calls are outstanding resumes the tool continuation with the caller's words.
- An optional write gate holds every consequential client tool call until the caller confirms a spoken summary of it.

The agent logic lives in `core/`. The `adapters/pipecat/` package runs that logic on the repository's shared Pipecat speech stack. This example replaces the previous implementation of the Frontend/Backend Agent, which was an airline assistant, and keeps its `frontend-backend-agent` example key and Compose profiles.

To deploy this example, follow [Deploy the Frontend/Backend Agent](../../../docs/how-to/deploy-frontend-backend-agent.md). For evaluation runs, such as tau-bench (Tau), deploy it in [eval mode](../../../docs/how-to/deploy-frontend-backend-agent.md#deploy-in-eval-mode), which serves plain `ws://` and writes the agent's event and filler logs to `logs/` on the host.

## Default Models

The `frontend-backend-agent` defaults in [`examples_registry.yaml`](../../../examples_registry.yaml) resolve to the following models for each Compose profile. The LLMs are the same in every profile. The ASR and TTS endpoints follow the platform that Compose sets in `REALTIME_SERVICE_PLATFORM`.

| Compose profile | ASR | Frontend LLM | Backend LLM | TTS |
| --- | --- | --- | --- | --- |
| Cloud (`frontend-backend-agent`) | Nemotron ASR Streaming on NVIDIA cloud functions (NVCF) | Nemotron 3.5 Lightning on the NVIDIA Inference Hub | Nemotron 3 Ultra on the NVIDIA Inference Hub | Magpie TTS Multilingual on NVCF, voice `Magpie-Multilingual.EN-US.Aria` |
| Server (`frontend-backend-agent/server`) | Nemotron ASR Streaming English NIM (`nemotron-asr-streaming-english:50052`) | Same | Same | Magpie TTS Multilingual NIM (`magpie-multilingual-tts-service:50051`), voice `Magpie-Multilingual.EN-US.Aria` |
| Single GPU (`frontend-backend-agent/single-gpu`) | `nemotron-speech-streaming-en-0.6b` through NeMo-Speech.cpp (`nemo-speech:50051`) | Same | Same | Magpie TTS Multilingual through NeMo-Speech.cpp (`nemo-speech:50051`), voice `John` |

The LLMs use the following request settings from `core/config/agent.yaml`:

| Role | Model ID | Endpoint | Request settings |
| --- | --- | --- | --- |
| Frontend LLM | `nvidia/nvidia/nemotron-3.5-lightning` | `https://inference-api.nvidia.com/v1` | Reasoning off, `max_tokens` 1,024, `temperature` 0 |
| Backend LLM | `nvidia/nvidia/nemotron-3-ultra` | `https://inference-api.nvidia.com/v1` | Reasoning on, `reasoning_budget` 1,024, `max_tokens` 4,096, `temperature` 0 |

Both LLMs use `NVIDIA_API_KEY`. For this example, set it to an NVIDIA Inference Hub key (`sk-...`). The doubled `nvidia/nvidia/` prefix in the model IDs is the Inference Hub naming and is intentional. On the cloud recipe, the NVCF speech functions also use `NVIDIA_API_KEY`. It has not been verified that one Inference Hub key works for NVCF speech. For details, refer to [Credentials](../../../docs/how-to/deploy-frontend-backend-agent.md#credentials).

The speech pipeline uses the shared `NvidiaForceEouSTTService` for ASR and the Pipecat `NvidiaTTSService` for TTS, with one request per sentence at 22,050 Hz.

## Request Flow

Both kinds of sessions run the same processors between the transport and the output. Only the transport and the event protocol differ.

The following diagram shows a browser UI session:

```text
Browser UI
  |  WebRTC, or the UI's WebSocket (protobuf), with RTVI
  v
src/server.py -> bot() (adapters/pipecat/pipeline.py) -> serve_ui() (adapters/pipecat/ui_session.py)
  - refuses the session when FBA_PROFILE=eval or when 8 UI sessions are open
  - reads the UI's LLM, ASR, TTS, voice, and prompt choices from the request body
  v
Pipecat pipeline (adapters/pipecat/ui_pipeline.py), with the `default` settings
  transport.input()                   create_transport() from src/examples/shared/
    -> VADProcessor (Silero) -> InputCursor -> STT -> UserTurnProcessor (Smart Turn v3)
    -> FbaTurnBridge -> TTS -> OutputTagger
    -> RealtimeFrameFilter            drops the Realtime-only event frames
  transport.output()                  audio, plus RTVI transcripts and user-to-bot latency
```

The following diagram shows an OpenAI Realtime session:

```text
Realtime client
  |  WS /v1/realtime?model=nvidia/nemotron-realtime-frontend-backend
  v
Realtime gateway (src/realtime/gateway.py)
  - authenticates the REALTIME_API_KEY Bearer credential or an ek_ client secret
  - resolves ?model= to frontend-backend-agent
  - hands the socket to the pipeline before it sends any event
  v
Pipecat pipeline (adapters/pipecat/pipeline.py), always with the `eval` settings
  FastAPIWebsocketTransport.input()   client events, input audio decoded to 16 kHz PCM16
    -> VADProcessor (Silero)          shared Pipecat VAD
    -> InputCursor                    input-audio clock, speech onset and offset
    -> NvidiaForceEouSTTService       shared streaming ASR (src/examples/shared/)
    -> UserTurnProcessor              Pipecat user turns
    -> FbaTurnBridge                  drives the core turn manager:
                                        frontend LLM -> backend LLM -> tool calls,
                                        barge-in verdict, filler, identifier normalization
    -> NvidiaTTSService               Pipecat TTS, one request per sentence
    -> OutputTagger                   transcript and audio order, playout progress
  FastAPIWebsocketTransport.output()  server events and output audio deltas
```

In both sessions, the frontend LLM answers or calls the backend. The backend LLM plans the work and calls tools. The server runs the tools of a browser UI session. A Realtime session sends its function calls to the client and waits for the outputs.

The example has the following parts.

| Path | Role |
| --- | --- |
| `pipeline.py` | Registry entry point. It re-exports `bot` from `adapters/pipecat/pipeline.py`. |
| `core/agent/` | Frontend/Backend agent, LLM clients, prompt loading, and agent session. |
| `core/frontend/`, `core/backend/` | Frontend delegation and the backend with its conversation history and tool protocol. |
| `core/engine/` | Turn manager, filler, barge-in history repair, write gate, response writer, and the Realtime server events. |
| `core/normalization/` | Transcript and tool-argument identifier normalization. |
| `core/tools/` | Tool specifications and Realtime client tools. |
| `core/audio/` | PCM16 and G.711 audio formats and resampling. |
| `core/config/` | Configuration loader, `voice_agent.yaml`, `agent.yaml`, the two profiles (`profiles/`), and the prompt catalogs (`prompts/`). `profiles.py` validates `FBA_PROFILE`, picks each profile's Pipecat settings file, maps speech to the deployment platform, and checks the WebSocket keepalive. |
| `adapters/pipecat/` | Pipecat host: the Realtime pipeline and session, the browser UI pipeline (`ui_pipeline.py`) and session (`ui_session.py`), serializer, turn bridge, speech warm-up, the session caps, and the shared speech services. |
| `adapters/pipecat/settings/` | Pipecat VAD, user-turn, transport-audio, and TTS settings: `default.yaml` for browser UI sessions and `eval.yaml` for Realtime sessions. Refer to [Pipecat Voice Settings](#pipecat-voice-settings). |
| `tools.yaml`, `tools.py`, `tool_handlers.py` | The five server-run tools of browser UI sessions. Refer to [Tools](#tools). |
| `prompts.yaml` | Registry-facing prompt catalog. It mirrors `core/config/prompts/generic.yaml`. |
| `services.cloud.yaml`, `services.local.yaml` | Example-local service catalogs for the cloud, `server`, and `singlegpu` platforms. |

## Run the Examples

Refer to [Deploy the Frontend/Backend Agent](../../../docs/how-to/deploy-frontend-backend-agent.md) for prerequisites, credentials, profile selection, verification, and troubleshooting. Run every command from the repository root.

### Configure Credentials

Preserve an existing `.env`. Otherwise, create it from the template:

```bash
test -f .env || cp .env.example .env
```

Set the following keys in `.env`:

- `NVIDIA_API_KEY`: an NVIDIA Inference Hub key for both LLMs. The cloud recipe also sends it to NVCF for ASR and TTS.
- `HF_TOKEN`: a Hugging Face token for the NeMo-Speech.cpp model download. Only the `/single-gpu` recipe needs it.
- `REALTIME_API_KEY`: the Bearer credential that Realtime clients send. Leave it unset only for unauthenticated local development.
- `WEATHERAPI_KEY`, `FINNHUB_API_KEY`, and `PERPLEXITY_API_KEY`: optional keys for the browser UI tools. Refer to [Tools](#tools).

### Run Host-Native

A host-native server must turn the uvicorn WebSocket keepalive off with `UVICORN_WS_PING_INTERVAL=0`. The keepalive applies to the whole server process, and Realtime sessions refuse to start when it is on.

For speech on the NeMo-Speech.cpp sidecar, start only the sidecar. It publishes port `50051` on the host.

```bash
docker compose --profile frontend-backend-agent/single-gpu up -d nemo-speech
```

Then start the server and point the example's speech endpoints at the host port:

```bash
EXAMPLE_SELECTION=frontend-backend-agent \
REALTIME_SERVICE_PLATFORM=singlegpu \
FBA_ASR_SERVER=localhost:50051 \
FBA_TTS_SERVER=localhost:50051 \
UVICORN_WS_PING_INTERVAL=0 \
UVICORN_WS_PING_TIMEOUT=20 \
uv run python src/server.py --host 0.0.0.0 --port 7860
```

Open `https://localhost:7860/` for the browser UI, or connect a Realtime client to `wss://localhost:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend`. Add `PIPELINE_TLS=false` to serve plain HTTP and `ws://` for an isolated local test. A browser allows the microphone over plain HTTP only on `localhost`. You can also start `src/realtime_server.py`, which serves only the Realtime API, with the same environment.

To use the NVCF speech endpoints of `services.cloud.yaml` instead of a sidecar, set `REALTIME_SERVICE_PLATFORM=cloud` and leave `FBA_ASR_SERVER` and `FBA_TTS_SERVER` unset.

### Run With Docker Compose

Start one recipe with the `default` profile:

```bash
docker compose --profile frontend-backend-agent up -d
docker compose --profile frontend-backend-agent/server up -d
docker compose --profile frontend-backend-agent/single-gpu up -d
```

The `/server` recipe needs `docker login nvcr.io` first. The `/single-gpu` recipe needs the speech weights first: run `bash scripts/download-nemo-speech-models.sh` once, as your user.

For evaluation runs, start the `/single-gpu` recipe with the eval override, which sets the `eval` profile:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fba-eval.yaml \
  --profile frontend-backend-agent/single-gpu up -d
```

The recipes start the following services.

| Recipe profile | App service | Sidecars |
| --- | --- | --- |
| `frontend-backend-agent` | `frontend-backend-agent` | None |
| `frontend-backend-agent/server` | `frontend-backend-agent-server` | `nemotron-asr-streaming-english` (ASR) and `magpie-multilingual-tts-service` (TTS) |
| `frontend-backend-agent/single-gpu` | `frontend-backend-agent-single-gpu` | `nemo-speech` (ASR and TTS) |

Each app service sets `EXAMPLE_SELECTION=frontend-backend-agent`, `REALTIME_SERVICE_PLATFORM` (`cloud`, `server`, or `singlegpu`), `FBA_PROFILE=default` unless you set another value, `UVICORN_WS_PING_INTERVAL=0`, and `UVICORN_WS_PING_TIMEOUT=20`. TLS is on by default, because browsers need HTTPS for the microphone. The eval override turns it off.

Open `https://<host>:7860` for the browser UI, or connect a Realtime client to `wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend`. Use `ws://` when `PIPELINE_TLS=false`.

Tear down with the same Compose profile, and with the override file for an eval-mode deployment:

```bash
docker compose --profile frontend-backend-agent/single-gpu down
```

## Configure the Example

### Profiles

The `FBA_PROFILE` environment variable selects one of two profiles in `core/config/profiles/`. It decides what the container serves. OpenAI Realtime sessions always use the `eval` profile.

| Profile | Browser UI sessions | OpenAI Realtime sessions | Set by |
| --- | --- | --- | --- |
| `default` | Served with the `default` profile | Served with the `eval` profile | Compose when `FBA_PROFILE` is unset, and the agent when the variable is unset |
| `eval` | Refused. The session ends, and the app log shows the error. | Served with the `eval` profile | `docker/docker-compose.fba-eval.yaml`, on `/single-gpu` only |

The agent supports no other value. Any other value, including an old `tau3_eval*` profile name, fails the first session with an error that names `default` and `eval`. The value is checked when the first session arrives, not at server start. The `SERVED_PROFILES` tuple in `core/config/profiles.py` defines the list.

`eval.yaml` extends `core/config/voice_agent.yaml`. It is the former default Tau profile, `tau3_eval_frontend_verdict_speak_history`, with its `extends` chain flattened into one file and its merged configuration unchanged. `default.yaml` extends `eval.yaml` and changes only what a live browser user needs. The following table lists the differences.

| Setting | `eval` | `default` |
| --- | --- | --- |
| Turn end | Silero VAD with a server-owned 800 ms silence (`honor_client_values: false`), which keeps spelled-out IDs in one turn | Smart Turn v3. A 200 ms VAD stop hands the turn to Smart Turn, with a 1 s silence fallback (`SMART_TURN_STOP_SECS`). |
| Pipecat settings | `adapters/pipecat/settings/eval.yaml` | `adapters/pipecat/settings/default.yaml` |
| Greeting | Off. The history is seeded with the client greeting `Hi! How can I help you today?`. | On, unless `ENABLE_WELCOME_MESSAGE=false`. The agent says `Hi! How can I help you today?` when the browser is ready. No greeting is seeded. |
| Tools | Client-owned, from `session.update` | The five server-run tools of `tools.yaml` |
| Instructions | The client's `session.instructions` fill the backend prompt's domain-policy slot | No client instructions. The UI can replace the `frontend` prompt for the session. |
| Tool-argument normalization | On, for `get_user_details.user_id`, with the retry guard | Off. It applies only to client tools. |
| Prompt catalog | `core/config/prompts/tau.yaml`, pinned | `core/config/prompts/generic.yaml`, free to tune |
| Event and filler logs | From `FBA_VOICE_EVENT_LOG` and `FBA_FILLER_LOG` | Off |
| Audio | The client's format, PCM16 at 24,000 Hz by default | 16 kHz in and 22,050 Hz out on the browser transport |

Both profiles share the rest of the agent: the frontend barge-in verdict, spoken filler, backend conversation history, transcript normalization, the write gate (off), the server keepalive (off), and a cap of 8 concurrent sessions. The browser UI sessions and the Realtime sessions have separate session counters.

### Prompt Variants

The profile selects the prompt catalog. The example ships two variants in `core/config/prompts/`.

| Variant | File | Used by | Change rule |
| --- | --- | --- | --- |
| Tau | `core/config/prompts/tau.yaml` | `eval`, so every OpenAI Realtime session | Pinned by SHA-256 in `tests/unit/frontend_backend_agent/test_fba_prompt_variants.py`. A change requires a new pin and an attached Tau A/B evaluation run. |
| Generic | `core/config/prompts/generic.yaml` | `default`, so every browser UI session | Free to tune. It currently has the same content as `tau.yaml`. |

The example's `prompts.yaml` mirrors `generic.yaml`, and a unit test checks that the two files are identical. The browser UI shows the `frontend` prompt and can edit it for one session. The agent's other prompt keys stay hidden (`agent_prompt_keys` in `examples_registry.yaml`).

The `FBA_VOICE_PROMPTS` environment variable replaces the catalog path of the `eval` profile. The `default` profile sets `generic.yaml` itself. A relative path resolves against `core/config/`. A missing or invalid catalog fails the session, and the application log shows an error that names `agent.overrides.prompts.path`.

The example loads a profile when the first session of that kind connects. The application log then records the profile in the `frontend/backend agent ready` line and the catalog in a line of the following form:

```text
prompts <catalog path> (sha256 <hash>)
```

A browser UI session logs `profile=default`, and a Realtime session logs `profile=eval`. The `session_start` record of the event log carries `prompts` with the catalog file name (`catalog`) and its SHA-256 hash (`sha256`), the adapter, and the Pipecat settings summary (`pipecat`).

### Pipecat Voice Settings

Each profile's Pipecat settings file in `adapters/pipecat/settings/` is fixed in code (`PIPECAT_SETTINGS` in `core/config/profiles.py`). No environment variable selects it. The comments in each file give the reason for each value.

The following table lists the values that differ between the two files.

| Key | `eval.yaml` | `default.yaml` | Effect |
| --- | --- | --- | --- |
| `vad.stop_secs` | `0.8` | Not set | The session applies the profile's `turn_detection.silence_duration_ms` as the VAD stop: 800 ms for `eval`, 200 ms for `default`. |
| `vad.audio_idle_timeout` | `0` | Not set (Pipecat's default) | `0` turns off the wall-clock forced speech stop, so a Realtime client that pauses its input does not end a turn. |
| `turn.user_speech_timeout` | `0` | Not set (Pipecat's default) | `0` adds no wall-clock wait after the VAD stop. |
| `turn.smart_turn` | Not set (off) | `true` | Turns end on the shared Smart Turn v3 stop strategy, as in the repository's other examples. |

Both files set the following values:

| Key | Value | Effect |
| --- | --- | --- |
| `turn.stt_ttfs_p99` | `3.0` | After the VAD stop, the turn waits for the final transcript. The final transcript ends the wait as soon as it arrives. |
| `turn.wait_for_transcript` | `false` | A turn with no transcript, such as noise, ends after that wait with an empty transcript. |
| `audio.in_sample_rate` | `16000` | The input transport runs at 16 kHz. |
| `audio.out_sample_rate` | `null` | The output transport runs at the TTS rate, so the agent's audio is resampled once, to the client format. |
| `audio.out_10ms_chunks` | `2` | The output transport sends 20 ms chunks. |
| `tts.synthesis_mode` | `per_sentence` | The TTS service sends one request per sentence. |

Other `VADParams` keep Pipecat's defaults unless the settings file sets them. A missing or malformed settings file fails the session, and the application log shows an error that names the file.

### Realtime Session Behavior

Every OpenAI Realtime session resolves to the following effective values. `core/config/profiles/eval.yaml`, `core/config/voice_agent.yaml`, and `adapters/pipecat/settings/eval.yaml` are the sources of truth.

| Area | Value |
| --- | --- |
| Turn detection | `server_vad` with `threshold` 0.5, `prefix_padding_ms` 300, and `silence_duration_ms` 800. The server owns these values (`honor_client_values: false`), so `session.updated` echoes 800 ms whatever the client sends. Turns end on the Pipecat Silero VAD with an 800 ms stop. |
| Speech start | Pipecat's Silero VAD confirms speech after its default `start_secs` of 200 ms. Pipecat's VAD also decides when caller speech interrupts agent audio. |
| Barge-in during backend work | `frontend_verdict`. The frontend decides whether the new speech continues the running task. The verdict deadline is 4,000 ms, after which `cancel_and_merge` applies. A query equal to the running one counts as a continuation (`same_query_guard: true`). |
| Barge-in history | `truncate_heard`. The history keeps only the heard part of the reply and appends ` [interrupted by the user]`. |
| Backend history | On, with `include: full` and the default guidance. |
| Filler | Spoken (`speak`) when the backend is still busy after 300 ms. |
| Identifier normalization | The transcript hook writes spoken identifiers in lowercase written form for the agent. The wire keeps the raw ASR transcript. The tool-argument rule canonicalizes `get_user_details.user_id` to the pattern `^[a-z]+_[a-z]+_\d{4}$` and answers a malformed ID locally. The retry guard answers a repeat of a call that failed with `Error: ... not found` locally. |
| Speech while tools are out | Goes to the outstanding tool wait. Refer to [Tool Continuation Resume](#tool-continuation-resume). |
| Write gate | Off. Refer to [Write Gate](#write-gate). |
| Tools | Client-owned. The client declares tools in `session.update` and returns each function call output. A call with no output after 120 seconds receives a synthesized error result. |
| Instructions | Client `session.instructions` fill the backend prompt's domain-policy slot. |
| Greeting | Off. The session seeds its history with the client greeting `Hi! How can I help you today?`. |
| Audio | PCM16 at 24,000 Hz by default. Output audio deltas follow Pipecat's 20 ms transport chunks, and Pipecat's WebSocket output transport sends them at playback pace. |
| Voice | The catalog voice. A client-requested voice is ignored, and the application log shows a warning. |
| Admission | At most 8 concurrent Realtime sessions (`server.max_sessions`). |

### Environment Variables

The example reads the following environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | None | NVIDIA Inference Hub key for both LLMs. NVCF speech on the cloud platform also uses it. |
| `FBA_PROFILE` | `default` | Selects the served profile: `default` or `eval`. |
| `REALTIME_SERVICE_PLATFORM` | Unset | Selects the speech catalog: `cloud` reads `services.cloud.yaml`, `server` reads the `server` section of `services.local.yaml`, and `singlegpu` or unset keeps the `singlegpu` section. Compose sets it per recipe. |
| `FBA_VOICE_PROMPTS` | Empty | Replaces the prompt catalog path of the `eval` profile. |
| `FBA_ASR_SERVER` | Empty | Replaces the ASR catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_TTS_SERVER` | Empty | Replaces the TTS catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_VOICE_EVENT_LOG` | Empty | Path of a JSONL event log for Realtime sessions. Empty turns the log off. Browser UI sessions never write it. |
| `FBA_FILLER_LOG` | Empty | Path of a JSONL filler log for Realtime sessions. Filler timing is also recorded in the application log and, when it is set, the event log. |
| `WEATHERAPI_KEY`, `FINNHUB_API_KEY`, `PERPLEXITY_API_KEY` | Empty | Keys for the browser UI tools. Refer to [Tools](#tools). |
| `ENABLE_TRACING` | `false` | Traces browser UI sessions. Refer to [Tracing](#tracing). |
| `FBA_REALTIME_TRACING` | `false` | With `ENABLE_TRACING=true`, also traces Realtime sessions. Ignored when `FBA_PROFILE=eval`. |
| `FBA_REALTIME_TURN_MODES` | Unset (`server_vad` only) | Comma-separated turn modes that a Realtime client can select: `server_vad`, `manual`, and `semantic_vad`. `server_vad` is always allowed. Ignored when `FBA_PROFILE=eval`. Refer to [Realtime Turn Modes](#realtime-turn-modes). |
| `FBA_REALTIME_MCP` | `false` | `true` lets Realtime clients declare Model Context Protocol (MCP) tools. It also needs `REALTIME_MCP_ALLOWED_SERVER_URLS`. Ignored when `FBA_PROFILE=eval`. Refer to [MCP Tools](#mcp-tools). |
| `ENABLE_WELCOME_MESSAGE` | Unset (greets) | `false` turns off the browser UI greeting. Refer to [Welcome Message](#welcome-message). |
| `UVICORN_WS_PING_INTERVAL` | Unset (uvicorn uses 20 seconds) | Server WebSocket keepalive ping interval in seconds. `0` turns the ping off. Compose sets `0` for every recipe of this example. |
| `UVICORN_WS_PING_TIMEOUT` | Unset (uvicorn uses 20 seconds) | Seconds to wait for a pong before the server closes the socket. Compose sets `20` for every recipe of this example. |

Relative log paths resolve against the working directory. That directory is the repository root for a host-native run and `/app` in the container.

Both profiles set `server.ws_ping_interval_s: 0` and `server.ws_ping_timeout_s: 20`. A Realtime session compares these values with `UVICORN_WS_PING_INTERVAL` and `UVICORN_WS_PING_TIMEOUT` when it builds its runtime. If they differ, the server sends an `error` event with code `server_error`, closes the socket with code 1011, and logs the values to set. The keepalive variables apply to the whole server process, so they also change the keepalive for browser sessions and for every other example in that process.

### Change the LLM Endpoints

The frontend LLM comes from the `llm` catalog entry that the session selects: the **LLM** menu of the browser UI, or the Realtime route's selection. A Realtime session takes the backend LLM from the `thinker-llm` entry. A browser UI session has no backend control and uses the backend LLM of `core/config/agent.yaml`. The example uses only `model_id` and `base_url` from the catalog. Other request settings, such as `temperature`, `max_tokens`, and reasoning switches, come from `core/config/agent.yaml`.

With `FBA_PROFILE=eval`, a Realtime session must use the registry default LLMs. When an `ek_` client secret selects another `llm_id` or `thinker_llm_id`, the server sends an `error` event with code `server_error` and closes the socket with code 1011.

### Run the Frontend LLM Locally

By default, both LLMs run on the NVIDIA Inference Hub, and no LLM container starts for this example. The optional `docker/docker-compose.fba-local-llm.yaml` overlay also starts Nemotron 3.5 Lightning on the host:

```bash
docker compose -f docker-compose.yml -f docker/docker-compose.fba-local-llm.yaml \
  --profile frontend-backend-agent/single-gpu up -d
```

The overlay adds the Frontend/Backend Agent recipes to existing LLM services. It does not change any other recipe. The following table lists the local LLM for each recipe.

| Recipe profile | LLM service | Catalog entry in `services.local.yaml` |
| --- | --- | --- |
| `frontend-backend-agent/server` | `nvidia-llm` (Nemotron 3.5 Lightning NIM) | `nemotron-3.5-lightning-local` in the `server` section, at `http://nvidia-llm:8000/v1` |
| `frontend-backend-agent/single-gpu` | `nvidia-llm-vllm-lightning` (vLLM) | `nemotron-3.5-lightning-local` in the `singlegpu` section, at `http://nvidia-llm-vllm:8000/v1` |

The browser UI's **LLM** menu lists the local entry only while its endpoint is reachable. The Inference Hub entry stays first and remains the default. The local LLM can replace only the frontend LLM. The backend LLM stays the profile's Nemotron 3 Ultra on the Inference Hub. Pass the overlay file in every Compose command for the deployment, including `down`.

On `/single-gpu`, the vLLM service sizes its memory from the free GPU memory when it starts. It needs about 28 GiB after the `VLLM_VRAM_HEADROOM_MIB` headroom, which is 4 GiB by default. Evaluation runs do not use the local LLM. Refer to the eval refusal described in [Change the LLM Endpoints](#change-the-llm-endpoints).

### Change the Speech Endpoints

The example resolves ASR and TTS with its own catalog reader, without reachability probing. `REALTIME_SERVICE_PLATFORM` selects the catalog for both kinds of sessions, as [Environment Variables](#environment-variables) describes. In the browser UI, the **ASR** and **TTS** menus can select another entry of the example's catalogs. Set `FBA_ASR_SERVER` or `FBA_TTS_SERVER` to point at another Riva-compatible gRPC server. The catalog's `synthesis_mode` is ignored, because the example always synthesizes one sentence per request. The application log shows a warning when the Realtime route's ASR or TTS selection differs from the server that the example uses.

### Tracing

The example uses the repository's OpenTelemetry tracing. Refer to [Enable OpenTelemetry Tracing](../../../docs/how-to/enable-opentelemetry-tracing.md) for the collector setup, for example with `--profile tracing` and Phoenix. The following table lists which sessions are traced.

| Session | Traced when |
| --- | --- |
| Browser UI | `ENABLE_TRACING=true` |
| OpenAI Realtime | `ENABLE_TRACING=true` and `FBA_REALTIME_TRACING=true`, and only when `FBA_PROFILE` is not `eval` |

An eval-mode container ignores every `FBA_REALTIME_*` switch and logs a warning when one is turned on. The `docker/docker-compose.fba-eval.yaml` override also sets `FBA_REALTIME_TRACING=false`, `FBA_REALTIME_TURN_MODES=server_vad`, and `FBA_REALTIME_MCP=false`. `adapters/pipecat/switches.py` defines these rules.

### Welcome Message

A browser UI session greets the user with `Hi! How can I help you today?` when the browser is ready. Set `ENABLE_WELCOME_MESSAGE=false` to turn the greeting off. When the variable is unset or `true`, the greeting stays on. Realtime sessions never greet, whatever the variable says.

### Tool Continuation Resume

This behavior applies to Realtime sessions, whose client runs the tools. It is always on when automatic responses are on, that is, when `protocol.auto_response` and the session's `turn_detection.create_response` are both true. A caller turn that commits while client function calls are outstanding goes to that tool wait instead of starting a new turn. The wait resumes after its tool-call response is done and every function call output is in. Either a `response.create` or the caller's speech triggers the resume, following the OpenAI Realtime `create_response` rule.

The resumed backend request contains the tool messages and then one user message with the caller's words. Several caller turns are joined in order. Timing details are kept as message metadata and are never sent to a model. The frontend history records the caller's words after the delegation result, in the same turn group.

After a resume that caller speech triggers, the session consumes one later `response.create` as a no-op. This applies only while the resumed step still runs and no new caller turn has committed. Any other `response.create` follows the usual rules. It starts a response from pending input when the session is idle, or it marks an outstanding tool wait for resume. During an active response, it returns the `conversation_already_has_active_response` error.

### Write Gate

The write gate holds consequential client tool calls until the caller confirms the exact call. It is off in both profiles. It needs the frontend, because the frontend judges the caller's reply, and it needs client tools (`tools.source: client`). Turning it on without the frontend raises a configuration error. To try it, set `write_gate.enabled: true` in a profile. Because Realtime sessions use `eval.yaml`, a change to that file changes Tau behavior and needs a Tau A/B run.

The gate works as follows:

1. The backend's call to a non-read tool is held and does not reach the client. The generic read/write classifier decides which tools are reads. A tool that the session did not offer is classified by its name.
2. The backend receives an internal `confirmation_required` result. It receives `arguments_mismatch` when a call differs from the pending proposal, and `summary_too_long` when the generated summary exceeds `max_summary_chars`.
3. The agent speaks the model's short framing sentence, a summary generated from the canonical call arguments, and the question "Shall I go ahead?"
4. Only the first committed caller turn after the summary played to its end, with no barge-in, can confirm. On that turn, the frontend must call `call_backend` with a required `confirmation` value of `yes`, `partial`, `no`, or `unclear`. A missing call or an invalid value counts as `unclear`.
5. Only `yes` confirms. After that, only a re-issued call equal to the proposal goes to the client.

The gate checks consent and argument consistency. It does not check whether the policy allows the action. That decision stays with the model.

The following `write_gate` keys in `core/config/voice_agent.yaml` configure the gate.

| Key | Default | Purpose |
| --- | --- | --- |
| `enabled` | `false` | Turns the gate on. |
| `default` | `non_read` | `non_read` holds every tool that the classifier does not call a read. `none` holds only the tools in `include`. |
| `exempt` | `[]` | Tool names that are never held, for example a handoff tool. |
| `include` | `[]` | Tool names that are held even when they are classified as reads. |
| `tools` | `{}` | Per-tool settings: `not_consequential` lists argument paths left out of the summary, and `labels` maps argument paths to spoken labels. |
| `max_summary_chars` | `600` | Longest generated summary. A longer summary blocks the call instead of being truncated. |
| `max_unconfirmed_reissues` | `2` | Held re-issues in one caller turn before a confirmation. After that, the agent asks the caller again. |

## Browser UI Metrics and Voice

The browser UI shows per-turn metrics and lets you change the TTS voice during a session.

### Metrics Panel

After each answered turn, a browser UI session sends one RTVI `metric-group` message. The group is named `Turn N`, and its source is `frontend-backend-agent`. The group carries the following metrics when the turn records them:

- **Voice to voice**: the time from the end of user speech to the first agent audio (`user_stop_to_first_audio_ms`).
- **ASR final**: the time from the end of user speech to the final transcript.
- **Agent start**: the time from the final transcript to the start of the agent.
- **Agent to first audio**: the time from the start of the agent to the first agent audio.
- **Frontend LLM** and **Backend LLM**: the LLM latency of each role, with its input and output tokens.

The existing `user-bot-latency` messages still feed the user-to-bot latency view. `adapters/pipecat/ui_metrics.py` builds the groups. Realtime sessions do not send them.

### Voice Picker

The UI's voice picker sends a `set-voice` message. The session changes the TTS voice, and the language when the message carries one, from the next sentence on.

## Tools

Browser UI sessions use five tools that the server runs. `tools.yaml` holds their OpenAI function schemas, and the UI's read-only tools panel lists them. All five are always enabled. `tool_handlers.py` holds the handlers, copied from the generic domain of the former Frontend/Backend Agent example.

| Tool | Key | Behavior without the key |
| --- | --- | --- |
| `get_weather` | `WEATHERAPI_KEY` | Returns an "unavailable" result |
| `get_stock_price` | `FINNHUB_API_KEY` | Returns an "unavailable" result |
| `web_search` | `PERPLEXITY_API_KEY` | Returns an "unavailable" result |
| `calculate_bmi` | None | Always available |
| `generate_random_number` | None | Always available |

A handler also returns an "unavailable" result when the upstream service fails, so the agent can tell the user that it cannot do that. The optional `WEATHERAPI_BASE_URL`, `FINNHUB_BASE_URL`, `PERPLEXITY_BASE_URL`, and `PERPLEXITY_MODEL` variables change the upstream services. Store the keys in `.env`. Do not put them in prompts, catalogs, or source code.

OpenAI Realtime sessions do not use these tools. A Realtime client declares its own tools in `session.update` and runs them.

## Event Log

Set `FBA_VOICE_EVENT_LOG` to record one JSONL record per internal, voice, and timing event of a Realtime session. Each record contains `timestamp`, `kind`, and `session_id`, plus event fields. The following list shows some of the record kinds:

- `session_start`: the session's model, backend history settings, normalization summary, adapter, Pipecat settings summary, and prompt catalog with its SHA-256 hash.
- `barge_in_verdict`: the frontend verdict for speech during backend work.
- `backend_context`: the history settings and history size of each delegated backend request.
- `transcript_normalized`: an ASR transcript after identifier normalization.
- `filler_timing`: when filler was ready, whether it was spoken, and its outcome.
- `wait_input` and `wait_resumed`: a caller turn added to an outstanding tool wait, and the resume of that wait with its trigger.
- `response_create`: how a client `response.create` was handled, for example the rule `consumed_after_inbox_resume` or `resume_wait`.
- `mcp_call`: one server-run MCP tool call with its server, tool, outcome, and latency, or its cancellation. Refer to [MCP Tools](#mcp-tools).
- `write_proposed`, `write_presented`, `write_presentation`, `write_presentation_heard`, `write_confirmation`, `write_confirmed`, `write_invalidated`, `write_held`, `write_gate_error`, and `write_gate_config_problem`: the write gate's proposals, spoken summaries, confirmation verdicts, and problems.

The log can record transcripts and tool payloads. Treat it as sensitive data.

## Use the OpenAI Realtime API

The `realtime_models` section of [`examples_registry.yaml`](../../../examples_registry.yaml) exposes this example under the following identifiers.

| Field | Value |
| --- | --- |
| Realtime model ID | `nvidia/nemotron-realtime-frontend-backend` |
| Example ID and pipeline mode | `frontend-backend-agent` |
| Registry entry point | `examples.frontend_backend_agent.pipeline:bot` |
| Transport | OpenAI Realtime WebSocket (`WS /v1/realtime?model=nvidia/nemotron-realtime-frontend-backend`) |
| Registry prompt | `frontend` |

The gateway hands the whole Realtime session to this pipeline before it sends any event. The session implements the OpenAI Realtime protocol. Note the following behavior:

- The session accepts the GA schema only. A `session.update` with a beta field, such as `turn_detection` at the top level or the `pcm16` format string, is rejected as a whole. The error names each field and its GA replacement.
- By default, a `semantic_vad` request is approximated by silence-based `server_vad`. Eagerness `high` maps to 300 ms of silence, `medium` and `auto` map to 500 ms, and `low` maps to 800 ms. The echoed object carries `x_nvidia_effective` with the effective values. Turns end on silence, not on semantic completeness. To end turns with the Smart Turn model instead, refer to [Realtime Turn Modes](#realtime-turn-modes).
- By default, `input_audio_buffer.commit` is refused with error code `unsupported_event`, because turns end on the server voice activity detection (VAD). To allow manual turns, refer to [Realtime Turn Modes](#realtime-turn-modes).
- By default, the session accepts function tools only. A `type: "mcp"` tool is refused with error code `invalid_value`. To allow MCP tools, refer to [MCP Tools](#mcp-tools).
- A `response.create` with per-response `instructions`, `tools`, or `tool_choice` is refused with error code `unsupported_response_override`.
- Add `?x_nvidia_filler=1` to the WebSocket URL to receive unspoken filler as non-standard `x_nvidia.filler` events. The `eval` profile speaks its filler, so these events apply only to a profile with `filler.mode: log_only`.
- The first session warms up speech with one short ASR request and one short TTS request. Over the session cap, or when the speech warm-up fails, the server sends an `error` event with code `server_busy` and closes the socket with code 1013. The next session retries the warm-up.

### Realtime Turn Modes

By default, every Realtime turn ends on the server VAD. Set `FBA_REALTIME_TURN_MODES` to a comma-separated list to let clients select other modes through `audio.input.turn_detection` in `session.update`. The following table lists the modes.

| Mode | Client setting | Behavior |
| --- | --- | --- |
| `server_vad` | `type: "server_vad"`, the default | Turns end after the configured silence. This mode is always allowed. |
| `manual` | `turn_detection: null` | Only `input_audio_buffer.commit` ends the user turn. The VAD still detects speech, and the turn-stop watchdog is suspended. |
| `semantic_vad` | `type: "semantic_vad"` | The Smart Turn model that the browser UI uses ends turns, with a 0.2 s VAD stop instead of the silence approximation. |

For example, set `FBA_REALTIME_TURN_MODES=manual,semantic_vad` to allow both opt-in modes. Unknown names are ignored with a warning. When a client requests a mode that the variable does not allow, the session uses `server_vad`.

In `manual` mode, a commit with no turn in progress returns error code `input_audio_buffer_commit_empty`. With `turn_detection: null`, `create_response` is false, so the client sends `response.create` after the commit, as in OpenAI semantics. When `manual` is not allowed, a commit still returns `unsupported_event`.

When the variable allows more than `server_vad`, the `turn_detection` object in `session.updated` gains the non-standard `x_nvidia_turn_strategy` field, `server_vad` or `smart_turn`, which names the strategy in use. An eval-mode container ignores the variable and keeps `server_vad` only. `adapters/pipecat/turn_modes.py` implements the modes.

### MCP Tools

By default, a Realtime session accepts function tools only. Set `FBA_REALTIME_MCP=true` and `REALTIME_MCP_ALLOWED_SERVER_URLS`, a JSON array of exact server URLs that the shared gateway also uses, to let clients declare remote MCP servers. An eval-mode container ignores `FBA_REALTIME_MCP`. When the allowlist is empty, MCP stays off and the server logs a warning.

A client declares a server in `session.tools`, as the following example shows.

```json
{
  "type": "mcp",
  "server_label": "docs",
  "server_url": "https://mcp.example.com/mcp",
  "require_approval": "never"
}
```

The entry can also carry `allowed_tools`, `headers`, and `authorization`. The server lists the tools of each declared server and offers them to the agent. It runs every MCP call itself, and the client never sees the calls. The session hides their `function_call` items and repairs `previous_item_id`, `output_index`, and the `response.done` output. `session.updated` echoes the client's `mcp` entries.

The session handles MCP calls in the following ways:

- A response that holds only MCP calls ends with an empty output. The server then continues with the answer in a new response, without a client `response.create`.
- When a response mixes MCP calls and client function calls, the client sees and answers only its own calls, and then sends `response.create` as usual.
- A barge-in, that is, a new committed caller turn, or a `response.cancel` stops running MCP calls. The agent gets a `cancelled` result.
- The event log records each call as an `mcp_call` record.

The following table lists the errors that a `session.update` with MCP tools can return.

| Error code | Cause |
| --- | --- |
| `unsupported_capability` | `require_approval` is not `"never"`, or the entry uses `connector_id` or `tunnel_id`. Calls are hidden, so the client has nothing to approve. |
| `mcp_server_not_allowed` | `server_url` is not in `REALTIME_MCP_ALLOWED_SERVER_URLS`. |
| `mcp_server_not_defined` | A label-only entry refers to a server that this session has not defined. |
| `mcp_server_unavailable` | The server cannot be reached or does not list its tools in time. |
| `invalid_value` | A client function name starts with the reserved `mcp__` prefix. |

The shared limits `REALTIME_MCP_MAX_SERVERS`, `REALTIME_MCP_MAX_TOOLS_PER_SERVER`, `REALTIME_MCP_DISCOVERY_TIMEOUT_SECONDS`, `REALTIME_MCP_CALL_TIMEOUT_SECONDS`, and `REALTIME_MCP_MAX_OUTPUT_BYTES` apply. This example uses a 30-second discovery timeout by default. `adapters/pipecat/mcp_tools.py` implements the MCP tools.

### Client Secrets

An `ek_` client secret from `POST /v1/realtime/client_secrets` logs in on this model's route, as the `REALTIME_API_KEY` Bearer credential does. The session ignores the template's standard session fields, such as `instructions`, `tools`, and `turn_detection`. The client still sends its tools and instructions in `session.update`. In an eval-mode container, a secret whose template selects an LLM other than the registry default is refused. Refer to [Change the LLM Endpoints](#change-the-llm-endpoints).

### Evaluate With Tau

Run tau-bench evaluation through `voice-agent-evaluation` with `execution.backend: evaluator_realtime` and `product.profile: openai`. The `tau_native` lane cannot select this model ID and expects a 500 ms turn-detection echo, while this example echoes 800 ms. Deploy in eval mode, so the container serves only Realtime sessions and writes the event and filler logs. Every Realtime session uses the `eval` profile and the pinned Tau prompt catalog. Results record `profile=eval`. For the deployment steps, refer to [Run a Tau Evaluation](../../../docs/how-to/deploy-frontend-backend-agent.md#run-a-tau-evaluation).

## Test the Example

Run the example's unit tests:

```bash
uv run pytest tests/unit/frontend_backend_agent -q
```

The unit tests use fake speech and LLM services. They include the prompt-variant pin in `test_fba_prompt_variants.py`, the golden comparisons in `test_fba_golden.py`, and the browser UI session tests in `test_fba_ui_session.py`. A live session requires a speech service, an Inference Hub key, and, for the local recipes, a GPU.

## Tips and Best Practices

### Choose the Profile for the Job

Use `default` for the browser UI and for prompt tuning. Use `eval`, through the eval override, for evaluation runs, so browser sessions cannot share the deployment with an evaluation.

### Tune the Generic Catalog, Not the Tau Catalog

Edit `core/config/prompts/generic.yaml` and the example's `prompts.yaml` together. A change to `tau.yaml` or `eval.yaml` changes every Realtime session and needs a Tau A/B run.

### Keep the Keepalive Off

Keep `UVICORN_WS_PING_INTERVAL=0` for every deployment of this example, including host-native runs. Realtime sessions refuse to start when the keepalive is on.

## Limitations

This example has the following limitations:

- The repository's `REALTIME_API_KEY` Bearer authentication applies to Realtime sessions. An `ek_` client secret from `POST /v1/realtime/client_secrets` also logs in. The session ignores the template's standard session fields, such as `instructions`, `tools`, and `turn_detection`, and behaves as it does with the Bearer credential. Refer to [Client Secrets](#client-secrets).
- The repository's `/health` route applies. It does not report the example's speech warm-up or session counts.
- Speech warm-up runs on the first session of each kind instead of at server start.
- On a newly created container, Pipecat downloads the NLTK `punkt_tab` data while it sets up the first pipeline. Sessions that start during the download wait for it, and they fail if it takes longer than 120 seconds. Download the data once after you start the container, for example with `docker compose exec <service> .venv/bin/python -c "import nltk; nltk.download('punkt_tab')"`, or keep `/root/nltk_data` on a volume.
- The browser UI shows the transcripts of both sides, the user-to-bot latency, and the per-turn metrics group, but not streamed LLM text, because the agent is not a Pipecat LLM service.
- The browser UI has no control for the backend LLM or for the tool set. The **LLM** menu selects only the frontend LLM.
- A refused browser UI session, for example on an `eval` container, shows no error in the UI. The app log shows the reason.
- Manual turn mode and Smart Turn `semantic_vad` are opt-in for Realtime sessions and are never available with `FBA_PROFILE=eval`. Without `FBA_REALTIME_TURN_MODES`, `input_audio_buffer.commit` returns the `unsupported_event` error, and `semantic_vad` is approximated by silence-based turn detection.
- MCP tools are opt-in for Realtime sessions and are never available with `FBA_PROFILE=eval`. They require `require_approval: "never"`, because the server runs the calls and hides them from the client. OpenAI connectors (`connector_id`) and tunnels (`tunnel_id`) are not available.
- A Realtime session cannot select its own ASR or TTS service. The speech endpoints follow `REALTIME_SERVICE_PLATFORM` and the `FBA_ASR_SERVER` and `FBA_TTS_SERVER` overrides.
- A Realtime client's requested voice is ignored. The TTS service uses the catalog voice and logs a warning.
- No served profile uses the agent's `backend_only` mode (`agent.mode` in `core/config/agent.yaml`). The unit tests cover that mode with a test profile.
- The `eval` profile is available only on the `/single-gpu` recipe.
