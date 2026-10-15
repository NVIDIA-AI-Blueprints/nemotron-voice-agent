# Frontend/Backend Verdict Agent Example

The Frontend/Backend Verdict Agent serves a voice Frontend/Backend Agent on the repository's OpenAI Realtime WebSocket endpoint. A fast frontend large language model (LLM) talks to the user and decides whether to answer or delegate. A reasoning backend LLM does the delegated work with the tools that the Realtime client defines.

This example adds the following behaviors to the cascaded voice loop:

- A frontend barge-in verdict decides whether speech during backend work continues the running task or starts a new one.
- The backend receives the full conversation history with each delegated request.
- Identifier normalization writes spoken identifiers, such as user IDs, in their written form before the agent sees them.
- Caller speech while client tool calls are outstanding resumes the tool continuation with the caller's words.
- An optional write gate holds every consequential tool call until the caller confirms a spoken summary of it.

The agent logic lives in `core/`. The `adapters/pipecat/` package runs that logic on the repository's shared Pipecat speech stack. This example is separate from the [Frontend/Backend Agent](../frontend_backend_agent/README.md) example and does not import its Talker/Thinker planner.

To deploy this example on a single-GPU host, follow [Deploy the Frontend/Backend Verdict Agent](../../../docs/how-to/deploy-frontend-backend-verdict-agent.md).
For evaluation runs, such as Tau, deploy it in [eval mode](../../../docs/how-to/deploy-frontend-backend-verdict-agent.md#deploy-in-eval-mode), which serves plain `ws://` and writes the agent's event and filler logs to `logs/` on the host.

## Realtime Model

The `realtime_models` section of [`examples_registry.yaml`](../../../examples_registry.yaml) exposes this example under the following identifiers.

| Field | Value |
| --- | --- |
| Realtime model ID | `nvidia/nemotron-realtime-frontend-backend-verdict` |
| Example ID and pipeline mode | `frontend-backend-verdict-agent` |
| Registry entry point | `examples.frontend_backend_verdict.pipeline:bot` |
| Transport | OpenAI Realtime WebSocket only (`WS /v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict`) |
| Registry prompt | `frontend` |

The bot refuses WebRTC, RTVI, and the Pipecat WebSocket transport. If you select this example in the browser UI, the bot raises an error that names the Realtime model, and the session does not start.

## Architecture

The gateway hands the whole Realtime session to this pipeline before it sends any event. The following diagram shows the request path.

```text
Realtime client
  |  WS /v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict
  v
Realtime gateway (src/realtime/gateway.py)
  - authenticates the REALTIME_API_KEY Bearer credential
  - resolves ?model= to frontend-backend-verdict-agent
  - hands the socket to the pipeline before it sends any event
  v
Pipecat pipeline (adapters/pipecat/pipeline.py)
  FastAPIWebsocketTransport.input()   client events, input audio decoded to 16 kHz PCM16
    -> VADProcessor (Silero)          shared Pipecat VAD
    -> InputCursor                    input-audio clock, speech onset and offset
    -> NvidiaForceEouSTTService       shared streaming ASR (src/examples/shared/)
    -> UserTurnProcessor              Pipecat user turns
    -> FbvTurnBridge                  drives the core turn manager:
                                        frontend LLM -> backend LLM -> client function calls,
                                        barge-in verdict, filler, identifier normalization
    -> NvidiaTTSService               Pipecat TTS, one request per sentence
    -> OutputTagger                   transcript and audio order, playout progress
  FastAPIWebsocketTransport.output()  server events and output audio deltas
```

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
| `core/config/` | Configuration loader, `voice_agent.yaml`, `agent.yaml`, the behavior profiles (`profiles/`), and the prompt variants (`prompts/`). `profiles.py` validates `FBV_PROFILE` and the WebSocket keepalive. |
| `adapters/pipecat/` | Pipecat host: pipeline, session, serializer, turn bridge, speech warm-up, the session cap, and the shared speech services. |
| `adapters/pipecat/settings/eval.yaml` | Pipecat VAD, user-turn, transport-audio, and TTS settings. Refer to [Pipecat Voice Settings](#pipecat-voice-settings). |
| `prompts.yaml` | Registry-facing prompt catalog. It mirrors `core/config/prompts/generic.yaml`. |
| `services.cloud.yaml`, `services.local.yaml` | Example-local service catalogs. |

The example has no command-line client, browser page, or standalone server of its own. Use a Realtime client against the repository server.

## Default Models

The LLM endpoints come from the example's service catalogs. The `llm` entry is the frontend, and the `thinker-llm` entry is the backend. Every other request parameter comes from `core/config/agent.yaml`. The ASR and TTS endpoints come from the example's speech loader, which reads the `singlegpu` section of `services.local.yaml`.

| Role | Model | Endpoint | Request settings |
| --- | --- | --- | --- |
| Frontend LLM | `nvidia/nvidia/nemotron-3.5-lightning` | NVIDIA Inference Hub, `https://inference-api.nvidia.com/v1` | Reasoning off, `max_tokens` 1,024, `temperature` 0 |
| Backend LLM | `nvidia/nvidia/nemotron-3-ultra` | NVIDIA Inference Hub, `https://inference-api.nvidia.com/v1` | Reasoning on, `reasoning_budget` 1,024, `max_tokens` 4,096, `temperature` 0 |
| ASR | `nemotron-speech-streaming-en-0.6b` | NeMo-Speech.cpp, `nemo-speech:50051` | Shared `NvidiaForceEouSTTService` |
| TTS | Magpie TTS Multilingual, voice `John` | NeMo-Speech.cpp, `nemo-speech:50051` | Pipecat `NvidiaTTSService`, one request per sentence at 22,050 Hz, resampled to the client format |

Both LLMs use `NVIDIA_API_KEY`. For this example, set it to an NVIDIA Inference Hub key. The doubled `nvidia/nvidia/` prefix in the model IDs is the Inference Hub naming and is intentional.

## Run the Example

Run commands from the repository root. This example has one Compose recipe, `frontend-backend-verdict-agent/single-gpu`. It has no cloud or server recipe. For the full procedure, including credentials, profile selection, verification, and troubleshooting, refer to [Deploy the Frontend/Backend Verdict Agent](../../../docs/how-to/deploy-frontend-backend-verdict-agent.md).

### Configure Credentials

Preserve an existing `.env`. Otherwise, create it from the template:

```bash
test -f .env || cp .env.example .env
```

Set the following keys in `.env`:

- `NVIDIA_API_KEY`: an NVIDIA Inference Hub key for both LLMs.
- `HF_TOKEN`: a Hugging Face token for the NeMo-Speech.cpp model download.
- `REALTIME_API_KEY`: the Bearer credential that Realtime clients send. Leave it unset only for unauthenticated local development.

### Run With Docker Compose

Download the NeMo-Speech.cpp weights once, as your user:

```bash
bash scripts/download-nemo-speech-models.sh
```

Start the recipe:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu up -d
```

The recipe starts the following services.

| Recipe profile | App service | Sidecars |
| --- | --- | --- |
| `frontend-backend-verdict-agent/single-gpu` | `frontend-backend-verdict-agent-single-gpu` | `nemo-speech` (ASR and TTS) |

The app service sets `EXAMPLE_SELECTION=frontend-backend-verdict-agent`, `REALTIME_SERVICE_PLATFORM=singlegpu`, `FBV_PROFILE=tau3_eval_frontend_verdict_speak_history`, `FBV_PIPECAT_SETTINGS=eval.yaml`, `UVICORN_WS_PING_INTERVAL=0`, and `UVICORN_WS_PING_TIMEOUT=20`. You can override `EXAMPLE_SELECTION`, `FBV_PROFILE`, `FBV_PIPECAT_SETTINGS`, and both keepalive values from `.env` or the shell.

Connect a Realtime client to `wss://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict`. Use `ws://` when `PIPELINE_TLS=false`.

Tear down with the same profile:

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu down
```

### Run Host-Native

Start only the speech sidecar. It publishes port `50051` on the host.

```bash
docker compose --profile frontend-backend-verdict-agent/single-gpu up -d nemo-speech
```

Then start the server and point the example's speech loader at the host port:

```bash
EXAMPLE_SELECTION=frontend-backend-verdict-agent \
REALTIME_SERVICE_PLATFORM=singlegpu \
FBA_ASR_SERVER=localhost:50051 \
FBA_TTS_SERVER=localhost:50051 \
UVICORN_WS_PING_INTERVAL=0 \
UVICORN_WS_PING_TIMEOUT=20 \
PIPELINE_TLS=false \
uv run python src/server.py --host 0.0.0.0 --port 7860
```

`PIPELINE_TLS=false` serves plain HTTP and WebSocket for an isolated local test. Remove it to keep TLS enabled. You can also start `src/realtime_server.py`, which serves only the Realtime API, with the same environment.

## Configure the Example

### Behavior Profiles

The `FBV_PROFILE` environment variable selects one profile from `core/config/profiles/`. When `FBV_PROFILE` is unset, the example uses `tau3_eval_frontend_verdict_speak_history`, and Compose sets the same default. Each `tau3_eval*` profile extends `tau3_eval.yaml`, and `tau3_eval.yaml` extends `core/config/voice_agent.yaml`.

The following table lists the served profiles and what each one changes.

| Profile | Prompt catalog | Changes |
| --- | --- | --- |
| `tau3_eval` | `tau.yaml` | Base evaluation arm. It pins server-owned turn detection with 800 ms silence, turns off the server WebSocket keepalive ping, turns off backend conversation history, logs filler without speaking it, cancels and merges on barge-in during backend work, and turns on identifier normalization. |
| `tau3_eval_normalization` | `tau.yaml` | Same as `tau3_eval`. The profile remains for existing commands. |
| `tau3_eval_backend_history` | `tau.yaml` | `tau3_eval` plus backend conversation history with `include: full`. |
| `tau3_eval_backend_history_noguide` | `tau.yaml` | `tau3_eval_backend_history` without the behavioral guidance (`guidance_key: ""`). The context note remains. |
| `tau3_eval_frontend_verdict` | `tau.yaml` | `tau3_eval` plus `barge_in.while_thinking: frontend_verdict`. |
| `tau3_eval_frontend_verdict_speak` | `tau.yaml` | `tau3_eval_frontend_verdict` plus spoken filler (`filler.mode: speak`). |
| `tau3_eval_frontend_verdict_speak_history` | `tau.yaml` | `tau3_eval_frontend_verdict_speak` plus backend conversation history with `include: full`. This is the default Tau arm. |
| `tau3_eval_frontend_verdict_speak_history_write_gate` | `tau.yaml` | `tau3_eval_frontend_verdict_speak_history` plus the write gate, with `transfer_to_human_agents` exempt. It also overrides the backend history guidance prompt. The added guidance tells the backend to re-check the policy before a call that changes data and how to handle a held call. |
| `default` | `generic.yaml` | `tau3_eval_frontend_verdict_speak_history` with the generic prompt catalog. The agent, turn taking, and Realtime settings are the same as the default Tau arm. |

An unknown profile name fails the session with an error that lists the served profiles. The `SERVED_PROFILES` tuple in `core/config/profiles.py` defines the list.

### Prompt Variants

The profile selects the prompt catalog. The example ships two variants in `core/config/prompts/`.

| Variant | File | Used by | Change rule |
| --- | --- | --- | --- |
| Tau | `core/config/prompts/tau.yaml` | Every `tau3_eval*` profile | Pinned by SHA-256 in `tests/unit/frontend_backend_verdict/test_fbv_prompt_variants.py`. A change requires a new pin and an attached Tau A/B evaluation run. |
| Generic | `core/config/prompts/generic.yaml` | `default` | Free to tune. It currently has the same content as `tau.yaml`. |

The root `prompts.yaml` mirrors `generic.yaml`, and a unit test checks that the two files are identical. To switch variants, change `FBV_PROFILE`. Do not edit the prompt files to switch.

The `FBA_VOICE_PROMPTS` environment variable replaces the catalog path for the profiles that do not set their own path, that is, every `tau3_eval*` profile. The `default` profile sets `generic.yaml` itself. A relative path resolves against `core/config/`. The example loads its configuration when the first session connects. A missing or invalid catalog then fails the session, and the application log shows an error that names `agent.overrides.prompts.path`.

The example becomes ready when the first session connects. The application log then records the profile in the `frontend/backend verdict agent ready` line and the catalog in a line of the following form:

```text
prompts <catalog path> (sha256 <hash>)
```

The `session_start` record of the event log carries `prompts` with the catalog file name (`catalog`) and its SHA-256 hash (`sha256`), `adapter: "pipecat"`, and the Pipecat settings summary (`pipecat`).

### Pipecat Voice Settings

The `FBV_PIPECAT_SETTINGS` environment variable selects the Pipecat settings file. The value is an absolute path or a file name relative to `adapters/pipecat/settings/`. The default is `eval.yaml`, which holds the Tau evaluation settings. The comments in that file give the reason for each value.

The following table lists the values in `eval.yaml`.

| Key | Value | Effect |
| --- | --- | --- |
| `vad.stop_secs` | `0.8` | Silero counts 800 ms of silence in audio, matching the profiles' server-owned silence. |
| `vad.audio_idle_timeout` | `0` | Turns off the wall-clock forced speech stop, so a client that pauses its input does not end a turn. |
| `turn.user_speech_timeout` | `0` | Adds no wall-clock wait after the VAD stop. |
| `turn.stt_ttfs_p99` | `3.0` | After the VAD stop, the turn waits up to `3.0 - stop_secs` seconds for the final transcript. The final transcript ends the wait as soon as it arrives. |
| `turn.wait_for_transcript` | `false` | A turn with no transcript, such as noise, ends after that wait with an empty transcript. |
| `audio.out_sample_rate` | `null` | The output transport runs at the TTS rate, so the agent's audio is resampled once, to the client format. |
| `audio.out_10ms_chunks` | `2` | The output transport sends 20 ms chunks. |
| `tts.synthesis_mode` | `per_sentence` | The TTS service sends one request per sentence. |

The session applies the effective `silence_duration_ms` of its turn detection as the VAD `stop_secs`. Other `VADParams` keep Pipecat's defaults unless the settings file sets them. A missing or malformed settings file fails the session, and the application log shows an error that names the file.

### Default Behavior

The default profile resolves to the following effective values. The profile files, `core/config/voice_agent.yaml`, and `adapters/pipecat/settings/eval.yaml` are the sources of truth.

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
| Admission | At most 8 concurrent sessions (`server.max_sessions`). |

### Environment Variables

The example reads the following environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | None | NVIDIA Inference Hub key for both LLMs. |
| `FBV_PROFILE` | `tau3_eval_frontend_verdict_speak_history` | Selects the served behavior profile and, with it, the prompt variant. |
| `FBV_PIPECAT_SETTINGS` | `eval.yaml` | Selects the Pipecat settings file, as an absolute path or a file name in `adapters/pipecat/settings/`. |
| `FBA_VOICE_PROMPTS` | Empty | Replaces the prompt catalog path of the `tau3_eval*` profiles. |
| `FBA_ASR_SERVER` | Empty | Replaces the ASR catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_TTS_SERVER` | Empty | Replaces the TTS catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_VOICE_EVENT_LOG` | Empty | Path of a JSONL event log. Empty turns the log off. |
| `FBA_FILLER_LOG` | Empty | Path of a JSONL filler log. Filler timing is also recorded in the application log and, when it is set, the event log. |
| `UVICORN_WS_PING_INTERVAL` | Unset (uvicorn uses 20 seconds) | Server WebSocket keepalive ping interval in seconds. `0` turns the ping off. Compose sets `0` for this recipe. |
| `UVICORN_WS_PING_TIMEOUT` | Unset (uvicorn uses 20 seconds) | Seconds to wait for a pong before the server closes the socket. Compose sets `20` for this recipe. |

Relative log paths resolve against the working directory. That directory is the repository root for a host-native run and `/app` in the container.

The served profiles set `server.ws_ping_interval_s: 0` and `server.ws_ping_timeout_s: 20`. The example compares these values with `UVICORN_WS_PING_INTERVAL` and `UVICORN_WS_PING_TIMEOUT` when it builds a profile's runtime. If they differ, the server sends an `error` event with code `server_error`, closes the socket with code 1011, and logs the values to set. The keepalive variables apply to the whole server process, so they also change the keepalive for every other example in that process.

### Tool Continuation Resume

This behavior is always on when automatic responses are on, that is, when `protocol.auto_response` and the session's `turn_detection.create_response` are both true. A caller turn that commits while client function calls are outstanding goes to that tool wait instead of starting a new turn. The wait resumes after its tool-call response is done and every function call output is in. Either a `response.create` or the caller's speech triggers the resume, following the OpenAI Realtime `create_response` rule.

The resumed backend request contains the tool messages and then one user message with the caller's words. Several caller turns are joined in order. Timing details are kept as message metadata and are never sent to a model. The frontend history records the caller's words after the delegation result, in the same turn group.

After a resume that caller speech triggers, the session consumes one later `response.create` as a no-op. This applies only while the resumed step still runs and no new caller turn has committed. Any other `response.create` follows the usual rules. It starts a response from pending input when the session is idle, or it marks an outstanding tool wait for resume. During an active response, it returns the `conversation_already_has_active_response` error.

### Write Gate

The write gate holds consequential tool calls until the caller confirms the exact call. It is off by default. Turn it on with the `tau3_eval_frontend_verdict_speak_history_write_gate` profile, or set `write_gate.enabled: true` in a profile. The gate needs the frontend, because the frontend judges the caller's reply. Turning it on without the frontend raises a configuration error.

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

### Change the LLM Endpoints

To change an LLM model or base URL, edit the `llm` or `thinker-llm` entry in the catalog section that `REALTIME_SERVICE_PLATFORM` selects. The example uses only `model_id` and `base_url` from the catalog. Other request settings, such as `temperature`, `max_tokens`, and reasoning switches, come from `core/config/agent.yaml`.

### Change the Speech Endpoints

The example's speech loader resolves ASR and TTS from the `singlegpu` section of `services.local.yaml`. Set `FBA_ASR_SERVER` or `FBA_TTS_SERVER` to point at another Riva-compatible gRPC server. The application log shows a warning when the Realtime route's ASR or TTS selection differs from the server that the speech loader uses.

## Event Log

Set `FBA_VOICE_EVENT_LOG` to record one JSONL record per internal, voice, and timing event. Each record contains `timestamp`, `kind`, and `session_id`, plus event fields. The following list shows some of the record kinds:

- `session_start`: the session's model, backend history settings, normalization summary, adapter, Pipecat settings summary, and prompt catalog with its SHA-256 hash.
- `barge_in_verdict`: the frontend verdict for speech during backend work.
- `backend_context`: the history settings and history size of each delegated backend request.
- `transcript_normalized`: an ASR transcript after identifier normalization.
- `filler_timing`: when filler was ready, whether it was spoken, and its outcome.
- `wait_input` and `wait_resumed`: a caller turn added to an outstanding tool wait, and the resume of that wait with its trigger.
- `response_create`: how a client `response.create` was handled, for example the rule `consumed_after_inbox_resume` or `resume_wait`.
- `write_proposed`, `write_presented`, `write_presentation`, `write_presentation_heard`, `write_confirmation`, `write_confirmed`, `write_invalidated`, `write_held`, `write_gate_error`, and `write_gate_config_problem`: the write gate's proposals, spoken summaries, confirmation verdicts, and problems.

The log can record transcripts and tool payloads. Treat it as sensitive data.

## Protocol Notes

The session implements the OpenAI Realtime protocol. Note the following behavior:

- The session accepts the GA schema only. A `session.update` with a beta field, such as `turn_detection` at the top level or the `pcm16` format string, is rejected as a whole. The error names each field and its GA replacement.
- A `semantic_vad` request is approximated by silence-based `server_vad`. Eagerness `high` maps to 300 ms of silence, `medium` and `auto` map to 500 ms, and `low` maps to 800 ms. The echoed object carries `x_nvidia_effective` with the effective values. Turns end on silence, not on semantic completeness.
- `input_audio_buffer.commit` is refused with error code `unsupported_event`. Turns end on the server VAD, so manual turn mode is not available.
- A `response.create` with per-response `instructions`, `tools`, or `tool_choice` is refused with error code `unsupported_response_override`.
- Add `?x_nvidia_filler=1` to the WebSocket URL to receive unspoken filler as non-standard `x_nvidia.filler` events. The default profile speaks its filler, so these events apply to profiles with `filler.mode: log_only`.
- The first session warms up speech with one short ASR request and one short TTS request. Over the session cap, or when the speech warm-up fails, the server sends an `error` event with code `server_busy` and closes the socket with code 1013. The next session retries the warm-up.

## Test the Example

Run the example's unit tests:

```bash
uv run pytest tests/unit/frontend_backend_verdict -q
```

The unit tests use fake speech and LLM services. They include the prompt-variant pin in `test_fbv_prompt_variants.py` and the golden comparisons in `test_fbv_golden.py`. A live session requires the NeMo-Speech.cpp sidecar, a GPU, and an Inference Hub key.

## Evaluate With Tau

Run tau-bench evaluation through `voice-agent-evaluation` with `execution.backend: evaluator_realtime` and `product.profile: openai`. The `tau_native` lane cannot select this model ID and expects a 500 ms turn-detection echo, while this example echoes 800 ms. Use a `tau3_eval*` profile, so the run uses the pinned Tau prompt catalog. For the deployment steps, refer to [Run a Tau Evaluation](../../../docs/how-to/deploy-frontend-backend-verdict-agent.md#run-a-tau-evaluation).

## Limitations

This example has the following limitations:

- The repository's `REALTIME_API_KEY` Bearer authentication applies.
- `ek_` client-secret session templates are not applied for this model, because the pipeline owns the session.
- The repository's `/health` route applies. It does not report the example's speech warm-up or session count.
- Speech warm-up runs on the first session instead of at server start.
- On a newly created container, Pipecat downloads the NLTK `punkt_tab` data while it sets up the first pipeline. Sessions that start during the download wait for it, and they fail if it takes longer than 120 seconds. Download the data once after you start the container, for example with `docker compose exec <service> .venv/bin/python -c "import nltk; nltk.download('punkt_tab')"`, or keep `/root/nltk_data` on a volume.
- Manual turn mode is not available. `input_audio_buffer.commit` returns the `unsupported_event` error.
- The client's requested voice is ignored. The TTS service uses the catalog voice and logs a warning.
- No served profile uses the agent's `backend_only` mode (`agent.mode` in `core/config/agent.yaml`). The unit tests cover that mode with a test profile.
- The example supports only the OpenAI Realtime WebSocket. WebRTC, RTVI, and the browser UI are not supported.
- The example has no cloud or server Compose recipe. The speech loader does not read `services.cloud.yaml`.
