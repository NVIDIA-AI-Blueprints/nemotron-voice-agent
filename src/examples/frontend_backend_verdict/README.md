# Frontend/Backend Verdict Agent Example

The Frontend/Backend Verdict Agent serves the voice Frontend/Backend Agent prototype on the repository's OpenAI Realtime WebSocket endpoint. A fast frontend large language model (LLM) talks to the user and decides whether to answer or delegate. A reasoning backend LLM does the delegated work with the tools that the Realtime client defines.

This example adds the following behaviors to the cascaded voice loop:

- A frontend barge-in verdict decides whether speech during backend work continues the running task or starts a new one.
- The backend receives the full conversation history with each delegated request.
- Identifier normalization writes spoken identifiers, such as user IDs, in their written form before the agent sees them.
- Caller speech while client tool calls are outstanding resumes the tool continuation with the caller's words.
- An optional write gate holds every consequential tool call until the caller confirms a spoken summary of it.

The behavior code is a copy of the prototype from the `nemotron-voice-agent-smasurekar` fork at commit `66f6c67`, with the local changes that `provenance.json` records. Refer to [Provenance](#provenance). Pipecat only hosts the prototype session on the socket that the Realtime gateway authenticated. This example is separate from the [Frontend/Backend Agent](../frontend_backend_agent/README.md) example and does not import its Talker/Thinker planner.

## Realtime Model

The `realtime_models` section of [`examples_registry.yaml`](../../../examples_registry.yaml) exposes this example under the following identifiers.

| Field | Value |
| --- | --- |
| Realtime model ID | `nvidia/nemotron-realtime-frontend-backend-verdict` |
| Example ID and pipeline mode | `frontend-backend-verdict-agent` |
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
Pipecat pipeline (pipeline.py)
  FastAPIWebsocketTransport.input()      client events, in arrival order
    -> PrototypeSessionProcessor         bridge/: runs the prototype RealtimeSession
         wire            GA event parsing, session.created, session.updated
         input path      Silero VAD + prototype segmenter
                         -> one Riva ASR stream per utterance
                         -> transcript identifier normalization
         turn manager    turns, barge-in, frontend verdict, filler
         text agent      frontend LLM -> backend LLM -> client function calls
                         -> tool-argument normalization
         output path     Riva TTS, one request per sentence
  FastAPIWebsocketTransport.output()     server events, in the session's order
```

The example has the following parts.

| Path | Role |
| --- | --- |
| `pipeline.py` | Pipecat entry point. It rejects non-Realtime transports, applies session admission, and builds the pipeline. |
| `bridge/runtime.py` | Process-wide state for one profile: event log, speech services, LLM clients, speech warm-up, and the session cap. It also validates `FBV_PROFILE` and the WebSocket keepalive. |
| `bridge/session_processor.py` | Pipecat processor that runs one prototype `RealtimeSession` per connection. |
| `bridge/wire.py` | Pass-through serializer. The prototype session parses and builds every Realtime event itself. |
| `text/` | Text Frontend/Backend Agent copied from the prototype: frontend, backend, delegation, conversation history, and prompts. |
| `voice/` | Voice layer copied from the prototype: wire protocol, engine, speech adapters, normalization, and configuration profiles. It also contains the example's own write gate (`voice/agent/write_gate.py`). |
| `prompts.yaml` | Registry-facing copy of the prototype voice prompt catalog (`voice/config/prompts.voice.yaml`). |
| `services.cloud.yaml`, `services.local.yaml` | Example-local service catalogs. |
| `provenance.json` | SHA-256 hash and source path of every copied prototype file, and a `diverged` record for every file that the example changes or adds. |

## Default Models

The LLM endpoints come from the example's service catalogs. The `llm` entry is the frontend, and the `thinker-llm` entry is the backend. Every request parameter comes from the prototype's `text/config/agent.yaml`. The ASR and TTS endpoints come from the prototype speech loader, which reads the `singlegpu` section of `services.local.yaml`.

| Role | Model | Endpoint | Request settings |
| --- | --- | --- | --- |
| Frontend LLM | `nvidia/nvidia/nemotron-3.5-lightning` | NVIDIA Inference Hub, `https://inference-api.nvidia.com/v1` | Reasoning off, `max_tokens` 1,024, `temperature` 0 |
| Backend LLM | `nvidia/nvidia/nemotron-3-ultra` | NVIDIA Inference Hub, `https://inference-api.nvidia.com/v1` | Reasoning on, `reasoning_budget` 1,024, `max_tokens` 4,096, `temperature` 0 |
| ASR | `nemotron-speech-streaming-en-0.6b` | NeMo-Speech.cpp, `nemo-speech:50051` | Interim results on |
| TTS | Magpie TTS Multilingual, voice `John` | NeMo-Speech.cpp, `nemo-speech:50051` | One `SynthesizeOnline` request per sentence at 22,050 Hz, resampled to the client format |

Both LLMs use `NVIDIA_API_KEY`. For this example, set it to an NVIDIA Inference Hub key. The doubled `nvidia/nvidia/` prefix in the model IDs is the Inference Hub naming and is intentional.

## Run the Example

Run commands from the repository root. This example has one Compose recipe. It has no cloud or server recipe.

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

The app service sets `EXAMPLE_SELECTION=frontend-backend-verdict-agent`, `REALTIME_SERVICE_PLATFORM=singlegpu`, `FBV_PROFILE=tau3_eval_frontend_verdict_speak_history`, `UVICORN_WS_PING_INTERVAL=0`, and `UVICORN_WS_PING_TIMEOUT=20`. You can override `EXAMPLE_SELECTION`, `FBV_PROFILE`, and both keepalive values from `.env` or the shell.

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

Then start the server and point the prototype speech loader at the host port:

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

The `FBV_PROFILE` environment variable selects one prototype profile from `voice/config/profiles/`. The default is `tau3_eval_frontend_verdict_speak_history`. Each served profile extends `tau3_eval.yaml`, and `tau3_eval.yaml` extends `voice/config/voice_agent.yaml`.

The following table lists the served profiles and what each one changes.

| Profile | Changes |
| --- | --- |
| `tau3_eval` | Base evaluation arm. It pins server-owned turn detection with 800 ms silence, turns off the server WebSocket keepalive ping, turns off backend conversation history, logs filler without speaking it, cancels and merges on barge-in during backend work, and turns on identifier normalization. |
| `tau3_eval_normalization` | Same as `tau3_eval`. The profile remains for existing commands. |
| `tau3_eval_backend_history` | `tau3_eval` plus backend conversation history with `include: full`. |
| `tau3_eval_backend_history_noguide` | `tau3_eval_backend_history` without the behavioral guidance (`guidance_key: ""`). The context note remains. |
| `tau3_eval_frontend_verdict` | `tau3_eval` plus `barge_in.while_thinking: frontend_verdict`. |
| `tau3_eval_frontend_verdict_speak` | `tau3_eval_frontend_verdict` plus spoken filler (`filler.mode: speak`). |
| `tau3_eval_frontend_verdict_speak_history` | `tau3_eval_frontend_verdict_speak` plus backend conversation history with `include: full`. This is the default. |
| `tau3_eval_frontend_verdict_speak_history_write_gate` | `tau3_eval_frontend_verdict_speak_history` plus the write gate, with `transfer_to_human_agents` exempt. It also overrides the backend history guidance prompt. The added guidance tells the backend to re-check the policy before a call that changes data and how to handle a held call. |

An unknown profile name raises an error that lists the served profiles. The `SHIPPED_PROFILES` tuple in `bridge/runtime.py` defines the list.

### Default Behavior

The default profile resolves to the following effective values. The profile files and `voice/config/voice_agent.yaml` are the sources of truth.

| Area | Value |
| --- | --- |
| Turn detection | `server_vad` with Silero VAD, `threshold` 0.5, `prefix_padding_ms` 300, and `silence_duration_ms` 800. The server owns these values (`honor_client_values: false`), so `session.updated` echoes 800 ms whatever the client sends. |
| Minimum speech | Bursts shorter than 120 ms are discarded as noise. They start no turn and no barge-in. |
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
| Audio | PCM16 at 24,000 Hz by default, with 100 ms output audio deltas, sent as synthesized without real-time pacing. |
| Admission | At most 8 concurrent sessions (`server.max_sessions`). |

### Environment Variables

The example reads the following environment variables.

| Variable | Default | Purpose |
| --- | --- | --- |
| `NVIDIA_API_KEY` | None | NVIDIA Inference Hub key for both LLMs. |
| `FBV_PROFILE` | `tau3_eval_frontend_verdict_speak_history` | Selects the served behavior profile. |
| `FBA_ASR_SERVER` | Empty | Replaces the ASR catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_TTS_SERVER` | Empty | Replaces the TTS catalog server, for example `localhost:50051` for a host-native run. |
| `FBA_VOICE_EVENT_LOG` | Empty | Path of a JSONL event log. Empty turns the log off. |
| `FBA_FILLER_LOG` | Empty | Path of a JSONL filler log. Filler timing is also recorded in the application log and, when it is set, the event log. |
| `UVICORN_WS_PING_INTERVAL` | Unset (uvicorn uses 20 seconds) | Server WebSocket keepalive ping interval in seconds. `0` turns the ping off. Compose sets `0` for this recipe. |
| `UVICORN_WS_PING_TIMEOUT` | Unset (uvicorn uses 20 seconds) | Seconds to wait for a pong before the server closes the socket. Compose sets `20` for this recipe. |

Relative log paths resolve against the working directory. That directory is the repository root for a host-native run and `/app` in the container.

The served profiles set `server.ws_ping_interval_s: 0` and `server.ws_ping_timeout_s: 20`. The example compares these values with `UVICORN_WS_PING_INTERVAL` and `UVICORN_WS_PING_TIMEOUT` when it builds a profile's runtime. If they differ, the session fails and the error names the values to set. The keepalive variables apply to the whole server process, so they also change the keepalive for every other example in that process.

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

The following `write_gate` keys in `voice/config/voice_agent.yaml` configure the gate.

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

To change an LLM model or base URL, edit the `llm` or `thinker-llm` entry in the catalog section that `REALTIME_SERVICE_PLATFORM` selects. The example uses only `model_id` and `base_url` from the catalog. Other request settings, such as `temperature`, `max_tokens`, and reasoning switches, come from `text/config/agent.yaml`.

### Change the Speech Endpoints

The prototype speech loader resolves ASR and TTS from the `singlegpu` section of `services.local.yaml`. Set `FBA_ASR_SERVER` or `FBA_TTS_SERVER` to point at another Riva-compatible gRPC server. The application log shows a warning when the Realtime route's ASR or TTS selection differs from the server that the speech loader uses.

## Event Log

Set `FBA_VOICE_EVENT_LOG` to record one JSONL record per internal, voice, and timing event. Each record contains `timestamp`, `kind`, and `session_id`, plus event fields. The record kinds use the prototype event names, for example:

- `barge_in_verdict`: the frontend verdict for speech during backend work.
- `backend_context`: the history settings and history size of each delegated backend request.
- `transcript_normalized`: an ASR transcript after identifier normalization.
- `filler_timing`: when filler was ready, whether it was spoken, and its outcome.
- `wait_input` and `wait_resumed`: a caller turn added to an outstanding tool wait, and the resume of that wait with its trigger.
- `response_create`: how a client `response.create` was handled, for example the rule `consumed_after_inbox_resume` or `resume_wait`.
- `write_proposed`, `write_presented`, `write_presentation`, `write_presentation_heard`, `write_confirmation`, `write_confirmed`, `write_invalidated`, `write_held`, `write_gate_error`, and `write_gate_config_problem`: the write gate's proposals, spoken summaries, confirmation verdicts, and problems.

The log can record transcripts and tool payloads. Treat it as sensitive data.

## Protocol Notes

The session implements the OpenAI Realtime protocol in the prototype's `voice/wire/` package. Note the following behavior:

- The session accepts the GA schema only. A `session.update` with a beta field, such as `turn_detection` at the top level or the `pcm16` format string, is rejected as a whole. The error names each field and its GA replacement.
- A `semantic_vad` request is approximated by silence-based `server_vad`. Eagerness `high` maps to 300 ms of silence, `medium` and `auto` map to 500 ms, and `low` maps to 800 ms. The echoed object carries `x_nvidia_effective` with the effective values. Turns end on silence, not on semantic completeness.
- `turn_detection: null` selects manual mode, in which the client commits audio and requests responses.
- Add `?x_nvidia_filler=1` to the WebSocket URL to receive unspoken filler as non-standard `x_nvidia.filler` events. The default profile speaks its filler, so these events apply to profiles with `filler.mode: log_only`.
- Over the session cap, or before the speech warm-up succeeds, the server sends an `error` event with code `server_busy` and closes the socket with code 1013. The next session retries the warm-up.

## Provenance

The `text/` and `voice/` directories are copies of the prototype's `src/prototypes/text_frontend_backend_agent` and `src/prototypes/voice_frontend_backend_agent` directories, with the following exceptions:

- The package prefix is rewritten to `examples.frontend_backend_verdict.text` and `examples.frontend_backend_verdict.voice`.
- `voice/config/voice_agent.yaml` has two listed value edits. They point the text agent configuration at `text/config/agent.yaml` and the speech catalogs at this example's `services.*.yaml`.
- The text terminal UI (`text/cli`), the voice `cli/tau2_gates` scripts, and the prototype READMEs are not copied.
- Files that the example changes for the tool continuation resume (`R1`) or the write gate (`R4`) carry a `diverged` record in `provenance.json` with the change ID and a reason. The two files that the example adds, `voice/agent/write_gate.py` and the `tau3_eval_frontend_verdict_speak_history_write_gate.yaml` profile, are recorded as `diverged` with `"source": null`.

The ported unit tests in `tests/unit/frontend_backend_verdict/` are also prototype copies, and some of them are diverged as well. The `provenance.json` file records the prototype repository, commit, and SHA-256 hash of every original file. The `test_fbv_provenance.py` test reverses the rewrites and checks every hash that is not diverged, so an unrecorded edit to a copied file fails CI. Each diverged entry must have a reason and a change ID, such as `R1` or `R1+R4`.

When updating copied files from a newer prototype commit, update `provenance.json` and preserve the recorded local changes. Run the provenance test below to verify the resulting file hashes.

## Test the Example

Run the example's unit tests, including the provenance check:

```bash
uv run pytest tests/unit/frontend_backend_verdict -v
```

The unit tests use fake speech and LLM services. A live session requires the NeMo-Speech.cpp sidecar, a GPU, and an Inference Hub key.

## Evaluate With Tau

Run tau-bench evaluation through `voice-agent-evaluation` with `execution.backend: evaluator_realtime` and `product.profile: openai`. The `tau_native` lane cannot select this model ID and expects a 500 ms turn-detection echo, while this example echoes 800 ms.

## Limitations

This example differs from the prototype server in the following ways:

- The repository's `REALTIME_API_KEY` Bearer authentication replaces the prototype's optional Bearer token.
- `ek_` client-secret session templates are not applied for this model, because the pipeline owns the session.
- The prototype's browser page at `/` and its `/health` session counters are not served. The repository's `/health` route applies.
- Speech warm-up runs on the first session instead of at server start.
- On a newly created container, Pipecat downloads the NLTK `punkt_tab` data while it sets up the first pipeline. Sessions that start during the download wait for it, and they fail if it takes longer than 120 seconds. Download the data once after you start the container, for example with `docker compose exec <service> .venv/bin/python -c "import nltk; nltk.download('punkt_tab')"`, or keep `/root/nltk_data` on a volume.
- The `browser_demo`, `browser_demo_frontend_verdict`, `browser_demo_slow_backend`, `live_demo`, `cloud_speech`, and `backend_only` profiles are not served.
- The example supports only the OpenAI Realtime WebSocket. WebRTC, RTVI, and the browser UI are not supported.
- The example has no cloud or server Compose recipe. The prototype speech loader does not read `services.cloud.yaml`.
