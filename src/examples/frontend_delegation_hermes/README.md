# Frontend Delegation Hermes Example

Frontend Delegation Hermes (FDH) is a voice agent with two agents that are decoupled in time:

- **The frontend** (`nvidia/nvidia/nemotron-3.5-lightning`, reasoning off) runs on every user turn and makes exactly one call: `delegate(delegate, filler_text, request)`. The server speaks `filler_text` immediately. `request=status` asks a working backend for progress without disturbing it.
- **The backend** is a Hermes `AIAgent` (`nvidia/nvidia/nemotron-3-ultra`, reasoning on), one per session, in its own Python 3.14 worker process behind a gateway. Depending on its state, it starts a session, continues it (`run_conversation`), steers the running task (`steer` or `redirect`), or reports progress (`get_activity_summary`). The controller follows [`fixtures/workflow.csv`](fixtures/workflow.csv).

The example is registered in [`examples_registry.yaml`](../../../examples_registry.yaml) as `frontend-delegation-hermes-agent`, with the slots `llm`, `asr`, and `tts`. It serves two kinds of sessions on the repository's shared server:

- Browser UI sessions over WebRTC or the UI's WebSocket transport, with the Real-Time Voice Inference (RTVI) protocol.
- OpenAI Realtime (GA) sessions on the shared `/v1/realtime` route with the model `nvidia/nemotron-realtime-frontend-delegation-hermes`.

It has one Docker Compose recipe, `frontend-delegation-hermes-agent/single-gpu`, an eval overlay, [`docker/docker-compose.fdh-eval.yaml`](../../../docker/docker-compose.fdh-eval.yaml), and a silent-frontend overlay for single-turn evaluations, such as VoiceBench, only, [`docker/docker-compose.fdh-silent-frontend.yaml`](../../../docker/docker-compose.fdh-silent-frontend.yaml), which goes after the eval overlay. `FDH_PROMPT_PROFILE` selects the [prompt profile](#prompt-profiles). For the step-by-step guide, refer to [Deploy the Frontend Delegation Hermes Agent](../../../docs/how-to/deploy-frontend-delegation-hermes-agent.md). AI agents load the [`nemotron-voice-agent-fdh-deploy`](../../../.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md) skill to build, deploy, or evaluate the example.

## Architecture

The following diagram shows the containers of the recipe:

```text
 browser UI (RTVI) ──┐
 Realtime client ────┴─► frontend-delegation-hermes-agent-single-gpu (shared app image, Python 3.12, :7860)
                          ├─ frontend decider ──► Inference Hub nemotron-3.5-lightning
                          ├─ speech ──► nemo-speech :50051 (ASR + TTS)
                          └─ ws://default--fdh-hermes.openshell.localhost:8080/v1/backend (FDH_BACKEND_URL)
                               └─ hermes-openshell-gateway (OpenShell gateway, 127.0.0.1:8080 on the host)
                                    └─ OpenShell sandbox fdh-hermes (nemotron-voice-agent-fdh:local, Python 3.14, UID 1500)
                                         └─ FDH gateway :8790 + one Hermes worker process per session
                                              └─ Inference Hub nemotron-3-ultra (the only allowed outbound host)
```

The voice side runs in the shared app container. Hermes needs Python 3.14 and its own packages, so the FDH gateway and the Hermes workers run in their own image, built from [`docker/Dockerfile.fdh`](../../../docker/Dockerfile.fdh) with the Hermes checkout in `HERMES_AGENT_DIR`. [`docker/fdh/gateway.sh`](../../../docker/fdh/gateway.sh) starts the gateway with the settings in [`docker/fdh/gateway.env`](../../../docker/fdh/gateway.env). By default, the gateway and its workers run in an NVIDIA OpenShell sandbox. The `fdh-hermes` service is a controller that creates the sandbox with the files in [`docker/fdh/openshell/`](../../../docker/fdh/openshell/). The sandbox can call only `inference-api.nvidia.com`, and it sees a placeholder in place of the real `NVIDIA_API_KEY`. The plain runtime overlay, [`docker/docker-compose.fdh-hermes-plain.yaml`](../../../docker/docker-compose.fdh-hermes-plain.yaml), runs the gateway in a plain `fdh-hermes` container at `ws://fdh-hermes:8790/v1/backend` instead. Use it for A/B checks, debugging, and hosts that cannot run OpenShell. Both runtimes run the same image and settings, and neither publishes the FDH gateway port to the host. The image does not mount `src/`, so rebuild it after a source change. It records a hash of its `src/` in `/app/SOURCE_SHA256`, and the gateway prints that hash at start.

Only the worker processes import Hermes, because `src/utils.py` would shadow Hermes's own `utils` module. The workers get a `PYTHONPATH` that holds only an `examples` symlink.

## Sessions and Profiles

`FDH_PROFILE` selects the deployment profile. Unset means `default`, and any other value than `default` or `eval` is an error. The following table lists what each profile runs. The paths are relative to `config/`.

| | `default` (browser UI) | `eval` (OpenAI Realtime, benchmarks) |
| --- | --- | --- |
| Delegation profile | `profiles/pipecat_demo.yaml` | `profiles/pipecat_realtime_eval.yaml` |
| Pipecat settings | `adapters/pipecat.yaml` | `adapters/pipecat.eval.yaml` |
| Gateway configuration (Hermes runtime) | `gateway.yaml` | `gateway.eval.yaml` |

The rules match the Frontend/Backend Agent example:

- OpenAI Realtime sessions always run the `eval` delegation profile and Pipecat settings, whatever `FDH_PROFILE` says.
- A deployment with `FDH_PROFILE=eval` refuses browser UI sessions.
- The gateway configuration belongs to the deployment, because one Hermes runtime serves every session. Only the eval overlay sets `gateway.eval.yaml`, so benchmark runs must use it.

[`adapters/pipecat/profiles.py`](adapters/pipecat/profiles.py) holds the mapping. `docker-compose.yml` sets every FDH variable of the voice service as a literal, so `.env` cannot change an evaluation run. The one exception is `FDH_PROMPT_PROFILE`. Refer to [Prompt Profiles](#prompt-profiles).

### Browser UI Sessions

A browser UI session runs the `default` profile on the repository's UI transport, with RTVI in place of the Realtime wire protocol. It is the former FDH demo: the greeting "Hi! How can I help you today?", the demo tools `get_order` and `cancel_order` run in the server, a 5-second delay before each backend run (`FDH_BACKEND_DELAY_S`), and Pipecat's default turn settings.

The UI's **LLM** menu selects the frontend model, and the **ASR**, **TTS**, and voice menus select the speech endpoints from this example's catalogs. The Hermes backend model stays the gateway's. The UI's prompt editor is not applied, because the frontend prompt is `config/prompts/<profile>/frontend.yaml` (`frontend_system`). Browser sessions share one cap of `server.max_sessions` sessions (4 in Compose), separate from the Realtime cap.

[`adapters/pipecat/ui_session.py`](adapters/pipecat/ui_session.py) and [`adapters/pipecat/ui_pipeline.py`](adapters/pipecat/ui_pipeline.py) implement these sessions. They reuse the Realtime session's turn manager, turn bridge, output tagger, delegation, and backend link.

### OpenAI Realtime Sessions

The shared `/v1/realtime` route authenticates the socket with `REALTIME_API_KEY`, resolves the model, and hands the socket over before it sends any event. `frontend-delegation-hermes-agent` is a session-owned pipeline in `src/realtime/gateway.py`, so FDH serves `session.created` and every later event with its own wire code. [`adapters/pipecat/bot.py`](adapters/pipecat/bot.py) builds the Realtime pipeline, and [`adapters/pipecat/runtime.py`](adapters/pipecat/runtime.py) admits the session.

Realtime sessions have their own cap of `server.max_sessions` sessions (4 in Compose). A refused session receives an `error` event with code `server_busy`, and the socket closes with code 1013. The first session waits for a one-time warm-up, one frontend LLM request and a gateway check, before it is admitted. A failed warm-up refuses that session with `server is starting`, and the next session retries it. The server must not ping WebSocket clients: Compose sets `UVICORN_WS_PING_INTERVAL=0`, and Realtime sessions refuse to start otherwise.

Pipecat owns turn taking: Silero VAD with `VADParams` from the settings file, the default user-turn start strategies, and `SpeechTimeoutUserTurnStopStrategy`, with Smart Turn disabled. Pipecat also owns interruptions and audio pacing. Speech uses `NvidiaForceEouSTTService` and `NvidiaTTSService` in `per_sentence` mode on the catalog endpoints. The FDH spelling hold is off in both profiles.

The Realtime sessions support the following OpenAI Realtime behavior:

- `session.update` with tools, instructions, input and output audio formats, and server VAD;
- `input_audio_buffer.append` and `input_audio_buffer.clear`;
- client-executed tools: `response.function_call_arguments.done` out, and `function_call_output` plus `response.create` in;
- transcript deltas before the audio of each sentence, and exactly one `response.done` per response.

The following limits apply:

- **Heard audio is an estimate.** Pipecat's WebSocket output writes audio at twice real time. The adapter advances its playout estimate at real time from the first written chunk, capped at the written amount. A client that pauses playback hears less than estimated.
- **Speech onset is late.** `audio_start_ms` is the cursor at Pipecat's VAD confirmation minus `start_secs`. Stub measurements put it about 200 ms after the actual onset.
- **No manual turns.** A client that asks for `turn_detection: null` is accepted with a logged warning, but Pipecat's VAD still ends audio turns and `input_audio_buffer.commit` is rejected.
- **Catalog voice only.** The TTS voice is the catalog voice, and the client's requested voice is ignored.

## Prompt Profiles

A prompt profile is a folder, `config/prompts/<profile>/`, with the frontend catalog (`frontend.yaml`, read by the voice side), the backend catalog (`backend.yaml`, read by the Hermes gateway), and the voice-layer messages (`voice.yaml`). The `FDH_PROMPT_PROFILE` environment variable selects it. The example ships the profiles in the following table.

| Prompt profile | Default for | Purpose |
| --- | --- | --- |
| `customer_service` | The eval overlay | Customer-service prompts for multi-turn tool use, such as Tau. Formerly `config/prompts.yaml`, `config/prompts.backend.yaml`, and `config/prompts.voice.yaml`, with unchanged content. The eval pins cover its prompt hashes. |
| `generic` | The `default` deployment, including its Realtime sessions | The browser demo prompts. Currently a copy of `customer_service/`, planned to become more generic. |
| `general_qa` | None | A general question-answering assistant for single-turn evaluations. The frontend delegates every request (`delegate=true`, filler "One moment."). The backend answers from the LLM with no tools: the answer first, "The answer is X." for multiple choice, formatting instructions followed literally, and a refusal that starts with "I'm sorry, but I can't help with that." for harmful requests. |

The following files read the variable. The browser-demo files default to `generic`, and the eval files default to `customer_service`:

| File in `config/` | Key | Value |
| --- | --- | --- |
| `delegation_agent.yaml` | `delegation.prompts.path` | `prompts/${FDH_PROMPT_PROFILE:-generic}/frontend.yaml` |
| `voice_agent.yaml` | `agent.overrides.prompts.path` | `prompts/${FDH_PROMPT_PROFILE:-generic}/voice.yaml` |
| `gateway.yaml` | `hermes.prompts.path` | `prompts/${FDH_PROMPT_PROFILE:-generic}/backend.yaml` |
| `profiles/realtime_eval.yaml` | `delegation.prompts.path` | `../prompts/${FDH_PROMPT_PROFILE:-customer_service}/frontend.yaml` |
| `voice/realtime_eval.yaml` | `agent.overrides.prompts.path` | `../prompts/${FDH_PROMPT_PROFILE:-customer_service}/voice.yaml` |
| `gateway.eval.yaml` | `hermes.prompts.path` | `prompts/${FDH_PROMPT_PROFILE:-customer_service}/backend.yaml` |

`FDH_PROMPT_PROFILE` is the one FDH setting that the shell or `.env` can set. Compose passes `${FDH_PROMPT_PROFILE:-generic}` to both the voice service and `fdh-hermes`, and the eval overlay passes `${FDH_PROMPT_PROFILE:-customer_service}` to both. `docker/fdh/gateway.env` lists `FDH_PROMPT_PROFILE=generic`, and the OpenShell controller passes the container's value into the sandbox, so both sides always use the same profile. Realtime sessions on a `default` deployment therefore use `generic`. Use the eval overlay for `customer_service`.

Never set the variable to an empty value. An unknown profile name fails with an error that names the missing prompt file. The Hermes image bakes in `src/`, so rerun `up -d --build` after a prompt change. To add a profile, copy `config/prompts/customer_service/` to `config/prompts/<name>/`, edit the three files, and set `FDH_PROMPT_PROFILE=<name>`. The unit tests in `test_fdh_prompt_profiles.py` load every profile in `config/prompts/`. For the deployment steps, refer to [Prompt Profiles](../../../docs/how-to/deploy-frontend-delegation-hermes-agent.md#prompt-profiles).

The prompt paths are part of the effective delegation and voice settings, so a `customer_service` run has another `config_hash` than a run before the move, or a run with another profile. The prompt hashes show whether the prompts are the same.

## Layout

The following table lists the example's files and folders:

| Path | Role |
| --- | --- |
| `pipeline.py` | The registry entry point. It re-exports `bot` from `adapters/pipecat/bot.py`. |
| `prompts.yaml` | The registry's prompt catalog: one public, informational `frontend` prompt for the UI's prompt panel. The agent does not read it. Its prompts are in `config/prompts/<profile>/`. |
| `tools.yaml` | The demo tools that the UI's tools panel lists. |
| `services.local.yaml` | The on-premises catalog. The `singlegpu` section holds the frontend LLM and the `nemo-speech` ASR and TTS entries. The `server` section exists for the registry's catalog checks. |
| `services.cloud.yaml` | NVIDIA cloud function (NVCF) ASR and TTS entries, and the Inference Hub frontend LLM, for browser UI selections. |
| `adapters/pipecat/` | The Pipecat adapter. `bot.py` routes a session, `runtime.py` holds the warm-up and admission, `profiles.py` maps `FDH_PROFILE`, `ui_session.py` and `ui_pipeline.py` serve the browser UI, and `session.py`, `pipeline.py`, `serializer.py`, `turn_bridge.py`, `speech_output.py`, `timing.py`, `services.py`, and `settings.py` serve the Realtime sessions. |
| `core/` | Shared, adapter-neutral code. `engine/` (turn manager, transcript, output scheduler, backend link, spelling hold, spoken-text cleanup), `frontend/` (the `delegate` contract, decider with repair, hedging, and guards, status verbalizer), `backend/` (protocol, controller state machine, context queue), `sidecar/` (gateway, session runtime, worker pool, Hermes homes, in-process stub), `worker/` (worker loop, tool futures, the only Hermes import), `tools/` (relay, result hints, format retry, field provenance, identity ledger), `normalization/`, `wire/` (Realtime events and session protocol), `audio/`, `text/` (demo tools), `catalog.py`, `config.py`, `voice_config.py`, `prompt_features.py`, `runtime.py`, `server_common.py`, and `head_api.py` |
| `config/` | `delegation_agent.yaml`, `voice_agent.yaml`, `gateway*.yaml`, `prompts/<profile>/` (the [prompt profiles](#prompt-profiles), each with `frontend.yaml`, `backend.yaml`, and `voice.yaml`), `voice/*.yaml` (voice profiles), `profiles/*.yaml`, `adapters/pipecat*.yaml` (Pipecat settings), and `text/` |
| `fixtures/` | `workflow.csv` (the controller specification) and `delegation_cases.jsonl` (frontend decision cases) |

`config/voice_agent.yaml` reads the speech endpoints from the `singlegpu` section of this example's own `services.local.yaml`. `FBA_ASR_SERVER` and `FBA_TTS_SERVER` replace the catalog server.

## Configuration Profiles

Unless noted otherwise, every profile has 800 ms end-of-turn silence (the client's value is ignored), transcript normalization on, and no WebSocket keepalive ping. The following table lists the delegation profiles in `config/profiles/`:

| Profile | Use |
| --- | --- |
| `pipecat_demo.yaml` | The `default` profile: `browser_demo.yaml` with Pipecat turn taking. |
| `pipecat_realtime_eval.yaml` | The `eval` profile: `realtime_eval.yaml` with Pipecat turn taking, client-executed tools (`tools.executor: wire`), the client's instructions as the policy, and no greeting. |
| `browser_demo.yaml` | The base of `pipecat_demo.yaml`: greeting, demo tools run in the server, and `FDH_BACKEND_DELAY_S` (default 5 s) before each backend run. |
| `realtime_eval.yaml` | The base of `pipecat_realtime_eval.yaml`: the single domain-agnostic evaluation profile. Tool-argument rules, result-hint targets, and spelling-hold patterns come from the session's tool schemas. |

The test-only stub profiles live in `tests/unit/frontend_delegation_hermes/core/_fdh_variants.py`.

The Pipecat turn settings live in their own files, outside the shared delegation config and its `config_hash`. The following table lists them:

| Settings File | Use |
| --- | --- |
| `config/adapters/pipecat.yaml` | The `default` profile: every VAD and turn key is `null`, so Pipecat's defaults apply. |
| `config/adapters/pipecat.eval.yaml` | The `eval` profile: Silero `stop_secs: 0.8`, `audio_idle_timeout: 0`, and `user_speech_timeout: 0`, so a turn ends after 0.8 s of silence counted in input audio. |

The skill's `config-knobs.md` reference lists every knob.

## Failure-Fix Switches

Each switch below is on in the YAML defaults (`delegation_agent.yaml`, `gateway.yaml`, `voice/browser.yaml`, `voice/realtime_eval.yaml`), except M2, which stays off. The Python code defaults are off. With every switch off, the rendered prompts are byte-identical to agent commit `3a7e04a` (`test_fdh_golden_prompts.py`). The following table lists the switches:

| Switch | Where | Fix |
| --- | --- | --- |
| `output.proactive_status` | `delegation_agent.yaml` | M1: one `status_proactive_lines` line after `after_s` of silence while WORKING (`FDH_PROACTIVE_AFTER_S`, default 15; calibrate it per host) |
| `normalization.transcript.spelled_runs`, `normalization.tool_arguments.escalate_invalid_message_key` | `voice/realtime_eval.yaml` (`voice/browser.yaml` has spelled runs only) | M3.1 and M3.3: spelled letters and digits are joined; the local "invalid ID" answer asks for a read-back, then for the whole ID |
| `delegation.spelling_hold` | `delegation_agent.yaml` | M3.2: wait `hold_ms` before deciding a turn that ends mid-spelling. With `complete_patterns: auto`, `realtime_eval.yaml` never holds a value in the strict shape of an `id`, `*_id`, or `*_ids` example from the tool schemas. Off in both Pipecat profiles |
| `delegation.replay_unheard_answer` and `prompt_features.replay_intent` | `delegation_agent.yaml` | M2 (experimental, off by default): replay an unheard answer on `request=status` in IDLE |
| `delegation.filler_dedupe` | `delegation_agent.yaml` | G1: drop or replace a repeated filler (`filler_alternatives`) |
| `output.clean_answers` | `delegation_agent.yaml` | G2: strip markdown and make one sentence per list item before TTS |
| `prompt_features.spelling_v2`, `spoken_output`, `write_consent` | `gateway.yaml` | M3.4, G2, and G4: backend prompt variants, all on |
| `prompt_features.phone_format` | `gateway.yaml`, `gateway.eval.yaml` | I1: a domain-generic `<phone_numbers>` block in `backend_system`. Lookups by a ten-digit phone number use XXX-XXX-XXXX unless the tool description shows another format, with one retry in the other common form. Tools that store or change a number get it unchanged |
| `prompt_features.domain_notes` and `domains` | `gateway.yaml`, `gateway.eval.yaml`, `prompts/<profile>/backend.yaml` (`backend_domain_notes`) | One telecom note, on by default. Telecom is detected by its tool `get_customer_by_phone`, and the note renders as `<domain_notes>` right after the policy in telecom sessions only |
| `normalization.tool_arguments.result_hints` | `voice/realtime_eval.yaml` (schema in `voice_agent.yaml`) | I3 and I4: after a failed lookup, append `lookup_not_found_hint` on the first miss and `lookup_not_found_hint_words` on later misses, at most 3 until a lookup succeeds. `voice/realtime_eval.yaml` uses `tools: auto` and fires only when the not-found text names a user, customer, member, client, or profile. Logged as `result_hint` events |

## Geval-Fix Switches

The switches below are on in `gateway.yaml`, `gateway.eval.yaml` (pinned), and `voice/realtime_eval.yaml`, and off in `voice_agent.yaml` and in the Python code, except A3, which is off in `voice/realtime_eval.yaml`. The following table lists the switches:

| Switch | Where | Fix |
| --- | --- | --- |
| `prompt_features.format_correction` | `gateway.yaml` | A2: inside the `spelling_v2` text, a change of separators or letter case only is a format correction, not a variant. Its voice half is `result_hints.message_key: lookup_not_found_hint_format` |
| `prompt_features.write_consent_v2` | `gateway.yaml` | B1: inside `<consent>`, the confirmation the policy requires, to the exact change, before a write |
| `prompt_features.policy_check`, `request_scope` | `gateway.yaml` | B2 and B3: a `<before_change>` block. B2 checks the policy for the exact change. B3 covers every record and option the request names |
| `prompt_features.transfer_check`, `capability_check` | `gateway.yaml` | C1 and C2: a `<before_transfer>` block. C1 transfers at once when the policy requires it. C2 checks the tool list before saying a request is impossible |
| `normalization.tool_arguments.format_retry` | `voice/realtime_eval.yaml` (schema in `voice_agent.yaml`) | A1 and A1-lite: after a not-found lookup by a ten-digit phone field, the same digits in the other format, once per value. `mode: call` makes the relay look it up itself, but only on a tool whose schema declares `annotations: {readOnlyHint: true, idempotentHint: true}`. Otherwise, and always with `mode: hint`, the relay appends `format_retry_hint` |
| `normalization.tool_arguments.result_hints.field_provenance` | `voice/realtime_eval.yaml` (off) | A3: from the second person-lookup miss on, the note also says which values the user spelled, which were heard as a word, and which are unknown |

The phone fields come from the session's tool schemas: string arguments of read tools whose name contains `phone`, `mobile`, `msisdn`, or `telephone`, or whose description says "phone number." On the τ³ tools, only telecom `get_customer_by_phone.phone_number` qualifies, and no tool declares read-only semantics, so A1 runs as A1-lite there. The relay's own call has the ID `<id>.fmt`. The found, not-found, and hint texts are the `format_retry_found`, `format_retry_not_found`, and `format_retry_hint` keys of `config/prompts/<profile>/voice.yaml`.

The gateway records a backend decision trace when `hermes.backend_trace` is on. The default is `${FDH_BACKEND_TRACE:-false}` in `gateway.yaml` and `${FDH_BACKEND_TRACE:-true}` in `gateway.eval.yaml`. Each worker appends a `session` record, then a `request` and `response` pair per model call, to `<workers.log_dir>/<session_id>.trace.jsonl`.

## F1, F2, F4, and F8 Switches

These switches are on in `realtime_eval.yaml`, `voice/realtime_eval.yaml`, `gateway.yaml`, and `gateway.eval.yaml`. The following table lists them:

| Switch | Where | Fix |
| --- | --- | --- |
| `delegation.guards.{liveness_reply,question_never_silent,ping_as_status}` and `delegation.replay_unheard_answer.triggers` | `profiles/realtime_eval.yaml` | F1 and F2: a silent reply to "are you still there?" or "could you repeat that?" is delegated or replaced by a replay of the unheard answer, and a ping while the backend works is a status request. Phrase lists are per ASR language, and only `en` ships |
| `prompt_features.situational_facts` | `gateway.yaml` | F4: when the policy makes a problem depend on where the user is and the user has not said, the backend asks once, early |
| `normalization.tool_arguments.result_hints.ledger` | `voice/realtime_eval.yaml` | F8: after a failed identification lookup, the note lists the lookups tried, what the user spelled, the user's corrections, and the other identification lookups the tools offer |
| `prompt_features.identity_from_identifier` | `gateway.yaml` | F8 name line: a name inside a spoken identifier, after it is confirmed |

## Silent-Frontend Switch

Single-turn evaluations, such as VoiceBench, score only the first response of each case, and this agent speaks each filler and proactive status line as its own response. The following switch is off by default in both profiles and in the Python code:

| Switch | Where | Effect |
| --- | --- | --- |
| `delegation.silent_frontend` | `delegation_agent.yaml` (`FDH_SILENT_FRONTEND`, default `false`) | When on, the filler of a delegated turn is logged as a `filler_silenced` event with the reason `silent_frontend` and never spoken. The M1 proactive status line is not spoken either, and a `proactive_status_silenced` event is logged once per backend run. Direct answers are still spoken. A turn that the frontend leaves silent (`delegate=false`, empty filler) but that holds a request is delegated as a `task` with an empty filler and logged as a `delegation_guard` event with `guard=silent_delegated`. Empty turns, backchannels (`delegation.guards.backchannel_words`), pauses, liveness turns, and turns while the backend is WORKING stay silent. Only `docker/docker-compose.fdh-silent-frontend.yaml` turns it on. With the switch off, the key is dropped from the effective settings, so the `config_hash` does not change. With it on, `fdh_session_start` carries `silent_frontend: true` and the `config_hash` differs |

For the deployment steps, refer to [Run a VoiceBench Evaluation](../../../docs/how-to/deploy-frontend-delegation-hermes-agent.md#run-a-voicebench-evaluation).

## Deployed Fingerprint

The voice server and the gateway log a deployed fingerprint: `fdh_session_start.features`, `frontend_prompt`, and `backend_configured`, which includes `backend_domain` and `session_tools_sha256`. `session_tools_sha256` is the first 16 hex characters of the SHA-256 of the session's sorted `session.update` tool schemas. With an `auto` switch on, the voice server also logs a `session_rules` event with the same hash and the derived rules.

`config_hash` values differ from the prototype's because the configuration paths moved. The prompt hashes and `session_tools_sha256` are unchanged.

## Run It

The following commands start the recipe, with the browser UI (`default`) or for evaluation runs (`eval`). Complete the prerequisites of the [deploy guide](../../../docs/how-to/deploy-frontend-delegation-hermes-agent.md) first: `NVIDIA_API_KEY` and `HF_TOKEN` in `.env`, the speech model weights, and the Hermes checkout in `HERMES_AGENT_DIR`. The default OpenShell runtime also needs Linux 6.2 or newer with Landlock, Docker Engine 28 or newer, Docker Compose 2.24.4 or newer, and a free host port `127.0.0.1:8080`.

```bash
# default: the browser UI at https://<host>:7860, and Realtime sessions with the eval settings
docker compose --profile frontend-delegation-hermes-agent/single-gpu up -d --build
# eval: OpenAI Realtime only, ws://<host>:7860/v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation-hermes
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build
# single-turn evaluations (VoiceBench) only: the eval overlay, then the silent-frontend overlay,
# with the general_qa prompt profile
FDH_PROMPT_PROFILE=general_qa docker compose -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml \
  -f docker/docker-compose.fdh-silent-frontend.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build
# plain runtime (baseline): add the plain overlay before the eval overlay, if any
docker compose -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml \
  --profile frontend-delegation-hermes-agent/single-gpu up -d --build
```

For a Realtime smoke test, run the repository's test client and open `http://localhost:8765`. Add `--upstream ws://localhost:7860` in eval mode:

```bash
uv run python realtime_test_client/proxy.py --model nvidia/nemotron-realtime-frontend-delegation-hermes
```

The voice event log is `logs/fdh_voice_events.pipecat.jsonl`. In the OpenShell runtime, `docker compose logs fdh-hermes` shows the controller and the gateway configuration, and the sandbox's gateway events, worker logs, and traces are saved to `logs/openshell/<time>/` when `fdh-hermes` stops. In the plain runtime, `docker compose logs fdh-hermes` shows the gateway configuration and the source hash, and the gateway writes `logs/fdh_gateway_events.pipecat.jsonl` and `logs/fdh_workers_pipecat/`. The deploy guide's [Read the Logs](../../../docs/how-to/deploy-frontend-delegation-hermes-agent.md#read-the-logs) section lists every log.

## Tests

The following command runs the unit tests on the host:

```bash
uv run pytest tests/unit/frontend_delegation_hermes -q
```

The suites cover the following areas:

- `core/` tests the shared code. `test_fdh_workflow_table.py` checks all 12 rows of `workflow.csv`. `test_fdh_gateway_processes.py` runs real worker processes with the fake agent. `test_fdh_golden_prompts.py` and `test_fdh_fix_plan.py` pin the baseline prompts and the `124acf7` control. `test_fdh_tau3_fixes.py`, `test_fdh_identity_fixes.py`, and `test_fdh_geval_fixes.py` cover the switches. `test_fdh_schema_rules.py` checks the derived rules against `fixtures/schema_rules_snapshot.json`. The core tests drive the turn manager through `core/_fdh_test_head/`, a test-only Realtime head with an input-audio clock, an energy VAD, and stub speech. It is not shipped.
- `voice/` tests the voice layer: configuration, normalization, the Realtime wire events, and playback.
- `pipecat/` tests the Pipecat adapter with an energy VAD, a stub ASR, and a tone TTS. `test_fdh_eval_pins.py` pins the eval behavior against `fixtures/eval_pins.json`: the prompt and tool hashes and the backend fingerprint per τ² domain (airline, retail, and telecom), the hashed delegation, voice, and gateway settings, the Pipecat settings and processors, and the server events of a scripted session per domain. `test_fdh_registry.py` checks the registry files, and `test_fdh_ui_session.py` checks the browser UI sessions.
- `test_fdh_compose.py` checks that the voice service sets every FDH variable as a literal, except `FDH_PROMPT_PROFILE`, which must have a non-empty default and the same value on the voice service and `fdh-hermes`, that `docker/fdh/gateway.env` holds plain `KEY=value` lines, that the eval overlay sets only `environment` keys on `fdh-hermes`, that OpenShell is the default runtime, that the plain overlay overrides single fields only, and that the controller passes every `gateway.env` key into the sandbox.
- `test_fdh_openshell_pins.py` checks that every OpenShell image and the CLI build use the one release in `docker/fdh/openshell/VERSION`, and that the OpenShell gateway's bind port matches the published port.
- `test_fdh_import_isolation.py` checks that the core imports no adapter, that no adapter imports another, and that the gateway never loads Pipecat.
- `test_fdh_dependencies.py` checks that the image lock pins the repository's Pipecat version and lists the same packages as the `fdh` extra.

`test_fdh_hermes_multisession.py` runs real Hermes workers against a fake OpenAI server. It is skipped when the Hermes interpreter (`FDH_HERMES_PYTHON`, default `~/.cache/fdh/hermes-venv-314/bin/python`) is missing. `test_fdh_hermes_adapter.py` also needs the Hermes interpreter. The skill's `prerequisites.md` reference describes the optional host Hermes environment.

After an intended baseline change, regenerate the golden files with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden.py --write`. Regenerate the eval pins only for an intended change of the evaluated behavior, with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/pipecat/_fdh_eval_pins.py --write`.
