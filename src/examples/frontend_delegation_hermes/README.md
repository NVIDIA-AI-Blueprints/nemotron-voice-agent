# Frontend Delegation Hermes Example

Frontend Delegation Hermes (FDH) is an OpenAI Realtime (GA) voice agent with two agents that are decoupled in time:

- **The frontend** (`nvidia/nvidia/nemotron-3.5-lightning`, reasoning off) runs on every user turn and makes exactly one call: `delegate(delegate, filler_text, request)`. The server speaks `filler_text` immediately. `request=status` asks a working backend for progress without disturbing it.
- **The backend** is a Hermes `AIAgent` (`nvidia/nvidia/nemotron-3-ultra`, reasoning on), one per Realtime session, in its own Python 3.14 worker process behind a gateway. Depending on its state, it starts a session, continues it (`run_conversation`), steers the running task (`steer` or `redirect`), or reports progress (`get_activity_summary`). The controller follows [`fixtures/workflow.csv`](fixtures/workflow.csv).

The example is self-contained. It is not registered in `examples_registry.yaml` and has no root Compose recipe. It runs from its own image and Compose file: [`docker/Dockerfile.fdh`](../../../docker/Dockerfile.fdh) and [`docker/docker-compose.frontend-delegation-hermes.yaml`](../../../docker/docker-compose.frontend-delegation-hermes.yaml).

To build, deploy, or evaluate the example, load the [`nemotron-voice-agent-fdh-deploy`](../../../.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md) skill. Its references hold the complete τ³ procedure, the arms, the gates, and the configuration knobs.

## Architecture

The following diagram shows one container. Both adapters use the same layout:

```text
 client ─ws─► voice server :7860 (Pipecat adapter or τ³ adapter)
               ├─ frontend decider ──► Inference Hub nemotron-3.5-lightning
               ├─ speech ──► nemo-speech :50051 (ASR + TTS)
               └─ ws://host.docker.internal:8790/v1/backend
                    └─ backend gateway 127.0.0.1:8790 (inside the container)
                         └─ one Hermes worker process per session ──► Inference Hub nemotron-3-ultra
```

`docker/fdh/entrypoint.sh` starts the gateway, waits for its `/health`, and then starts the adapter named by `FDH_ADAPTER`. The gateway is reachable only on the container's loopback. Only the worker processes import Hermes, because `src/utils.py` would shadow Hermes's own `utils` module. The workers get a `PYTHONPATH` that holds only an `examples` symlink.

## Adapters

The example has two server adapters over one shared core. The following table compares them:

| Adapter | Entry Point | Service and Port | Use When |
| --- | --- | --- | --- |
| Pipecat (default) | `server.py` | `fdh-voice`, 8776 | You want the browser demo, or a τ² evaluation with Pipecat-native turn taking |
| τ³ | `tau_server.py` | `fdh-tau` (Compose profile `tau`), 8775 | You run τ³ benchmarks and need the behavior the τ³ scores were measured with |

The **Pipecat adapter** serves WebSocket and OpenAI Realtime only, without WebRTC or RTVI. Pipecat owns turn taking: Silero VAD with `VADParams` from the settings file, the default user-turn start strategies, and `SpeechTimeoutUserTurnStopStrategy`, with Smart Turn disabled. Pipecat also owns interruptions and audio pacing. Speech uses `NvidiaForceEouSTTService` and `NvidiaTTSService` in `per_sentence` mode on the catalog endpoints. The FDH spelling hold is off in both Pipecat profiles.

The **τ³ adapter** keeps the prototype server unchanged: an audio clock driven by the client's audio, Silero VAD with 800 ms silence, tracking of sent versus heard audio, and each transcript delta sent just before its audio. It never imports Pipecat adapter code.

### Pipecat Adapter Realtime Subset

The Pipecat adapter supports the following OpenAI Realtime behavior:

- `session.update` with tools, instructions, input and output audio formats, and server VAD;
- `input_audio_buffer.append` and `input_audio_buffer.clear`;
- client-executed tools: `response.function_call_arguments.done` out, and `function_call_output` plus `response.create` in;
- transcript deltas before the audio of each sentence, and exactly one `response.done` per response.

The following limits apply:

- **Heard audio is an estimate.** Pipecat's WebSocket output writes audio at twice real time. The adapter advances its playout estimate at real time from the first written chunk, capped at the written amount. A client that pauses playback hears less than estimated.
- **Speech onset is late.** `audio_start_ms` is the cursor at Pipecat's VAD confirmation minus `start_secs`. Stub measurements put it about 200 ms after the actual onset.
- **No manual turns.** A client that asks for `turn_detection: null` is accepted with a logged warning, but Pipecat's VAD still ends audio turns and `input_audio_buffer.commit` is rejected.
- **Catalog voice only.** The TTS voice is the catalog voice, and the client's requested voice is ignored.

## Layout

The following table lists the example's folders and entry points:

| Path | Role |
| --- | --- |
| `server.py` | Default entry point: the Pipecat adapter |
| `tau_server.py` | τ³ entry point: the τ³ adapter |
| `core/` | Shared, adapter-neutral code. `engine/` (turn manager, transcript, output scheduler, backend link, spelling hold, spoken-text cleanup), `frontend/` (the `delegate` contract, decider with repair, hedging, and guards, status verbalizer), `backend/` (protocol, controller state machine, context queue), `sidecar/` (gateway, session runtime, worker pool, Hermes homes, in-process stub), `worker/` (worker loop, tool futures, the only Hermes import), `tools/` (relay, result hints, format retry, field provenance, identity ledger), `normalization/`, `wire/` (Realtime events and session protocol), `audio/`, `text/` (demo tools), `config.py`, `voice_config.py`, `prompt_features.py`, `runtime.py`, `head_api.py` |
| `adapters/pipecat/` | The Pipecat adapter: app, serializer, per-connection pipeline, turn bridge, speech output and tagger, timing estimate, services, and settings |
| `adapters/tau/` | The τ³ adapter: app, server, session, input and output paths, audio clock, segmenter, and Riva, Silero, and stub speech |
| `cli/` | Tools: `voice_chat` and `audio_io` (Realtime client), `client_tools`, `backend_probe` (gateway without audio), `delegation_replay` (frontend decision gate), `fingerprint_check` (deployed fingerprint of an arm), `report_adapter` (event log to `fba_voice_metrics.py` records), `spelling_hold_replay`, `relay_replay` (gate G1), `backend_decision_replay` (gate G2), `normalization_replay`, `tau2_replay`, and `tau2_gates/` (Gate A and Gate B) |
| `config/` | `delegation_agent.yaml`, `voice_agent.yaml`, `gateway*.yaml`, `prompts.yaml` (frontend), `prompts.backend.yaml` (backend), `prompts.voice.yaml`, `voice/*.yaml` (voice profiles), `profiles/*.yaml`, `adapters/pipecat*.yaml` (Pipecat settings), and `text/` |
| `fixtures/` | `workflow.csv` (the controller specification) and `delegation_cases.jsonl` (the `delegation_replay` cases) |
| `web/index.html` | The browser page served at `/` |

## Profiles

Pass a profile with `--config`, or with `FDH_PROFILE` or `FDH_PIPECAT_PROFILE` in Compose. Unless noted otherwise, every profile has 800 ms end-of-turn silence (the client's value is ignored), transcript normalization on, and no WebSocket keepalive ping. The following table lists the profiles:

| Profile | Use |
| --- | --- |
| `profiles/pipecat_demo.yaml` | Pipecat demo: `browser_demo.yaml` with Pipecat turn taking, browser page, greeting, demo tools run in the server, and `FDH_BACKEND_DELAY_S` (default 5 s) before each backend run. Pair it with `gateway.yaml` and `adapters/pipecat.yaml` |
| `profiles/pipecat_realtime_eval.yaml` | Pipecat eval: `realtime_eval.yaml` with Pipecat turn taking, client-executed tools (`tools.executor: wire`), the client's instructions as the policy, and no greeting. Pair it with `gateway.eval.yaml` and `adapters/pipecat.eval.yaml` |
| `profiles/realtime_eval.yaml` | τ³ default: the single domain-agnostic evaluation profile. Tool-argument rules, result-hint targets, and spelling-hold patterns come from the session's tool schemas. Pair it with `gateway.eval.yaml` |
| `profiles/tau3_eval.yaml` | Historical τ³ profile, kept as a control and as the `dlg` arm: every failure fix except M2 and every identity fix. Pair it with `gateway.yaml` |
| `profiles/browser_demo.yaml` | The browser profile for the τ³ adapter, with demo tools in the server |
| `profiles/tau3_eval_silent_ack.yaml` | The `silentack` ablation: the filler of a delegated turn is not spoken |
| `profiles/stub.yaml`, `profiles/stub_gate.yaml` | No GPU, LLM, or Hermes (with `--stub-speech`). `stub_gate` makes the fake backend call a tool, so τ³ Gate A passes |
| `profiles/tau3_eval_baseline.yaml` | Control of the failure fixes, with every switch off. Pair it with `gateway.baseline.yaml` |
| `profiles/tau3_arm_*.yaml` | One failure fix or identity fix each |
| `profiles/tau3_identity_control.yaml` | Control of the identity fixes. Pair it with `gateway.identity_control.yaml` |
| `profiles/tau3_airline_canary.yaml` | M1 and M3 together (with `gateway.spelling_v2.yaml`) |
| `profiles/realtime_eval_control.yaml` | Control of the geval fixes. Pair it with `gateway.eval_control.yaml` |
| `profiles/realtime_eval_arm_*.yaml` | One geval fix each over `realtime_eval_control.yaml` |
| `profiles/realtime_eval_124acf7.yaml` | Control of the F1, F2, F4, and F8 fixes (agent `124acf7`). Pair it with `gateway.eval_124acf7.yaml` |
| `profiles/realtime_eval_liveness.yaml`, `profiles/realtime_eval_identity.yaml` | The F1 and F2 arm and the F8 arm over that control |

The Pipecat adapter's turn settings live in their own files, outside the shared delegation config and its `config_hash`. The following table lists them:

| Settings File | Use |
| --- | --- |
| `config/adapters/pipecat.yaml` | Demo and deployment: every VAD and turn key is `null`, so Pipecat's defaults apply |
| `config/adapters/pipecat.eval.yaml` | Eval only: Silero `stop_secs: 0.8`, `audio_idle_timeout: 0` and `user_speech_timeout: 0`, so a turn ends after 0.8 s of silence counted in input audio, as on the τ³ adapter |

Select a settings file with `FDH_PIPECAT_SETTINGS` or `--pipecat-settings`. The skill's `config-knobs.md` reference and its `tau.md` reference list every knob and every arm pair.

## Failure-Fix Switches

Each switch below is on in the YAML defaults (`delegation_agent.yaml`, `gateway.yaml`, `voice/browser.yaml`, `profiles/tau3_eval.yaml`, `voice/tau3_fixes.yaml`), except M2, which stays off. The Python code defaults are off. `tau3_eval_baseline.yaml` with `gateway.baseline.yaml` pins every switch off. With every switch off, the rendered prompts are byte-identical to agent commit `3a7e04a` (`test_fdh_golden_prompts.py`). The following table lists the switches:

| Switch | Where | Fix |
| --- | --- | --- |
| `output.proactive_status` | `delegation_agent.yaml` | M1: one `status_proactive_lines` line after `after_s` of silence while WORKING (`FDH_PROACTIVE_AFTER_S`, default 15; calibrate it per host) |
| `normalization.transcript.spelled_runs`, `normalization.tool_arguments.escalate_invalid_message_key` | `voice/tau3_spelling.yaml` (extended by `voice/tau3_fixes.yaml`; `voice/browser.yaml` has spelled runs only) | M3.1 and M3.3: spelled letters and digits are joined; the local "invalid ID" answer asks for a read-back, then for the whole ID |
| `delegation.spelling_hold` | `delegation_agent.yaml` | M3.2: wait `hold_ms` before deciding a turn that ends mid-spelling. `tau3_eval.yaml` never holds a complete six-character code. With `complete_patterns: auto`, `realtime_eval.yaml` never holds a value in the strict shape of an `id`, `*_id`, or `*_ids` example from the tool schemas. Off in both Pipecat profiles |
| `delegation.replay_unheard_answer` and `prompt_features.replay_intent` | `delegation_agent.yaml` | M2 (experimental, off by default): replay an unheard answer on `request=status` in IDLE |
| `delegation.filler_dedupe` | `delegation_agent.yaml` | G1: drop or replace a repeated filler (`filler_alternatives`) |
| `output.clean_answers` | `delegation_agent.yaml` | G2: strip markdown and make one sentence per list item before TTS |
| `prompt_features.spelling_v2`, `spoken_output`, `write_consent` | `gateway.yaml` | M3.4, G2, and G4: backend prompt variants, all on. `gateway.baseline.yaml` turns all off, and `gateway.<name>.yaml` turns one on over it |
| `prompt_features.phone_format` | `gateway.yaml`, `gateway.eval.yaml` (off in `gateway.baseline.yaml` and `gateway.identity_control.yaml`) | I1: a domain-generic `<phone_numbers>` block in `backend_system`. Lookups by a ten-digit phone number use XXX-XXX-XXXX unless the tool description shows another format, with one retry in the other common form. Tools that store or change a number get it unchanged |
| `prompt_features.domain_notes` and `domains` | `gateway.yaml`, `gateway.eval.yaml`, `prompts.backend.yaml` (`backend_domain_notes`) | One telecom note, on by default. Telecom is detected by its tool `get_customer_by_phone`, and the note renders as `<domain_notes>` right after the policy in telecom sessions only. Off in the controls and in the ablation `gateway.eval_no_note.yaml` |
| `normalization.tool_arguments.result_hints` | `voice/tau3_fixes.yaml`, `voice/realtime_eval.yaml` (schema in `voice_agent.yaml`) | I3 and I4: after a failed lookup, append `lookup_not_found_hint` on the first miss and `lookup_not_found_hint_words` on later misses, at most 3 until a lookup succeeds. `voice/realtime_eval.yaml` uses `tools: auto` and fires only when the not-found text names a user, customer, member, client, or profile. Logged as `result_hint` events |

## Geval-Fix Switches

The switches below are on in `gateway.yaml`, `gateway.eval.yaml` (pinned), and `voice/realtime_eval.yaml`, and off in `voice_agent.yaml` and in the Python code. The control (`gateway.eval_control.yaml`, `voice/realtime_eval_control.yaml`) turns them off. Each `gateway.eval_<fix>.yaml` or `voice/realtime_eval_<fix>.yaml` turns one on over it. The following table lists the switches:

| Switch | Where | Fix |
| --- | --- | --- |
| `prompt_features.format_correction` | `gateway.yaml` | A2: inside the `spelling_v2` text, a change of separators or letter case only is a format correction, not a variant. Its voice half is `result_hints.message_key: lookup_not_found_hint_format` |
| `prompt_features.write_consent_v2` | `gateway.yaml` | B1: inside `<consent>`, the confirmation the policy requires, to the exact change, before a write |
| `prompt_features.policy_check`, `request_scope` | `gateway.yaml` | B2 and B3: a `<before_change>` block. B2 checks the policy for the exact change. B3 covers every record and option the request names |
| `prompt_features.transfer_check`, `capability_check` | `gateway.yaml` | C1 and C2: a `<before_transfer>` block. C1 transfers at once when the policy requires it. C2 checks the tool list before saying a request is impossible |
| `normalization.tool_arguments.format_retry` | `voice/realtime_eval_a1_format_retry.yaml`, `voice/realtime_eval_a1_lite_format_hint.yaml` (schema in `voice_agent.yaml`) | A1 and A1-lite: after a not-found lookup by a ten-digit phone field, the same digits in the other format, once per value. `mode: call` makes the relay look it up itself, but only on a tool whose schema declares `annotations: {readOnlyHint: true, idempotentHint: true}`. Otherwise, and always with `mode: hint`, the relay appends `format_retry_hint` |
| `normalization.tool_arguments.result_hints.field_provenance` | `voice/realtime_eval_a3_field_provenance.yaml` | A3: from the second person-lookup miss on, the note also says which values the user spelled, which were heard as a word, and which are unknown |

The phone fields come from the session's tool schemas: string arguments of read tools whose name contains `phone`, `mobile`, `msisdn`, or `telephone`, or whose description says "phone number." On the τ³ tools, only telecom `get_customer_by_phone.phone_number` qualifies, and no tool declares read-only semantics, so A1 runs as A1-lite there. The relay's own call has the ID `<id>.fmt`. The found, not-found, and hint texts are the `format_retry_found`, `format_retry_not_found`, and `format_retry_hint` keys of `config/prompts.voice.yaml`.

The gateway records a backend decision trace when `hermes.backend_trace` is on. The default is `${FDH_BACKEND_TRACE:-false}` in `gateway.yaml` and `${FDH_BACKEND_TRACE:-true}` in `gateway.eval.yaml`, so every arm records it. Each worker appends a `session` record, then a `request` and `response` pair per model call, to `<workers.log_dir>/<session_id>.trace.jsonl`.

Two replay tools check an arm without a live run:

- `cli/relay_replay.py` (G1) feeds every recorded tool call of a voice event log through the control relay and the arm relay, and exits 1 on an unexpected difference. It needs the session's tool schemas as `--session-update` files, such as [`tau2_session_update/telecom.json`](../../../tests/unit/frontend_delegation_hermes/tau/fixtures/tau2_session_update/telecom.json).
- `cli/backend_decision_replay.py` (G2) runs `extract` to list the write and transfer decisions of control traces for labelling, and `replay` to re-send them with an arm's `--gateway-config`.

## F1, F2, F4, and F8 Switches

These switches are on in `realtime_eval.yaml`, `voice/realtime_eval.yaml`, `gateway.yaml`, and `gateway.eval.yaml`, and off in the control `realtime_eval_124acf7.yaml` with `gateway.eval_124acf7.yaml` and in every older control. The following table lists them:

| Switch | Where | Fix |
| --- | --- | --- |
| `delegation.guards.{liveness_reply,question_never_silent,ping_as_status}` and `delegation.replay_unheard_answer.triggers` | `profiles/realtime_eval.yaml`, `profiles/realtime_eval_liveness.yaml` | F1 and F2: a silent reply to "are you still there?" or "could you repeat that?" is delegated or replaced by a replay of the unheard answer, and a ping while the backend works is a status request. Phrase lists are per ASR language, and only `en` ships |
| `prompt_features.situational_facts` | `gateway.yaml` | F4: when the policy makes a problem depend on where the user is and the user has not said, the backend asks once, early |
| `normalization.tool_arguments.result_hints.ledger` | `voice/realtime_eval.yaml`, `voice/realtime_eval_identity.yaml` | F8: after a failed identification lookup, the note lists the lookups tried, what the user spelled, the user's corrections, and the other identification lookups the tools offer |
| `prompt_features.identity_from_identifier` | `gateway.yaml` | F8 name line: a name inside a spoken identifier, after it is confirmed |

## Deployed Fingerprint

The voice server and the gateway log a deployed fingerprint: `fdh_session_start.features`, `frontend_prompt`, and `backend_configured`, which includes `backend_domain` and `session_tools_sha256`. `session_tools_sha256` is the first 16 hex characters of the SHA-256 of the session's sorted `session.update` tool schemas. With an `auto` switch on, the voice server also logs a `session_rules` event with the same hash and the derived rules.

`config_hash` values differ from the prototype's because the configuration paths moved. The prompt hashes and `session_tools_sha256` are unchanged. To check that every session of an arm ran the expected switches and prompts, run `fingerprint_check`:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.fingerprint_check \
  logs/fdh_voice_events.realtime_eval.jsonl \
  --profile src/examples/frontend_delegation_hermes/config/profiles/realtime_eval.yaml \
  --gateway-config src/examples/frontend_delegation_hermes/config/gateway.eval.yaml
```

## Run It

Use the Compose services through the [deploy skill](../../../.agents/skills/nemotron-voice-agent-fdh-deploy/SKILL.md). The following commands start each adapter after `nemo-speech` is up and `HERMES_AGENT_DIR` is set in `.env`:

```bash
F=docker/docker-compose.frontend-delegation-hermes.yaml
docker compose --profile frontend-backend-agent/single-gpu up -d nemo-speech
docker compose --env-file .env -f $F up -d --build fdh-voice                     # Pipecat demo, https://<host>:8776/
docker compose --env-file .env -f $F --profile tau up -d --build fdh-tau         # τ³, ws://<host>:8775/v1/realtime
docker compose -f $F --profile tau down                                          # stop both
```

You can also launch the voice servers directly on the host for development. They need a running gateway (refer to the host gateway appendix of the skill's `tau.md` reference) and the speech endpoint on the host. The following commands show the direct launch:

```bash
export FBA_ASR_SERVER=localhost:50051 FBA_TTS_SERVER=localhost:50051
# Pipecat adapter, demo
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.server \
  --config src/examples/frontend_delegation_hermes/config/profiles/pipecat_demo.yaml \
  --backend-url ws://127.0.0.1:8790/v1/backend --tls
# Pipecat adapter, eval
FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.server \
  --config src/examples/frontend_delegation_hermes/config/profiles/pipecat_realtime_eval.yaml \
  --backend-url ws://127.0.0.1:8790/v1/backend
# τ³ adapter
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
  --config src/examples/frontend_delegation_hermes/config/profiles/realtime_eval.yaml \
  --backend-url ws://127.0.0.1:8790/v1/backend
```

Both adapters accept `--host`, `--port`, `--max-sessions`, `--backend-url`, `--tls`, `--tls-cert`, `--tls-key`, and `--stub-backend` (an in-process fake backend and the rule-based frontend). The τ³ adapter also accepts `--stub-speech` (energy VAD, stub ASR, and tone TTS, no GPU). The no-GPU smoke run is:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
  --config src/examples/frontend_delegation_hermes/config/profiles/stub_gate.yaml \
  --stub-speech --stub-backend --port 8779
```

## Tests

The following command runs the unit tests on the host:

```bash
uv run pytest tests/unit/frontend_delegation_hermes -q
```

The suites cover the following areas:

- `core/` tests the shared code. `test_fdh_workflow_table.py` checks all 12 rows of `workflow.csv`. `test_fdh_gateway_processes.py` runs real worker processes with the fake agent. `test_fdh_golden_prompts.py` and `test_fdh_fix_plan.py` pin the baseline prompts and the `124acf7` control. `test_fdh_tau3_fixes.py`, `test_fdh_identity_fixes.py`, and `test_fdh_geval_fixes.py` cover the switches. `test_fdh_schema_rules.py` checks the derived rules against `fixtures/schema_rules_snapshot.json`.
- `tau/` tests the τ³ adapter, including the τ² contract and an end-to-end WebSocket run.
- `pipecat/` tests the Pipecat adapter with an energy VAD, a stub ASR, and a tone TTS.
- `test_fdh_import_isolation.py` checks that the core imports no adapter, that each adapter imports no other adapter, and that the τ³ adapter imports no Pipecat pipeline code.
- `test_fdh_dependencies.py` checks that the image lock pins the repository's Pipecat version and lists the same packages as the `fdh` extra.

`test_fdh_hermes_multisession.py` runs real Hermes workers against a fake OpenAI server. It is skipped when the Hermes interpreter (`FDH_HERMES_PYTHON`, default `~/.cache/fdh/hermes-venv-314/bin/python`) is missing. `test_fdh_hermes_adapter.py` needs the Hermes interpreter, and the skill's `tau.md` reference has the in-image command that runs every suite with real Hermes.

After an intended baseline change, regenerate the golden files with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden.py --write`.
