# τ³ Adapter

The τ³ adapter (`src/examples/frontend_delegation_hermes/tau_server.py`) is used only for τ³ benchmarking. It keeps the behavior the τ³ scores were measured with:

- an audio clock driven by the client's audio;
- Silero VAD with 800 ms end-of-turn silence (`honor_client_values: false`, so the client's 500 ms is ignored);
- tracking of sent versus heard audio;
- each transcript delta sent just before its audio.

It never imports the Pipecat adapter. The service is `fdh-tau`, started only with the Compose profile `tau`, on host port 8775.

Follow the sections in order. Complete `prerequisites.md` first, then set `F=docker/docker-compose.frontend-delegation-hermes.yaml` in every shell.

## 1. Topology

The following diagram shows what runs where:

```text
 τ² / Realtime ─ws─► fdh-tau (nemotron-voice-agent-fdh:latest, Python 3.14, host 8775 → 7860, client tools)
                      ├─ voice server :7860 (tau_server)
                      ├─ backend gateway 127.0.0.1:8790 (Hermes-free)
                      └─ one Hermes worker process per Realtime session
                           └─ AIAgent → Inference Hub nvidia/nvidia/nemotron-3-ultra (reasoning on)

 speech:       nemo-speech (root project, :50051, ASR + TTS)
 frontend LLM: Inference Hub nvidia/nvidia/nemotron-3.5-lightning (reasoning off)
```

The entrypoint `docker/fdh/entrypoint.sh` starts the gateway on the container's loopback with `--max-sessions` from `FDH_GATEWAY_MAX_SESSIONS` (default 8), waits until its `/health` answers, then starts the voice server. If either process exits, the container exits, and Docker restarts it (`restart: unless-stopped`). The gateway is not reachable from the host or the LAN. Only the worker processes import Hermes, because `src/utils.py` would shadow Hermes's own `utils` module.

The voice server reaches the gateway at `ws://host.docker.internal:8790/v1/backend`, the same string as a host run. Inside the container, `host.docker.internal` maps to `127.0.0.1`.

The default pair, `realtime_eval.yaml` with `gateway.eval.yaml`, uses:

- transcript normalization, including spelled-run joining (M3.1);
- the failure fixes M1 (proactive status), M3.2 (spelling hold), G1 (filler de-duplication), and G2 (answer cleanup), with M2 (replay on a status request) off;
- tool-argument rules, result-hint targets, and spelling-hold patterns derived from each session's tool schemas;
- the frontend tool `delegate(delegate, filler_text, request)`, with the filler spoken.

### Config Hash Note

`config_hash` values differ from the prototype's because the configuration paths moved. Pair runs by profile name, not by an old `config_hash`. The prompt hashes (`frontend_prompt`, `backend_system_sha256`, `backend_catalog_sha256`) and `session_tools_sha256` are unchanged.

## 2. Check That the Pipecat Adapter Is Stopped

A reportable τ³ run needs `fdh-voice` stopped, because both adapters share `nemo-speech` and the Inference Hub. The following commands check and stop it:

```bash
docker ps --filter name='^fdh-voice$' --format '{{.Names}} {{.Status}}'   # expect no output
docker compose -f $F stop fdh-voice                                        # only if it runs
```

## 3. Start fdh-tau

`FDH_PROFILE` selects the voice profile and `FDH_GATEWAY_CONFIG` the gateway config. `FDH_EVENT_LOG` names the voice event log. The default is `logs/fdh_voice_events.realtime_eval.jsonl`, which the τ³ campaign tooling expects for `realtime_eval.yaml` and its arms. `FDH_PROACTIVE_AFTER_S` sets the M1 delay. Calibrate it per host (refer to section 9).

Every setting is a `${VAR:-default}` entry in the Compose file. A value exported in the shell wins over `.env`, which wins over the default. `up -d` recreates the container when a value changes, so the same command switches profiles and arms.

```bash
export FDH_PROFILE=${FDH_PROFILE:-realtime_eval.yaml} FDH_GATEWAY_CONFIG=${FDH_GATEWAY_CONFIG:-gateway.eval.yaml}
docker compose --env-file .env -f $F --profile tau up -d --build fdh-tau   # --build only for a rebuild trigger
sleep 30; curl -s localhost:8775/health; echo
docker logs fdh-tau 2>&1 | grep -E 'backend gateway|warm-up|VAD|config hash'
```

The defaults are voice `FDH_MAX_SESSIONS=4`, gateway `FDH_GATEWAY_MAX_SESSIONS=8`, `PIPELINE_TLS=false`, and the frontend and backend models on the Inference Hub.

Expected `/health`:

- `"status": "ok"`, `"prototype": "frontend-delegation-hermes"`, and `"backend": {"link": "websocket", ...}`;
- `"features"` with `realtime_eval.yaml`: `proactive_status`, `clean_answers`, `spelling_hold`, `filler_dedupe`, `spelled_runs`, `result_hints`, `schema_rules`, `schema_result_hint_tools`, `schema_spelling_hold_patterns`, `schema_format_retry_fields`, and `replay_unheard_answer` (F1) are `true`, `format_retry` is `"call"`, and `field_provenance` (A3) and `prompt_features.replay_intent` are `false`;
- with `tau3_eval.yaml`, the first six of those switches are `true`, without the schema-derived switches;
- with `tau3_eval_baseline.yaml`, every switch is `false`. With an arm profile, only the arm's own switches are `true` (`format_retry` is `"call"` or `"hint"` in the A1 arms).

Expected logs are `frontend warm-up done in …s`, `backend gateway http://host.docker.internal:8790/health: {... 'ok': True ...}`, and `VAD silence=800ms`. The warm-up opens the HTTPS connection to the frontend endpoint at startup, so the first user turn of a fresh server does not time out.

Keep the voice server's `FDH_MAX_SESSIONS` (4) below the gateway's `FDH_GATEWAY_MAX_SESSIONS` (8). A voice server refuses to start when its limit exceeds the gateway's. The gateway frees a slot only after the worker stops (up to 15 s), so equal limits can refuse a back-to-back τ³ session with `capacity`.

## 4. Check the Gateway

The following command queries the gateway inside the container:

```bash
docker exec fdh-tau curl -s 127.0.0.1:8790/health; echo
```

Expected health:

- `"ok": true`, `"max_sessions": 8`, and `"agent_kind": "hermes"`;
- `"hermes"` with `"model": "nvidia/nvidia/nemotron-3-ultra"`, `"base_url": "https://inference-api.nvidia.com/v1"`, `"reasoning": true`, and `"backend_trace"` (`true` with `gateway.eval.yaml`, `gateway.eval_control.yaml`, and the arms, unless `FDH_BACKEND_TRACE=false`);
- `"backend_features"` with every backend variant `true` for `gateway.yaml` and `gateway.eval.yaml`, all `false` for `gateway.baseline.yaml`, and one variant `true` for a single-variant arm;
- `"domains"` listing telecom, detected by `get_customer_by_phone`, and `"backend_catalog_sha256"`, the hash of `prompts.backend.yaml`.

The gateway loads its code and prompts at startup. Recreate `fdh-tau` after a code or prompt change, or to switch `FDH_GATEWAY_CONFIG`.

The following optional check drives the gateway directly with a real Hermes worker, text only, in about 30 s. It covers a new task with a tool call, a continue, a status request while WORKING, and a steer during a slow tool:

```bash
docker exec fdh-tau python -m examples.frontend_delegation_hermes.cli.backend_probe \
  --url ws://127.0.0.1:8790/v1/backend
```

## 5. Smoke Tests

### Text Smoke Test

The client registers the demo tools and executes them itself, as τ³ does. It exercises the frontend decision, the spoken filler, the delegation, the Hermes tool call over the Realtime wire, and the spoken answer:

```bash
( sleep 2; echo "can you check the status of order 5512"; sleep 25; echo "okay thanks"; sleep 6; \
  echo "what about order 5513"; sleep 3; echo "what's the update so far"; sleep 25 ) | \
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.voice_chat \
  --url ws://localhost:8775/v1/realtime --model pine-fdh-smoke --io text --ui plain \
  --tools examples.frontend_delegation_hermes.core.text.demo_tools:TOOLS
```

This client uses manual-response mode: every line is followed by `response.create`. The following list shows the expected order:

1. a filler, such as "Sure, let me check that...";
2. a `get_order` function call answered by the client;
3. the order status;
4. nothing for "okay thanks";
5. a filler for order 5513;
6. a spoken status update for "what's the update so far" while the backend still works, or an answer from history after it finishes.

The following command shows each step in the event log:

```bash
grep -E '"kind": "(delegation_decision|backend_action|tool_output_in|backend_answer|status_spoken)"' \
  logs/fdh_voice_events.realtime_eval.jsonl
```

### Audio Smoke Test

This test uses the real ASR, VAD, and TTS. It synthesizes three user turns with the local TTS and plays them in τ³'s wire format (μ-law, 8 kHz). Expect the same flow as the text test, with the user lines transcribed by the ASR:

```bash
mkdir -p /tmp/fdh-wav && uv run python - <<'PY'
import wave, riva.client
tts = riva.client.SpeechSynthesisService(riva.client.Auth(uri="localhost:50051", use_ssl=False))
for i, text in enumerate(["Can you give me the status of order five five one two?", "Okay, thanks.",
                          "What about order five five one three?"], 1):
    audio = tts.synthesize(text, voice_name="", language_code="en-US", sample_rate_hz=22050).audio
    with wave.open(f"/tmp/fdh-wav/u{i}.wav", "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050); w.writeframes(audio)
PY
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.voice_chat \
  --url ws://localhost:8775/v1/realtime --model pine-fdh-audio --io wav --format pcmu \
  --in /tmp/fdh-wav/u1.wav --in /tmp/fdh-wav/u2.wav --in /tmp/fdh-wav/u3.wav --out /tmp/fdh-wav/agent.wav \
  --ui plain --tools examples.frontend_delegation_hermes.core.text.demo_tools:TOOLS
```

## 6. Start a Second Container

A second τ³ container, such as the `silentack` arm on port 8777, runs next to `fdh-tau`. Give it its own container name, host port, Compose project name (`-p`), and log files. Without `-p`, the second `up` replaces the first container. Prefix the variables instead of exporting them, so the main container's variables stay untouched:

```bash
FDH_CONTAINER_NAME=fdh-tau-silentack FDH_VOICE_HOST_PORT=8777 \
FDH_PROFILE=tau3_eval_silent_ack.yaml FDH_GATEWAY_CONFIG=gateway.yaml \
FDH_EVENT_LOG=logs/fdh_voice_silentack_events.jsonl FDH_GATEWAY_LOG=logs/fdh_gateway_events.silentack.jsonl \
  docker compose --env-file .env -p fdh-silentack -f $F --profile tau up -d fdh-tau
curl -s localhost:8777/health; echo
docker exec fdh-tau-silentack curl -s 127.0.0.1:8790/health; echo
```

Each container runs its own gateway and workers, so both gateway limits apply separately. Stop the second container with the same project name: `docker compose -p fdh-silentack -f $F --profile tau down`.

## 7. Point τ³ at the Server

Run this command in the τ² checkout, after its own setup (`uv sync --extra voice --extra dev` and the Inference Hub variables). Model tags must start with `pine-`. Use `--max-concurrency 1` for the smoke run:

```bash
IHUB=https://inference-api.nvidia.com/v1
RUN=fdh_voice_dlg_mock_control_smoke
PINE_REALTIME_BASE_URL=ws://localhost:8775/v1/realtime PINE_API_KEY=unused \
uv run python misc/prototypes/fba_voice_eval/tau2_ihub.py run --domain mock --audio-native \
  --audio-native-provider openai --audio-native-model pine-fdh-voice-dlg-mock-control-smoke \
  --speech-complexity control --user-llm openai/azure/openai/gpt-5.2 \
  --user-llm-args "{\"temperature\": 0.0, \"api_base\": \"$IHUB\"}" --review-model openai/azure/openai/gpt-5.2 \
  --task-split-name base --num-trials 1 --max-concurrency 1 --num-tasks 1 --save-to "$RUN" --verbose-logs
```

The τ² fork's campaign helper `fdh_run <arm> <domain> regular` in `misc/prototypes/fdh_voice_eval/fdh_lib.sh` builds the same command with the arm's port and a `pine-fdh-voice-<arm>-*` tag. It must use the `fdh-tau` container names.

### Gate A

Gate A runs τ²'s real provider against the server with one tool call, without a full run. Against the real server it needs audio the ASR transcribes, because it streams a tone. Use the stub server (section 14) for Gate A. Run it in the τ² checkout:

```bash
PINE_REALTIME_BASE_URL=ws://localhost:8775/v1/realtime PINE_API_KEY=unused \
  uv run python <path-to-this-repository>/src/examples/frontend_delegation_hermes/cli/tau2_gates/gate_a_handshake.py
# last line: GATE A PASSED
```

## 8. Evaluation Arms

An arm is a voice profile (`FDH_PROFILE`) plus a gateway config (`FDH_GATEWAY_CONFIG`). Each arm turns on one fix over its control and is compared with a fresh control on the same host. To switch arms, export the two variables and run the section 3 command again. Use one event log per arm and domain, because the prompt hashes include the τ² policy and tools.

```bash
export FDH_PROFILE=realtime_eval_arm_a2_format_correction.yaml FDH_GATEWAY_CONFIG=gateway.eval_a2_format_correction.yaml
export FDH_EVENT_LOG=logs/fdh_voice_events.realtime_eval.jsonl FDH_GATEWAY_LOG=logs/fdh_gateway_events.realtime_eval.jsonl
docker compose --env-file .env -f $F --profile tau up -d fdh-tau
```

### 8.1 Campaign Arms

The τ² campaign tooling names three arms. The following table lists them:

| Arm | Container and Port | `FDH_PROFILE` | `FDH_GATEWAY_CONFIG` | Event Log |
| --- | --- | --- | --- | --- |
| `geval` (default) | `fdh-tau`, 8775 | `realtime_eval.yaml` | `gateway.eval.yaml` | `logs/fdh_voice_events.realtime_eval.jsonl` |
| `dlg` (historical) | `fdh-tau`, 8775 | `tau3_eval.yaml` | `gateway.yaml` | `logs/fdh_voice_events.jsonl` |
| `silentack` (ablation) | `fdh-tau-silentack`, 8777 (section 6) | `tau3_eval_silent_ack.yaml` | `gateway.yaml` | `logs/fdh_voice_silentack_events.jsonl` |

`geval` and `dlg` share port 8775, so only one of them runs at a time. `silentack` does not speak the filler of a delegated turn. Direct replies are still spoken.

### 8.2 Failure-Fix and Identity-Fix Arms

These arms pair `tau3_*` profiles. `tau3_eval_baseline.yaml` must run with `gateway.baseline.yaml`, and `tau3_identity_control.yaml` with `gateway.identity_control.yaml`. The following table lists the pairs:

| Arm | `FDH_PROFILE` | `FDH_GATEWAY_CONFIG` | Turns On |
| --- | --- | --- | --- |
| Default (every fix except M2) | `tau3_eval.yaml` | `gateway.yaml` | M1, M3, G1, G2, G4, I1, I3, and I4 together. Not an evaluated arm |
| Control | `tau3_eval_baseline.yaml` | `gateway.baseline.yaml` | Nothing. Prompts byte-identical to agent commit `3a7e04a` |
| M1 proactive status | `tau3_arm_m1_status.yaml` | `gateway.baseline.yaml` | One short status line per run after `FDH_PROACTIVE_AFTER_S` (default 15) seconds of silence while WORKING |
| M3 spelling | `tau3_arm_m3_spelling.yaml` | `gateway.spelling_v2.yaml` | Spelled-run joining, the spelling hold, escalating local "invalid ID" wording, and backend variant `spelling_v2` |
| Airline canary (M1 and M3) | `tau3_airline_canary.yaml` | `gateway.spelling_v2.yaml` | M1 and M3 together, after each passed its own arm |
| M2 replay (experimental) | `tau3_arm_m2_replay.yaml` | `gateway.baseline.yaml` | Replays an unheard answer on a status request in IDLE, with frontend variant `replay_intent` |
| G1 filler de-duplication | `tau3_arm_g1_filler_dedupe.yaml` | `gateway.baseline.yaml` | Drops or replaces a repeated filler |
| G2 short answers | `tau3_arm_g2_short_answers.yaml` | `gateway.spoken_output.yaml` | Markdown cleanup before TTS and backend variant `spoken_output` |
| G4 write consent | `tau3_eval_baseline.yaml` | `gateway.write_consent.yaml` | Backend variant `write_consent` |
| Identity control | `tau3_identity_control.yaml` | `gateway.identity_control.yaml` | The default without I1, I3, and I4 |
| I1 phone-number rule | `tau3_identity_control.yaml` | `gateway.yaml` | Backend variant `phone_format`: a `<phone_numbers>` block in every session |
| I3 recovery note | `tau3_arm_i3_recovery_hint.yaml` | `gateway.identity_control.yaml` | The I3 note after a failed lookup of `find_user_id_by_name_zip`, `find_user_id_by_email`, or `get_user_details` |
| I4 word-assisted spelling | `tau3_arm_i4_word_spelling.yaml` | `gateway.identity_control.yaml` | I3 on the first miss and the I4 note ("S as in Sam") on later misses. Compare it with the I3 arm |

For I1, I3, and I4, run I1 on telecom and I3 or I4 on airline and retail, each against a fresh identity control. In `backend_configured`, `session_tools_sha256` must be the same for every session of one domain. `result_hint` events (`message_key`, `n`) show each appended note.

### 8.3 Domain-Agnostic Profile

`realtime_eval.yaml` with `gateway.eval.yaml` is the single frozen profile for every domain. No setting names a tool or a domain, except the disclosed telecom note's detection rule. The voice profile `voice/realtime_eval.yaml` derives the following from the tool schemas in the client's `session.update`:

- **Tool-argument rules** (`normalization.tool_arguments.rules: auto`): for read tools only, each `id` or `*_id` string argument whose description quotes an example gets a rule from that example. Write tools are never rewritten.
- **Result-hint targets** (`result_hints.tools: auto`): every read tool. A note is appended only when the not-found text names a user, customer, member, client, or profile.
- **Spelling-hold patterns** (`delegation.spelling_hold.complete_patterns: auto`): the strict shapes of the `id`, `*_id`, and `*_ids` examples, matched case-insensitively. Examples with a name-like part give no pattern.

The retry guard covers every tool with the pattern `(?i)\b(not found|no such|does not exist)\b`. In the event log, each session has a `session_rules` event with `session_tools_sha256`, the `schema_auto` switches, and the derived rules. Review them per domain.

### 8.4 Geval-Fix Arms

Every geval fix is on in the default pair and off in the control, `realtime_eval_control.yaml` with `gateway.eval_control.yaml`. The following table lists the arms:

| Arm | `FDH_PROFILE` | `FDH_GATEWAY_CONFIG` | Turns On |
| --- | --- | --- | --- |
| Control | `realtime_eval_control.yaml` | `gateway.eval_control.yaml` | Nothing new. The backend decision trace (P0.3) is on |
| A1 format retry | `realtime_eval_arm_a1_format_retry.yaml` | `gateway.eval_control.yaml` | `format_retry` `mode: call`. On τ³, no tool declares read-only semantics, so every tool gets the hint |
| A1-lite format hint | `realtime_eval_arm_a1_lite_format_hint.yaml` | `gateway.eval_control.yaml` | `format_retry` `mode: hint`. The relay never calls a tool itself |
| A2 format correction | `realtime_eval_arm_a2_format_correction.yaml` | `gateway.eval_a2_format_correction.yaml` | Backend variant `format_correction` and the recovery note `lookup_not_found_hint_format` |
| A3 field provenance | `realtime_eval_arm_a3_field_provenance.yaml` | `gateway.eval_control.yaml` | `result_hints.field_provenance` |
| B1 consent to the exact change | `realtime_eval_control.yaml` | `gateway.eval_b1_write_consent_v2.yaml` | Backend variant `write_consent_v2` |
| B2 policy check | `realtime_eval_control.yaml` | `gateway.eval_b2_policy_check.yaml` | Backend variant `policy_check` |
| B3 request scope | `realtime_eval_control.yaml` | `gateway.eval_b3_request_scope.yaml` | Backend variant `request_scope` |
| C1 transfer check | `realtime_eval_control.yaml` | `gateway.eval_c1_transfer_check.yaml` | Backend variant `transfer_check` |
| C2 capability check | `realtime_eval_control.yaml` | `gateway.eval_c2_capability_check.yaml` | Backend variant `capability_check` |
| All fixes (default) | `realtime_eval.yaml` | `gateway.eval.yaml` | A1 (`mode: call`), A2, A3, B1, B2, B3, C1, and C2 together |

### 8.5 F1, F2, F4, and F8 Arms

The control for these arms is `realtime_eval_124acf7.yaml` with `gateway.eval_124acf7.yaml`, which renders and decides exactly as agent `124acf7`. Every arm adds one fix to it. The following table lists the arms:

| Arm | `FDH_PROFILE` | `FDH_GATEWAY_CONFIG` | Turns On |
| --- | --- | --- | --- |
| All fixes (default) | `realtime_eval.yaml` | `gateway.eval.yaml` | F1 and F2, F4, F8, and the F8 name line together |
| Control | `realtime_eval_124acf7.yaml` | `gateway.eval_124acf7.yaml` | Nothing new |
| Liveness (F1 and F2) | `realtime_eval_liveness.yaml` | `gateway.eval_124acf7.yaml` | `delegation.guards.{liveness_reply, question_never_silent, ping_as_status}`, and `replay_unheard_answer` with `triggers: [backchannel, ping, repeat]`, `ttl_s: 60`, and `whole_if_heard_below: 0.2` |
| Situational (F4) | `realtime_eval_124acf7.yaml` | `gateway.eval_situational.yaml` | Backend variant `situational_facts` |
| Identity (F8) | `realtime_eval_identity.yaml` | `gateway.eval_124acf7.yaml` | `result_hints.ledger` (`identification_tools: auto`), `max_hints: 6`, and A3's `field_provenance` off |
| Identity name (F8 sub-arm) | `realtime_eval_124acf7.yaml` | `gateway.eval_identity_name.yaml` | Backend variant `identity_from_identifier` |

If an arm does not pass, turn its switch off in the default files (`profiles/realtime_eval.yaml`, `voice/realtime_eval.yaml`, `gateway.yaml`, and `gateway.eval.yaml`) and re-run the default. Check these events in an arm's log:

- **Liveness:** `delegation_guard` with `guard` `liveness_reply`, `question_never_silent`, or `ping_as_status`, and `answer_replayed` or `answer_replay_skipped` with `trigger`. A `lexicon_missing` event means the session's ASR language has no phrase list, so every F1 and F2 rule is inactive.
- **Identity:** `session_rules.identification_tools` (a bill or usage lookup must not be there) and `identity_ledger` per identification output.
- **Situational and identity name:** the gateway `/health` and `backend_configured.backend_features` show the variant.

### 8.6 Telecom Note Ablation

The default pair runs a telecom domain note. Its ablation changes only `domain_notes`. The following table lists the two runs:

| Run | `FDH_PROFILE` | `FDH_GATEWAY_CONFIG` | Telecom Note |
| --- | --- | --- | --- |
| Note on (default) | `realtime_eval.yaml` | `gateway.eval.yaml` | On |
| Note off (ablation) | `realtime_eval.yaml` | `gateway.eval_no_note.yaml` | Off |

Check the following in each run:

- The gateway `/health` shows `backend_features.domain_notes` `true` or `false`, and `domains` lists telecom with `get_customer_by_phone`.
- In telecom sessions, `backend_configured.backend_domain` is `"telecom"`. With the note on, the backend trace's system prompt has one `<domain_notes>` block right after `</policy>`, and `backend_system_sha256` differs between the runs.
- In every other domain, `backend_domain` is `""`, and `backend_system_sha256` is the same in both runs.

The note is disclosed word for word with a submission. Any change to its text needs a new held-out validation on telecom tasks outside the scored `base` split.

## 9. Calibrations and Re-Measures

Run these checks before the arms they gate.

**M1: calibrate `FDH_PROACTIVE_AFTER_S` on the target host.** From a control run, take the wall-clock and τ² simulated gaps of WORKING spans with no audio. The status line must play at about 20 s simulated or earlier, and not before the 75th percentile of normal filler-to-answer gaps.

**M3: re-measure the spelling hold** on the run's own events. The canary is blocked if the held or continued counts move by more than 10%:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.spelling_hold_replay \
  <events.jsonl> --config src/examples/frontend_delegation_hermes/config/profiles/tau3_arm_m3_spelling.yaml
```

**M2: frontend decisions.** `delegation_replay` runs three more cases (`replay_unheard|IDLE`) when the profile turns `replay_intent` on:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.delegation_replay \
  --config src/examples/frontend_delegation_hermes/config/profiles/tau3_arm_m2_replay.yaml --only replay
```

## 10. Fingerprint Check

Before you score an arm, check its deployed fingerprint. Every session must have the arm's switches, the gateway's variants, and one set of prompt hashes. Exit status 1 means you re-run the arm instead of scoring it:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.fingerprint_check \
  $FDH_EVENT_LOG \
  --profile src/examples/frontend_delegation_hermes/config/profiles/$FDH_PROFILE \
  --gateway-config src/examples/frontend_delegation_hermes/config/$FDH_GATEWAY_CONFIG
# last line: fingerprint: OK
```

Use one event log per arm and domain, for example `logs/fdh_voice_events.realtime_eval.<domain>.jsonl`. Export `FDH_EVENT_LOG` before the section 3 command.

## 11. Gates

Run the gates in this order. A fix that fails one is not shipped:

- **P0.1, a clean run:** the τ³ `report.sh` marks a run `INVALID` when any of its sessions hit a backend 429 (`Error code: 429` or `Retrying API call` in `logs/fdh_workers/<session>-<n>.log`). Re-run an invalid run, preferably in quiet hours.
- **G0:** `uv run pytest tests/unit/frontend_delegation_hermes/core/test_fdh_geval_fixes.py -q`.
- **G1, the code paths (A1 and A1-lite):** replay a clean control's recorded tool calls through the arm's relay. Every found-after-retry result is printed for the identity check. A3 needs the user turns, so review the `field_provenance` events of an A3 run against a hand-labelled sample instead.
- **G2, the backend decisions (A1-lite, A2, A3, and B1 through C2):** list the write and transfer decisions of a clean control's traces, label them **before** any arm is replayed (`U` unchanged, `P` expected policy correction, `T` targeted, `skip`), then replay them with the arm's backend prompt (3 samples at temperature 0). `--dry-run` only checks that the arm's prompt replaces the control's in every request.
- **G3 and G4:** the mock domain and invented conversations (G3), then the paired A/B comparison (G4: an A/A control run first, five trials per domain, the arm interleaved with the control). Check every arm's fingerprint (section 10).

The following commands run G1 and G2:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.relay_replay \
  logs/fdh_voice_events.realtime_eval.jsonl \
  --profile src/examples/frontend_delegation_hermes/config/profiles/realtime_eval_arm_a1_format_retry.yaml \
  --session-update tests/unit/frontend_delegation_hermes/tau/fixtures/tau2_session_update/{airline,retail,telecom}.json \
  --out logs/g1_a1.json                                    # last line: G1: OK
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_decision_replay \
  extract logs/fdh_workers --out logs/g2_decisions.jsonl   # then edit the labels
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_decision_replay \
  replay logs/g2_decisions.jsonl --traces logs/fdh_workers \
  --gateway-config src/examples/frontend_delegation_hermes/config/gateway.eval_b1_write_consent_v2.yaml \
  --out logs/g2_b1.json                                    # last line: G2: OK
```

Keep each control run's traces apart by moving `logs/fdh_workers/*.trace.jsonl` aside per run. A trace is named by session, and the replay reads every trace in the directory. The G2 replay compares a `U` decision's complete, ordered tool-call list.

The offline frontend decision check (gate G3, live frontend model, about 1 minute, 51 cases) runs as follows:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.delegation_replay \
  --concurrency 2 --out logs/fdh_delegation_replay.json
```

## 12. Report

`fba_voice_metrics.py` expects one backend operation per turn. Convert the event log first. The adapter credits each run to the turn that started it and exits 1 on an orphaned tool call:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.report_adapter \
  logs/fdh_voice_events.realtime_eval.jsonl --out logs/fdh_voice_events.legacy.jsonl
```

## 13. Logs

The following table lists the log files of `fdh-tau` with its defaults:

| File | Content |
| --- | --- |
| `logs/fdh_voice_events.realtime_eval.jsonl` (`FDH_EVENT_LOG`) | Voice server events per session: `fdh_session_start` (config hash, `features`), `frontend_prompt` (prompt hash), `backend_configured` (backend fingerprint, `backend_domain`, `session_tools_sha256`), `delegation_decision`, `backend_action`, `tool_calls_out` (with `call_ids` and `arguments`), `tool_output_in`, `call_answered_locally`, `backend_answer`, `status_spoken`, `playback_outcome`, `delivery_note`, `history_sync`, `turn_latency`, `filler_timing`, and `barge_in`. With a fix on: `status_proactive`, `spelling_hold`, `answer_replayed`, `filler_deduped`, `answer_cleaned`, `result_hint`, `session_rules`, `format_retry`, `field_provenance`, `delegation_guard`, and `identity_ledger` |
| `logs/fdh_gateway_events.realtime_eval.jsonl` (`FDH_GATEWAY_LOG`) | Gateway: `gateway_start` (`backend_features`, `backend_catalog_sha256`), sessions, `backend_fingerprint`, worker spawn, ready, and exit (`start_ms`, `rss_mb`), run epochs and outcomes, context delivery states, watchdog, and respawns |
| `logs/fdh_workers/*.log` (`FDH_WORKER_LOG_DIR`) | Each Hermes worker's output. "Auxiliary Nous client unavailable" lines are harmless. `Error code: 429` or `Retrying API call` makes the run invalid (P0.1) |
| `logs/fdh_workers/<session>.trace.jsonl` | With `hermes.backend_trace` on: one `session` record (tool schemas, the rendered `backend_system`), then a `request` and a `response` per model call, joined by `api_request_id`. The input of the G2 replay. The file grows with each model call |
| `docker logs fdh-tau` | Voice server and gateway output (latency breakdowns and errors) |

Files written by the container are owned by root on the host. Run `sudo chown -R "$USER" logs` before you move or delete them.

## 14. No-GPU Stub Mode

Stub mode runs without speech, an LLM, or Hermes: stub ASR and TTS, the rule-based frontend, and an in-process fake backend. `stub_gate.yaml` makes the fake backend call the first tool on every run, so Gate A sees a function call. Run it on the host:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
  --config src/examples/frontend_delegation_hermes/config/profiles/stub_gate.yaml \
  --stub-speech --stub-backend --port 8779 &
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.tau2_replay \
  --url ws://localhost:8779/v1/realtime --synthetic-turns 3 --pause-s 1   # last line: tau2 contract: OK
```

Then run Gate A in the τ² checkout:

```bash
PINE_REALTIME_BASE_URL=ws://localhost:8779/v1/realtime PINE_API_KEY=unused \
  uv run python <path-to-this-repository>/src/examples/frontend_delegation_hermes/cli/tau2_gates/gate_a_handshake.py
# last line: GATE A PASSED
```

Stop the stub server with `kill %1` when you are done.

## 15. Tests

The following command runs the unit tests on the host:

```bash
uv run pytest tests/unit/frontend_delegation_hermes -q
```

`test_fdh_fix_plan.py` checks that the control `realtime_eval_124acf7.yaml` with `gateway.eval_124acf7.yaml` renders and decides exactly as agent `124acf7`. Regenerate its golden file only for an intended change of the deployed behavior with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden_eval.py --write`. `test_fdh_golden_prompts.py` checks that, with every prompt variant off, every prompt is byte-identical to agent commit `3a7e04a`. Regenerate it only for an intended baseline change with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden.py --write`.

The following command runs the same suites in the image, with its Python 3.14 and real Hermes. The fixtures live in the example, so only `tests/` and `pyproject.toml` are mounted. The Hermes adapter test runs with `unittest` and the worker's `examples`-only `PYTHONPATH`, because `src/utils.py` shadows Hermes's `utils` under pytest:

```bash
docker compose --env-file .env -f $F --profile tau run --rm --no-deps --entrypoint "" \
  -v "$PWD/tests:/app/tests:ro" -v "$PWD/pyproject.toml:/app/pyproject.toml:ro" \
  -e FDH_HERMES_PYTHON=/app/.venv/bin/python \
  fdh-tau sh -c 'uv pip install --python /app/.venv/bin/python pytest &&
    PYTHONPATH=src python -m pytest tests/unit/frontend_delegation_hermes -q \
      --deselect tests/unit/frontend_delegation_hermes/core/test_fdh_hermes_adapter.py &&
    mkdir -p /tmp/fdh-workers/pythonpath && ln -sfn /app/src/examples /tmp/fdh-workers/pythonpath/ &&
    PYTHONPATH=/tmp/fdh-workers/pythonpath:tests/unit/frontend_delegation_hermes/core python -m unittest test_fdh_hermes_adapter'
```

## 16. Stop and Clean Up

Always pass `--profile tau`, or `fdh-tau` keeps running. Take the FDH containers down before the root project, because a root `down` cannot remove `nemotron-voice-agent_default` while an FDH container is attached ("active endpoints"):

```bash
docker compose -f $F --profile tau down                            # stops gateways and workers (up to 75 s)
docker compose -p fdh-silentack -f $F --profile tau down           # if a second container runs (section 6)
kill "$(cat logs/fdh_gateway.pid)"                                 # if the host gateway runs (appendix)
docker compose --profile frontend-backend-agent/single-gpu stop nemo-speech   # optional
```

Never commit runtime state: `.cache/fdh-hermes-homes/` (per-worker `HERMES_HOME` of the host gateway), `/tmp/fdh-workers/` (sockets), and `logs/`.

## Appendix: Host Gateway

Use a host gateway only for development, for example to run the voice server outside Docker or to probe Hermes from the host. It needs the host Hermes environment and `DOCKER_HOST_IP` from `prerequisites.md`. It has no authentication, so bind it to the `docker0` address or `127.0.0.1`, never to the LAN. Give it its own logs, so it never appends to a container's gateway log:

```bash
export FDH_GATEWAY_CONFIG=gateway.yaml
nohup env PYTHONPATH=src \
  FDH_GATEWAY_HOST=$DOCKER_HOST_IP FDH_GATEWAY_PORT=8790 FDH_MAX_SESSIONS=8 \
  FDH_HERMES_PYTHON=${FDH_HERMES_VENV:?set it in prerequisites.md}/bin/python \
  BACKEND_LLM_MODEL=nvidia/nvidia/nemotron-3-ultra BACKEND_LLM_BASE_URL=https://inference-api.nvidia.com/v1 \
  FDH_GATEWAY_LOG=logs/fdh_gateway_events.host.jsonl FDH_WORKER_LOG_DIR=logs/fdh_workers_host \
  uv run python -m examples.frontend_delegation_hermes.core.sidecar.gateway_server \
    --config src/examples/frontend_delegation_hermes/config/$FDH_GATEWAY_CONFIG \
  > logs/fdh_gateway.out 2>&1 &
echo $! > logs/fdh_gateway.pid
sleep 3; curl -s http://$DOCKER_HOST_IP:8790/health; echo
```

The expected health is the same as in section 4. The backend probe runs from the host with `PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.backend_probe --url ws://$DOCKER_HOST_IP:8790/v1/backend`. A host voice server reaches this gateway with `--backend-url ws://$DOCKER_HOST_IP:8790/v1/backend`. Restart the gateway after a code or prompt change, or to switch `FDH_GATEWAY_CONFIG`. If `kill` leaves the gateway running because the PID was only the `uv` wrapper, run `pkill -f sidecar.gateway_server`.
