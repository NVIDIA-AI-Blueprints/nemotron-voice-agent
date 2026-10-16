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

`FDH_PROFILE` selects the voice profile and `FDH_GATEWAY_CONFIG` the gateway config. `FDH_EVENT_LOG` names the voice event log. The default is `logs/fdh_voice_events.realtime_eval.jsonl`, which the τ³ campaign tooling expects for `realtime_eval.yaml`. `FDH_PROACTIVE_AFTER_S` sets the M1 delay. Calibrate it per host (refer to section 8).

Every setting is a `${VAR:-default}` entry in the Compose file. A value exported in the shell wins over `.env`, which wins over the default. `up -d` recreates the container when a value changes, so the same command switches profiles.

```bash
export FDH_PROFILE=${FDH_PROFILE:-realtime_eval.yaml} FDH_GATEWAY_CONFIG=${FDH_GATEWAY_CONFIG:-gateway.eval.yaml}
docker compose --env-file .env -f $F --profile tau up -d --build fdh-tau   # --build only for a rebuild trigger
sleep 30; curl -s localhost:8775/health; echo
docker logs fdh-tau 2>&1 | grep -E 'backend gateway|warm-up|VAD|config hash'
```

The defaults are voice `FDH_MAX_SESSIONS=4`, gateway `FDH_GATEWAY_MAX_SESSIONS=8`, `PIPELINE_TLS=false`, and the frontend and backend models on the Inference Hub.

Expected `/health`:

- `"status": "ok"`, `"prototype": "frontend-delegation-hermes"`, and `"backend": {"link": "websocket", ...}`;
- `"features"` with `realtime_eval.yaml`: `proactive_status`, `clean_answers`, `spelling_hold`, `filler_dedupe`, `spelled_runs`, `result_hints`, `schema_rules`, `schema_result_hint_tools`, `schema_spelling_hold_patterns`, `schema_format_retry_fields`, and `replay_unheard_answer` (F1) are `true`, `format_retry` is `"call"`, and `field_provenance` (A3) and `prompt_features.replay_intent` are `false`.

Expected logs are `frontend warm-up done in …s`, `backend gateway http://host.docker.internal:8790/health: {... 'ok': True ...}`, and `VAD silence=800ms`. The warm-up opens the HTTPS connection to the frontend endpoint at startup, so the first user turn of a fresh server does not time out.

Keep the voice server's `FDH_MAX_SESSIONS` (4) below the gateway's `FDH_GATEWAY_MAX_SESSIONS` (8). A voice server refuses to start when its limit exceeds the gateway's. The gateway frees a slot only after the worker stops (up to 15 s), so equal limits can refuse a back-to-back τ³ session with `capacity`.

## 4. Check the Gateway

The following command queries the gateway inside the container:

```bash
docker exec fdh-tau curl -s 127.0.0.1:8790/health; echo
```

Expected health:

- `"ok": true`, `"max_sessions": 8`, and `"agent_kind": "hermes"`;
- `"hermes"` with `"model": "nvidia/nvidia/nemotron-3-ultra"`, `"base_url": "https://inference-api.nvidia.com/v1"`, `"reasoning": true`, and `"backend_trace"` (`true` with `gateway.eval.yaml`, unless `FDH_BACKEND_TRACE=false`);
- `"backend_features"` with every backend variant `true` for `gateway.yaml` and `gateway.eval.yaml`;
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

A second τ³ container, such as one on port 8777, runs next to `fdh-tau`. Give it its own container name, host port, Compose project name (`-p`), and log files. Without `-p`, the second `up` replaces the first container. Prefix the variables instead of exporting them, so the main container's variables stay untouched:

```bash
FDH_CONTAINER_NAME=fdh-tau-2 FDH_VOICE_HOST_PORT=8777 \
FDH_EVENT_LOG=logs/fdh_voice_events.2.jsonl FDH_GATEWAY_LOG=logs/fdh_gateway_events.2.jsonl \
  docker compose --env-file .env -p fdh-2 -f $F --profile tau up -d fdh-tau
curl -s localhost:8777/health; echo
docker exec fdh-tau-2 curl -s 127.0.0.1:8790/health; echo
```

Each container runs its own gateway and workers, so both gateway limits apply separately. Stop the second container with the same project name: `docker compose -p fdh-2 -f $F --profile tau down`.

## 7. Point τ³ at the Server

Run the τ² client in the τ² checkout, after its own setup (`uv sync --extra voice --extra dev` and the Inference Hub variables). A τ² client connects to `ws://localhost:8775/v1/realtime` (`PINE_REALTIME_BASE_URL`). Model tags must start with `pine-`. Use `--max-concurrency 1` for the smoke run.

### Gate A

Gate A runs τ²'s real provider against the server with one tool call, without a full run. Against the real server it needs audio the ASR transcribes, because it streams a tone. Use the stub server (section 12) for Gate A. Run it in the τ² checkout:

```bash
PINE_REALTIME_BASE_URL=ws://localhost:8775/v1/realtime PINE_API_KEY=unused \
  uv run python <path-to-this-repository>/src/examples/frontend_delegation_hermes/cli/tau2_gates/gate_a_handshake.py
# last line: GATE A PASSED
```

## 8. Calibration

**M1: calibrate `FDH_PROACTIVE_AFTER_S` on the target host.** From a control run, take the wall-clock and τ² simulated gaps of WORKING spans with no audio. The status line must play at about 20 s simulated or earlier, and not before the 75th percentile of normal filler-to-answer gaps.

## 9. Gates

Run the gates in this order. A fix that fails one is not shipped:

- **P0.1, a clean run:** the τ³ `report.sh` marks a run `INVALID` when any of its sessions hit a backend 429 (`Error code: 429` or `Retrying API call` in `logs/fdh_workers/<session>-<n>.log`). Re-run an invalid run, preferably in quiet hours.
- **G0:** `uv run pytest tests/unit/frontend_delegation_hermes/core/test_fdh_geval_fixes.py -q`.
- **G3 and G4:** the mock domain and invented conversations (G3), then the paired A/B comparison (G4: an A/A control run first, five trials per domain, the arm interleaved with the control).

## 10. Report

`fba_voice_metrics.py` expects one backend operation per turn. Convert the event log first. The adapter credits each run to the turn that started it and exits 1 on an orphaned tool call:

```bash
PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.cli.report_adapter \
  logs/fdh_voice_events.realtime_eval.jsonl --out logs/fdh_voice_events.legacy.jsonl
```

## 11. Logs

The following table lists the log files of `fdh-tau` with its defaults:

| File | Content |
| --- | --- |
| `logs/fdh_voice_events.realtime_eval.jsonl` (`FDH_EVENT_LOG`) | Voice server events per session: `fdh_session_start` (config hash, `features`), `frontend_prompt` (prompt hash), `backend_configured` (backend fingerprint, `backend_domain`, `session_tools_sha256`), `delegation_decision`, `backend_action`, `tool_calls_out` (with `call_ids` and `arguments`), `tool_output_in`, `call_answered_locally`, `backend_answer`, `status_spoken`, `playback_outcome`, `delivery_note`, `history_sync`, `turn_latency`, `filler_timing`, and `barge_in`. With a fix on: `status_proactive`, `spelling_hold`, `answer_replayed`, `filler_deduped`, `answer_cleaned`, `result_hint`, `session_rules`, `format_retry`, `field_provenance`, `delegation_guard`, and `identity_ledger` |
| `logs/fdh_gateway_events.realtime_eval.jsonl` (`FDH_GATEWAY_LOG`) | Gateway: `gateway_start` (`backend_features`, `backend_catalog_sha256`), sessions, `backend_fingerprint`, worker spawn, ready, and exit (`start_ms`, `rss_mb`), run epochs and outcomes, context delivery states, watchdog, and respawns |
| `logs/fdh_workers/*.log` (`FDH_WORKER_LOG_DIR`) | Each Hermes worker's output. "Auxiliary Nous client unavailable" lines are harmless. `Error code: 429` or `Retrying API call` makes the run invalid (P0.1) |
| `logs/fdh_workers/<session>.trace.jsonl` | With `hermes.backend_trace` on: one `session` record (tool schemas, the rendered `backend_system`), then a `request` and a `response` per model call, joined by `api_request_id`. The file grows with each model call |
| `docker logs fdh-tau` | Voice server and gateway output (latency breakdowns and errors) |

Files written by the container are owned by root on the host. Run `sudo chown -R "$USER" logs` before you move or delete them.

## 12. No-GPU Stub Mode

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

## 13. Tests

The following command runs the unit tests on the host:

```bash
uv run pytest tests/unit/frontend_delegation_hermes -q
```

`test_fdh_fix_plan.py` checks that the `124acf7` control renders and decides exactly as agent `124acf7`. Regenerate its golden file only for an intended change of the deployed behavior with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden_eval.py --write`. `test_fdh_golden_prompts.py` checks that, with every prompt variant off, every prompt is byte-identical to agent commit `3a7e04a`. Regenerate it only for an intended baseline change with `PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/core/_fdh_golden.py --write`.

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

## 14. Stop and Clean Up

Always pass `--profile tau`, or `fdh-tau` keeps running. Take the FDH containers down before the root project, because a root `down` cannot remove `nemotron-voice-agent_default` while an FDH container is attached ("active endpoints"):

```bash
docker compose -f $F --profile tau down                            # stops gateways and workers (up to 75 s)
docker compose -p fdh-2 -f $F --profile tau down                   # if a second container runs (section 6)
kill "$(cat logs/fdh_gateway.pid)"                                 # if the host gateway runs (appendix)
docker compose --profile frontend-backend-verdict-agent/single-gpu stop nemo-speech   # optional
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
