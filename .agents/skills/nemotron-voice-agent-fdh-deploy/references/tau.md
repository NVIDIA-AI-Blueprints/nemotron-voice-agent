# Eval Mode and Benchmark Runs

Eval mode deploys the `eval` profile for OpenAI Realtime evaluation runs, such as τ². Evaluation clients reach FDH through the shared `/v1/realtime` route of the app container. The former FDH server, its τ³ adapter, its `cli/` tools, and the τ² gate scripts no longer exist.

Complete `prerequisites.md` first, then set the following variables in every shell:

```bash
P=frontend-delegation-hermes-agent/single-gpu
E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml"
# plain runtime (baseline) instead: the plain overlay goes before the eval overlay
# E="-f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml -f docker/docker-compose.fdh-eval.yaml"
# LLM backend instead: the LLM overlay goes after the eval overlay
# E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-llm.yaml"
```

The backend must be the user's choice (ask, suggest `hermes`; refer to `SKILL.md`). Record the backend with every run, and do not mix backends inside one comparison unless the comparison is Hermes against `llm`.

The eval overlay defaults the prompt profile to `customer_service` on both services. Do not set `FDH_PROMPT_PROFILE` for Tau; check that the shell and `.env` do not set it (`printenv FDH_PROMPT_PROFILE` in the voice container prints `customer_service`). Only a VoiceBench run sets it (refer to the rule in `SKILL.md`).

The Hermes runtime is OpenShell by default. Use the plain runtime only when the user asks for it, for example for an A/B slice. Record which runtime a run used, and do not mix runtimes inside one comparison.

## 1. Topology

```text
 τ² / Realtime client ─ws─► frontend-delegation-hermes-agent-single-gpu (shared app image, :7860)
                             /v1/realtime?model=nvidia/nemotron-realtime-frontend-delegation-hermes
                             ├─ frontend decider ──► Inference Hub nvidia/nvidia/nemotron-3.5-lightning (reasoning off)
                             ├─ speech ──► nemo-speech :50051
                             └─ ws://default--fdh-hermes.openshell.localhost:8080/v1/backend
                                  └─ hermes-openshell-gateway (routes by Host header)
                                       └─ OpenShell sandbox fdh-hermes (nemotron-voice-agent-fdh:local, UID 1500):
                                          FDH gateway + one Hermes worker per session
                                            └─ AIAgent ──► Inference Hub nvidia/nvidia/nemotron-3-ultra (reasoning on)
```

The `fdh-hermes` service is the controller that creates the sandbox. With the plain overlay, the URL is `ws://fdh-hermes:8790/v1/backend`, and `fdh-hermes` is the FDH gateway container itself. With the LLM backend, the WebSocket hop and the Hermes containers are gone: the voice container runs the gateway and one OpenAI SDK tool loop per session in-process (`FDH_BACKEND_LINK=in_process`), calling the same Inference Hub `nvidia/nvidia/nemotron-3-ultra`.

The shared route checks the Bearer credential (`REALTIME_API_KEY`), resolves the model from the registry, and hands the socket to FDH before it sends any event. FDH serves `session.created` and every later event with its own wire code (`frontend-delegation-hermes-agent` is in `_SESSION_OWNED_PIPELINES` of `src/realtime/gateway.py`).

## 2. What Eval Mode Changes

The overlay `docker/docker-compose.fdh-eval.yaml` sets only the following keys:

| Service | Key | Value |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `FDH_PROFILE` | `eval`: Realtime sessions only, browser sessions refused |
| `frontend-delegation-hermes-agent-single-gpu` | `PIPELINE_TLS` | `false`: plain HTTP and `ws://` |
| `fdh-hermes` | `FDH_GATEWAY_CONFIG` | `gateway.eval.yaml`: every backend prompt variant pinned, the telecom domain note, and the backend decision trace on |
| `frontend-delegation-hermes-agent-single-gpu` | `FDH_GATEWAY_CONFIG` | `gateway.eval.yaml`: the same file for the in-process LLM backend (unused with Hermes) |
| Both | `FDH_PROMPT_PROFILE` | `${FDH_PROMPT_PROFILE:-customer_service}` |

Every Realtime session, in both profiles, runs `profiles/pipecat_realtime_eval.yaml` (client-executed tools with `tools.executor: wire`, the client's `session.update` instructions as the policy, no greeting) with `config/adapters/pipecat.eval.yaml`. The eval Pipecat settings measure the end of a user turn in input audio only. τ² pauses its input between ticks, and some ticks stall for more than 2 s while its user simulator runs, so any wall-clock timer in turn ending cuts users off or adds latency:

| Key | `default` (Pipecat default) | `eval` | Why |
| --- | --- | --- | --- |
| `vad.stop_secs` | 0.2 | 0.8 | Silero counts 800 ms of silence in audio chunks |
| `vad.audio_idle_timeout` | 1.0 (wall clock) | 0 (off) | `VADProcessor` forces speech-stopped after 1 s without input audio, which ends the turn mid-utterance when a τ² tick stalls |
| `turn.user_speech_timeout` | 0.6 (wall clock) | 0 | `SpeechTimeoutUserTurnStopStrategy` waits this long after the VAD stop |

`user_turn_stop_timeout` keeps its default (5 s, wall clock). It fires only while the user is not speaking, as a safety net. Do not raise `stop_secs` to make up for tick stalls: a longer wait only adds latency, which τ²'s user simulator answers with pings and talk-over.

## 3. Start and Check

```bash
mkdir -p logs
docker compose $E --profile $P up -d --build
docker compose $E --profile $P exec frontend-delegation-hermes-agent-single-gpu \
  .venv/bin/python -c "import nltk; print(nltk.download('punkt_tab', quiet=True))"   # True
curl -f http://localhost:7860/health                                                    # {"status":"ok"}
docker compose $E --profile $P exec frontend-delegation-hermes-agent-single-gpu printenv FDH_PROFILE PIPELINE_TLS UVICORN_WS_PING_INTERVAL
docker compose $E --profile $P logs fdh-hermes | grep 'gateway config'                 # controller: gateway config gateway.eval.yaml
W=$(docker ps --filter ancestor=nemotron-voice-agent-fdh:local --format '{{.Names}}' | head -1)   # sandbox workload
docker logs "$W" 2>&1 | grep 'FDH gateway'                                              # config gateway.eval.yaml, source <hash>
docker compose $E --profile $P exec fdh-hermes \
  curl -s -H 'Host: default--fdh-hermes.openshell.localhost' http://hermes-openshell-gateway:8080/health; echo
```

Plain runtime: `logs fdh-hermes | grep 'FDH gateway'` and `exec fdh-hermes curl -s http://localhost:8790/health`.

LLM backend: skip the `fdh-hermes`, `W`, and gateway health lines (none exist). Run instead:

```bash
docker compose $E --profile $P exec frontend-delegation-hermes-agent-single-gpu \
  printenv FDH_BACKEND FDH_BACKEND_LINK FDH_GATEWAY_CONFIG                       # llm, in_process, gateway.eval.yaml
```

Expect `eval`, `false`, and `0`. Hermes: the gateway health shows `"ok": true`, `"max_sessions": 8`, and the pinned `"backend_features"`. Compare the source hash with the checkout, as `SKILL.md` describes.

## 4. Smoke Test

Run the repository's Realtime test client against the plain `ws://` upstream. It reads `REALTIME_API_KEY` from `.env`:

```bash
uv run python realtime_test_client/proxy.py --upstream ws://localhost:7860 \
  --model nvidia/nemotron-realtime-frontend-delegation-hermes
```

Open `http://localhost:8765`, select **Connect**, and speak or send text. The first session waits for the one-time warm-up (one frontend LLM request and, with Hermes, the gateway check). Then the app log shows `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=eval)` with the `config hash`, and `logs/fdh_voice_events.pipecat.jsonl` grows:

```bash
docker compose $E --profile $P logs frontend-delegation-hermes-agent-single-gpu | grep -E 'ready \(adapter|Pipecat adapter settings|backend gateway'
tail -n 1 logs/fdh_voice_events.pipecat.jsonl
```

The `Pipecat adapter settings` line must show a `settings_file` that ends in `adapters/pipecat.eval.yaml`, `"vad"` with `stop_secs` 0.8 and `audio_idle_timeout` 0.0, and `"turn"` with `user_speech_timeout` 0.0. The app log must show no `forcing speech stop` warning.

## 5. Point the Evaluation Client at FDH

- URL: `ws://<host>:7860/v1/realtime`, model `nvidia/nemotron-realtime-frontend-delegation-hermes`.
- Credential: `Authorization: Bearer $REALTIME_API_KEY`. Do not print or rotate the key.
- The former `pine-*` model names and ports 8775 and 8776 no longer reach FDH. The evaluator configuration must use the registry model ID.
- Keep concurrency within the Realtime cap of 4 sessions. A fifth session receives `server_busy` and close code 1013.

Report FDH runs on the shared stack separately from runs on the former FDH containers until an A/B slice shows the same rewards.

### VoiceBench

VoiceBench (in `voice-agent-evaluation`) never sends `response.create`, ends each case on the first `response.done`, and scores only that response's audio transcript. FDH speaks each filler and proactive status line as its own response. Only when the user says the run is VoiceBench, ask the two questions of the rule in `SKILL.md`: keep the frontend silent (default no), and which prompt profile (suggested `general_qa`). If silent, add the silent-frontend overlay last. Set the chosen profile on every command that recreates containers:

```bash
E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-silent-frontend.yaml"
# plain runtime: -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-silent-frontend.yaml
# LLM backend: -f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-llm.yaml -f docker/docker-compose.fdh-silent-frontend.yaml
# not silent: leave out the silent-frontend overlay
FDH_PROMPT_PROFILE=general_qa docker compose $E --profile $P up -d --build
docker compose $E --profile $P exec frontend-delegation-hermes-agent-single-gpu printenv FDH_SILENT_FRONTEND FDH_PROMPT_PROFILE   # true, general_qa
docker compose $E --profile $P exec fdh-hermes printenv FDH_PROMPT_PROFILE                                                         # general_qa (Hermes only)
```

- `general_qa` makes the frontend delegate every request (`delegate=true`, filler "One moment.") and the backend answer from the LLM with no tools.
- With the silent frontend, a turn the frontend model leaves silent (`delegate=false`, empty filler) that holds a request is delegated anyway as a `task` with an empty filler, logged as a `delegation_guard` event with `guard=silent_delegated`. Empty turns, backchannels (`backchannel_words`, such as "okay"), pauses ("hold on"), liveness pings (R1), and turns while the backend is WORKING stay silent.
- After one session, `logs/fdh_voice_events.pipecat.jsonl` has `silent_frontend: true` in the `fdh_session_start` settings and a `filler_silenced` record (reason `silent_frontend`) per delegated turn. `proactive_status_silenced` is logged once per backend run instead of the M1 status line. Direct answers are spoken.
- Harness: profile `nemotron-fdh-local`, run configuration `configs/runs/voicebench-fdh-realtime-smoke.yaml` in `voice-agent-evaluation`. Do not change the evaluator.
- The `config hash` of a silent run differs from a normal eval run, and so does a run with another prompt profile (the prompt paths are hashed). With the switch off and no profile set, the hash is unchanged.
- To switch back, rerun `up -d --build` without the silent-frontend overlay and without `FDH_PROMPT_PROFILE`. Never use the overlay or another profile for Tau.

## 6. Fingerprint and Hashes

Record the following with every run:

- the backend (`hermes` or `llm`);
- Hermes: the runtime (OpenShell or plain), and for OpenShell the release in `docker/fdh/openshell/VERSION`;
- Hermes: the gateway configuration and source hash, from the `FDH gateway` line of `docker logs "$W"` (plain: `docker compose $E --profile $P logs fdh-hermes`). LLM backend: `printenv FDH_BACKEND FDH_BACKEND_LINK FDH_GATEWAY_CONFIG` in the voice container and `git rev-parse --short HEAD` (the voice service runs the mounted `src/`);
- the profile and `config hash` from the ready line of the app log, and the prompt profile (`printenv FDH_PROMPT_PROFILE`);
- the fingerprint records in `logs/fdh_voice_events.pipecat.jsonl`: `fdh_session_start` (`features`), `frontend_prompt`, and `backend_configured`, which carries the backend fingerprint the gateway reports (`backend_system_sha256`, `backend_catalog_sha256`, `backend_domain`) and `session_tools_sha256`.

`config_hash` includes `backend.link` and `backend.gateway_config`, so it differs between the backends. It also includes `backend.url`, which is now `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` (plain runtime: `ws://fdh-hermes:8790/v1/backend`; formerly `ws://host.docker.internal:8790/v1/backend`), so it differs between runtimes and from earlier runs. Group and compare runs by the prompt hashes and `session_tools_sha256`. `backend_system_sha256` differs between `gateway.yaml` and `gateway.eval.yaml`, so it shows a run that missed the eval overlay.

The pins of the eval behavior are `tests/unit/frontend_delegation_hermes/pipecat/fixtures/eval_pins.json`, checked by `test_fdh_eval_pins.py`. Regenerate them only for an intended change of the evaluated behavior.

## 7. Logs

| Log | Content |
| --- | --- |
| `docker compose $E --profile $P logs frontend-delegation-hermes-agent-single-gpu` | App log: ready line, Pipecat settings, session errors; with `llm`, also the backend output |
| `docker compose $E --profile $P logs fdh-hermes` | OpenShell: controller log (image, gateway configuration, OpenShell CLI version, setup errors). Plain: gateway log (configuration, source hash, gateway errors) |
| `docker logs "$W"` | OpenShell: FDH gateway and worker output, and Landlock denials |
| `docker compose $E --profile $P exec fdh-hermes openshell logs fdh-hermes` | OpenShell: policy decisions (allowed and denied outbound calls) |
| `logs/fdh_voice_events.pipecat.jsonl` | Voice events of every session (`FDH_EVENT_LOG`) |
| `logs/fdh_gateway_events.pipecat.jsonl` | Plain and LLM backend: gateway events (`FDH_GATEWAY_LOG` in `docker/fdh/gateway.env`, or in the LLM overlay); the `session_open` record names the backend in `workers.agent_kind` |
| `logs/fdh_workers_pipecat/` | Plain: one log per Hermes worker. Plain and LLM backend: with `gateway.eval.yaml`, one `<session_id>.trace.jsonl` backend decision trace per session (same format, `core/backends/trace.py`) |
| `logs/openshell/<time>/` | OpenShell: the sandbox's `/app/logs` (gateway events, worker logs, traces), saved when `fdh-hermes` stops |

In the OpenShell runtime, the gateway events, worker logs, and traces stay in the sandbox's `/app/logs` until `fdh-hermes` stops (`down`, or `restart fdh-hermes`). To copy them during a run, use `docker compose $E --profile $P exec fdh-hermes openshell sandbox download fdh-hermes /app/logs /out/now`; the copy lands in `logs/openshell/now/`. Copy or stop before you read a run's traces.

The containers write the logs as `root`. Before you move or delete them, run `sudo chown -R "$USER" logs`.

## 8. Tests

```bash
uv run pytest tests/unit/frontend_delegation_hermes -q
```

The Hermes suites (`test_fdh_hermes_multisession.py`, `test_fdh_hermes_adapter.py`) need the host Hermes environment from `prerequisites.md`, section 5. Point `FDH_HERMES_PYTHON` at its interpreter when it is not at the default `~/.cache/fdh/hermes-venv-314/bin/python`.

## 9. Stop

```bash
docker compose $E --profile $P down
```

To switch back to the browser UI, run `up -d --build` with the `default` `$E` of the same backend (without the eval overlay).
