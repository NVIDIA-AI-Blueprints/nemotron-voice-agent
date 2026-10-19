# Eval Mode and Benchmark Runs

Eval mode deploys the `eval` profile for OpenAI Realtime evaluation runs, such as τ². Evaluation clients reach FDH through the shared `/v1/realtime` route of the app container. The former FDH server, its τ³ adapter, its `cli/` tools, and the τ² gate scripts no longer exist.

Complete `prerequisites.md` first, then set the following variables in every shell:

```bash
P=frontend-delegation-hermes-agent/single-gpu
E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml"
# plain runtime (baseline) instead: the plain overlay goes before the eval overlay
# E="-f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml -f docker/docker-compose.fdh-eval.yaml"
```

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

The `fdh-hermes` service is the controller that creates the sandbox. With the plain overlay, the URL is `ws://fdh-hermes:8790/v1/backend`, and `fdh-hermes` is the FDH gateway container itself.

The shared route checks the Bearer credential (`REALTIME_API_KEY`), resolves the model from the registry, and hands the socket to FDH before it sends any event. FDH serves `session.created` and every later event with its own wire code (`frontend-delegation-hermes-agent` is in `_SESSION_OWNED_PIPELINES` of `src/realtime/gateway.py`).

## 2. What Eval Mode Changes

The overlay `docker/docker-compose.fdh-eval.yaml` sets only the following keys:

| Service | Key | Value |
| --- | --- | --- |
| `frontend-delegation-hermes-agent-single-gpu` | `FDH_PROFILE` | `eval`: Realtime sessions only, browser sessions refused |
| `frontend-delegation-hermes-agent-single-gpu` | `PIPELINE_TLS` | `false`: plain HTTP and `ws://` |
| `fdh-hermes` | `FDH_GATEWAY_CONFIG` | `gateway.eval.yaml`: every backend prompt variant pinned, the telecom domain note, and the backend decision trace on |

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

Expect `eval`, `false`, and `0`. The gateway health shows `"ok": true`, `"max_sessions": 8`, and the pinned `"backend_features"`. Compare the source hash with the checkout, as `SKILL.md` describes.

## 4. Smoke Test

Run the repository's Realtime test client against the plain `ws://` upstream. It reads `REALTIME_API_KEY` from `.env`:

```bash
uv run python realtime_test_client/proxy.py --upstream ws://localhost:7860 \
  --model nvidia/nemotron-realtime-frontend-delegation-hermes
```

Open `http://localhost:8765`, select **Connect**, and speak or send text. The first session waits for the one-time warm-up (one frontend LLM request and the gateway check). Then the app log shows `frontend-delegation-hermes-agent ready (adapter=pipecat, profile=eval)` with the `config hash`, and `logs/fdh_voice_events.pipecat.jsonl` grows:

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

VoiceBench (in `voice-agent-evaluation`) never sends `response.create`, ends each case on the first `response.done`, and scores only that response's audio transcript. FDH speaks each filler and proactive status line as its own response. Ask the user whether to keep the frontend silent only when they say the run is VoiceBench (refer to the rule in `SKILL.md`). If yes, add the VoiceBench overlay last:

```bash
E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-voicebench.yaml"
# plain runtime: -f docker-compose.yml -f docker/docker-compose.fdh-hermes-plain.yaml -f docker/docker-compose.fdh-eval.yaml -f docker/docker-compose.fdh-voicebench.yaml
docker compose $E --profile $P up -d --build
docker compose $E --profile $P exec frontend-delegation-hermes-agent-single-gpu printenv FDH_SILENT_FRONTEND   # true
```

- After one session, `logs/fdh_voice_events.pipecat.jsonl` has `silent_frontend: true` in the `fdh_session_start` settings and a `filler_silenced` record (reason `silent_frontend`) per delegated turn. `proactive_status_silenced` is logged once per backend run instead of the M1 status line. Direct answers are spoken.
- Harness: profile `nemotron-fdh-local`, run configuration `configs/runs/voicebench-fdh-realtime-smoke.yaml` in `voice-agent-evaluation`. Do not change the evaluator.
- The `config hash` of a silent run differs from a normal eval run. With the switch off, the hash is unchanged.
- To switch back, rerun `up -d --build` without the VoiceBench overlay. Never use the overlay for Tau.

## 6. Fingerprint and Hashes

Record the following with every run:

- the Hermes runtime (OpenShell or plain), and for OpenShell the release in `docker/fdh/openshell/VERSION`;
- the gateway configuration and source hash: the `FDH gateway` line from `docker logs "$W"` (plain: `docker compose $E --profile $P logs fdh-hermes`);
- the profile and `config hash` from the ready line of the app log;
- the fingerprint records in `logs/fdh_voice_events.pipecat.jsonl`: `fdh_session_start` (`features`), `frontend_prompt`, and `backend_configured`, which carries the backend fingerprint the gateway reports (`backend_system_sha256`, `backend_catalog_sha256`, `backend_domain`) and `session_tools_sha256`.

`config_hash` includes `backend.url`, which is now `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` (plain runtime: `ws://fdh-hermes:8790/v1/backend`; formerly `ws://host.docker.internal:8790/v1/backend`), so it differs between runtimes and from earlier runs. Group and compare runs by the prompt hashes and `session_tools_sha256`. `backend_system_sha256` differs between `gateway.yaml` and `gateway.eval.yaml`, so it shows a run that missed the eval overlay.

The pins of the eval behavior are `tests/unit/frontend_delegation_hermes/pipecat/fixtures/eval_pins.json`, checked by `test_fdh_eval_pins.py`. Regenerate them only for an intended change of the evaluated behavior.

## 7. Logs

| Log | Content |
| --- | --- |
| `docker compose $E --profile $P logs frontend-delegation-hermes-agent-single-gpu` | App log: ready line, Pipecat settings, session errors |
| `docker compose $E --profile $P logs fdh-hermes` | OpenShell: controller log (image, gateway configuration, OpenShell CLI version, setup errors). Plain: gateway log (configuration, source hash, gateway errors) |
| `docker logs "$W"` | OpenShell: FDH gateway and worker output, and Landlock denials |
| `docker compose $E --profile $P exec fdh-hermes openshell logs fdh-hermes` | OpenShell: policy decisions (allowed and denied outbound calls) |
| `logs/fdh_voice_events.pipecat.jsonl` | Voice events of every session (`FDH_EVENT_LOG`) |
| `logs/fdh_gateway_events.pipecat.jsonl` | Plain: gateway events (`FDH_GATEWAY_LOG` in `docker/fdh/gateway.env`) |
| `logs/fdh_workers_pipecat/` | Plain: one log per Hermes worker, and with `gateway.eval.yaml` one `<session_id>.trace.jsonl` backend decision trace per session |
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

To switch back to the browser UI, run `docker compose --profile $P up -d --build` without the overlay.
