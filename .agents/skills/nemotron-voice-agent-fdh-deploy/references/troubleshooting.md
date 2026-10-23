# Troubleshooting

Set `P=frontend-delegation-hermes-agent/single-gpu`, and `E="-f docker-compose.yml -f docker/docker-compose.fdh-eval.yaml"` for eval mode, before you run the commands below. `[$E]` means: pass `$E` for an eval deployment. For the plain runtime, `$E` also holds `-f docker/docker-compose.fdh-hermes-plain.yaml` (refer to `SKILL.md`, Rules). `W` is the OpenShell sandbox workload container: `W=$(docker ps --filter ancestor=nemotron-voice-agent-fdh:local --format '{{.Names}}' | head -1)`.

## Deployment

The following table lists deployment symptoms and fixes:

| Symptom | Cause and Fix |
| --- | --- |
| `up` fails while it builds `fdh-hermes-image`, and the error names the `hermes` build context | The Hermes checkout is missing at `HERMES_AGENT_DIR`. Complete `prerequisites.md`, section 3, with an absolute path |
| The image build fails in the `uv pip check` step | The lock no longer matches Hermes's pins. Check out the commit from the lock header in a clean checkout, or regenerate `docker/fdh/requirements.txt` as its header says (with `--no-config`) |
| The lock cannot resolve Pipecat on Python 3.14, or `test_fdh_dependencies.py` fails | The lock must pin the repository's Pipecat version (1.11.0) for Python 3.14, and its header must list the same packages as the `fdh` extra in `pyproject.toml`. Fix both together and regenerate the lock. Load `nemotron-voice-agent-upgrade-pipecat` before you change the Pipecat version |
| `fdh-hermes` exits at start with `NVIDIA_API_KEY is not set`, and the voice service never starts | `.env` has no `NVIDIA_API_KEY`. Set it (presence only, do not print it) and rerun `up -d` |
| `fdh-hermes` exits with `FDH_GATEWAY_CONFIG is not set` or a configuration error | `docker/fdh/gateway.env` or the eval overlay names a missing file. Read `docker compose [$E] --profile $P logs fdh-hermes` and fix the name |
| `fdh-hermes` never turns healthy, so the voice service waits | Read `docker compose [$E] --profile $P logs fdh-hermes` and `docker logs "$W"`. OpenShell: the health check calls `http://hermes-openshell-gateway:8080/health` with `Host: default--fdh-hermes.openshell.localhost` and a 180 s start period. Plain: it calls `http://localhost:8790/health` with a 60 s start period |
| `fdh-hermes` exits with `OpenShell needs Linux 6.2+ (Landlock ABI 3)` | The host kernel is too old. Use the plain overlay, or a host with Linux 6.2+ |
| The sandbox does not start, and `docker logs "$W"` shows a Landlock error | Landlock is off in the host kernel (`cat /sys/kernel/security/lsm` does not list `landlock`). `policy.yaml` requires it (`hard_requirement`). Turn it on, or use the plain overlay. Do not change the policy to `best_effort` without the user's approval |
| `fdh-hermes` exits with `OpenShell gateway at ... not connected after 120 s` | `hermes-openshell-gateway` did not start. Read `docker compose [$E] --profile $P logs hermes-openshell-init hermes-openshell-gateway`. Check `/var/run/docker.sock` and `/var/lib/openshell` |
| `up -d` fails with `port is already allocated` on `127.0.0.1:8080` | Another process or another OpenShell gateway (for example a second checkout) uses the port. Only one OpenShell gateway can run per host. Stop the other one |
| `up` fails on an unknown `!override` or `!reset` tag | Docker Compose is older than 2.24.4. Update it |
| `fdh-hermes` log: `sandbox fdh-hermes still exists 60 s after delete` | OpenShell did not finish deleting the old sandbox. Wait, then rerun `up -d`. Check `docker compose [$E] --profile $P exec fdh-hermes openshell sandbox get fdh-hermes` |
| Hermes fails on a call to a host other than `inference-api.nvidia.com` | The sandbox policy blocks it. Read denials with `docker compose [$E] --profile $P exec fdh-hermes openshell logs fdh-hermes`. Report the host to the user; adding it under `network_policies` in `docker/fdh/openshell/policy.yaml` changes what Hermes can reach, so ask first |
| A code or backend prompt change has no effect in Hermes, or the `FDH gateway: ... source <hash>` line differs from the checkout's hash | The Hermes runtime runs the source built into the Hermes image. Rerun `up -d --build` |
| The eval settings are lost after a later command | A Compose command ran without `$E`, which recreates the containers with `default` and `gateway.yaml`. Pass `$E` in every command and check `printenv FDH_PROFILE` and the controller's `gateway config` line (plain: the `FDH gateway` line) |
| `FDH_PROFILE` or another `FDH_*` key in `.env` has no effect | Expected. Compose sets them as literals on the voice service. Use the eval overlay for `eval` |
| A browser session ends at once, and the app log shows `runs with FDH_PROFILE=eval` | The deployment runs the eval overlay, which refuses browser sessions. Redeploy without `$E` |
| A browser session ends at once, and the app log shows `browser session refused` | 4 browser sessions are open, or the warm-up failed (`startup failed` in the app log). Close idle sessions, or fix the cause and start a new session |
| The app log shows `server.max_sessions (N) exceeds the gateway's max_sessions` | The voice side allows more sessions than the gateway. Keep the Compose values: 4 per session kind and 8 in `gateway.env` |
| The app log shows `backend gateway not reachable` | `fdh-hermes` is down or not healthy. Sessions fail until it is up. Check its log and health |
| Worker log: `ModuleNotFoundError` for `examples`, or a Hermes `utils` import error | The worker's `PYTHONPATH` must hold only the `examples` symlink in `<socket_dir>/pythonpath` (default `/tmp/fdh-workers/pythonpath`), never `src`, because `src/utils.py` shadows Hermes's `utils`. Recreate the runtime: `docker compose [$E] --profile $P up -d --force-recreate fdh-hermes` (OpenShell: this deletes and recreates the sandbox) |
| Worker log: `tool surface ... tool_search` | Stale `HERMES_HOME`. Homes are regenerated per worker with `tools.tool_search.enabled: "off"`. Recreate `fdh-hermes` |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable` | A Hermes worker did not start. Read the worker logs: plain `logs/fdh_workers_pipecat/*.log`; OpenShell `docker logs "$W"`, or copy `/app/logs` with `docker compose [$E] --profile $P exec fdh-hermes openshell sandbox download fdh-hermes /app/logs /out/now` and read `logs/openshell/now/`. Rebuild if the Hermes checkout changed |
| Browser: no microphone | Open `https://<IP>:7860`, not `http` and not a proxied host name, and accept the certificate. Do not use the eval overlay for browser sessions |
| `up -d` fails with `port is already allocated` on 7860 or 50051 | Another process uses port 7860 or 50051. Set `PIPELINE_APP_PORT`, or stop the other recipe that runs a speech sidecar |
| `mv` or `rm` in `logs/` fails with permission denied | The containers write logs as root. Run `sudo chown -R "$USER" logs` |

## Realtime Sessions

The following table lists symptoms of OpenAI Realtime sessions:

| Symptom | Cause and Fix |
| --- | --- |
| The handshake closes with code 1008 and reason `realtime authentication failed` | The client sent no Bearer credential or the wrong one. Send `Authorization: Bearer $REALTIME_API_KEY` |
| The handshake closes with code 1008 and reason `invalid realtime model` | The model is not `nvidia/nemotron-realtime-frontend-delegation-hermes`. The former `pine-*` names no longer route to FDH |
| `error` with code `server_busy`, close code 1013, message `server at capacity (server.max_sessions)` | 4 Realtime sessions are open. Lower the client's concurrency |
| `error` with code `server_busy`, close code 1013, message `server is starting` | The one-time warm-up failed. Read `startup failed` in the app log. The next session retries |
| `error` with code `server_error`, close code 1011, and the message asks for `UVICORN_WS_PING_INTERVAL=0` | The server pings WebSocket clients. Compose sets `0`; a host-native run must set it too |
| `error` with code `server_error`, close code 1011, and the message names `FDH_PROFILE` | `FDH_PROFILE` is not `default` or `eval`. Use the Compose recipe, which sets it |
| The client gets an error on `input_audio_buffer.commit` | Expected. FDH ends turns with Pipecat's VAD and rejects manual commits |
| Users are cut off mid-utterance, or the app log shows `no audio received while speaking, forcing speech stop` | The session did not run `adapters/pipecat.eval.yaml`. Check the `Pipecat adapter settings` line for `audio_idle_timeout` 0.0. Do not raise `stop_secs` instead |
| The requested voice is not used | Expected. FDH keeps the catalog TTS voice and logs a warning |
| The client fails with `ssl argument is incompatible with a ws:// URI`, or a `wss://` handshake fails | The eval overlay serves plain `ws://`; the `default` profile serves `wss://` |

## Evaluation Runs

The following table lists symptoms of evaluation runs:

| Symptom | Cause and Fix |
| --- | --- |
| No reply to a real question | Check `delegation_decision`. `repair: timeout` means the frontend endpoint was slow (4 s budget, hedged at 1.5 s). `guard_backchannel` means the turn was a pure backchannel |
| An answer is never spoken | The user was still speaking, and answers are held until they stop (`output_hold`). Check `playback_outcome` for `not_heard` |
| Voice server fails with `replay_unheard_answer.enabled needs prompt_features.replay_intent` | M2 needs its frontend prompt variant |
| A status update says "still working" after the answer | A status line is dropped when its run has already finished (`status_dropped`). If it occurs, report the session ID |
| A status sentence sounds templated | The status LLM call exceeded `backend.status_verbalizer.timeout_ms` (3000). `status_spoken.source` is `template_fallback` |
| No `*.trace.jsonl` in `logs/fdh_workers_pipecat/` (plain) or the worker logs saved under `logs/openshell/<time>/` (OpenShell) with `gateway.eval.yaml` | Check that the `FDH gateway` line names `gateway.eval.yaml`, then the worker log for `backend trace:` warnings. A Hermes version without the plugin hook registry writes no trace |
| An F1 or F2 arm shows no `delegation_guard` or `answer_replayed` events | Look for `lexicon_missing`. The session's ASR `language_code` has no `delegation.guards.lexicon` entry, so the rules are inactive by design |
| `identity_ledger` never appears in an identity arm | `session_rules.identification_tools` is empty or misses the lookup. Check that the tool's name or description says it returns a person, or list the tools in `result_hints.identification_tools` |
| A manual-mode client gets `conversation_already_has_active_response` | It sent `response.create` while the previous turn was still deciding or speaking. Wait for `response.done`, as Realtime requires |
| A `config_hash` does not match an earlier run | `config_hash` includes the backend URL: `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` (OpenShell) or `ws://fdh-hermes:8790/v1/backend` (plain). Pair runs by the prompt hashes and `session_tools_sha256` |
| `backend_system_sha256` differs from the eval pins | The deployment ran `gateway.yaml`. Redeploy with the eval overlay |
