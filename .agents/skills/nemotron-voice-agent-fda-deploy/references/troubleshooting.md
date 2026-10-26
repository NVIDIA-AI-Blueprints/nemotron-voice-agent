# Troubleshooting

Set `P=frontend-delegation-agent/single-gpu`, and `E` to the Compose files of the deployment (the table in `SKILL.md`, Containers), before you run the commands below. `[$E]` means: pass `$E` whenever it holds an overlay (eval, plain runtime, LLM backend, or silent frontend). Rows that name `fda-hermes`, OpenShell, the sandbox, or the Hermes image apply to the Hermes backend only. `W` is the OpenShell sandbox workload container: `W=$(docker ps --filter ancestor=nemotron-voice-agent-fda:local --format '{{.Names}}' | head -1)`.

## Deployment

The following table lists deployment symptoms and fixes:

| Symptom | Cause and Fix |
| --- | --- |
| `up` fails while it builds `fda-hermes-image`, and the error names the `hermes` build context | The Hermes checkout is missing at `HERMES_AGENT_DIR`. Complete `prerequisites.md`, section 3, with an absolute path |
| The image build fails in the `uv pip check` step | The lock no longer matches Hermes's pins. Check out the commit from the lock header in a clean checkout, or regenerate `docker/fda/requirements.txt` as its header says (with `--no-config`) |
| The lock cannot resolve Pipecat on Python 3.14, or `test_fda_dependencies.py` fails | The lock must pin the repository's Pipecat version (1.11.0) for Python 3.14, and its header must list the same packages as the `fda` extra in `pyproject.toml`. Fix both together and regenerate the lock. Load `nemotron-voice-agent-upgrade-pipecat` before you change the Pipecat version |
| `fda-hermes` exits at start with `NVIDIA_API_KEY is not set`, and the voice service never starts | `.env` has no `NVIDIA_API_KEY`. Set it (presence only, do not print it) and rerun `up -d` |
| `fda-hermes` exits with `FDA_GATEWAY_CONFIG is not set` or a configuration error | `docker/fda/gateway.env` or the eval overlay names a missing file. Read `docker compose [$E] --profile $P logs fda-hermes` and fix the name |
| `fda-hermes` never turns healthy, so the voice service waits | Read `docker compose [$E] --profile $P logs fda-hermes` and `docker logs "$W"`. OpenShell: the health check calls `http://hermes-openshell-gateway:8080/health` with `Host: default--fda-hermes.openshell.localhost` and a 180 s start period. Plain: it calls `http://localhost:8790/health` with a 60 s start period |
| `fda-hermes` exits with `OpenShell needs Linux 6.2+ (Landlock ABI 3)` | The host kernel is too old. Use the plain overlay, or a host with Linux 6.2+ |
| The sandbox does not start, and `docker logs "$W"` shows a Landlock error | Landlock is off in the host kernel (`cat /sys/kernel/security/lsm` does not list `landlock`). `policy.yaml` requires it (`hard_requirement`). Turn it on, or use the plain overlay. Do not change the policy to `best_effort` without the user's approval |
| `fda-hermes` exits with `OpenShell gateway at ... not connected after 120 s` | `hermes-openshell-gateway` did not start. Read `docker compose [$E] --profile $P logs hermes-openshell-init hermes-openshell-gateway`. Check `/var/run/docker.sock` and `/var/lib/openshell` |
| `up -d` fails with `port is already allocated` on `127.0.0.1:8080` | Another process or another OpenShell gateway (for example a second checkout) uses the port. Only one OpenShell gateway can run per host. Stop the other one |
| `up` fails on an unknown `!override` or `!reset` tag | Docker Compose is older than 2.24.4. Update it |
| `fda-hermes` log: `sandbox fda-hermes still exists 60 s after delete` | OpenShell did not finish deleting the old sandbox. Wait, then rerun `up -d`. Check `docker compose [$E] --profile $P exec fda-hermes openshell sandbox get fda-hermes` |
| Hermes fails on a call to a host other than `inference-api.nvidia.com` | The sandbox policy blocks it. Read denials with `docker compose [$E] --profile $P exec fda-hermes openshell logs fda-hermes`. Report the host to the user; adding it under `network_policies` in `docker/fda/openshell/policy.yaml` changes what Hermes can reach, so ask first |
| Hermes or the voice side logs `protocol version 2 is not supported (expected 1)` (or the reverse) | The Hermes image was built from source older than the voice side (the worker protocol is version 2). Rerun `up -d --build` |
| `fda-hermes` or an OpenShell service starts although the user chose the LLM backend, or `printenv FDA_BACKEND` prints nothing | The `up` ran without `docker/docker-compose.fda-llm.yaml`. Rerun with the LLM `$E` from `SKILL.md`, and pass it in every command |
| After switching to `llm`, `fda-hermes` or `hermes-openshell-gateway` still runs, or `127.0.0.1:8080` stays in use | The Hermes deployment was not stopped first. The LLM overlay moves the Hermes services to an inactive profile, so an `up` or `down` with it neither stops nor starts them. Stop them with the Hermes `$E`: `docker compose <Hermes $E> --profile $P down`, then rerun the LLM `up` |
| Every session is refused: a Realtime client gets `error` with code `server_busy`, close code 1013, message `server is starting`; a browser session ends at once (`browser session refused`); the app log shows `frontend-delegation-agent: startup failed: backend.kind: hermes cannot run in the voice process (link: in_process); ...` | `FDA_BACKEND_LINK=in_process` without `FDA_BACKEND=llm` (for example only `FDA_BACKEND_LINK` in `.env`, or a modified Compose file). The check runs at startup, before the warm-up, so no session starts until it is fixed. Remove `FDA_BACKEND` and `FDA_BACKEND_LINK` from `.env` (or undo the Compose change), then redeploy the backend the user chose: Hermes with the Hermes `$E` from `SKILL.md`, or `llm` with `docker/docker-compose.fda-llm.yaml` unchanged, which sets both values |
| The gateway fails with `workers.agent_kind was replaced by backend.kind`, `workers.mode: in_process_fake was renamed to in_process`, or `hermes.<key> moved to backend.<key>` | A modified gateway config uses the former names. Rename as the error says (`FDH_AGENT_KIND` is now `FDA_BACKEND`; shared keys live under `backend:`). Refer to `config-knobs.md`, section Backends |
| `unknown backend kind` | `FDA_BACKEND` or `backend.kind` is not `hermes`, `llm`, or `emulator`. Select the backend with the overlay only |
| LLM backend: a session fails, or a `harness_*` key is reported missing | Read the app log (the backend runs in the voice process). A prompt profile you added needs `backend/llm.yaml` with every `harness_*` key; copy it from `config/prompts/customer_service/backend/llm.yaml`. Check `NVIDIA_API_KEY` presence (do not print it) |
| A code or backend prompt change has no effect in Hermes, or the `FDA gateway: ... source <hash>` line differs from the checkout's hash | The Hermes runtime runs the source built into the Hermes image. Rerun `up -d --build` |
| The eval settings are lost after a later command | A Compose command ran without `$E`, which recreates the containers with `default` and `gateway.yaml`. Pass `$E` in every command and check `printenv FDA_PROFILE` and the controller's `gateway config` line (plain: the `FDA gateway` line) |
| `FDA_PROFILE` or another `FDA_*` key in `.env` has no effect | Expected. Compose sets them as literals on the voice service. Use the eval overlay for `eval` |
| A browser session ends at once, and the app log shows `runs with FDA_PROFILE=eval` | The deployment runs the eval overlay, which refuses browser sessions. Redeploy with the `default` `$E` of the same backend (`SKILL.md`, section 2): drop only `docker/docker-compose.fda-eval.yaml`, and keep `docker/docker-compose.fda-llm.yaml` for the LLM backend |
| A browser session ends at once, and the app log shows `browser session refused` | 4 browser sessions are open, or the warm-up failed (`startup failed` in the app log). Close idle sessions, or fix the cause and start a new session |
| The app log shows `server.max_sessions (N) exceeds the gateway's max_sessions` | The voice side allows more sessions than the gateway. Keep the Compose values: 4 per session kind and 8 in `gateway.env` |
| The app log shows `backend gateway not reachable` | Hermes: `fda-hermes` is down or not healthy. Sessions fail until it is up. Check its log and health |
| Worker log: `ModuleNotFoundError` for `examples`, or a Hermes `utils` import error | The worker's `PYTHONPATH` must hold only the `examples` symlink in `<socket_dir>/pythonpath` (default `/tmp/fda-workers/pythonpath`), never `src`, because `src/utils.py` shadows Hermes's `utils`. Recreate the runtime: `docker compose [$E] --profile $P up -d --force-recreate fda-hermes` (OpenShell: this deletes and recreates the sandbox) |
| Worker log: `tool surface ... tool_search` | Stale `HERMES_HOME`. Homes are regenerated per worker with `tools.tool_search.enabled: "off"`. Recreate `fda-hermes` |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable` | A backend worker did not start. LLM backend: read the app log. Hermes: read the worker logs: plain `logs/fda_workers_pipecat/*.log`; OpenShell `docker logs "$W"`, or copy `/app/logs` with `docker compose [$E] --profile $P exec fda-hermes openshell sandbox download fda-hermes /app/logs /out/now` and read `logs/openshell/now/`. Rebuild if the Hermes checkout changed |
| Browser: no microphone | Open `https://<IP>:7860`, not `http` and not a proxied host name, and accept the certificate. Do not use the eval overlay for browser sessions |
| `up -d` fails with `port is already allocated` on 7860 or 50051 | Another process uses port 7860 or 50051. Set `PIPELINE_APP_PORT`, or stop the other recipe that runs a speech sidecar |
| `mv` or `rm` in `logs/` fails with permission denied | The containers write logs as root. Run `sudo chown -R "$USER" logs` |

## Realtime Sessions

The following table lists symptoms of OpenAI Realtime sessions:

| Symptom | Cause and Fix |
| --- | --- |
| The handshake closes with code 1008 and reason `realtime authentication failed` | The client sent no Bearer credential or the wrong one. Send `Authorization: Bearer $REALTIME_API_KEY` |
| The handshake closes with code 1008 and reason `invalid realtime model` | The model is not `nvidia/nemotron-realtime-frontend-delegation`. The former `pine-*` names no longer route to FDA |
| `error` with code `server_busy`, close code 1013, message `server at capacity (server.max_sessions)` | 4 Realtime sessions are open. Lower the client's concurrency |
| `error` with code `server_busy`, close code 1013, message `server is starting` | The one-time warm-up failed. Read `startup failed` in the app log. The next session retries |
| `error` with code `server_error`, close code 1011, and the message asks for `UVICORN_WS_PING_INTERVAL=0` | The server pings WebSocket clients. Compose sets `0`; a host-native run must set it too |
| `error` with code `server_error`, close code 1011, and the message names `FDA_PROFILE` | `FDA_PROFILE` is not `default` or `eval`. Use the Compose recipe, which sets it |
| The client gets an error on `input_audio_buffer.commit` | Expected. FDA ends turns with Pipecat's VAD and rejects manual commits |
| Users are cut off mid-utterance, or the app log shows `no audio received while speaking, forcing speech stop` | The session did not run `adapters/pipecat.eval.yaml`. Check the `Pipecat adapter settings` line for `audio_idle_timeout` 0.0. Do not raise `stop_secs` instead |
| The requested voice is not used | Expected. FDA keeps the catalog TTS voice and logs a warning |
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
| No `*.trace.jsonl` in `logs/fda_workers_pipecat/` (plain or LLM backend) or the worker logs saved under `logs/openshell/<time>/` (OpenShell) with `gateway.eval.yaml` | Hermes: check that the `FDA gateway` line names `gateway.eval.yaml`, then the worker log for `backend trace:` warnings. A Hermes version without the plugin hook registry writes no trace. LLM backend: check `printenv FDA_GATEWAY_CONFIG` (`gateway.eval.yaml`) and `FDA_WORKER_LOG_DIR` in the voice container |
| An F1 or F2 arm shows no `delegation_guard` or `answer_replayed` events | Look for `lexicon_missing`. The session's ASR `language_code` has no `delegation.guards.lexicon` entry, so the rules are inactive by design |
| `identity_ledger` never appears in an identity arm | `session_rules.identification_tools` is empty or misses the lookup. Check that the tool's name or description says it returns a person, or list the tools in `result_hints.identification_tools` |
| A manual-mode client gets `conversation_already_has_active_response` | It sent `response.create` while the previous turn was still deciding or speaking. Wait for `response.done`, as Realtime requires |
| A `config_hash` does not match an earlier run | `config_hash` includes the backend link, the gateway file, and the backend URL: `ws://default--fda-hermes.openshell.localhost:8080/v1/backend` (OpenShell) or `ws://fda-hermes:8790/v1/backend` (plain). It differs between backends and runtimes. Pair runs by the prompt hashes and `session_tools_sha256` |
| `backend_system_sha256` differs from the eval pins | The deployment ran `gateway.yaml`. Redeploy with the eval overlay |
