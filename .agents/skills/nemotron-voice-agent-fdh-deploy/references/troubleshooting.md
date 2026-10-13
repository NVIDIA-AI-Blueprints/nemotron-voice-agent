# Troubleshooting

Set `F=docker/docker-compose.frontend-delegation-hermes.yaml` before you run the commands below.

## Deployment

The following table lists deployment symptoms and fixes:

| Symptom | Cause and Fix |
| --- | --- |
| The wrong adapter started | The Pipecat adapter is `fdh-voice` and the τ³ adapter is `fdh-tau`. `/health` shows `"adapter": "pipecat"` only on the Pipecat adapter, and `docker logs <container>` shows `starting adapter '<name>'`. Start the right service by name |
| A client cannot connect, or reaches the wrong adapter | τ³ is on 8775 (`ws://`), and Pipecat is on 8776 (`wss://` for the demo, `ws://` in eval mode with `FDH_PIPECAT_TLS=false`) |
| `--profile tau up` started both services | `up` without a service name starts every service of the enabled profiles. Stop `fdh-voice` with `docker compose -f $F stop fdh-voice`, and always name `fdh-tau` |
| `fdh-tau` still runs after `down` | `down` without `--profile tau` ignores the profiled service. Run `docker compose -f $F --profile tau down` |
| Pipecat eval settings are lost after a later `up` | The `FDH_PIPECAT_*` variables were prefixed to one command only. Export them, or prefix them again, then check `/health` for `"vad": {"stop_secs": 0.8, "audio_idle_timeout": 0.0}` and `"turn": {"user_speech_timeout": 0.0}` |
| Pipecat eval cuts users off mid-utterance, or `docker logs fdh-voice` shows `no audio received while speaking, forcing speech stop` | The container runs the demo settings, whose `VADProcessor` forces a speech stop after 1 s without input audio. Start the eval mode with `FDH_PIPECAT_SETTINGS=adapters/pipecat.eval.yaml` and check `/health` for `"audio_idle_timeout": 0.0`. Do not raise `stop_secs` instead |
| A container exits at start with `gateway not healthy` | The in-container gateway did not answer within 30 s. `docker logs <container>` shows its error, usually a wrong config name in `FDH_GATEWAY_CONFIG` or `FDH_PIPECAT_GATEWAY_CONFIG` |
| A container exits with `unknown adapter` or a usage message | `FDH_ADAPTER`, `FDH_PROFILE`, or `FDH_GATEWAY_CONFIG` is empty or names no `<adapter>_server` module. Use the Compose services, which set all three |
| `fdh-tau` restarts again and again | The speech warm-up fails until `nemo-speech` is up, or the external network is missing. Start `nemo-speech` first |
| `up` fails with `network nemotron-voice-agent_default … not found` | Start `nemo-speech` from the root project first (`prerequisites.md`, section 2) |
| A root `docker compose … down` fails with "active endpoints" | An FDH container is still attached to `nemotron-voice-agent_default`. Take the FDH containers down first |
| The image build fails in the `uv pip check` step | The lock no longer matches Hermes's pins. Regenerate `docker/fdh/requirements.txt` as its header says (with `--no-config`), from a clean Hermes checkout at the recorded commit |
| The lock cannot resolve Pipecat on Python 3.14, or `test_fdh_dependencies.py` fails | The lock must pin the repository's Pipecat version (1.11.0) for Python 3.14, and its header must list the same packages as the `fdh` extra in `pyproject.toml`. Fix both together and regenerate the lock. Load `nemotron-voice-agent-upgrade-pipecat` before you change the Pipecat version |
| `session.update` fails with `worker_start_timeout` or `backend_unavailable` | Read `logs/fdh_workers/*.log`. Rebuild the image if the Hermes checkout changed. With the host gateway, the Hermes environment is usually missing or wrong: check `FDH_HERMES_PYTHON`, and set `FDH_HERMES_REPO` |
| Worker log: `ModuleNotFoundError` for `examples`, or a Hermes `utils` import error | The worker's `PYTHONPATH` must hold only the `examples` symlink in `<socket_dir>/pythonpath` (default `/tmp/fdh-workers/pythonpath`), never `src`, because `src/utils.py` shadows Hermes's `utils`. Delete a stale `pythonpath` directory. The gateway recreates the symlink |
| Worker log: `tool surface ... tool_search` | Stale `HERMES_HOME`. Homes are regenerated per worker with `tools.tool_search.enabled: "off"`. Delete `.cache/fdh-hermes-homes/` |
| Voice server exits with `server.max_sessions (N) exceeds the gateway's max_sessions` | Lower the voice server's limit (`FDH_MAX_SESSIONS` or `FDH_PIPECAT_MAX_SESSIONS`) or raise the gateway's |
| Browser: no microphone | Use `https://<IP>:8776/`, not `http` and not a proxied hostname, and accept the certificate |

## Pipecat Adapter

The following table lists symptoms of the Pipecat adapter:

| Symptom | Cause and Fix |
| --- | --- |
| A Realtime client gets an error on `input_audio_buffer.commit` | The Pipecat adapter ends turns with its own VAD and rejects manual commits |
| `voice_chat` fails against the demo | The client has no TLS option. Run the smoke test against a `ws://` server (eval mode, or the demo with `FDH_PIPECAT_TLS=false`) |
| The client receives RTVI messages, or a τ² session closes during a long pause | RTVI or the idle timeout was turned on. `adapters/pipecat/pipeline.py` must keep `enable_rtvi=False`, `idle_timeout_secs=None`, and `cancel_on_idle_timeout=False`, and the server runs with WebSocket pings off |
| The requested voice is not used | The Pipecat adapter keeps the catalog TTS voice and logs a warning |

## τ³ Runs

The following table lists symptoms of τ³ runs and their tools:

| Symptom | Cause and Fix |
| --- | --- |
| τ³: `Session configuration failed` | The server sent `error` on `session.update`. The voice log's `backend_error` has the code |
| τ³: `RuntimeError: Not connected to API` | Keepalive pings closed the session while τ² was frozen. The τ³ profiles set `ws_ping_interval_s: 0`, so check the profile in use |
| No reply to a real question | Check `delegation_decision`. `repair: timeout` means the frontend endpoint was slow (4 s budget, hedged at 1.5 s). `guard_backchannel` means the turn was a pure backchannel |
| An answer is never spoken | The user was still speaking, and answers are held until they stop (`output_hold`). Check `playback_outcome` for `not_heard` |
| τ²: `No module named 'pyaudio'` | In the τ² checkout, run `uv sync --extra voice --extra dev` (needs the `portaudio19-dev` package) |
| Gate A against the real server fails on the transcript | Gate A streams a tone, which the real ASR does not transcribe. Use the stub server for Gate A, and the smoke tests or a τ³ run for the real stack |
| `fingerprint_check` fails with `missing [...]` | The voice server or the gateway runs code from before the fingerprint change. Recreate `fdh-tau` |
| `fingerprint_check` fails because sessions disagree | The log holds several domains or older sessions. Use one fresh `FDH_EVENT_LOG` per arm and domain |
| `fingerprint_check` fails with `the arm declares` | A server runs another arm's profile or gateway config. Check `FDH_PROFILE` and `FDH_GATEWAY_CONFIG`, and query both `/health` endpoints. The control pairs with `gateway.baseline.yaml`, not `gateway.yaml` |
| Voice server fails with `replay_unheard_answer.enabled needs prompt_features.replay_intent` | M2 needs its frontend prompt variant. Use `tau3_arm_m2_replay.yaml` |
| A status update says "still working" after the answer | A status line is dropped when its run has already finished (`status_dropped`). If it occurs, report the session ID |
| A status sentence sounds templated | The status LLM call exceeded `backend.status_verbalizer.timeout_ms` (3000). `status_spoken.source` is `template_fallback` |
| `report.sh` prints `P0.1 … INVALID (re-run it)` | A backend call of that run was rate-limited (429) and waited for `Retry-After`. Re-run it in a quieter window |
| No `logs/fdh_workers/*.trace.jsonl` with `gateway.eval.yaml` | Check `"backend_trace": true` in the gateway `/health`, then the worker log for `backend trace:` warnings. A Hermes version without the plugin hook registry writes no trace |
| `relay_replay` reports every session `skipped` | The event log has `tool_calls_out` without `call_ids` and `arguments`, or no `--session-update` matches the sessions' `session_tools_sha256` |
| `backend_decision_replay replay` fails with `occurs 0 times in the system messages` | The decisions or traces come from another control, or Hermes changed how it embeds the ephemeral system prompt. Re-extract from the control's own traces |
| An F1 or F2 arm shows no `delegation_guard` or `answer_replayed` events | Look for `lexicon_missing`. The session's ASR `language_code` has no `delegation.guards.lexicon` entry, so the rules are inactive by design |
| `identity_ledger` never appears in an identity arm | `session_rules.identification_tools` is empty or misses the lookup. Check that the tool's name or description says it returns a person, or list the tools in `result_hints.identification_tools` |
| A manual-mode client gets `conversation_already_has_active_response` | It sent `response.create` while the previous turn was still deciding or speaking. Wait for `response.done`, as Realtime requires |
| A `config_hash` does not match an earlier run | The configuration paths moved from the prototype, so every `config_hash` changed. Pair runs by profile name. The prompt hashes are unchanged |
| `mv` or `rm` in `logs/` fails with permission denied | The container writes logs as root. Run `sudo chown -R "$USER" logs` |
