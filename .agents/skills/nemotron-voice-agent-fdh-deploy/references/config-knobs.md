# Configuration Knobs

Profiles and gateway configurations are in `src/examples/frontend_delegation_hermes/config/`. Unknown keys fail at load time. The voice service mounts `src/` read-only, so a voice-side configuration change needs a restart: `docker compose [$E] --profile $P restart frontend-delegation-hermes-agent-single-gpu`. The Hermes runtime (the OpenShell sandbox, or the plain `fdh-hermes` container) runs the source built into the Hermes image, so a change to a gateway configuration, a backend prompt, or other gateway code needs `up -d --build`.

## Profiles

`FDH_PROFILE` (`adapters/pipecat/profiles.py`) selects the files in the following table. Unset means `default`. Any other value than `default` or `eval` fails each session with an error that names both.

| | `default` (browser UI) | `eval` (OpenAI Realtime, benchmarks) |
| --- | --- | --- |
| Delegation profile | `profiles/pipecat_demo.yaml` | `profiles/pipecat_realtime_eval.yaml` |
| Pipecat settings | `adapters/pipecat.yaml` | `adapters/pipecat.eval.yaml` |
| Gateway configuration (Hermes runtime, `FDH_GATEWAY_CONFIG`) | `gateway.yaml` | `gateway.eval.yaml` (eval overlay) |

Realtime sessions always use the `eval` delegation profile and Pipecat settings. The code passes each profile's Pipecat settings file explicitly, so `FDH_PIPECAT_SETTINGS` has no effect on the Compose recipe.

## Compose Variables

`docker-compose.yml` sets every FDH variable of the voice service as a literal, so `.env` cannot change them. Only the eval overlay, `docker/docker-compose.fdh-eval.yaml`, changes `FDH_PROFILE`, `PIPELINE_TLS`, and the gateway's `FDH_GATEWAY_CONFIG`. The following table lists the voice service values:

| Variable | Value | Meaning |
| --- | --- | --- |
| `EXAMPLE_SELECTION` | `frontend-delegation-hermes-agent` | The UI shows only this example |
| `REALTIME_SERVICE_PLATFORM` | `singlegpu` | The catalog section of Realtime sessions |
| `FDH_PROFILE` | `default` (`eval` with the overlay) | The deployment profile |
| `FDH_MAX_SESSIONS` | `4` | `server.max_sessions`: one cap for browser sessions and a separate cap for Realtime sessions |
| `FDH_BACKEND_DELAY_S` | `5` | The demo delay before each backend run (browser sessions only) |
| `FDH_PROACTIVE_AFTER_S` | `15` | M1: one proactive status line after this many seconds of silence while the backend works |
| `FDH_EVENT_LOG` | `logs/fdh_voice_events.pipecat.jsonl` | The voice event log |
| `FDH_PROMPTS` | `prompts.yaml` | The frontend prompt catalog in `config/` |
| `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3.5-lightning`, `https://inference-api.nvidia.com/v1` | The frontend model of Realtime sessions and the default of browser sessions |
| `FDH_BACKEND_URL` | `ws://default--fdh-hermes.openshell.localhost:8080/v1/backend` (`ws://fdh-hermes:8790/v1/backend` with the plain overlay) | The gateway, through the OpenShell gateway's network alias by default |
| `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | Empty | Replace the catalog speech server |
| `FBA_VOICE_EVENT_LOG`, `FBA_FILLER_LOG` | Empty | Voice-layer logs, off |
| `FBA_VOICE_PROMPTS` | `prompts.voice.yaml` | The voice-layer prompt catalog |
| `FBA_BACKEND_HISTORY` | `false` | Unchanged from the former FDH containers |
| `UVICORN_WS_PING_INTERVAL`, `UVICORN_WS_PING_TIMEOUT` | `0`, `20` | No server keepalive pings. Realtime sessions refuse to start otherwise |

The voice service also reads the shared `.env` (`NVIDIA_API_KEY`, `REALTIME_API_KEY`, `PIPELINE_TLS`, `ENABLE_WELCOME_MESSAGE`, and so on) and publishes `${PIPELINE_APP_PORT:-7860}`.

The `fdh-hermes` service reads `docker/fdh/gateway.env` and receives only `NVIDIA_API_KEY` from `.env`. In the OpenShell runtime, the controller passes every `gateway.env` key into the sandbox with the value from its own environment, so an eval overlay value wins. It gives `NVIDIA_API_KEY` to the OpenShell provider only; the sandbox sees a placeholder. `gateway.env` holds plain `KEY=value` lines:

| Variable | Value |
| --- | --- |
| `FDH_GATEWAY_CONFIG` | `gateway.yaml` (`gateway.eval.yaml` with the overlay) |
| `FDH_GATEWAY_MAX_SESSIONS` | `8`, passed as `--max-sessions` |
| `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `https://inference-api.nvidia.com/v1` |
| `FDH_GATEWAY_LOG` | `logs/fdh_gateway_events.pipecat.jsonl` |
| `FDH_WORKER_LOG_DIR` | `logs/fdh_workers_pipecat` |

`gateway.env` deliberately does not set `FDH_BACKEND_TRACE`, `FDH_WARM_WORKERS`, or `FDH_AGENT_KIND`, because their defaults differ by gateway configuration. The plain overlay also sets `FDH_GATEWAY_HOST=0.0.0.0`. In the sandbox the gateway binds the default, `127.0.0.1`, and OpenShell reaches it there. `docker/fdh/gateway.sh` starts the gateway on port 8790 and prints the configuration and the image's source hash. `tests/unit/frontend_delegation_hermes/test_fdh_compose.py` checks these rules.

## OpenShell Runtime

Every OpenShell file is in `docker/fdh/openshell/`. The following table lists them:

| File | Purpose |
| --- | --- |
| `VERSION` | The one pinned OpenShell release (`v0.1.5-pre.4`), its commit, and the image digests. `test_fdh_openshell_pins.py` checks that the 5 pins in Compose, `gateway.toml`, and the CLI build arguments match it |
| `gateway.toml` | The OpenShell gateway settings: bind `0.0.0.0:8080`, no TLS, no user authentication, pinned sandbox and supervisor images, `image_pull_policy = "if_not_present"` |
| `policy.yaml` | The sandbox policy: UID/GID 1500, read-only `/app` except `/app/logs` and `/app/.cache`, Landlock required, and `network_policies` for extra outbound hosts (none today) |
| `nvidia-fdh.yaml` | The provider profile: the real `NVIDIA_API_KEY`, only for `inference-api.nvidia.com:443`, only for Hermes' Python interpreter |
| `Dockerfile.cli` | The controller image: the OpenShell CLI built from source at the pinned commit, plus `curl`, `jq`, and `bash` |
| `controller.env` | Controller-only settings: `OPENSHELL_GATEWAY_ENDPOINT`, `FDH_IMAGE` |
| `controller.sh` | The `fdh-hermes` controller: kernel check, provider setup, sandbox create, log save and delete on stop |

`policy.yaml`, `nvidia-fdh.yaml`, and `controller.sh` are mounted into the controller, so a change to them needs `docker compose [$E] --profile $P restart fdh-hermes`, which saves the logs, deletes the sandbox, and creates a new one. A change to `controller.env` needs `up -d`. A change to `gateway.toml` needs `docker compose [$E] --profile $P restart hermes-openshell-gateway`. A change to `VERSION` must update all 5 pins together.

## Image

The following table lists the Hermes image inputs:

| Setting | Where | Default |
| --- | --- | --- |
| Hermes checkout baked into the image | `.env` `HERMES_AGENT_DIR` (the build context `hermes` of `fdh-hermes-image`) | `../hermes-agent`, relative to the repository root |
| Image dependencies | `docker/fdh/requirements.txt` (Python 3.14 lock with hashes, regenerated as its header says) and `docker/fdh/overrides.txt` | The Hermes commit in the lock header, Pipecat 1.11.0 |
| Repository extra | `pyproject.toml` `fdh` extra, installed in the shared app image with `uv sync --frozen --extra fdh` | Must name the same packages as the lock header |
| Source hash | `/app/SOURCE_SHA256`, written at build time | Printed by `gateway.sh` at start |

## Pipecat Adapter Settings

The Pipecat settings files are `config/adapters/pipecat.yaml` (`default`) and `config/adapters/pipecat.eval.yaml` (`eval`). The profile selects one. These keys never enter the shared delegation config or its `config_hash`. A `null` keeps Pipecat's own default. The following table lists the keys:

| Key | `default` | `eval` |
| --- | --- | --- |
| `vad.confidence`, `vad.start_secs`, `vad.min_volume` | `null` | `null` |
| `vad.stop_secs` | `null` | 0.8 |
| `vad.audio_idle_timeout` (`VADProcessor`; 0 disables the wall-clock forced speech stop) | `null` | 0.0 |
| `turn.user_speech_timeout` (`SpeechTimeoutUserTurnStopStrategy`) | `null` | 0.0 |
| `turn.user_turn_stop_timeout` (`UserTurnProcessor`) | `null` | `null` |
| `audio.in_sample_rate`, `audio.out_sample_rate` | 16000, 24000 | 16000, 24000 |
| `audio.out_10ms_chunks` | 10 (100 ms output chunks) | 10 |
| `audio.write_timeout_s` | 120.0 | 120.0 |
| `tts.synthesis_mode` | `per_sentence` | `per_sentence` |

The app log reports the active settings in the `Pipecat adapter settings` line after the first session of each profile.

## Delegation and Voice Settings

The following table lists the shared knobs:

| Setting | Where | Default |
| --- | --- | --- |
| Transcript and tool-argument normalization | `voice/base.yaml`, `voice/browser.yaml`, `voice/realtime_eval.yaml` `normalization.*` | Transcript normalization on; tool arguments off in the browser profiles; `rules: auto` in `voice/realtime_eval.yaml` |
| Simulated backend delay | `backend.simulated_delay.{seconds,where}`, env `FDH_BACKEND_DELAY_S` | 0; 5 in `browser_demo.yaml` and `pipecat_demo.yaml` |
| Frontend model | env `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL`; `frontend.llm.*` | `nvidia/nvidia/nemotron-3.5-lightning`, thinking off |
| Frontend tool choice, hedging, and timeout | `frontend.tool_choice`, `frontend.hedge_after_ms`, `frontend.timeout_ms` | `named`, 1500, 4000 |
| Filler spoken when delegating | `delegation.speak_when_delegating` | `true` |
| Silent frontend for VoiceBench (filler of delegated turns and M1 status line logged, never spoken; direct answers spoken) | `delegation.silent_frontend`, env `FDH_SILENT_FRONTEND` | `false`; only `docker/docker-compose.fdh-voicebench.yaml` sets `true`. Never set it empty. With it on, the `config hash` differs |
| Guards | `delegation.guards.backend_question_needs_delegate`, `backchannel_words` | On, 17 words |
| Steering | `backend.steer_mode` | `auto` (redirect during a model call, else steer) |
| Status speech | `backend.status_verbalizer.{mode,timeout_ms}` | `llm` (template fallback), 3000 |
| Output hold and stale policy | `output.hold_max_ms`, `output.on_stale` | 4000; keep answers and apologies, drop stale status lines and fillers |
| Tool or instruction change after the agent exists | `session.update_after_start` | `error` |
| Tool execution | `tools.executor` | `wire` (client executes, `eval`); `local` (server executes, browser profiles) |
| Backend model | `gateway.yaml` `hermes.model`, `base_url`, `request_overrides`; env `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `enable_thinking: true`, budget 1024 |
| Hermes budgets | `hermes.max_iterations`, `run_budget_seconds`, `run_hard_deadline_s` | 30, 300 (soft), 330 (hard, kills the worker) |
| Worker processes | `workers.start_timeout_s`, `stop_timeout_s`, `kill_grace_s`, `warm`, `recovery.*` | 15, 12, 3, 0; respawn at most two per session |
| Speech endpoints | `voice_agent.yaml` `asr.catalog`, `tts.catalog` (the `singlegpu` section of the example's own `services.local.yaml`); env `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | `nemo-speech:50051` |

## Fix Switches

The following table lists the switches of the evaluation fixes. The Python code defaults are off for all of them:

| Fix | Where | Default |
| --- | --- | --- |
| M1 proactive status | `output.proactive_status.{enabled,after_s,max_per_run}` (`after_s` from `FDH_PROACTIVE_AFTER_S`); lines in `prompts.yaml` `status_proactive_lines` | On, 15 s, 1 per run |
| M3 spelled runs | `voice/*.yaml` `normalization.transcript.spelled_runs.{enabled,min_tokens,case}` | On, 3, `keep` |
| M3 spelling hold | `delegation.spelling_hold.{enabled,hold_ms,complete_patterns}` | On, 1500 ms, `[]`; `auto` in `realtime_eval.yaml`; off in both Pipecat profiles |
| M3 local "invalid ID" wording | `voice/*.yaml` `normalization.tool_arguments.invalid_message_key`, `escalate_invalid_message_key` | `tool_argument_invalid_readback`, `tool_argument_invalid_spell_all` (`voice/realtime_eval.yaml`) |
| I3 and I4 recovery notes | `voice/*.yaml` `normalization.tool_arguments.result_hints.*`; texts in `config/prompts.voice.yaml` | On in `voice/realtime_eval.yaml` (`tools: auto`) |
| I1 phone numbers | `gateway.yaml` `prompt_features.phone_format` | On |
| Domain notes | `gateway.yaml` `prompt_features.domain_notes`, `domains.<name>.tools_any`; texts in `prompts.backend.yaml` `backend_domain_notes` | On, telecom only (`get_customer_by_phone`) |
| M2 replay of an unheard answer | `delegation.replay_unheard_answer.{enabled,ttl_s}` (needs `prompt_features.replay_intent`) | Off, 30 s |
| G1 filler de-duplication | `delegation.filler_dedupe.{enabled,recent}`; lines in `prompts.yaml` `filler_alternatives` | On, 3 |
| G2 answer cleanup | `output.clean_answers` | On |
| Frontend prompt variants | `delegation_agent.yaml` `prompt_features.replay_intent` | Off |
| Backend prompt variants | `gateway.yaml` `prompt_features.{spelling_v2,spoken_output,write_consent,phone_format,domain_notes}` | All on; `gateway.eval.yaml` pins them |
| Geval backend variants (A2, B1 through C2) | `gateway.yaml` `prompt_features.{format_correction,write_consent_v2,policy_check,request_scope,transfer_check,capability_check}` | On |
| A1 and A1-lite format retry | `voice/*.yaml` `normalization.tool_arguments.format_retry.*`; texts `format_retry_*` in `config/prompts.voice.yaml` | On in `voice/realtime_eval.yaml` (`mode: call`, `fields: phone`) |
| A3 field provenance | `voice/*.yaml` `normalization.tool_arguments.result_hints.field_provenance` | Off in `voice/realtime_eval.yaml` (the identification ledger replaces it) |
| Backend decision trace (P0.3) | `gateway.yaml` `hermes.backend_trace`, env `FDH_BACKEND_TRACE` | Off; on in `gateway.eval.yaml` and the configs that extend it |
| F2 no-silence guards | `delegation.guards.{liveness_reply,question_never_silent,ping_as_status}`; phrase lists `delegation.guards.lexicon.<language>`; lines in `prompts.yaml` `liveness_lines` | On in `realtime_eval.yaml`; `en` ships |
| F1 replay triggers | `delegation.replay_unheard_answer.{triggers,whole_if_heard_below}` | `[status_intent]`, 0 in `delegation_agent.yaml`; `[backchannel, ping, repeat]`, 0.2, `ttl_s: 60` in `realtime_eval.yaml` |
| F4 situational facts | `gateway.yaml` `prompt_features.situational_facts` | On |
| F8 identification ledger | `voice/*.yaml` `normalization.tool_arguments.result_hints.{ledger,identification_tools,...}`; text `lookup_ledger` in `config/prompts.voice.yaml` | On in `voice/realtime_eval.yaml` |
| F8 name inside an identifier | `gateway.yaml` `prompt_features.identity_from_identifier` | On |
