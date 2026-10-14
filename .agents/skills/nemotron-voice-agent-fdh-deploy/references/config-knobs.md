# Configuration Knobs

Profiles and gateway configs are in `src/examples/frontend_delegation_hermes/config/`. Unknown keys fail at load time. `src/` is mounted read-only into the containers, so a configuration change needs a restart, not a rebuild. An edit to a file under `src/` does not make `up -d` recreate the container: run `docker compose -f $F restart <service>`, or `up -d --force-recreate <service>`. A changed environment variable, such as a new profile, does make `up -d` recreate it.

## Containers and Ports

The two services read separate variables, so a τ³ arm switch can never retarget the Pipecat service. The following table lists the Compose variables of `docker/docker-compose.frontend-delegation-hermes.yaml`:

| Setting | `fdh-voice` (Pipecat) | `fdh-tau` (τ³) |
| --- | --- | --- |
| Compose profile | None (always) | `tau` |
| Adapter (`FDH_ADAPTER`) | `pipecat` | `tau` |
| Container name | `FDH_PIPECAT_CONTAINER_NAME`, default `fdh-voice` | `FDH_CONTAINER_NAME`, default `fdh-tau` |
| Host port (container port 7860) | `FDH_PIPECAT_HOST_PORT`, default 8776 | `FDH_VOICE_HOST_PORT`, default 8775 |
| Voice profile | `FDH_PIPECAT_PROFILE`, default `pipecat_demo.yaml` | `FDH_PROFILE`, default `realtime_eval.yaml` |
| Gateway config | `FDH_PIPECAT_GATEWAY_CONFIG`, default `gateway.yaml` | `FDH_GATEWAY_CONFIG`, default `gateway.eval.yaml` |
| Pipecat settings | `FDH_PIPECAT_SETTINGS`, default `adapters/pipecat.yaml` | Not used |
| TLS | `FDH_PIPECAT_TLS`, default `true`; `FDH_PIPECAT_TLS_CERT` and `FDH_PIPECAT_TLS_KEY`, or a self-signed pair in the `fdh_certs` volume | `PIPELINE_TLS: "false"` |
| Voice sessions | `FDH_PIPECAT_MAX_SESSIONS`, default 4 | `FDH_MAX_SESSIONS`, default 4 |
| Gateway sessions | `FDH_PIPECAT_GATEWAY_MAX_SESSIONS`, default 8 | `FDH_GATEWAY_MAX_SESSIONS`, default 8 |
| Backend delay | `FDH_PIPECAT_BACKEND_DELAY_S`, default 5 (demo profile only) | Not set; the τ³ profiles never delay |
| M1 delay | `FDH_PIPECAT_PROACTIVE_AFTER_S`, default 15 | `FDH_PROACTIVE_AFTER_S`, default 15 |
| Frontend model | `FDH_PIPECAT_FRONTEND_LLM_MODEL`, `FDH_PIPECAT_FRONTEND_LLM_BASE_URL` | `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL` |
| Backend model | `FDH_PIPECAT_BACKEND_LLM_MODEL`, `FDH_PIPECAT_BACKEND_LLM_BASE_URL` | `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` |
| Gateway URL | `FDH_PIPECAT_BACKEND_URL` | `FDH_BACKEND_URL` |
| Voice event log | `FDH_PIPECAT_EVENT_LOG`, default `logs/fdh_voice_events.pipecat.jsonl` | `FDH_EVENT_LOG`, default `logs/fdh_voice_events.realtime_eval.jsonl` |
| Gateway event log | `FDH_PIPECAT_GATEWAY_LOG`, default `logs/fdh_gateway_events.pipecat.jsonl` | `FDH_GATEWAY_LOG`, default `logs/fdh_gateway_events.realtime_eval.jsonl` |
| Worker logs | `FDH_PIPECAT_WORKER_LOG_DIR`, default `logs/fdh_workers_pipecat` | `FDH_WORKER_LOG_DIR`, default `logs/fdh_workers` |
| Restart policy | `"no"` | `unless-stopped` |

Both models default to the Inference Hub (`https://inference-api.nvidia.com/v1`): `nvidia/nvidia/nemotron-3.5-lightning` for the frontend and `nvidia/nvidia/nemotron-3-ultra` for the backend. The gateway URL default is `ws://host.docker.internal:8790/v1/backend`, which maps to the in-container gateway.

The entrypoint `docker/fdh/entrypoint.sh` needs `FDH_ADAPTER` (or an argument), `FDH_PROFILE`, and `FDH_GATEWAY_CONFIG`. It runs `examples.frontend_delegation_hermes.server` for `pipecat` and `examples.frontend_delegation_hermes.<adapter>_server` for any other adapter. Each container runs its gateway on `127.0.0.1:8790`.

## Image

The following table lists the image inputs:

| Setting | Where | Default |
| --- | --- | --- |
| Hermes checkout baked into the image | `.env` `HERMES_AGENT_DIR` (the build context `hermes`) | `../../hermes-agent-smasurekar`, relative to `docker/` |
| Image dependencies | `docker/fdh/requirements.txt` (Python 3.14 lock with hashes, regenerated as its header says) and `docker/fdh/overrides.txt` | The Hermes commit in the lock header, Pipecat 1.11.0 |
| Repository extra | `pyproject.toml` `fdh` extra, installed by the `dev` group | Must name the same packages as the lock header |

## Pipecat Adapter Settings

The Pipecat settings files are `config/adapters/pipecat.yaml` (demo) and `config/adapters/pipecat.eval.yaml` (eval). Select one with `FDH_PIPECAT_SETTINGS` or `--pipecat-settings`. The value is a path, or a name relative to `config/`. These keys never enter the shared delegation config or its `config_hash`. A `null` keeps Pipecat's own default. The following table lists the keys:

| Key | Demo | Eval |
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

`/health` reports the active settings under `"pipecat"`.

## Delegation and Voice Settings

The following table lists the shared knobs:

| Setting | Where | Default |
| --- | --- | --- |
| End-of-turn silence (τ³ adapter) | `voice/base.yaml` `turn_detection.silence_duration_ms` | 800 (client values ignored) |
| Transcript and tool-argument normalization | `voice/base.yaml`, `voice/tau3.yaml`, `voice/tau3_spelling.yaml`, `voice/tau3_fixes.yaml`, `voice/realtime_eval.yaml` `normalization.*` | On for τ³; tool arguments off in the browser profiles; `rules: auto` in `voice/realtime_eval.yaml` |
| Simulated backend delay | `backend.simulated_delay.{seconds,where}`, env `FDH_BACKEND_DELAY_S` | 0; 5 in `browser_demo.yaml` and `pipecat_demo.yaml` |
| Frontend model | env `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL`; `frontend.llm.*` | `nvidia/nvidia/nemotron-3.5-lightning`, thinking off |
| Frontend tool choice, hedging, and timeout | `frontend.tool_choice`, `frontend.hedge_after_ms`, `frontend.timeout_ms` | `named`, 1500, 4000 |
| Filler spoken when delegating | `delegation.speak_when_delegating` (`tau3_eval_silent_ack.yaml` sets `false`) | `true` |
| Guards | `delegation.guards.backend_question_needs_delegate`, `backchannel_words` | On, 17 words |
| Steering | `backend.steer_mode` | `auto` (redirect during a model call, else steer) |
| Status speech | `backend.status_verbalizer.{mode,timeout_ms}` | `llm` (template fallback), 3000 |
| Output hold and stale policy | `output.hold_max_ms`, `output.on_stale` | 4000; keep answers and apologies, drop stale status lines and fillers |
| Tool or instruction change after the agent exists | `session.update_after_start` | `error` |
| Tool execution | `tools.executor` | `wire` (client executes, τ³ and Pipecat eval); `local` (server executes, browser profiles) |
| Backend model | `gateway.yaml` `hermes.model`, `base_url`, `request_overrides`; env `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `enable_thinking: true`, budget 1024 |
| Hermes budgets | `hermes.max_iterations`, `run_budget_seconds`, `run_hard_deadline_s` | 30, 300 (soft), 330 (hard, kills the worker) |
| Worker processes | `workers.start_timeout_s`, `stop_timeout_s`, `kill_grace_s`, `warm`, `recovery.*` | 15, 12, 3, 0; respawn at most two per session |
| Speech endpoints | `voice_agent.yaml` `asr.catalog`, `tts.catalog` (the single-GPU section of `frontend_backend_agent/services.local.yaml`); env `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | `nemo-speech:50051`; set `localhost:50051` for a host run |

## Fix Switches

The following table lists the switches of the τ³ fixes. The Python code defaults are off for all of them:

| Fix | Where | Default |
| --- | --- | --- |
| M1 proactive status | `output.proactive_status.{enabled,after_s,max_per_run}` (`after_s` from `FDH_PROACTIVE_AFTER_S`); lines in `prompts.yaml` `status_proactive_lines` | On, 15 s, 1 per run |
| M3 spelled runs | `voice/*.yaml` `normalization.transcript.spelled_runs.{enabled,min_tokens,case}` | On, 3, `keep` (off in `voice/tau3.yaml`) |
| M3 spelling hold | `delegation.spelling_hold.{enabled,hold_ms,complete_patterns}` | On, 1500 ms, `[]`; `auto` in `realtime_eval.yaml`; off in both Pipecat profiles |
| M3 local "invalid ID" wording | `voice/*.yaml` `normalization.tool_arguments.invalid_message_key`, `escalate_invalid_message_key` | `tool_argument_invalid_readback`, `tool_argument_invalid_spell_all` (`voice/tau3_spelling.yaml`, `voice/tau3_fixes.yaml`, `voice/realtime_eval.yaml`) |
| I3 and I4 recovery notes | `voice/*.yaml` `normalization.tool_arguments.result_hints.*`; texts in `config/prompts.voice.yaml` | On in `voice/tau3_fixes.yaml` and `voice/realtime_eval.yaml` (`tools: auto`) |
| I1 phone numbers | `gateway.yaml` `prompt_features.phone_format` | On (off in `gateway.baseline.yaml` and `gateway.identity_control.yaml`) |
| Domain notes | `gateway.yaml` `prompt_features.domain_notes`, `domains.<name>.tools_any`; texts in `prompts.backend.yaml` `backend_domain_notes` | On, telecom only (`get_customer_by_phone`); off in the controls and `gateway.eval_no_note.yaml` |
| M2 replay of an unheard answer | `delegation.replay_unheard_answer.{enabled,ttl_s}` (needs `prompt_features.replay_intent`) | Off, 30 s |
| G1 filler de-duplication | `delegation.filler_dedupe.{enabled,recent}`; lines in `prompts.yaml` `filler_alternatives` | On, 3 |
| G2 answer cleanup | `output.clean_answers` | On |
| Frontend prompt variants | `delegation_agent.yaml` `prompt_features.replay_intent` | Off |
| Backend prompt variants | `gateway.yaml` `prompt_features.{spelling_v2,spoken_output,write_consent,phone_format,domain_notes}` | All on; `gateway.eval.yaml` pins them |
| Geval backend variants (A2, B1 through C2) | `gateway.yaml` `prompt_features.{format_correction,write_consent_v2,policy_check,request_scope,transfer_check,capability_check}` | On; off in `gateway.eval_control.yaml` |
| A1 and A1-lite format retry | `voice/*.yaml` `normalization.tool_arguments.format_retry.*`; texts `format_retry_*` in `config/prompts.voice.yaml` | On in `voice/realtime_eval.yaml` (`mode: call`, `fields: phone`) |
| A3 field provenance | `voice/*.yaml` `normalization.tool_arguments.result_hints.field_provenance` | On in `voice/realtime_eval.yaml` |
| Backend decision trace (P0.3) | `gateway.yaml` `hermes.backend_trace`, env `FDH_BACKEND_TRACE` | Off; on in `gateway.eval.yaml` and the configs that extend it |
| F2 no-silence guards | `delegation.guards.{liveness_reply,question_never_silent,ping_as_status}`; phrase lists `delegation.guards.lexicon.<language>`; lines in `prompts.yaml` `liveness_lines` | On in `realtime_eval.yaml` and `realtime_eval_liveness.yaml`; `en` ships |
| F1 replay triggers | `delegation.replay_unheard_answer.{triggers,whole_if_heard_below}` | `[status_intent]`, 0 in the controls; `[backchannel, ping, repeat]`, 0.2, `ttl_s: 60` in `realtime_eval.yaml` |
| F4 situational facts | `gateway.yaml` `prompt_features.situational_facts` | On; off in `gateway.eval_124acf7.yaml` and every other control |
| F8 identification ledger | `voice/*.yaml` `normalization.tool_arguments.result_hints.{ledger,identification_tools,...}`; text `lookup_ledger` in `config/prompts.voice.yaml` | On in `voice/realtime_eval.yaml` and `voice/realtime_eval_identity.yaml` |
| F8 name inside an identifier | `gateway.yaml` `prompt_features.identity_from_identifier` | On; off in `gateway.eval_124acf7.yaml` and every other control |
