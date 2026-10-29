# Configuration Knobs

Profiles and gateway configurations are in `src/examples/frontend_delegation_agent/config/`. Unknown keys fail at load time. The voice service mounts `src/` read-only, so a voice-side configuration change needs a restart: `docker compose [$E] --profile $P restart frontend-delegation-agent-single-gpu`. The Hermes runtime (the OpenShell sandbox, or the plain `fda-hermes` container) runs the source built into the Hermes image, so with Hermes a change to a gateway configuration, a prompt, or other gateway code needs `up -d --build`. The LLM backend runs in the voice service, so a restart is enough for it.

## Backends

`backend.kind` in `config/gateway*.yaml`, from `FDA_BACKEND`, selects the backend (registry: `core/backends/__init__.py`, one folder per backend in `core/backends/<kind>/`):

| `backend.kind` | Runs | How to select | Backend prompt file |
| --- | --- | --- | --- |
| `hermes` (default) | One Hermes worker process per session in `fda-hermes` (OpenShell sandbox or plain) | No backend overlay | `prompts/<profile>/backend/hermes.yaml` |
| `llm` | One OpenAI SDK tool loop per session in the voice process | `docker/docker-compose.fda-llm.yaml` (after the eval overlay; never with the plain overlay) | `prompts/<profile>/backend/llm.yaml` |
| `emulator` | In-process scripted agent without a model | Tests only (`gateway.emulator.yaml`) | None |

The LLM overlay sets the following on the voice service and removes `depends_on` and the four Hermes and OpenShell services (`hermes-openshell-init`, `hermes-openshell-gateway`, `fda-hermes`, `fda-hermes-image`) from the recipe profile:

| Variable | Value | Meaning |
| --- | --- | --- |
| `FDA_BACKEND` | `llm` | `backend.kind` |
| `FDA_BACKEND_LINK` | `in_process` | The voice side's `backend.link`: the gateway and the worker run in the voice process (default `websocket`) |
| `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `https://inference-api.nvidia.com/v1` | The backend model, as `gateway.env` sets it for Hermes |
| `FDA_GATEWAY_LOG` | `logs/fda_gateway_events.pipecat.jsonl` | The gateway event log, at the Hermes path |
| `FDA_WORKER_LOG_DIR` | `logs/fda_workers_pipecat` | The backend traces, at the Hermes path |

With `link: in_process`, the voice side loads the gateway file in `backend.gateway_config`, `${FDA_GATEWAY_CONFIG:-gateway.emulator.yaml}` in `delegation_agent.yaml`. Compose sets `FDA_GATEWAY_CONFIG=gateway.yaml` on the voice service, and the eval overlay sets `gateway.eval.yaml`; Hermes ignores it there. Both backends therefore read the same gateway settings, prompt features, and domains. `profiles/realtime_eval.yaml` also reads `FDA_BACKEND_LINK`.

Gateway configuration sections (`config/gateway.yaml`):

- `backend:` (shared by every backend): `kind`, `model`, `base_url`, `api_key_env`, `agent_construct`, `max_iterations`, `run_budget_seconds`, `run_hard_deadline_s`, `unwind_timeout_s`, `tool_result_timeout_s`, `backend_trace`, `steer_timeout_s`, `status_timeout_s`, `request_overrides`, and `prompts.path` (`prompts/${FDA_PROMPT_PROFILE:-generic}/backend/{kind}.yaml`).
- `hermes:` (Hermes only): `provider`, `context_length`, `load_soul_identity`, `disable_streaming`.
- `llm:` (LLM only): `reasoning_effort` (`medium`), `request_timeout_s` (180), `max_retries` (3), `verify_policy_follow_up_steps` (`${FDA_VERIFY_FOLLOW_UP_STEPS:-true}`), `follow_up_rules` (`[]`), `follow_up_ignore_tools` (`[]`).

Renamed keys fail at load with an error that names the new one: `workers.agent_kind` and `FDH_AGENT_KIND` are now `backend.kind` and `FDA_BACKEND`; shared keys under `hermes:` moved to `backend:`; `workers.mode: in_process_fake` and the voice `backend.link: in_process_fake` are now `in_process`. The backend worker protocol is version 2, so a Hermes image built from older source needs `up -d --build`.

## Profiles

`FDA_PROFILE` (`adapters/pipecat/profiles.py`) selects the files in the following table. Unset means `default`. Any other value than `default` or `eval` fails each session with an error that names both.

| | `default` (browser UI) | `eval` (OpenAI Realtime, benchmarks) |
| --- | --- | --- |
| Delegation profile | `profiles/pipecat_demo.yaml` | `profiles/pipecat_realtime_eval.yaml` |
| Pipecat settings | `adapters/pipecat.yaml` | `adapters/pipecat.eval.yaml` |
| Gateway configuration (`FDA_GATEWAY_CONFIG`; on `fda-hermes` for Hermes, on the voice service for `llm`) | `gateway.yaml` | `gateway.eval.yaml` (eval overlay) |

Realtime sessions always use the `eval` delegation profile and Pipecat settings. The code passes each profile's Pipecat settings file explicitly, so `FDA_PIPECAT_SETTINGS` has no effect on the Compose recipe.

## Prompt Profiles

A prompt profile is a folder `config/prompts/<profile>/` with `frontend.yaml` (voice side, `frontend_system` and the frontend lines), `backend/hermes.yaml` (Hermes backend; the former `backend.yaml`, byte-identical), `backend/llm.yaml` (LLM backend: an independent full copy of the backend prompts plus the `harness_*` keys, the text the LLM backend adds to the conversation itself, including `harness_follow_up_check` and `harness_follow_up_draft`; a missing key fails to load), and `voice.yaml` (voice-layer messages). The two backend files are independent: edit both when a change applies to both backends. `FDA_PROMPT_PROFILE` selects it. The following table lists the shipped profiles:

| Profile | Default for | Content |
| --- | --- | --- |
| `customer_service` | Eval overlay (`${FDA_PROMPT_PROFILE:-customer_service}`; the eval files `profiles/realtime_eval.yaml`, `voice/realtime_eval.yaml`, and `gateway.eval.yaml` default to it too) | The customer-service prompts, formerly `config/prompts.yaml`, `prompts.backend.yaml`, and `prompts.voice.yaml`, content unchanged. Pinned by the eval pins |
| `generic` | `default` deployment, including its Realtime sessions (`${FDA_PROMPT_PROFILE:-generic}`; `delegation_agent.yaml`, `voice_agent.yaml`, and `gateway.yaml` default to it) | A copy of `customer_service/` today; to become more generic |
| `general_qa` | None | General question answering: the frontend delegates every request (`delegate=true`, filler "One moment."); the backend answers from the LLM with no tools |

`FDA_PROMPT_PROFILE` is the one FDA setting the shell or `.env` can set. Compose passes it to the voice service and `fda-hermes` (and the OpenShell controller into the sandbox), so both sides use the same profile. Never set it empty; an unknown name fails with an error that names the missing prompt file. Choose one per run with `FDA_PROMPT_PROFILE=<name> docker compose $E --profile $P up -d --build`. To add one, copy `config/prompts/customer_service/` (including `backend/`) to `config/prompts/<name>/`, edit, and set the variable; `test_fda_prompt_profiles.py` loads every profile. The prompt paths are part of the delegation and voice settings, so another profile changes the `config hash`.

## Compose Variables

`docker-compose.yml` sets every FDA variable of the voice service as a literal, so `.env` cannot change them, except `FDA_PROMPT_PROFILE`. Only the overlays change them. The eval overlay, `docker/docker-compose.fda-eval.yaml`, changes `FDA_PROFILE`, `PIPELINE_TLS`, the default of `FDA_PROMPT_PROFILE` on both services, and `FDA_GATEWAY_CONFIG` on both services. The LLM overlay sets the values in Backends above. The following table lists the voice service values:

| Variable | Value | Meaning |
| --- | --- | --- |
| `EXAMPLE_SELECTION` | `frontend-delegation-agent` | The UI shows only this example |
| `REALTIME_SERVICE_PLATFORM` | `singlegpu` | The catalog section of Realtime sessions |
| `FDA_PROFILE` | `default` (`eval` with the overlay) | The deployment profile |
| `FDA_MAX_SESSIONS` | `4` | `server.max_sessions`: one cap for browser sessions and a separate cap for Realtime sessions |
| `FDA_BACKEND_DELAY_S` | `5` | The demo delay before each backend run (browser sessions only) |
| `FDA_PROACTIVE_AFTER_S` | `15` | M1: one proactive status line after this many seconds of silence while the backend works |
| `FDA_EVENT_LOG` | `logs/fda_voice_events.pipecat.jsonl` | The voice event log |
| `FDA_SILENT_FRONTEND` | `false` (`true` with `docker/docker-compose.fda-silent-frontend.yaml`) | `delegation.silent_frontend` |
| `FDA_ACKNOWLEDGE_CALLER` | `false` (`true` with `docker/docker-compose.fda-acknowledge-caller.yaml`) | `delegation.guards.acknowledge_caller_while_working` (R4 acknowledgement lines) |
| `FDA_VERIFY_FOLLOW_UP_STEPS` | `true` | `llm.verify_policy_follow_up_steps` (LLM backend only). To turn it off, edit the gateway file |
| `FDA_PROMPT_PROFILE` | `${FDA_PROMPT_PROFILE:-generic}` (`${FDA_PROMPT_PROFILE:-customer_service}` with the eval overlay) | The prompt profile, `config/prompts/<profile>/`. The only value `.env` can change |
| `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3.5-lightning`, `https://inference-api.nvidia.com/v1` | The frontend model of Realtime sessions and the default of browser sessions |
| `FDA_BACKEND_URL` | `ws://default--fda-hermes.openshell.localhost:8080/v1/backend` (`ws://fda-hermes:8790/v1/backend` with the plain overlay) | The gateway, through the OpenShell gateway's network alias by default. Unused with `llm` |
| `FDA_GATEWAY_CONFIG` | `gateway.yaml` (`gateway.eval.yaml` with the eval overlay) | The gateway file of the in-process backend (`llm`). Unused with Hermes |
| `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | Empty | Replace the catalog speech server |
| `FBA_VOICE_EVENT_LOG`, `FBA_FILLER_LOG` | Empty | Voice-layer logs, off |
| `FBA_BACKEND_HISTORY` | `false` | Unchanged from the former FDA containers |
| `UVICORN_WS_PING_INTERVAL`, `UVICORN_WS_PING_TIMEOUT` | `0`, `20` | No server keepalive pings. Realtime sessions refuse to start otherwise |

The voice service also reads the shared `.env` (`NVIDIA_API_KEY`, `REALTIME_API_KEY`, `PIPELINE_TLS`, `ENABLE_WELCOME_MESSAGE`, and so on) and publishes `${PIPELINE_APP_PORT:-7860}`.

The `fda-hermes` service reads `docker/fda/gateway.env` and receives only `NVIDIA_API_KEY` and `FDA_PROMPT_PROFILE` from `.env`. In the OpenShell runtime, the controller passes every `gateway.env` key into the sandbox with the value from its own environment, so an eval overlay value wins. It gives `NVIDIA_API_KEY` to the OpenShell provider only; the sandbox sees a placeholder. `gateway.env` holds plain `KEY=value` lines:

| Variable | Value |
| --- | --- |
| `FDA_GATEWAY_CONFIG` | `gateway.yaml` (`gateway.eval.yaml` with the overlay) |
| `FDA_PROMPT_PROFILE` | `generic`; replaced by the Compose value (`customer_service` with the eval overlay, or the shell or `.env` value) |
| `FDA_GATEWAY_MAX_SESSIONS` | `8`, passed as `--max-sessions` |
| `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `https://inference-api.nvidia.com/v1` |
| `FDA_GATEWAY_LOG` | `logs/fda_gateway_events.pipecat.jsonl` |
| `FDA_WORKER_LOG_DIR` | `logs/fda_workers_pipecat` |

`gateway.env` deliberately does not set `FDA_BACKEND_TRACE`, `FDA_WARM_WORKERS`, or `FDA_BACKEND`, because their defaults differ by gateway configuration. The plain overlay also sets `FDA_GATEWAY_HOST=0.0.0.0`. In the sandbox the gateway binds the default, `127.0.0.1`, and OpenShell reaches it there. `docker/fda/gateway.sh` starts the gateway on port 8790 and prints the configuration and the image's source hash. `tests/unit/frontend_delegation_agent/test_fda_compose.py` checks these rules.

## OpenShell Runtime

Every OpenShell file is in `docker/fda/openshell/`. The following table lists them:

| File | Purpose |
| --- | --- |
| `VERSION` | The one pinned OpenShell release (`v0.1.5-pre.4`), its commit, and the image digests. `test_fda_openshell_pins.py` checks that the 5 pins in Compose, `gateway.toml`, and the CLI build arguments match it |
| `gateway.toml` | The OpenShell gateway settings: bind `0.0.0.0:8080`, no TLS, no user authentication, pinned sandbox and supervisor images, `image_pull_policy = "if_not_present"` |
| `policy.yaml` | The sandbox policy: UID/GID 1500, read-only `/app` except `/app/logs` and `/app/.cache`, Landlock required, and `network_policies` for extra outbound hosts (none today) |
| `nvidia-fda.yaml` | The provider profile: the real `NVIDIA_API_KEY`, only for `inference-api.nvidia.com:443`, only for Hermes' Python interpreter |
| `Dockerfile.cli` | The controller image: the OpenShell CLI built from source at the pinned commit, plus `curl`, `jq`, and `bash` |
| `controller.env` | Controller-only settings: `OPENSHELL_GATEWAY_ENDPOINT`, `FDA_IMAGE` |
| `controller.sh` | The `fda-hermes` controller: kernel check, provider setup, sandbox create, log save and delete on stop |

`policy.yaml`, `nvidia-fda.yaml`, and `controller.sh` are mounted into the controller, so a change to them needs `docker compose [$E] --profile $P restart fda-hermes`, which saves the logs, deletes the sandbox, and creates a new one. A change to `controller.env` needs `up -d`. A change to `gateway.toml` needs `docker compose [$E] --profile $P restart hermes-openshell-gateway`. A change to `VERSION` must update all 5 pins together.

## Image

The following table lists the Hermes image inputs:

| Setting | Where | Default |
| --- | --- | --- |
| Hermes checkout baked into the image | `.env` `HERMES_AGENT_DIR` (the build context `hermes` of `fda-hermes-image`) | `../hermes-agent`, relative to the repository root |
| Image dependencies | `docker/fda/requirements.txt` (Python 3.14 lock with hashes, regenerated as its header says) and `docker/fda/overrides.txt` | The Hermes commit in the lock header, Pipecat 1.11.0 |
| Repository extra | `pyproject.toml` `fda` extra, installed in the shared app image with `uv sync --frozen --extra fda` | Must name the same packages as the lock header |
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
| Simulated backend delay | `backend.simulated_delay.{seconds,where}`, env `FDA_BACKEND_DELAY_S` | 0; 5 in `browser_demo.yaml` and `pipecat_demo.yaml` |
| Frontend model | env `FRONTEND_LLM_MODEL`, `FRONTEND_LLM_BASE_URL`; `frontend.llm.*` | `nvidia/nvidia/nemotron-3.5-lightning`, thinking off |
| Frontend tool choice, hedging, and timeout | `frontend.tool_choice`, `frontend.hedge_after_ms`, `frontend.timeout_ms` | `named`, 1500, 4000 |
| Filler spoken when delegating | `delegation.speak_when_delegating` | `true` |
| Silent frontend for single-turn evaluations such as VoiceBench (filler of delegated turns and M1 status line logged, never spoken; direct answers spoken; a silent turn that holds a request is delegated as a `task` with an empty filler, logged as `delegation_guard` with `guard=silent_delegated`, except empty turns, backchannels, pauses, liveness turns, and turns while WORKING) | `delegation.silent_frontend`, env `FDA_SILENT_FRONTEND` | `false`; only `docker/docker-compose.fda-silent-frontend.yaml` sets `true`. Never set it empty. With it on, the `config hash` differs |
| Guards | `delegation.guards.backend_question_needs_delegate`, `backchannel_words` | On, 17 words |
| R4 acknowledgement lines (a backchannel while WORKING, with nothing else pending, gets a short line in place of silence, logged as `delegation_guard` with `guard=acknowledge_caller`; a repeated delegated filler while WORKING is replaced from the same pool, logged as `filler_deduped` with the reason `acknowledgement`; never for a backend question, a hold, a liveness ping, or with the silent frontend; only when the session's ASR language has a lexicon, so localize `backchannel_words` and `acknowledgement_lines` together) | `delegation.guards.{acknowledge_caller_while_working,acknowledgement_max_per_run,acknowledgement_min_interval_s,acknowledgement_min_since_agent_s}`, env `FDA_ACKNOWLEDGE_CALLER`; lines in `prompts/<profile>/frontend.yaml` `acknowledgement_lines` | Off, 2 per run, 5 s, 0 s; only `docker/docker-compose.fda-acknowledge-caller.yaml` sets `true`. Never set it empty. With it on, the `config hash` differs |
| Steering | `backend.steer_mode` | `auto` (redirect during a model call, else steer) |
| Status speech | `backend.status_verbalizer.{mode,timeout_ms}` | `llm` (template fallback), 3000 |
| Output hold and stale policy | `output.hold_max_ms`, `output.on_stale` | 4000; keep answers and apologies, drop stale status lines and fillers |
| Tool or instruction change after the agent exists | `session.update_after_start` | `error` |
| Tool execution | `tools.executor` | `wire` (client executes, `eval`); `local` (server executes, browser profiles) |
| Backend model (both backends) | `gateway.yaml` `backend.model`, `base_url`, `request_overrides`; env `BACKEND_LLM_MODEL`, `BACKEND_LLM_BASE_URL` | `nvidia/nvidia/nemotron-3-ultra`, `enable_thinking: true`, budget 1024 |
| Backend budgets (both backends) | `backend.max_iterations`, `run_budget_seconds`, `run_hard_deadline_s` | 30, 300 (soft), 330 (hard, gateway watchdog; kills a Hermes worker process) |
| LLM backend client | `llm.reasoning_effort`, `request_timeout_s`, `max_retries` | `medium`, 180, 3 |
| LLM backend policy follow-up check (after a successful write that matches an English "After X, always/must ..." rule of the client's `session.update` instructions, the draft answer is withheld once and `harness_follow_up_check` asks for the rule's follow-up steps; nothing without a matching rule; gateway events `follow_up_obligations` at configure and `follow_up_check` per check) | `llm.verify_policy_follow_up_steps`, env `FDA_VERIFY_FOLLOW_UP_STEPS`; `llm.follow_up_rules` (`{tools: [...], text: "...", domain?: "..."}`, for policies the extraction cannot read: other languages, tables, catch-all tools); `llm.follow_up_ignore_tools` (tools that never open an extracted rule) | On in `gateway.yaml` and `gateway.eval.yaml`; `[]`, `[]`. Not used by Hermes |
| Worker processes | `workers.start_timeout_s`, `stop_timeout_s`, `kill_grace_s`, `warm`, `recovery.*` | 15, 12, 3, 0; respawn at most two per session |
| Speech endpoints | `voice_agent.yaml` `asr.catalog`, `tts.catalog` (the `singlegpu` section of the example's own `services.local.yaml`); env `FBA_ASR_SERVER`, `FBA_TTS_SERVER` | `nemo-speech:50051` |

## Fix Switches

The following table lists the switches of the evaluation fixes. The Python code defaults are off for all of them:

| Fix | Where | Default |
| --- | --- | --- |
| M1 proactive status | `output.proactive_status.{enabled,after_s,max_per_run}` (`after_s` from `FDA_PROACTIVE_AFTER_S`); lines in `prompts/<profile>/frontend.yaml` `status_proactive_lines` | On, 15 s, 1 per run |
| M3 spelled runs | `voice/*.yaml` `normalization.transcript.spelled_runs.{enabled,min_tokens,case}` | On, 3, `keep` |
| M3 spelling hold | `delegation.spelling_hold.{enabled,hold_ms,complete_patterns}` | On, 1500 ms, `[]`; `auto` in `realtime_eval.yaml`; off in both Pipecat profiles |
| M3 local "invalid ID" wording | `voice/*.yaml` `normalization.tool_arguments.invalid_message_key`, `escalate_invalid_message_key` | `tool_argument_invalid_readback`, `tool_argument_invalid_spell_all` (`voice/realtime_eval.yaml`) |
| I3 and I4 recovery notes | `voice/*.yaml` `normalization.tool_arguments.result_hints.*`; texts in `config/prompts/<profile>/voice.yaml` | On in `voice/realtime_eval.yaml` (`tools: auto`) |
| I1 phone numbers | `gateway.yaml` `prompt_features.phone_format` | On |
| Domain notes | `gateway.yaml` `prompt_features.domain_notes`, `domains.<name>.tools_any`; texts in `prompts/<profile>/backend/<kind>.yaml` `backend_domain_notes` | On, telecom only (`get_customer_by_phone`) |
| M2 replay of an unheard answer | `delegation.replay_unheard_answer.{enabled,ttl_s}` (needs `prompt_features.replay_intent`) | Off, 30 s |
| G1 filler de-duplication | `delegation.filler_dedupe.{enabled,recent}`; lines in `prompts/<profile>/frontend.yaml` `filler_alternatives` | On, 3 |
| G2 answer cleanup | `output.clean_answers` | On |
| Frontend prompt variants | `delegation_agent.yaml` `prompt_features.replay_intent` | Off |
| Backend prompt variants | `gateway.yaml` `prompt_features.{spelling_v2,spoken_output,write_consent,phone_format,domain_notes}` | All on; `gateway.eval.yaml` pins them |
| Geval backend variants (A2, B1 through C2) | `gateway.yaml` `prompt_features.{format_correction,write_consent_v2,policy_check,request_scope,transfer_check,capability_check}` | On |
| A1 and A1-lite format retry | `voice/*.yaml` `normalization.tool_arguments.format_retry.*`; texts `format_retry_*` in `config/prompts/<profile>/voice.yaml` | On in `voice/realtime_eval.yaml` (`mode: call`, `fields: phone`) |
| A3 field provenance | `voice/*.yaml` `normalization.tool_arguments.result_hints.field_provenance` | Off in `voice/realtime_eval.yaml` (the identification ledger replaces it) |
| Backend decision trace (P0.3), both backends (`core/backends/trace.py`) | `gateway.yaml` `backend.backend_trace`, env `FDA_BACKEND_TRACE` | Off; on in `gateway.eval.yaml` and the configs that extend it |
| F2 no-silence guards | `delegation.guards.{liveness_reply,question_never_silent,ping_as_status}`; phrase lists `delegation.guards.lexicon.<language>`; lines in `prompts/<profile>/frontend.yaml` `liveness_lines` | On in `realtime_eval.yaml`; `en` ships |
| F1 replay triggers | `delegation.replay_unheard_answer.{triggers,whole_if_heard_below}` | `[status_intent]`, 0 in `delegation_agent.yaml`; `[backchannel, ping, repeat]`, 0.2, `ttl_s: 60` in `realtime_eval.yaml` |
| F4 situational facts | `gateway.yaml` `prompt_features.situational_facts` | On |
| F8 identification ledger | `voice/*.yaml` `normalization.tool_arguments.result_hints.{ledger,identification_tools,...}`; text `lookup_ledger` in `config/prompts/<profile>/voice.yaml` | On in `voice/realtime_eval.yaml` |
| F8 name inside an identifier | `gateway.yaml` `prompt_features.identity_from_identifier` | On |
