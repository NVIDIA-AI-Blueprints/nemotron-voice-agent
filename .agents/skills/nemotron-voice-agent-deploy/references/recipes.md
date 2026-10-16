# Recipes

Sidecars are the only per-example difference. Apply the same auth and preflight for the family. Do not invent extra steps.

`*/server` and `generic-assistant/server-perf` are **workstation only**. They are not supported on DGX Spark or Jetson Thor. On those hosts use `*/single-gpu` or cloud.

Per-example catalogs at `src/examples/<example>/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

| Recipe | App | Sidecars |
| --- | --- | --- |
| `generic-assistant` | `generic-assistant` | — |
| `generic-assistant/server` | `generic-assistant-server` | `nvidia-llm`, `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service` |
| `generic-assistant/server-perf` | `generic-assistant-server-perf` | `nvidia-llm-perf`, `nemotron-asr-streaming-english-perf`, `magpie-multilingual-tts-service-perf` |
| `generic-assistant/single-gpu` | `generic-assistant-single-gpu` | `nvidia-llm-vllm-lightning`, `nemo-speech` (ASR + TTS) |
| `multilingual-assistant` | `multilingual-assistant` | — |
| `multilingual-assistant/server` | `multilingual-assistant-server` | `nvidia-llm`, `nemotron-asr-streaming-multilingual`, `magpie-multilingual-tts-service` |
| `multilingual-assistant/single-gpu` | `multilingual-assistant-single-gpu` | `nvidia-llm-vllm-lightning`, `nemo-speech-multilingual` (multilingual ASR + TTS) |
| `omni-assistant` | `omni-assistant` | — |
| `omni-assistant/server` | `omni-assistant-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
| `omni-assistant/single-gpu` | `omni-assistant-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` (TTS only) |
| `omni-assistant-subagents` | `omni-assistant-subagents` | — |
| `omni-assistant-subagents/server` | `omni-assistant-subagents-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
| `omni-assistant-subagents/single-gpu` | `omni-assistant-subagents-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` (TTS only). **Not supported on Jetson Thor.** |
| `frontend-backend-verdict-agent` | `frontend-backend-verdict-agent` | — (NVCF ASR + TTS). LLMs use the NVIDIA Inference Hub. Browser UI and OpenAI Realtime. Ask for the profile before `up`. |
| `frontend-backend-verdict-agent/server` | `frontend-backend-verdict-agent-server` | `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service`. LLMs use the NVIDIA Inference Hub. Browser UI and OpenAI Realtime. Ask for the profile before `up`. |
| `frontend-backend-verdict-agent/single-gpu` | `frontend-backend-verdict-agent-single-gpu` | `nemo-speech` (ASR + TTS). LLMs use the NVIDIA Inference Hub. Browser UI and OpenAI Realtime; the only FBV recipe for the `eval` profile. Ask for the profile before `up`. |
| `frontend-delegation-hermes-agent/single-gpu` | `frontend-delegation-hermes-agent-single-gpu` | `nemo-speech` (ASR + TTS), `fdh-hermes` (FDH gateway and Hermes workers), `fdh-hermes-image` (build only). LLMs use the NVIDIA Inference Hub. Browser UI and OpenAI Realtime. x86_64 workstation only. Use `nemotron-voice-agent-fdh-deploy`. |

`NVIDIA_API_KEY` is required for cloud-only, `*/server`, and `generic-assistant/server-perf`. `HF_TOKEN` is required for `*/single-gpu`. Cloud catalog entries appear only when `NVIDIA_API_KEY` is set. Local catalogs merge by TCP reachability: NIM sidecars (`*/server`) and NeMo-Speech.cpp (`*/single-gpu`) appear when those endpoints are up. Host-native `uv run` uses the same rule.

UI is `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`.

## Example Deltas

Do not treat these as extra preflights.

- **Generic:** cascaded NVIDIA STT, NIM/vLLM LLM, NVIDIA TTS, with function calling.
- **Omni:** Nemotron 3 Nano Omni handles ASR and LLM in one multimodal chat-completions call. No ASR sidecar. Magpie or `nemo-speech-tts` speaks the reply. Companion `omni-assistant-subagents` is a separate recipe. First vLLM start downloads `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` from Hugging Face (`HF_TOKEN`) and can take ~30 minutes. Orin-class Jetson is unsupported.
  - Server NIM health: `curl -f http://localhost:18002/v1/health/ready`
  - Single-GPU vLLM health: `curl -f http://localhost:8002/health`
- **Omni Subagents:** same Omni sidecars. Five Pipecat workers (transport, speaker, media analyzer, webcam, thinker) share a `WorkerBus`. Declares `capabilities: [attachments, webcam]` in `examples_registry.yaml`. Backend: `POST /api/sessions/{id}/attachments`, `POST /api/sessions/{id}/webcam/frames`, `GET /api/webcam-config`. `omni-assistant-subagents/single-gpu` is **not supported on Jetson Thor** (workstation and DGX Spark only). On Thor use cloud (`omni-assistant-subagents`).
  - App logs should show `Starting Nemotron Omni Assistant Subagents pipeline ... agents=transport,speaker,media,webcam,thinker`.
  - Attachment check with the default self-signed certificate: `curl -fk -F file=@image.jpg "https://<host>:7860/api/sessions/<session_id>/attachments?kind=image"`
  - Webcam config: `curl -fk https://<host>:7860/api/webcam-config`
  - Missing webcam/attachment UI → `EXAMPLE_SELECTION` is not `omni-assistant-subagents`, or the registry lost those capabilities.
  - Webcam uploads fail silently → page is not HTTPS (`PIPELINE_TLS=true`) and not `http://localhost`.
  - Media analyzer never runs after upload → speaker did not set `selected_input_source=uploaded_attachment`. Look for `Speaker Omni queued media analysis trigger`. If absent, `src/examples/omni_assistant_subagents/prompts.yaml` was overridden.
  - `ModuleNotFoundError: pipecat_subagents` → rebuild: `docker compose --profile omni-assistant-subagents build`.
- **Frontend/Backend Verdict:** three recipes: `frontend-backend-verdict-agent` (cloud, NVCF speech), `frontend-backend-verdict-agent/server` (ASR and TTS NIM sidecars), and `frontend-backend-verdict-agent/single-gpu` (`nemo-speech`). Every recipe needs `NVIDIA_API_KEY` set to an NVIDIA Inference Hub key for both LLMs, plus the family rule: cloud sends the same key to NVCF speech, `/server` also uses it for `docker login nvcr.io` and the NIM sidecars, and `/single-gpu` also needs `HF_TOKEN`. Whether one Inference Hub key also works for NVCF speech or NGC has not been verified; if it fails, use `/single-gpu`. `REALTIME_API_KEY` protects the Realtime socket. Optional `WEATHERAPI_KEY`, `FINNHUB_API_KEY`, and `PERPLEXITY_API_KEY` enable the browser tools. It does not use `NVIDIA_SPEECH_API_KEY`.
  - Profile: before `up`, determine `default` or `eval`. If the request does not say it, ask "Which profile: `default` (for the browser UI) or `eval` (for evaluation runs such as Tau)?" and wait. Do not choose silently, and do not treat an existing `.env` `FBV_PROFILE` as the answer: show it and ask the user to confirm or change it.
  - `default` (browser UI; Compose default when `FBV_PROFILE` is unset): `docker compose --profile <recipe> up -d` for the recipe the user wants. It serves the browser UI (WebRTC and WebSocket) and OpenAI Realtime. Do not write `FBV_PROFILE` to `.env`.
  - `eval` (evaluation runs such as Tau): `/single-gpu` only, `docker compose -f docker-compose.yml -f docker/docker-compose.fbv-eval.yaml --profile frontend-backend-verdict-agent/single-gpu up -d`. Pass both `-f` files in every Compose command for that deployment. It serves only `WS /v1/realtime?model=nvidia/nemotron-realtime-frontend-backend-verdict`, over plain `ws://`, and writes event and filler logs to `logs/`. Browser sessions are refused; the UI shows nothing and the app log shows the error.
  - Optional local frontend LLM (`default` only): add `-f docker-compose.yml -f docker/docker-compose.fbv-local-llm.yaml` to every Compose command. It adds `nvidia-llm-vllm-lightning` to `/single-gpu` and `nvidia-llm` (NIM) to `/server`; without it no LLM container starts. The UI **LLM** menu lists `nemotron-3.5-lightning-local` only while reachable; the Inference Hub entry stays the default, and the backend LLM stays on the Inference Hub. Do not use it for `eval`: an eval container refuses a Realtime session whose `ek_` key selects a non-default LLM (`server_error`, close 1011).
  - Tracing: `ENABLE_TRACING=true` plus `--profile tracing` traces browser UI sessions. Realtime sessions also need `FBV_REALTIME_TRACING=true` and are never traced with `FBV_PROFILE=eval`, which ignores every `FBV_REALTIME_*` switch. Tau runs must use `eval`.
  - Other Realtime opt-ins (`default` only; never set them for an evaluation): `FBV_REALTIME_TURN_MODES` (comma list of `server_vad`, `manual`, `semantic_vad`) allows `input_audio_buffer.commit` turns and Smart Turn `semantic_vad`; `FBV_REALTIME_MCP=true` plus `REALTIME_MCP_ALLOWED_SERVER_URLS` allows server-run, hidden MCP tools with `require_approval: "never"`. The eval override sets `FBV_REALTIME_TURN_MODES=server_vad` and `FBV_REALTIME_MCP=false`.
  - `ENABLE_WELCOME_MESSAGE=false` turns off the browser UI greeting; Realtime sessions never greet. `ek_` client secrets log in on the FBV route; their standard session fields are ignored.
  - Realtime sessions use the `eval` settings in both profiles. Only `default` and `eval` are valid; old values such as `tau3_eval*` fail the first session with an error that names both. `FBV_PIPECAT_SETTINGS` no longer exists.
  - Do not edit `src/examples/frontend_backend_verdict/core/config/prompts/tau.yaml` to tune prompts. It is pinned by SHA-256, and a change needs a Tau A/B run. Tune `generic.yaml` for the browser UI.
  - Verify: the ready line and `prompts <path> (sha256 <hash>)` appear in `docker compose logs <app-service>` after the first session connects. Browser sessions log `profile=default` and `generic.yaml`; Realtime sessions log `profile=eval` and `tau.yaml`. Report the profile, catalog path, and SHA-256.
  - Compose sets `UVICORN_WS_PING_INTERVAL=0` and `UVICORN_WS_PING_TIMEOUT=20` on every FBV recipe; host-native runs must set them too. A mismatch fails Realtime sessions with `server_error` and close code 1011. `server_busy` with close code 1013 means the 8-session cap or a failed speech warm-up.
  - Full procedure: `docs/how-to/deploy-frontend-backend-verdict-agent.md`. Example details: `src/examples/frontend_backend_verdict/README.md`.
- **Frontend Delegation Hermes:** one recipe, `frontend-delegation-hermes-agent/single-gpu`, plus the `docker/docker-compose.fdh-eval.yaml` overlay for `eval`. The Hermes backend runs in its own `fdh-hermes` container (Python 3.14), built from a Hermes checkout (`HERMES_AGENT_DIR`), so start it with `up -d --build`. Needs `HF_TOKEN` and `NVIDIA_API_KEY` (Inference Hub key). Load `nemotron-voice-agent-fdh-deploy` for the procedure; the user guide is `docs/how-to/deploy-frontend-delegation-hermes-agent.md`.
- **`generic-assistant/server-perf`:** four-GPU Blackwell layout, pinned NVFP4 TP2 LLM profile, 200 Uvicorn workers, not a browser UI session. On older hardware, select a compatible TP2 profile first. See `benchmarking_tools/scaling-perf/README.md`.
