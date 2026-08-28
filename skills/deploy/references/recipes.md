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
| `frontend-backend-agent` | `frontend-backend-agent` | `booking-server` |
| `frontend-backend-agent/server` | `frontend-backend-agent-server` | `booking-server`, `nvidia-llm`, `nemotron-asr-streaming-english`, `magpie-multilingual-tts-service` |
| `frontend-backend-agent/single-gpu` | `frontend-backend-agent-single-gpu` | `booking-server`, `nvidia-llm-vllm-lightning`, `nemo-speech` (ASR + TTS) |

Local catalogs merge by TCP reachability: NIM sidecars (`*/server`) and NeMo-Speech.cpp (`*/single-gpu`) appear when those endpoints are up. Host-native `uv run` uses the same rule.

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
- **Frontend/Backend:** every recipe includes `booking-server`. Also check `docker compose logs --tail 200 booking-server`.
- **`generic-assistant/server-perf`:** four-GPU layout, 200 Uvicorn workers, not a browser UI session. See `benchmarking_tools/scaling-perf/README.md`.
