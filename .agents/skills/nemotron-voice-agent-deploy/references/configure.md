# Configure a Running Deployment

Open this file only to change the configuration of a deployed agent. Edit the smallest surface that satisfies the request and preserve unrelated keys and comments.

## Where Configuration Lives

| File | Owns |
| --- | --- |
| `.env` | Feature flags, tracing, chat history, audio debugging, buffering, supported vLLM memory overrides |
| `examples_registry.yaml` | `selection` (`all` or one example id, overridden by the `EXAMPLE_SELECTION` env var that each Compose app service sets), `transports`, and per example: `services` (offered keys per slot, first available is the default), `settings.<slot>.<key>` (setting defaults), `categories.<slot>` (slot reads another catalog category), `defaults.prompt` |
| `services.yaml` | Built-in LLM, ASR, and TTS entries under `server` (NIM) and `single-gpu` (vLLM and NeMo-Speech.cpp). An entry with `nvcf` is also served by NVIDIA Cloud |
| `settings.yaml` | Editable LLM setting profiles referenced by `settings: <profile>` in `services.yaml` |
| `src/examples/<package>/prompts.yaml` | Built-in prompts for that example |

Example ids map to packages under `src/examples/`: `generic-assistant` → `generic`, `multilingual-assistant` → `multilingual`, `omni-assistant` → `omni_assistant`, `omni-assistant-subagents` → `omni_assistant_subagents`, `frontend-backend-agent` → `frontend_backend_agent`.

## Validate

- Every key under an example's `services` exists in `services.yaml` for the active `SERVICE_RECIPE`, or has `nvcf` for the cloud recipe.
- Multilingual prompts pair with multilingual ASR (`nemotron-asr-streaming-multilingual`, `parakeet-rnnt`) and TTS (`magpie-multilingual-tts`, `magpie-zeroshot-tts`, `chatterbox-multilingual-tts`).
- Every TTS entry sets `synthesis_mode`: `stitched` for Magpie Multilingual and Magpie Zeroshot, `per_sentence` for Chatterbox.
- Local endpoints use Compose service names or aliases (`nvidia-llm:8000`, `nvidia-llm-vllm:8000`, `nvidia-llm-vllm-omni:8002`, `nemo-speech:50051`, `nemotron-asr-streaming-english:50052`, `magpie-multilingual-tts-service:50051`, and so on). Host-native runs rewrite them to `localhost`.
- Alternate local ASR and TTS (`parakeet-ctc-asr`, `parakeet-rnnt-asr`, `chatterbox-tts`, `magpie-zeroshot-tts` profiles) publish their own host ports, so they can run beside the default. Scale the default to zero only to free its GPU, for example `--scale magpie-multilingual-tts-service=0`.

## Apply

- `.env` change: re-run `docker compose --profile <recipe> up -d` with the same overlays (`tracing`, `turn`). The environment is read at container start.
- YAML change: `docker compose restart <app-service>`, then refresh the browser. `./src` and the root YAML files are bind-mounted, so no rebuild is needed.
- Dependency, `Dockerfile`, or `pipeline.py` import changes: add `--build`.
- UI-only services and prompts stay in browser localStorage. Write only repository files.

## Examples

- **Switch the default LLM:** move the key to the front of the example's `services.llm`, for example `llm: [nemotron-super, nemotron-lightning]`. Restart the app service.
- **Turn reasoning on for one service:** add `settings: {llm: {nemotron-lightning: {enable_thinking: true}}}` to the example. Restart the app service.
- **Add a multilingual persona prompt:** add it to the example's `prompts.yaml`, set it in `defaults.prompt`, and confirm multilingual ASR and TTS lead the example's `services`. Restart the app service.
- **Word-level TTS input streaming:** `NvidiaWordTTSService` is a source-level opt-in. Change only the import and constructor in `pipeline.py` as documented in `docs/how-to/configure-tts.md#word-level-input-streaming-and-timestamps`.
- **Turn speaker labels on:** Generic Assistant only, with an ASR entry that sets `speaker_diarization_supported: true`, for example `nemotron-asr-streaming-english`. In the browser, switch **Speaker Labels** > **Diarization** before you connect. To set the default for API clients, set `ENABLE_SPEAKER_DIARIZATION=true` in `.env` and re-run `docker compose --profile <recipe> up -d` with the same overlays. Verify with `docker compose logs <app-service> | grep "Speaker diarization enabled"`. Refer to `docs/how-to/enable-speaker-labels.md`.
- **Finalize ASR at each voice pause (`force_eou`):** set `ASR_FORCE_EOU=true` in `.env` and re-run `docker compose --profile <recipe> up -d` with the same overlays. Turns close about 0.3 s sooner. It is off by default. Refer to `docs/how-to/tune-pipeline-performance.md`.

## Troubleshooting

- **Change does not appear in the UI** -> restart the app service and refresh the browser.
- **Service missing from the Services tab** -> the key is not under the example's `services`, or it has no entry in the `services.yaml` section for the active `SERVICE_RECIPE`. Host-native runs list only reachable sidecars.
- **Local LLM returns 400 (`auto tool choice requires ...`) or speaks `<think>`** -> the reasoning and tool-call parsers are missing. They are set in `docker/docker-compose.nemotron3-*.yaml`. Refer to `docs/06-troubleshooting.md`.
- **Raw vLLM lacks `nemotron_v3` or Super (`MIXED_PRECISION`) does not load** -> the vLLM image is too old. Use NGC `vllm:26.07-py3` (vLLM 0.20 or later).
- **No Speaker Labels panel** -> select Generic Assistant and an ASR entry that sets `speaker_diarization_supported: true` in `services.yaml`.
