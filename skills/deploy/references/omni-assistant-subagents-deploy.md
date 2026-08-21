# Omni Assistant Subagents Cascaded Example — Deployment Reference

Use this reference from the `deploy` skill when deploying the examples/omni_assistant_subagents example — Nemotron 3 Omni split across five cooperating Pipecat workers (transport, speaker, media analyzer, webcam, thinker) that share a single `WorkerBus`.

## When to use

Pinning a Docker Compose deployment to the Omni Assistant Subagents example. Use `omni-assistant-subagents` for cloud, `omni-assistant-subagents/server` for Omni NIM and NIM TTS, or `omni-assistant-subagents/single-gpu` for Omni vLLM and NeMo-Speech.cpp TTS. The companion `omni-assistant` example is a separate recipe. Selector modes are host-native only and are not exposed as Compose profiles.

This example declares `capabilities: [attachments, webcam]` in `examples_registry.yaml`. The browser UI gates the attachment upload control and the webcam panel on these capabilities, and the backend exposes `POST /api/sessions/{id}/attachments`, `POST /api/sessions/{id}/webcam/frames`, and `GET /api/webcam-config` for them.

Per-example catalogs at `src/examples/omni_assistant_subagents/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

Hardware support: cloud-only, `server`, and `single-gpu` on workstations and DGX Spark.

## Compose deploy

```bash
# Cloud (NVCF)
docker compose --profile omni-assistant-subagents up -d

# Server (Omni NIM + NIM TTS, recommended for scaling)
docker compose --profile omni-assistant-subagents/server up -d

# One GPU on a workstation or DGX Spark
docker compose --profile omni-assistant-subagents/single-gpu up -d
```

| Recipe profile | App service | Sidecars from `docker/` |
| --- | --- | --- |
| `omni-assistant-subagents` | `omni-assistant-subagents` | none (cloud NVCF) |
| `omni-assistant-subagents/server` | `omni-assistant-subagents-server` | `nvidia-llm-omni`, `magpie-multilingual-tts-service` |
| `omni-assistant-subagents/single-gpu` | `omni-assistant-subagents-single-gpu` | `nvidia-llm-vllm-omni`, `nemo-speech-tts` |

Tear down with the same recipe used at `up` time.

## Verify

- UI at `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`. The sidebar shows a webcam panel and the conversation panel shows an attachment upload control.
- Cloud app logs: `docker compose logs --tail 200 omni-assistant-subagents`.
- Server app logs: `docker compose logs --tail 200 omni-assistant-subagents-server`.
- Single-GPU app logs: `docker compose logs --tail 200 omni-assistant-subagents-single-gpu`.
- In the active app logs, look for `Starting Nemotron Omni Assistant Subagents pipeline ... agents=transport,speaker,media,webcam,thinker`.
- Attachment upload check: `curl -F file=@image.jpg "https://<host>:7860/api/sessions/<session_id>/attachments?kind=image"` (use a session id from a live session).
- Webcam config check: `curl -fk https://<host>:7860/api/webcam-config`.

## Common failures

- **`pull access denied` / `unauthorized`** -> NGC login was not done or expired. See the root `deploy` skill.
- **App container exits with `ModuleNotFoundError: pipecat_subagents`** -> dependency desync. Rebuild with `docker compose --profile omni-assistant-subagents build`.
- **UI is missing the webcam / attachment surfaces** -> the active example does not declare the `attachments` / `webcam` capability. Verify `EXAMPLE_SELECTION` resolves to `omni-assistant-subagents` and `examples_registry.yaml` still lists `capabilities: [attachments, webcam]` on that entry.
- **Webcam panel uploads silently fail** -> browser blocked camera access. Confirm the page is served over HTTPS (`PIPELINE_TLS=true`) or `http://localhost` on the same host.
- **Media analyzer never runs after an upload** -> the speaker LLM did not set `selected_input_source=uploaded_attachment`. Check `omni-assistant-subagents` logs for `Speaker Omni queued media analysis trigger`. If absent, the prompt routing rules in `src/examples/omni_assistant_subagents/prompts.yaml` were overridden.
- **Omni vLLM issues** -> see `omni-assistant-deploy.md` (same sidecar).
- **Tear-down leaves orphan services after a service rename** -> rerun `up` or `down` with `--remove-orphans`.
