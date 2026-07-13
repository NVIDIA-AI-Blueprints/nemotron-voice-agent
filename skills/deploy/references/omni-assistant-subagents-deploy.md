# Omni Assistant Subagents Cascaded Example — Deployment Reference

Use this reference from the `deploy` skill when deploying the examples/omni_assistant_subagents example — Nemotron 3 Omni split across five cooperating Pipecat workers (transport, speaker, media analyzer, webcam, thinker) that share a single `WorkerBus`.

## When to use

Pinning a Docker Compose deployment to the Omni Assistant Subagents example. Recipe profile names are `<example>` for cloud-only and `<example>/<hardware>` for on-prem. The companion `omni-assistant` example is a separate recipe (see its deploy reference). Selector modes (`all`, or a single `<example>`) are host-native only — they are not exposed as compose profiles.

This example declares `capabilities: [attachments, webcam]` in `examples_registry.yaml`. The browser UI gates the attachment upload control and the webcam panel on these capabilities, and the backend exposes `POST /api/sessions/{id}/attachments`, `POST /api/sessions/{id}/webcam/frames`, and `GET /api/webcam-config` for them.

Per-example catalogs at `src/examples/omni_assistant_subagents/services.{cloud,local}.yaml` are auto-selected on container startup because the registry resolves the example for the active recipe.

Hardware support: cloud-only, workstation, and `dgx-spark`, matching `omni-assistant`.

## Compose deploy

```bash
# Cloud (NVCF)
docker compose --profile omni-assistant-subagents up -d

# Workstation (local Omni vLLM + NIM TTS)
docker compose --profile omni-assistant-subagents/workstation up -d

# DGX Spark (local Omni vLLM + NIM TTS)
docker compose --profile omni-assistant-subagents/dgx-spark up -d
```

| Recipe profile | App service | Sidecars from `docker/` |
| --- | --- | --- |
| `omni-assistant-subagents` | `omni-assistant-subagents` | none (cloud NVCF) |
| `omni-assistant-subagents/workstation` | `omni-assistant-subagents` | `nvidia-llm-vllm-omni`, `tts-service` |
| `omni-assistant-subagents/dgx-spark` | `omni-assistant-subagents` | `nvidia-llm-vllm-omni`, `tts-service` |

Tear down with the same recipe used at `up` time.

## Verify

- UI at `https://<host>:7860/` by default, or `http://<host>:7860/` when `PIPELINE_TLS=false`. The sidebar shows a webcam panel and the conversation panel shows an attachment upload control.
- App logs: `docker compose logs --tail 200 omni-assistant-subagents`. Look for `Starting Nemotron Omni Assistant Subagents pipeline ... agents=transport,speaker,media,webcam,thinker`.
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
