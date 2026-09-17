# Phase 4 — Deploy & Validate

Run the upgraded agent end-to-end to confirm runtime (not just import/test) correctness. Use the repo's
`deploy` skill for mechanics. Needs a deploy env (host `uv`, or Docker + `NVIDIA_API_KEY`). If unavailable,
report DEFERRED with the recipe list left to validate.

## Step 1 — Pick surface

Validate at least one cloud-only recipe end-to-end, plus every example whose `pipeline.py` changed. Order by
cost: `generic-assistant` (fast) → `omni-assistant`, `omni-assistant-subagents` → `frontend-backend-agent`,
`multilingual-assistant`.

## Step 2 — Host-native smoke (fastest)

```bash
test -f .env || cp .env.example .env   # NVIDIA_API_KEY
uv sync --dev
uv run <entrypoint>                     # discover via EXAMPLE_SELECTION; confirm pipeline builds + binds
cd client && npm ci && npm run build    # client must compile against the bumped @pipecat-ai/* + RTVI APIs
```

Watch: server import/transport/runner/RTVI-setup errors; client `tsc`/build errors on renamed RTVI APIs.
Failures → back to Phase 3.

## Step 3 — Compose deploy

```bash
docker compose --profile generic-assistant up -d --build   # --build: src/deps changed
docker compose ps
docker compose logs --tail 200 generic-assistant
```

Add `--profile tracing` for spans. Validate a websocket/on-prem path too if transports/serializers changed.

## Step 4 — Live session loop

Per example (browser `client/` or benchmarking harness):

- **Connect**: start a WebRTC session; confirm the transport handshake (most version-sensitive: SmallWebRTC /
  serializer / runner).
- **Exercise**: VAD/turn fires → ASR transcript → LLM response → TTS playback → RTVI messages render. Also
  `omni_assistant_subagents` (subagent bus routing); `frontend_backend_agent` (planner + tools + TTS filter).
- **Observe**: server/Compose logs + browser console for frame exceptions, RTVI mismatches, transport drops,
  observer errors.
- **Fix & re-run** (max 3/example): diagnose (query `pipecat-docs` MCP if unclear) → fix code/`.env` →
  redeploy → re-run; if shared code changed, re-validate passing examples.

## Step 5 — Report

```text
Server pipecat-ai {old}→{new} (+ subpackages) | Client @pipecat-ai/* {old}→{new} | Mode {host|compose} | Recipes {...}
  <recipe>: ✅/❌ {note}
Client build: ✅/❌ | Runtime issues fixed: N | Remaining: M | server↔client RTVI contract changes: {...}
```

After all pass, do one final run of the touched examples to confirm no regressions.
