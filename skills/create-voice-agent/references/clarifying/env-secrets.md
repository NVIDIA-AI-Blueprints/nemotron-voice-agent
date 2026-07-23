# Env secrets gate | .env validation before scaffold

Trigger: any create/scaffold path (Pipecat or LiveKit, cloud or local). Run **after** hardware probe, **before** main gate questions (or in parallel with probe). FORBID scaffold until required secrets are present.

```
PROC: locate .env → probe keys → missing? guide + STOP WAIT → user confirms populated → continue gate
REQ: HF_TOKEN WAIT for any local Omni compose deployment | FORBID skip HF_TOKEN gate on local Omni
```

## Locate and probe
```bash
ENV_FILE="${ENV_FILE:-.env}"
test -f "$ENV_FILE" || echo "NO_ENV_FILE"
grep -E '^(NVIDIA_API_KEY|HF_TOKEN|LIVEKIT_URL|LIVEKIT_API_KEY|LIVEKIT_API_SECRET)=' "$ENV_FILE" 2>/dev/null \
  | sed 's/=.*/=***/' || true
# also check shell (may be exported without .env)
[ -n "${NVIDIA_API_KEY:-}" ] && echo "NVIDIA_API_KEY=set(shell)" || echo "NVIDIA_API_KEY=MISSING"
[ -n "${HF_TOKEN:-}" ] && echo "HF_TOKEN=set(shell)" || echo "HF_TOKEN=MISSING"
[ -n "${LIVEKIT_URL:-}" ] && echo "LIVEKIT_URL=set(shell)" || echo "LIVEKIT_URL=MISSING"
[ -n "${LIVEKIT_API_KEY:-}" ] && echo "LIVEKIT_API_KEY=set(shell)" || echo "LIVEKIT_API_KEY=MISSING"
[ -n "${LIVEKIT_API_SECRET:-}" ] && echo "LIVEKIT_API_SECRET=set(shell)" || echo "LIVEKIT_API_SECRET=MISSING"
```

Treat a key as **present** when either shell export or `.env` line holds a non-placeholder value. A key is **missing** when both are absent/empty/placeholder (`your-key-here`, `changeme`, `API_KEY_REQUIRED`). Shell-only credentials count even when `.env` is absent.

## NVIDIA_API_KEY (REQ)
Required for: cloud NVCF (LLM/ASR/TTS), local NIM image pull (`nvcr.io` login), catalog discovery (`GET /v1/models`).

When missing — one message STOP WAIT:
```
NVIDIA_API_KEY is not set in .env (or your shell). Cloud deployment and local NIM pulls both need it.

How to obtain:
1. Sign in at https://build.nvidia.com/
2. Open API keys: https://build.nvidia.com/settings/api-keys
   (or any model page → Get API Key / View Code → Generate API Key)
3. Copy the key (starts with nvapi-; shown once)
4. Add to .env: NVIDIA_API_KEY=nvapi-...

Docs: https://docs.api.nvidia.com/nim/docs/api-quickstart

Reply when .env is updated (or export NVIDIA_API_KEY in your shell). No scaffold until this is set.
```

Verify after user reply (do not `source .env` — parse without executing shell):
```bash
_is_usable() { case "$1" in ''|your-key-here|changeme|API_KEY_REQUIRED) return 1 ;; *) return 0 ;; esac; }
FILE_KEY=$(grep -E '^NVIDIA_API_KEY=' "$ENV_FILE" 2>/dev/null | cut -d= -f2- | tr -d '"'"' '"'"')
if _is_usable "${NVIDIA_API_KEY:-}" || _is_usable "$FILE_KEY"; then echo OK; else echo STILL_MISSING; fi
```

Apply the same `_is_usable` rule to shell **or** file values for `HF_TOKEN`, `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`.

## HF_TOKEN (REQ on all local Omni deployments)
Trigger: `PIPELINE_MODE=omni` with local deployment (workstation/DGX/Jetson) using `docker-compose.nim.omni.yml` or any local vLLM Omni sidecar that pulls HF models.

When missing — STOP WAIT:
```
HF_TOKEN is required for local Omni vLLM deployments (workstation, DGX Spark, or Jetson Thor).

How to obtain:
1. Sign in at https://huggingface.co/
2. Settings → Access Tokens → New token (read access is enough for public models)
3. Add to .env: HF_TOKEN=hf_...

Reply when .env is updated. No scaffold until this is set.
```

## LiveKit Cloud keys (REQ for LiveKit)
Default path: **LiveKit Cloud** (online server), not self-hosted. REQ: `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` in `.env`.

When missing — STOP WAIT; use **LiveKit Docs MCP first** (`user-livekit-docs`) to fetch current setup steps, then guide:

```
LiveKit agent needs LiveKit Cloud credentials in .env:

LIVEKIT_URL=wss://<project-subdomain>.livekit.cloud
LIVEKIT_API_KEY=API...
LIVEKIT_API_SECRET=...

Recommended (CLI):
1. Install: brew install livekit-cli  (or https://docs.livekit.io/reference/developer-tools/livekit-cli/)
2. Sign up: https://cloud.livekit.io/
3. Link project: lk cloud auth  (browser flow → selects project → writes keys)
4. Copy from ~/.livekit/cli.yaml or project Settings → Keys into .env

Manual (dashboard):
1. https://cloud.livekit.io/ → create/select project
2. Project Settings → API Keys → Create Key
3. Copy URL, API Key, Secret into .env

Docs: https://docs.livekit.io/agents/start/voice-ai/ (LiveKit Cloud section)
MCP: docs_search "LiveKit Cloud API keys" if MCP available

Reply when .env is updated. No scaffold until all three are set.
```

Self-hosted LiveKit is **override only** — user must explicitly request; same three vars with `http://localhost:7880` or custom URL.

## Gate integration
| path | keys required before scaffold |
| --- | --- |
| Pipecat cloud | NVIDIA_API_KEY |
| Pipecat local GPU (cascaded) | NVIDIA_API_KEY |
| Pipecat local GPU (omni) | NVIDIA_API_KEY + HF_TOKEN |
| Jetson Thor local | NVIDIA_API_KEY + HF_TOKEN (omni or cascaded vLLM) |
| LiveKit cloud (default) | NVIDIA_API_KEY + LIVEKIT_* |
| LiveKit local NIM | NVIDIA_API_KEY + LIVEKIT_* |

FORBID model ids / function_id in `.env` — secrets only (see run.md).

## Anti-patterns
scaffold then ask for keys | assume keys from .env.example placeholders | skip NVIDIA_API_KEY on cloud | skip LIVEKIT_* for LiveKit | proceed after guide without user confirm keys are set
