# LiveKit | cascaded only

Defaults: agent.py cascaded STT→LLM→TTS | models constants | DEPLOYMENT_PLATFORM cloud|workstation|jetson profiles dgx_spark/jetson_thor | **LiveKit server: LiveKit Cloud (online)** — not self-hosted unless user overrides | connect **console only** not Playground/custom UI

Run: `uv run python agent.py console` terminal mic.

LiveKit+omni→redirect Pipecat omni pipeline/omni.md or cascaded LiveKit.
LiveKit+WebRTC→console redirect browser WebRTC=Pipecat. FORBID bot.py refs. State agent.py console not :7860/client.

## Doc source order (REQ)

1. **MCP `user-livekit-docs` FIRST** when the host agent exposes that MCP server. Use `docs_search`, `get_pages`, `code_search`, `get_python_agent_example` — do not assume API from memory. Query: voice-ai quickstart, console, STT/LLM/TTS plugins, NVIDIA models, deployment, LiveKit Cloud API keys, AgentSession.
2. **No MCP:** fetch docs.livekit.io/llms.txt then pages per topic below.
3. Examples last resort: github livekit-examples fetch one file.

Index: docs.livekit.io/llms.txt | MCP endpoint: docs.livekit.io/mcp
Topics: agents,voice-ai quickstart,console,models/plugins STT/LLM/TTS,audio,audio customization,prompting,deployment,LiveKit Cloud,lk cloud auth

FORBID agent.py from memory | iterate→iterate.md fetch first

## LiveKit Cloud (default server path)

Default: agent connects to **LiveKit Cloud** (`wss://<project>.livekit.cloud`), not local `localhost:7880`.

**Before scaffold** — `clarifying/env-secrets.md` §LiveKit Cloud keys. If `LIVEKIT_URL`, `LIVEKIT_API_KEY`, or `LIVEKIT_API_SECRET` missing → guide + STOP WAIT.

Guide user (use MCP `docs_search` "LiveKit Cloud API keys" for current steps):

1. Sign up: https://cloud.livekit.io/
1. Create or select a project
1. Obtain credentials — **preferred:** `lk cloud auth` (install CLI → browser link → imports keys)
   1. Install: `brew install livekit-cli` or https://docs.livekit.io/reference/developer-tools/livekit-cli/
   1. Auth: `lk cloud auth`
   1. Docs: https://docs.livekit.io/reference/developer-tools/livekit-cli/projects/
1. **Manual:** Project Settings → API Keys → Create Key → copy URL, API Key, Secret
1. Add to `.env` (secrets only — not model ids):

```text
LIVEKIT_URL=wss://<your-project-subdomain>.livekit.cloud
LIVEKIT_API_KEY=APIxxxxxxxx
LIVEKIT_API_SECRET=xxxxxxxx
```

1. Quickstart reference: https://docs.livekit.io/agents/start/voice-ai/

Self-hosted LiveKit (`http://localhost:7880`) is **override only** — user must explicitly request; still need all three `LIVEKIT_*` vars.

## NVIDIA models on LiveKit

Use NVIDIA cloud NIMs for STT/LLM/TTS (same as Pipecat cloud): `NVIDIA_API_KEY` in `.env` per env-secrets.md. Model ids in `agent.py` constants, not `.env`.

Local LLM compose→nim-llm-profiles after model-selection.

## Layout

agent.py, pyproject.toml, .env.example, README(console). Flow: LIVEKIT_*+NVIDIA_API_KEY→console→STT→LLM→TTS AgentSession. No mix frameworks one file.

## Pre-scaffold checklist (LiveKit)

```text
0 hardware probe + env-secrets (NVIDIA_API_KEY + LIVEKIT_*)
1 MCP livekit-docs: voice-ai + console + plugin STT/LLM/TTS for NVIDIA
2 gate: framework LiveKit, deployment (probe-driven), models
3 scaffold agent.py with NVIDIA model constants
4 .env.example lists LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, NVIDIA_API_KEY only
```
