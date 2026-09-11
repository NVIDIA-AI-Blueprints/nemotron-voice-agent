# Phase 1 — Explore & Discover

Build the change inventory before touching code. Steps: **Resolve → Read notes → Diff → Scan repo → Synthesize**.

## Step 1 — Resolve inputs

```text
OLD (always inspectable):
  - pyproject.toml pinned spec: pipecat-ai[<extras>]==X.Y.Z
  - discover EVERY pipecat dependency on BOTH surfaces — don't assume which exist:
      grep -nE '^\s*"?pipecat[-_a-z0-9]*' pyproject.toml          # server: pipecat-ai + every pipecat-* subpackage
      grep -nE '"@pipecat-ai/' client/package.json                # client: every @pipecat-ai/* npm package
      then read server versions from uv.lock and client versions from client/package-lock.json (or node_modules)
  - installed source = ground truth: .venv/lib/python3.12/site-packages/ (pipecat + every pipecat_* package)
NEW (target pipecat-ai):
  - latest version + notes ALWAYS from https://github.com/pipecat-ai/pipecat/releases
  - local repo path → tag matching target; git tag → use with repo
  - PyPI version → the releases page above (+ CHANGELOG.md); pip download wheel to read module tree if needed
  - docs URL → WebFetch index + API pages
```

The only target you choose is the `pipecat-ai` version. The fate of every other pipecat subpackage (bump /
rename / remove / fold-into-core) is decided by the release notes read in Step 2 — not chosen here.

Record: `OLD_REF`, `NEW_REF`, `OLD_PKG_PATH`, `NEW_SRC`, and the full list of pipecat packages on both surfaces
(server `pipecat*` and client `@pipecat-ai/*`) with each current version.

## Step 2 — Read release notes & analyze

An upgrade crosses multiple releases (`1.2.1 → 1.5.0` = `1.3.x`/`1.4.x`/`1.5.x`). Read them **cumulatively**.

### Core `pipecat-ai` notes — sources (priority order)

1. **GitHub Releases (canonical)** — <https://github.com/pipecat-ai/pipecat/releases> — read EVERY release in
   `(OLD_REF, NEW_REF]`: `gh release view <tag> -R pipecat-ai/pipecat` or WebFetch the tag pages.
2. `CHANGELOG.md` — `sed -n '/## \[<NEW>\]/,/## \[<OLD>\]/p' <NEW_SRC>/CHANGELOG.md`.
3. Docs — <https://docs.pipecat.ai/> (search "migration", "breaking changes").
4. `pipecat-docs` MCP (`search_daily_knowledge_sources`) — clarify any note you don't fully understand.

### Every other pipecat package — read its notes too (server + client)

For each package found in Step 1:

- **Server `pipecat-*`** (e.g. `pipecat-ai-subagents`, `pipecat-ai-flows`): find its own repo/releases
  (typically `pipecat-ai/<subpackage>`) and read notes/CHANGELOG between current and the version compatible with
  target `pipecat-ai`. If it has **no separate notes / is archived / its repo says it merged into `pipecat-ai`**,
  treat it as folded-into-core — the `pipecat-ai` notes then say where its API moved. Check PyPI / MCP if unsure.
- **Client `@pipecat-ai/*`** (e.g. `client-js`, `client-react`, `*-transport`): read each package's notes from
  its npm page (`npm view <pkg> versions`) and its GitHub repo (the JS client/transports live under
  `pipecat-ai/*` repos). The `pipecat-ai` server notes call out RTVI client-API renames explicitly (e.g.
  event/message/provider renames) — these are the client changes that must land in `client/src/`. Pick client
  versions that match the RTVI protocol of the target server version.

### Extract per release (core + each subpackage)

Breaking changes, deprecations, new required args, behavioral changes (VAD/smart-turn defaults, turn-taking,
transport/serializer handshake, RTVI shapes), extras changes, and **dependency changes** — including any
subpackage being renamed, removed, or folded into `pipecat-ai` core (if so, note the new in-core modules its
functionality moved to).

Then map each note to the repo (grep `src/` for the affected symbol) into the candidate table:

| Release | Note | Affected symbol | Used in repo? | Action |
|---------|------|-----------------|---------------|--------|
| 1.4.0 | (paraphrase) | (symbol) | yes/no/unclear | migrate / advisory / n/a |

This is the first draft of the change inventory; Step 3 confirms and completes it.

## Step 3 — Diff old vs new

Catches what the notes omit (module moves, undocumented signature changes).

### 3a — Map the module layout (both versions)

```bash
find .venv/lib/python3.12/site-packages/pipecat -maxdepth 3 -name "*.py" | sort   # old
find <NEW_SRC> -maxdepth 4 -name "*.py" -path "*/pipecat/*" | sort                # new (if available)
```

Verify each subtree this repo imports (still exists / where it moved):

- `services.nvidia.{llm,stt,tts}` (`Nvidia*Service`, `Nvidia*Settings`)
- `frames.frames` (`LLMRunFrame`, `LLMTextFrame`, `LLMFullResponse*Frame`, `TTSUpdateSettingsFrame`)
- `processors.frame_processor` (`FrameProcessor`, `FrameDirection`)
- `processors.aggregators.{llm_context,llm_response_universal}` (`LLMContext`)
- `turns.*` (`user_mute`, `user_stop`, `user_start`, `user_turn_processor`, `user_turn_strategies`)
- `audio.turn.smart_turn.*` (`SmartTurnParams`, `LocalSmartTurnAnalyzerV3`), `audio.vad.*`
- `transports.{base_transport,smallwebrtc.*,websocket.fastapi}`, `serializers.*`
- `runner.types` (`RunnerArguments`, `SmallWebRTCRunnerArguments`)
- `observers.*`, `processors.frameworks.rtvi.*`, `processors.audio.*`
- `adapters.schemas.tools_schema`, `utils.text.*`, `utils.context.*`
- whatever the repo's import scan (Step 4) shows it uses from any companion package — map where each symbol
  lives now (still in the companion package, or moved into `pipecat-ai` core)

### 3b — Parallel diff agents

Give each the Step 2 table.

- **Core/source diff**: `git -C <NEW_SRC> diff <OLD_REF>..<NEW_REF> -- src/pipecat/ | head -4000` (or compare
  old `.venv` tree vs new wheel). Extract: moved/renamed modules/classes/methods, changed constructor/`Settings`
  params, renamed frames, changed `FrameProcessor` lifecycle, changed pipeline/task/transport params, extras.
- **Companion-package diff** (for each companion package the repo imports): per the Step 2 notes, either diff
  old vs new of that package, OR — if it was folded into `pipecat-ai` core — locate the symbols the repo uses
  in their new in-core home and record old→new module paths so Phase 2 can migrate and drop the dependency.

Reconcile: resolve every "unclear" note; any diff finding not in the table is an undocumented change — add it.

### 3c — Change inventory

| Category | Old | New |
|----------|-----|-----|
| Module / import root | | |
| Service / Settings | | |
| Frame names | | |
| FrameProcessor API | | |
| Aggregator / context | | |
| Turn strategy | | |
| Transport / serializer | | |
| Runner args | | |
| RTVI / observer | | |
| Pipeline / PipelineWorker | | |
| Extras `[nvidia,...]` | | |
| Server deps (bump/rename/remove/fold-into-core) | | |
| Companion-package symbols → new home | | |
| Client `@pipecat-ai/*` versions | | |
| Client RTVI API (events/messages/providers) | | |
| Removed | | (gone) |
| New required args | (n/a) | |

Source of truth for all later phases.

## Step 4 — Scan the repo

### 4a — Structure (one Explore agent)

Server: `src/examples/shared/` plumbing (`pipeline_utils.py`, `prewarm.py`, `audio_recorder.py`, text filters);
`src/server.py` (runner/transport/serializer); each `src/examples/*/pipeline.py`; `pyproject.toml`/`uv.lock`
pins+extras; every `from pipecat*` import; custom subclasses of
`FrameProcessor`/`BaseUserMuteStrategy`/`BaseTextFilter`/`PipelineWorker`/`BaseWorker`.
Always inspect custom service subclasses against their new parent implementations, especially
`src/examples/shared/nvidia_word_tts.py` (`NvidiaWordTTSService`) and
`src/examples/omni_assistant/nvidia_omni_multimodal_service.py`.
Client: `client/package.json` `@pipecat-ai/*` deps; every `@pipecat-ai/*` import in `client/src/**` and where
RTVI is used (client init/transport setup, RTVI event/message handlers, providers/hooks). Note files for the
change matrix.

### 4b — Pipeline build trace (one agent per example)

For `generic`, `multilingual`, `omni_assistant`, `omni_assistant_subagents`, `frontend_backend_agent`, trace:

```text
pipeline.py → service construction (Nvidia{LLM,STT,TTS}Service+Settings) → VAD/smart-turn/turn-strategy
  → context aggregator (LLMContext, llm_response_universal) → custom FrameProcessors (frames+direction)
  → Pipeline([...]) + PipelineWorker(PipelineParams) → transport/serializer (server.py) → runner
  → RTVI server messages, observers, audio buffer
```

`omni_assistant_subagents` also: `WorkerRunner`, `BaseWorker`, `WorkerBus`, `BusBridgeProcessor`, `bus.messages`.
Return every Pipecat symbol, import path, and arguments.

### 4c — Map changes to code

Cross-reference the inventory (Step 3) with the per-example maps (4b): each inventory change → affected call
sites; each repo symbol → confirm in inventory or verify unchanged against the new tree. Produces the **change
matrix** (what changes, where, to what, which examples).

## Step 5 — Synthesize & present

| Change | Old (repo) | New (Pipecat) | File:Symbol | Surface (example / shared / server / client) | Priority |
|--------|------------|---------------|-------------|----------------------------------------------|----------|

Also list: obsolete code to remove; extras changes; per-dependency decision (bump / rename / remove +
migrate-to-core) for BOTH server and client packages, with the release-note entry that justifies it; behavioral
changes; release-note coverage
(cite the entry behind each change; confirm every breaking change is in the matrix or marked "not used").
Present for review before Phase 2.
