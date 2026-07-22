# Phase 2 — Plan & Implement

Turn the change matrix into code changes, bottom-up so the tree stays consistent.

## Step 1 — Implementation order

Derive from the change matrix; skip layers with no changes. Default bottom-up order:

1. Import paths / module moves
2. Service constructors & `*Settings` (`Nvidia{LLM,STT,TTS}Service`)
3. Shared plumbing (`src/examples/shared/`: `pipeline_utils.py`, `prewarm.py`, text filters, audio recorder)
4. Frames, `FrameProcessor` subclasses, aggregators / `LLMContext`
5. Turn strategies, VAD, smart-turn
6. Transports & serializers (`src/server.py`: SmallWebRTC / FastAPI websocket)
7. Per-example `pipeline.py` (`Pipeline`, pipeline-task/runner, params) — use the new names from the notes
8. Companion-package usage (e.g. the `omni_assistant_subagents` example): if its package was folded into
   `pipecat-ai` core, rewrite imports to the new in-core modules; otherwise apply that package's own changes
9. `pyproject.toml` extras + server dependency changes (bump / **remove a folded-in package** / add), then
   `uv.lock` last, so tests/lint run against the new pin
10. Client (`client/`): bump `@pipecat-ai/*` in `client/package.json` to RTVI-compatible versions, migrate
    `client/src/**` RTVI usage (renamed events/messages/providers/hooks, transport setup), then refresh the
    lockfile — do last so the client builds against the matching server contract

## Step 2 — Implement each layer

Per layer: re-read the current code → apply changes → trace propagation → verify imports.

Apply rules:

- Match the new API exactly (new source/docs = truth); keep repo style (Google docstrings, ruff line-length 120).
- Moved import → update import + all usages. Changed constructor/`Settings` → update all instantiation sites.
  Renamed/reshaped frame → update producers and consumers.
- Unclear signature / moved module / replacement for a removed API → query the `pipecat-docs` MCP
  (`search_daily_knowledge_sources`) and verify against installed new source. Don't invent APIs.

Propagation grep after each change — check for: old kwargs/`Settings` fields, `process_frame` branches on old
frame names, old import paths, changed `pipecat_subagents` bus message shapes. Fix before the next layer.

Verify (real module paths from Phase 1):

```bash
uv run python -c "import src.server"
uv run python -c "from src.examples.shared import pipeline_utils"
uv run ruff check src/ | head -40
```

## Step 2e — Parallel agents for large rewrites

If an example needs more than renames (reworked turn-taking, transport handshake), give one agent the new
source/docs + the example's `pipeline.py` and custom processors. Likely candidates: `omni_assistant`
(smart-turn + user-mute + multimodal), `omni_assistant_subagents` (subagents bus/runner), `frontend_backend_agent`
(planner + TTS filter + tools). Add a translation layer if a frame/RTVI payload shape changed rather than
dropping fields.

## Step 3 — Cross-cutting sweep

- **Import roots**: no `src/` import from an old module path.
- **Frames**: every referenced frame (`LLMRunFrame`, `LLMTextFrame`, `LLMFullResponse{Start,End}Frame`,
  `TTSUpdateSettingsFrame`, `RTVIServerMessageFrame`, …) exists; `isinstance` checks use new names.
- **Services**: `Nvidia{LLM,STT,TTS}Service` + `*Settings` constructors/fields + update-settings frames align.
- **Turns/VAD**: `turns.*`, `audio.turn.smart_turn.*`, `audio.vad.*` (`UserTurnStrategies`, `UserTurnProcessor`,
  `VADParams`, `SileroVADAnalyzer`, smart-turn v3) classes/params/wiring match.
- **Transports/serializers**: `src/server.py` `SmallWebRTCConnection`/`SmallWebRTCTransport`,
  `ProtobufFrameSerializer`, runner arg types (`RunnerArguments`, `SmallWebRTCRunnerArguments`).
- **RTVI/observers (server)**: `processors.frameworks.rtvi.*`, `observers.*` (`UserBotLatencyObserver`).
- **RTVI contract (client)**: any RTVI event/message/provider rename in the notes must be applied in
  `client/src/**` so the client matches the server wire protocol — keep the two in sync.
- **Companion-package usage** (e.g. the `omni_assistant_subagents` example): every symbol it imports resolves to
  its new home per the change inventory — either the companion package's new API, or the in-core `pipecat.*`
  modules it was folded into.
- **Extras/dead code**: update `pipecat-ai[...]` extras; remove old-version shims and `F401` imports.

## Step 4 — Dependencies

**Server** — apply the change-matrix decisions to `pyproject.toml`: set `pipecat-ai[<extras>]==<NEW_REF>`; for
each companion package **bump / rename / or remove it if folded into core**; keep `[tool.uv]
override-dependencies` (e.g. `cryptography` CVE pin) unless newly resolved. Then:

```bash
uv sync --dev
uv run python -c "import pipecat; print(pipecat.__version__)"   # + import each companion pkg still depended on
```

Confirm `uv.lock` matches intended pins, no removed package lingers, nothing pulled a conflicting `pipecat-ai`.

**Client** — bump every `@pipecat-ai/*` in `client/package.json` to the RTVI-compatible versions from the
matrix; remove any client package whose capability was dropped/renamed. Then:

```bash
cd client && npm install && npm run lint && npm run build   # tsc surfaces RTVI API renames as type errors
```

Fix client type/lint errors using the renamed RTVI APIs from the notes. Confirm the lockfile updated.

## Step 5 — Tests

Unit tests live under `tests/unit/` (`test_service_catalog.py`, `test_prompt_catalog.py`,
`test_frontend_backend_agent.py`, booking-state helpers). Update plumbing only; preserve intent.

- Scan: `grep -rn "pipecat\|<old_import_root>\|<OldClass>\|<OldFrame>" tests/unit/`.
- Change only import paths, mock targets, constructor/`Settings` kwargs, frame names, fixture shapes. Never
  change assertions, remove tests, or reduce coverage.
- Run/fix (max 3 cycles):
  `set +e; uv run pytest tests/unit -x -q > /tmp/pytest.log 2>&1; status=$?; head -80 /tmp/pytest.log; exit $status`.
  Classify each failure as source bug (fix source, feed back to gap analysis) vs test plumbing (fix test).
- Lint (CI parity): `uv run ruff format . && uv run ruff check . --fix && uv run ruff check .`.

## Deliverable

Summary: server files changed (by example/shared), client files changed, test files changed, server + client
dependency pins, change categories, failures + fixes, RTVI contract changes (server↔client), concerns for
review. Proceed to Phase 3.
