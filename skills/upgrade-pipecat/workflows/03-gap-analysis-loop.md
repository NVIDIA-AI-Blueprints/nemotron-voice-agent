# Phase 3 — Gap Analysis Loop

Validate each example's pipeline against the new API; loop until zero breaking gaps.

```text
spawn agents (1/example + cross-cut) → collect/triage → gaps>0 ? fix & loop : done
```

## Step 1 — Spawn validation agents (parallel, pass N)

Each agent: resolve uncertainty via the `pipecat-docs` MCP (`search_daily_knowledge_sources`) before reporting,
and cite the `source_url`.

**Per-example** (`generic`, `multilingual`, `omni_assistant`, `omni_assistant_subagents`, `frontend_backend_agent`):

> Validate the **{example}** pipeline against the new Pipecat API.
> A. Trace `pipeline.py`: services → VAD/turn → context aggregator → custom processors → `Pipeline` →
> `PipelineWorker(PipelineParams)` → transport/serializer (server.py) → runner; plus prewarm, audio recorder,
> RTVI messages, observers.
> B. Per symbol, check: import path valid; class/constructor/`*Settings` names+types; frame names+payload
> fields in `process_frame`/`push_frame`; `FrameProcessor` lifecycle + direction; turn/VAD/smart-turn params;
> `Pipeline`/`PipelineWorker`/`PipelineParams` args; transport/serializer/runner arg types.
> C. Frame & RTVI consumption chains: producer/consumer agree on new shape; RTVI matches `client/`.
> D. Example-specific: `omni_assistant` (smart-turn, user-mute, multimodal); `omni_assistant_subagents` (its
> companion-package symbols resolve to their new home per the change inventory — companion API or in-core
> `pipecat.*` if folded in); `frontend_backend_agent` (planner, TTS filter, tool handlers, tools schema);
> `multilingual` (processor + ASR/TTS settings); dead code.
> E. Report `✅ PASS: {what}` or `❌ GAP: {what} — repo has X, new uses Y — File:Symbol`.

**Cross-cutting** (one agent): shared plumbing (`src/examples/shared/`), `src/server.py` wiring,
`Nvidia{LLM,STT,TTS}Service`+`*Settings`, `turns.*`+`audio.turn.smart_turn.*`, all `frames.frames` imports,
extras + every dependency change (bump/rename/removed-if-folded-in) in `pyproject.toml`/`uv.lock`, dead imports
(`ruff` F401/F811). Report PASS/GAP.

**Client** (one agent): every `@pipecat-ai/*` import in `client/src/**` resolves; RTVI usage (events, messages,
providers/hooks, transport setup) matches the renamed client APIs from the notes; `client/package.json` versions
are RTVI-compatible with the target server. Report PASS/GAP.

## Step 2 — Triage

Deduplicate; classify BREAKING (import/runtime error, wrong behavior) vs NON-BREAKING (advisory); count breaking.

## Step 3 — Fix or loop

- Breaking > 0: fix each (read → change → trace propagation across callers/consumers/examples), log, re-run
  Step 1 for affected examples + cross-cutting.
- Breaking == 0: report advisories, generate deliverables.

## Step 4 — Static & test gates

```bash
# Import smoke (real paths)
uv run python -c "import src.server"
uv run python -c "from src.examples.generic import pipeline; from src.examples.multilingual import pipeline"
uv run python -c "from src.examples.omni_assistant import pipeline; from src.examples.frontend_backend_agent import pipeline"
uv run python -c "from src.examples.omni_assistant_subagents import pipeline"
# Lint (CI parity)
uv run ruff format --check . && uv run ruff check .
# Tests
set +e; uv run pytest tests/unit -x -q > /tmp/pytest.log 2>&1; status=$?; head -100 /tmp/pytest.log; exit $status
# Client (CI parity) — tsc catches RTVI API renames, eslint catches the rest
cd client && npm ci && npm run lint && npm run build
```

`ImportError`/`AttributeError` or a client `tsc`/`eslint` error = missed gap → feed back into the loop. Test
failure → source bug (fix source, re-run gap analysis) vs plumbing (fix test). Optional skill self-check if
`nv-base` present:
`nv-base validate skills --no-llm --no-dedup --checks schema,pii,code-integrity,unicode,quality,lint`.

## Pass strategy

Pass 1 = all examples + cross-cutting + client; pass 2 = surfaces with gaps; pass 3+ = remaining. Then import
smoke → lint → tests → client build. Max 5 code passes + 3 test cycles, else escalate.

## Deliverables

1. **Gap report**: passes, gaps found/fixed/advisories, per-example + cross-cutting + client status, advisory list.
2. **Change log** `docs/pipecat-upgrade-changelog.md`: per change — file:symbol, old→new, why (CHANGELOG/docs
   ref), examples affected. Plus: `pipecat-ai` old→new, every server + client dependency decision
   (bumped/renamed/removed + migrated-to-core), extras changes, files changed, removed/renamed APIs,
   server↔client RTVI contract notes,
   advisories.
