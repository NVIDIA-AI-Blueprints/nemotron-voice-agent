# Frontend/Backend Agent: ToolSpec Design Plan

| Field | Value |
| --- | --- |
| Goal | Cut the cost of changing `examples/frontend_backend_agent`: one definition per tool, and a new read-only flavour that reuses existing tools becomes a registry entry rather than a Python package |
| Status | **Design only — nothing in this document is implemented.** `grep -rn "ToolSpec\|ParamSpec\|render_tool_block" src/ tests/` returns no matches, and `src/examples/frontend_backend_agent/src/tools.py` does not exist |
| Branch | `dev/nikkulkarni/domain-configurable-frontend-backend-agent` |
| Scope | `src/examples/frontend_backend_agent/**`, plus registry/server wiring already added by `77b23456` (the pre-rebase SHA `e260967b` cited in the first draft is no longer on this branch) |
| Supersedes | `docs/generic-frontend-backend-agent-implementation-plan.md` (see §8) |
| Non-goals | ReAct replanning, deterministic pre-router, write-capable generic tools, NVCF/Astra rollout, changes to `src/examples/generic/` (the standalone Generic Assistant) |

---

## 0. Architecture diagrams

**Engineering detail** — every file, every duplication site:

| Diagram | File |
| --- | --- |
| Current architecture — the shared pipeline plus the two hand-written domain packages, with each duplicated per-tool definition site tagged | [`images/frontend-backend-agent-current.svg`](images/frontend-backend-agent-current.svg) |
| ToolSpec architecture — the same pipeline with the Generic Assistant flavour wired through a single `ToolSpec` registry | [`images/frontend-backend-agent-toolspec.svg`](images/frontend-backend-agent-toolspec.svg) |

**Stakeholder slides** — 16:9, one idea per slide, no file names except the seven that make the point:

| Slide | Files |
| --- | --- |
| How it works today, and what one new tool costs | [`images/frontend-backend-agent-today-slide.svg`](images/frontend-backend-agent-today-slide.svg) · [`.png`](images/frontend-backend-agent-today-slide.png) |
| What changes: one definition per tool | [`images/frontend-backend-agent-proposed-slide.svg`](images/frontend-backend-agent-proposed-slide.svg) · [`.png`](images/frontend-backend-agent-proposed-slide.png) |

Read them side by side: bands 2 and 3 (the Pipecat voice path and the Talker
tool boundary) are pixel-identical between the two, which is the point — the
change is confined to how a domain declares its tools.

Both are generated SVGs; open them in a browser or any SVG viewer to zoom.

---

## 1. The problem, stated as a cost

**What already works, so the plan is not mistaken for it.** `77b23456` already made
the flavour selectable from configuration: `examples_registry.yaml` carries two
entries (`frontend-backend-agent`, `generic-frontend-backend-agent`) that name the
*same* `bot: examples.frontend_backend_agent.pipeline:bot` and differ only by
`domain_profile` plus slot defaults. Starting the example as airline or as generic is
a config row today. This plan does not add that; it attacks what is still expensive.

Adding one tool today means editing the same tool name in seven places, in two
different shapes depending on which domain you are in.

**`frontend_backend_agent/generic/` (read-only toolset domain)**

| Site | Holds |
| --- | --- |
| `generic/tools.py` → `INTERNAL_TOOL_NAMES` | enablement allowlist |
| `generic/services.py` → `TOOL_SERVICES` | the callable |
| `generic/dispatcher.py` → `_TOOL_TIMEOUTS` | deadline |
| `generic/dispatcher.py` → `_PARAMS` | required/optional field names |
| `generic/dispatcher.py` → `_validate_values` | per-tool `elif` of value checks |
| `generic/result_formatters.py` → `labels`, `format_tool_result` | per-tool `elif` of speech text |
| `prompts.yaml` → `generic_thinker` | contracts, routing precedence, examples |

**`frontend_backend_agent/airline/` (stateful domain)**

| Site | Holds |
| --- | --- |
| `airline/thinker.py` → `_dispatch_tool_call` | `if tool_name == ...` chain over 4 tools |
| `airline/thinker.py` → `_READ_ONLY_TOOLS` | parallel-vs-serialized decision |
| `prompts.yaml` → `thinker` | "Available internal tools:" contracts |

Neither list is derived from the other, and nothing checks that they agree.

### 1.1 Failure modes this shape already produces

1. **Capability lie on partial enablement.** `generic/result_formatters.py`
   `unsupported_request()` hardcodes all five capabilities. A session running
   `tools_available: [calculate_bmi]` still tells the user it can search the live
   web and check stock prices. The enable path is per-session; the sentence is not.
2. **Silent degradation on partial registration.** Register in `TOOL_SERVICES`
   but not `_PARAMS` → `KeyError` in `validate_plan` → swallowed by the broad
   `except Exception` in `generic/backend.py::_run_call` → user hears the generic
   `planner_failure()` retry line. Omit the `format_tool_result` branch → falls
   through to `"I completed the check."` and the real result is never spoken.
   Neither surfaces as an error in logs or tests.
3. **Prompt/code drift.** `generic_thinker` enumerates five tool contracts and a
   seven-step routing precedence as prose. Disabling a tool at runtime does not
   remove it from the model's world; the model keeps planning calls that the
   dispatcher then rejects as `disabled tool`.

### 1.2 Duplicated lifecycle

`airline/thinker.py` (301 lines) and `generic/backend.py` (134 lines) both
implement latest-request-wins cancel, supersede-and-await-previous, plan
dispatch, parallel gather, and combine. They have since diverged:

| Behavior | `airline/thinker.py` | `generic/backend.py` |
| --- | --- | --- |
| Overall deadline | none | `GENERIC_BACKEND_TIMEOUT_SECONDS`, default 40s |
| Planner deadline | none | `GENERIC_PLANNER_TIMEOUT_SECONDS`, default 15s |
| Stale-generation guard | none | `active_call_id` re-check after await |
| Param validation | none — merged slots passed through | `_PARAMS` + `_validate_values` |
| `response_hint` speech | model-authored `response_text` used verbatim | closed vocabulary, Python-generated |
| Synthetic tool delay | yes (`_next_tool_delay_seconds`) | no |
| `slots` from `call_backend` | merged into every tool call | discarded (`del slots`) |

The divergence is the point: two copies drifted, and the airline copy is the one
missing the deadlines and the stale guard.

### 1.3 Sites the first draft of this plan missed

The seven-site audit counts *per-tool* duplication. Three further sites keep the
airline privileged inside supposedly domain-neutral code, and a fourth is a live
trust gap. All were verified against the tree at `77b23456`.

| Site | What is wrong | Fix |
| --- | --- | --- |
| `src/tts_filter.py:12` | The shared `src/` layer imports `examples.frontend_backend_agent.airline.airports`. Its only consumer is `airline/domain.py:17`. | §Step 0 — move the module into `airline/`. |
| `pipeline.py:29` | Module-scope `from ...airline import domain as airline_domain`, used only by the two legacy facades at lines 379-386. Every generic session pays an airline import. | §Step 0 — drop it. |
| `pipeline.py:72` | `_build_context_messages` falls back to `resolve_domain_spec("airline").runtime_context()`. Its only caller (line 273) always passes `domain.runtime_context()`, so the fallback is **unreachable**. | §Step 0 — require the argument, delete the fallback. |
| `utils.py:65` + `utils.py:488` | `tools_available` sits in `_SLOT_AGNOSTIC_KEYS` and the `filter_session_config` allowlist, and unlike `domain_profile` it is **not** overwritten from the registry in `server.py`. A client can enable a tool the selected prompt never declared — bounded to the five built-ins, but `web_search` is a metered Perplexity call. | §3.7 |

### 1.4 The filler-text asymmetry — a second instance of §4.3

§4.3 identifies one place where the airline domain lets the model author user-facing
speech and the toolset domain does not (`response_hint`). **The same split exists in
filler text, and §3.6 as first drafted would have deleted the safe side of it.**

- `airline/tools.py:29-37` declares a `filler_text` parameter on `call_backend`; the
  generic schema declares none.
- `src/tool_handlers.py:69-72`:

  ```python
  if filler_selector is not None:
      filler_text = filler_selector(query)                 # generic: Python-authored
  else:
      filler_text = str(arguments.get("filler_text", ""))   # airline: MODEL-authored
  ```

- `airline/domain.py:66-77` sets no `filler_selector`, so airline speaks
  model-authored filler; `generic/domain.py:64` sets `select_filler`, so generic
  does not.

This is a trust boundary, not a latency heuristic. See §3.6.

---

## 2. Design

One idea: **collapse the per-tool tables into one `ToolSpec`, and make a flavor a
registry entry rather than a Python package.**

### 2.1 `ToolSpec` — new file `src/examples/frontend_backend_agent/src/tools.py`

```python
@dataclass(frozen=True, slots=True)
class ParamSpec:
    kind: type                              # str | int | float | bool
    required: bool = True
    label: str = ""                         # "the city or location" — used by missing_parameters()
    bounds: tuple[float, float] | None = None
    choices: frozenset[str] | None = None
    max_len: int = 200
    default: Any = None

@dataclass(frozen=True, slots=True)
class ToolContext:
    """Per-call dependencies. Pure tools ignore it."""
    state: Any = None                       # domain session state (airline)
    backend: Any = None                     # domain service client (airline BookingBackend)

@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    contract: str                           # one prose line, rendered into the Thinker prompt
    params: dict[str, ParamSpec]
    run: Callable[[Mapping[str, Any], ToolContext], Awaitable[dict]]
    speak: Callable[[dict, dict], str] | None = None   # None → tool returns its own protocol payload
    mutates: bool = False                   # replaces _READ_ONLY_TOOLS
    timeout_s: float = 12.0
    capability: str = ""                    # phrase for the generated capability sentence
```

`ToolSpec` subsumes `INTERNAL_TOOL_NAMES`, `TOOL_SERVICES`, `_TOOL_TIMEOUTS`,
`_PARAMS`, the `_validate_values` chain, the `labels` dict, the
`format_tool_result` chain, and `_READ_ONLY_TOOLS`.

Two fields carry the airline/generic difference:

- **`run(arguments, ctx)`** — generic tools ignore `ctx`; airline tools read
  `ctx.state` and `ctx.backend`. Adapters are one-line lambdas (§3.2, §4.2).
- **`mutates`** — drives serialize-vs-parallel. Generic tools are all
  `mutates=False`; `flight_search` and `booking` are `mutates=True`.

`speak=None` means the tool already returns a finished protocol payload
(`airline` tools do); a callable means the dispatcher formats speech from the
validated arguments and the returned data (`generic` tools).

### 2.2 Generated Thinker prompt block

The tool contracts move out of `prompts.yaml` prose and are rendered from the
**enabled** specs at session start, appended the same way `runtime_context()`
already is in `pipeline.py::_build_context_messages`:

```python
def render_tool_block(specs: Sequence[ToolSpec]) -> str:
    lines = ["\n\nAvailable internal tools (only these names are permitted):"]
    for spec in specs:
        req = [n for n, p in spec.params.items() if p.required]
        opt = [n for n, p in spec.params.items() if not p.required]
        lines.append(f"- {spec.name}: {spec.contract}")
        lines.append(f"  Required params: {', '.join(req) or 'none'}."
                     f" Optional params: {', '.join(opt) or 'none'}.")
    return "\n".join(lines)
```

This is the change that pays for the rest. It removes the drift class entirely
and makes disable actually mean disabled. The prompt keeps its trust-boundary
rules, output schema, routing precedence, and examples — only the per-tool
contract enumeration is generated.

`unsupported_request()` likewise builds its sentence from the enabled specs'
`capability` fields, which fixes failure mode §1.1(1) for free.

### 2.3 Flavor as configuration

Keep `DomainSpec` and the `_DOMAIN_FACTORIES` allowlist in `src/domain.py` — a
domain still owns state, backend construction, TTS transforms, and its tool
registry. What moves to the registry entry is the part you want to change
without code:

```yaml
generic-frontend-backend-agent:
  label: Generic Frontend Backend Agent
  domain_profile: generic                # unchanged — see §2.5
  thinker_prompt: generic_thinker        # NEW — was frozen in DomainSpec
  tools: [get_weather, get_stock_price, web_search, calculate_bmi, generate_random_number]
  defaults:
    prompt: [generic_talker]             # Talker prompt — already selectable
```

`DomainSpec.thinker_prompt_key` becomes a default that the registry entry
overrides. `tools:` is server-owned config at the same trust level as
`domain_profile` (bound in `server.py::_sanitize_session_config`), resolved
against the domain's `ToolSpec` registry — a name not in the registry is
dropped with a warning, exactly as `resolve_enabled_tools` does today.

**Result:** a new read-only flavour *that reuses tools which already exist* is a
registry entry plus two prompts, with no new Python. A new **capability** still needs
Python. Those are different claims and this plan must not blur them: `tools:` selects
from what the domain already implements, it does not implement anything.

### 2.4 Where configuration stops — a deliberate boundary

`ToolSpec.run` is a Python callable and stays one.

| Expressed in configuration | Expressed in Python |
| --- | --- |
| which tools a flavour exposes (`tools:`) | what a tool *does* (`run=`) |
| which Talker / Thinker prompt it uses | how a result becomes speech (`speak=`) |
| model, ASR, TTS and timeouts per slot | the domain's session state and service clients |

Pushing `run=` bodies into YAML means inventing a config language for HTTP verbs,
auth, retries and JSON reshaping — unreviewable, untestable, and a new injection
surface for anything that can write config. **The boundary is the design, not a
shortfall.** State it plainly when presenting: "just config" invites the question,
and the honest answer is stronger than the slogan.

### 2.5 Naming: keep `generic`, do not rename to `toolset`

The first draft wrote `domain_profile: toolset` in §2.3 while §3 kept referring to the
`generic/` package. Renaming the profile changes a key in the `_DOMAIN_FACTORIES`
allowlist (`src/domain.py:63`) and invalidates any stored session config or NVCF body
that names `generic`, for no functional gain.

**Decision: the profile key stays `generic`.** "Toolset domain" is a useful phrase for
the *shape* — read-only, stateless, tools declared as data — and this document uses it
that way, but it is not an identifier. Every `domain_profile:` value here reads
`generic`.

---

## Step 0 — de-privilege the airline (land this first)

Independent of `ToolSpec`, small, and the most on-goal work in the plan: after this
commit the shared pipeline no longer names a domain anywhere except its allowlist.

1. `git mv src/examples/frontend_backend_agent/src/tts_filter.py \
        src/examples/frontend_backend_agent/airline/tts_filter.py` — update the import
   in `airline/domain.py:17` and in `tests/unit/test_frontend_backend_agent.py:35`.
2. Delete `pipeline.py:29` and the two facades at `pipeline.py:379-386`. Any caller
   that still needs them can use `airline/domain.py`, which already exports
   `default_booking_backend_url()` and `booking_backend_url()`.
3. Make `runtime_context` a required argument of `_build_context_messages` and delete
   the `resolve_domain_spec("airline")` fallback (`pipeline.py:72`). The single caller
   at line 273 already passes it.

**Gate:** grepping `pipeline.py` and `src/` for `airline` returns only the
`_DOMAIN_FACTORIES` entry in `src/domain.py` and the backward-compatible default
profile string at `pipeline.py:103`. Full unit suite green. No behaviour change.

---

## 3. Step 1 — toolset domain

Additive. Airline untouched and still on its own code path.

### 3.1 `src/tools.py` (new, ~90 lines)

`ParamSpec`, `ToolContext`, `ToolSpec`, `render_tool_block`, and one generic
validator replacing the per-tool `elif` chain:

```python
def validate_arguments(spec: ToolSpec, arguments: Mapping[str, Any]) -> list[str]:
    """Return missing required names; raise ValueError on an invalid value."""
    unexpected = set(arguments) - set(spec.params)
    if unexpected:
        raise ValueError(f"unexpected params: {sorted(unexpected)}")
    missing = [n for n, p in spec.params.items()
               if p.required and arguments.get(n) in (None, "")]
    if missing:
        return missing
    for name, p in spec.params.items():
        if name not in arguments:
            continue
        value = arguments[name]
        if p.kind is str:
            if not isinstance(value, str) or not 0 < len(value.strip()) <= p.max_len:
                raise ValueError(f"invalid {name}")
            if p.choices and value.lower() not in p.choices:
                raise ValueError(f"invalid {name}")
        else:
            if isinstance(value, bool) or not isinstance(value, p.kind):
                raise ValueError(f"invalid {name}")
            if p.bounds and not p.bounds[0] <= float(value) <= p.bounds[1]:
                raise ValueError(f"invalid {name}")
    return []
```

### 3.2 `generic/tools.py` → the registry

`CALL_BACKEND_TOOL`, `CANCEL_BACKEND_TOOL`, `TOOLS_SCHEMA`, and
`resolve_enabled_tools` stay as-is. `INTERNAL_TOOL_NAMES` is replaced by:

```python
TOOLS: dict[str, ToolSpec] = {
    spec.name: spec for spec in (
        ToolSpec(
            name="get_weather",
            contract="live CURRENT conditions; never forecast, history, or weather news",
            capability="check current weather",
            params={
                "city":  ParamSpec(str, label="the city or location"),
                "units": ParamSpec(str, required=False, default="celsius",
                                   choices=frozenset({"celsius", "fahrenheit"})),
            },
            run=lambda args, ctx: services.get_weather(args),
            speak=speech.weather,
            timeout_s=12.0,
        ),
        ...
    )
}
```

`resolve_enabled_tools` filters against `TOOLS` instead of `INTERNAL_TOOL_NAME_SET`.

### 3.3 `generic/services.py` — unchanged

The five async functions are already clean, credential-scoped, and mock-free.
Only the trailing `TOOL_SERVICES` dict is deleted (its content now lives in the
`run=` fields).

### 3.4 `generic/dispatcher.py` — **−~70 lines**

- Delete `_TOOL_TIMEOUTS`, `_PARAMS`, and the `_validate_values` `elif` chain.
- `validate_plan` looks up `TOOLS[name]` and calls `validate_arguments`.
- `_execute` reads `spec.timeout_s` and calls `spec.run(args, ctx)`.
- Add the `mutates` partition now (unused by generic, where all specs are
  `mutates=False`) so Step 2 does not touch this function again.
- **Unchanged:** `MAX_PARALLEL_TOOL_CALLS`, the atomic preflight-before-first-side-effect
  ordering, `PlanValidationError` semantics, timeout/failure envelopes, and
  `_response_hint`'s closed vocabulary.

### 3.5 `generic/result_formatters.py` — **−~45 lines**

- `format_tool_result` keeps the `unavailable` / `not_found` / non-success
  branches verbatim; the success branch becomes `spec.speak(arguments, data)`.
- The five success branches move to a `generic/speech.py` module as small named
  functions (`weather`, `stock`, `search`, `bmi`, `random_number`).
- `missing_parameters` reads `ParamSpec.label` instead of its local `labels` dict.
- `unsupported_request(specs)` takes the enabled specs and builds its sentence.
- **Unchanged:** `_speech_text` sanitization, `combine_tool_results`, and all
  `response_hint` builders.

### 3.6 `generic/domain.py`

- `_build_backend` resolves specs from `TOOLS` and passes them to the planner
  (for `render_tool_block`) and the dispatcher.
- `thinker_prompt_key` accepts the registry override.
- `select_filler` becomes a `DomainSpec` field mirroring `response_hint_policy` (§4.3):

  ```python
  filler_policy: Literal["code_authored", "planner_authored"] = "code_authored"
  ```

  `code_authored` keeps a Python-selected phrase (today's generic behaviour);
  `planner_authored` reads the model's `filler_text` argument (today's airline
  behaviour). **Do not simply delete `select_filler`** — per §1.4 that silently moves
  the toolset domain onto the model-authored path, which is a trust regression, not a
  cleanup. Its *keyword heuristic* can collapse to one default string
  (`"Let me check that."`); the policy field is what must survive.

### 3.7 `prompts.yaml` / `examples_registry.yaml`

- `generic_thinker`: delete the "Tool contracts" block and the tool-specific
  rows of "Routing precedence". Keep the trust boundary, runtime inputs,
  enabled-tool rule, missing-information rule, multi-tool rule, output forms,
  and examples.
- Registry: add `thinker_prompt` and `tools` to the
  `generic-frontend-backend-agent` entry; extend `examples_registry.py`
  `ExampleEntry` and `_load_examples` the same way `domain_profile` was added.
- **Close the `tools_available` gap in the same commit** (§1.3). Moving the tool set
  to the registry only makes it server-owned if the body field stops winning:
  1. remove `"tools_available"` from `_SLOT_AGNOSTIC_KEYS` (`utils.py:65`) and from
     the `filter_session_config` allowlist (`utils.py:488`);
  2. in `_sanitize_session_config`, set `config["tools"]` from the registry entry the
     way `config["domain_profile"]` is set at `server.py:220`;
  3. `resolve_enabled_tools` reads the registry-bound value instead of `body`.

  Call this out in review as a privilege reduction: today a client can switch on a
  metered `web_search` that the selected prompt never declared.

**Step 1 net:** ~+130 new, ~−115 deleted. Duplication factor per tool: 7 → 1.

---

## 4. Step 2 — fold airline onto the same spec

Behavior-preserving refactor of the one domain that cannot break.

### 4.0 Split this step: ship the bug fixes now, gate the refactor

Step 2 as first drafted bundles two things with very different risk profiles.

**Step 2a — backport the two real defects. Independent of `ToolSpec`; do it now.**
`airline/thinker.py` lacks both safety features `generic/backend.py` has (§1.2), and
neither needs a refactor to add:

- wrap `_run_call`'s dispatch in `asyncio.timeout(...)` and the planner call in
  `asyncio.wait_for(...)`, with env-tunable defaults matching today's effective 30 s
  `THINKER_TOOL_TIMEOUT_SECONDS` ceiling;
- re-check `state.active_call_id` after the await in `call()` and raise
  `CancelledError` when the generation has moved on, as `generic/backend.py:75` does.

A small, independently reviewable diff against a well-tested file that captures most
of Step 2's user-visible value.

**Step 2b — unify the lifecycle as `PlannedBackend`. Gate this.**

| | |
| --- | --- |
| Cost | Touches the shipped airline demo; `tests/unit/test_frontend_backend_agent.py` is 1,549 lines / 53 tests; three intentional behaviour changes (§4.4) |
| Benefit once 2a has landed | ~240 lines removed and one lifecycle instead of two — the two defects are already fixed |

§4.4 also concedes airline should start with "permissive `ParamSpec`s that mirror what
each tool already accepts", so the parameter-validation benefit is deferred too.

**Recommendation: do 2a now; do 2b when a third stateful domain actually appears.**
Two implementations of one lifecycle is a tolerable cost. Three is not.

### 4.1 Unify the lifecycle

Move `generic/backend.py` to `src/backend.py` as `PlannedBackend` (it contains
nothing generic-specific). Add the two airline-only features behind
constructor arguments:

- `tool_delay_seconds` / `tool_delay_min_seconds` — the synthetic demo delay.
- `slots` passthrough — airline merges `call_backend` extra arguments into every
  tool call; generic passes `{}`.

Delete `airline/thinker.py`'s lifecycle half (`call`, `cancel_active`,
`_run_call`, `_next_tool_delay_seconds`, `_dispatch_parallel_tool_calls`,
`_dispatch_tool_call_safely`). Keep `cancel_pending_booking` /
`cancel_pending_work` as a domain hook on the airline `DomainSpec` — it is real
airline state semantics, not lifecycle.

### 4.2 Airline tools become specs

```python
TOOLS: dict[str, ToolSpec] = {spec.name: spec for spec in (
    ToolSpec(name="flight_search",
             contract="search flights",
             params={"origin_airport": ParamSpec(str), "dest_airport": ParamSpec(str),
                     "date": ParamSpec(str), "sorting": ParamSpec(str, required=False)},
             run=lambda args, ctx: flight_search(state=ctx.state, backend=ctx.backend, slots=args),
             mutates=True),
    ToolSpec(name="pnr_status",
             contract="check a booking",
             params={"pnr_code": ParamSpec(str)},
             run=lambda args, ctx: pnr_status(backend=ctx.backend, slots=args)),
    ToolSpec(name="booking",
             contract="continue booking a selected flight",
             params={...},
             run=lambda args, ctx: BookingTool(state=ctx.state, backend=ctx.backend).continue_booking(args),
             mutates=True),
)}
```

`booking_tool.py`, `flight_search.py`, `pnr_status.py`, `state.py`,
`transform.py`, `slot_parsing.py`, `backend.py`, `airports.py` are **unchanged** —
that is the code that earns its place.

### 4.3 The one asymmetry that must not collapse

`airline/thinker.py::_planner_response_hint` passes the model's `response_text`
through verbatim. `generic/dispatcher.py::_response_hint` accepts only a closed
vocabulary and generates the speech in Python.

**Do not unify these by weakening generic to airline's posture.** Make it a
`DomainSpec` field:

```python
response_hint_policy: Literal["closed_vocabulary", "planner_authored"] = "closed_vocabulary"
```

Airline sets `planner_authored` to preserve current behavior; the toolset domain
keeps `closed_vocabulary`. Tightening airline is a separate, prompt-tested change
and is explicitly **not** in this plan.

### 4.4 Behavior changes airline will acquire

These are improvements, but they are changes and need to be called out in review:

| Change | Effect | Mitigation |
| --- | --- | --- |
| Overall + planner deadlines | a hung Thinker now returns `timeout_failure()` instead of hanging until the 30s Pipecat `THINKER_TOOL_TIMEOUT_SECONDS` fires | default airline deadlines to match today's effective ceiling; make them env-tunable as generic's already are |
| Stale-generation guard | a superseded call can no longer deliver a late payload | this is the fix for a real interruption bug; verify against the interruption tests |
| Param validation | malformed planner params now fail with `missing_parameters` / `invalid_parameters` before the tool runs | airline tools currently self-validate; start with permissive `ParamSpec`s that mirror what each tool already accepts, and tighten later |

If Step 2a (§4.0) has landed, the first two rows are no longer new behaviour here —
they shipped separately and were reviewed on their own merits. Only the
parameter-validation row remains as a Step 2b delta.

### 4.5 Prompt

`thinker`: delete the "Available internal tools:" enumeration; keep planning
rules, output schema, and both worked examples.

**Step 2 net:** `airline/thinker.py` 301 → ~60 lines (`DomainSpec` glue plus
`cancel_pending_booking`); one dispatcher and one lifecycle serve both domains.

---

## 5. Test plan

### 5.0 Step 0 and Step 2a

- Step 0: the full existing suite passes unmodified except the two `tts_filter`
  import paths. Add a guard test (or an import-linter rule) asserting that
  `pipeline.py` and `src/` contain no imports from a domain package other than the
  `_DOMAIN_FACTORIES` string table.
- Step 2a: `test_frontend_backend_agent.py` green; add one test that a superseded
  airline call cannot deliver a late payload, and one that a hung planner returns a
  timeout payload rather than blocking until the Pipecat function timeout.

### 5.1 Step 1

Existing — mechanical updates:
- `tests/unit/test_frontend_backend_domains.py` patches `dispatcher.TOOL_SERVICES`
  at five sites; these become patches of the domain `TOOLS` registry.
- `format_tool_result` assertions move to the `generic/speech.py` functions.

New:
- `validate_arguments` table test: required/optional, bounds, choices, `max_len`,
  `bool`-is-not-`int`, unexpected-param rejection. One parameterized case per
  `ParamSpec` field, not per tool.
- `render_tool_block` renders only enabled specs, and the rendered names are
  exactly `resolve_enabled_tools`' output.
- **Registry consistency test** (this is the one that retires failure mode
  §1.1(2)): for every domain, every `ToolSpec` has a non-empty `contract`,
  `capability`, a `speak` callable or a self-formatting tool, and every name in
  the registry entry's `tools:` list resolves. This single test replaces the
  seven-site manual discipline.
- `unsupported_request` names only enabled capabilities — asserts §1.1(1) fixed.
- `filler_policy` (§1.4): `code_authored` ignores a model-supplied `filler_text`;
  `planner_authored` uses it. Assert the generic domain resolves to `code_authored`
  even when the model supplies `filler_text`.
- `tools_available` in a session body no longer widens the enabled set (§3.7).

### 5.2 Step 2

- `tests/unit/test_frontend_backend_agent.py` (1,549 lines, 16 test functions)
  is the acceptance gate. It must pass **unmodified** except where §4.4 changes
  behavior intentionally; each such edit needs a comment naming the row in §4.4.
- Add: `mutates=True` tools serialize in planner order while `mutates=False`
  tools run concurrently — the invariant currently asserted only by the comment
  at `airline/thinker.py::_dispatch_parallel_tool_calls`.
- Add: `response_hint_policy` — `planner_authored` passes model text through,
  `closed_vocabulary` rejects an out-of-vocabulary reason.

### 5.3 Manual

Per step, one voice session per domain: happy path, missing-parameter
clarification, mid-request interruption, `cancel_backend`, and a disabled-tool
request. Confirm the filler still fires at the 0.3s threshold and that the
capability sentence matches the enabled set.

---

## 6. Sequencing

| # | Commit | Step | Gate |
| --- | --- | --- | --- |
| 0 | de-privilege the airline (§Step 0) | 0 | full suite green; no domain names in `pipeline.py` / `src/` |
| 0b | airline deadlines + stale guard (§4.0) | 2a | `test_frontend_backend_agent.py` green; interruption tests reviewed |
| 1 | `src/tools.py` + registry consistency test (§3.1) | 1 | unit tests green; nothing wired yet |
| 2 | toolset domain onto `ToolSpec` (§3.2–3.6) | 1 | `test_frontend_backend_domains.py` green |
| 3 | generated prompt block + `unsupported_request` (§2.2) | 1 | manual: disable a tool, confirm the model stops planning it |
| 4 | registry `thinker_prompt` + `tools`, `tools_available` closed (§3.7) | 1 | new flavour with no Python; a body override no longer widens the set |
| 5 | `PlannedBackend` extraction (§4.1) | 2b | full unit suite green, both domains |
| 6 | airline tools → specs (§4.2–4.5) | 2b | `test_frontend_backend_agent.py` green; §4.4 deltas reviewed |

Commits 0 and 0b depend on nothing else here and can land immediately.

**Commit 3 is the payoff, not commit 1 or 2.** Commits 1-2 are a pure refactor with
nothing a user or a stakeholder can observe; commit 3 is what makes "disabled" mean
disabled and ends the drift class. Stopping after commit 4 is a coherent outcome —
**stopping after commit 2 is not**, because it leaves the tree mid-refactor for no
delivered benefit.

Commits 5-6 are gated on §4.0.

---

## 7. Acceptance

**Step 0**
- No module under `src/examples/frontend_backend_agent/src/` or `pipeline.py` imports
  a domain package; `_DOMAIN_FACTORIES` is the only place a domain is named.
- Behaviour identical; suite green with only import-path edits.

**Step 2a**
- A hung airline planner returns a timeout payload instead of blocking to the Pipecat
  function timeout.
- A superseded airline call cannot deliver a late payload.

**Step 1**
- Adding a tool touches exactly two files (`services.py`, the domain registry)
  plus one name in `examples_registry.yaml`. No prompt edit.
- Removing a tool is deleting one name; the Thinker prompt and the capability
  sentence both follow automatically.
- A new read-only flavour over **existing** tools requires no new Python module; a
  new capability still does, by design (§2.4).
- Swapping either prompt is a registry key change.
- A session body can no longer widen the enabled tool set (§3.7).

**Step 2**
- `test_frontend_backend_agent.py` passes with only §4.4-justified edits.
- One dispatcher and one lifecycle serve both domains.
- `airline/thinker.py` is under 80 lines.
- The `closed_vocabulary` posture of the toolset domain is unchanged.

---

## 8. Relationship to the prior plan

`docs/generic-frontend-backend-agent-implementation-plan.md` is correct that the
airline domain needs real code (its §3.3), and its §10 safety rules and §14
routing matrix are worth keeping. Two things are replaced:

- **Its §12.2 makes "domain" the unit of extension** — an eight-file package per
  flavor. The unit that actually changes is the *tool*. This plan makes a
  read-only flavor configuration and keeps packages only where state and
  side effects justify them.
- **Its §12.4 defers the structural work** ("reduce to fewer modules", "extract
  shared services", "extract a reusable planned-backend lifecycle base") to an
  unscheduled follow-up. Those are §3 and §4 here. Shipping the duplication and
  promising to collapse it later is how seven sites become fourteen.

Its §13–18 (phases, acceptance matrices, rollback, promotion gate) are rollout
process and are deliberately not reproduced here.
