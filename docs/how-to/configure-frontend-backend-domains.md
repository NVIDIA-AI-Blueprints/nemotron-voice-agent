# Configure Frontend/Backend Agent Domains

Use a Frontend/Backend Agent domain when you want to keep one real-time voice pipeline and replace the task-specific backend. The repository includes airline and generic domains.

## Understand the Shared Architecture

The shared Pipecat pipeline separates low-latency conversation from slower task execution:

1. The Talker LLM receives the transcript and conversation history.
2. The Talker answers stable conversational questions directly.
3. For domain work, the Talker emits `call_backend`. It emits `cancel_backend` when the user withdraws pending work.
4. A session-local backend sends the self-contained request to the registry-selected hidden Thinker prompt.
5. For the generic domain, the planner appends a generated contract block for only the registry-enabled tools. The airline domain keeps its existing prompt-owned contracts. Python validates the plan before dispatch.
6. The backend returns structured response text to the Talker.
7. The Talker produces the final text-to-speech (TTS) response.

The Talker sees only 2 functions. Internal functions, credentials, backend state, and tool results remain behind the domain boundary. The Thinker produces a bounded plan; the implementation does not use a ReAct observe-and-replan loop.

## Choose a Built-In Domain

The following registry entries use the same `examples.frontend_backend_agent.pipeline:bot` implementation:

| Registry Example | `domain_profile` | User-Facing Prompt | `thinker_prompt` | Internal Tools |
| --- | --- | --- | --- | --- |
| `frontend-backend-agent` | `airline` | `talker` | `thinker` | Domain-owned flight search, booking, and passenger name record (PNR) status |
| `generic-frontend-backend-agent` | `generic` | `generic_talker` | `generic_thinker` | Registry-selected weather, stock price, web search, body mass index (BMI), and random-number tools |

The registry loader normalizes `domain_profile`, `thinker_prompt`, and `tools` as fields on each `ExampleEntry`. During `_sanitize_session_config`, the server binds those fields from the selected entry and overwrites client-supplied values. Domain resolution uses the fixed `_DOMAIN_FACTORIES` allowlist in `src/examples/frontend_backend_agent/src/domain.py`. Tool resolution uses the selected domain's code-owned registry.

This design prevents a client from changing the backend domain, selecting a hidden prompt, enabling a metered tool, or requesting an arbitrary Python module independently of the selected example. The session allowlist does not accept `tools_available`.

## Select a Domain Locally

The default `selection: all` setting in `examples_registry.yaml` exposes both built-ins in the UI. Start the server from the repository root:

```bash
uv run python3 src/server.py --host 0.0.0.0 --port 7860
```

To expose only the generic variant, use the registry environment override:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent uv run python3 src/server.py --host 0.0.0.0 --port 7860
```

The airline domain also needs its booking service. Start it in another shell for a host-native run:

```bash
PYTHONPATH=src uv run python3 -m examples.frontend_backend_agent.airline.database.server
```

The existing Compose recipes select the airline entry by default. Set `EXAMPLE_SELECTION` to use the generic entry through the same application and model recipe:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent up -d
```

For a local model deployment on workstation or server GPUs, use:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent/server up -d
```

For a supported single-GPU deployment, use:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent/single-gpu up -d
```

The Compose profile still starts the booking server. The generic backend does not load or call it because its registry entry does not include the `booking-server` slot.

## Understand Generic Tool Specifications

`src/examples/frontend_backend_agent/src/tools.py` defines the `ToolSpec` contract, and `src/examples/frontend_backend_agent/generic/tools.py` declares every generic internal capability in one specification. Each specification owns its planner contract, parameters, executable adapter, speech formatter, deadline, mutation flag, and capability phrase. Validation, dispatch, result formatting, the hidden Thinker context, and user-facing capability text consume the same definition.

The built-in generic registry entry enables these internal tools:

| Tool | Purpose | Credential | Important Boundary |
| --- | --- | --- | --- |
| `get_weather` | Current conditions for a city or location | `WEATHERAPI_KEY` | Does not provide forecasts or historical weather |
| `get_stock_price` | Current public-company quote | `FINNHUB_API_KEY` | Does not provide predictions, crypto, commodities, or historical prices |
| `web_search` | Current or externally verifiable information | `PERPLEXITY_API_KEY` | Requests concise spoken text without URLs and strips numeric citation markers |
| `calculate_bmi` | Metric adult BMI screening calculation | None | Requires explicit weight in kilograms and height in meters |
| `generate_random_number` | Inclusive random integer | None | Accepts a bounded minimum and maximum |

Add live-tool credentials to `.env` for Docker Compose or to the process environment for a host-native run:

```bash
WEATHERAPI_KEY=<weatherapi-key>
FINNHUB_API_KEY=<finnhub-key>
PERPLEXITY_API_KEY=<perplexity-key>
```

The services also recognize these optional endpoint settings:

| Environment Variable | Default |
| --- | --- |
| `WEATHERAPI_BASE_URL` | `https://api.weatherapi.com/v1` |
| `FINNHUB_BASE_URL` | `https://finnhub.io/api/v1` |
| `PERPLEXITY_BASE_URL` | `https://api.perplexity.ai` |
| `PERPLEXITY_MODEL` | `sonar` |

Do not pass credentials through `call_backend`, Thinker plans, prompt text, or client session configuration. Each service reads its credential directly from the process environment.

When a credential is absent, the service returns an unavailable result. It does not use sample data or a stale fallback. This lets the Talker report the failure without presenting fabricated live information.

### Restrict the Generic Tool Set

Use the trusted `tools` list in `examples_registry.yaml` to choose a subset of the 5 registered generic tools. The server resolves every name against the generic domain's code-owned registry. Unknown names do not create executable capabilities.

```yaml
examples:
  search-frontend-backend-agent:
    label: Search Frontend Backend Agent
    domain_profile: generic
    thinker_prompt: generic_thinker
    tools: [web_search]
    agent_prompt_keys:
      - talker
      - thinker
      - generic_thinker
    slots: [llm, thinker-llm, asr, tts]
    defaults:
      prompt: [search_talker]
      llm: [nemotron-lightning-talker]
      thinker-llm: [nemotron-super-reasoning]
      asr: [nemotron-asr-streaming-english]
      tts: [magpie-multilingual-tts]
    bot: examples.frontend_backend_agent.pipeline:bot
```

Add `search_talker` to `prompts.yaml`, or use another compatible Talker prompt. Preserve the trust, grounding, delegation, cancellation, and spoken-output rules. You can also select a different hidden Thinker prompt through `thinker_prompt`; keep its output envelope and trust-boundary rules compatible with the planner parser.

At session startup, the generic planner renders an available-tool contract block from only the registry-enabled `ToolSpec` objects. Its runtime `enabled_tools` list uses the same subset, and Python rejects plans outside that subset. Static output examples can still mention built-in names, but they do not enable those tools. Unsupported-request capability text also uses only the enabled set.

Keep the user-facing Talker prompt and its hidden Thinker prompt separate. The registry's `agent_prompt_keys` hides internal prompts from the prompt selector. Prompt `tools_available` metadata can describe a prompt to the catalog and user interface, but it does not select generic backend tools. Client session data cannot widen the registry-owned set.

### Add a Generic Tool

Configuration can reuse an existing capability, but it cannot implement one. To add a generic capability:

1. Implement the service function in Python.
2. Add one `ToolSpec` to the generic tool registry.
3. Add the tool name to each trusted registry flavor that should expose it.
4. Add focused tests for validation, credential failures, timeouts, speech formatting, generated prompt content, and capability text.

Do not repeat parameter schemas, deadlines, capability descriptions, or success-formatting branches in dispatcher tables or prompt prose. The `ToolSpec` is the single declaration that coordinates these behaviors.

## Understand the Domain Contract

A domain factory returns a frozen `DomainSpec`. The shared pipeline consumes the following fields:

| Field | Domain Responsibility |
| --- | --- |
| `key` | Match the allowlisted `domain_profile` key |
| `label` | Identify the domain in logs and diagnostics |
| `thinker_prompt_key` | Provide the default hidden planner prompt; a trusted registry entry can select another catalog key |
| `talker_tools_schema` | Define the domain-specific descriptions for `call_backend` and `cancel_backend` |
| `build_backend` | Create a backend and isolated state for one session |
| `runtime_context` | Append trusted date, time, timezone, or domain context |
| `intro_prompt` | Define the welcome-turn instruction |
| `tts_text_transform` | Apply optional pronunciation handling |
| `filler_policy` | Choose code-authored or planner-authored progress speech |
| `filler_selector` | Select trusted code-authored progress speech when the policy requires it |
| `tool_registry` | Publish the domain's code-owned `ToolSpec` allowlist for registry-selected capabilities |
| `max_query_chars` | Bound delegated input length |

`build_backend` receives a `DomainBuildContext` with `thinker_llm`, the resolved `thinker_prompt`, `thinker_max_tokens`, registry-owned `tool_names`, `tool_delay_seconds`, `tool_delay_min_seconds`, and `load_service_entry`. It does not receive the raw session body or prompt metadata.

The returned backend must implement:

- `call(query, slots, on_started=...)` to plan and execute one request.
- `cancel_active(reason)` to cancel an active request.
- `cancel_pending_work()` to clear domain state that remains after active execution.

Keep the backend session-scoped. Do not store mutable conversation state in a module-level object.

## Add a Read-Only Flavor

If a new flavor reuses the generic domain's tools and behavior, add or reuse Talker and Thinker prompts, then add a registry entry. Set `domain_profile: generic`, select `thinker_prompt`, and list only the existing `tools` that the flavor can use. You do not need a new Python package.

Use this path only when the flavor keeps the same parameter schemas, service adapters, state model, validation, side effects, cancellation behavior, result envelope, filler policy, and concurrency rules.

## Add a Stateful Domain

Follow these steps when the new use case requires state, side effects, or different enforcement behavior:

1. Create `src/examples/frontend_backend_agent/<domain>/`.
2. Implement domain state, service adapters, parameter validation, result formatting, and the backend protocol.
3. Add a `create_domain_spec()` factory that returns a complete `DomainSpec`.
4. Add the factory import target to `_DOMAIN_FACTORIES`. Unknown keys must continue to fail closed.
5. Add a user-facing Talker prompt and a hidden Thinker prompt to the example-local `prompts.yaml`.
6. Add a new entry to `examples_registry.yaml`. Use the shared `bot`, set `domain_profile` and `thinker_prompt`, list only required service slots, and hide internal prompt keys. Add `tools` when the domain supports registry-selected capabilities.
7. Add service-catalog entries and deployment sidecars only when the domain needs them.
8. Add unit tests for domain selection, client override rejection, prompt isolation, tool validation, timeouts, cancellation, concurrent requests, session isolation, credential failures, and safe result text.

The shared `pipeline.py` should not import the new backend directly. It resolves the domain through `DomainSpec`.

## Decide Between Configuration and Python

Use a registry-configured flavor when all executable behavior remains the same. Prompts and registry fields can adjust:

- Persona, tone, response length, and TTS-ready wording.
- Which stable questions the Talker answers directly.
- Routing examples and delegation wording.
- The hidden Thinker prompt and enabled subset of already registered domain tools.

Add or change a domain plugin when the flavor needs:

- New tool names, parameters, credentials, services, or side effects.
- New session state, confirmation, authorization, or business workflows.
- New validation, privacy, grounding, or result-formatting rules.
- Different cancellation, timeout, concurrency, filler, or pronunciation behavior.
- A new registry service slot or deployment dependency.

Prompts and registry entries select trusted behavior, but Python remains the enforcement boundary. Do not encode HTTP requests, authentication, retries, or result parsing in configuration.

## Preserve Safety and Concurrency Guarantees

The generic domain applies the following controls:

- It validates the structure of every call in a multi-tool plan before it starts any tool.
- It rejects unknown tools, disabled tools, unexpected parameters, invalid values at the individual tool boundary, and plans with more than 3 calls.
- It builds the generated available-tool block and runtime `enabled_tools` list from registry-enabled specifications. Python rejects calls outside that subset, and unsupported-request responses name only enabled capabilities.
- It runs up to 3 validated read-only tools concurrently and preserves planner order in the combined result.
- It bounds planner, backend, and individual tool execution with timeouts.
- It treats the user request and retrieved webpages as untrusted input.
- It creates final spoken text from validated arguments and returned service data.
- It cancels and replaces an unfinished request when the same session sends newer delegated work.
- It invalidates the active call identifier before cancellation, which suppresses late stale results.
- It uses code-authored progress speech and ignores model-supplied filler text.

The airline backend keeps its stateful booking workflow, booking-server integration, planner-authored filler, and shared call/cancellation contract for backward compatibility. `AIRLINE_PLANNER_TIMEOUT_SECONDS` and `AIRLINE_BACKEND_TIMEOUT_SECONDS` both default to `30.0` seconds. The planner deadline cannot exceed the overall deadline. A newer generation suppresses a superseded call's late result.

## Validate a Domain Change

At minimum, test the following behavior:

- Stable Talker questions do not invoke the backend.
- Tool-backed questions invoke `call_backend` with a self-contained request.
- Stop, cancel, and topic-switch turns invoke `cancel_backend` when work is pending.
- Missing parameters return a clarification without starting a tool.
- Unknown, disabled, malformed, and over-limit plans run no tools.
- Missing credentials and upstream failures produce unavailable responses without mock data.
- Multi-tool requests execute concurrently and return results in planner order.
- New requests cancel old work, and concurrent sessions do not share mutable state.
- Airline search, booking, passenger name record status, pronunciation, and booking-server behavior remain unchanged.

Refer to the [Frontend/Backend Agent README](../../src/examples/frontend_backend_agent/README.md) for runtime variables and the [Configure Prompts guide](configure-prompts.md) for general prompt-catalog behavior.
