# Frontend/Backend Agent Cascaded Example

The Frontend/Backend Agent is one shared Pipecat voice pipeline with replaceable domain behavior. A fast Talker large language model (LLM) owns the conversation. A separate Thinker LLM plans work that requires tools, state, or domain policy. The pipeline includes airline and generic-assistant domains. You can add a read-only generic flavor without copying the audio pipeline or its tool implementation.

The agent is not a ReAct agent. The Talker can call only `call_backend` and `cancel_backend`. A session-local backend asks the Thinker for a bounded plan, validates that plan in Python, runs the allowed domain tools, and returns a structured result to the Talker.

![Frontend/Backend Agent architecture](images/frontend-backend-agent-architecture.png)

## Request Flow

Each request follows the same path for every domain:

1. The transport receives user audio, and automatic speech recognition (ASR) produces a transcript.
2. The Talker answers stable conversational requests directly or calls `call_backend` with a self-contained request.
3. The session-local backend asks the Thinker for a plan. The selected registry entry controls the hidden Thinker prompt and, for the generic domain, the enabled internal tools.
4. The generic planner appends a generated tool-contract block for only those enabled tools. The airline domain keeps its existing prompt-owned contracts. Domain code validates each plan before dispatch.
5. The backend runs the approved tools and returns a structured `response_hint` or `tool_result`. The generic domain also generates user-facing capability text from the enabled tool specifications.
6. The Talker converts `response_text` into a concise spoken reply, and text-to-speech (TTS) produces audio.
7. `cancel_backend` or a newer superseding request cancels pending work and prevents stale results from reaching the conversation.

The React client is only the user interface. The agent orchestration runs in the Python Pipecat pipeline.

## Built-In Domains

Both built-ins point to `examples.frontend_backend_agent.pipeline:bot`. The selected example registry entry supplies the trusted domain and hidden Thinker prompt. The generic entry also supplies its enabled tool set.

| Registry Example | Domain Profile | Talker Prompt | Thinker Prompt | Internal Tools | Extra Dependency |
| --- | --- | --- | --- | --- | --- |
| `frontend-backend-agent` | `airline` | `talker` | `thinker` | Airline domain defaults | `booking-server` |
| `generic-frontend-backend-agent` | `generic` | `generic_talker` | `generic_thinker` | `get_weather`, `get_stock_price`, `web_search`, `calculate_bmi`, and `generate_random_number` | WeatherAPI, Finnhub, and Perplexity credentials for their respective live tools |

`domain_profile`, `thinker_prompt`, and `tools` are registry-owned. The server binds these values from `examples_registry.yaml`; a client session cannot replace the hidden prompt or widen the enabled tool set. `tools_available` is not accepted as session configuration. The pipeline resolves `domain_profile` through the code allowlist in `src/domain.py`. It never imports a client-provided module or path.

The existing `frontend-backend-agent` identifier remains the airline example. Existing airline prompts, booking behavior, booking-server selection, pronunciation handling, and call/cancel contract remain compatible.

## Run the Examples

Refer to the [Getting Started guide](../../../docs/01-getting-started.md) for prerequisites and hardware details. Run every command from the repository root.

### Configure Credentials

Create `.env` from the template, then add the credentials required by the domain you plan to run:

```bash
cp .env.example .env
```

Both domains require `NVIDIA_API_KEY` for the default NVIDIA cloud model services. The generic domain also recognizes these variables:

| Environment Variable | Required For | Default or Behavior |
| --- | --- | --- |
| `WEATHERAPI_KEY` | `get_weather` | No live weather result when unset |
| `WEATHERAPI_BASE_URL` | Custom WeatherAPI endpoint | `https://api.weatherapi.com/v1` |
| `FINNHUB_API_KEY` | `get_stock_price` | No live stock result when unset |
| `FINNHUB_BASE_URL` | Custom Finnhub endpoint | `https://finnhub.io/api/v1` |
| `PERPLEXITY_API_KEY` | `web_search` | No live web result when unset |
| `PERPLEXITY_BASE_URL` | Custom Perplexity endpoint | `https://api.perplexity.ai` |
| `PERPLEXITY_MODEL` | Perplexity model selection | `sonar` |

Store credentials in `.env` for Docker Compose. Do not put secrets in `prompts.yaml`, `examples_registry.yaml`, client session configuration, or source code. BMI calculation and random-number generation do not require an extra credential.

### Run Host-Native

The default registry selection is `all`, so the UI exposes both domain variants. Start the airline booking server only when you use the airline domain:

```bash
PYTHONPATH=src uv run python3 -m examples.frontend_backend_agent.airline.database.server
```

Start the application in another shell:

```bash
uv run python3 src/server.py --host 0.0.0.0 --port 7860
```

Open `https://localhost:7860/` and select **Airline Frontend Backend Agent** or **Generic Frontend Backend Agent**. To expose only one variant, set `EXAMPLE_SELECTION` before startup:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent uv run python3 src/server.py --host 0.0.0.0 --port 7860
```

### Run With Docker Compose

The existing `frontend-backend-agent` recipes provide the shared application and model services:

```bash
docker compose --profile frontend-backend-agent up -d
docker compose --profile frontend-backend-agent/server up -d
docker compose --profile frontend-backend-agent/single-gpu up -d
```

These commands select the airline domain by default. Override the selected registry example to run the generic domain through the same application recipe:

```bash
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent up -d
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent/server up -d
EXAMPLE_SELECTION=generic-frontend-backend-agent docker compose --profile frontend-backend-agent/single-gpu up -d
```

The existing Compose profile also starts the booking-server sidecar. The generic domain does not load or call that service. Clean up with the same profile that you started:

```bash
docker compose --profile frontend-backend-agent down
docker compose --profile frontend-backend-agent/server down
docker compose --profile frontend-backend-agent/single-gpu down
```

## Configure the Shared Pipeline

The following environment variables bound shared and domain-specific orchestration:

| Environment Variable | Default | Purpose |
| --- | --- | --- |
| `CHAT_HISTORY_RECENT_TURNS` | `20` | Retains this many recent non-prompt messages in the Talker context |
| `THINKER_FILLER_THRESHOLD_SECONDS` | `0.3` | Delays progress speech until delegated work remains active past the threshold |
| `THINKER_TOOL_TIMEOUT_SECONDS` | `30.0` | Bounds the shared Talker-to-backend function handler |
| `GENERIC_PLANNER_TIMEOUT_SECONDS` | `15.0` | Bounds generic Thinker planning |
| `GENERIC_BACKEND_TIMEOUT_SECONDS` | `40.0` | Bounds the generic planner and tool execution together |
| `AIRLINE_PLANNER_TIMEOUT_SECONDS` | `30.0` | Bounds airline Thinker planning; capped at the overall airline deadline |
| `AIRLINE_BACKEND_TIMEOUT_SECONDS` | `30.0` | Bounds airline planning and tool execution together |

The `generic-frontend-backend-agent` registry entry enables all 5 built-in generic tools. To expose a subset, create or edit a trusted registry entry. Client session data and Talker prompt metadata do not widen that set.

For model and catalog settings, refer to [Configure LLM](../../../docs/how-to/configure-llm.md) and [Configure Services](../../../docs/how-to/configure-services.md). For prompt behavior, tool subsets, and domain extension, refer to [Configure Frontend/Backend Agent Domains](../../../docs/how-to/configure-frontend-backend-domains.md).

## Domain Contract

`src/domain.py` defines the shared contract. A trusted domain factory returns one `DomainSpec` with these values:

| Field | Responsibility |
| --- | --- |
| `key` and `label` | Stable domain identity and human-readable name |
| `thinker_prompt_key` | Default hidden prompt that constrains the Thinker plan; the trusted registry entry can select another catalog key |
| `talker_tools_schema` | Talker-visible `call_backend` and `cancel_backend` definitions |
| `build_backend` | Session-scoped factory for the domain backend and state |
| `runtime_context` | Trusted date, time, or domain context appended to the Talker prompt |
| `intro_prompt` | Initial Talker instruction when welcome messages are enabled |
| `tts_text_transform` | Optional domain pronunciation transformation |
| `filler_policy` and `filler_selector` | Choose code-authored or planner-authored progress speech and provide the trusted selector when required |
| `tool_registry` | Publish the domain's code-owned `ToolSpec` allowlist for registry-selected capabilities |
| `max_query_chars` | Maximum delegated query length |

`build_backend` receives a `DomainBuildContext` with `thinker_llm`, the resolved `thinker_prompt`, `thinker_max_tokens`, registry-owned `tool_names`, `tool_delay_seconds`, `tool_delay_min_seconds`, and `load_service_entry`. The context does not expose the raw session body or prompt metadata to domain code.

The backend returned by `build_backend` implements 3 operations: `call`, `cancel_active`, and `cancel_pending_work`. The pipeline does not need to know the domain's state machine, external services, or result format.

## Understand Tool Specifications

`src/tools.py` defines the `ToolSpec` contract, and `generic/tools.py` declares each generic internal capability in one specification. The specification owns the tool name, planner contract, parameters, executor, speech formatter, deadline, mutation flag, and user-facing capability phrase. Generic validation, dispatch, result formatting, and Thinker context consume this same definition.

The executable service function remains Python code. Configuration selects existing capabilities; it does not define network requests, authentication, retries, or response parsing. This boundary keeps executable behavior reviewable and prevents registry data from becoming a code-injection surface.

At session startup, the generic planner renders an available-tool contract block from only the registry-enabled specifications. Its runtime `enabled_tools` list uses the same subset, and Python rejects plans outside that subset. Static output examples can still mention built-in names, but they do not enable those tools. The unsupported-request response also names only enabled capabilities.

## Add a Read-Only Flavor

You do not need a new Python package when a flavor reuses the generic domain's existing read-only tools, validation, services, result formatters, and concurrency rules:

1. Add or reuse a user-facing Talker prompt and a hidden Thinker prompt in `prompts.yaml`.
2. Add an entry to `examples_registry.yaml` that uses the shared `bot`, sets `domain_profile: generic`, selects `thinker_prompt`, and lists the allowed `tools`.
3. Declare only the model, automatic speech recognition (ASR), and text-to-speech (TTS) service slots that the flavor needs.
4. Hide internal prompts with `agent_prompt_keys`.
5. Add tests that verify the registry selection, generated tool block, capability response, and disabled-tool behavior.

The server treats the registry entry as trusted application configuration. Do not accept `domain_profile`, `thinker_prompt`, or `tools` from a client request.

## Add a Capability or Stateful Domain

Use the following sequence when you need a new executable capability or a stateful business domain:

1. For a new generic capability, implement the service function and add one `ToolSpec` to the generic tool registry. Select its name in the relevant `examples_registry.yaml` entries. Do not duplicate its schema, deadline, or speech policy in dispatcher tables or prompt prose.
2. For a stateful business domain, add a package under `src/examples/frontend_backend_agent/<domain>/` for its backend, services, state, validation, and result formatting.
3. Implement the `DomainBackend` call and cancellation contract.
4. Return a `DomainSpec` from a `create_domain_spec()` factory.
5. Add the factory to `_DOMAIN_FACTORIES` in `src/domain.py`. This explicit allowlist is required.
6. Add separate Talker and hidden Thinker prompts to `prompts.yaml`.
7. Add an example entry to `examples_registry.yaml`. Point `bot` at the shared pipeline, set `domain_profile` and `thinker_prompt`, declare only required service slots, and hide internal prompts with `agent_prompt_keys`. Add `tools` when the domain supports registry-selected capabilities.
8. Add model or sidecar entries to the example-local service catalogs when the domain needs another registered service.
9. Add tests for registry isolation, unknown and disabled tools, malformed plans, cancellation, concurrent requests, timeouts, credential failures, and deterministic spoken output.

Do not derive `domain_profile` from a user prompt or allow a request to provide a Python import path.

## Registry Configuration Versus Domain Code

A registry-configured flavor is appropriate when you keep the same internal tool names, parameter schemas, service adapters, state, validation, side effects, cancellation behavior, and result format. You can change the persona, spoken style, routing examples, direct-answer policy, hidden Thinker prompt, and enabled subset of existing tools without adding Python.

Add or change domain code when you introduce any of the following behavior:

- A new tool name, parameter, credential, external service, or side effect.
- New state, such as an order draft, account context, or confirmation workflow.
- New validation, authorization, grounding, privacy, or business-policy rules.
- A different result envelope, pronunciation policy, filler policy, timeout, or concurrency rule.
- A different service slot or deployment dependency.

Prompt text cannot safely implement those controls because model output is untrusted.

## Safety, Grounding, and Concurrency

The pipeline enforces the following boundaries:

- The Talker sees only `call_backend` and `cancel_backend`; internal domain tools remain hidden.
- The server owns `domain_profile`, `thinker_prompt`, and `tools`. Code restricts the domain to registered factories and resolves tool names against that domain's registry.
- The generic generated tool-contract block and user-facing capability sentence contain only enabled tool specifications.
- Generic tool plans are validated atomically before any tool runs. Unknown tools, disabled tools, unexpected parameters, and more than 3 calls fail closed.
- Up to 3 validated generic read-only tools can run concurrently. Results return in planner order.
- A backend instance and its state belong to one voice session. A new delegated request cancels and replaces unfinished work in that session.
- Cancellation invalidates the active call identifier, so a late result cannot become the current response.
- Airline planning and overall backend execution have bounded deadlines. A superseded airline generation cannot deliver a late result.
- Live-data tools read credentials from the process environment. Credentials never enter the Thinker request or tool parameters.
- Missing credentials, timeouts, invalid responses, and upstream failures return bounded unavailable responses. The generic tools do not substitute fabricated data.
- Deterministic Python formatters produce TTS-safe result text from validated inputs and returned service data.
- The generic domain uses code-authored progress speech and ignores model-supplied filler text. The airline domain retains planner-authored filler for backward compatibility.

After a prompt or domain change, test direct Talker replies, delegation, cancellation, parameter clarification, disabled tools, unavailable credentials, parallel calls, session isolation, and repeated tool-calling behavior.
