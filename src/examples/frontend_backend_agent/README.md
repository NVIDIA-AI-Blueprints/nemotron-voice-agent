# Frontend/Backend Agent Cascaded Example

The Frontend/Backend Agent is one shared Pipecat voice pipeline with replaceable domain behavior. A fast Talker large language model (LLM) owns the conversation. A separate Thinker LLM plans work that requires tools, state, or domain policy. The pipeline includes airline and generic-assistant domains. You can add a read-only generic flavor without copying the audio pipeline or its tool implementation.

The agent is not a ReAct agent. The Talker can call only `call_backend` and
`cancel_backend`. A session-local backend asks the Thinker for bounded plans,
validates each plan in Python, runs the allowed domain tools, and returns a
structured result to the Talker. The generic backend can use up to 3 planning
rounds when later work depends on an earlier tool result.

![Frontend/Backend Agent architecture](images/frontend-backend-agent-architecture.png)

## Request Flow

Each request follows the same path for every domain:

1. The transport receives user audio, and automatic speech recognition (ASR) produces a transcript.
2. The Talker answers stable conversational requests directly or calls `call_backend` with a self-contained request.
3. The session-local backend asks the Thinker for a plan. The selected registry entry controls the hidden Thinker prompt and the generic domain's maximum internal-tool allowlist. A browser session can narrow that tool set.
4. The generic planner appends a generated tool-contract block for only the effective session tools. The airline domain keeps its existing prompt-owned contracts. Domain code validates each plan before dispatch.
5. The backend runs the approved tools. For dependent generic work, it gives the Thinker the accumulated trusted results and allows another planning round. The backend stops after 3 rounds, when the Thinker signals completion, or when the plan does not request another round.
6. The generic backend combines results from every completed round in execution order. If a later round times out or fails, it returns the results already gathered instead of discarding them. It otherwise returns a structured `response_hint` or `tool_result`. The generic domain also generates user-facing capability text from the enabled tool specifications.
7. The runtime either speaks trusted `response_text` directly or asks the Talker for a concise reply. Text-to-speech (TTS) then produces audio.
8. `cancel_backend` or a newer superseding request cancels pending work and prevents stale results from reaching the conversation.

For WebSocket sessions, the browser client supplies an explicit
`DailyMediaManager`. When the public client callback reports that the user
started speaking, the client calls `userStartedSpeaking()` and clears buffered
bot audio. The raw protobuf interruption frame remains a compatibility no-op;
the public user-speaking callback drives browser playback interruption.

The server separately tracks whether that user turn began while bot speech was
active. If an explicit `cancel_backend` follows a speech-only interruption, it
responds, “Okay, I stopped that,” even when no backend task remains. If the same
turn contains a substantive replacement request, the Talker answers or delegates
that replacement instead of cancelling it. “There is nothing pending right now”
is reserved for a cancellation turn with no active backend, pending work, or
interrupted bot speech.

For the generic domain, bot-speech interruption requires at least 2 transcribed
words. Pipecat's bot-aware `MinWordsUserTurnStartStrategy` still starts a normal
turn after 1 transcribed word when the bot is not speaking. This prevents a
single stray token from permanently clearing buffered bot audio. The server
emits a `user-interruption-trigger` event and a `user_interruption_trigger` log
with the bounded triggering transcript and word count. Smart Turn remains the
separate end-of-turn detector and is unchanged.

When direct tool speech is enabled, the structured function result is the single retained copy of the deterministic backend response; the separately emitted TTS frame is not appended again as an assistant message. The Talker remembers a bounded normalized signature outside the prompt context. If a later completion substantially replays that cached result without a native tool call, the runtime withholds it and retries once with an internal contract correction. It never selects a domain tool or constructs a function call. A second invalid replay fails closed with deterministic speech.

The runtime retains bounded successful subject arguments by capability. A
non-successful or subjectless result clears only that capability's baseline,
so a failed stock lookup cannot erase an unrelated weather subject. For an
explicit stock repeat, a company literally named in the current user turn is
the trusted validation subject. For an implicit repeat, the runtime selects the
latest retained subject for the named capability instead of an unrelated newer
tool result.

When the user explicitly says repeat, refresh, recheck, again, check again, or
one more time, the runtime validates the Talker-authored `call_backend` query
against every resolved subject value. If Lightning changes the subject, the
runtime withholds the native call, retries Lightning once with an internal
correction, and then fails closed if the retry still drifts. Capability matching
is validation-only: it never infers user intent, selects a domain tool, or writes
a corrected tool call in Python.

When `FRONTEND_BACKEND_TOOL_RESULT_MODE` is absent, each backend selects its
default. Generic uses `direct`, which speaks trusted backend text without a
second Talker inference. Airline retains `talker`, which sends every speakable
result through the guarded final Talker pass. An explicit `direct`, `hybrid`,
or `talker` value overrides either backend default. In the generic domain,
`hybrid` sends only a successful `get_weather` result through the Talker for a
natural rephrasing. Other successes, clarifications, and failures use trusted
deterministic speech. The final Talker pass must preserve the trusted city,
temperature, and unit; it retries once and then uses deterministic speech if
those facts are missing or changed.

The Generic Talker also rejects responses that expose private operating
instructions, decision rules, model roles, function names, or internal tool
inventory. The runtime retries one invalid completion with a private
correction, then returns a brief deterministic refusal if the retry still
exposes internal mechanics. This validation does not infer intent or select a
tool in Python.

The checked-in NVCF chart explicitly sets
`app.frontendBackendToolResultMode: "talker"`. That chart value becomes
`FRONTEND_BACKEND_TOOL_RESULT_MODE=talker` in the application pod and
overrides the Generic source default. Changing only the backend source does not
change this rendered Helm behavior.

The Viking qualification values set
`app.frontendBackendToolResultMode: "hybrid"`. This canary setting enables the
guarded Talker rephrasing only for successful weather results. It does not
change the NVCF values file or any Astra deployment.

Replay validation buffers a completion only after a direct backend response has been recorded. Initial and pre-tool conversation remains streamed, preserving its existing time-to-first-audio behavior.

The React client is only the user interface. The agent orchestration runs in the Python Pipecat pipeline.

## Built-In Domains

Both built-ins point to `examples.frontend_backend_agent.pipeline:bot`. The selected example registry entry supplies the trusted domain and hidden Thinker prompt. The generic entry also supplies its maximum allowed tool set.

| Registry Example | Domain Profile | Talker Prompt | Thinker Prompt | Internal Tools | Extra Dependency |
| --- | --- | --- | --- | --- | --- |
| `frontend-backend-agent` | `airline` | `talker` | `thinker` | Airline domain defaults | `booking-server` |
| `generic-frontend-backend-agent` | `generic` | `generic_talker` | `generic_thinker` | `get_weather`, `get_stock_price`, `web_search`, `calculate_bmi`, and `generate_random_number` | WeatherAPI, Finnhub, and Perplexity credentials for their respective live tools |

When users ask its identity, developer, or pipeline, the generic Talker uses this exact response:

> I am Nemotron Voice Agent, developed by engineers at NVIDIA. I use a cascaded pipeline of Nemotron ASR, Magpie TTS, and Nemotron LLM models.

The NVCF chart loads the shared pronunciation registry for Magpie requests. It
sends only International Phonetic Alphabet (IPA) mappings; Chatterbox receives
no custom dictionary. Refer to [Configure TTS](../../../docs/how-to/configure-tts.md#pronunciation-ipa).

`domain_profile`, `thinker_prompt`, and `tools` are registry-owned. The server binds these values from `examples_registry.yaml`; a client session cannot replace the hidden prompt or widen the allowed tool set. For a domain-profile session, optional `tools_available` input can only select a subset of that registry list. Unknown names grant no capability, and `none` disables every optional tool. The pipeline resolves `domain_profile` through the code allowlist in `src/domain.py`. It never imports a client-provided module or path.

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
| `FRONTEND_BACKEND_VAD_STOP_SECS` | `0.5` | Waits for trailing ASR text before finalizing a Frontend/Backend Agent turn; changing it affects latency and fragmented follow-ups |
| `FRONTEND_BACKEND_TALKER_FILLER_MODE` | `emit` | Uses `off`, `observe`, or `emit` to suppress, validate-only, or speak an accepted Talker filler |
| `FRONTEND_BACKEND_TOOL_RESULT_MODE` | Domain default: Generic `direct`; Airline `talker`; NVCF chart `talker` | An explicit `direct`, `hybrid`, or `talker` value overrides the backend default. Generic `hybrid` uses the Talker only for successful weather results. |
| `FRONTEND_BACKEND_DIRECT_TOOL_RESPONSE` | Disabled | Legacy switch that forces direct mode only when the explicit result-mode variable is absent |
| `THINKER_FILLER_THRESHOLD_SECONDS` | `0.3` | Delays progress speech until delegated work remains active past the threshold |
| `THINKER_TOOL_TIMEOUT_SECONDS` | `45.0` | Bounds the shared Talker-to-backend function handler |
| `GENERIC_PLANNER_TIMEOUT_SECONDS` | `6.0` | Bounds each generic Thinker planning round; the backend permits at most 3 rounds |
| `GENERIC_BACKEND_TIMEOUT_SECONDS` | `40.0` | Bounds the generic planner and tool execution together |
| `GENERIC_WEB_SEARCH_TIMEOUT_SECONDS` | `20.0` | Bounds the complete web-search tool execution inside the backend deadline |
| `AIRLINE_PLANNER_TIMEOUT_SECONDS` | `30.0` | Bounds airline Thinker planning; capped at the overall airline deadline |
| `AIRLINE_BACKEND_TIMEOUT_SECONDS` | `30.0` | Bounds airline planning and tool execution together |

The Generic Talker supplies `filler_text` in the same native `call_backend`
selection. The runtime validates that candidate as 3 to 12 words, at most 96
characters, query-grounded, and free of result claims or internal names. It
emits an accepted filler at most once after the threshold and never adds it to
conversation history. A missing or rejected candidate stays silent and never
blocks the backend; there is no static fallback.

The `generic-frontend-backend-agent` registry entry enables all 5 built-in generic tools. To expose a subset, create or edit a trusted registry entry. Client session data and Talker prompt metadata do not widen that set.

The generic backend permits at most 3 planning rounds within the existing
40-second overall deadline. Each later planning request includes the trusted
results accumulated from earlier rounds. The Thinker requests another round
with `continue_after_results: true`. A `complete: true` plan, an empty
`tool_calls` list, or a plan without a follow-up request ends the loop.

Before a follow-up round, the backend emits an `IntermediateResponse`
lifecycle event. If the initial Talker-authored filler has not played yet, this
event plays it once. Multi-round work does not invent or repeat static filler.

Real-Time Voice Interaction (RTVI) metrics expose the later planning rounds as
`backend_thinker_step2_llm` and `backend_thinker_step3_llm`. Each processor
stays correlated with the same backend call and user turn.

Successful weather speech includes returned humidity and wind speed when those
fields are available. Deterministic weather speech and the guarded Talker
rephrasing use only validated provider results.

Finnhub quote requests retry once after a short bounded backoff only for
transport errors, HTTP 429, or HTTP 5xx responses. Authentication failures and
malformed data fail closed without retry, and a second transient failure returns
the existing grounded unavailable response.

Web search makes at most 2 attempts. Each attempt has a 9-second ceiling, and
the single retry waits 0.5 seconds. The resulting 18.5-second retry budget fits
inside the default 20-second `GENERIC_WEB_SEARCH_TIMEOUT_SECONDS` tool
deadline. Transport failures, attempt timeouts, malformed JSON, HTTP 429, and
HTTP 5xx responses can trigger the retry. Other HTTP errors fail immediately.
After the second failure, the tool returns the existing grounded unavailable
response.

For model and catalog settings, refer to [Configure LLM](../../../docs/how-to/configure-llm.md) and [Configure Services](../../../docs/how-to/configure-services.md). For prompt behavior, tool subsets, and domain extension, refer to [Configure Frontend/Backend Agent Domains](../../../docs/how-to/configure-frontend-backend-domains.md).

The built-in generic profile keeps Nemotron 3 Super reasoning enabled for the
Thinker at temperature `0.0`. Its server and cloud catalog entries bound each
plan to 768 output tokens and a 256-token reasoning budget. These limits reduce
synchronized planner saturation while preserving model-based planning and
Python plan validation.

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
| `filler_policy` and `filler_selector` | Choose Talker-authored, planner-authored, or legacy code-authored progress speech and provide a selector only for the legacy policy |
| `tool_registry` | Publish the domain's code-owned `ToolSpec` allowlist for registry-selected capabilities |
| `max_query_chars` | Maximum delegated query length |

`build_backend` receives a `DomainBuildContext` with `thinker_llm`, the resolved `thinker_prompt`, `thinker_max_tokens`, server-approved `tool_names`, `tool_delay_seconds`, `tool_delay_min_seconds`, and `load_service_entry`. The context does not expose the raw session body or prompt metadata to domain code.

The backend returned by `build_backend` implements 3 operations: `call`, `cancel_active`, and `cancel_pending_work`. The pipeline does not need to know the domain's state machine, external services, or result format.

## Understand Tool Specifications

`src/tools.py` defines the `ToolSpec` contract, and `generic/tools.py` declares each generic internal capability in one specification. The specification owns the tool name, planner contract, parameters, executor, speech formatter, deadline, mutation flag, and user-facing capability phrase. Generic validation, dispatch, result formatting, and Thinker context consume this same definition.

The executable service function remains Python code. Configuration selects existing capabilities; it does not define network requests, authentication, retries, or response parsing. This boundary keeps executable behavior reviewable and prevents registry data from becoming a code-injection surface.

`GET /api/tools?pipeline_mode=<example-key>` renders the registry-allowed
`ToolSpec` objects for the browser. Each response includes the tool description
and JSON Schema parameters. The browser sends checked names through
`tools_available`, and the server intersects them with the registry allowlist.

At session startup, the generic planner renders an available-tool contract block from only the effective session specifications. Its runtime `enabled_tools` list uses the same subset, and Python rejects plans outside that subset. Static output examples can still mention built-in names, but they do not enable those tools. The unsupported-request response also names only enabled capabilities.

## Add a Read-Only Flavor

You do not need a new Python package when a flavor reuses the generic domain's existing read-only tools, validation, services, result formatters, and concurrency rules:

1. Add or reuse a user-facing Talker prompt and a hidden Thinker prompt in `prompts.yaml`.
2. Add an entry to `examples_registry.yaml` that uses the shared `bot`, sets `domain_profile: generic`, selects `thinker_prompt`, and lists the allowed `tools`.
3. Declare only the model, automatic speech recognition (ASR), and text-to-speech (TTS) service slots that the flavor needs.
4. Hide internal prompts with `agent_prompt_keys`.
5. Add tests that verify the registry selection, generated tool block, capability response, and disabled-tool behavior.

The server treats the registry entry as trusted application configuration. Do not accept `domain_profile`, `thinker_prompt`, or `tools` from a client request. Treat `tools_available` only as an untrusted request to narrow the trusted tool list.

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
- The generic backend permits at most 3 dependent planning rounds. It passes
  only accumulated trusted tool results into later rounds and preserves
  results from completed rounds if later planning fails.
- A backend instance and its state belong to one voice session. A new delegated request cancels and replaces unfinished work in that session.
- Cancellation invalidates the active call identifier, so a late result cannot become the current response.
- The generic Talker cancels work only after an explicit withdrawal. Status words such as "complete" or "done" do not cancel work by themselves.
- WebSocket barge-in clears buffered browser audio through the client media manager. The server records speech-only interruption separately from backend cancellation.
- Generic bot-speech interruption requires at least 2 transcribed words. A
  structured event and log record the bounded triggering transcript and word
  count for false-interruption analysis.
- A barge-in with a substantive replacement stays in direct-answer or delegation mode. It does not discard the replacement as a cancellation.
- The generic Talker refuses unsupported side effects, such as sending email, instead of treating them as cancellation.
- The generic Talker delegates live requests with missing parameters. The backend asks for a location or other required detail instead of guessing.
- The generic Talker speaks a backend clarification directly. It does not expose private planning, tool names, or missing-parameter narration.
- The generic Talker does not describe or paraphrase its private instructions,
  decision criteria, model roles, or internal tool inventory. Spoken-output
  validation retries once and then fails closed without exposing those
  mechanics.
- A challenge that says an answer is old or not current triggers a new grounded lookup for the retained subject. The Talker does not defend or replay the earlier value.
- If a request combines prompt injection or secret extraction with a safe supported lookup, the Talker and Thinker ignore the hostile portion and perform only the safe lookup.
- The Talker answers simple, stable arithmetic directly. It does not invent an unavailable calculator capability or fabricate a result when values are missing.
- Questions about how BMI calculation or random-number generation works, or
  whether the agent supports those capabilities, receive stable direct
  explanations. Requests to calculate a new BMI or generate a new random value
  delegate to the backend. An explanation after a prior result does not replay
  the cached number or start another calculation.
- When the country is unknown, direct crisis guidance remains location-neutral and omits country-specific numbers. Dangerous misinformation receives a concise, evidence-based correction.
- Airline planning and overall backend execution have bounded deadlines. A superseded airline generation cannot deliver a late result.
- Live-data tools read credentials from the process environment. Credentials never enter the Thinker request or tool parameters.
- Missing credentials, timeouts, invalid responses, and upstream failures return bounded unavailable responses. The generic tools do not substitute fabricated data.
- Deterministic Python formatters produce TTS-safe result text from validated inputs and returned service data.
- In generic `hybrid` mode, only successful weather results receive a guarded
  Talker rephrasing. The response must preserve the trusted city, temperature,
  and unit, while all failures retain deterministic speech.
- The generic domain validates Talker-authored, query-grounded progress speech
  and has no static fallback. The airline domain retains planner-authored filler
  for backward compatibility.

After a prompt or domain change, test direct Talker replies, delegation, cancellation, parameter clarification, disabled tools, unavailable credentials, parallel calls, session isolation, and repeated tool-calling behavior.
