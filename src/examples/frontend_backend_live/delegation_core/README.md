# Delegation Core

The delegation core runs the work that a voice frontend delegates. It accepts each request through the OpenAI Responses API, runs it on the backend adapter that its settings name, and streams the response's events back. By default, the backend adapter is the backend large language model (LLM), directly connected. The core is a separate Python package in the example's folder. It imports nothing from the Live gateway ([`src/gateways/live/`](../../../gateways/live/)), the example's frontend adapter, or Pipecat. Only the example's composition roots, `factory.py` and `pipeline.py`, import it, and it keeps its records in memory.

## How Callers Reach the Core

The core has no HTTP routes of its own. A caller reaches it in process through `LocalResponsesClient` ([`api/local.py`](api/local.py)), which serves the official Responses API endpoints as Python methods. Every request, event, and refusal passes through JSON on the way, as it would over HTTP. The following table lists the endpoints:

| Endpoint | What the core does |
| --- | --- |
| `POST /v1/conversations` | Opens a conversation, served by an interaction manager (IM) session of its own. |
| `POST /v1/conversations/{id}` | Attaches a later call to a kept conversation, with the call's metadata. |
| `POST /v1/responses` | Admits the request and streams its one response, or refuses it before any event. |
| `POST /v1/responses/input_tokens` | Checks a request's settings without running it, and counts its input. |
| `GET /v1/conversations/{id}/items` | Lists the conversation's backend history. The core sets the `id` of each message in a request's input: a request's messages take its `live_delegation_id`, then that ID with `.2`, `.3`, and so on. A request that names no delegation, and typed messages that come with function outputs, take an ID that the core makes instead. |
| `GET /v1/models/{model}` | Returns the model object when the backend serves `model`, and refuses any other model. |
| `DELETE /v1/conversations/{id}` | Ends the conversation's IM session, which stops its work and ends every open response. A conversation that keeps work after its call answers `deleted: false` instead. |

The core reads three `metadata` keys: `live_delegation_id` names the delegation, `live_typed_run: "true"` marks a typed run, and `live_store: "true"` on a conversation lets its work outlive a call. A request that names its delegation, brings no function outputs, and is not a typed run is a delegation, and any other request is the client's own. A request with no `conversation` is stateless: a one-shot IM session serves it and keeps nothing.

A conversation with `live_store: "true"` offers its tasks the `continue_after_call` work tool on a backend above `max_running: 1`. When its call ends while its records list work that a task marked with this tool, open or among the newest ended, `DELETE` keeps the conversation. Such work that is still open goes on, and a later call attaches with `POST /v1/conversations/{id}`. The gateway guide's [Work That Outlives the Call](../../../../docs/how-to/use-live-gateway.md#work-that-outlives-the-call) describes the flow.

In the [Frontend/Backend Live example](../README.md), the composition root builds one core per process, on first use, for the gateway. Each Live session in `responses` mode reaches it through the gateway's [server delegation](../../../../docs/how-to/use-live-gateway.md#server-delegation), in a conversation of its own. A client-mode app can send each backend round to `POST /v1/live/delegate`, which reaches the core as one stateless request. The repository UI bot builds a core of its own for each bot session, from the core's settings with `ui_bot.core` merged over them, which set the cafe's `backend.prompt` and `backend.tools`.

The in-process client exposes `set_attachment(conversation_id, attached)` for [WebSocket reconnect](../../../../docs/how-to/use-live-gateway.md#resume-a-dropped-websocket-call). It fences caller-owned actions while the audio attachment is absent and checks which app calls remain pending.

## Build and Configure the Core

`build_core(settings, *, endpoints, defaults)` in [`app.py`](app.py) builds one `DelegationCore`, choosing every implementation by name. The example's composition root, [`factory.py`](../factory.py), reads the settings and the core's texts with `CoreSettings.parse(settings, texts)`. It resolves the default model of `backend.slot` from the service catalog and the instructions of `backend.prompt` from the example's `prompts.yaml`. It hands the core to the gateway through `LocalResponsesClient`.

At final shutdown, `await core.aclose()` starts every distinct backend source close and waits for all of them. It then propagates the first original error; native sources join only their own cleanup.

The core's settings are the `core` block of the example's settings, in the `config` block of its entry in [`examples_registry.yaml`](../../../../examples_registry.yaml), as the example's [Configuration](../README.md#configuration) describes. The server reads them once, so an edit takes effect at the next restart. The block must set `backend.slot` and `backend.prompt`. Every other key that it leaves out keeps its default, and an unknown key stops the load. The block has the following parts:

- `delegation`: the IM's settings, with their defaults in `DelegationSettings` ([`config.py`](config.py)).
- `backend`: the adapter (`adapter`) and the one that serves stateless requests (`one_shot_adapter`), both `llm` by default, and the default model's slot and prompt (`slot` and `prompt`). It also sets the server tool sources (`tools`) and the steps that the app's outputs take before the backend reads them (`app_output_steps`, none by default). It also has each adapter's own block, whose defaults are in the adapter's `settings.py`. The registry block shows, commented out, how to turn on the ID-shape hints step (`app_output_steps.shape_hints` with its `not_found_pattern`). The hint's words are the `backend.app_output_steps.shape_hints.hint` entry of `agent_prompts.Core`.

The core records each app result under its original `call_id`. If an output step fails, drops a result, or changes a call ID, the core skips that step and continues with the last valid outputs.

The core's texts are the lines that the IM and the backend write or add to a model's input. They are the `agent_prompts.Core` entries of the example's [`prompts.yaml`](../prompts.yaml), which the UI's **Prompts** tab shows. Each entry is named by its place in the settings. `texts.<key>` names the core's own lines, such as `texts.conductor_prompt`. `backend.<block>.<key>` names the texts of a backend block, such as the harness's prompt templates (`backend.harness.prompts`). A block's texts join it only when the settings have the block, so you can remove an app output step without editing `prompts.yaml`. Settings that set one of these places, such as `backend.harness.prompts.user`, stop the load with a message that names `agent_prompts.Core`.

The following table lists the `delegation` settings that you are most likely to change. `DelegationSettings` comments every key:

| Setting | Default | What it sets |
| --- | --- | --- |
| `max_pending_delegations` | `16` | The most requests admitted and not yet running. The next one is refused. |
| `client_tool_timeout_seconds` | `60` | How long the app's function outputs may take, counted from the end of the response with their calls. `null` waits until the session closes. |
| `work_timeout_seconds` | `600` | The backstop on one Work. `null` turns it off. Work that outlives the call has none. |
| `after_call_seconds` | `172800` | How long a conversation kept for work that outlives its call stays after its last call, and again after that work ends. When that time runs out, it closes, and work still running stops. |
| `max_after_call_conversations` | `8` | The most conversations at once whose work still runs after their call. A kept conversation whose work has ended does not count: it holds only memory until its keep time ends. |
| `result_format` | `plain` | `plain`, or `speak_display`, which adds a `show` item for the screen before the answer to speak. |
| `progress_note_seconds` | `1.0` | The shortest time between two progress notes of one delegation. `0` writes none. |
| `context_tokens` | `24000` | The token budget for backend history and private task state. The core drops the oldest requests as whole units to make room. |

Every task of a conversation gets the `cancel_work`, `set_work_state`, `caller_message`, and `resolve_questions` work tools. It also gets `show` under `speak_display`, `continue_after_call` as described above, and the caller and voice notes (`texts.caller_note` and `texts.voice_note`). An app function with one of these names is refused together with the work tool. A server tool source must not use them, because its tool would replace the work tool. A one-shot session offers no work tool.

Use `caller_message(kind: question|status)` to publish a question or status while work continues. Each question has a stable ID. The worker calls `resolve_questions` with the IDs it wants to resolve, using an applied caller input reference for `answered` or the `no_longer_needed` disposition. An exact continuation inherits unresolved questions. Unrelated input and dropped publications leave the question open.

The frontend tracks playback. A published message shows its wording and source. Caller hearing and business authorization have separate evidence.

## Packages

The following table lists the core's packages:

| Package | Owns |
| --- | --- |
| [`api/`](api/) | The Responses API endpoints (`ResponsesService`) and the in-process client (`LocalResponsesClient`). |
| [`im/`](im/) | The interaction manager: admission, placement, Works and their tasks, steering, stops, progress notes, results, one-shot sessions, and the close. Only the IM writes a response. |
| [`ledger/`](ledger/) | The work ledger: the IM's records and the moves that each record allows. |
| [`backend/`](backend/) | The backend port ([`port.py`](backend/port.py)), the adapter table, the conductors, the model providers, the server tool sources, the app output steps, and the three backend adapters. |

## Backend Adapters

Use [Choose a Backend Adapter](../README.md#choose-a-backend-adapter) for backend configuration and [Connect an Agent Harness](../../../../docs/how-to/connect-an-agent-harness.md) for ACP agent setup.

### Complete Routing Decisions

The model conductor submits one `submit_decision` with all actions for its supplied inputs. The following actions can appear together:

| Action | Purpose |
| --- | --- |
| `start` | Assign a focused task brief. Optional `continues` selects an eligible ended task, and `verifies` selects a separate read-only check. |
| `steer` | Update exact tasks, including eligible continuations. Provide a scoped update for each task when steering multiple tasks or sharing the input with another action. |
| `stop` | Fence new effects and request an immediate stop for the targets. |
| `pause`, `resume` | Retain the task and set its desired state. Resumed work requires settled activity and available capacity. |
| `caller_message` | Ask a routing question or select work for a status update. |

Core generates status updates from the selected tasks' records. The worker that owns a task reports its business results from tool receipts. For a routing question, the conductor supplies the question text.

A caller can say “Pause dinner and check the late showing.” One decision can pause the dinner task and update the movie task. The conductor selects caller lines by source ID and sequence, and Core copies the relevant words from the transcript. Include shared details in each relevant brief. Keep dependent steps in one task.

Core checks exact targets and coverage of the bound input IDs. A decision rejected before any action starts gets one repair attempt. An LLM Agent stream that ends without a complete decision, or a harness disconnect before submission, can use the same repair. If the repair also fails before any action starts, Core can ask the caller to clarify. An invalid target or insufficient worker capacity can produce a notice for the affected request. Accepted actions keep their recorded outcomes.

Harness job previews use caller text, with playback recorded separately.

### Stops and Pending Input

Core can hold tool calls that could change state or end the interaction while the frontend routes a completed turn or Core places a newer request. If a routing action remains unresolved, even within a partially accepted decision, Core can mark affected pending calls and late proposals as `not_run`. The restriction clears for a task after it receives applicable caller input at least as recent as the unresolved request.

A stop fences new effects immediately. Started business actions can still settle, and their real outputs remain recorded. A pause preserves the task and its questions while activity settles. A verification task receives retained evidence and only authorized read-only tools.

### Shared Responses and Work Ownership

Several tasks can contribute to one response with independent histories and separate input, tool, and result ownership. An update moves only the selected task's obligations. For the client-visible tool batch and late result rules, refer to [Continue a Shared Response](../../../../docs/how-to/use-live-gateway.md#continue-a-shared-response).

### Usage Observations

Private `backend.provider_attempt.ended` records cover direct model reader attempts, including failed and incomplete attempts. Harness and conductor runs record their own `harness.round.usage` and `conductor.round.usage`. Use correlation IDs to join attempts to tasks and retain the latest reported observation per native invocation. Core round summaries can mirror the adapter observations.

Count only reported input, output, reasoning, and cache tokens. Missing usage stays unknown. A reader attempt can include provider opening retries, and a native counter snapshot is per-invocation usage only if the runtime supplies that scope.

## Extend the Core

Each part of the core that you can add is one class and one entry in a name table. The core's settings, or a model's `services.yaml` entry, choose each part by name. The following table lists where to add each part. The module of each table describes what its classes implement:

| Part | Table | Chosen by |
| --- | --- | --- |
| Backend adapter | `ADAPTERS` in [`backend/adapters.py`](backend/adapters.py), with the backend port in [`backend/port.py`](backend/port.py) | `backend.adapter` |
| Backend model provider | `BACKENDS` in [`backend/providers/__init__.py`](backend/providers/__init__.py) | `api` in the model's `services.yaml` entry, `chat-completions` when the entry has none |
| Server tool source | `TOOL_SOURCES` in [`backend/tools/__init__.py`](backend/tools/__init__.py) | `backend.tools` |
| App output step | `APP_OUTPUT_STEPS` in [`backend/output_steps.py`](backend/output_steps.py) | `backend.app_output_steps` |
| Harness protocol | `PROTOCOLS` in [`backend/harness/protocol.py`](backend/harness/protocol.py) | `backend.harness.protocol` |
| ACP agent class | `AGENTS` in [`backend/harness/acp/agents.py`](backend/harness/acp/agents.py) | `backend.harness.agent` with `protocol: acp` |

Use the [backend port](backend/port.py) for task input, tool admission, and results. Keep harness-specific behavior in the matching agent class. The Core owns task routing and records.
