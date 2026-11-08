# Delegation Core

The delegation core runs the work that a voice frontend delegates. It accepts each request through the OpenAI Responses API, runs it on the backend adapter that its settings name, and streams the response's events back. By default, the backend adapter is the backend large language model (LLM), directly connected. The core is a separate Python package in the example's folder. It imports nothing from the Live gateway ([`src/gateways/live/`](../../../gateways/live/)), the example's frontend adapter, or Pipecat. Only the example's composition roots, `factory.py` and `pipeline.py`, import it, and it keeps its records in memory.

## How Callers Reach the Core

The core has no HTTP routes of its own. A caller reaches it in process through `LocalResponsesClient` ([`api/local.py`](api/local.py)), which serves the official Responses API endpoints as Python methods. Every request, event, and refusal passes through JSON on the way, as it would over HTTP. The following table lists the endpoints:

| Endpoint | What the core does |
| --- | --- |
| `POST /v1/conversations` | Opens a conversation, served by an interaction manager (IM) session of its own. |
| `POST /v1/responses` | Admits the request and streams its one response, or refuses it before any event. |
| `POST /v1/responses/input_tokens` | Checks a request's settings without running it, and counts its input. |
| `GET /v1/conversations/{id}/items` | Lists the conversation's backend history. |
| `GET /v1/models/{model}` | Returns the model object when the backend serves `model`, and refuses any other model. |
| `DELETE /v1/conversations/{id}` | Ends the conversation's IM session, which stops its work and ends every open response. |

The core reads three `metadata` keys: `live_delegation_id` names the delegation, `live_typed_run: "true"` marks a typed run, and `live_reply_unheard` states what of the backend's answers the caller had not heard when they spoke: `none`, `answer` (part of an answer, with no question in that part), or `question` (an unheard sentence asks). The example's frontend states `live_reply_unheard` on each delegation, and the [write gate](#the-write-gate) reads it. A request that states no value, or a value outside these three, states no fact. A request that names its delegation, brings no function outputs, and is not a typed run is a delegation, and any other request is the client's own. A request with no `conversation` is stateless: a one-shot IM session serves it and keeps nothing.

In the [Frontend/Backend Live example](../README.md), the composition root builds one core per process, on first use, for the gateway. Each Live session in `responses` mode reaches it through the gateway's [server delegation](../../../../docs/how-to/use-live-gateway.md#server-delegation), in a conversation of its own. A client-mode app can send each backend round to `POST /v1/live/delegate`, which reaches the core as one stateless request. The repository UI bot builds a core of its own for each bot session, from the core's settings with `ui_bot.core` merged over them, which set the cafe's `backend.prompt` and `backend.tools`.

## Build and Configure the Core

`build_core(settings, *, endpoints, defaults)` in [`app.py`](app.py) builds one `DelegationCore`, choosing every implementation by name. The example's composition root, [`factory.py`](../factory.py), reads the settings and the core's texts with `CoreSettings.parse(settings, texts)`. It resolves the default model of `backend.slot` from the service catalog and the instructions of `backend.prompt` from the example's `prompts.yaml`. It hands the core to the gateway through `LocalResponsesClient`.

The core's settings are the `core` block of the example's settings, in the `config` block of its entry in [`examples_registry.yaml`](../../../../examples_registry.yaml), as the example's [Configuration](../README.md#configuration) describes. The server reads them once, so an edit takes effect at the next restart. The block must set `backend.slot` and `backend.prompt`. Every other key that it leaves out keeps its default, and an unknown key stops the load. The block has the following parts:

- `delegation`: the IM's settings, with their defaults in `DelegationSettings` ([`config.py`](config.py)).
- `backend`: the adapter (`adapter`) and the one that serves stateless requests (`one_shot_adapter`), both `llm` by default, and the default model's slot and prompt (`slot` and `prompt`). It also sets the server tool sources (`tools`) and the steps that the app's outputs take before the backend reads them (`app_output_steps`, none by default). It also has each adapter's own block, whose defaults are in the adapter's `settings.py`. The registry block shows, commented out, how to turn on the ID-shape hints step (`app_output_steps.shape_hints` with its `not_found_pattern`), which stays off by default. The step suggests each other form of an ID once. With `stop_when_futile` (`true`), after a form that it suggested also finds nothing, the step gives no more hints for that field and adds one line that asks the backend to read the value back to the caller. The texts are the `backend.app_output_steps.shape_hints.hint` and `backend.app_output_steps.shape_hints.read_back` entries of `agent_prompts.Core`. Without `read_back`, no line is added.

The core's texts are the lines that the IM and the backend write or add to a model's input. They are the `agent_prompts.Core` entries of the example's [`prompts.yaml`](../prompts.yaml), which the UI's **Prompts** tab shows. Each entry is named by its place in the settings. `texts.<key>` names the core's own lines, such as `texts.conductor_prompt`. `backend.<block>.<key>` names the texts of a backend block, such as the harness's prompt templates (`backend.harness.prompts`). A block's texts join it only when the settings have the block, so you can remove an app output step without editing `prompts.yaml`. Settings that set one of these places, such as `backend.harness.prompts.user`, stop the load with a message that names `agent_prompts.Core`.

The following table lists the `delegation` settings that you are most likely to change. `DelegationSettings` comments every key:

| Setting | Default | What it sets |
| --- | --- | --- |
| `max_pending_delegations` | `16` | The most requests admitted and not yet running. The next one is refused. |
| `client_tool_timeout_seconds` | `60` | How long the app's function outputs may take, counted from the end of the response with their calls. `null` waits until the session closes. |
| `work_timeout_seconds` | `600` | The backstop on one Work. `null` turns it off. |
| `result_format` | `plain` | `plain`, or `speak_display`, which adds a `show` item for the screen before the answer to speak. |
| `progress_note_seconds` | `1.0` | The shortest time between two progress notes of one delegation. `0` writes none. |
| `context_tokens` | `24000` | The backend history's budget. Over it, the oldest requests leave the history, each one whole. |
| `questions.shown` | `3` | The backend's newest questions that the questions block lists. `0` turns the block off. |

A request that carries the work records also carries the questions block after them (`texts.questions_block`). It
lists the backend's newest questions to the caller in this conversation, newest last. Each question shows how many of
the backend's replies asked it (`asked`) and, when the frontend stated it, how many times the caller had heard it to the
end when they next spoke (`heard_in_full`). A prompt file without `texts.questions_block` loads with a
`prompts.missing_text` warning, and no block is shown.

Every task of a conversation gets the `cancel_work` work tool and the caller and voice notes (`texts.caller_note` and `texts.voice_note`). A one-shot session offers no work tool.

## The Write Gate

The write gate keeps the backend from changing the app's data on a caller's words that answered something else. It is
off by default, and `delegation.write_gate.enabled: true` turns it on. While it is on, the IM holds an app function
call that can change data, and answers it with a not-run output instead of sending it to the app, when one of the
following is true:

- The caller had not heard the backend's answers in full when they last spoke, as the request's `live_reply_unheard`
  states. `write_gate.unheard` (experimental) sets the lowest level that holds a call: `answer` (the default) or
  `question`.
- The backend asked a question after the caller's request was admitted.
- A question asked earlier in the call was never heard. The gate holds a call once for each such question, and the
  not-run output names it (`texts.unheard_question_output`).

A request that states no `live_reply_unheard` is never held, so a frontend or a delegation port without the key keeps
working as before. The gate reads what each tool does from its MCP annotations, then from the operator's lists:

- An app declares a function tool's kind with an MCP `annotations` object on the tool, such as
  `{"readOnlyHint": true}`. `readOnlyHint: true` is a read, which the gate never holds. `readOnlyHint: false` or
  `destructiveHint: true` is a write. A write with `destructiveHint: false` is reversible, and `write_gate.reversible`
  (`run`) lets it run. The core reads the annotations and strips them before every model call.
- `delegation.tool_kinds.read_tools` and `delegation.tool_kinds.write_tools` list tool names or `fnmatch` patterns for
  tools that carry no annotations. A tool's own annotations win, and a name must not be in both lists.
- A tool that neither declares is undeclared. With `write_gate.undeclared: hold`, the default, the gate holds it at most
  once per caller request for each tool, and it runs when the backend calls it again. Declare your read-only tools,
  with annotations or `read_tools`, to spare them that extra backend round.

When the backend calls a held declared write again for the same request, the call is not run either, and the IM closes
the task's tools: its next round offers no tool and must answer (port rule E9 in
[`backend/port.py`](backend/port.py)). A newer request reopens them. The not-run outputs are the
`texts.unheard_write_output`, `texts.unheard_undeclared_output`, and `texts.unheard_question_output` entries of
`agent_prompts.Core`. A prompt file that lacks one still loads with a `prompts.missing_text` warning: without
`texts.unheard_write_output`, the gate stays off; without the other two, undeclared tools run and the open-question
reason is off. The registry's `core` block shows the gate's keys, commented out. With the harness adapter,
`backend.harness.bridge.tool_annotations: true` also lists the app's functions to the agent with their annotations.
It is off by default, since an agent may act on them, for example by asking for permission.

## Packages

The following table lists the core's packages:

| Package | Owns |
| --- | --- |
| [`api/`](api/) | The Responses API endpoints (`ResponsesService`) and the in-process client (`LocalResponsesClient`). |
| [`im/`](im/) | The interaction manager: admission, placement, Works and their tasks, steering, stops, progress notes, results, one-shot sessions, and the close. Only the IM writes a response. |
| [`ledger/`](ledger/) | The work ledger: the IM's records and the moves that each record allows. |
| [`backend/`](backend/) | The backend port ([`port.py`](backend/port.py)), the adapter table, the conductors, the model providers, the server tool sources, the app output steps, and the three backend adapters. |

## Backend Adapters

`backend.adapter` chooses one adapter from the table in [`backend/adapters.py`](backend/adapters.py), which loads an adapter's package only when a core is built with it. The following table lists the adapters:

| Adapter | Package | What it runs |
| --- | --- | --- |
| `llm` (default) | [`backend/llm/`](backend/llm/) | The backend LLM, directly connected. The steer rule places each request: it joins the newest open Work, or starts one. It also serves stateless requests. |
| `llm-agent` | [`backend/agent/`](backend/agent/) | Backend LLM runs as workers. Above `max_running: 1`, a model conductor places every request, with one model call per decision. At `max_running: 1`, the steer rule places each request. |
| `harness` | [`backend/harness/`](backend/harness/) | An external agent that its operator deploys, such as Hermes, over the Agent Client Protocol (ACP). Above `max_running: 1`, a conductor session of the agent places every request. At `max_running: 1`, the steer rule places each request. |

To choose an adapter, refer to [Choose a Backend Adapter](../README.md#choose-a-backend-adapter). To deploy Hermes or connect another ACP agent, refer to [Connect an Agent Harness](../../../../docs/how-to/connect-an-agent-harness.md). To add a backend adapter, a model provider, or a tool source, refer to [Extend the Core](#extend-the-core).

## Extend the Core

Each part of the core that you can add is one class and one entry in a name table. The core's settings, or a model's `services.yaml` entry, choose each part by name. The following table lists where to add each part. The module of each table describes what its classes implement:

| Part | Table | Chosen by |
| --- | --- | --- |
| Backend adapter | `ADAPTERS` in [`backend/adapters.py`](backend/adapters.py), with the backend port in [`backend/port.py`](backend/port.py) | `backend.adapter` |
| Backend model provider | `BACKENDS` in [`backend/providers/__init__.py`](backend/providers/__init__.py) | `api` in the model's `services.yaml` entry, `chat-completions` when the entry has none |
| Server tool source | `TOOL_SOURCES` in [`backend/tools/__init__.py`](backend/tools/__init__.py) | `backend.tools` |
| App output step | `APP_OUTPUT_STEPS` in [`backend/output_steps.py`](backend/output_steps.py) | `backend.app_output_steps` |
| Harness agent class | `AGENTS` in [`backend/harness/acp/agents.py`](backend/harness/acp/agents.py) | `backend.harness.agent` |
