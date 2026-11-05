# Delegation Core

The delegation core runs the work that a voice frontend delegates. It accepts each request through the OpenAI
Responses API, runs it on the backend adapter that its settings name, and streams the response's events back. By
default, the backend adapter is the backend large language model (LLM), directly connected. The core is a separate
Python package in the example's folder: it imports nothing from the Live gateway
([`src/gateways/live/`](../../../gateways/live/)), the example's frontend adapter, or Pipecat. Only the example's
composition roots import it: they build it in process, and they hand the frontend adapter the helpers that the two
share, the one copy of `pick` ([`tables.py`](tables.py)), the profile loader ([`profiles.py`](profiles.py)), and
`message_text` ([`items.py`](items.py)). It has its own settings file (`core.yaml`), tests, and README, and it keeps
its records in memory. The core, the gateway, and the frontend adapter share one copy of the ids, the token estimate,
the delegation contract's constants, and the model endpoint type. These helpers are in modules of
[`src/examples/shared/`](../../shared/) that import only the standard library.

For the design, including the interaction manager (IM), the backend port, the three backend adapters, and how to extend
each part, refer to the [Frontend/Backend Live Architecture](../../../../docs/design/architecture.md) page.

## How Callers Reach the Core

The core has no HTTP routes of its own. A caller reaches it in process through `LocalResponsesClient`
([`api/local.py`](api/local.py)), which serves the official Responses API endpoints as Python methods. Every request,
event, and refusal passes through JSON on the way, as it would over HTTP. The following table lists the endpoints:

| Endpoint | What the core does |
| --- | --- |
| `POST /v1/conversations` | Opens a conversation, served by an IM session of its own. |
| `POST /v1/responses` | Admits the request and streams its one response, or refuses it before any event. |
| `POST /v1/responses/input_tokens` | Checks a request's settings without running it, and counts its input. |
| `GET /v1/conversations/{id}/items` | Lists the conversation's backend history. |
| `GET /v1/models/{model}` | Returns the model object when the backend serves `model`, and refuses any other model. |
| `DELETE /v1/conversations/{id}` | Ends the conversation's IM session, which stops its work and ends every open response. |

The core reads two `metadata` keys: `live_delegation_id` names the delegation, and `live_typed_run: "true"` marks a
typed run. A request that names its delegation, brings no function outputs, and is not a typed run is a delegation, and
any other request is the client's own. A request with no `conversation` is stateless: a one-shot IM session serves it
and keeps nothing.

In the [Frontend/Backend Live example](../README.md), the composition root builds one
core per process, on first use, for the gateway. Each Live session in `responses` mode reaches it through the
gateway's [server delegation](../../../../docs/how-to/use-live-gateway.md#server-delegation), in a conversation of its
own. A client-mode app can send each backend round to `POST /v1/live/delegate`, which reaches the core as one
stateless request. The repository UI bot builds a core of its own for each bot session, from `core.ui-bot.yaml`,
which sets the cafe's `backend.prompt` and `backend.tools`.

## Build and Configure the Core

`build_core(settings, *, endpoints, defaults)` in [`app.py`](app.py) builds one `DelegationCore`, choosing every
implementation by name. The example's composition root, [`factory.py`](../factory.py),
loads the settings and the core's texts with `CoreSettings.load`, resolves the default model of `backend.slot` and the
instructions of `backend.prompt` from the example's `prompts.yaml`, and hands the core to the gateway through
`LocalResponsesClient`.

The core's settings are their own file, [`core.yaml`](../core.yaml) in the example's
folder. Set `CORE_CONFIG_PATH` to use another file, such as one of the profiles that
[Configuration](../../../../docs/design/architecture.md#configuration) lists. The root reads the file once, on first
use, so an edit takes effect at the next restart. A profile names the file it builds on with `extends` and sets only
what differs, and an unknown key stops the load. The file has the following blocks:

- `delegation`: the IM's settings, with their defaults in `DelegationSettings` ([`config.py`](config.py)).
- `backend`: the adapter (`adapter`) and the one that serves stateless requests (`one_shot_adapter`), both `llm` by
  default, the default model's slot and prompt (`slot` and `prompt`), the server tool sources (`tools`), the steps
  that the app's outputs take before the backend reads them (`app_output_steps`, the ID-shape hints by default), and
  each adapter's own block.

The core's texts, the lines that the IM and the backend write or add to a model's input, are the `agent_prompts.Core`
entries of the example's [`prompts.yaml`](../prompts.yaml). The root reads them with the file and hands them to
`CoreSettings.load`, and the UI's **Prompts** tab shows them. Each entry is named by its place in the settings:
`texts.<key>` for the core's own lines, such as `texts.conductor_prompt`, and `backend.<block>.<key>` for the texts of
a backend block, such as the harness's prompt templates (`backend.harness.prompts`). A block's texts join it only when
the settings have the block, so you can remove an app output step without editing `prompts.yaml`. A core file that
sets one of these places, such as `backend.harness.prompts.user`, stops the load with a message that names
`agent_prompts.Core`.

The following table lists the `delegation` settings that you are most likely to change. `core.yaml` comments every key:

| Setting | Default | What it sets |
| --- | --- | --- |
| `max_pending_delegations` | `16` | The most requests admitted and not yet running. The next one is refused. |
| `client_tool_timeout_seconds` | `60` | How long a task waits for the app's function outputs. `null` waits until the session closes. |
| `work_timeout_seconds` | `600` | The backstop on one Work. `null` turns it off. |
| `result_format` | `plain` | `plain`, or `speak_display`, which adds a `show` item for the screen before the answer to speak. |
| `progress_note_seconds` | `1.0` | The shortest time between two progress notes of one delegation. `0` writes none. |
| `context_tokens` | `24000` | The backend history's budget. Over it, the oldest requests leave the history, each one whole. |

## Packages

The following table lists the core's packages:

| Package | Owns |
| --- | --- |
| [`api/`](api/) | The Responses API endpoints (`ResponsesService`) and the in-process client (`LocalResponsesClient`). |
| [`im/`](im/) | The interaction manager: admission, placement, Works and their tasks, steering, stops, progress notes, results, one-shot sessions, and the close. Only the IM writes a response. |
| [`ledger/`](ledger/) | The work ledger: the IM's records and the moves that each record allows. |
| [`backend/`](backend/) | The backend port ([`port.py`](backend/port.py)), the adapter table, the conductors, the model providers, the server tool sources, the app output steps, and the three backend adapters. |

## Backend Adapters

`backend.adapter` chooses one adapter from the table in [`backend/adapters.py`](backend/adapters.py), which loads an
adapter's package only when a core is built with it. The following table lists the adapters:

| Adapter | Package | What it runs | Profile |
| --- | --- | --- | --- |
| `llm` (default) | [`backend/llm/`](backend/llm/) | The backend LLM, directly connected, with a rule conductor. It also serves stateless requests. | `core.yaml` |
| `llm-agent` | [`backend/agent/`](backend/agent/) | A model conductor that makes one model call per decision, with backend LLM runs as its workers. While no Work is open, a request starts one with no model call (`conductor.when_idle: start`). With `max_running: 1`, the `steer` rule places each request instead. | `core.agent.yaml` |
| `harness` | [`backend/harness/`](backend/harness/) | An external agent that its operator deploys, such as Hermes, over the Agent Client Protocol (ACP). | `core.harness.yaml` |

To deploy Hermes or connect another ACP agent, refer to
[Connect an Agent Harness](../../../../docs/how-to/connect-an-agent-harness.md). To add a backend adapter,
a model provider, or a tool source, refer to
[Extension Recipes](../../../../docs/design/architecture.md#extension-recipes).

## Tests

The core's tests in `tests/unit/delegation_core/` and the port contract suites in `tests/unit/contracts/` run offline
with scripted backends. Run them from the repository root:

```bash
uv run pytest tests/unit/delegation_core tests/unit/contracts -v
```

For the other test commands and the import rule that keeps the core apart from the gateway and the frontend adapter,
refer to [Tests](../../../../docs/design/architecture.md#tests) and
[The Core's Import Rule](../../../../docs/design/architecture.md#the-cores-import-rule).
