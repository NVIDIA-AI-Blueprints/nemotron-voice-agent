# Delegation Core

The delegation core runs backend work for the [Frontend/Backend Live example](../README.md). Its interaction manager (IM) assigns caller input, tracks tasks and tool results, and streams Responses events. The frontend owns the voice pipeline and records what played. The core keeps conversation and execution records in memory.

Use [`LocalResponsesClient`](api/local.py) to call the core in process. Include a conversation ID to keep task state across requests. Omit it to run a stateless request. For network clients, refer to [Use the Live Gateway](../../../../../../docs/how-to/use-live-gateway.md).

## Build and Configure the Core

[`build_core`](app.py) takes the core settings, model endpoint resolver, and backend defaults. The Live gateway uses one core per process and a conversation per session in Responses mode. The browser UI bot builds its own core for each session using the `ui_bot.core` overrides.

Edit the `core` block of the `pipecat.examples.frontend-backend-live` entry in [`examples_registry.yaml`](../../../../../../examples_registry.yaml), then restart the server. [`CoreSettings`](config.py) defines the available keys and defaults. Backend instructions and Core text live under `agent_prompts.Backend` and `agent_prompts.Core` in [`prompts.yaml`](../prompts.yaml). For setup, refer to [Choose a Backend Adapter](../README.md#choose-a-backend-adapter) and [Connect an Agent Harness](../../../../../../docs/how-to/connect-an-agent-harness.md).

Core reserves the work-tool names `cancel_work`, `set_work_state`, `caller_message`, `resolve_questions`, `show`, and `continue_after_call`. Choose different names for app and server functions. The result format controls availability of `show`, and the conversation and adapter settings control availability of `continue_after_call`.

## Complete Routing Decisions

The single-worker rule places new input on the newest open work or an eligible continuation. With more than one worker slot, the `llm-agent` and `harness` adapters use a model conductor. The conductor sends one `submit_decision` covering its assigned inputs. One decision can combine the following actions.

| Action | Purpose |
| --- | --- |
| `start` | Start one or more focused tasks. A task can continue prior work or verify a recorded outcome with approved read-only tools. |
| `steer` | Apply a caller update to the selected tasks or continue an eligible task. |
| `stop` | Block new effects and request a stop for selected work. Use `scope: all` to include all eligible work and earlier queued requests. |
| `pause`, `resume` | Change the state of a task while keeping its history and open questions. |
| `caller_message` | Ask a routing question or request a status update from the selected task records. |

A routing question asks only what the conductor needs to choose an action or target. The frontend presents it using the session's speaking instructions. Workers ask for the details they need to carry out their assigned tasks.

If the caller says “Pause dinner and check the late showing,” one decision can pause dinner and update the movie task. Core keeps the caller's transcript in the conversation and gives each worker its task brief as routing context. When one line covers several jobs, the conductor supplies each worker's excerpt, and Core validates the line's message ID and sequence. Shared facts go in each relevant task brief or update.

Core checks input coverage, exact targets, and worker capacity before admitting work. Core can request one repair for a rejected decision when no action ran. Before the conductor submits a decision, `llm-agent` can use the same repair if an open stream fails, a stream times out, or the provider reports a transient failure. Core preserves the recorded outcomes of accepted actions. A task reports business results using its own tool receipts.

## Tool Calls and Delivery

- Core can hold a new app call while the frontend routes a completed caller turn or Core places newer input. When Core cannot confirm delivery of caller input, it blocks new app calls for that task and its continuation. Calls can resume after the worker applies that input or a newer one.
- After a stop, Core blocks new effects and records the outcomes of calls already in flight. Resuming paused work requires available capacity and settled activity.
- Workers use `caller_message` for questions or updates while work continues. A question has a stable ID. To resolve it as answered, the worker supplies a newer applied caller input reference through `resolve_questions`. The frontend tracks playback separately.
- Each task keeps its own history, input, and tool ownership even when several tasks contribute to one response. Refer to the [gateway guide](../../../../../../docs/how-to/use-live-gateway.md) for returning function results and using reconnect.

## Usage Observations

For direct model calls, `backend.provider_attempt.ended` records usage per logical provider attempt, including failed or incomplete attempts. For native calls, use the latest `harness.invocation.usage` observation per invocation ID. Worker and conductor summaries can repeat those observations.

Count reported counters for input, output, reasoning, and cache usage. Missing counters stay unknown. A cumulative native snapshot gives per-invocation usage only when the runtime identifies a comparable counter scope.

## Extend the Core

Use the [backend port](backend/port.py) for worker input, tool admission, and results. The IM owns routing and task records. Each adapter owns model calls and worker sessions. Keep Agent Client Protocol (ACP) behavior specific to an agent in its agent class.

| Component | Registry |
| --- | --- |
| Backend adapter | `ADAPTERS` in [`backend/adapters.py`](backend/adapters.py) |
| Model provider | `BACKENDS` in [`backend/providers/__init__.py`](backend/providers/__init__.py) |
| Server tool source | `TOOL_SOURCES` in [`backend/tools/__init__.py`](backend/tools/__init__.py) |
| App output step | `APP_OUTPUT_STEPS` in [`backend/output_steps.py`](backend/output_steps.py) |
| Harness protocol | `PROTOCOLS` in [`backend/harness/protocol.py`](backend/harness/protocol.py) |
| ACP agent | `AGENTS` in [`backend/harness/acp/agents.py`](backend/harness/acp/agents.py) |
