# Frontend/Backend Live Architecture

This page describes how the [Frontend/Backend Live example](../../src/examples/frontend_backend_live/README.md) is
built: the four boxes, the contract between the frontend media plane and the interaction manager, how the interaction
manager handles work, the backend port and its three backend adapters, the extension recipes, the configuration files,
the deployment, and the rules that the tests enforce. It is written for engineers who change or
extend the example. To run and use the example, refer to its README.

The live session protocol follows GPT-Live. Every event and item inside the pipeline uses GPT-Live's schema and the
official Responses API, with no custom events or fields. In client delegation mode, the app runs the delegated work
itself. The shared Nemotron Voice Agent pipeline (`src/examples/shared/`: voice activity detection, automatic speech
recognition (ASR), text-to-speech (TTS), and the turn models) is outside this design.

## Terms

The following terms have a fixed meaning on this page and in the code:

| Term | Meaning |
| --- | --- |
| The app | The client application on the Live wire: the Live Console, a benchmark client, or your own app. It declares function tools and runs the calls that the backend routes to it. |
| Client mode, Responses mode | The resolved `delegation.type` of a session. In client mode, the gateway only announces each delegation, and the app does the work. An omitted, `null`, or `{}` delegation is client mode. In Responses mode, the delegation core does the work. |
| Backend LLM | The backend large language model (LLM), directly connected: the default backend adapter, `llm`. |
| Talker | The frontend model call that decides each caller turn: speak a reply, delegate, or wait. |
| Writer | The frontend model call that words ready results into one spoken line (`result_wording: writer`). `as_written` says them unchanged. |
| Delegation, request | One `delegate()` call, sent as one `POST /v1/responses`, with its responses. The request is what it asks for. |
| Work | One piece of backend work. One request, or several joined requests, is one Work. A Work holds one or more tasks. |
| Task, worker | A task is one worker: one run of the backend (`BackendRun`). |
| Conductor | The part of a backend adapter that places new requests: start a Work, steer one, cancel one, or answer a status question. It decides, and the interaction manager (IM) carries each decision out. |
| Steering | A newer request that joins a Work that already started. It lands at the backend's next safe point. |
| Commit gate | While newer input waits to land, or waits for a model conductor's decision, no new call of the work it can concern is dispatched. |
| Carrier, taker | The carrier is the delegation whose open response carries a Work's output. When a join lands, the newer delegation, the taker, becomes the carrier. |
| Envelope | The speak and display shape of a result: a `show` item for the screen, then the `final_answer` message to speak. |
| Voice note | Text the IM adds to the backend's instructions about the voice channel. `v2`, the default, asks for plain spoken sentences and adds rules for spoken IDs and for complete requests. `v1` adds nothing. |
| One-shot session | A request with no conversation, the official stateless use. The Live Console's client-mode rounds use it. |

## The Four Boxes

The example is four boxes. Each box owns its decisions, and none reaches into another's records. The following
diagram shows a call from the caller to the backend and back:

```text
 caller --- app (Live client) --- LIVE GATEWAY (src/live): sessions, media, command and close order, the wire schema
                                       |                                    |
                                       | frontend part                      | delegation part, by delegation.type
                                       v                                    v
              FRONTEND ADAPTER (Live to Pipecat), hosting the cascade        client: announce only; the app works
                talker: speak | delegate | wait                              responses: the server delegation
                heard log -> the next request's input                          ordered sender -> POST /v1/responses
                presenter's inbox: every ready result at idle                  paced relay of official events
                display -> the screen at once                                       |
                                       ^ official events                            | Responses API (in process)
                                       |                                            v
                                       +------------------------------- DELEGATION CORE (src/delegation_core)
                                                                          Responses API surface (api/)
                                                                          interaction manager (im/)
                                                                          work ledger (ledger/)
                                                                                    | backend port
                                                                                    v
                                                                        BACKEND ADAPTER (backend/), by backend.adapter
                                                                          llm (default)   rule conductor, 1 LlmRun
                                                                          llm-agent       model conductor, N LlmRuns
                                                                          harness         runtime conductor, N sessions
```

The following table lists the boxes, what each owns, and what each never knows:

| Box | Path | Owns | Never knows |
| --- | --- | --- | --- |
| Live gateway | [`src/live/`](../../src/live/) | Sessions, the order of commands and of the close, the wire schema, the relay of delegation events, and the choice of the delegation part by the resolved `delegation.type` | Pipecat, the cascade, and the IM's records |
| Frontend adapter (Live to Pipecat), hosting the cascade | [`adapter/`](../../src/examples/frontend_backend_live/adapter/) and [`cascade/`](../../src/examples/frontend_backend_live/cascade/) of the example, with its `models/`, `prompts/`, and `config_manager/` | Speech, caller turns, playout, the talker's memory, when a line may start (the floor), the heard log, the inbox and idle delivery, and the display | The IM's records, the backend's kind, and backend model calls |
| Delegation core: the IM, the work ledger, and the Responses API surface | [`im/`](../../src/delegation_core/im/), [`ledger/`](../../src/delegation_core/ledger/), and [`api/`](../../src/delegation_core/api/) of [`src/delegation_core/`](../../src/delegation_core/) | Admission, placement through the conductor, Works and their tasks, steering, the commit gate, stops, progress notes, results, endings, one-shot sessions, every record, and the official endpoints | Speech, playout, and backend protocols |
| Backend adapter | [`src/delegation_core/backend/`](../../src/delegation_core/backend/) | Model rounds, model providers, tool sources, conductors, worker sessions, and harness protocols | Speech and the voice wire |

The backend adapter lives in the `backend/` package of the delegation core, but it is a box of its own: the IM reaches
it only through the backend port, [`backend/port.py`](../../src/delegation_core/backend/port.py).

Four places turn names into objects, and none of them is a box. They are the composition roots:

- [`factory.py`](../../src/examples/frontend_backend_live/factory.py) builds each live session's frontend adapter and
  the gateway's one delegation core.
- [`pipeline.py`](../../src/examples/frontend_backend_live/pipeline.py) builds the repository UI bot, with a core of
  its own.
- [`app.py`](../../src/delegation_core/app.py) is the delegation core's builder, `build_core`.
- [`src/server.py`](../../src/server.py) mounts the Live routes from the registry's `live:` block.

## How the Frontend Media Plane Meets the IM

The frontend media plane is the gateway's media and session handling, the frontend adapter, and the cascade. It meets
the IM side, which is the delegation core, over two channels:

- Toward the core, the official Responses API. The gateway's server delegation sends each request through a
  `ResponsesClient` ([`responses_client.py`](../../src/live/server_delegation/responses_client.py)). In process, that
  client is `LocalResponsesClient` ([`api/local.py`](../../src/delegation_core/api/local.py)), which sends every
  request, result, and event through a JSON round trip, as HTTP would.
- Back to the media plane, the official Responses streaming events, plus one transport fact,
  `on_delegation_lost`, when a stream ends with no terminal event ([`live/ports.py`](../../src/live/ports.py),
  `DelegationEvents`).

### The Four Meeting Points

The two channels carry four meeting points. The following table lists each one with its official event or item and
the part of each side that handles it:

| Point | Direction | Official event or item | Media side | IM side |
| --- | --- | --- | --- | --- |
| 1. A delegation creates work | Media to IM | `POST /v1/responses`; its response opens `response.created` (status `queued`), then `response.queued` | The caller's turns and the server delegation's ordered sender | Admission, then the conductor |
| 1b. A request joins open work | IM to media | The taker's `response.in_progress` names the older response in `previous_response_id`; then the older response gets one `commentary` message and `response.completed` | The reader moves the older delegation's words, lead-in, and clock to the taker | `WorkRun`, when the steering lands |
| 2. Short status | IM to media | `message` items with `phase: commentary`; `mcp_call` items for the tools the backend runs | The work note keeps the newest line of each delegation | `ProgressNotes`, at most one note per delegation per `progress_note_seconds` |
| 3. A ready result | IM to media | One `message` with `phase: final_answer` per task; `show` items before it; then the terminal event | The inbox, then idle, then memory and speech together; the display at once | `WorkRun` writes it at once, and the IM never waits for speech |
| 3b. A wait for the app | Both | `function_call` items end the response; the app answers with `function_call_output` items and `response.create` | The app-wait clock and its line | The task waits, matched by `call_id` |
| 3c. Endings | IM to media | `response.failed` with an official error code; `response.incomplete` at close; an empty `response.completed` for a late continuation | The voice's own fixed lines | The stop door and the failure path |
| 4. Playback | None | Nothing crosses | Playout facts correct what the cascade wrote; heard words go into the next request | None |

Nothing crosses at point 4, because GPT-Live has no receipt event that confirms what the caller heard. Playout facts
stay in the frontend: they correct the talker's memory and the heard log, and the heard words reach the backend in the
next request.

Naming the taker of point 1b in `previous_response_id` of its `response.in_progress` is this server's convention. The
IM names it on every join, so the reader never guesses which delegation took the work.

### The Result Lifecycle

The core writes each result at once, and the voice side decides when the caller hears it. The following steps take a
result from the backend to the caller:

1. The core writes the task's `final_answer` item into the carrier's open response. While that response waits for the
   app, the item waits in order and is written when the next response opens.
2. The cascade's reader ([`delegations.py`](../../src/examples/frontend_backend_live/cascade/delegations.py)) reads the
   item. The display sink shows each completed `show` item at once, and the presenter's inbox keeps every ready result
   in arrival order.
3. When the caller is silent and no turn waits, the presenter words every ready result in one call. When no line is
   playing, the results and the spoken line enter the talker's memory and the heard log together, and speech starts.
4. The playout facts settle the line. A cut line keeps what played and ends with `[cut off]`. A line that nobody heard
   leaves the memory and the heard log, and its results go back to the inbox.

### Endings

A delegation ends in one of the following ways. The cascade reads each ending by structure, never by text, and says
its own fixed line from `lines:` in the example's [`prompts.yaml`](../../src/examples/frontend_backend_live/prompts.yaml)
where one applies:

| Ending | What the core writes | What the voice says |
| --- | --- | --- |
| Answered | Each task's `final_answer` message, then `response.completed` | The results, worded together at idle |
| Handed over | The taker's `response.in_progress` names this response; then one `commentary` line (`texts.handover_line`) and `response.completed` | Nothing of its own: the taker carries the work, its words, and its lead-in |
| Stopped for a newer request | `texts.cancelled_line`, or `texts.unconfirmed_line` when the stop was not confirmed, as a `final_answer`, then `response.completed` | That line, as a result |
| Failed | `response.failed` with `rate_limit_exceeded` or `server_error`, or `response.incomplete` with an official reason such as `max_output_tokens` | `apology` when no tool ran; `unclear` after a tool ran |
| Late | A continuation after its Work ended gets an empty response: `response.created`, `response.queued`, `response.in_progress`, then `response.completed` | `unclear` |
| Cancelled at close | `response.incomplete` with status `cancelled` and `incomplete_details: null` | Nothing: the session is closing |
| Lost | No terminal event: the stream ended, and the gateway calls `on_delegation_lost` and sends the client a `response.failed` | `lost` |

The core writes no apology and never retries a Work. A failed Work ends its response with `response.failed`, so a
client that stops on that event stops there.

## The Interaction Manager

The IM ([`im/`](../../src/delegation_core/im/)) owns everything that happens to backend work after a delegation. One IM
session serves one conversation, and the Responses API surface opens one for each conversation.

### Admission

Admission is one instant step ([`session.py`](../../src/delegation_core/im/session.py), `IMSession.admit`). The IM
refuses a request before anything changes when the session is closing, when the delegation ID is in use, or when
`delegation.max_pending_delegations` requests already wait. Otherwise, it records the delegation, opens its response at
once with `response.created` (status `queued`) and `response.queued`, and keeps the request as an arrival.

A request that the app makes itself goes through `IMSession.admit_client_request`. Function outputs continue the
response whose calls they answer, matched by `call_id`, and they must answer every call of that response, or the IM
refuses them with the official code `function_call_outputs_required`. Outputs for calls whose Work has ended get an
empty `response.completed`. Anything else is a typed run: an arrival like any delegation, which brings its own
settings.

### Placement: The Conductor Decides, the IM Applies

The backend adapter's conductor places every arrival. The IM asks the adapter for the conductor once per session and
calls `decide()` one call at a time:

- `decide()` gets the new arrivals, the open Works and the newest `delegation.records_max_works` ended ones, and the
  conversation, with the position of the first item that this conductor has not read yet.
- The conductor places each decision through `place()`: `Start`, `Steer`, `Cancel`, or `Answer`. The IM applies each
  one, re-checks it against the current state, and returns a one-line outcome from `texts.outcomes`, which a model
  conductor reads as its control tool's result.
- An arrival that comes while a `decide()` runs restarts that call once with every arrival, if it has placed nothing
  yet. Otherwise, the arrival waits for the next `decide()`, which sees the earlier requests as Works.

The IM re-checks every decision:

- A `Steer` to a Work or task that has ended starts a new Work instead.
- A `Cancel` of a Work or task that has ended does nothing.
- A delegation placed twice keeps its first placement.
- A decision that names a Work or task the IM does not know is refused.

What a conductor leaves unplaced is placed by the `steer` rule: it joins the newest open Work, or starts a Work when
none is open, so no request starts a second Work beside open work. The two model conductors apply this rule themselves
after a failed, timed-out, or skipped decision. The IM applies it, through the same re-checks, to whatever is still
unplaced when a `decide()` ends: after a call that raised, was cut short, or skipped a request.

### Works and Tasks

A Work ([`run.py`](../../src/delegation_core/im/run.py), `WorkRun`) carries one or more requests, oldest first, and the
newest one carries its response. A Work holds one or more tasks (`TaskRun`), and each task is one worker:

- A model conductor can split a request into independent tasks with `Start(tasks=...)`, each with a brief, and can
  steer or cancel one task.
- The scheduler starts tasks in the order they became ready while fewer than the adapter's `max_running` run. The
  limit counts tasks across every Work of the session. A task that waits for the app holds no slot.
- Each task's answer is its own `final_answer` item. Nothing merges or rewords the answers.
- The Work's response ends when its last task ends.
- When one task waits for the app, the response ends with that task's calls, as GPT-Live requires, while the other
  tasks run on. The IM holds their items in order and writes them when the continuation opens the next response.

Each task reads the conversation within `delegation.context_tokens`, then its Work's requests, then its brief, if the
conductor wrote one (`texts.task_line`). The work records go into the last user message of a request, as extra text
parts: one JSON line per open or recently ended Work, inside `texts.records_block`, followed by `texts.closing_line`,
which asks the backend to say where the work stands and carry on. The newest request of a starting Work carries the
records of the other Works, when there are any, and every steered request carries the records of every Work.

### Steering

A request that joins a Work that started is steering. The IM decides when it lands, by the state of the Work or task,
and the adapter decides how. The following table lists when steering lands:

| State | When the steering lands |
| --- | --- |
| The Work, or the task, has not started | At once. The newest request carries the Work. |
| The task runs | At the adapter's next safe point: the IM calls `run.steer()`, the adapter takes the batch with `take_boundary_input()`, and it reports `applied` or `not_applied` through `settle()` |
| The task waits for the app | After the app's outputs, at the next safe point |
| No task applied it before the Work ended | It starts a follow-up Work, so nothing the caller said is lost |

When a batch lands, each of its requests takes the Work over in turn, as point 1b describes. Unapplied input never
enters the backend's history. A joined request keeps the Work's settings, and an explicit request (a continuation or a
typed run) brings its own settings, which the next model call uses.

### The Commit Gate

The commit gate keeps the backend from acting on an intent that the caller changed. While a newer request waits to
land in a task, every call that the task routes is refused: it is not run, and its output is `texts.not_run_output`,
so the backend plans again with the newer request after it lands. A round's calls for the app go out only after the
round's backend calls ran, so newer input that comes meanwhile keeps them back too. Calls that run already are never
cut.

The gate also covers the time that a model conductor (`llm-agent` or `harness`) takes to decide. While an arrival waits
for its placement, every call that a task of any open Work routes is refused the same way, because the conductor has
not yet said which Work the arrival concerns. A task whose calls were refused then waits for the decision before its
next round, so that round reads the not-run outputs together with what the decision brings. When the decision stops
the task, no next round starts. With `llm`, the rule conductor decides at once, and the request then waits to land as
steering.

Each refused call's output says why it did not run:

| Reason | Output |
| --- | --- |
| Newer input waits to land, or for the conductor's decision | `texts.not_run_output` |
| The task is stopping, by a stop or by its own `cancel_work` | `texts.stopped_output` |
| The app and a work tool both have a function by the call's name | `texts.clash_output` |

### Stops

The IM stops work only through one door, `WorkRun.stop`. After a stop is asked, nothing more of the task is written,
and its calls are refused. A queued task leaves the scheduler. A running task gets `await run.stop()`, which the IM
waits for at most `delegation.stop_timeout_seconds`, and the stop ends `stopped` or `unconfirmed`. The reason writes
the ending ([`stop.py`](../../src/delegation_core/im/stop.py)):

| Reason | Ending |
| --- | --- |
| A conductor's `Cancel`, for a newer request | `texts.cancelled_line` (stop confirmed) or `texts.unconfirmed_line` as the stopped task's `final_answer`, then `response.completed` |
| The task's own `cancel_work` work tool | Not a stop of the run: the call is a control call that writes no item on the wire, later calls are refused with `texts.stopped_output`, and its next answer is delivered |
| The client-tool deadline (`delegation.client_tool_timeout_seconds`) on a task that waits for the app | Nothing is written, because the response already ended with its calls. The Work fails, its calls become unclear, and waiting steering becomes a follow-up Work. When a newer request already landed in another task of the Work, that request takes the Work over instead: its response goes in progress and names the parked response in `previous_response_id`, the other tasks' held items go into it, the stuck task's answer is `texts.owed_output`, and the Work ends as its tasks end |
| The backstop (`delegation.work_timeout_seconds`) | `response.failed` with `server_error` |
| The session's close | The close writes every open response's terminal itself |

A caller's "stop" is a request like any other. A model conductor decides a `Cancel`. With `llm`, the request joins the
running Work, and the backend LLM reads it.

### Failures and Time

When a task fails, its Work's response ends, after the Work's last task, with `response.failed` and an official error
code: `rate_limit_exceeded` for an upstream 429, the failed attempt's own code, or `server_error`. The detail goes in
`message`. A round that its model cut at `max_output_tokens` or by a content filter ends the response with
`response.incomplete` and that reason. A task whose run ends with neither an answer nor the app's calls fails with
`server_error`.

Every run ends or raises within its adapter's own limits: `llm` has a first-event limit, a silence limit, and a tool
limit, and `harness` has a turn limit. The IM's `delegation.work_timeout_seconds` (600 seconds) is only the backstop.

### Progress Notes and Status

The backend's `commentary` messages and the harness's activity become progress notes on the carrier's open response,
at most one per delegation every `delegation.progress_note_seconds`. Notes never join a backend history. A status
question takes one of three paths:

1. The talker answers it from the newest progress note in its work line, and nothing is delegated.
2. If the talker delegates it, a model conductor can answer it from the work records with `Answer` (the
   `answer_status` control tool), which ends the new delegation while the Work keeps running.
3. With `llm`, whose conductor is a rule, the question joins the running Work. The records block and the closing line
   ask the backend to say where the work stands and carry on. A message written in a round that also has calls is
   `commentary`, so the status line is progress and the Work goes on.

### One-Shot Sessions

A request with no conversation is served by a one-shot session ([`oneshot.py`](../../src/delegation_core/im/oneshot.py))
on an adapter from the source that `backend.one_shot_adapter` names. Its input is used as sent: no placement, no
records block, no result-format note, no work tool, and no trim, although the voice note still joins the
instructions. Its response opens in progress. Its round's calls end its response, and nothing waits for their outputs.
A failure ends it with `response.failed`. The session ends with its response, and nothing of the request is kept.

### Close

When a conversation is deleted, its IM session closes ([`session.py`](../../src/delegation_core/im/session.py),
`IMSession.close`):

1. No new request is accepted.
2. Queued Works and Works that wait for the app stop at once.
3. Running Works get `delegation.close_grace_seconds` (5 seconds) to end.
4. Every task still open stops, and every response still open ends once, in the order its request was accepted, with
   `response.incomplete`, status `cancelled`, and `incomplete_details: null`.
5. The backend adapter closes last, whatever happened before.

The work ledger ([`ledger/`](../../src/delegation_core/ledger/)) keeps every record in memory, and nothing is written
to disk.

## The Backend Port

The backend port, [`backend/port.py`](../../src/delegation_core/backend/port.py), is the only module that both the IM
and every backend adapter import. Only Responses shapes and the port's types pass between them.

### Three Levels

The port has three levels:

- `AdapterSource`: one per delegation core, chosen by `backend.adapter` and built from its own block of `core.yaml`
  (`from_settings`). It answers the model check (`check_model`), the settings check (`unsupported`), and `/info`
  (`describe`). A source that can serve stateless requests is also a `OneShotSource`.
- `BackendAdapter`: one per IM session. It gives the session's `Conductor` and `max_running`, and `start(task)` runs
  one task.
- `BackendRun`: one per task, which is one worker. `events()` yields the task's Responses events, `steer()` wakes the
  adapter for waiting steering, `resume()` continues after the app's outputs, and `stop()` returns `stopped` or
  `unconfirmed`.

The IM implements `TaskContext` for each task and `DecisionTarget` for the conductor. A task reads its settings
(`request()`, with the composed instructions) and its history (`history()`), asks who runs each call (`route()`:
`app`, `backend`, or `refused`), reports the calls the backend runs (`tool_started`, `tool_outcome`) and its other
activity (`progress`), takes the app's outputs and waiting steering at each safe point (`take_boundary_input()`), and
reports what became of steering (`settle()`).

### The Conductor

A `Conductor` has one method, `decide(given, place)`. The decisions are `Start(delegations, tasks)`,
`Steer(work_id, delegations, task_id)`, `Cancel(work_id, task_id)`, and `Answer(delegation_id, text)`. A model
conductor reads its instructions (`texts.conductor_prompt`) and its four control tools (`start_work`, `steer_work`,
`cancel_work`, and `answer_status`, from `texts.control_tools`) through `ControlTools`
([`control.py`](../../src/delegation_core/backend/control.py)). A conductor never runs work and never words a result.

### The Event Rules

Every `BackendRun.events()` keeps eight rules, and the contract suite checks them:

| Rule | What it says |
| --- | --- |
| E1 Rounds and attempts | Each attempt opens with `response.created` and ends with exactly one terminal event. `response.completed` ends the round; `response.failed` or `response.incomplete` ends only the attempt. |
| E2 Items | Each item is announced by `response.output_item.added` and finished by `response.output_item.done`. The IM reads items only from `output_item.done`. |
| E3 Phase | Every message carries `phase`: `commentary` is progress, and `final_answer` is the task's answer. |
| E4 Calls last | A round's calls come after its messages, and a round with calls carries no `final_answer`. |
| E5 How the iteration ends | After a round with a `final_answer`, after a round with the app's calls (the task waits for `resume()`), or by raising. After `stop()`, nothing more. |
| E6 Every step the IM must see | Each call the backend runs is reported before it runs and when it ends, a `call_id` never runs twice, and steering reaches the IM only through `settle()`. |
| E7 Reading and retries | An adapter that owns its model rounds reads `request()` and `history()` at each model call. A harness sends `history()` in the worker's first prompt. No attempt that dispatched a call or delivered an item is retried. |
| E8 Time | A task ends or raises within the adapter's own limits, and `stop()` returns within its stop limit. |

## The Three Backend Adapters

Three backend adapters stand behind one port. `backend.adapter` in `core.yaml` chooses one, by name, from the table in
[`backend/adapters.py`](../../src/delegation_core/backend/adapters.py). The table loads an adapter's package only when a
core is built with it. The following table compares the three adapters:

| | `llm` (the default) | `llm-agent` | `harness` |
| --- | --- | --- | --- |
| What it is | The backend LLM, directly connected | The backend LLM agent: a model conductor and LLM workers | An external agent that its operator deploys, over the Agent Client Protocol (ACP): any ACP agent, with Hermes as the first agent class |
| Package | [`backend/llm/`](../../src/delegation_core/backend/llm/) | [`backend/agent/`](../../src/delegation_core/backend/agent/) | [`backend/harness/`](../../src/delegation_core/backend/harness/) |
| Conductor | A rule, with no model call: `steer` (the default) or `start_each` (`backend.llm.conductor`) | One model call per `decide()`, with the four control tools, `tool_choice: required`, and parallel calls | Its own agent session, in a process of its own, with exactly the four control tools: one prompt per `decide()` |
| Workers | One `LlmRun`: one task per Work | `LlmRun`s; a request can split into parallel tasks | One ACP session per task, in the shared workers' process |
| `max_running` | 1 | `max_running` (3) | `max_running` (3), at most the agent's turns per process (4 for Hermes) |
| Steering | A queued Work joins at once; a running task redoes a round that still streams and has dispatched nothing, once per task, or else takes the steering at its next round; a waiting task takes it after the outputs | The same as `llm` | `backend.harness.steer`: `native_command` (the agent's own mid-turn steer, such as Hermes's `/steer`, which `core.yaml` sets), `next_prompt` (the default), or `cancel_reprompt` |
| Stop | Cuts a round that still streams, lets started tool calls finish, and confirms within `stop_timeout_seconds` | The same as `llm` | `session/cancel` while a prompt runs, confirmed when the turn ends `cancelled` within `stop_timeout_seconds`; the session's bridge door is closed |
| Status answers | Never: the question joins the Work | `answer_status` | `answer_status` |
| Instructions and the envelope | Every round's instructions | Every round's instructions of each worker | Where the agent class puts them: Hermes reads a `.hermes.md` file in the worker's own session folder, and a plain ACP agent gets them at the top of the first prompt |
| Stateless requests | Serves them | Served by `backend.one_shot_adapter` (`llm`) | Served by `backend.one_shot_adapter` (`llm`) |
| Model | The request's model, through the catalog (`unlisted_models`) | Workers: the request's model; conductor: `conductor.model`, or the default | The model that the agent's operator configures; requests can name only the default of `backend.slot`, which the agent should run |
| Profile | `core.yaml` | `core.agent.yaml` | `core.harness.yaml` |

`llm` is the default: it adds no conductor model call before work starts, and it needs no agent of its own.

### The Backend LLM, Directly Connected

`llm` ([`backend/llm/`](../../src/delegation_core/backend/llm/)) is the backend LLM loop on the port. `LlmSource` builds
it from `backend.llm` (defaults in [`settings.py`](../../src/delegation_core/backend/llm/settings.py)), and each IM
session gets an `LlmAdapter` with one model provider per model. `LlmRun` runs one task:

- At each safe point, it takes the app's outputs and waiting steering, and settles the steering `applied` at once.
- Each round reads `request()` and `history()`, sends the whole history, and chains nothing.
- The round's tools are the request's function tools, then each server tool source's, then the IM's work tools.
- In a round with calls, every message is `commentary`, and the IM routes each call before the round is passed on.
  Calls that the backend runs start at once after the round, and a call cut at its time limit is `unclear`.
- An attempt that failed before it dispatched a call or delivered an item is tried once more. A 400 is never retried.
- After `max_tool_rounds` rounds, one more round runs with `tool_choice: none` and the `round_cap_note`, so the task
  answers with what it has.

A model that the catalog does not list follows `backend.llm.unlisted_models`. With `report`, the default, the core
refuses the model, and the gateway's settings check reports it to the client as the official nonfatal
`responses_validation_failed`. With `fallback`, which only the benchmark profile sets, it runs on the configured
endpoint under its own name, and the server logs it.

### The Backend LLM Agent

`llm-agent` ([`backend/agent/`](../../src/delegation_core/backend/agent/)) adds a model conductor to LLM workers:

- `AgentSource` parses `backend.llm-agent`: `max_running`, `conductor` (`provider`, `model`, `timeout_seconds` of 20,
  `memory_tokens` of 8,000, `max_output_tokens_per_request` of 256, `temperature` of 0, and `reasoning_effort` of
  `none`), and `worker`, which takes the `backend.llm` keys except `conductor`.
- `LlmConductor` makes one model call per `decide()`. Its input is one user message whose text is one JSON object:
  the new part of the conversation, the work records, and the new requests. Every control call the model makes is
  placed at once, and the IM's outcome line becomes that call's output in the conductor's own memory, which keeps
  whole decision rounds within `memory_tokens`.
- What the model leaves unplaced, after a failure, a timeout, a round with no call, or a skipped request, is placed by
  the `steer` rule.
- The workers are `LlmRun`s of the session's `LlmAdapter`, so a request's model, settings, and tool sources reach them
  exactly as they reach `llm`.

[`core.agent.yaml`](../../src/examples/frontend_backend_live/core.agent.yaml) selects it: `backend.adapter: llm-agent`,
with `delegation.result_format: speak_display`.

### The Harness

`harness` ([`backend/harness/`](../../src/delegation_core/backend/harness/)) runs the backend work on an external
agent that its operator deploys, such as Hermes. The operator gives the agent its model, key, toolsets, and home, and
`backend.harness` says only which agent class reads the agent and how to reach it. When the IM asks for the conductor,
the adapter starts or connects two runtime processes for the IM session: one for the conductor and one for the
workers. To deploy an agent and connect it, refer to [Connect Your Own Harness](#connect-your-own-harness).

The harness has three layers. Each one is an interface whose classes a key of `backend.harness` names, as the
following table shows:

| Layer | Package | What it owns | Chosen by |
| --- | --- | --- | --- |
| The generic harness | [`harness/`](../../src/delegation_core/backend/harness/) | The conductor session and its control tools, the worker runs and their tasks, the steering policy, stops and the commit gate, the tool bridge, the results, and the envelope. It is the same for any protocol. | `protocol`, `steer`, `permission` |
| The ACP client | [`harness/acp/`](../../src/delegation_core/backend/harness/acp/) | Initialize, sessions, prompts, cancel, session updates, the MCP servers given at `session/new`, and the transports: `stdio` starts `command`, and `websocket` connects to the `url` that the operator serves. | `transport` |
| The agent classes | [`acp/agent.py`](../../src/delegation_core/backend/harness/acp/agent.py) and [`acp/hermes/`](../../src/delegation_core/backend/harness/acp/hermes/) | Everything of the connection that is one agent's own. `acp`, the default, is a plain ACP agent, and `hermes` is Hermes. | `agent`, with that agent's own `agents.<name>` block |

The generic harness reaches the agent through two seams in
[`harness/protocol.py`](../../src/delegation_core/backend/harness/protocol.py): `HarnessProtocol`, which opens a
client for each runtime process, and the `Agent` that it holds. The following table compares the two agent classes:

| | `acp` (the default) | `hermes` |
| --- | --- | --- |
| Instructions | At the top of the session's first prompt, because the ACP `session/new` request has no instructions field | In `.hermes.md` in the session's own folder, which is pinned with an empty `.git`, so no parent folder's file reaches the session |
| Mid-turn steer | None: `steer` is `next_prompt` or `cancel_reprompt` | `/steer` followed by the batch, which Hermes takes at the turn's next model boundary |
| A prompt's text | As written | A prompt that starts with `/` gets `Request: ` first, so it never runs as a Hermes command. A steer's text gets the same guard after `/steer `, because an idle Hermes runs that text as a prompt |
| A message chunk with no message ID | Part of the message being streamed, until a tool call, a plan, or a prompt comes between; reasoning does not end it | Hermes's own reply, such as an acknowledgment, which is never output |
| Tool cards | Every card is a call of the session's bridged tools, which the bridge reports | A card named `mcp__<server>__<tool>` is a bridged call; any other card is a Hermes tool, reported with `runtime_label` |
| A turn's answer | The last message of a turn that ended `end_turn` | Also needs the turn's usage to be reported and to move, and the message to be streamed by a model call |
| Turns per process | No limit | 4 |
| A started process's environment | Nothing more | `HERMES_ACP_SKIP_CONFIGURED_MCP=1`: Hermes then starts the MCP servers that its home lists at its first session instead of at its startup |

`agents.hermes` sets `instructions_file` (`.hermes.md`), `steer_command` (`/steer `), and `command_guard`
(`Request: `). The core's build refuses a value that Hermes cannot use:

- `instructions_file` must be a file that Hermes reads in a session's folder: `.hermes.md`, `HERMES.md`,
  `AGENTS.override.md`, `AGENTS.md`, `agents.md`, `CLAUDE.md`, `claude.md`, or `.cursorrules`.
- `steer_command` must end with a space, because Hermes reads a command's name up to the first space. Quote it in
  YAML, which drops the trailing space of an unquoted value.
- `command_guard` must not start with `/`, because Hermes would run it as a command.

The tool bridge (`McpBridge`) is one Model Context Protocol (MCP) server over streamable HTTP, one per core. Each ACP
session gets its own door: a server name unique in its process, a bearer token, and its tools. The conductor's door
offers exactly the four control tools. A worker's door offers the app's functions, the session's tool sources, and the
IM's work tools, and it asks the IM's `route()` before anything runs, so the commit gate covers the agent's calls of
these tools. The bridge never marks a result as an error, because an agent can pause a server after a few failed
calls in a row. The following table shows how ACP maps onto the port:

| ACP | Port |
| --- | --- |
| `agent_message_chunk` | Held: `commentary` when tool activity follows, `final_answer` at the turn's end. Only text that the agent class reads as its model's is passed on. A reply that the agent wrote itself, such as a provider error, is logged and never spoken, and as a turn's answer it fails the task. |
| The agent's own `tool_call` and `tool_call_update` | With an agent class that tells its own tools apart, such as `hermes`: `tool_started` and `tool_outcome`, as `mcp_call` items labeled with `runtime_label` (`hermes` in `core.yaml`). With `acp`, every card counts as a call of the session's own MCP server, so the agent's own tool calls are not reported at all. The bridge reports the calls of the session's own MCP server. |
| `plan` | `progress()` |
| An app function called through the bridge | Held by the bridge, one app call per round; the round's `function_call` and terminal follow after every other call of the turn has reported, and `resume()` answers the held MCP request |
| `request_permission` | Answered by `backend.harness.permission`; no approval request reaches the app |
| `end_turn` | The answer, unless steering waits: then the text is `commentary`, and the steering is the next prompt |
| `cancelled` | Confirms a stop |
| Another stop reason, or a lost connection | Fails the task; the outputs it owes say the outcome is unknown |

A cancel goes only to a prompt that runs, and a cancelled session is never used again, because an agent can carry a
cancelled turn into its next prompt. Each session works in a folder of its own under `backend.harness.work_root`
(`~/.cache/voiceclaw/harness` by default), outside any git checkout, and the folder is deleted at the end of the call.
A server that crashes can leave its folders behind, and the next core built with the same `work_root` deletes them. A
running server holds a lock on each of its work folders, so a core deletes only the folders that no server holds. The
servers of one user on one machine, containers included, can therefore share a `work_root`.

The harness has the following limits:

- With Hermes, a one-chunk reply that is a turn's only text fails the task, because on the wire it cannot be told
  apart from Hermes's own text.
- The commit gate cannot hold the agent's own tools, such as Hermes toolsets that its operator turns on. They run
  inside the agent. With `hermes`, they are reported after the fact, and with `acp`, they are not reported at all.
  Give the agent no tools of its own where the gate matters, and connect a plain agent only with its own tools turned
  off.
- The agent runs one model, the one its operator configures. The model check accepts only the default of
  `backend.slot`, and the settings check (`unsupported()`) refuses a reasoning effort. A request's instructions and
  functions are fixed when its worker starts, so a later change reaches only new workers.

### The Envelope

The envelope is the speak and display shape of a result. `delegation.result_format: speak_display` turns it on:

1. The IM adds `texts.result_formats.speak_display` to every task's instructions, after the voice note, and offers the
   `show` work tool.
2. The backend puts screen-only detail in one `show` call. The IM writes it as an `mcp_call` item with server label
   `delegation.display_label` (`voiceclaw`) and name `show`, whose `output` is the display.
3. The backend answers with speakable sentences, the `final_answer` message.

On the voice side, the display sink shows a completed `show` item at once and never speaks it, and the `final_answer`
waits in the inbox and is said at idle, together with every other ready result. The instructions reach every worker
where its backend takes a system prompt (every round for `llm` and `llm-agent`, and where the agent class puts a
session's instructions for `harness`), and never the conductor. The UI bot's profile, `core.agent.yaml`, and the
harness profile turn it on. The default is `plain`, because a Live client can be voice only.

## Extension Recipes

Each extension point is an interface in its box's port module, its implementations sit in one plain table, and the
composition root picks one by the name in the configuration. Adding one is one class, one table entry, and the name in
config. No code compares a kind against a name: the guard test fails on such a branch (refer to
[Guard Test and Import Rules](#guard-test-and-import-rules)).

### Add a Frontend Adapter

A frontend adapter serves a live session's voice. To add one:

1. Write a class with the class contract `FrontendAdapterClass`
   ([`adapter/contract.py`](../../src/examples/frontend_backend_live/adapter/contract.py)): `Settings.parse(block)`,
   `readiness(settings, catalog)`, `from_config(session, transport, connection, settings, catalog)`,
   `describe(settings, catalog)`, and `hosts_rtvi`. Each instance is a `FrontendAdapter`
   ([`live/ports.py`](../../src/live/ports.py)): `start`, `close`, `set_muted`, `feed_audio`, and the three appends,
   each of which returns quickly and never waits on a model.
2. Add the class to `FRONTEND_ADAPTERS` in
   [`adapter/registry.py`](../../src/examples/frontend_backend_live/adapter/registry.py).
3. Set `frontend.adapter` to its name in `config.yaml`, and give it its own `frontend.<name>` block.

It delegates through the `DelegationPort` that the session hands it and learns about its delegations only from the
official events (`DelegationEvents`). It decides by `reports_events`, never by comparing `target`. The UI bot refuses
an adapter whose `hosts_rtvi` is false.

### Add a Backend Adapter

A backend adapter runs the work behind the port. To add one:

1. Write a package under `src/delegation_core/backend/` with an `AdapterSource` whose `from_settings(block, *,
   endpoints, tools, vocabulary)` validates its block and connects nothing, a `BackendAdapter` with `conductor()` and
   `max_running`, and a `BackendRun` that keeps the event rules E1 to E8.
2. Give it a conductor: a rule, a model through `ControlTools`, or a runtime session. What the conductor leaves
   unplaced goes to the `steer` rule (`SteerOpenWork` in
   [`conductors.py`](../../src/delegation_core/backend/conductors.py)).
3. Add one entry to `ADAPTERS` in [`adapters.py`](../../src/delegation_core/backend/adapters.py), which loads the
   package only when a core is built with it.
4. Bind the contract suites `BackendAdapterContract` and `ConductorContract`
   ([`tests/unit/contracts/backend.py`](../../tests/unit/contracts/backend.py)) to it, and add its group to the import
   rules.
5. Name it in `backend.adapter` and give it its block in a `core.yaml` profile.

### Add an Agent, a Protocol, a Transport, or a Steering Strategy to the Harness

The harness's seams are private to the harness, so each new one stays inside `src/delegation_core/backend/harness/`:

- An agent that keeps to the Agent Client Protocol and meets the conditions in
  [Connect Another ACP Agent](#connect-another-acp-agent) is configuration only: `agent: acp`, a `steer` that it can
  take, and its `command` or `url`. For an agent with ways of its own, such as a file it reads instructions from, a
  mid-turn steer, or replies of its own that must never be spoken, subclass `AcpAgent`
  ([`acp/agent.py`](../../src/delegation_core/backend/harness/acp/agent.py)) in a package of its own beside
  `acp/hermes/`. Set its `name`, its key in `AGENTS`, which its error messages name, and its `keys`, the keys that its
  `agents.<name>` block takes, which the inherited `check_keys` checks. Set `turns_per_process` too when one of its
  processes runs a limited number of turns at once. Give it a `from_settings(block)` that reads its `agents.<name>`
  block, add it to `AGENTS` in [`acp/agents.py`](../../src/delegation_core/backend/harness/acp/agents.py), and name it
  in `backend.harness.agent`.
- A protocol is a `HarnessProtocol` with a `from_settings(block)`: its `client()` makes a `HarnessClient` with its
  `HarnessSession`, and its `agent` is an `Agent`. Add it to `PROTOCOLS` and name it in `backend.harness.protocol`.
  Its packages load only when a core is built with it.
- A transport of the ACP client is a `Transport` (`check`, `for_launch`, `open`, and `close`), added to `TRANSPORTS` in
  [`acp/transports.py`](../../src/delegation_core/backend/harness/acp/transports.py) and named in
  `backend.harness.transport`.
- A steering strategy is a `SteerStrategy` (`native` and `deliver`) with a `from_settings(block)`, added to `STEERING`
  and named in `backend.harness.steer`. A strategy that needs the agent's own mid-turn steer is refused for an agent
  without one.

### Add a Model Provider

A backend model provider is private to `llm` and `llm-agent`. Subclass `Backend`
([`providers/base.py`](../../src/delegation_core/backend/providers/base.py)), give it a `name`, implement
`_events(payload)` to yield one round's Responses events, list the settings it cannot honor in `unsupported()`, add
it to `BACKENDS` in [`providers/__init__.py`](../../src/delegation_core/backend/providers/__init__.py), and bind
`ProviderContract`. Then name it in `backend.llm.provider` or `backend.llm-agent.conductor.provider`. The template
`Backend.stream()` adds the retries and the time limits.

A frontend model provider serves the talker, the writer, and the normalizer. Subclass `Frontend`
([`models/base.py`](../../src/examples/frontend_backend_live/models/base.py)) with its `name`, implement
`complete_once(instructions, history, spec)`, add it to `FRONTENDS` in
[`models/__init__.py`](../../src/examples/frontend_backend_live/models/__init__.py), and set
`frontend.cascade.provider`.

### Add a Tool Source

A tool source runs tools on the server for the backend. Write a `ToolSource`
([`backend/port.py`](../../src/delegation_core/backend/port.py)) with a `label`, `tools()`, an idempotent
`execute(call_id, name, arguments)`, and `from_settings(block)`, add it to `TOOL_SOURCES` in
[`backend/tools/__init__.py`](../../src/delegation_core/backend/tools/__init__.py), and name it under `backend.tools`
with its block. The root makes a fresh set for each IM session, so per-session state, such as a cart, is never
shared. The cafe tools in [`tools/cafe/`](../../src/delegation_core/backend/tools/cafe/) are the reference, and
`core.ui-bot.yaml` turns them on.

### Add a Prompt Set

A prompt set holds every text that the cascade's own model calls read. Write a module that builds its `PROMPT_SET`
([`prompts/promptset.py`](../../src/examples/frontend_backend_live/prompts/promptset.py)), add it to `PROMPT_SETS` in
[`prompts/__init__.py`](../../src/examples/frontend_backend_live/prompts/__init__.py), and set
`frontend.cascade.prompt_version`. Behavior that differs between sets is data in the set, so no code asks which set it
runs. `v2` is the default set, and `v1` and `v3` are the alternatives.

### Extension Points at a Glance

The following table lists every extension point, its interface, what is built, and the key that chooses it:

| Point | Interface (module) | Built | Chosen by |
| --- | --- | --- | --- |
| Backend adapter | `AdapterSource`, `BackendAdapter`, `BackendRun`, `Conductor` (`core/backend/port.py`) | `llm`, `llm-agent`, `harness` | `backend.adapter`, `backend.one_shot_adapter` |
| Rule conductor | `Conductor` | `steer`, `start_each` | `backend.llm.conductor` |
| Backend model provider | `Backend` (`core/backend/providers/base.py`) | `chat-completions`, `openai-responses` | `backend.llm.provider` |
| Harness seams | `HarnessProtocol`, `Agent`, `SteerStrategy`, `PermissionAnswer` (`harness/protocol.py`); `AcpAgent`, `Transport` (`harness/acp/`) | `acp`; `acp`, `hermes`; `stdio`, `websocket`; `native_command`, `next_prompt`, `cancel_reprompt`; `reject` | `backend.harness.protocol`, `agent`, `transport`, `steer`, `permission` |
| Tool source | `ToolSource` (`core/backend/port.py`) | `cafe` | `backend.tools` |
| Result format; backend call reports | `ResultFormat`, `CallReports` (`core/im/results.py`) | `plain`, `speak_display`; `items`, `notes` | `delegation.result_format`, `delegation.backend_calls` |
| Session kind | `IMSession`, `OneShotSession` (`core/im/`) | Both | The official rule: a request with no conversation is one-shot |
| Frontend adapter | `FrontendAdapter` (`live/ports.py`), `FrontendAdapterClass` (`FBL/adapter/contract.py`) | `cascade` | `frontend.adapter` |
| Delegation part | `DelegationPart` (`live/ports.py`) | `client`, `responses` | The resolved `delegation.type` of each session |
| TTS mode | `SpeechMode` (`FBL/adapter/services.py`) | `per_sentence`, `stitched` | `frontend.cascade.speech.tts_mode` |
| Frontend model provider; prompt set | `Frontend` (`FBL/models/base.py`); `PromptSet` (`FBL/prompts/promptset.py`) | `chat-completions`, `openai-responses`; `v1`, `v2`, `v3` | `frontend.cascade.provider`; `frontend.cascade.prompt_version` |
| Result wording; display sink | `ResultWording`, `DisplaySink` (`FBL/cascade/ports.py`) | `writer`, `as_written`; `NullDisplay`, `RtviTaskCards` | `frontend.cascade.delegation.result_wording`; the root |
| Live example | The registry's `live:` block | `frontend-backend-live` | `examples_registry.yaml` |

## Configuration

Two files configure the example, each with profiles. A root reads each file once, on first use, so an edit takes
effect at the next restart. An unknown key stops the load.

### Files and Profiles

The following table lists the files in the example's folder:

| File | Holds | Selected by |
| --- | --- | --- |
| [`config.yaml`](../../src/examples/frontend_backend_live/config.yaml) | The frontend only: `frontend.adapter`, each adapter's block (`frontend.cascade`), and `ui_bot.core_config` | The default; `LIVE_CONFIG_PATH` names another file |
| [`config.bench.yaml`](../../src/examples/frontend_backend_live/config.bench.yaml) | `config.yaml` with no app-wait line and no spoken progress lines, for a benchmark client | `LIVE_CONFIG_PATH` |
| [`core.yaml`](../../src/examples/frontend_backend_live/core.yaml) | The delegation core: `delegation` (the IM's settings), `texts` (every line the core writes or adds to a model's input), and `backend` (the adapter, the default model's slot and prompt, the tool sources, and each adapter's block) | The default; `CORE_CONFIG_PATH` names another file |
| [`core.ui-bot.yaml`](../../src/examples/frontend_backend_live/core.ui-bot.yaml) | `core.yaml` with the cafe tools and the envelope, for the repository UI bot | `ui_bot.core_config` in `config.yaml` |
| [`core.bench.yaml`](../../src/examples/frontend_backend_live/core.bench.yaml) | `core.yaml` with no client-tool deadline, no progress notes, plain results, and `unlisted_models: fallback`, for a benchmark client | `CORE_CONFIG_PATH` |
| [`core.agent.yaml`](../../src/examples/frontend_backend_live/core.agent.yaml) | `core.yaml` with `backend.adapter: llm-agent` and the envelope | `CORE_CONFIG_PATH` |
| [`core.harness.yaml`](../../src/examples/frontend_backend_live/core.harness.yaml) | `core.agent.yaml` with `backend.adapter: harness`. The `backend.harness` block of `core.yaml` names the Hermes agent class, starts `hermes acp` over stdio, and binds the bridge on `127.0.0.1:8791` | `CORE_CONFIG_PATH` |

The model behind each role comes from a services slot of the example's registry entry: the talker's and the writer's
model from `frontend.cascade.slot` (`llm`), and the backend's default model from `backend.slot` (`thinker-llm`). The
prompts come from the example's `prompts.yaml`: `frontend.cascade.prompt` (`talker`) when a session sets no
instructions, and `backend.prompt` (`thinker`) when a request sets none.

### How Profiles Extend

A profile names the file it builds on with `extends`, a path relative to the profile's own folder, and sets only what
differs. A mapping merges key by key into the one it extends, and any other value replaces it, `null` included, so a
profile can turn a limit off. Both files follow the same rule.

### Environment Variables

The following variables configure the Live routes and the example's files:

| Variable | What it sets |
| --- | --- |
| `LIVE_ENABLED` | Whether the Live routes (`/v1/live`) are served. When it is set, it decides (`1`, `true`, `yes`, or `on` serve them). Otherwise, the routes are served when `LIVE_API_TOKEN` is set or when the deployment runs this example alone. The example's Compose profiles default it to `true`. |
| `LIVE_API_TOKEN` | The bearer token that every Live route requires. Without one, anyone who can reach the server can use the routes, and the server logs a warning at startup. |
| `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS` | The session limits (8 sessions, 1,800 seconds). |
| `LIVE_ICE_URLS` | More ICE servers for the Live gateway's WebRTC peer, comma-separated, beside the TURN server that `TURN_USERNAME`, `TURN_PASSWORD`, and `TURN_URL` configure. |
| `LIVE_MAX_CONSOLE_ROUNDS`, `LIVE_ALLOWED_HOSTS` | The console route's limits: rounds at once (2), and, with no token, the host names besides IP addresses and `localhost` that it answers to. |
| `LIVE_CONFIG_PATH`, `CORE_CONFIG_PATH` | Another `config.yaml` or `core.yaml`, such as a profile. |
| `LIVE_ADAPTER_FACTORY` | Another factory of each session's frontend adapter, as `module:function`: an async function of the session's settings, transport, and WebRTC peer that returns `SessionParts` (`live/ports.py`). `LIVE_ENGINE_FACTORY` is a deprecated alias, read with a warning when `LIVE_ADAPTER_FACTORY` is unset. |
| `UVICORN_WORKERS` | The number of server workers. With more than one, the Live routes are not served, because a session lives in one process. |

When the Live routes are served, a value of `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS`, or
`LIVE_MAX_CONSOLE_ROUNDS` that is not a whole number of at least 1 stops the server at startup. `UVICORN_WORKERS` is
read before that check: for a value that is not a whole number of at least 1, the server logs a warning and runs one
worker.

## Deployment

### Compose Profiles and Processes

Each recipe profile runs one container with one server process and one uvicorn worker. The example has three Compose
profiles, and each one serves the Live routes by default (`LIVE_ENABLED` defaults to `true` in them):

```bash
docker compose --profile frontend-backend-live up -d              # NVIDIA cloud endpoints
docker compose --profile frontend-backend-live/server up -d       # local NIM services on a workstation
docker compose --profile frontend-backend-live/single-gpu up -d   # vLLM and NeMo-Speech.cpp on one GPU
```

The delegation core runs in that process, and the gateway reaches it only through its Responses API:

- The gateway holds one delegation core, built on first use from `core.yaml` (or the file that `CORE_CONFIG_PATH`
  names) by the `live.core` hook. Every Responses-mode session and every console round use it, and
  `GET /v1/live/info` reports its settings. A client-mode session never asks for it.
- When the server shuts down, the gateway closes every session and then the core, which releases the core's backend
  resources, such as the harness's tool bridge.
- A Work ends with its session.
- The repository UI bot builds a core of its own for each bot session, from `core.ui-bot.yaml`, and closes it when the
  session ends.

The server has three ways in for this example:

- The repository UI (`/`) runs the UI bot over Pipecat's RTVI protocol, on WebRTC or WebSocket.
- The Realtime gateway (`WS /v1/realtime`) hands its socket to the same UI bot. Refer to
  [Use the Realtime Gateway](../how-to/use-realtime-gateway.md).
- The Live API (`/v1/live/...`) serves live sessions through the Live gateway, the Live Console, and the console's
  client-mode rounds (`POST /v1/live/delegate`).

The following table lists where each recipe's models run. The talker, the writer, and the backend use the same model
with different settings:

| Recipe | ASR | Talker and backend LLM | TTS |
| --- | --- | --- | --- |
| `frontend-backend-live` | NVIDIA cloud endpoints | NVIDIA cloud endpoint (Nemotron 3.5 Lightning) | NVIDIA cloud endpoints |
| `frontend-backend-live/server` | Nemotron ASR NIM | Nemotron 3.5 Lightning NIM | Magpie TTS NIM |
| `frontend-backend-live/single-gpu` | NeMo-Speech.cpp | Nemotron 3.5 Lightning through vLLM | NeMo-Speech.cpp |

To run another backend adapter, point `CORE_CONFIG_PATH` at its profile. The path is relative to the working
directory: the repository root on the host, and `/app` in the container, where `./src` is mounted. For example,
`CORE_CONFIG_PATH=src/examples/frontend_backend_live/core.agent.yaml` in `.env` runs the backend LLM agent in a Compose
deployment. The harness also needs an agent that you deploy yourself, as the next section describes.

### Connect Your Own Harness

The repository ships the harness adapter and its configuration, not the agent: you deploy the agent, and the core
connects to it through `backend.harness`. The shipped configuration connects to Hermes: the `backend.harness` block of
`core.yaml` names the Hermes agent class (`agent: hermes`) and starts `hermes acp` over stdio, and `core.harness.yaml`
switches the core to the harness.

#### Deploy Hermes

Hermes runs in its own Python environment, outside the app's image. To deploy it:

1. Install Hermes from `https://github.com/nousresearch/hermes-agent` at the tested commit,
   `af26acab737db4382815aadcd73c06755471ab52`, with its `acp` and `mcp` extras, in a virtual environment of its own.
   `hermes acp --check` prints `Hermes ACP check OK`. From commit `eb75627faf` on, `hermes acp` connects to the Hermes
   gateway daemon instead of running the agent itself, so test a later commit before you use it.
2. Give Hermes a home for the voice agent alone, because every session of the core reads the home's identity and
   configuration: the folder that `HERMES_HOME` names, or `~/.hermes` by default. The home holds Hermes's own
   `config.yaml`, its `SOUL.md`, and its model key in the home's `.env` (`NVIDIA_API_KEY` for the `nvidia` provider),
   which Hermes loads itself. The core never passes it a key.
3. Write the home's `SOUL.md`, the identity that opens the system prompt of every Hermes session. Without one, Hermes
   writes its own persona there on its first run ("You are Hermes Agent, built by Nous Research ..."), and every
   conductor and worker session reads that persona beside the core's instructions. Use the following identity, which
   the real-Hermes tests deploy:

   ```text
   You work behind a voice assistant, for its callers. Follow the instructions of your project context exactly, use only the tools you are given, and do not narrate your tool calls.
   ```

4. Configure Hermes in its `config.yaml`. The following example runs the default model of `backend.slot` on NVIDIA's
   cloud endpoint:

   ```yaml
   model:
     provider: nvidia
     default: nvidia/nemotron-3.5-lightning-30b-a3b
     base_url: https://integrate.api.nvidia.com/v1
     context_length: 131072
   platform_toolsets:
     acp: [no_mcp]
   tools:
     tool_search:
       enabled: "off"
   agent:
     coding_context: "off"
     task_completion_guidance: false
     parallel_tool_call_guidance: false
     tool_use_enforcement: false
     execution_guidance: false
   memory:
     memory_enabled: false
     user_profile_enabled: false
   auxiliary:
     title_generation:
       enabled: false
     background_review:
       enabled: false
   auth:
     adopt_external_logins: false
   timeouts:
     mcp:
       tool_call: 3600
   ```

The following table explains why each setting matters to the core:

| Setting | Why |
| --- | --- |
| `model.default` and `model.base_url` | The model that Hermes runs. Run the default model of `backend.slot`, because the core accepts requests for that model only. Keep `base_url`: it pins the route that `context_length` describes. Without it, Hermes drops `context_length` when it runs on a route other than the provider's default, such as a local NIM that `NVIDIA_BASE_URL` names. |
| `model.context_length` | The context window that Hermes compresses its own transcript against. Give it at least 65,536 tokens. |
| `platform_toolsets.acp: [no_mcp]` | Only the session's own MCP server, the core's tool bridge, reaches a session. A list without `no_mcp` merges in every MCP server of the configuration. Toolsets that you add after `no_mcp` reach the conductor too, and the commit gate cannot hold them. Hermes warns that the `acp` platform has no valid toolsets (`unknown name(s): no_mcp`) and that tools will be unavailable. This is expected, because the session's own MCP server still reaches the session. Keep `no_mcp`, and do not save the `acp` platform from `hermes tools`, which removes it. |
| No `mcp_servers` | Hermes starts every MCP server that its home lists when it opens its first session, even with `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, which skips only their start at Hermes's startup. `no_mcp` keeps their tools out of the sessions, but not the servers out of the process, and a server that never connects makes each new session wait up to `mcp_discovery_timeout` (1.5 seconds). |
| `tools.tool_search.enabled: "off"` | The session's tools stay in front of the model instead of behind a search tool. |
| `agent` guidance settings and `coding_context` | No coding-agent brief in the system prompt: the core's instructions are the brief. |
| `memory` and `auxiliary` | No memory between calls, and no hidden model calls to title or review each session. |
| `auth.adopt_external_logins: false` | Hermes never borrows another tool's login that it finds on the host. |
| `timeouts.mcp.tool_call` | Above `delegation.client_tool_timeout_seconds` (60 seconds), so a held app call ends by the core's deadline instead of a Hermes timeout. |

At the tested commit, a Hermes ACP session reads no `agent.max_turns`, so the example sets none, and nothing caps the
model calls of one turn. `backend.harness.turn_timeout_seconds` (300 seconds, not counting waits for the app) cancels a
turn that runs longer.

Hermes keeps each session's transcript in the `state.db` file of its home, which holds the instructions and the
conversation of every call. When a provider error fails a model call, it also writes the whole request, with the
instructions and the conversation, to a `request_dump_*.json` file in the `sessions` folder of its home. Clear both as
your data policy requires.

#### Connect over Stdio

With the `stdio` transport, the core starts `hermes acp` itself: one process for the conductor and one for the workers
of each conversation, and one more when every one of a process's four turns is taken. To connect:

1. Put `hermes` on the server's `PATH`, or set `backend.harness.command` to its full path.
2. If the home of Hermes is not `~/.hermes`, export `HERMES_HOME` in the server's environment. A started process gets
   only the variables that `backend.harness.pass_env` names (`PATH`, `LANG`, `LC_ALL`, `TZ`, and `HERMES_HOME` in
   `core.yaml`), `HERMES_ACP_SKIP_CONFIGURED_MCP=1` from the Hermes agent class, and the values of
   `backend.harness.env`.
3. Point `CORE_CONFIG_PATH` at `src/examples/frontend_backend_live/core.harness.yaml`, and start the server.

#### Connect over WebSocket

To run Hermes on another host or in a container of its own, serve it over WebSocket. For each connection, your server
must do the following:

- Start one `hermes acp` process, with Hermes's own environment and `HERMES_ACP_SKIP_CONFIGURED_MCP=1`, as the Hermes
  agent class gives a process that the core starts.
- Pass each text frame to the process's stdin as one line, and each line of its stdout back as one text frame.
- When the connection closes, end its process: close the process's stdin, and Hermes ends its sessions and exits.
  Terminate the process if it still runs a few seconds later. The core ends a runtime process only by closing its
  connection, so a server that keeps the process leaves the processes of every conversation running.
- When the process exits, close the connection. The core sees that a runtime is lost only when its connection closes.
  A connection left open keeps a running turn waiting until its timeout, and each new session on the process that is
  gone waits `start_timeout_seconds` and fails.

Then point the harness at it in a profile of your own that sets `transport: websocket` together with `url`, such as
the following one. A `url` with the `stdio` transport fails the core's build.

```yaml
extends: core.harness.yaml

backend:
  harness:
    transport: websocket
    url: ws://hermes.internal:8790/acp
    work_root: /var/lib/voiceclaw/harness
    bridge:
      bind: "0.0.0.0:8791"
      advertise: "http://voice-agent.internal:8791"
```

The core waits up to `backend.harness.start_timeout_seconds` (60 seconds) for each connection to open, finish its
WebSocket handshake, and answer `initialize`, and as long again for each `session/new`. Over stdio, the same limit
covers starting the process. `stop_grace_seconds` (5 seconds) bounds only the steps of a close.

Two paths must also reach across:

- Hermes reads each session's instructions from `.hermes.md` in the session's folder, which the core makes under
  `work_root`. Mount one folder at the same path on both sides, and set `work_root` to that path, because the default,
  `~/.cache/voiceclaw/harness`, is a folder on the core's side only. Run Hermes as the same user as the voice-agent
  server, or as root. The core makes each runtime process's work folder with mode 0700 for its own user, so a Hermes
  of another user cannot enter it. Hermes's own image, for example, runs Hermes as the user `hermes`, with UID 10000
  unless its `HERMES_UID` names another.
- Hermes calls the core's tools through the tool bridge. Bind the bridge where Hermes can reach it, and set
  `advertise` to the URL that Hermes uses. A bridge bound on a loopback address, such as the `127.0.0.1` of
  `core.yaml`, answers only requests whose `Host` header names a loopback address. The MCP SDK's protection against
  DNS rebinding answers any other request with 421, and Hermes's MCP connect then fails silently, so every task fails.
  When Hermes reaches the bridge by a name, or through a proxy that keeps the `Host` header, bind a non-loopback
  address, as the example's `0.0.0.0` does. The bridge serves POST only and answers any other method with 405 at once.
  Keep that behavior through any proxy, because Hermes probes the URL with HEAD and GET before it connects.

When Hermes cannot see or enter a session's folder, nothing fails. Hermes logs only the warning
`configured working directory does not exist` and runs the session without its instructions: the conductor without
`texts.conductor_prompt`, and each worker without its composed instructions and the envelope. It can read the project
files of its own launch folder instead, such as an `AGENTS.md` there. After you deploy, run one delegation and check
that Hermes's log has no such warning.

The core adds no credential to the connection, so serve the agent only where the voice agent alone can reach it.
`GET /v1/live/info` shows `backend.harness.url` without its user, password, query, or fragment, because any of them can
carry a credential.

#### Connect Another ACP Agent

An agent that keeps to the Agent Client Protocol needs no code when it does the following, which the plain agent class
relies on:

- It takes an HTTP MCP server with headers in `session/new`, which is an optional capability in ACP. The core gives
  each session one such server, its door to the tool bridge, with a bearer token in the `Authorization` header.
- It connects to that server and lists its tools within `ready_timeout_seconds` (15 seconds) of `session/new`, before
  any prompt.
- It needs no `authenticate` call, because its credentials are its own configuration.

An agent that does not do all three fails every delegation. Connect the agent with its own tools turned off: with
`agent: acp`, every tool card counts as a call of the session's own MCP server, so neither the commit gate nor the
reports see a tool that the agent runs itself. Name the plain agent class, a steering strategy that it can take, and
the way to reach it, as the following profile does:

```yaml
extends: core.harness.yaml

backend:
  harness:
    agent: acp
    steer: next_prompt
    command: [my-agent, --acp]
```

The plain agent class puts the instructions at the top of each session's first prompt, takes no mid-turn steer, and
passes on every message that the agent streams. An agent with ways of its own, such as a file that it reads its
instructions from, a mid-turn steer, or replies of its own that must never be spoken, needs an agent class beside
`HermesAgent`. An agent that does not meet the three conditions needs more than configuration too. Refer to
[Add an Agent, a Protocol, a Transport, or a Steering Strategy to the Harness](#add-an-agent-a-protocol-a-transport-or-a-steering-strategy-to-the-harness).

## Guard Test and Import Rules

Two tests keep the boxes apart and the extension points honest. Both read the code with Python's `ast` module and never
import it.

### The Guard Test

The guard ([`tests/unit/ratchet/guard.py`](../../tests/unit/ratchet/guard.py), run by `test_guard.py`) walks `src/live`,
the example, `src/delegation_core`, and `src/server.py`, and fails on any of four kinds of site:

- `compare`: a kind, such as `.mode`, `.adapter`, `.agent`, or `delegation.type`, compared or matched with a string or
  a constant outside the table modules. `KIND_ATTRIBUTES` and `KIND_NAMES` list the kinds.
- `isinstance`: an `isinstance` or `issubclass` check against a class of another module of the repository, unless the
  class is a port type or an exception, or the check sits in a composition root.
- `flag`: a flag name that `REMOVED_FLAGS` bans, as an attribute, an argument, a keyword, or a field.
- `pick`: a `pick()` from a table outside a composition root and outside a `from_settings`, `from_config`, or `parse`
  method.

Each failure names the file, the line, and the site.

### The Import Rules

[`tests/unit/test_import_boundaries.py`](../../tests/unit/test_import_boundaries.py) puts each module in a group by the
longest prefix of its name, and each group may import only its own modules and the groups it lists. The following
table summarizes the rules:

| Group | May import |
| --- | --- |
| `live.protocol` | Nothing else of the gateway |
| `live.events`, `live.ports` | `live.protocol` |
| `live.server_delegation`, `live.client_delegation` | `live.protocol`, `live.ports` |
| `live.session` | `live.protocol`, `live.events`, `live.ports`, and both delegation parts |
| `live.gateway`, `live.media`, `live.extras`, `live.mount` | Every gateway group |
| The frontend adapter (the example's package, except its roots) | `live.ports`, `live.protocol`, `examples.shared`, and `utils` |
| `delegation_core.api` | `delegation_core.app`, `delegation_core.im`, `delegation_core.config`, and `delegation_core.ids` |
| `delegation_core.app` | `delegation_core.im`, `delegation_core.ledger`, `delegation_core.backend`, the port, `delegation_core.config`, `delegation_core.ids`, and `delegation_core.tables` |
| `delegation_core.im` | `delegation_core.ledger`, `delegation_core.backend.port`, `delegation_core.config`, and `delegation_core.ids` |
| `delegation_core.backend.port` | `delegation_core.config` and `delegation_core.ids` |
| `delegation_core.backend` | The port, the three adapter packages, `delegation_core.config`, `delegation_core.ids`, and `delegation_core.tables` |
| Each adapter package (`backend.llm`, `backend.agent`, `backend.harness`) | The shared backend, the port, `delegation_core.config`, `delegation_core.ids`, and `delegation_core.tables`; `backend.agent` also imports `backend.llm` |
| `delegation_core.ledger` | `delegation_core.ids` |
| `delegation_core.config`, `delegation_core.ids`, `delegation_core.tables` | Nothing else of the core |
| The composition roots | Anything |

The same test checks where outside packages may be imported: Pipecat only in the frontend adapter's Pipecat modules and
the WebRTC peer; `openai` only in the core's providers and the frontend's models; FastAPI and Starlette only at the
HTTP edges; `mcp`, `websockets`, and `acp` only in the harness. Only `delegation_core.backend` may reach a model
provider's `stream()`, so every model call stays inside the backend adapter. Only the agents table,
[`acp/agents.py`](../../src/delegation_core/backend/harness/acp/agents.py), imports an agent class's package, such as
`acp/hermes/`, so no other module depends on one agent. No module of the boxes imports the tests, and the core's own
tests import nothing from the gateway or the example.

## Tests

The tests run offline with scripted models. The following table lists the commands, each run from the repository root:

| What | Command |
| --- | --- |
| Every Live box, the contracts, the guard, the goldens, and the import rules | `uv run pytest tests/unit/live tests/unit/frontend_backend_live tests/unit/delegation_core tests/unit/cross_box tests/unit/contracts tests/unit/ratchet tests/unit/goldens tests/unit/test_import_boundaries.py -q` |
| One box | `uv run pytest tests/unit/delegation_core -q`, and the same for the box's own folder |
| Re-record a golden after an intended behavior change | `LIVE_GOLDEN_RECORD=1 uv run pytest tests/unit/goldens/<test file> -q -p no:cacheprovider` |
| The whole suite, as CI runs it | `uv run pytest tests/unit -v` |
| The Live Console's JavaScript modules, with Node.js 20 or later, as CI runs them | `node --test client/test/live/*.test.js` |

Each box has its own test folder: `tests/unit/live/` for the gateway, `tests/unit/frontend_backend_live/` for the
frontend adapter, the cascade, and the composition roots, and `tests/unit/delegation_core/` for the core, with each
backend adapter under `backend/`. Tests that join boxes sit in `tests/unit/cross_box/`, and the Live Console's tests
sit in `client/test/live/`.

Each port has one contract suite in [`tests/unit/contracts/`](../../tests/unit/contracts/), which the tests of every
implementation bind. The goldens play whole calls through the real gateway, the cascade, and the core, and compare
them with the recordings in `tests/unit/goldens/recordings/`.

The harness also has a module that runs against a real `hermes acp`, deployed the way an operator would deploy it:
`test_harness_real_hermes.py`. It gives Hermes a home of its own, as [Deploy Hermes](#deploy-hermes) describes, on a
scripted model endpoint, and connects to it through the core's settings, so it runs the contracts and the core's flows
with no model cost. It is skipped unless `HARNESS_HERMES_BIN` names the Hermes executable.
`HARNESS_TRANSPORT=websocket` serves Hermes over WebSocket instead of starting it over stdio, and `HARNESS_HERMES_HOME`,
a home that you configured with a real model and its key, adds one run on that model.
