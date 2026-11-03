# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the harness (``port.BackendRun``): one worker session of the agent, under the port's event rules.

**Opening.** The first ``events()`` takes a turn of the workers' process, readies the session's folder with the
instructions the IM composed, opens the session's bridge door and ``session/new``, and requires the agent to list the
door's tools (one retry under a new server name, then the task fails). The first prompt is the task's ``history()`` as
a transcript, after any blocks the agent class puts first. A session the session policy kept
(``backend.harness.session``) takes the task instead when it fits (:meth:`WorkerRun._adopt`), and its prompt is only
the items of ``history()`` that follow the ones it was shown or wrote.

**Rounds.** The agent runs its own model calls; a round is what the IM reads as one, from ``response.created`` to one
terminal. A round ends when a new model message begins after tool activity (a round's calls come after its messages);
when an app call is ready, after every other call in flight has reported (one app call per round, and ``route()`` with
its commit gate runs when the call is forwarded); when a native steer was delivered, or steering waits while the round
holds calls, so no output joins the history before its call; or when every prompt of the session has returned.

**Answer.** The answer is the last message after the turn's last tool activity, only when the turn ended ``end_turn``,
``Agent.no_answer`` finds it answered and ``Agent.model_text`` says a model call wrote it. Anything else fails the task
(``response.failed``, ``server_error``) and the text is only logged. A message no model call wrote never becomes
commentary either, since commentary can be spoken. Usage is how far the session's counters moved, since ACP counts
usage across all of a session's turns.

**Steering.** An agent's own mid-turn steer joins the running turn at its next model boundary, so delivery is reported
``applied`` once a tool result came after the ack and a new model call began, or a new model message began while the
session's transcript held a tool row. A batch not taken by the turn's end goes as the next prompt. A steer that reaches
an idle session runs as a turn of its own, so the run reads until every prompt has returned.
A turn that answered ends the task even while steering waits: each batch it did not take goes back to the IM
``not_applied``, and the IM starts it as the next request; only a turn without an answer goes on with it.

**Stops** are level-triggered and never cleared: ``session/cancel`` only while a prompt is outstanding, the door
retired, then a wait for every prompt to end within ``stop_timeout_seconds`` (``stopped``, else ``unconfirmed``); the
session is never used again. A turn past ``turn_timeout_seconds`` of reading time is cancelled the same way and fails
the task. Only the session of a task that answered, with nothing of the task left in it, is ever kept
(:meth:`WorkerRun._kept`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from delegation_core.backend.port import BoundaryInput, Steering, StopOutcome, TaskContext
from delegation_core.ids import uid

from .bridge import WorkerDoor
from .protocol import (
    AppCall,
    CallRunning,
    CallSettled,
    HarnessSession,
    HeldCall,
    Mailbox,
    MessageChunk,
    PlanUpdate,
    PromptEnded,
    Runtime,
    SessionEvent,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
    UserText,
    Wake,
)
from .runtime import give_back
from .sessions import KeptSession, keys

if TYPE_CHECKING:
    from .adapter import HarnessAdapter

# Tries to open a session whose door the agent lists: one more under a new server name.
OPEN_ATTEMPTS = 2
# How much of a refused turn's text the log keeps.
LOGGED_TEXT_CHARS = 300


class HarnessTurnError(RuntimeError):
    """The agent's turn gave no answer the run can pass on (its text is logged, never spoken)."""


def usage_total(usage: Mapping[str, Any] | None) -> int | None:
    """The session's token total an agent reported (ACP ``totalTokens``), or ``None``."""
    if not usage:
        return None
    total = usage.get("totalTokens")
    return total if isinstance(total, int) else None


def output_text(item: Mapping[str, Any] | None) -> str | None:
    """The text of a ``function_call_output``: its output as given, or as JSON when it is not text."""
    if item is None:
        return None
    output = item.get("output")
    return output if isinstance(output, str) else json.dumps(output)


@dataclass
class _Message:
    """One assistant message of the agent: its text so far, and how it streamed (``protocol.MessageFacts``)."""

    id: str
    parts: list[str] = field(default_factory=list)
    chunks: int = 0
    thought: bool = False  # a reasoning chunk carried its id
    after_text: bool = False  # the agent's turn had streamed text before this message began

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()


@dataclass
class _SentSteer:
    """A batch sent as the agent's own mid-turn steer, until its delivery is known."""

    steering: Steering
    prompt: int
    acked: bool = False  # its steer returned while the turn ran: the agent holds it for the turn
    ack_message: str | None = None  # the model message running when it was acked
    tool_rows_at_ack: bool = False
    result_after_ack: bool = False  # a tool result came after the ack
    settled: bool = False


@dataclass
class _Turn:
    """One turn of the session: from a prompt until every prompt of the session has returned."""

    main: int
    total_before: int | None
    usage_before: Mapping[str, Any] | None = None
    started: float = field(default_factory=time.monotonic)
    used: float = 0.0  # seconds spent reading it (waiting for the app does not count)
    main_open: bool = True
    stop: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    activity_after_main: bool = False  # model activity after the main prompt returned (a steer ran as a turn)
    cancelled_by_us: bool = False
    seen: bool = False  # the agent reported anything of the turn: its prompt holds the session
    early: list[Steering] = field(default_factory=list)  # native steers waiting for that, so none races the prompt


class _TimedOut:
    """The turn ran past its limit (the run's own event)."""

    def apply(self, reader: WorkerRun) -> None:
        """Hand the event to ``reader``."""
        reader.timed_out()


class WorkerRun:
    """One task's run on the harness (``port.BackendRun``)."""

    def __init__(self, task: TaskContext, host: HarnessAdapter):
        """Run ``task`` on ``host``'s agent; nothing is opened before :meth:`events`."""
        self.task, self.host = task, host
        self.settings, self.agent = host.settings, host.settings.agent
        self.round = 0
        self.done = False  # the task answered or failed: events() yields nothing more
        self.answered = False  # the task answered
        self.stopped = False  # level: set by stop(), never cleared
        self.waits = False  # the last round ended with an app call: the iteration ends until resume()
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: WorkerDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.turn: _Turn | None = None
        self.instructions = ""  # the composed instructions the session was opened with
        self.shown: list[tuple] = []  # the keys of every item the session was shown or wrote, in the IM's order
        # The prompts sent whose end the run has not read yet: the turn is over when none is left, in mailbox order
        # (the session's own outstanding set runs ahead of the mailbox).
        self.open_prompts: set[int] = set()
        self.total: int | None = None  # the session's cumulative token total, as last reported
        self.usage: Mapping[str, Any] | None = None  # the session's cumulative usage, as last reported
        self.tool_rows = False  # the session's transcript holds a tool result (a steer is then taken sooner)
        self.text_streamed = False  # the agent's current turn streamed text so far
        self.current_message: str | None = None  # the newest model message's id
        self.sent: dict[int, _SentSteer] = {}  # native steers by their prompt number
        self.next_prompts: list[Steering] = []  # batches that go as the next prompt
        self.reprompts: list[Steering] = []  # batches that wait for a cancel and a fresh session
        self.app_queue: deque[HeldCall] = deque()  # app calls held, not forwarded yet
        self.forwarded: list[HeldCall] = []  # the app call the last round forwarded, answered at resume()
        self.running: set[str] = set()  # calls the bridge runs now
        self.cards_open: set[str] = set()  # tool cards announced and not ended
        self.runtime_calls: dict[str, dict] = {}  # the agent's own tool calls (not the bridge's), by card id
        self.boundary: Callable[[], AsyncIterator[dict]] | None = None
        self.pending: SessionEvent | _TimedOut | None = None
        self.delivering: list[_SentSteer] = []
        self.response_id = ""
        self._stopping: asyncio.Future | None = None
        self._new_round()

    # ------------------------------------------------------------------------------------------------ the port
    async def events(self) -> AsyncIterator[dict]:
        """The task's Responses events, round by round, by the port's event rules; called again after :meth:`resume`.

        Raises:
            HarnessTurnError: The turn gave no answer the run can pass on, or ran past its limit.
            Exception: The session could not be opened (the agent did not start or did not list the tools).
        """
        self.waits = False
        if self.done or self.stopped:
            return
        try:
            if self.session is None:
                await self._open()
            while not (self.done or self.stopped or self.waits):
                async for event in self._round():
                    yield event
        except BaseException:
            self.done = not self.stopped
            await self.release()
            raise
        if self.done:
            await self.release()

    def steer(self) -> None:
        """Steering waits: wake the reader, which takes it at once (closing the open round first)."""
        if not (self.done or self.stopped):
            self.mailbox.put_nowait(Wake())

    async def resume(self) -> None:
        """The app's outputs are in: deliver waiting steering first, then answer the held call with its output.

        A native steer waits for its ack before the held call is answered, so the agent holds the steer when the tool
        batch ends and puts it after the results.
        """
        self.task.trace("tools.completed", round=self.round)
        boundary = self._boundary()
        outputs = {item.get("call_id"): item for item in boundary.outputs}
        before = set(self.sent)
        steering = boundary.steering
        while steering is not None:
            self.settings.steering.deliver(self, steering)
            steering = self._boundary().steering
        steers = [number for number in self.sent if number not in before]
        if steers and self.session is not None:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.settings.ready_timeout_seconds):
                    await asyncio.gather(*(self.session.ended(number) for number in steers))
        for held in self.forwarded:
            held.reply(output_text(outputs.get(held.call_id)) or self.task.texts.owed_output)
        self.forwarded = []

    async def stop(self) -> StopOutcome:
        """Stop the task: cancel while a prompt is outstanding, retire the door and the session, wait for the end.

        Returns ``stopped`` once no prompt of the session runs, or ``unconfirmed`` at ``stop_timeout_seconds``.
        """
        if self._stopping is None:
            self._stopping = asyncio.ensure_future(self._stop())
        return await asyncio.shield(self._stopping)

    async def _stop(self) -> StopOutcome:
        self.stopped = True
        self.mailbox.put_nowait(Wake())
        outcome: StopOutcome = "stopped"
        session = self.session
        if session is not None and session.outstanding:
            await session.cancel()
        self._close_door(self.task.texts.stopped_output)
        if session is not None:
            try:
                async with asyncio.timeout(self.settings.stop_timeout_seconds):
                    await session.idle()
            except TimeoutError:
                outcome = "unconfirmed"
        await self.release()
        self.task.trace("backend.stopped", round=self.round, outcome=outcome)
        return outcome

    async def release(self) -> None:
        """Keep the session for a later task when the session policy takes it; else retire the door and the session.

        Either way the run's turn of its runtime process is given back: a kept session holds none while it waits. Safe
        to call twice: the second call finds no runtime.
        """
        kept = self._kept()
        if kept is not None and self.settings.sessions.keep(self.host, kept):
            self.runtime = self.session = self.door = None
            kept.runtime.leave()
            self.task.trace("harness.session_kept", server=kept.door.name, session=kept.session.id)
            return
        self._close_door(self.task.texts.stopped_output)
        session = self.session
        if session is not None:
            session.retire()
        runtime, self.runtime = self.runtime, None
        if runtime is not None:
            give_back(runtime, session)

    def _close_door(self, text: str) -> None:
        """Close the session's door; the app calls it holds end with ``text``."""
        if self.door is not None:
            self.host.bridge.close(self.door, text)

    def _kept(self) -> KeptSession | None:
        """The session as a later task may take it: only once its task answered and nothing of the task is left in it.

        No stopped, failed or timed-out task's session qualifies, nor one with a prompt, a batch or a call of the task
        still open, or whose client has ended. Nor does one sent a cancel: its turn may answer before the cancel
        reaches the agent, and an idle agent takes a cancel for its next prompt (Hermes ends that one unprocessed).
        """
        session, door, runtime = self.session, self.door, self.runtime
        if session is None or door is None or runtime is None or not self.answered or self.stopped:
            return None
        if self.turn is not None and self.turn.cancelled_by_us:
            return None
        left = (session.outstanding, self.reprompts, self.next_prompts, self.sent, self.forwarded, self.app_queue)
        if any(left) or self.running or self.cards_open or door.held or not runtime.client.alive:
            return None
        shown = list(self.shown)
        return KeptSession(
            runtime, session, door, self.mailbox, self.instructions, shown, self.total, self.usage, self.tool_rows
        )

    # ------------------------------------------------------------------------------------------------ opening
    async def _open(self) -> None:
        """Take the session the session policy kept when it fits, else open one (the module's docstring); prompt it."""
        kept = self.settings.sessions.take(self.host)
        if kept is not None and self._adopt(kept):
            return
        self.runtime = await self.host.shared(self.host.workers)
        await self._open_session(self.task.request().get("instructions") or "")

    def _adopt(self, kept: KeptSession) -> bool:
        """Take over ``kept`` when it fits, and prompt it with the items it has not read; else retire it (``False``).

        It fits when its client is alive, it was opened with this task's instructions, its process has a turn free, its
        door serves this task's toolset (:meth:`WorkerDoor.rebind`), and this task's ``history()``, steering that came
        before the prompt included, begins with exactly the items it was shown or wrote, with more after them. The
        session's usage counters carry over, since ACP counts usage across all of a session's turns.
        """
        instructions = self.task.request().get("instructions") or ""
        fits = kept.runtime.client.alive and kept.instructions == instructions
        if not (fits and self.host.take_place(kept.runtime)):
            kept.retire(self.host.bridge)
            return False
        if not kept.door.rebind(self.task, self.host.tools):
            return self._retire_kept(kept)
        try:
            history = self.task.history()
            # Steering joins at the end: it is applied only if the history before it begins with what the session read.
            if keys(history)[: len(kept.shown)] == kept.shown:
                self._apply_before_prompt([])
                history = self.task.history()
        except BaseException:  # the task fails (its history is over the budget): the session leaves nothing behind
            self._retire_kept(kept)
            raise
        shown = keys(history)
        if shown[: len(kept.shown)] != kept.shown or len(shown) == len(kept.shown):
            return self._retire_kept(kept)
        self.runtime, self.session, self.door, self.mailbox = kept.runtime, kept.session, kept.door, kept.mailbox
        while not self.mailbox.empty():  # what reached it after its task's run ended: nothing of this task
            self.mailbox.get_nowait()
        self.instructions, self.shown = instructions, shown
        self.total, self.usage, self.tool_rows = kept.total, kept.usage, kept.tool_rows
        rest = history[len(kept.shown) :]
        self.task.trace("harness.session_reused", server=kept.door.name, session=kept.session.id, items=len(rest))
        self._start_turn(self.session.prompt(self.host.renderer.transcript(rest)))
        return True

    def _retire_kept(self, kept: KeptSession) -> bool:
        """A kept session that does not fit gives back the place it took and is retired; ``False``."""
        kept.runtime.leave()
        kept.retire(self.host.bridge)
        return False

    async def _open_session(self, instructions: str, batches: tuple[Steering, ...] = ()) -> None:
        """A fresh session in the run's runtime, whose first prompt is the history with ``batches`` applied."""
        runtime, started = self.runtime, time.monotonic()
        for attempt in range(1, OPEN_ATTEMPTS + 1):
            name = runtime.server_name(self.host.workers.prefix)
            mailbox: Mailbox = asyncio.Queue()
            door = WorkerDoor(
                name,
                self.task,
                self.host.tools,
                mailbox,
                tool_timeout_seconds=self.settings.tool_timeout_seconds,
            )
            server, session = self.host.bridge.open(door), None
            try:
                folder = runtime.session_folder(name)
                blocks = self.agent.prepare(folder, instructions)
                session = await runtime.client.new_session(folder, server, mailbox)
                listed = await self._listed(door)
            except BaseException:  # a failed or cancelled opening leaves no door and no session behind
                if session is not None:
                    session.retire()
                self.host.bridge.close(door, self.task.texts.stopped_output)
                raise
            if listed:
                break
            session.retire()
            self.host.bridge.close(door, self.task.texts.stopped_output)
            logger.warning(f"harness.session_not_ready task={self.task.task_id} server={name} attempt={attempt}")
        else:
            raise RuntimeError("The agent did not list the session's tools: its MCP connection to the bridge failed")
        self.mailbox, self.door, self.session = mailbox, door, session
        self.total, self.usage = 0, None  # a fresh session's counters start at 0 (ACP: usage across all turns)
        self.instructions = instructions
        self.task.trace(
            "harness.session_opened", server=name, session=session.id, seconds=round(time.monotonic() - started, 3)
        )
        if self.stopped:
            await self.release()
            return
        self._apply_before_prompt(list(batches))
        history = self.task.history()
        self.shown = keys(history)
        text = "\n\n".join([*blocks, self.host.renderer.worker_input(history)])
        self._start_turn(session.prompt(text))

    def _apply_before_prompt(self, batches: list[Steering]) -> None:
        """Before a fresh session's first prompt, settle ``batches`` and any steering that waits ``applied``.

        The owed outputs join first, then each batch (one that came while the session opened too), so the prompt's
        history carries them in that order.
        """
        for steering in self._collect(batches):
            self._applied(steering)
            self.task.trace("harness.steer_applied", round=self.round, steering=steering.id, how="first_prompt")

    async def _listed(self, door: WorkerDoor) -> bool:
        try:
            async with asyncio.timeout(self.settings.ready_timeout_seconds):
                await door.listed.wait()
            return True
        except TimeoutError:
            return False

    def _start_turn(self, prompt: int) -> None:
        self.open_prompts.add(prompt)
        self.turn = _Turn(prompt, self.total, self.usage)
        self.text_streamed = False
        self.task.trace("harness.prompt_sent", round=self.round, prompt=prompt)

    # ------------------------------------------------------------------------------------------------ a round
    def _new_round(self) -> None:
        self.messages: dict[str, _Message] = {}  # the round's messages, in order
        self.calls: list[dict] = []  # the round's calls, in order
        self.activity = False  # tool activity in this round
        self.activity_since_message = False  # tool activity since the newest message began

    async def _round(self) -> AsyncIterator[dict]:
        """One round: ``response.created``, the session's events up to a boundary, then the boundary's ending."""
        self.round += 1
        self._new_round()
        self.response_id = uid("resp")
        yield {"type": "response.created", "response": {"id": self.response_id, "status": "in_progress", "output": []}}
        self._ready_app()  # an app call held while the last round ended goes now
        self._cancel_for_reprompt()
        while self.boundary is None:
            event, self.pending = self.pending, None
            if event is None:
                event = await self._next()
            event.apply(self)
        finish, self.boundary = self.boundary, None
        async for event in finish():
            yield event

    async def _next(self) -> SessionEvent | _TimedOut:
        """The session's next event; a turn past its limit gives its own event."""
        turn = self.turn
        if turn is None:
            return await self.mailbox.get()
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(self.settings.turn_timeout_seconds - turn.used, 0.001)):
                return await self.mailbox.get()
        except TimeoutError:
            return _TimedOut()
        finally:
            turn.used += time.monotonic() - started

    def _has_items(self) -> bool:
        return bool(self.calls) or any(message.text for message in self.messages.values())

    async def _close(self, final: _Message | None, usage: Mapping[str, Any] | None = None) -> AsyncIterator[dict]:
        """End the open round: its messages (``final`` is the answer, the rest commentary), then its calls.

        A message no model call wrote is the agent's own text: commentary may be spoken as progress, so it is dropped
        and logged, never passed on.
        """
        items = []
        for message in self.messages.values():
            if message.text and not self.agent.model_text(message):
                logger.warning(
                    f"harness.text_dropped task={self.task.task_id} text={message.text[:LOGGED_TEXT_CHARS]!r}"
                )
                self.task.trace("harness.text_dropped", round=self.round)
            elif message.text:
                phase = "final_answer" if message is final else "commentary"
                items.append(
                    {
                        "id": uid("msg"),
                        "type": "message",
                        "status": "completed",
                        "role": "assistant",
                        "phase": phase,
                        "content": [{"type": "output_text", "text": message.text, "annotations": []}],
                    }
                )
        items += [dict(call) for call in self.calls]
        self.shown += keys(items)
        for index, item in enumerate(items):
            opening = {**item, "status": "in_progress"}
            if item["type"] == "message":
                opening["content"] = []
            yield {"type": "response.output_item.added", "output_index": index, "item": opening}
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        response = {"id": self.response_id, "status": "completed", "output": items}
        if usage:
            response["usage"] = usage
        yield {"type": "response.completed", "response": response}
        self._new_round()

    async def _fail(self, reason: str, text: str = "") -> AsyncIterator[dict]:
        """End the round ``response.failed`` and raise: the task failed; the agent's text is only logged."""
        logger.warning(
            f"harness.turn_refused task={self.task.task_id} reason={reason!r} text={text[:LOGGED_TEXT_CHARS]!r}"
        )
        self.task.trace("harness.turn_refused", round=self.round, reason=reason)
        error = {"code": "server_error", "message": reason}
        yield {
            "type": "response.failed",
            "response": {"id": self.response_id, "status": "failed", "output": [], "error": error},
        }
        self.done = True
        raise HarnessTurnError(reason)

    def _between_rounds(self) -> None:
        """A safe point: outputs join the history, and every waiting steering batch goes to the strategy."""
        steering = self._boundary().steering
        while steering is not None:
            self.settings.steering.deliver(self, steering)
            steering = self._boundary().steering

    # ------------------------------------------------------------------------------------------------ endings
    async def _finish_next_call(self) -> AsyncIterator[dict]:
        """A new model call began after tool activity: the round so far is commentary and calls."""
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_woken(self) -> AsyncIterator[dict]:
        """Steering waits: close the round, then take it."""
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_delivered(self) -> AsyncIterator[dict]:
        """A native steer was delivered: close the round, then settle it, so it joins after what came before."""
        async for event in self._close(None):
            yield event
        self._settle_delivered()

    async def _finish_app(self) -> AsyncIterator[dict]:
        """An app call goes to the app: the round ends with it, and the task waits for its output."""
        async for event in self._close(None):
            yield event
        self.waits = True
        self.task.trace("tools.waiting", round=self.round, call_ids=[held.call_id for held in self.forwarded])

    async def _finish_stopped(self) -> AsyncIterator[dict]:
        """Stopped: nothing more is yielded."""
        return
        yield

    async def _finish_timeout(self) -> AsyncIterator[dict]:
        """The turn ran past ``turn_timeout_seconds``: cancel it while it runs, and fail the task."""
        if self.session is not None and self.session.outstanding:
            await self.session.cancel()
        async for event in self._fail(f"the turn ran past {self.settings.turn_timeout_seconds:g} s"):
            yield event

    async def _finish_turn(self) -> AsyncIterator[dict]:
        """Every prompt has returned: the answer, the next prompt, a fresh session, or a failure."""
        turn = self.turn
        if self.stopped:
            return
        failure = self._turn_failure(turn)
        candidate = self._candidate()
        reason = failure or self._unparsed(turn, candidate)
        if reason is None:  # the answer, also while steering waits: what the turn did not take goes back to the IM
            self._leave_unapplied(turn)
            self.task.trace("harness.answered", round=self.round, seconds=round(time.monotonic() - turn.started, 3))
            async for event in self._close(candidate, self._turn_usage(turn)):
                yield event
            self.done = self.answered = True
            return
        if self.reprompts and (turn.cancelled_by_us or turn.stop == "end_turn"):
            async for event in self._close(None):
                yield event
            await self._reopen()
            return
        if failure is not None:
            async for event in self._fail(failure, self._last_text()):
                yield event
        if not self.calls:
            self._between_rounds()  # safe: the open round holds no call whose output could join before it
        self.next_prompts += turn.early  # native steers that never saw the turn begin go as the next prompt
        turn.early = []
        waiting = self.next_prompts or any(not sent.settled for sent in self.sent.values())
        if waiting:
            async for event in self._close(None):
                yield event
            self._prompt_next_now()
            return
        async for event in self._fail(reason, self._last_text()):
            yield event

    def _leave_unapplied(self, turn: _Turn) -> None:
        """The turn answered: every batch it took and did not apply goes back to the IM ``not_applied``, once."""
        batches = [*self.next_prompts, *turn.early, *self.reprompts]
        batches += [sent.steering for sent in self.sent.values() if not sent.settled]
        self.next_prompts, self.reprompts, self.sent, turn.early = [], [], {}, []
        for steering in batches:
            self.task.settle(steering, "not_applied")
        if batches:
            self.task.trace("harness.steer_left", round=self.round, batches=len(batches))

    def _turn_failure(self, turn: _Turn) -> str | None:
        if turn.error is not None:
            return f"the agent's prompt failed: {turn.error}"
        if turn.stop != "end_turn":
            return f"the agent ended the turn {turn.stop}"
        return None

    def _candidate(self) -> _Message | None:
        """The last message, unless tool activity followed it (then the turn ended without an answer)."""
        if self.activity_since_message or not self.messages:
            return None
        return list(self.messages.values())[-1]

    def _unparsed(self, turn: _Turn, candidate: _Message | None) -> str | None:
        """Why the turn's last message is not an answer the run may pass on; ``None`` when it is."""
        if candidate is None or not candidate.text:
            return "the turn ended without a message after its last tool call"
        reason = self.agent.no_answer(usage_total(turn.usage), turn.total_before)
        if reason is not None:
            return reason
        if not self.agent.model_text(candidate):
            return "the last message was not streamed by a model call (the agent's own text)"
        return None

    def _last_text(self) -> str:
        texts = [message.text for message in self.messages.values() if message.text]
        return texts[-1] if texts else ""

    def _turn_usage(self, turn: _Turn) -> dict | None:
        """The turn's own usage in the Responses shape: how far the session's running counters moved."""
        if not turn.usage:
            return None
        before = turn.usage_before or {}
        keys = (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"), ("total_tokens", "totalTokens"))
        return {name: max(int(turn.usage.get(key) or 0) - int(before.get(key) or 0), 0) for name, key in keys}

    # ------------------------------------------------------------------------------------------------ prompts
    def _collect(self, batches: list[Steering]) -> list[Steering]:
        """A safe point (the round closed): the owed outputs join the history; every batch that waits joins ``batches``.

        So a batch settled next joins after the outputs of the calls before it, in the agent's own order.
        """
        steering = self._boundary().steering
        while steering is not None:
            batches.append(steering)
            steering = self._boundary().steering
        return batches

    def _boundary(self) -> BoundaryInput:
        """A safe point (``TaskContext.take_boundary_input``): the outputs that join are items the session has read."""
        boundary = self.task.take_boundary_input()
        self.shown += keys(boundary.outputs)
        return boundary

    def _applied(self, steering: Steering) -> None:
        """Settle ``steering`` ``applied``: its items join the history, and the session has read them."""
        self.task.settle(steering, "applied")
        self.shown += keys(steering.items)

    def _prompt_next_now(self) -> None:
        """Send every batch the turn did not take as the next prompt; each is ``applied`` once sent."""
        batches = [*self.next_prompts, *(sent.steering for sent in self.sent.values() if not sent.settled)]
        self.next_prompts, self.sent = [], {}
        batches = self._collect(batches)
        text = "\n\n".join(self.host.renderer.steering(steering) for steering in batches)
        prompt = self.session.prompt(text)
        for steering in batches:
            self._applied(steering)
        self.task.trace("harness.steer_prompted", round=self.round, batches=len(batches))
        self._start_turn(prompt)

    async def _reopen(self) -> None:
        """After a cancel for steering: a fresh session, its first prompt the history with the batches."""
        batches, self.reprompts = self.reprompts, []
        self._close_door(self.task.texts.not_run_output)
        if self.session is not None:
            self.session.retire()
        self.sent, self.next_prompts, self.app_queue, self.forwarded = {}, [], deque(), []
        self.open_prompts = set()
        self.running, self.cards_open, self.runtime_calls = set(), set(), {}
        self.tool_rows, self.current_message = False, None
        await self._open_session(self.task.request().get("instructions") or "", tuple(batches))

    # ------------------------------------------------------------------------------------------------ Steerable
    def steer_in_turn(self, steering: Steering) -> None:
        """While the turn's prompt runs, the agent's own steer; otherwise the next prompt.

        A steer sent before the agent reported anything of the turn could reach it before the prompt holds the
        session, and run as a prompt of its own; it waits for the turn's first event instead.
        """
        turn = self.turn
        if self.session is None or turn is None or not turn.main_open or self.stopped:
            self.prompt_next(steering)
        elif not turn.seen:
            turn.early.append(steering)
        else:
            self._steer(steering)

    def _steer(self, steering: Steering) -> None:
        prompt = self.session.steer(self.host.renderer.steering(steering))
        self.open_prompts.add(prompt)
        self.sent[prompt] = _SentSteer(steering, prompt)
        self.task.trace("harness.steer_sent", round=self.round, steering=steering.id)

    def _seen(self) -> None:
        """The agent reported something of the turn: native steers that waited for it go now."""
        turn = self.turn
        if turn is None or turn.seen:
            return
        turn.seen = True
        early, turn.early = turn.early, []
        for steering in early:
            self.settings.steering.deliver(self, steering)

    def prompt_next(self, steering: Steering) -> None:
        """The batch goes as the next prompt, once the turn ends."""
        self.next_prompts.append(steering)

    def reprompt(self, steering: Steering) -> None:
        """Cancel the turn once no bridged call runs; then a fresh session carries the batch."""
        self.reprompts.append(steering)
        self._cancel_for_reprompt()

    def _cancel_for_reprompt(self) -> None:
        turn, session = self.turn, self.session
        if not self.reprompts or turn is None or turn.cancelled_by_us or session is None:
            return
        if self.running or self.cards_open or self.forwarded or not session.outstanding:
            return
        turn.cancelled_by_us = True
        task = asyncio.ensure_future(session.cancel())
        task.add_done_callback(lambda done: done.exception())

    # ------------------------------------------------------------------------------------------------ steer delivery
    def _acked(self) -> list[_SentSteer]:
        return [sent for sent in self.sent.values() if sent.acked and not sent.settled]

    def _delivered_by(self, message_id: str | None) -> list[_SentSteer]:
        """The acked steers the model output now beginning proves delivered (see the module's docstring)."""
        delivered = []
        for sent in self._acked():
            new_call = message_id is not None and message_id != sent.ack_message
            if sent.result_after_ack or (new_call and (sent.tool_rows_at_ack or self.tool_rows)):
                delivered.append(sent)
        return delivered

    def _settle_delivered(self) -> None:
        """Settle the delivered steers ``applied`` once the outputs of the calls before them joined (a safe point).

        The agent takes a steer after the tool batch's results, so the task's history has the outputs first; no call of
        the open round is waiting for its output here (the round closed, or it holds no call).
        """
        delivering, self.delivering = self.delivering, []
        self._between_rounds()
        for sent in delivering:
            if not sent.settled:
                sent.settled = True
                self._applied(sent.steering)
                self.task.trace("harness.steer_applied", round=self.round, steering=sent.steering.id)
        self.sent = {prompt: sent for prompt, sent in self.sent.items() if not sent.settled}

    def _model_output(self, event: SessionEvent, message_id: str | None) -> bool:
        """Model output begins; return whether a boundary waits for it (the event is read again in the next round)."""
        delivered = self._delivered_by(message_id)
        if self.turn is not None and not self.turn.main_open:
            self.turn.activity_after_main = True
        new_message = message_id is not None and message_id not in self.messages
        if delivered:
            self.delivering = delivered
            if self._has_items():
                self.boundary, self.pending = self._finish_delivered, event
                return True
            self._settle_delivered()
        if new_message and self.activity and self._has_items():
            self.boundary, self.pending = self._finish_next_call, event
            return True
        return False

    def _begin_message(self, message_id: str) -> _Message:
        message = self.messages.get(message_id)
        if message is None:
            message = self.messages[message_id] = _Message(message_id, after_text=self.text_streamed)
            self.activity_since_message = False
        self.current_message = message_id
        return message

    # ------------------------------------------------------------------------------------------------ EventReader
    def message_chunk(self, event: MessageChunk) -> None:
        """Model text; a chunk with no id is the agent's own reply (an ack): never output."""
        self._seen()
        if event.message_id is None or self._model_output(event, event.message_id):
            return
        message = self._begin_message(event.message_id)
        message.parts.append(event.text)
        message.chunks += 1
        self.text_streamed = self.text_streamed or bool(event.text)

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Reasoning: its id marks the message as streamed by a model call."""
        self._seen()
        if event.message_id is None or self._model_output(event, event.message_id):
            return
        self._begin_message(event.message_id).thought = True

    def tool_card(self, event: ToolCard) -> None:
        """A tool call starts: the bridged ones are the bridge's to report; the agent's own are reported here."""
        self._seen()
        if self._model_output(event, None):
            return
        self.activity = self.activity_since_message = True
        self.cards_open.add(event.call_id)
        if event.bridged:
            return
        call = {
            "type": "function_call",
            "id": uid("fc"),
            "call_id": event.call_id,
            "name": event.name,
            "arguments": json.dumps(dict(event.arguments or {})),
            "status": "completed",
        }
        if self.task.route(call) == "refused":  # the agent runs its own tools: the gate cannot hold them
            logger.warning(f"harness.runtime_tool_not_held task={self.task.task_id} name={event.name}")
        self.task.tool_started(call, self.settings.runtime_label)
        self.runtime_calls[event.call_id] = call
        self.calls.append(call)

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """A tool call ended: a tool row now exists; the agent's own call reports its outcome."""
        self._seen()
        self.activity = self.activity_since_message = True
        self.tool_rows = True
        self.cards_open.discard(event.call_id)
        for sent in self._acked():
            sent.result_after_ack = True
        call = self.runtime_calls.pop(event.call_id, None)
        if call is not None:
            self.task.tool_outcome(event.call_id, event.output, "failed" if event.failed else "done")
        self._ready_app()
        self._cancel_for_reprompt()

    def plan(self, event: PlanUpdate) -> None:
        """The agent's plan: a progress note (never history)."""
        self._seen()
        if event.entries:
            self.task.progress("; ".join(event.entries))

    def user_text(self, event: UserText) -> None:
        """The agent drains or replays a prompt of its own: a boundary between model calls."""
        self._seen()
        self.activity = self.activity_since_message = True

    def prompt_ended(self, event: PromptEnded) -> None:
        """A prompt returned: a steer's ack, the turn's end, or a steer that ran as a turn of its own."""
        turn = self.turn
        total = usage_total(event.usage)
        if total is not None:
            self.total, self.usage = total, event.usage
        sent = self.sent.get(event.prompt)
        if sent is not None and turn is not None:
            self._steer_ended(sent, turn, event)
        if turn is not None and event.prompt == turn.main:
            turn.main_open, turn.stop, turn.error = False, event.stop_reason, event.error
            turn.usage = event.usage
            self.text_streamed = False  # model text from now on is a turn of the agent's own (a steer run as one)
            self.task.trace("harness.turn_ended", round=self.round, stop=event.stop_reason, error=event.error)
        elif turn is not None and event.usage is not None and turn.activity_after_main:
            turn.stop, turn.error, turn.usage = event.stop_reason, event.error, event.usage
        self.open_prompts.discard(event.prompt)
        if not self.open_prompts and turn is not None and not turn.main_open:
            self.boundary = self._finish_turn

    def _steer_ended(self, sent: _SentSteer, turn: _Turn, event: PromptEnded) -> None:
        if event.error is not None:  # it never reached the agent: it goes as the next prompt
            return
        if turn.main_open:  # it returned while the turn ran: the agent holds it for this turn
            sent.acked, sent.ack_message, sent.tool_rows_at_ack = True, self.current_message, self.tool_rows
            self.task.trace("harness.steer_acked", round=self.round, steering=sent.steering.id)
        elif turn.activity_after_main and not sent.settled:  # it ran as a turn of its own after the turn ended
            sent.settled = True
            self._applied(sent.steering)
            self.task.trace("harness.steer_applied", round=self.round, steering=sent.steering.id, how="turn")

    def app_call(self, event: AppCall) -> None:
        """The bridge holds an app call: it goes to the app once nothing else of the batch runs."""
        self._seen()
        self.activity = self.activity_since_message = True
        self.app_queue.append(event.held)
        self._ready_app()

    def call_running(self, event: CallRunning) -> None:
        """The bridge took a call: refused (not run) or run by the bridge; either is a call of this round."""
        self._seen()
        self.activity = self.activity_since_message = True
        self.calls.append(event.call)
        if not event.refused:
            self.running.add(event.call["call_id"])

    def call_settled(self, event: CallSettled) -> None:
        """A call the bridge ran has reported."""
        self.running.discard(event.call_id)
        self._ready_app()
        self._cancel_for_reprompt()

    def wake(self, event: Wake) -> None:
        """A stop ends the round with nothing; waiting steering is taken at once, after the round if it has calls.

        Only calls force the round to close first (their outputs must join after them); a message still streaming
        is not cut in two.
        """
        if self.stopped:
            self.boundary = self._finish_stopped
        elif self.calls:
            self.boundary = self._finish_woken
        else:
            self._between_rounds()

    def timed_out(self) -> None:
        """The turn ran past its limit."""
        self.boundary = self._finish_timeout

    # ------------------------------------------------------------------------------------------------ app calls
    def _ready_app(self) -> None:
        """Forward the oldest held app call once no other call of the batch runs (one per round)."""
        while self.app_queue and not self.forwarded and self.boundary is None:
            held_count = len(self.door.held) if self.door is not None else 0
            if self.running or len(self.cards_open) > held_count:
                return
            held = self.app_queue.popleft()
            {"app": self._forward, "refused": self._refuse, "backend": self._misrouted}[self.task.route(held.call)](
                held
            )

    def _forward(self, held: HeldCall) -> None:
        self.calls.append(held.call)
        self.forwarded = [held]
        self.boundary = self._finish_app

    def _refuse(self, held: HeldCall) -> None:
        self.calls.append(held.call)
        held.reply(self.task.refusal(held.call_id))

    def _misrouted(self, held: HeldCall) -> None:
        """An app function the IM routes to the backend (it never should): reported failed, answered with the error.

        The error is the core's ``texts.misrouted_output``.
        """
        output = json.dumps({"error": self.task.texts.misrouted_output.format(name=held.call["name"])})
        self.calls.append(held.call)
        self.task.tool_started(held.call, self.settings.runtime_label)
        self.task.tool_outcome(held.call_id, output, "failed")
        held.reply(output)
