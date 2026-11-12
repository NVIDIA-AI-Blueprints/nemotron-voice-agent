# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the harness (``port.BackendRun``): a worker session of the agent, read as the IM's rounds."""

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

from examples.frontend_backend_live.delegation_core.backend.port import (
    BoundaryInput,
    Steering,
    StopOutcome,
    TaskContext,
)
from examples.shared.ids import uid

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
from .render import output_text
from .runtime import give_back
from .sessions import KeptSession, keys

if TYPE_CHECKING:
    from .adapter import HarnessAdapter

OPEN_ATTEMPTS = 2
LOGGED_TEXT_CHARS = 300

# The cancels sent in the background, kept until sent or timed out so none is collected before it is sent.
_CANCELS: set[asyncio.Future] = set()


class HarnessTurnError(RuntimeError):
    """The agent's turn gave no answer the run can pass on (its text is logged, never spoken)."""


def usage_total(usage: Mapping[str, Any] | None) -> int | None:
    """Return the session's token total an agent reported (ACP ``totalTokens``), or ``None``."""
    if not usage:
        return None
    total = usage.get("totalTokens")
    return total if isinstance(total, int) else None


@dataclass
class _Message:
    """One assistant message of the agent: its text so far, and whether it is the agent's own."""

    id: str
    parts: list[str] = field(default_factory=list)
    own: bool = False

    @property
    def text(self) -> str:
        return "".join(self.parts).strip()


@dataclass
class _SentSteer:
    """A batch sent as the agent's own mid-turn steer, until its delivery is known."""

    steering: Steering
    prompt: int
    acked: bool = False
    ack_message: str | None = None
    tool_rows_at_ack: bool = False
    result_after_ack: bool = False
    settled: bool = False


@dataclass
class _Turn:
    """One turn of the session: from a prompt until every prompt of the session has returned."""

    main: int
    total_before: int | None
    usage_before: Mapping[str, Any] | None = None
    started: float = field(default_factory=time.monotonic)
    # Seconds spent reading the turn: the wait for the app does not count toward its limit.
    used: float = 0.0
    main_open: bool = True
    stop: str | None = None
    usage: Mapping[str, Any] | None = None
    error: str | None = None
    activity_after_main: bool = False
    cancelled_by_us: bool = False
    seen: bool = False
    early: list[Steering] = field(default_factory=list)

    @property
    def cancel_pending(self) -> bool:
        """Our cancel went out and the turn did not end ``cancelled``: the agent may take it for its next prompt."""
        return self.cancelled_by_us and self.stop != "cancelled"


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
        self.done = False
        self.answered = False
        self.stopped = False
        self.waits = False
        self.runtime: Runtime | None = None
        self.session: HarnessSession | None = None
        self.door: WorkerDoor | None = None
        self.mailbox: Mailbox = asyncio.Queue()
        self.turn: _Turn | None = None
        self.instructions = ""
        # The keys of every item the session was shown or wrote, in the IM's order.
        self.shown: list[tuple] = []
        # Prompts whose end the run has not read, in mailbox order: the session's own set runs ahead of the mailbox.
        self.open_prompts: set[int] = set()
        self.total: int | None = None
        self.usage: Mapping[str, Any] | None = None
        self.tool_rows = False
        self.current_message: str | None = None
        self.sent: dict[int, _SentSteer] = {}
        self.next_prompts: list[Steering] = []
        self.reprompts: list[Steering] = []
        self.app_queue: deque[HeldCall] = deque()
        self.forwarded: list[HeldCall] = []
        self.running: set[str] = set()
        self.cards_open: set[str] = set()
        self.runtime_calls: dict[str, dict] = {}
        self.boundary: Callable[[], AsyncIterator[dict]] | None = None
        self.pending: SessionEvent | _TimedOut | None = None
        self.delivering: list[_SentSteer] = []
        self.response_id = ""
        self._stopping: asyncio.Future | None = None
        self._new_round()

    async def events(self) -> AsyncIterator[dict]:
        """Yield the task's Responses events round by round; call it again after :meth:`resume`.

        Raises:
            HarnessTurnError: The turn gave no answer the run can pass on, or ran past its limit.
            Exception: The session could not be opened: the agent did not start or did not list its tools.
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
            if not self.stopped:
                # The turn would run on with no reader; a stop cancels its own.
                self._cancel_turn()
            await self.release()
            raise
        if self.done:
            await self.release()

    def steer(self) -> None:
        """Wake the reader, which takes the waiting steering at once."""
        if not (self.done or self.stopped):
            self.mailbox.put_nowait(Wake())

    async def resume(self) -> None:
        """Deliver waiting steering, then answer the held app call; a native steer's ack is awaited first."""
        self.task.trace("tools.completed", round=self.round)
        boundary = self._boundary()
        outputs = {item.get("call_id"): item for item in boundary.outputs}
        before = set(self.sent)
        steering = boundary.steering
        while steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering
        steers = [number for number in self.sent if number not in before]
        # Awaiting the acks first makes the agent hold the steers when the tool batch ends, after its results.
        if steers and self.session is not None:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self.settings.ready_timeout_seconds):
                    await asyncio.gather(*(self.session.ended(number) for number in steers))
        for held in self.forwarded:
            text = output_text(outputs.get(held.call_id))
            held.reply(self.task.texts.owed_output if text is None else text)
        self.forwarded = []

    async def stop(self) -> StopOutcome:
        """Stop the task: cancel a running prompt, retire the door and the session, and wait for the end."""
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
        """Keep the session for a later task if the adapter takes it, else retire it; either way give back its place."""
        kept = self._kept()
        if kept is not None and self.host.keep(kept):
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
        if self.door is not None:
            self.host.bridge.close(self.door, text)

    def _kept(self) -> KeptSession | None:
        """Return the session for a later task: only once its task answered and nothing of the task is left in it."""
        session, door, runtime = self.session, self.door, self.runtime
        if session is None or door is None or runtime is None or not self.answered or self.stopped:
            return None
        if self.turn is not None and self.turn.cancel_pending:
            return None
        left = (session.outstanding, self.reprompts, self.next_prompts, self.sent, self.forwarded, self.app_queue)
        if any(left) or self.running or self.cards_open or door.held or not runtime.client.alive:
            return None
        shown = list(self.shown)
        return KeptSession(
            runtime, session, door, self.mailbox, self.instructions, shown, self.total, self.usage, self.tool_rows
        )

    async def _open(self) -> None:
        """Take the kept session when it fits (:meth:`_adopt`), else open a fresh one, and prompt it."""
        kept = self.host.take_kept()
        if kept is not None and self._adopt(kept):
            return
        self.runtime = await self.host.shared(self.host.workers)
        await self._open_session(self.task.request().get("instructions") or "")

    def _adopt(self, kept: KeptSession) -> bool:
        """Take over ``kept`` when it fits, and prompt it with the items it has not read; else retire it (``False``)."""
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
        except BaseException:
            # The task fails, its history over the budget: the session leaves nothing behind.
            self._retire_kept(kept)
            raise
        shown = keys(history)
        if shown[: len(kept.shown)] != kept.shown or len(shown) == len(kept.shown):
            return self._retire_kept(kept)
        self.runtime, self.session, self.door, self.mailbox = kept.runtime, kept.session, kept.door, kept.mailbox
        while not self.mailbox.empty():
            self.mailbox.get_nowait()
        self.instructions, self.shown = instructions, shown
        self.total, self.usage, self.tool_rows = kept.total, kept.usage, kept.tool_rows
        rest = history[len(kept.shown) :]
        self.task.trace("harness.session_reused", server=kept.door.name, session=kept.session.id, items=len(rest))
        self._start_turn(self.session.prompt(self.host.renderer.transcript(rest)))
        return True

    def _retire_kept(self, kept: KeptSession) -> bool:
        kept.runtime.leave()
        kept.retire(self.host.bridge)
        return False

    async def _open_session(self, instructions: str, batches: tuple[Steering, ...] = ()) -> None:
        """Open a fresh session in the run's runtime and prompt it with the history, ``batches`` applied."""
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
            except BaseException:
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
        self.total, self.usage = 0, None
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
        """Settle ``batches`` and any waiting steering ``applied`` before a fresh session's first prompt."""
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
        self.task.trace("harness.prompt_sent", round=self.round, prompt=prompt)

    def _new_round(self) -> None:
        self.messages: dict[str, _Message] = {}
        self.calls: list[dict] = []
        self.activity = False
        self.activity_since_message = False

    async def _round(self) -> AsyncIterator[dict]:
        """Yield one round: ``response.created``, the session's events up to a boundary, then the boundary's ending.

        A round ends when a model message begins after tool activity, when an app call is forwarded, when steering needs
        a safe point, when every prompt has returned, or on a stop or a timeout.
        """
        self.round += 1
        self._new_round()
        self.response_id = uid("resp")
        yield {"type": "response.created", "response": {"id": self.response_id, "status": "in_progress", "output": []}}
        self._ready_app()
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
        """Return the session's next event, or a :class:`_TimedOut` once the turn's reading time is up."""
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
        """End the round: messages (``final`` is the answer), then calls; the agent's own text is only logged."""
        items = []
        for message in self.messages.values():
            # Never commentary either, which may be spoken as progress.
            if message.own:
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
        """End the round ``response.failed`` and raise :class:`HarnessTurnError`; the agent's text is only logged."""
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
        """Let the owed outputs join at a safe point, then hand every waiting batch to the steering strategy."""
        steering = self._boundary().steering
        while steering is not None:
            self._deliver(steering)
            steering = self._boundary().steering

    def _deliver(self, steering: Steering) -> None:
        """Hand ``steering`` to the agent class's strategy (``steering.py``); a batch with no text applies at once."""
        if self.host.renderer.steering(steering):
            self.settings.steering.deliver(self, steering)
            return
        self._applied(steering)
        self.task.trace("harness.steer_applied", round=self.round, steering=steering.id, how="no_text")

    async def _finish_next_call(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_woken(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self._between_rounds()

    async def _finish_delivered(self) -> AsyncIterator[dict]:
        """Close the round, then settle the delivered steer, so it joins after what came before."""
        async for event in self._close(None):
            yield event
        self._settle_delivered()

    async def _finish_app(self) -> AsyncIterator[dict]:
        async for event in self._close(None):
            yield event
        self.waits = True
        self.task.trace("tools.waiting", round=self.round, call_ids=[held.call_id for held in self.forwarded])

    async def _finish_stopped(self) -> AsyncIterator[dict]:
        return
        yield

    async def _finish_timeout(self) -> AsyncIterator[dict]:
        """Fail the task whose turn ran past ``turn_timeout_seconds``; :meth:`events` cancels the turn."""
        async for event in self._fail(f"the turn ran past {self.settings.turn_timeout_seconds:g} s"):
            yield event

    async def _finish_turn(self) -> AsyncIterator[dict]:
        """End the turn with the answer, the next prompt, a fresh session, or a failure.

        The answer is the last message after the turn's last tool activity, when the turn ended ``end_turn`` and
        ``Agent.no_answer`` finds it answered; a turn without one goes on with waiting steering, or fails the task.
        """
        turn = self.turn
        if self.stopped:
            return
        failure = self._turn_failure(turn)
        candidate = self._candidate()
        reason = failure or self._unparsed(turn, candidate)
        if reason is None:
            self._leave_unapplied(turn)
            self.task.trace("harness.answered", round=self.round, seconds=round(time.monotonic() - turn.started, 3))
            async for event in self._close(candidate, self._turn_usage(turn)):
                yield event
            self.done = self.answered = True
            return
        if self.reprompts and (turn.cancelled_by_us or turn.stop == "end_turn"):
            async for event in self._close(None):
                yield event
            if turn.cancel_pending:
                await self._reopen()
            else:
                self._reprompt_now()
            return
        if failure is not None:
            async for event in self._fail(failure, self._last_text()):
                yield event
        if not self.calls:
            # Safe: the open round holds no call whose output could join before it.
            self._between_rounds()
        # Steers that never saw the turn begin are the oldest: they go first in the next prompt.
        self.next_prompts[:0] = turn.early
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
        """Settle every batch the answered turn did not apply ``not_applied``, once, so the IM starts it next."""
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
        if self.activity_since_message or not self.messages:
            return None
        return list(self.messages.values())[-1]

    def _unparsed(self, turn: _Turn, candidate: _Message | None) -> str | None:
        """Return why the turn's last message is no answer the run may pass on, or ``None``."""
        if candidate is None or not candidate.text:
            return "the turn ended without a message after its last tool call"
        reason = self.agent.no_answer(usage_total(turn.usage), turn.total_before)
        candidate.own = reason is not None
        return reason

    def _last_text(self) -> str:
        texts = [message.text for message in self.messages.values() if message.text]
        return texts[-1] if texts else ""

    def _turn_usage(self, turn: _Turn) -> dict | None:
        """Return the turn's own usage: how far the session's running counters moved, as ACP counts across turns."""
        if not turn.usage:
            return None
        before = turn.usage_before or {}
        keys = (("input_tokens", "inputTokens"), ("output_tokens", "outputTokens"), ("total_tokens", "totalTokens"))
        return {name: max(int(turn.usage.get(key) or 0) - int(before.get(key) or 0), 0) for name, key in keys}

    def _collect(self, batches: list[Steering]) -> list[Steering]:
        """Let the owed outputs join at a safe point, then add every waiting batch to ``batches`` after them."""
        steering = self._boundary().steering
        while steering is not None:
            batches.append(steering)
            steering = self._boundary().steering
        return batches

    def _boundary(self) -> BoundaryInput:
        """Take the boundary input at a safe point; the outputs that join are items the session has read."""
        boundary = self.task.take_boundary_input()
        self.shown += keys(boundary.outputs)
        return boundary

    def _applied(self, steering: Steering) -> None:
        self.task.settle(steering, "applied")
        self.shown += keys(steering.items)

    def _prompt_next_now(self) -> None:
        """Send every batch the turn did not take as the next prompt, oldest first; each is ``applied`` once sent."""
        batches = [*(sent.steering for sent in self.sent.values() if not sent.settled), *self.next_prompts]
        self.next_prompts, self.sent = [], {}
        batches = self._collect(batches)
        text = "\n\n".join(self.host.renderer.steering(steering) for steering in batches)
        prompt = self.session.prompt(text)
        for steering in batches:
            self._applied(steering)
        self.task.trace("harness.steer_prompted", round=self.round, batches=len(batches))
        self._start_turn(prompt)

    def _reprompt_now(self) -> None:
        """Send the batches as the session's next prompt after a turn that our cancel cannot still reach."""
        # What the cancelled turn left open ends with it: ACP has the client mark its open tool calls cancelled.
        for held in self.app_queue:
            held.reply(self.task.texts.not_run_output)
        self.app_queue, self.cards_open = deque(), set()
        self.next_prompts, self.reprompts = [*self.next_prompts, *self.reprompts], []
        self._prompt_next_now()

    async def _reopen(self) -> None:
        """Open a fresh session after a turn our cancel may still reach, prompted with the history and the batches."""
        batches, self.reprompts = self.reprompts, []
        self._close_door(self.task.texts.not_run_output)
        if self.session is not None:
            self.session.retire()
        self.sent, self.next_prompts, self.app_queue, self.forwarded = {}, [], deque(), []
        self.open_prompts = set()
        self.running, self.cards_open, self.runtime_calls = set(), set(), {}
        self.tool_rows, self.current_message = False, None
        await self._open_session(self.task.request().get("instructions") or "", tuple(batches))

    def steer_in_turn(self, steering: Steering) -> None:
        """Send the batch as the agent's own steer while the turn's prompt runs, else as the next prompt."""
        turn = self.turn
        if self.session is None or turn is None or not turn.main_open or self.stopped:
            self.prompt_next(steering)
        elif not turn.seen:
            # Sent now, the steer could reach the agent before the prompt holds the session, and run as a prompt.
            turn.early.append(steering)
        else:
            self._steer(steering)

    def _steer(self, steering: Steering) -> None:
        prompt = self.session.steer(self.host.renderer.steering(steering))
        self.open_prompts.add(prompt)
        self.sent[prompt] = _SentSteer(steering, prompt)
        self.task.trace("harness.steer_sent", round=self.round, steering=steering.id)

    def _seen(self) -> None:
        turn = self.turn
        if turn is None or turn.seen:
            return
        turn.seen = True
        early, turn.early = turn.early, []
        for steering in early:
            self.settings.steering.deliver(self, steering)

    def prompt_next(self, steering: Steering) -> None:
        """Queue the batch as the next prompt, sent once the turn ends."""
        self.next_prompts.append(steering)

    def reprompt(self, steering: Steering) -> None:
        """Cancel the turn once no bridged call runs; then the batch goes as the session's next prompt."""
        self.reprompts.append(steering)
        self._cancel_for_reprompt()

    def _cancel_for_reprompt(self) -> None:
        if self.reprompts and not (self.running or self.cards_open or self.forwarded):
            self._cancel_turn()

    def _cancel_turn(self) -> None:
        """Cancel the running turn once, in the background, within ``stop_timeout_seconds``."""
        turn, session = self.turn, self.session
        if turn is None or turn.cancelled_by_us or session is None or not session.outstanding:
            return
        turn.cancelled_by_us = True
        cancel = asyncio.ensure_future(asyncio.wait_for(session.cancel(), self.settings.stop_timeout_seconds))
        _CANCELS.add(cancel)
        cancel.add_done_callback(_CANCELS.discard)
        cancel.add_done_callback(lambda done: done.cancelled() or done.exception())

    def _acked(self) -> list[_SentSteer]:
        return [sent for sent in self.sent.values() if sent.acked and not sent.settled]

    def _delivered_by(self, message_id: str | None) -> list[_SentSteer]:
        """Return the acked steers that the model output now beginning proves delivered."""
        delivered = []
        for sent in self._acked():
            new_call = message_id is not None and message_id != sent.ack_message
            if sent.result_after_ack or (new_call and (sent.tool_rows_at_ack or self.tool_rows)):
                delivered.append(sent)
        return delivered

    def _settle_delivered(self) -> None:
        """Settle the delivered steers ``applied`` at a safe point, after the outputs of the calls before them."""
        delivering, self.delivering = self.delivering, []
        self._between_rounds()
        for sent in delivering:
            if not sent.settled:
                sent.settled = True
                self._applied(sent.steering)
                self.task.trace("harness.steer_applied", round=self.round, steering=sent.steering.id)
        self.sent = {prompt: sent for prompt, sent in self.sent.items() if not sent.settled}

    def _model_output(self, event: SessionEvent, message_id: str | None) -> bool:
        """Note that model output begins; return whether a boundary waits for it, the event read again next round."""
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
            message = self.messages[message_id] = _Message(message_id)
            self.activity_since_message = False
        self.current_message = message_id
        return message

    def message_chunk(self, event: MessageChunk) -> None:
        """Read model text; a chunk with no id is the agent's own reply, such as an ack, and is never output."""
        self._seen()
        if event.message_id is None or self._model_output(event, event.message_id):
            return
        message = self._begin_message(event.message_id)
        message.parts.append(event.text)

    def thought_chunk(self, event: ThoughtChunk) -> None:
        """Read reasoning: its id begins the message its text will join."""
        self._seen()
        if event.message_id is None or self._model_output(event, event.message_id):
            return
        self._begin_message(event.message_id)

    def tool_card(self, event: ToolCard) -> None:
        """Read a tool call's start; the bridge reports its own calls, and the agent's own are reported here."""
        self._seen()
        if self._model_output(event, None):
            return
        self.activity = self.activity_since_message = True
        self.cards_open.add(event.call_id)
        self.pending = event.ended
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
        # Reported as it runs: the commit gate cannot hold the agent's own tools.
        self.task.tool_started(call, self.settings.runtime_label)
        self.runtime_calls[event.call_id] = call
        self.calls.append(call)

    def tool_card_end(self, event: ToolCardEnd) -> None:
        """Read a tool call's end: the transcript holds a tool row, and an agent's own call reports its outcome."""
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
        """Read the agent's plan as a progress note, never history."""
        self._seen()
        if event.entries:
            self.task.progress("; ".join(event.entries))

    def user_text(self, event: UserText) -> None:
        """Read a prompt the agent drains or replays itself: a boundary between model calls."""
        self._seen()
        self.activity = self.activity_since_message = True

    def prompt_ended(self, event: PromptEnded) -> None:
        """Read a prompt's end: a steer's ack, the turn's end, or a steer that ran as a turn of its own."""
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
            self.task.trace("harness.turn_ended", round=self.round, stop=event.stop_reason, error=event.error)
        elif turn is not None and event.usage is not None and turn.activity_after_main:
            turn.stop, turn.error, turn.usage = event.stop_reason, event.error, event.usage
        self.open_prompts.discard(event.prompt)
        if not self.open_prompts and turn is not None and not turn.main_open:
            self.boundary = self._finish_turn

    def _steer_ended(self, sent: _SentSteer, turn: _Turn, event: PromptEnded) -> None:
        # An error means it never reached the agent, so it goes as the next prompt; an end while the turn ran is the
        # agent's ack, and an end after the turn, with model activity, a turn of its own.
        if event.error is not None:
            return
        if turn.main_open:
            sent.acked, sent.ack_message, sent.tool_rows_at_ack = True, self.current_message, self.tool_rows
            self.task.trace("harness.steer_acked", round=self.round, steering=sent.steering.id)
        elif turn.activity_after_main and not sent.settled:
            sent.settled = True
            self._applied(sent.steering)
            self.task.trace("harness.steer_applied", round=self.round, steering=sent.steering.id, how="turn")

    def app_call(self, event: AppCall) -> None:
        """Read an app call the bridge holds; it goes to the app once nothing else of the batch runs."""
        self._seen()
        self.activity = self.activity_since_message = True
        self.app_queue.append(event.held)
        self._ready_app()

    def call_running(self, event: CallRunning) -> None:
        """Read a call the bridge took, refused or run by the bridge; either is a call of this round."""
        self._seen()
        self.activity = self.activity_since_message = True
        self.calls.append(event.call)
        if not event.refused:
            self.running.add(event.call["call_id"])

    def call_settled(self, event: CallSettled) -> None:
        """Read the end of a call the bridge ran."""
        self.running.discard(event.call_id)
        self._ready_app()
        self._cancel_for_reprompt()

    def wake(self, event: Wake) -> None:
        """End the round on a stop; else take waiting steering, closing the round first only when it holds calls."""
        if self.stopped:
            self.boundary = self._finish_stopped
        elif self.calls:
            self.boundary = self._finish_woken
        else:
            self._between_rounds()

    def timed_out(self) -> None:
        """End the round for a turn that ran past its limit."""
        self.boundary = self._finish_timeout

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
        """Fail an app function the IM routes to the backend, answering with ``texts.misrouted_output``."""
        output = json.dumps({"error": self.task.texts.misrouted_output.format(name=held.call["name"])})
        self.calls.append(held.call)
        self.task.tool_started(held.call, self.settings.runtime_label)
        self.task.tool_outcome(held.call_id, output, "failed")
        held.reply(output)
