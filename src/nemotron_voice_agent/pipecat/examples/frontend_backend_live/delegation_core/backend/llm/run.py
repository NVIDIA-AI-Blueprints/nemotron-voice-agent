# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the backend LLM, directly connected: the rounds of one worker, by the port's event rules."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing, suppress
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, get_args

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    Delivery,
    Phase,
    Route,
    Steering,
    StopOutcome,
    TaskContext,
    ToolNotExecuted,
    ToolSource,
    allowed_tools,
    listed,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.providers import (
    Backend,
    BrokenStreamError,
    back_off,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import CHAT_COMPLETIONS, TERMINALS
from nemotron_voice_agent.pipecat.shared.ids import uid

from .settings import LlmSettings

# The source label of a call no tool source offered, which the run answers with an error output.
OWN_LABEL = "llm"

PHASES = frozenset(get_args(Phase))


class RoundFailedError(RuntimeError):
    """The backend's round ended ``failed``, or completed with nothing in it: an IM session tries it again."""


@dataclass(frozen=True)
class _End:
    """The reader is done before a terminal: the stream ended (``error`` None) or raised ``error``."""

    error: BaseException | None = None


# Wakes the run when it cuts an attempt, so it stops waiting for the attempt's events.
_CUT = object()


@dataclass(frozen=True)
class _Sent:
    """What a round was sent, written out only when the debug log takes it: the history is not serialized else."""

    payload: dict

    def __str__(self) -> str:
        """Return the model, the tool choice, the tools' names, the instructions and the input."""
        payload = self.payload
        tools = [tool.get("name") or tool.get("type") for tool in payload.get("tools") or []]
        return (
            f"model={payload.get('model')} tool_choice={payload.get('tool_choice')} tools={tools} "
            f"instructions={payload.get('instructions')!r} input={json.dumps(payload.get('input'))}"
        )


class _AttemptObservation:
    """One logical provider reader's private observations, independent of the consumer's event drain."""

    def __init__(self, trace: Callable[..., None], identity: dict):
        """Bind the opening identity before scheduling the provider reader."""
        self.trace, self.identity = trace, identity
        self.started = time.monotonic()
        self.reader_entered = self.created = self.finished = False
        self.response_id: str | None = None
        self.response_model: str | None = None
        self.terminal: str | None = None
        self.usage: dict | None = None
        self.error_type: str | None = None
        self._emit("started")

    def _emit(self, event_phase: str, **fields: Any) -> None:
        """An observation sink cannot change a model attempt or escape its done callback."""
        try:
            self.trace(f"backend.provider_attempt.{event_phase}", **self.identity, **fields)
        except BaseException as exc:
            self.failed_observation(event_phase, exc)

    def failed_observation(self, event_phase: str, error: BaseException) -> None:
        """Report only the error type; diagnostic failures never escape a reader done callback."""
        with suppress(BaseException):
            logger.warning(
                "backend.provider_attempt.observation_failed observation_id={} event={} error_type={}",
                self.identity["observation_id"],
                event_phase,
                type(error).__name__,
            )

    def observe(self, event: dict) -> None:
        """Retain only provider-observed envelope metadata; never alter the event passed to the consumer."""
        kind = event.get("type")
        self.created = self.created or kind == "response.created"
        response = event.get("response")
        if isinstance(response, dict):
            for field in ("id", "model"):
                value = response.get(field)
                if isinstance(value, str) and value.strip():
                    setattr(self, "response_id" if field == "id" else "response_model", value)
            if kind in TERMINALS:
                usage = response.get("usage")
                self.usage = deepcopy(usage) if isinstance(usage, dict) else None
        if kind in TERMINALS:
            self.terminal = kind

    def end(self, reader: asyncio.Task | None, cut_by: str | None, error: BaseException | None = None) -> None:
        """Emit once when the reader actually ends, including cancellation before its coroutine first runs."""
        if self.finished:
            return
        self.finished = True
        cancelled = reader is not None and reader.cancelled()
        failure = error or (reader.exception() if reader is not None and not cancelled else None)
        error_type = self.error_type or (type(failure).__name__ if failure is not None else None)
        ending = cut_by or ("error" if error_type else "cancelled" if cancelled else self.terminal or "stream_ended")
        translated = self.identity["provider_api"] == CHAT_COMPLETIONS
        self._emit(
            "ended",
            carrier_scope="event_time",
            reader_entered=self.reader_entered,
            created_observed=self.created,
            response_id=self.response_id,
            response_id_source=("translated_response" if translated else "provider_event")
            if self.response_id
            else None,
            response_model=self.response_model,
            response_model_source=("translated_request" if translated else "provider_event")
            if self.response_model
            else None,
            reported_model=None if translated else self.response_model,
            provider_terminal=self.terminal,
            usage=self.usage,
            usage_missing=self.usage is None,
            usage_authority="logical_provider_attempt",
            cost=None,
            cut_by=cut_by,
            ending=ending,
            error_type=error_type,
            duration_ms=round((time.monotonic() - self.started) * 1000, 3),
        )


class _Attempt:
    """One try of a round: the provider's events, read by a task into a queue, so the run can cut it at any time."""

    def __init__(self, provider: Backend, payload: dict, *, trace: Callable[..., None], identity: dict):
        """Start reading one round of ``provider``, opened once: the run itself tries a failed round again."""
        self.provider = provider
        self.queue: asyncio.Queue = asyncio.Queue()
        self.created: dict | None = None
        self.items: list[dict] = []
        self.terminal: dict | None = None
        self.error: BaseException | None = None
        self.cut_by: str | None = None
        self.started = time.monotonic()
        self.observation = _AttemptObservation(trace, identity)
        try:
            self.reader = asyncio.create_task(self._read(provider.stream(payload)), name="llm-round")
        except BaseException as error:
            self.observation.end(None, None, error)
            raise
        self.reader.add_done_callback(self._observed_end)

    def _observed_end(self, reader: asyncio.Task) -> None:
        try:
            self.observation.end(reader, self.cut_by)
        except BaseException as error:
            self.observation.failed_observation("ended", error)

    async def _read(self, events: AsyncIterator[dict]) -> None:
        """Queue the round's events up to its first terminal, then stop reading."""
        self.observation.reader_entered = True
        try:
            async with aclosing(events) as stream:
                async for event in stream:
                    self.observation.observe(event)
                    self.queue.put_nowait(event)
                    if event.get("type") in TERMINALS:
                        return
            self.queue.put_nowait(_End())
        except Exception as exc:
            self.observation.error_type = type(exc).__name__
            self.queue.put_nowait(_End(exc))

    def cut(self, by: str) -> None:
        """End the attempt: stop reading the provider, and wake the run (once)."""
        if self.cut_by is None:
            self.cut_by = by
            self.reader.cancel()
            self.queue.put_nowait(_CUT)

    async def opened(self) -> AsyncIterator[dict]:
        """Read the attempt to its end: pass on its ``response.created``, hold its items, keep its terminal or error."""
        while True:
            got = await self.queue.get()
            if got is _CUT:
                return
            if isinstance(got, _End):
                self.error = got.error or BrokenStreamError("The backend's stream ended without a terminal event")
                return
            kind = got.get("type", "")
            if kind == "response.created" and self.created is None:
                self.created = got
                yield got
            elif kind == "response.created" or self.created is None:
                self.error = BrokenStreamError(f"The backend's stream is out of order at {kind}")
                self.cut("broken")
                return
            elif kind in TERMINALS:
                self.terminal = got
                return
            elif kind == "response.output_item.done":
                item = dict(got["item"])
                item.setdefault("id", uid("item"))
                self.items.append(item)

    def closed(self, kind: str, **fields: Any) -> dict:
        """Return a terminal the run writes for this attempt (failed or incomplete), with no items."""
        response = dict((self.created or {}).get("response") or {})
        reported = (self.terminal or {}).get("response") or {}
        if "usage" in reported:
            response["usage"] = reported["usage"]
        status = "failed" if kind == "response.failed" else "incomplete"
        return {"type": kind, "response": {**response, "status": status, "output": [], **fields}}

    def failure(self) -> BaseException | None:
        """Return why the attempt gave no round, or ``None`` for a completed one with content, or one cut short."""
        if self.error is not None:
            return self.error
        terminal = self.terminal or {}
        if terminal.get("type") == "response.failed":
            error = (terminal.get("response") or {}).get("error") or {}
            detail = error.get("message") or error.get("code") or "no detail"
            return RoundFailedError(f"The backend's round failed: {detail}")
        if terminal.get("type") == "response.completed" and not _has_content(self.items):
            return RoundFailedError("The backend's round answered nothing")
        return None

    def failed(self, failure: BaseException) -> dict:
        """Return the failed attempt's terminal: the provider's own ``response.failed`` without items, or the run's."""
        if self.terminal is not None and self.terminal.get("type") == "response.failed":
            return {**self.terminal, "response": {**(self.terminal.get("response") or {}), "output": []}}
        return self.closed("response.failed", error={"code": "server_error", "message": str(failure)})


Retryable = Callable[[_Attempt, BaseException], bool]


def can_clear_up(attempt: _Attempt, failure: BaseException) -> bool:
    """Return whether a failed attempt may be tried again: a transient error, a time limit, a broken or failed round."""
    return isinstance(failure, (*attempt.provider.transient_errors, TimeoutError, BrokenStreamError, RoundFailedError))


def failed_to_open(attempt: _Attempt, failure: BaseException) -> bool:
    """Return whether a failed attempt may be tried again: a transient error before its round's first event."""
    return attempt.created is None and isinstance(failure, attempt.provider.transient_errors)


def _text_and_refusals(item: dict) -> str:
    """Return the text of a message item: its content's text and refusal parts, of any part type."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get(field) or "") for part in content or [] if isinstance(part, dict) for field in ("text", "refusal")
    )


def _has_content(items: list[dict]) -> bool:
    """Return whether a completed round holds a call or a message with text: an IM session tries an empty one again."""
    return any(
        item.get("type") == "function_call" or (item.get("type") == "message" and _text_and_refusals(item).strip())
        for item in items
    )


def _labeled(items: list[dict], commentary: bool) -> list[dict]:
    """Return the round's items other than calls, each message with its phase: all commentary in a round with calls."""
    labeled = [dict(item) for item in items]
    messages = [item for item in labeled if item.get("type") == "message"]
    for message in messages:
        if commentary or message.get("phase") not in PHASES:
            message["phase"] = "commentary" if commentary else "final_answer"
    if messages and not commentary and all(message["phase"] == "commentary" for message in messages):
        messages[-1]["phase"] = "final_answer"
    return labeled


def _opening(item: dict) -> dict:
    """Return an item as ``response.output_item.added`` announces it: in progress, a message with no content yet."""
    opening = {**item, "status": "in_progress"}
    if item.get("type") == "message":
        opening["content"] = []
    return opening


class LlmRun:
    """One task's rounds on the backend LLM (``port.BackendRun``): one worker."""

    def __init__(
        self,
        task: TaskContext,
        *,
        providers: Callable[[str | None], Backend],
        tools: Sequence[ToolSource],
        settings: LlmSettings,
        retryable: Retryable,
        parent: BackendRun | None = None,
        prepare: Callable[[], Awaitable[None]] | None = None,
    ):
        """Run ``task``; nothing is called before :meth:`events`."""
        self.task, self.providers, self.sources = task, providers, tuple(tools)
        self.parent = parent
        self._prepare = prepare
        self.settings, self.retryable = settings, retryable
        self.round = 0
        self.done = False
        self.stopped = False
        self.suspended = False
        self.waits = False
        self.held: list[Steering] = []
        self.owners: dict[str, ToolSource] = {}
        self.offered: list[str] = []
        self._attempt: _Attempt | None = None
        self._readers: set[asyncio.Task] = set()
        self._calls: asyncio.Task | None = None
        self._events_readers = 0
        self._calls_unconfirmed = False
        self._quiescent: asyncio.Future[None] | None = None

    async def events(self) -> AsyncIterator[dict]:
        """Yield the task's Responses events, attempt by attempt; called again after :meth:`resume`.

        A failed round is tried again, up to ``provider_retries`` times after a backoff, if ``retryable`` accepts its
        failure. The last failure raises: a provider's error unchanged, so a 429 stays one, or the run's own.
        """
        self.waits = False
        self._events_readers += 1
        try:
            if self._prepare is not None and not self.stopped:
                await self._prepare()
                self._prepare = None
            while not (self.done or self.stopped or self.suspended or self.waits):
                self.round += 1
                tries = 0
                while True:
                    attempt = self._start_attempt(tries)
                    if attempt is None:
                        return
                    async with aclosing(attempt.opened()) as opened:
                        async for created in opened:
                            yield created
                            if self.stopped:
                                return
                            if self.suspended:
                                yield attempt.closed("response.incomplete", incomplete_details={"reason": "paused"})
                                return
                    self._attempt = None
                    if self.stopped:
                        return
                    if self.suspended:
                        if attempt.created is not None:
                            yield attempt.closed("response.incomplete", incomplete_details={"reason": "paused"})
                        return
                    if attempt.cut_by == "steer" and attempt.terminal is None and attempt.error is None:
                        self.task.trace("backend.steered", round=self.round)
                        if attempt.created is not None:
                            yield attempt.closed("response.incomplete", incomplete_details={"reason": "steered"})
                        continue
                    failure = attempt.failure()
                    if failure is None:
                        break
                    tries += 1
                    if attempt.created is not None:
                        yield attempt.failed(failure)
                        if self.stopped:
                            return
                    if tries > self.settings.provider_retries or not self.retryable(attempt, failure):
                        self.done = True
                        raise failure
                    self.task.trace(
                        "backend.recovered", round=self.round, error_type=type(failure).__name__, action="retry_round"
                    )
                    await back_off(failure, tries, role="backend")
                if attempt.terminal["type"] == "response.incomplete":
                    self.done = True
                    details = (attempt.terminal.get("response") or {}).get("incomplete_details")
                    yield attempt.closed("response.incomplete", incomplete_details=details)
                    raise RuntimeError(f"The backend's round was cut short: {(details or {}).get('reason')}")
                async with aclosing(self._round(attempt)) as round_events:
                    async for event in round_events:
                        yield event
                        if self.stopped:
                            return
        finally:
            try:
                if self._attempt is not None:
                    self._attempt.cut("close")
                    self._attempt = None
                self._settle_held("not_applied")
            finally:
                self._events_readers -= 1
                self._notify_quiescent()

    def steer(self) -> None:
        """Restart a streaming round, which has dispatched nothing; else the steering waits for the next safe point.

        A round that answers first ends the task, and the IM starts the steering as the next request.
        """
        attempt = self._attempt
        if attempt is not None and not attempt.reader.done() and not self.stopped:
            attempt.cut("steer")

    async def resume(self) -> None:
        """Go on after the app's outputs: the next round's safe point takes them."""
        if self.stopped:
            return
        if self.suspended:
            if await self.suspend() != "stopped":
                raise RuntimeError("The worker has not settled its suspended invocation")
            if self.stopped:
                return
            self.suspended = self.done = False
        self.task.trace("tools.completed", round=self.round)

    async def suspend(self) -> StopOutcome:
        """Fence inference without ending the run or inventing a result for an outstanding call."""
        self.suspended = True
        if self._attempt is not None:
            self._attempt.cut("suspend")
        running = [*self._readers, *([self._calls] if self._calls is not None else [])]
        pending = {task for task in running if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self.settings.stop_timeout_seconds)
        outcome: StopOutcome = "unconfirmed" if pending or not self._calls_quiescent() else "stopped"
        self.task.trace("backend.suspended", round=self.round, outcome=outcome)
        return outcome

    async def stop(self) -> StopOutcome:
        """Stop the task: cut a streaming round, let started calls finish, and confirm within the stop limit."""
        self.stopped = True
        if self._attempt is not None:
            self._attempt.cut("stop")
        running = [*self._readers, *([self._calls] if self._calls is not None else [])]
        pending = {task for task in running if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self.settings.stop_timeout_seconds)
        outcome: StopOutcome = "unconfirmed" if pending or not self._calls_quiescent() else "stopped"
        self.task.trace("backend.stopped", round=self.round, outcome=outcome)
        self._notify_quiescent()
        return outcome

    def on_quiescent(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Report positive late local settlement without another stop, provider request, or business action."""
        if self._quiescent is None:
            self._quiescent = asyncio.get_running_loop().create_future()
        future = self._quiescent
        subscribed = True

        def settled(done: asyncio.Future) -> None:
            nonlocal subscribed
            if subscribed and not done.cancelled() and done.exception() is None:
                subscribed = False
                callback()

        def unsubscribe() -> None:
            nonlocal subscribed
            subscribed = False
            future.remove_done_callback(settled)

        future.add_done_callback(settled)
        self._notify_quiescent()
        return unsubscribe

    def _notify_quiescent(self) -> None:
        """A stopped run cannot open another attempt; both iterator exit and physical owners are required."""
        future = self._quiescent
        if (
            future is None
            or future.done()
            or not self.stopped
            or self._events_readers
            or self._readers
            or not self._calls_quiescent()
        ):
            return
        future.set_result(None)

    def _calls_quiescent(self) -> bool:
        """A cancelled or exceptional aggregate is not positive proof that all its effects settled."""
        calls = self._calls
        return not self._calls_unconfirmed and (
            calls is None or calls.done() and not calls.cancelled() and calls.exception() is None
        )

    def _reader_finished(self, reader: asyncio.Task) -> None:
        """Provider reader completion includes its stream-close cleanup, including a requested local cut."""
        self._readers.discard(reader)
        self._notify_quiescent()

    def _calls_finished(self, calls: asyncio.Task) -> None:
        """An abandoned tool aggregate cannot establish positive settlement merely by being done."""
        if calls.cancelled() or calls.exception() is not None:
            self._calls_unconfirmed = True
        self._notify_quiescent()

    def _start_attempt(self, tries: int) -> _Attempt | None:
        """Open an attempt of the round at a safe point, or ``None`` once the task was stopped."""
        if self.stopped or self.suspended:
            return None
        if self.task.business_access != "full" and not self.task.can_dispatch():
            return None
        self._take_boundary()
        # Settled first: the IM joins it, and a typed run that lands brings its settings, which this round reads.
        self._settle_held("applied")
        request = self.task.request()
        provider = self.providers(request.get("model"))
        payload = self._payload(request, self.task.history())
        self.task.trace("backend.started", round=self.round, attempt=tries + 1, model=payload.get("model"))
        logger.debug(
            "backend.request work={} task={} round={} attempt={} {}",
            self.task.work_id,
            self.task.task_id,
            self.round,
            tries + 1,
            _Sent(payload),
        )
        attempt = self._attempt = _Attempt(
            provider,
            payload,
            trace=self.task.trace,
            identity={
                "observation_id": uid("provider_attempt"),
                "conversation_ref": getattr(self.task, "conversation_ref", None),
                "worker_implementation": "llm",
                "role": "worker",
                "provider_api": provider.name,
                "round": self.round,
                "attempt": tries + 1,
                "requested_model": payload.get("model"),
                "configured_model": provider.endpoint.model,
            },
        )
        self._readers.add(attempt.reader)
        attempt.reader.add_done_callback(self._reader_finished)
        return attempt

    def _payload(self, request: dict, history: list[dict]) -> dict:
        """Return the round's request: the current settings, the whole history, and every tool the round offers.

        The capped round after ``max_tool_rounds`` offers no tool, so its model has none to call or to write out.
        """
        capped = self.round > self.settings.max_tool_rounds
        tools, self.owners = ([], {}) if capped else self._round_tools(request)
        self.offered = [tool["name"] for tool in tools if tool.get("name")]
        payload = {**request, "tools": tools, "input": deepcopy(history)}
        if self.task.business_access != "full":
            payload["_restricted_tools"] = True
        if capped:
            note = self.task.texts.round_cap_note
            payload["tool_choice"] = "none"
            payload["instructions"] = f"{request['instructions']}\n\n{note}" if request.get("instructions") else note
        elif allowed_tools(request.get("tool_choice")) is not None:
            payload["tool_choice"] = request["tool_choice"].get("mode") or "auto"
        return payload

    def _round_tools(self, request: dict) -> tuple[list[dict], dict[str, ToolSource]]:
        """Return the request's, the server sources' and the work tools, each name once (the first wins).

        Under an ``allowed_tools`` tool choice, only the tools it lists, but the work tools all the same.
        """
        allowed = allowed_tools(request.get("tool_choice"))
        tools = [tool for tool in deepcopy(list(request.get("tools") or [])) if listed(tool, allowed)]
        names = {tool.get("name") for tool in tools}
        owners: dict[str, ToolSource] = {}
        work_tools = self.task.tools()
        for source in (*self.sources, work_tools):
            for tool in source.tools():
                name = tool.get("name")
                if name not in names and (source is work_tools or listed(tool, allowed)):
                    names.add(name)
                    tools.append(tool)
                    owners[name] = source
        return tools, owners

    async def _round(self, attempt: _Attempt) -> AsyncIterator[dict]:
        """Pass on a completed round, labeled, then run the backend's calls; the app's calls end the iteration."""
        calls = [item for item in attempt.items if item.get("type") == "function_call"]
        rest = [item for item in attempt.items if item.get("type") != "function_call"]
        self.task.trace(
            "backend.completed",
            round=self.round,
            calls=len(calls),
            duration_ms=round((time.monotonic() - attempt.started) * 1000, 3),
        )
        if calls and self.round > self.settings.max_tool_rounds:
            self.done = True
            yield attempt.closed("response.failed", error={"code": "server_error", "message": "past the round cap"})
            raise RuntimeError("The backend kept calling tools past the round limit")
        while any(self.task.placement_pending(call) for call in calls):
            await self.task.wait_for_placement()
            if self.stopped:
                return
            if self.suspended:
                yield attempt.closed("response.incomplete", incomplete_details={"reason": "paused"})
                return
        by_route: dict[Route, list[dict]] = {"app": [], "backend": [], "refused": []}
        for call in calls:
            by_route[self.task.route(call)].append(call)
        items = [*_labeled(rest, bool(calls)), *calls]
        for index, item in enumerate(items):
            yield {"type": "response.output_item.added", "output_index": index, "item": _opening(item)}
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        yield {"type": "response.completed", "response": {**attempt.terminal["response"], "output": items}}
        if not calls:
            self.done = True
            return
        if by_route["backend"] and not self.stopped:
            await self._run_calls(by_route["backend"])
        if by_route["app"]:
            self.waits = True
            self.task.trace("tools.waiting", round=self.round, call_ids=[call["call_id"] for call in by_route["app"]])

    async def _run_calls(self, calls: list[dict]) -> None:
        """Run the completed round's calls, settling any that never start because the invocation was suspended."""
        if self.suspended:
            for call in calls:
                self.task.dispatch(call, suspended=True)
            return
        self.task.trace(
            "tools.started",
            round=self.round,
            call_ids=[call.get("call_id") for call in calls],
            names=[call.get("name", "") for call in calls],
        )
        self._calls = asyncio.create_task(self._run_all(calls), name="llm-calls")
        self._calls.add_done_callback(self._calls_finished)
        await asyncio.shield(self._calls)
        self._calls = None
        self.task.trace("tools.completed", round=self.round)

    async def _run_all(self, calls: list[dict]) -> None:
        """Apply task-stop controls before independently dispatching the remaining calls."""
        controls = [
            call
            for call in calls
            if call.get("name") in ("cancel_work", "set_work_state")
            and self.owners.get(call["name"]) is self.task.tools()
        ]
        for call in controls:
            await self._run_call(call)
        results = await asyncio.gather(
            *(self._run_call(call) for call in calls if call not in controls), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in results):
            self._calls_unconfirmed = True

    async def _run_call(self, call: dict) -> None:
        """Run one call on the source that offered it, and report every outcome with an output."""
        call_id, name, texts = call.get("call_id", ""), call.get("name", ""), self.task.texts
        while not self.suspended and self.task.placement_pending(call):
            await self.task.wait_for_placement()
        if self.task.dispatch(call, suspended=self.suspended) == "refused":
            return
        source = self.owners.get(name)
        self.task.tool_started(call, source.label if source is not None else OWN_LABEL)
        limit = self.settings.server_tool_timeout_seconds
        if source is None:
            unknown = texts.unknown_tool_output.format(name=name, offered=", ".join(self.offered) or texts.none)
            output, outcome = json.dumps({"error": unknown}), "failed"
        else:
            try:
                async with asyncio.timeout(limit):
                    output, outcome = await source.execute(call_id, name, call.get("arguments") or "{}"), "done"
            except TimeoutError:
                # Cut after its effect may have happened: whether it ran is not known.
                output, outcome = json.dumps({"error": f"{name} did not answer within {limit:g} s"}), "unclear"
            except ToolNotExecuted as exc:
                failed = texts.failed_output.format(name=name, error=f"{type(exc).__name__}: {exc}")
                output, outcome = json.dumps({"error": failed}), "failed"
            except asyncio.CancelledError as exc:
                self._calls_unconfirmed = True
                failed = texts.failed_output.format(name=name, error=type(exc).__name__)
                self.task.tool_outcome(call_id, json.dumps({"error": failed}), "unclear")
                raise
            except Exception as exc:
                failed = texts.failed_output.format(name=name, error=f"{type(exc).__name__}: {exc}")
                output, outcome = json.dumps({"error": failed}), "unclear"
        self.task.tool_outcome(call_id, output, outcome)
        if outcome == "done" and name == "cancel_work" and source is self.task.tools():
            self.done = True

    def _take_boundary(self) -> None:
        """Take the app's outputs, and hold every waiting steering batch for the next attempt: a safe point."""
        while True:
            boundary = self.task.take_boundary_input()
            if boundary.steering is None:
                return
            self.held.append(boundary.steering)

    def _settle_held(self, delivery: Delivery) -> None:
        """Settle every held steering batch once: ``applied`` into an attempt, ``not_applied`` when none follows."""
        held, self.held = self.held, []
        for steering in held:
            self.task.settle(steering, delivery)
