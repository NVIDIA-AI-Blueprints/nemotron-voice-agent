# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One task's run on the backend LLM, directly connected: the rounds of one worker, by the port's event rules.

Each attempt reads ``request()`` and ``history()`` after the safe point settles waiting steering, sends the whole
history, and holds its items until its terminal, so an attempt that fails delivers nothing. A failed, timed-out or
empty attempt is tried once more (never in a one-shot session); a completed round with content is never retried.
Steering restarts a round still streaming, once per Work, because such a round has dispatched nothing (the cut attempt
ends ``incomplete`` with the reason ``steered``); otherwise it lands at the next safe point. A stop cuts a streaming
round but lets started calls finish. After ``max_tool_rounds`` rounds, one more runs with ``tool_choice: none``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, get_args

from loguru import logger

from delegation_core.backend.port import (
    TERMINALS,
    Delivery,
    Phase,
    Route,
    Steering,
    StopOutcome,
    TaskContext,
    ToolSource,
)
from delegation_core.backend.providers import Backend, BrokenStreamError
from delegation_core.ids import uid

from .settings import LlmSettings

# The label a call no tool source offered is reported with: the backend LLM answered it with an error output.
OWN_LABEL = "llm"


def _form(name: str) -> str:
    """A tool name's form for comparing names: CamelCase split, lower case, no separators, no trailing "tool"."""
    key = re.sub(r"[\s_.\-]", "", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)).lower()
    return key[: -len("tool")] if key.endswith("tool") and len(key) > len("tool") else key


PHASES = frozenset(get_args(Phase))


class RoundFailedError(RuntimeError):
    """The backend's round ended ``failed``, or completed with nothing in it: worth one more try."""


@dataclass(frozen=True)
class _End:
    """The reader is done before a terminal: the stream ended (``error`` None) or raised ``error``."""

    error: BaseException | None = None


# Put into an attempt's queue when the run cuts it (a stop or a steer), so the run stops waiting for its events.
_CUT = object()


@dataclass(frozen=True)
class _Sent:
    """What a round was sent, written out only when the debug log takes it: the history is not serialized else."""

    payload: dict

    def __str__(self) -> str:
        """The model, the tool choice, the tools' names, the instructions and the input."""
        payload = self.payload
        tools = [tool.get("name") or tool.get("type") for tool in payload.get("tools") or []]
        return (
            f"model={payload.get('model')} tool_choice={payload.get('tool_choice')} tools={tools} "
            f"instructions={payload.get('instructions')!r} input={json.dumps(payload.get('input'))}"
        )


class _Attempt:
    """One try of a round: the provider's events, read by a task into a queue, so the run can cut it at any time."""

    def __init__(self, provider: Backend, payload: dict):
        """Start reading one round of ``provider``."""
        self.provider = provider
        self.queue: asyncio.Queue = asyncio.Queue()
        self.created: dict | None = None  # its response.created, once passed on
        self.items: list[dict] = []  # its finished items, held to its terminal
        self.terminal: dict | None = None
        self.error: BaseException | None = None
        self.cut_by: str | None = None  # "stop", "steer" or "close"
        self.started = time.monotonic()
        self.reader = asyncio.create_task(self._read(provider.stream(payload)), name="llm-round")

    async def _read(self, events: AsyncIterator[dict]) -> None:
        """Queue the round's events up to its first terminal, then stop reading."""
        try:
            async with aclosing(events) as stream:
                async for event in stream:
                    self.queue.put_nowait(event)
                    if event.get("type") in TERMINALS:
                        return
            self.queue.put_nowait(_End())
        except Exception as exc:
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
        """A terminal of this attempt written by the run (``response.failed`` or ``response.incomplete``), no items."""
        response = dict((self.created or {}).get("response") or {})
        status = "failed" if kind == "response.failed" else "incomplete"
        return {"type": kind, "response": {**response, "status": status, "output": [], **fields}}

    def failure(self) -> BaseException | None:
        """Why the attempt gave no round, or ``None`` for a completed one with content, or one cut short."""
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

    def retryable(self, failure: BaseException) -> bool:
        """Whether the failure can clear up on its own: the provider's transient errors, a limit, a broken round."""
        return isinstance(failure, (*self.provider.transient_errors, TimeoutError, BrokenStreamError, RoundFailedError))

    def failed(self, failure: BaseException) -> dict:
        """The failed attempt's terminal: the provider's own ``response.failed`` without items, or the run's."""
        if self.terminal is not None and self.terminal.get("type") == "response.failed":
            return {**self.terminal, "response": {**(self.terminal.get("response") or {}), "output": []}}
        return self.closed("response.failed", error={"code": "server_error", "message": str(failure)})


def message_text(item: dict) -> str:
    """The text of a message item: its content's text and refusal parts."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        str(part.get(field) or "") for part in content or [] if isinstance(part, dict) for field in ("text", "refusal")
    )


def _has_content(items: list[dict]) -> bool:
    """Whether a completed round holds a call or a message with text: an empty one is retried once."""
    return any(
        item.get("type") == "function_call" or (item.get("type") == "message" and message_text(item).strip())
        for item in items
    )


def _labeled(items: list[dict], commentary: bool) -> list[dict]:
    """The round's items other than calls, each message with its phase.

    ``commentary`` (a round with calls, or one that steering crossed) labels every message so. Otherwise a message keeps
    a phase its model gave, the rest are the ``final_answer``, and a round whose messages are all commentary answers
    with its last one.
    """
    labeled = [dict(item) for item in items]
    messages = [item for item in labeled if item.get("type") == "message"]
    for message in messages:
        if commentary or message.get("phase") not in PHASES:
            message["phase"] = "commentary" if commentary else "final_answer"
    if messages and not commentary and all(message["phase"] == "commentary" for message in messages):
        messages[-1]["phase"] = "final_answer"
    return labeled


def _opening(item: dict) -> dict:
    """An item as ``response.output_item.added`` announces it: in progress, a message with no content yet."""
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
        attempts: int,
    ):
        """Run ``task``; nothing is called before :meth:`events`.

        Args:
            task: What the IM gives the task: what to read and where to report.
            providers: The session's provider of a model (one per model).
            tools: The session's server tool sources.
            settings: The backend LLM's settings: the round cap, the tool and stop limits.
            attempts: Tries per round: 2 (one retry), or 1 in a one-shot session.
        """
        self.task, self.providers, self.sources = task, providers, tuple(tools)
        self.settings, self.attempts = settings, attempts
        self.round = 0  # the current round, from 1
        self.done = False  # the task answered or failed: events() yields nothing more
        self.stopped = False  # level: set by stop(), never cleared
        self.redone = False  # the one restart of a streaming round this Work may have had
        self.waits = False  # the last round ended with the app's calls: the iteration ends until resume()
        self.held: list[Steering] = []  # steering taken and not settled yet
        self.owners: dict[str, ToolSource] = {}  # the round's offered tools by name: which source runs each
        self.offered: list[str] = []  # every tool name the round offered, the app's included
        self._attempt: _Attempt | None = None  # the attempt that is streaming
        self._readers: set[asyncio.Task] = set()  # every attempt's reader still running, a cut one included
        self._calls: asyncio.Task | None = None  # the backend's calls that are running

    # ------------------------------------------------------------------------------------------------ the port
    async def events(self) -> AsyncIterator[dict]:
        """The task's Responses events, attempt by attempt, by the event rules; called again after :meth:`resume`.

        Raises:
            Exception: The task failed: a provider's error (unchanged, so a 429 stays one), a time limit, a round that
                stayed empty or broken after its retry, a round cut short (after its ``response.incomplete``), the
                round cap passed, or a model the catalog does not list (``unlisted_models: report``).
        """
        self.waits = False
        try:
            while not (self.done or self.stopped or self.waits):
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
                    self._attempt = None
                    if self.stopped:
                        return
                    if attempt.cut_by == "steer" and attempt.terminal is None and attempt.error is None:
                        self.redone = True  # the restart before acting, once per Work
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
                    if tries >= self.attempts or not attempt.retryable(failure):
                        self.done = True
                        raise failure
                    self.task.trace(
                        "backend.recovered", round=self.round, error_type=type(failure).__name__, action="retry_round"
                    )
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
            if self._attempt is not None:
                self._attempt.cut("close")
                self._attempt = None
            self._settle_held("not_applied")

    def steer(self) -> None:
        """Steering waits: restart the round if it still streams and this Work has not been restarted.

        Otherwise the steering lands at the next safe point: after the round's calls, or after the app's outputs.
        """
        attempt = self._attempt
        if attempt is not None and not attempt.reader.done() and not self.redone and not self.stopped:
            attempt.cut("steer")

    async def resume(self) -> None:
        """The app's outputs are in: the next round's safe point takes them, then :meth:`events` goes on."""
        self.task.trace("tools.completed", round=self.round)

    async def stop(self) -> StopOutcome:
        """Stop the task: cut a streaming round, let started calls finish, and confirm within the stop limit.

        Returns ``stopped`` once nothing of the task runs, or ``unconfirmed`` when the round or a call still runs at
        ``stop_timeout_seconds`` (a hung provider). The stop is never cleared: nothing is dispatched after it.
        """
        self.stopped = True
        if self._attempt is not None:
            self._attempt.cut("stop")
        running = [*self._readers, *([self._calls] if self._calls is not None else [])]
        pending = {task for task in running if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self.settings.stop_timeout_seconds)
        outcome: StopOutcome = "unconfirmed" if pending else "stopped"
        self.task.trace("backend.stopped", round=self.round, outcome=outcome)
        return outcome

    # ------------------------------------------------------------------------------------------------ a round
    def _start_attempt(self, tries: int) -> _Attempt | None:
        """Open an attempt of the round at a safe point, or ``None`` once the task was stopped."""
        if self.stopped:
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
        attempt = self._attempt = _Attempt(provider, payload)
        self._readers.add(attempt.reader)
        attempt.reader.add_done_callback(self._readers.discard)
        return attempt

    def _payload(self, request: dict, history: list[dict]) -> dict:
        """The round's request: the current settings, the whole history, and every tool the round offers."""
        tools, self.owners = self._round_tools(request)
        self.offered = [tool["name"] for tool in tools if tool.get("name")]
        payload = {**request, "tools": tools, "input": deepcopy(history)}
        if self.round > self.settings.max_tool_rounds:  # the capped round: answer with what there is
            note = self.settings.round_cap_note
            payload["tool_choice"] = "none"
            payload["instructions"] = f"{request['instructions']}\n\n{note}" if request.get("instructions") else note
        return payload

    def _round_tools(self, request: dict) -> tuple[list[dict], dict[str, ToolSource]]:
        """The request's tools, then the server tool sources', then the work tools'; each name once, the first wins."""
        tools = deepcopy(list(request.get("tools") or []))
        names = {tool.get("name") for tool in tools}
        owners: dict[str, ToolSource] = {}
        for source in (*self.sources, self.task.tools()):
            for tool in source.tools():
                name = tool.get("name")
                if name not in names:
                    names.add(name)
                    tools.append(tool)
                    owners[name] = source
        return tools, owners

    async def _round(self, attempt: _Attempt) -> AsyncIterator[dict]:
        """Pass on a completed round, labeled, then run the backend's calls; the app's calls end the iteration."""
        calls = [self._named(item) for item in attempt.items if item.get("type") == "function_call"]
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
        by_route: dict[Route, list[dict]] = {"app": [], "backend": [], "refused": []}
        for call in calls:
            by_route[self.task.route(call)].append(call)
        if not calls:
            self._take_boundary()  # steering that waits: this round's words are not the answer
        items = [*_labeled(rest, bool(calls) or bool(self.held)), *calls]
        for index, item in enumerate(items):
            yield {"type": "response.output_item.added", "output_index": index, "item": _opening(item)}
            yield {"type": "response.output_item.done", "output_index": index, "item": item}
        yield {"type": "response.completed", "response": {**attempt.terminal["response"], "output": items}}
        if not calls:
            self.done = not self.held
            return
        if by_route["backend"] and not self.stopped:
            await self._run_calls(by_route["backend"])
        if by_route["app"]:
            self.waits = True  # the task waits for the app's outputs (resume, then events again)
            self.task.trace("tools.waiting", round=self.round, call_ids=[call["call_id"] for call in by_route["app"]])

    async def _run_calls(self, calls: list[dict]) -> None:
        """Run the backend's calls at once; a stop never cuts them, it waits for them."""
        self.task.trace(
            "tools.started",
            round=self.round,
            call_ids=[call.get("call_id") for call in calls],
            names=[call.get("name", "") for call in calls],
        )
        self._calls = asyncio.create_task(self._run_all(calls), name="llm-calls")
        await asyncio.shield(self._calls)
        self._calls = None
        self.task.trace("tools.completed", round=self.round)

    async def _run_all(self, calls: list[dict]) -> None:
        """Every call of the round together; one that raises never takes the others' outputs with it."""
        await asyncio.gather(*(self._run_call(call) for call in calls), return_exceptions=True)

    async def _run_call(self, call: dict) -> None:
        """Run one call on the source that offered it; every outcome is reported with an output."""
        call_id, name = call.get("call_id", ""), call.get("name", "")
        source = self.owners.get(name)
        self.task.tool_started(call, source.label if source is not None else OWN_LABEL)
        limit = self.settings.server_tool_timeout_seconds
        if source is None:
            offered = ", ".join(self.offered) or "none"
            output, outcome = json.dumps({"error": f"Unknown tool: {name}. The tools offered: {offered}"}), "failed"
        else:
            try:
                async with asyncio.timeout(limit):
                    output, outcome = await source.execute(call_id, name, call.get("arguments") or "{}"), "done"
            except TimeoutError:  # cut after its effect may have happened: whether it ran is not known
                output, outcome = json.dumps({"error": f"{name} did not answer within {limit:g} s"}), "unclear"
            except Exception as exc:
                output, outcome = json.dumps({"error": f"{name} failed: {type(exc).__name__}: {exc}"}), "failed"
        self.task.tool_outcome(call_id, output, outcome)

    def _named(self, call: dict) -> dict:
        """``call``, under the offered tool's name when it names one in all but its form; otherwise unchanged.

        Only the exact form counts, and only one offered tool may have it: a guessed name could run a write tool the
        model never named.
        """
        name = call.get("name", "")
        if name in self.offered:
            return call
        same = [offered for offered in self.offered if _form(offered) == _form(name)]
        if len(same) != 1:
            return call
        self.task.trace("tools.renamed", round=self.round, call_id=call.get("call_id"), name=name, to=same[0])
        return {**call, "name": same[0]}

    # ------------------------------------------------------------------------------------------------ steering
    def _take_boundary(self) -> None:
        """A safe point: the app's outputs join the conversation, and every waiting steering batch is held.

        Every waiting batch reaches the next attempt, not one per round; each ``take_boundary_input()`` hands over the
        next batch.
        """
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
