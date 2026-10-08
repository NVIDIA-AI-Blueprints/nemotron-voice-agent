# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serialized backend rounds with server-side tool execution.

Only one backend round runs at a time. Each task runs rounds until the model answers without calling a
tool: after a round that called tools, every call runs, the results join the history, and the next round
starts. A failed or uncertain round is never retried once a function call was emitted, because an
application action may already have happened.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy

from loguru import logger

from examples.frontend_backend_live.common.ids import approx_tokens, uid
from examples.frontend_backend_live.config_manager.schema import ReliabilityConfig
from examples.frontend_backend_live.delegation.tasks import BackendTask, DelegatedTask
from examples.frontend_backend_live.delegation.thinker import Thinker
from examples.frontend_backend_live.tool_calling.executor import ToolExecutor

APOLOGY = "Sorry, I ran into a problem with that. Could you say it again?"
MAX_COMPLETED_CALLS = 512


class DelegationError(Exception):
    """Raised when a task cannot be accepted (closed worker, full queue)."""


def _starts_a_request(item: dict) -> bool:
    """Whether a history item opens a new request: a user message, as the delegate's input always begins with one."""
    return item.get("role") == "user" and item.get("type", "message") == "message"


def _noop(kind: str, **fields) -> None:
    """Ignore observations."""


class DelegationWorker:
    """Runs delegated tasks one at a time and reports answers through callbacks."""

    def __init__(
        self,
        thinker: Thinker,
        task: BackendTask,
        tools: ToolExecutor,
        reliability: ReliabilityConfig,
        *,
        on_answer: Callable[[str, DelegatedTask], Awaitable[None]],
        on_abandoned: Callable[[str], None] = lambda task_id: None,
        on_event: Callable[[DelegatedTask, dict], Awaitable[None]] | None = None,
        observe: Callable[..., None] = _noop,
        context_tokens: int = 100_000,
    ):
        """Wire the thinker, its tools and the answer callback.

        Args:
            thinker: The backend model wrapper.
            task: The model, instructions and tool schemas every round uses.
            tools: Executes the model's function calls.
            reliability: Timeouts, round limit, queue size and the recovery switch.
            on_answer: Awaited with ``(text, task)`` for each answer, including the apology after a failure.
            on_abandoned: Called with a task id that will never produce an answer.
            on_event: Optional awaited hook with ``(task, raw_event)`` for every backend event of a round, so a
                protocol layer can forward the stream.
            observe: Optional ``observe(kind, **fields)`` hook for traces.
            context_tokens: Budget for the history resent each round.
        """
        self.thinker, self.task, self.tools, self.reliability = thinker, task, tools, reliability
        self.on_answer, self.on_abandoned, self.observe = on_answer, on_abandoned, observe
        self.on_event = on_event
        self.context_tokens = context_tokens
        self.queue: asyncio.Queue[DelegatedTask] = asyncio.Queue(maxsize=reliability.max_pending_delegations)
        self.active: DelegatedTask | None = None
        self.worker: asyncio.Task | None = None
        self.history: list[dict] = []
        self.previous_response_id: str | None = None
        self.completed_calls: dict[str, dict] = {}
        self.closed = False
        self.carry: list[dict] = []  # tool outputs a failed round left owing, sent with the next task
        self.cache_key = "live:" + uid("thread")

    def start(self) -> None:
        """Start the worker task."""
        self.worker = asyncio.create_task(self._run(), name="delegation-worker")

    @property
    def has_newer(self) -> bool:
        """Whether another task is waiting, so the running task's answer would be superseded."""
        return not self.queue.empty()

    async def submit(self, input_items: list[dict]) -> str:
        """Queue a task and return its id."""
        if self.closed:
            raise DelegationError("Session is closing")
        task = DelegatedTask(uid("item"), deepcopy(input_items))
        try:
            self.queue.put_nowait(task)
        except asyncio.QueueFull as exc:
            raise DelegationError("Delegation queue is full") from exc
        self.observe("backend.queued", delegation_id=task.id, queue_depth=self.queue.qsize())
        return task.id

    async def _run(self) -> None:
        while True:
            task = await self.queue.get()
            self.active = task
            self.observe(
                "backend.dequeue",
                delegation_id=task.id,
                duration_ms=round((time.monotonic() - task.queued_at) * 1000, 3),
            )
            try:
                await self._task(task)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Never retry a failed round automatically once a call was emitted: actions might have happened.
                logger.warning(f"Delegated task {task.id} failed in round {task.round}: {type(exc).__name__}: {exc}")
                self.observe(
                    "backend.failed",
                    delegation_id=task.id,
                    round=task.round,
                    error_type=type(exc).__name__,
                    status_code=getattr(exc, "status_code", None),
                )
                self.on_abandoned(task.id)
                self._answer_open_calls()
                if self.reliability.recover_failures:
                    self.observe("backend.recovered", delegation_id=task.id, action="apology")
                    await self._deliver(APOLOGY, task)
            finally:
                self.active = None

    def _round_payload(self, task: DelegatedTask, incoming: list[dict]) -> dict:
        """Build one round's request: a stored chain for stateful backends, the full history otherwise."""
        payload = self.task.payload()
        if self.thinker.stateful:
            payload.update(input=deepcopy(incoming), store=True)
            payload["prompt_cache_key"] = self.cache_key
            if self.previous_response_id:
                payload["previous_response_id"] = self.previous_response_id
        else:
            # A stateless chat API has no stored chain: send the audited history, which already holds
            # earlier rounds' items and function results.
            payload["input"] = deepcopy(self.history)
        return payload

    async def _stream_round(self, task: DelegatedTask, payload: dict) -> tuple[str, list[dict], dict[str, dict]]:
        """Run one backend round; return its text, output items and function calls."""
        for recovery_attempt in range(2):
            final_text, terminal, output_items, calls = "", None, [], {}
            try:
                async with asyncio.timeout(self.reliability.api_timeout_seconds):
                    async for raw in self.thinker.stream(payload):
                        kind = raw.get("type", "")
                        if self.on_event is not None:
                            await self.on_event(task, raw)
                        if raw.get("item", {}).get("type") == "function_call" or kind.startswith(
                            "response.function_call"
                        ):
                            task.emitted_function = True
                        if kind == "response.output_item.done":
                            output_items.append(raw["item"])
                            if raw["item"].get("type") == "function_call":
                                calls[raw["item"]["call_id"]] = raw["item"]
                        elif kind == "response.output_text.delta":
                            final_text += raw.get("delta", "")
                        elif kind in {"response.completed", "response.incomplete", "response.failed"}:
                            terminal = kind
                            if kind == "response.completed":
                                # Only a completed response can be chained from; a failed one is not stored usefully.
                                self.previous_response_id = raw["response"]["id"]
                if terminal != "response.completed":
                    raise RuntimeError("Backend did not complete successfully")
                return final_text, output_items, calls
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if not self.reliability.recover_failures or recovery_attempt or task.emitted_function:
                    raise
                self.observe(
                    "backend.recovered",
                    delegation_id=task.id,
                    round=task.round,
                    error_type=type(exc).__name__,
                    action="retry_round",
                )
        raise RuntimeError("unreachable")

    async def _run_tools(self, calls: dict[str, dict]) -> list[dict]:
        """Run a round's calls and return the input items for the next round.

        An executor that owns whole batches (the client's tools) receives the batch; otherwise every call runs
        concurrently and yields a ``function_call_output`` item.
        """
        if hasattr(self.tools, "execute_batch"):
            return await self.tools.execute_batch(calls)

        async def one(call_id: str, call: dict) -> dict:
            if call_id in self.completed_calls:
                return self.completed_calls[call_id]
            async with asyncio.timeout(self.reliability.tool_timeout_seconds):
                output = await self.tools.execute(call_id, call.get("name", ""), call.get("arguments") or "{}")
            return {"type": "function_call_output", "call_id": call_id, "output": output}

        results = await asyncio.gather(*(one(call_id, call) for call_id, call in calls.items()))
        for item in results:
            self.completed_calls[item["call_id"]] = item
        if len(self.completed_calls) > MAX_COMPLETED_CALLS:
            self.completed_calls = dict(list(self.completed_calls.items())[-MAX_COMPLETED_CALLS // 2 :])
        return list(results)

    async def _task(self, task: DelegatedTask) -> None:
        incoming = [*self.carry, *task.input]
        self.carry = []
        for index in range(self.reliability.max_tool_rounds):
            task.round = index + 1
            task.emitted_function = False
            self.history.extend(incoming)
            self._trim_history()
            if approx_tokens(json.dumps(self.history)) > self.context_tokens:
                raise RuntimeError("Backend context budget exceeded")
            payload = self._round_payload(task, incoming)
            started = time.monotonic()
            self.observe(
                "backend.started",
                delegation_id=task.id,
                round=task.round,
                model=self.task.model,
                history_tokens=approx_tokens(json.dumps(self.history)),
            )
            final_text, output_items, calls = await self._stream_round(task, payload)
            self.observe(
                "backend.completed",
                delegation_id=task.id,
                round=task.round,
                calls=len(calls),
                duration_ms=round((time.monotonic() - started) * 1000, 3),
            )
            self.history.extend(output_items)
            if not calls:
                if final_text.strip():
                    await self._deliver(final_text, task)
                else:
                    # The caller was told to wait; silence now would leave them waiting.
                    self.observe("backend.empty", delegation_id=task.id, round=task.round)
                    self.on_abandoned(task.id)
                    await self._deliver(APOLOGY, task)
                return
            self.observe("tools.started", delegation_id=task.id, round=task.round, call_ids=list(calls))
            incoming = await self._run_tools(calls)
            self.observe("tools.completed", delegation_id=task.id, round=task.round)
        raise RuntimeError("Maximum tool rounds exceeded")

    async def _deliver(self, text: str, task: DelegatedTask) -> None:
        """Hand a result to ``on_answer``; a failure there is logged and never stops the worker."""
        try:
            await self.on_answer(text, task)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"Delivering the answer for {task.id} failed: {type(exc).__name__}: {exc}")
            self.observe("answer.failed", delegation_id=task.id, error_type=type(exc).__name__)

    def _trim_history(self) -> None:
        """Drop the oldest tasks' items, whole, until the history fits the budget (the newest task always stays)."""
        while approx_tokens(json.dumps(self.history)) > self.context_tokens:
            start = next((i for i, item in enumerate(self.history) if i and _starts_a_request(item)), None)
            if start is None:
                return
            del self.history[:start]

    def _answer_open_calls(self) -> None:
        """Answer function calls a failed round left without outputs, so the history stays valid for the next task.

        A stateless backend is sent the history, a stateful one chains from its last stored response: either way a
        call without an output is rejected. The outputs say the call did not complete.
        """
        answered = {item.get("call_id") for item in self.history if item.get("type") == "function_call_output"}
        owed = [
            {
                "type": "function_call_output",
                "call_id": item["call_id"],
                "output": json.dumps({"error": "The call did not complete."}),
            }
            for item in self.history
            if item.get("type") == "function_call" and item.get("call_id") not in answered
        ]
        if self.thinker.stateful:
            self.carry.extend(owed)
        else:
            self.history.extend(owed)

    async def close(self) -> None:
        """Stop the worker and mark every unfinished task abandoned."""
        self.closed = True
        if self.active:
            self.on_abandoned(self.active.id)
        while not self.queue.empty():
            self.on_abandoned(self.queue.get_nowait().id)
        if self.worker:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
