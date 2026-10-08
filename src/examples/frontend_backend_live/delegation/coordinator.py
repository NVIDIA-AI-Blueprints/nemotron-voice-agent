# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delegation, independent of whichever frontend speaks to the caller.

The coordinator owns everything about delegated work: the backend worker, delegation ids, the client-owned
mode, the voice turns the delegate is shown, what the caller was already told, and whether an answer has been
superseded. A frontend talks to it in two directions:

* it calls :meth:`delegate` when the caller's request needs the backend, and reports the caller-visible
  conversation with :meth:`remember_voice`;
* it implements :class:`ResultSink`, which receives each backend result.

How a result is phrased and spoken is the frontend's concern, so the same coordinator serves the Pipecat cascade
and any other frontend.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Protocol

from loguru import logger

from examples.frontend_backend_live.common.ids import uid
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation.tasks import RESPONSES, BackendTask, DelegatedTask
from examples.frontend_backend_live.delegation.thinker import Thinker
from examples.frontend_backend_live.delegation.worker import DelegationWorker

Observe = Callable[..., None]


def log_observation(kind: str, **fields) -> None:
    """Default observation hook: one debug log line per event."""
    logger.debug(f"live.{kind} {fields}")


class ResultSink(Protocol):
    """The frontend's side of a delegation: it receives each backend result."""

    async def on_result(self, text: str, task: DelegatedTask, told: str | None, *, superseded: bool) -> None:
        """Handle a verified backend result (or the recovery apology).

        Args:
            text: The result text.
            task: The delegation it answers.
            told: What the caller was already told for this delegation, if anything.
            superseded: True when a newer delegation is queued, so the result should be recorded but not spoken.
        """


class DelegationCoordinator:
    """Runs delegated work for one call and tracks what the caller has been told."""

    def __init__(
        self,
        thinker: Thinker,
        config: LiveConfig,
        *,
        backend_task: BackendTask | None,
        tools,
        sink: ResultSink,
        mode: str = RESPONSES,
        on_event: Callable[[DelegatedTask, dict], Awaitable[None]] | None = None,
        on_client_delegation: Callable[[str], Awaitable[None]] | None = None,
        observe: Observe = log_observation,
    ):
        """Wire the backend worker (``responses`` mode) and the result sink.

        Args:
            thinker: The backend model wrapper.
            config: The validated example configuration.
            backend_task: The model, instructions and tool schemas every backend round uses (``None`` in
                ``client`` mode).
            tools: A ``ToolExecutor`` for the thinker's function calls (unused in ``client`` mode).
            sink: Receives each backend result.
            mode: ``responses`` (server-owned backend) or ``client`` (the client owns delegated work).
            on_event: Optional awaited hook with ``(task, raw_event)`` for every backend event.
            on_client_delegation: Awaited with the id of each delegation announced in ``client`` mode.
            observe: Optional ``observe(kind, **fields)`` hook for traces.
        """
        self.thinker, self.sink, self.observe, self.mode = thinker, sink, observe, mode
        self.on_client_delegation = on_client_delegation
        self.worker = (
            DelegationWorker(
                thinker,
                backend_task,
                tools,
                config.reliability,
                on_answer=self.on_answer,
                on_abandoned=lambda task_id: self.told.pop(task_id, None),
                on_event=on_event,
                observe=observe,
                context_tokens=config.backend_context_tokens,
            )
            if mode == RESPONSES
            else None
        )
        # What the caller was already told for a delegation (its lead-in), so its answer does not repeat it.
        self.told: dict[str, str] = {}
        # Delegation ids issued in ``client`` mode; commentary may refer to them.
        self.client_delegations: set[str] = set()
        # Voice turns since the previous delegation, for the delegate's context.
        self.pending_voice: list[dict] = []

    def validate_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""
        self.thinker.backend.validate_model(model)

    def start(self) -> None:
        """Start the delegation worker (``responses`` mode)."""
        if self.worker:
            self.worker.start()

    async def close(self) -> None:
        """Stop the worker."""
        if self.worker:
            await self.worker.close()

    # ----------------------------------------------------------------------- the conversation
    def remember_voice(self, role: str, text: str) -> None:
        """Record one caller-visible turn for the next delegation's context."""
        self.pending_voice.append({"role": role, "content": text})

    def _delegate_input(self, text: str, *, instruction: bool = False) -> list[dict]:
        """The delegate's input: the voice turns since its last delegation, then the current request."""
        context = "Voice conversation context (reference data):\n" + json.dumps(self.pending_voice)
        label = "Current instruction" if instruction else "Current request"
        return [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": f"{context}\n{label}: {text}"}],
            }
        ]

    # ------------------------------------------------------------------------- delegation
    async def delegate(self, text: str, *, instruction: bool = False, record: bool = True) -> str:
        """Delegate ``text`` and return the delegation id.

        In ``responses`` mode the backend worker takes the task; in ``client`` mode the delegation is announced
        and the client does the work. ``record=False`` is for a frontend that has already recorded the caller's
        words itself, so ``text`` is only the request.
        """
        if record and not instruction:
            self.remember_voice("user", text)
        if self.mode == RESPONSES:
            delegation_id = await self.worker.submit(self._delegate_input(text, instruction=instruction))
        else:
            delegation_id = uid("item")
            self.client_delegations.add(delegation_id)
            if self.on_client_delegation:
                await self.on_client_delegation(delegation_id)
        self.pending_voice.clear()
        return delegation_id

    async def submit_items(self, items: list[dict]) -> str:
        """Start a backend task from typed input items (``response.create`` with queued messages)."""
        task_id = await self.worker.submit(items)
        self.pending_voice.clear()
        return task_id

    @property
    def backend_busy(self) -> bool:
        """Whether a backend task is running or queued."""
        return bool(self.worker) and (self.worker.active is not None or not self.worker.queue.empty())

    async def on_answer(self, text: str, task: DelegatedTask) -> None:
        """Hand a backend result to the frontend, noting whether a newer delegation supersedes it."""
        superseded = self.worker.has_newer
        if superseded:
            # The newer delegation's answer supersedes this one; the frontend still records the verified text.
            self.observe("commentary.suppressed", delegation_id=task.id, reason="newer_delegation")
        await self.sink.on_result(text, task, self.told.pop(task.id, None), superseded=superseded)
