# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Works and tasks: what runs, what it reads, and how it reaches the wire."""

from __future__ import annotations

import asyncio
import json
import time
from bisect import bisect_left
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    BackendRun,
    BoundaryInput,
    Delivery,
    Outcome,
    RefusalDetails,
    Route,
    Steering,
    StopOutcome,
    TaskContext,
    TaskSummary,
    WorkSummary,
    allowed_tools,
    listed,
    message_phase,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.access import read_only_sources
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.conversation import (
    TaskHistory,
    cut_output,
    item_size,
    with_ids,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.input import (
    InputScope,
    routing_context,
    scoped_items,
    settings_for,
    task_request,
    user_message,
    with_parts,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.results import (
    CallReports,
    ControlCalls,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stop import (
    AtBackstop,
    AtDeadline,
    StopReason,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stream import (
    ResponseWriter,
    output_message,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.task_lifecycle import TaskLifecycle
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.task_presentation import (
    TaskPresentation,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.tools import (
    WorkTools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.worker_control import (
    WorkerControl,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import (
    WorkerData,
    call_arguments,
    message_text,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import (
    InputReceipt,
    Runner,
    Task,
    Work,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import (
    Steering as SteeringRecord,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import CHAT_COMPLETIONS
from nemotron_voice_agent.pipecat.shared.ids import uid

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.config import Texts
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import IMSession

# The reasons of a cut attempt that end the response incomplete; any other failure ends it failed.
OFFICIAL_REASONS = frozenset({"max_output_tokens", "content_filter"})
TOO_MANY_REQUESTS = 429
RECORD_TEXT_CHARS = 300
RECORD_STEPS = 10
ITEM_REPORTS = CallReports()
DISPLAY_REPORTS = CallReports(retain_output=True)
CONTROL_REPORTS = ControlCalls()


@dataclass(frozen=True)
class Failure:
    """How a failed task ends its Work's response: ``response.failed`` or ``response.incomplete``."""

    kind: str
    fields: dict

    @classmethod
    def of(cls, error: BaseException, terminal: dict | None) -> Failure:
        """Return the failure ``error`` gives, read from the task's last attempt when it ended with one."""
        response = (terminal or {}).get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason")
        if reason in OFFICIAL_REASONS:
            return cls("response.incomplete", {"incomplete_details": {"reason": reason}})
        detail = f"{type(error).__name__}: {error}"[:300]
        # A 429 wins over the attempt's code, which the run may have set to server_error before it retried.
        limited = getattr(error, "status_code", None) == TOO_MANY_REQUESTS
        code = "rate_limit_exceeded" if limited else (response.get("error") or {}).get("code") or "server_error"
        return cls("response.failed", {"error": {"code": code, "message": detail}})


NO_ANSWER = Failure("response.failed", {"error": {"code": "server_error", "message": "The backend gave no answer."}})


@dataclass(eq=False)
class Batch:
    """Requests steered into a Work: one ``Steering`` record per task, landed once by the first task to apply it."""

    delegations: list[str]
    items: list[dict]
    brings: dict | None
    landed: bool = False
    caller_input: bool = True
    placement_refs: tuple[str, ...] = ()
    call_generation: int = 0
    control: bool = True
    scopes: tuple[InputScope, ...] = ()

    def worker_items(self) -> list[dict]:
        """Return the task-scoped input, keeping receipts tied to the original items."""
        return scoped_items(self.items, self.scopes)


@dataclass(eq=False)
class HeldEmission:
    """One original output operation; a never-forwarded app batch still requires its commit gate."""

    action: Callable[[ResponseWriter], None]
    task: TaskRun | None = None
    kind: str = "output"
    calls: tuple[dict, ...] = ()
    publication_call_id: str | None = None

    @property
    def discarded(self) -> bool:
        """Whether the caller-message receipt already settled separately from this held item."""
        if self.task is None or self.publication_call_id is None:
            return False
        return not self.task.presentation.is_pending(self.publication_call_id)

    def discard(self, reason: str) -> None:
        """Settle the exact caller-message receipt, when this item has one."""
        if self.task is not None and self.publication_call_id is not None:
            self.task.presentation.drop(self.publication_call_id, reason)


@dataclass(eq=False)
class Waiting:
    """A batch waiting in one task: the batch, and that task's ``Steering`` record of it."""

    batch: Batch
    record: SteeringRecord


@dataclass(eq=False)
class Parked:
    """A response that ended with the app's calls, and the tasks that wait for the calls' outputs."""

    work: WorkRun
    delegation_id: str
    response_id: str
    calls: dict[str, tuple[TaskRun, dict]]
    input_revision: int
    response_ended: bool = False
    window_opened: bool = False
    timer: asyncio.TimerHandle | None = field(default=None, repr=False)


def request_text(items: list[dict]) -> str:
    """Return the newest words of a request's user messages, as the work records show them."""
    texts = []
    for item in items:
        content = item.get("content") if item.get("role") == "user" else None
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                text
                for part in content
                if isinstance(part, dict) and isinstance(text := part.get("text"), str) and text
            )
    return " ".join(texts)[-RECORD_TEXT_CHARS:]


class TaskRun:
    """One task: one worker's run, and the ``TaskContext`` it reads and reports through."""

    def __init__(self, work: WorkRun, record: Task):
        """Make a task of ``work``, run for its ``Task`` record."""
        self.work, self.session, self.record = work, work.session, record
        self.work_id, self.task_id, self.brief = work.id, record.id, record.brief
        self.conversation_ref = self.session.conversation_ref
        self.execution_ref = self.task_id
        self.context_budget = self.session.settings.context_tokens
        self.business_access = "read_only" if record.verifies is not None else "full"
        self._read_only_tools: frozenset[str] = frozenset()
        self.continues = record.continues or work.record.continues
        self.execution_owners = {self.task_id}
        references = [reference for reference in (self.continues, record.verifies) if reference is not None]
        while references:
            reference = references.pop()
            if reference in self.execution_owners:
                continue
            predecessor = self.session.ledger.tasks.get(reference)
            if predecessor is None:
                continue
            self.execution_owners.add(reference)
            references.extend(
                identity for identity in (predecessor.continues, predecessor.verifies) if identity is not None
            )
        self.history_mode = record.history_mode
        self.uncertain_inputs: list[tuple[dict, WorkerData]] = []
        self._metadata_text = ""
        self._metadata_revision = 0
        self.thread_items: list[dict] = []
        self.ancestor_size = 0
        self._execution_fact_ids: set[str] = set()
        self.initial_requested = deepcopy(work.requested)
        self.initial_scopes: tuple[InputScope, ...] = ()
        self.pending_scopes: list[InputScope] = []
        self.scope_applied = False
        self.start_settings: dict | None = None
        self.request_ids = list(work.record.requests)
        self.ended = False
        self.run: BackendRun | None = None
        self.completed_rounds = 0
        self._round_identity: dict | None = None
        self.view: TaskHistory | None = None
        self.pending: list[dict] = []
        self.steering: deque[Waiting] = deque()
        self.taken: dict[str, Waiting] = {}
        self.outputs: dict[str, dict] = {}
        self.call_order: list[str] = []
        self.routes: dict[str, Route] = {}
        self._call_bindings: dict[str, tuple[str, str | None]] = {}
        self._held_input_revisions: dict[str, int] = {}
        self._observed_tools: set[str] = set()
        self.refusals: dict[str, str] = {}
        self._refusal_details: dict[str, RefusalDetails] = {}
        self.app_calls: dict[str, dict] = {}
        self.started_calls: dict[str, CallReports] = {}
        self.reported: set[str] = set()
        self.steps: dict[str, list[str]] = {}
        self.attempt: list[dict] = []
        self.last_terminal: dict | None = None
        self.answered = False
        self.held = False
        self.completed = False
        self.failure: Failure | None = None
        self.grant: asyncio.Future | None = None
        self.outputs_in: asyncio.Future | None = None
        self.driver: asyncio.Task | None = None
        self.wake = asyncio.Event()
        self.lifecycle = TaskLifecycle(self)
        self.control = WorkerControl(self)
        self.presentation = TaskPresentation(self)
        shown = self.session.result_format.tools()
        self.displays = frozenset(shown)
        names = tuple(dict.fromkeys((*shown, *self.session.work_tools)))
        label = self.session.settings.display_label
        self.work_tools = WorkTools(
            names,
            self.texts,
            self.lifecycle.cancel_own,
            self.presentation.caller_message,
            self.continue_after_call,
            label,
            resolve=self.presentation.resolve_questions,
            set_state=self.lifecycle.set_work_state,
        )
        self._readers: dict[str, Callable[[dict], None]] = {
            "response.created": self._attempt_opened,
            "response.output_item.done": self._item_done,
            "response.completed": self._round_completed,
            "response.failed": self._attempt_failed,
            "response.incomplete": self._attempt_failed,
        }

    def begin(self) -> None:
        """Start the task's driver: it waits for a slot, then runs."""
        self.report_state(self.record.state)
        self.driver = asyncio.create_task(self._drive(), name=f"task-{self.task_id}")

    def report_state(self, state: str) -> None:
        """Trace the task's visible lifecycle state."""
        self.trace("im.task.state", source_delegation_id=self.request_ids[0], brief=self.brief, state=state)

    def move(self, state: str) -> None:
        """Update the task record and report a changed state."""
        changed = self.record.state != state
        self.record.move(state)
        if changed:
            self.report_state(
                "stop_unconfirmed" if state == "cancelled" and self.lifecycle.stopped == "unconfirmed" else state
            )

    async def _drive(self) -> None:
        """Wait for routing to settle before committing app calls that Core still owns."""
        try:
            await self._slot()
            self.run = self.work.task_started(self)
            while True:
                await self._read_events()
                while any(self.app_placement_pending(call) for call in self.app_calls.values()):
                    await self.wait_for_placement()
                if self.lifecycle.stop_reason is not None:
                    break
                if not self.lifecycle.suspension_active:
                    if not self.app_calls:
                        break
                    if self._gated():
                        await self.session.decided(require_attachment=not self.after_call)
                    elif self.session.waits_for_app(self):
                        await self._wait_for_app()
                    else:
                        break
                if self.lifecycle.suspension_active and self.lifecycle.stop_reason is None:
                    self.refuse_unpublished_app("paused", self.texts.paused_output)
                    await self.lifecycle.await_resumption()
                    if self.lifecycle.stop_reason is None:
                        await self._slot()
                if self.lifecycle.stop_reason is not None:
                    break
                await self.run.resume()
            if self.lifecycle.stop_reason is None:
                self.completed = self.answered or bool(self.app_calls)
                self.failure = None if self.completed else NO_ANSWER
        except asyncio.CancelledError:
            if self.lifecycle.stop_reason is None:
                raise
        except Exception as exc:
            if self.lifecycle.stop_reason is None:
                logger.warning(f"Task {self.task_id} failed: {type(exc).__name__}: {exc}")
                self.failure = Failure.of(exc, self.last_terminal)
                self.trace("backend.failed", error_type=type(exc).__name__, ending=self.failure.kind)
        finally:
            self.session.release(self)
            if self.lifecycle.stop_reason is None:
                self.end()

    @property
    def running(self) -> bool:
        """Whether its run holds a slot."""
        return self.record.state == "running"

    @property
    def after_call(self) -> bool:
        """Whether its Work goes on after the call, so the call's per-turn limit does not apply to its turns."""
        return self.work.record.after_call

    async def _slot(self) -> None:
        """Reserve capacity, then check that the task can actually start or resume."""
        if self.work.start_barrier is not None:
            # A logical assignment exists before its physical dependency settles. Cancelling one selected task
            # must not cancel the shared dependency for its siblings.
            await asyncio.shield(self.work.start_barrier)
        while True:
            if self.lifecycle.stop_reason is not None or self.session.closed:
                raise asyncio.CancelledError
            if not (self.after_call or self.session.attachment_ready):
                await self.session.wait_attached()
                continue
            if self.lifecycle.suspension_active:
                await self.lifecycle.await_resumption()
                continue
            self.grant = self.session.request_slot(self)
            await self.grant
            if self.lifecycle.stop_reason is not None or self.session.closed:
                raise asyncio.CancelledError
            if not self.lifecycle.suspension_active and (self.after_call or self.session.attachment_ready):
                break
            self.session.release(self)
        self.move("running")
        self.work.execution_changed()

    async def _read_events(self) -> None:
        self.answered = False
        events = self.run.events()
        try:
            async for raw in events:
                reader = self._readers.get(raw.get("type", ""))
                if reader is not None:
                    reader(raw)
                # The round's calls were refused while the conductor decides: read on once it has.
                if self.held:
                    self.held = False
                    await self.session.dispatch_placement(require_attachment=not self.after_call)
                    if self.lifecycle.stop_reason is not None:
                        return
        finally:
            close = getattr(events, "aclose", None)
            if close is not None:
                await close()

    def _gated(self) -> bool:
        """Refuse the round's app calls after the call or at the commit gate; return whether it refused them.

        Their route was decided when the round completed, and newer input may have come, or the call may have ended,
        while its backend calls ran.
        """
        texts = self.session.texts
        if self.lifecycle.stop_reason is not None or self.lifecycle.own_cancel:
            reason, output = "stopped", texts.stopped_output
        elif self.newer_input_waits():
            reason, output = "not_run", texts.not_run_output
        elif not self.session.caller_attached:
            reason, output = "attachment", texts.unavailable_output
        elif self.work.unheard():
            reason, output = "after_call", texts.after_call_output
        else:
            return False
        for call in self.app_calls.values():
            self.remember_refusal(call, reason, output)
        self.app_calls = {}
        return True

    def gate_held_app(self, calls: Sequence[dict]) -> list[dict]:
        """Recheck only app calls that have never crossed the response boundary, retaining their exact IDs."""
        allowed = []
        for call in calls:
            call_id = call.get("call_id", "")
            refusal = self.call_refusal(call)
            if refusal is None and self.routes.get(call_id) != "refused":
                allowed.append(call)
            else:
                if refusal is not None:
                    self.remember_refusal(call, *refusal)
                self.app_calls.pop(call_id, None)
        if not allowed and self.outputs_in is not None and not self.outputs_in.done():
            self.outputs_in.set_result(None)
            self.wake.set()
        return allowed

    async def _wait_for_app(self) -> None:
        """Wait, holding no slot, for the app's outputs; then for a slot again."""
        self.session.release(self)
        if not self.lifecycle.suspension_active:
            self.move("waiting_client")
        self.outputs_in = asyncio.get_running_loop().create_future()
        self.record.acted = True
        self.work.park(self, list(self.app_calls.values()))
        while (
            not self.outputs_in.done()
            or self.lifecycle.suspension_active
            or self.control.run is not None
            or self.control.unconfirmed
        ):
            if self.lifecycle.stop_reason is not None:
                return
            if self.lifecycle.suspension_active:
                await self.lifecycle.await_resumption()
                continue
            if self.control.ready():
                await self.control.process_input()
                continue
            self.wake.clear()
            if (
                (self.outputs_in.done() and self.control.run is None and not self.control.unconfirmed)
                or self.control.ready()
                or self.lifecycle.suspension_active
            ):
                continue
            await self.wake.wait()
        self.outputs_in = None
        await self._slot()
        self.work.resumed()

    def give(self, outputs: list[dict]) -> None:
        """Take the app's outputs for this task's calls."""
        for output in outputs:
            self.outputs[output["call_id"]] = output
            self.record_step(output["call_id"], "done")
        if self.outputs_in is not None and not self.outputs_in.done():
            self.outputs_in.set_result(None)
        self.wake.set()

    def take_steering(self) -> Steering:
        """Take the oldest input with the same source references for business and control calls."""
        waiting = self.steering[0]
        items = list(self.work.steering_items(waiting.batch, self))
        if waiting.batch.caller_input and any(q.open for q in self.presentation.owned_questions()):
            sources = [item["id"] for item in waiting.batch.items if item.get("role") == "user" and item.get("id")]
            if sources:
                items.append(
                    {
                        "type": "message",
                        "role": "developer",
                        "id": f"{self.task_id}.{waiting.record.id}.sources",
                        "content": [
                            {
                                "type": "input_text",
                                "text": json.dumps(
                                    {
                                        "private_backend_data": {
                                            "source_message_ids": sources,
                                            "proves_delivery": False,
                                            "caller_order": [
                                                (identity, self.session.caller_order(identity)) for identity in sources
                                            ],
                                        }
                                    }
                                ),
                            }
                        ],
                    }
                )
        self.steering.popleft()
        waiting.record.move("taken")
        steering = Steering(waiting.record.id, tuple(items))
        self.taken[steering.id] = waiting
        return steering

    def _requeue_steering(self, waiting: Waiting) -> None:
        """Restore a batch to its original position even if the adapter returns batches out of order."""
        index = bisect_left(self.steering, waiting.record.enqueue_order, key=lambda queued: queued.record.enqueue_order)
        self.steering.insert(index, waiting)

    def settle_control(self, steering: Steering, delivery: Delivery) -> None:
        """Record actual control-input application independently of the business provider's pending calls."""
        waiting = self.taken.pop(steering.id, None)
        if waiting is None:
            return
        waiting.record.move(delivery)
        if delivery == "applied":
            if waiting.batch.caller_input:
                self.presentation.landed_input(waiting.batch.items)
                self.request_ids.extend(key for key in waiting.batch.delegations if key not in self.request_ids)
            self._applied_ownership(waiting.batch)
        elif delivery == "not_applied":
            self._requeue_steering(waiting)
        else:
            self._remember_uncertain_steering(steering, waiting.batch, control_only=True)

    def _remember_uncertain_steering(self, steering: Steering, batch: Batch, *, control_only: bool = False) -> None:
        if batch.caller_input and (order := self.session.latest_input_order(batch.items)):
            self.fence_unresolved_input(order)
        receipt = {"steering_id": steering.id, "source_delegation_ids": list(batch.delegations)}
        if control_only:
            receipt["control_only"] = True
        self.remember_uncertain_input(receipt, {"steering_id": steering.id, "items": steering.items})
        self.session.ownership.delivery_unknown(self, batch)

    def remember_uncertain_input(self, receipt: dict, details: dict) -> None:
        """Keep the delivery receipt separate from the message and tool content it describes."""
        data = WorkerData(
            user_message(json.dumps({"uncertain_input_evidence": details}, ensure_ascii=False)),
            id=f"{self.task_id}.uncertain.{len(self.uncertain_inputs) + 1}",
        )
        self.uncertain_inputs.append((receipt, data))

    def end(self) -> None:
        """End the task: its record moves to how it ended."""
        if self.ended:
            return
        self.ended = True
        record = self.record
        state = (
            "cancelled"
            if self.lifecycle.stop_reason is not None
            else "failed"
            if self.failure is not None
            else "answered"
            if self.completed
            else "cancelled"
        )
        self.move(state)
        self.presentation.drop_all("task_ended")
        if state in {"cancelled", "failed"}:
            self.presentation.close_questions(state)
        self.presentation.prune()
        record.ended_at = time.monotonic()
        # No response's end waits for calls it never reported ended.
        self.session.book.drop(self.unreported())
        self.session.task_ended(self, continue_pending=True)
        self.work.task_ended()

    def refuse_unpublished_app(self, reason: str, output: str) -> None:
        """Settle calls still owned by Core and preserve any result already owed by the app."""
        for call_id, call in self.app_calls.items():
            action = self.session.ledger.actions.get(call_id)
            if action is not None and not (action.ended or action.app_published):
                self.remember_refusal(call, reason, output)

    def fence_unresolved_input(self, order: int) -> None:
        """Keep existing work from committing new effects without a newer applied caller input."""
        if not self.ended and self.record.record_unresolved_input(order):
            self.refuse_unpublished_app("routing_unresolved", self.texts.not_run_output)

    def unreported(self) -> list[str]:
        """Return the backend calls it reported started and not ended."""
        return [call_id for call_id in self.started_calls if call_id not in self.reported]

    def _applied_ownership(self, batch: Batch) -> None:
        """Actual input evidence can arrive after stopping; it cannot reopen a settled publication obligation."""
        if self.ended or self.lifecycle.stop_reason is not None:
            for control in self.session.ledger.controls.values():
                if self.task_id in control.delivery and set(batch.delegations).intersection(control.sources):
                    control.receipt.setdefault("late_input_delivery", {})[self.task_id] = "applied"
            self.trace("im.input.applied_after_stop", sources=list(batch.delegations))
            return
        self.session.ownership.delivered(self, batch)
        self.work.land(batch)

    def continue_after_call(self, message: str) -> str:
        """Hand ``message`` to the Work for the task's own ``continue_after_call``; return what the Work answers."""
        return self.work.continue_after_call(self, message)

    def _attempt_opened(self, raw: dict) -> None:
        self.attempt, self.last_terminal = [], None
        self._round_identity = None
        self._round_identity = self._usage_metadata(raw)

    def _item_done(self, raw: dict) -> None:
        item = raw.get("item")
        if isinstance(item, dict):
            self.attempt.append(item)

    def _attempt_failed(self, raw: dict) -> None:
        # A failed or cut attempt writes nothing.
        self.trace_round_usage(raw)
        self.attempt, self.last_terminal = [], raw

    def trace_round_usage(self, raw: dict, *, context: TaskContext | None = None) -> None:
        """Retain a terminal attempt's reported usage, including failed or cut attempts that wrote no items."""
        usage = (raw.get("response") or {}).get("usage")
        self.completed_rounds += 1
        self.trace(
            "backend.round.usage",
            **self._usage_metadata(raw, context=context),
            role="worker",
            round=self.completed_rounds,
            terminal=raw.get("type"),
            conversation_ref=self.conversation_ref,
            response_id=(raw.get("response") or {}).get("id"),
            usage=usage,
            usage_missing=usage is None,
        )

    def _usage_metadata(self, raw: dict, *, context: TaskContext | None = None) -> dict:
        """Keep the attempt's resolved identity, preferring the model recorded in its response envelope."""
        task = self if context is None else context
        if context is not None or self._round_identity is None:
            identity = getattr(self.session.backend, "usage_identity", None)
            metadata = identity(task) if identity is not None else {"backend": None, "model": None}
        else:
            metadata = dict(self._round_identity)
        metadata.setdefault("requested_model", task.request().get("model"))
        metadata.setdefault("configured_model", metadata.get("model"))
        model = (raw.get("response") or {}).get("model")
        if isinstance(model, str) and model.strip():
            translated = metadata.get("provider_api") == CHAT_COMPLETIONS
            metadata.update(
                response_model=model,
                response_model_source="translated_request" if translated else "provider_event",
                reported_model=None if translated else model,
            )
            if not translated:
                metadata.update(model=model, model_source="provider_event")
        return metadata

    def _round_completed(self, raw: dict) -> None:
        items, self.attempt, self.last_terminal = self.attempt, [], None
        self.trace_round_usage(raw)
        usage = (raw.get("response") or {}).get("usage")
        writer = self.session.book.writer(self.work.carrier)
        if usage and writer is not None and not self.session.ownership.managed(self.work):
            writer.usage = usage
        calls = [item for item in items if item.get("type") == "function_call"]
        messages = [item for item in items if item.get("type") == "message"]
        finals = [] if calls else [item for item in messages if message_phase(item) == "final_answer"]
        answer = finals[-1] if finals else None
        if answer is not None and not answer.get("id"):
            answer["id"] = uid("msg")
        self.view.join(items)
        self.thread_items.extend(deepcopy(items))
        for message in messages:
            if message is not answer:
                self.session.commentary(self, message)
        self.call_order = [call.get("call_id", "") for call in calls]
        self.app_calls = {}
        for call in calls:
            if call.get("call_id") not in self._observed_tools and self.route(call) == "app":
                self.app_calls[call.get("call_id", "")] = call
                self.session.action(self, call)
        refused = any(self.routes.get(call.get("call_id", "")) == "refused" for call in calls)
        self.held = (
            refused
            and (self.session.waits_for_decision or (not self.session.caller_attached and not self.after_call))
            and self.lifecycle.stop_reason is None
        )
        if answer is not None:
            self._answer(answer)

    def _answer(self, message: dict) -> None:
        """Mark the round answered, keep its text on the task's record, and have the Work write ``message``."""
        self.answered = True
        self.record.answer = message_text(message)
        self.record.answer_item_id = message["id"]
        self.record.answer_publication = "pending"
        self.trace("im.answer.ready")
        self.work.answer(self, message)

    @property
    def texts(self) -> Texts:
        """The core's texts."""
        return self.session.texts

    def request(self) -> dict:
        """Return the settings of the task's next model call, its instructions composed."""
        if self.start_settings is None:
            self.start_settings = self.work.request()
        request = deepcopy(self.start_settings)
        if self.business_access != "full":
            request.update(tools=[], tool_choice="auto")
        return request

    def history(self, extra: Sequence[dict] = ()) -> list[dict]:
        """Return task history, private state and required extras within the same budget, without applying input."""
        return self.view.history((*self.backend_state(extra), *extra))

    def refresh_history(self, *, unconfirmed_calls: Sequence[str] = ()) -> None:
        """Refresh exact inherited facts without replacing this task's own history or admitting new caller input."""
        if self.view is None:
            return
        unconfirmed = set(unconfirmed_calls) & self.session.ledger.actions.keys()
        if any(self.session.ledger.actions[call_id].task not in self.execution_owners for call_id in unconfirmed):
            raise ValueError("Execution facts must name only recorded calls of this exact task thread")
        prefix = self.view.items[: self.view.base]
        own = self.view.own()
        ancestor = self.session.retained_thread(self.continues)
        if ancestor is not None:
            if self.history_mode == "focused":
                prefix = [*ancestor, *prefix[self.ancestor_size :]]
            self.thread_items[: self.ancestor_size] = deepcopy(ancestor)
            self.ancestor_size = len(ancestor)

        def reconcile(items: list[dict]) -> list[dict]:
            refreshed = []
            for item in items:
                if item.get("type") == "function_call_output":
                    action = self.session.ledger.actions.get(item.get("call_id"))
                    if action is not None:
                        output = (
                            action.late_output
                            if action.late_outcome is not None
                            else action.output
                            if action.state in ("done", "failed")
                            else None
                        )
                        if output is not None:
                            item = {**item, "output": output}
                refreshed.append(item)
            return refreshed

        view = TaskHistory(reconcile(prefix), self.view.budget, self.view.output_chars, self.view.marker)
        view.join(reconcile(own))
        self.view = view
        self.thread_items = reconcile(self.thread_items)
        calls = {item.get("call_id") for item in view.items if item.get("type") == "function_call"}
        outputs = {item.get("call_id") for item in view.items if item.get("type") == "function_call_output"}
        represented = calls & outputs
        self._execution_fact_ids = unconfirmed | {
            action.id
            for action in self.session.ledger.actions.values()
            if action.task in self.execution_owners
            and action.runner != "im"
            and (action.id not in represented or not action.ended)
        }
        self.history()

    def _execution_facts(self) -> list[dict]:
        """Read current exact execution receipts that ordinary native history cannot establish."""
        facts = []
        for action in self.session.ledger.actions.values():
            if action.task not in self.execution_owners or action.runner == "im":
                continue
            if action.id not in self._execution_fact_ids and action.late_outcome is None:
                continue
            recorded = action.execution_fact()
            output = recorded["output"]
            fact = {key: recorded[key] for key in ("call_id", "task_id", "name", "arguments", "outcome", "has_result")}
            fact["output"] = None
            if action.late_outcome is not None:
                fact["late_evidence"] = True
            if fact["has_result"] and output is not None:
                trimmed = cut_output(
                    {"type": "function_call_output", "output": output},
                    self.session.settings.tool_output_chars,
                    self.texts.cut_output,
                )["output"]
                fact.update(output=trimmed, output_truncated=trimmed != output)
            facts.append(fact)
        return facts

    def backend_state(self, extra: Sequence[dict] = ()) -> tuple[dict, ...]:
        """Fresh host state for adapters; it grants no input application or business authority."""
        budget = self.view.extra_budget(extra) if self.view is not None else self.context_budget
        return self.backend_state_for_budget(budget)

    def backend_state_for_budget(self, budget: int | None) -> tuple[dict, ...]:
        """Render canonical task evidence for a backend request with this budget."""
        questions = self.presentation.owned_questions()
        executions = self._execution_facts()
        if not self.session.work_tools and not executions:
            return ()
        if (
            self.brief
            or questions
            or self.uncertain_inputs
            or self.presentation.receipts
            or self.presentation.input_revision > 1
            or executions
            or self.lifecycle.suspension_active
            or self.business_access != "full"
        ):
            evidence = {
                "private_backend_state": {
                    "task_id": self.task_id,
                    "business_access": self.business_access,
                    "verification_of": self.record.verifies,
                    "execution_state": self.record.state,
                    "pause_requested": self.lifecycle.want_paused,
                    "pause_confirmation": self.lifecycle.pause_outcome,
                    "initial_assignment": self.brief,
                    "continues": self.continues,
                    "questions": [
                        {
                            "question_id": q.id,
                            "text": q.text,
                            "creator_task_id": q.creator_task_id,
                            "owner_task_id": q.owner_task_id,
                            "previous_owners": q.previous_owners,
                            "created_input_revision": q.created_input_revision,
                            "dispatch": q.dispatch,
                            "resolution": q.resolution,
                            "input_revision": q.input_revision,
                            "input_ref": q.input_ref,
                            "response_id": q.response_id,
                            "item_id": q.item_id,
                            "publication_call_id": q.publication_call_id,
                        }
                        for q in questions
                        if q.open
                    ],
                    "recent_resolutions": [
                        {"question_id": q.id, "disposition": q.resolution, "input_ref": q.input_ref}
                        for q in questions
                        if not q.open
                    ][-16:],
                    "publications": list(self.presentation.receipts.values()),
                    "uncertain_inputs": [receipt for receipt, _ in self.uncertain_inputs],
                    "applied_inputs": [
                        {
                            "input_ref": row.id,
                            "revision": row.revision,
                            "source_message_ids": row.source_message_ids,
                            "caller_order": row.caller_order,
                        }
                        for row in self.presentation.input_receipts.values()
                    ],
                }
            }
            if executions:
                evidence["private_backend_state"]["execution_facts"] = executions
            details = tuple(data for _, data in self.uncertain_inputs)
            if budget is not None:
                budget -= sum(map(item_size, details))
            item = self._fit_backend_state(evidence, budget)
            encoded = item["content"][0]["text"]
            if encoded != self._metadata_text:
                self._metadata_text = encoded
                self._metadata_revision += 1
            item["id"] = f"{self.task_id}.state.{self._metadata_revision}"
            return (item, *details)
        return ()

    @staticmethod
    def _fit_backend_state(evidence: dict, budget: int | None) -> dict:
        """Make room by omitting historical rows from the view without changing their canonical receipts."""
        content = {"type": "input_text"}
        item = {"type": "message", "role": "developer", "content": [content]}

        def fits() -> bool:
            content["text"] = json.dumps(evidence)
            return budget is None or item_size(item) <= budget

        if fits():
            return item
        state = evidence["private_backend_state"]
        publications = state["publications"]
        inputs = state["applied_inputs"]
        resolutions = state["recent_resolutions"]
        latest = inputs[-1:]
        referenced = {row["input_ref"] for row in resolutions}
        for rows, optional in (
            (publications, [row for row in publications if row["state"] != "pending"]),
            (inputs, [row for row in inputs[:-1] if row["input_ref"] not in referenced]),
        ):
            for row in optional:
                rows.remove(row)
                if fits():
                    return item
        while resolutions:
            ref = resolutions.pop(0)["input_ref"]
            if not any(row["input_ref"] == ref for row in [*latest, *resolutions]):
                inputs[:] = [row for row in inputs if row["input_ref"] != ref]
            if fits():
                return item
        raise RuntimeError("Backend context budget exceeded")

    def tools(self) -> WorkTools:
        """Return the IM's work tools for this task."""
        return self.work_tools

    def tool_sources(self, sources: Sequence) -> tuple:
        """A verification execution can use only host-declared read-only source capabilities."""
        selected, self._read_only_tools = read_only_sources(sources)
        return tuple(sources) if self.business_access == "full" else selected

    def route(self, call: dict) -> Route:
        """Return who runs ``call``, decided once per call, when the adapter or the round reader first asks."""
        call_id = self._bind_call(call)
        if call_id not in self.routes:
            name = call.get("name", "")
            refusal = self.call_refusal(call)
            self.routes[call_id] = "refused" if refusal is not None else self._route(name)
            self.record_step(call_id, "started", name)
            if refusal is not None:
                self.remember_refusal(call, *refusal)
            else:
                self.session.call_routed(call)
        return self.routes[call_id]

    def _bind_call(self, call: dict) -> str:
        """An actual call identity has one owner and argument binding, including rejected or failed attempts."""
        identity, name = call.get("call_id"), call.get("name")
        if not isinstance(identity, str) or not identity or not isinstance(name, str) or not name:
            raise ValueError("A tool call requires nonempty call and tool identities")
        if identity in self._observed_tools:
            raise ValueError("An observed external operation cannot become a Core execution request")
        signature = (name, call_arguments(call, preserve_invalid=True))
        previous = self._call_bindings.get(identity)
        if previous is not None and previous != signature:
            raise ValueError("A tool call identity cannot be reused with different arguments")
        action = self.session.ledger.actions.get(identity)
        if action is not None:
            if (action.task, action.work, action.name) != (self.task_id, self.work_id, name):
                raise ValueError("A tool call identity cannot be reused by another task or tool")
            if action.arguments is not None:
                recorded = call_arguments({"arguments": action.arguments}, preserve_invalid=True)
                if recorded != signature[1]:
                    raise ValueError("A recorded tool call cannot be reused with different arguments")
        self._call_bindings[identity] = signature
        return identity

    def permits_output(
        self,
        *,
        allow_ended: bool = False,
        allow_suspended: bool = False,
        allow_detached: bool = False,
        revision: int | None = None,
    ) -> bool:
        """Check task, caller and input state for the requested kind of output."""
        return (
            self.stop_reason is None
            and not self.own_cancel
            and (allow_ended or not self.ended)
            and (allow_suspended or not self.lifecycle.suspension_active)
            and (allow_detached or self.session.caller_attached)
            and (revision is None or revision == self.input_revision)
        )

    def can_dispatch(self) -> bool:
        """Check whether the business worker may dispatch a call."""
        return self.permits_output(allow_detached=self.after_call)

    def publication_guard(self) -> Callable[[], bool]:
        """Capture immutable input scope for text that may be delayed before presentation."""
        revision = self.input_revision
        attribution = self.session.ownership.attribution(self.work)
        return lambda: (
            self.permits_output(allow_ended=True, allow_suspended=True, revision=revision)
            and not self.newer_input_waits()
            and not self.work.unheard()
            and attribution == self.session.ownership.attribution(self.work)
        )

    def dispatch(self, call: dict, *, suspended: bool = False) -> Route:
        """Recheck actual dispatch permission without changing an earlier refusal into a grant."""
        if suspended:
            call_id = self._bind_call(call)
            action = self.session.ledger.actions.get(call_id)
            if call_id not in self.started_calls and not (action and (action.ended or action.app_published)):
                self.remember_refusal(call, "paused", self.texts.paused_output)
            return "refused"
        route = self.route(call)
        if route != "refused":
            refusal = self.call_refusal(call)
            if refusal is not None:
                self.remember_refusal(call, *refusal)
                return "refused"
        return route

    def placement_pending(self, call: dict) -> bool:
        """Inspect a call before dispatch without recording a route, refusal, or action."""
        route = self.routes.get(call.get("call_id", ""))
        return (
            route != "refused"
            and (route is None or self.session.caller_turn_pending)
            and self.app_placement_pending(call)
        )

    def app_placement_pending(self, call: dict) -> bool:
        """Whether frontend routing or global placement is all that prevents dispatch."""
        if (
            self.lifecycle.stop_reason is not None
            or self.steering
            or self.taken
            or self._held_input_changed(call)
            or not (self.session.waits_for_decision or self.session.caller_turn_pending)
        ):
            return False
        refusal = self.call_refusal(call)
        if refusal is None or refusal[0] != "not_run":
            return False
        if self.session.caller_turn_pending and (call_id := call.get("call_id")):
            self._held_input_revisions.setdefault(call_id, self.presentation.input_revision)
        return True

    def _held_input_changed(self, call: dict) -> bool:
        return (
            self._held_input_revisions.get(call.get("call_id", ""), self.presentation.input_revision)
            != self.presentation.input_revision
        )

    async def wait_for_placement(self) -> None:
        """Use the existing placement barrier; attachment and every call permission are checked afterward."""
        await self.session.dispatch_placement()

    def refusal(self, call_id: str) -> str:
        """Return the not-run output the IM gave a call it refused."""
        return self.refusals[call_id]

    def refusal_details(self, call_id: str) -> RefusalDetails:
        """Return the original route-time snapshot; later inputs, stops, and repeated calls do not change it."""
        return self._refusal_details[call_id]

    def pending_input_ready(self) -> bool:
        """Recheck current owned input eligibility; the original refusal remains an immutable snapshot."""
        return (
            bool(self.steering or self.taken)
            and (self.session.caller_attached or self.after_call)
            and not (self.ended or self.lifecycle.stop_reason is not None or self.lifecycle.own_cancel)
        )

    def remember_refusal(self, call: dict, reason: str, output: str) -> None:
        """Capture the first actual refusal, including a call newly refused at the late commit gate."""
        call_id = call.get("call_id", "")
        if call_id in self._refusal_details:
            return
        scope = "none"
        if reason == "not_run":
            scope = "owned" if self.steering or self.taken or self._held_input_changed(call) else "global"
        self.routes[call_id], self.refusals[call_id] = "refused", output
        self._refusal_details[call_id] = RefusalDetails(
            self.work_id,
            self.task_id,
            call_id,
            len(self._refusal_details) + 1,
            reason,
            scope,
            self.session.caller_attached,
        )
        self._refuse(call, reason)

    def call_refusal(self, call: dict) -> tuple[str, str] | None:
        """Return why the IM will not run ``call``, as the reason and the not-run output, or ``None`` when it runs.

        Stops, reserved-name clashes and tool restrictions take precedence. Then the commit gate reports newer
        input before attachment or after-call absence. Repeated application calls reuse only this task's confirmed
        output when no request arrived since.
        """
        texts, name = self.session.texts, call.get("name", "")
        settings = (
            settings_for(self.start_settings, self.session.defaults)
            if self.start_settings is not None
            else self.work.settings
        )
        declared = [tool for tool in settings.tools if tool.get("type") == "function" and tool.get("name") == name]
        if self.ended or self.lifecycle.stop_reason is not None or self.lifecycle.own_cancel:
            return "stopped", texts.stopped_output
        if self.lifecycle.suspension_active and name not in self.work_tools.names:
            return "paused", texts.paused_output
        if self.business_access != "full" and name not in self.work_tools.names and name not in self._read_only_tools:
            return "unlisted", texts.unlisted_output
        if declared and name in self.work_tools.names:
            return "clash", texts.clash_output
        if declared:
            for choice in (settings.tool_choice, self.work.settings.tool_choice):
                allowed = allowed_tools(choice)
                forced_elsewhere = (
                    isinstance(choice, dict) and choice.get("type") == "function" and choice.get("name") != name
                )
                if choice == "none" or forced_elsewhere or not any(listed(tool, allowed) for tool in declared):
                    return "unlisted", texts.unlisted_output
        if (
            self.newer_input_waits()
            or self._held_input_changed(call)
            or self.session.caller_turn_pending
            and (declared or name not in self._read_only_tools)
        ):
            return "not_run", texts.not_run_output
        if (
            self.record.unresolved_input_order
            and name not in self.work_tools.names
            and (declared or name not in self._read_only_tools)
        ):
            return "routing_unresolved", texts.not_run_output
        held_at_end = (self.session.closed or self.session.detached) and (
            call.get("call_id") in self._held_input_revisions
        )
        if (held_at_end or not self.session.caller_attached) and (
            declared or name == "caller_message" or not self.after_call
        ):
            return "attachment", texts.unavailable_output
        if (declared or name == "caller_message") and self.work.unheard():
            return "after_call", texts.after_call_output
        # A Work that took another request may ask for the same call again.
        if not declared or len(self.work.requests) > 1:
            return None
        output = self.session.repeated_output(call, self.task_id)
        return ("repeated", output) if output is not None else None

    def _route(self, name: str) -> Route:
        """Return who runs a call the IM lets run: the app for a function the request declared, else the backend."""
        declared = any(
            tool.get("type") == "function" and tool.get("name") == name for tool in self.request().get("tools", [])
        )
        return "app" if declared else "backend"

    def newer_input_waits(self) -> bool:
        """Return whether newer input waits: to land in this task, or for the conductor's decision on any Work.

        This is the commit gate: the backend re-plans with that input first, and a call already running is never cut.
        """
        return bool(self.steering or self.taken) or self.session.waits_for_decision

    def runner(self, name: str) -> Runner:
        """Return who runs a call of ``name``, as its action records it, also for a refused call."""
        if name in self.work_tools.names:
            return "im"
        declared = any(
            tool.get("type") == "function" and tool.get("name") == name for tool in self.request().get("tools", [])
        )
        return "app" if declared else "backend"

    def _refuse(self, call: dict, reason: str) -> None:
        """Refuse ``call``: its action is ``not_run``, and its output is the not-run output that says why.

        Its trace names ``reason``, never the output: a repeated call's output holds the app's earlier answer.
        """
        call_id = call.get("call_id", "")
        output = self.refusals[call_id]
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self.session.action(self, call, "not_run")
        self.record_step(call_id, "not_run", call.get("name", ""))
        self.trace("tools.refused", call_id=call_id, name=call.get("name", ""), reason=reason)

    def tool_started(self, call: dict, source: str) -> None:
        """Report a backend call as it starts: an ``mcp_call`` item, unless it is one of the IM's control tools."""
        call_id = self._bind_call(call)
        if call_id in self.started_calls or self.routes.get(call_id) == "refused":
            return
        if call_id not in self.routes:
            self.routes[call_id] = "backend"
            self.session.call_routed(call)
        self.started_calls[call_id] = self._reports(call.get("name", ""))
        self.record.acted = True
        self.session.action(self, call)
        self.record_step(call_id, "started", call.get("name", ""))
        if self.lifecycle.stop_reason is None:
            self.started_calls[call_id].started(self, call, source)

    def tool_observed(self, call: dict, source: str) -> None:
        """Retain an external adapter tool's actual activity without claiming it passed Core permission checks."""
        call_id, name = call.get("call_id"), call.get("name")
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
            raise ValueError("An external observation requires its actual call and tool identities")
        if call_id not in self._observed_tools and (
            call_id in self.routes or call_id in self.started_calls or call_id in self._call_bindings
        ):
            raise ValueError("An external observation cannot replace a Core-admitted operation")
        self.session.action(self, call, runner="backend")
        self._observed_tools.add(call_id)
        if call_id in self.started_calls:
            return
        self.routes[call_id] = "backend"
        self.started_calls[call_id] = ITEM_REPORTS
        self.record.acted = True
        self.record_step(call_id, "started", call.get("name", ""))
        self.trace("tools.observed", call_id=call_id, name=call.get("name"), source=source, core_admitted=False)
        if self.lifecycle.stop_reason is None:
            ITEM_REPORTS.started(self, call, source)

    def tool_outcome(self, call_id: str, output: str, outcome: Outcome) -> None:
        """Report that a backend call ended; its output joins at the next safe point, once."""
        if call_id in self.reported:
            if outcome != "unclear":
                self.session.call_ended(call_id, output, outcome)
            return
        self.reported.add(call_id)
        if outcome == "unclear":
            self.trace("tools.unclear", call_id=call_id, detail=output[:300])
            output = self.session.texts.owed_output
        self.session.call_ended(call_id, output, outcome)
        self.outputs[call_id] = {"type": "function_call_output", "call_id": call_id, "output": output}
        self.record_step(call_id, outcome)
        self.wake.set()
        reports = self.started_calls.get(call_id) or self._reports(self.steps[call_id][0])
        if self.lifecycle.stop_reason is None:
            reports.ended(self, call_id, output, outcome)

    def _reports(self, name: str) -> CallReports:
        """Return how a call reaches the wire: as an item, except the IM's control tools, which never do."""
        if name in self.displays:
            return DISPLAY_REPORTS
        return CONTROL_REPORTS if name in self.work_tools.names else ITEM_REPORTS

    def progress(self, text: str) -> None:
        """Report activity that is not model text: a progress note, never history."""
        if text.strip():
            self.record.progress = text
            self.session.progress(self, text)

    def take_boundary_input(self) -> BoundaryInput:
        """Join the outputs to the history and hand over waiting steering: a safe point."""
        order = {call_id: index for index, call_id in enumerate(self.call_order)}
        represented = {item.get("call_id") for item in self.view.items if item.get("type") == "function_call"}
        outputs = sorted(
            (item for identity, item in self.outputs.items() if identity in represented),
            key=lambda item: order.get(item.get("call_id"), len(order)),
        )
        self.outputs = {identity: item for identity, item in self.outputs.items() if identity not in represented}
        # A native tool can finish before its containing model round commits. Keep its real receipt as host
        # evidence until the matching call is actually in this provider history; never insert an orphan output.
        self._execution_fact_ids.update(self.outputs)
        if outputs:
            self.view.join(outputs)
            self.thread_items.extend(deepcopy(outputs))
        represented_outputs = {
            item.get("call_id") for item in self.view.items if item.get("type") == "function_call_output"
        }
        represented_calls = {item.get("call_id") for item in self.view.items if item.get("type") == "function_call"}
        if self.control.items and represented_calls.issubset(represented_outputs) and self.control.run is None:
            self.view.join(self.control.items)
            self.thread_items.extend(deepcopy(self.control.items))
            self.control.items.clear()
        steering = None
        if (
            self.steering
            and self.lifecycle.stop_reason is None
            and not self.lifecycle.suspension_active
            and self.control.run is None
        ):
            steering = self.take_steering()
        return BoundaryInput(tuple(outputs), steering)

    def settle(self, steering: Steering, delivery: Delivery) -> None:
        """Settle a batch the adapter took: ``applied`` lands it, ``not_applied`` keeps it for later.

        A batch that lands records input evidence; only explicit worker resolution answers its question.
        """
        waiting = self.taken.pop(steering.id, None)
        if waiting is None:
            return
        waiting.record.move(delivery)
        if delivery == "applied":
            self.view.join(list(steering.items))
            if waiting.batch.caller_input:
                self.presentation.landed_input(waiting.batch.items)
                self.request_ids.extend(d for d in waiting.batch.delegations if d not in self.request_ids)
            self.scope_applied |= bool(waiting.batch.scopes)
            self.thread_items.extend(deepcopy(waiting.batch.worker_items()))
            self._applied_ownership(waiting.batch)
        elif delivery == "unknown":
            self._remember_uncertain_steering(steering, waiting.batch)
            self.trace("im.input.uncertain", steering_id=steering.id)
        else:
            self._requeue_steering(waiting)

    def unapplied(self) -> list[Waiting]:
        """Return the batches still waiting in this task, or taken and not settled, oldest first."""
        return sorted((*self.taken.values(), *self.steering), key=lambda waiting: waiting.record.enqueue_order)

    def trace(self, kind: str, **fields: Any) -> None:
        """Note a trace of this task, with its Work, the task and the delegation that carries it."""
        where = {"work_id": self.work_id, "task_id": self.task_id, "delegation_id": self.work.carrier}
        self.session.observe(kind, **where, **fields)

    def record_step(self, call_id: str, outcome: str, name: str | None = None) -> None:
        """Record the result of an operation in the task's step history."""
        step = self.steps.setdefault(call_id, [name or "", outcome])
        step[1] = outcome
        if name:
            step[0] = name

    def step_line(self, call_id: str) -> str:
        """Return one call's line: its name and its outcome so far."""
        name, outcome = self.steps[call_id]
        return f"{name}: {outcome}"

    def step_lines(self) -> tuple[str, ...]:
        """Return one line per call it made, with the call's outcome so far."""
        return tuple(self.step_line(call_id) for call_id in self.steps)

    def record_steps(self) -> tuple[str, ...]:
        """Return the step lines the records show: only the newest ``RECORD_STEPS`` once its Work outlives the call."""
        lines = self.step_lines()
        return lines[-RECORD_STEPS:] if self.after_call else lines

    def record_answer(self) -> str | None:
        """Return the start of its answer as the records show it: only once its Work outlives the call."""
        answer = self.record.retained_answer
        return answer[:RECORD_TEXT_CHARS] if self.after_call and answer else None

    def visible_tools(self) -> list[dict]:
        """Retain complete caller-visible call/output pairs, preserving identities and order."""
        if self.view is None:
            return []
        items = self.view.own()
        names = self.displays | {"caller_message"}
        actions = self.session.ledger.actions
        calls = {
            call_id
            for call_id in self.presentation.written_calls
            if (action := actions.get(call_id)) is not None
            and action.task == self.task_id
            and action.name in names
            and action.state == "done"
        }
        calls &= {item.get("call_id") for item in items if item.get("type") == "function_call"}
        outputs = {item.get("call_id") for item in items if item.get("type") == "function_call_output"}
        paired = calls & outputs
        return [deepcopy(item) for item in items if item.get("call_id") in paired]

    def summary(self) -> TaskSummary:
        """Return the task as a conductor and the work records read it; an ended task says how it ended."""
        record = self.record
        answer = record.retained_answer
        end = record.ended_at if record.ended_at is not None else time.monotonic()
        seconds = end - record.started_at if record.started_at is not None else 0.0
        results, omitted = self.session.task_results(self.task_id)
        return TaskSummary(
            record.id,
            record.state,
            record.brief,
            self.record_steps(),
            record.acted,
            record.progress,
            seconds,
            record.question,
            self.record_answer(),
            answer[-RECORD_TEXT_CHARS:] if record.state == "answered" and not self.after_call and answer else None,
            questions=tuple(
                {
                    "question_id": q.id,
                    "text": q.text,
                    "dispatch": q.dispatch,
                    "resolution": q.resolution,
                    "owner_task_id": q.owner_task_id,
                }
                for q in self.presentation.owned_questions()
                if q.open
            ),
            stop="stopped" if self.record.stop.quiescent_later else self.record.stop.state,
            answer_publication=self.record.answer_publication,
            tool_results=results,
            tool_results_omitted=omitted,
            continues=record.continues,
            successor_task_id=record.successor_task_id,
            successor_work_id=record.successor_work_id,
            steering=self.session.task_steering_state(record),
            verifies=record.verifies,
            stop_initial=self.record.stop.state if self.record.stop.quiescent_later else None,
        )

    def assignment_message(self) -> WorkerData:
        """Render the task's assigned responsibility."""
        return routing_context({"task_input": self.texts.task_line.format(task=self.brief)}, starts_request=False)

    @property
    def stop_reason(self) -> StopReason | None:
        """Return the active stop request, if one exists."""
        return self.lifecycle.stop_reason

    @property
    def stopped(self) -> StopOutcome | None:
        """Return the native stop result, when known."""
        return self.lifecycle.stopped

    @property
    def stop_quiescent(self) -> bool:
        """Return whether a later receipt confirmed an initially unconfirmed stop."""
        return self.lifecycle.stop_quiescent

    @property
    def want_paused(self) -> bool:
        """Return whether caller work is still paused by request."""
        return self.lifecycle.want_paused

    @property
    def pause_outcome(self) -> StopOutcome | None:
        """Return the native suspension result, when known."""
        return self.lifecycle.pause_outcome

    @property
    def own_cancel(self) -> bool:
        """Return whether the worker requested its own cancellation."""
        return self.lifecycle.own_cancel

    @property
    def control_run(self) -> BackendRun | None:
        """Return the current isolated control invocation."""
        return self.control.run

    @property
    def control_items(self) -> list[dict]:
        """Return the settled control transcript for thread retention."""
        return self.control.items

    @property
    def inference_unconfirmed(self) -> bool:
        """Return whether control inference still holds its permit."""
        return self.control.unconfirmed

    @property
    def input_revision(self) -> int:
        """Return the task's applied caller input revision."""
        return self.presentation.input_revision

    @property
    def input_receipts(self) -> dict[str, InputReceipt]:
        """Return the task's receipts for applied caller input."""
        return self.presentation.input_receipts

    @property
    def published_calls(self) -> set[str]:
        """Return the tool identities that published a message or result."""
        return self.presentation.written_calls

    def request_pause(self) -> bool:
        """Fence business execution and start pause confirmation."""
        return self.lifecycle.request_pause()

    async def wait_paused(self) -> StopOutcome:
        """Wait for the current pause confirmation."""
        return await self.lifecycle.wait_paused()

    def request_resume(self) -> bool:
        """Allow the suspended task to resume on confirmed settlement."""
        return self.lifecycle.request_resume()

    def request_stop(self, reason: StopReason) -> bool:
        """Fence task execution and record why stopping was requested."""
        return self.lifecycle.request_stop(reason)

    async def confirm_stop(self) -> StopOutcome:
        """Wait for the current stop confirmation."""
        return await self.lifecycle.confirm_stop()

    def clear_quiescence_observers(self) -> None:
        """Detach native settlement callbacks."""
        self.lifecycle.clear_quiescence_observers()


class WorkRun:
    """One Work: its tasks, its carrier's response, its settings, its steering, its ending."""

    def __init__(self, session: IMSession, record: Work):
        """Make the run of ``record``, with one task per ``Task`` record."""
        self.session, self.record, self.id = session, record, record.id
        self.requested = [item for delegation in record.requests for item in session.request_items(delegation)]
        self.settings = settings_for(record.settings, session.defaults)
        self.tasks = [TaskRun(self, task) for task in record.tasks]
        self.input: list[dict] = []
        self.batches: list[Batch] = []
        self.deferred: list[str] = []
        self.held: list[HeldEmission] = []
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.stop_reason: StopReason | None = None
        self.ended = False
        self.done = asyncio.Event()
        self.start_barrier: asyncio.Future[None] | None = None
        self._backstop: asyncio.TimerHandle | None = None
        self.branch_ref = session.ownership.register(self)

    @property
    def carrier(self) -> str:
        """The delegation whose response carries the Work's output now."""
        return self.record.carrier

    @property
    def requests(self) -> list[str]:
        """The requests it carries, in the order they joined it, handed-over ones included."""
        return self.record.requests

    def take_settings(self, settings: dict) -> None:
        """Take an explicit request's own settings: the Work's response is written with them from then on."""
        self.record.settings = deepcopy(settings)
        self.settings = settings_for(self.record.settings, self.session.defaults)

    def open_tasks(self) -> list[TaskRun]:
        """Return the tasks that have not ended."""
        return [task for task in self.tasks if not task.ended]

    def execution_changed(self) -> None:
        """Aggregate actual task execution without completing or cancelling a paused sibling."""
        if self.ended or self.stop_reason is not None:
            return
        states = {task.record.state for task in self.open_tasks()}
        if not states:
            return
        state = (
            "waiting_client"
            if "waiting_client" in states
            else "running"
            if "running" in states
            else "pausing"
            if "pausing" in states
            else "queued"
            if "queued" in states
            else "paused"
        )
        if self.record.state != state:
            self.record.move(state)
            self.session.ownership.publication_changed(self)

    def parked(self) -> set[Parked]:
        """Return its responses that ended with the app's calls and wait for their outputs."""
        return {parked for parked in self.session.parked.values() if parked.work is self}

    def begin(self) -> None:
        """Start every task's driver; each waits for a slot."""
        for task in self.tasks:
            task.begin()

    def task_started(self, task: TaskRun) -> BackendRun:
        """Start ``task``'s run on its first slot, and the Work with it if the Work has not started."""
        if self.started_at is None:
            self.started_at = time.monotonic()
            carrier = self.session.ledger.delegations[self.carrier]
            if carrier.state == "queued" or (
                carrier.state == "waiting_client" and self.session.book.writer(self.carrier) is not None
            ):
                carrier.move("running")
            self.input = self.session.work_input(self)
            self.start_response()
            limit = self.session.settings.work_timeout_seconds
            if limit is not None:
                self._backstop = asyncio.get_running_loop().call_later(limit, self.stop_soon, AtBackstop())
            self.session.observe(
                "backend.dequeue", delegation_id=self.carrier, work_id=self.id, continues=self.record.continues
            )
        task.record.started_at = time.monotonic()
        if task.start_settings is None:
            task.start_settings = self.request()
        initial = task.initial_requested if task.history_mode == "focused" else self.requested
        task.presentation.landed_input([*initial, *task.pending])
        self.session.ownership.initial_applied(task)
        task.view = self.session.history_for(self, task)
        task.refresh_history()
        return task.run if task.run is not None else self.session.backend.start(task)

    def request(self) -> dict:
        """Return the settings of the next model call of any of its tasks, the instructions composed."""
        return task_request(self.settings, *self.session.instruction_notes())

    def steering_items(self, batch: Batch, task: TaskRun | None = None) -> list[dict]:
        """Return a batch's requests as a task takes them: the records block and the closing line in the newest one.

        The records list the batch's requests under this Work, which they join.
        """
        if not batch.delegations:
            return batch.worker_items()
        if task is not None and (task.history_mode == "focused" or batch.scopes):
            records = json.dumps(
                self.record_line(batch.delegations, task, scoped=bool(batch.scopes)), ensure_ascii=False
            )
            return with_parts(
                batch.worker_items(),
                self.session.texts.records_block.format(records=records),
                self.session.texts.closing_line,
            )
        return with_parts(batch.items, *self.session.records_parts(joining={self.id: batch.delegations}))

    def write(
        self,
        action: Callable[[ResponseWriter], None],
        *,
        task: TaskRun | None = None,
        kind: str = "output",
        calls: Sequence[dict] = (),
        publication_call_id: str | None = None,
    ) -> None:
        """Write into the carrier's response now, or hold it, in order, until a response of it is in progress.

        While nobody hears the Work (:meth:`unheard`), what it writes is dropped.
        """
        emission = HeldEmission(action, task, kind, tuple(calls), publication_call_id)
        writer = self.session.book.running(self.carrier)
        if writer is None or self.held:
            if not self.unheard():
                self.held.append(emission)
                self.session.ownership.publication_changed(self)
            else:
                emission.discard("no_listener")
            return
        if not self._emit(emission, writer):
            self.held.append(emission)
            self.session.ownership.publication_changed(self)

    def _emit(self, emission: HeldEmission, writer: ResponseWriter) -> bool:
        if emission.discarded:
            return True
        if emission.task is not None and self.session.waits_for_decision:
            self.session.ownership.wait_to_flush(self.carrier)
            return False
        if emission.kind == "app" and emission.task is not None:
            if any(emission.task.app_placement_pending(call) for call in emission.calls):
                self.session.ownership.wait_to_flush(self.carrier)
                return False
            calls = emission.task.gate_held_app(emission.calls)
            if not calls:
                return True
            with writer.attributed(self.session.ownership.attribution(self)):
                self._send_calls(emission.task, calls, writer)
            return True
        task_ref = emission.task.task_id if emission.task is not None else None
        with writer.attributed(self.session.ownership.attribution(self, task_ref=task_ref)):
            emission.action(writer)
        return True

    def unheard(self) -> bool:
        """Return whether nobody hears the Work now, as after the call's end until a newer request takes it over.

        It outlives the call with no stop asked, the call has ended or its carrier has no open response, and it owes the
        app no output.
        """
        return (
            self.record.after_call
            and self.stop_reason is None
            and (self.session.detached or self.session.book.writer(self.carrier) is None)
            and not self.parked()
            and not (
                self.session.ownership.managed(self)
                and any(parked.delegation_id == self.carrier for parked in self.session.parked.values())
            )
        )

    def flush(self) -> None:
        """Write what was held, in order, while the carrier's response is in progress."""
        self.session.ownership.flush(self.carrier)

    def prune_discarded_publications(self) -> None:
        """Remove settled caller messages without changing the order of other held output."""
        remaining = [emission for emission in self.held if not emission.discarded]
        if len(remaining) != len(self.held):
            self.held = remaining
            self.session.ownership.publication_changed(self)

    def drop_held(self, reason: str) -> None:
        """Preserve never-written receipts when an entire held queue loses its owner."""
        if self.held:
            for emission in self.held:
                emission.discard(reason)
            self.held.clear()
            self.session.ownership.publication_changed(self)

    def _flush(self) -> None:
        while self.held:
            writer = self.session.book.running(self.carrier)
            if writer is None:
                return
            if not self._emit(self.held[0], writer):
                return
            self.held.pop(0)
            self.session.ownership.publication_changed(self)

    def answer(self, task: TaskRun, message: dict) -> None:
        """Write a task's answer as its own ``final_answer`` item, at once or held; it is never merged with another."""
        result = task.record.result
        if result.state == "none":
            result.move("final")
        if task.stop_reason is None:
            text = message_text(message)

            revision = task.input_revision

            def written(writer: ResponseWriter) -> None:
                if (
                    not task.permits_output(allow_ended=True, allow_detached=True, revision=revision)
                    or task.record.answer_publication == "superseded"
                    or task.steering
                    or task.taken
                ):
                    self.session.supersede_answer(task.record)
                    task.trace("im.answer.superseded")
                    return
                self.session.notes.drop(self.carrier, branch_ref=self.branch_ref)
                item = writer.message(text, "final_answer")
                task.record.answer_publication = "written"
                task.trace("im.answer.written", item_id=item["id"], response_id=writer.id)
                if result.state == "final":
                    result.move("pushed")

            self.write(written, task=task)

    def stop_answer(self, line: str) -> None:
        """Write a stopped task's answer as a ``final_answer`` on the carrier's open response, after what was held."""
        if line and self.session.book.writer(self.carrier) is not None:
            self.start_response()
            writer = self.session.book.running(self.carrier)
            with writer.attributed(self.session.ownership.attribution(self)):
                writer.message(line, "final_answer")

    def control_diagnostic(
        self, task: TaskRun, sources: Sequence[str], line: str, *, frontier: frozenset[str], call_generation: int
    ) -> bool:
        """Publish the outcome only on the exact still-current control sources."""
        current = {waiting.record.id for waiting in task.unapplied()}
        if (
            current != frontier
            or self.session.waits_for_decision
            or not self.session.caller_attached
            or self.session.closed
            or call_generation != self.session.call_generation
        ):
            return False
        written = False
        for source in sources:
            record = self.session.ledger.delegations.get(source)
            if record is None or record.ended or record.call_generation != call_generation:
                continue
            self.session.book.start(source)
            writer = self.session.book.running(source)
            if writer is not None:
                writer.message(line, "final_answer", source_item=True)
                written = True
        return written

    def continue_after_call(self, task: TaskRun, message: str) -> str:
        """Let the Work outlive the call, for a task's ``continue_after_call``; return what the task is told.

        In one step it writes ``message`` as a ``final_answer``, while the response stays open, marks the Work and
        lifts its backstop. It is refused while the message would be held, for a task that is stopping, and when the
        session has no room for more work after the call (``IMSession.room_after_call``). A Work marked already writes
        nothing again.
        """
        texts = self.session.texts
        if self.record.after_call:
            return texts.continued_output
        held = self.held or self.session.book.running(self.carrier) is None
        stopping = task.stop_reason is not None or task.own_cancel
        if held or stopping or not self.session.room_after_call(self):
            return texts.not_continued_output
        self.write(lambda writer: writer.message(message, "final_answer"), task=task)
        self.record.after_call = True
        if self._backstop is not None:
            self._backstop.cancel()
        task.trace("im.work.after_call")
        return texts.continued_output

    def start_response(self) -> None:
        """Put the carrier's open response in progress (once); what was held while it was closed follows."""
        self.session.book.start(self.carrier)
        self.flush()

    def park(self, task: TaskRun, calls: list[dict]) -> None:
        """Record the app wait even when publication must wait for the next response frame."""
        self.execution_changed()
        self.write(lambda writer: self._send_calls(task, calls, writer), task=task, kind="app", calls=calls)

    def _send_calls(self, task: TaskRun, calls: list[dict], writer: ResponseWriter) -> None:
        # Held while another call waited, and stopped since: nothing of it reaches the app.
        if task.stop_reason is not None:
            return
        for call in calls:
            writer.function_call(call)
            self.session.ledger.actions[call["call_id"]].app_published = True
        carrier = self.carrier
        self.session.notes.drop(carrier, branch_ref=self.branch_ref if self.session.ownership.managed(self) else None)
        self.session.ledger.delegations[carrier].move("waiting_client")
        pending = {call["call_id"]: (task, call) for call in calls}
        parked = Parked(self, carrier, writer.id, pending, task.input_revision)
        self.session.park(parked)
        self.session.book.end_after_calls(
            carrier, "response.completed", ended=lambda: self.session.app_calls_ended(parked)
        )

    def continue_with(
        self, parked: Parked, outputs: dict[str, dict], messages: list[dict], brings: dict | None
    ) -> None:
        """Go on with ``parked``'s tasks in the carrier's next response: their outputs, then the typed messages."""
        if brings is not None:
            self.take_settings(brings)
        if not self.ended:
            self.session.ledger.delegations[parked.delegation_id].move("running")
        by_task: dict[TaskRun, list[dict]] = {}
        for call_id, (task, _call) in parked.calls.items():
            by_task.setdefault(task, []).append(outputs[call_id])
        for task, given in by_task.items():
            task.give(given)
        self.execution_changed()
        if messages:
            self.take_typed(list(by_task), messages)
        if self.deferred or self.held or any(task.running for task in self.tasks):
            self.resumed()
        if not self.open_tasks():
            self._finish()

    def resumed(self) -> None:
        """Reopen the carrier after app output and settle deferred handovers before flushing held items."""
        self.session.book.start(self.carrier)
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._take_over(taker)
        self.flush()

    def steer(
        self,
        delegations: list[str],
        task: TaskRun | None,
        brings: dict | None,
        extra: Sequence[dict] = (),
        placement_refs: tuple[str, ...] = (),
        *,
        selected: Sequence[TaskRun] | None = None,
        control: bool = True,
        caller_input: bool = True,
        scopes: tuple[InputScope, ...] = (),
    ) -> None:
        """Join these requests to the Work's open tasks, or to one; a running task takes them at its next safe point."""
        items = [
            *deepcopy(list(extra)),
            *[item for delegation in delegations for item in self.session.request_items(delegation)],
        ]
        batch = Batch(
            list(delegations),
            items,
            brings,
            placement_refs=placement_refs,
            call_generation=self.session.call_generation,
            control=control,
            caller_input=caller_input,
            scopes=scopes,
        )
        self._steer(
            batch, list(selected) if selected is not None else [task] if task is not None else self.open_tasks()
        )

    def take_typed(self, owners: list[TaskRun], messages: list[dict]) -> None:
        """Give companion text to the tasks that own the returned calls; retain any untaken text in history."""
        items = with_ids(messages, uid("item"))
        self.session.remember_input_order(items)
        batch = Batch([], items, None, call_generation=self.session.call_generation)
        self._steer(batch, [task for task in owners if not task.ended and task.stop_reason is None])

    def _steer(self, batch: Batch, targets: list[TaskRun]) -> None:
        """Join ``batch`` to ``targets``, or to the Work's input before the Work starts."""
        if batch.scopes:
            for target in targets:
                if target.view is None:
                    target.history_mode = target.record.history_mode = "focused"
        if self.started_at is None and not any(t.control_run for t in targets):
            if not batch.scopes:
                self.requested.extend(batch.items)
            for target in targets:
                if target.history_mode == "focused":
                    target.pending.extend(deepcopy(batch.items))
                    target.pending_scopes.extend(batch.scopes)
                else:
                    target.initial_requested.extend(deepcopy(batch.items))
                target.request_ids.extend(d for d in batch.delegations if d not in target.request_ids)
            self.land(batch)
            return
        self.batches.append(batch)
        for target in targets:
            if target.view is None and target.control_run is None:
                target.pending.extend(batch.items)
                target.pending_scopes.extend(batch.scopes)
                target.request_ids.extend(d for d in batch.delegations if d not in target.request_ids)
                self.land(batch)
            else:
                record = self.session.ledger.new_steering(target.task_id)
                target.steering.append(Waiting(batch, record))
                if target.control_run is not None:
                    target.control_run.steer()
                elif target.running:
                    target.run.steer()
                target.wake.set()

    def land(self, batch: Batch) -> None:
        """Land a batch: each of its requests takes the Work over in turn, and the Work keeps or takes settings."""
        if batch.landed:
            return
        batch.landed = True
        if batch in self.batches:
            self.batches.remove(batch)
        if batch.scopes:
            existing = {item.get("id") for item in self.requested}
            self.requested.extend(item for item in batch.items if not item.get("id") or item["id"] not in existing)
        for taker in batch.delegations:
            if taker not in self.requests:
                self.requests.append(taker)
                self.session.ownership.input_joined(self)
            self._take_over(taker)
        if batch.brings is not None:
            self.take_settings(batch.brings)

    def take_over_at_start(self, takers: list[str]) -> None:
        """Let the newer requests of a queued Work take it over in turn, each bringing its input."""
        for taker in takers:
            self.requests.append(taker)
            self.session.ownership.input_joined(self)
            self.requested.extend(self.session.request_items(taker))
            for task in self.tasks:
                task.initial_requested.extend(self.session.request_items(taker))
                task.request_ids.append(taker)
            self._take_over(taker)

    def _take_over(self, taker: str) -> None:
        """Make ``taker`` carry the Work: hand over at once, or later if the carrier's response waits for the app.

        A Work that nobody hears (:meth:`unheard`) waits for no app, so ``taker`` takes it over at once. A taker whose
        call has ended never takes the Work from a carrier that a call hears: it joins the Work and ends
        ``handed_over``.
        """
        if self.session.ownership.managed(self, taker) and self.session.ownership.older_input(self, taker):
            return
        composite_frame = self.session.ownership.listening(taker)
        if (
            self.session.book.writer(taker) is None
            and self.session.book.writer(self.carrier) is not None
            and not composite_frame
        ):
            self.session.ledger.delegations[taker].move("handed_over")
            return
        other_parked = self.session.ownership.managed(self, taker) and not self.parked()
        if self.session.book.writer(self.carrier) is None and not self.unheard() and not other_parked:
            self.deferred.append(taker)
            return
        self._hand_over(taker)

    def _hand_over(self, taker: str) -> None:
        """Hand the Work to ``taker``: its response starts, and the carrier's open one ends with the handover line."""
        if self.session.ownership.hand_over(self, taker):
            return
        book, ledger, older = self.session.book, self.session.ledger, self.carrier
        book.start(taker, previous=older)
        if book.writer(older) is not None:
            book.start(older)
            line = self.session.texts.handover_line.format(delegation=taker)
            self.session.notes.drop(older)
            book.end_after_calls(
                older, "response.completed", last=lambda writer: writer.message(line, "commentary", source_item=True)
            )
        ledger.delegations[older].move("handed_over")
        self.record.carrier = taker
        if self.started_at is not None:
            ledger.delegations[taker].move("running")
        self.session.observe("backend.handed_over", delegation_id=older, taker=taker, work_id=self.id)

    def stop_soon(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """Ask the stop door at once (nothing more is written, calls are refused); confirm it in the background."""
        asked = self.ask_stop(reason, task)
        if asked:
            self.session.spawn(self.confirm_stop(asked))

    async def stop(self, reason: StopReason, task: TaskRun | None = None) -> None:
        """Stop the whole Work, or one task of it, through the stop door."""
        await self.confirm_stop(self.ask_stop(reason, task))

    def ask_stop(self, reason: StopReason, task: TaskRun | None = None) -> list[TaskRun]:
        """Ask the tasks to stop for ``reason``; return those that were still running and are stopping now.

        Once no task goes on, its responses stop waiting for the app, a request that landed meanwhile takes the Work
        over, and the Work ends at once when no task is open; while another task goes on, they keep waiting.
        """
        targets = [task] if task is not None else self.open_tasks()
        whole = all(each in targets for each in self.open_tasks())
        asked = [each for each in targets if each.request_stop(reason)]
        if asked and whole:
            self.stop_reason = reason
        parked = self.parked()
        if parked and all(each.stop_reason is not None for each in self.open_tasks()):
            if self.stop_reason is None:
                self.stop_reason = reason
            for each in parked:
                self.session.unpark(each, stop_state=self.stop_reason.work_state)
            for taker in self.deferred[:]:
                self.deferred.remove(taker)
                self._hand_over(taker)
            self.flush()
            if not self.open_tasks():
                self._finish()
        return asked

    async def confirm_stop(self, asked: list[TaskRun]) -> None:
        """Wait for each asked task to stop (``stopped`` or ``unconfirmed``)."""
        await asyncio.gather(*(each.confirm_stop() for each in asked))

    def deadline(self, parked: Parked) -> None:
        """Stop the Work when the app never answered ``parked``'s calls in time, unless another request carries it.

        A request that landed meanwhile takes the Work over, and the stuck tasks stop, so nothing it asked runs twice.
        """
        if self.ended:
            return
        self.session.ownership.expire(parked)
        if not self.deferred:
            if all(work is self for work in self.session.ownership.members(self.carrier)) and self.held:
                self.drop_held("expired")
            self.stop_soon(AtDeadline())
            return
        self.session.unpark(parked)
        for taker in self.deferred[:]:
            self.deferred.remove(taker)
            self._hand_over(taker)
        self.flush()
        waiting = dict.fromkeys(task for task, _ in parked.calls.values())
        stuck = [task for task in waiting if task.request_stop(AtDeadline())]
        if stuck:
            self.session.spawn(self.confirm_stop(stuck))
        if not self.open_tasks():
            self._finish()

    def task_ended(self) -> None:
        """End the Work with its last task, unless held items wait for the response the app's outputs open next."""
        if not self.open_tasks() and not (self.held and self.parked()):
            self._finish()

    def _finish(self) -> None:
        if self.ended:
            return
        self.ended, self.ended_at = True, time.monotonic()
        if self._backstop is not None:
            self._backstop.cancel()
        session, book, carrier = self.session, self.session.book, self.carrier
        failure = next((task.failure for task in self.tasks if task.failure is not None), None)
        completed = any(task.completed and task.record.state == "answered" for task in self.tasks)
        state = (
            self.stop_reason.work_state
            if self.stop_reason is not None
            else "failed"
            if failure
            else "answered"
            if completed
            else "cancelled"
        )
        managed = session.ownership.work_finished(self, state)
        session.notes.drop(carrier, branch_ref=self.branch_ref if managed else None)
        if not managed:
            if self.stop_reason is not None:
                self.stop_reason.end(book, carrier)
            elif failure is not None:
                self.start_response()
                book.end(carrier, failure.kind, **failure.fields)
            else:
                self.start_response()
                book.end(carrier, "response.completed")
        self.record.move(state)
        record = session.ledger.delegations.get(carrier)
        if record is not None and not record.ended and not managed:
            record.move(state)
        discard_pending = self.stop_reason is not None and self.stop_reason.discard_pending_requests
        for task in self.tasks:
            for waiting in task.unapplied():
                if not waiting.batch.landed:
                    waiting.record.move("cancelled" if discard_pending else "follow_up")
        leftovers = list(self.deferred)
        if discard_pending:
            if managed:
                session.ownership.discard_pending(self, leftovers)
            else:
                for delegation in dict.fromkeys(leftovers):
                    record = session.ledger.delegations[delegation]
                    if not record.ended:
                        book.cancelled(delegation)
                        record.move("cancelled")
            leftovers = []
        session.work_ended(self, leftovers)
        self.done.set()

    def segment(self) -> list[dict]:
        """Return what the ended Work adds to the conversation: its requests, then each task's own items.

        A call left without an output gets the one its task holds, else ``texts.owed_output``; untaken typed messages
        come last. A Work that outlived the call adds its requests, then the answer of each task that answered.
        """
        untaken = [item for batch in self.batches if not batch.delegations for item in batch.items]
        if self.record.after_call:
            items = [item for delegation in self.requests for item in self.session.request_items(delegation)]
            items += [item for task in self.tasks for item in task.visible_tools()]
            for task in self.tasks:
                if task.record.state == "answered" and (answer := task.record.retained_answer):
                    items.append(output_message(task.record.answer_item_id or uid("msg"), answer, "final_answer"))
            return items + untaken
        items = list(self.requested)
        for task in self.tasks:
            if task.view is not None:
                task.view.join(task.view.owed(self.session.texts.owed_output, task.outputs))
                items.extend(
                    item
                    for item in task.record.without_superseded_answer(task.view.own())
                    if not isinstance(item, WorkerData)
                )
        return items + untaken

    def summary(self) -> WorkSummary:
        """Return the Work, open or ended, as a conductor reads it: its requests and every task."""
        pending = [d for batch in self.batches for d in batch.delegations]
        requests = tuple(
            self.session.request_view(d, pending=d in pending) for d in dict.fromkeys([*self.requests, *pending])
        )
        tasks = tuple(task.summary() for task in self.tasks)
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        seconds = end - self.record.queued_at
        return WorkSummary(self.id, self.record.state, self.carrier, requests, tasks, seconds, self.record.after_call)

    def record_line(self, joining: Sequence[str] = (), task: TaskRun | None = None, *, scoped: bool = False) -> dict:
        """Return the Work as the records block shows it to the backend: one JSON object.

        A Work that outlives the call says so, and shows the start of each task's answer. ``joining`` names requests
        that join it with the steer being read, listed after its own.
        """
        tasks = [task] if task is not None else self.tasks
        steps = [line for task in tasks for line in task.record_steps()]
        notes = [task.record.progress for task in tasks if task.record.progress]
        questions = [
            {"question_id": q.id, "text": q.text, "dispatch": q.dispatch, "owner_task_id": q.owner_task_id}
            for task in tasks
            for q in task.presentation.owned_questions()
            if q.open
        ]
        answers = [answer for task in tasks if (answer := task.record_answer())]
        line: dict[str, Any] = {"work": self.id, "state": self.record.state, "steps": steps}
        if task is None or not (scoped or task.initial_scopes or task.pending_scopes or task.scope_applied):
            request_ids = task.request_ids if task is not None else self.requests
            line["requests"] = [
                request_text(self.session.request_items(d)) for d in dict.fromkeys([*request_ids, *joining])
            ]
        stops = [
            {
                "task_id": task.task_id,
                "initial_result": task.record.stop.state,
                "quiescent_later": task.record.stop.quiescent_later,
            }
            for task in tasks
            if task.record.stop.state != "none"
        ]
        if stops:
            line["stops"] = stops
        if notes:
            line["progress"] = notes[-1]
        if questions:
            line["questions"] = questions
        if self.record.after_call:
            line["continues_after_call"] = True
        if answers:
            line["answers"] = answers
        return line
