# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task questions, input receipts and caller-facing publications."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import ToolNotExecuted
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.stream import ResponseWriter
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.tools import (
    CallerMessage,
    Invocation,
    QuestionResolutions,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.ledger.records import (
    InputReceipt,
    Question,
    Task,
)
from nemotron_voice_agent.pipecat.shared.ids import approx_tokens

if TYPE_CHECKING:
    from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.run import TaskRun


class TaskPresentation:
    """Keep questions and their publication and input evidence under one owner."""

    def __init__(self, task: TaskRun):
        """Bind question and publication evidence to the task record."""
        self.task = task
        self.receipts = task.record.publications
        self.input_receipts = task.record.input_receipts
        self.written_calls: set[str] = set()

    @property
    def input_revision(self) -> int:
        """Return the current applied input revision."""
        return self.task.record.input_revision

    def inherit(self, predecessor: Task) -> None:
        """Carry the predecessor's applied input and publication receipts into a continuation."""
        self.task.record.input_revision = predecessor.input_revision
        self.input_receipts.update(predecessor.input_receipts)
        self.receipts.update(deepcopy(predecessor.publications))

    def is_pending(self, call_id: str) -> bool:
        """Check whether the named caller message still awaits publication."""
        return self.receipts.get(call_id, {}).get("state") == "pending"

    def caller_message(self, invocation: Invocation) -> str:
        """Publish a typed status or question through the same current-input gate."""
        task = self.task
        given = CallerMessage.model_validate(invocation.arguments)
        if task.ended or task.lifecycle.stop_reason is not None or task.lifecycle.own_cancel:
            raise ToolNotExecuted("caller_message: task has ended or is stopping")
        return self.publish(given, invocation.call_id)

    def owned_questions(self) -> list[Question]:
        """Return canonical questions this task currently owns, including after its ordinary answer."""
        task = self.task
        return [q for q in task.record.questions if q.owner_task_id == task.task_id]

    def refresh_question_summary(self) -> None:
        """Refresh the display summary without replacing canonical question records."""
        task = self.task
        texts = [q.text for q in self.owned_questions() if q.open]
        task.record.question = "\n".join(texts) or None

    def prune(self) -> None:
        """Retain every pending publication and the recent settled receipts."""
        closed = [key for key, row in self.receipts.items() if row["state"] != "pending"]
        for key in closed[:-32]:
            del self.receipts[key]

    def drop(self, call_id: str, reason: str) -> None:
        """Settle a never-written caller message and its exact question publication together."""
        task = self.task
        row = self.receipts.get(call_id)
        if row is None or row["state"] != "pending":
            return
        row.update(state="dropped", reason=reason)
        if row["kind"] == "question":
            question = next(
                (q for q in self.owned_questions() if q.id == row["question_id"] and q.publication_call_id == call_id),
                None,
            )
            if question is not None and question.dispatch == "pending":
                question.dispatch = "dropped"
            task.trace(
                "im.question.publication_dropped", question_id=row["question_id"], call_id=call_id, reason=reason
            )
        else:
            task.trace("im.status.publication_dropped", call_id=call_id, reason=reason)

    def drop_all(self, reason: str) -> None:
        """Drop outstanding publications and remove their held writes."""
        task = self.task
        for call_id in self.receipts:
            self.drop(call_id, reason)
        task.work.prune_discarded_publications()

    def close_questions(self, disposition: str) -> None:
        """Cancel owned questions or retire questions that failed before they were written."""
        task = self.task
        for question in task.record.close_questions(disposition):
            task.trace(
                "im.question.closed",
                question_id=question.id,
                disposition=disposition,
                dispatch=question.dispatch,
                input_ref=question.input_ref,
            )

    def landed_input(self, items: Sequence[dict]) -> None:
        """Only actual caller-message application creates answer evidence."""
        task = self.task
        if not any(item.get("role") == "user" for item in items):
            return
        task.record.input_revision += 1
        identity = f"{task.task_id}.I{self.input_revision}"
        self.input_receipts[identity] = InputReceipt(
            identity,
            self.input_revision,
            tuple(item["id"] for item in items if item.get("role") == "user" and item.get("id")),
            tuple(
                (item["id"], task.session.caller_order(item["id"]))
                for item in items
                if item.get("role") == "user" and item.get("id")
            ),
        )
        if task.record.unresolved_input_order and any(
            order is not None and order >= task.record.unresolved_input_order
            for _, order in self.input_receipts[identity].caller_order
        ):
            task.record.unresolved_input_order = 0
        task.trace(
            "im.input.landed",
            input_ref=identity,
            input_revision=self.input_revision,
            source_message_ids=list(self.input_receipts[identity].source_message_ids),
        )
        # Preserve evidence referenced by retained questions; expired unreferenced receipts fail explicitly.
        required = {q.input_ref for q in self.owned_questions() if q.input_ref}
        for key in list(self.input_receipts)[:-64]:
            if key not in required:
                del self.input_receipts[key]
        self.drop_all("newer_input")

    def resolve_questions(self, invocation: Invocation) -> str:
        """Validate every selected disposition before changing any canonical question."""
        task = self.task
        if task.ended or task.lifecycle.stop_reason is not None or task.lifecycle.own_cancel:
            raise ToolNotExecuted("resolve_questions: task has ended or is stopping")
        given = QuestionResolutions.model_validate(invocation.arguments)
        prepared = []
        for resolution in given.resolutions:
            question = next((q for q in self.owned_questions() if q.id == resolution.question_id), None)
            if question is None or not question.open:
                raise ToolNotExecuted(
                    f"resolve_questions: unknown, closed or transferred question {resolution.question_id}"
                )
            receipt = None
            if resolution.disposition == "answered":
                reference = resolution.input_ref
                exact = self.input_receipts.get(reference)
                candidates = (exact,) if exact is not None else reversed(self.input_receipts.values())
                receipt = next(
                    (
                        row
                        for row in candidates
                        if row.revision > question.input_revision
                        and any(
                            order is not None
                            and order > question.input_caller_order
                            and (exact is not None or message_id == reference)
                            for message_id, order in row.caller_order
                        )
                    ),
                    None,
                )
                if receipt is None:
                    raise ToolNotExecuted(
                        f"resolve_questions: {question.id} requires a newer applied caller input reference"
                    )
            prepared.append((question, resolution, receipt))
        for question, resolution, receipt in prepared:
            question.resolution = resolution.disposition
            question.input_ref = receipt.id if receipt is not None else None
            self.drop(question.publication_call_id, "question_resolved")
            task.trace(
                "im.question.resolved",
                question_id=question.id,
                call_id=invocation.call_id,
                disposition=resolution.disposition,
                input_ref=question.input_ref,
            )
        self.refresh_question_summary()
        task.work.prune_discarded_publications()
        return json.dumps(
            {
                "resolutions": [
                    {"question_id": q.id, "disposition": q.resolution, "input_ref": q.input_ref} for q, _, _ in prepared
                ]
            }
        )

    def publish(self, given: CallerMessage, call_id: str, *, source_id: str | None = None) -> str:
        """Share publication eligibility while preserving distinct question/status responsibility."""
        task = self.task
        revision = self.input_revision
        record = None
        previous_publication = None
        if given.kind == "question":
            if given.question_id is not None:
                record = next((q for q in self.owned_questions() if q.id == given.question_id), None)
                if record is None or not record.open or record.dispatch != "dropped" or record.text != given.message:
                    raise ToolNotExecuted("caller_message: reissue requires the unchanged owned unpublished question")
                previous_publication = record.dispatch, record.publication_revision, record.publication_call_id
                record.dispatch = "pending"
                record.publication_revision = revision
                record.publication_call_id = call_id
            else:
                identity = f"{task.task_id}.Q{task.record.question_sequence + 1}"
                record = Question(
                    identity,
                    given.message,
                    call_id,
                    revision,
                    creator_task_id=task.task_id,
                    owner_task_id=task.task_id,
                    publication_revision=revision,
                    publication_call_id=call_id,
                    created_input_revision=revision,
                    input_caller_order=max(
                        (
                            order
                            for input_receipt in self.input_receipts.values()
                            for _, order in input_receipt.caller_order
                            if order is not None
                        ),
                        default=0,
                    ),
                )
                # Active identities must fit; closed history may be compacted, never silently evict active ones.
                active = {q.id: q for owner in task.session.ledger.tasks.values() for q in owner.questions if q.open}
                active[record.id] = record
                evidence = [
                    {"id": q.id, "text": q.text, "owner": q.owner_task_id, "previous_owners": q.previous_owners}
                    for q in active.values()
                ]
                if approx_tokens(json.dumps(evidence)) > task.session.settings.context_tokens:
                    raise ToolNotExecuted("caller_message: active question state exceeds context budget")
                task.record.questions.append(record)

        publication = {
            "call_id": call_id,
            "task_id": task.task_id,
            "kind": given.kind,
            "question_id": record.id if record else None,
            "state": "pending",
            "input_revision": revision,
        }
        self.receipts[call_id] = publication
        if record is not None:
            try:
                if task.view is not None:
                    task.backend_state()
            except RuntimeError:
                del self.receipts[call_id]
                if previous_publication is None:
                    task.record.questions.remove(record)
                else:
                    record.dispatch, record.publication_revision, record.publication_call_id = previous_publication
                raise ToolNotExecuted("caller_message: required question evidence exceeds context budget") from None
            if previous_publication is None:
                task.record.question_sequence += 1
                self.refresh_question_summary()
        self.prune()

        def written(writer: ResponseWriter) -> None:
            eligible = (
                publication["state"] == "pending"
                and task.permits_output(allow_suspended=True, revision=revision)
                and not task.newer_input_waits()
                and not task.work.unheard()
            )
            if record is not None:
                eligible = (
                    eligible
                    and record.open
                    and record.owner_task_id == task.task_id
                    and record.dispatch == "pending"
                    and record.publication_call_id == call_id
                )
            if not eligible:
                self.drop(call_id, "publication_ineligible")
                return
            item = writer.message(given.message, "final_answer", source_item=source_id is not None)
            self.written_calls.add(call_id)
            publication.update(state="written", response_id=writer.id, item_id=item["id"])
            if record is not None:
                record.dispatch, record.response_id, record.item_id = "written", writer.id, item["id"]
                task.trace(
                    "im.question.written",
                    question_id=record.id,
                    call_id=call_id,
                    input_revision=revision,
                    item_id=item["id"],
                    response_id=writer.id,
                )
            else:
                task.record.progress = given.message
                task.trace("im.status.written", call_id=call_id, item_id=item["id"], response_id=writer.id)

        if source_id is None:
            task.work.write(written, task=task, kind="caller_message", publication_call_id=call_id)
        else:
            task.session.book.start(source_id)
            writer = task.session.book.running(source_id)
            if writer is not None:
                written(writer)
            else:
                self.drop(call_id, "no_listener")
        return json.dumps(
            {"kind": given.kind, "question_id": record.id if record else None, "publication": publication["state"]}
        )
