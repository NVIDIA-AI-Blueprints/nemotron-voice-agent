# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation: passes one ``responses`` session's delegations through to the delegation core.

It keeps only the session's conversation, a reader per open stream, and the call routes and pending client input; the
core decides what a request means, whether a result is right and how a delegation ends.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from loguru import logger

from examples.shared.delegation_contract import (
    CALLER_CONTEXT,
    CALLER_EXCERPT,
    DELEGATION_ID,
    OPTIONS,
    STORE,
    TERMINALS,
    TYPED_RUN,
)
from examples.shared.ids import uid
from examples.shared.observe import log_observation
from examples.shared.work_provenance import WORK_PROVENANCE_KEY, strip_provenance
from gateways.live.protocol import CONVERSATION_PARAM, BackendConfig, ProtocolError, SessionConfig
from gateways.live.server_delegation.responses_client import CallerContextClient
from gateways.live.server_delegation.retention import DUPLICATE, DeliveryMailbox, MailboxFull
from gateways.live.server_delegation.sender import Request, RequestSender

if TYPE_CHECKING:
    from gateways.live.ports import DelegationEvents, DelegationRequestState, DelegationView
    from gateways.live.server_delegation.responses_client import ResponsesClient, ResponseStream

Observe = Callable[..., None]
LIVE_SESSION_ID = "live_session_id"
SETTINGS = "session.delegation.responses"
MAX_OPEN_ROUTES = 512
MAX_PENDING_RESULTS = 512
MAX_QUEUED_ITEMS = 16
CONVERSATION_SECONDS = 10
RELAY_SECONDS = 5.0
CLOSE_SECONDS = 10.0


def _log_conversation_failure(failure: Exception, operation: str) -> None:
    """Mark an authority-bearing failure for safe outer logging without replacing the exception."""
    reference = uid(operation)
    # Messages, causes, contexts and traceback locals may all contain the attachment authority. Keep the original
    # exception for the internal caller, but never attach its traceback to a production log record.
    failure._live_attachment_reference = reference
    logger.error(f"core.conversation_{operation}_failed reference={reference} error_type={type(failure).__name__}")


def request_settings(responses: BackendConfig | None) -> dict:
    """Return the fields of a Responses request that the session's ``delegation.responses`` block sets."""
    if responses is None:
        return {}
    settings = {
        "model": responses.model,
        "tools": deepcopy(responses.tools),
        "tool_choice": deepcopy(responses.tool_choice),
    }
    if responses.instructions:
        settings["instructions"] = responses.instructions
    if responses.max_output_tokens:
        settings["max_output_tokens"] = responses.max_output_tokens
    settings.update(responses.model_dump(include=set(OPTIONS), exclude_defaults=True, exclude_none=True))
    return settings


COMPOSED_SOURCE = "the core: the newest request's instructions, as composed (instructions or the default, then notes)"


def sanitize_response_event(raw: dict) -> dict:
    """Remove private ownership and redact response instructions, input, tools and output from the public copy."""
    public = strip_provenance(raw)
    response = public.get("response")
    if not isinstance(response, dict):
        return public
    snapshot = {name: value for name, value in response.items() if name != "input"}
    snapshot.update(instructions=None, tools=[], output=[])
    return {**public, "response": snapshot}


def lost_response(response_id: str, sequence_number: int | None) -> dict:
    """Return the ``response.failed`` (``server_error``) that ends a response whose stream was lost."""
    sequence = {} if sequence_number is None else {"sequence_number": sequence_number + 1}
    return {
        "type": "response.failed",
        **sequence,
        "response": {
            "id": response_id,
            "object": "response",
            "status": "failed",
            "error": {"code": "server_error", "message": "The response's stream ended before the response did"},
        },
    }


class BackendEventRelay:
    """Tells the client what its delegations do: each delegation as it starts, and every event of its responses."""

    def __init__(self, view: DelegationView, routes: ClientRoutes):
        """Relay through the session's delegation ``view``; ``routes`` hears of each call and each end."""
        self.view, self.routes = view, routes
        self.responses: dict[str, str | None] = {}
        self.forward = True

    async def announce(
        self, delegation_id: str, response_id: str, offset_ms: int, client_event_id: str | None = None
    ) -> None:
        """Announce a delegation with ``session.delegation.created`` at the first response of its starting request."""
        if not self.forward:
            return
        await self.view.emit_paced(
            "session.delegation.created",
            offset_ms=offset_ms,
            delegation={"id": delegation_id, "type": "delegation", "target": "responses", "response_id": response_id},
            client_event_id=client_event_id,
        )

    async def backend_event(
        self,
        delegation_id: str,
        event: dict,
        client_event_id: str | None = None,
        *,
        before_publish: Callable[[], bool] | None = None,
    ) -> None:
        """Relay one event of a delegation to the client, and route the calls the client must answer."""
        kind = event.get("type", "")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        if kind == "response.created":
            self.responses[delegation_id] = response.get("id")
        elif kind == "response.output_item.done" and event.get("item", {}).get("type") == "function_call":
            self.routes.note_call(delegation_id, self.responses.get(delegation_id), event["item"])
        elif kind in TERMINALS:
            self.responses.pop(delegation_id, None)
            self.routes.response_ended(delegation_id, response.get("id"))
        if not self.forward:
            return
        await self.view.emit_paced(
            "response.event",
            before_publish=before_publish,
            delegation_id=delegation_id,
            event=sanitize_response_event(event),
            client_event_id=client_event_id,
        )

    async def lost(self, delegation_id: str, sequence_number: int | None, client_event_id: str | None = None) -> None:
        """End the response the client was told of as failed, after the delegation's stream ended before it did."""
        response_id = self.responses.get(delegation_id)
        if response_id is not None:
            await self.backend_event(delegation_id, lost_response(response_id, sequence_number), client_event_id)


@dataclass(frozen=True)
class Route:
    """An open route: the delegation a relayed call belongs to, the response that relayed it, and the call's name."""

    delegation_id: str
    response_id: str | None
    name: str


@dataclass(eq=False)
class Result:
    """A result the client sent for the call ``key``, not yet passed on: each is its own entry, a repeat included."""

    key: str
    item: dict


@dataclass(eq=False)
class Typed:
    """A typed user message the client sent, not yet passed on: each is its own entry, so a request knows its own."""

    item: dict


@dataclass
class Continuation:
    """One request of a ``response.create``: the delegation it continues (``None``: a typed run) and what it carries."""

    delegation_id: str | None
    results: list[Result]
    messages: list[Typed]

    @property
    def input(self) -> list[dict]:
        """The request's input: its results in call order, then its typed messages."""
        return deepcopy([result.item for result in self.results] + [message.item for message in self.messages])


class ClientRoutes:
    """The call routes and the pending client input of one session's delegations.

    A result is taken for any routed call, repeats included: the core judges every result.
    """

    def __init__(self, observe: Observe):
        """Start with nothing routed or pending; ``observe(kind, **fields)`` gets the traces."""
        self.observe = observe
        self.routes: dict[str, Route] = {}
        self.answered: set[str] = set()
        self.results: list[Result] = []
        self.messages: list[Typed] = []
        self.in_flight: set[Result] = set()
        # Each message is known by itself, not by its place: an older queued request may be refused while a newer one is
        # accepted.
        self.messages_in_flight: set[Typed] = set()

    def note_call(self, delegation_id: str, response_id: str | None, item: dict) -> None:
        """Open the route of a call that the delegation's response ``response_id`` relayed."""
        key = item["call_id"]
        # A key relayed again routes to its newest call.
        self.routes.pop(key, None)
        self.answered.discard(key)
        self.routes[key] = Route(delegation_id, response_id, item.get("name", ""))
        while len(self.routes) > MAX_OPEN_ROUTES:
            self._close(next(iter(self.routes)))

    def response_ended(self, delegation_id: str, response_id: str | None) -> None:
        """Close the routes of the delegation's other responses, once ``response_id`` ended."""
        for key, route in list(self.routes.items()):
            if route.delegation_id == delegation_id and route.response_id != response_id:
                self._close(key)

    def _close(self, key: str) -> None:
        self.routes.pop(key, None)
        self.answered.discard(key)
        # With no route, a result can never be passed on; one a queued request carries goes with that request.
        self.results = [result for result in self.results if result.key != key or result in self.in_flight]

    def close(self) -> None:
        """Close every route and drop all pending input."""
        self.routes.clear()
        self.answered.clear()
        self.results.clear()
        self.messages.clear()
        self.in_flight.clear()
        self.messages_in_flight.clear()

    def prepare_item(self, item: dict) -> Result | Typed:
        """Validate and copy one exact input without consuming routes, result buffers or typed messages."""
        kind = item.get("type", "message")
        if kind == "function_call_output":
            key = item.get("call_id")
            if not isinstance(key, str) or key not in self.routes:
                raise ProtocolError("Unknown pending call", param="item.call_id")
            if len(self.results) >= MAX_PENDING_RESULTS:
                raise ProtocolError(f"Too many pending results: at most {MAX_PENDING_RESULTS}", "rate_limit_exceeded")
            return Result(key, deepcopy(item))
        if kind != "message":
            raise ProtocolError("Unsupported response item", param="item.type")
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            raise ProtocolError("Expected a user message", param="item")
        parts = item["content"]
        if not parts or any(
            not isinstance(p, dict)
            or p.get("type") not in {"input_text", "input_image"}
            or (p["type"] == "input_text" and not isinstance(p.get("text"), str))
            for p in parts
        ):
            raise ProtocolError("Unsupported message content", param="item.content")
        if len(self.messages) >= MAX_QUEUED_ITEMS:
            raise ProtocolError(f"Too many queued response items: at most {MAX_QUEUED_ITEMS}", "rate_limit_exceeded")
        return Typed(deepcopy(item))

    def commit_item(self, prepared: Result | Typed) -> None:
        """Consume the already checked exact item, without an intervening await or frontier re-read."""
        if isinstance(prepared, Result):
            self.results.append(prepared)
            self.observe("tool.result.received", call_id=prepared.key, name=self.routes[prepared.key].name)
        else:
            self.messages.append(prepared)
            text = "\n".join(p.get("text", "") for p in prepared.item["content"] if p.get("type") == "input_text")
            self.observe("input.typed", text=text, chars=len(json.dumps(prepared.item)))

    def item(self, item: dict) -> None:
        """Take a result or a typed user message from ``response.item.create``; raise ``ProtocolError`` to refuse it."""
        kind = item.get("type", "message")
        if kind == "function_call_output":
            self._result(item)
        elif kind == "message":
            self._message(item)
        else:
            raise ProtocolError("Unsupported response item", param="item.type")

    def _result(self, item: dict) -> None:
        key = item.get("call_id")
        route = self.routes.get(key) if isinstance(key, str) else None
        if route is None:
            raise ProtocolError("Unknown pending call", param="item.call_id")
        if len(self.results) >= MAX_PENDING_RESULTS:
            raise ProtocolError(f"Too many pending results: at most {MAX_PENDING_RESULTS}", "rate_limit_exceeded")
        self.results.append(Result(key, deepcopy(item)))
        self.observe("tool.result.received", call_id=key, name=route.name)

    def _message(self, item: dict) -> None:
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            raise ProtocolError("Expected a user message", param="item")
        parts = item["content"]
        if not parts or any(
            not isinstance(p, dict)
            or p.get("type") not in {"input_text", "input_image"}
            or (p["type"] == "input_text" and not isinstance(p.get("text"), str))
            for p in parts
        ):
            raise ProtocolError("Unsupported message content", param="item.content")
        if len(self.messages) >= MAX_QUEUED_ITEMS:
            raise ProtocolError(f"Too many queued response items: at most {MAX_QUEUED_ITEMS}", "rate_limit_exceeded")
        self.messages.append(Typed(deepcopy(item)))
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "input_text")
        self.observe("input.typed", text=text, chars=len(json.dumps(item)))

    def requests(self) -> list[Continuation]:
        """Return what a ``response.create`` sends, by the Live API's rule for a continuation.

        Each response whose calls all have results is continued, oldest first, and the newest also takes the typed
        messages; with none, it is a typed run. Results still owed with nothing to continue raise ``ProtocolError``.
        """
        pending: dict[str, list[Result]] = {}
        for result in self.results:
            if result not in self.in_flight:
                pending.setdefault(result.key, []).append(result)
        carried = {result.key for result in self.in_flight}
        responses: dict[tuple[str, str | None], list[str]] = {}
        for key, route in self.routes.items():
            responses.setdefault((route.delegation_id, route.response_id), []).append(key)
        continued: list[Continuation] = []
        missing: list[str] = []
        for (delegation_id, _), keys in responses.items():
            owed = [key for key in keys if key not in self.answered and key not in carried and key not in pending]
            results = [result for key in keys for result in pending.get(key, [])]
            if owed:
                missing += owed
            elif results:
                continued.append(Continuation(delegation_id, results, []))
        messages = [message for message in self.messages if message not in self.messages_in_flight]
        if not continued and missing:
            raise ProtocolError(
                f"Return the results of the pending calls before response.create: {', '.join(missing)}",
                "function_call_outputs_required",
            )
        if not continued:
            return [Continuation(None, [], messages)]
        continued[-1].messages = messages
        return continued

    def sending(self, continuation: Continuation) -> None:
        """Mark what ``continuation`` carries as in flight while its request is queued."""
        self.in_flight.update(continuation.results)
        self.messages_in_flight.update(continuation.messages)

    def release(self, continuation: Continuation) -> None:
        """Keep pending, while routed, what a request carried that was not sent or was refused but not as invalid."""
        self.in_flight.difference_update(continuation.results)
        self.messages_in_flight.difference_update(continuation.messages)
        self.results = [result for result in self.results if result.key in self.routes or result in self.in_flight]

    def refused(self, continuation: Continuation) -> None:
        """Drop what a request refused as invalid carries, its calls unanswered: it would be refused again."""
        carried, typed = set(continuation.results), set(continuation.messages)
        self.in_flight -= carried
        self.results = [result for result in self.results if result not in carried]
        self.messages_in_flight -= typed
        self.messages = [message for message in self.messages if message not in typed]

    def sent(self, continuation: Continuation) -> None:
        """Drop what an accepted request carries from the pending input, and mark its calls answered."""
        carried, typed = set(continuation.results), set(continuation.messages)
        self.in_flight -= carried
        self.results = [result for result in self.results if result not in carried]
        self.answered.update(result.key for result in carried if result.key in self.routes)
        self.messages_in_flight -= typed
        self.messages = [message for message in self.messages if message not in typed]


class ServerDelegation:
    """Passes one session's delegations through to its delegation core, relays their events, answers the commands."""

    target: Literal["responses"] = "responses"
    reports_events = True

    def __init__(self, client: ResponsesClient, *, observe: Observe = log_observation):
        """Reach the delegation core through ``client``; ``observe`` gets the traces of the client's items."""
        self.client, self.observe = client, observe
        self.view: DelegationView | None = None
        self.conversation: str | None = None
        self.routes: ClientRoutes | None = None
        self.relay: BackendEventRelay | None = None
        self.listener: DelegationEvents | None = None
        self.sender = RequestSender(self._send)
        self.checks: set[asyncio.Task] = set()
        self.readers: set[asyncio.Task] = set()
        self._newest_reader: dict[str, asyncio.Task] = {}
        self.composed: str | None = None
        self._closing = False
        self._opened = self._closed = False
        self.mailbox: DeliveryMailbox | None = None
        self._reconciled_output = False
        self._attached = True
        self.on_mailbox_full: Callable[[], None] | None = None
        self._attachment_ready = asyncio.Event()
        self._attachment_ready.set()
        self._control_openings: dict[str, asyncio.Task] = {}

    async def reconnect_supported(self) -> bool:
        """Require an explicit attachment boundary on the actual core client."""
        check = getattr(self.client, "attachment_supported", None)
        return await check() if check is not None else callable(getattr(self.client, "set_attachment", None))

    def retain_attachments(self) -> None:
        """Enable bounded canonical delivery evidence for this logical call."""
        self.mailbox = DeliveryMailbox()

    async def suspend_attachment(self) -> None:
        """Fence the frontend and effects, retaining accepted requests, readers, results and the core lease."""
        self._attached = False
        self._attachment_ready.clear()
        self.listener = None
        if self.relay is not None:
            self.relay.forward = False
        if self.mailbox is not None and self.routes is not None:
            self.mailbox.disconnected(self.routes)
        if self.conversation is not None:
            await self.client.set_attachment(self.conversation, False)

    async def resume_attachment(self) -> dict:
        """Restore the retained frontier to the new listener while public forwarding remains fenced."""
        current = {}
        if self.mailbox is not None and self.mailbox.exhausted:
            raise MailboxFull()
        if self.conversation is not None:
            current = await self.client.set_attachment(self.conversation, True)
        if self.mailbox is not None:
            if self.mailbox.exhausted:
                raise MailboxFull()
            self.mailbox.revalidate_calls(set(current.get("pending_calls", [])))
        if self.mailbox is not None and self.listener is not None:
            calls: set[str] = set()
            for entry in self.mailbox.replay():
                if entry.request_state is not None:
                    self._hand_request(entry.request_ref, entry.request_state, entry.delegation_id)
                    continue
                restore_provenance = getattr(self.listener, "restore_work_provenance", None)
                if entry.forwarded and WORK_PROVENANCE_KEY in entry.event and callable(restore_provenance):
                    restore_provenance(entry.delegation_id, deepcopy(entry.event))
                item = entry.event.get("item") or {}
                kind = entry.event.get("type")
                if kind == "response.created":
                    calls.discard(entry.delegation_id)
                elif item.get("type") == "function_call":
                    calls.add(entry.delegation_id)
                terminal = kind in TERMINALS
                if terminal and entry.forwarded and (kind != "response.completed" or entry.delegation_id not in calls):
                    restore_terminal = getattr(self.listener, "restore_terminal", None)
                    if restore_terminal is not None:
                        restored = {**entry.event, "_nva_lost": True} if entry.lost else entry.event
                        restore_terminal(entry.delegation_id, restored)
                        continue
                if entry.lost:
                    entry.forwarded = True
                    self.listener.on_delegation_lost(entry.delegation_id)
                    continue
                if item.get("type") == "message" and entry.forwarded:
                    restore_progress = getattr(self.listener, "restore_progress", None)
                    if restore_progress is not None and item.get("phase") == "commentary":
                        restore_progress(entry.delegation_id, item)
                    restore_final = getattr(self.listener, "restore_final", None)
                    if restore_final is not None and item.get("phase", "final_answer") != "commentary":
                        restore_final(entry.delegation_id)
                    continue
                entry.forwarded = True
                self.listener.on_delegation_event(entry.delegation_id, deepcopy(entry.event))
        self._attached = True
        self._attachment_ready.set()
        return current

    async def relay_pending_calls(self, current: dict) -> None:
        """Enable public forwarding after admission and send only revalidated, never-emitted calls."""
        if self.mailbox is not None and self.mailbox.exhausted:
            raise MailboxFull()
        if self.relay is not None:
            self.relay.forward = True
        if self.mailbox is None or self.view is None:
            return
        # Revalidate after normal delivery is active: an asynchronous core client can wake a worker during
        # resume_attachment(), after its first snapshot but before this attachment can receive that call.
        if self.conversation is not None:
            current = await self.client.set_attachment(self.conversation, True)
        if self.mailbox.exhausted:
            raise MailboxFull()
        allowed = set(current.get("pending_calls", []))
        self.mailbox.revalidate_calls(allowed)
        for entry in self.mailbox.entries:
            item = entry.event.get("item") or {}
            key = item.get("call_id")
            if item.get("type") != "function_call" or key in self.mailbox.emitted or key not in allowed:
                continue
            await self.view.emit_paced(
                "response.event",
                before_publish=partial(self.mailbox.claim_call, key),
                delegation_id=entry.delegation_id,
                event=sanitize_response_event(entry.event),
            )

    async def open(self, view: DelegationView | None) -> None:
        """Open the session's conversation with the core, then build the call routes and the relay.

        With no ``view`` nothing is relayed and no settings are sent; a conversation not created in time raises. A
        ``store: true`` session that names ``delegation.responses.conversation`` attaches to that conversation instead,
        waiting up to the core's close budget, since the call before may still be closing; a conversation the core
        refuses (unknown, ended or serving another call) raises ``ProtocolError`` ``invalid_value``, naming no id.
        """
        if self._opened or self._closed:
            return
        self._opened = True
        self.view = view
        metadata = {LIVE_SESSION_ID: view.id} if view is not None else {}
        if view is not None and view.store:
            metadata[STORE] = "true"
        try:
            if view is None or view.responses is None or view.responses.conversation is None:
                async with asyncio.timeout(CONVERSATION_SECONDS):
                    conversation = await self.client.create_conversation(metadata)
            else:
                conversation = await self._attach(metadata)
        except TimeoutError:
            raise RuntimeError(
                f"The delegation core did not open a conversation within {CONVERSATION_SECONDS} s"
            ) from None
        # Closed while the core created or attached it: nothing else would delete it (a kept one stays kept).
        if self._closed:
            await self._delete(conversation["id"])
            return
        self.conversation = conversation["id"]
        self.routes = ClientRoutes(self.observe)
        if view is not None:
            self.relay = BackendEventRelay(view, self.routes)
        self.sender.start()

    async def _attach(self, metadata: dict[str, str]) -> dict:
        """Attach to the core's kept conversation that the session names (``POST /v1/conversations/{id}``).

        It waits up to the core's close budget, at least ``CONVERSATION_SECONDS``, since the call before may still be
        closing. A refusal raises ``ProtocolError`` ``invalid_value``. Neither it nor a line of its traceback names
        the id, since the id is what attaches.
        """
        conversation_id = self.view.responses.conversation
        budget = getattr(self.client, "close_seconds", None)
        limit = max(CONVERSATION_SECONDS, budget or 0)
        try:
            async with asyncio.timeout(limit):
                return await self.client.update_conversation(conversation_id, metadata)
        except TimeoutError:
            failure = RuntimeError(f"The delegation core did not attach the conversation within {limit:g} s")
            _log_conversation_failure(failure, "attach")
            raise failure from None
        except LookupError:
            raise ProtocolError(
                "The conversation to continue is unknown, has ended, or serves another call",
                param=CONVERSATION_PARAM,
            ) from None
        except Exception as refused:
            _log_conversation_failure(refused, "attach")
            raise

    async def _delete(self, conversation_id: str) -> dict:
        """Delete a conversation, retaining the original failure only for the internal caller."""
        try:
            return await self.client.delete_conversation(conversation_id)
        except Exception as failure:
            _log_conversation_failure(failure, "delete")
            raise

    def stop_handing(self) -> None:
        """Close step 2: send no new request and hand the frontend nothing; the readers still relay to the client."""
        self._closing = True

    async def close(self) -> None:
        """Close step 4: refuse unsent requests, delete the conversation, let the readers end, close the call routes.

        A conversation that the core keeps for work outliving the call (``deleted: false``) stays named, so
        ``session.closed`` tells the app its id.
        """
        if self._closed:
            return
        self._closed = self._closing = True
        self._attachment_ready.set()
        conversation, self.conversation = self.conversation, None
        for check in list(self.checks):
            check.cancel()
        try:
            await self.sender.close()
            openings = tuple(self._control_openings.values())
            for opening in openings:
                opening.cancel()
            await asyncio.gather(*openings, return_exceptions=True)
            if conversation is not None:
                deleted = await self._delete(conversation)
                if deleted.get("deleted") is False:
                    self.conversation = conversation
            if self.readers:
                await asyncio.gather(*self.readers, return_exceptions=True)
        finally:
            for reader in list(self.readers):
                reader.cancel()
            if self.routes is not None:
                self.routes.close()

    @property
    def close_seconds(self) -> float:
        """How long its close may take: the core's close budget, as the client says, plus the relay of the ends."""
        budget = getattr(self.client, "close_seconds", None)
        return CLOSE_SECONDS if budget is None else budget + RELAY_SECONDS

    @property
    def acknowledgement_delay_ms(self) -> int:
        """The session's acknowledgement delay, read when asked; 0 (at once) without a live session."""
        return self.view.acknowledgement_delay_ms if self.view is not None else 0

    def begin_caller_turn(self) -> Callable[[], None] | None:
        """Use the in-process routing hold when this delegation client offers it."""
        begin = getattr(self.client, "begin_caller_turn", None)
        if callable(begin) and self.conversation is not None and not self._closing and self._attached:
            settle = begin(self.conversation)
            if settle is not None:

                def attached_settle() -> None:
                    if not self._closing and self._attached:
                        settle()

                return attached_settle
        return None

    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its delegation id once the core accepts it; a refusal raises."""
        self._check_open()
        if self.listener is None:
            raise RuntimeError("listen() comes before the first delegate(): its events would reach nobody")
        request = Request(deepcopy(input), uid("item"), offset_ms=self._clock(), starts=True)
        self.sender.put(request)
        try:
            return await request.outcome
        finally:
            request.returned.set()

    def _clock(self) -> int:
        return self.view.input_ms if self.view is not None else 0

    def _check_open(self) -> None:
        if self._closing:
            raise RuntimeError("Session is closing")
        if self.conversation is None:
            raise RuntimeError("The server delegation is not open")
        if not self._attached:
            raise ProtocolError("The caller attachment is unavailable", "session_reconnecting")

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation, typed runs included, to ``listener``."""
        self.listener = listener

    async def _send(self, request: Request) -> bool:
        """Send one request for the sender, settle its outcome, and return whether the core accepted it."""
        await self._attachment_ready.wait()
        if self.conversation is None or self._closing:
            if not request.outcome.done():
                request.outcome.set_exception(RuntimeError("Session is closing"))
            return False
        metadata = {DELEGATION_ID: request.delegation_id}
        if self.view is not None:
            metadata = {LIVE_SESSION_ID: self.view.id, **metadata}
        if request.typed_run:
            metadata[TYPED_RUN] = "true"
        input_items = deepcopy(request.input)
        context = None
        if request.starts and input_items:
            context = input_items[0].pop(CALLER_CONTEXT, None)
            if isinstance(excerpt := input_items[0].pop(CALLER_EXCERPT, None), str):
                metadata[CALLER_EXCERPT] = excerpt
        params = {
            **request_settings(self.view.responses if self.view is not None else None),
            "input": input_items,
            "conversation": self.conversation,
            "metadata": metadata,
            "stream": True,
            "background": True,
            "store": False,
        }
        if request.starts:
            try:
                self._request_state(request.delegation_id, "submitted", request.delegation_id)
            except MailboxFull as error:
                if not request.outcome.done():
                    request.outcome.set_exception(error)
                return False
        try:
            client = self.client
            stream = (
                await client.create_response_with_caller_context(params, context)
                if context is not None and isinstance(client, CallerContextClient)
                else await client.create_response(params)
            )
        except asyncio.CancelledError:
            if request.starts:
                self._request_state(request.delegation_id, "cancelled", request.delegation_id)
            raise
        except Exception as refused:
            if request.starts:
                self._request_state(request.delegation_id, "failed", request.delegation_id)
            code = getattr(refused, "code", None)
            if isinstance(code, str):
                refusal = ProtocolError(str(refused), code)
            elif isinstance(refused, ValueError):
                refusal = ProtocolError(str(refused))
            else:
                refusal = refused
            if request.outcome.done():
                logger.info(f"The core refused delegation {request.delegation_id}, whose caller had gone: {refused}")
            else:
                request.outcome.set_exception(refusal)
            return False
        # An accepted request gets its reader even when its caller gave up, so no accepted work goes unread.
        try:
            if request.starts:
                self._request_state(request.delegation_id, "accepted", request.delegation_id)
        finally:
            self._start_reader(request, stream)
        if not request.outcome.done():
            request.outcome.set_result(request.delegation_id)
        return True

    def _request_state(self, request_ref: str, state: DelegationRequestState, delegation_id: str | None = None) -> None:
        """Journal new-source submission before dispatch, then report only descriptive transport facts."""
        entry = None
        try:
            if self.mailbox is not None:
                entry = self.mailbox.record_request(request_ref, state, delegation_id)
                if entry is None:
                    return
        except MailboxFull:
            self._mailbox_full()
            if state == "submitted":
                raise  # No unjournalled request has been sent yet.
            # Acceptance cannot be undone. The accepted stream still gets its owned reader as the call closes.
        if not self._closing and self._attached:
            if entry is not None:
                entry.forwarded = True
            if not self._hand_request(request_ref, state, delegation_id) and state == "submitted":
                self._request_state(request_ref, "failed", delegation_id)
                raise RuntimeError("The frontend could not register the submitted request")

    def _mailbox_full(self) -> None:
        """Fence new work once before the session's close owner runs."""
        if not self._closing:
            self.stop_handing()
            if self.on_mailbox_full is not None:
                self.on_mailbox_full()

    def _hand_request(self, request_ref: str, state: DelegationRequestState, delegation_id: str | None = None) -> bool:
        callback = getattr(self.listener, "on_delegation_request", None)
        if callable(callback):
            try:
                callback(request_ref, state, delegation_id)
            except Exception:
                logger.exception(f"The frontend could not take delegation request state {request_ref}")
                return False
        return True

    def _start_reader(self, request: Request, stream: ResponseStream) -> None:
        """Keep every producer on the same per-source response ordering and cleanup path."""
        previous = self._newest_reader.get(request.delegation_id)
        reader = asyncio.create_task(self._read(request, stream, previous), name="delegation-reader")
        self._newest_reader[request.delegation_id] = reader
        self.readers.add(reader)
        reader.add_done_callback(self.readers.discard)
        reader.add_done_callback(lambda done: self._reader_ended(request.delegation_id, done))

    def _reader_ended(self, delegation_id: str, reader: asyncio.Task) -> None:
        if self._newest_reader.get(delegation_id) is reader:
            del self._newest_reader[delegation_id]

    async def _read(self, request: Request, stream: ResponseStream, previous: asyncio.Task | None) -> None:
        """Hand each event of a request to the frontend, then relay it to the client, until its stream ends.

        It starts once ``delegate()`` returned and the delegation's ``previous`` reader ended, so events stay in order.
        """
        await request.returned.wait()
        if previous is not None and not previous.done():
            await asyncio.wait({previous})
        delegation_id = request.delegation_id
        announcing = request.starts
        ended, sequence = False, None
        response_id = None
        try:
            if self.mailbox is not None and self.mailbox.exhausted:
                raise MailboxFull()
            async for event in stream:
                kind = event.get("type") if isinstance(event, dict) else None
                ended = ended or kind in TERMINALS
                if isinstance(event, dict) and isinstance(event.get("sequence_number"), int):
                    sequence = event["sequence_number"]
                if kind == "response.created" and isinstance(event.get("response"), dict):
                    self.composed = event["response"].get("instructions") or self.composed
                    response_id = event["response"].get("id")
                entry = None
                if self.mailbox is not None:
                    if self.mailbox.exhausted:
                        raise MailboxFull()
                    entry = self.mailbox.record(delegation_id, event, response_id=response_id)
                    if entry == DUPLICATE:
                        if kind == "response.created":
                            announcing = False
                        continue
                if not self._closing and self._attached:
                    if entry is not None and self.listener is not None:
                        entry.forwarded = True  # A mutating callback may fail after queuing presentation.
                    self._hand(delegation_id, event)
                announce = announcing and kind == "response.created"
                if announce:
                    announcing = False
                if self.relay is not None:
                    await self._relay(request, event, announce)
        except MailboxFull:
            self._mailbox_full()
            try:
                await stream.aclose()
            except Exception as error:
                logger.warning(
                    f"Could not close a response stream after exhausting its journal: {type(error).__name__}"
                )
            return
        except Exception:
            logger.exception(f"The stream of delegation {delegation_id} failed")
        if not ended and not self._closing:
            await self._lost(request, sequence)
        if request.starts and announcing and not self._closing:
            # No source frame reached the listener; do not retain a first-event hold after its stream ended.
            self._request_state(request.request_ref or request.delegation_id, "lost", request.delegation_id)

    def _hand(self, delegation_id: str, event: dict) -> bool:
        """Hand one event to the frontend; its failure must not stop the delegation's relay."""
        if self.listener is None:
            return False
        try:
            self.listener.on_delegation_event(delegation_id, event)
            return True
        except Exception:
            logger.exception(f"The frontend could not take an event of delegation {delegation_id}")
            return False

    async def _relay(self, request: Request, event: dict, announce: bool) -> None:
        delegation_id = request.delegation_id
        try:
            before_publish = None
            if self.mailbox is not None and event.get("type") == "response.output_item.done":
                item = event.get("item") or {}
                if item.get("type") == "function_call":
                    before_publish = partial(self.mailbox.claim_call, item["call_id"])
            if announce:
                response_id = event["response"]["id"]
                await self.relay.announce(delegation_id, response_id, request.offset_ms, request.client_event_id)
            await self.relay.backend_event(delegation_id, event, request.client_event_id, before_publish=before_publish)
        except Exception:
            logger.exception(f"An event of delegation {delegation_id} could not be relayed; it is skipped")

    async def _lost(self, request: Request, sequence: int | None) -> None:
        delegation_id = request.delegation_id
        logger.warning(f"The stream of delegation {delegation_id} ended before its response did")
        entry = None
        if self.mailbox is not None and self.relay is not None:
            response_id = self.relay.responses.get(delegation_id)
            if response_id is not None:
                try:
                    entry = self.mailbox.record(delegation_id, lost_response(response_id, sequence))
                    if entry == DUPLICATE:
                        return
                    if entry is not None:
                        entry.lost = True
                except MailboxFull:
                    self._mailbox_full()
                    return
        if self.listener is not None:
            try:
                if entry is not None:
                    entry.forwarded = True
                self.listener.on_delegation_lost(delegation_id)
            except Exception:
                logger.exception(f"The frontend could not take the loss of delegation {delegation_id}")
        if self.relay is not None:
            try:
                await self.relay.lost(delegation_id, sequence, request.client_event_id)
            except Exception:
                logger.exception(f"The loss of delegation {delegation_id} could not be relayed")

    def knows(self, delegation_id: object) -> bool:
        """Return ``False``: in ``responses`` mode the core reports its own work, so no append names a delegation."""
        return False

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch and return the new settings; the core checks them in the background.

        A ``delegation.responses.conversation`` that names the session's own conversation changes nothing.
        """
        delegation = patch.get("delegation") if isinstance(patch, dict) else None
        responses = delegation.get("responses") if isinstance(delegation, dict) else None
        own = isinstance(responses, dict) and self.conversation is not None
        if own and responses.get("conversation") == self.conversation:
            responses = {key: value for key, value in responses.items() if key != "conversation"}
            patch = {**patch, "delegation": {**delegation, "responses": responses}}
        updated = config.updated(patch)
        self._check(updated.delegation.responses, client_event_id)
        return updated

    def _check(self, responses: BackendConfig, client_event_id: str | None) -> None:
        """Ask the core in the background whether it can honor ``responses`` (``POST /v1/responses/input_tokens``)."""
        if self.conversation is None or self.view is None:
            return
        request = {**request_settings(responses), "conversation": self.conversation, "input": []}
        check = asyncio.create_task(self._checked(request, client_event_id), name="settings-check")
        self.checks.add(check)
        check.add_done_callback(self.checks.discard)

    async def _checked(self, request: dict, client_event_id: str | None) -> None:
        try:
            await self.client.input_tokens(request)
        except ValueError as refused:
            if not self._closing:
                param = getattr(refused, "param", None)
                where = f"{SETTINGS}.{param}" if param else SETTINGS
                self.view.report_refusal(str(refused), "responses_validation_failed", client_event_id, where)
        except Exception:
            logger.exception("The delegation core could not check the session's settings")

    def prepare_item_create(self, item: dict) -> Callable:
        """Prepare input for the original waiting frame or preserve late evidence on a retained call."""
        key = item.get("call_id")
        call = self.mailbox.recovering_calls.get(key) if self.mailbox is not None and isinstance(key, str) else None
        if (
            call is not None
            and call.state == "waiting"
            and key in self.routes.routes
            and key not in self.routes.answered
        ):
            if item.get("type") != "function_call_output" or not isinstance(item.get("output"), (str, list)):
                raise ProtocolError("A recovered call requires its function output", param="item")
            existing = next((result for result in self.routes.results if result.key == key), None)
            if existing is not None:
                if existing.item["output"] != item["output"]:
                    raise ProtocolError("The recovered call already has a different result", param="item.output")
                return lambda: None
        elif call is not None:
            if item.get("type") != "function_call_output" or not isinstance(item.get("output"), str):
                raise ProtocolError("Late evidence requires the exact call and a string output", param="item")
            if not callable(getattr(self.client, "reconcile_tool_outputs", None)):
                raise ProtocolError("This Core cannot reconcile late action evidence", "action_outcome_unclear")
            frozen = deepcopy(item)

            async def evidence_only():
                try:
                    receipt = await self.client.reconcile_tool_outputs(self.conversation, [frozen])
                except ValueError as refused:
                    raise ProtocolError(str(refused), getattr(refused, "code", None) or "invalid_value") from None
                if key not in receipt.get("recorded_call_ids", ()) or receipt.get("resumed") is not False:
                    raise ProtocolError("Core did not confirm the original result", "action_outcome_unclear")
                self.mailbox.record_result(key)
                if key in self.routes.routes:
                    self.routes.answered.add(key)
                self._reconciled_output = True
                self.observe("tool.late_evidence.recorded", call_id=key, resumed=False)

            return evidence_only
        prepared = self.routes.prepare_item(item)
        return lambda: self.routes.commit_item(prepared)

    def prepare_response_create(self, client_event_id: str | None) -> Callable[[], None]:
        """Capture the exact ready continuations without consuming any result or request."""
        self._check_open()
        continuations = self.routes.requests()
        if (
            self._reconciled_output
            and len(continuations) == 1
            and continuations[0].delegation_id is None
            and not continuations[0].messages
        ):
            # The result already reached Core through its evidence-only path.
            continuations = []

        def commit() -> None:
            self._reconciled_output = False
            batch = []
            for continuation in continuations:
                self.routes.sending(continuation)
                request = Request(
                    continuation.input,
                    continuation.delegation_id or uid("item"),
                    offset_ms=self._clock(),
                    starts=continuation.delegation_id is None,
                    typed_run=continuation.delegation_id is None,
                    client_event_id=client_event_id,
                )
                request.returned.set()
                request.outcome.add_done_callback(partial(self._settled, continuation, request), context=copy_context())
                batch.append(request)
            if batch:
                self.sender.put(*batch)

        return commit

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """Take ordinary input, or reconcile trusted late evidence without resuming business work."""
        commit = self.prepare_item_create(item)
        result = commit()
        if inspect.isawaitable(result):
            await result

    async def response_create(self, client_event_id: str | None) -> None:
        """Queue a continuation for each waiting response whose results the client completed, or a typed run.

        It returns once the requests are queued, so the core's answer never holds the command lock.
        """
        self.prepare_response_create(client_event_id)()

    def _settled(self, continuation: Continuation, request: Request, outcome: asyncio.Future) -> None:
        """Settle a request of the client's own: drop or keep what it carries, and report a refusal to the client."""
        if not outcome.cancelled() and outcome.exception() is None:
            self.routes.sent(continuation)
            if self.mailbox is not None:
                for result in continuation.results:
                    self.mailbox.record_result(result.key)
            return
        refusal = None if outcome.cancelled() else outcome.exception()
        if isinstance(refusal, ProtocolError) and refusal.code == "invalid_value":
            self.routes.refused(continuation)
        else:
            self.routes.release(continuation)
        if refusal is None or self.view is None or self._closing:
            return
        if isinstance(refusal, ProtocolError):
            self.view.report_refusal(str(refusal), refusal.code, request.client_event_id)
        else:
            logger.warning(f"The core refused a request of the client's own: {refusal}")
            self.view.report_refusal("The request could not be sent.", "server_error", request.client_event_id)

    async def history(self) -> dict | None:
        """Return the part's half of ``/history``: the settings each request carries and the conversation's items."""
        settings = request_settings(self.view.responses if self.view is not None else None)
        items = []
        if self.conversation is not None:
            items = (await self.client.list_conversation_items(self.conversation))["data"]
        instructions = settings.get("instructions")
        prompt = [{"source": "delegation.responses.instructions", "text": instructions}] if instructions else []
        if self.composed:
            prompt.append({"source": COMPOSED_SOURCE, "text": self.composed})
        return {"model": settings.get("model"), "prompt": prompt, "items": items}
