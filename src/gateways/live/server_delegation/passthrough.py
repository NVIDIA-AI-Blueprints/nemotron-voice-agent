# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation: passes one ``responses`` session's delegations through to the delegation core.

It keeps only the session's conversation, a reader per open stream, and the call routes and pending client input; the
core decides what a request means, whether a result is right and how a delegation ends. A delegation's request carries
what the frontend stated about it (:func:`stated`) beside the gateway's own metadata keys.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from loguru import logger

from examples.shared.delegation_contract import DELEGATION_ID, OPTIONS, TERMINALS, TYPED_RUN
from examples.shared.ids import uid
from examples.shared.observe import log_observation
from gateways.live.protocol import BackendConfig, ProtocolError, SessionConfig
from gateways.live.server_delegation.sender import Request, RequestSender

if TYPE_CHECKING:
    from gateways.live.ports import DelegationEvents, DelegationView
    from gateways.live.server_delegation.responses_client import ResponsesClient

Observe = Callable[..., None]
LIVE_SESSION_ID = "live_session_id"
SETTINGS = "session.delegation.responses"
MAX_OPEN_ROUTES = 512
MAX_PENDING_RESULTS = 512
MAX_QUEUED_ITEMS = 16
# What the frontend may state about a delegation (its metadata pairs): at most this many pairs, each key and value a
# string of at most these lengths, the official Responses limits per pair.
MAX_STATED_METADATA = 8
MAX_METADATA_KEY = 64
MAX_METADATA_VALUE = 512


def stated(metadata: Mapping[str, str] | None, observe: Callable[..., None] = log_observation) -> dict[str, str]:
    """Return the pairs of ``metadata`` within the limits; any other pair is dropped with a warning and an observation.

    It never raises: a refused hand-off would be spoken as a refusal, and a metadata fault must never refuse a
    caller's request.
    """
    kept: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        reason = ""
        if not isinstance(key, str) or not isinstance(value, str):
            reason = "not_text"
        elif not key or len(key) > MAX_METADATA_KEY or len(value) > MAX_METADATA_VALUE:
            reason = "too_long"
        elif len(kept) >= MAX_STATED_METADATA:
            reason = "too_many"
        if reason:
            logger.warning(f"A delegation's metadata pair {key!r} was dropped: {reason}")
            observe("delegation.metadata_dropped", key=str(key)[:MAX_METADATA_KEY], reason=reason)
        else:
            kept[key] = value
    return kept


CONVERSATION_SECONDS = 10
RELAY_SECONDS = 5.0
CLOSE_SECONDS = 10.0


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
    """Return ``raw`` with its response snapshot redacted: no ``instructions``, ``input``, ``tools`` or ``output``."""
    response = raw.get("response")
    if not isinstance(response, dict):
        return raw
    snapshot = {name: value for name, value in response.items() if name != "input"}
    snapshot.update(instructions=None, tools=[], output=[])
    return {**raw, "response": snapshot}


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

    async def announce(
        self, delegation_id: str, response_id: str, offset_ms: int, client_event_id: str | None = None
    ) -> None:
        """Announce a delegation with ``session.delegation.created`` at the first response of its starting request."""
        await self.view.emit_paced(
            "session.delegation.created",
            offset_ms=offset_ms,
            delegation={"id": delegation_id, "type": "delegation", "target": "responses", "response_id": response_id},
            client_event_id=client_event_id,
        )

    async def backend_event(self, delegation_id: str, event: dict, client_event_id: str | None = None) -> None:
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
        await self.view.emit_paced(
            "response.event",
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

    async def open(self, view: DelegationView | None) -> None:
        """Open the session's conversation with the core, then build the call routes and the relay.

        With no ``view`` nothing is relayed and no settings are sent; a conversation not created in time raises.
        """
        if self._opened or self._closed:
            return
        self._opened = True
        self.view = view
        metadata = {LIVE_SESSION_ID: view.id} if view is not None else {}
        try:
            async with asyncio.timeout(CONVERSATION_SECONDS):
                conversation = await self.client.create_conversation(metadata)
        except TimeoutError:
            raise RuntimeError(
                f"The delegation core did not open a conversation within {CONVERSATION_SECONDS} s"
            ) from None
        # Closed while the core created it: nothing else would delete it.
        if self._closed:
            await self.client.delete_conversation(conversation["id"])
            return
        self.conversation = conversation["id"]
        self.routes = ClientRoutes(self.observe)
        if view is not None:
            self.relay = BackendEventRelay(view, self.routes)
        self.sender.start()

    def stop_handing(self) -> None:
        """Close step 2: send no new request and hand the frontend nothing; the readers still relay to the client."""
        self._closing = True

    async def close(self) -> None:
        """Close step 4: refuse unsent requests, delete the conversation, let the readers end, close the call routes."""
        if self._closed:
            return
        self._closed = self._closing = True
        conversation, self.conversation = self.conversation, None
        for check in list(self.checks):
            check.cancel()
        try:
            await self.sender.close()
            if conversation is not None:
                await self.client.delete_conversation(conversation)
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

    async def delegate(self, input: list[dict], *, metadata: Mapping[str, str] | None = None) -> str:
        """Delegate a request and return its delegation id once the core accepts it; a refusal raises.

        ``metadata``: what the frontend states about the request, added under the gateway's own keys (:func:`stated`).
        """
        self._check_open()
        if self.listener is None:
            raise RuntimeError("listen() comes before the first delegate(): its events would reach nobody")
        pairs = stated(metadata, self.observe)
        request = Request(deepcopy(input), uid("item"), offset_ms=self._clock(), starts=True, metadata=pairs)
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

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation, typed runs included, to ``listener``."""
        self.listener = listener

    async def _send(self, request: Request) -> bool:
        """Send one request for the sender, settle its outcome, and return whether the core accepted it."""
        if self.conversation is None or self._closing:
            if not request.outcome.done():
                request.outcome.set_exception(RuntimeError("Session is closing"))
            return False
        metadata = dict(request.metadata)  # the frontend's pairs first: the gateway's own keys, written last, win
        if self.view is not None:
            metadata[LIVE_SESSION_ID] = self.view.id
        metadata[DELEGATION_ID] = request.delegation_id
        if request.typed_run:
            metadata[TYPED_RUN] = "true"
        params = {
            **request_settings(self.view.responses if self.view is not None else None),
            "input": deepcopy(request.input),
            "conversation": self.conversation,
            "metadata": metadata,
            "stream": True,
            "background": True,
            "store": False,
        }
        try:
            stream = await self.client.create_response(params)
        except Exception as refused:
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
        previous = self._newest_reader.get(request.delegation_id)
        reader = asyncio.create_task(self._read(request, stream, previous), name="delegation-reader")
        self._newest_reader[request.delegation_id] = reader
        self.readers.add(reader)
        reader.add_done_callback(self.readers.discard)
        reader.add_done_callback(lambda done: self._reader_ended(request.delegation_id, done))
        if not request.outcome.done():
            request.outcome.set_result(request.delegation_id)
        return True

    def _reader_ended(self, delegation_id: str, reader: asyncio.Task) -> None:
        if self._newest_reader.get(delegation_id) is reader:
            del self._newest_reader[delegation_id]

    async def _read(self, request: Request, stream: AsyncIterator[dict], previous: asyncio.Task | None) -> None:
        """Hand each event of a request to the frontend, then relay it to the client, until its stream ends.

        It starts once ``delegate()`` returned and the delegation's ``previous`` reader ended, so events stay in order.
        """
        await request.returned.wait()
        if previous is not None and not previous.done():
            await asyncio.wait({previous})
        delegation_id = request.delegation_id
        announcing = request.starts
        ended, sequence = False, None
        try:
            async for event in stream:
                kind = event.get("type") if isinstance(event, dict) else None
                ended = ended or kind in TERMINALS
                if isinstance(event, dict) and isinstance(event.get("sequence_number"), int):
                    sequence = event["sequence_number"]
                if kind == "response.created" and isinstance(event.get("response"), dict):
                    self.composed = event["response"].get("instructions") or self.composed
                if not self._closing:
                    self._hand(delegation_id, event)
                announce = announcing and kind == "response.created"
                if announce:
                    announcing = False
                if self.relay is not None:
                    await self._relay(request, event, announce)
        except Exception:
            logger.exception(f"The stream of delegation {delegation_id} failed")
        if not ended and not self._closing:
            await self._lost(request, sequence)

    def _hand(self, delegation_id: str, event: dict) -> None:
        """Hand one event to the frontend; its failure must not stop the delegation's relay."""
        if self.listener is None:
            return
        try:
            self.listener.on_delegation_event(delegation_id, event)
        except Exception:
            logger.exception(f"The frontend could not take an event of delegation {delegation_id}")

    async def _relay(self, request: Request, event: dict, announce: bool) -> None:
        delegation_id = request.delegation_id
        try:
            if announce:
                response_id = event["response"]["id"]
                await self.relay.announce(delegation_id, response_id, request.offset_ms, request.client_event_id)
            await self.relay.backend_event(delegation_id, event, request.client_event_id)
        except Exception:
            logger.exception(f"An event of delegation {delegation_id} could not be relayed; it is skipped")

    async def _lost(self, request: Request, sequence: int | None) -> None:
        delegation_id = request.delegation_id
        logger.warning(f"The stream of delegation {delegation_id} ended before its response did")
        if self.listener is not None:
            try:
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
        """Apply a ``session.update`` patch and return the new settings; the core checks them in the background."""
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

    async def item_create(self, item: dict, client_event_id: str | None) -> None:
        """Take a result for a relayed call, or a typed user message, or refuse it at once (``ProtocolError``)."""
        self.routes.item(item)

    async def response_create(self, client_event_id: str | None) -> None:
        """Queue a continuation for each waiting response whose results the client completed, or a typed run.

        It returns once the requests are queued, so the core's answer never holds the command lock.
        """
        self._check_open()
        batch = []
        for continuation in self.routes.requests():
            self.routes.sending(continuation)
            request = Request(
                continuation.input,
                continuation.delegation_id or uid("item"),
                offset_ms=self._clock(),
                starts=continuation.delegation_id is None,
                typed_run=continuation.delegation_id is None,
                client_event_id=client_event_id,
            )
            # The client's own request: no frontend waits for its id.
            request.returned.set()
            # Settled in this command's context, so a refusal reaches the connection that sent the command.
            request.outcome.add_done_callback(partial(self._settled, continuation, request), context=copy_context())
            batch.append(request)
        self.sender.put(*batch)

    def _settled(self, continuation: Continuation, request: Request, outcome: asyncio.Future) -> None:
        """Settle a request of the client's own: drop or keep what it carries, and report a refusal to the client."""
        if not outcome.cancelled() and outcome.exception() is None:
            self.routes.sent(continuation)
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
