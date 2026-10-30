# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation: passes one ``responses`` session's delegations through to the delegation core.

:class:`ServerDelegation` serves the session (``live.ports.DelegationPart``), the frontend adapter
(``live.ports.DelegationPort``) and the Live client (through ``live.ports.DelegationView``) over the core's official
Responses API (``ResponsesClient``); a client-mode session gets ``live.client_delegation`` instead. It mints each
delegation's id, since the gateway mints every Live ID. Each request carries the session's ``delegation.responses``
settings as they are when it is sent (:func:`request_settings`), the session's conversation and ``metadata`` naming the
session and the delegation, and goes through one ordered sender (``live.server_delegation.sender``). A refused request
raises before any event: to the frontend for a new delegation, so its lead-in is never spoken, and to the client as an
``error`` with the refusal's code for a request of its own.

The only state it keeps is the session's conversation (created at start, so a core that cannot create it fails the
start, and deleted at close, which also stops its work), one reader per open stream, and the call routes and pending
client input (:class:`ClientRoutes`). A reader starts as its request's last step, so no event reaches the frontend
before the frontend has the delegation's id, and a delegation's readers run one after another, so a continuation's
events never overtake the response it continues. A stream that ends before its response's terminal event while the
session is open is lost: the frontend gets ``on_delegation_lost`` and the client a ``response.failed``. What a request
means, whether a result is right, timers and how a delegation ends belong to the core and the frontend. The UI bot
opens it with no view, so nothing is relayed to a client and no settings are sent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextvars import copy_context
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from loguru import logger

from live.protocol import BackendConfig, ProtocolError, SessionConfig, uid
from live.server_delegation.sender import Request, RequestSender

if TYPE_CHECKING:
    from live.ports import DelegationEvents, DelegationView
    from live.server_delegation.responses_client import ResponsesClient

Observe = Callable[..., None]
# The options a request carries only when the client set a value other than its default.
OPTIONS = ("parallel_tool_calls", "reasoning", "text", "service_tier")
# The metadata keys of the gateway-to-core convention: the live session, the delegation a request starts
# or continues, and the mark of a typed run (the client's typed messages, or nothing, run as a delegation of its own).
LIVE_SESSION_ID = "live_session_id"
LIVE_DELEGATION_ID = "live_delegation_id"
TYPED_RUN = "live_typed_run"
TERMINALS = frozenset({"response.completed", "response.failed", "response.incomplete"})
SETTINGS = "session.delegation.responses"  # where a setting the core refuses lives in the session's settings
MAX_OPEN_ROUTES = 512
MAX_PENDING_RESULTS = 512
MAX_QUEUED_ITEMS = 16
CONVERSATION_SECONDS = 10  # how long the core may take to open a session's conversation
# The close's limit: its core's close budget (the client's close_seconds: running work's grace, then the stop wait),
# plus the relay of the ends the core writes; CLOSE_SECONDS when the client does not say.
RELAY_SECONDS = 5.0
CLOSE_SECONDS = 10.0


def log_observation(kind: str, **fields) -> None:
    """Default observation hook: one debug log line per event."""
    logger.debug(f"live.{kind} {fields}")


def client_settings(config: SessionConfig | None) -> dict:
    """What a session's settings make each request carry: :func:`request_settings` of its ``delegation.responses``.

    A client delegation has no ``responses`` block, so it sets nothing.
    """
    return request_settings(config.delegation.responses if config is not None else None)


def request_settings(responses: BackendConfig | None) -> dict:
    """What the client set in the session's ``delegation.responses`` block, as the fields of a Responses request.

    The model, the tools and the tool choice always; the instructions and the output limit when set; the other options
    only when the client set a value other than its default. Without a ``responses`` block (the UI bot's server
    delegation has no session) it sets nothing, and the core's defaults apply.
    """
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


# The /history source of the instructions the core composed for the newest response.
COMPOSED_SOURCE = "the core: the newest request's instructions, as composed (instructions or the default, then notes)"


# ----------------------------------------------------------------------------------------- what the client is told
def sanitize_response_event(raw: dict) -> dict:
    """Return ``raw`` with the response snapshot redacted, as lifecycle events are.

    Lifecycle snapshots drop ``instructions``, ``input``, ``tools`` and ``output``; everything else stays as the core
    wrote it, ``previous_response_id`` included. Individual item events keep their completed function calls, so
    a client can collect every call of a round. Only a new top level and a new snapshot are built: the rest is the
    core's event, which the fan-out copies once when it sends it.
    """
    response = raw.get("response")
    if not isinstance(response, dict):
        return raw
    snapshot = {name: value for name, value in response.items() if name != "input"}
    snapshot.update(instructions=None, tools=[], output=[])
    return {**raw, "response": snapshot}


def lost_response(response_id: str, sequence_number: int | None) -> dict:
    """The ``response.failed`` that ends a response whose stream was lost: ``server_error``, after its last event."""
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
    """Tells the client what its delegations do: each delegation as it starts, and every event of its responses.

    It announces a delegation with ``session.delegation.created`` (:meth:`announce`), then wraps every event in
    ``response.event`` with the delegation's id, through the view's paced emit, so a burst waits for room in the primary
    connection's queue. It drops nothing beyond Live's snapshot redaction, so a client sees each response's one
    terminal; a response whose stream was lost ends with a synthesized ``response.failed`` (:meth:`lost`). Events of a
    request the client made itself carry its ``response.create``'s ``client_event_id``. The relay opens a call route
    for each relayed call the client answers (:data:`CALLS`) and, when a delegation's response ends, closes the routes
    of its earlier responses.
    """

    def __init__(self, view: DelegationView, routes: ClientRoutes):
        """Relay through the session's delegation ``view``; ``routes`` hears of each call and each end."""
        self.view, self.routes = view, routes
        self.responses: dict[str, str | None] = {}  # each delegation's open response, which its relayed calls name

    async def announce(
        self, delegation_id: str, response_id: str, offset_ms: int, client_event_id: str | None = None
    ) -> None:
        """Announce a delegation at the first response of the request that started it.

        ``offset_ms`` is where that request was made; ``client_event_id`` names the ``response.create``
        that made a typed run.
        """
        await self.view.emit_paced(
            "session.delegation.created",
            offset_ms=offset_ms,
            delegation={"id": delegation_id, "type": "delegation", "target": "responses", "response_id": response_id},
            client_event_id=client_event_id,
        )

    async def backend_event(self, delegation_id: str, event: dict, client_event_id: str | None = None) -> None:
        """Relay one event of a delegation to the client; route the calls it must answer.

        ``client_event_id`` names the ``response.create`` that made the request, for a request of the client's own.
        """
        kind = event.get("type", "")
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        if kind == "response.created":
            self.responses[delegation_id] = response.get("id")
        elif kind == "response.output_item.done" and event.get("item", {}).get("type") in CALLS:
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
        """The delegation's stream ended before its response did: end the response the client was told of, failed.

        Nothing is sent when the stream relayed no ``response.created``: the client was told of no response to end.
        """
        response_id = self.responses.get(delegation_id)
        if response_id is not None:
            await self.backend_event(delegation_id, lost_response(response_id, sequence_number), client_event_id)


# ---------------------------------------------------------------------------------------------- the client's side
# The calls a client answers, by the item type the core relays, and the field each call is known by.
CALLS = {
    "function_call": "call_id",
    "custom_tool_call": "call_id",
    "computer_call": "call_id",
    "shell_call": "call_id",
    "apply_patch_call": "call_id",
    "tool_search_call": "call_id",
    "mcp_approval_request": "id",
}
# The result items a client sends with response.item.create, and the field that names the call each answers.
RESULTS = {
    "function_call_output": "call_id",
    "custom_tool_call_output": "call_id",
    "computer_call_output": "call_id",
    "shell_call_output": "call_id",
    "apply_patch_call_output": "call_id",
    "tool_search_output": "call_id",
    "mcp_approval_response": "approval_request_id",
}


@dataclass(frozen=True)
class Route:
    """An open route: the delegation a relayed call belongs to, the response that relayed it, and the call."""

    delegation_id: str
    response_id: str | None
    call: str  # the call item's type, a key of CALLS
    name: str  # the called tool's name, when the call has one


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
    """The call routes and the pending client input, for one session's delegations.

    * The call routes. For each call the client answers (:data:`CALLS`) that was relayed to it, the delegation and
      the response it belongs to. A route opens when the call's item is relayed, and closes when a later response of
      its delegation ends, or at close. A route whose result was passed on stays, answered, so a repeat still reaches
      the core, which judges it. At most ``MAX_OPEN_ROUTES`` are open; the oldest is dropped first.
    * The pending client input: the results (:data:`RESULTS`) and typed messages received and not yet passed on. A
      result is taken for any call a route names, whatever its output, repeats and mismatched types included: the core
      judges them. They leave when the core accepts the request that carries them, and a result also when its route
      closes. While a queued request carries them they are in flight, and no other ``response.create`` sends them again;
      a refusal keeps them. At most ``MAX_PENDING_RESULTS`` results and ``MAX_QUEUED_ITEMS`` typed messages wait.

    It keeps no rule about a request beyond the official ones of ``response.create`` (:meth:`requests`).
    """

    def __init__(self, observe: Observe):
        """Start with nothing routed or pending; ``observe(kind, **fields)`` gets the traces."""
        self.observe = observe
        self.routes: dict[str, Route] = {}  # the open routes by call key, in the order their calls were relayed
        self.answered: set[str] = set()  # the routed calls whose results were passed on
        self.results: list[Result] = []  # the results not yet passed on, in the order they came
        self.messages: list[Typed] = []  # the typed messages not yet passed on, in order
        self.in_flight: set[Result] = set()  # the pending results a queued request carries now
        # The pending typed messages a queued request carries now. Several requests may be queued at once, and an
        # older one may be refused while a newer one is accepted, so each message is known by itself, not by its place.
        self.messages_in_flight: set[Typed] = set()

    # ------------------------------------------------------------------------------------------- the relayed calls
    def note_call(self, delegation_id: str, response_id: str | None, item: dict) -> None:
        """A call of the delegation's response ``response_id`` was relayed: its route opens."""
        key = item[CALLS[item["type"]]]
        self.routes.pop(key, None)  # a key relayed again routes to its newest call
        self.answered.discard(key)
        self.routes[key] = Route(delegation_id, response_id, item["type"], item.get("name", ""))
        while len(self.routes) > MAX_OPEN_ROUTES:
            self._close(next(iter(self.routes)))

    def response_ended(self, delegation_id: str, response_id: str | None) -> None:
        """A response of the delegation ended: the routes of its earlier responses close."""
        for key, route in list(self.routes.items()):
            if route.delegation_id == delegation_id and route.response_id != response_id:
                self._close(key)

    def _close(self, key: str) -> None:
        self.routes.pop(key, None)
        self.answered.discard(key)
        # with no route, a result can never be passed on; one a queued request carries goes with it
        self.results = [result for result in self.results if result.key != key or result in self.in_flight]

    def close(self) -> None:
        """The session closes: every route closes, and nothing is pending any more."""
        self.routes.clear()
        self.answered.clear()
        self.results.clear()
        self.messages.clear()
        self.in_flight.clear()
        self.messages_in_flight.clear()

    # ------------------------------------------------------------------------------------------ the client's items
    def item(self, item: dict) -> None:
        """Take a result or a typed user message from ``response.item.create``, or refuse it.

        Raises:
            ProtocolError: A result for a call no route names, a typed message that is not a user message of text or
                image parts, any other item type, or too many items waiting.
        """
        kind = item.get("type", "message")
        if kind in RESULTS:
            self._result(item, RESULTS[kind])
        elif kind == "message":
            self._message(item)
        else:
            raise ProtocolError("Unsupported response item", param="item.type")

    def _result(self, item: dict, field: str) -> None:
        key = item.get(field)
        route = self.routes.get(key) if isinstance(key, str) else None
        if route is None:
            raise ProtocolError("Unknown pending call", param=f"item.{field}")
        if len(self.results) >= MAX_PENDING_RESULTS:
            raise ProtocolError(f"Too many pending results: at most {MAX_PENDING_RESULTS}", "rate_limit_exceeded")
        self.results.append(Result(key, deepcopy(item)))
        self.observe("tool.result.received", call_id=key, name=route.name)

    def _message(self, item: dict) -> None:
        if item.get("role") != "user" or not isinstance(item.get("content"), list):
            raise ProtocolError("Expected a user message", param="item")
        parts = item["content"]
        if not parts or any(
            not isinstance(p, dict) or p.get("type") not in {"input_text", "input_image"} for p in parts
        ):
            raise ProtocolError("Unsupported message content", param="item.content")
        if len(self.messages) >= MAX_QUEUED_ITEMS:
            raise ProtocolError(f"Too many queued response items: at most {MAX_QUEUED_ITEMS}", "rate_limit_exceeded")
        self.messages.append(Typed(deepcopy(item)))
        text = "\n".join(p.get("text", "") for p in parts if p.get("type") == "input_text")
        self.observe("input.typed", text=text, chars=len(json.dumps(item)))

    # ---------------------------------------------------------------------------------- response.create
    def requests(self) -> list[Continuation]:
        """What a ``response.create`` sends, by GPT-Live's rule for a continuation.

        Every waiting response whose calls all have results is continued, one request each, the oldest first, with its
        results in call order; the newest also takes the typed messages, after its results. A response with only some
        of its results waits, with no error. With no response waiting, it is a typed run of the typed messages, which
        may be none. What a queued request already carries is not sent again.

        Raises:
            ProtocolError: It would continue nothing while results are owed: ``function_call_outputs_required`` when
                only function results are missing, ``tool_outputs_required`` otherwise. Nothing pending changes.
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
            only_functions = all(self.routes[key].call == "function_call" for key in missing)
            raise ProtocolError(
                f"Return the results of the pending calls before response.create: {', '.join(missing)}",
                "function_call_outputs_required" if only_functions else "tool_outputs_required",
            )
        if not continued:
            return [Continuation(None, [], messages)]
        continued[-1].messages = messages
        return continued

    def sending(self, continuation: Continuation) -> None:
        """A request that carries ``continuation`` is queued: what it carries is in flight until the core answers."""
        self.in_flight.update(continuation.results)
        self.messages_in_flight.update(continuation.messages)

    def release(self, continuation: Continuation) -> None:
        """The request was refused or not sent: what it carries stays pending, to be sent again, while routed."""
        self.in_flight.difference_update(continuation.results)
        self.messages_in_flight.difference_update(continuation.messages)
        self.results = [result for result in self.results if result.key in self.routes or result in self.in_flight]

    def sent(self, continuation: Continuation) -> None:
        """The core accepted the request: what it carries leaves the pending input, and its calls are answered."""
        carried, typed = set(continuation.results), set(continuation.messages)
        self.in_flight -= carried
        self.results = [result for result in self.results if result not in carried]
        self.answered.update(result.key for result in carried if result.key in self.routes)
        self.messages_in_flight -= typed
        self.messages = [message for message in self.messages if message not in typed]


# ----------------------------------------------------------------------------------------- the pass-through
class ServerDelegation:
    """Passes one session's delegations through to its delegation core, relays their events, answers the commands."""

    target: Literal["responses"] = "responses"  # where the work goes, as session.delegation.created names it
    reports_events = True  # each delegation's events follow (live.ports.DelegationPort)

    def __init__(self, client: ResponsesClient, *, observe: Observe = log_observation):
        """Reach the delegation core through ``client``; nothing is sent until :meth:`open`.

        Args:
            client: The session's delegation core, through the official Responses API; the composition root built it.
            observe: ``observe(kind, **fields)`` hook for the traces of the client's items.
        """
        self.client, self.observe = client, observe
        self.view: DelegationView | None = None  # the session's delegation view, from open; None for the UI bot
        self.conversation: str | None = None  # the session's conversation, from open until close
        self.routes: ClientRoutes | None = None  # the call routes and pending client input, from open
        self.relay: BackendEventRelay | None = None  # what the client is told, from open with a view
        self.listener: DelegationEvents | None = None  # the frontend, from listen
        self.sender = RequestSender(self._send)  # every request, in the order it was made, off the command lock
        self.checks: set[asyncio.Task] = set()  # the background checks of the session's settings
        self.readers: set[asyncio.Task] = set()  # one reader per open stream
        self._newest_reader: dict[str, asyncio.Task] = {}  # each delegation's newest reader, which its next one follows
        # The instructions of the newest response the core opened, as it composed them (the response object's own
        # field): /history shows them; the client never gets them (lifecycle snapshots are redacted).
        self.composed: str | None = None
        self._closing = False  # from close step 2: no new request, nothing more for the frontend
        self._opened = self._closed = False

    # ------------------------------------------------------------------------------ life cycle
    async def open(self, view: DelegationView | None) -> None:
        """Open the session's conversation with the core, then build the call routes and the relay.

        ``view`` is the session's delegation view; ``None`` (the UI bot) means that no Live client listens, so nothing
        is relayed and no settings are sent. A conversation the core cannot create, or does not create within
        ``CONVERSATION_SECONDS``, raises, which fails the session's start. Safe to call twice; a server delegation that
        was closed opens nothing.
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
        if self._closed:  # closed while the core created it: nothing else would delete it
            await self.client.delete_conversation(conversation["id"])
            return
        self.conversation = conversation["id"]
        self.routes = ClientRoutes(self.observe)
        if view is not None:
            self.relay = BackendEventRelay(view, self.routes)
        self.sender.start()

    def stop_handing(self) -> None:
        """Close step 2: send no new request, and hand nothing more to the frontend.

        The readers still relay to the client what the core writes, so the ends the core writes at close step 4 reach
        it before ``session.closed``.
        """
        self._closing = True

    async def close(self) -> None:
        """Close step 4: send nothing more, delete the conversation, let the readers end, then close the call routes.

        Every request not yet accepted is refused. Deleting the conversation stops its work in the core, which ends
        every response it opened, so each reader relays what its stream still holds and ends. Readers still
        running when the delete fails, or when the session's close limit cuts the close short, are stopped. Safe
        twice, and before, during or after :meth:`open`.
        """
        if self._closed:
            return
        self._closed = self._closing = True
        conversation, self.conversation = self.conversation, None
        for check in list(self.checks):  # a check still running has nothing left to report to
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
        """The session's acknowledgement delay, read when asked; 0 (at once) without a live session (the UI bot)."""
        return self.view.acknowledgement_delay_ms if self.view is not None else 0

    # ------------------------------------------------------------------------------ the frontend
    async def delegate(self, input: list[dict]) -> str:
        """Delegate a request and return its delegation id once the core accepted it; a refusal raises.

        The request takes its place in the session's order now, where ``offset_ms`` is read; the sender sends it in
        turn. A caller that gives up before it is sent cancels it; one that gives up later still has its events read, so
        no accepted work is left unread. No event reaches the frontend before this returns.

        Raises:
            RuntimeError: The part is not open, is closing, or has no listener yet (``listen()`` comes first).
            ProtocolError: The core refused the request with a code, or as invalid, before any event.
            Exception: The core refused it otherwise (a closing session, a full queue), before any event.
        """
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
        """The session's input clock now; 0 without a live session (the UI bot)."""
        return self.view.input_ms if self.view is not None else 0

    def _check_open(self) -> None:
        """Each request names the session's conversation, which exists from open until close; none from close step 2."""
        if self._closing:
            raise RuntimeError("Session is closing")
        if self.conversation is None:
            raise RuntimeError("The server delegation is not open")

    def listen(self, listener: DelegationEvents) -> None:
        """Hand every event of every delegation, typed runs included, to ``listener``."""
        self.listener = listener

    async def _send(self, request: Request) -> bool:
        """Send one request (the sender's step), settle its outcome, and say whether the core accepted it.

        The request carries the session's ``delegation.responses`` settings as they are now, its conversation and
        its metadata. A refusal comes before any event: a code, or an invalid request (a ``ValueError``, the
        ``ResponsesClient`` port's), makes it the ``ProtocolError`` the request's maker gets. An accepted request
        always gets its reader, also when its caller gave up meanwhile. From close step 2 a request still queued is
        refused, not sent.
        """
        if self.conversation is None or self._closing:
            if not request.outcome.done():
                request.outcome.set_exception(RuntimeError("Session is closing"))
            return False
        metadata = {LIVE_DELEGATION_ID: request.delegation_id}
        if self.view is not None:
            metadata = {LIVE_SESSION_ID: self.view.id, **metadata}
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
            elif isinstance(refused, ValueError):  # the port's refusal of an invalid request: its maker's mistake
                refusal = ProtocolError(str(refused))
            else:
                refusal = refused
            if request.outcome.done():  # its caller gave up while it was sent
                logger.info(f"The core refused delegation {request.delegation_id}, whose caller had gone: {refused}")
            else:
                request.outcome.set_exception(refusal)
            return False
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

        It starts once the frontend has the delegation's id (``delegate()`` returned) and the delegation's
        ``previous`` reader has ended, so over a network a continuation's stream cannot overtake the end of the
        response it continues. From close step 2 the frontend gets nothing more, and the client still gets every event,
        the ends the core writes at close included. A stream that ends, or fails, before its response's terminal while
        the session is open is lost (:meth:`_lost`); at close it is not.
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
                announce = announcing and kind == "response.created"  # the stream's one response opens
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
        """Tell the client of one event, after announcing the delegation when ``announce``; a bad event is skipped.

        The events of a request the client made itself carry its ``response.create``'s ``client_event_id``.
        """
        delegation_id = request.delegation_id
        try:
            if announce:
                response_id = event["response"]["id"]
                await self.relay.announce(delegation_id, response_id, request.offset_ms, request.client_event_id)
            await self.relay.backend_event(delegation_id, event, request.client_event_id)
        except Exception:
            logger.exception(f"An event of delegation {delegation_id} could not be relayed; it is skipped")

    async def _lost(self, request: Request, sequence: int | None) -> None:
        """A stream ended before its response did: the frontend hears of it, then the client's response ends failed."""
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

    # --------------------------------------------------------------------- the session's commands
    def knows(self, delegation_id: object) -> bool:
        """No append may name a delegation in ``responses`` mode: the core reports its own work."""
        return False

    async def update(self, config: SessionConfig, patch: dict, client_event_id: str | None) -> SessionConfig:
        """Apply a ``session.update`` patch by the Responses delegation's rule; the core checks the result afterwards.

        The new settings are returned at once, for the session to store and answer with ``session.updated``. The core
        is then asked, in the background, whether it can honor them (:meth:`_check`).

        Raises:
            ProtocolError: The patch is refused.
        """
        updated = config.updated(patch)
        self._check(updated.delegation.responses, client_event_id)
        return updated

    def _check(self, responses: BackendConfig, client_event_id: str | None) -> None:
        """Ask the core whether it can honor ``responses``, running nothing (``POST /v1/responses/input_tokens``).

        It runs in the background, so settings never hold up the session's commands or audio. A setting the core
        refuses comes back as a nonfatal ``responses_validation_failed`` with its ``param`` and the settings event's
        ``client_event_id``; the setting is kept. A core that cannot answer is logged, and nothing is reported: the
        settings may be right.
        """
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

        :meth:`ClientRoutes.requests` says what goes; one that would continue nothing while results are owed raises
        the official code. It returns once the requests are queued, so the core's answer never holds the session's
        command lock. The sender sends them in turn, oldest response first; each that the core accepts leaves
        the pending input. The first refusal is reported to the client with the refusal's code (``invalid_value`` for a
        request the core refuses as invalid, such as a repeated result) and this command's ``client_event_id``, and it
        and the requests after it stay pending.

        Raises:
            ProtocolError: It would continue nothing while results are owed.
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
            request.returned.set()  # the client's own request: no frontend waits for its id
            # Settled in this command's context, so a refusal reaches the connection that sent the command.
            request.outcome.add_done_callback(partial(self._settled, continuation, request), context=copy_context())
            batch.append(request)
        self.sender.put(*batch)

    def _settled(self, continuation: Continuation, request: Request, outcome: asyncio.Future) -> None:
        """A request of the client's own was answered: what it carries leaves the pending input, or stays.

        A refusal reaches the client as the client's mistake (a ``ProtocolError``, :meth:`_send`) when the core refused
        the request with a code, or as an invalid request: a repeated result, a result of another type than its call,
        results for several responses at once. Any other failure is the server's, said without detail.
        """
        if not outcome.cancelled() and outcome.exception() is None:
            self.routes.sent(continuation)
            return
        self.routes.release(continuation)
        if outcome.cancelled() or self.view is None or self._closing:  # not sent, or the session is closing
            return
        refusal = outcome.exception()
        if isinstance(refusal, ProtocolError):
            self.view.report_refusal(str(refusal), refusal.code, request.client_event_id)
        else:
            logger.warning(f"The core refused a request of the client's own: {refusal}")
            self.view.report_refusal("The request could not be sent.", "server_error", request.client_event_id)

    async def history(self) -> dict | None:
        """The part's half of ``/history``: what each request carries, and the conversation's items in the core.

        ``model`` and the first ``prompt`` part are what the client set; the core fills what it left out with its
        defaults, which the gateway learns only from the core's answers. Once the core opened a response, the last part
        is that newest response's instructions as the core composed them (the request's or the default, then the core's
        notes). ``items`` is the backend's history, none before the open or after the close.
        """
        settings = request_settings(self.view.responses if self.view is not None else None)
        items = []
        if self.conversation is not None:
            items = (await self.client.list_conversation_items(self.conversation))["data"]
        instructions = settings.get("instructions")
        prompt = [{"source": "delegation.responses.instructions", "text": instructions}] if instructions else []
        if self.composed:
            prompt.append({"source": COMPOSED_SOURCE, "text": self.composed})
        return {"model": settings.get("model"), "prompt": prompt, "items": items}
