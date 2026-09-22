# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""A framework-neutral OpenAI Realtime facade for VoiceClaw.

The facade is the only browser-facing realtime socket.  It forwards ordinary
Realtime events to a configured speech frontend, while keeping server-owned
tools, backend credentials, and upstream identifiers behind the service
boundary.  UI projections are ordinary, out-of-band Realtime responses.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import inspect
import json
import logging
import math
import secrets
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

from voiceclaw.application.context import FrontendInstructionBuilder
from voiceclaw.domain.models import FrontendActivity, InputActivity, ModelActivity, OutputActivity
from voiceclaw.domain.response_only import (
    ResponseOnlyRequestState,
    ResponseOnlyResultEventKind,
    ResponseOnlyUpdateKind,
)
from voiceclaw.interaction_profiles import InteractionProfile, load_interaction_profile_catalog
from voiceclaw.model_contracts import (
    FailureCopy,
    ModelContractCatalog,
    ModelContractError,
    load_model_contract_catalog,
)
from voiceclaw.ports.runtime import (
    FrontendConversationDeliveryState,
    FrontendConversationTurn,
    FrontendPlaybackReceipt,
    FrontendPlaybackReceiptState,
    FrontendResponse,
    FrontendResponsePurpose,
    InteractionUpdate,
    RealtimeSessionRuntimePort,
    SessionSnapshot,
    TurnDirectiveKind,
)
from voiceclaw.ports.turns import MAX_COMMITTED_TURN_GOAL_BYTES
from voiceclaw.realtime.events import (
    Projection,
    ProjectionEventFactory,
    ProjectionEventStream,
    ProjectionStreamAbortStatus,
)
from voiceclaw.realtime.tools import ParsedToolCall, ProtectedTool, VoiceClawToolRegistry

_DEFAULT_MAX_EVENT_BYTES = 4 * 1024 * 1024
_MAX_CONFIGURED_EVENT_BYTES = 16 * 1024 * 1024
_MAX_IDENTIFIER_CHARACTERS = 512
_MAX_INSTRUCTIONS_CHARACTERS = 64_000
_DEFAULT_CONTEXT_CHARACTER_BUDGET = 16_000
_MAX_CONFIGURED_CONTEXT_CHARACTER_BUDGET = 64_000
_MAX_OUTPUT_CHARACTERS = 128_000
_MAX_REALTIME_RESPONSE_TOKENS = 4_096
_MAX_PROTECTED_TASKS = 4
_DEFAULT_MAX_PENDING_SPEECH = 32
_MAX_CONFIGURED_PENDING_SPEECH = 1024
_MAX_TRACKED_RESPONSES = 32
_MAX_TRACKED_ITEMS = 512
_MAX_TRACKED_CALLS = 128
_MAX_BUFFERED_FUNCTION_ITEMS = 64
_MAX_BUFFERED_FUNCTION_EVENTS_PER_ITEM = 4
_DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 15.0
_FINALIZED_AUDIO_TURN_TIMEOUT_SECONDS = 15.0
_DEFAULT_PCM_SAMPLE_RATE = 24_000
_PCM16_BYTES_PER_SAMPLE = 2
_LOGGER = logging.getLogger(__name__)
_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "client_secret",
        "credential",
        "deployment_bearer",
        "grant",
        "password",
        "secret",
        "token",
    }
)


class RealtimeTransport(Protocol):
    """Minimal text-message transport implemented by WebSocket adapters."""

    async def receive(self) -> str | bytes:
        """Receive one complete text message from the peer."""
        ...

    async def send(self, message: str) -> None:
        """Send one complete text message to the peer."""
        ...


class ActivityDirection(StrEnum):
    """An exact observation point at the facade transport boundary."""

    DOWNSTREAM_RECEIVED = "downstream.received"
    UPSTREAM_SENT = "upstream.sent"
    UPSTREAM_RECEIVED = "upstream.received"
    DOWNSTREAM_SENT = "downstream.sent"


@dataclass(frozen=True, slots=True)
class RealtimeActivity:
    """A credential-free event observation for metrics and diagnostics."""

    direction: ActivityDirection
    event_type: str


class ActivityObserver(Protocol):
    """Observe accepted events without receiving their payloads."""

    def __call__(self, activity: RealtimeActivity) -> Awaitable[None] | None:
        """Record one exact boundary activity."""
        ...


class FacadeProtocolError(RuntimeError):
    """A safe protocol failure suitable for terminating one facade session."""

    def __init__(self, code: str, message: str) -> None:
        """Store only a bounded public code and message."""
        self.code = code
        self.public_message = message
        super().__init__(code)


class _Owner(StrEnum):
    CLIENT = "client"
    SERVER = "server"


@dataclass(slots=True)
class _ItemRecord:
    local_id: str
    response_id: str | None = None
    call_id: str | None = None
    owner: _Owner | None = None
    completed: bool = False


@dataclass(slots=True)
class _CallRecord:
    local_id: str
    item_id: str
    response_id: str
    owner: _Owner
    completed: bool = False
    executed: bool = False
    output_sent: bool = False


class _ResponsePurpose(StrEnum):
    """Internal response classes used by the single upstream arbiter."""

    ACKNOWLEDGEMENT = "acknowledgement"
    DIRECT_REPLY = "direct_reply"
    INTERACTIVE = "interactive"
    DELIVERY = "delivery"


_SERVER_SPEECH_PURPOSES = frozenset(
    {
        _ResponsePurpose.ACKNOWLEDGEMENT,
        _ResponsePurpose.DIRECT_REPLY,
        _ResponsePurpose.DELIVERY,
    }
)
_APPLICATION_SPEECH_PURPOSES = frozenset(purpose.value for purpose in FrontendResponsePurpose)


@dataclass(slots=True)
class _DirectReplyTemplate:
    """A response body plus the original untrusted client instruction text."""

    event: dict[str, Any]
    client_response_instructions: str


@dataclass(slots=True)
class _QueuedResponseCreate:
    """One wire-level Realtime response request awaiting the single-response slot.

    This is protocol arbitration, not the semantic Work/result/presentation
    queue owned by the Interaction Manager.
    """

    event: dict[str, Any]
    response_template: dict[str, Any]
    purpose: _ResponsePurpose
    priority: int
    requires_speech_floor: bool
    speech_purpose: str | None = None
    local_request_id: str | None = None
    presentation_id: str | None = None
    playback_receipt_id: str | None = None
    client_response_instructions: str = ""
    server_response_context: Mapping[str, str] | None = None
    finalized_user_text: str | None = None
    finalized_user_item_id: str | None = None
    finalized_item_id: str | None = None
    awaits_manual_audio_item: bool = False
    awaits_finalized_user_text: bool = False
    routing_applied: bool = False
    direct_route: bool = False
    expected_protected_tool: ProtectedTool | None = None
    model_route_selection: bool = False
    direct_reply: _DirectReplyTemplate | None = None
    prerequisite_item_creates: list[_DeferredClientItemCreate] = field(default_factory=list)
    sent_upstream: bool = False


@dataclass(slots=True)
class _PlaybackLease:
    """One browser-playback lease held after frontend audio generation."""

    response_id: str
    public_response_id: str
    presentation_id: str
    receipt_id: str
    local_request_id: str | None
    speech_purpose: str | None
    sample_rate: int
    item_id: str | None = None
    public_item_id: str | None = None
    content_index: int | None = None
    generated_samples: int = 0
    audio_done: bool = False
    response_done: bool = False
    transcript: str = ""
    pending_receipt_id: str | None = None
    terminal: bool = False

    @property
    def audio_end_ms(self) -> int:
        """Return the exact complete-frame duration exposed on the wire."""
        return (self.generated_samples * 1000) // self.sample_rate


@dataclass(frozen=True, slots=True)
class _PendingPlaybackReceipt:
    """A validated standard truncation awaiting upstream acknowledgement."""

    receipt_id: str
    response_id: str
    presentation_id: str
    public_item_id: str
    local_request_id: str | None
    heard_through_ms: int
    audio_end_ms: int
    fully_played: bool


@dataclass(slots=True)
class _DeferredClientItemCreate:
    """A client item held behind an earlier, not-yet-sent response request."""

    event: dict[str, Any]
    event_id: str
    item_id: str


@dataclass(slots=True)
class _PreparedResponseCreate:
    """A response request plus its response-local finalized-turn binding."""

    event: dict[str, Any]
    finalized_user_text: str | None
    finalized_user_item_id: str | None
    has_response_local_input: bool
    client_response_instructions: str


@dataclass(slots=True)
class _ManualAudioCommit:
    """One client commit paired by order with its next interactive response."""

    event_id: str
    input_generation: int | None = None
    finalized_item_id: str | None = None
    finalized_user_text: str | None = None
    response: _QueuedResponseCreate | None = None
    rejected: bool = False
    rejection_code: str = "upstream_error"
    rejection_message: str = "The realtime frontend rejected the associated audio commit."


class _OwnershipRegistry:
    """Map private upstream identities to facade identities and enforce ownership."""

    def __init__(self, id_factory: Callable[[str], str]) -> None:
        self._id_factory = id_factory
        self._responses: dict[str, str] = {}
        self._local_responses: dict[str, str] = {}
        self._retired_local_responses: deque[str] = deque()
        self._retired_local_response_set: set[str] = set()
        self._response_done: set[str] = set()
        self._items: dict[str, _ItemRecord] = {}
        self._local_items: dict[str, str] = {}
        self._hidden_public_predecessors: dict[str, str | None] = {}
        self._output_items_added: set[str] = set()
        self._calls: dict[str, _CallRecord] = {}
        self._local_calls: dict[str, str] = {}

    def declare_response(self, upstream_id: object) -> str:
        value = _identifier(upstream_id, "upstream response id")
        if value in self._responses:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend reused a response id.")
        if len(self._responses) >= _MAX_TRACKED_RESPONSES:
            raise FacadeProtocolError(
                "session_capacity_exceeded", "The realtime session has too many tracked responses."
            )
        local = self._id_factory("resp_vc")
        self._responses[value] = local
        self._local_responses[local] = value
        return local

    def response(self, upstream_id: object) -> str:
        value = _identifier(upstream_id, "upstream response id")
        try:
            return self._responses[value]
        except KeyError as error:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend referenced an unknown response."
            ) from error

    def upstream_response(self, local_id: object) -> str:
        value = _identifier(local_id, "response id")
        try:
            return self._local_responses[value]
        except KeyError as error:
            raise FacadeProtocolError("invalid_request", "The response id is not owned by this session.") from error

    def cancel_target(self, local_id: object) -> str | None:
        """Resolve an active response, treating a just-terminal cancel as idempotent.

        ``response.done`` and a browser barge-in cross opposite WebSocket
        directions, so a cancel can legitimately arrive just after the
        terminal response retired its private alias.  Retain only a bounded
        set of facade-owned terminal ids so that race is a no-op without
        weakening ownership checks for arbitrary ids.
        """
        value = _identifier(local_id, "response id")
        if value in self._retired_local_response_set:
            return None
        upstream = self._local_responses.get(value)
        if upstream is not None:
            return upstream
        raise FacadeProtocolError("invalid_request", "The response id is not owned by this session.")

    def complete_response(self, upstream_id: object) -> None:
        value = _identifier(upstream_id, "upstream response id")
        local_response_id = self.response(value)
        if value in self._response_done:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend completed a response twice.")
        self._response_done.add(value)
        self._remember_terminal_response(local_response_id)

    def _remember_terminal_response(self, local_response_id: str) -> None:
        if local_response_id in self._retired_local_response_set:
            return
        self._retired_local_responses.append(local_response_id)
        self._retired_local_response_set.add(local_response_id)
        while len(self._retired_local_responses) > _MAX_TRACKED_RESPONSES:
            expired = self._retired_local_responses.popleft()
            self._retired_local_response_set.discard(expired)

    def ensure_item(self, upstream_id: object) -> _ItemRecord:
        value = _identifier(upstream_id, "upstream item id")
        record = self._items.get(value)
        if record is None:
            if value in self._hidden_public_predecessors:
                raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend reused a hidden item id.")
            if len(self._items) >= _MAX_TRACKED_ITEMS:
                raise FacadeProtocolError(
                    "session_capacity_exceeded", "The realtime session has too many tracked items."
                )
            local = self._id_factory("item_vc")
            record = _ItemRecord(local_id=local)
            self._items[value] = record
            self._local_items[local] = value
        return record

    def add_output_item(self, upstream_id: object, response_id: object) -> _ItemRecord:
        item_id = _identifier(upstream_id, "upstream item id")
        response = _identifier(response_id, "upstream response id")
        self.response(response)
        if item_id in self._output_items_added:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend added an output item twice.")
        record = self.ensure_item(item_id)
        if record.response_id not in {None, response}:
            raise FacadeProtocolError("upstream_protocol_error", "An output item changed response ownership.")
        record.response_id = response
        self._output_items_added.add(item_id)
        return record

    def item(self, upstream_id: object) -> _ItemRecord:
        value = _identifier(upstream_id, "upstream item id")
        try:
            return self._items[value]
        except KeyError as error:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend referenced an unknown item."
            ) from error

    def upstream_item(self, local_id: object) -> str:
        value = _identifier(local_id, "item id")
        try:
            return self._local_items[value]
        except KeyError as error:
            raise FacadeProtocolError("invalid_request", "The item id is not owned by this session.") from error

    def upstream_client_item(self, local_id: object) -> str:
        upstream = self.upstream_item(local_id)
        if self._items[upstream].owner is _Owner.SERVER:
            raise FacadeProtocolError("invalid_request", "The item is server-owned.")
        return upstream

    def register_client_item(self, local_id: object) -> None:
        value = _identifier(local_id, "item id")
        if value in self._local_items or value in self._hidden_public_predecessors:
            raise FacadeProtocolError("invalid_request", "The client reused an item id.")
        if len(self._items) >= _MAX_TRACKED_ITEMS:
            raise FacadeProtocolError("session_capacity_exceeded", "The realtime session has too many tracked items.")
        self._local_items[value] = value
        self._items[value] = _ItemRecord(local_id=value)

    def hide_item(self, upstream_id: object, previous_upstream_id: object) -> None:
        """Retain a bounded public predecessor for one server-hidden item.

        Realtime frontend responses append their assistant message after the
        server-owned function output. That output and its protected call are
        intentionally absent from the browser conversation, so their private
        identifiers must collapse to the nearest browser-visible predecessor.
        """
        item_id = _identifier(upstream_id, "upstream item id")
        self.item(item_id)
        if previous_upstream_id is None:
            public_predecessor = None
        else:
            previous_item_id = _identifier(previous_upstream_id, "upstream previous item id")
            if previous_item_id == item_id:
                raise FacadeProtocolError("upstream_protocol_error", "A hidden item cannot be its own predecessor.")
            if previous_item_id in self._hidden_public_predecessors:
                public_predecessor = self._hidden_public_predecessors[previous_item_id]
            else:
                public_predecessor = self.item(previous_item_id).local_id
        if item_id in self._hidden_public_predecessors:
            if self._hidden_public_predecessors[item_id] != public_predecessor:
                raise FacadeProtocolError("upstream_protocol_error", "A hidden item changed its public predecessor.")
            return
        if len(self._hidden_public_predecessors) >= _MAX_TRACKED_ITEMS:
            raise FacadeProtocolError(
                "session_capacity_exceeded", "The realtime session has too many hidden conversation items."
            )
        self._hidden_public_predecessors[item_id] = public_predecessor

    def public_predecessor(self, upstream_id: object) -> str | None:
        """Resolve an upstream predecessor without exposing hidden aliases."""
        item_id = _identifier(upstream_id, "upstream previous item id")
        if item_id in self._hidden_public_predecessors:
            public_predecessor = self._hidden_public_predecessors[item_id]
            if public_predecessor is not None and public_predecessor not in self._local_items:
                raise FacadeProtocolError(
                    "upstream_protocol_error", "A hidden item referenced a deleted public predecessor."
                )
            return public_predecessor
        return self.item(item_id).local_id

    def declare_call(
        self,
        upstream_call_id: object,
        *,
        item_id: object,
        response_id: object,
        owner: _Owner,
    ) -> _CallRecord:
        call_id = _identifier(upstream_call_id, "upstream call id")
        upstream_item_id = _identifier(item_id, "upstream item id")
        upstream_response_id = _identifier(response_id, "upstream response id")
        if call_id in self._calls:
            existing = self._calls[call_id]
            if (
                existing.item_id != upstream_item_id
                or existing.response_id != upstream_response_id
                or existing.owner is not owner
            ):
                raise FacadeProtocolError("upstream_protocol_error", "A function call changed ownership.")
            return existing
        if len(self._calls) >= _MAX_TRACKED_CALLS:
            raise FacadeProtocolError(
                "session_capacity_exceeded", "The realtime session has too many tracked function calls."
            )
        local = self._id_factory("call_vc")
        record = _CallRecord(
            local_id=local,
            item_id=upstream_item_id,
            response_id=upstream_response_id,
            owner=owner,
        )
        self._calls[call_id] = record
        self._local_calls[local] = call_id
        item = self.item(upstream_item_id)
        item.call_id = call_id
        item.owner = owner
        return record

    def call(self, upstream_call_id: object) -> _CallRecord:
        value = _identifier(upstream_call_id, "upstream call id")
        try:
            return self._calls[value]
        except KeyError as error:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend referenced an unknown function call."
            ) from error

    def upstream_call(self, local_call_id: object) -> tuple[str, _CallRecord]:
        value = _identifier(local_call_id, "call id")
        try:
            upstream = self._local_calls[value]
        except KeyError as error:
            raise FacadeProtocolError("invalid_request", "The function call is not owned by the client.") from error
        return upstream, self._calls[upstream]

    def complete_call(self, upstream_call_id: object) -> _CallRecord:
        record = self.call(upstream_call_id)
        if record.completed:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend completed a call twice.")
        record.completed = True
        return record

    def claim_server_call_output(self, upstream_call_id: object) -> _CallRecord | None:
        """Atomically reserve the sole function output allowed for a protected call."""
        record = self.call(upstream_call_id)
        if record.owner is not _Owner.SERVER:
            raise FacadeProtocolError("upstream_protocol_error", "A client function call changed ownership.")
        if record.output_sent:
            return None
        # Claim before the transport await.  A failed or ambiguous send must
        # never be retried as a second function_call_output for the same call.
        record.output_sent = True
        return record

    def protected_calls_for_response(self, upstream_response_id: str) -> tuple[tuple[str, _CallRecord], ...]:
        return tuple(
            (call_id, record)
            for call_id, record in self._calls.items()
            if record.response_id == upstream_response_id and record.owner is _Owner.SERVER and record.completed
        )

    def is_primary_protected_call(self, upstream_response_id: str, upstream_call_id: str) -> bool:
        """Return whether this is the first protected action declared by a response."""
        response_id = _identifier(upstream_response_id, "upstream response id")
        call_id = _identifier(upstream_call_id, "upstream call id")
        self.response(response_id)
        self.call(call_id)
        return next(
            (
                candidate_call_id == call_id
                for candidate_call_id, record in self._calls.items()
                if record.response_id == response_id and record.owner is _Owner.SERVER
            ),
            False,
        )

    def retire_response(self, upstream_response_id: object) -> None:
        """Release terminal response state and its private server-owned objects."""
        response_id = _identifier(upstream_response_id, "upstream response id")
        local_response_id = self._responses.pop(response_id, None)
        if local_response_id is not None:
            self._local_responses.pop(local_response_id, None)
        self._response_done.discard(response_id)

        for item_id, record in tuple(self._items.items()):
            if record.response_id != response_id:
                continue
            if record.owner is _Owner.CLIENT and record.call_id is not None:
                record.response_id = None
            elif record.owner is _Owner.SERVER and record.call_id is not None:
                # Keep the call alias until the upstream acknowledges the
                # server-created function_call_output item. The originating
                # response and function-call item can retire independently.
                self._output_items_added.discard(item_id)
                self._items.pop(item_id, None)
                self._local_items.pop(record.local_id, None)
            else:
                # Response completion does not delete public conversation
                # items. A later user/audio item can name this assistant item
                # as its predecessor, so keep the bounded alias until an
                # explicit deletion or session teardown.
                record.response_id = None

    def retire_item(self, upstream_item_id: object) -> None:
        """Release one item alias and any server-owned call attached to it."""
        item_id = _identifier(upstream_item_id, "upstream item id")
        record = self._items.pop(item_id, None)
        self._output_items_added.discard(item_id)
        if record is None:
            return
        self._local_items.pop(record.local_id, None)
        if record.call_id is not None:
            call = self._calls.get(record.call_id)
            if call is not None and call.owner is _Owner.SERVER:
                self._remove_call(record.call_id)

    def retire_terminal_item(self, upstream_item_id: object) -> None:
        """Release hidden response state while retaining public item aliases."""
        item_id = _identifier(upstream_item_id, "upstream item id")
        record = self._items.get(item_id)
        if record is None:
            return
        if item_id not in self._hidden_public_predecessors:
            record.response_id = None
            return
        self.retire_item(item_id)

    def retire_client_call(self, upstream_call_id: object) -> None:
        """Release a client-owned call after its output item reaches terminal state."""
        call_id = _identifier(upstream_call_id, "upstream call id")
        record = self._calls.get(call_id)
        if record is None or record.owner is not _Owner.CLIENT:
            return
        self._remove_call(call_id)
        item = self._items.get(record.item_id)
        if item is not None:
            item.response_id = None

    def retire_server_call(self, upstream_call_id: object, output_item_id: object) -> None:
        """Release a protected call after its hidden output item is terminal."""
        call_id = _identifier(upstream_call_id, "upstream call id")
        record = self.call(call_id)
        if record.owner is not _Owner.SERVER:
            raise FacadeProtocolError("upstream_protocol_error", "A client function call changed ownership.")
        self._remove_call(call_id)
        self.retire_item(record.item_id)
        self.retire_item(output_item_id)

    def clear(self) -> None:
        """Release all aliases when the facade session terminates."""
        self._responses.clear()
        self._local_responses.clear()
        self._retired_local_responses.clear()
        self._retired_local_response_set.clear()
        self._response_done.clear()
        self._items.clear()
        self._local_items.clear()
        self._hidden_public_predecessors.clear()
        self._output_items_added.clear()
        self._calls.clear()
        self._local_calls.clear()

    def _remove_call(self, upstream_call_id: str) -> None:
        record = self._calls.pop(upstream_call_id, None)
        if record is not None:
            self._local_calls.pop(record.local_id, None)


def _identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_IDENTIFIER_CHARACTERS
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise FacadeProtocolError("upstream_protocol_error", f"The {name} is invalid.")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise FacadeProtocolError("invalid_json", "Realtime events must not contain duplicate JSON keys.")
        value[key] = item
    return value


def _decode_event(message: str | bytes, *, maximum_bytes: int) -> dict[str, Any]:
    if not isinstance(message, (str, bytes)):
        raise FacadeProtocolError("invalid_json", "Realtime messages must be JSON text.")
    try:
        raw = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    except (UnicodeEncodeError, ValueError) as error:
        raise FacadeProtocolError("invalid_json", "Realtime messages must be valid UTF-8 JSON.") from error
    if not raw or len(raw) > maximum_bytes:
        raise FacadeProtocolError("event_too_large", "The realtime event exceeds the configured size limit.")
    try:
        event = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except FacadeProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise FacadeProtocolError("invalid_json", "Realtime messages must be valid JSON objects.") from error
    if not isinstance(event, dict):
        raise FacadeProtocolError("invalid_event", "Realtime events must be JSON objects.")
    event_type = event.get("type")
    if not isinstance(event_type, str) or not event_type or len(event_type) > 128:
        raise FacadeProtocolError("invalid_event", "Realtime events require a valid type.")
    return event


def _encode_event(event: Mapping[str, Any], *, maximum_bytes: int) -> str:
    try:
        message = json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        size = len(message.encode("utf-8"))
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as error:
        raise FacadeProtocolError("invalid_event", "A realtime event could not be serialized.") from error
    if size > maximum_bytes:
        raise FacadeProtocolError("event_too_large", "The realtime event exceeds the configured size limit.")
    return message


def _strip_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_sensitive(item)
            for key, item in value.items()
            if isinstance(key, str) and key.lower() not in _SENSITIVE_KEYS
        }
    if isinstance(value, list):
        return [_strip_sensitive(item) for item in value]
    return copy.deepcopy(value)


def _merge_object_patch(base: Mapping[str, Any], patch: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in patch.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _merge_object_patch(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _reserved_metadata_key(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.lower()
    return normalized == "voiceclaw" or normalized.startswith(("voiceclaw_", "voiceclaw."))


def _has_reserved_metadata(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "metadata" and isinstance(item, dict) and any(_reserved_metadata_key(name) for name in item):
                return True
            if _has_reserved_metadata(item):
                return True
    elif isinstance(value, list):
        return any(_has_reserved_metadata(item) for item in value)
    return False


class VoiceClawRealtimeFacade:
    """Bridge one browser Realtime connection to one configured speech frontend."""

    def __init__(
        self,
        *,
        downstream: RealtimeTransport,
        upstream: RealtimeTransport,
        runtime: RealtimeSessionRuntimePort,
        model_contracts: ModelContractCatalog | None = None,
        interaction_profile: InteractionProfile | None = None,
        static_instructions: str | None = None,
        projection_events: ProjectionEventFactory | None = None,
        activity_observer: ActivityObserver | None = None,
        id_factory: Callable[[str], str] | None = None,
        public_model: str = "voiceclaw",
        max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
        bootstrap_timeout_seconds: float = _DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS,
        max_pending_speech: int = _DEFAULT_MAX_PENDING_SPEECH,
        context_character_budget: int = _DEFAULT_CONTEXT_CHARACTER_BUDGET,
    ) -> None:
        """Configure a single-session facade without opening either transport."""
        contracts = model_contracts or load_model_contract_catalog()
        profile = interaction_profile or load_interaction_profile_catalog().resolve("stateless")
        policy = contracts.static_instructions if static_instructions is None else static_instructions
        if not isinstance(policy, str) or not policy.strip():
            raise ValueError("static_instructions must be a non-empty string")
        if len(policy) > _MAX_INSTRUCTIONS_CHARACTERS or "\x00" in policy:
            raise ValueError("static_instructions is invalid")
        if (
            not isinstance(public_model, str)
            or not public_model.strip()
            or len(public_model) > 256
            or "\x00" in public_model
        ):
            raise ValueError("public_model is invalid")
        if (
            isinstance(max_event_bytes, bool)
            or not isinstance(max_event_bytes, int)
            or not 1024 <= max_event_bytes <= _MAX_CONFIGURED_EVENT_BYTES
        ):
            raise ValueError("max_event_bytes must be between 1024 and 16777216")
        if (
            isinstance(bootstrap_timeout_seconds, bool)
            or not isinstance(bootstrap_timeout_seconds, (int, float))
            or not math.isfinite(bootstrap_timeout_seconds)
            or bootstrap_timeout_seconds <= 0
        ):
            raise ValueError("bootstrap_timeout_seconds must be a finite positive number")
        if (
            isinstance(max_pending_speech, bool)
            or not isinstance(max_pending_speech, int)
            or not 1 <= max_pending_speech <= _MAX_CONFIGURED_PENDING_SPEECH
        ):
            raise ValueError("max_pending_speech must be between 1 and 1024")
        if (
            isinstance(context_character_budget, bool)
            or not isinstance(context_character_budget, int)
            or not 1 <= context_character_budget <= _MAX_CONFIGURED_CONTEXT_CHARACTER_BUDGET
        ):
            raise ValueError("context_character_budget must be between 1 and 64000")
        self._downstream = downstream
        self._upstream = upstream
        self._model_contracts = contracts
        self._interaction_profile = profile
        self._static_instructions = policy.strip()
        self._public_model = public_model.strip()
        self._max_event_bytes = max_event_bytes
        self._bootstrap_timeout_seconds = float(bootstrap_timeout_seconds)
        self._max_pending_speech = max_pending_speech
        self._context_character_budget = context_character_budget
        self._runtime = runtime
        self._id_factory = id_factory or self._new_id
        self._projection_events = projection_events or ProjectionEventFactory(id_factory=self._id_factory)
        self._observer = activity_observer
        self._tools = VoiceClawToolRegistry(contracts=contracts, interaction_profile=profile)
        self._instruction_builder = FrontendInstructionBuilder(
            maximum_characters=_MAX_INSTRUCTIONS_CHARACTERS,
            contracts=contracts,
        )
        self._registry = _OwnershipRegistry(self._id_factory)
        self._downstream_send_lock = asyncio.Lock()
        self._upstream_send_lock = asyncio.Lock()
        self._facade_session_id = self._id_factory("sess_vc")
        self._facade_conversation_id = self._id_factory("conv_vc")
        self._upstream_session_id: str | None = None
        self._upstream_conversation_id: str | None = None
        self._pending_finalized_user_text: str | None = None
        self._pending_typed_user_turns: deque[tuple[str, str]] = deque()
        self._pending_client_item_creates: dict[str, str] = {}
        self._deferred_client_item_creates: deque[_DeferredClientItemCreate] = deque()
        self._response_finalized_user_text: dict[str, str] = {}
        self._response_finalized_user_item: dict[str, str] = {}
        self._unbound_audio_items: deque[str] = deque()
        self._manual_audio_commits: deque[_ManualAudioCommit] = deque()
        self._input_generation = 0
        self._active_input_generation: int | None = None
        self._input_buffer_open = False
        self._audio_item_input_generation: dict[str, int] = {}
        self._deferred_active_truncations: dict[str, deque[dict[str, Any]]] = {}
        self._pending_passthrough_client_events: dict[str, tuple[str, str]] = {}
        self._failed_audio_items: set[str] = set()
        self._audio_item_response: dict[str, str] = {}
        self._response_audio_item: dict[str, str] = {}
        self._response_expected_protected_tool: dict[str, ProtectedTool] = {}
        self._response_protected_calls_forbidden: set[str] = set()
        self._response_model_route_selection: set[str] = set()
        self._response_direct_reply: dict[str, _DirectReplyTemplate] = {}
        self._late_finalized_text_waiters: dict[str, asyncio.Future[str | None]] = {}
        self._finalized_audio_timeout_tasks: dict[str, asyncio.Task[None]] = {}
        self._pending_protected_calls: dict[str, ParsedToolCall] = {}
        self._buffered_function_item_events: dict[str, list[dict[str, Any]]] = {}
        self._tool_tasks: set[asyncio.Task[None]] = set()
        self._response_create_queue: deque[_QueuedResponseCreate] = deque()
        self._response_state_lock = asyncio.Lock()
        # Serializes the final safe-to-speak check with input/model activity
        # transitions. A response may wait in the arbiter for a long time;
        # eligibility must still be true at the moment it is sent upstream.
        self._speech_admission_lock = asyncio.Lock()
        # A protected call holds this short causal fence until its local
        # function outcome and follow-up acknowledgement/failure have both
        # been reserved. Backend execution continues after the fence drops.
        self._response_admission_barriers: set[str] = set()
        self._response_create_in_flight: _QueuedResponseCreate | None = None
        self._active_upstream_response_id: str | None = None
        self._active_upstream_response_purpose: _ResponsePurpose | None = None
        self._active_speech_purpose: str | None = None
        self._active_speech_local_request_id: str | None = None
        self._active_speech_delivery_mode: str | None = None
        self._active_presentation_id: str | None = None
        self._active_playback_receipt_id: str | None = None
        self._output_sample_rate = _DEFAULT_PCM_SAMPLE_RATE
        self._playback_leases_by_response: dict[str, _PlaybackLease] = {}
        self._playback_leases_by_item: dict[str, _PlaybackLease] = {}
        self._pending_playback_receipts: dict[str, _PendingPlaybackReceipt] = {}
        self._retired_playback_receipt_ids: deque[str] = deque()
        self._retired_playback_receipt_id_set: set[str] = set()
        self._recorded_user_turn_ids: set[str] = set()
        self._response_terminal_fence = False
        self._delivery_queue_projection_lock = asyncio.Lock()
        self._reported_delivery_queue_state = (0, 0, False)
        self._automatic_turn_detection = False
        self._automatic_response = False
        self._client_session_instructions = ""
        self._accepted_client_session: dict[str, Any] = {}
        self._pending_client_session: dict[str, Any] | None = None
        self._pending_client_session_event_id: str | None = None
        self._pending_previous_automatic_turn_detection: bool | None = None
        self._pending_previous_automatic_response: bool | None = None
        self._activity = FrontendActivity(connected=True, output=OutputActivity.IDLE)
        self._session_snapshot: SessionSnapshot | None = None
        self._runtime_open = False
        self._bootstrapped = False

    @property
    def session_id(self) -> str:
        """Return the browser-visible facade session identity."""
        return self._facade_session_id

    @property
    def conversation_id(self) -> str:
        """Return the browser-visible facade conversation identity."""
        return self._facade_conversation_id

    async def serve(self) -> None:
        """Run the facade until either transport closes or violates the protocol."""
        close_reason = "client_disconnected"
        try:
            await self.bootstrap()
            downstream_task = asyncio.create_task(self._pump_downstream(), name="voiceclaw-downstream")
            upstream_task = asyncio.create_task(self._pump_upstream(), name="voiceclaw-upstream")
            done, pending = await asyncio.wait({downstream_task, upstream_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            completed = await asyncio.gather(*done, return_exceptions=True)
            for result in completed:
                if isinstance(result, BaseException):
                    raise result
        except asyncio.CancelledError:
            close_reason = "server_cancelled"
            raise
        except (EOFError, StopAsyncIteration):
            return
        except FacadeProtocolError as error:
            close_reason = error.code
            await self._send_public_error(error.code, error.public_message)
        except Exception:
            close_reason = "voiceclaw_internal_error"
            await self._send_public_error("voiceclaw_internal_error", "The VoiceClaw session could not continue.")
        finally:
            await self._cancel_tool_tasks()
            self._response_create_queue.clear()
            self._response_admission_barriers.clear()
            self._response_create_in_flight = None
            self._active_upstream_response_id = None
            self._active_upstream_response_purpose = None
            self._active_speech_purpose = None
            self._active_speech_local_request_id = None
            self._active_speech_delivery_mode = None
            self._active_presentation_id = None
            self._active_playback_receipt_id = None
            self._playback_leases_by_response.clear()
            self._playback_leases_by_item.clear()
            self._pending_playback_receipts.clear()
            self._retired_playback_receipt_ids.clear()
            self._retired_playback_receipt_id_set.clear()
            self._recorded_user_turn_ids.clear()
            self._response_terminal_fence = False
            self._automatic_turn_detection = False
            self._automatic_response = False
            self._pending_previous_automatic_turn_detection = None
            self._pending_previous_automatic_response = None
            self._pending_finalized_user_text = None
            self._pending_typed_user_turns.clear()
            self._pending_client_item_creates.clear()
            self._deferred_client_item_creates.clear()
            self._response_finalized_user_text.clear()
            self._response_finalized_user_item.clear()
            self._unbound_audio_items.clear()
            self._manual_audio_commits.clear()
            self._active_input_generation = None
            self._input_buffer_open = False
            self._audio_item_input_generation.clear()
            self._deferred_active_truncations.clear()
            self._pending_passthrough_client_events.clear()
            self._failed_audio_items.clear()
            self._audio_item_response.clear()
            self._response_audio_item.clear()
            self._response_expected_protected_tool.clear()
            self._response_protected_calls_forbidden.clear()
            self._response_model_route_selection.clear()
            self._response_direct_reply.clear()
            for waiter in self._late_finalized_text_waiters.values():
                if not waiter.done():
                    waiter.cancel()
            self._late_finalized_text_waiters.clear()
            timeout_tasks = tuple(self._finalized_audio_timeout_tasks.values())
            self._finalized_audio_timeout_tasks.clear()
            for task in timeout_tasks:
                task.cancel()
            if timeout_tasks:
                await asyncio.gather(*timeout_tasks, return_exceptions=True)
            self._pending_protected_calls.clear()
            self._buffered_function_item_events.clear()
            self._registry.clear()
            if self._runtime_open:
                with suppress(Exception):
                    await self._runtime.close_session(self._facade_session_id, close_reason)

    async def wait_for_pending_tools(self) -> None:
        """Wait for currently admitted server-owned tool calls to settle."""
        while self._tool_tasks:
            await asyncio.gather(*tuple(self._tool_tasks))

    async def bootstrap(self) -> None:
        """Consume the private upstream bootstrap and emit a sanitized facade bootstrap."""
        if self._bootstrapped:
            raise FacadeProtocolError("invalid_state", "The realtime session was already bootstrapped.")
        try:
            self._session_snapshot = await self._runtime.open_session(
                self._facade_session_id,
                self._facade_conversation_id,
            )
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not open the session."
            ) from error
        projected_tools = self._session_snapshot.frontend_tools if self._session_snapshot.gateway_reachable else ()
        self._tools = VoiceClawToolRegistry(
            tools=projected_tools,
            include_direct_route=bool(projected_tools),
            contracts=self._model_contracts,
            interaction_profile=self._interaction_profile,
        )
        self._runtime_open = True
        try:
            await self._runtime.update_activity(self._facade_session_id, self._activity)
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not initialize session activity."
            ) from error
        try:
            async with asyncio.timeout(self._bootstrap_timeout_seconds):
                session_event: dict[str, Any] | None = None
                conversation_event: dict[str, Any] | None = None
                while session_event is None or conversation_event is None:
                    event = await self._receive_upstream()
                    event_type = event["type"]
                    if event_type == "session.created" and session_event is None:
                        session_event = event
                        continue
                    if event_type == "conversation.created" and conversation_event is None:
                        conversation_event = event
                        continue
                    if event_type == "error":
                        raise FacadeProtocolError("upstream_unavailable", "The realtime frontend rejected the session.")
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "The realtime frontend sent an invalid bootstrap."
                    )

                session = session_event.get("session")
                conversation = conversation_event.get("conversation")
                if not isinstance(session, dict) or not isinstance(conversation, dict):
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "The realtime frontend bootstrap was malformed."
                    )
                self._upstream_session_id = _identifier(session.get("id"), "upstream session id")
                self._upstream_conversation_id = _identifier(conversation.get("id"), "upstream conversation id")

                bootstrap_update = self._tools.merge_session_update(
                    {
                        "event_id": self._id_factory("event_vc"),
                        "type": "session.update",
                        "session": {},
                    },
                    static_instructions=self._static_instructions,
                    dynamic_projection=self._current_projection(),
                )
                self._force_projection_boundary(bootstrap_update["session"], baseline=session)
                await self._send_upstream(bootstrap_update)
                bootstrap_ack = await self._receive_upstream()
                if bootstrap_ack.get("type") == "error":
                    raise FacadeProtocolError(
                        "upstream_unavailable", "The realtime frontend rejected VoiceClaw policy."
                    )
                if bootstrap_ack.get("type") != "session.updated" or not isinstance(bootstrap_ack.get("session"), dict):
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "The realtime frontend did not acknowledge VoiceClaw policy."
                    )
                acknowledged_session = bootstrap_ack["session"]
                self._capture_output_sample_rate(acknowledged_session)
                acknowledged_audio = acknowledged_session.get("audio")
                acknowledged_input = acknowledged_audio.get("input") if isinstance(acknowledged_audio, dict) else None
                baseline_audio = session.get("audio")
                baseline_input = baseline_audio.get("input") if isinstance(baseline_audio, dict) else None
                baseline_supplied = isinstance(baseline_input, dict) and "turn_detection" in baseline_input
                baseline_turn_detection = (
                    baseline_input.get("turn_detection") if isinstance(baseline_input, dict) else None
                )
                acknowledged_turn_detection = (
                    acknowledged_input.get("turn_detection") if isinstance(acknowledged_input, dict) else None
                )
                if baseline_supplied and baseline_turn_detection is None:
                    if not isinstance(acknowledged_input, dict) or acknowledged_turn_detection is not None:
                        raise FacadeProtocolError(
                            "upstream_protocol_error",
                            "The realtime frontend did not retain VoiceClaw manual turn control.",
                        )
                elif baseline_supplied and isinstance(baseline_turn_detection, dict):
                    expected_turn_detection = copy.deepcopy(baseline_turn_detection)
                    expected_turn_detection["create_response"] = False
                    if acknowledged_turn_detection != expected_turn_detection:
                        raise FacadeProtocolError(
                            "upstream_protocol_error",
                            "The realtime frontend did not retain VoiceClaw automatic turn control.",
                        )
                elif baseline_supplied:
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "The realtime frontend advertised invalid turn detection."
                    )
                acknowledged_tools = acknowledged_session.get("tools")
                if not isinstance(acknowledged_tools, list):
                    raise FacadeProtocolError(
                        "upstream_protocol_error",
                        "The realtime frontend did not acknowledge VoiceClaw tools.",
                    )
                acknowledged_by_name = {
                    tool.get("name"): tool
                    for tool in acknowledged_tools
                    if isinstance(tool, dict) and isinstance(tool.get("name"), str)
                }
                for protected_schema in self._tools.schemas():
                    acknowledged_schema = acknowledged_by_name.get(protected_schema["name"])
                    if (
                        not isinstance(acknowledged_schema, dict)
                        or acknowledged_schema.get("type") != "function"
                        or acknowledged_schema.get("parameters") != protected_schema["parameters"]
                    ):
                        raise FacadeProtocolError(
                            "upstream_protocol_error",
                            "The realtime frontend did not retain a VoiceClaw protected tool.",
                        )

                safe_session = _strip_sensitive(acknowledged_session)
                assert isinstance(safe_session, dict)
                safe_session["id"] = self._facade_session_id
                safe_session["model"] = self._public_model
                safe_session["instructions"] = ""
                safe_session["tools"] = []
                safe_audio = safe_session.get("audio")
                safe_input = safe_audio.get("input") if isinstance(safe_audio, dict) else None
                safe_turn_detection = safe_input.get("turn_detection") if isinstance(safe_input, dict) else None
                if isinstance(safe_turn_detection, dict):
                    safe_turn_detection["create_response"] = self._automatic_response
        except TimeoutError as error:
            raise FacadeProtocolError(
                "upstream_timeout", "The realtime frontend did not complete its session bootstrap in time."
            ) from error
        await self._send_downstream(
            {"event_id": self._id_factory("event_vc"), "type": "session.created", "session": safe_session}
        )
        await self._send_downstream(
            {
                "event_id": self._id_factory("event_vc"),
                "type": "conversation.created",
                "conversation": {
                    "id": self._facade_conversation_id,
                    "object": "realtime.conversation",
                },
            }
        )
        self._bootstrapped = True
        await self._emit_runtime_attachment()

    async def _emit_runtime_attachment(self) -> None:
        """Project server-side gateway reachability without claiming agent readiness."""
        snapshot = self._session_snapshot
        if snapshot is None:
            raise FacadeProtocolError("runtime_unavailable", "VoiceClaw did not receive a runtime snapshot.")
        ready = snapshot.gateway_reachable
        payload = {
            "backend": snapshot.backend_label,
            "mode": snapshot.backend_mode,
            "target": snapshot.target_ref,
            "agent_readiness": "unknown" if ready else "unavailable",
            "capabilities": list(snapshot.capabilities),
            "durability": snapshot.durability.value if ready else "unavailable",
            "event_delivery": snapshot.event_delivery.value if ready else "unavailable",
            "max_parallel_work": snapshot.max_parallel_work if ready else 0,
        }
        if ready and snapshot.capability_source is not None:
            payload["capability_source"] = snapshot.capability_source.value
        if ready and snapshot.capability_source_id is not None:
            payload["capability_source_id"] = snapshot.capability_source_id
        if ready and snapshot.capability_revision is not None:
            payload["capability_revision"] = snapshot.capability_revision
        if ready and snapshot.capability_hash is not None:
            payload["capability_hash"] = snapshot.capability_hash
        if snapshot.model_contract_schema is not None:
            payload["model_contract_schema"] = snapshot.model_contract_schema
        if snapshot.model_contract_profile is not None:
            payload["model_contract_profile"] = snapshot.model_contract_profile
        if snapshot.model_contract_hash is not None:
            payload["model_contract_hash"] = snapshot.model_contract_hash
        correlation = {
            "backend_name": snapshot.backend_label,
            "backend_mode": snapshot.backend_mode,
            "target_name": snapshot.backend_label,
            "target_ref": snapshot.target_ref,
            "target_state": "gateway_reachable" if ready else "unavailable",
            "agent_readiness": "unknown" if ready else "unavailable",
            "durability": snapshot.durability.value if ready else "unavailable",
            "event_delivery": snapshot.event_delivery.value if ready else "unavailable",
            "max_parallel_work": str(snapshot.max_parallel_work if ready else 0),
        }
        if ready and snapshot.frontend_tools:
            correlation["frontend_tools"] = ",".join(tool.name for tool in snapshot.frontend_tools)
        if snapshot.capabilities:
            correlation["capabilities"] = ",".join(snapshot.capabilities)
        await self._emit_projection(
            kind="backend_target",
            phase="reachable" if ready else "unavailable",
            title=(f"{snapshot.backend_label} gateway reachable" if ready else "Agent backend unavailable"),
            text=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            correlation=correlation,
        )

    async def handle_downstream_event(self, event: Mapping[str, Any]) -> None:
        """Validate and forward one already-decoded browser event."""
        value = copy.deepcopy(dict(event))
        queue_response_create = False
        prepared_response: _PreparedResponseCreate | None = None
        pending_client_item_create: tuple[str, str] | None = None
        manual_audio_commit_event_id: str | None = None
        passthrough_client_event_id: str | None = None
        pending_playback_receipt: _PendingPlaybackReceipt | None = None
        defer_upstream = False
        if _has_reserved_metadata(value):
            raise FacadeProtocolError("invalid_request", "voiceclaw_* metadata is reserved for the server.")
        event_type = value.get("type")
        if not isinstance(event_type, str):
            raise FacadeProtocolError("invalid_event", "Realtime events require a valid type.")
        if event_type in {
            "session.created",
            "session.updated",
            "conversation.created",
            "response.created",
            "response.done",
            "response.output_item.added",
            "response.output_item.done",
            "error",
        }:
            raise FacadeProtocolError("invalid_request", "The client submitted a server-only realtime event.")
        if event_type == "session.update":
            try:
                if self._pending_client_session is not None:
                    raise FacadeProtocolError(
                        "session_update_in_progress", "Wait for the current session update to be acknowledged."
                    )
                requested_session = value.get("session")
                if not isinstance(requested_session, dict):
                    raise ValueError("session.update requires a session object")
                client_event_id = value.get("event_id")
                if client_event_id is None:
                    client_event_id = self._id_factory("event_vc")
                    value["event_id"] = client_event_id
                if (
                    not isinstance(client_event_id, str)
                    or not client_event_id
                    or len(client_event_id) > _MAX_IDENTIFIER_CHARACTERS
                    or any(ord(character) < 32 or ord(character) == 127 for character in client_event_id)
                ):
                    raise ValueError("session.update requires a valid event id")
                candidate = _merge_object_patch(self._accepted_client_session, requested_session)
                requested_instructions = candidate.get("instructions", "")
                if requested_instructions is not None and not isinstance(requested_instructions, str):
                    raise ValueError("session instructions must be text or null")
                value = self._tools.merge_session_update(
                    {**value, "session": candidate},
                    static_instructions=self._static_instructions,
                    dynamic_projection=self._current_projection(),
                )
                previous_automatic_response = self._automatic_response
                previous_automatic_turn_detection = self._automatic_turn_detection
                self._force_projection_boundary(value["session"])
                self._pending_previous_automatic_turn_detection = previous_automatic_turn_detection
                self._pending_previous_automatic_response = previous_automatic_response
                self._pending_client_session = candidate
                self._pending_client_session_event_id = client_event_id
            except (TypeError, ValueError) as error:
                raise FacadeProtocolError("invalid_request", "The session update is invalid.") from error
        elif event_type == "response.create":
            prepared_response = self._prepare_response_create(value)
            value = prepared_response.event
            queue_response_create = True
        elif event_type == "input_audio_buffer.commit" and not self._automatic_turn_detection:
            event_id = value.get("event_id")
            if event_id is None:
                event_id = self._id_factory("event_vc")
                value["event_id"] = event_id
            await self._register_manual_audio_commit(event_id)
            manual_audio_commit_event_id = event_id
        elif event_type == "response.cancel" and value.get("response_id") is not None:
            upstream_response_id = self._registry.cancel_target(value["response_id"])
            if upstream_response_id is None:
                return
            value["response_id"] = upstream_response_id
            passthrough_client_event_id = self._track_passthrough_client_event(
                value,
                event_type="response.cancel",
                target_id=upstream_response_id,
            )
        elif event_type in {"conversation.item.delete", "conversation.item.truncate", "conversation.item.retrieve"}:
            if event_type == "conversation.item.delete":
                await self._discard_finalized_typed_item(_identifier(value.get("item_id"), "item id"))
            upstream_item_id = self._registry.upstream_client_item(value.get("item_id"))
            value["item_id"] = upstream_item_id
            if event_type == "conversation.item.truncate":
                pending_playback_receipt = self._validate_playback_receipt(value, upstream_item_id)
                passthrough_client_event_id = self._track_passthrough_client_event(
                    value,
                    event_type="conversation.item.truncate",
                    target_id=upstream_item_id,
                )
                if pending_playback_receipt is not None:
                    lease = self._playback_leases_by_response[pending_playback_receipt.response_id]
                    lease.pending_receipt_id = pending_playback_receipt.receipt_id
                    self._pending_playback_receipts[pending_playback_receipt.receipt_id] = pending_playback_receipt
                response_id = self._registry.item(upstream_item_id).response_id
                if response_id is not None:
                    defer_upstream = await self._defer_active_truncation(response_id, value)
        elif event_type == "conversation.item.create":
            value = self._prepare_client_item(value)
            item = value.get("item")
            assert isinstance(item, dict)
            item_id = _identifier(item.get("id"), "client item id")
            event_id = value.get("event_id")
            if event_id is None:
                event_id = self._id_factory("event_vc")
                value["event_id"] = event_id
            if (
                not isinstance(event_id, str)
                or not event_id
                or len(event_id) > _MAX_IDENTIFIER_CHARACTERS
                or any(ord(character) < 32 or ord(character) == 127 for character in event_id)
                or self._client_item_event_id_in_use(event_id)
            ):
                self._registry.retire_item(item_id)
                raise FacadeProtocolError("invalid_request", "conversation.item.create requires a unique event id.")
            try:
                finalized_text = self._finalized_text_from_client_item(item)
            except BaseException:
                self._registry.retire_item(item_id)
                raise
            if finalized_text is not None:
                if len(self._pending_typed_user_turns) >= self._max_pending_speech:
                    self._registry.retire_item(item_id)
                    raise FacadeProtocolError(
                        "session_capacity_exceeded", "The realtime session has too many pending user turns."
                    )
                await self._retire_rejected_unpaired_manual_commits()
                self._pending_typed_user_turns.append((item_id, finalized_text))
            pending_client_item_create = (event_id, item_id)
            try:
                defer_upstream = await self._defer_client_item_create_if_response_blocked(
                    value,
                    event_id=event_id,
                    item_id=item_id,
                )
            except BaseException:
                await self._discard_finalized_typed_item(item_id)
                self._registry.retire_item(item_id)
                raise
        await self._update_downstream_activity(event_type)
        if queue_response_create:
            assert prepared_response is not None
            await self._enqueue_response_create(
                value,
                finalized_user_text=prepared_response.finalized_user_text,
                finalized_user_item_id=prepared_response.finalized_user_item_id,
                bind_pending_user_turn=not prepared_response.has_response_local_input,
                client_response_instructions=prepared_response.client_response_instructions,
            )
        else:
            if defer_upstream:
                return
            if pending_client_item_create is not None:
                create_event_id, item_id = pending_client_item_create
                self._pending_client_item_creates[create_event_id] = item_id
            try:
                await self._send_upstream(value)
            except BaseException:
                if passthrough_client_event_id is not None:
                    self._pending_passthrough_client_events.pop(passthrough_client_event_id, None)
                    self._abandon_playback_receipt(passthrough_client_event_id)
                if pending_client_item_create is not None:
                    create_event_id, item_id = pending_client_item_create
                    self._pending_client_item_creates.pop(create_event_id, None)
                    await self._discard_finalized_typed_item(item_id)
                    self._registry.retire_item(item_id)
                if manual_audio_commit_event_id is not None:
                    await self._reject_manual_audio_commit(manual_audio_commit_event_id)
                if event_type == "session.update":
                    self._pending_client_session = None
                    self._pending_client_session_event_id = None
                    if self._pending_previous_automatic_turn_detection is not None:
                        self._automatic_turn_detection = self._pending_previous_automatic_turn_detection
                    self._pending_previous_automatic_turn_detection = None
                    if self._pending_previous_automatic_response is not None:
                        self._automatic_response = self._pending_previous_automatic_response
                    self._pending_previous_automatic_response = None
                raise

    async def handle_upstream_event(self, event: Mapping[str, Any]) -> None:
        """Rewrite, filter, and forward one already-decoded frontend event."""
        value = copy.deepcopy(dict(event))
        event_type = value.get("type")
        if not isinstance(event_type, str):
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend event has no type.")
        await self._update_upstream_activity(event_type, value)
        if event_type == "conversation.item.input_audio_transcription.completed":
            finalized_item_id = _identifier(value.get("item_id"), "transcribed audio item id")
            transcript = self._normalize_finalized_user_text(
                value.get("transcript"),
                code="upstream_protocol_error",
                message="The realtime frontend returned an invalid finalized transcription.",
            )
            if transcript is not None:
                await self._finish_audio_input(finalized_item_id)
                await self._record_finalized_user_text(
                    transcript,
                    allow_active_binding=True,
                    finalized_item_id=finalized_item_id,
                )
            else:
                await self._fail_finalized_audio_text(
                    finalized_item_id,
                    code="empty_audio_turn",
                    message="No speech was detected. Please try again.",
                )
        elif event_type == "conversation.item.input_audio_transcription.failed":
            await self._fail_finalized_audio_text(
                _identifier(value.get("item_id"), "transcribed audio item id"),
                code="audio_transcription_failed",
                message="The audio request could not be transcribed. Please try again.",
            )
        if event_type in {"session.created", "conversation.created"}:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend repeated its bootstrap.")
        if event_type == "error":
            upstream_error = value.get("error")
            upstream_code = upstream_error.get("code") if isinstance(upstream_error, Mapping) else None
            upstream_param = upstream_error.get("param") if isinstance(upstream_error, Mapping) else None
            failed_client_event_id = await self._correlated_client_error_event_id(value)
            if failed_client_event_id is not None:
                await self._fail_playback_receipt(failed_client_event_id)
            _LOGGER.warning(
                "Realtime frontend rejected an event code=%s param=%s",
                upstream_code if isinstance(upstream_code, str) else "unknown",
                upstream_param if isinstance(upstream_param, str) else "unknown",
            )
            failed_event_id = upstream_error.get("event_id") if isinstance(upstream_error, Mapping) else None
            if self._pending_client_session is not None and failed_event_id == self._pending_client_session_event_id:
                self._pending_client_session = None
                self._pending_client_session_event_id = None
                if self._pending_previous_automatic_turn_detection is not None:
                    self._automatic_turn_detection = self._pending_previous_automatic_turn_detection
                self._pending_previous_automatic_turn_detection = None
                if self._pending_previous_automatic_response is not None:
                    self._automatic_response = self._pending_previous_automatic_response
                self._pending_previous_automatic_response = None
            await self._handle_client_item_create_error(value)
            await self._handle_manual_audio_commit_error(value)
            await self._handle_response_create_error(value)
            await self._send_public_error(
                "upstream_error",
                "The realtime frontend rejected an event.",
                event_id=failed_client_event_id,
            )
            return
        if event_type == "session.updated":
            await self._handle_session_updated(value)
            return
        if event_type in {"conversation.item.added", "conversation.item.created"}:
            item = value.get("item")
            if isinstance(item, dict) and item.get("id") is not None:
                self._acknowledge_client_item_create(item.get("id"))
            if isinstance(item, dict) and item.get("type") == "function_call":
                item_id = _identifier(item.get("id"), "upstream item id")
                self._registry.ensure_item(item_id)
                self._buffer_function_item_event(item_id, value)
                return
        nested_item = value.get("item")
        conversation_item_event = event_type.startswith("conversation.item.")
        if conversation_item_event and isinstance(nested_item, dict) and nested_item.get("type") == "function_call":
            item_record = self._registry.ensure_item(nested_item.get("id"))
            if item_record.owner is _Owner.SERVER:
                return
            name = nested_item.get("name")
            if isinstance(name, str) and name in {tool.value for tool in ProtectedTool}:
                raise FacadeProtocolError(
                    "upstream_protocol_error", "A protected function item arrived before ownership was established."
                )
        if (
            conversation_item_event
            and isinstance(nested_item, dict)
            and nested_item.get("type") == "function_call_output"
        ):
            call = self._registry.call(nested_item.get("call_id"))
            item_record = self._registry.ensure_item(nested_item.get("id"))
            item_record.owner = call.owner
            item_record.call_id = _identifier(nested_item.get("call_id"), "upstream call id")
            if call.owner is _Owner.SERVER:
                if "previous_item_id" not in value:
                    raise FacadeProtocolError(
                        "upstream_protocol_error",
                        "A protected function output did not identify its predecessor.",
                    )
                self._registry.hide_item(nested_item.get("id"), value["previous_item_id"])
                if event_type == "conversation.item.done":
                    self._registry.retire_server_call(nested_item.get("call_id"), nested_item.get("id"))
                return
        if event_type == "response.created":
            response = value.get("response")
            if not isinstance(response, dict):
                raise FacadeProtocolError("upstream_protocol_error", "A response event was malformed.")
            response_id = _identifier(response.get("id"), "upstream response id")
            self._registry.declare_response(response_id)
            is_delivery = await self._mark_response_created(response_id)
            if response_id in self._response_model_route_selection:
                # The structured selector is internal control-plane work. Its
                # response envelope and any model output never cross the
                # browser boundary; only the chosen public path does.
                await self._publish_delivery_queue_state()
                return
            rewritten = self._rewrite_upstream_event(value)
            rewritten_response = rewritten.get("response")
            if is_delivery and isinstance(rewritten_response, dict):
                metadata = rewritten_response.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata["voiceclaw_delivery"] = "true"
            if isinstance(rewritten_response, dict):
                self._mark_active_speech_metadata(rewritten_response)
            await self._send_downstream(rewritten)
            await self._publish_delivery_queue_state()
            return
        if event_type == "response.output_item.added":
            response_id = value.get("response_id")
            item = value.get("item")
            if not isinstance(item, dict):
                raise FacadeProtocolError("upstream_protocol_error", "An output item event was malformed.")
            checked_response_id = _identifier(response_id, "upstream response id")
            expected_tool = self._response_expected_protected_tool.get(checked_response_id)
            protected_call = (
                item.get("type") == "function_call"
                and isinstance(item.get("name"), str)
                and item["name"].startswith("voiceclaw_")
            )
            if protected_call and checked_response_id in self._response_protected_calls_forbidden:
                raise FacadeProtocolError(
                    "upstream_protocol_error",
                    "The realtime frontend emitted a protected tool outside an authorized route.",
                )
            if expected_tool is not None and (
                item.get("type") != "function_call" or item.get("name") != expected_tool.value
            ):
                raise FacadeProtocolError(
                    "required_tool_not_called",
                    "The realtime frontend did not follow the required backend route.",
                )
            if checked_response_id in self._response_model_route_selection and (
                item.get("type") != "function_call"
                or item.get("name") not in {tool.value for tool in self._tools.enabled}
            ):
                raise FacadeProtocolError(
                    "required_tool_not_called",
                    "The realtime frontend did not return a structured VoiceClaw route.",
                )
            item_id = _identifier(item.get("id"), "upstream item id")
            buffered_events = self._buffered_function_item_events.get(item_id, [])
            for buffered in buffered_events:
                buffered_item = buffered.get("item")
                if not isinstance(buffered_item, dict) or any(
                    buffered_item.get(field) != item.get(field) for field in ("id", "type", "call_id", "name")
                ):
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "A buffered function item changed immutable identity."
                    )
            record = self._registry.add_output_item(item.get("id"), response_id)
            owner = self._classify_item(item)
            if owner is not None:
                call = self._registry.declare_call(
                    item.get("call_id"), item_id=item.get("id"), response_id=response_id, owner=owner
                )
                record.call_id = _identifier(item.get("call_id"), "upstream call id")
                record.owner = owner
                await self._set_activity(model=ModelActivity.WAITING_FOR_TOOL)
                if call.owner is _Owner.SERVER:
                    for buffered in buffered_events:
                        if "previous_item_id" not in buffered:
                            raise FacadeProtocolError(
                                "upstream_protocol_error",
                                "A protected function item did not identify its predecessor.",
                            )
                        self._registry.hide_item(item_id, buffered["previous_item_id"])
                    self._buffered_function_item_events.pop(item_id, None)
                    return
                for buffered in self._buffered_function_item_events.pop(item_id, []):
                    await self._send_downstream(self._rewrite_upstream_event(buffered))
            await self._send_downstream(self._rewrite_upstream_event(value))
            return
        if event_type == "response.output_item.done":
            await self._handle_output_item_done(value)
            return
        if event_type == "response.done":
            await self._handle_response_done(value)
            return

        associated_response_id = value.get("response_id")
        if isinstance(associated_response_id, str) and associated_response_id in self._response_model_route_selection:
            return

        if event_type == "response.output_audio.delta":
            await self._observe_output_audio_delta(value)
        elif event_type == "response.output_audio.done":
            self._observe_output_audio_done(value)
        elif event_type in {
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
        }:
            self._observe_output_audio_transcript(value)

        item_id = value.get("item_id")
        deleted_local_item_id: str | None = None
        if item_id is not None:
            record = self._registry.ensure_item(item_id)
            if record.owner is _Owner.SERVER:
                return
            if event_type == "conversation.item.deleted":
                deleted_local_item_id = record.local_id
            elif event_type == "conversation.item.truncated":
                receipt_id = self._acknowledge_passthrough_client_event(
                    event_type="conversation.item.truncate",
                    target_id=_identifier(item_id, "upstream item id"),
                )
                if receipt_id is not None:
                    await self._acknowledge_playback_receipt(receipt_id)
        call_id = value.get("call_id")
        if call_id is not None:
            call = self._registry.call(call_id)
            if call.owner is _Owner.SERVER:
                return
        await self._send_downstream(self._rewrite_upstream_event(value))
        if event_type == "conversation.item.deleted" and item_id is not None:
            assert deleted_local_item_id is not None
            await self._discard_finalized_typed_item(deleted_local_item_id)
            self._registry.retire_item(item_id)
        if (
            event_type == "conversation.item.done"
            and isinstance(nested_item, dict)
            and nested_item.get("type") == "function_call_output"
        ):
            self._registry.retire_client_call(nested_item.get("call_id"))
            if nested_item.get("id") is not None:
                self._registry.retire_terminal_item(nested_item.get("id"))
        if event_type == "input_audio_buffer.committed":
            finalized_item_id = _identifier(value.get("item_id"), "committed audio item id")
            if finalized_item_id not in self._unbound_audio_items:
                self._unbound_audio_items.append(finalized_item_id)
            manual_response = await self._bind_manual_audio_commit(finalized_item_id)
            if not manual_response and self._active_input_generation is not None:
                self._audio_item_input_generation[finalized_item_id] = self._active_input_generation
            if self._automatic_response and not manual_response:
                prepared = self._prepare_response_create(
                    {"event_id": self._id_factory("event_vc"), "type": "response.create", "response": {}}
                )
                await self._enqueue_response_create(
                    prepared.event,
                    finalized_user_text=prepared.finalized_user_text,
                    bind_pending_user_turn=not prepared.has_response_local_input,
                    finalized_item_id=finalized_item_id,
                    client_response_instructions=prepared.client_response_instructions,
                )

    def _playback_lease_for_audio_event(self, event: Mapping[str, Any]) -> _PlaybackLease:
        """Resolve and bind an audio event to its single response playback lease."""
        response_id = _identifier(event.get("response_id"), "upstream response id")
        item_id = _identifier(event.get("item_id"), "upstream item id")
        content_index = event.get("content_index", 0)
        if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
            raise FacadeProtocolError("upstream_protocol_error", "An output audio content index was invalid.")
        lease = self._playback_leases_by_response.get(response_id)
        if lease is None:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend emitted audio without a public playback lease."
            )
        if lease.item_id is None:
            item_record = self._registry.item(item_id)
            if item_record.owner is _Owner.SERVER:
                raise FacadeProtocolError("upstream_protocol_error", "A protected item emitted public audio.")
            lease.item_id = item_id
            lease.public_item_id = item_record.local_id
            lease.content_index = content_index
            self._playback_leases_by_item[item_id] = lease
        elif lease.item_id != item_id or lease.content_index != content_index:
            raise FacadeProtocolError(
                "upstream_protocol_error", "A realtime response emitted more than one public audio stream."
            )
        return lease

    async def _observe_output_audio_delta(self, event: Mapping[str, Any]) -> None:
        """Track exact generated PCM duration and hold the local speech floor."""
        lease = self._playback_lease_for_audio_event(event)
        delta = event.get("delta")
        if not isinstance(delta, str) or not delta:
            raise FacadeProtocolError("upstream_protocol_error", "An output audio delta was invalid.")
        try:
            raw_audio = base64.b64decode(delta, validate=True)
        except (binascii.Error, ValueError) as error:
            raise FacadeProtocolError("upstream_protocol_error", "An output audio delta was invalid.") from error
        if not raw_audio or len(raw_audio) % _PCM16_BYTES_PER_SAMPLE:
            raise FacadeProtocolError("upstream_protocol_error", "An output PCM delta was not frame aligned.")
        lease.generated_samples += len(raw_audio) // _PCM16_BYTES_PER_SAMPLE
        if self._activity.output is not OutputActivity.SPEAKING:
            await self._set_activity(output=OutputActivity.SPEAKING)

    def _observe_output_audio_done(self, event: Mapping[str, Any]) -> None:
        """Record audio-generation completion without claiming browser playout."""
        self._playback_lease_for_audio_event(event).audio_done = True

    def _observe_output_audio_transcript(self, event: Mapping[str, Any]) -> None:
        """Retain bounded conversational text until browser playout is acknowledged."""
        lease = self._playback_lease_for_audio_event(event)
        if event.get("type") == "response.output_audio_transcript.done":
            text = event.get("transcript", event.get("text", ""))
            if isinstance(text, str) and "\x00" not in text:
                lease.transcript = text[:_MAX_OUTPUT_CHARACTERS]
            return
        delta = event.get("delta")
        if isinstance(delta, str) and "\x00" not in delta:
            lease.transcript = f"{lease.transcript}{delta}"[:_MAX_OUTPUT_CHARACTERS]

    async def _pump_downstream(self) -> None:
        while True:
            event = await self._receive_downstream()
            try:
                await self.handle_downstream_event(event)
            except FacadeProtocolError as error:
                await self._send_public_error(error.code, error.public_message, event_id=event.get("event_id"))

    async def _pump_upstream(self) -> None:
        while True:
            event = await self._receive_upstream()
            try:
                await self.handle_upstream_event(event)
            except FacadeProtocolError as error:
                # Event type and normalized code are safe operational evidence;
                # never log the upstream payload, prompts, identifiers, or media.
                _LOGGER.warning(
                    "Rejected upstream Realtime event type=%s code=%s",
                    event.get("type", "invalid"),
                    error.code,
                )
                raise

    async def _handle_session_updated(self, event: dict[str, Any]) -> None:
        session = event.get("session")
        candidate = self._pending_client_session
        if not isinstance(session, dict) or candidate is None:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend sent an unexpected session acknowledgement."
            )
        self._accepted_client_session = candidate
        self._pending_client_session = None
        self._pending_client_session_event_id = None
        self._pending_previous_automatic_turn_detection = None
        self._pending_previous_automatic_response = None
        instructions = candidate.get("instructions", "")
        self._client_session_instructions = instructions.strip() if isinstance(instructions, str) else ""
        self._capture_output_sample_rate(session)

        public_session = _strip_sensitive(session)
        assert isinstance(public_session, dict)
        public_session["id"] = self._facade_session_id
        public_session["model"] = self._public_model
        public_session["instructions"] = copy.deepcopy(candidate.get("instructions", ""))
        public_session["tools"] = copy.deepcopy(candidate.get("tools", []))
        # Keep the upstream acknowledgement's effective tool choice.  Protected
        # server-owned tools require ``auto`` at the frontend even when an
        # untrusted client requested ``none``.  Echoing the client candidate
        # here would make the public session acknowledgement contradict the
        # configuration that is actually active upstream.
        audio = public_session.get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        turn_detection = input_audio.get("turn_detection") if isinstance(input_audio, dict) else None
        if isinstance(turn_detection, dict):
            turn_detection["create_response"] = self._automatic_response
        await self._send_downstream(
            {
                "event_id": self._id_factory("event_vc"),
                "type": "session.updated",
                "session": public_session,
            }
        )

    def _capture_output_sample_rate(self, session: Mapping[str, Any]) -> None:
        """Capture the negotiated PCM rate used to validate playout boundaries."""
        audio = session.get("audio")
        output = audio.get("output") if isinstance(audio, Mapping) else None
        audio_format = output.get("format") if isinstance(output, Mapping) else None
        if not isinstance(audio_format, Mapping):
            return
        format_type = audio_format.get("type")
        rate = audio_format.get("rate", audio_format.get("sample_rate"))
        if format_type != "audio/pcm" or isinstance(rate, bool) or not isinstance(rate, int):
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend negotiated an unsupported output format."
            )
        if not 8_000 <= rate <= 192_000:
            raise FacadeProtocolError(
                "upstream_protocol_error", "The realtime frontend negotiated an invalid output sample rate."
            )
        self._output_sample_rate = rate

    async def _enqueue_response_create(
        self,
        event: dict[str, Any],
        *,
        purpose: _ResponsePurpose = _ResponsePurpose.INTERACTIVE,
        requires_speech_floor: bool = False,
        capacity_reserved: bool = False,
        speech_purpose: str | None = None,
        local_request_id: str | None = None,
        client_response_instructions: str = "",
        server_response_context: Mapping[str, str] | None = None,
        finalized_item_id: str | None = None,
        finalized_user_text: str | None = None,
        finalized_user_item_id: str | None = None,
        bind_pending_user_turn: bool = True,
    ) -> None:
        """Queue one response request and send it only when the upstream is idle."""
        event_id = event.get("event_id")
        if event_id is None:
            event_id = self._id_factory("event_vc")
            event["event_id"] = event_id
        if (
            not isinstance(event_id, str)
            or not event_id
            or len(event_id) > _MAX_IDENTIFIER_CHARACTERS
            or any(ord(character) < 32 or ord(character) == 127 for character in event_id)
        ):
            raise FacadeProtocolError("invalid_request", "response.create requires a valid event id.")
        response_template = event.get("response")
        if not isinstance(response_template, dict):
            raise FacadeProtocolError("invalid_request", "response.create requires a response object.")
        queued = _QueuedResponseCreate(
            event=event,
            response_template=copy.deepcopy(response_template),
            purpose=purpose,
            priority={
                _ResponsePurpose.ACKNOWLEDGEMENT: 0,
                _ResponsePurpose.DIRECT_REPLY: 0,
                _ResponsePurpose.INTERACTIVE: 10,
                _ResponsePurpose.DELIVERY: 100,
            }[purpose],
            requires_speech_floor=requires_speech_floor,
            speech_purpose=speech_purpose,
            local_request_id=local_request_id,
            presentation_id=self._id_factory("pres_vc"),
            playback_receipt_id=self._id_factory("receipt_vc"),
            client_response_instructions=client_response_instructions,
            server_response_context=(None if server_response_context is None else dict(server_response_context)),
            finalized_item_id=finalized_item_id,
            finalized_user_text=finalized_user_text,
            finalized_user_item_id=finalized_user_item_id,
            awaits_finalized_user_text=finalized_item_id is not None and finalized_user_text is None,
        )
        rejected_manual_response: tuple[str, str] | None = None
        async with self._response_state_lock:
            if (
                purpose is _ResponsePurpose.INTERACTIVE
                and finalized_user_text is not None
                and not bind_pending_user_turn
            ):
                self._retire_rejected_unpaired_manual_commits_locked()
            manual_commit: _ManualAudioCommit | None = None
            bind_interactive_turn = (
                purpose is _ResponsePurpose.INTERACTIVE and queued.finalized_item_id is None and bind_pending_user_turn
            )
            if bind_interactive_turn:
                manual_commit = next(
                    (commit for commit in self._manual_audio_commits if commit.response is None),
                    None,
                )
                if manual_commit is not None and manual_commit.rejected:
                    self._manual_audio_commits.remove(manual_commit)
                    if manual_commit.finalized_item_id is not None:
                        with suppress(ValueError):
                            self._unbound_audio_items.remove(manual_commit.finalized_item_id)
                    rejected_manual_response = (
                        manual_commit.rejection_code,
                        manual_commit.rejection_message,
                    )
            if rejected_manual_response is None:
                bounded_purposes = (
                    _SERVER_SPEECH_PURPOSES
                    if purpose in _SERVER_SPEECH_PURPOSES
                    else frozenset({_ResponsePurpose.INTERACTIVE})
                )
                pending_depth = sum(item.purpose in bounded_purposes for item in self._response_create_queue)
                if not capacity_reserved and pending_depth >= self._max_pending_speech:
                    raise FacadeProtocolError(
                        "session_capacity_exceeded", "The realtime session has too many pending responses."
                    )
                if finalized_user_item_id is not None and not bind_pending_user_turn:
                    with suppress(ValueError):
                        self._pending_typed_user_turns.remove((finalized_user_item_id, finalized_user_text))
                if bind_interactive_turn:
                    if manual_commit is not None:
                        manual_commit.response = queued
                        if manual_commit.finalized_item_id is None:
                            queued.awaits_manual_audio_item = True
                        else:
                            queued.finalized_item_id = manual_commit.finalized_item_id
                            queued.finalized_user_text = manual_commit.finalized_user_text
                            queued.awaits_finalized_user_text = manual_commit.finalized_user_text is None
                            self._manual_audio_commits.remove(manual_commit)
                    elif self._pending_typed_user_turns:
                        queued.finalized_user_item_id, queued.finalized_user_text = (
                            self._pending_typed_user_turns.popleft()
                        )
                    elif self._pending_finalized_user_text is not None:
                        queued.finalized_user_text = self._pending_finalized_user_text
                        self._pending_finalized_user_text = None
                    elif self._unbound_audio_items:
                        queued.finalized_item_id = self._unbound_audio_items[0]
                        queued.awaits_finalized_user_text = True
                if queued.finalized_item_id is not None:
                    with suppress(ValueError):
                        self._unbound_audio_items.remove(queued.finalized_item_id)
                if purpose is _ResponsePurpose.INTERACTIVE and self._deferred_client_item_creates:
                    queued.prerequisite_item_creates = list(self._deferred_client_item_creates)
                    self._deferred_client_item_creates.clear()
                insertion = next(
                    (
                        index
                        for index, existing in enumerate(self._response_create_queue)
                        if existing.priority > queued.priority
                    ),
                    len(self._response_create_queue),
                )
                self._response_create_queue.insert(insertion, queued)
        if rejected_manual_response is not None:
            await self._send_public_error(
                rejected_manual_response[0],
                rejected_manual_response[1],
                event_id=event_id,
            )
        elif queued.awaits_finalized_user_text and queued.finalized_item_id is not None:
            self._arm_finalized_audio_timeout(queued.finalized_item_id)
        await self._dispatch_next_response_create()

    def _client_item_event_id_in_use(self, event_id: str) -> bool:
        """Return whether an item-create id is pending on either side of a response barrier."""
        if event_id in self._pending_client_item_creates:
            return True
        if any(item.event_id == event_id for item in self._deferred_client_item_creates):
            return True
        candidates = [*self._response_create_queue]
        if self._response_create_in_flight is not None:
            candidates.append(self._response_create_in_flight)
        return any(item.event_id == event_id for queued in candidates for item in queued.prerequisite_item_creates)

    async def _defer_client_item_create_if_response_blocked(
        self,
        event: dict[str, Any],
        *,
        event_id: str,
        item_id: str,
    ) -> bool:
        """Keep a later item from overtaking an earlier unsent response.create."""
        async with self._response_state_lock:
            in_flight = self._response_create_in_flight
            response_barrier = (
                bool(self._response_admission_barriers)
                or bool(self._response_create_queue)
                or (in_flight is not None and not in_flight.sent_upstream)
            )
            if not response_barrier:
                return False
            self._deferred_client_item_creates.append(
                _DeferredClientItemCreate(event=event, event_id=event_id, item_id=item_id)
            )
            return True

    async def _flush_deferred_client_item_creates_if_unblocked(self) -> None:
        """Forward loose client items once every earlier response request is on the wire."""
        async with self._response_state_lock:
            in_flight = self._response_create_in_flight
            if (
                self._response_admission_barriers
                or self._response_create_queue
                or (in_flight is not None and not in_flight.sent_upstream)
            ):
                return
            deferred = list(self._deferred_client_item_creates)
            self._deferred_client_item_creates.clear()
        if not deferred:
            return
        for item in deferred:
            self._pending_client_item_creates[item.event_id] = item.item_id
        try:
            await self._send_upstream_batch([item.event for item in deferred])
        except BaseException:
            for item in deferred:
                self._pending_client_item_creates.pop(item.event_id, None)
                await self._discard_finalized_typed_item(item.item_id)
                self._registry.retire_item(item.item_id)
            raise

    def _apply_turn_routing(self, queued: _QueuedResponseCreate) -> None:
        """Apply one Interaction Manager directive to the private response only."""
        if queued.routing_applied or queued.purpose is not _ResponsePurpose.INTERACTIVE:
            return
        text = queued.finalized_user_text
        if text is None:
            # Client-created responses without a committed user turn retain
            # standard generation behavior but cannot invoke protected Work.
            response = queued.event.get("response")
            if not isinstance(response, dict):
                raise FacadeProtocolError("invalid_request", "response.create requires a response object.")
            response["tool_choice"] = "none"
            queued.direct_route = True
            queued.routing_applied = True
            return
        try:
            directive = self._runtime.route_finalized_turn(self._facade_session_id, text)
        except Exception as error:
            raise FacadeProtocolError(
                "routing_unavailable",
                "VoiceClaw could not route the finalized request.",
            ) from error
        if directive.kind is TurnDirectiveKind.REJECT:
            raise FacadeProtocolError(
                directive.reason_code,
                "VoiceClaw could not admit the finalized request.",
            )
        response = queued.event.get("response")
        if not isinstance(response, dict):
            raise FacadeProtocolError("invalid_request", "response.create requires a response object.")
        if directive.kind is TurnDirectiveKind.DIRECT:
            response["tool_choice"] = "none"
            queued.direct_route = True
        elif directive.kind is TurnDirectiveKind.TOOL:
            if directive.logical_tool is None:
                raise FacadeProtocolError("routing_unavailable", "VoiceClaw returned an invalid route.")
            try:
                protected = ProtectedTool.from_logical_name(directive.logical_tool)
            except ValueError as error:
                raise FacadeProtocolError("routing_unavailable", "VoiceClaw returned an invalid route.") from error
            if protected not in self._tools.enabled:
                raise FacadeProtocolError(
                    "capability_unsupported",
                    "The selected backend operation is unavailable.",
                )
            # A live runtime update may replace a protected schema after the
            # upstream session was bootstrapped. Bind the complete current
            # snapshot to this response instead of relying on stale session
            # defaults.
            response["tools"] = list(copy.deepcopy(self._tools.schemas()))
            response["tool_choice"] = {"type": "function", "name": protected.value}
            queued.expected_protected_tool = protected
        elif directive.kind is TurnDirectiveKind.AUTO:
            # Run classification as a silent structured model turn. A mixed
            # speech/tool response cannot be made safe after streaming starts:
            # the model may narrate private routing reasoning before it calls
            # the backend tool. ``required`` makes the same frontend model pick
            # either the direct-route control or one capability-gated Work
            # operation without generating user-facing audio.
            if ProtectedTool.CONVERSATION_RESPOND not in self._tools.enabled:
                response["tool_choice"] = "none"
            else:
                direct_response = copy.deepcopy(response)
                direct_response.pop("instructions", None)
                direct_response.pop("tools", None)
                direct_response.pop("tool_choice", None)
                direct_response.pop("parallel_tool_calls", None)
                queued.direct_reply = _DirectReplyTemplate(
                    event={
                        "event_id": self._id_factory("event_vc"),
                        "type": "response.create",
                        "response": direct_response,
                    },
                    client_response_instructions=queued.client_response_instructions,
                )
                response.pop("audio", None)
                response["output_modalities"] = ["text"]
                # Allow a standalone multi-turn goal without truncating protected-call arguments.
                response["max_output_tokens"] = _MAX_REALTIME_RESPONSE_TOKENS
                response["tools"] = list(copy.deepcopy(self._tools.schemas()))
                response["tool_choice"] = "required"
                response["parallel_tool_calls"] = False
                queued.model_route_selection = True
        else:
            raise FacadeProtocolError("routing_unavailable", "VoiceClaw returned an invalid route.")
        queued.routing_applied = True

    @staticmethod
    def _reset_response_create_for_retry(queued: _QueuedResponseCreate) -> None:
        """Discard a stale prepared route before returning a response to the queue."""
        queued.event["response"] = copy.deepcopy(queued.response_template)
        queued.routing_applied = False
        queued.direct_route = False
        queued.expected_protected_tool = None
        queued.model_route_selection = False
        queued.direct_reply = None

    def _refresh_response_instructions(self, queued: _QueuedResponseCreate) -> None:
        """Build the authoritative prompt at the actual model boundary."""
        response = queued.event.get("response")
        if not isinstance(response, dict):
            raise FacadeProtocolError("invalid_request", "response.create requires a response object.")
        if queued.purpose in _SERVER_SPEECH_PURPOSES:
            response["tools"] = []
            response["tool_choice"] = "none"
        try:
            response["instructions"] = self._instruction_builder.build(
                static_instructions=self._static_instructions,
                client_session_instructions=self._client_session_instructions,
                client_response_instructions=queued.client_response_instructions,
                dynamic_projection=self._current_projection(),
                response_context=queued.server_response_context,
            )
        except ValueError as error:
            raise FacadeProtocolError("invalid_request", "response instructions are too large.") from error

    def _claim_next_response_create_locked(self) -> _QueuedResponseCreate | None:
        if (
            self._response_terminal_fence
            or self._response_admission_barriers
            or self._active_upstream_response_id is not None
            or self._response_create_in_flight is not None
            or not self._response_create_queue
        ):
            return None
        queued = self._response_create_queue[0]
        if queued.awaits_manual_audio_item or queued.awaits_finalized_user_text:
            return None
        if queued.requires_speech_floor and not self._activity.speech_floor_available:
            return None
        self._refresh_response_instructions(queued)
        self._apply_turn_routing(queued)
        self._response_create_queue.popleft()
        self._response_create_in_flight = queued
        return queued

    async def _register_manual_audio_commit(self, event_id: object) -> None:
        """Reserve the next interactive response for one explicit audio commit."""
        if (
            not isinstance(event_id, str)
            or not event_id
            or len(event_id) > _MAX_IDENTIFIER_CHARACTERS
            or any(ord(character) < 32 or ord(character) == 127 for character in event_id)
        ):
            raise FacadeProtocolError("invalid_request", "input_audio_buffer.commit requires a valid event id.")
        async with self._response_state_lock:
            self._retire_rejected_unpaired_manual_commits_locked()
            if len(self._manual_audio_commits) >= self._max_pending_speech:
                raise FacadeProtocolError(
                    "session_capacity_exceeded", "The realtime session has too many pending audio commits."
                )
            if any(commit.event_id == event_id for commit in self._manual_audio_commits):
                raise FacadeProtocolError("invalid_request", "input_audio_buffer.commit requires a unique event id.")
            self._manual_audio_commits.append(
                _ManualAudioCommit(event_id=event_id, input_generation=self._active_input_generation)
            )

    async def _retire_rejected_unpaired_manual_commits(self) -> None:
        """Retire rejected commits when an ordered newer user-turn boundary arrives."""
        async with self._response_state_lock:
            self._retire_rejected_unpaired_manual_commits_locked()

    def _retire_rejected_unpaired_manual_commits_locked(self) -> None:
        """Retire rejected, response-less commit tails while holding the response lock."""
        # Downstream WebSocket events are ordered. If a definite newer user
        # turn arrives before a rejected commit's response.create, the older
        # response was never sent; discard that tombstone so the new response
        # cannot be consumed by stale state. A true late paired response is
        # ordered before the newer turn and consumes the tombstone normally.
        for commit in tuple(self._manual_audio_commits):
            if not commit.rejected or commit.response is not None:
                continue
            self._manual_audio_commits.remove(commit)
            if commit.finalized_item_id is not None:
                with suppress(ValueError):
                    self._unbound_audio_items.remove(commit.finalized_item_id)
                self._cancel_finalized_audio_timeout(commit.finalized_item_id)
                self._failed_audio_items.discard(commit.finalized_item_id)

    async def _bind_manual_audio_commit(self, finalized_item_id: str) -> bool:
        """Bind a commit acknowledgement only to its order-paired response."""
        response_ready = False
        awaiting_transcript = False
        async with self._response_state_lock:
            manual_commit = next(
                (commit for commit in self._manual_audio_commits if commit.finalized_item_id is None),
                None,
            )
            if manual_commit is None:
                return False
            manual_commit.finalized_item_id = finalized_item_id
            if manual_commit.input_generation is not None:
                self._audio_item_input_generation[finalized_item_id] = manual_commit.input_generation
            queued = manual_commit.response
            if queued is not None:
                queued.finalized_item_id = finalized_item_id
                queued.awaits_manual_audio_item = False
                queued.awaits_finalized_user_text = queued.finalized_user_text is None
                with suppress(ValueError):
                    self._unbound_audio_items.remove(finalized_item_id)
                self._manual_audio_commits.remove(manual_commit)
                response_ready = not queued.awaits_finalized_user_text
                awaiting_transcript = queued.awaits_finalized_user_text
        if awaiting_transcript:
            self._arm_finalized_audio_timeout(finalized_item_id)
        if response_ready:
            await self._dispatch_next_response_create()
        return True

    async def _dispatch_next_response_create(self) -> None:
        routing_failure: tuple[_QueuedResponseCreate, FacadeProtocolError] | None = None
        async with self._response_state_lock:
            try:
                queued = self._claim_next_response_create_locked()
            except FacadeProtocolError as error:
                failed = self._response_create_queue.popleft()
                routing_failure = (failed, error)
                queued = None
        if routing_failure is not None:
            failed, error = routing_failure
            await self._retire_unsent_response_create(failed)
            await self._publish_delivery_queue_state()
            event_id = failed.event.get("event_id")
            await self._send_public_error(
                error.code,
                error.public_message,
                event_id=event_id if isinstance(event_id, str) else None,
            )
            await self._dispatch_next_response_create()
            return
        await self._publish_delivery_queue_state()
        if queued is None:
            return
        async with self._speech_admission_lock:
            if queued.requires_speech_floor and not self._activity.speech_floor_available:
                # Activity changed after the earlier claim but before the wire
                # send. Return the same prepared response to its priority slot;
                # a later idle transition will retry it with a fresh check.
                async with self._response_state_lock:
                    if self._response_create_in_flight is queued:
                        self._response_create_in_flight = None
                        self._reset_response_create_for_retry(queued)
                        insertion = next(
                            (
                                index
                                for index, existing in enumerate(self._response_create_queue)
                                if existing.priority > queued.priority
                            ),
                            len(self._response_create_queue),
                        )
                        self._response_create_queue.insert(insertion, queued)
                return
            for prerequisite in queued.prerequisite_item_creates:
                self._pending_client_item_creates[prerequisite.event_id] = prerequisite.item_id
            try:
                await self._send_upstream_batch(
                    [*(item.event for item in queued.prerequisite_item_creates), queued.event]
                )
            except BaseException:
                async with self._response_state_lock:
                    if self._response_create_in_flight is queued:
                        self._response_create_in_flight = None
                for prerequisite in queued.prerequisite_item_creates:
                    self._pending_client_item_creates.pop(prerequisite.event_id, None)
                    await self._discard_finalized_typed_item(prerequisite.item_id)
                    self._registry.retire_item(prerequisite.item_id)
                await self._publish_delivery_queue_state()
                raise
        async with self._response_state_lock:
            queued.sent_upstream = True
            queued.prerequisite_item_creates.clear()
        await self._flush_deferred_client_item_creates_if_unblocked()

    async def _reserve_admission_barrier(self, upstream_call_id: str) -> None:
        """Fence later turns behind one protected call's prompt local outcome."""
        async with self._response_state_lock:
            if upstream_call_id in self._response_admission_barriers:
                raise FacadeProtocolError("upstream_protocol_error", "A protected call reused its admission boundary.")
            self._response_admission_barriers.add(upstream_call_id)

    async def _release_admission_barrier(self, upstream_call_id: str) -> None:
        """Release a call fence after its local outcome and follow-up are queued."""
        async with self._response_state_lock:
            removed = upstream_call_id in self._response_admission_barriers
            self._response_admission_barriers.discard(upstream_call_id)
        if removed:
            await self._dispatch_next_response_create()
            await self._flush_deferred_client_item_creates_if_unblocked()

    async def _retire_unsent_response_create(self, queued: _QueuedResponseCreate) -> None:
        """Release private reservations owned by one response rejected before send."""
        if queued.finalized_item_id is not None:
            self._cancel_finalized_audio_timeout(queued.finalized_item_id)
            self._failed_audio_items.discard(queued.finalized_item_id)
        for prerequisite in queued.prerequisite_item_creates:
            await self._discard_finalized_typed_item(prerequisite.item_id)
            self._registry.retire_item(prerequisite.item_id)
        queued.prerequisite_item_creates.clear()

    async def _mark_response_created(self, upstream_response_id: str) -> bool:
        """Bind one response to its prompt boundary and optional browser playout lease."""
        finalized_turn: tuple[str, str] | None = None
        async with self._response_state_lock:
            if self._active_upstream_response_id is not None:
                raise FacadeProtocolError(
                    "upstream_protocol_error", "The realtime frontend started overlapping responses."
                )
            in_flight = self._response_create_in_flight
            self._response_create_in_flight = None
            self._active_upstream_response_id = upstream_response_id
            self._active_upstream_response_purpose = None if in_flight is None else in_flight.purpose
            self._active_speech_purpose = None if in_flight is None else in_flight.speech_purpose
            self._active_speech_local_request_id = None if in_flight is None else in_flight.local_request_id
            self._active_presentation_id = (
                self._id_factory("pres_vc")
                if in_flight is None or in_flight.presentation_id is None
                else in_flight.presentation_id
            )
            self._active_playback_receipt_id = (
                self._id_factory("receipt_vc")
                if in_flight is None or in_flight.playback_receipt_id is None
                else in_flight.playback_receipt_id
            )
            self._active_speech_delivery_mode = (
                None
                if in_flight is None or in_flight.speech_purpose not in _APPLICATION_SPEECH_PURPOSES
                else "model_mediated"
            )
            if in_flight is not None and in_flight.expected_protected_tool is not None:
                self._response_expected_protected_tool[upstream_response_id] = in_flight.expected_protected_tool
            if in_flight is not None and (in_flight.direct_route or in_flight.purpose in _SERVER_SPEECH_PURPOSES):
                self._response_protected_calls_forbidden.add(upstream_response_id)
            if in_flight is not None and in_flight.model_route_selection:
                self._response_model_route_selection.add(upstream_response_id)
                if in_flight.direct_reply is None:
                    raise FacadeProtocolError(
                        "upstream_protocol_error", "A model-selected route lost its direct-response template."
                    )
                self._response_direct_reply[upstream_response_id] = in_flight.direct_reply
            if (
                in_flight is not None
                and in_flight.purpose is _ResponsePurpose.INTERACTIVE
                and in_flight.finalized_item_id is not None
            ):
                source_item_id = in_flight.finalized_item_id
                self._audio_item_response[source_item_id] = upstream_response_id
                self._response_audio_item[upstream_response_id] = source_item_id
                if in_flight.finalized_user_text is None:
                    waiter = asyncio.get_running_loop().create_future()
                    self._late_finalized_text_waiters[upstream_response_id] = waiter
                    if source_item_id in self._failed_audio_items:
                        waiter.set_result(None)
            if (
                in_flight is not None
                and in_flight.purpose is _ResponsePurpose.INTERACTIVE
                and in_flight.finalized_user_text is not None
            ):
                self._response_finalized_user_text[upstream_response_id] = in_flight.finalized_user_text
                if in_flight.finalized_user_item_id is not None:
                    self._response_finalized_user_item[upstream_response_id] = in_flight.finalized_user_item_id
                    finalized_turn = (in_flight.finalized_user_item_id, in_flight.finalized_user_text)
                else:
                    finalized_turn = (self._id_factory("turn_vc"), in_flight.finalized_user_text)
            elif in_flight is None and self._pending_typed_user_turns:
                # Preserve ordered item identity even for a compatible frontend
                # that starts the response before acknowledging our create.
                item_id, text = self._pending_typed_user_turns.popleft()
                self._response_finalized_user_text[upstream_response_id] = text
                self._response_finalized_user_item[upstream_response_id] = item_id
                finalized_turn = (item_id, text)
            elif in_flight is None and self._pending_finalized_user_text is not None:
                # Some compatible frontends create a response automatically
                # after a committed item instead of acknowledging an explicit
                # response.create request. Correlate that response only when
                # there is no facade-owned request in flight.
                self._response_finalized_user_text[upstream_response_id] = self._pending_finalized_user_text
                finalized_turn = (self._id_factory("turn_vc"), self._pending_finalized_user_text)
                self._pending_finalized_user_text = None
            is_delivery = self._active_upstream_response_purpose is _ResponsePurpose.DELIVERY
            model_route_selection = upstream_response_id in self._response_model_route_selection
            presentation_id = self._active_presentation_id
            receipt_id = self._active_playback_receipt_id
            local_request_id = self._active_speech_local_request_id
            speech_purpose = self._active_speech_purpose

        if not model_route_selection and presentation_id is not None and receipt_id is not None:
            if len(self._playback_leases_by_response) >= _MAX_TRACKED_RESPONSES:
                raise FacadeProtocolError(
                    "session_capacity_exceeded", "The realtime session has too many pending playback receipts."
                )
            self._playback_leases_by_response[upstream_response_id] = _PlaybackLease(
                response_id=upstream_response_id,
                public_response_id=self._registry.response(upstream_response_id),
                presentation_id=presentation_id,
                receipt_id=receipt_id,
                local_request_id=local_request_id,
                speech_purpose=speech_purpose,
                sample_rate=self._output_sample_rate,
            )
        if finalized_turn is not None:
            self._record_user_turn(*finalized_turn)
        return is_delivery

    def _record_user_turn(self, item_id: str, text: str) -> None:
        """Record one finalized public user turn exactly once."""
        try:
            public_item_id = self._registry.item(item_id).local_id
        except FacadeProtocolError:
            public_item_id = item_id
        if public_item_id in self._recorded_user_turn_ids:
            return
        try:
            self._runtime.record_conversation_turn(
                self._facade_session_id,
                FrontendConversationTurn(
                    turn_id=public_item_id,
                    role="user",
                    text=text,
                    delivery_state=FrontendConversationDeliveryState.COMMITTED,
                ),
            )
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not record the user turn."
            ) from error
        self._recorded_user_turn_ids.add(public_item_id)

    def _record_assistant_turn(self, lease: _PlaybackLease) -> None:
        """Record transcript-backed assistant delivery without claiming it was heard."""
        text = lease.transcript.strip()
        if not text or lease.public_item_id is None:
            return
        try:
            self._runtime.record_conversation_turn(
                self._facade_session_id,
                FrontendConversationTurn(
                    turn_id=lease.public_item_id,
                    role="assistant",
                    text=text,
                    delivery_state=FrontendConversationDeliveryState.DELIVERED,
                    presentation_id=lease.presentation_id,
                    local_request_id=lease.local_request_id,
                ),
            )
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not record the assistant turn."
            ) from error

    async def _mark_response_terminal(self, upstream_response_id: str) -> None:
        async with self._response_state_lock:
            if self._active_upstream_response_id != upstream_response_id:
                raise FacadeProtocolError(
                    "upstream_protocol_error", "The realtime frontend completed an inactive response."
                )
            # Close the response and take any truncations that arrived after
            # the earlier response.done drain as one atomic boundary.  Without
            # this final take, the downstream and upstream pumps can interleave
            # between that drain and terminal retirement, leaving a late
            # truncation stranded while the next response is dispatched.
            pending_truncations = self._deferred_active_truncations.pop(upstream_response_id, deque())
            self._response_finalized_user_text.pop(upstream_response_id, None)
            self._response_finalized_user_item.pop(upstream_response_id, None)
            self._response_expected_protected_tool.pop(upstream_response_id, None)
            self._response_protected_calls_forbidden.discard(upstream_response_id)
            self._response_model_route_selection.discard(upstream_response_id)
            self._response_direct_reply.pop(upstream_response_id, None)
            self._active_upstream_response_id = None
            self._active_upstream_response_purpose = None
            self._active_speech_purpose = None
            self._active_speech_local_request_id = None
            self._active_speech_delivery_mode = None
            self._active_presentation_id = None
            self._active_playback_receipt_id = None
            self._response_terminal_fence = True
        try:
            await self._send_deferred_truncations(pending_truncations)
        finally:
            async with self._response_state_lock:
                self._response_terminal_fence = False
        await self._dispatch_next_response_create()

    async def _handle_response_create_error(self, event: Mapping[str, Any]) -> None:
        error = event.get("error")
        failed_event_id = error.get("event_id") if isinstance(error, Mapping) else None
        rejected: _QueuedResponseCreate | None = None
        async with self._response_state_lock:
            in_flight = self._response_create_in_flight
            if in_flight is None or failed_event_id != in_flight.event.get("event_id"):
                return
            rejected = in_flight
            self._response_create_in_flight = None
        await self._emit_speech_delivery_failure(
            purpose=rejected.purpose,
            speech_purpose=rejected.speech_purpose,
            local_request_id=rejected.local_request_id,
            phase="failed",
            reason="response_create_rejected",
        )
        await self._publish_delivery_queue_state()
        await self._dispatch_next_response_create()

    async def _handle_manual_audio_commit_error(self, event: Mapping[str, Any]) -> None:
        """Discard only the reservation and response paired with a rejected commit."""
        error = event.get("error")
        failed_event_id = error.get("event_id") if isinstance(error, Mapping) else None
        if not isinstance(failed_event_id, str):
            return
        removed_response_event_id = await self._reject_manual_audio_commit(failed_event_id)
        if removed_response_event_id is not None:
            await self._send_public_error(
                "upstream_error",
                "The realtime frontend rejected the associated audio commit.",
                event_id=removed_response_event_id,
            )
        await self._dispatch_next_response_create()

    async def _reject_manual_audio_commit(self, failed_event_id: str) -> str | None:
        """Mark an exact rejected commit and detach any response already paired to it."""
        removed_response_event_id: str | None = None
        rejected_generation: int | None = None
        async with self._response_state_lock:
            manual_commit = next(
                (commit for commit in self._manual_audio_commits if commit.event_id == failed_event_id),
                None,
            )
            if manual_commit is None:
                return None
            rejected_generation = manual_commit.input_generation
            queued = manual_commit.response
            if queued is None:
                # The upstream and browser pumps run concurrently. Retain an
                # ordered tombstone until the already-sent paired response.create
                # reaches the facade, then consume both without forwarding it.
                manual_commit.rejected = True
            else:
                self._manual_audio_commits.remove(manual_commit)
                try:
                    self._response_create_queue.remove(queued)
                except ValueError:
                    pass
                else:
                    event_id = queued.event.get("event_id")
                    if isinstance(event_id, str):
                        removed_response_event_id = event_id
        if rejected_generation is not None:
            await self._finish_input_generation(rejected_generation)
        return removed_response_event_id

    async def _correlated_client_error_event_id(self, event: Mapping[str, Any]) -> str | None:
        """Return an upstream error correlation only when it names a pending client event."""
        error = event.get("error")
        failed_event_id = error.get("event_id") if isinstance(error, Mapping) else None
        if (
            not isinstance(failed_event_id, str)
            or not failed_event_id
            or len(failed_event_id) > _MAX_IDENTIFIER_CHARACTERS
            or any(ord(character) < 32 or ord(character) == 127 for character in failed_event_id)
        ):
            return None
        if failed_event_id in self._pending_client_item_creates:
            return failed_event_id
        if failed_event_id == self._pending_client_session_event_id:
            return failed_event_id
        if failed_event_id in self._pending_passthrough_client_events:
            self._pending_passthrough_client_events.pop(failed_event_id, None)
            return failed_event_id
        async with self._response_state_lock:
            if any(commit.event_id == failed_event_id for commit in self._manual_audio_commits):
                return failed_event_id
            in_flight = self._response_create_in_flight
            if in_flight is not None and in_flight.event.get("event_id") == failed_event_id:
                return failed_event_id
        return None

    def _track_passthrough_client_event(
        self,
        event: dict[str, Any],
        *,
        event_type: str,
        target_id: str,
    ) -> str:
        """Remember a bounded cancel/truncate correlation until acknowledgement."""
        event_id = event.get("event_id")
        if event_id is None:
            event_id = self._id_factory("event_vc")
            event["event_id"] = event_id
        if (
            not isinstance(event_id, str)
            or not event_id
            or len(event_id) > _MAX_IDENTIFIER_CHARACTERS
            or any(ord(character) < 32 or ord(character) == 127 for character in event_id)
            or event_id in self._pending_passthrough_client_events
        ):
            raise FacadeProtocolError("invalid_request", f"{event_type} requires a unique valid event id.")
        if len(self._pending_passthrough_client_events) >= _MAX_TRACKED_ITEMS:
            raise FacadeProtocolError(
                "session_capacity_exceeded", "The realtime session has too many pending client operations."
            )
        self._pending_passthrough_client_events[event_id] = (event_type, target_id)
        return event_id

    async def _defer_active_truncation(self, response_id: str, event: dict[str, Any]) -> bool:
        """Hold an active-item truncation behind its response terminal boundary."""
        async with self._response_state_lock:
            if self._active_upstream_response_id != response_id:
                return False
            pending = self._deferred_active_truncations.setdefault(response_id, deque())
            pending.append(copy.deepcopy(event))
            return True

    async def _flush_deferred_truncations(self, response_id: str) -> None:
        """Send terminal-safe truncations before admitting the next response."""
        async with self._response_state_lock:
            pending = self._deferred_active_truncations.pop(response_id, deque())
        await self._send_deferred_truncations(pending)

    async def _send_deferred_truncations(self, pending: deque[dict[str, Any]]) -> None:
        """Forward an already-claimed truncation batch in client event order."""
        for event in pending:
            try:
                await self._send_upstream(event)
            except BaseException:
                event_id = event.get("event_id")
                if isinstance(event_id, str):
                    self._pending_passthrough_client_events.pop(event_id, None)
                    self._abandon_playback_receipt(event_id)
                raise

    def _acknowledge_passthrough_client_event(self, *, event_type: str, target_id: str) -> str | None:
        """Retire accepted passthrough operations without relying on echoed event ids."""
        for event_id, operation in tuple(self._pending_passthrough_client_events.items()):
            if operation == (event_type, target_id):
                self._pending_passthrough_client_events.pop(event_id, None)
                return event_id
        return None

    def _validate_playback_receipt(
        self,
        event: Mapping[str, Any],
        upstream_item_id: str,
    ) -> _PendingPlaybackReceipt | None:
        """Validate a standard truncation as same-session browser playout evidence."""
        lease = self._playback_leases_by_item.get(upstream_item_id)
        if lease is None:
            return None
        event_id = event.get("event_id")
        if event_id != lease.receipt_id or event_id in self._retired_playback_receipt_id_set:
            raise FacadeProtocolError(
                "invalid_playback_receipt", "The playback receipt does not belong to this audio response."
            )
        if lease.terminal or lease.pending_receipt_id is not None:
            raise FacadeProtocolError("invalid_playback_receipt", "The playback receipt is already terminal.")
        content_index = event.get("content_index")
        if (
            isinstance(content_index, bool)
            or not isinstance(content_index, int)
            or content_index != lease.content_index
        ):
            raise FacadeProtocolError("invalid_playback_receipt", "The playback receipt content index is invalid.")
        heard_through_ms = event.get("audio_end_ms")
        if isinstance(heard_through_ms, bool) or not isinstance(heard_through_ms, int) or heard_through_ms < 0:
            raise FacadeProtocolError("invalid_playback_receipt", "The playback receipt boundary is invalid.")
        audio_end_ms = lease.audio_end_ms
        if heard_through_ms > audio_end_ms:
            raise FacadeProtocolError("invalid_playback_receipt", "The playback receipt exceeds generated audio.")
        if lease.public_item_id is None:
            raise FacadeProtocolError("invalid_playback_receipt", "The playback receipt has no public audio item.")
        fully_played = (
            lease.response_done
            and lease.audio_done
            and lease.generated_samples > 0
            and heard_through_ms == audio_end_ms
        )
        return _PendingPlaybackReceipt(
            receipt_id=event_id,
            response_id=lease.response_id,
            presentation_id=lease.presentation_id,
            public_item_id=lease.public_item_id,
            local_request_id=lease.local_request_id,
            heard_through_ms=heard_through_ms,
            audio_end_ms=audio_end_ms,
            fully_played=fully_played,
        )

    def _abandon_playback_receipt(self, receipt_id: str) -> None:
        """Release a locally claimed receipt when its wire send did not occur."""
        pending = self._pending_playback_receipts.pop(receipt_id, None)
        if pending is None:
            return
        lease = self._playback_leases_by_response.get(pending.response_id)
        if lease is not None and lease.pending_receipt_id == receipt_id:
            lease.pending_receipt_id = None

    async def _acknowledge_playback_receipt(self, receipt_id: str) -> None:
        """Apply only an upstream-acknowledged standard playback boundary."""
        pending = self._pending_playback_receipts.pop(receipt_id, None)
        if pending is None:
            return
        lease = self._playback_leases_by_response.get(pending.response_id)
        if lease is None or lease.pending_receipt_id != receipt_id or lease.terminal:
            raise FacadeProtocolError("upstream_protocol_error", "A playback receipt lost its active lease.")
        lease.pending_receipt_id = None
        lease.terminal = True
        receipt_state = (
            FrontendPlaybackReceiptState.HEARD if pending.fully_played else FrontendPlaybackReceiptState.INTERRUPTED
        )
        try:
            self._runtime.record_playback_receipt(
                self._facade_session_id,
                FrontendPlaybackReceipt(
                    turn_id=pending.public_item_id,
                    state=receipt_state,
                    presentation_id=pending.presentation_id,
                    local_request_id=pending.local_request_id,
                    heard_through_ms=pending.heard_through_ms,
                    audio_end_ms=pending.audio_end_ms,
                ),
            )
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not record playback."
            ) from error
        self._retire_playback_lease(lease)
        if not pending.fully_played:
            await self._set_activity(output=OutputActivity.INTERRUPTED)
        await self._set_activity(output=OutputActivity.IDLE)

    async def _fail_playback_receipt(self, receipt_id: str) -> None:
        """Fail an upstream-rejected receipt without ever claiming audio was heard."""
        pending = self._pending_playback_receipts.pop(receipt_id, None)
        if pending is None:
            return
        lease = self._playback_leases_by_response.get(pending.response_id)
        if lease is None or lease.terminal:
            return
        lease.pending_receipt_id = None
        lease.terminal = True
        try:
            self._runtime.record_playback_receipt(
                self._facade_session_id,
                FrontendPlaybackReceipt(
                    turn_id=pending.public_item_id,
                    state=FrontendPlaybackReceiptState.FAILED,
                    presentation_id=pending.presentation_id,
                    local_request_id=pending.local_request_id,
                    heard_through_ms=pending.heard_through_ms,
                    audio_end_ms=pending.audio_end_ms,
                ),
            )
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not record playback."
            ) from error
        self._retire_playback_lease(lease)
        await self._set_activity(output=OutputActivity.INTERRUPTED)
        await self._set_activity(output=OutputActivity.IDLE)

    def _retire_playback_lease(self, lease: _PlaybackLease) -> None:
        """Release one terminal/no-audio lease while retaining bounded replay evidence."""
        self._playback_leases_by_response.pop(lease.response_id, None)
        if lease.item_id is not None:
            self._playback_leases_by_item.pop(lease.item_id, None)
        self._pending_playback_receipts.pop(lease.receipt_id, None)
        if lease.receipt_id not in self._retired_playback_receipt_id_set:
            self._retired_playback_receipt_ids.append(lease.receipt_id)
            self._retired_playback_receipt_id_set.add(lease.receipt_id)
        while len(self._retired_playback_receipt_ids) > _MAX_TRACKED_RESPONSES:
            expired = self._retired_playback_receipt_ids.popleft()
            self._retired_playback_receipt_id_set.discard(expired)

    def _acknowledge_client_item_create(self, upstream_item_id: object) -> None:
        """Stop awaiting rejection once the frontend accepts a client item."""
        item_id = _identifier(upstream_item_id, "upstream item id")
        for event_id, pending_item_id in tuple(self._pending_client_item_creates.items()):
            if pending_item_id == item_id:
                self._pending_client_item_creates.pop(event_id, None)

    async def _handle_client_item_create_error(self, event: Mapping[str, Any]) -> None:
        """Invalidate a finalized typed item rejected by its exact create event."""
        error = event.get("error")
        failed_event_id = error.get("event_id") if isinstance(error, Mapping) else None
        if not isinstance(failed_event_id, str):
            return
        item_id = self._pending_client_item_creates.pop(failed_event_id, None)
        if item_id is None:
            return
        await self._discard_finalized_typed_item(item_id)
        self._registry.retire_item(item_id)

    async def _discard_finalized_typed_item(self, local_item_id: str) -> None:
        """Fail closed every unconsumed binding to one deleted or rejected item."""
        async with self._response_state_lock:
            self._pending_typed_user_turns = deque(
                (item_id, text) for item_id, text in self._pending_typed_user_turns if item_id != local_item_id
            )
            candidates = [*self._response_create_queue]
            if self._response_create_in_flight is not None:
                candidates.append(self._response_create_in_flight)
            for queued in candidates:
                if queued.finalized_user_item_id == local_item_id:
                    queued.finalized_user_item_id = None
                    queued.finalized_user_text = None
            for response_id, item_id in tuple(self._response_finalized_user_item.items()):
                if item_id != local_item_id:
                    continue
                self._response_finalized_user_item.pop(response_id, None)
                self._response_finalized_user_text.pop(response_id, None)

    async def _publish_delivery_queue_state(self) -> None:
        """Publish waiting speech separately from the response currently speaking."""
        if not self._bootstrapped:
            return
        async with self._delivery_queue_projection_lock:
            async with self._response_state_lock:
                server_speech = {
                    _ResponsePurpose.ACKNOWLEDGEMENT,
                    _ResponsePurpose.DIRECT_REPLY,
                    _ResponsePurpose.DELIVERY,
                }
                waiting_depth = sum(item.purpose in server_speech for item in self._response_create_queue) + int(
                    self._response_create_in_flight is not None
                    and self._response_create_in_flight.purpose in server_speech
                )
                active_speech = self._active_upstream_response_purpose in server_speech
                depth = waiting_depth + int(active_speech)
                delivery_state = (depth, waiting_depth, active_speech)
            if delivery_state == self._reported_delivery_queue_state:
                return
            await self._emit_projection(
                kind="delivery_queue",
                phase="updated",
                title="Speech delivery queue",
                text=json.dumps(
                    {
                        "queue_depth": depth,
                        "waiting_depth": waiting_depth,
                        "active_speech": active_speech,
                    },
                    separators=(",", ":"),
                ),
                correlation={
                    "queue_depth": str(depth),
                    "waiting_depth": str(waiting_depth),
                    "active_speech": str(active_speech).lower(),
                },
            )
            self._reported_delivery_queue_state = delivery_state

    def _buffer_function_item_event(self, item_id: str, event: dict[str, Any]) -> None:
        events = self._buffered_function_item_events.get(item_id)
        if events is None:
            if len(self._buffered_function_item_events) >= _MAX_BUFFERED_FUNCTION_ITEMS:
                raise FacadeProtocolError(
                    "session_capacity_exceeded", "The realtime session has too many buffered function items."
                )
            events = []
            self._buffered_function_item_events[item_id] = events
        if len(events) >= _MAX_BUFFERED_FUNCTION_EVENTS_PER_ITEM:
            raise FacadeProtocolError(
                "session_capacity_exceeded", "The realtime session buffered too many events for a function item."
            )
        events.append(event)

    async def _handle_output_item_done(self, event: dict[str, Any]) -> None:
        response_id = _identifier(event.get("response_id"), "upstream response id")
        self._registry.response(response_id)
        item = event.get("item")
        if not isinstance(item, dict):
            raise FacadeProtocolError("upstream_protocol_error", "An output item event was malformed.")
        record = self._registry.item(item.get("id"))
        if record.completed:
            raise FacadeProtocolError("upstream_protocol_error", "The realtime frontend completed an item twice.")
        record.completed = True
        if record.owner is _Owner.SERVER:
            try:
                parsed = self._tools.parse_call(item)
            except (TypeError, ValueError) as error:
                raise FacadeProtocolError("upstream_protocol_error", "A protected tool call was invalid.") from error
            if parsed is None or record.call_id is None or parsed.call_id != record.call_id:
                raise FacadeProtocolError("upstream_protocol_error", "A protected tool call changed identity.")
            if parsed.name in {
                ProtectedTool.WORK_DELEGATE,
                ProtectedTool.WORK_ANSWER_AGENT,
            } and self._registry.is_primary_protected_call(response_id, parsed.call_id):
                finalized_text = self._response_finalized_user_text.pop(response_id, None)
                self._response_finalized_user_item.pop(response_id, None)
                if finalized_text is None:
                    if response_id not in self._late_finalized_text_waiters:
                        raise FacadeProtocolError(
                            "missing_finalized_user_turn",
                            "VoiceClaw cannot delegate without a finalized user request.",
                        )
                else:
                    parsed = replace(parsed, finalized_user_text=finalized_text)
            self._registry.complete_call(parsed.call_id)
            if (
                parsed.call_id not in self._pending_protected_calls
                and len(self._pending_protected_calls) >= _MAX_TRACKED_CALLS
            ):
                raise FacadeProtocolError(
                    "session_capacity_exceeded", "The realtime session has too many pending protected calls."
                )
            self._pending_protected_calls[parsed.call_id] = parsed
            return
        if record.call_id is not None:
            self._registry.complete_call(record.call_id)
        await self._send_downstream(self._rewrite_upstream_event(event))

    async def _handle_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response")
        if not isinstance(response, dict):
            raise FacadeProtocolError("upstream_protocol_error", "A response event was malformed.")
        response_id = _identifier(response.get("id"), "upstream response id")
        self._registry.complete_response(response_id)
        playback_lease = self._playback_leases_by_response.get(response_id)
        if playback_lease is not None:
            playback_lease.response_done = True
            # response.done is a generation boundary, never a playout receipt.
            # It does guarantee that no later audio delta belongs to this response.
            playback_lease.audio_done = True
        self._acknowledge_passthrough_client_event(event_type="response.cancel", target_id=response_id)
        response_output = response.get("output")
        if playback_lease is not None and not playback_lease.transcript:
            playback_lease.transcript = self._audio_transcript_from_response(response, playback_lease)
        terminal_item_ids = (
            tuple(
                _identifier(item.get("id"), "upstream item id")
                for item in response_output
                if isinstance(item, dict) and item.get("id") is not None
            )
            if isinstance(response_output, list)
            else ()
        )
        # Generation has ended even when the completed response contains a
        # protected tool call. Backend work continues in its own task; keeping
        # the frontend marked WAITING_FOR_TOOL would deadlock the speech-floor
        # arbiter when that task later queues its delivery response.
        await self._set_activity(model=ModelActivity.IDLE)
        protected_calls = self._registry.protected_calls_for_response(response_id)
        expected_tool = self._response_expected_protected_tool.get(response_id)
        if response.get("status") == "completed" and expected_tool is not None:
            matched_required_tool = any(
                (parsed := self._pending_protected_calls.get(upstream_call_id)) is not None
                and parsed.name is expected_tool
                for upstream_call_id, _record in protected_calls
            )
            if not matched_required_tool:
                raise FacadeProtocolError(
                    "required_tool_not_called",
                    "The realtime frontend did not follow the required backend route.",
                )
        if (
            response.get("status") == "completed"
            and response_id in self._response_model_route_selection
            and len(protected_calls) != 1
        ):
            raise FacadeProtocolError(
                "required_tool_not_called",
                "The realtime frontend did not return exactly one structured VoiceClaw route.",
            )
        model_route_selection = response_id in self._response_model_route_selection
        rewritten = self._rewrite_upstream_event(event)
        rewritten_response = rewritten.get("response")
        if self._active_upstream_response_purpose is _ResponsePurpose.DELIVERY and isinstance(rewritten_response, dict):
            metadata = rewritten_response.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["voiceclaw_delivery"] = "true"
        if isinstance(rewritten_response, dict):
            self._mark_active_speech_metadata(rewritten_response)
        if isinstance(rewritten_response, dict) and isinstance(rewritten_response.get("output"), list):
            rewritten_response["output"] = [
                item
                for item in rewritten_response["output"]
                if not (isinstance(item, dict) and self._is_protected_item(item, ids_are_local=True))
            ]
        if not model_route_selection:
            await self._send_downstream(rewritten)
        if playback_lease is not None and playback_lease.generated_samples > 0:
            self._record_assistant_turn(playback_lease)
        if playback_lease is not None and playback_lease.generated_samples == 0:
            self._retire_playback_lease(playback_lease)
        await self._flush_deferred_truncations(response_id)
        self._buffered_function_item_events.clear()
        if response.get("status") != "completed":
            response_status = response.get("status")
            delivery_phase = "cancelled" if response_status in {"cancelled", "canceled"} else "failed"
            await self._emit_speech_delivery_failure(
                purpose=self._active_upstream_response_purpose,
                speech_purpose=self._active_speech_purpose,
                local_request_id=self._active_speech_local_request_id,
                phase=delivery_phase,
                reason=f"frontend_response_{delivery_phase}",
            )
            if model_route_selection:
                await self._send_public_error(
                    "routing_unavailable",
                    "VoiceClaw could not select a safe response route for this turn.",
                )
            for upstream_call_id, record in protected_calls:
                record.executed = True
                self._pending_protected_calls.pop(upstream_call_id, None)
            self._release_finalized_audio_turn(response_id)
            self._registry.retire_response(response_id)
            for item_id in terminal_item_ids:
                self._registry.retire_terminal_item(item_id)
            await self._mark_response_terminal(response_id)
            return
        task_started = False
        for call_index, (upstream_call_id, record) in enumerate(protected_calls):
            if record.executed:
                continue
            record.executed = True
            parsed = self._pending_protected_calls.pop(upstream_call_id, None)
            if parsed is None:
                raise FacadeProtocolError("upstream_protocol_error", "A protected tool result was lost.")
            if call_index > 0:
                await self._send_public_error(
                    "multiple_work_actions",
                    "The realtime frontend requested more than one Work action in a single response.",
                )
                await self._send_tool_output(
                    upstream_call_id,
                    {"status": "failed"},
                )
                continue
            if len(self._tool_tasks) >= _MAX_PROTECTED_TASKS:
                failure_copy = self._failure_copy("session_busy")
                await self._send_public_error(
                    "session_busy",
                    failure_copy.display,
                )
                await self._return_tool_result(
                    upstream_call_id,
                    {"status": "failed", "error": "session_busy"},
                    "session_busy",
                )
                continue
            await self._reserve_admission_barrier(upstream_call_id)
            direct_reply = (
                copy.deepcopy(self._response_direct_reply.get(response_id))
                if parsed.name is ProtectedTool.CONVERSATION_RESPOND
                else None
            )
            if parsed.name is ProtectedTool.CONVERSATION_RESPOND and direct_reply is None:
                await self._release_admission_barrier(upstream_call_id)
                raise FacadeProtocolError(
                    "upstream_protocol_error", "The selected direct route lost its response template."
                )
            try:
                task = asyncio.create_task(
                    self._run_protected_task(
                        response_id,
                        upstream_call_id,
                        record.local_id,
                        parsed,
                        direct_reply=direct_reply,
                    ),
                    name=f"voiceclaw-tool-{record.local_id}",
                )
            except BaseException:
                await self._release_admission_barrier(upstream_call_id)
                raise
            task_started = True
            self._tool_tasks.add(task)
            task.add_done_callback(self._tool_tasks.discard)
        if not task_started:
            self._release_finalized_audio_turn(response_id)
        self._registry.retire_response(response_id)
        for item_id in terminal_item_ids:
            self._registry.retire_terminal_item(item_id)
        await self._mark_response_terminal(response_id)

    @staticmethod
    def _audio_transcript_from_response(response: Mapping[str, Any], lease: _PlaybackLease) -> str:
        """Extract the bounded public transcript from one terminal audio item."""
        output = response.get("output")
        if not isinstance(output, list):
            return ""
        for item in output:
            if not isinstance(item, Mapping) or item.get("id") != lease.item_id:
                continue
            content = item.get("content")
            if not isinstance(content, list):
                return ""
            parts = [
                part.get("transcript", "")
                for part in content
                if isinstance(part, Mapping)
                and part.get("type") in {"audio", "output_audio"}
                and isinstance(part.get("transcript"), str)
            ]
            return "".join(parts)[:_MAX_OUTPUT_CHARACTERS]
        return ""

    def _mark_active_speech_metadata(self, response: dict[str, Any]) -> None:
        """Describe model-mediated speech and its standard browser receipt contract."""
        metadata = response.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            return
        lease = (
            self._playback_leases_by_response.get(self._active_upstream_response_id)
            if self._active_upstream_response_id is not None
            else None
        )
        if lease is not None:
            metadata.update(
                {
                    "voiceclaw_presentation_id": lease.presentation_id,
                    "voiceclaw_playback_receipt": "conversation.item.truncate.v1",
                    "voiceclaw_playback_receipt_id": lease.receipt_id,
                    "voiceclaw_playback_receipt_required": "true",
                }
            )
        speech_purpose = self._active_speech_purpose
        if speech_purpose not in _APPLICATION_SPEECH_PURPOSES:
            return
        metadata["voiceclaw_speech_purpose"] = speech_purpose
        if self._active_speech_delivery_mode is not None:
            metadata["voiceclaw_speech_delivery"] = self._active_speech_delivery_mode

    async def _run_protected_task(
        self,
        response_id: str,
        upstream_call_id: str,
        local_call_id: str,
        parsed: ParsedToolCall,
        *,
        direct_reply: _DirectReplyTemplate | None = None,
    ) -> None:
        try:
            parsed = await self._await_finalized_audio_turn(response_id, parsed)
            await self._execute_protected_call(
                upstream_call_id,
                local_call_id,
                parsed,
                direct_reply=direct_reply,
            )
        except asyncio.CancelledError:
            raise
        except FacadeProtocolError as error:
            await self._send_public_error(error.code, error.public_message)
            with suppress(Exception):
                await self._return_tool_result(
                    upstream_call_id,
                    {"status": "failed", "error": error.code},
                    error.code,
                )
        except Exception:
            failure_copy = self._failure_copy("runtime_unavailable")
            await self._send_public_error(
                "runtime_unavailable",
                failure_copy.display,
            )
            with suppress(Exception):
                await self._return_tool_result(
                    upstream_call_id,
                    {"status": "failed", "error": "runtime_unavailable"},
                    "runtime_unavailable",
                )
        finally:
            self._release_finalized_audio_turn(response_id)
            await self._release_admission_barrier(upstream_call_id)

    async def _await_finalized_audio_turn(
        self,
        response_id: str,
        parsed: ParsedToolCall,
    ) -> ParsedToolCall:
        """Wait only for a transcript explicitly correlated to this audio response."""
        if parsed.name not in {ProtectedTool.WORK_DELEGATE, ProtectedTool.WORK_ANSWER_AGENT}:
            return parsed
        if parsed.finalized_user_text is not None:
            return parsed
        waiter = self._late_finalized_text_waiters.get(response_id)
        if waiter is None:
            raise FacadeProtocolError(
                "missing_finalized_user_turn",
                "VoiceClaw cannot delegate without a finalized user request.",
            )
        try:
            finalized_text = await asyncio.wait_for(
                asyncio.shield(waiter),
                timeout=_FINALIZED_AUDIO_TURN_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise FacadeProtocolError(
                "finalized_user_turn_timeout",
                "The audio transcript was not finalized in time for delegation.",
            ) from error
        if finalized_text is None:
            raise FacadeProtocolError(
                "missing_finalized_user_turn",
                "The audio request could not be transcribed for delegation.",
            )
        return replace(parsed, finalized_user_text=finalized_text)

    def _release_finalized_audio_turn(self, response_id: str) -> None:
        """Release one response-to-audio correlation after terminal handling."""
        waiter = self._late_finalized_text_waiters.pop(response_id, None)
        if waiter is not None and not waiter.done():
            waiter.cancel()
        item_id = self._response_audio_item.pop(response_id, None)
        if item_id is not None:
            self._cancel_finalized_audio_timeout(item_id)
            self._audio_item_response.pop(item_id, None)
            self._audio_item_input_generation.pop(item_id, None)
            self._failed_audio_items.discard(item_id)
            with suppress(ValueError):
                self._unbound_audio_items.remove(item_id)
        self._response_finalized_user_text.pop(response_id, None)
        self._response_finalized_user_item.pop(response_id, None)

    async def _cancel_tool_tasks(self) -> None:
        tasks = tuple(self._tool_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tool_tasks.clear()

    def _classify_item(self, item: dict[str, Any]) -> _Owner | None:
        if item.get("type") != "function_call":
            return None
        name = item.get("name")
        if isinstance(name, str) and name.startswith("voiceclaw_"):
            return _Owner.SERVER
        return _Owner.CLIENT

    def _is_protected_item(self, item: dict[str, Any], *, ids_are_local: bool = False) -> bool:
        if item.get("type") != "function_call":
            return False
        call_id = item.get("call_id")
        try:
            if ids_are_local:
                _, record = self._registry.upstream_call(call_id)
            else:
                record = self._registry.call(call_id)
        except FacadeProtocolError:
            return False
        return record.owner is _Owner.SERVER

    def _prepare_client_item(self, event: dict[str, Any]) -> dict[str, Any]:
        item = event.get("item")
        if not isinstance(item, dict):
            raise FacadeProtocolError("invalid_request", "conversation.item.create requires an item object.")
        item_type = item.get("type")
        if item_type == "function_call_output":
            upstream_call_id, record = self._registry.upstream_call(item.get("call_id"))
            if record.owner is _Owner.SERVER:
                raise FacadeProtocolError("invalid_request", "Protected tool output is server-owned.")
            item["call_id"] = upstream_call_id
        elif item_type == "function_call":
            name = item.get("name")
            if isinstance(name, str) and name.startswith("voiceclaw_"):
                raise FacadeProtocolError("invalid_request", "Protected tool calls are server-owned.")
        elif item_type == "message" and item.get("role") in {"system", "developer"}:
            raise FacadeProtocolError("invalid_request", "Structured system messages are server-owned.")
        item_id = item.get("id")
        if item_id is None:
            item_id = self._id_factory("item_input_vc")
            item["id"] = item_id
        self._registry.register_client_item(item_id)
        previous = event.get("previous_item_id")
        if previous is not None:
            event["previous_item_id"] = self._registry.upstream_client_item(previous)
        return event

    def _finalized_text_from_client_item(self, item: object) -> str | None:
        """Extract a committed typed user turn without trusting model arguments."""
        if not isinstance(item, Mapping) or item.get("type") != "message" or item.get("role") != "user":
            return None
        status = item.get("status")
        if status is not None and status != "completed":
            return None
        content = item.get("content")
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "input_text":
                continue
            text = part.get("text")
            if not isinstance(text, str):
                raise FacadeProtocolError("invalid_request", "The finalized user text is invalid.")
            parts.append(text)
        if not parts:
            return None
        return self._normalize_finalized_user_text(
            "\n".join(parts),
            code="invalid_request",
            message="The finalized user text is invalid.",
        )

    @staticmethod
    def _normalize_finalized_user_text(value: object, *, code: str, message: str) -> str | None:
        """Validate one bounded finalized turn without rewriting its content."""
        if not isinstance(value, str):
            raise FacadeProtocolError(code, message)
        if not value.strip():
            return None
        if "\x00" in value or len(value.encode("utf-8")) > MAX_COMMITTED_TURN_GOAL_BYTES:
            raise FacadeProtocolError(code, message)
        return value

    def _arm_finalized_audio_timeout(self, finalized_item_id: str) -> None:
        """Bound how long one queued response may wait for its ASR terminal event."""
        if finalized_item_id in self._finalized_audio_timeout_tasks:
            return
        task = asyncio.create_task(
            self._expire_finalized_audio_text(finalized_item_id),
            name=f"voiceclaw-audio-transcript-{finalized_item_id}",
        )
        self._finalized_audio_timeout_tasks[finalized_item_id] = task
        task.add_done_callback(
            lambda completed, item_id=finalized_item_id: (
                self._finalized_audio_timeout_tasks.pop(item_id, None)
                if self._finalized_audio_timeout_tasks.get(item_id) is completed
                else None
            )
        )

    def _cancel_finalized_audio_timeout(self, finalized_item_id: str) -> None:
        """Cancel one pending ASR deadline without cancelling its own callback."""
        task = self._finalized_audio_timeout_tasks.pop(finalized_item_id, None)
        if task is not None and task is not asyncio.current_task() and not task.done():
            task.cancel()

    async def _expire_finalized_audio_text(self, finalized_item_id: str) -> None:
        """Reject an exact audio turn when its frontend never finalizes ASR."""
        try:
            await asyncio.sleep(_FINALIZED_AUDIO_TURN_TIMEOUT_SECONDS)
            await self._fail_finalized_audio_text(
                finalized_item_id,
                code="finalized_user_turn_timeout",
                message="The audio transcript was not finalized in time. Please try again.",
            )
        except asyncio.CancelledError:
            return
        except Exception:
            _LOGGER.exception("Could not retire a timed-out finalized audio turn")

    async def _finish_audio_input(self, finalized_item_id: str) -> None:
        """Retire one item without letting a stale ASR event clobber a newer turn."""
        generation = self._audio_item_input_generation.pop(finalized_item_id, None)
        if generation is None:
            # A manual push-to-talk commit may contain no preceding append in
            # a compatibility client/test, so it has no generation mapping.
            # Its terminal transcript still closes TRANSCRIBING when no newer
            # capture exists.
            if self._active_input_generation is None and self._activity.input is InputActivity.TRANSCRIBING:
                await self._set_activity(input=InputActivity.IDLE)
            return
        await self._finish_input_generation(generation)

    async def _finish_input_generation(self, generation: int) -> None:
        """Set input idle only when the completed generation is still current."""
        if generation != self._active_input_generation:
            return
        self._active_input_generation = None
        if self._activity.input is InputActivity.TRANSCRIBING:
            await self._set_activity(input=InputActivity.IDLE)

    async def _record_finalized_user_text(
        self,
        text: str,
        *,
        allow_active_binding: bool,
        finalized_item_id: str | None = None,
    ) -> None:
        """Bind committed text to its exact response or retain it for the next one."""
        if finalized_item_id is not None:
            self._cancel_finalized_audio_timeout(finalized_item_id)
            response_id = self._audio_item_response.get(finalized_item_id)
            if response_id is not None:
                self._response_finalized_user_text[response_id] = text
                self._record_user_turn(finalized_item_id, text)
                waiter = self._late_finalized_text_waiters.get(response_id)
                if waiter is not None and not waiter.done():
                    waiter.set_result(text)
                self._failed_audio_items.discard(finalized_item_id)
                return
            in_flight = self._response_create_in_flight
            if in_flight is not None and in_flight.finalized_item_id == finalized_item_id:
                in_flight.finalized_user_text = text
                self._failed_audio_items.discard(finalized_item_id)
                return
            for queued in self._response_create_queue:
                if queued.finalized_item_id == finalized_item_id:
                    queued.finalized_user_text = text
                    queued.awaits_finalized_user_text = False
                    self._failed_audio_items.discard(finalized_item_id)
                    await self._dispatch_next_response_create()
                    return
            manual_commit = next(
                (commit for commit in self._manual_audio_commits if commit.finalized_item_id == finalized_item_id),
                None,
            )
            if manual_commit is not None:
                manual_commit.finalized_user_text = text
                with suppress(ValueError):
                    self._unbound_audio_items.remove(finalized_item_id)
                self._failed_audio_items.discard(finalized_item_id)
                return
            if finalized_item_id in self._unbound_audio_items:
                self._unbound_audio_items.remove(finalized_item_id)
                self._pending_finalized_user_text = text
            self._failed_audio_items.discard(finalized_item_id)
            return

        active_response_id = self._active_upstream_response_id
        if (
            allow_active_binding
            and active_response_id is not None
            and self._active_upstream_response_purpose is _ResponsePurpose.INTERACTIVE
            and active_response_id not in self._response_finalized_user_text
        ):
            self._response_finalized_user_text[active_response_id] = text
            return
        if allow_active_binding:
            in_flight = self._response_create_in_flight
            if (
                in_flight is not None
                and in_flight.purpose is _ResponsePurpose.INTERACTIVE
                and in_flight.finalized_user_text is None
            ):
                in_flight.finalized_user_text = text
                return
            for queued in self._response_create_queue:
                if queued.purpose is _ResponsePurpose.INTERACTIVE and queued.finalized_user_text is None:
                    queued.finalized_user_text = text
                    return
        self._pending_finalized_user_text = text

    async def _fail_finalized_audio_text(
        self,
        finalized_item_id: str,
        *,
        code: str,
        message: str,
    ) -> None:
        """Retire one untranscribed audio turn without generating an empty response."""
        self._cancel_finalized_audio_timeout(finalized_item_id)
        await self._finish_audio_input(finalized_item_id)
        self._failed_audio_items.add(finalized_item_id)
        with suppress(ValueError):
            self._unbound_audio_items.remove(finalized_item_id)
        rejected_event_id: str | None = None
        rejected_response: _QueuedResponseCreate | None = None
        async with self._response_state_lock:
            for queued in tuple(self._response_create_queue):
                if queued.finalized_item_id != finalized_item_id:
                    continue
                self._response_create_queue.remove(queued)
                rejected_response = queued
                candidate = queued.event.get("event_id")
                rejected_event_id = candidate if isinstance(candidate, str) else None
                break
            manual_commit = next(
                (commit for commit in self._manual_audio_commits if commit.finalized_item_id == finalized_item_id),
                None,
            )
            if manual_commit is not None:
                manual_commit.rejected = True
                manual_commit.rejection_code = code
                manual_commit.rejection_message = message
        if rejected_response is not None:
            await self._retire_unsent_response_create(rejected_response)
        response_id = self._audio_item_response.get(finalized_item_id)
        if response_id is not None:
            waiter = self._late_finalized_text_waiters.get(response_id)
            if waiter is not None and not waiter.done():
                waiter.set_result(None)
        else:
            self._failed_audio_items.discard(finalized_item_id)
        if rejected_event_id is not None:
            await self._send_public_error(code, message, event_id=rejected_event_id)
        await self._dispatch_next_response_create()

    def _prepare_response_create(
        self,
        event: dict[str, Any],
    ) -> _PreparedResponseCreate:
        response = event.get("response", {})
        finalized_text: str | None = None
        finalized_item_id: str | None = None
        if response is None:
            response = {}
        if not isinstance(response, dict):
            raise FacadeProtocolError("invalid_request", "response.create requires a response object.")
        if "tools" in response:
            client_tools = response["tools"]
            if not isinstance(client_tools, list):
                raise FacadeProtocolError("invalid_request", "response tools must be an array.")
            for tool in client_tools:
                if not isinstance(tool, dict):
                    raise FacadeProtocolError("invalid_request", "response tools must be objects.")
                name = tool.get("name")
                if isinstance(name, str) and name.startswith("voiceclaw_"):
                    raise FacadeProtocolError("invalid_request", "The client cannot redefine a protected tool.")
            response["tools"] = [*client_tools, *copy.deepcopy(self._tools.schemas())]
        if "tool_choice" in response:
            choice = response["tool_choice"]
            name: str | None = None
            if isinstance(choice, str):
                name = choice if choice.startswith("voiceclaw_") else None
            elif isinstance(choice, dict):
                candidate_name = choice.get("name")
                name = candidate_name if isinstance(candidate_name, str) else None
            else:
                raise FacadeProtocolError("invalid_request", "response tool_choice is invalid.")
            if name is not None:
                raise FacadeProtocolError("invalid_request", "Protected response tool selection is server-owned.")
        elif self._tools.enabled:
            response["tool_choice"] = "auto"
        has_response_local_input = "input" in response
        if has_response_local_input:
            finalized_text, finalized_item_id = self._finalized_turn_from_response_input(response["input"])
            response["input"] = self._prepare_response_input(response["input"])
        client_instructions = response.get("instructions", "")
        if client_instructions is None:
            client_instructions = ""
        if not isinstance(client_instructions, str):
            raise FacadeProtocolError("invalid_request", "response instructions must be text.")
        # Keep untrusted client text separate from application-owned policy.
        # The complete boundary is rebuilt only when this response actually
        # owns the upstream slot, so its runtime projection cannot go stale in
        # the arbiter queue.
        response.pop("instructions", None)
        event["response"] = response
        return _PreparedResponseCreate(
            event=event,
            finalized_user_text=finalized_text,
            finalized_user_item_id=finalized_item_id,
            has_response_local_input=has_response_local_input,
            client_response_instructions=client_instructions.strip(),
        )

    def _prepare_response_input(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._prepare_response_input(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        item = copy.deepcopy(value)
        item_type = item.get("type")
        if item_type == "function_call":
            name = item.get("name")
            if isinstance(name, str) and name.startswith("voiceclaw_"):
                raise FacadeProtocolError("invalid_request", "Protected inline tool calls are server-owned.")
        elif item_type == "function_call_output":
            upstream_call_id, record = self._registry.upstream_call(item.get("call_id"))
            if record.owner is _Owner.SERVER:
                raise FacadeProtocolError("invalid_request", "Protected inline tool output is server-owned.")
            item["call_id"] = upstream_call_id
        elif item_type == "item_reference":
            item["id"] = self._registry.upstream_client_item(item.get("id"))
        elif item_type == "message" and item.get("role") in {"system", "developer"}:
            raise FacadeProtocolError("invalid_request", "Structured system messages are server-owned.")
        for key, child in tuple(item.items()):
            if key not in {"call_id", "id", "name", "type"}:
                item[key] = self._prepare_response_input(child)
        return item

    def _finalized_turn_from_response_input(self, value: object) -> tuple[str | None, str | None]:
        """Return the last finalized user turn and any tracked item identity."""
        if not isinstance(value, list):
            return None, None
        finalized: str | None = None
        finalized_item_id: str | None = None
        for item in value:
            if isinstance(item, Mapping) and item.get("type") == "item_reference":
                candidate_id = item.get("id")
                if isinstance(candidate_id, str):
                    pending = next(
                        (
                            (item_id, text)
                            for item_id, text in reversed(self._pending_typed_user_turns)
                            if item_id == candidate_id
                        ),
                        None,
                    )
                    if pending is not None:
                        finalized_item_id, finalized = pending
                continue
            candidate = self._finalized_text_from_client_item(item)
            if candidate is not None:
                finalized = candidate
                finalized_item_id = None
        return finalized, finalized_item_id

    def _force_projection_boundary(
        self,
        session: dict[str, Any],
        *,
        baseline: Mapping[str, Any] | None = None,
    ) -> None:
        """Own response creation while preserving advertised turn detection.

        The upstream keeps its negotiated server/semantic VAD, but does not
        create a response itself.  The facade recreates the requested public
        automatic response after the committed-audio event so it can inject
        the current bounded projection first. Manual mode remains manual.
        """
        source: Mapping[str, Any] = session
        source_audio = source.get("audio")
        source_input = source_audio.get("input") if isinstance(source_audio, Mapping) else None
        supplied = isinstance(source_input, Mapping) and "turn_detection" in source_input
        turn_detection = source_input.get("turn_detection") if isinstance(source_input, Mapping) else None
        if not supplied and baseline is not None:
            baseline_audio = baseline.get("audio")
            baseline_input = baseline_audio.get("input") if isinstance(baseline_audio, Mapping) else None
            turn_detection = baseline_input.get("turn_detection") if isinstance(baseline_input, Mapping) else None
            supplied = isinstance(baseline_input, Mapping) and "turn_detection" in baseline_input
        if not supplied:
            return
        if turn_detection is None:
            self._automatic_turn_detection = False
            self._automatic_response = False
            if isinstance(source_input, dict):
                source_input["turn_detection"] = None
            return
        if not isinstance(turn_detection, Mapping):
            raise FacadeProtocolError("invalid_request", "turn detection must be an object or null.")
        requested = turn_detection.get("create_response", True)
        if not isinstance(requested, bool):
            raise FacadeProtocolError("invalid_request", "turn detection create_response must be boolean.")
        self._automatic_turn_detection = True
        self._automatic_response = requested
        audio = session.setdefault("audio", {})
        if not isinstance(audio, dict):
            raise FacadeProtocolError("invalid_request", "session audio settings must be an object.")
        input_audio = audio.setdefault("input", {})
        if not isinstance(input_audio, dict):
            raise FacadeProtocolError("invalid_request", "session input audio settings must be an object.")
        upstream_turn_detection = copy.deepcopy(dict(turn_detection))
        upstream_turn_detection["create_response"] = False
        input_audio["turn_detection"] = upstream_turn_detection

    def _current_projection(self) -> str:
        try:
            projection = self._runtime.projection(self._facade_session_id)
        except Exception as error:
            raise FacadeProtocolError(
                "projection_unavailable", "VoiceClaw context is temporarily unavailable."
            ) from error
        if not isinstance(projection, str) or "\x00" in projection:
            raise FacadeProtocolError("projection_unavailable", "VoiceClaw context is invalid.")
        if len(projection) > self._context_character_budget:
            raise FacadeProtocolError("projection_too_large", "VoiceClaw context exceeds its bounded projection.")
        return projection

    async def _update_downstream_activity(self, event_type: str) -> None:
        if event_type == "input_audio_buffer.append":
            if self._automatic_turn_detection:
                # Continuous VAD clients stream silence as well as speech.
                # Only upstream speech boundary events may claim LISTENING or
                # allocate a semantic input generation in automatic mode.
                return
            if not self._input_buffer_open:
                self._input_generation += 1
                self._active_input_generation = self._input_generation
                self._input_buffer_open = True
            await self._set_activity(input=InputActivity.LISTENING)
        elif event_type == "input_audio_buffer.commit":
            self._input_buffer_open = False
            await self._set_activity(input=InputActivity.TRANSCRIBING)
        elif event_type == "input_audio_buffer.clear":
            self._input_buffer_open = False
            self._active_input_generation = None
            await self._set_activity(input=InputActivity.IDLE)

    async def _update_upstream_activity(self, event_type: str, event: Mapping[str, Any]) -> None:
        if event_type == "input_audio_buffer.speech_started":
            item_id = event.get("item_id")
            generation = self._audio_item_input_generation.get(item_id) if isinstance(item_id, str) else None
            if generation is None:
                self._input_generation += 1
                generation = self._input_generation
            if isinstance(item_id, str) and item_id:
                self._audio_item_input_generation[item_id] = generation
            self._active_input_generation = generation
            await self._set_activity(input=InputActivity.LISTENING)
        elif event_type == "input_audio_buffer.speech_stopped":
            item_id = event.get("item_id")
            generation = self._audio_item_input_generation.get(item_id) if isinstance(item_id, str) else None
            if generation is not None and generation == self._active_input_generation:
                self._input_buffer_open = False
                await self._set_activity(input=InputActivity.TRANSCRIBING)
        elif event_type == "response.created":
            await self._set_activity(model=ModelActivity.GENERATING)

    async def _set_activity(
        self,
        *,
        input: InputActivity | None = None,
        model: ModelActivity | None = None,
        output: OutputActivity | None = None,
    ) -> None:
        async with self._speech_admission_lock:
            activity = replace(
                self._activity,
                input=self._activity.input if input is None else input,
                model=self._activity.model if model is None else model,
                output=self._activity.output if output is None else output,
            )
            if activity == self._activity:
                return
            self._activity = activity
        if not self._runtime_open:
            return
        try:
            await self._runtime.update_activity(self._facade_session_id, activity)
        except Exception as error:
            raise FacadeProtocolError(
                "runtime_unavailable", "The VoiceClaw interaction manager could not update session activity."
            ) from error
        # The browser reports its played-through boundary with the standard
        # conversation.item.truncate transaction. Until that receipt closes the
        # output lease, queued application speech cannot claim the floor.
        if activity.speech_floor_available:
            await self._dispatch_next_response_create()

    async def _execute_protected_call(
        self,
        upstream_call_id: str,
        local_call_id: str,
        parsed: ParsedToolCall,
        *,
        direct_reply: _DirectReplyTemplate | None = None,
    ) -> None:
        if parsed.name is ProtectedTool.CONVERSATION_RESPOND:
            if direct_reply is None:
                raise FacadeProtocolError(
                    "runtime_protocol_error", "The direct conversational route has no response template."
                )
            if not await self._send_tool_output(
                upstream_call_id,
                {"status": "ok"},
            ):
                raise FacadeProtocolError(
                    "upstream_protocol_error", "The direct conversational route repeated its outcome."
                )
            await self._enqueue_response_create(
                direct_reply.event,
                purpose=_ResponsePurpose.DIRECT_REPLY,
                requires_speech_floor=True,
                speech_purpose="direct_reply",
                client_response_instructions=direct_reply.client_response_instructions,
                bind_pending_user_turn=False,
            )
            await self._release_admission_barrier(upstream_call_id)
            return
        if parsed.name not in self._tools.enabled:
            raise FacadeProtocolError(
                "capability_unsupported",
                "The selected Work operation is not supported by the current backend attachment.",
            )
        commit_id = self._id_factory("commit_vc")
        updates = self._runtime.execute_tool(
            session_id=self._facade_session_id,
            commit_id=commit_id,
            call_id=local_call_id,
            tool_name=parsed.logical_name,
            arguments=parsed.arguments,
            finalized_user_text=parsed.finalized_user_text,
        )
        await self._consume_runtime_updates(
            upstream_call_id,
            updates,
            local_request_id=commit_id,
            failure_code="backend_unavailable",
        )

    async def _consume_runtime_updates(
        self,
        upstream_call_id: str,
        updates: AsyncIterator[InteractionUpdate],
        *,
        local_request_id: str,
        failure_code: str,
    ) -> None:
        terminal: InteractionUpdate | None = None
        pending_terminal: InteractionUpdate | None = None
        tool_output_sent = False
        acknowledgement_queued = False
        admission_released = False
        display_stream: ProjectionEventStream | None = None
        display_stream_correlation: dict[str, str] | None = None
        display_stream_closed = False
        display_result_committed = False

        async def discard_provisional_display(
            *,
            status: ProjectionStreamAbortStatus = ProjectionStreamAbortStatus.FAILED,
            reason: str = "runtime_protocol_error",
        ) -> None:
            """Close an open projection while marking its text non-authoritative."""
            nonlocal display_result_committed, display_stream_closed
            if display_stream is None:
                return
            if display_stream_closed:
                if display_result_committed:
                    await self._emit_projection(
                        kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                        phase=ResponseOnlyResultEventKind.DISCARDED,
                        title="Response discarded",
                        text="",
                        correlation=display_stream_correlation or {},
                    )
                    display_result_committed = False
                return
            projection = Projection(
                session_id=self._facade_session_id,
                kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                phase=ResponseOnlyResultEventKind.DISCARDED,
                title="Response discarded",
                text=display_stream.text,
                correlation=display_stream_correlation or {},
            )
            for event in display_stream.abort(projection, status=status, reason=reason):
                await self._send_downstream(event)
            display_stream_closed = True

        async def apply_update(update: InteractionUpdate) -> None:
            nonlocal acknowledgement_queued, admission_released, tool_output_sent
            nonlocal display_result_committed, display_stream, display_stream_closed, display_stream_correlation
            correlation = dict(update.correlation)
            for identity_key in ("local_request_id", "commit_id"):
                projected_identity = correlation.get(identity_key)
                if projected_identity is not None and projected_identity != local_request_id:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned inconsistent request identity.",
                    )
            identity_authority = correlation.get("identity_authority")
            if identity_authority is not None and identity_authority != "voiceclaw_local":
                raise FacadeProtocolError(
                    "runtime_protocol_error",
                    "The VoiceClaw interaction manager returned inconsistent request authority.",
                )
            correlation.setdefault("local_request_id", local_request_id)
            correlation.setdefault("identity_authority", "voiceclaw_local")

            if update.frontend_tools is not None:
                try:
                    self._tools = VoiceClawToolRegistry(
                        tools=update.frontend_tools,
                        include_direct_route=bool(update.frontend_tools),
                        contracts=self._model_contracts,
                        interaction_profile=self._interaction_profile,
                    )
                except ValueError as error:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned an invalid tool snapshot.",
                    ) from error

            output = dict(update.tool_output) if update.tool_output is not None else None
            if output is not None:
                if tool_output_sent:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned more than one tool outcome.",
                    )
                if not all(isinstance(key, str) and isinstance(value, str) for key, value in output.items()):
                    raise FacadeProtocolError(
                        "runtime_protocol_error", "The VoiceClaw interaction manager returned an invalid outcome."
                    )
                output_request_id = output.get("local_request_id")
                if output_request_id is not None and output_request_id != local_request_id:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned inconsistent request identity.",
                    )

            frontend_response = update.frontend_response
            if frontend_response is not None:
                if frontend_response.local_request_id is not None and (
                    frontend_response.local_request_id != local_request_id
                ):
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned inconsistent response identity.",
                    )
                if frontend_response.local_request_id is None:
                    frontend_response = replace(frontend_response, local_request_id=local_request_id)

            if update.kind is ResponseOnlyUpdateKind.RESULT_DISPLAY:
                if output is not None or frontend_response is not None or update.terminal:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned an invalid display update.",
                    )
                if display_stream_correlation is not None and correlation != display_stream_correlation:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager changed display correlation.",
                    )
                if display_stream is None:
                    display_stream_correlation = correlation
                    display_stream = self._projection_events.stream(
                        Projection(
                            session_id=self._facade_session_id,
                            kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                            phase=ResponseOnlyResultEventKind.DISPLAY_DELTA,
                            title=update.title,
                            text="",
                            correlation=correlation,
                        )
                    )
                    for event in display_stream.start():
                        await self._send_downstream(event)
                if display_stream_closed:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager updated a closed display stream.",
                    )
                if update.phase is ResponseOnlyResultEventKind.DISPLAY_DELTA:
                    await self._send_downstream(display_stream.delta(update.text))
                    return
                if update.phase is not ResponseOnlyResultEventKind.COMPLETED:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned an unsupported display phase.",
                    )
                projection = Projection(
                    session_id=self._facade_session_id,
                    kind=ResponseOnlyUpdateKind.RESULT_DISPLAY,
                    phase=ResponseOnlyResultEventKind.COMPLETED,
                    title=update.title,
                    text=update.text,
                    correlation=correlation,
                )
                for event in display_stream.finish(projection):
                    await self._send_downstream(event)
                display_stream_closed = True
                display_result_committed = True
                return

            if update.terminal:
                expected_purpose = (
                    FrontendResponsePurpose.RESULT_DELIVERY
                    if update.phase is ResponseOnlyRequestState.SUCCEEDED
                    else FrontendResponsePurpose.FAILURE_DELIVERY
                )
                if frontend_response is not None and frontend_response.purpose is not expected_purpose:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned a mismatched terminal response.",
                    )
                if update.phase is ResponseOnlyRequestState.FAILED and frontend_response is None:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager omitted failure speech.",
                    )
                if update.phase is ResponseOnlyRequestState.SUCCEEDED and not acknowledgement_queued:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager completed Work without an acknowledgement boundary.",
                    )
                if update.phase is ResponseOnlyRequestState.SUCCEEDED and not display_result_committed:
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager completed a request without a committed display result.",
                    )
                if update.phase is ResponseOnlyRequestState.FAILED:
                    await discard_provisional_display(
                        reason=correlation.get("error_code", failure_code),
                    )
                # Failure speech remains part of the terminal failure contract.
                # Result speech is optional delivery attached to an already
                # committed display and is validated independently below.
                if (
                    frontend_response is not None
                    and frontend_response.purpose is not FrontendResponsePurpose.RESULT_DELIVERY
                ):
                    self._frontend_response_context(frontend_response)
            elif frontend_response is not None:
                if (
                    frontend_response.purpose is not FrontendResponsePurpose.DELEGATION_ACK
                    or output is None
                    or acknowledgement_queued
                ):
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager returned an invalid acknowledgement boundary.",
                    )
            elif output is not None:
                raise FacadeProtocolError(
                    "runtime_protocol_error",
                    "The VoiceClaw interaction manager returned a receipt without an acknowledgement.",
                )

            # Reserve the acknowledgement (or an immediate terminal failure)
            # before publishing a tool outcome. The admission barrier keeps the
            # queued response off the wire until the function output is sent.
            response_prequeued = frontend_response is not None and output is not None
            if response_prequeued:
                await self._queue_frontend_response(frontend_response)
                if frontend_response.purpose is FrontendResponsePurpose.DELEGATION_ACK:
                    acknowledgement_queued = True
            await self._emit_projection(
                kind=update.kind,
                phase=update.phase,
                title=update.title,
                text=update.text,
                correlation=correlation,
                request_summary=update.request_summary,
            )
            if output is not None:
                private_outcome = {
                    "status": ("failed" if update.terminal or output.get("status") == "failed" else "ok")
                }
                if not await self._send_tool_output(upstream_call_id, private_outcome):
                    raise FacadeProtocolError(
                        "runtime_protocol_error",
                        "The VoiceClaw interaction manager repeated a tool outcome.",
                    )
                tool_output_sent = True
            if frontend_response is not None and not response_prequeued:
                # An acknowledged request owns one bounded terminal speech slot.
                # This prevents its result/failure from being dropped merely
                # because the acknowledgement is still waiting to be spoken.
                if frontend_response.purpose is FrontendResponsePurpose.RESULT_DELIVERY:
                    try:
                        await self._queue_frontend_response(
                            frontend_response,
                            capacity_reserved=acknowledgement_queued,
                        )
                    except FacadeProtocolError as error:
                        await self._emit_speech_delivery_failure(
                            purpose=_ResponsePurpose.DELIVERY,
                            speech_purpose=frontend_response.purpose.value,
                            local_request_id=frontend_response.local_request_id,
                            phase="failed",
                            reason=error.code,
                        )
                else:
                    await self._queue_frontend_response(
                        frontend_response,
                        capacity_reserved=acknowledgement_queued,
                    )
            if tool_output_sent and not admission_released:
                # The Realtime tool has a local outcome and the natural
                # follow-up response now owns its place in the arbiter.
                # The backend iterator remains live after this point.
                await self._release_admission_barrier(upstream_call_id)
                admission_released = True

        try:
            async for update in updates:
                if not isinstance(update, InteractionUpdate) or pending_terminal is not None:
                    raise FacadeProtocolError(
                        "runtime_protocol_error", "The VoiceClaw interaction manager returned invalid updates."
                    )
                if len(update.text) > _MAX_OUTPUT_CHARACTERS or "\x00" in update.text:
                    raise FacadeProtocolError(
                        "runtime_protocol_error", "The VoiceClaw interaction manager returned an invalid result."
                    )
                if update.terminal:
                    # A terminal update is a commit boundary. Hold its display
                    # and speech effects until the iterator proves that no
                    # contradictory update follows it.
                    pending_terminal = update
                    continue
                await apply_update(update)
            terminal = pending_terminal
            if terminal is not None:
                has_one_tool_outcome = tool_output_sent != (terminal.tool_output is not None)
                if not has_one_tool_outcome:
                    terminal = None
                else:
                    await apply_update(terminal)
        except asyncio.CancelledError:
            with suppress(Exception):
                await discard_provisional_display(
                    status=ProjectionStreamAbortStatus.CANCELLED,
                    reason="client_cancelled",
                )
            close = getattr(updates, "aclose", None)
            if callable(close):
                with suppress(Exception):
                    closing = close()
                    if inspect.isawaitable(closing):
                        await closing
            raise
        except Exception:
            terminal = None
            close = getattr(updates, "aclose", None)
            if callable(close):
                with suppress(Exception):
                    closing = close()
                    if inspect.isawaitable(closing):
                        await closing
        if terminal is None or not tool_output_sent:
            failure_copy = self._failure_copy(failure_code)
            await discard_provisional_display(reason=failure_code)
            await self._send_public_error(failure_code, failure_copy.display)
            failure_correlation = {
                "local_request_id": local_request_id,
                "identity_authority": "voiceclaw_local",
                "error_code": failure_code,
            }
            await self._emit_projection(
                kind=ResponseOnlyUpdateKind.BACKEND_TURN,
                phase=ResponseOnlyRequestState.FAILED,
                title=failure_copy.title,
                text=failure_copy.display,
                correlation=failure_correlation,
            )
            failure_response = FrontendResponse(
                purpose=FrontendResponsePurpose.FAILURE_DELIVERY,
                local_request_id=local_request_id,
                payload_text=failure_copy.speech,
            )
            await self._queue_frontend_response(failure_response, capacity_reserved=True)
            if not tool_output_sent:
                await self._send_tool_output(
                    upstream_call_id,
                    {"status": "failed"},
                )
            if not admission_released:
                await self._release_admission_barrier(upstream_call_id)
            return
        if terminal.phase is ResponseOnlyRequestState.FAILED:
            terminal_error_code = terminal.correlation.get("error_code")
            if (
                not isinstance(terminal_error_code, str)
                or not terminal_error_code
                or len(terminal_error_code) > 128
                or not terminal_error_code.replace("_", "").isalnum()
            ):
                terminal_error_code = failure_code
            await self._send_public_error(
                terminal_error_code,
                terminal.text,
            )

    def _failure_copy(self, code: str) -> FailureCopy:
        """Resolve only stable machine codes; malformed internal codes use the reviewed default."""
        try:
            return self._model_contracts.failure_copy(code)
        except ModelContractError:
            return self._model_contracts.default_failure_copy

    async def _return_tool_result(self, upstream_call_id: str, output: dict[str, str], failure_code: str) -> None:
        """Return an immediate pre-runtime failure and schedule natural speech."""
        failure_copy = self._failure_copy(failure_code)
        await self._send_tool_output(upstream_call_id, output)
        await self._queue_frontend_response(
            FrontendResponse(
                purpose=FrontendResponsePurpose.FAILURE_DELIVERY,
                payload_text=failure_copy.speech,
            )
        )

    async def _queue_frontend_response(
        self,
        response: FrontendResponse,
        *,
        capacity_reserved: bool = False,
    ) -> None:
        """Queue one typed application-owned speech turn."""
        context = self._frontend_response_context(response)
        max_output_tokens = (
            min(_MAX_REALTIME_RESPONSE_TOKENS, len(response.payload_text.encode("utf-8")) + 16)
            if response.purpose is FrontendResponsePurpose.RESULT_DELIVERY
            else 96
        )
        # Application-owned speech must not continue the private canonical
        # tool tail.  After an acknowledgement that tail ends in the earlier
        # assistant response, which can make a later result generation replay
        # it even though the fresh response instructions contain the result.
        # Use response-local input so inference sees only the authoritative
        # instructions (including live projection and typed payload) plus this
        # generic server-owned turn trigger.  The generated assistant item is
        # still committed to the normal Realtime conversation.
        response_body: dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": self._model_contracts.render_instruction("application_response_turn"),
                        }
                    ],
                }
            ],
        }
        event = {
            "event_id": self._id_factory("event_vc"),
            "type": "response.create",
            "response": response_body,
        }
        await self._enqueue_response_create(
            event,
            purpose=(
                _ResponsePurpose.ACKNOWLEDGEMENT
                if response.purpose is FrontendResponsePurpose.DELEGATION_ACK
                else _ResponsePurpose.DELIVERY
            ),
            requires_speech_floor=True,
            capacity_reserved=capacity_reserved,
            speech_purpose=response.purpose.value,
            local_request_id=response.local_request_id,
            server_response_context=context,
        )

    def _frontend_response_context(self, response: FrontendResponse) -> dict[str, str]:
        """Serialize the already validated typed payload for one speech handoff."""
        context = {
            "response_purpose": response.purpose.value,
            "payload_text": response.payload_text,
        }
        serialized = json.dumps(context, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        if len(serialized) > self._context_character_budget or "\x00" in serialized:
            raise FacadeProtocolError("runtime_protocol_error", "The VoiceClaw frontend response context is invalid.")
        return context

    async def _emit_speech_delivery_failure(
        self,
        *,
        purpose: _ResponsePurpose | None,
        speech_purpose: str | None,
        local_request_id: str | None,
        phase: str,
        reason: str,
    ) -> None:
        """Project a speech-channel failure without changing backend Work state."""
        if purpose not in _SERVER_SPEECH_PURPOSES:
            return
        correlation = {
            "speech_purpose": speech_purpose or purpose.value,
            "delivery_state": phase,
            "reason": reason,
        }
        if local_request_id is not None:
            correlation["local_request_id"] = local_request_id
        cancelled = phase == "cancelled"
        await self._emit_projection(
            kind="speech_delivery",
            phase=phase,
            title="Speech delivery cancelled" if cancelled else "Speech delivery failed",
            text=(
                "The queued spoken response was cancelled before completion."
                if cancelled
                else "The realtime frontend could not generate the queued spoken response."
            ),
            correlation=correlation,
        )

    async def _send_tool_output(self, upstream_call_id: str, output: dict[str, str]) -> bool:
        """Return one local dispatch receipt immediately after its originating call."""
        call = self._registry.claim_server_call_output(upstream_call_id)
        if call is None:
            return False
        await self._send_upstream(
            {
                "event_id": self._id_factory("event_vc"),
                "type": "conversation.item.create",
                "previous_item_id": call.item_id,
                "item": {
                    "type": "function_call_output",
                    "call_id": upstream_call_id,
                    "output": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
                },
            }
        )
        return True

    async def _emit_projection(
        self,
        *,
        kind: str,
        phase: str,
        title: str,
        text: str,
        correlation: Mapping[str, str],
        request_summary: str | None = None,
    ) -> None:
        projection = Projection(
            session_id=self._facade_session_id,
            kind=kind,
            phase=phase,
            title=title,
            text=text,
            request_summary=request_summary,
            correlation=correlation,
        )
        for event in self._projection_events.render(projection):
            await self._send_downstream(event)

    def _rewrite_upstream_event(self, event: dict[str, Any]) -> dict[str, Any]:
        value = _strip_sensitive(event)
        assert isinstance(value, dict)
        value["event_id"] = self._id_factory("event_vc")
        if "response_id" in value:
            value["response_id"] = self._registry.response(value["response_id"])
        if "item_id" in value:
            value["item_id"] = self._registry.ensure_item(value["item_id"]).local_id
        if "call_id" in value:
            value["call_id"] = self._registry.call(value["call_id"]).local_id
        if isinstance(value.get("response"), dict):
            value["response"] = self._rewrite_response(value["response"])
        if isinstance(value.get("item"), dict):
            value["item"] = self._rewrite_item(value["item"])
        if isinstance(value.get("session"), dict):
            value["session"]["id"] = self._facade_session_id
            value["session"]["model"] = self._public_model
            audio = value["session"].get("audio")
            input_audio = audio.get("input") if isinstance(audio, dict) else None
            turn_detection = input_audio.get("turn_detection") if isinstance(input_audio, dict) else None
            if isinstance(turn_detection, dict):
                turn_detection["create_response"] = self._automatic_response
        if isinstance(value.get("conversation"), dict):
            value["conversation"]["id"] = self._facade_conversation_id
        if "previous_item_id" in value and value["previous_item_id"] is not None:
            value["previous_item_id"] = self._registry.public_predecessor(value["previous_item_id"])
        return value

    def _rewrite_response(self, response: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(response)
        value["id"] = self._registry.response(value.get("id"))
        # Preserve ordinary client correlation metadata, but upstreams can
        # never author the server-owned VoiceClaw projection namespace.
        metadata = value.get("metadata")
        value["metadata"] = (
            {key: item for key, item in metadata.items() if not _reserved_metadata_key(key)}
            if isinstance(metadata, dict)
            else {}
        )
        if value.get("conversation_id") is not None:
            value["conversation_id"] = self._facade_conversation_id
        output = value.get("output")
        if isinstance(output, list):
            value["output"] = [self._rewrite_item(item) if isinstance(item, dict) else item for item in output]
        return value

    def _rewrite_item(self, item: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(item)
        if value.get("id") is not None:
            value["id"] = self._registry.ensure_item(value["id"]).local_id
        if value.get("call_id") is not None:
            value["call_id"] = self._registry.call(value["call_id"]).local_id
        return value

    async def _receive_downstream(self) -> dict[str, Any]:
        event = _decode_event(await self._downstream.receive(), maximum_bytes=self._max_event_bytes)
        await self._observe(ActivityDirection.DOWNSTREAM_RECEIVED, event["type"])
        return event

    async def _receive_upstream(self) -> dict[str, Any]:
        event = _decode_event(await self._upstream.receive(), maximum_bytes=self._max_event_bytes)
        await self._observe(ActivityDirection.UPSTREAM_RECEIVED, event["type"])
        return event

    async def _send_downstream(self, event: Mapping[str, Any]) -> None:
        message = _encode_event(event, maximum_bytes=self._max_event_bytes)
        async with self._downstream_send_lock:
            await self._downstream.send(message)
        await self._observe(ActivityDirection.DOWNSTREAM_SENT, str(event.get("type", "")))

    async def _send_upstream(self, event: Mapping[str, Any]) -> None:
        message = _encode_event(event, maximum_bytes=self._max_event_bytes)
        async with self._upstream_send_lock:
            await self._upstream.send(message)
        await self._observe(ActivityDirection.UPSTREAM_SENT, str(event.get("type", "")))

    async def _send_upstream_batch(self, events: list[Mapping[str, Any]]) -> None:
        """Write an ordered group without allowing another producer to interleave."""
        messages = [_encode_event(event, maximum_bytes=self._max_event_bytes) for event in events]
        async with self._upstream_send_lock:
            for message in messages:
                await self._upstream.send(message)
        for event in events:
            await self._observe(ActivityDirection.UPSTREAM_SENT, str(event.get("type", "")))

    async def _send_public_error(self, code: str, message: str, *, event_id: object = None) -> None:
        safe_code = code if isinstance(code, str) and code.replace("_", "").isalnum() else "voiceclaw_error"
        safe_message = message if isinstance(message, str) and len(message) <= 512 else "The realtime request failed."
        error: dict[str, Any] = {
            "type": "error",
            "event_id": self._id_factory("event_vc"),
            "error": {"type": safe_code, "code": safe_code, "message": safe_message, "param": None},
        }
        if isinstance(event_id, str) and len(event_id) <= _MAX_IDENTIFIER_CHARACTERS:
            error["error"]["event_id"] = event_id
        try:
            await self._send_downstream(error)
        except Exception:
            return

    async def _observe(self, direction: ActivityDirection, event_type: str) -> None:
        if self._observer is None:
            return
        try:
            result = self._observer(RealtimeActivity(direction=direction, event_type=event_type))
            if inspect.isawaitable(result):
                await result
        except Exception:
            return

    @staticmethod
    def _new_id(prefix: str) -> str:
        return f"{prefix}_{secrets.token_urlsafe(12)}"


__all__ = [
    "ActivityDirection",
    "ActivityObserver",
    "FacadeProtocolError",
    "RealtimeActivity",
    "RealtimeTransport",
    "VoiceClawRealtimeFacade",
]
