# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core's Responses API endpoints as Python functions."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Protocol

from examples.frontend_backend_live.delegation_core.im.input import request_settings
from examples.frontend_backend_live.delegation_core.im.session import DelegationError, IMSession
from examples.frontend_backend_live.delegation_core.im.stream import ResponseStream
from examples.shared.delegation_contract import DELEGATION_ID, STORE, TYPED_RUN
from examples.shared.ids import approx_tokens

MODEL_OWNER = "delegation_core"
MAX_REQUEST_DEPTH = 64
# A refusal never names a conversation: its id is what lets a later call attach to it.
NOT_FOUND = "Conversation not found"


def nested_deeper_than(value: object, limit: int) -> bool:
    """Return whether ``value`` nests objects and lists more than ``limit`` levels deep, read level by level."""
    level = [value]
    for _ in range(limit):
        level = [
            child
            for node in level
            for child in (node.values() if isinstance(node, dict) else node if isinstance(node, list) else ())
        ]
        if not level:
            return False
    return any(isinstance(node, (dict, list)) for node in level)


class InvalidRequest(ValueError):
    """A request the core refuses: an official ``invalid_request_error``, with the ``param`` at fault and a ``code``."""

    def __init__(self, message: str, *, param: str | None = None, code: str | None = None):
        """Refuse a request with ``message``, naming the field at fault and the official code when there is one."""
        super().__init__(message)
        self.param, self.code = param, code


class NotFound(LookupError):
    """Raised for a conversation that does not exist (HTTP 404)."""


class BackendRules(Protocol):
    """The two ``AdapterSource`` members the API reads, named here since the API may not import the backend port."""

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings it cannot honor, as ``(param, message)``."""


class ResponsesService:
    """The official Responses endpoints of one delegation core: its conversations, its model rule and its responses."""

    def __init__(
        self,
        new_session: Callable[..., IMSession],
        *,
        stateless_session: Callable[[], IMSession],
        backend: BackendRules,
    ):
        """Serve the conversations of a core whose IM sessions ``new_session`` makes, one for each conversation.

        ``new_session`` gets each session's ``room`` and ``on_closed`` hooks as keywords.
        """
        self.new_session, self.stateless_session, self.backend = new_session, stateless_session, backend
        self.sessions: dict[str, IMSession] = {}
        self.conversations: dict[str, dict] = {}
        self.created_at = int(time.time())

    async def create_conversation(self, metadata: dict[str, str] | None = None) -> dict:
        """Start a new IM session for a new conversation (``POST /v1/conversations``); return its object.

        ``live_store: "true"`` in its metadata lets its work outlive a call, as :meth:`delete_conversation` says.
        """
        metadata = _metadata(metadata)
        session = self.new_session(room=self._room, on_closed=self._closed)
        conversation_id = session.ledger.session.id
        conversation = {
            "id": conversation_id,
            "object": "conversation",
            "created_at": int(time.time()),
            "metadata": deepcopy(metadata),
        }
        self.sessions[conversation_id], self.conversations[conversation_id] = session, conversation
        session.continuable = metadata.get(STORE) == "true"
        session.start()
        return deepcopy(conversation)

    async def update_conversation(self, conversation_id: str, metadata: dict[str, str] | None = None) -> dict:
        """Attach a later call to a kept conversation, with its new metadata (``POST /v1/conversations/{id}``).

        Return the conversation's object. ``NotFound`` for a conversation that is unknown or has ended, and
        ``DelegationError`` for one that another call holds.
        """
        metadata = _metadata(metadata)
        session = self._session(conversation_id)
        try:
            await session.attach()
        except DelegationError:
            if session.closed:
                raise NotFound(NOT_FOUND) from None
            raise
        conversation = self.conversations[session.ledger.session.id]
        conversation["metadata"] = deepcopy(metadata)
        session.continuable = metadata.get(STORE) == "true"
        return deepcopy(conversation)

    async def delete_conversation(self, conversation_id: str) -> dict:
        """End a conversation's IM session, then forget it (``DELETE /v1/conversations/{id}``).

        A conversation with ``live_store: "true"`` in its metadata is kept instead while its records list work that
        outlives the call, open or among the newest ended (``IMSession.leave``): such work still open goes on, it
        answers ``deleted: false``, and a later call can attach with :meth:`update_conversation`. A kept conversation is
        forgotten once its session closes.
        """
        session = self._session(conversation_id)
        if session.continuable:
            kept = await session.leave()
            return {"id": conversation_id, "object": "conversation.deleted", "deleted": not kept}
        try:
            await session.close()
        finally:
            self._closed(session)
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    def _room(self, session: IMSession) -> bool:
        """Return whether ``session``'s work may outlive its call by the cap of such conversations.

        Fewer than its ``max_after_call_conversations`` other conversations have work open that outlives their call.
        """
        others = sum(other.runs_after_call for other in self.sessions.values() if other is not session)
        return others < session.settings.max_after_call_conversations

    def _closed(self, session: IMSession) -> None:
        """Forget ``session``'s conversation while the session serves it."""
        conversation_id = session.ledger.session.id
        if self.sessions.get(conversation_id) is session:
            del self.sessions[conversation_id], self.conversations[conversation_id]

    async def aclose(self) -> None:
        """Close every conversation's IM session at once, kept ones included, so no work outlives the server."""
        await asyncio.gather(*(session.close() for session in list(self.sessions.values())), return_exceptions=True)

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """Return the backend history, oldest first, as one list page (``GET /v1/conversations/{id}/items``)."""
        items = deepcopy(self._session(conversation_id).history)
        return {
            "object": "list",
            "data": items,
            "first_id": items[0].get("id") if items else None,
            "last_id": items[-1].get("id") if items else None,
            "has_more": False,
        }

    def _session(self, conversation_id: object) -> IMSession:
        # A request may give its conversation as an object with its id.
        if isinstance(conversation_id, dict):
            conversation_id = conversation_id.get("id")
        session = self.sessions.get(conversation_id) if isinstance(conversation_id, str) else None
        if session is None:
            raise NotFound(NOT_FOUND)
        return session

    async def retrieve_model(self, model: str) -> dict:
        """Return the model object (``GET /v1/models/{model}``); ``ValueError`` if the backend does not serve it."""
        await self.backend.check_model(model)
        return {"id": model, "object": "model", "created": self.created_at, "owned_by": MODEL_OWNER}

    async def create_response(self, params: dict) -> ResponseStream | StatelessStream:
        """Check a request, admit it, and return its stream (``POST /v1/responses``, streamed).

        A request with no conversation is served in a one-shot IM session of its own. A refusal raises before any
        event: ``InvalidRequest``, ``NotFound``, ``DelegationError`` (with its official ``code``) or ``ValueError``.
        """
        input_items, metadata = checked(params)
        if params.get("conversation") is None:
            return await self._stateless(params, input_items)
        im = self._session(params["conversation"])
        delegation_id, settings = metadata.get(DELEGATION_ID), request_settings(params)
        if clients_own(input_items, metadata):
            admission = im.admit_client_request(input_items, delegation_id=delegation_id, settings=settings)
        else:
            admission = im.admit(input_items, delegation_id=delegation_id, settings=settings)
        return admission.stream

    async def input_tokens(self, params: dict) -> dict:
        """Count a request's input, checking its settings without running it (``POST /v1/responses/input_tokens``).

        ``InvalidRequest`` names a model the backend does not serve, or the first setting it cannot honor, as ``param``.
        """
        input_items, _ = checked({"input": [], **params})
        history = [] if params.get("conversation") is None else self._session(params["conversation"]).history
        settings = request_settings(params)
        if "model" in settings:
            try:
                await self.backend.check_model(settings["model"])
            except ValueError as refused:
                raise InvalidRequest(str(refused), param="model") from None
        unsupported = self.backend.unsupported(settings)
        if unsupported:
            named = "; ".join(f"{param}: {message}" for param, message in unsupported)
            raise InvalidRequest(f"Unsupported settings: {named}", param=unsupported[0][0])
        counted = [settings.get("instructions") or "", settings.get("tools") or [], history, input_items]
        return {"object": "response.input_tokens", "input_tokens": approx_tokens(json.dumps(counted))}

    async def _stateless(self, params: dict, input_items: list[dict]) -> StatelessStream:
        """Admit a stateless request in a one-shot IM session kept nowhere, which its stream ends with the response."""
        session = self.stateless_session()
        session.start()
        try:
            admission = session.admit(input_items, settings=request_settings(params))
        except Exception:
            await session.close()
            raise
        return StatelessStream(admission.stream, session)


def _metadata(metadata: object) -> dict[str, str]:
    """Return a conversation's metadata, ``{}`` for none; ``InvalidRequest`` unless it is an object of strings."""
    metadata = {} if metadata is None else metadata
    if not isinstance(metadata, dict) or not all(isinstance(value, str) for value in metadata.values()):
        raise InvalidRequest("A conversation's metadata must be an object of strings", param="metadata")
    return metadata


def checked(params: dict) -> tuple[list[dict], dict]:
    """Return a request's input items and metadata, once the request passes the official checks (``InvalidRequest``)."""
    # First: the IM copies a request, and Python's copy recurses.
    if nested_deeper_than(params, MAX_REQUEST_DEPTH):
        raise InvalidRequest(f"A request may nest objects and lists at most {MAX_REQUEST_DEPTH} levels deep")
    input_items, metadata = params.get("input"), params.get("metadata") or {}
    if not isinstance(input_items, list) or not all(isinstance(item, dict) for item in input_items):
        raise InvalidRequest("A request's input must be a list of input items", param="input")
    if not isinstance(metadata, dict):
        raise InvalidRequest("A request's metadata must be an object", param="metadata")
    previous = params.get("previous_response_id")
    if previous is not None and params.get("conversation") is not None:
        raise InvalidRequest(
            "previous_response_id cannot be used together with conversation", param="previous_response_id"
        )
    if previous is not None:
        raise InvalidRequest(
            f"Previous response with id {previous!r} not found: the core stores no responses",
            param="previous_response_id",
            code="previous_response_not_found",
        )
    return input_items, metadata


def clients_own(input_items: list[dict], metadata: dict) -> bool:
    """Return whether a request is the client's own: with function outputs, a typed run, or naming no delegation."""
    return (
        any(item.get("type") == "function_call_output" for item in input_items)
        or metadata.get(TYPED_RUN) == "true"
        or metadata.get(DELEGATION_ID) is None
    )


class StatelessStream:
    """The stream of a stateless request's response; its one-shot session ends with the response or the reader."""

    def __init__(self, stream: ResponseStream, session: IMSession):
        """Read ``stream``, the response of the one request that ``session`` serves."""
        self.stream, self.session = stream, session

    def __aiter__(self) -> StatelessStream:
        """Return the stream itself."""
        return self

    async def __anext__(self) -> dict:
        """Return the next event; after the last, the session ends before the iteration stops."""
        try:
            return await self.stream.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """End the one-shot session now; safe to call twice."""
        await self.session.close()
