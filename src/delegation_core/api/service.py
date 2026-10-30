# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core's Responses API endpoints as Python functions.

Each conversation is served by one IM session, and the conversation's id is that session's. ``create_response``
checks and admits a request before it returns the stream, so a refusal is raised before any event. A request that
brings function outputs, is a typed run (``metadata.live_typed_run``) or names no ``metadata.live_delegation_id`` is
the client's own; any other request is a delegation. A request that names no conversation is stateless: a one-shot IM
session serves it and ends with its response, so nothing of it is kept.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any, Protocol

from delegation_core.ids import approx_tokens
from delegation_core.im.input import request_settings
from delegation_core.im.session import IMSession
from delegation_core.im.stream import ResponseStream

# The metadata key that names a request's delegation, and the one that marks a typed run, with its value.
DELEGATION_ID = "live_delegation_id"
TYPED_RUN = "live_typed_run"
# Who owns the models the core serves, as the model object names it.
MODEL_OWNER = "delegation_core"
# How deep a request may nest objects and lists: the IM copies a request, and Python's copy recurses.
MAX_REQUEST_DEPTH = 64


def nested_deeper_than(value: object, limit: int) -> bool:
    """Whether ``value`` nests objects and lists more than ``limit`` levels deep; it reads them level by level."""
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
    """A request the core refuses: the official ``invalid_request_error`` (HTTP 400).

    ``param`` names the request's field at fault; ``code`` is the official code when there is one
    (``previous_response_not_found``), ``None`` otherwise.
    """

    def __init__(self, message: str, *, param: str | None = None, code: str | None = None):
        """Refuse a request with ``message``, naming the field at fault and the official code when there is one."""
        super().__init__(message)
        self.param, self.code = param, code


class NotFound(LookupError):
    """Raised for a conversation that does not exist (HTTP 404)."""


class BackendRules(Protocol):
    """What the core's backend says about a request: the models it serves, and the settings it cannot honor.

    The API may not import the backend port, so this names the two members of its ``AdapterSource`` that it reads.
    """

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings it cannot honor, as ``(param, message)``."""


class ResponsesService:
    """The official Responses endpoints of one delegation core: its conversations, its model rule and its responses."""

    def __init__(
        self,
        new_session: Callable[[], IMSession],
        *,
        stateless_session: Callable[[], IMSession] | None = None,
        backend: BackendRules,
    ):
        """Serve the conversations of a core whose IM sessions ``new_session`` makes, one for each conversation.

        Args:
            new_session: Makes the IM session of each new conversation.
            stateless_session: Makes the one-shot IM session of each stateless request; without it, a request that
                names no conversation is refused.
            backend: The core's backend rules (its ``AdapterSource``), which answer the model check, with or without an
                open conversation, and name the settings it cannot honor.
        """
        self.new_session, self.stateless_session, self.backend = new_session, stateless_session, backend
        self.sessions: dict[str, IMSession] = {}  # each open conversation's IM session, by conversation id
        self.conversations: dict[str, dict] = {}  # each open conversation's object, by its id
        self.created_at = int(time.time())

    # ------------------------------------------------------------------------------ conversations
    async def create_conversation(self, metadata: dict[str, str] | None = None) -> dict:
        """``POST /v1/conversations``: a new IM session, started at once; return the conversation object.

        Raises:
            InvalidRequest: ``metadata`` is not an object of strings.
        """
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict) or not all(isinstance(value, str) for value in metadata.values()):
            raise InvalidRequest("A conversation's metadata must be an object of strings", param="metadata")
        session = self.new_session()
        conversation_id = session.ledger.session.id
        conversation = {
            "id": conversation_id,
            "object": "conversation",
            "created_at": int(time.time()),
            "metadata": deepcopy(metadata),
        }
        self.sessions[conversation_id], self.conversations[conversation_id] = session, conversation
        session.start()
        return deepcopy(conversation)

    async def delete_conversation(self, conversation_id: str) -> dict:
        """``DELETE /v1/conversations/{id}``: end its IM session, then forget it; return the deleted object.

        The IM session's close stops the conversation's work and ends every open response first, so the conversation
        is forgotten only once its last terminal is written. A close that raises or is cut short still forgets it.

        Raises:
            NotFound: No open conversation has this id.
        """
        session = self._session(conversation_id)
        try:
            await session.close()
        finally:
            if self.sessions.get(conversation_id) is session:
                del self.sessions[conversation_id], self.conversations[conversation_id]
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """``GET /v1/conversations/{id}/items``: the conversation's backend history, oldest first, as one list page.

        Raises:
            NotFound: No open conversation has this id.
        """
        items = deepcopy(self._session(conversation_id).history)
        return {
            "object": "list",
            "data": items,
            "first_id": items[0].get("id") if items else None,
            "last_id": items[-1].get("id") if items else None,
            "has_more": False,
        }

    def _session(self, conversation_id: object) -> IMSession:
        if isinstance(conversation_id, dict):  # the request's conversation may be given as an object with its id
            conversation_id = conversation_id.get("id")
        session = self.sessions.get(conversation_id) if isinstance(conversation_id, str) else None
        if session is None:
            raise NotFound(f"Conversation {conversation_id!r} not found")
        return session

    # ------------------------------------------------------------------------------ models
    async def retrieve_model(self, model: str) -> dict:
        """``GET /v1/models/{model}``: the model object, if the backend serves ``model``.

        Raises:
            ValueError: The backend does not serve ``model``.
        """
        await self.backend.check_model(model)
        return {"id": model, "object": "model", "created": self.created_at, "owned_by": MODEL_OWNER}

    # ------------------------------------------------------------------------------ responses
    async def create_response(self, params: dict) -> ResponseStream | StatelessStream:
        """``POST /v1/responses`` (stream): check the request, admit it, then return its stream.

        A request that names no conversation is stateless: it is served in a one-shot IM session of its own
        (:meth:`_stateless`).

        Raises:
            InvalidRequest: The request is malformed (see :func:`checked`), or it names no ``conversation`` and
                the core serves no stateless requests.
            NotFound: The request's conversation does not exist.
            DelegationError: The IM session refuses the request (``im/session.py``), before any event, with the
                official ``code`` when one names the refusal.
            ValueError: The function outputs the request brings are malformed (``IMSession.admit_client_request``).
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
        """``POST /v1/responses/input_tokens``: check a request's settings without running it; count its input.

        It makes the checks ``create_response`` makes and asks the backend about the model and settings; nothing is
        admitted. The count estimates the instructions, tools, input and the named conversation's items.

        Raises:
            InvalidRequest: The request is malformed; or the backend does not serve its model (``param: model``); or
                it cannot honor some of its settings: the first is the error's ``param``, and the message names each.
            NotFound: The request's conversation does not exist.
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
        """A stateless request: a one-shot IM session admits it, and its stream ends that session with the response.

        The session is kept nowhere, so nothing of the request outlives its response, and many such requests run at
        once.
        """
        if self.stateless_session is None:
            raise InvalidRequest("A request must name its conversation", param="conversation")
        session = self.stateless_session()
        session.start()
        try:
            admission = session.admit(input_items, settings=request_settings(params))
        except Exception:
            await session.close()
            raise
        return StatelessStream(admission.stream, session)


def checked(params: dict) -> tuple[list[dict], dict]:
    """A request's input items and metadata, once the request passes the official checks.

    Raises:
        InvalidRequest: It nests too deeply, its ``input`` is not a list of objects or its ``metadata`` not an object,
            or it names a ``previous_response_id`` (the core stores no responses).
    """
    if nested_deeper_than(params, MAX_REQUEST_DEPTH):  # first: the IM copies the request, and Python's copy recurses
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
    """Whether a request is the client's own, which the IM judges by its own rules.

    It is when it brings function outputs, when the client marks it a typed run, or when it names no delegation (a
    plain Responses client's request); a request that names its delegation and brings no outputs is a delegation.
    """
    return (
        any(item.get("type") == "function_call_output" for item in input_items)
        or metadata.get(TYPED_RUN) == "true"
        or metadata.get(DELEGATION_ID) is None
    )


class StatelessStream:
    """The stream of a stateless request's one response; its one-shot IM session ends with it.

    The session ends once the response's last event was read, or when the reader stops reading (:meth:`aclose`; over
    HTTP, the client's disconnect). Ending it stops the request's work if it still runs (its open response then ends
    ``response.incomplete`` with status ``cancelled``) and closes its backend adapter.
    """

    def __init__(self, stream: ResponseStream, session: IMSession):
        """Read ``stream``, the response of the one request that ``session`` serves."""
        self.stream, self.session = stream, session

    def __aiter__(self) -> StatelessStream:
        """Return the stream itself."""
        return self

    async def __anext__(self) -> dict:
        """The next event; after the last, the session ends before the iteration stops."""
        try:
            return await self.stream.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        """End the one-shot session now; safe to call twice."""
        await self.session.close()
