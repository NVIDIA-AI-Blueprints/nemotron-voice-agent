# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process client of the delegation core's Responses API endpoints, through a JSON round trip as over HTTP."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from examples.frontend_backend_live.delegation_core.api.service import (
    InvalidRequest,
    NotFound,
    ResponsesService,
    StatelessStream,
)
from examples.frontend_backend_live.delegation_core.im.session import DelegationError
from examples.frontend_backend_live.delegation_core.im.stream import ResponseStream


def round_trip(value):
    """Return ``value`` as JSON would carry it: encoded and decoded, so nothing of the original is shared."""
    return json.loads(json.dumps(value))


async def _refusals_as_json(call: Awaitable):
    """Await ``call``; a refusal is raised again as a new exception rebuilt from its JSON-safe message and code."""
    try:
        return await call
    except DelegationError as refused:
        message, code = round_trip([str(refused), refused.code])
        raise DelegationError(message, code) from None
    except NotFound as refused:
        raise NotFound(round_trip(str(refused))) from None
    except InvalidRequest as refused:
        message, param, code = round_trip([str(refused), refused.param, refused.code])
        raise InvalidRequest(message, param=param, code=code) from None
    except ValueError as refused:
        raise ValueError(round_trip(str(refused))) from None


class LocalStream:
    """The events of one response, each through a JSON round trip, in order; it ends after the response's terminal."""

    def __init__(self, stream: ResponseStream | StatelessStream):
        """Read ``stream``, the core's stream of the request's response."""
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[dict]:
        """Return the stream itself."""
        return self

    async def __anext__(self) -> dict:
        """Return the next event, as JSON carries it; ``StopAsyncIteration`` once the response has ended."""
        return round_trip(await self._stream.__anext__())

    async def aclose(self) -> None:
        """Stop reading, as the stream's kind closes: a one-shot session ends now; a conversation's response goes on."""
        await self._stream.aclose()


class LocalResponsesClient:
    """The delegation core's endpoints, reached in process."""

    def __init__(
        self,
        service: ResponsesService,
        *,
        close_seconds: float | None,
        close_core: Callable[[], Awaitable[None]],
    ):
        """Reach the core whose endpoints ``service`` serves, with that core's close budget and its close."""
        self.service, self.close_seconds, self._close_core = service, close_seconds, close_core

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """Open a conversation (``POST /v1/conversations``) and return the conversation object."""
        return round_trip(await _refusals_as_json(self.service.create_conversation(round_trip(metadata))))

    async def update_conversation(self, conversation_id: str, metadata: dict[str, str]) -> dict:
        """Attach a later call to a kept conversation (``POST /v1/conversations/{id}``) and return the conversation."""
        update = self.service.update_conversation(round_trip(conversation_id), round_trip(metadata))
        return round_trip(await _refusals_as_json(update))

    async def delete_conversation(self, conversation_id: str) -> dict:
        """Delete a conversation (``DELETE /v1/conversations/{id}``); return the deleted one.

        Its work stops, unless work that outlives the call keeps the conversation: then ``deleted`` is ``false``.
        """
        return round_trip(await _refusals_as_json(self.service.delete_conversation(round_trip(conversation_id))))

    async def retrieve_model(self, model: str) -> dict:
        """Return the model object (``GET /v1/models/{model}``); ``ValueError`` if the core does not serve it."""
        return round_trip(await _refusals_as_json(self.service.retrieve_model(round_trip(model))))

    async def create_response(self, params: dict) -> LocalStream:
        """Return a request's stream once the core has accepted the request (``POST /v1/responses``, streamed)."""
        return LocalStream(await _refusals_as_json(self.service.create_response(round_trip(params))))

    async def input_tokens(self, params: dict) -> dict:
        """Count a request's input (``POST /v1/responses/input_tokens``); ``InvalidRequest`` for what it refuses."""
        return round_trip(await _refusals_as_json(self.service.input_tokens(round_trip(params))))

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """Return a conversation's items as an official list object (``GET /v1/conversations/{id}/items``)."""
        return round_trip(await _refusals_as_json(self.service.list_conversation_items(round_trip(conversation_id))))

    async def aclose(self) -> None:
        """Close every conversation the core still serves, kept ones included, then release the core by its close."""
        try:
            await self.service.aclose()
        finally:
            await self._close_core()
