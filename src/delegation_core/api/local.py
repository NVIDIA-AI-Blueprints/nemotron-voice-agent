# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process client of the delegation core's Responses API endpoints.

Every request, result, streamed event and refusal crosses through a JSON round trip, as over HTTP, so the two sides
never share an object.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable

from delegation_core.api.service import InvalidRequest, NotFound, ResponsesService, StatelessStream
from delegation_core.im.session import DelegationError
from delegation_core.im.stream import ResponseStream


def round_trip(value):
    """``value`` as JSON would carry it: encoded and decoded, so nothing of the original is shared."""
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
        """The next event, as JSON carries it; ``StopAsyncIteration`` once the response has ended."""
        return round_trip(await self._stream.__anext__())

    async def aclose(self) -> None:
        """Stop reading, as the stream's kind closes: a one-shot session ends now; a conversation's response goes on."""
        await self._stream.aclose()


async def _nothing_to_release() -> None:
    """The close of a client whose root handed it none: it holds nothing to release."""


class LocalResponsesClient:
    """The delegation core's endpoints, reached in process.

    ``close_seconds`` is how long the core's close of a conversation may take, which the gateway waits for when a
    session closes (``None`` when unknown). ``close_core`` releases the core and its backend sources on :meth:`aclose`.
    """

    def __init__(
        self,
        service: ResponsesService,
        *,
        close_seconds: float | None = None,
        close_core: Callable[[], Awaitable[None]] = _nothing_to_release,
    ):
        """Reach the core whose endpoints ``service`` serves, with that core's close budget and its close."""
        self.service, self.close_seconds, self._close_core = service, close_seconds, close_core

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """``POST /v1/conversations``: open a conversation; return the conversation object."""
        return round_trip(await _refusals_as_json(self.service.create_conversation(round_trip(metadata))))

    async def delete_conversation(self, conversation_id: str) -> dict:
        """``DELETE /v1/conversations/{id}``: delete it, which stops its work; return the deleted object."""
        return round_trip(await _refusals_as_json(self.service.delete_conversation(round_trip(conversation_id))))

    async def retrieve_model(self, model: str) -> dict:
        """``GET /v1/models/{model}``: the model object; ``ValueError`` if the core does not serve ``model``."""
        return round_trip(await _refusals_as_json(self.service.retrieve_model(round_trip(model))))

    async def create_response(self, params: dict) -> LocalStream:
        """``POST /v1/responses`` with ``stream: true``: return the stream once the core has accepted the request."""
        return LocalStream(await _refusals_as_json(self.service.create_response(round_trip(params))))

    async def input_tokens(self, params: dict) -> dict:
        """``POST /v1/responses/input_tokens``: the request's input count; ``InvalidRequest`` for what it refuses."""
        return round_trip(await _refusals_as_json(self.service.input_tokens(round_trip(params))))

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """``GET /v1/conversations/{id}/items``: the conversation's items, as an official list object."""
        return round_trip(await _refusals_as_json(self.service.list_conversation_items(round_trip(conversation_id))))

    async def aclose(self) -> None:
        """Release what the client holds: its core, by the close the root handed it (``close_core``)."""
        await self._close_core()
