# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation's port to the delegation core: the official Responses API, types only."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class ResponseStream(Protocol):
    """The official streaming events of one response, in order; the iteration ends after its terminal."""

    def __aiter__(self) -> AsyncIterator[dict]:
        """Iterate the events."""

    async def aclose(self) -> None:
        """Stop reading; a stateless request's one-shot work ends with it."""


class ResponsesClient(Protocol):
    """The delegation core's official endpoints, as the server delegation calls them.

    A client may set ``close_seconds`` (or ``None``): how long its core's close of a conversation may take.
    """

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """Open a conversation (``POST /v1/conversations``) and return the conversation object."""

    async def delete_conversation(self, conversation_id: str) -> dict:
        """Delete the conversation (``DELETE /v1/conversations/{id}``); the core also stops its work."""

    async def retrieve_model(self, model: str) -> dict:
        """Return the model object (``GET /v1/models/{model}``); raise ``ValueError`` if the core does not serve it."""

    async def create_response(self, params: dict) -> ResponseStream:
        """Send ``POST /v1/responses`` with ``stream: true`` and return the stream once the core accepts the request.

        Requests are accepted one at a time, in order. A refusal raises before any event: an invalid request (malformed,
        a repeated result, a result of another type than its call, results for several responses at once) as
        ``ValueError``, any other with a ``code`` attribute, an official code or ``None``.
        """

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """Return the conversation's items as an official list object (``GET /v1/conversations/{id}/items``)."""

    async def input_tokens(self, params: dict) -> dict:
        """Count a request's input without running it (``POST /v1/responses/input_tokens``).

        A refused setting raises ``ValueError`` naming every setting at fault, with an optional ``param`` attribute.
        """

    async def aclose(self) -> None:
        """Release what the client holds; the gateway calls it once, at server shutdown, after every session closed."""
