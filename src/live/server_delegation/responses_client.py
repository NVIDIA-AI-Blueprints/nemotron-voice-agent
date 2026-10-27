# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The server delegation's port to the delegation core: the official Responses API, types only.

The gateway holds one client, built on first use by the composition root's ``core`` hook
(``live.gateway.gateway.CoreClient``); the in-process core passes every request, result and event through a JSON round
trip (``delegation_core/api/local.py``). Every client meets the contract in ``tests/unit/contracts/responses.py``:

* a refusal comes before any event. ``create_response`` returns once the core accepts the request and raises on a
  refusal, with the refusal's message and a ``code`` attribute: the official code for a refused request the client
  made itself (``function_call_outputs_required``), else ``None``. An invalid request (malformed, a repeated result, a
  result of another type than its call, or results for several responses at once) raises ``ValueError``, as does
  ``retrieve_model`` for a model the core does not serve;
* each request reads one stream, which ends after its response's terminal event;
* requests are accepted one at a time, in the order they are sent;
* a response opens at acceptance (``response.created`` with status ``queued``, then ``response.queued``) and goes
  ``in_progress`` when work runs for it; a response that takes over an older delegation's work names that delegation's
  response in ``previous_response_id`` before the older response ends;
* ``input_tokens`` checks a request's settings without running it. A refused setting raises ``ValueError`` naming every
  setting at fault, with an optional ``param`` attribute (``model``, ``tools[2].version``) that the gateway reports as
  ``session.delegation.responses.<param>``.

A client may set ``close_seconds`` (or ``None``), how long its core's close of a conversation may take; the server
delegation sizes its own close limit from it. ``aclose`` releases what the client holds; the gateway calls it once, at
server shutdown, after every session has closed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol


class ResponseStream(Protocol):
    """The official streaming events of one response, in order; the iteration ends after its terminal."""

    def __aiter__(self) -> AsyncIterator[dict]:
        """Iterate the events."""

    async def aclose(self) -> None:
        """Stop reading: a stateless request's one-shot work ends (over HTTP, the client's disconnect would end it)."""


class ResponsesClient(Protocol):
    """The delegation core's official endpoints, as the server delegation calls them."""

    async def create_conversation(self, metadata: dict[str, str]) -> dict:
        """``POST /v1/conversations``: open a conversation; return the conversation object."""

    async def delete_conversation(self, conversation_id: str) -> dict:
        """``DELETE /v1/conversations/{id}``: delete it; the core also stops the conversation's work."""

    async def retrieve_model(self, model: str) -> dict:
        """``GET /v1/models/{model}``: the model object; raise ``ValueError`` if the core does not serve ``model``."""

    async def create_response(self, params: dict) -> ResponseStream:
        """``POST /v1/responses`` with ``stream: true``: return the stream once the core has accepted the request."""

    async def list_conversation_items(self, conversation_id: str) -> dict:
        """``GET /v1/conversations/{id}/items``: the conversation's items, as an official list object."""

    async def input_tokens(self, params: dict) -> dict:
        """``POST /v1/responses/input_tokens``: count a request's input; ``ValueError`` (with ``param``) if refused."""

    async def aclose(self) -> None:
        """Release what the client holds (the in-process client closes its core); called once, at server shutdown."""
