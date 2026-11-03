# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""The gateway's two delegation parts against ``live.ports.DelegationPort``.

Both parts keep the contract; the server delegation refuses a delegation made before ``listen()``.
"""

import unittest

from live.client_delegation import ClientDelegation
from live.server_delegation import ServerDelegation
from tests.unit.contracts.delegation_port import DelegationPortContract


class View:
    """The session's delegation view (``live.ports.DelegationView``): clock, settings, events and refusals."""

    id, input_ms, responses, acknowledgement_delay_ms = "live_test", 0, None, 0

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, kind, **fields):
        self.events.append((kind, fields))

    async def emit_paced(self, kind, **fields):
        self.events.append((kind, fields))

    def report_refusal(self, message, code, client_event_id, param=None):
        self.events.append(("error", {"message": message, "code": code, "client_event_id": client_event_id}))


class Answers:
    """A delegation core's official endpoints, as a ``ResponsesClient``, that answer each request at once."""

    async def create_conversation(self, metadata):
        return {"id": "conv_test", "object": "conversation", "created_at": 0, "metadata": dict(metadata)}

    async def delete_conversation(self, conversation_id):
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    async def retrieve_model(self, model):
        return {"id": model, "object": "model", "created": 0, "owned_by": "test"}

    async def create_response(self, params):
        return Stream(
            [
                {"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}},
                {"type": "response.completed", "response": {"id": "resp_1", "status": "completed", "output": []}},
            ]
        )

    async def list_conversation_items(self, conversation_id):
        return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}

    async def aclose(self):
        """The client's own close: it holds nothing to release."""


class Stream:
    def __init__(self, events):
        self.events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)


class ClientDelegationPortTests(DelegationPortContract, unittest.IsolatedAsyncioTestCase):
    async def make_port(self):
        part = ClientDelegation()
        await part.open(View())
        self.addAsyncCleanup(part.close)
        return part


class ServerDelegationPortTests(DelegationPortContract, unittest.IsolatedAsyncioTestCase):
    async def make_port(self):
        part = ServerDelegation(Answers())
        await part.open(View())
        self.addAsyncCleanup(part.close)
        return part
