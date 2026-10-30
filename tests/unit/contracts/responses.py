# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D103, D107

"""The contract of a ``ResponsesClient`` (``live/server_delegation/responses_client.py``), the port to the core.

The gateway reaches the delegation core only through it. The in-process ``LocalResponsesClient`` binds it; an HTTP
client gets the same contract.

Its guarantees: a refusal comes before any event; each request reads one stream, which ends after its terminal;
requests are accepted one at a time; a response opens ``queued`` at acceptance; ``input_tokens`` validates a request's
settings in the background.
"""

from __future__ import annotations

from typing import Any

from tests.unit.contracts.base import ContractCase, collect

TERMINALS = ("response.completed", "response.failed", "response.incomplete")


def a_request(conversation: str, text: str = "Find teas.", delegation_id: str = "item_1") -> dict:
    return {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}],
        "conversation": conversation,
        "metadata": {"live_delegation_id": delegation_id},
        "stream": True,
        "background": True,
        "store": False,
    }


class ResponsesClientContract(ContractCase):
    """What every ``ResponsesClient`` guarantees the gateway.

    A subclass is an ``IsolatedAsyncioTestCase`` that implements ``async make_client()``: a client of a core whose
    backend answers every request with "Done.".
    """

    async def make_client(self) -> Any:
        raise NotImplementedError

    async def a_conversation(self) -> tuple[Any, str]:
        client = await self.make_client()
        conversation = await client.create_conversation({"live_session_id": "live_test"})
        self.addAsyncCleanup(client.delete_conversation, conversation["id"])
        return client, conversation["id"]

    async def test_a_conversation_opens_with_an_id(self):
        _, conversation = await self.a_conversation()
        self.assertTrue(conversation)

    async def test_a_malformed_request_is_refused_before_any_event(self):
        client, conversation = await self.a_conversation()
        with self.assertRaises(ValueError):
            await client.create_response({**a_request(conversation), "input": "not a list of items"})

    async def test_each_request_reads_one_stream_that_ends_after_its_terminal(self):
        client, conversation = await self.a_conversation()
        events = await collect(await client.create_response(a_request(conversation)))
        terminals = [event for event in events if event["type"] in TERMINALS]
        self.assertEqual(len(terminals), 1)
        self.assertIs(events[-1], terminals[0])

    async def test_a_response_opens_queued_at_acceptance(self):
        client, conversation = await self.a_conversation()
        events = await collect(await client.create_response(a_request(conversation)))
        self.assertEqual([event["type"] for event in events[:2]], ["response.created", "response.queued"])
        self.assertEqual(events[0]["response"]["status"], "queued")

    async def test_input_tokens_validates_a_request(self):
        client, conversation = await self.a_conversation()
        counted = await client.input_tokens(a_request(conversation))
        self.assertEqual(counted["object"], "response.input_tokens")
        self.assertGreater(counted["input_tokens"], 0)
