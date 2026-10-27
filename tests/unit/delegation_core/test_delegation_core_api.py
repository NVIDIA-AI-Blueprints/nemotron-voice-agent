# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""The delegation core's Responses endpoints end to end, on a core built from the shipped ``core.yaml``."""

import asyncio
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from delegation_core.api.local import LocalResponsesClient
from delegation_core.api.service import NotFound, ResponsesService
from delegation_core.app import build_core
from delegation_core.backend.adapters import ADAPTERS
from delegation_core.backend.llm import LlmSettings, LlmSource
from delegation_core.config import CoreSettings, ModelEndpoint
from delegation_core.im.input import BackendTask
from delegation_core.im.stream import ResponseStream
from tests.unit.contracts.backend import ScriptedBackend, text_round
from tests.unit.core_helpers import answer_text

CORE_YAML = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/core.yaml"
ENDPOINT = ModelEndpoint(model="org/test-model", base_url="http://localhost:1/v1")
DEFAULTS = BackendTask("org/test-model", "Default.", [])
TERMINALS = {"response.completed", "response.failed", "response.incomplete"}


def user(text):
    """A user message's input items."""
    return [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}]


class Provider(ScriptedBackend):
    """A session's scripted model provider: it plays its rounds, and notes when its session releases it."""

    def __init__(self, *rounds):
        super().__init__(list(rounds), ENDPOINT)
        self.closes = 0

    async def close(self):
        self.closes += 1


def scripted_llm(providers: list[Provider]):
    """The adapter table, its backend LLM making each IM session's provider from ``providers``, in turn."""

    class ScriptedLlm(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            return cls(
                LlmSettings.parse(block), endpoints=endpoints, tools=tools, make_provider=lambda _: providers.pop(0)
            )

    return {**ADAPTERS, "llm": lambda: ScriptedLlm}  # the shipped file has blocks for the other adapters too


class CoreCase(unittest.IsolatedAsyncioTestCase):
    def core(self, *providers: Provider, **delegation):
        """A core from the shipped file, its conversations on ``providers``, reached through its in-process client."""
        settings = CoreSettings.load(CORE_YAML)
        settings = replace(settings, delegation=replace(settings.delegation, close_grace_seconds=0.01, **delegation))
        core = build_core(
            settings, endpoints=lambda model: ENDPOINT, defaults=DEFAULTS, adapters=scripted_llm(list(providers))
        )
        self.addAsyncCleanup(core.aclose)
        self.service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
        self.client = LocalResponsesClient(self.service)
        return self.client

    async def open(self, metadata=None):
        """Open a conversation; it is deleted when the test ends, unless the test deleted it."""
        conversation = await self.client.create_conversation(metadata or {"live_session_id": "live_1"})

        async def delete():
            if conversation["id"] in self.service.sessions:
                await self.client.delete_conversation(conversation["id"])

        self.addAsyncCleanup(delete)
        return conversation

    @staticmethod
    def request(conversation, text="Hi.", **fields):
        metadata = {"live_session_id": "live_1", "live_delegation_id": fields.pop("delegation_id", "item_1")}
        return {"input": user(text), "conversation": conversation, "metadata": metadata, "stream": True, **fields}

    @staticmethod
    async def read(stream):
        async with asyncio.timeout(5):
            return [event async for event in stream]


# ------------------------------------------------------------------------- DELETE /v1/conversations/{id}
class DeleteTests(CoreCase):
    async def test_a_deleted_conversation_stops_its_running_work_and_releases_its_provider(self):
        gate = asyncio.Event()  # the round still runs when the conversation is deleted
        provider = Provider([gate, *text_round("Done.")])
        self.core(provider)
        conversation = (await self.open())["id"]
        stream = await self.client.create_response(self.request(conversation))
        im = self.service.sessions[conversation]
        async with asyncio.timeout(2):
            while not provider.payloads:
                await asyncio.sleep(0.005)
        deleted = await self.client.delete_conversation(conversation)
        self.assertEqual(deleted, {"id": conversation, "object": "conversation.deleted", "deleted": True})
        events = await self.read(stream)
        self.assertEqual((events[-1]["type"], events[-1]["response"]["status"]), ("response.incomplete", "cancelled"))
        self.assertEqual((im.ledger.session.state, im.ledger.works["W1"].state), ("ended", "cancelled"))
        self.assertEqual(provider.closes, 1)

    async def test_a_deleted_conversation_is_gone_from_every_endpoint(self):
        self.core(Provider())
        conversation = (await self.open())["id"]
        await self.client.delete_conversation(conversation)
        for name, call in (
            ("delete", self.client.delete_conversation(conversation)),
            ("items", self.client.list_conversation_items(conversation)),
            ("respond", self.client.create_response(self.request(conversation))),
        ):
            with self.subTest(endpoint=name), self.assertRaises(NotFound):
                await call


# ------------------------------------------------------------------------------------- POST /v1/responses
class ResponseTests(CoreCase):
    async def test_a_response_opens_queued_runs_and_ends_once_numbered_from_one(self):
        provider = Provider(text_round("Done."))
        self.core(provider)
        conversation = (await self.open())["id"]
        events = await self.read(await self.client.create_response(self.request(conversation)))
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[:3], ["response.created", "response.queued", "response.in_progress"])
        self.assertEqual([kind for kind in kinds if kind in TERMINALS], ["response.completed"])
        self.assertEqual(kinds[-1], "response.completed")
        self.assertEqual([event["sequence_number"] for event in events], list(range(1, len(events) + 1)))
        created, completed = events[0]["response"], events[-1]["response"]
        self.assertEqual((created["status"], created["id"]), ("queued", completed["id"]))
        self.assertEqual((completed["status"], answer_text(events[-1])), ("completed", "Done."))
        self.assertEqual(provider.payloads[0]["model"], DEFAULTS.model)  # a request that sets nothing: the defaults

    async def test_a_request_names_an_open_conversation_by_its_id_or_as_an_object(self):
        self.core(Provider(text_round("Done.")))
        conversation = (await self.open())["id"]
        with self.assertRaises(NotFound):
            await self.client.create_response(self.request("conv_unknown"))
        stream = await self.client.create_response(self.request({"id": conversation}))
        self.assertEqual(answer_text((await self.read(stream))[-1]), "Done.")

    async def test_a_request_is_admitted_before_its_stream_is_returned(self):
        gate = asyncio.Event()
        self.core(Provider([gate, *text_round("Done.")]))
        conversation = (await self.open())["id"]
        stream = await self.client.create_response(self.request(conversation))
        self.assertIn("item_1", self.service.sessions[conversation].ledger.delegations)
        with self.assertRaises(ValueError):
            await self.client.create_response({**self.request(conversation), "input": "not a list"})
        gate.set()
        self.assertEqual(answer_text((await self.read(stream))[-1]), "Done.")


# ------------------------------------------------------------------------ GET /v1/conversations/{id}/items
class ItemsTests(CoreCase):
    async def test_the_items_are_the_conversations_backend_history_as_one_list_page(self):
        self.core(Provider(text_round("We open at nine.")))
        conversation = (await self.open())["id"]
        await self.read(await self.client.create_response(self.request(conversation, "When do you open?")))
        page = await self.client.list_conversation_items(conversation)
        self.assertEqual(sorted(page), ["data", "first_id", "has_more", "last_id", "object"])
        self.assertEqual((page["object"], page["has_more"]), ("list", False))
        self.assertEqual(page["data"], self.service.sessions[conversation].history)
        self.assertEqual([item.get("role") for item in page["data"]], ["user", "assistant"])


# --------------------------------------------------------------------------------------- the in-process client
class InProcessClientTests(CoreCase):
    async def test_every_request_result_and_event_crosses_as_a_copy(self):
        self.core(Provider(text_round("Done.")))
        metadata = {"live_session_id": "live_1"}
        conversation = await self.client.create_conversation(metadata)
        self.addAsyncCleanup(self.client.delete_conversation, conversation["id"])
        conversation["metadata"]["live_session_id"] = "changed"
        metadata["live_session_id"] = "changed too"
        self.assertEqual(self.service.conversations[conversation["id"]]["metadata"], {"live_session_id": "live_1"})
        written = []
        put = ResponseStream.put

        def noted_put(stream, event):
            written.append(event)
            put(stream, event)

        request = self.request(conversation["id"], tools=({"type": "function", "name": "get_menu"},))
        with patch.object(ResponseStream, "put", noted_put):
            stream = await self.client.create_response(request)
            request["input"][0]["content"][0]["text"] = "changed after it was sent"
            events = await self.read(stream)
        im = self.service.sessions[conversation["id"]]
        self.assertEqual(im.ledger.delegations["item_1"].settings["tools"], [{"type": "function", "name": "get_menu"}])
        self.assertEqual(im.history[0], user("Hi.")[0])
        self.assertEqual(events, written)  # the same events, in order ...
        self.assertTrue(all(read is not original for read, original in zip(events, written, strict=True)))  # copied


# ---------------------------------------------------------------------------------- two conversations, one core
class TwoConversationsTests(CoreCase):
    async def test_two_conversations_have_their_own_im_session_adapter_provider_work_and_history(self):
        gate = asyncio.Event()
        first_provider, second_provider = Provider([gate, *text_round("First.")]), Provider(text_round("Second."))
        self.core(first_provider, second_provider)
        first, second = (await self.open())["id"], (await self.open({"live_session_id": "live_2"}))["id"]
        ims = [self.service.sessions[first], self.service.sessions[second]]
        self.assertIsNot(ims[0], ims[1])
        self.assertIsNot(ims[0].backend, ims[1].backend)
        held = await self.client.create_response(self.request(first, "One."))
        answered = await self.client.create_response(self.request(second, "Two."))  # not queued behind the first
        self.assertEqual(answer_text((await self.read(answered))[-1]), "Second.")
        self.assertEqual(ims[1].history[0], user("Two.")[0])
        await self.client.delete_conversation(second)  # one conversation ends; the other's work goes on
        self.assertEqual(list(self.service.sessions), [first])
        self.assertEqual((first_provider.closes, second_provider.closes), (0, 1))
        gate.set()
        self.assertEqual(answer_text((await self.read(held))[-1]), "First.")
        self.assertEqual(len(first_provider.payloads), 1)


if __name__ == "__main__":
    unittest.main()
