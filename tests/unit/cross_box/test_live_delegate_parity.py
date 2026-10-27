# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D105, D107

"""``POST /v1/live/delegate`` against recorded answers to ten requests.

The fixture holds, for each request, the HTTP status, the reply (response, item and call ids normalized), every request
that reached the scripted model client, and whether the client was closed; a difference is a change of the route's
behavior, which updates the fixture. The service catalog is pinned, so the answers do not depend on which model servers
this machine can reach. The recording's default voice note is v1, so a core profile over the shipped file sets it.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletionChunk

import server
from delegation_core.backend.providers import base as providers
from delegation_core.config import CoreSettings
from delegation_core.im.input import compose
from examples.frontend_backend_live import factory
from tests.unit.frontend_backend_live.root.root_helpers import clear_root_caches
from utils import clear_service_context, set_active_services

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "live_delegate_da43c83.json"
CORE = Path(factory.__file__).with_name("core.yaml")


def chunk(content=None, tool_calls=None, finish=None):
    delta = {"role": "assistant"}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


class Stream:
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.chunks:
            raise StopAsyncIteration
        return self.chunks.pop(0)

    async def close(self):
        pass


class FakeChat:
    """The model client, scripted as when the fixture was recorded: it notes each request and plays one step."""

    def __init__(self, *script):
        self.script, self.requests, self.closed = list(script), [], False
        self.chat = self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return Stream(step)

    async def close(self):
        self.closed = True


CALL = [
    chunk(
        tool_calls=[
            {"index": 0, "id": "call_1", "type": "function", "function": {"name": "get_menu", "arguments": '{"cate'}}
        ]
    ),
    chunk(
        tool_calls=[{"index": 0, "type": "function", "function": {"arguments": 'gory":"tea"}'}}], finish="tool_calls"
    ),
]
ANSWER = [chunk("We have Earl Grey and a matcha latte."), chunk(finish="stop")]
CUT = [chunk("We have Earl"), chunk(finish="length")]
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "What teas do you have?"}]}
TOOLS = [
    {"type": "function", "name": "get_menu", "description": "Menu", "parameters": {"type": "object", "properties": {}}},
    {"type": "web_search"},
]
CALL_ITEM = {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": '{"category":"tea"}'}
OUTPUT_ITEM = {"type": "function_call_output", "call_id": "call_1", "output": '{"items": ["Earl Grey"]}'}


class DelegateParityTests(unittest.TestCase):
    """The route answers as recorded, request for request."""

    @classmethod
    def setUpClass(cls):
        # The recording's default voice note, v1: a core profile over the shipped file (v2) that sets only the note.
        folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(folder.cleanup)
        cls.core_v1 = Path(folder.name) / "core.v1.yaml"
        cls.core_v1.write_text(f"extends: {CORE}\ndelegation:\n  voice_note: v1\n", encoding="utf-8")

    def setUp(self):
        # The service catalog as when the fixture was recorded: the self-hosted "server" section and no NVIDIA Cloud
        # key, named outright so that no reachability probe, cached by an earlier test, picks another section. The test
        # client sends Host: testserver, a name the console route answers to with no token only when it is allowed.
        live = {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "LIVE_ALLOWED_HOSTS": "testserver"}
        env = patch.dict(os.environ, {**live, "SERVICE_RECIPE": "server", "NVIDIA_API_KEY": ""})
        env.start()
        self.addCleanup(env.stop)
        # The bundled files and the registry's own hooks.
        for name in (
            "SERVICES_PATH",
            "LIVE_CONFIG_PATH",
            "CORE_CONFIG_PATH",
            "LIVE_ADAPTER_FACTORY",
            "LIVE_ENGINE_FACTORY",
        ):
            os.environ.pop(name, None)
        # create_app binds the selected example's services process-wide; leave no binding behind.
        self.addCleanup(clear_service_context)
        self.addCleanup(set_active_services, None)

    def play(self, body, *script, core=None):
        chat = FakeChat(*script)
        clear_root_caches(self)  # each play builds its own core, whose provider makes its client from the patch
        with (
            patch.dict(os.environ, {"CORE_CONFIG_PATH": str(core or self.core_v1)}),
            patch.object(providers, "AsyncOpenAI", return_value=chat),
            TestClient(server.create_app()) as client,
        ):
            response = client.post("/v1/live/delegate", json=body)
        reply = response.json()
        if isinstance(reply.get("response"), dict) and reply["response"].get("id"):
            reply["response"]["id"] = "<resp id>"
        for item in (reply.get("response") or {}).get("output", []):
            if item.get("id"):
                item["id"] = "<item id>"
            if item.get("call_id"):
                item["call_id"] = "<call id>"  # the core gives each call its own id
        return {"status": response.status_code, "reply": reply, "model_requests": chat.requests, "closed": chat.closed}

    def played(self):
        """Every recorded request played on the route, by case: what came back and what reached the model."""
        results = {}
        for version, core in (("v1", self.core_v1), ("v2", CORE)):
            results[f"{version} round 1, defaults"] = self.play({"input": [USER], "tools": TOOLS}, CALL, core=core)
            results[f"{version} round 2, whole history"] = self.play(
                {"input": [USER, CALL_ITEM, OUTPUT_ITEM], "tools": TOOLS}, ANSWER, core=core
            )
            results[f"{version} client model and instructions"] = self.play(
                {"input": [USER], "model": "org/other", "instructions": "Be terse."}, ANSWER, core=core
            )
        results["a failed model call"] = self.play({"input": [USER]}, RuntimeError("model down"))
        results["a round cut off by the token limit"] = self.play({"input": [USER]}, CUT)
        results["no input"] = self.play({}, ANSWER)
        results["a tool with no name"] = self.play({"input": [USER], "tools": [{"type": "function"}]}, ANSWER)
        return json.loads(json.dumps(results, sort_keys=True, default=str))

    def test_every_recorded_request_gets_nikhils_answer(self):
        recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
        played = self.played()
        self.assertEqual(sorted(played), sorted(recorded))
        for case in recorded:
            with self.subTest(case):
                self.assertEqual(played[case], recorded[case])

    def test_a_listed_client_model_gets_the_clients_instructions_and_then_the_voice_note(self):
        body = {"input": [USER], "model": "nvidia/nemotron-3-super-120b-a12b", "instructions": "Be terse."}
        played = self.play(body, ANSWER, core=CORE)
        [request] = played["model_requests"]
        note = CoreSettings.load(CORE).texts.voice_notes["v2"]
        self.assertEqual((played["status"], played["closed"], request["model"]), (200, True, body["model"]))
        self.assertEqual(request["messages"][0], {"role": "system", "content": compose("Be terse.", note)})
        self.assertEqual(played["reply"]["sent_request"], {"model": body["model"], "tools": []})


if __name__ == "__main__":
    unittest.main()
