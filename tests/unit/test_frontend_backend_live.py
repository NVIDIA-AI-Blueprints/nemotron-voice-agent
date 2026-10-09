# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D105, D107

import asyncio
import hashlib
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from openai import APITimeoutError, BadRequestError
from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import run_test

from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.commentary.guards import (
    SAFE_LEAD_IN,
    asserts_completion,
    plain_speech,
    safe_lead_in,
)
from examples.frontend_backend_live.cascade.conversation import ContextConversation, split_messages
from examples.frontend_backend_live.cascade.processor import LiveTalkerProcessor
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.talker.router import TurnRouter
from examples.frontend_backend_live.cascade.talker.talker import never_silent
from examples.frontend_backend_live.common.history import BACKEND_RESULT_PREFIX
from examples.frontend_backend_live.config_manager.loader import ConfigError, load_live_config, parse_config
from examples.frontend_backend_live.config_manager.schema import (
    DelegationConfig,
    LiveConfig,
    ModelEndpoint,
    ReliabilityConfig,
    RoleConfig,
)
from examples.frontend_backend_live.delegation.tasks import backend_task_for
from examples.frontend_backend_live.delegation.thinker import Thinker
from examples.frontend_backend_live.delegation.worker import APOLOGY, BackendTask, DelegationWorker
from examples.frontend_backend_live.models import BACKENDS, FRONTENDS, Backend, Frontend
from examples.frontend_backend_live.models.chat_completions.backend import ChatCompletionsBackend
from examples.frontend_backend_live.models.chat_completions.frontend import (
    ChatCompletionsFrontend,
    chat_messages,
    extract_json,
)
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.prompts import get_prompt_set, v1, v2
from examples.frontend_backend_live.prompts.catalog import catalog_prompt, role_instructions
from examples.frontend_backend_live.tool_calling.cafe.schemas import CAFE_TOOLS
from examples.frontend_backend_live.tool_calling.cafe.tools import CafeTools
from live.protocol import SessionConfig
from tests.unit.test_frontend_backend_live_translation import chunk, tool_fragment

ENDPOINT = ModelEndpoint(model="test/model", base_url="http://localhost:1/v1")
FAST = ReliabilityConfig(provider_retries=1, talker_attempt_timeout_seconds=1.0, api_timeout_seconds=2.0)


def _config(**changes) -> LiveConfig:
    base = LiveConfig(
        frontend=RoleConfig("scripted", "llm"), backend=RoleConfig("scripted", "thinker-llm"), reliability=FAST
    )
    return replace(base, **changes)


# ----------------------------------------------------------------------------- test doubles
class ScriptedFrontend(Frontend):
    """A talker model that plays a script: each call pops the next decision, or raises it."""

    def __init__(self, script, endpoint=ENDPOINT, reliability=FAST):
        super().__init__(endpoint, reliability)
        self.script = list(script)
        self.calls: list[dict] = []

    async def complete_once(self, instructions, history, actions, max_output_tokens):
        self.calls.append(
            {"instructions": instructions, "history": json.loads(json.dumps(history)), "actions": actions}
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def _text_round(text: str, response_id: str = "resp") -> list[dict]:
    return [
        {"type": "response.output_item.done", "item": {"type": "message", "role": "assistant", "content": text}},
        {"type": "response.output_text.delta", "delta": text},
        {"type": "response.completed", "response": {"id": response_id}},
    ]


def _call_round(name: str, arguments: dict, call_id: str = "call_1") -> list[dict]:
    item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}
    return [
        {"type": "response.output_item.done", "item": item},
        {"type": "response.completed", "response": {"id": "resp_call"}},
    ]


class ScriptedBackend(Backend):
    """A thinker model that plays rounds of Responses events. An Exception entry raises mid-round."""

    stateful = False

    def __init__(self, rounds, endpoint=ENDPOINT, reliability=FAST):
        super().__init__(endpoint, reliability)
        self.rounds = list(rounds)
        self.payloads: list[dict] = []
        self.gate: asyncio.Event | None = None

    async def stream(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        if self.gate is not None:
            await self.gate.wait()
        step = self.rounds.pop(0)
        for event in step:
            if isinstance(event, Exception):
                raise event
            yield event


class ListConversation:
    def __init__(self, messages):
        self._messages = list(messages)

    def messages(self):
        return list(self._messages)

    def add(self, message):
        self._messages.append(message)

    def trim(self, limit):
        pass


class RecordingTools:
    def __init__(self, result='{"ok": true}'):
        self.result, self.calls = result, []

    async def execute(self, call_id, name, arguments):
        self.calls.append((call_id, name, arguments))
        return self.result


def _agent(frontend, backend, prompt_version="v1"):
    from examples.frontend_backend_live.cascade.agent import LiveAgent
    from examples.frontend_backend_live.cascade.commentary.writer import CommentaryWriter
    from examples.frontend_backend_live.cascade.talker.talker import Talker

    prompts = get_prompt_set(prompt_version)
    return LiveAgent(
        frontend,
        backend,
        prompts,
        Talker(frontend, prompts),
        CommentaryWriter(frontend, prompts),
        Thinker(backend, prompts),
    )


def _messages(*turns):
    out = [{"role": "system", "content": "You are Remy."}]
    out += [{"role": role, "content": text} for role, text in turns]
    return out


async def _settle(session, spoken, count, timeout=2.0):
    """Wait until ``count`` utterances were spoken."""
    async with asyncio.timeout(timeout):
        while len(spoken) < count:
            await asyncio.sleep(0.005)


class _SessionCase(unittest.IsolatedAsyncioTestCase):
    def make(self, frontend_script, backend_rounds, messages, *, config=None, tools=None):
        self.frontend = ScriptedFrontend(frontend_script)
        self.backend = ScriptedBackend(backend_rounds)
        self.conversation = ListConversation(messages)
        self.spoken: list[str] = []
        self.events: list[tuple[str, dict]] = []

        async def say(text):
            self.spoken.append(text)

        self.session = CascadeSession(
            _agent(self.frontend, self.backend),
            config or _config(),
            self.conversation,
            backend_task=BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            tools=tools or RecordingTools(),
            say=say,
            observe=lambda kind, **fields: self.events.append((kind, fields)),
        )
        self.session.start()
        self.addAsyncCleanup(self.session.worker.close)

    def kinds(self):
        return [kind for kind, _ in self.events]


# ------------------------------------------------------------------------------ configuration
class ConfigTests(unittest.TestCase):
    def test_bundled_config_loads_and_names_registered_providers(self):
        config = load_live_config()
        self.assertEqual((config.frontend.slot, config.backend.slot), ("llm", "thinker-llm"))
        self.assertIn(config.frontend.provider, FRONTENDS.names())
        self.assertIn(config.backend.provider, BACKENDS.names())
        self.assertEqual(config.prompt_version, "v1")

    def test_each_role_names_its_static_prompt_and_defaults_to_the_catalog_entry(self):
        config = load_live_config()
        self.assertEqual((config.frontend.prompt, config.backend.prompt), ("talker", "thinker"))
        for role in (config.frontend, config.backend):
            self.assertTrue(catalog_prompt(role.prompt))
        bare = {"frontend": {"provider": "p", "slot": "llm"}, "backend": {"provider": "p", "slot": "t"}}
        self.assertEqual(parse_config(bare).backend.prompt, "thinker")
        named = {**bare, "backend": {"provider": "p", "slot": "t", "prompt": "custom"}}
        self.assertEqual(parse_config(named).backend.prompt, "custom")
        with self.assertRaisesRegex(ConfigError, "prompts.yaml"):
            parse_config({**bare, "frontend": {"provider": "p", "slot": "llm", "prompt": ""}})

    def test_a_server_prompt_file_does_not_replace_the_examples_own_prompts(self):
        with TemporaryDirectory() as tmp:
            other = Path(tmp) / "other.yaml"
            other.write_text("custom:\n  content: not the talker prompt\n")
            with patch.dict(os.environ, {"PROMPT_FILE_PATH": str(other)}):
                self.assertTrue(catalog_prompt("talker"))
                self.assertTrue(role_instructions("thinker").startswith(catalog_prompt("thinker")))

    def test_a_zero_is_only_allowed_for_the_retry_count(self):
        bare = {"frontend": {"provider": "p", "slot": "llm"}, "backend": {"provider": "p", "slot": "t"}}
        self.assertEqual(parse_config({**bare, "reliability": {"provider_retries": 0}}).reliability.provider_retries, 0)
        for field in (
            "talker_attempt_timeout_seconds",
            "api_timeout_seconds",
            "tool_timeout_seconds",
            "max_tool_rounds",
            "max_pending_delegations",
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ConfigError, "positive"):
                parse_config({**bare, "reliability": {field: 0}})

    def test_role_instructions_append_the_channel_context(self):
        text = role_instructions("thinker")
        self.assertTrue(text.startswith(catalog_prompt("thinker")))
        self.assertTrue(text.endswith(catalog_prompt("channel")))
        with self.assertRaises(KeyError):
            catalog_prompt("no-such-prompt")

    def test_unknown_keys_and_bad_values_are_errors(self):
        good = {"frontend": {"provider": "p", "slot": "llm"}, "backend": {"provider": "p", "slot": "t"}}
        for bad, message in (
            ({**good, "oops": 1}, "unknown keys"),
            ({**good, "prompt_version": "v9"}, "prompt_version"),
            ({**good, "reliability": {"provider_retries": "two"}}, "non-negative"),
            ({**good, "reliability": {"nope": 1}}, "unknown keys"),
            ({**good, "guards": {"repeat_guard": "yes"}}, "true or false"),
            ({**good, "guards": {"continuers": "okay"}}, "list of strings"),
            ({**good, "frontend": {"provider": "p"}}, "provider"),
            ({**good, "chat_history_recent_turns": 0}, "positive"),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(ConfigError, message):
                parse_config(bad)

    def test_values_override_defaults(self):
        config = parse_config(
            {
                "frontend": {"provider": "chat-completions", "slot": "llm"},
                "backend": {"provider": "openai-responses", "slot": "thinker-llm"},
                "prompt_version": "v2",
                "reliability": {"provider_retries": 0, "talker_attempt_timeout_seconds": 3},
                "guards": {"turn_router_shadow": True, "continuers": ["okay"]},
            }
        )
        self.assertEqual(config.reliability.provider_retries, 0)
        self.assertEqual(config.reliability.talker_attempt_timeout_seconds, 3.0)
        self.assertTrue(config.guards.turn_router_shadow)
        self.assertEqual(config.guards.continuers, ("okay",))
        self.assertEqual(config.backend.provider, "openai-responses")

    def test_unreadable_file_is_a_config_error(self):
        with TemporaryDirectory() as tmp, self.assertRaises(ConfigError):
            load_live_config(Path(tmp) / "missing.yaml")


# ---------------------------------------------------------------------------- pluggable models
class RegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_shipped_providers_are_registered_for_both_roles(self):
        self.assertEqual(FRONTENDS.names(), ["chat-completions", "openai-responses"])
        self.assertEqual(BACKENDS.names(), ["chat-completions", "openai-responses"])

    async def test_a_new_frontend_is_one_subclass_and_a_name(self):
        @FRONTENDS.register("scripted-test")
        class Echo(Frontend):
            async def complete_once(self, instructions, history, actions, max_output_tokens):
                return TalkerDecision("speak", f"heard {len(history)}")

        try:
            config = _config(frontend=RoleConfig("scripted-test", "llm"), backend=RoleConfig("chat-completions", "t"))
            agent = build_agent(config, ENDPOINT, ENDPOINT)
            decision = await agent.talker.decide("Be kind.", [{"role": "user", "content": "hi"}])
            self.assertEqual((decision.action, decision.speech), ("speak", "heard 1"))
            await agent.close()
        finally:
            FRONTENDS._items.pop("scripted-test")

    def test_unknown_provider_names_the_choices(self):
        config = _config(frontend=RoleConfig("nope", "llm"), backend=RoleConfig("chat-completions", "t"))
        with self.assertRaisesRegex(ValueError, "registered: chat-completions, openai-responses"):
            build_agent(config, ENDPOINT, ENDPOINT)

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "already registered"):
            FRONTENDS.register("chat-completions")(ChatCompletionsFrontend)


# ------------------------------------------------------------------------------------ prompts
class PromptTests(unittest.TestCase):
    # sha256 of each prompt text: an unintended prompt change fails here. Update on purpose when a prompt changes.
    DIGESTS = {
        (v1, "ROUTING_INSTRUCTIONS"): "40365a9f3d1220147e84a8b2c989095f7aad440b15832b14dd20c91f92a8b5ea",
        (v1, "COMMENTARY_INSTRUCTIONS"): "e50f058f61635840fc3f6a54b46044c0669c5ec78f99373968d632b382d9e055",
        (v1, "ANSWER_INSTRUCTIONS"): "cf6778d693ac771d41001d4bbe7eb3aedb88970195d0d36203294f0c339a73b2",
        (v2, "ROUTING_INSTRUCTIONS"): "6547b7fc7b545b164ff788f5865b62feb12e8fed27dc4cac186f14854d28025d",
        (v2, "COMMENTARY_INSTRUCTIONS"): "dd3ce14d515204c756dce38a0a14c9564057e4a69ddf2e6ecb792e1d28ce5c68",
        (v2, "ANSWER_INSTRUCTIONS"): "91a3f2b0efe7de3adf2fd64d394bc3c59dee09f129bcaa09a0370d8759383666",
        (v2, "BACKEND_VOICE_NOTES"): "c9c4771eae16a4da1421d5d149efab26ffd260fa80f2d36c327fdf0f6101f2f1",
    }

    def test_prompt_texts_are_unchanged(self):
        for (module, name), digest in self.DIGESTS.items():
            with self.subTest(prompt=f"{module.__name__.rsplit('.', 1)[-1]}.{name}"):
                self.assertEqual(hashlib.sha256(getattr(module, name).encode()).hexdigest(), digest)

    def test_v1_gives_the_backend_no_voice_note_and_v2_does(self):
        self.assertEqual(get_prompt_set("v1").backend_notes, "")
        self.assertIn("Voice channel", get_prompt_set("v2").backend_notes)
        with self.assertRaises(ValueError):
            get_prompt_set("v3")

    def test_routing_has_no_listen_and_the_schema_only_allows_speak_or_delegate(self):
        self.assertNotIn("listen", v1.ROUTING_INSTRUCTIONS.lower())
        self.assertEqual(decision_schema()["properties"]["action"]["enum"], ["speak", "delegate"])
        self.assertEqual(decision_schema(("speak", "listen"))["properties"]["action"]["enum"], ["speak", "listen"])

    def test_the_example_prompt_catalog_has_talker_thinker_and_channel(self):
        import yaml

        catalog = yaml.safe_load(
            (Path(__file__).parents[2] / "src/examples/frontend_backend_live/prompts.yaml").read_text()
        )
        self.assertTrue(catalog["talker"]["default"])
        for key in ("talker", "thinker", "channel"):
            self.assertTrue(catalog[key]["content"].strip(), key)
        self.assertNotIn("web_search", catalog["thinker"]["content"])


class ThinkerNoteTests(unittest.IsolatedAsyncioTestCase):
    async def test_thinker_adds_the_voice_note_only_under_v2(self):
        backend = ScriptedBackend([_text_round("ok"), _text_round("ok")])
        for version, expected in (("v1", "Do the task."), ("v2", None)):
            thinker = Thinker(backend, get_prompt_set(version))
            [e async for e in thinker.stream({"instructions": "Do the task.", "input": []})]
            sent = backend.payloads[-1]["instructions"]
            if expected:
                self.assertEqual(sent, expected)
            else:
                self.assertTrue(sent.startswith("Do the task.\n"))
                self.assertIn("Voice channel", sent)


# ------------------------------------------------------------------------------------ guards
class GuardTests(unittest.TestCase):
    def test_never_silent_delegates_listen_and_empty_speak(self):
        spoken = TalkerDecision("speak", "Hello.")
        self.assertIs(never_silent(spoken), spoken)
        for decision, reason in (
            (TalkerDecision("listen", ""), "listen_removed"),
            (TalkerDecision("speak", "  "), "empty_speak"),
        ):
            floored = never_silent(decision)
            self.assertEqual(
                (floored.action, floored.speech, floored.diagnostics["recovery"]), ("delegate", "", reason)
            )

    def test_lead_in_never_claims_an_outcome_or_asks_a_question(self):
        self.assertEqual(safe_lead_in("Your order is confirmed."), SAFE_LEAD_IN)
        self.assertTrue(asserts_completion("It is all set"))
        self.assertEqual(safe_lead_in("Let me check. Which size?"), "Let me check.")
        self.assertEqual(safe_lead_in("Which size?"), SAFE_LEAD_IN)
        self.assertEqual(safe_lead_in("Let me look that up."), "Let me look that up.")
        self.assertEqual(plain_speech("- one\n- two"), "one. two.")

    def test_router_drops_backchannels_but_keeps_answers(self):
        router = TurnRouter()
        self.assertFalse(router.verdict("Okay.", "Your latte is ready.").keep)
        self.assertTrue(router.verdict("Yes.", "Please confirm the order.").keep)
        self.assertTrue(router.verdict("Okay.", None).keep)
        self.assertTrue(router.verdict("Two lattes", "Anything else?").keep)


# --------------------------------------------------------------------- chat-completions models
class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.chunks:
            yield item

    async def close(self):
        self.closed = True


class FakeChat:
    """Scripted ``chat.completions.create``: each entry is a chunk list or an exception."""

    def __init__(self, *script):
        self.script, self.requests, self.closed = list(script), [], False
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return FakeStream(step)

    async def close(self):
        self.closed = True


def _decision_chunks(text, finish="stop"):
    half = len(text) // 2
    return [chunk(text[:half]), chunk(text[half:]), chunk(finish=finish)]


def _bad_request(message):
    response = httpx.Response(400, request=httpx.Request("POST", "https://hub.example/v1"))
    return BadRequestError(message, response=response, body=None)


def _timeout():
    return APITimeoutError(request=httpx.Request("POST", "https://hub.example/v1"))


class ChatFrontendTests(unittest.IsolatedAsyncioTestCase):
    def frontend(self, client, **endpoint):
        return ChatCompletionsFrontend(replace(ENDPOINT, **endpoint), FAST, client)

    async def test_request_uses_sampling_defaults_and_the_endpoint_extra_params(self):
        client = FakeChat(_decision_chunks('{"action":"speak","speech":"Hello."}'))
        decision = await self.frontend(client).complete("You are Remy.", [{"role": "user", "content": "hi"}])
        self.assertEqual((decision.action, decision.speech), ("speak", "Hello."))
        request = client.requests[0]
        self.assertEqual((request["model"], request["temperature"], request["max_tokens"]), ("test/model", 0.0, 512))
        self.assertTrue(request["stream"])
        schema = request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["action"]["enum"], ["speak", "delegate"])

    async def test_extra_params_override_defaults_without_duplicate_keyword_errors(self):
        client = FakeChat(_decision_chunks('{"action":"delegate","speech":""}'))
        extra = {"temperature": 0.2, "max_tokens": 64, "extra_body": {"repetition_penalty": 1.05}}
        await self.frontend(client, extra_params=extra).complete("x", [{"role": "user", "content": "hi"}])
        request = client.requests[0]
        self.assertEqual((request["temperature"], request["max_tokens"]), (0.2, 64))
        self.assertEqual(request["extra_body"], {"repetition_penalty": 1.05})

    async def test_response_format_steps_down_after_a_rejection_and_is_remembered(self):
        client = FakeChat(
            _bad_request("response_format json_schema is not supported"),
            _decision_chunks('{"action":"speak","speech":"Hi."}'),
            _decision_chunks('{"action":"speak","speech":"Again."}'),
        )
        frontend = self.frontend(client)
        history = [{"role": "user", "content": "hi"}]
        self.assertEqual((await frontend.complete("x", history)).speech, "Hi.")
        self.assertEqual(client.requests[1]["response_format"], {"type": "json_object"})
        await frontend.complete("x", history)
        self.assertEqual(client.requests[2]["response_format"], {"type": "json_object"})

    async def test_an_unrelated_bad_request_is_not_retried_as_a_format_problem(self):
        frontend = self.frontend(FakeChat(_bad_request("model not found")))
        with self.assertRaises(BadRequestError):
            await frontend.complete("x", [{"role": "user", "content": "hi"}])

    async def test_a_transient_failure_is_retried_and_then_succeeds(self):
        client = FakeChat(_timeout(), _decision_chunks('{"action":"speak","speech":"Back."}'))
        decision = await self.frontend(client).complete("x", [{"role": "user", "content": "hi"}])
        self.assertEqual((decision.speech, decision.diagnostics["attempt"]), ("Back.", 2))

    async def test_unusable_output_delegates_the_original_request(self):
        for text, reason in (("not json", "invalid_json"), ('{"action":"dance","speech":"x"}', "invalid_decision")):
            client = FakeChat(_decision_chunks(text))
            decision = await self.frontend(client).complete("x", [{"role": "user", "content": "hi"}])
            self.assertEqual(
                (decision.action, decision.speech, decision.diagnostics["recovery"]), ("delegate", "", reason)
            )

    async def test_truncated_output_is_incomplete_not_parsed(self):
        client = FakeChat(_decision_chunks('{"action":"speak","speech":"Hel', finish="length"))
        decision = await self.frontend(client).complete("x", [{"role": "user", "content": "hi"}])
        self.assertEqual(decision.diagnostics["recovery"], "incomplete_output")

    def test_messages_fold_developer_context_and_merge_roles(self):
        messages = chat_messages(
            "Be Remy.",
            [
                {"role": "developer", "content": "Kiosk."},
                {"role": "user", "content": "hi"},
                {"role": "user", "content": "menu?"},
                {"role": "assistant", "content": "Sure."},
            ],
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertIn("Kiosk.", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "hi\nmenu?")

    def test_json_extraction_tolerates_fences_and_leaked_reasoning(self):
        self.assertEqual(
            extract_json('```json\n{"action":"speak","speech":"Hi"}\n```'), '{"action":"speak","speech":"Hi"}'
        )
        self.assertEqual(
            extract_json('<think>hmm</think>{"action":"delegate","speech":""}'), '{"action":"delegate","speech":""}'
        )


class ChatBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_sends_tools_history_and_translates_a_function_call(self):
        client = FakeChat(
            [
                chunk(tool_calls=[tool_fragment(0, "call_1", "get_menu", '{"category":')]),
                chunk(tool_calls=[tool_fragment(0, arguments='"tea"}')], finish="tool_calls"),
            ]
        )
        backend = ChatCompletionsBackend(ENDPOINT, FAST, client)
        payload = {
            "model": "test/model",
            "instructions": "Use tools.",
            "tools": CAFE_TOOLS + [{"type": "web_search"}],
            "tool_choice": "auto",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "teas?"}]}],
        }
        events = [e async for e in backend.stream(payload)]
        request = client.requests[0]
        self.assertEqual((request["temperature"], request["max_tokens"], request["tool_choice"]), (0.7, 1024, "auto"))
        self.assertEqual([t["function"]["name"] for t in request["tools"]], ["get_menu", "manage_cart", "place_order"])
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])
        calls = [
            e["item"]
            for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ]
        self.assertEqual((calls[0]["name"], json.loads(calls[0]["arguments"])), ("get_menu", {"category": "tea"}))
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertFalse(backend.stateful)

    async def test_failure_before_the_first_chunk_is_retried_but_a_later_one_is_not(self):
        backend = ChatCompletionsBackend(ENDPOINT, FAST, FakeChat(_timeout(), [chunk("Hello", finish="stop")]))
        events = [e async for e in backend.stream({"input": []})]
        self.assertIn("response.output_text.delta", [e["type"] for e in events])

        failing = ChatCompletionsBackend(ENDPOINT, replace(FAST, provider_retries=0), FakeChat(_timeout()))
        with self.assertRaises(APITimeoutError):
            [e async for e in failing.stream({"input": []})]


# ----------------------------------------------------------------------- the session's routing
class RoutingTests(_SessionCase):
    async def test_a_spoken_reply_is_said_and_nothing_is_delegated(self):
        self.make([TalkerDecision("speak", "We open at seven.")], [], _messages(("user", "When do you open?")))
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["We open at seven."])
        self.assertEqual(self.backend.payloads, [])
        self.assertEqual(self.frontend.calls[0]["history"], [{"role": "user", "content": "When do you open?"}])
        self.assertIn("You are Remy.", self.frontend.calls[0]["instructions"])
        self.assertIn(v1.ROUTING_INSTRUCTIONS, self.frontend.calls[0]["instructions"])

    async def test_delegation_speaks_a_lead_in_then_the_phrased_answer(self):
        self.make(
            [TalkerDecision("delegate", "Let me look that up."), TalkerDecision("speak", "A latte is four fifty.")],
            [_text_round("Latte: $4.50.")],
            _messages(("user", "How much is a latte?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(self.spoken, ["Let me look that up.", "A latte is four fifty."])
        # the answer is phrased as an answer, and told what the caller already heard
        phrasing = self.frontend.calls[1]
        self.assertEqual(phrasing["actions"], ("speak",))
        self.assertIn(v1.ANSWER_INSTRUCTIONS, phrasing["instructions"])
        update = phrasing["history"][-1]["content"]
        self.assertIn("Latte: $4.50.", update)
        self.assertIn('you already told the caller: "Let me look that up."', update)

    async def test_the_result_is_in_the_history_in_order_and_phrasing_sees_it_once(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Done.")],
            [_text_round("Order BB-42.")],
            _messages(("user", "Order a latte.")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        stored = [m["content"] for m in self.conversation.messages()]
        self.assertEqual(stored[-1], BACKEND_RESULT_PREFIX + "Order BB-42.")
        sent = [m["content"] for m in self.frontend.calls[1]["history"]]
        self.assertEqual(sum("Order BB-42." in item for item in sent), 1)
        self.assertNotIn(BACKEND_RESULT_PREFIX + "Order BB-42.", sent)

    async def test_the_delegate_gets_only_the_turns_since_its_last_delegation(self):
        self.make(
            [
                TalkerDecision("speak", "Sure, what size?"),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "Added."),
            ],
            [_text_round("Added a large latte.")],
            _messages(("user", "I want a latte.")),
        )
        await self.session.on_turn()
        self.conversation.add({"role": "assistant", "content": "Sure, what size?"})
        self.conversation.add({"role": "user", "content": "Large, please."})
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 2)
        text = self.backend.payloads[0]["input"][0]["content"][0]["text"]
        self.assertIn("I want a latte.", text)
        self.assertIn("Sure, what size?", text)
        self.assertTrue(text.endswith("Current request: Large, please."))
        self.assertNotIn(BACKEND_RESULT_PREFIX, text)

    async def test_a_lead_in_that_claims_an_outcome_or_asks_a_question_is_cut(self):
        for lead_in, expected in (("Your order is confirmed!", SAFE_LEAD_IN), ("Checking. Which size?", "Checking.")):
            self.make(
                [TalkerDecision("delegate", lead_in), TalkerDecision("speak", "Ok.")],
                [_text_round("fine")],
                _messages(("user", "Order.")),
            )
            await self.session.on_turn()
            await _settle(self.session, self.spoken, 2)
            self.assertEqual(self.spoken[0], expected)
            self.assertIn("talker.lead_in_replaced", self.kinds())
            await self.session.worker.close()

    async def test_a_reply_that_repeats_an_earlier_one_becomes_a_delegation(self):
        stale = "I am still unable to locate your account with the information provided."
        self.make(
            [TalkerDecision("speak", stale), TalkerDecision("speak", "Found it.")],
            [_text_round("Account found.")],
            _messages(("user", "Check my account."), ("assistant", stale), ("user", "It is 4472.")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertIn("talker.repeat_replaced", self.kinds())
        self.assertEqual(len(self.backend.payloads), 1)
        self.assertNotIn(stale, self.spoken)

    async def test_the_repeat_guard_can_be_turned_off(self):
        stale = "I am still unable to locate your account with the information provided."
        config = _config(guards=replace(_config().guards, repeat_guard=False))
        self.make(
            [TalkerDecision("speak", stale)], [], _messages(("assistant", stale), ("user", "Again?")), config=config
        )
        await self.session.on_turn()
        self.assertEqual(self.spoken, [stale])

    async def test_a_failed_routing_call_delegates_instead_of_surfacing_an_error(self):
        self.make(
            [RuntimeError("hub down"), TalkerDecision("speak", "Here you go.")],
            [_text_round("Menu.")],
            _messages(("user", "What teas do you have?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertIn("talker.recovered", self.kinds())
        self.assertEqual(self.spoken, ["Here you go."])

    async def test_a_failed_routing_call_raises_when_recovery_is_off(self):
        config = _config(reliability=replace(FAST, recover_failures=False))
        self.make([RuntimeError("hub down")], [], _messages(("user", "Hi")), config=config)
        with self.assertRaises(RuntimeError):
            await self.session.on_turn()

    async def test_a_silent_decision_is_floored_to_a_delegation(self):
        self.make(
            [TalkerDecision("listen", ""), TalkerDecision("speak", "Menu.")],
            [_text_round("Menu.")],
            _messages(("user", "Teas?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(len(self.backend.payloads), 1)


class RouterModeTests(_SessionCase):
    def messages(self):
        return _messages(("user", "Hi"), ("assistant", "Your latte is ready."), ("user", "Okay."))

    async def test_enforced_router_drops_an_acknowledgment_before_the_talker(self):
        config = _config(guards=replace(_config().guards, turn_router=True))
        self.make([], [], self.messages(), config=config)
        await self.session.on_turn()
        self.assertEqual((self.frontend.calls, self.spoken), ([], []))
        self.assertIn("router.dropped", self.kinds())

    async def test_shadow_router_only_traces_and_still_asks_the_talker(self):
        config = _config(guards=replace(_config().guards, turn_router_shadow=True))
        self.make([TalkerDecision("speak", "Anything else?")], [], self.messages(), config=config)
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["Anything else?"])
        self.assertIn("router.would_drop", self.kinds())
        self.assertNotIn("router.dropped", self.kinds())

    async def test_router_is_off_by_default(self):
        self.make([TalkerDecision("speak", "Anything else?")], [], self.messages())
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["Anything else?"])
        self.assertEqual(self.kinds().count("router.would_drop"), 0)


class AnswerDeliveryTests(_SessionCase):
    async def test_every_delegations_answer_is_spoken(self):
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "The latte is added."),
                TalkerDecision("speak", "The muffin is added."),
            ],
            [_text_round("first"), _text_round("second")],
            _messages(("user", "Add a latte.")),
            config=_config(delegation=DelegationConfig(merge_pending=False)),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        self.conversation.add({"role": "user", "content": "And a muffin."})
        await self.session.on_turn()
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(self.spoken, ["The latte is added.", "The muffin is added."])
        self.assertNotIn("commentary.suppressed", self.kinds())
        results = [m["content"] for m in self.conversation.messages()]
        self.assertEqual(results.count(BACKEND_RESULT_PREFIX + "first"), 1)

    async def test_a_backchannel_in_the_meantime_does_not_discard_the_answer(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Your total is five.")],
            [_text_round("Total: $5.00.")],
            _messages(("user", "What's my total?")),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        self.conversation.add({"role": "user", "content": "Okay."})  # said while the backend works
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["Your total is five."])

    async def test_an_answer_waits_for_the_caller_to_finish_speaking(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Ready.")],
            [_text_round("done")],
            _messages(("user", "Order.")),
        )
        self.session.set_user_speaking(True)
        await self.session.on_turn()
        await asyncio.sleep(0.1)
        self.assertEqual(self.spoken, [])
        self.assertIn("commentary.held_for_caller", self.kinds())
        self.session.set_user_speaking(False)
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["Ready."])

    async def test_a_failed_phrasing_call_speaks_the_verified_text_as_written(self):
        self.make(
            [TalkerDecision("delegate", ""), RuntimeError("hub down"), RuntimeError("hub down")],
            [_text_round("**Total:** $5.00")],
            _messages(("user", "Total?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["Total: $5.00."])
        self.assertIn("commentary.recovered", self.kinds())

    async def test_a_silent_phrasing_decision_still_speaks_the_answer(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("listen", "")],
            [_text_round("Your order is BB-42.")],
            _messages(("user", "Order number?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["Your order is BB-42."])
        self.assertIn("commentary.answer_forced", self.kinds())

    async def test_commentary_the_application_sends_is_always_spoken(self):
        # The writer may word it, but commentary is "say this aloud": it cannot be dropped as irrelevant.
        self.make([TalkerDecision("listen", "")], [], _messages(("user", "Hi"), ("assistant", "Welcome.")))
        await self.session.on_commentary("The kitchen is running about five minutes behind today.", None)
        self.assertEqual(self.spoken, ["The kitchen is running about five minutes behind today."])
        self.assertIn("commentary.forced", self.kinds())

    async def test_commentary_is_spoken_in_the_writers_words_when_it_has_some(self):
        self.make(
            [TalkerDecision("speak", "Heads up, the kitchen is a few minutes behind.")],
            [],
            _messages(("user", "Hi"), ("assistant", "Welcome.")),
        )
        await self.session.on_commentary("The kitchen is running about five minutes behind today.", None)
        self.assertEqual(self.spoken, ["Heads up, the kitchen is a few minutes behind."])
        request = self.frontend.calls[0]
        self.assertIn("asked for this to be said", str(request))  # the writer is told it must speak

    async def test_a_phrasing_decision_to_delegate_speaks_the_verified_text(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("delegate", "")],
            [_text_round("Your order is BB-42.")],
            _messages(("user", "Order number?")),
        )
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["Your order is BB-42."])


# --------------------------------------------------------------------------- delegation worker
class WorkerTests(unittest.IsolatedAsyncioTestCase):
    def worker(self, rounds, tools=None, reliability=FAST, policy=None):
        self.backend = ScriptedBackend(rounds, reliability=reliability)
        self.answers: list[str] = []
        self.abandoned: list[str] = []

        async def on_answer(text, task):
            self.answers.append(text)

        self.tools = tools or RecordingTools('{"items": []}')
        worker = DelegationWorker(
            Thinker(self.backend, get_prompt_set("v1")),
            BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            self.tools,
            reliability,
            on_answer=on_answer,
            on_abandoned=self.abandoned.append,
            policy=policy,
        )
        worker.start()
        self.addAsyncCleanup(worker.close)
        return worker

    def user(self, text):
        return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]

    async def wait_for_answer(self):
        async with asyncio.timeout(2):
            while not self.answers:
                await asyncio.sleep(0.005)

    async def test_tool_round_then_answer_with_the_full_history_resent(self):
        worker = self.worker([_call_round("get_menu", {"category": "tea"}), _text_round("We have Earl Grey.")])
        await worker.submit(self.user("Teas?"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, ["We have Earl Grey."])
        self.assertEqual(self.tools.calls, [("call_1", "get_menu", '{"category": "tea"}')])
        second = self.backend.payloads[1]["input"]
        self.assertEqual(second[0]["role"], "user")
        self.assertEqual([item.get("type") for item in second[1:]], ["function_call", "function_call_output"])
        self.assertEqual(second[-1]["output"], '{"items": []}')
        self.assertEqual(self.backend.payloads[1]["tools"], CAFE_TOOLS)

    async def test_a_repeated_call_id_does_not_run_the_tool_twice(self):
        worker = self.worker(
            [_call_round("get_menu", {}, "dup"), _call_round("get_menu", {}, "dup"), _text_round("done")]
        )
        await worker.submit(self.user("Menu?"))
        await self.wait_for_answer()
        self.assertEqual(len(self.tools.calls), 1)

    async def test_only_one_backend_round_runs_at_a_time_and_the_queue_is_bounded(self):
        worker = self.worker(
            [_text_round("one"), _text_round("two")],
            reliability=replace(FAST, max_pending_delegations=1),
            policy=DelegationConfig(merge_pending=False),
        )
        self.backend.gate = asyncio.Event()
        first = await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)
        await worker.submit(self.user("b"))
        self.assertEqual(len(worker.pending), 1)
        with self.assertRaisesRegex(Exception, "full"):
            await worker.submit(self.user("c"))
        self.assertIsNotNone(first)
        self.backend.gate.set()

    async def test_a_failure_before_any_function_call_is_retried_once(self):
        worker = self.worker([[RuntimeError("blip")], _text_round("recovered")])
        await worker.submit(self.user("hi"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, ["recovered"])
        self.assertEqual(len(self.backend.payloads), 2)

    async def test_a_failure_in_a_round_that_emitted_a_function_call_is_never_retried(self):
        item = {"type": "function_call", "call_id": "c1", "name": "manage_cart", "arguments": "{}"}
        broken = [{"type": "response.output_item.done", "item": item}, RuntimeError("stream broke")]
        worker = self.worker([broken, _text_round("must not be asked")])
        task_id = await worker.submit(self.user("Add a latte."))
        await self.wait_for_answer()
        self.assertEqual(self.answers, [APOLOGY])
        self.assertEqual(len(self.backend.payloads), 1)  # no second attempt: an action may be in flight
        self.assertEqual(self.tools.calls, [])
        self.assertEqual(self.abandoned, [task_id])

    async def test_a_round_whose_tools_failed_leaves_the_history_valid_for_the_next_task(self):
        class DownTools(RecordingTools):
            async def execute(self, call_id, name, arguments):
                raise RuntimeError("the tool service is down")

        worker = self.worker(
            [_call_round("get_menu", {"category": "tea"}), _text_round("Back again.")], tools=DownTools()
        )
        await worker.submit(self.user("Teas?"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, [APOLOGY])
        self.answers.clear()
        await worker.submit(self.user("Anything now?"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, ["Back again."])
        sent = self.backend.payloads[1]["input"]
        asked = [i["call_id"] for i in sent if i.get("type") == "function_call"]
        answered = [i["call_id"] for i in sent if i.get("type") == "function_call_output"]
        self.assertEqual((asked, answered), (["call_1"], ["call_1"]))  # no call is left without its output

    async def test_a_backend_that_answers_nothing_does_not_leave_the_caller_waiting(self):
        worker = self.worker([[{"type": "response.completed", "response": {"id": "r"}}]])
        task_id = await worker.submit(self.user("hi"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, [APOLOGY])
        self.assertEqual(self.abandoned, [task_id])

    async def test_a_failure_while_delivering_an_answer_does_not_stop_the_worker(self):
        self.backend = ScriptedBackend([_text_round("first"), _text_round("second")])
        delivered: list[str] = []

        async def on_answer(text, task):
            if text == "first":
                raise RuntimeError("the speech service failed")
            delivered.append(text)

        worker = DelegationWorker(
            Thinker(self.backend, get_prompt_set("v1")),
            BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            RecordingTools(),
            FAST,
            on_answer=on_answer,
            on_abandoned=lambda task_id: None,
            policy=DelegationConfig(merge_pending=False),
        )
        worker.start()
        self.addAsyncCleanup(worker.close)
        await worker.submit(self.user("one"))
        await worker.submit(self.user("two"))
        async with asyncio.timeout(2):
            while not delivered:
                await asyncio.sleep(0.005)
        self.assertEqual(delivered, ["second"])  # no apology on top of the answer that failed to play

    async def test_old_requests_are_dropped_whole_when_the_history_outgrows_its_budget(self):
        worker = self.worker([_text_round("a" * 400), _text_round("b" * 400), _text_round("c")])
        worker.context_tokens = 200
        for text in ("one", "two", "three"):
            self.answers.clear()
            await worker.submit(self.user(text))
            await self.wait_for_answer()
        self.assertEqual(self.answers, ["c"])  # the third request still ran: the budget is kept by dropping the oldest
        last = self.backend.payloads[-1]["input"]
        self.assertEqual(last[0]["role"], "user")  # what is kept starts at a request, never mid-way through one
        self.assertNotIn("one", json.dumps(last))

    async def test_only_a_completed_response_can_be_chained_from(self):
        class Stateful(ScriptedBackend):
            stateful = True

        failed = [{"type": "response.incomplete", "response": {"id": "resp_failed"}}]
        self.backend = Stateful([failed, failed, _text_round("ok", "resp_ok"), _text_round("again", "resp_2")])
        self.answers = []
        self.abandoned = []

        async def on_answer(text, task):
            self.answers.append(text)

        worker = DelegationWorker(
            Thinker(self.backend, get_prompt_set("v1")),
            BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            RecordingTools(),
            FAST,
            on_answer=on_answer,
            on_abandoned=self.abandoned.append,
        )
        worker.start()
        self.addAsyncCleanup(worker.close)
        await worker.submit(self.user("first"))
        await self.wait_for_answer()
        self.answers.clear()
        await worker.submit(self.user("second"))
        await self.wait_for_answer()
        self.assertNotIn("resp_failed", [p.get("previous_response_id") for p in self.backend.payloads])

    async def test_a_failure_after_tools_ran_only_re_asks_the_model_once(self):
        worker = self.worker(
            [_call_round("manage_cart", {"action": "add"}), [RuntimeError("blip")], _text_round("Added.")]
        )
        await worker.submit(self.user("Add a latte."))
        await self.wait_for_answer()
        self.assertEqual(self.answers, ["Added."])
        self.assertEqual(len(self.tools.calls), 1)  # the action ran once and is not repeated
        self.assertEqual(len(self.backend.payloads), 3)

    async def test_a_second_failure_in_the_same_round_apologizes(self):
        worker = self.worker([[RuntimeError("one")], [RuntimeError("two")]])
        task_id = await worker.submit(self.user("hi"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, [APOLOGY])
        self.assertEqual(self.abandoned, [task_id])

    async def test_the_round_limit_stops_a_tool_loop(self):
        worker = self.worker(
            [_call_round("get_menu", {}, f"c{i}") for i in range(3)], reliability=replace(FAST, max_tool_rounds=3)
        )
        await worker.submit(self.user("loop"))
        await self.wait_for_answer()
        self.assertEqual(self.answers, [APOLOGY])

    async def test_a_context_over_budget_fails_the_task_instead_of_sending_it(self):
        worker = self.worker([_text_round("never")])
        worker.context_tokens = 5
        await worker.submit(self.user("x" * 400))
        await self.wait_for_answer()
        self.assertEqual(self.backend.payloads, [])

    async def test_closing_marks_unfinished_tasks_abandoned(self):
        worker = self.worker([_text_round("a"), _text_round("b")], policy=DelegationConfig(merge_pending=False))
        self.backend.gate = asyncio.Event()
        first = await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)
        second = await worker.submit(self.user("b"))
        await worker.close()
        self.assertEqual(sorted(self.abandoned), sorted([first, second]))


# ------------------------------------------------------------------------------- cafe tools
class CafeToolTests(unittest.IsolatedAsyncioTestCase):
    async def run_tool(self, tools, call_id, name, **arguments):
        return json.loads(await tools.execute(call_id, name, json.dumps(arguments)))

    async def test_a_bad_call_cannot_poison_the_cart(self):
        tools = CafeTools()
        bad = [
            ("manage_cart", {"action": "add", "item_id": "nope"}),
            ("manage_cart", {"action": "add", "item_id": None}),
            ("manage_cart", {"action": "remove", "line_id": "line_99"}),
            ("manage_cart", {"action": "update_quantity"}),
        ]
        for i, (name, args) in enumerate(bad):
            result = await self.run_tool(tools, f"bad{i}", name, **args)
            self.assertEqual(result["status"], "error")
        added = await self.run_tool(tools, "a", "manage_cart", action="add", item_id="lat", size="huge", quantity="x")
        line = added["cart"]["lines"][0]
        self.assertEqual((line["size"], line["quantity"]), ("medium", 1))  # defaults, not a crash later
        self.assertEqual((await self.run_tool(tools, "v", "manage_cart", action="view"))["cart"]["item_count"], 1)
        self.assertEqual(
            (await self.run_tool(tools, "m", "get_menu", query=None, category=None))["items"][0]["id"], "esp"
        )

    async def test_cart_total_includes_size_modifiers_and_tax(self):
        tools = CafeTools()
        cart = (
            await self.run_tool(
                tools, "1", "manage_cart", action="add", item_id="lat", size="large", modifiers=["oat_milk"]
            )
        )["cart"]
        self.assertEqual((cart["subtotal"], cart["tax"], cart["total"]), (6.25, 0.53, 6.78))

    async def test_a_repeated_call_id_neither_adds_a_second_item_nor_places_a_second_order(self):
        tools = CafeTools()
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="esp")
        self.assertEqual(tools.snapshot()["item_count"], 1)
        first = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        again = await self.run_tool(tools, "p", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual((first["status"], first["order"]["order_id"]), ("confirmed", again["order"]["order_id"]))

    async def test_an_order_needs_explicit_confirmation(self):
        tools = CafeTools()
        await self.run_tool(tools, "a", "manage_cart", action="add", item_id="crs", quantity=2)
        pending = await self.run_tool(tools, "b", "place_order", customer_name="Sam", confirmed=False)
        self.assertEqual(pending["status"], "needs_confirmation")
        self.assertEqual(tools.last_order, None)
        empty = await self.run_tool(CafeTools(), "c", "place_order", customer_name="Sam", confirmed=True)
        self.assertEqual(empty["status"], "error")

    async def test_bad_input_returns_an_error_result_not_an_exception(self):
        tools = CafeTools()
        self.assertEqual((await self.run_tool(tools, "x", "nope"))["status"], "error")
        self.assertEqual(
            (await self.run_tool(tools, "y", "manage_cart", action="add", item_id="zzz"))["status"], "error"
        )
        self.assertEqual(json.loads(await tools.execute("z", "manage_cart", "not json"))["status"], "error")

    async def test_menu_filters_by_category(self):
        menu = await self.run_tool(CafeTools(), "m", "get_menu", category="tea")
        self.assertEqual([item["name"] for item in menu["items"]], ["Matcha Latte", "Earl Grey"])

    def test_tool_schemas_match_the_executor(self):
        self.assertEqual({tool["name"] for tool in CAFE_TOOLS}, {"get_menu", "manage_cart", "place_order"})


class EndToEndTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_cafe_order_flows_through_the_registered_chat_providers_and_real_tools(self):
        # One fake endpoint serves every model call, in order: routing, tool round, answer round, phrasing.
        client = FakeChat(
            _decision_chunks('{"action":"delegate","speech":"Let me add that."}'),
            [
                chunk(
                    tool_calls=[
                        tool_fragment(0, "call_1", "manage_cart", '{"action":"add","item_id":"lat","size":"large"}')
                    ]
                ),
                chunk(finish="tool_calls"),
            ],
            [chunk("Added one large Latte. Your total is $5.97 with tax.", finish="stop")],
            _decision_chunks(
                '{"action":"speak","speech":"One large latte is in your cart, five ninety-seven with tax."}'
            ),
        )
        config = _config(
            frontend=RoleConfig("chat-completions", "llm"), backend=RoleConfig("chat-completions", "thinker-llm")
        )
        agent = build_agent(config, ENDPOINT, ENDPOINT, frontend_client=client, backend_client=client)
        spoken: list[str] = []

        async def say(text):
            spoken.append(text)

        tools = CafeTools()
        conversation = ListConversation(_messages(("user", "A large latte, please.")))
        session = CascadeSession(
            agent,
            config,
            conversation,
            backend_task=BackendTask("test/model", "Take orders.", CAFE_TOOLS),
            tools=tools,
            say=say,
        )
        session.start()
        try:
            await session.on_turn()
            async with asyncio.timeout(2):
                while len(spoken) < 2:
                    await asyncio.sleep(0.005)
        finally:
            await session.worker.close()
        self.assertEqual(spoken, ["Let me add that.", "One large latte is in your cart, five ninety-seven with tax."])
        self.assertEqual(tools.snapshot()["item_count"], 1)
        self.assertEqual(tools.snapshot()["total"], 5.97)
        tool_result = next(m for m in client.requests[2]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_result["content"])["status"], "added")


# --------------------------------------------------------------------------------- pipecat glue
class HistoryWindowTests(unittest.IsolatedAsyncioTestCase):
    def test_a_context_keeps_its_prompt_and_only_the_latest_messages(self):
        context = LLMContext([{"role": "system", "content": "You are Remy."}])
        for i in range(10):
            context.add_message({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
        ContextConversation(context).trim(4)
        self.assertEqual([m["content"] for m in context.get_messages()], ["You are Remy.", "m6", "m7", "m8", "m9"])

    def test_a_short_conversation_is_left_alone(self):
        context = LLMContext([{"role": "system", "content": "p"}, {"role": "user", "content": "hi"}])
        ContextConversation(context).trim(20)
        self.assertEqual(len(context.get_messages()), 2)

    async def test_a_live_turn_trims_the_conversation_to_the_configured_window(self):
        context = LLMContext([{"role": "system", "content": "You are Remy."}])
        for i in range(30):
            context.add_message({"role": "assistant" if i % 2 else "user", "content": f"old {i}"})
        context.add_message({"role": "user", "content": "What time do you close?"})
        spoken: list[str] = []

        async def say(text):
            spoken.append(text)

        session = CascadeSession(
            _agent(ScriptedFrontend([TalkerDecision("speak", "At nine.")]), ScriptedBackend([])),
            _config(chat_history_recent_turns=6),
            ContextConversation(context),
            backend_task=BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            tools=RecordingTools(),
            say=say,
        )
        session.start()
        self.addAsyncCleanup(session.close)
        await session.on_turn()
        messages = context.get_messages()
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(len(messages), 1 + 6)
        self.assertEqual(messages[-1]["content"], "What time do you close?")
        self.assertEqual(spoken, ["At nine."])


class BackendSettingsTests(unittest.TestCase):
    def task(self, **responses):
        config = SessionConfig.model_validate(
            {"model": "live-1", "delegation": {"type": "responses", "responses": {"model": "m", **responses}}}
        )
        return backend_task_for(config, "fallback", "static")

    def test_what_the_client_set_reaches_the_backend_request(self):
        task = self.task(
            reasoning={"effort": "high"},
            service_tier="priority",
            tool_choice={"type": "function", "name": "get_menu"},
            tools=[{"type": "function", "name": "get_menu"}],
        )
        payload = task.payload()
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["service_tier"], "priority")
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "get_menu"})

    def test_settings_the_client_left_at_their_defaults_are_not_sent(self):
        payload = self.task().payload()
        for key in ("reasoning", "service_tier", "parallel_tool_calls", "text"):
            self.assertNotIn(key, payload)

    def test_a_backend_without_a_model_check_accepts_any_model(self):
        from examples.frontend_backend_live.models.openai_responses.backend import OpenAIResponsesBackend

        backend = OpenAIResponsesBackend(ENDPOINT, FAST, client=object())
        backend.validate_model("any-model-at-all")  # the base class accepts; only chat-completions is selective


class ConversationTests(unittest.TestCase):
    def test_split_messages_separates_instructions_from_voice_history(self):
        instructions, history = split_messages(
            [
                {"role": "system", "content": "Be Remy."},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": "Hello."},
                {"role": "developer", "content": "Kiosk."},
            ]
        )
        self.assertEqual(instructions, "Be Remy.\n\nKiosk.")
        self.assertEqual(history, [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello."}])

    def test_context_adapter_reads_and_appends(self):
        context = LLMContext([{"role": "system", "content": "s"}])
        conversation = ContextConversation(context)
        conversation.add({"role": "user", "content": "hi"})
        self.assertEqual([m["role"] for m in conversation.messages()], ["system", "user"])


class ProcessorTests(unittest.IsolatedAsyncioTestCase):
    def processor(self, script):
        context = LLMContext(_messages(("user", "When do you open?")))
        frontend = ScriptedFrontend(script)
        backend = ScriptedBackend([])
        processor = LiveTalkerProcessor()
        session = CascadeSession(
            _agent(frontend, backend),
            _config(),
            ContextConversation(context),
            backend_task=BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            tools=RecordingTools(),
            say=processor.speak,
        )
        processor.attach(session)
        self.backend = backend
        return processor, context, frontend

    async def test_a_turn_is_spoken_as_one_llm_response(self):
        processor, context, frontend = self.processor([TalkerDecision("speak", "We open at seven.")])
        down, _ = await run_test(
            processor,
            frames_to_send=[LLMContextFrame(context=context)],
            expected_down_frames=[LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame],
        )
        self.assertEqual(down[1].text, "We open at seven.")
        self.assertEqual(len(frontend.calls), 1)

    async def test_utterances_spoken_at_the_same_time_do_not_interleave_their_frames(self):
        processor, context, frontend = self.processor([])
        sent = []

        async def record(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append(type(frame).__name__)
            await asyncio.sleep(0)  # let the other speaker run between frames

        processor.push_frame = record
        await asyncio.gather(
            processor.speak("A lead-in."), processor.speak("An answer."), processor.speak("An update.")
        )
        self.assertEqual(sent, ["LLMFullResponseStartFrame", "LLMTextFrame", "LLMFullResponseEndFrame"] * 3)

    async def test_a_speculative_context_is_ignored(self):
        processor, context, frontend = self.processor([])
        await run_test(
            processor,
            frames_to_send=[LLMContextFrame(context=context, speculation=True)],
            expected_down_frames=[],
        )
        self.assertEqual(frontend.calls, [])

    async def test_a_delegating_turn_speaks_nothing_until_the_answer(self):
        processor, context, frontend = self.processor([TalkerDecision("delegate", "")])
        self.backend.gate = asyncio.Event()  # the backend is still working
        await run_test(processor, frames_to_send=[LLMContextFrame(context=context)], expected_down_frames=[])
        self.assertEqual(frontend.calls[0]["actions"], ("speak", "delegate"))
        self.assertEqual(len(self.backend.payloads), 1)

    async def test_inside_real_aggregators_the_spoken_reply_lands_in_the_context(self):
        context = LLMContext([{"role": "system", "content": "You are Remy."}])
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
        frontend = ScriptedFrontend([TalkerDecision("speak", "We open at seven.")])
        processor = LiveTalkerProcessor()
        session = CascadeSession(
            _agent(frontend, ScriptedBackend([])),
            _config(),
            ContextConversation(context),
            backend_task=BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            tools=RecordingTools(),
            say=processor.speak,
        )
        processor.attach(session)
        await run_test(
            Pipeline([user_aggregator, processor, assistant_aggregator]),
            frames_to_send=[
                LLMMessagesAppendFrame(messages=[{"role": "user", "content": "When do you open?"}], run_llm=True)
            ],
            expected_down_frames=None,  # the assistant aggregator consumes the response frames
        )
        self.assertEqual(frontend.calls[0]["history"], [{"role": "user", "content": "When do you open?"}])
        self.assertEqual(
            [(m["role"], m["content"]) for m in context.get_messages()[1:]],
            [("user", "When do you open?"), ("assistant", "We open at seven.")],
        )


if __name__ == "__main__":
    unittest.main()
