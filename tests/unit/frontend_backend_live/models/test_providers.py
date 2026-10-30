# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The frontend model providers."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx
from openai import BadRequestError
from openai.types.responses import Response

from examples.frontend_backend_live.cascade.agent import LiveAgent, build_agent
from examples.frontend_backend_live.cascade.normalizer import TurnNormalizer
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.talker.talker import PromptPart, Talker
from examples.frontend_backend_live.cascade.writer.writer import CommentaryWriter
from examples.frontend_backend_live.config_manager.schema import NormalizerConfig
from examples.frontend_backend_live.models import FRONTENDS, CallSpec, Frontend
from examples.frontend_backend_live.models.base import DEFAULT_MAX_OUTPUT_TOKENS
from examples.frontend_backend_live.models.chat_completions import (
    JSON_RULES,
    ChatCompletionsFrontend,
    chat_messages,
)
from examples.frontend_backend_live.models.decision import TalkerDecision, decision_schema
from examples.frontend_backend_live.models.openai_responses import OpenAIResponsesFrontend
from examples.frontend_backend_live.prompts import get_prompt_set
from tests.unit.cascade_helpers import CASCADE_ENDPOINT as ENDPOINT
from tests.unit.cascade_helpers import (
    FAST,
    ListConversation,
    _config,
    _decision_chunks,
    _delegation,
    _messages,
    _settle,
)
from tests.unit.contracts.backend import ScriptedBackend
from tests.unit.core_helpers import FakeChat, _text_round, _timeout

DEFAULT = get_prompt_set("default")
TALKER = CallSpec("talker", DEFAULT.actions, DEFAULT.call_hints.talker)
HISTORY = [{"role": "user", "content": "hi"}]


def _bad_request(message):
    response = httpx.Response(400, request=httpx.Request("POST", "https://hub.example/v1"))
    return BadRequestError(message, response=response, body=None)


def _response(text="", *, status="completed", refusal=None, reason=None):
    """One Responses result whose message carries ``text`` (or a refusal)."""
    part = (
        {"type": "refusal", "refusal": refusal} if refusal else {"type": "output_text", "text": text, "annotations": []}
    )
    return Response.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "model": "test/model",
            "status": status,
            "output": [
                {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "content": [part]}
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "incomplete_details": {"reason": reason} if reason else None,
        }
    )


class FakeResponses:
    """Scripted ``responses.create``: each entry is a ``Response`` or an exception; each request is kept."""

    def __init__(self, *script):
        self.script, self.requests, self.closed = list(script), [], False
        self.responses = self

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step

    async def close(self):
        self.closed = True


def _decision(action, speech=""):
    return _response(f'{{"action":"{action}","speech":"{speech}"}}')


# ---------------------------------------------------------------------------------------------- the table and recipe
class ProviderTableTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_new_frontend_is_one_subclass_and_one_entry(self):
        class Echo(Frontend):
            name = "scripted-test"

            async def complete_once(self, instructions, history, spec):
                return TalkerDecision("speak", f"heard {len(history)}")

        config = _config(provider="scripted-test")
        with patch.dict(FRONTENDS, {Echo.name: Echo}):
            agent = build_agent(config, ENDPOINT)
        decision = await agent.talker.decide("Be kind.", HISTORY)
        self.assertEqual((decision.action, decision.speech), ("speak", "heard 2"))  # the history, then the runtime data
        await agent.close()

    def test_an_unknown_provider_names_the_known_ones(self):
        config = _config(provider="nope")
        with self.assertRaisesRegex(ValueError, r"Unknown frontend provider 'nope'; known: chat-completions, openai-"):
            build_agent(config, ENDPOINT)

    def test_every_provider_takes_the_same_constructor_so_the_agent_always_passes_the_client(self):
        for name in FRONTENDS:
            with self.subTest(provider=name):
                client = FakeChat() if name == "chat-completions" else FakeResponses()
                config = _config(provider=name)
                agent = LiveAgent.from_config(config, ENDPOINT, frontend_client=client)
                self.assertIs(agent.frontend.client, client)
                self.assertEqual(build_agent, LiveAgent.from_config)

    def test_the_agent_builds_the_normalizer_on_its_prompt_set_only_when_enabled(self):
        config = _config(normalizer=NormalizerConfig(enabled=True), prompt_version="default")
        config = replace(config, provider="chat-completions")
        agent = build_agent(config, ENDPOINT, frontend_client=FakeChat())
        self.assertIsInstance(agent.normalizer, TurnNormalizer)
        self.assertIs(agent.normalizer.prompts, agent.prompts)
        self.assertIs(agent.prompts, DEFAULT)
        plain = replace(config, normalizer=NormalizerConfig())
        self.assertIsNone(build_agent(plain, ENDPOINT, frontend_client=FakeChat()).normalizer)


# ------------------------------------------------------------------------------------------------- call specs
class CallSpecTests(unittest.IsolatedAsyncioTestCase):
    def test_the_schema_is_the_specs_actions_and_hint(self):
        schema = decision_schema(TALKER)
        self.assertEqual(schema["properties"]["action"]["enum"], ["speak", "delegate", "wait"])
        self.assertEqual(schema["properties"]["speech"]["description"], DEFAULT.call_hints.talker)
        self.assertEqual(schema["required"], ["action", "speech"])
        self.assertNotIn("description", decision_schema(CallSpec("x", ("speak",), ""))["properties"]["speech"])

    def test_each_caller_states_its_role_actions_and_hint(self):
        frontend = ChatCompletionsFrontend(ENDPOINT, FAST, FakeChat())
        talker, writer = Talker(frontend, DEFAULT), CommentaryWriter(frontend, DEFAULT)
        normalizer = TurnNormalizer(frontend, DEFAULT, NormalizerConfig(enabled=True))
        self.assertEqual((talker.spec.role, talker.spec.actions), ("talker", ("speak", "delegate", "wait")))
        self.assertEqual((writer.answer_spec.role, writer.answer_spec.actions), ("writer", ("speak",)))
        self.assertEqual(writer.answer_spec.hint, DEFAULT.call_hints.answer)
        self.assertEqual((normalizer.spec.role, normalizer.spec.hint), ("normalizer", DEFAULT.call_hints.normalizer))

    async def test_the_retry_log_names_the_callers_role_and_the_spec_sets_the_attempts(self):
        roles = []

        async def recording(call, *, attempts, role):
            roles.append((role, attempts))
            return await call()

        client = FakeChat(*[_decision_chunks('{"action":"speak","speech":"x"}') for _ in range(3)])
        frontend = ChatCompletionsFrontend(ENDPOINT, FAST, client)
        with patch("examples.frontend_backend_live.models.base.with_retries", recording):
            await TurnNormalizer(frontend, DEFAULT, NormalizerConfig(enabled=True)).normalize([], "two bags")
            await CommentaryWriter(frontend, DEFAULT).render("Be Remy.", HISTORY, "Done.", attempts=1)
            await Talker(frontend, DEFAULT).decide("Be Remy.", HISTORY)
        self.assertEqual(roles, [("normalizer", 2), ("writer", 1), ("talker", 2)])


# ------------------------------------------------------------------------------------- the slot's output limit
class OutputLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_slots_max_tokens_reaches_every_chat_call_and_512_without_one(self):
        for extra, expected in (({}, DEFAULT_MAX_OUTPUT_TOKENS), ({"max_tokens": 64}, 64)):
            with self.subTest(extra_params=extra):
                client = FakeChat(
                    _decision_chunks('{"action":"speak","speech":"Hi."}'),
                    _decision_chunks('{"action":"speak","speech":"Done."}'),
                )
                frontend = ChatCompletionsFrontend(replace(ENDPOINT, extra_params=extra), FAST, client)
                await Talker(frontend, DEFAULT).decide("Be Remy.", HISTORY)
                await CommentaryWriter(frontend, DEFAULT).render("Be Remy.", HISTORY, "Done.")
                self.assertEqual([request["max_tokens"] for request in client.requests], [expected, expected])

    async def test_the_slots_max_tokens_reaches_every_responses_call(self):
        for extra, expected in (({}, DEFAULT_MAX_OUTPUT_TOKENS), ({"max_tokens": 64}, 64)):
            with self.subTest(extra_params=extra):
                client = FakeResponses(_decision("speak", "Hi."), _decision("speak", "Done."))
                frontend = OpenAIResponsesFrontend(replace(ENDPOINT, extra_params=extra), FAST, client)
                await Talker(frontend, DEFAULT).decide("Be Remy.", HISTORY)
                await CommentaryWriter(frontend, DEFAULT).render("Be Remy.", HISTORY, "Done.")
                self.assertEqual([request["max_output_tokens"] for request in client.requests], [expected, expected])


# ------------------------------------------------------------------------------- the talker's layout, as sent
class LayoutTests(unittest.IsolatedAsyncioTestCase):
    def parts(self):
        return [
            PromptPart("session.instructions", "Be Remy."),
            PromptPart("session.instructions.append", "Speak slowly.", "\n\n"),
        ]

    def test_the_parts_name_their_sources_and_join_into_what_decide_sends(self):
        talker = Talker(ChatCompletionsFrontend(ENDPOINT, FAST, FakeChat()), DEFAULT)
        parts = talker.routing_parts(self.parts())
        self.assertEqual(
            [part.source for part in parts],
            [
                "prompt set default: the voice's role",
                "session.instructions",
                "session.instructions.append",
                "prompt set default: how the instructions apply",
                "prompt set default: routing rules",
            ],
        )
        joined = "".join(part.sep + part.text for part in parts)
        self.assertEqual(joined, DEFAULT.framed("Be Remy.\n\nSpeak slowly.") + "\n" + DEFAULT.routing)
        # The work line goes after the conversation, in the runtime data, never in the instructions.
        runtime = talker.runtime_data("WORK")
        self.assertEqual(runtime, DEFAULT.runtime_label + "\nWORK\n\n" + DEFAULT.decision_cue)
        self.assertEqual(talker.runtime_data(), DEFAULT.runtime_label + "\n" + DEFAULT.decision_cue)

    async def test_chat_completions_sends_the_layout_then_its_reply_format(self):
        client = FakeChat(_decision_chunks('{"action":"speak","speech":"Hi."}'))
        talker = Talker(ChatCompletionsFrontend(ENDPOINT, FAST, client), DEFAULT)
        await talker.decide("Be Remy.\n\nSpeak slowly.", HISTORY, "WORK")
        joined = "".join(part.sep + part.text for part in talker.routing_parts(self.parts()))
        self.assertEqual(ChatCompletionsFrontend.reply_format, JSON_RULES)
        messages = client.requests[0]["messages"]
        self.assertEqual(messages[0]["content"], joined + "\n" + JSON_RULES)
        # The caller's newest turn, then the runtime data a blank line after it, in one user message.
        self.assertEqual(messages[-1], {"role": "user", "content": "hi\n\n" + talker.runtime_data("WORK")})

    async def test_openai_responses_sends_the_layout_as_its_instructions_with_no_reply_format(self):
        client = FakeResponses(_decision("speak", "Hi."))
        talker = Talker(OpenAIResponsesFrontend(ENDPOINT, FAST, client), DEFAULT)
        await talker.decide("Be Remy.\n\nSpeak slowly.", HISTORY, "WORK")
        joined = "".join(part.sep + part.text for part in talker.routing_parts(self.parts()))
        self.assertEqual(OpenAIResponsesFrontend.reply_format, "")
        self.assertEqual(client.requests[0]["instructions"], joined)
        runtime = {"role": "user", "content": "\n" + talker.runtime_data("WORK")}
        self.assertEqual(client.requests[0]["input"], [*HISTORY, runtime])


# --------------------------------------------------------------------------------------- chat-completions model
class ChatFrontendTests(unittest.IsolatedAsyncioTestCase):
    def frontend(self, client, **endpoint):
        return ChatCompletionsFrontend(replace(ENDPOINT, **endpoint), FAST, client)

    async def test_request_uses_sampling_defaults_and_the_endpoint_extra_params(self):
        client = FakeChat(_decision_chunks('{"action":"speak","speech":"Hello."}'))
        decision = await self.frontend(client).complete("You are Remy.", HISTORY, TALKER)
        self.assertEqual((decision.action, decision.speech), ("speak", "Hello."))
        request = client.requests[0]
        self.assertEqual((request["model"], request["temperature"], request["max_tokens"]), ("test/model", 0.0, 512))
        self.assertTrue(request["stream"])
        schema = request["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertEqual(schema["schema"]["properties"]["action"]["enum"], ["speak", "delegate", "wait"])

    async def test_extra_params_override_defaults_without_duplicate_keyword_errors(self):
        client = FakeChat(_decision_chunks('{"action":"delegate","speech":""}'))
        extra = {"temperature": 0.2, "max_tokens": 64, "extra_body": {"repetition_penalty": 1.05}}
        await self.frontend(client, extra_params=extra).complete("x", HISTORY, TALKER)
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
        self.assertEqual((await frontend.complete("x", HISTORY, TALKER)).speech, "Hi.")
        self.assertEqual(client.requests[1]["response_format"], {"type": "json_object"})
        await frontend.complete("x", HISTORY, TALKER)
        self.assertEqual(client.requests[2]["response_format"], {"type": "json_object"})

    async def test_an_unrelated_bad_request_is_not_retried_as_a_format_problem(self):
        frontend = self.frontend(FakeChat(_bad_request("model not found")))
        with self.assertRaises(BadRequestError):
            await frontend.complete("x", HISTORY, TALKER)

    async def test_a_transient_failure_is_retried_and_then_succeeds(self):
        client = FakeChat(_timeout(), _decision_chunks('{"action":"speak","speech":"Back."}'))
        decision = await self.frontend(client).complete("x", HISTORY, TALKER)
        self.assertEqual((decision.speech, decision.diagnostics["attempt"]), ("Back.", 2))

    async def test_unusable_output_delegates_the_original_request(self):
        for text, reason in (("not json", "invalid_json"), ('{"action":"dance","speech":"x"}', "invalid_decision")):
            client = FakeChat(_decision_chunks(text))
            decision = await self.frontend(client).complete("x", HISTORY, TALKER)
            self.assertEqual(
                (decision.action, decision.speech, decision.diagnostics["recovery"]), ("delegate", "", reason)
            )

    async def test_truncated_output_is_incomplete_not_parsed(self):
        client = FakeChat(_decision_chunks('{"action":"speak","speech":"Hel', finish="length"))
        decision = await self.frontend(client).complete("x", HISTORY, TALKER)
        self.assertEqual(decision.diagnostics["recovery"], "incomplete_output")

    async def test_fences_and_leaked_reasoning_around_the_decision_are_tolerated(self):
        for text in (
            '```json\n{"action":"speak","speech":"Hi"}\n```',
            '<think>hmm</think>{"action":"speak","speech":"Hi"}',
        ):
            with self.subTest(text=text):
                decision = await self.frontend(FakeChat(_decision_chunks(text))).complete("x", HISTORY, TALKER)
                self.assertEqual((decision.action, decision.speech), ("speak", "Hi"))

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


# ------------------------------------------------------------------------------------------ the Responses talker
class ResponsesFrontendTests(unittest.IsolatedAsyncioTestCase):
    def frontend(self, client):
        return OpenAIResponsesFrontend(ENDPOINT, FAST, client)

    async def test_one_call_has_reasoning_off_no_storage_and_the_strict_schema(self):
        client = FakeResponses(_decision("speak", "Hello."))
        decision = await self.frontend(client).complete("Be Remy.", HISTORY, TALKER)
        self.assertEqual((decision.action, decision.speech), ("speak", "Hello."))
        request = client.requests[0]
        self.assertEqual(
            (request["model"], request["reasoning"], request["store"]), ("test/model", {"effort": "none"}, False)
        )
        schema = request["text"]["format"]
        self.assertEqual((schema["type"], schema["name"], schema["strict"]), ("json_schema", "voice_decision", True))
        self.assertEqual(schema["schema"], decision_schema(TALKER))

    async def test_a_refusal_is_spoken_and_unusable_output_delegates(self):
        cases = (
            (_response(refusal="I can't help with that."), ("speak", "I can't help with that.", "refusal")),
            (_response(""), ("delegate", "", "empty_output")),
            (_response('{"action":"sing","speech":"x"}'), ("delegate", "", "invalid_decision")),
            (_response(status="incomplete", reason="max_output_tokens"), ("delegate", "", "incomplete_output")),
        )
        for response, expected in cases:
            with self.subTest(expected=expected):
                decision = await self.frontend(FakeResponses(response)).complete("x", HISTORY, TALKER)
                self.assertEqual((decision.action, decision.speech, decision.diagnostics["recovery"]), expected)

    async def test_a_transient_failure_is_retried(self):
        client = FakeResponses(_timeout(), _decision("delegate", "One moment."))
        decision = await self.frontend(client).complete("x", HISTORY, TALKER)
        self.assertEqual((decision.action, len(client.requests)), ("delegate", 2))

    async def test_the_responses_talker_decides_a_turn_and_a_delegation_in_a_call(self):
        # The Responses talker serves a whole cascade call: it speaks a reply, delegates the next turn, and words the
        # backend's answer, which is said.
        client = FakeResponses(
            _decision("speak", "We open at seven."),
            _decision("delegate", "Let me check."),
            _decision("speak", "A latte is four fifty."),
        )
        config = _config(provider="openai-responses", prompt_version="default")
        agent = build_agent(config, ENDPOINT, frontend_client=client)
        backend = ScriptedBackend([_text_round("Latte: $4.50.")])
        delegation = _delegation(backend)
        conversation = ListConversation(_messages())
        spoken: list[str] = []

        async def say(text):
            spoken.append(text)

        session = CascadeSession(agent, config, conversation, delegation=delegation, say=say)
        await delegation.open(None)
        session.start()
        self.addAsyncCleanup(delegation.close)
        self.addAsyncCleanup(session.close)
        conversation.add({"role": "user", "content": "When do you open?"})
        await session.on_turn()
        conversation.add({"role": "user", "content": "How much is a latte?"})
        await session.on_turn()
        await _settle(session, spoken, 3)
        self.assertEqual(spoken, ["We open at seven.", "Let me check.", "A latte is four fifty."])
        talker_request, _, writer_request = client.requests
        self.assertTrue(talker_request["instructions"].startswith(DEFAULT.frame_before))
        self.assertEqual(
            talker_request["text"]["format"]["schema"]["properties"]["action"]["enum"], ["speak", "delegate", "wait"]
        )
        self.assertEqual(writer_request["text"]["format"]["schema"]["properties"]["action"]["enum"], ["speak"])
        self.assertIn("Latte: $4.50.", writer_request["input"][-1]["content"])
        self.assertEqual(len(backend.payloads), 1)


if __name__ == "__main__":
    unittest.main()
