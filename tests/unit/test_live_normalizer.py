# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The turn normalizer: which turns it rewrites, what it keeps, and what the talker and the backend see."""

import asyncio
import unittest

from examples.frontend_backend_live.cascade.conversation import ContextConversation
from examples.frontend_backend_live.cascade.normalizer import (
    NEWEST_TURN,
    TurnNormalizer,
    has_details,
    plausible,
)
from examples.frontend_backend_live.common.history import BACKEND_RESULT_PREFIX
from examples.frontend_backend_live.config_manager.loader import ConfigError, parse_config
from examples.frontend_backend_live.config_manager.schema import NormalizerConfig
from examples.frontend_backend_live.models.decision import TalkerDecision
from tests.unit.test_frontend_backend_live import ScriptedFrontend, _messages, _SessionCase, _settle, _text_round

ON = NormalizerConfig(enabled=True)


def speak(text):
    return TalkerDecision("speak", text, {})


class DetailTests(unittest.TestCase):
    def test_turns_with_details_are_normalized(self):
        for text in (
            "my user id is m i a underscore l i three six six eight",
            "reservation H A T zero four five",
            "I have 2 bags",
            "no checked bags, zero",
            "the twentieth of May",
            "it's spelled L I",
        ):
            self.assertTrue(has_details(text), text)

    def test_plain_turns_are_left_alone(self):
        for text in ("I want to change my flight.", "Yes please.", "Can I talk to a person?", "I need a seat."):
            self.assertFalse(has_details(text), text)

    def test_a_rewrite_that_adds_content_is_implausible(self):
        self.assertTrue(plausible("m i a underscore l i", "mia_li"))
        self.assertFalse(plausible("two bags", ""))
        self.assertFalse(plausible("two bags", "2 bags, and I also want to upgrade to business and add insurance"))


class NormalizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_newest_turn_is_framed_once_after_the_history(self):
        frontend = ScriptedFrontend([speak("my user id is mia_li_3668")])
        history = [
            {"role": "assistant", "content": "What is your user ID?"},
            {"role": "user", "content": "m i a underscore l i three six six eight"},
        ]
        text, diagnostics = await TurnNormalizer(frontend, ON).normalize(history, history[-1]["content"])
        self.assertEqual(text, "my user id is mia_li_3668")
        self.assertEqual(diagnostics["outcome"], "changed")
        sent = frontend.calls[0]["history"]
        self.assertEqual(sent[0], history[0])
        self.assertEqual(sent[-1]["content"], NEWEST_TURN + "m i a underscore l i three six six eight")
        self.assertEqual(len(sent), 2)
        self.assertEqual(frontend.calls[0]["actions"], ("speak",))

    async def test_failures_and_implausible_rewrites_keep_the_turn_as_heard(self):
        heard = "two bags"
        for script, outcome in (
            ([RuntimeError("down"), RuntimeError("down")], "failed"),
            ([speak("")], "rejected"),
            ([speak("2 bags " + "and more " * 20)], "rejected"),
            ([speak("two bags")], "unchanged"),
        ):
            text, diagnostics = await TurnNormalizer(ScriptedFrontend(script), ON).normalize([], heard)
            self.assertEqual((text, diagnostics["outcome"]), (heard, outcome))

    async def test_a_slow_normalization_keeps_the_turn_as_heard(self):
        class Slow(ScriptedFrontend):
            async def complete_once(self, *args):
                await asyncio.sleep(5)

        config = NormalizerConfig(enabled=True, timeout_seconds=0.05)
        text, diagnostics = await TurnNormalizer(Slow([]), config).normalize([], "two bags")
        self.assertEqual((text, diagnostics["outcome"]), ("two bags", "failed"))


class SessionTests(_SessionCase):
    def enable(self, config=ON):
        self.session.agent.normalizer = TurnNormalizer(self.frontend, config)

    async def test_the_talker_and_the_backend_see_the_normalized_turn(self):
        heard = "it's m i a underscore l i three six six eight"
        self.make(
            [speak("it's mia_li_3668"), TalkerDecision("delegate", "One moment.", {}), speak("Found you, Mia.")],
            [_text_round("Mia Li, user mia_li_3668.")],
            _messages(("assistant", "What is your user ID?"), ("user", heard)),
        )
        self.enable()
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(self.conversation.messages()[-2:][0]["content"], "it's mia_li_3668")
        self.assertEqual(self.frontend.calls[1]["history"][-1]["content"], "it's mia_li_3668")
        backend_input = str(self.backend.payloads[0])
        self.assertIn("mia_li_3668", backend_input)
        self.assertNotIn("underscore", backend_input)
        completed = [fields for kind, fields in self.events if kind == "normalizer.completed"]
        self.assertEqual(completed[0]["heard"], heard)
        self.assertEqual(completed[0]["outcome"], "changed")

    async def test_a_turn_without_details_skips_the_normalizer(self):
        self.make(
            [speak("Sure, what would you like to change?")], [], _messages(("user", "I want to change my flight."))
        )
        self.enable()
        await self.session.on_turn()
        self.assertEqual(len(self.frontend.calls), 1)
        self.assertNotIn("normalizer.completed", self.kinds())

    async def test_every_turn_mode_normalizes_plain_turns_too(self):
        self.make(
            [speak("I want to change my flight."), speak("Sure.")],
            [],
            _messages(("user", "I want to change my flight.")),
        )
        self.enable(NormalizerConfig(enabled=True, every_turn=True))
        await self.session.on_turn()
        self.assertEqual(len(self.frontend.calls), 2)
        self.assertEqual(self.kinds().count("normalizer.completed"), 1)


class ConversationTests(unittest.TestCase):
    def test_the_newest_matching_caller_turn_is_replaced(self):
        class Context:
            def __init__(self, messages):
                self.items = messages

            def get_messages(self):
                return self.items

            def set_messages(self, messages):
                self.items = messages

        context = Context(
            [
                {"role": "user", "content": "two bags"},
                {"role": "user", "content": BACKEND_RESULT_PREFIX + "two bags"},
                {"role": "user", "content": "two bags"},
                {"role": "assistant", "content": "Okay."},
            ]
        )
        ContextConversation(context).replace_caller_turn("two bags", "2 bags")
        contents = [m["content"] for m in context.items][:3]
        self.assertEqual(contents, ["two bags", BACKEND_RESULT_PREFIX + "two bags", "2 bags"])
        ContextConversation(context).replace_caller_turn("missing", "x")
        self.assertEqual(len(context.items), 4)


class ConfigTests(unittest.TestCase):
    def test_the_normalizer_section_is_optional_and_validated(self):
        bare = {"frontend": {"provider": "p", "slot": "llm"}, "backend": {"provider": "p", "slot": "t"}}
        self.assertFalse(parse_config(bare).normalizer.enabled)
        config = parse_config({**bare, "normalizer": {"enabled": True, "timeout_seconds": 2}})
        self.assertEqual((config.normalizer.enabled, config.normalizer.timeout_seconds), (True, 2.0))
        with self.assertRaises(ConfigError):
            parse_config({**bare, "normalizer": {"enabled": "yes"}})
        with self.assertRaises(ConfigError):
            parse_config({**bare, "normalizer": {"model": "x"}})
