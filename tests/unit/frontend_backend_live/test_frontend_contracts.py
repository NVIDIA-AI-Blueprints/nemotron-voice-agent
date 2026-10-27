# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The frontend adapter's contracts: the cascade's ports, the class contract, the tables and the fixed lines."""

import inspect
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.adapter.contract import FrontendAdapterClass, ServiceStarting
from examples.frontend_backend_live.cascade.ports import DisplaySink, Ending, ReadyResult, ResultWording
from examples.frontend_backend_live.models.decision import TalkerAction, TalkerDecision
from examples.frontend_backend_live.tables import pick
from live.ports import FrontendAdapter
from tests.unit.contracts.endings import ENDINGS

PROMPTS = Path(__file__).resolve().parents[3] / "src/examples/frontend_backend_live/prompts.yaml"
LINES = ("apology", "unclear", "lost", "refusal", "app_wait_line", "lead_in", "holding", "still_working", "intro")


def members(protocol) -> set[str]:
    return {name for name in vars(protocol) if not name.startswith("_")} | set(getattr(protocol, "__annotations__", {}))


class CascadePortTests(unittest.TestCase):
    def test_the_endings_are_the_fixtures(self):
        self.assertEqual([ending.value for ending in Ending], list(ENDINGS))

    def test_a_ready_result_names_its_item(self):
        result = ReadyResult("item_d1", "msg_1", "Your latte is ready.")
        self.assertEqual((result.delegation_id, result.item_id), ("item_d1", "msg_1"))

    def test_the_ports_hold_the_frozen_members(self):
        self.assertEqual(members(ResultWording), {"words"})
        self.assertEqual(members(DisplaySink), {"started", "progress", "shown", "answered", "spoken", "ended"})
        self.assertLessEqual(
            {"Settings", "hosts_rtvi", "from_config", "describe", "readiness"}, members(FrontendAdapterClass)
        )
        self.assertEqual(ServiceStarting("tts").name, "tts")


class TalkerDecisionTests(unittest.TestCase):
    def test_the_action_type_includes_wait(self):
        self.assertEqual(TalkerAction.__args__, ("speak", "delegate", "wait", "listen"))
        self.assertEqual(TalkerDecision("wait").action, "wait")


class TableTests(unittest.TestCase):
    def test_pick_names_what_is_known(self):
        self.assertEqual(pick({"cascade": 1}, "cascade", "frontend adapter"), 1)
        with self.assertRaisesRegex(ValueError, r"Unknown frontend adapter 'realtime'; known: cascade"):
            pick({"cascade": 1}, "realtime", "frontend adapter")


class LineTests(unittest.TestCase):
    def test_every_line_said_word_for_word_is_named_and_none_is_a_prompt(self):
        catalog = yaml.safe_load(PROMPTS.read_text(encoding="utf-8"))
        self.assertEqual(sorted(catalog["lines"]), sorted(LINES))
        self.assertNotIn("content", catalog["lines"])  # so the prompt list (/api/prompts) leaves it out
        self.assertEqual(catalog["lines"]["apology"], "Sorry, I ran into a problem with that. Could you say it again?")
        self.assertEqual(catalog["lines"]["still_working"], "I'm still working on that, thanks for your patience.")


class StubTests(unittest.TestCase):
    """The cascade adapter implements every member of the live port."""

    def test_the_adapter_has_every_frozen_member(self):
        self.assertLessEqual(members(FrontendAdapter), set(dir(CascadeAdapter)))
        for append in ("append_instructions", "append_thinking", "append_commentary"):
            with self.subTest(append):
                self.assertIn("delegation_id", inspect.signature(getattr(CascadeAdapter, append)).parameters)
        # The context meter's window is the talker's memory budget; none before the cascade is built.
        self.assertIsNone(CascadeAdapter.context_window_tokens(SimpleNamespace(cascade=None)))
        started = SimpleNamespace(cascade=SimpleNamespace(memory=SimpleNamespace(context_tokens=16_000)))
        self.assertEqual(CascadeAdapter.context_window_tokens(started), 16_000)
