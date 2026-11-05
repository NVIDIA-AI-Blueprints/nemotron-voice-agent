# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Whole calls through the repository UI bot, as its root (``pipeline.py``) runs it, on its own core."""

import asyncio
import json
from unittest.mock import patch

from pipecat.frames.frames import VADUserStoppedSpeakingFrame

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.factory import core_client
from gateways.live.server_delegation import ServerDelegation
from tests.unit.cascade_helpers import chat_adapters
from tests.unit.core_helpers import chunk, tool_fragment
from tests.unit.frontend_backend_live.root.test_root_ui_bot import BotCase
from tests.unit.frontend_helpers import ScriptedChat, decision


def heard(frontend) -> list[str]:
    """The lines the talker's context records as heard."""
    return [m["content"] for m in frontend.context.get_messages() if m["role"] == "assistant"]


class BotCallTests(BotCase):
    """The talker and the backend model each scripted; the calls start with the caller."""

    def setUp(self):
        super().setUp()
        self.welcome, self.backend = False, ScriptedChat()

        def core(settings, *, endpoints, defaults):
            return build_core(settings, endpoints=endpoints, defaults=defaults, adapters=chat_adapters(self.backend))

        for name, value in {
            "build_core": core,
            "core_client": core_client,
            "ServerDelegation": ServerDelegation,
        }.items():
            self.enterContext(patch.object(pipeline, name, value))

    def cards(self) -> dict:
        """Each delegation's newest task card, by its task id."""
        return {
            data["task_id"]: data
            for data in self.transport.outbound.messages
            if data.get("type") == "agent-task-update"
        }

    async def test_each_cafe_order_runs_the_cafes_tool_its_result_is_worded_and_its_card_shows_the_answer(self):
        first, second = asyncio.Event(), asyncio.Event()  # each answer comes once its lead-in has played
        worded = "One large latte is in your cart, five ninety-seven with tax."
        self.chat.script += [
            decision("delegate", "Let me add that."),
            decision("speak", worded),
            decision("delegate", "One moment."),
            decision("speak", "Your Matcha is ordered."),
        ]
        arguments = '{"action":"add","item_id":"lat","size":"large"}'
        self.backend.script += [
            [chunk(tool_calls=[tool_fragment(0, "call_1", "manage_cart", arguments)]), chunk(finish="tool_calls")],
            first,
            [chunk("Added one large Latte. Your total is $5.97 with tax.", finish="stop")],
            second,
            [chunk("Ordered one Matcha.", finish="stop")],
        ]

        async def during_run(frontend):
            await self.stt.propose("A large latte, please.")
            await self.until(lambda: self.tts.spoken, "the first lead-in")
            first.set()
            await self.until(lambda: len(heard(frontend)) == 2, "the first answer heard")
            await self.stt.propose("One Matcha, please.")
            await self.until(lambda: len(self.tts.spoken) == 3, "the second lead-in")
            second.set()
            await self.until(lambda: len(self.tts.spoken) == 4 and len(self.cards()) == 2, "the second answer")

        frontend = await self.run_bot(during_run)
        self.assertEqual(self.tts.spoken, ["Let me add that.", worded, "One moment.", "Your Matcha is ordered."])
        tool_result = next(m for m in self.backend.requests[1]["messages"] if m["role"] == "tool")
        self.assertEqual(json.loads(tool_result["content"])["status"], "added")  # the cafe's own cart tool ran
        relay = self.chat.requests[1]["messages"][-1]["content"]  # the talker words the backend's answer
        self.assertIn("Added one large Latte. Your total is $5.97 with tax.", relay)
        self.assertEqual(heard(frontend), self.tts.spoken)
        self.assertEqual(
            [(card["query"], card["status"], card["response"]) for card in self.cards().values()],
            [
                ("A large latte, please.", "done", "Added one large Latte. Your total is $5.97 with tax."),
                ("One Matcha, please.", "done", "Ordered one Matcha."),
            ],
        )

    async def test_a_finished_caller_turn_and_its_latency_reach_the_client_as_generic_sends_them(self):
        self.chat.script.append(decision("speak", "Earl Grey it is."))

        async def during_run(frontend):
            await self.stt.push_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2))  # the caller's speech ended
            await self.stt.propose("A pot of Earl Grey.")
            await self.until(lambda: len(self.transport.outbound.messages) >= 4, "the turn's messages")

        await self.run_bot(during_run)
        sent = self.transport.outbound.messages
        kinds = ["user-turn-finalized", "user-bot-latency", "metric-group", "latency-breakdown"]
        self.assertEqual([data["type"] for data in sent], kinds)
        self.assertEqual((sent[0]["transcript"], sent[2]["group_id"]), ("A pot of Earl Grey.", "turn-1"))
