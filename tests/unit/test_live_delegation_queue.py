# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Several delegations in one call: merging, corrections, the ledger, status answers and progress speech."""

import asyncio
import unittest

from examples.frontend_backend_live.config_manager.loader import ConfigError, load_live_config, parse_config
from examples.frontend_backend_live.config_manager.schema import DelegationConfig
from examples.frontend_backend_live.delegation.ledger import (
    MAX_KEPT_CLOSED,
    STATUS_LINES,
    DelegationLedger,
    asks_for_status,
)
from examples.frontend_backend_live.delegation.thinker import Thinker
from examples.frontend_backend_live.delegation.worker import BackendTask, DelegationWorker
from examples.frontend_backend_live.models.decision import TalkerDecision
from examples.frontend_backend_live.prompts import get_prompt_set
from examples.frontend_backend_live.tool_calling.cafe.schemas import CAFE_TOOLS
from tests.unit.test_frontend_backend_live import (
    FAST,
    RecordingTools,
    ScriptedBackend,
    _call_round,
    _config,
    _messages,
    _SessionCase,
    _settle,
    _text_round,
)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


# ------------------------------------------------------------------------------------- ledger
class LedgerTests(unittest.TestCase):
    def test_nothing_open_describes_nothing(self):
        ledger = DelegationLedger()
        self.assertEqual(ledger.describe(), "")
        ledger.add("a", "Add a latte.")
        ledger.answered("a")
        self.assertEqual(ledger.describe(), "")

    def test_the_open_work_is_listed_with_its_state_and_step(self):
        clock = Clock()
        ledger = DelegationLedger(clock)
        ledger.add("a", "Look up my reservation.")
        ledger.add("b", "x" * 300)
        ledger.start("a")
        ledger.step("a", ["search_direct_flight", "get_user_details"])
        clock.now += 14
        text = ledger.describe()
        self.assertIn('1. running for 14 s, now using search direct flight, get user details: "Look up', text)
        self.assertIn("2. waiting to start", text)
        self.assertNotIn("x" * 200, text)

    def test_the_newer_open_requests_are_the_ones_made_after_a_delegation(self):
        ledger = DelegationLedger()
        ledger.add("a", "Book Friday.")
        ledger.add("b", "What teas do you have?")
        ledger.add("c", "And the oat milk price?")
        self.assertEqual([e.id for e in ledger.open_after("a")], ["b", "c"])
        self.assertEqual(ledger.open_after("c"), [])
        ledger.answered("b")
        self.assertEqual([e.id for e in ledger.open_after("a")], ["c"])
        self.assertEqual(ledger.open_after("missing"), [])
        self.assertEqual(ledger.newest_open().id, "c")

    def test_a_request_that_joins_a_waiting_one_extends_its_text(self):
        ledger = DelegationLedger()
        ledger.add("a", "My ID is ivan")
        ledger.merge("a", "eight five five five")
        self.assertEqual(ledger.entries["a"].request, "My ID is ivan / eight five five five")

    def test_closed_entries_are_not_kept_forever(self):
        ledger = DelegationLedger()
        for number in range(MAX_KEPT_CLOSED + 10):
            ledger.add(str(number), "r")
            ledger.answered(str(number))
        self.assertLessEqual(len(ledger.entries), MAX_KEPT_CLOSED)

    def test_the_caller_has_waited_since_the_oldest_open_request(self):
        clock = Clock()
        ledger = DelegationLedger(clock)
        self.assertIsNone(ledger.oldest_open_since())
        ledger.add("a", "r")
        clock.now += 5
        ledger.add("b", "r")
        self.assertEqual(ledger.oldest_open_since(), 100.0)

    def test_short_progress_questions_are_status_requests(self):
        for text in ("Any update?", "Hello?", "Are you still there?", "How much longer?", "Did you find it?"):
            with self.subTest(text=text):
                self.assertTrue(asks_for_status(text))

    def test_other_turns_are_not_status_requests(self):
        for text in (
            "Yes, please.",
            "My user ID is ivan underscore rossi",
            "Actually make it Saturday",
            "Hello, I would like to book a flight to Denver tomorrow morning",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(asks_for_status(text))


# ------------------------------------------------------------------------------- the decision


class ConfigTests(unittest.TestCase):
    BARE = {"frontend": {"provider": "p", "slot": "llm"}, "backend": {"provider": "p", "slot": "t"}}

    def test_the_bundled_config_enables_the_behaviour(self):
        delegation = load_live_config().delegation
        self.assertTrue(delegation.merge_pending and delegation.fold_before_acting)
        self.assertTrue(delegation.status_from_ledger and delegation.progress_speech)
        self.assertGreater(delegation.progress_after_seconds, 0)

    def test_each_setting_can_be_changed(self):
        config = parse_config({**self.BARE, "delegation": {"merge_pending": False, "progress_after_seconds": 3}})
        self.assertFalse(config.delegation.merge_pending)
        self.assertEqual(config.delegation.progress_after_seconds, 3.0)

    def test_the_new_settings_load_and_validate(self):
        config = parse_config(
            {
                **self.BARE,
                "delegation": {
                    "fold_before_acting": False,
                    "progress_max_per_delegation": 3,
                },
            }
        )
        self.assertFalse(config.delegation.fold_before_acting)
        self.assertEqual(config.delegation.progress_max_per_delegation, 3)
        with self.assertRaisesRegex(ConfigError, "positive"):
            parse_config({**self.BARE, "delegation": {"progress_after_seconds": 0}})

    def test_the_config_path_can_be_overridden(self):
        import os
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.yaml"
            path.write_text(
                "frontend: {provider: p, slot: llm}\nbackend: {provider: p, slot: t}\n"
                "delegation: {fold_before_acting: false}\n"
            )
            with patch.dict(os.environ, {"LIVE_CONFIG_PATH": str(path)}):
                self.assertFalse(load_live_config().delegation.fold_before_acting)

    def test_a_typo_or_a_bad_value_is_an_error(self):
        with self.assertRaisesRegex(ConfigError, "unknown keys"):
            parse_config({**self.BARE, "delegation": {"merge": True}})
        with self.assertRaisesRegex(ConfigError, "true or false"):
            parse_config({**self.BARE, "delegation": {"fold_before_acting": "yes"}})
        with self.assertRaisesRegex(ConfigError, "positive"):
            parse_config({**self.BARE, "delegation": {"progress_after_seconds": 0}})


# ------------------------------------------------------------------------------------ the worker
class SlowTools(RecordingTools):
    """Tool execution that waits until released, so a task stays in its tool step."""

    def __init__(self):
        super().__init__('{"ok": true}')
        self.gate = asyncio.Event()

    async def execute(self, call_id, name, arguments):
        await self.gate.wait()
        return await super().execute(call_id, name, arguments)


class WorkerQueueTests(unittest.IsolatedAsyncioTestCase):
    def worker(self, rounds, *, tools=None, policy=None, on_answer=None):
        self.backend = ScriptedBackend(rounds)
        self.answers: list[str] = []
        self.abandoned: list[str] = []

        async def record(text, task):
            self.answers.append(text)

        worker = DelegationWorker(
            Thinker(self.backend, get_prompt_set("v1")),
            BackendTask("test/model", "Do the task.", CAFE_TOOLS),
            tools or RecordingTools(),
            FAST,
            on_answer=on_answer or record,
            on_abandoned=self.abandoned.append,
            policy=policy,
        )
        worker.start()
        self.addAsyncCleanup(worker.close)
        return worker

    @staticmethod
    def user(text):
        return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]

    @staticmethod
    def user_texts(items):
        return [i["content"][0]["text"] for i in items if i.get("role") == "user"]

    async def answered(self, count):
        async with asyncio.timeout(2):
            while len(self.answers) < count:
                await asyncio.sleep(0.005)

    async def test_a_request_that_arrives_while_another_waits_joins_it(self):
        worker = self.worker(
            [_text_round("first"), _text_round("both")], policy=DelegationConfig(fold_before_acting=False)
        )
        self.backend.gate = asyncio.Event()
        await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)  # a is running
        waiting = await worker.submit(self.user("b"))
        joined = await worker.submit(self.user("c"))
        self.assertEqual(joined, waiting)
        self.assertEqual(len(worker.pending), 1)
        self.backend.gate.set()
        await self.answered(2)
        self.assertEqual(self.answers, ["first", "both"])
        self.assertEqual(self.user_texts(self.backend.payloads[1]["input"])[-2:], ["b", "c"])

    async def test_a_correction_never_cancels_a_task_that_already_acted(self):
        tools = SlowTools()
        worker = self.worker(
            [_call_round("manage_cart", {}), _text_round("booked friday"), _text_round("moved to saturday")],
            tools=tools,
        )
        first = await worker.submit(self.user("book friday"))
        await asyncio.sleep(0.05)  # a has emitted its call and is running the tool
        second = await worker.submit(self.user("make it saturday"))
        tools.gate.set()
        await self.answered(2)
        self.assertNotEqual(first, second)
        self.assertEqual(self.abandoned, [])
        self.assertEqual(self.answers, ["booked friday", "moved to saturday"])
        self.assertEqual(self.user_texts(self.backend.payloads[-1]["input"])[-1], "make it saturday")

    async def test_a_correction_never_interrupts_an_answer_being_delivered(self):
        release = asyncio.Event()
        delivered: list[str] = []

        async def slow(text, task):
            await release.wait()
            delivered.append(text)

        worker = self.worker([_text_round("found it"), _text_round("updated")], on_answer=slow)
        await worker.submit(self.user("look it up"))
        await asyncio.sleep(0.05)  # a is handing over its answer
        await worker.submit(self.user("actually the other one"))
        release.set()
        async with asyncio.timeout(2):
            while len(delivered) < 2:
                await asyncio.sleep(0.005)
        self.assertEqual(delivered, ["found it", "updated"])
        self.assertEqual(self.abandoned, [])

    async def test_a_request_arriving_before_the_task_acts_restarts_it_once_with_both(self):
        worker = self.worker([_text_round("both answered")])
        self.backend.gate = asyncio.Event()
        first = await worker.submit(self.user("book friday"))
        await asyncio.sleep(0.02)
        second = await worker.submit(self.user("and add a bag"))  # nothing has started yet
        self.assertNotEqual(first, second)
        self.assertEqual(worker.last_replaced, first)
        self.backend.gate.set()
        await self.answered(1)
        self.assertEqual(self.abandoned, [first])
        self.assertEqual(self.user_texts(self.backend.payloads[-1]["input"]), ["book friday", "and add a bag"])

    async def test_a_task_is_restarted_only_once(self):
        worker = self.worker([_text_round("one"), _text_round("two")])
        self.backend.gate = asyncio.Event()
        await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)
        folded = await worker.submit(self.user("b"))
        await asyncio.sleep(0.02)  # the restarted task is running
        later = await worker.submit(self.user("c"))
        self.assertNotEqual(folded, later)
        self.assertIsNone(worker.last_replaced)  # the restarted task is left to finish
        self.assertEqual(len(worker.pending), 1)
        self.backend.gate.set()
        await self.answered(2)
        self.assertEqual(len(self.abandoned), 1)  # only the first run was replaced

    async def test_a_request_never_restarts_a_task_that_acted(self):
        tools = SlowTools()
        worker = self.worker([_call_round("manage_cart", {}), _text_round("done"), _text_round("next")], tools=tools)
        await worker.submit(self.user("add a latte"))
        await asyncio.sleep(0.05)
        await worker.submit(self.user("and a muffin"))
        self.assertIsNone(worker.last_replaced)
        tools.gate.set()
        await self.answered(2)
        self.assertEqual(self.abandoned, [])

    async def test_folding_can_be_turned_off(self):
        worker = self.worker(
            [_text_round("one"), _text_round("two")], policy=DelegationConfig(fold_before_acting=False)
        )
        self.backend.gate = asyncio.Event()
        await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)
        await worker.submit(self.user("b"))
        self.assertIsNone(worker.last_replaced)
        self.backend.gate.set()
        await self.answered(2)
        self.assertEqual(self.abandoned, [])

    async def test_typed_input_is_never_merged(self):
        worker = self.worker([_text_round("one"), _text_round("two")])
        self.backend.gate = asyncio.Event()
        await worker.submit(self.user("a"))
        await asyncio.sleep(0.02)
        first = await worker.submit(self.user("b"), coalesce=False)
        second = await worker.submit(self.user("c"), coalesce=False)
        self.assertNotEqual(first, second)
        self.assertEqual(len(worker.pending), 2)


# ---------------------------------------------------------------------------- the whole session
class DelegationSessionTests(_SessionCase):
    async def test_the_talker_sees_the_work_in_progress(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Okay.")],
            [_text_round("done")],
            _messages(("user", "Look up my reservation.")),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "And my flights next week."})
        await self.session.on_turn()
        first, second = (call["instructions"] for call in self.frontend.calls)
        self.assertNotIn("Work in progress", first)
        self.assertIn("Work in progress", second)
        self.assertIn("running for", second)
        self.assertIn("Look up my reservation.", second)
        self.backend.gate.set()

    async def test_a_status_question_is_answered_without_a_model_call_or_new_work(self):
        self.make([TalkerDecision("delegate", "")], [_text_round("done")], _messages(("user", "Look it up.")))
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "Any update?"})
        await self.session.on_turn()
        self.assertEqual(self.spoken, [STATUS_LINES[0]])
        self.assertEqual(len(self.frontend.calls), 1)  # only the first turn reached the talker
        self.assertEqual(self.session.worker.pending, [])
        self.assertIn("status.answered", self.kinds())
        self.backend.gate.set()

    async def test_a_status_question_with_nothing_open_goes_to_the_talker(self):
        self.make([TalkerDecision("speak", "Hi, how can I help?")], [], _messages(("user", "Hello?")))
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["Hi, how can I help?"])

    async def test_a_status_answer_can_be_turned_off(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Still looking.")],
            [_text_round("done")],
            _messages(("user", "Look it up.")),
            config=_config(delegation=DelegationConfig(status_from_ledger=False, progress_speech=False)),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "Any update?"})
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["Still looking."])
        self.backend.gate.set()

    async def test_a_request_before_any_tool_call_replaces_the_run_and_only_one_answer_is_spoken(self):
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "It is booked for Saturday."),
            ],
            [_text_round("Booked for Saturday.")],
            _messages(("user", "Book me on Friday.")),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "Actually make it Saturday."})
        await self.session.on_turn()
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["It is booked for Saturday."])
        self.assertIn("backend.cancelled", self.kinds())
        self.assertEqual(len(self.backend.payloads), 2)  # the replaced run's request, then the combined one
        self.assertEqual(self.session.coordinator.ledger.open(), [])

    async def test_an_answer_is_phrased_knowing_the_newer_requests_still_in_progress(self):
        tools = SlowTools()
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "I found the Friday flight; I'm updating it for Saturday."),
                TalkerDecision("speak", "It is now Saturday."),
            ],
            [_call_round("get_menu", {}), _text_round("Friday flight found."), _text_round("Moved to Saturday.")],
            _messages(("user", "Book me on Friday.")),
            tools=tools,
        )
        await self.session.on_turn()
        await asyncio.sleep(0.05)  # the first request is in its tool step: it cannot be restarted
        self.conversation.add({"role": "user", "content": "Actually make it Saturday."})
        await self.session.on_turn()
        tools.gate.set()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(
            self.spoken, ["I found the Friday flight; I'm updating it for Saturday.", "It is now Saturday."]
        )
        first = self.frontend.calls[2]["history"][-1]["content"]  # phrasing the first answer
        self.assertIn("Actually make it Saturday.", first)
        self.assertIn("otherwise report this result normally", first)
        self.assertNotIn("since asked", self.frontend.calls[3]["history"][-1]["content"])  # nothing newer remains
        self.assertIn("answer.with_newer_requests", self.kinds())
        self.assertNotIn("backend.cancelled", self.kinds())

    async def test_unrelated_requests_each_get_their_own_spoken_answer(self):
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "Your flight is on Friday."),
                TalkerDecision("speak", "Your balance is fifty dollars."),
            ],
            [_text_round("Flight Friday."), _text_round("Balance $50.")],
            _messages(("user", "When is my flight?")),
            config=_config(
                delegation=DelegationConfig(merge_pending=False, fold_before_acting=False, progress_speech=False)
            ),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "What is my balance?"})
        await self.session.on_turn()
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(self.spoken, ["Your flight is on Friday.", "Your balance is fifty dollars."])
        self.assertNotIn("answer.outdated", self.kinds())

    async def test_a_second_request_before_any_tool_call_is_answered_together_with_the_first(self):
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "The latte and the muffin are added."),
            ],
            [_text_round("Latte and muffin added.")],
            _messages(("user", "Add a latte.")),
            config=_config(delegation=DelegationConfig(progress_speech=False)),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        self.conversation.add({"role": "user", "content": "And a muffin."})
        await self.session.on_turn()
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 1)
        self.assertEqual(self.spoken, ["The latte and the muffin are added."])
        self.assertIn("backend.cancelled", self.kinds())
        entry = next(e for e in self.session.coordinator.ledger.entries.values() if e.state == "answered")
        self.assertIn(" / ", entry.request)  # the ledger entry covers both requests

    async def test_requests_that_wait_together_share_one_run_and_one_answer(self):
        self.make(
            [
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("delegate", ""),
                TalkerDecision("speak", "Done with the first."),
                TalkerDecision("speak", "Both are done."),
            ],
            [_text_round("first"), _text_round("both")],
            _messages(("user", "Check my user ID.")),
            config=_config(delegation=DelegationConfig(fold_before_acting=False, progress_speech=False)),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.02)
        for text in ("And my flight.", "And my bags."):
            self.conversation.add({"role": "user", "content": text})
            await self.session.on_turn()
        self.assertEqual(len(self.session.worker.pending), 1)
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 2)
        self.assertEqual(len(self.backend.payloads), 2)
        self.assertEqual(self.spoken, ["Done with the first.", "Both are done."])

    async def test_long_silent_work_gets_a_status_line_and_then_the_answer(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Here is your answer.")],
            [_text_round("done")],
            _messages(("user", "Look it up.")),
            config=_config(
                delegation=DelegationConfig(
                    progress_after_seconds=0.1, progress_interval_seconds=0.1, progress_max_per_delegation=3
                )
            ),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 2, timeout=3)
        self.assertEqual(self.spoken[:2], [STATUS_LINES[0], STATUS_LINES[1]])
        self.backend.gate.set()
        await _settle(self.session, self.spoken, 3)
        self.assertEqual(self.spoken[-1], "Here is your answer.")
        count = len(self.spoken)
        await asyncio.sleep(0.4)
        self.assertEqual(len(self.spoken), count)  # nothing open: no more progress lines

    async def test_progress_is_not_spoken_while_the_caller_is_talking(self):
        self.make(
            [TalkerDecision("delegate", "")],
            [_text_round("done")],
            _messages(("user", "Look it up.")),
            config=_config(delegation=DelegationConfig(progress_after_seconds=0.1, progress_interval_seconds=0.1)),
        )
        self.backend.gate = asyncio.Event()
        self.session.set_user_speaking(True)
        await self.session.on_turn()
        await asyncio.sleep(0.5)
        self.assertEqual(self.spoken, [])
        self.backend.gate.set()
        self.session.set_user_speaking(False)

    async def test_progress_can_be_turned_off(self):
        self.make(
            [TalkerDecision("delegate", "")],
            [_text_round("done")],
            _messages(("user", "Look it up.")),
            config=_config(delegation=DelegationConfig(progress_speech=False, progress_after_seconds=0.1)),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.5)
        self.assertEqual(self.spoken, [])
        self.backend.gate.set()


class ProgressCapTests(_SessionCase):
    async def test_a_delegation_gets_at_most_the_configured_number_of_status_lines(self):
        self.make(
            [TalkerDecision("delegate", "")],
            [_text_round("done")],
            _messages(("user", "Look it up.")),
            config=_config(
                delegation=DelegationConfig(
                    progress_after_seconds=0.1, progress_interval_seconds=0.1, progress_max_per_delegation=1
                )
            ),
        )
        self.backend.gate = asyncio.Event()
        await self.session.on_turn()
        await asyncio.sleep(0.8)
        self.assertEqual(self.spoken, [STATUS_LINES[0]])
        self.backend.gate.set()


# ----------------------------------------------------------------- application input and settings
class ApplicationContextTests(_SessionCase):
    async def test_silent_context_reaches_the_next_delegation_once(self):
        self.make(
            [TalkerDecision("delegate", ""), TalkerDecision("speak", "Done."), TalkerDecision("delegate", "")],
            [_text_round("done"), _text_round("again")],
            _messages(("user", "Look up my order.")),
            config=_config(delegation=DelegationConfig(progress_speech=False)),
        )
        self.session.on_thinking("The caller is a gold member.")
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 1)
        first = self.backend.payloads[0]["input"][0]["content"][0]["text"]
        self.assertIn("Application context (reference data):", first)
        self.assertIn("The caller is a gold member.", first)
        self.conversation.add({"role": "user", "content": "And another thing."})
        await self.session.on_turn()
        await _settle(self.session, self.spoken, 2)
        second = self.backend.payloads[1]["input"][-1]["content"][0]["text"]
        self.assertNotIn("gold member", second)  # sent once; the backend keeps it in its own history

    async def test_thinking_is_still_visible_to_the_talker_and_is_not_spoken(self):
        self.make([TalkerDecision("speak", "Okay.")], [], _messages(("user", "Hi.")))
        self.session.on_thinking("The cart starts empty.")
        await self.session.on_turn()
        self.assertEqual(self.spoken, ["Okay."])
        self.assertIn("The cart starts empty.", str(self.frontend.calls[0]["history"]))

    async def test_an_instruction_interrupts_and_is_decided_on_at_once(self):
        self.make([TalkerDecision("speak", "Understood, formal tone.")], [], _messages(("user", "Hi.")))
        interrupted = []

        async def interrupt():
            interrupted.append(True)

        self.session.application.interrupt = interrupt
        await self.session.on_instruction("Be formal.")
        self.assertEqual((interrupted, self.spoken), ([True], ["Understood, formal tone."]))
        self.assertEqual(self.conversation.messages()[-1], {"role": "developer", "content": "Be formal."})


class SettingsUpdateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_session_update_changes_the_backend_settings_from_the_next_round(self):
        from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator
        from live.protocol import SessionConfig

        backend = ScriptedBackend([])
        coordinator = DelegationCoordinator(
            Thinker(backend, get_prompt_set("v1")),
            _config(),
            backend_task=BackendTask("old/model", "Old.", CAFE_TOOLS),
            tools=RecordingTools(),
            sink=type("Sink", (), {"on_result": staticmethod(lambda *a, **k: None)})(),
        )
        self.addAsyncCleanup(coordinator.close)
        config = SessionConfig.model_validate(
            {
                "model": "live-1",
                "delegation": {"type": "responses", "responses": {"model": "new/model", "instructions": "New."}},
            }
        )
        coordinator.update_settings(config, "fallback/model", "default instructions")
        self.assertEqual((coordinator.worker.task.model, coordinator.worker.task.instructions), ("new/model", "New."))

    async def test_a_client_mode_session_has_nothing_to_update(self):
        from examples.frontend_backend_live.delegation.coordinator import DelegationCoordinator
        from examples.frontend_backend_live.delegation.tasks import CLIENT
        from live.protocol import SessionConfig

        coordinator = DelegationCoordinator(None, _config(), backend_task=None, tools=None, sink=object(), mode=CLIENT)
        coordinator.update_settings(SessionConfig.model_validate({"model": "live-1"}), "m", "i")
        self.assertIsNone(coordinator.worker)
