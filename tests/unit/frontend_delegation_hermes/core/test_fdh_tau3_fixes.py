# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""tau3-failure-fixes-plan.md: M1 proactive status, M3 spelling hold and wording, M2 replay, G1, G2, fingerprints."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from _fdh_backend_fakes import Harness as ControllerHarness
from _fdh_variants import variant  # noqa: E402
from _fdh_voice_fakes import DelegationHarness, profile_config, tau2_session_update, tau3_config

from examples.frontend_delegation_hermes.core.config import (
    DelegationConfigError,
    FillerDedupeSettings,
    ProactiveStatusSettings,
    ReplaySettings,
    SpellingHoldSettings,
    load_delegation_config,
)
from examples.frontend_delegation_hermes.core.engine.spelling_hold import SpellingHoldPredicate
from examples.frontend_delegation_hermes.core.engine.spoken_text import clean_for_speech
from examples.frontend_delegation_hermes.core.engine.transcript import NOT_HEARD, PARTIAL
from examples.frontend_delegation_hermes.core.engine.turn_manager import _unheard_part
from examples.frontend_delegation_hermes.core.frontend.decider import RuleDecider

FIXTURES = Path(__file__).resolve().parent / "fixtures"
AIRLINE_CODE = ("^[A-Za-z0-9]{6}$",)


def with_proactive(after_s: float = 0.3, max_per_run: int = 1):
    config = tau3_config()
    output = replace(config.output, proactive_status=ProactiveStatusSettings(True, after_s, max_per_run))
    return replace(config, output=output)


def with_hold(hold_ms: int = 400):
    config = profile_config(variant("profiles/tau3_arm_m3_spelling.yaml"))
    delegation = replace(config.delegation, spelling_hold=SpellingHoldSettings(True, hold_ms, AIRLINE_CODE))
    return replace(config, delegation=delegation)


def with_replay(ttl_s: float = 30.0):
    config = tau3_config()
    delegation = replace(config.delegation, replay=ReplaySettings(True, ttl_s))
    return replace(config, delegation=delegation, prompt_features={**config.prompt_features, "replay_intent": True})


def predicate(config=None) -> SpellingHoldPredicate:
    config = config or with_hold()
    harness = DelegationHarness(config)
    parts = harness.runtime.parts(None)  # type: ignore[arg-type]
    assert parts.spelling_hold is not None
    return parts.spelling_hold


TASK = {"delegate": True, "filler_text": "Sure, one moment.", "request": "task"}
STATUS = {"delegate": True, "filler_text": "Let me check.", "request": "status"}
ANSWER = "Your flight QV101 is confirmed for May third. Would you like to add a checked bag?"


# -- M1 ---------------------------------------------------------------------------------------------


class ProactiveStatusTests(unittest.IsolatedAsyncioTestCase):
    async def _working(self, harness: DelegationHarness) -> None:
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.wait_for("response.done")
        await harness.idle(1500)  # the filler is played out

    async def test_fires_once_per_run_after_silence_and_is_heard_as_a_status(self) -> None:
        harness = DelegationHarness(with_proactive(0.3))
        await self._working(harness)
        await asyncio.sleep(0.6)
        await harness.idle(1500)
        await asyncio.sleep(0.6)
        fired = harness.logged("status_proactive")
        self.assertEqual(len(fired), 1)
        self.assertEqual((fired[0]["run_id"], fired[0]["epoch"]), ("run_1", 1))
        self.assertGreaterEqual(fired[0]["silence_wall_s"], 0.3)
        self.assertEqual(harness.transcripts()[-1], "Still checking, one moment.")
        entries = [e for m in harness.link.of_type("history.append") for e in m["entries"]]
        self.assertIn(("status_speech", "Still checking, one moment."), [(e["kind"], e["text"]) for e in entries])
        # A new run gets a new budget.
        harness.link.deliver("state", state="WORKING", epoch=2, run_id="run_2")
        await asyncio.sleep(0.6)
        await harness.idle(1500)
        self.assertEqual(len(harness.logged("status_proactive")), 2)
        await harness.close()

    async def test_not_while_user_speaks_output_is_queued_or_idle(self) -> None:
        harness = DelegationHarness(with_proactive(0.05))
        await self._working(harness)
        manager = harness.manager
        assert manager is not None
        manager._proactive_task.cancel()  # drive the check by hand
        manager._quiet_since = manager._ctx.clock.monotonic() - 10
        manager._awaiting = 1
        manager._maybe_proactive_status()
        self.assertEqual(harness.logged("status_proactive"), [])
        manager._awaiting = 0  # speaking just ended: the timer restarted
        manager._maybe_proactive_status()
        self.assertEqual(harness.logged("status_proactive"), [])
        harness.link.deliver("state", state="IDLE", epoch=1, run_id=None)
        manager._quiet_since = manager._ctx.clock.monotonic() - 10
        manager._maybe_proactive_status()
        self.assertEqual(harness.logged("status_proactive"), [])
        await harness.close()

    async def test_dropped_when_the_run_finished_before_it_played(self) -> None:
        harness = DelegationHarness(with_proactive(0.05))
        await self._working(harness)
        manager = harness.manager
        assert manager is not None
        manager._proactive_task.cancel()
        manager._quiet_since = manager._ctx.clock.monotonic() - 10
        manager._maybe_proactive_status()
        harness.link.deliver("state", state="IDLE", epoch=1, run_id=None)  # same tick, before the output loop
        await harness.settle()
        self.assertEqual(len(harness.logged("status_proactive")), 1)
        self.assertEqual(harness.logged("status_dropped")[-1]["reason"], "run_finished")
        await harness.close()

    async def test_off_by_default(self) -> None:
        harness = DelegationHarness()
        await harness.start(tau2_session_update("mock"))
        self.assertIsNone(harness.manager._proactive_task)  # type: ignore[union-attr]
        await harness.close()


# -- M3 ---------------------------------------------------------------------------------------------


class SpellingHoldPredicateTests(unittest.TestCase):
    def test_plan_examples(self) -> None:
        check = predicate().check
        for text in ("A, A, L, T, O, underscore, A, H", "Q U I N N", "Z P four K nine"):
            with self.subTest(text=text):
                self.assertTrue(check(text).hold)
        for text in (
            "my id is sofia underscore nguyen underscore six six nine nine",
            "S O F I A underscore K I M underscore seven two eight seven .",
            "I F O Y Y Z",
            "I",
            "one",
            "five zero zero",
            "I want a flight.",
        ):
            with self.subTest(text=text):
                self.assertFalse(check(text).hold)

    def test_accumulator_holds_a_single_letter_follow_up(self) -> None:
        check = predicate().check
        self.assertFalse(check("H.").hold)
        self.assertEqual(check("H.", accumulator=True).evidence, "accumulator")

    def test_without_the_code_pattern_a_complete_code_is_held(self) -> None:
        config = with_hold()
        config = replace(config, delegation=replace(config.delegation, spelling_hold=SpellingHoldSettings(True)))
        self.assertTrue(predicate(config).check("I F O Y Y Z").hold)

    def test_airline_asr_finals_fixture(self) -> None:
        check = predicate().check
        rows = [json.loads(line) for line in (FIXTURES / "spelling_hold_airline_finals.jsonl").read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 40)
        for row in rows:
            with self.subTest(text=row["text"]):
                result = check(row["text"])
                self.assertEqual(result.hold, row["held"])
                self.assertEqual(result.evidence if result.hold else result.reason, row["why"])


class SpellingHoldTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_speech_during_the_hold_merges_and_decides_once(self) -> None:
        harness = DelegationHarness(
            with_hold(1500), transcripts=["A, A, L, T, O, underscore, A, H", "underscore one two three four"]
        )
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.speak()
        delegate = await harness.wait_sent("delegate")
        self.assertEqual(delegate["run_input"][0]["text"], "aalto_ah_1234")
        await asyncio.sleep(0.1)
        self.assertEqual(len(harness.link.of_type("delegate")), 1)
        holds = harness.logged("spelling_hold")
        self.assertEqual([h["merged"] for h in holds], [True])
        self.assertEqual(holds[0]["evidence"], "trailing_run")
        await harness.close()

    async def test_a_hold_without_speech_decides_after_hold_ms(self) -> None:
        harness = DelegationHarness(with_hold(200), transcripts=["Q U I N N"])
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        delegate = await harness.wait_sent("delegate")
        self.assertEqual(delegate["run_input"][0]["text"], "QUINN")  # spelled run joined, ASR case kept
        self.assertEqual(harness.logged("spelling_hold")[0]["merged"], False)
        await harness.close()

    async def test_off_never_holds_and_keeps_the_letters(self) -> None:
        harness = DelegationHarness(transcripts=["Q U I N N"])
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        delegate = await harness.wait_sent("delegate")
        self.assertEqual(delegate["run_input"][0]["text"], "Q U I N N")
        self.assertEqual(harness.logged("spelling_hold"), [])
        await harness.close()


class InvalidWordingTests(unittest.IsolatedAsyncioTestCase):
    async def test_first_invalid_reads_back_then_asks_to_spell_all_and_never_sends(self) -> None:
        harness = DelegationHarness(
            profile_config(variant("profiles/tau3_arm_m3_spelling.yaml")), transcripts=["my user id"]
        )
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        for index in (1, 2):
            harness.link.deliver(
                "tool.call", call_id=f"c{index}", epoch=1, name="get_user_details", arguments='{"user_id": "aalto_ah"}'
            )
            await harness.wait_sent("tool.result", count=index)
        results = harness.link.of_type("tool.result")
        self.assertTrue(all(result["local"] for result in results))
        self.assertIn("character by character (a, a, l, t, o, underscore, a, h)", results[0]["output"])
        self.assertIn("spell the whole user ID slowly", results[1]["output"])
        keys = [log["message_key"] for log in harness.logged("call_answered_locally")]
        self.assertEqual(keys, ["tool_argument_invalid_readback", "tool_argument_invalid_spell_all"])
        self.assertEqual(harness.of_type("response.function_call_arguments.done"), [])
        await harness.close()


# -- M2 ---------------------------------------------------------------------------------------------


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def _answered(self, harness: DelegationHarness, *, finish: bool = True) -> None:
        """Turn 1 delegated, run_1 answers; the answer is marked not heard after its release."""
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(1500)
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes", text=ANSWER, turn_ids=[1]
        )
        await harness.wait_for("response.done", count=2)
        manager = harness.manager
        assert manager is not None
        entry = manager.transcript.last_backend_answer()
        assert entry is not None and entry.released_mono is not None
        manager._unheard.clear()
        entry.outcome, entry.heard_text = NOT_HEARD, ""
        if finish:
            harness.link.deliver("backend_run_done", run_id="run_1", epoch=1, status="ok", history="committed",
                                 turn_ids=[1])  # fmt: skip
            harness.link.deliver("state", state="IDLE", epoch=1, run_id=None)

    async def test_status_in_idle_replays_the_unheard_answer_instead_of_delegating(self) -> None:
        harness = DelegationHarness(
            with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["book it", "did you find anything", "ok"]
        )
        await self._answered(harness)
        await harness.speak()
        await harness.wait_for("response.done", count=3)
        self.assertEqual(len(harness.link.of_type("delegate")), 1)
        replayed = harness.logged("answer_replayed")
        self.assertEqual((replayed[0]["run_id"], replayed[0]["outcome_before"]), ("run_1", "not_heard"))
        self.assertEqual(harness.transcripts()[-1], ANSWER)
        await harness.idle(3000)  # the replay is played out, heard, and then travels as context
        await harness.speak()  # the next turn flushes context
        entries = [e for m in harness.link.of_type("history.append") for e in m["entries"]]
        self.assertIn(("user", "did you find anything"), [(e["kind"], e["text"]) for e in entries])
        self.assertIn(("frontend_speech", ANSWER), [(e["kind"], e["text"]) for e in entries])
        await harness.close()

    async def test_the_frontend_is_told_only_that_the_answer_was_not_heard(self) -> None:
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness)
        manager = harness.manager
        assert manager is not None
        self.assertTrue(manager._last_answer_unheard())
        manager.transcript.last_backend_answer().replayed = True  # type: ignore[union-attr]
        self.assertFalse(manager._last_answer_unheard())
        await harness.close()

    async def _status_turn(self, harness: DelegationHarness) -> None:
        await harness.speak()
        await harness.wait_sent("delegate", count=2)

    async def test_guards_skip_and_delegate_as_today(self) -> None:
        cases = {
            "ttl_expired": lambda m, e: setattr(e, "released_mono", m._ctx.clock.monotonic() - 100),
            "already_replayed": lambda m, e: setattr(e, "replayed", True),
            "task_since_answer": lambda m, e: setattr(m._backend, "last_task_turn_id", 5),
            "answer_heard": lambda m, e: setattr(e, "outcome", "heard"),
            "never_released": lambda m, e: setattr(e, "released_mono", None),
        }
        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
                await self._answered(harness)
                manager = harness.manager
                assert manager is not None
                mutate(manager, manager.transcript.last_backend_answer())
                await self._status_turn(harness)
                self.assertEqual(harness.logged("answer_replay_skipped")[-1]["reason"], reason)
                self.assertEqual(harness.logged("answer_replayed"), [])
                await harness.close()

    async def test_partial_replays_from_the_cut_sentence_or_not_when_the_question_was_heard(self) -> None:
        self.assertEqual(_unheard_part(ANSWER, "Your flight"), ANSWER)
        self.assertEqual(
            _unheard_part(ANSWER, "Your flight QV101 is confirmed for May third. Would you"),
            "Would you like to add a checked bag?",
        )
        self.assertEqual(_unheard_part(ANSWER, ANSWER.rstrip("?")), "")
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness)
        entry = harness.manager.transcript.last_backend_answer()  # type: ignore[union-attr]
        entry.outcome, entry.heard_text = PARTIAL, ANSWER[:-1]
        await self._status_turn(harness)
        self.assertEqual(harness.logged("answer_replay_skipped")[-1]["reason"], "question_heard")
        await harness.close()

    async def test_task_request_is_never_a_replay(self) -> None:
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, TASK]), transcripts=["a", "b"])
        await self._answered(harness)
        await self._status_turn(harness)
        self.assertEqual(harness.logged("answer_replayed"), [])
        self.assertEqual(harness.logged("answer_replay_skipped"), [])
        await harness.close()

    async def test_status_before_run_done_is_a_normal_status_request(self) -> None:
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness, finish=False)  # answer arrived, backend_run_done / IDLE not yet
        await self._status_turn(harness)
        self.assertEqual(harness.link.of_type("delegate")[-1]["request"], "status")
        self.assertEqual(harness.logged("answer_replayed"), [])
        await harness.close()

    async def test_pending_steer_run_is_working_so_no_replay(self) -> None:
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness, finish=False)
        harness.link.deliver("backend_run_done", run_id="run_1", epoch=1, status="ok", history="committed",
                             turn_ids=[1])  # fmt: skip
        harness.link.deliver("state", state="WORKING", epoch=2, run_id="run_2")  # pending_steer
        await self._status_turn(harness)
        self.assertEqual(harness.link.of_type("delegate")[-1]["request"], "status")
        self.assertEqual(harness.logged("answer_replayed"), [])
        await harness.close()

    async def test_newer_run_without_a_hermes_answer_blocks_the_older_answer(self) -> None:
        harness = DelegationHarness(with_replay(), decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness)
        harness.link.deliver("state", state="WORKING", epoch=2, run_id="run_2")
        harness.link.deliver("backend_run_done", run_id="run_2", epoch=2, status="ok", history="committed",
                             turn_ids=[1])  # fmt: skip
        harness.link.deliver("state", state="IDLE", epoch=2, run_id=None)
        await self._status_turn(harness)
        self.assertEqual(harness.logged("answer_replay_skipped")[-1]["reason"], "other_run")
        await harness.close()

    async def test_off_by_default_status_in_idle_is_delegated(self) -> None:
        harness = DelegationHarness(decider=RuleDecider([TASK, STATUS]), transcripts=["a", "b"])
        await self._answered(harness)
        await self._status_turn(harness)
        self.assertEqual(harness.logged("answer_replay_skipped"), [])
        await harness.close()

    def test_replay_needs_the_frontend_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.yaml"
            base = variant("profiles/tau3_eval.yaml")
            path.write_text(f"extends: {base}\ndelegation: {{replay_unheard_answer: {{enabled: true}}}}\n")
            with self.assertRaisesRegex(DelegationConfigError, "replay_intent"):
                load_delegation_config(path)


# -- G1, G2 -----------------------------------------------------------------------------------------


class FillerDedupeTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeat_dropped_while_working_and_replaced_otherwise(self) -> None:
        config = tau3_config()
        config = replace(config, delegation=replace(config.delegation, filler_dedupe=FillerDedupeSettings(True, 3)))
        harness = DelegationHarness(config, transcripts=["check my flight", "and my bags", "and my seat"])
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.wait_for("response.done")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.speak()
        await harness.wait_sent("delegate", count=2)
        harness.link.deliver("state", state="IDLE", epoch=1, run_id=None)
        await harness.speak()
        await harness.wait_sent("delegate", count=3)
        await harness.wait_for("response.done", count=2)
        deduped = harness.logged("filler_deduped")
        self.assertEqual(
            [(d["action"], d.get("reason")) for d in deduped], [("dropped", "working"), ("replaced", None)]
        )
        self.assertEqual(harness.transcripts(), ["Sure, one moment.", "One moment."])
        await harness.close()


CONFIRMATIONS = [
    (
        "Here is the booking:\n1. **Flight QV101** from JFK to SFO on May 3, economy, $120\n"
        "2. **Flight QV102** from SFO to JFK on May 9, economy, $135\n"
        "Payment: credit_card_1029, total $255. Shall I book it?",
        [["QV101", "JFK", "May 3", "$120"], ["QV102", "May 9", "$135"], ["credit_card_1029", "$255"]],
    ),
    (
        "To modify reservation XB7TP2:\n- New flight: *QV215* on May 12\n- Fare difference: $80 to gift_card_42\n"
        "Do you confirm?",
        [["QV215", "May 12"], ["$80", "gift_card_42"]],
    ),
    (
        "Cancelling reservation ZT4M9Q: refund of $310 to credit_card_1029. Proceed?",
        [["ZT4M9Q", "$310", "credit_card_1029"]],
    ),
    (
        "Return for order #W2378156:\n- Water Bottle (item 4021937755), $48.02\n- Desk Lamp (item 5530172984), "
        "$137.94\nRefund of $185.96 to paypal_3024827. Confirm?",
        [
            ["Water Bottle", "4021937755", "$48.02"],
            ["Desk Lamp", "5530172984", "$137.94"],
            ["$185.96", "paypal_3024827"],
        ],
    ),
    (
        "Exchange in order #W4817420:\n* **T-Shirt** size M -> size L (item 1234567890), no price difference\n"
        "* **Sneakers** 1 pair, $12.50 more, paid with credit_card_3261838. Shall I proceed?",
        [["T-Shirt", "size L", "1234567890"], ["Sneakers", "$12.50", "credit_card_3261838"]],
    ),
]


class SpokenTextTests(unittest.TestCase):
    def test_every_confirmation_detail_survives_attached_to_its_own_item(self) -> None:
        for text, groups in CONFIRMATIONS:
            cleaned = clean_for_speech(text)
            sentences = [s for s in cleaned.replace("? ", "?|").replace(". ", ".|").split("|")]
            with self.subTest(text=text[:30]):
                self.assertNotIn("*", cleaned)
                self.assertNotIn("\n", cleaned)
                for group in groups:
                    self.assertTrue(
                        any(all(value in sentence for value in group) for sentence in sentences),
                        f"{group} not in one sentence of {cleaned!r}",
                    )

    def test_markdown_urls_and_identifiers(self) -> None:
        self.assertEqual(
            clean_for_speech("## Summary\n`jordan_lee_8264` is **verified**."), "Summary. jordan_lee_8264 is verified."
        )
        self.assertEqual(clean_for_speech("See https://example.com/x for more."), "See for more.")
        self.assertEqual(clean_for_speech("Plain answer."), "Plain answer.")


class CleanAnswersTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_cleaned_text_is_spoken_and_recorded(self) -> None:
        config = tau3_config()
        config = replace(config, output=replace(config.output, clean_answers=True))
        harness = DelegationHarness(config, transcripts=["options please"])
        await harness.start(tau2_session_update("airline"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.idle(1500)
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes", turn_ids=[1],
            text="Options:\n- **QV101** at $120\n- **QV102** at $135",
        )  # fmt: skip
        await harness.wait_for("response.done", count=2)
        self.assertEqual(harness.transcripts()[-1], "Options. QV101 at $120. QV102 at $135.")
        self.assertEqual(len(harness.logged("answer_cleaned")), 1)
        await harness.close()


# -- fingerprints -----------------------------------------------------------------------------------


class FingerprintTests(unittest.IsolatedAsyncioTestCase):
    async def test_voice_server_logs_features_and_the_frontend_prompt_hash(self) -> None:
        harness = DelegationHarness(profile_config(variant("profiles/tau3_arm_m3_spelling.yaml")))
        await harness.start(tau2_session_update("airline"))
        start = harness.logged("fdh_session_start")[0]
        self.assertTrue(start["features"]["spelling_hold"])
        self.assertTrue(start["features"]["spelled_runs"])
        self.assertEqual(start["invalid_message_keys"]["escalate_invalid"], "tool_argument_invalid_spell_all")
        prompt = harness.logged("frontend_prompt")
        self.assertEqual(len(prompt), 1)
        self.assertEqual(len(prompt[0]["frontend_prompt_sha256"]), 16)
        await harness.close()

    async def test_session_configured_carries_the_backend_fingerprint(self) -> None:
        h = ControllerHarness()
        await h.start()
        configured = h.of_type("session.configured")[0]
        for key in (
            "backend_features",
            "backend_catalog_sha256",
            "backend_soul_sha256",
            "backend_system_sha256",
            "backend_domain",
        ):
            self.assertIn(key, configured)
        self.assertEqual(configured["backend_features"]["spelling_v2"], False)


if __name__ == "__main__":
    unittest.main()
