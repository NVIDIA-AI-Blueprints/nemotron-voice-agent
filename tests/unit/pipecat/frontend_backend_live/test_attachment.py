# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107
"""Private audio-attachment replacement preserves context without replaying actions or claiming playout."""

import asyncio
import json
import unittest
from dataclasses import replace

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.attachment import (
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_ITEMS,
    export_attachment,
    restore_attachment,
    resume_presentation,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import ReadyResult
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.heard_log import Heard
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.models.decision import TalkerDecision
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.spoken import Line, LineKind
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.work_provenance import (
    Attribution,
    Binding,
    BranchSources,
)
from tests.unit.pipecat.frontend_backend_live.test_delegations import mcp_call, message
from tests.unit.pipecat.frontend_backend_live.test_work_ownership import edge, response_opened, snapshot, tagged
from tests.unit.pipecat.frontend_helpers import CallHarness, decision


class AttachmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_reconnect_restores_pending_progress_without_reoffering_old_tool_completion(self):
        proof = snapshot()
        scoped = replace(proof, item_attribution=Attribution("A", "D3", 0))
        words = "Checking dinner availability."
        update = tagged(message(words, phase="commentary", item_id="progress"), scoped)
        tool = tagged(mcp_call("app", "get_availability"), scoped)

        for previously_offered, fresh_update in ((False, False), (True, False), (True, True)):
            with self.subTest(previously_offered=previously_offered, fresh_update=fresh_update):
                old, replacement = CallHarness(), CallHarness()
                for event in (*response_opened(proof), update, tool):
                    old.voice.reader.on_delegation_event("D3", event)
                note = old.voice.worknote
                if previously_offered:
                    sources, _ = note.prepare_progress((note.branches["A"].progress_key,), 3)
                    note.progress_offered(sources)
                    note.progress_worded(sources)

                restore_attachment(replacement.voice, export_attachment(old.voice))
                reader = replacement.voice.reader
                for event in response_opened(proof):
                    reader.restore_delegation_event("D3", event)
                reader.restore_work_provenance("D3", update)
                reader.restore_progress("D3", update["item"])
                reader.restore_delegation_event("D3", tool)
                if fresh_update:
                    reader.on_delegation_event(
                        "D3", tagged(message(words, phase="commentary", item_id="fresh"), scoped)
                    )
                resume_presentation(replacement.voice)
                restored = replacement.voice.worknote
                self.assertEqual(restored.branches["A"].progress, words)
                self.assertEqual(restored.branches["A"].progress_offers, int(previously_offered))
                self.assertEqual(
                    [entry.branch_ref for entry in restored.quiet_candidates(3)],
                    ["A"] if not previously_offered or fresh_update else [],
                )

    async def test_reconnect_uses_complete_accepted_caller_wording_for_the_exact_work(self):
        old = CallHarness()
        turns = [
            {"role": "user", "content": "Check the evening flight."},
            {"role": "user", "content": "After eight, please."},
        ]
        old.context.set_messages(turns)
        plan = old.voice._routing(turns).plan
        old.voice._delegated(plan, None, TalkerDecision("delegate", "", {}))
        await asyncio.gather(*old.voice.handoffs.tasks)
        self.assertEqual(len(old.port.inputs), 1)
        original = export_attachment(old.voice)
        self.assertEqual([row["delivery"]["state"] for row in original["messages"]], ["accepted", "accepted"])

        proof = replace(
            snapshot(),
            request_sources=(BranchSources("A", ("del_1",)), BranchSources("B", ("typed_other",))),
        )
        for complete in (True, False):
            with self.subTest(complete=complete):
                fresh = CallHarness()
                saved = original if complete else {**original, "messages": original["messages"][1:]}
                restore_attachment(fresh.voice, saved)
                for event in response_opened(proof):
                    fresh.voice.reader.on_delegation_event(proof.owner_ref, event)
                resume_presentation(fresh.voice)
                context = fresh.voice.worknote.newer_requests(("D3",), ("A",))
                self.assertEqual([row["branch_ref"] for row in context], ["A"] if complete else [])
                if complete:
                    self.assertEqual(context[0]["caller_requests"], ["Check the evening flight. After eight, please."])
                self.assertEqual((fresh.chat.requests, fresh.tts.spoken, fresh.port.inputs), ([], [], []))

    async def test_spent_progress_follows_exact_work_reconstructed_on_another_carrier(self):
        def replay(voice, value):
            for event in response_opened(value):
                voice.reader.on_delegation_event(value.owner_ref, event)

        original = snapshot(bindings=[Binding("A", "D3", 0)], obligations=[edge("A")])
        for branch in ("A", "A2"):
            with self.subTest(branch):
                old, fresh = CallHarness(), CallHarness()
                replay(old.voice, original)
                note = old.voice.worknote
                note.progress("D3", "Checking dinner availability.", attribution=Attribution("A", "D3", 0))
                sources, _ = note.prepare_progress((note.branches["A"].progress_key,), 1)
                note.progress_offered(sources)
                self.assertEqual(len(sources), 1)
                note.progress_worded(sources)

                restore_attachment(fresh.voice, export_attachment(old.voice))
                replay(fresh.voice, original)
                revision = int(branch == "A")
                replay(
                    fresh.voice,
                    snapshot(
                        "D4",
                        "resp4",
                        bindings=[Binding(branch, "D4", revision), Binding("B", "D4", 0)],
                        obligations=[replace(edge("A"), branch_ref=branch), edge("B", owner="D4")],
                    ),
                )
                note = fresh.voice.worknote
                note.progress("D4", "Checking dinner availability.", attribution=Attribution(branch, "D4", revision))
                note.progress("D4", "Checking the movie times.", attribution=Attribution("B", "D4", 0))
                resume_presentation(fresh.voice)

                self.assertEqual((note.branches[branch].progress_offers, note.branches["B"].progress_offers), (1, 0))
                self.assertEqual([entry.branch_ref for entry in note.quiet_candidates(1)], ["B"])
                self.assertTrue(note.branches[branch].news)
                self.assertEqual((fresh.chat.requests, fresh.tts.spoken, fresh.port.inputs), ([], [], []))

    async def test_new_speech_does_not_reuse_a_restored_lines_identity(self):
        for pending_audio in (False, True):
            with self.subTest(pending_audio=pending_audio):
                await self.check_restored_playback(pending_audio)

    async def check_restored_playback(self, pending_audio):
        heard = "Thursday at 09:15 costs $20."
        async with CallHarness(decision("speak", heard), decision("wait")) as old:
            await old.say("Repeat the appointment.")
            await old.until(lambda: old.said() == [heard], "the completed spoken turn")
            await old.quiet()
            await old.say("Give me a moment.")
            await old.until(lambda: len(old.chat.requests) == 2, "the silent response")
            await old.quiet()
            response_number = old.voice.responses
            if pending_audio:
                pending = Line(LineKind.RESULT, "Thursday", results=True, number=99)
                old.voice.spoken.handed(pending)
                old.voice.spoken.started(pending.number)
                old.voice.spoken.ended(pending.number)
            snapshot = export_attachment(old.voice)
            self.assertTrue(any(row["content"] == heard and row["kind"] == "spoken" for row in snapshot["messages"]))
            if pending_audio:
                self.assertIn(pending.text, snapshot["uncertain"])
                self.assertNotIn(pending.number, [entry["line"] for entry in snapshot["heard"]])

        async with CallHarness(decision("speak", "Welcome back.")) as fresh:
            restore_attachment(fresh.voice, snapshot)
            self.assertEqual(fresh.voice.responses, pending.number if pending_audio else response_number)
            self.assertIn(heard, [message["content"] for message in fresh.context.get_messages()])
            await fresh.quiet()
            self.assertEqual((fresh.chat.requests, fresh.tts.spoken, fresh.port.inputs), ([], [], []))
            await fresh.say("Hello.")
            await fresh.until(lambda: fresh.said() == [heard, "Welcome back."], "new speech after restoring")
            self.assertIn(heard, [entry.text for entry in fresh.voice.heard_log.entries])
            lines = [entry.line for entry in fresh.voice.heard_log.entries if entry.kind is Heard.AGENT]
            self.assertEqual(len(lines), len(set(lines)))

    async def test_snapshot_has_no_shared_objects_pending_audio_or_automatic_actions(self):
        old = CallHarness()
        old.context.set_messages([{"role": "user", "content": "Check the reservation."}])
        old.voice.heard_log.caller("Check the reservation.")
        omitted = old.voice.heard_log.not_conveyed(
            [{"delegation_id": "D1", "item_id": "old", "branch_ref": "B1", "kind": "result"}],
            "writer_omitted",
        )
        pending_line = Line(LineKind.RESULT, "Thursday is available. Shall I book it?", number=1)
        old.voice.spoken.handed(pending_line)
        old.voice.heard_log.presented(pending_line, old.voice.spoken.presentation(pending_line))
        old.voice.worknote.confirm(old.voice.worknote.reserve("reservation"), "D1")
        raw, queued = "The proposed exchange would refund $16.63.", "Please confirm the exchange."
        old.voice.worknote.answered("D1", item_id="pending-ref")
        old.voice.ready(ReadyResult("D1", "pending-ref", raw))
        batch = old.voice._take()
        candidate = old.voice.guard_presentation(old.voice.delivered(batch, queued), batch)
        old.voice.say(candidate)
        snapshot = export_attachment(old.voice)
        json.dumps(snapshot)
        self.assertNotIn("work", snapshot)  # Canonical journal owns work-state restoration.
        self.assertIn(raw, snapshot["uncertain"])
        self.assertIn(queued, snapshot["uncertain"])
        self.assertNotIn(candidate, old.voice.spoken.presentations)
        self.assertEqual(old.voice.worknote.entries["D1"].ready, 1, "a queued candidate has not settled its item")
        self.assertIn(id(batch[0].message), old.voice.unsaid)
        snapshot["instructions"].append("extra instruction")
        self.assertNotIn("extra instruction", old.voice.instructions)
        async with CallHarness() as fresh:
            restore_attachment(fresh.voice, snapshot)
            await asyncio.sleep(0.05)
            self.assertEqual(fresh.chat.requests, [])
            self.assertEqual(fresh.tts.spoken, [])
            self.assertEqual(fresh.port.inputs, [])
            self.assertEqual(fresh.voice.worknote.open(), [])
            self.assertEqual(
                [entry.kind for entry in fresh.voice.heard_log.entries],
                [Heard.CALLER, Heard.CONTEXT, Heard.CONTEXT],
            )
            input, _ = fresh.voice.heard_log.delegation_input()
            transcript = input[0]["content"][0]["text"]
            self.assertIn('"state": "unknown", "reason": "reconnect"', transcript)
            self.assertIn(omitted, transcript)
            self.assertNotIn(raw, transcript)
            self.assertNotIn(queued, transcript)
            context = str(fresh.context.get_messages())
            self.assertIn("Some or none may have been heard", context)
            self.assertIn("Shall I book it?", context)
            self.assertIn(raw, context)
            self.assertIn(queued, context)
            self.assertEqual([item["content"] for item in fresh.voice._undecided()], ["Check the reservation."])

    async def test_oversized_or_nonfresh_snapshots_are_refused(self):
        old = CallHarness()
        old.voice.instructions = ["x" * MAX_ATTACHMENT_BYTES]
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            export_attachment(old.voice)
        old.voice.instructions = ["Speak briefly."]
        for index in range(MAX_ATTACHMENT_ITEMS + 1):
            old.voice.heard_log.caller(f"Turn {index}.")
        snapshot = export_attachment(old.voice)
        self.assertEqual(len(snapshot["heard"]), MAX_ATTACHMENT_ITEMS)
        self.assertEqual(snapshot["heard"][-1]["text"], f"Turn {MAX_ATTACHMENT_ITEMS}.")
        fresh = CallHarness()
        restore_attachment(fresh.voice, snapshot)
        carried, receipt = fresh.voice.heard_log.delegation_input()
        self.assertIn(fresh.voice.heard_log.words.omitted, carried[0]["content"][0]["text"])
        self.assertIsNotNone(fresh.voice.heard_log.omitted_before)
        fresh.voice.heard_log.sent(receipt)
        self.assertIsNone(fresh.voice.heard_log.omitted_before)
        consumed = export_attachment(fresh.voice)
        self.assertEqual(consumed["heard"], [])
        restored = CallHarness()
        restore_attachment(restored.voice, consumed)
        restored.voice.heard_log.caller("A new request.")
        self.assertGreaterEqual(restored.voice.heard_log.entries[0].seq, consumed["heard_sent"])
        old.voice.spoken.handed(Line(LineKind.REPLY, "Speaking.", number=1))
        with self.assertRaisesRegex(ValueError, "fresh voice"):
            restore_attachment(old.voice, snapshot)

    async def test_attachment_byte_limit_keeps_pending_input_and_its_tool_group_before_optional_history(self):
        def call(call_id):
            function = {"name": "delegate", "arguments": "{}"}
            return {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "function": function}]}

        def play(voice, line):
            voice.spoken.handed(line)
            voice.spoken.started(line.number)
            voice.spoken.ended(line.number)
            voice.spoken.end_played(line.number, line.text)

        old, fresh = CallHarness(), CallHarness()
        proof = snapshot()
        for event in response_opened(proof):
            old.voice.reader.on_delegation_event(proof.owner_ref, event)
        lead_in = Line(
            LineKind.LEAD_IN,
            "I will check the booking.",
            number=1,
            delegation_ids=("D3",),
            tells=("D3",),
            work_tells=frozenset(("A",)),
        )
        play(old.voice, lead_in)
        for number in range(2, 66):
            text = f"Reply {number}. " + "Receipt information. " * 100 + "🚆"
            play(old.voice, Line(LineKind.REPLY, text, number=number))
        self.assertNotIn(lead_in, old.voice.spoken.presentations)
        told = old.voice.spoken.context_text(work_refs=("A",))
        self.assertIn(lead_in.text, told)
        _, carried = old.voice.heard_log.delegation_input()
        old.voice.heard_log.sent(carried)
        playing = Line(LineKind.RESULT, "This result was still playing.", number=66)
        old.voice.spoken.handed(playing)
        old.voice.spoken.started(playing.number)
        unknown = {"role": "user", "content": "Change the booking to eight."}
        pending = {"role": "user", "content": "Actually, make it nine."}
        result = {"role": "tool", "tool_call_id": "correction", "content": '{"status":"unknown"}'}
        large = "🚆" * 60_000
        prior_context = "Retained frontend context. " * 1000
        history = [
            unknown,
            call("correction"),
            result,
            call("settled"),
            {"role": "tool", "tool_call_id": "settled", "content": large},
            {"role": "assistant", "content": "I have the restaurant details."},
            pending,
        ]
        old.context.set_messages(history)
        old.voice.decided[id(unknown)] = unknown
        old.voice.input_delivery[id(unknown)] = {"state": "unknown", "call_id": "correction"}
        old.voice.heard_log.context(prior_context)
        old.voice.bound()
        self.assertIn(history[4], old.context.get_messages())

        saved = export_attachment(old.voice)
        self.assertLessEqual(len(json.dumps(saved, ensure_ascii=False).encode()), MAX_ATTACHMENT_BYTES)
        presentations = saved["presentations"]
        self.assertLess(len(presentations), 65)
        self.assertEqual([row["line_id"] for row in presentations[-2:]], [65, playing.number])
        self.assertEqual(presentations[-2]["recorded_sentence_prefix"], text)
        self.assertTrue(presentations[-2]["complete_conveyance"])
        self.assertFalse(presentations[-1]["complete_conveyance"])
        self.assertEqual(presentations[-1]["recorded_sentence_prefix"], "")
        self.assertEqual(saved["uncertain"], [playing.text])
        restore_attachment(fresh.voice, saved)
        for event in response_opened(proof):
            fresh.voice.reader.on_delegation_event(proof.owner_ref, event)
        resume_presentation(fresh.voice)
        self.assertFalse(fresh.voice.spoken.in_flight())
        for number in range(67, 131):
            play(fresh.voice, Line(LineKind.REPLY, "An unrelated reply.", number=number))
        self.assertEqual(fresh.voice.spoken.context_text(work_refs=("A",)), told)
        self.assertIsNone(fresh.voice.spoken.context_text(work_refs=("B",)))
        self.assertEqual(fresh.voice.spoken.told_text("D3"), old.voice.spoken.told_text("D3"))
        retained = fresh.context.get_messages()
        self.assertEqual([message["content"] for message in fresh.voice._undecided()], [pending["content"]])
        self.assertIn(unknown, retained)
        self.assertIn(history[1], retained)
        self.assertIn(result, retained)
        self.assertIn(history[-2], retained)
        self.assertEqual(len([message for message in retained if message.get("tool_calls")]), 1)
        self.assertEqual([message["tool_call_id"] for message in retained if "tool_call_id" in message], ["correction"])
        self.assertIn(prior_context, [entry.text for entry in fresh.voice.heard_log.entries])
        self.assertIn(fresh.voice.heard_log.words.omitted, [message["content"] for message in retained])
        self.assertEqual((fresh.chat.requests, fresh.port.inputs, fresh.tts.spoken), ([], [], []))

        result["content"] = large
        with self.assertRaisesRegex(ValueError, "256 KiB"):
            export_attachment(old.voice)

    async def test_response_sequence_defaults_safely_for_an_older_snapshot(self):
        old = CallHarness()
        line = Line(LineKind.REPLY, "The earlier response.", number=7)
        old.voice.heard_log.presented(line, old.voice.spoken.presentation(line))
        snapshot = export_attachment(old.voice)
        snapshot["version"] = 2
        snapshot.pop("response_number")
        snapshot.pop("heard_omitted_before", None)
        for entry in snapshot["heard"]:
            for field in ("line_kind", "delegation_ids", "work_refs", "results"):
                entry.pop(field, None)
        fresh = CallHarness()
        restore_attachment(fresh.voice, snapshot)
        self.assertEqual(fresh.voice.responses, 7)
        unknown = json.loads(fresh.voice.heard_log.lines()[0][1].partition(": ")[2])
        self.assertIsNone(unknown["kind"])
        self.assertEqual((unknown["delegation_ids"], unknown["work_refs"], unknown["results"]), ([], [], []))
        self.assertEqual(unknown["recorded_sentence_prefix"], "")
        self.assertFalse(unknown["complete_conveyance"])
        for invalid in (-1, True, "7"):
            with self.subTest(invalid=invalid):
                malformed = {**snapshot, "response_number": invalid}
                with self.assertRaisesRegex(ValueError, "response number"):
                    restore_attachment(CallHarness().voice, malformed)
