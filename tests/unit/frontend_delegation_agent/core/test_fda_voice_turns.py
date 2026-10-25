# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""DelegationTurnManager over a real RealtimeSession (stub speech, rule-based frontend, fake gateway link)."""

from __future__ import annotations

import json
import unittest

from _fda_voice_fakes import DelegationHarness, FakeLink, tau2_session_update, tau3_config

from examples.frontend_delegation_agent.core.frontend.decider import RuleDecider


class DelegatedTurnTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_update_opens_and_configures_the_backend(self) -> None:
        harness = DelegationHarness()
        await harness.start(tau2_session_update("mock"))
        opened = harness.link.of_type("session.open")[0]
        self.assertEqual(opened["settings"]["simulated_delay"], {"seconds": 0.0, "where": "per_delegation"})
        configure = harness.link.of_type("session.configure")[0]
        self.assertTrue(configure["tools"])
        self.assertIn("instructions", configure)
        await harness.close()
        self.assertTrue(harness.link.closed)
        self.assertEqual(harness.link.sent[-1]["type"], "session.close")

    async def test_new_task_is_delegated_and_the_filler_is_spoken(self) -> None:
        harness = DelegationHarness(transcripts=["what is the status of order 1234"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        delegate = await harness.wait_sent("delegate")
        self.assertEqual(delegate["request"], "task")
        self.assertEqual(delegate["run_input"][0]["text"], "what is the status of order 1234")
        await harness.wait_for("response.done")
        self.assertEqual(harness.transcripts(), ["Sure, one moment."])
        decision = harness.logged("delegation_decision")[0]
        self.assertTrue(decision["delegate"])
        self.assertEqual(decision["backend_state"], "NO_SESSION")
        await harness.close()

    async def test_acknowledgement_is_not_delegated_and_creates_no_response(self) -> None:
        harness = DelegationHarness(transcripts=["okay sure"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.idle(500)
        self.assertEqual(harness.link.of_type("delegate"), [])
        self.assertEqual(harness.of_type("response.created"), [])
        await harness.wait_sent("history.append", count=2)
        kinds = [(e["kind"], e["text"]) for m in harness.link.of_type("history.append") for e in m["entries"]]
        self.assertEqual(kinds[0], ("frontend_speech", "Hi! How can I help you today?"))  # tau2's greeting, seq 0
        self.assertIn(("user", "okay sure"), kinds)
        await harness.close()

    async def test_tool_call_round_trip_and_response_create_is_an_ack(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1234"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        harness.link.deliver(
            "tool.call",
            call_id="fda_s_1_1",
            epoch=1,
            run_id="run_1",
            name="get_order",
            arguments='{"order_id": "1234"}',
        )
        done = await harness.wait_for("response.function_call_arguments.done")
        self.assertEqual(done["call_id"], "fda_s_1_1")
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "fda_s_1_1", "output": "shipped"},
            }
        )
        await harness.send({"type": "response.create"})
        result = await harness.wait_sent("tool.result")
        self.assertEqual(
            (result["call_id"], result["epoch"], result["output"], result["local"]), ("fda_s_1_1", 1, "shipped", False)
        )
        self.assertEqual([e for e in harness.of_type("error")], [])
        await harness.close()

    async def test_answer_is_spoken_heard_and_a_cut_answer_gets_a_delivery_note(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1234", "wait"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.idle(1500)  # the filler is played and heard
        harness.link.deliver(
            "answer", answer_id="a1", run_id="run_1", epoch=1, kind="hermes",
            text="Your order 1234 was shipped yesterday and should arrive on Friday morning.",
            usage={"input_tokens": 100, "output_tokens": 20}, turn_ids=[1],
        )  # fmt: skip
        await harness.wait_for("response.output_audio.delta", count=3)
        items = {e["item"]["id"] for e in harness.of_type("response.output_item.added")}
        self.assertEqual(len(items), 2)  # filler and answer are separate items
        await harness.idle(200)
        await harness.speak(400)  # barge-in while the answer plays
        outcomes = [o for o in harness.logged("playback_outcome") if o["entry_kind"] == "backend_answer"]
        self.assertEqual(outcomes[-1]["outcome"], "partial")
        notes = [
            e for m in harness.link.of_type("history.append") for e in m["entries"] if e["kind"] == "delivery_note"
        ]
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["outcome"], "partial")
        self.assertTrue(notes[0]["heard_text"])
        fillers = [
            e for m in harness.link.of_type("history.append") for e in m["entries"] if e["kind"] == "frontend_speech"
        ]
        self.assertEqual([f["outcome"] for f in fillers if f["text"] == "Sure, one moment."], ["heard"])
        await harness.close()

    async def test_update_request_while_working_asks_for_status_and_speaks_it(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1234", "what's the update so far"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(1500)
        await harness.speak()
        second = await harness.wait_sent("delegate", count=2)
        self.assertEqual(second["request"], "status")
        harness.link.deliver(
            "status", turn_id=2, summary={"task": "check order 1234", "current_tool": "get_order", "tools_done": []}
        )
        await harness.wait_for("response.output_audio_transcript.done", count=3)
        self.assertIn("I'm still working on it", harness.transcripts()[-1])
        await harness.close()

    async def test_status_is_dropped_when_the_run_already_finished(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1234", "what's the update so far"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        harness.link.deliver("state", state="WORKING", epoch=1, run_id="run_1")
        await harness.idle(1500)
        await harness.speak()
        await harness.wait_sent("delegate", count=2)
        # The run finishes before the status report is spoken.
        harness.link.deliver("state", state="IDLE", epoch=1, run_id="")
        harness.link.deliver("status", turn_id=2, summary={"current_tool": "get_order"})
        await harness.idle(1500)
        self.assertFalse(any("still working" in text for text in harness.transcripts()))
        self.assertEqual(harness.logged("status_dropped")[0]["reason"], "run_finished")
        await harness.close()

    async def test_backend_answer_is_held_while_the_user_speaks(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1234", "and order 1235 as well"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.idle(1500)
        before = len(harness.of_type("response.created"))
        # Start speaking, deliver the answer mid-utterance: nothing may be sent until the user stops.
        from _voice_fakes import pcmu_silence, pcmu_speech

        await harness.feed(pcmu_speech(400))
        harness.link.deliver(
            "answer", answer_id="a1", run_id="r", epoch=1, kind="hermes", text="It shipped.", usage={}, turn_ids=[1]
        )
        await harness.settle()
        self.assertEqual(len(harness.of_type("response.created")), before)
        await harness.feed(pcmu_silence(900))
        await harness.wait_for("response.output_audio_transcript.done", count=3)
        # The new turn's filler goes before the held answer.
        self.assertEqual(harness.transcripts()[-2:], ["Sure, one moment.", "It shipped."])
        await harness.close()


class ResponseCreateOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_mode_text_input_starts_on_response_create(self) -> None:
        harness = DelegationHarness()
        await harness.start(tau2_session_update("mock"))
        await harness.send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "check order 9"}],
                },
            }
        )
        self.assertEqual(harness.link.of_type("delegate"), [])
        await harness.send({"type": "response.create"})
        delegate = await harness.wait_sent("delegate")
        self.assertEqual(delegate["run_input"][0]["text"], "check order 9")
        await harness.close()

    async def test_manual_turn_without_speech_still_gets_a_response(self) -> None:
        harness = DelegationHarness()
        await harness.start(tau2_session_update("mock"))
        item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "thanks"}]}
        await harness.send({"type": "conversation.item.create", "item": item})
        await harness.send({"type": "response.create"})
        done = await harness.wait_for("response.done")
        self.assertEqual((done["response"]["status"], done["response"]["output"]), ("completed", []))
        self.assertEqual(harness.link.of_type("delegate"), [])
        # The next manual turn is accepted (no "active response" error).
        item["content"][0]["text"] = "check order 7"
        await harness.send({"type": "conversation.item.create", "item": item})
        await harness.send({"type": "response.create"})
        await harness.wait_sent("delegate")
        self.assertEqual(harness.of_type("error"), [])
        await harness.close()

    async def test_response_create_without_input_is_an_error(self) -> None:
        harness = DelegationHarness()
        await harness.start(tau2_session_update("mock"))
        await harness.send({"type": "response.create"})
        self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "invalid_value")
        await harness.close()


class SessionUpdateTransactionTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejected_tool_change_leaves_the_session_unchanged(self) -> None:
        harness = DelegationHarness(link=FakeLink(configure_error="tools_locked"))
        update = tau2_session_update("mock")
        await harness.start(update)
        before = harness.session.view.public()
        changed = json.loads(json.dumps(update))
        changed["session"]["tools"] = changed["session"]["tools"][:1]
        await harness.send(changed)
        error = harness.of_type("error")[-1]["error"]
        self.assertEqual(error["code"], "tools_locked")
        self.assertEqual(harness.session.view.public(), before)
        self.assertEqual(len(harness.of_type("session.updated")), 1)
        await harness.close()

    async def test_unreachable_gateway_fails_the_session_update(self) -> None:
        harness = DelegationHarness(link=FakeLink(fail_open=True))
        await harness.start()
        await harness.send(tau2_session_update("mock"))
        self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "backend_unavailable")
        self.assertEqual(harness.of_type("session.updated"), [])
        await harness.close()


class GatewayFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_fatal_gateway_error_then_delegation_speaks_an_apology(self) -> None:
        harness = DelegationHarness(transcripts=["check order 1"])
        await harness.start(tau2_session_update("mock"))
        harness.link.deliver("error", code="link_lost", message="gone", fatal=True)
        await harness.settle()
        self.assertEqual(harness.of_type("error")[-1]["error"]["code"], "backend_unavailable")
        await harness.speak()
        await harness.wait_for("response.output_audio_transcript.done")
        self.assertIn("can't reach", harness.transcripts()[-1])
        self.assertEqual(harness.link.of_type("delegate"), [])
        await harness.close()


class SilentAckTests(unittest.IsolatedAsyncioTestCase):
    async def test_speak_when_delegating_false_keeps_the_filler_silent(self) -> None:
        from dataclasses import replace

        config = tau3_config()
        config = replace(config, delegation=replace(config.delegation, speak_when_delegating=False))
        harness = DelegationHarness(config, decider=RuleDecider(), transcripts=["check order 1"])
        await harness.start(tau2_session_update("mock"))
        await harness.speak()
        await harness.wait_sent("delegate")
        await harness.idle(300)
        self.assertEqual(harness.of_type("response.created"), [])
        self.assertEqual(len(harness.logged("filler_silenced")), 1)
        await harness.close()


if __name__ == "__main__":
    unittest.main()
