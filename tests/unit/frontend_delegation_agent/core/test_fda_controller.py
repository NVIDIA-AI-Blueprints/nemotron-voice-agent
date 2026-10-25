# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""BackendController: epochs and settling, outcome table, steering, delay, watchdog, recovery (plan section 5)."""

from __future__ import annotations

import asyncio
import unittest

from _fda_backend_fakes import Harness, RecordingWorker, history_after

from examples.frontend_delegation_agent.core.delegation import protocol as proto


class ControllerTest(unittest.IsolatedAsyncioTestCase):
    async def test_open_and_configure(self) -> None:
        h = Harness()
        await h.start()
        self.assertEqual([m["type"] for m in h.emitted[:3]], ["worker", "session.ready", "session.configured"])
        configure = h.worker.of_type("configure")[0]
        self.assertEqual([t["name"] for t in configure["tools"]], ["get_order"])
        self.assertIn("<policy>\nPOLICY\n</policy>", configure["instructions"])

    async def test_start_run_carries_context_block_and_task_id(self) -> None:
        h = Harness()
        await h.start()
        h.controller.append(
            [{"seq": 1, "origin": "frontend", "kind": "frontend_speech", "text": "Hi!", "outcome": "heard"}]
        )
        await h.delegate(1, "status of order 1234", seq=2)
        run = h.worker.of_type("run")[0]
        self.assertEqual((run["epoch"], run["task_id"], run["conversation_history"]), (1, "s1:1", []))
        self.assertTrue(run["user_message"].startswith('[the voice frontend told the user] "Hi!"'))
        self.assertTrue(run["user_message"].endswith("status of order 1234"))
        self.assertEqual(h.controller.state, proto.WORKING)
        self.assertEqual(h.of_type("action")[-1]["kind"], "start")

    async def test_delegation_racing_run_end_is_classified_against_the_settled_state(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1")
        # The outcome arrives, and a delegation is handled before the scheduled settle task runs.
        h.worker.outcome(1, "ok", final_response="Done.", messages=history_after("order 1", "Done."))
        await h.delegate(2, "and order 2")
        self.assertEqual(h.worker.of_type("steer"), [])
        runs = h.worker.of_type("run")
        self.assertEqual([r["epoch"] for r in runs], [1, 2])
        self.assertEqual(len(runs[1]["conversation_history"]), 2)
        self.assertEqual(h.of_type("action")[-1]["kind"], "continue")
        await h.settle()
        self.assertEqual(len(h.of_type("backend_run_done")), 1)  # settle is idempotent

    async def test_stale_epoch_outcomes_and_tool_calls_are_dropped(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "x")
        h.worker.outcome(7, "ok", final_response="nope", messages=[])
        h.worker.reply({"type": "tool.call", "call_id": "c9", "epoch": 3, "name": "get_order", "arguments": "{}"})
        await h.settle()
        self.assertEqual(h.of_type("backend_run_done"), [])
        self.assertEqual(h.of_type("tool.call"), [])
        self.assertEqual(h.controller.state, proto.WORKING)

    async def test_pending_steer_reruns_without_idle_and_moves_its_context(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1", seq=1)
        await h.delegate(2, "also order 2", seq=2)
        steer = h.worker.of_type("steer")[0]
        self.assertIn("also order 2", steer["text"])
        messages = history_after("order 1", "Order 1 is shipped.")
        h.worker.outcome(1, "ok", final_response="Order 1 is shipped.", messages=messages, pending_steer=steer["text"])
        await h.settle()
        runs = h.worker.of_type("run")
        self.assertEqual(len(runs), 2)
        self.assertEqual((runs[1]["epoch"], runs[1]["user_message"]), (2, steer["text"]))
        states = [m["state"] for m in h.of_type("state")]
        self.assertNotIn(proto.IDLE, states)
        self.assertEqual(h.of_type("action")[-1]["kind"], "pending_steer")
        self.assertEqual(h.controller.context.committed_upto, 1)  # the steer's words are still in flight
        h.worker.outcome(
            2, "ok", final_response="Order 2 too.", messages=history_after(steer["text"], "Order 2 too.", messages)
        )
        await h.settle()
        self.assertEqual(h.controller.context.committed_upto, 2)
        self.assertEqual(h.controller.state, proto.IDLE)

    async def test_delivered_steer_is_not_rerun(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1")
        await h.delegate(2, "also order 2")
        steer = h.worker.of_type("steer")[0]["text"]
        messages = [
            *history_after("order 1", "")[:1],
            {"role": "user", "content": f"[OUT-OF-BAND]\n{steer}"},
            {"role": "assistant", "content": "Both done."},
        ]
        h.worker.outcome(1, "ok", final_response="Both done.", messages=messages, pending_steer="")
        await h.settle()
        self.assertEqual(len(h.worker.of_type("run")), 1)
        self.assertEqual(h.of_type("answer")[-1]["turn_ids"], [1, 2])

    async def test_rejected_steer_after_run_end_starts_a_new_run(self) -> None:
        h = Harness(worker_kwargs={"steer_accepted": False})
        await h.start()
        await h.delegate(1, "order 1")
        loop = asyncio.get_running_loop()
        loop.call_later(
            0.05, lambda: h.worker.outcome(1, "ok", final_response="Done.", messages=history_after("order 1", "Done."))
        )
        await h.delegate(2, "and order 2", seq=20)
        runs = h.worker.of_type("run")
        self.assertEqual(len(runs), 2)
        self.assertTrue(runs[1]["user_message"].endswith("and order 2"))

    async def test_status_does_not_touch_the_run(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1")
        h.worker.reply(
            {"type": "tool.call", "call_id": "c1", "epoch": 1, "name": "get_order", "arguments": '{"order_id": "1"}'}
        )
        await h.delegate(2, "what's the update?", request="status")
        status = h.of_type("status")[-1]
        self.assertEqual(status["summary"]["outstanding"], ["get_order"])
        self.assertEqual(status["summary"]["task"], "order 1")
        self.assertEqual(h.worker.of_type("steer"), [])
        self.assertEqual(len(h.worker.of_type("run")), 1)
        # The question reaches Hermes later, as context.
        self.assertEqual([e.text for e in h.controller.context.queued()], ["what's the update?"])

    async def test_rollback_requeues_context_and_notes_side_effects(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "order 1", seq=1)
        h.worker.outcome(1, "ok", final_response="Done.", messages=history_after("order 1", "Done."))
        await h.settle()
        settled = list(h.controller.history)
        h.controller.append([{"seq": 2, "origin": "user", "kind": "user", "text": "thanks", "outcome": "heard"}])
        await h.delegate(2, "cancel order 5", seq=3)
        h.worker.reply(
            {"type": "tool.call", "call_id": "c1", "epoch": 2, "name": "cancel_order", "arguments": '{"id": 5}'}
        )
        h.controller.tool_result("c1", '{"cancelled": true}', 2, local=False)
        h.worker.outcome(2, "error", error="boom")
        await h.settle()
        self.assertEqual(h.controller.history, settled)
        done = h.of_type("backend_run_done")[-1]
        self.assertEqual((done["status"], done["history"]), ("error", "rolled_back"))
        self.assertEqual(h.of_type("answer")[-1]["kind"], "apology")
        kinds = [e.kind for e in h.controller.context.queued()]
        self.assertEqual(kinds, ["side_effect_note", "user", "requeued_request"])
        await h.delegate(3, "hello?", seq=4)
        message = h.worker.of_type("run")[-1]["user_message"]
        self.assertIn("already ran: cancel_order", message)
        self.assertIn('[earlier request, not completed yet] user: "cancel order 5"', message)
        self.assertEqual(h.controller.state, proto.WORKING)

    async def test_interrupted_run_with_messages_is_committed(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "x", seq=1)
        messages = history_after("x", "")
        h.worker.outcome(1, "interrupted", messages=messages)
        await h.settle()
        self.assertEqual(h.controller.history, messages)
        self.assertEqual(h.of_type("backend_run_done")[-1]["history"], "committed")
        self.assertEqual(h.of_type("answer")[-1]["kind"], "apology")

    async def test_simulated_delay_absorbs_steers(self) -> None:
        h = Harness()
        await h.start(delay_s=0.2)
        await h.delegate(1, "order 1")
        await h.delegate(2, "and order 2")
        self.assertEqual(h.worker.of_type("run"), [])
        self.assertEqual(h.of_type("action")[-1]["kind"], "queued_in_delay")
        await asyncio.sleep(0.3)
        run = h.worker.of_type("run")[0]
        self.assertTrue(run["user_message"].endswith("order 1\nand order 2"))
        self.assertEqual(h.worker.of_type("steer"), [])

    async def test_usage_is_passed_as_per_run_delta(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "x")
        h.worker.outcome(
            1,
            "ok",
            final_response="y",
            messages=history_after("x", "y"),
            usage_delta={"input_tokens": 12, "output_tokens": 3, "reasoning_tokens": 1},
        )
        await h.settle()
        self.assertEqual(
            h.of_type("answer")[-1]["usage"], {"input_tokens": 12, "output_tokens": 3, "reasoning_tokens": 1}
        )

    async def test_watchdog_interrupts_then_kills_and_respawns(self) -> None:
        h = Harness(run_hard_deadline_s=0.05, unwind_timeout_s=0.05)
        await h.start()
        first = h.worker
        await h.delegate(1, "[hang]")
        await asyncio.sleep(0.25)
        self.assertEqual(first.of_type("interrupt"), [{"type": "interrupt", "hard": True}])
        self.assertEqual(first.killed[:1], ["deadline"])
        done = h.of_type("backend_run_done")[-1]
        self.assertEqual((done["status"], done["history"]), ("deadline", "rolled_back"))
        first.die("exited: -15")
        await asyncio.sleep(0.05)
        self.assertEqual(len(RecordingWorker.instances), 2)
        self.assertEqual(len(h.worker.of_type("configure")), 1)  # the new worker was configured
        await h.delegate(2, "hello", seq=30)
        self.assertEqual(len(h.worker.of_type("run")), 1)

    async def test_worker_death_mid_run_rolls_back_and_respawn_limit_fails_the_backend(self) -> None:
        h = Harness(max_respawns=1)
        await h.start()
        await h.delegate(1, "x")
        h.worker.die()
        await asyncio.sleep(0.05)
        self.assertEqual(h.of_type("backend_run_done")[-1]["status"], "worker_died")
        self.assertEqual([m["event"] for m in h.of_type("worker")][-2:], ["ready", "respawned"])
        h.worker.die()
        await asyncio.sleep(0.05)
        await h.delegate(2, "y")
        self.assertEqual(h.of_type("action")[-1]["kind"], "unavailable")
        self.assertEqual(h.of_type("answer")[-1]["reason"], "backend_unavailable")

    async def test_construct_failure_reports_and_blocks_runs(self) -> None:
        h = Harness(worker_kwargs={"configure_error": "ToolSurfaceError: collision"})
        await h.start()
        self.assertEqual(h.of_type("error")[-1]["code"], "construct_failed")
        await h.delegate(1, "x")
        self.assertEqual(h.worker.of_type("run"), [])
        self.assertEqual(h.of_type("action")[-1]["kind"], "unavailable")

    async def test_tools_locked_after_the_agent_exists_and_ignore_policy(self) -> None:
        h = Harness()
        await h.start()
        await h.controller.configure([{"name": "other"}], "POLICY")
        self.assertEqual(h.of_type("error")[-1]["code"], "tools_locked")
        h2 = Harness()
        await h2.start(update_after_start="ignore")
        await h2.controller.configure([{"name": "other"}], "POLICY")
        configured = h2.of_type("session.configured")[-1]
        self.assertEqual((configured["applied"], configured["tools"]), (False, ["get_order"]))

    async def test_close_stops_the_worker_and_suppresses_answers(self) -> None:
        h = Harness()
        await h.start()
        await h.delegate(1, "x")
        await h.controller.close()
        self.assertTrue(h.worker.stopped)
        self.assertEqual(h.controller.state, proto.CLOSED)
        self.assertEqual(h.of_type("answer"), [])
        self.assertEqual(h.of_type("backend_run_done")[-1]["status"], "closed")


if __name__ == "__main__":
    unittest.main()
