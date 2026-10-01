# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Tests for the Frontend/Backend backend conversation history."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any
from unittest.mock import patch

from examples.frontend_backend_agent.airline.backend import RecordingBookingBackend
from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.generic import planner as generic_planner_module
from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs
from examples.frontend_backend_agent.generic.planner import NvidiaGenericPlanner
from examples.frontend_backend_agent.src.delegation import DelegationRun
from examples.frontend_backend_agent.src.history import (
    MAX_HISTORY_CHARS,
    ConversationTranscript,
    DelegationLedger,
    classify_result,
)
from examples.frontend_backend_agent.src.planner import NvidiaThinkerPlanner
from realtime.conversation import ConversationJournal


def _client_tools(*names: str):
    return build_client_tool_specs(
        tuple(
            {
                "type": "function",
                "name": name,
                "description": f"Run {name}.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
            }
            for name in names
        )
    )


class _ScriptedPlanner:
    """Return one client call per delegation, then complete; record what each plan received."""

    def __init__(self, tool: str = "update_record") -> None:
        self.tool = tool
        self.calls: list[dict[str, Any]] = []

    async def plan(self, *, query: str, state: dict, **kwargs: Any) -> dict:
        self.calls.append({"query": query, "round": state["planning_round"], **kwargs})
        if state["planning_round"] > 1:
            return {"complete": True}
        return {"tool": self.tool, "params": {"record_id": query.split()[-1].rstrip(".")}}


def _backend(planner, execute, ledger: DelegationLedger | None, *names: str) -> GenericThinkerBackend:
    return GenericThinkerBackend(
        planner=planner,
        tools={},
        enabled_tools=names or ("update_record",),
        client_tools=_client_tools(*(names or ("update_record",))),
        client_tool_executor=execute,
        overall_timeout_seconds=5,
        planner_timeout_seconds=5,
        max_planning_rounds=2,
        conversation_ledger=ledger,
    )


async def _ok(calls, _timeout):
    return [{"ok": True, "updated": True} for _call in calls]


class PlannerPayloadTests(unittest.IsolatedAsyncioTestCase):
    """Without history the planner input is byte-identical to the previous layout."""

    async def test_generic_payload_layout_with_and_without_history(self) -> None:
        captured: list = []

        async def fake_inference(_llm, context, _span, *, max_tokens):
            captured.append(json.loads(context.get_messages()[-1]["content"]))
            return '{"complete": true}'

        planner = NvidiaGenericPlanner(llm=object(), system_prompt="Plan.", enabled_tools=())
        with patch.object(generic_planner_module, "run_streamed_inference", fake_inference):
            await planner.plan(query="q", state={"s": 1})
            await planner.plan(query="q", state={"s": 1}, history=[{"run": 1}])

        self.assertEqual(
            list(captured[0]), ["untrusted_user_request", "enabled_tools", "session_state", "runtime_context"]
        )
        self.assertEqual(
            list(captured[1]),
            ["untrusted_user_request", "conversation_history", "enabled_tools", "session_state", "runtime_context"],
        )

    async def test_airline_payload_layout_with_and_without_history(self) -> None:
        captured: list = []

        class LLM:
            async def run_inference(self, context, *, max_tokens):
                captured.append(json.loads(context.get_messages()[-1]["content"]))
                return '{"tool": "response_hint"}'

        planner = NvidiaThinkerPlanner(llm=LLM(), system_prompt="Plan.")
        await planner.plan(query="q", slots={}, state={})
        await planner.plan(query="q", slots={}, state={}, history=[{"run": 1}])

        self.assertEqual(list(captured[0]), ["query", "structured_fields", "session_state", "runtime_context"])
        self.assertEqual(
            list(captured[1]),
            ["query", "conversation_history", "structured_fields", "session_state", "runtime_context"],
        )


class GenericHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_flag_off_sends_no_history_keyword(self) -> None:
        planner = _ScriptedPlanner()
        backend = _backend(planner, _ok, None)

        await backend.call("Update record R1.")
        await backend.call("Update record R2.")

        self.assertTrue(all("history" not in call for call in planner.calls))

    async def test_first_plan_has_no_history_and_the_second_sees_the_confirmed_write(self) -> None:
        transcript = ConversationTranscript()
        ledger = DelegationLedger(transcript)
        planner = _ScriptedPlanner()
        backend = _backend(planner, _ok, ledger)

        transcript.record_user("Please update record R1.")
        await backend.call("Update record R1.")
        ledger.record_delivery(ledger._entries[-1].run_id, "via_talker")
        transcript.record_assistant("Record R1 is updated.")
        transcript.record_user("Now R2.")
        await backend.call("Update record R2.")

        first_round_calls = [call for call in planner.calls if call["round"] == 1]
        self.assertNotIn("history", first_round_calls[0])
        history = first_round_calls[1]["history"]
        self.assertEqual([entry["run"] for entry in history], [1, 2])
        earlier, current = history
        self.assertEqual(earlier["transcript"], [{"role": "user", "text": "Please update record R1."}])
        self.assertEqual(
            earlier["tool_calls"],
            [
                {
                    "tool": "update_record",
                    "kind": "write",
                    "arguments": '{"record_id": "R1"}',
                    "state": "confirmed",
                    "status": "success",
                }
            ],
        )
        self.assertEqual(earlier["result"], "answered")
        self.assertEqual(earlier["delivery"], "via_talker")
        self.assertTrue(current["current"])
        self.assertEqual(
            current["transcript"],
            [{"role": "assistant", "text": "Record R1 is updated."}, {"role": "user", "text": "Now R2."}],
        )

    async def test_a_detached_write_in_flight_is_visible_to_the_replacement_plan(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        planner = _ScriptedPlanner()
        published = asyncio.Event()
        release = asyncio.Event()

        async def gated(calls, _timeout):
            published.set()
            await release.wait()
            return [{"ok": True} for _call in calls]

        backend = _backend(planner, gated, ledger)
        first = asyncio.create_task(backend.call("Update record R1."))
        await published.wait()
        second = asyncio.create_task(backend.call("Update record R9."))
        await asyncio.sleep(0.01)
        seen_while_running = [call for call in planner.calls if call["round"] == 1][1]["history"][0]["tool_calls"]
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await first
        await second

        self.assertEqual(seen_while_running[0]["state"], "started")
        rendered = ledger.render()
        self.assertEqual(rendered[0]["tool_calls"][0]["state"], "confirmed")
        self.assertNotIn("delivery", rendered[0])

    async def test_timeouts_cancellations_and_errors_leave_writes_unconfirmed(self) -> None:
        async def timed_out(calls, _timeout):
            return [{"ok": False, "error": {"code": "client_tool_timeout", "message": "late"}} for _call in calls]

        async def broken(calls, _timeout):
            raise RuntimeError("socket closed")

        for execute in (timed_out, broken):
            with self.subTest(execute=execute.__name__):
                ledger = DelegationLedger(ConversationTranscript())
                await _backend(_ScriptedPlanner(), execute, ledger).call("Update record R1.")
                self.assertEqual(ledger.render()[0]["tool_calls"][0]["state"], "unconfirmed")

        ledger = DelegationLedger(ConversationTranscript())
        blocked = asyncio.Event()

        async def hanging(calls, _timeout):
            blocked.set()
            await asyncio.Event().wait()

        backend = _backend(_ScriptedPlanner(), hanging, ledger)
        call = asyncio.create_task(backend.call("Update record R1."))
        await blocked.wait()
        backend.cancel_active("user_cancelled")
        with self.assertRaises(asyncio.CancelledError):
            await call
        entry = ledger.render()[0]
        self.assertEqual(entry["tool_calls"][0]["state"], "unconfirmed")
        self.assertEqual(entry["result"], "cancelled")

    async def test_a_suppressed_duplicate_is_not_recorded_as_executed(self) -> None:
        async def failing(calls, _timeout):
            return [{"ok": False, "error": {"code": "bad", "message": "no"}} for _call in calls]

        ledger = DelegationLedger(ConversationTranscript())
        backend = _backend(_ScriptedPlanner(), failing, ledger)
        await backend.call("Update record R1.")
        await backend.call("Update record R1.")

        calls = [call for entry in ledger.render() for call in entry["tool_calls"]]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["status"], "unavailable")

    async def test_an_adopted_run_is_one_entry_with_the_adopters_delivery(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        release = asyncio.Event()

        class Planner(_ScriptedPlanner):
            async def plan(self, *, query, state, **kwargs):
                if state["planning_round"] == 1:
                    await release.wait()
                return await super().plan(query=query, state=state, **kwargs)

        backend = _backend(Planner(), _ok, ledger)
        first = asyncio.create_task(backend.call("Update record R1."))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(backend.call("Update record R1.", continue_active=True))
        with self.assertRaises(asyncio.CancelledError):
            await first
        run_id = ledger._entries[-1].run_id
        ledger.record_delivery(run_id, "not_delivered")
        release.set()
        await second
        ledger.record_delivery(run_id, "via_talker")

        self.assertEqual(len(ledger._entries), 1)
        self.assertEqual(ledger.render()[0]["delivery"], "via_talker")


class _Booking:
    def __init__(self, *, hang: bool = False) -> None:
        self.hang = hang
        self.started = asyncio.Event()

    async def create_booking(self, **kwargs):
        self.started.set()
        if self.hang:
            await asyncio.Event().wait()
        return {"pnr": "ABC123"}

    async def search_flights(self, **kwargs):
        return []

    async def get_pnr(self, pnr_code):
        return None


class AirlineWriteBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_create_booking_is_a_write(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        run = DelegationRun("run-1", "book", lambda _progress: _create(RecordingBookingBackend(_Booking(), ledger)))
        ledger.open("run-1", "book it")
        await run.wait(run.attach())

        calls = ledger.render()[0]["tool_calls"]
        self.assertEqual(
            [(call["tool"], call["state"], call["status"]) for call in calls],
            [("create_booking", "confirmed", "success")],
        )

    async def test_a_cancelled_booking_request_is_unconfirmed(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        booking = _Booking(hang=True)
        ledger.open("run-1", "book it")
        run = DelegationRun("run-1", "book", lambda _progress: _create(RecordingBookingBackend(booking, ledger)))
        future = run.attach()
        await booking.started.wait()
        run.task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await run.wait(future)

        self.assertEqual(ledger.render()[0]["tool_calls"][0]["state"], "unconfirmed")

    async def test_a_confirmation_prompt_records_no_write(self) -> None:
        class Planner:
            async def plan(self, *, query, slots, state, **kwargs):
                return {"tool": "booking", "params": {"flight_selected": "1"}}

        ledger = DelegationLedger(ConversationTranscript())
        thinker = ThinkerBackend(planner=Planner(), backend=_Booking(), conversation_ledger=ledger)

        payload = await thinker.call("Book the first flight.")

        self.assertNotEqual(payload.get("tool"), "booking")
        self.assertFalse([call for call in ledger._entries[0].calls if call.kind == "write"])


async def _create(backend: RecordingBookingBackend) -> dict:
    await backend.create_booking(passenger_name="Ava", flight={"flight_id": "AA311", "date": "2026-10-02"})
    return {"type": "tool_result", "status": "success", "response_text": "Booked."}


class ClassifyResultTests(unittest.TestCase):
    def test_results_come_from_payloads(self) -> None:
        cases = {
            "answered": {"type": "tool_result", "status": "success"},
            "failed": {"type": "tool_result", "status": "unavailable"},
            "needs_input": {"type": "response_hint", "reason": "params_missing"},
            "timeout": {"type": "response_hint", "reason": "timeout"},
            "planner_failed": {"type": "response_hint", "reason": "planner_error"},
            "unsupported": {"type": "response_hint", "reason": "unsupported_request"},
        }
        for expected, payload in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_result(payload), expected)


def _message(role: str, part_type: str, text: str, item_id: str) -> dict:
    field = "text" if part_type.endswith("text") else "transcript"
    return {
        "id": item_id,
        "type": "message",
        "role": role,
        "status": "completed",
        "content": [{"type": part_type, field: text}],
    }


class TranscriptTests(unittest.TestCase):
    def test_realtime_journal_follows_truncation_deletion_and_insertion(self) -> None:
        journal = ConversationJournal()
        transcript = ConversationTranscript(lambda: journal)
        start = transcript.cursor()
        transcript.record_user("Check record R1.")
        journal.add_item(_message("assistant", "output_audio", "Record R1 is active and was renewed.", "a1"))
        journal.add_item(_message("assistant", "output_audio", "Anything else?", "a2"))
        journal.add_item(_message("user", "input_text", "typed note", "u1"), previous_item_id=None)
        transcript.record_assistant("ignored: the journal is authoritative")

        journal.patch_item("a1", {"content": [{"type": "output_audio", "transcript": "Record R1 is"}]})
        journal.delete_item("a2")
        messages, truncated = transcript.between(start)

        self.assertFalse(truncated)
        self.assertEqual(
            messages,
            [
                {"role": "user", "text": "Check record R1."},
                {"role": "assistant", "text": "Record R1 is"},
                {"role": "user", "text": "typed note"},
            ],
        )
        self.assertEqual(transcript.latest_user_text(), "typed note")

    def test_a_cursor_older_than_the_retained_journal_is_marked_truncated(self) -> None:
        class RetainedJournal:
            def last_sequence(self) -> int:
                return 5000

            def oldest_retained_sequence(self) -> int:
                return 905

            def ordered_item_ids(self):
                return ()

            def added_after(self, _sequence):
                return ()

        transcript = ConversationTranscript(lambda: RetainedJournal())

        self.assertTrue(transcript.between((10, 0))[1])
        self.assertFalse(transcript.between((904, 0))[1])

    def test_webrtc_direct_speech_is_emitted_text_marked_when_possibly_interrupted(self) -> None:
        interruptions = [0]
        ledger = DelegationLedger(ConversationTranscript(), interruption_count=lambda: interruptions[0])
        ledger.open("run-1", "weather")
        ledger.record_delivery("run-1", "spoken_direct", "It is sunny.")
        ledger.open("run-2", "stock")
        ledger.record_delivery("run-2", "spoken_direct", "Up two percent.")
        interruptions[0] = 1
        ledger.open("run-3", "news")
        ledger.record_delivery("run-3", "spoken_direct", "Markets are calm.")

        first, second, third = ledger.render()
        self.assertEqual(first["delivered_text"], "It is sunny.")
        self.assertNotIn("possibly_interrupted", first)
        self.assertTrue(second["possibly_interrupted"])
        self.assertNotIn("possibly_interrupted", third)
        interruptions[0] = 2
        self.assertTrue(ledger.render()[2]["possibly_interrupted"])

    def test_realtime_does_not_store_delivered_text(self) -> None:
        journal = ConversationJournal()
        ledger = DelegationLedger(ConversationTranscript(lambda: journal))
        ledger.open("run-1", "weather")
        ledger.record_delivery("run-1", "spoken_direct", "It is sunny.")
        ledger.open("run-2", "stock")

        self.assertNotIn("delivered_text", ledger.render()[0])


class BoundTests(unittest.TestCase):
    def test_two_hundred_distinct_long_tool_names_fit_through_the_fallback(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        for run in ("run-1", "run-2"):
            ledger.open(run, "do many things")
            for index in range(200):
                record = ledger.write_started(run, f"{run}-{index:03d}" + "x" * 55, {"record_id": "R" * 400})
                ledger.write_confirmed(record, "success")

        rendered = ledger.render()

        self.assertLessEqual(len(json.dumps(rendered, ensure_ascii=False)), MAX_HISTORY_CHARS)
        self.assertEqual(set(rendered), {"history_truncated", "omitted_entries", "write_counts", "recent_writes"})
        self.assertEqual(rendered["omitted_entries"], 2)
        self.assertEqual(rendered["write_counts"], {"confirmed/success": 400})
        self.assertTrue(rendered["recent_writes"])
        self.assertEqual(rendered["recent_writes"][0]["tool"], "run-2-199" + "x" * 55)
        self.assertNotIn("arguments", rendered["recent_writes"][0])

    def test_an_oversized_earlier_entry_is_summarized_without_tool_names(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        ledger.open("run-1", "do many things")
        for index in range(200):
            record = ledger.write_started("run-1", f"{index:03d}" + "x" * 61, {"record_id": "R" * 400})
            ledger.write_confirmed(record, "success")
        ledger.open("run-2", "next")

        rendered = ledger.render()

        self.assertLessEqual(len(json.dumps(rendered, ensure_ascii=False)), MAX_HISTORY_CHARS)
        self.assertTrue(rendered["history_truncated"])
        self.assertEqual(rendered["omitted_entries"], 1)
        self.assertEqual(rendered["write_counts"], {"confirmed/success": 200})
        self.assertEqual([entry["run"] for entry in rendered["entries"]], [2])

    def test_eight_maximal_entries_fit_and_keep_write_status(self) -> None:
        transcript = ConversationTranscript()
        ledger = DelegationLedger(transcript)
        for run in range(10):
            for _turn in range(10):
                transcript.record_user("word " * 100)
            ledger.open(f"run-{run}", "request " * 100)
            record = ledger.write_started(f"run-{run}", "update_record", {"value": "v" * 500})
            ledger.write_confirmed(record, "success")
            ledger.read(f"run-{run}", "get_record", {"value": "v" * 500}, "success")
            ledger.close(f"run-{run}", {"type": "tool_result", "status": "success", "response_text": "done " * 100})

        rendered = ledger.render()

        self.assertLessEqual(len(json.dumps(rendered, ensure_ascii=False)), MAX_HISTORY_CHARS)
        self.assertTrue(rendered["history_truncated"])
        self.assertEqual(rendered["omitted_entries"], 2)
        self.assertEqual(rendered["write_counts"], {"confirmed/success": 2})
        for entry in rendered["entries"]:
            writes = [call for call in entry["tool_calls"] if call["kind"] == "write"]
            self.assertEqual([(call["state"], call["status"]) for call in writes], [("confirmed", "success")])

    def test_small_histories_are_untouched(self) -> None:
        ledger = DelegationLedger(ConversationTranscript())
        ledger.open("run-1", "one")
        ledger.open("run-2", "two")

        rendered = ledger.render("run-2")

        self.assertIsInstance(rendered, list)
        self.assertNotIn("history_truncated", rendered[0])
        self.assertTrue(rendered[1]["current"])
        self.assertIsNone(DelegationLedger().render())


if __name__ == "__main__":
    unittest.main()
