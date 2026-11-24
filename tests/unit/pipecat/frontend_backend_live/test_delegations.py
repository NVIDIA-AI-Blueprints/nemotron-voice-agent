# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The delegation reader and its work note: every ending read from official fields, by structure."""

import asyncio
import unittest
from collections import Counter
from functools import partial

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.items import message_text
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegations import DelegationReader
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.prompts import load_prompts
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.worknote import WorkNote
from tests.unit.pipecat.core_helpers import load, official_events

LOADED = load_prompts()
LINES = LOADED.lines
TEXTS = {
    "handover_line": "Handled together with your other request (D2).",
    "cancelled_line": "Stopped for your newer request.",
    "unconfirmed_line": "I could not confirm that the earlier request stopped; check before trying it again.",
}
LABEL = "delegation_core"


def response(kind, response_id, *, status, **fields):
    body = {"id": response_id, "object": "response", "status": status, "output": [], **fields}
    return {"type": f"response.{kind}", "response": body}


created, queued = partial(response, "created", status="queued"), partial(response, "queued", status="queued")
completed = partial(response, "completed", status="completed")
failed = partial(response, "failed", status="failed", error={"code": "server_error", "message": "The backend failed."})


def in_progress(response_id, previous=None):
    named = {"previous_response_id": previous} if previous else {}
    return response("in_progress", response_id, status="in_progress", **named)


def incomplete(response_id, reason=None):
    if reason is None:
        return response("incomplete", response_id, status="cancelled", incomplete_details=None)
    return response("incomplete", response_id, status="incomplete", incomplete_details={"reason": reason})


def done(item):
    return {"type": "response.output_item.done", "item": item}


def message(text, phase="final_answer", item_id="msg_1", status="completed"):
    part = {"type": "output_text", "text": text, "annotations": []}
    item = {"id": item_id, "type": "message", "role": "assistant", "status": status, "content": [part]}
    return done({**item, "phase": phase} if phase else item)


def function_call(name, call_id="call_1"):
    return done({"id": f"fc_{call_id}", "type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"})


def mcp_call(label, name, item_id="mcp_1", output="{}", status="completed"):
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return done({**item, "output": output, "status": status})


def mcp_started(label, name, item_id="mcp_1"):
    """Return the event that starts a backend call, as the IM writes it: its item in progress, with no output yet."""
    item = {"id": item_id, "type": "mcp_call", "server_label": label, "name": name, "arguments": "{}"}
    return {"type": "response.output_item.added", "item": {**item, "output": None, "status": "in_progress"}}


HANDOVER = message(TEXTS["handover_line"], phase="commentary", item_id="handover")


class Results:
    """The reader's output, as the voice takes it."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.ready_results = []

    def ready(self, result):
        self.calls.append(("ready", result.delegation_id, result.speak))
        self.ready_results.append(result)

    def notice(self, delegation_id, text):
        self.calls.append(("notice", delegation_id, text))

    def ended(self, delegation_id):
        self.calls.append(("results.ended", delegation_id))

    def handed_over(self, delegation_id, taker):
        self.calls.append(("handed_over", delegation_id, taker))


class Display(Results):
    """The reader's display side; each card's ending names how it ended."""

    def started(self, delegation_id, request):
        self.calls.append(("started", delegation_id, request))

    def progress(self, delegation_id, line):
        self.calls.append(("progress", delegation_id, line))

    def shown(self, delegation_id, item):
        self.calls.append(("shown", delegation_id, item.get("name")))

    def answered(self, result):
        self.calls.append(("answered", result.delegation_id, result.speak))

    def ended(self, delegation_id, how):
        self.calls.append(("ended", delegation_id, how.value))


class Reading:
    """A reader with recording doubles around it, for the frontend's own delegations ``delegation_ids``."""

    def __init__(self, *delegation_ids, app_wait_seconds=None, client_tool_timeout_seconds=None):
        self.events: list[tuple[str, dict]] = []
        self.results, self.display, self.arrived = Results(), Display(), []
        note = WorkNote(LOADED.cascade, tracks=True, clock=lambda: 0.0, observe=self.observe)
        for delegation_id in delegation_ids:
            note.confirm(note.reserve(f"request {delegation_id}"), delegation_id)
        self.reader = DelegationReader(
            note,
            self.results,
            self.display,
            LINES,
            display_label=LABEL,
            app_wait_seconds=app_wait_seconds,
            item_text=message_text,
            observe=self.observe,
            outcome_arrived=self.arrived.append,
            client_tool_timeout_seconds=client_tool_timeout_seconds,
        )

    def observe(self, kind, **fields):
        self.events.append((kind, fields))

    def hand(self, delegation_id, *events):
        for event in events:
            self.reader.on_delegation_event(delegation_id, event)

    def calls(self, kind):
        """Return what the reader told the voice or the display of ``kind``, in order."""
        return [call[1:] for call in self.results.calls + self.display.calls if call[0] == kind]

    def notices(self):
        return [text for _, text in self.calls("notice")]

    def endings(self) -> dict:
        return dict(self.calls("ended"))

    def outcome(self) -> dict:
        """Return the reader's outcome in the fixture's terms."""

        def traced(kind):
            return [fields["delegation_id"] for name, fields in self.events if name == kind]

        def count(calls):
            return dict(Counter(delegation_id for delegation_id, *_ in calls))

        return {
            "results": count(self.calls("ready")),
            "endings": self.endings(),
            "takers": dict(self.calls("handed_over")),
            "waits": sorted(set(traced("delegation.waits_for_app"))),
            "typed": sorted(set(traced("delegation.typed"))),
            "progress": count(self.calls("progress")),
            "steps": dict(Counter(traced("delegation.backend_step"))),
            "shown": count(self.calls("shown")),
        }


class EndingTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_fixture_scenario_ends_as_expected(self):
        for name, scenario in load().items():
            events = official_events(scenario, TEXTS)
            reworded = official_events(scenario, dict.fromkeys(TEXTS, "anything at all"))
            variants = [("configured", events)]
            if events != reworded:
                variants.append(("reworded", reworded))
            for wording, frames in variants:
                with self.subTest(name, wording=wording):
                    expected = scenario["reader"]
                    typed = set(expected.get("typed", ()))
                    reading = Reading(*sorted(set(scenario["responses"].values()) - typed))
                    for delegation_id, event in frames:
                        reading.hand(delegation_id, event)
                    outcome = reading.outcome()
                    for key in ("results", "endings", "takers", "steps", "shown"):
                        self.assertEqual(outcome[key], expected.get(key, {}), key)
                    for key in ("waits", "typed"):
                        self.assertEqual(outcome[key], sorted(expected.get(key, ())), key)
                    if "progress" in expected:
                        self.assertEqual(outcome["progress"], expected["progress"], "progress")
                    reading.reader.close()

    async def test_a_failure_says_the_apology_or_the_unclear_line(self):
        for events, notices, ending in (
            ([failed("r1")], [LINES.apology], "failed"),
            ([mcp_call("cafe", "manage_cart"), failed("r1")], [LINES.unclear], "failed"),  # once a tool ran
            ([incomplete("r1", "max_output_tokens")], [LINES.apology], "failed"),
            ([incomplete("r1")], [], "cancelled"),
        ):
            with self.subTest(ending=ending, notices=notices):
                reading = Reading("D1")
                reading.hand("D1", created("r1"), in_progress("r1"), *events)
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))

    async def test_a_lost_stream_says_its_line_once(self):
        reading = Reading("D1", "D2")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.on_delegation_lost("D1")
        reading.reader.on_delegation_lost("D1")
        reading.hand("D1", message("Too late."), completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([LINES.lost], {"D1": "lost"}))
        reading.reader.close()
        reading.hand("D2", created("r2"), in_progress("r2"), message("Done."), completed("r2"))
        self.assertEqual(reading.calls("ready"), [])

    async def test_the_app_wait_line_then_the_unclear_one_at_the_deadline(self):
        for case, (continued, deadline, notices, endings) in {
            "the app-wait time runs out": (False, None, [LINES.app_wait_line], {}),
            "a continuation comes first": (True, None, [], {}),
            # The core writes nothing at its deadline, so the reader ends the wait at the same deadline.
            "the core's deadline passes": (False, 0.1, [LINES.app_wait_line, LINES.unclear], {"D1": "failed"}),
        }.items():
            with self.subTest(case):
                reading = Reading("D1", app_wait_seconds=0.05, client_tool_timeout_seconds=deadline)
                reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
                if continued:
                    reading.hand("D1", created("r2"))
                await asyncio.sleep(0.2)
                self.assertEqual((reading.notices(), reading.endings()), (notices, endings))
                open_work = [work.id for work in reading.reader.worknote.open()]
                self.assertEqual(open_work, [] if endings else ["D1"])

    async def test_a_taker_named_after_the_deadline_takes_the_work(self):
        for previous, notices, ending in (("r1", [], "handed_over"), (None, [LINES.unclear], "failed")):
            with self.subTest(previous=previous):
                reading = Reading("D1", "D2", client_tool_timeout_seconds=0.05)
                reading.hand("D1", created("r1"), in_progress("r1"), function_call("get_menu"), completed("r1"))
                reading.hand("D2", created("r2"), queued("r2"))
                await asyncio.sleep(0.1)
                self.assertEqual((reading.notices(), reading.endings()), ([], {}))
                reading.hand("D2", in_progress("r2", previous=previous))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))
                reading.hand("D1", created("r3"), in_progress("r3"), completed("r3"))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))


class ResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_final_answer_is_a_result_and_a_partial_is_not(self):
        reading = Reading("D1")
        reading.hand("D1", created("r1"), in_progress("r1"), message("Tea: yes.", item_id="m1"))
        self.assertEqual(reading.arrived, ["D1"])
        reading.hand("D1", message("Half", phase=None, status="incomplete", item_id="m2"))
        reading.hand("D1", message("Cake: no.", phase=None, item_id="m3"), completed("r1"))
        self.assertEqual(reading.calls("ready"), [("D1", "Tea: yes."), ("D1", "Cake: no.")])

    async def test_only_a_completed_display_item_is_shown_never_said(self):
        reading = Reading("D1")
        error = '{"error": "show failed: display must be non-empty text"}'
        reading.hand("D1", created("r1"), in_progress("r1"), mcp_call(LABEL, "show", output=error, status="failed"))
        self.assertEqual(reading.display.calls, [])
        reading.hand("D1", mcp_call(LABEL, "show", item_id="mcp_2"))
        self.assertEqual(reading.display.calls, [("shown", "D1", "show")])
        reading.hand("D1", message("Done."), completed("r1"))
        self.assertEqual([call[0] for call in reading.display.calls], ["shown", "answered", "ended"])
        self.assertNotIn("show", str(reading.results.calls))

    async def test_a_line_waits_while_another_delegation_may_name_it(self):
        # A commentary line waits while D2 or D3 may name r1, unless its response went on after it.
        reading = Reading("D1", "D2", "D3")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D3", created("r3"), queued("r3"))
        reading.hand("D1", created("r1"), in_progress("r1"), message("Checking the menu.", phase="commentary"))
        self.assertEqual(reading.calls("progress"), [])
        reading.hand("D1", mcp_started("cafe", "get_menu"))
        reading.hand("D1", message("Pricing your order.", phase="commentary", item_id="m2"))
        reading.hand("D2", in_progress("r2"))
        self.assertEqual(reading.calls("progress"), [("D1", "Checking the menu.")])
        reading.hand("D3", in_progress("r3"))
        self.assertEqual(reading.calls("progress"), [("D1", "Checking the menu."), ("D1", "Pricing your order.")])


class HandoverTests(unittest.IsolatedAsyncioTestCase):
    async def test_server_request_holds_replay_privately_before_a_delayed_first_response(self):
        from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation import ServerDelegation
        from tests.unit.pipecat.live_helpers import FakeAdapter, Stream

        old_end, new_first, new_end = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Delayed(Stream):
            async def __anext__(self):
                await new_first.wait()
                return await super().__anext__()

        first_stream = Stream(
            [created("r1"), in_progress("r1"), message("Which size?", item_id="question")],
            until=old_end,
            then=[completed("r1")],
        )
        second_stream = Delayed(
            [created("r2"), in_progress("r2", previous="r1")],
            until=new_end,
            then=[message("Done.", item_id="done"), completed("r2")],
        )
        core = FakeAdapter(streams=[first_stream, second_stream])

        async def attachment(conversation, attached):
            return {}

        core.set_attachment = attachment
        part = ServerDelegation(core)
        reading = Reading()
        part.listen(reading.reader)
        part.retain_attachments()
        await part.open(None)

        async def close():
            old_end.set()
            new_first.set()
            new_end.set()
            await part.close()

        self.addAsyncCleanup(close)

        async def until(condition):
            async with asyncio.timeout(2):
                while not condition():
                    await asyncio.sleep(0)

        note = reading.reader.worknote
        reservation = note.reserve("original request")
        first = await part.delegate([{"role": "user", "content": "First request."}])
        note.confirm(reservation, first)
        await until(
            lambda: reading.reader.open[first].response is not None and reading.reader.open[first].response.finals
        )
        note.said(first, item_id="question")
        reservation = note.reserve("new request")
        second = await part.delegate([{"role": "user", "content": "A correction."}])
        note.confirm(reservation, second)
        self.assertIsNone(reading.reader.open[second].response)
        old_end.set()
        await until(lambda: reading.reader.open[first].awaits_taker)
        self.assertIn(first, note)

        await part.suspend_attachment()
        restored = Reading()
        received = []
        receive = restored.reader.on_delegation_event

        def record(delegation_id, event):
            received.append(event["type"])
            receive(delegation_id, event)

        restored.reader.on_delegation_event = record
        part.listen(restored.reader)
        await part.resume_attachment()
        self.assertTrue(restored.reader.open[first].awaits_taker)
        self.assertEqual(restored.calls("ready"), [])
        new_first.set()
        await until(lambda: restored.calls("handed_over") == [(first, second)])
        self.assertIn(first, restored.reader.worknote.entries[second].carries)
        self.assertEqual(restored.notices(), [])
        self.assertTrue(all(kind.startswith("response.") for kind in received))
        self.assertEqual(
            [
                entry.request_state
                for entry in part.mailbox.entries
                if entry.delegation_id == second and entry.request_state is not None
            ],
            ["accepted"],
        )
        new_end.set()
        await until(lambda: second in restored.reader.ended)
        self.assertEqual(restored.calls("ready"), [(second, "Done.")])
        restored.reader.worknote.said(second, item_id="done")
        self.assertEqual(restored.reader.worknote.open(), [])
        self.assertEqual(restored.reader.requests, {})

    async def test_actual_submission_holds_a_completed_predecessor_until_the_request_outcome(self):
        for outcome in ("handover", "independent", "failed", "cancelled", "lost"):
            with self.subTest(outcome=outcome):
                reading = Reading("D1")
                reader, note = reading.reader, reading.reader.worknote
                reservation = note.reserve("the new request")
                reader.on_delegation_request("request-2", "submitted")
                reading.hand("D1", created("r1"), in_progress("r1"), message("Which size?", item_id="question"))
                note.said("D1", item_id="question")
                reading.hand("D1", completed("r1"))
                self.assertTrue(reader.open["D1"].awaits_taker)
                self.assertIn("D1", note)
                if outcome in ("failed", "cancelled"):
                    reader.on_delegation_request("request-2", outcome)
                    note.release(reservation)
                    self.assertNotIn("D1", note)
                    self.assertEqual(reading.endings(), {"D1": "answered"})
                else:
                    reader.on_delegation_request("request-2", "accepted", "D2")
                    self.assertNotIn("D2", note)
                    if outcome == "lost":
                        reader.on_delegation_lost("D2")
                        note.confirm(reservation, "D2")
                        self.assertEqual(reading.notices(), [LINES.lost])
                    else:
                        note.confirm(reservation, "D2")
                        previous = "r1" if outcome == "handover" else None
                        reading.hand("D2", created("r2"), queued("r2"), in_progress("r2", previous=previous))
                        self.assertEqual(reading.calls("handed_over"), [("D1", "D2")] if previous else [])
                        if previous:
                            self.assertEqual(note.entries["D2"].request, "request D1 / the new request")
                        reading.hand("D2", message("Done.", item_id="done"), completed("r2"))
                        note.said("D2", item_id="done")
                self.assertEqual(reader.requests, {})
                self.assertEqual(note.open(), [])
                self.assertEqual(reading.notices(), [LINES.lost] if outcome == "lost" else [])

    async def test_reversed_handover_links_reach_the_current_carrier_without_losing_original_results(self):
        for older_created_first in (True, False):
            with self.subTest(older_created_first=older_created_first):
                reading = Reading("D1")
                reader, note = reading.reader, reading.reader.worknote
                earlier_reservation = note.reserve("request D2")
                reservation = note.reserve("request D3")
                for key in ("D1", "D2", "D3"):
                    reader.on_delegation_request(key, "submitted", key)
                    reader.on_delegation_request(key, "accepted", key)
                reading.hand("D3", created("r3"), queued("r3"))
                reading.hand("D2", created("r2"), queued("r2"))
                for index, key in enumerate(("D1", "D2", "D3"), 1):
                    note.running(key)
                    entry = note.entries[key]
                    entry.created_at, entry.started_at, entry.progress_offers = index, index + 2, 1
                note.entries[reservation].created_at = 3
                note.entries[earlier_reservation].created_at = 2
                if older_created_first:
                    reading.hand("D1", created("r1"), in_progress("r1"), message("Which size?", item_id="question"))
                reading.hand("D3", in_progress("r3", previous="r2"))
                reading.hand("D2", in_progress("r2", previous="r1"), HANDOVER, completed("r2"))
                if not older_created_first:
                    self.assertEqual(reader.named, {"r1": "D3"})
                    reading.hand("D1", created("r1"), in_progress("r1"), message("Which size?", item_id="question"))
                note.confirm(earlier_reservation, "D2")
                note.confirm(reservation, "D3")
                carrier = note.entries["D3"]
                self.assertEqual(carrier.request, "request D1 / request D2 / request D3")
                self.assertEqual((carrier.created_at, carrier.started_at, carrier.progress_offers), (1, 3, 3))
                self.assertEqual(set(carrier.carries), {"D1", "D2"})
                self.assertEqual(reading.calls("handed_over"), [("D2", "D3"), ("D1", "D3")])
                self.assertEqual(note.pending_publications(), frozenset({("D1", "question")}))
                before = carrier.parts, carrier.carries, carrier.ready, carrier.progress_offers
                self.assertIsNone(note.hand_over("D3", "D1"))
                self.assertIsNone(note.hand_over("D3", "missing"))
                reading.hand("D1", in_progress("r1", previous="r3"))
                self.assertEqual((carrier.parts, carrier.carries, carrier.ready, carrier.progress_offers), before)
                reading.hand("D1", completed("r1"))
                note.said("D1", item_id="question")
                note.said("D1", item_id="question")
                self.assertEqual(carrier.ready, 0)
                reading.hand("D3", message("Done.", item_id="done"), completed("r3"))
                note.said("D3", item_id="done")
                self.assertEqual(note.open(), [])
                self.assertEqual(reading.notices(), [])

    async def test_real_multi_source_clarification_is_one_result_in_both_stream_orders(self):
        from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
            CallerMessage,
        )
        from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.im.session import IMSession
        from tests.unit.pipecat.delegation_core.helpers import (
            CORE_SETTINGS,
            DEFAULTS,
            ScriptedAdapter,
            ask,
            placing,
            planned,
            read_all,
        )
        from tests.unit.pipecat.delegation_core.helpers import (
            TEXTS as CORE_TEXTS,
        )

        backend = ScriptedAdapter(
            conductor=planned(placing(CallerMessage(("D1", "D2"), "question", "Which booking?"))), max_running=2
        )
        im = IMSession(backend, DEFAULTS, CORE_SETTINGS.delegation, CORE_TEXTS)
        im.start()
        self.addAsyncCleanup(im.close)
        first = im.admit(ask("Change it."), delegation_id="D1")
        second = im.admit(ask("To six."), delegation_id="D2")
        older, newer = await asyncio.gather(read_all(first.stream), read_all(second.stream))
        for older_first in (True, False):
            with self.subTest(older_first=older_first):
                reading = Reading("D1", "D2")
                # Both delegations are admitted/queued before the conductor places this shared question.
                # Their starts, handover, question and terminals can still arrive in either stream order.
                reading.hand("D1", *older[:2])
                reading.hand("D2", *newer[:2])
                streams = [("D1", older[2:]), ("D2", newer[2:])]
                for delegation_id, events in streams if older_first else reversed(streams):
                    reading.hand(delegation_id, *events)
                self.assertEqual(reading.calls("ready"), [("D2", "Which booking?")])
                self.assertTrue(reading.results.ready_results[0].routing_question)
                self.assertEqual(reading.calls("handed_over"), [("D1", "D2")])
                self.assertEqual(reading.endings(), {"D1": "handed_over", "D2": "answered"})
                self.assertEqual(reading.notices(), [])
                self.assertEqual(reading.calls("progress"), [])

    async def test_a_handovers_line_is_never_progress(self):
        # Each response has its own stream, so the naming can come before or after the older line and its end.
        for taker_first in (True, False):
            with self.subTest(taker_first=taker_first):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"))
                naming, older = ("D2", [in_progress("r2", previous="r1")]), ("D1", [HANDOVER, completed("r1")])
                for delegation_id, events in [naming, older] if taker_first else [older, naming]:
                    reading.hand(delegation_id, *events)
                self.assertEqual((reading.calls("progress"), reading.notices()), ([], []))
                self.assertEqual(reading.endings(), {"D1": "handed_over"})
                self.assertEqual(reading.calls("handed_over"), [("D1", "D2")])

    async def test_an_answered_handover_keeps_its_answer(self):
        # D2 carries that a tool ran, so its failure leaves the outcome unclear.
        for older_first in (False, True):
            with self.subTest(older_first=older_first):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), mcp_call("cafe", "lookup"))
                reading.hand("D1", message("Task A done.", item_id="a"))
                naming, older = ("D2", [in_progress("r2", previous="r1")]), ("D1", [HANDOVER, completed("r1")])
                for delegation_id, events in [older, naming] if older_first else [naming, older]:
                    reading.hand(delegation_id, *events)
                self.assertEqual(reading.calls("handed_over"), [("D1", "D2")])
                self.assertEqual(reading.endings(), {"D1": "answered"})
                reading.hand("D2", failed("r2"))
                self.assertEqual(reading.notices(), [LINES.unclear])
                self.assertEqual(reading.calls("ready"), [("D1", "Task A done.")])

    async def test_an_early_naming_waits_for_its_response(self):
        reading = Reading("D1", "D2")
        reading.hand("D2", created("r2"), in_progress("r2", previous="r1"))
        reading.hand("D1", created("r1"), in_progress("r1"), HANDOVER, completed("r1"))
        self.assertEqual((reading.notices(), reading.endings()), ([], {"D1": "handed_over"}))

    async def test_a_spoken_question_survives_both_handover_stream_orders(self):
        for older_first in (False, True):
            with self.subTest(older_first=older_first):
                reading = Reading("D1", "D2")
                note = reading.reader.worknote
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), message("Which size?"))
                note.said("D1")
                naming = ("D2", [in_progress("r2", previous="r1")])
                older = ("D1", [message("I will continue after this call.", phase="commentary"), completed("r1")])
                for delegation_id, events in [older, naming] if older_first else [naming, older]:
                    reading.hand(delegation_id, *events)
                carrier = note.entries["D2"]
                self.assertEqual(carrier.request, "request D1 / request D2")
                self.assertIn("D1", carrier.carries)
                self.assertFalse(carrier.ended)
                self.assertEqual(reading.calls("ready"), [("D1", "Which size?")])
                self.assertEqual(reading.endings(), {"D1": "answered"})

    async def test_a_response_no_taker_names_ends_when_none_can(self):
        for answer, previous, notices, ending in (
            ([message("Tea: yes.")], None, [], "answered"),
            ([], None, [LINES.unclear], "late"),
            ([], "unknown", [LINES.unclear], "late"),
        ):
            with self.subTest(ending=ending, previous=previous):
                reading = Reading("D1", "D2")
                reading.hand("D2", created("r2"), queued("r2"))
                reading.hand("D1", created("r1"), in_progress("r1"), *answer, completed("r1"))
                self.assertEqual(reading.endings(), {})
                reading.hand("D2", in_progress("r2", previous=previous))
                self.assertEqual((reading.notices(), reading.endings()), (notices, {"D1": ending}))
                self.assertEqual(reading.calls("handed_over"), [])

    async def test_restored_forwarded_final_counts_as_answered_without_speaking_or_false_failure(self):
        reading = Reading("D1")
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.restore_final("D1")
        reading.hand("D1", completed("r1"))
        self.assertEqual(reading.endings(), {"D1": "answered"})
        self.assertEqual(reading.calls("ready"), [])
        self.assertEqual(reading.notices(), [])

    async def test_restored_progress_is_private_work_note_evidence_without_replaying_events(self):
        reading = Reading("D1")
        reading.hand("D1", created("r1"), in_progress("r1"))
        item = message("Checking the available dates.", phase="commentary")["item"]
        reading.reader.restore_progress("D1", item)
        self.assertEqual(reading.reader.worknote.entries["D1"].progress, "Checking the available dates.")
        self.assertEqual(reading.reader.worknote.quiet_candidates(3), ())
        self.assertEqual(reading.calls("progress"), [])
        self.assertEqual(reading.calls("ready"), [])
        self.assertEqual(reading.notices(), [])

    async def test_restored_forwarded_failure_cancel_and_loss_close_silently(self):
        for terminal, expected in (
            (failed("r1"), "failed"),
            (incomplete("r1"), "cancelled"),
            (incomplete("r1", "max_output_tokens"), "failed"),
            ({**failed("r1"), "_nva_lost": True}, "lost"),
        ):
            with self.subTest(expected=expected):
                reading = Reading("D1")
                reading.hand("D1", created("r1"), in_progress("r1"))
                reading.reader.restore_terminal("D1", terminal)
                reading.reader.restore_terminal("D1", terminal)
                self.assertNotIn("D1", reading.reader.worknote)
                self.assertNotIn("D1", reading.reader.open)
                self.assertEqual(reading.results.calls, [])
                self.assertEqual(reading.calls("ended"), [("D1", expected)])

    async def test_restored_terminal_silently_settles_an_older_response_waiting_for_a_taker(self):
        reading = Reading("D1", "D2")
        reading.hand("D2", created("r2"), queued("r2"))
        reading.hand("D1", created("r1"), in_progress("r1"))
        reading.reader.restore_final("D1")
        reading.hand("D1", completed("r1"))
        self.assertTrue(reading.reader.open["D1"].awaits_taker)
        reading.reader.restore_terminal("D2", failed("r2"))
        self.assertEqual(reading.reader.open, {})
        self.assertEqual(reading.reader.worknote.open(), [])
        self.assertEqual(reading.results.calls, [])
        self.assertEqual(reading.endings(), {"D1": "answered", "D2": "failed"})
