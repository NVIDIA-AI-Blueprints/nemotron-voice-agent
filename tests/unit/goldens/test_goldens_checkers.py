# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Three checks beside the goldens: a strict client, the console's reads and the close order.

* The strict client copies the rules by which tau2's Live client ends or stalls a benchmark run (tau2-bench 2174a60,
  ``src/tau2/voice/audio_native/openai/live_provider.py``, cited as ``T2`` below, and its ``events.py``), with no
  tau2 import. It reads the event channels of every recorded golden: each golden passes it, or each violation is
  listed, in ``KNOWN_VIOLATIONS``, or in ``PROVOKED_REFUSALS`` when the golden provokes the refusal with a command
  tau2 never sends. A whole call answered with tau2's own timing passes it too.
* The console contract reads ``GET /v1/live/sessions/{id}/history``, ``GET /v1/live/info``, the WebRTC ``201`` answer
  and the usage report as the browser console does (``client/public/live/app.js``), from the real gateway and its
  HTTP routes with the cascade engine. What the history panel reads through one whole call is pinned in
  ``recordings/console-history.json``.
* The close-order pin: on every close path, the session's own tasks end first; then the engine stops the pipeline and
  closes the worker and both model clients, in a fixed order; then ``session.closed`` comes once and last, before the
  socket or the data channel closes.

``console-history`` is recorded like a golden: ``LIVE_GOLDEN_RECORD=1`` writes it, a normal run only compares, and its
tags follow the harness's rules (``harness.py``, "Tag").
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from collections import Counter
from dataclasses import dataclass, field
from unittest.mock import patch

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack
from fastapi import FastAPI
from pipecat.frames.frames import EndFrame
from pydantic import BaseModel, ValidationError

import server
import utils
from delegation_core.im.session import IMSession
from examples.frontend_backend_live.adapter.adapter import CascadeAdapter
from examples.frontend_backend_live.adapter.pipeline_host import PipelineHost
from live.gateway import create_live_router
from live.mount import mount_live
from live.protocol import approx_tokens
from live.session import lifecycle
from tests.unit.cascade_helpers import FakeSTT, FakeTTS, cascade_parts, chat_core, core_hook
from tests.unit.frontend_backend_live.root.root_helpers import clear_root_caches
from tests.unit.goldens import harness
from tests.unit.goldens.harness import (
    AUDIO_EVENT,
    CLOCK,
    GOLDEN_CORE,
    GOLDEN_DIR,
    GOLDEN_LIVE_CONFIG,
    KNOWN_ROWS,
    RECORD_VARIABLE,
    RESPONSES_SESSION,
    TRANSCRIPT_KINDS,
    USAGE_INTERVAL_SECONDS,
    WAIT_SECONDS,
    WORK_RUNNING,
    GoldenCase,
    HeldRound,
    HoldingChat,
    Normalizer,
    RecordingWebSocket,
    ScriptedChat,
    answer_round,
    canonical,
    channel_diff,
    decision,
    line_matches,
    result_item,
    tool_call_round,
)

TEAS = '{"teas": ["Earl Grey", "Matcha"]}'


# --------------------------------------------------------------------------------------------- the strict client
# What ends or stalls a tau2 run, rule by rule (T2:395-468, _normalize_event; T2:175-185, _accept_event; T2:371-393,
# the continuation of a function-call batch).
RULES = {
    "error": "a top-level error event: tau2 raises RuntimeError (T2:432-433)",
    "unexpected_close": "session.closed before the client asked to close: tau2 raises ConnectionError (T2:428-431)",
    "nested_failure": "a relayed response.failed, response.incomplete or error: tau2 raises RuntimeError (T2:466-467)",
    "call_before_created": (
        "a function-call item with no relayed response.created of its own delegation before it: tau2 files each call "
        "under its delegation's latest response id, and that lookup raises KeyError (T2:438-446)"
    ),
    "stale_completed": (
        "a response.completed that carries another id than its delegation's latest response while calls are filed "
        "under that latest response: tau2 continues only the calls filed under the id a response.completed carries "
        "(T2:459-465), so those calls are not continued"
    ),
    "uncontinued_calls": (
        "a function call filed under a response that never completes: tau2 sends response.create only once the "
        "response its calls are filed under has completed (T2:384-393, 459-465), so the call waits forever"
    ),
    "missing_field": (
        "a field tau2 reads is missing, or has a value its parser refuses (T2:177, 406-424, 435-457; events.py:62-66)"
    ),
}
NESTED_FAILURES = frozenset({"response.failed", "response.incomplete", "error"})


# The fields tau2's own models parse from the events it reads, copied as models (not imported) so that the checker
# parses them the same way: pydantic's lax mode, which takes "0", 0.0 and True as integers. The transcript models are
# LiveTranscriptDelta and LiveInputTranscriptDelta; a call's fields fill FunctionCallArgumentsDoneEvent
# (events.py:62-66) the first time tau2 passes the call to its agent. The wire's integer times are pinned by
# test_live_pipeline.py, not here.
class TranscriptFields(BaseModel):
    delta: str
    start_ms: int
    end_ms: int


class InputTranscriptFields(TranscriptFields):
    event_id: str | None = None


class CallFields(BaseModel):
    call_id: str | None = None
    name: str | None = None
    arguments: str = "{}"


def parse_problem(model, values):
    """Why tau2's model refuses ``values``, or None when it parses them."""
    try:
        model.model_validate(values)
    except ValidationError as error:
        return "; ".join(f"{'.'.join(map(str, problem['loc']))}: {problem['msg']}" for problem in error.errors())
    return None


TRANSCRIPT_EVENTS = frozenset(TRANSCRIPT_KINDS.values())
EVENT_CHANNELS = ("session", "delegations", *TRANSCRIPT_KINDS)  # a golden's channels of Live events
# A golden records no client commands. The gateway answers a client's session.close, and only that, with this reason
# (live/session/session.py), so a recorded session.closed with it is one the client asked for.
CLIENT_CLOSE_REASON = "close_requested"

# The violations the recorded goldens are known to carry, by golden. Each entry names its rule, its channel
# ("delegations" for any delegation), the violating event as a tag's ``contains`` matches a line (harness.py, "Tag"),
# a row of ``KNOWN_ROWS`` (harness.py) and a note. An entry that no violation matches fails the test.
KNOWN_VIOLATIONS: dict[str, list[dict]] = {}

# Refusals that a golden provokes on purpose, with a command tau2's client never sends: tau2 would stop on them, but
# no tau2 run can meet them. tau2 sends response.create only to continue a response that has completed and whose every
# call it has answered, and it sends no typed messages. Each entry names the client command by
# its event id and says what it is; ``row`` is None when the refusal is the intended behavior.
PROVOKED_REFUSALS: dict[str, list[dict]] = {
    "G11": [
        {
            "rule": "error",
            "channel": "session",
            "contains": {"error": {"code": "function_call_outputs_required", "client_event_id": "too_early"}},
            "row": None,
            "note": "A response.create sent with a typed message while a call's output is still owed: it would "
            "continue nothing.",
        },
    ],
    "G16": [
        {
            "rule": "error",
            "channel": "session",
            "contains": {
                "error": {"code": "function_call_outputs_required", "client_event_id": "continue_without_output"}
            },
            "row": None,
            "note": "A response.create without the owed output, after the core's deadline: the call's route is still "
            "open, so it would continue nothing.",
        },
    ],
}

# Endings a golden scripts on purpose: a backend round that fails or is cut by the output limit, or a close while work
# runs. The core reports each with its official terminal (the gateway relays the close's ends before
# session.closed), and tau2's client stops on any relayed failure, as it would on a real backend's. They are the
# protocol, not a defect of the server.
SCRIPTED_ENDINGS: dict[str, list[dict]] = {
    name: [
        {
            "rule": "nested_failure",
            "channel": "delegations",
            "contains": {"event": {"type": kind}},
            "row": None,
            "note": note,
        }
    ]
    for name, kind, note in (
        ("G3", "response.incomplete", "The round is cut by the output limit: incomplete, max_output_tokens."),
        ("G4", "response.failed", "Every try of the round fails with HTTP 500: failed, server_error."),
        ("G13", "response.incomplete", "The client closes while the round runs: incomplete, cancelled."),
        ("X2", "response.failed", "The joined Work's round times out on every try: the taker's response fails."),
    )
}


@dataclass(frozen=True)
class Violation:
    """One broken rule: where it was broken, and the event that broke it."""

    rule: str
    channel: str
    index: int
    detail: str
    event: dict = field(compare=False, repr=False)

    def __str__(self):
        """Where the rule was broken, which rule, and how."""
        return f"{self.channel}[{self.index}] {self.rule}: {self.detail}"


def response_id_of(inner):
    response = inner.get("response")
    response_id = response.get("id") if isinstance(response, dict) else None
    return response_id if isinstance(response_id, str) else None


class StrictClient:
    """tau2's Live client as a checker: feed it the server's events in order, then call :meth:`end`.

    It keeps each rule the events break. tau2 raises at the first; the checker goes on, so one run shows every
    violation. ``recorded=True`` reads a golden's normalized events, whose clock values are ``<clock>``: their presence
    is checked, not their value.
    """

    def __init__(self, *, recorded=False):
        self.recorded = recorded
        self.closing = False
        self.latest_response: dict[str, str] = {}  # delegation id -> the id of its latest relayed response.created
        # Response id -> {call id: where the call's item arrived}: the calls tau2 files under that response and
        # continues once a response.completed carries its id.
        self.filed: dict[str, dict] = {}
        self.surfaced: set = set()  # the call ids tau2 has passed to its agent; it reads name and arguments only then
        self.violations: list[Violation] = []

    @classmethod
    def read(cls, events, *, asked_to_close_at=None, channel="stream"):
        """The violations in a raw stream; the client sent session.close just before event ``asked_to_close_at``."""
        client = cls()
        for index, event in enumerate(events):
            if index == asked_to_close_at:
                client.close_requested()
            client.receive(event, channel, index)
        client.end()
        return client.violations

    def close_requested(self):
        """The client is closing: tau2 sets this before it sends session.close."""
        self.closing = True

    def end(self):
        """The call is over: a call still filed under a response that never completed was never continued."""
        for response_id, calls in self.filed.items():
            for call_id, (channel, index, event) in calls.items():
                detail = f"call {call_id!r} is filed under {response_id}, which never completed"
                self.violations.append(Violation("uncontinued_calls", channel, index, detail, event))
        self.filed.clear()

    def receive(self, event, channel="stream", index=0):
        def broke(rule, detail):
            self.violations.append(Violation(rule, channel, index, detail, event))

        kind = event.get("type") if isinstance(event, dict) else None
        if not isinstance(kind, str):
            broke("missing_field", "an event with no type (T2:177)")
        elif kind in TRANSCRIPT_EVENTS:
            self.transcript(event, broke)
        elif kind == "session.closed" and not self.closing:
            broke("unexpected_close", f"session.closed ({event.get('reason')}) before the client asked to close")
        elif kind == "error":
            broke("error", json.dumps(event.get("error"))[:300])
        elif kind == "response.event":
            self.response_event(event, broke, (channel, index, event))

    def transcript(self, event, broke):
        # tau2 reads the text and the times with [], and an input delta's event id with get(), into its models.
        missing = [key for key in ("delta", "start_ms", "end_ms") if key not in event]
        if missing:
            broke("missing_field", f"{event['type']} without {', '.join(missing)}")
            return
        values = {key: event[key] for key in ("delta", "start_ms", "end_ms")}
        if self.recorded:
            values.update({key: 0 for key in ("start_ms", "end_ms") if values[key] == CLOCK})
        model = TranscriptFields
        if event["type"] == "session.input_transcript.delta":
            model, values["event_id"] = InputTranscriptFields, event.get("event_id")
        if problem := parse_problem(model, values):
            broke("missing_field", f"{event['type']} that tau2 cannot parse: {problem}")

    def response_event(self, event, broke, where):
        delegation_id, inner = event.get("delegation_id"), event.get("event")
        if not isinstance(delegation_id, str) or not isinstance(inner, dict) or not isinstance(inner.get("type"), str):
            broke("missing_field", "a response.event without delegation_id, event or event.type (T2:435-437)")
            return
        kind = inner["type"]
        if kind == "response.created":
            response_id = response_id_of(inner)
            if response_id is None:
                broke("missing_field", "response.created without response.id (T2:439)")
            else:
                self.latest_response[delegation_id] = response_id
        elif kind == "response.output_item.done":
            item = inner.get("item")
            if not isinstance(item, dict) or "type" not in item:
                broke("missing_field", "response.output_item.done without item.type (T2:442)")
            elif item["type"] == "function_call":
                self.function_call(delegation_id, item, broke, where)
        elif kind == "response.completed":
            response_id = response_id_of(inner)
            if response_id is None:
                broke("missing_field", "response.completed without response.id (T2:460)")
                return
            self.filed.pop(response_id, None)  # tau2 continues the calls filed under it
            latest = self.latest_response.get(delegation_id)
            if response_id != latest and self.filed.get(latest):
                waiting = sorted(self.filed[latest])
                broke("stale_completed", f"{delegation_id} completes {response_id}; {waiting} wait under {latest}")
        elif kind in NESTED_FAILURES:
            broke("nested_failure", f"{kind} relayed for {delegation_id}")

    def function_call(self, delegation_id, item, broke, where):
        # tau2 files the call under its delegation's latest response id, then reads the call's id; it reads the name
        # and the arguments only the first time it passes the call to its agent.
        if delegation_id not in self.latest_response:
            broke(
                "call_before_created",
                f"call {item.get('call_id')!r} of {delegation_id} has no response.created before it",
            )
        if "call_id" not in item:
            broke("missing_field", "a function-call item without call_id (T2:446)")
            return
        call_id = item["call_id"]
        if problem := parse_problem(CallFields, {"call_id": call_id}):
            broke("missing_field", f"a function-call item whose call_id tau2 cannot parse: {problem}")
            return
        if delegation_id in self.latest_response:
            self.filed.setdefault(self.latest_response[delegation_id], {})[call_id] = where
        if call_id in self.surfaced:
            return
        self.surfaced.add(call_id)
        missing = [key for key in ("name", "arguments") if key not in item]
        if missing:
            broke("missing_field", f"a function-call item without {', '.join(missing)} (T2:447-455)")
        elif problem := parse_problem(CallFields, {key: item[key] for key in ("call_id", "name", "arguments")}):
            broke("missing_field", f"a function-call item that tau2 cannot parse: {problem}")


def read_recorded(golden):
    """The strict client's violations in a golden's event channels, each channel read in its own order.

    Every rule depends on one channel only: the closing rule on the session channel, a delegation's rules on its own
    events, a transcript rule on one event. So reading the channels one by one finds what the client finds in the
    call's real interleaving, which a golden does not keep. A golden ends at quiet, when the server waits for the
    client: a call still filed then is one tau2 would never continue.
    """
    channels = golden["channels"]
    client = StrictClient(recorded=True)
    for index, event in enumerate(channels.get("session", [])):
        if event.get("type") == "session.closed" and event.get("reason") == CLIENT_CLOSE_REASON:
            client.close_requested()
        client.receive(event, "session", index)
    for delegation_id, events in channels.get("delegations", {}).items():
        for index, event in enumerate(events):
            client.receive(event, f"delegations/{delegation_id}", index)
    for channel, kind in TRANSCRIPT_KINDS.items():
        for index, line in enumerate(channels.get(channel, [])):
            # A text line is a delta with exactly the plain keys (harness.transcript_line); any other line is the event.
            event = {"type": kind, "delta": line, "start_ms": CLOCK, "end_ms": CLOCK} if isinstance(line, str) else line
            client.receive(event, channel, index)
    client.end()
    return client.violations


def known(entry, violation):
    channel = entry["channel"]
    return (
        entry["rule"] == violation.rule
        and (channel == violation.channel or (channel == "delegations" and violation.channel.startswith(channel + "/")))
        and line_matches(violation.event, entry["contains"])
    )


def unexplained(name, violations):
    """Each violation that no entry explains, and each entry that no violation matches."""
    entries = KNOWN_VIOLATIONS.get(name, []) + PROVOKED_REFUSALS.get(name, []) + SCRIPTED_ENDINGS.get(name, [])
    problems = [
        f"{violation} (rule: {RULES[violation.rule]}); fix it, or list it in KNOWN_VIOLATIONS with the row that "
        "will remove it (or, for a refusal the golden provokes with a command tau2 never sends, in PROVOKED_REFUSALS)"
        for violation in violations
        if not any(known(entry, violation) for entry in entries)
    ]
    problems += [
        f"stale entry {entry}: no violation matches it any more; remove it in the step that removed the violation"
        for entry in entries
        if not any(known(entry, violation) for violation in violations)
    ]
    return problems


def rules(violations):
    return [violation.rule for violation in violations]


def relayed(delegation_id, inner):
    return {"type": "response.event", "event_id": "event_1", "delegation_id": delegation_id, "event": inner}


def created(response_id):
    return {"type": "response.created", "response": {"id": response_id}}


def completed(response_id):
    return {"type": "response.completed", "response": {"id": response_id}}


def call_done(call_id="call_1"):
    item = {"type": "function_call", "call_id": call_id, "name": "get_menu", "arguments": "{}"}
    return {"type": "response.output_item.done", "item": item}


class StrictClientRuleTests(unittest.TestCase):
    """The checker itself, on short streams written by hand: each rule fires where tau2 would stop or stall."""

    def test_a_delegation_with_a_client_tool_round_breaks_no_rule(self):
        events = [
            {"type": "session.started", "session": {"id": "live_1"}},
            {"type": "session.input_transcript.delta", "delta": "Teas?", "start_ms": 0, "end_ms": 600},
            {"type": "session.delegation.created", "delegation": {"id": "item_1"}},
            relayed("item_1", created("resp_1")),
            relayed("item_1", call_done()),
            relayed("item_1", completed("resp_1")),
            relayed("item_1", created("resp_2")),
            relayed("item_1", completed("resp_2")),
            {"type": "session.output_transcript.delta", "delta": "Matcha.", "start_ms": 900, "end_ms": 1300},
            {"type": "session.closed", "reason": "close_requested"},
        ]
        self.assertEqual(StrictClient.read(events, asked_to_close_at=len(events) - 1), [])

    def test_any_top_level_error_ends_the_run(self):
        error = {"type": "error", "error": {"code": "missing_tool_results", "message": "Return every result."}}
        self.assertEqual(rules(StrictClient.read([error])), ["error"])

    def test_a_session_closed_the_client_did_not_ask_for_ends_the_run(self):
        closed = {"type": "session.closed", "reason": "expired"}
        self.assertEqual(rules(StrictClient.read([closed])), ["unexpected_close"])
        self.assertEqual(StrictClient.read([closed], asked_to_close_at=0), [])

    def test_a_relayed_failure_ends_the_run(self):
        for kind in ("response.failed", "response.incomplete", "error"):
            with self.subTest(kind=kind):
                events = [relayed("item_1", created("resp_1")), relayed("item_1", {"type": kind})]
                self.assertEqual(rules(StrictClient.read(events)), ["nested_failure"])

    def test_a_function_call_needs_a_response_created_of_its_own_delegation_before_it(self):
        self.assertEqual(rules(StrictClient.read([relayed("item_1", call_done())])), ["call_before_created"])
        elsewhere = [relayed("item_2", created("resp_1")), relayed("item_1", call_done())]
        self.assertEqual(rules(StrictClient.read(elsewhere)), ["call_before_created"])

    def test_response_completed_continues_only_the_calls_filed_under_its_id(self):
        # tau2 files each call under its delegation's latest response.created id, and continues a response's calls
        # only when a response.completed carries that id.
        retried = [
            relayed("item_1", created("resp_1")),
            relayed("item_1", created("resp_2")),
            relayed("item_1", call_done()),
        ]
        self.assertEqual(
            rules(StrictClient.read([*retried, relayed("item_1", completed("resp_1"))])),
            ["stale_completed", "uncontinued_calls"],
        )
        self.assertEqual(StrictClient.read([*retried, relayed("item_1", completed("resp_2"))]), [])
        # An older response's response.completed continues the calls filed under it.
        older = [
            relayed("item_1", created("resp_1")),
            relayed("item_1", call_done()),
            relayed("item_1", created("resp_2")),
        ]
        self.assertEqual(StrictClient.read([*older, relayed("item_1", completed("resp_1"))]), [])
        # A response.completed under which no call waits stops nothing, whatever its id, even with no response.created
        # of its own. (The empty one that answers a late continuation has its own: G16 holds it.)
        answered = [
            relayed("item_1", created("resp_1")),
            relayed("item_1", call_done()),
            relayed("item_1", completed("resp_1")),
        ]
        self.assertEqual(StrictClient.read([*answered, relayed("item_1", completed("resp_2"))]), [])
        self.assertEqual(StrictClient.read([relayed("item_1", completed("resp_1"))]), [])
        # Response ids are kept per delegation: another delegation's newer response leaves this one current.
        apart = [
            relayed("item_1", created("resp_1")),
            relayed("item_1", call_done()),
            relayed("item_2", created("resp_2")),
        ]
        self.assertEqual(StrictClient.read([*apart, relayed("item_1", completed("resp_1"))]), [])

    def test_a_call_whose_response_never_completes_is_never_continued(self):
        # For example a call whose response never gets a terminal on the wire: tau2 never sends its response.create,
        # and the call waits until the server's tool wait runs out.
        events = [relayed("item_1", created("resp_1")), relayed("item_1", call_done())]
        self.assertEqual(rules(StrictClient.read(events)), ["uncontinued_calls"])
        relayed_end = [*events, relayed("item_1", {"type": "response.incomplete"})]
        self.assertEqual(rules(StrictClient.read(relayed_end)), ["nested_failure", "uncontinued_calls"])

    def test_every_field_tau2_reads_must_be_there(self):
        for event in (
            {"delta": "no type"},
            {"type": "response.event", "event": created("resp_1")},
            relayed("item_1", {"type": "response.created"}),
            relayed("item_1", {"type": "response.output_item.done"}),
            {"type": "session.output_transcript.delta", "delta": "Hi.", "start_ms": 0},
            {"type": "session.input_transcript.delta", "delta": "Hi.", "start_ms": 1.5, "end_ms": 300},
            {"type": "session.input_transcript.delta", "event_id": 7, "delta": "Hi.", "start_ms": 0, "end_ms": 300},
        ):
            with self.subTest(event=event):
                self.assertEqual(rules(StrictClient.read([event])), ["missing_field"])
        for item in (
            {"type": "function_call"},
            {"type": "function_call", "call_id": "call_1", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_1", "name": "get_menu", "arguments": None},
        ):
            with self.subTest(item=item):
                done = relayed("item_1", {"type": "response.output_item.done", "item": item})
                complete = relayed("item_1", completed("resp_1"))
                self.assertEqual(
                    rules(StrictClient.read([relayed("item_1", created("resp_1")), done, complete])), ["missing_field"]
                )

    def test_values_are_parsed_as_tau2s_models_parse_them(self):
        # tau2's models parse in pydantic's lax mode: an integer may come as text, a whole float or a
        # boolean, and an input delta may have no event id. A call's name and arguments are read only when tau2 first
        # passes the call to its agent, and a null name is allowed (T2:447-457; events.py:62-66).
        lax = [
            {"type": "session.input_transcript.delta", "delta": "Hi.", "start_ms": "0", "end_ms": 300.0},
            {"type": "session.output_transcript.delta", "delta": "Hi.", "start_ms": True, "end_ms": 300},
        ]
        self.assertEqual(StrictClient.read(lax), [])
        first = {"type": "function_call", "call_id": "call_1", "name": None, "arguments": "{}"}
        again = {"type": "function_call", "call_id": "call_1"}
        events = [
            relayed("item_1", created("resp_1")),
            relayed("item_1", {"type": "response.output_item.done", "item": first}),
            relayed("item_1", {"type": "response.output_item.done", "item": again}),
            relayed("item_1", completed("resp_1")),
        ]
        self.assertEqual(StrictClient.read(events), [])

    def test_a_recorded_golden_is_read_channel_by_channel(self):
        golden = {
            "channels": {
                "session": [{"type": "session.closed", "reason": CLIENT_CLOSE_REASON}],
                "delegations": {
                    "<item_1>": [
                        relayed("<item_1>", created("<resp_1>")),
                        relayed("<item_1>", call_done()),
                        relayed("<item_1>", completed("<resp_1>")),
                    ]
                },
                "input_transcript": ["Hi."],
                "output_transcript": [
                    {"type": "session.output_transcript.delta", "delta": "Hello.", "start_ms": CLOCK}
                ],
            }
        }
        violations = read_recorded(golden)
        self.assertEqual(
            [(v.channel, v.index, v.rule) for v in violations], [("output_transcript", 0, "missing_field")]
        )
        golden["channels"]["session"][0]["reason"] = "expired"
        self.assertEqual(rules(read_recorded(golden)), ["unexpected_close", "missing_field"])
        # The golden ends at quiet: a call still filed then is one tau2 would never continue.
        golden["channels"]["delegations"]["<item_1>"].pop()
        self.assertEqual(
            [(v.channel, v.index, v.rule) for v in read_recorded(golden)][-1],
            ("delegations/<item_1>", 1, "uncontinued_calls"),
        )


class StrictClientGoldenTests(unittest.TestCase):
    """Every recorded golden, read as tau2's Live client reads a call."""

    def test_every_recorded_golden_passes_or_each_violation_is_known(self):
        paths = sorted(GOLDEN_DIR.glob("*.json"))
        self.assertLessEqual({"G1", "G2"}, {path.stem for path in paths}, f"the pilot goldens are not in {GOLDEN_DIR}")
        for path in paths:
            with self.subTest(golden=path.stem):
                golden = json.loads(path.read_text(encoding="utf-8"))
                self.assertIsInstance(golden.get("channels"), dict, f"{path.name} has no channels")
                # A call through the gateway has all four event channels; a UI-bot golden (X5, X7) has none to read.
                present = [channel for channel in EVENT_CHANNELS if channel in golden["channels"]]
                self.assertIn(present, ([], list(EVENT_CHANNELS)), f"{path.name} lacks an event channel")
                problems = unexplained(path.stem, read_recorded(golden))
                if problems:
                    self.fail(f"tau2's Live client would stop on {path.name}:\n" + "\n".join(problems))

    def test_each_listed_violation_names_a_recorded_golden_and_why_it_is_there(self):
        tables = (
            (KNOWN_VIOLATIONS, KNOWN_ROWS),
            (PROVOKED_REFUSALS, {*KNOWN_ROWS, None}),
            (SCRIPTED_ENDINGS, {None}),
        )
        for table, rows in tables:
            for name, entries in table.items():
                self.assertTrue((GOLDEN_DIR / f"{name}.json").exists(), f"an entry names a missing golden {name}")
                for entry in entries:
                    with self.subTest(golden=name, entry=entry):
                        self.assertEqual(set(entry), {"rule", "channel", "contains", "row", "note"})
                        self.assertIn(entry["rule"], RULES)
                        self.assertIn(entry["row"], rows)
                        self.assertTrue(entry["note"].strip())
        for name, entries in PROVOKED_REFUSALS.items():
            for entry in entries:
                # A provoked refusal answers one client command, named by its event id.
                self.assertEqual(entry["rule"], "error")
                self.assertIsInstance(entry["contains"]["error"]["client_event_id"], str, name)


class StrictClientCallTests(GoldenCase):
    """A whole call through the real gateway and engine, answered with tau2's own timing."""

    async def test_a_client_that_answers_and_continues_as_early_as_tau2_does_is_never_refused(self):
        await self.serve(
            frontend=[decision("delegate", "Let me check."), decision("speak", "We have Earl Grey and Matcha.")],
            backend=[tool_call_round(), answer_round("We have Earl Grey and Matcha.")],
        )
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        # tau2 returns a call's output as soon as its tool has run: here at once, which can be before the
        # round's response.completed reaches the client.
        self.send(result_item(call["call_id"], TEAS))
        # It asks to continue while it handles that response.completed, with no event id. The IM marks the work
        # waiting for the client before it writes that terminal, so the continue is never refused.
        await self.wait_for_response_event("response.completed")
        self.send({"type": "response.create"})
        await self.wait_quiet()
        asked = len(self.ws.from_server)
        await self.close()
        self.check_scripts_used()
        self.assertEqual(self.tts.utterances, ["Let me check.", "We have Earl Grey and Matcha."])
        violations = StrictClient.read(self.ws.from_server, asked_to_close_at=asked)
        self.assertEqual(violations, [], [str(violation) for violation in violations])


# ---------------------------------------------------------------------------------------- the console's reads
# The browser console polls GET /v1/live/sessions/{id}/history every second for its history panel (``refreshHistory`` in
# app.js), which shows what the talker and the backend hold (``historyView`` in history.js). A talker message that
# starts with this prefix shows as application context (history.js); the server writes context appends with it.
CONSOLE_CONTEXT_PREFIX = "Application context (reference data): "


class HistoryPanel:
    """Reads a /history snapshot as the console's history panel does (``historyView`` in history.js).

    ``lines`` holds each value the panel reads, in the panel's order: the talker's model, prompt parts and messages,
    then the backend's model, busy flag, prompt parts, items and pending entries. A content is kept as the panel shows
    it (``contentText``): a text, or each part's text, a part without text whole. An item is kept by what
    the panel reads for its type (``backendEntry``); an item with no type shows as a message.
    ``problems`` names each field the panel reads that is missing or of a type it cannot show.
    """

    def __init__(self, snapshot):
        self.lines: list[dict] = []
        self.problems: list[str] = []
        if self.expect("snapshot", snapshot, dict):
            self.frontend(snapshot.get("frontend"))
            self.backend(snapshot.get("backend"))

    def read(self, name, value):
        self.lines.append({"read": name, "value": value})

    def expect(self, where, value, kind):
        if isinstance(value, kind):
            return True
        self.problems.append(f"{where}: expected {kind.__name__}, got {value!r}"[:300])
        return False

    def frontend(self, frontend):
        # The talker's model, prompt and messages; the first message, the system prompt, is counted but not shown.
        if not self.expect("frontend", frontend, dict):
            return
        if self.expect("frontend.model", frontend.get("model"), str):
            self.read("frontend.model", frontend["model"])
        self.prompt("frontend.prompt", frontend.get("prompt"))
        messages = frontend.get("messages")
        if not self.expect("frontend.messages", messages, list):
            return
        for number, message in enumerate(messages):
            where = f"frontend.messages[{number}]"
            if self.expect(where, message, dict) and self.expect(f"{where}.role", message.get("role"), str):
                content = self.content(where, message.get("content"))
                self.read("frontend.messages", {"role": message["role"], "content": content})

    def backend(self, backend):
        # No backend shows "No server backend in this session." (``historyView``).
        if backend is None:
            self.read("backend", None)
            return
        if not self.expect("backend", backend, dict):
            return
        if self.expect("backend.model", backend.get("model"), str):
            self.read("backend.model", backend["model"])
        if self.expect("backend.busy", backend.get("busy"), bool):  # " · working"
            self.read("backend.busy", backend["busy"])
        self.prompt("backend.prompt", backend.get("prompt"))
        items = backend.get("items")
        if self.expect("backend.items", items, list):
            for number, item in enumerate(items):
                self.item(f"backend.items[{number}]", item)
        # The labeled lines the next delegation carries (``pendingEntry`` in history.js): "Caller: ...", "Agent: ...".
        pending = backend.get("pending")
        if not self.expect("backend.pending", pending, list):
            return
        for number, line in enumerate(pending):
            if self.expect(f"backend.pending[{number}]", line, str):
                self.read("backend.pending", line)

    def prompt(self, name, parts):
        # ``promptEntries``; a part's source also says whose it is (``promptOrigin``).
        if not self.expect(name, parts, list):
            return
        for number, part in enumerate(parts):
            where = f"{name}[{number}]"
            if self.expect(where, part, dict) and self.strings(where, part, ("source", "text")):
                self.read(name, {"source": part["source"], "text": part["text"]})

    def item(self, where, item):
        if not self.expect(where, item, dict):
            return
        kind = item.get("type")
        if kind in (None, "message"):
            role = item.get("role")  # no role shows as "message"
            if role is None or self.expect(f"{where}.role", role, str):
                content = self.content(where, item.get("content"))
                self.read("backend.items", {"type": "message", "role": role, "content": content})
        elif kind == "reasoning":
            summary = item.get("summary") or []
            if self.expect(f"{where}.summary", summary, list):
                texts = [part.get("text") if isinstance(part, dict) else part for part in summary]
                self.read("backend.items", {"type": kind, "summary": texts})
        elif kind in ("function_call", "function_call_output"):
            keys = ("call_id", "name", "arguments") if kind == "function_call" else ("call_id", "output")
            if self.strings(where, item, keys):
                self.read("backend.items", {"type": kind, **{key: item[key] for key in keys}})
        else:
            self.read("backend.items", {"type": kind, "item": item})  # shown whole

    def strings(self, where, value, keys):
        return all([self.expect(f"{where}.{key}", value.get(key), str) for key in keys])

    def content(self, where, content):
        if isinstance(content, str) or not self.expect(f"{where}.content", content, list):
            return content
        return [part["text"] if isinstance(part, dict) and part.get("text") is not None else part for part in content]


def values(lines, name):
    return [line["value"] for line in lines if line["read"] == name]


def recorded_diffs(name, expected, recorded):
    return [
        channel_diff(name, channel, expected.get(channel), recorded.get(channel))
        for channel in sorted(set(expected) | set(recorded))
        if canonical(expected.get(channel)) != canonical(recorded.get(channel))
    ]


# ------------------------------------------------------------------------------------------- the close order
# What the close has done when session.closed reaches the client, on every close path (close steps 3 to 5 of
# ``live/session/lifecycle.py``). The session leaves the registry once the final event is out. The backend's model
# client belongs to the IM session, which makes it at its first backend round and closes it (``LlmAdapter.close``), so
# it closes only after a backend round ran (:data:`AFTER_A_BACKEND_ROUND`).
AT_THE_FINAL_EVENT = {
    "pipeline stopped": True,
    "worker closes": 1,
    "frontend client closes": 1,  # one close owner per side
    "backend client closes": 0,
    "still registered": True,
    "socket closed": False,
}
AFTER_A_BACKEND_ROUND = {**AT_THE_FINAL_EVENT, "backend client closes": 1}
ENGINE_CLOSE = ("pipeline stopped", "worker closes", "frontend client closes", "backend client closes")
# The steps of the parts' close, in order. It starts once the session's own tasks have ended. The
# adapter's close comes first (close step 4): it stops the pipeline, then closes the cascade (the talker's model
# client). Then the delegation part's close ends the conversation in the core, whose IM session closes (and with it the
# provider of a backend round that ran).
ENGINE_CLOSE_ORDER = (
    "engine close starts",
    "pipeline stopped",
    "frontend client closes",
    "worker closes",
    "engine close ends",
)
# Then the final event, then the client's socket closes.
CLOSE_ORDER = (*ENGINE_CLOSE_ORDER, "session.closed", "socket closed")
_WORKER_CLOSED = CLOSE_ORDER.index("worker closes") + 1
AFTER_A_BACKEND_ROUND_ORDER = (*CLOSE_ORDER[:_WORKER_CLOSED], "backend client closes", *CLOSE_ORDER[_WORKER_CLOSED:])


class LeavingWebSocket(RecordingWebSocket):
    """A primary WebSocket whose client can leave without session.close, as a closed browser tab does."""

    LEAVE = "\x00leave"  # not JSON, so no client event can look like it

    def leave(self):
        self.to_server.put_nowait(self.LEAVE)

    async def receive(self):
        frame = await super().receive()
        if frame.get("text") == self.LEAVE:
            return {"type": "websocket.disconnect", "code": 1001}  # going away, as a closed tab's socket closes
        return frame


class CloseWatch:
    """Counts what the engine closes and notes each step of the close in order; shared by every session of the test.

    The spies call the original methods unchanged. ``steps`` holds the engine's close as ``ENGINE_CLOSE_ORDER`` names
    it; the test adds the final event and the socket's close. ``running_at_engine_close`` names the session's tasks
    that were still running when the engine's close started.
    """

    def __init__(self, test):
        self.closes = Counter()
        self.steps: list[str] = []
        self.running_at_engine_close: list[str] | None = None
        worker_close, engine_close = IMSession.close, lifecycle.close_parts
        stop_pipeline = PipelineHost.stop_pipeline

        async def counted_worker_close(worker):
            self.closes["worker"] += 1
            self.steps.append("worker closes")
            await worker_close(worker)

        async def watched_engine_close(session):
            # The session names its tasks "<session id>:<name>" (live/session/session.py).
            tasks = session.tasks
            running = [task.get_name().split(":", 1)[-1] for task in tasks if not task.done()]
            self.running_at_engine_close = sorted(running)
            self.steps.append("engine close starts")
            await engine_close(session)
            self.steps.append("engine close ends")

        async def watched_stop_pipeline(engine):
            await stop_pipeline(engine)
            self.steps.append("pipeline stopped")

        test.enterContext(patch.object(IMSession, "close", counted_worker_close))
        test.enterContext(patch.object(lifecycle, "close_parts", watched_engine_close))
        test.enterContext(patch.object(PipelineHost, "stop_pipeline", watched_stop_pipeline))

    def count(self, client, name):
        """Count the closes of a model client, which its owner calls by the instance's ``close``."""
        close = client.close

        async def counted():
            self.closes[name] += 1
            self.steps.append(f"{name} closes")
            await close()

        client.close = counted

    def engine_state(self, engine):
        """What the engine's close has done so far.

        With ``CheckedCall.close_state`` and the spies above, the only reads of internals in this file: re-point them
        when a change moves what they read (the pipeline's task, the engine's close, the worker).
        """
        runner = engine.runner_task
        return {
            "pipeline stopped": runner is not None and runner.done(),
            "worker closes": self.closes["worker"],
            "frontend client closes": self.closes["frontend client"],
            "backend client closes": self.closes["backend client"],
        }


class CheckedCall(GoldenCase):
    """A golden call whose gateway also serves its HTTP routes, and whose close is watched step by step.

    Its model clients are :class:`HoldingChat`, so a backend step may be a :class:`HeldRound`; a held round waits for
    the test, as a client-tool round waits for the client, so the call can go quiet while it is held.
    """

    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.close_watch = CloseWatch(self)
        self.final_state = None  # what the close had done when session.closed reached the client
        self.events_at_socket_close = None

    async def serve_checked(self, *, websocket=RecordingWebSocket, **kwargs):
        """``serve`` on a gateway that also serves its HTTP routes, with ``websocket`` as the client's socket.

        The gateway is built by the live router, as ``mount_live`` builds it. Once the session has started, the model
        clients' closes are counted and the socket's close is noted.
        """
        built = []

        def routed_gateway(factory, settings, *, core=None, ice_servers=None):
            router, gateway = create_live_router(factory, settings, core=core, ice_servers=ice_servers)
            self.app = FastAPI()
            self.app.include_router(router)
            built.append(gateway)
            return gateway

        # serve builds these by their names in the harness module; if that changes, re-point these patches.
        with (
            patch.object(harness, "LiveGateway", routed_gateway),
            patch.object(harness, "RecordingWebSocket", websocket),
            patch.object(harness, "ScriptedChat", HoldingChat),
        ):
            started = await self.serve(**kwargs)
        self.assertEqual(built, [self.gateway])
        self.assertIs(type(self.ws), websocket)
        self.session_id = started["session"]["id"]
        self.close_watch.count(self.frontend, "frontend client")
        self.close_watch.count(self.backend, "backend client")
        socket_close = self.ws.close

        async def noted_close(code=1000):
            self.events_at_socket_close = len(self.ws.from_server)
            self.close_watch.steps.append("socket closed")
            await socket_close(code)

        self.ws.close = noted_close
        return started

    def engine_pending(self):
        """The harness's reasons, except busy work while a backend round is held for the test."""
        reasons = super().engine_pending()
        if self.backend.holding:
            reasons = [reason for reason in reasons if reason != WORK_RUNNING]
        return reasons

    def _changed(self):
        super()._changed()
        # The client's socket reports each event as it is sent, so the close's state is read as session.closed goes out.
        events = getattr(self, "ws", None) and self.ws.from_server
        if events and self.final_state is None and events[-1]["type"] == "session.closed":
            self.final_state = self.close_state()
            self.close_watch.steps.append("session.closed")

    def close_state(self):
        return {
            **self.close_watch.engine_state(self.engine),
            "still registered": self.session_id in self.gateway.sessions,
            "socket closed": self.ws.closed,
        }

    async def assert_close_order(
        self,
        reason,
        client_event_id=None,
        *,
        asked_to_close_at=None,
        tau2_stops_on=(),
        backend_ran=False,
        steps=None,
        still_running=(),
    ):
        """The close order every path keeps; ``tau2_stops_on`` lists the strict client's violations, in order.

        ``backend_ran`` says the call's backend ran a round, whose model client the IM session's close closes.
        ``steps`` is the order of the close's steps, when it is not the path's own; ``still_running`` names the
        session's tasks still running when the engine's close starts: none, or the task that asked for the close, which
        the close keeps running.
        """
        try:
            async with asyncio.timeout(WAIT_SECONDS):
                await self.server
        except TimeoutError:
            self.fail("The gateway loop did not end after the close")
        events = self.ws.from_server
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds.count("session.closed"), 1, "session.closed must come once")
        self.assertEqual(kinds[-1], "session.closed", "session.closed must be the last event")
        # An optional field is omitted when it has no value: session.closed has client_event_id only when the
        # client's session.close asked for it.
        self.assertEqual((events[-1]["reason"], events[-1].get("client_event_id")), (reason, client_event_id))
        self.assertEqual(self.final_state, AFTER_A_BACKEND_ROUND if backend_ran else AT_THE_FINAL_EVENT)
        steps = steps or (AFTER_A_BACKEND_ROUND_ORDER if backend_ran else CLOSE_ORDER)
        self.assertEqual(self.close_watch.steps, list(steps))
        self.assertEqual(self.close_watch.running_at_engine_close, list(still_running))
        # The socket closed after the final event was sent, and nothing was sent after it closed.
        self.assertTrue(self.ws.closed)
        self.assertEqual(self.events_at_socket_close, len(events))
        self.assertNotIn(self.session_id, self.gateway.sessions)
        violations = StrictClient.read(events, asked_to_close_at=asked_to_close_at)
        self.assertEqual(rules(violations), list(tau2_stops_on), [str(violation) for violation in violations])

    # ------------------------------------------------------------------------------------- the history route
    async def get_history(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://test") as http:
            return await http.get(f"/v1/live/sessions/{self.session_id}/history")

    async def panel(self):
        """What the history panel reads from /history now; fails on any field it reads that is missing or mistyped."""
        response = await self.get_history()
        self.assertEqual(response.status_code, 200, response.text)
        panel = HistoryPanel(response.json())
        self.assertEqual(panel.problems, [], "the console cannot read /history")
        return panel.lines

    def assert_recorded(self, name, channels):
        """Compare ``channels`` with ``recordings/<name>.json``, or record them, as ``assert_golden`` does a call's."""
        recorded = {channel: Normalizer()(lines) for channel, lines in channels.items()}
        path = GOLDEN_DIR / f"{name}.json"
        golden = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if os.environ.get(RECORD_VARIABLE) == "1":
            previous, golden = (
                golden,
                {
                    "name": name,
                    "scenario": self.scenario(),
                    "tags": golden["tags"] if golden else [],
                    "channels": recorded,
                },
            )
            path.write_text(json.dumps(golden, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            diffs = recorded_diffs(name, previous["channels"], recorded) if previous else []
            if diffs:
                self.fail(
                    f"Recorded {path.name} again: {len(diffs)} channel(s) changed, and the file now holds this run. "
                    f"Check the diff, tag the changed lines, then run without {RECORD_VARIABLE}.\n\n"
                    + "\n\n".join(diffs)
                )
        elif golden is None:
            self.fail(f"{path} does not exist; record it with {RECORD_VARIABLE}=1")
        else:
            diffs = recorded_diffs(name, golden["channels"], recorded)
            if diffs:
                self.fail(
                    f"{name} changed in {len(diffs)} channel(s). If the change is intended, name it (a row and a "
                    f"note, see harness.py), record again with {RECORD_VARIABLE}=1 and tag the changed lines.\n\n"
                    + "\n\n".join(diffs)
                )
        self.check_tags(name, golden)


class ConsoleHistoryTests(CheckedCall):
    async def test_the_history_panel_through_a_reply_a_client_tool_round_and_the_answer(self):
        """console-history: what the console's history panel reads from /history at four moments of one call.

        The real route serves each read. The moments are the start; after a thinking append and a direct reply; while
        the backend waits for the client's tool output; and after the answer. Each moment is a channel, and each line
        one value the panel reads (``historyView``).
        """
        await self.serve_checked(
            frontend=[
                decision("speak", "Hello there."),
                decision("delegate", "Let me check."),
                decision("speak", "We have Earl Grey and Matcha."),
            ],
            backend=[tool_call_round(), answer_round("We have Earl Grey and Matcha.")],
        )
        await self.wait_quiet()
        moments = {"at_start": await self.panel()}
        context = {"type": "session.thinking.append", "event_id": "context", "delegation_id": None}
        self.send({**context, "content": "The caller is a regular."})
        self.feed_silence(6)  # its acknowledgment waits for 200 ms of caller audio
        await self.wait_for("session.thinking.appended")
        await self.say("Hi.")
        await self.wait_quiet()
        moments["after_a_reply"] = await self.panel()
        await self.say("What teas do you have?")
        call = await self.wait_for_call("get_menu")
        await self.wait_quiet()
        moments["while_the_backend_waits_for_the_client"] = await self.panel()
        self.send(result_item(call["call_id"], TEAS))
        self.send({"type": "response.create", "event_id": "continue"})
        await self.wait_quiet()
        moments["after_the_answer"] = await self.panel()
        await self.close()
        # The panel stops polling once the session is gone (``refreshHistory``).
        self.assertEqual((await self.get_history()).status_code, 404)
        self.check_scripts_used()

        # What the panel branches on, checked by meaning as well as recorded.
        self.assertEqual(values(moments["at_start"], "frontend.messages"), [])  # the prompt is its own part
        talker = values(moments["after_a_reply"], "frontend.messages")
        self.assertIn({"role": "user", "content": CONSOLE_CONTEXT_PREFIX + "The caller is a regular."}, talker)
        self.assertEqual(
            values(moments["after_a_reply"], "backend.pending"),
            ["App context: The caller is a regular.", "Caller: Hi.", "Agent: Hello there."],
        )
        self.assertEqual(
            [values(lines, "backend.busy") for lines in moments.values()], [[False], [False], [True], [False]]
        )
        self.assertEqual(
            [item["type"] for item in values(moments["after_the_answer"], "backend.items")],
            ["message", "function_call", "function_call_output", "message"],
        )
        self.assert_recorded("console-history", moments)


class ConsoleWebRTCTests(unittest.IsolatedAsyncioTestCase):
    """The console's own transport: a WebRTC session with the cascade over loopback, through the real HTTP routes.

    The console (``connect`` in app.js) and tau2 open a session the same way: they post an SDP offer with the
    session config, read the 201 answer, then exchange events on the data channel.
    """

    async def asyncSetUp(self):
        # The usage report is a clock (harness.USAGE_INTERVAL_SECONDS): an hour keeps it off the data channel.
        self.enterContext(patch.object(lifecycle, "USAGE_INTERVAL_SECONDS", USAGE_INTERVAL_SECONDS))
        self.close_watch = CloseWatch(self)
        self.engines: list[CascadeAdapter] = []
        self.frontend = ScriptedChat("frontend", decision("speak", "Hello there."))
        self.backend = ScriptedChat("backend")
        self.close_watch.count(self.frontend, "frontend client")
        self.close_watch.count(self.backend, "backend client")
        self.stt = FakeSTT()
        self.events: list[dict] = []
        self.arrived = asyncio.Event()

    async def factory(self, config, transport, connection):
        parts = cascade_parts(
            config,
            transport,
            connection,
            chat=self.frontend,
            stt=self.stt,
            tts=FakeTTS(),
            live_config=GOLDEN_LIVE_CONFIG,
        )
        self.engines.append(parts.frontend)
        return parts

    def mount(self, app):
        """The live routes on ``app``, with the gateway's one core on the backend's script, as the root mounts them."""
        core = chat_core(self.backend, **GOLDEN_CORE)
        self.addAsyncCleanup(core.aclose)
        return mount_live(app, factory=self.factory, core=core_hook(core))

    async def wait_for(self, kind, *, after=0, where=None):
        async with asyncio.timeout(WAIT_SECONDS):
            while True:
                for event in self.events[after:]:
                    if event["type"] == kind and (where is None or where(event)):
                        return event
                self.arrived.clear()
                await self.arrived.wait()

    async def test_the_answer_the_history_and_the_close_as_the_console_and_tau2_read_them(self):
        app = FastAPI()
        gateway = self.mount(app)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.addAsyncCleanup(http.aclose)
        client = RTCPeerConnection()
        self.addAsyncCleanup(client.close)
        client.addTrack(AudioStreamTrack())
        channel = client.createDataChannel("oai-events")  # the console's label (``connect``)
        final_state, channel_closed = {}, asyncio.Event()

        @channel.on("message")
        def on_message(raw):
            event = json.loads(raw)
            self.events.append(event)
            self.arrived.set()
            if event["type"] == "session.closed":
                self.close_watch.steps.append("session.closed")
                final_state.update(self.close_watch.engine_state(self.engines[0]))

        @channel.on("close")
        def on_close():
            self.close_watch.steps.append("channel closed")
            channel_closed.set()

        await client.setLocalDescription(await client.createOffer())
        offer = {"type": "webrtc", "sdp": client.localDescription.sdp}
        response = await http.post("/v1/live/sessions", json={"session": RESPONSES_SESSION, "transport": offer})
        self.assertEqual(response.status_code, 201, response.text)  # response.ok (``connect``)
        answer = response.json()
        session = answer["session"]
        self.assertIsInstance(session["id"], str)  # logged by the console, kept by tau2
        # The answer is the official one, the session's id and the SDP answer, which the console and
        # tau2 set as the remote answer. The resolved configuration comes on session.started.
        sdp = answer["transport"]["sdp"]
        self.assertIsInstance(sdp, str)
        self.assertEqual(answer, {"session": {"id": session["id"]}, "transport": {"type": "webrtc", "sdp": sdp}})
        await client.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="answer"))

        # The console takes the session id for /history from session.started, and keeps its session as the config the
        # server started with: the delegation cards show its backend model.
        started = await self.wait_for("session.started")
        self.assertEqual(started["session"]["id"], session["id"])
        sent_model = RESPONSES_SESSION["delegation"]["responses"]["model"]
        self.assertEqual(started["session"]["delegation"]["responses"]["model"], sent_model)
        history = f"/v1/live/sessions/{session['id']}/history"
        response = await http.get(history)
        self.assertEqual(response.status_code, 200, response.text)
        panel = HistoryPanel(response.json())
        self.assertEqual(panel.problems, [])
        self.assertEqual((values(panel.lines, "backend.busy"), values(panel.lines, "backend.items")), ([False], []))
        await self.stt.say("Hi.")
        await self.wait_for("session.output_transcript.delta", where=lambda event: "there." in event["delta"])
        self.assertEqual(HistoryPanel((await http.get(history)).json()).problems, [])

        asked = len(self.events)
        channel.send(json.dumps({"type": "session.close", "event_id": "bye"}))
        closed = await self.wait_for("session.closed", after=asked)
        async with asyncio.timeout(WAIT_SECONDS):
            await channel_closed.wait()
        # session.closed came once and last, after the engine closed everything in order (the talker's model
        # client once, see ENGINE_CLOSE_ORDER), and before the data channel closed.
        self.assertEqual(self.close_watch.steps, [*ENGINE_CLOSE_ORDER, "session.closed", "channel closed"])
        self.assertEqual(self.close_watch.running_at_engine_close, [])
        kinds = [event["type"] for event in self.events]
        self.assertEqual((kinds.count("session.closed"), kinds[-1]), (1, "session.closed"))
        self.assertEqual((closed["reason"], closed["client_event_id"]), ("close_requested", "bye"))
        self.assertEqual(final_state, {key: AT_THE_FINAL_EVENT[key] for key in ENGINE_CLOSE})
        # The session left the registry before its channel closed, so the history panel gets its 404
        # (``refreshHistory``).
        self.assertEqual(gateway.sessions, {})
        self.assertEqual((await http.get(history)).status_code, 404)
        self.assertEqual((self.frontend.script, self.frontend.overrun, self.backend.requests), ([], [], []))
        violations = StrictClient.read(self.events, asked_to_close_at=asked)
        self.assertEqual(violations, [], [str(violation) for violation in violations])

    async def test_a_refused_offer_says_why_in_a_field_the_console_shows(self):
        # The console shows result.error?.message || result.detail (``connect``). Nothing is negotiated here.
        app = FastAPI()
        gateway = self.mount(app)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            offer = {"type": "webrtc", "sdp": "not an sdp"}
            no_audio = await http.post("/v1/live/sessions", json={"session": RESPONSES_SESSION, "transport": offer})
            no_model = await http.post(
                "/v1/live/sessions", json={"session": {"instructions": "Hi."}, "transport": offer}
            )
        self.assertEqual(no_audio.status_code, 400)
        self.assertEqual(no_audio.json(), {"detail": "The SDP offer must include an audio track"})
        self.assertEqual(no_model.status_code, 400)
        self.assertIn("model", no_model.json()["error"]["message"])
        self.assertEqual((self.engines, gateway.sessions), ([], {}))


class ConsoleInfoTests(unittest.IsolatedAsyncioTestCase):
    """``GET /v1/live/info`` as the console reads it to fill its form (``applyInfo``, ``updateBackendBadge``).

    The route is ``live/extras.py``'s. The values come from the deployment's config, so only what the console reads is
    pinned.
    """

    async def test_info_has_every_field_the_console_reads(self):
        # Mounted as the server mounts it, with the registry's hooks (the root's describe()), the bundled files and no
        # token to send; the console's host is allowed. The catalog's self-hosted section is named outright and
        # no NVIDIA Cloud key is set; the service catalog would dial each endpoint to see which ones are up
        # (utils.is_endpoint_reachable), and none counts as up here, so nothing is dialed.
        live = {"LIVE_ENABLED": "true", "LIVE_API_TOKEN": "", "LIVE_ALLOWED_HOSTS": "test"}
        self.enterContext(patch.dict(os.environ, {**live, "SERVICE_RECIPE": "server", "NVIDIA_API_KEY": ""}))
        for name in ("SERVICES_PATH", "LIVE_CONFIG_PATH", "CORE_CONFIG_PATH", "LIVE_ADAPTER_FACTORY"):
            os.environ.pop(name, None)
        self.enterContext(patch.object(utils, "is_endpoint_reachable", lambda server: False))
        # create_app binds the selected example's services process-wide; leave no binding behind.
        self.addCleanup(utils.clear_service_context)
        self.addCleanup(utils.set_active_services, None)
        clear_root_caches(self)
        app = server.create_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
            response = await http.get("/v1/live/info")
        self.assertEqual(response.status_code, 200, response.text)
        info = response.json()

        def strings(value):
            return isinstance(value, list) and all(isinstance(item, str) for item in value)

        # The voice list fills a select and must hold at least one voice (.includes and [0], ``applyInfo``).
        self.assertTrue(strings(info.get("voices")) and info["voices"], info.get("voices"))
        # The backend models offered as choices (``applyInfo``); each warning is logged as text.
        self.assertTrue(strings(info.get("backend_models")), info.get("backend_models"))
        self.assertTrue(strings(info.get("warnings")), info.get("warnings"))
        # The texts the form and the badges show (``applyInfo``, ``updateBackendBadge``).
        for path in (
            ("backend", "model"),
            ("backend", "provider"),
            ("frontend", "model"),
            ("frontend", "provider"),
            ("prompt_version",),
            ("instructions", "frontend"),
            ("instructions", "backend"),
        ):
            value = info
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            with self.subTest(field=".".join(path)):
                self.assertIsInstance(value, str)
        # The sample tools fill the tools box as JSON (``applyInfo``), which the console sends back as the session's
        # tools.
        tools = info.get("sample_tools")
        self.assertTrue(isinstance(tools, list) and tools, tools)
        for tool in tools:
            self.assertIsInstance(tool.get("name") if isinstance(tool, dict) else None, str, tool)


class CloseOrderTests(CheckedCall):
    """session.closed comes once and last, after the engine closed everything in order, and before the socket closes."""

    async def test_a_client_close_after_a_reply(self):
        await self.serve_checked(frontend=[decision("speak", "Hello there.")])
        await self.say("Hi.")
        await self.wait_quiet()
        asked = len(self.ws.from_server)
        await self.close("bye")
        await self.assert_close_order("close_requested", "bye", asked_to_close_at=asked)

    async def test_a_client_close_while_the_backend_waits_for_the_clients_tool_output(self):
        await self.serve_checked(frontend=[decision("delegate", "Let me check.")], backend=[tool_call_round()])
        await self.say("What teas do you have?")
        await self.wait_for_call("get_menu")
        await self.wait_quiet()
        asked = len(self.ws.from_server)
        await self.close("bye")
        # Nothing more is said about the open delegation: its round has completed, and the worker is cancelled
        # while it waits for the output.
        later = [event["type"] for event in self.ws.from_server[asked:] if event["type"] != AUDIO_EVENT]
        self.assertEqual(later, ["session.closed"])
        await self.assert_close_order("close_requested", "bye", asked_to_close_at=asked, backend_ran=True)

    async def test_a_client_close_while_a_backend_round_runs(self):
        # The backend round is held: its request is sent and its opening events are relayed, and the model has not
        # answered when the client closes. The IM session's close cancels the round, after the pipeline has stopped,
        # and ends the delegation's open response as the close's end (incomplete, status cancelled), which the
        # gateway relays before session.closed (G13 pins the delegation's events). tau2 stops on that relayed
        # ending (SCRIPTED_ENDINGS: G13).
        held = HeldRound()
        held.on_cancel = lambda: self.close_watch.steps.append("backend round cancelled")
        await self.serve_checked(frontend=[decision("delegate", "Let me check.")], backend=[held])
        await self.say("What teas do you have?")
        await self.wait_until(lambda: held.waiting.is_set(), "the backend round to wait for its model")
        await self.wait_quiet()
        asked = len(self.ws.from_server)
        await self.close("bye")
        later = [event for event in self.ws.from_server[asked:] if event["type"] != AUDIO_EVENT]
        self.assertEqual([event["type"] for event in later], ["response.event", "session.closed"])
        ending = later[0]["event"]
        self.assertEqual(
            (ending["type"], ending["response"]["status"], ending["response"]["incomplete_details"]),
            ("response.incomplete", "cancelled", None),
        )
        self.assertEqual((held.cancelled, held.closed), (True, True))
        cancelled_at = AFTER_A_BACKEND_ROUND_ORDER.index("worker closes") + 1
        steps = [
            *AFTER_A_BACKEND_ROUND_ORDER[:cancelled_at],
            "backend round cancelled",
            *AFTER_A_BACKEND_ROUND_ORDER[cancelled_at:],
        ]
        await self.assert_close_order(
            "close_requested",
            "bye",
            asked_to_close_at=asked,
            tau2_stops_on=["nested_failure"],
            backend_ran=True,
            steps=steps,
        )

    async def test_an_append_is_acknowledged_at_once_so_the_close_refuses_nothing(self):
        # An append before any caller audio is acknowledged at once, as a zero-length span at the input clock;
        # no acknowledgment waits for audio, so the close has nothing to refuse.
        await self.serve_checked()
        self.send(
            {"type": "session.thinking.append", "event_id": "early", "delegation_id": None, "content": "A regular."}
        )
        await self.wait_quiet()
        appended = [event for event in self.ws.from_server if event["type"] == "session.thinking.appended"]
        self.assertEqual(
            [(event["client_event_id"], event["start_ms"], event["end_ms"]) for event in appended], [("early", 0, 0)]
        )
        asked = len(self.ws.from_server)
        await self.close("bye")
        later = [event["type"] for event in self.ws.from_server[asked:] if event["type"] != AUDIO_EVENT]
        self.assertEqual(later, ["session.closed"])
        await self.assert_close_order("close_requested", "bye", asked_to_close_at=asked)

    async def test_a_client_that_leaves_without_closing(self):
        await self.serve_checked(frontend=[decision("speak", "Hello there.")], websocket=LeavingWebSocket)
        await self.say("Hi.")
        await self.wait_quiet()
        left = len(self.ws.from_server)
        self.ws.leave()
        # A client whose socket closes cleanly (1000, or 1001 going away) hung up (remote_hangup).
        await self.assert_close_order("remote_hangup", asked_to_close_at=left)

    async def test_a_session_that_expires(self):
        # Test-local timers: the usage report, which also checks the session's age, comes every 50 ms instead of every
        # 5 s, and the session's lifetime is set to zero once it has started, so the next report closes it.
        self.usage_interval_seconds = 0.05
        await self.serve_checked()
        session = self.gateway.sessions[self.session_id]
        session.max_session_seconds = 0
        await self.wait_for("session.closed")
        # tau2 stops on a close it did not ask for, by its own rule. The task that asked for the close, which the
        # usage report spawned (live/session/lifecycle.py), is the one session task still running then.
        await self.assert_close_order("expired", tau2_stops_on=["unexpected_close"], still_running=["expiry"])
        # The usage report as the console reads it (app.js): usage.seconds, and context_window.usage_ratio, which
        # comes from the adapter's own count of the talker's context out of the talker's own window (the ContextMeter),
        # not from the fallbacks, the size of the session's config out of the protocol's window
        # (lifecycle.context_ratio). The call has no turns, so the count does not move.
        reports = [event for event in self.ws.from_server if event["type"] == "session.usage.updated"]
        self.assertTrue(reports, "the usage report that found the session expired")
        window = self.engine.context_window_tokens()
        self.assertNotIn(window, (None, lifecycle.CONTEXT_WINDOW_TOKENS))
        meter = min(self.engine.context_tokens() / window, 1)
        fallback = min(approx_tokens(json.dumps(session.config.model_dump())) / lifecycle.CONTEXT_WINDOW_TOKENS, 1)
        self.assertNotEqual(meter, fallback)
        for report in reports:
            seconds = report["usage"]["seconds"]
            self.assertTrue(isinstance(seconds, int | float) and not isinstance(seconds, bool), seconds)
            self.assertEqual(report["context_window"]["usage_ratio"], meter)

    async def test_a_server_shutdown(self):
        await self.serve_checked(frontend=[decision("speak", "Hello there.")])
        await self.say("Hi.")
        await self.wait_quiet()
        await self.gateway.close_all()
        # A server shutdown is reported as connection_lost, the nearest official reason. tau2
        # stops on a close it did not ask for.
        await self.assert_close_order("connection_lost", tau2_stops_on=["unexpected_close"])

    async def test_a_pipeline_that_ends_on_its_own(self):
        await self.serve_checked(frontend=[decision("speak", "Hello there.")])
        await self.say("Hi.")
        await self.wait_quiet()
        # The pipeline ends from inside, as its worker ends it when a processor cannot do its job
        # (ProcessorUnusablePolicy.END queues an EndFrame). The pipeline host's on_pipeline_finished handler then asks
        # the session to close, and the close runs in order: the cascade closes once, in the adapter's close. The task
        # that asked for the close is the one session task still running when the parts' close starts, and tau2 stops on
        # a close it did not ask for.
        await self.engine.task.queue_frame(EndFrame())
        await self.assert_close_order(
            "connection_lost", tau2_stops_on=["unexpected_close"], still_running=["adapter-close"]
        )


if __name__ == "__main__":
    unittest.main()
