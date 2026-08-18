# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import asyncio
import json
import sqlite3
import unittest
from contextlib import suppress
from datetime import date, timedelta
from typing import Any
from unittest.mock import patch

from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.services.llm_service import FunctionCallParams

from examples.frontend_backend_agent import pipeline as frontend_backend_pipeline
from examples.frontend_backend_agent.airline.airports import spoken_time
from examples.frontend_backend_agent.airline.database.api import BookingAPI
from examples.frontend_backend_agent.airline.database.db import apply_schema
from examples.frontend_backend_agent.airline.state import MAX_LIFECYCLE_EVENTS, ThinkerSessionState
from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.airline.tools import CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL
from examples.frontend_backend_agent.airline.transform import _server_booking_to_record, _server_flight_to_option
from examples.frontend_backend_agent.src.planner import NvidiaThinkerPlanner
from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent, is_speakable_payload
from examples.frontend_backend_agent.src.runtime_context import runtime_today
from examples.frontend_backend_agent.src.tool_handlers import (
    _emit_talker_response,
    _normalize_arguments,
    build_handlers,
)
from examples.frontend_backend_agent.src.tts_filter import FrontendBackendAgentPronunciationTextFilter
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter


def _today() -> date:
    return date(2026, 5, 25)


class _StructuredTestPlanner:
    async def plan(self, *, query: str, slots: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if _filled(slots.get("pnr_code")):
            return {"tool": "pnr_status", "params": dict(slots)}
        if _has_route_fields(slots):
            return {"tool": "flight_search", "params": dict(slots)}
        if state.get("waiting_for_confirmation") or state.get("waiting_for_preferences"):
            return {"tool": "booking", "params": dict(slots)}
        if state.get("search_results") and _filled(slots.get("flight_selected")):
            return {"tool": "booking", "params": dict(slots)}
        if _filled(slots.get("flight_selected")):
            return {"tool": "booking", "params": dict(slots)}
        return {
            "tool": "response_hint",
            "reason": "unsupported_request",
            "action": "answer_directly",
            "context": "general",
            "response_text": "I can help search flights, book a selected flight, or check a PNR status.",
        }


class _StaticPlanner:
    def __init__(self, plan: dict[str, Any]) -> None:
        self._plan = plan

    async def plan(self, *, query: str, slots: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        return self._plan


_TEST_CITY_NAMES = {
    "JFK": "New York",
    "SEA": "Seattle",
    "SFO": "San Francisco",
    "LAX": "Los Angeles",
}

_TEST_FLIGHT_TEMPLATES = [
    {
        "flight_id": "AA311",
        "carrier": "Booking Server",
        "origin": "JFK",
        "destination": "SEA",
        "depart_time": "07:30",
        "arrive_time": "11:15",
        "duration_minutes": 375,
        "price_usd": 219,
    },
    {
        "flight_id": "AA315",
        "carrier": "Delta",
        "origin": "JFK",
        "destination": "SEA",
        "depart_time": "15:00",
        "arrive_time": "18:50",
        "duration_minutes": 410,
        "price_usd": 241,
    },
    {
        "flight_id": "AA2080",
        "carrier": "United",
        "origin": "SFO",
        "destination": "LAX",
        "depart_time": "06:30",
        "arrive_time": "07:50",
        "duration_minutes": 80,
        "price_usd": 84,
    },
    {
        "flight_id": "AA2081",
        "carrier": "American",
        "origin": "SFO",
        "destination": "LAX",
        "depart_time": "12:45",
        "arrive_time": "14:05",
        "duration_minutes": 80,
        "price_usd": 91,
    },
]


class _TestBookingBackend:
    def __init__(self) -> None:
        today = _today().isoformat()
        self._records: dict[str, dict[str, Any]] = {
            "ABC123": {
                "pnr": "ABC123",
                "passenger_name": "Ava Chen",
                "status": "confirmed",
                "flight_id": "AA311",
                "carrier": "United",
                "origin_city": "New York",
                "dest_city": "Seattle",
                "date": today,
                "seat_pref": "14A",
                "meal_pref": "vegetarian",
            },
            "GHI789": {
                "pnr": "GHI789",
                "passenger_name": "Maria Garcia",
                "status": "cancelled_weather",
                "flight_id": "AA789",
                "carrier": "Booking Server",
                "origin_city": "Atlanta",
                "dest_city": "Miami",
                "date": today,
                "seat_pref": "28F",
                "meal_pref": None,
            },
        }
        self._next_pnr_index = 1

    async def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        travel_date: str,
        sorting: str | None = None,
    ) -> list[dict[str, Any]]:
        matches = [
            _materialize_test_flight(template, travel_date)
            for template in _TEST_FLIGHT_TEMPLATES
            if template["origin"] == origin and template["destination"] == destination
        ]
        if sorting == "duration":
            matches.sort(key=lambda item: item["duration_minutes"])
        elif sorting == "departure_time":
            matches.sort(key=lambda item: item["departure_time"])
        else:
            matches.sort(key=lambda item: item["price_usd"])
        return matches[:5]

    async def create_booking(
        self,
        *,
        passenger_name: str | None,
        flight: dict[str, Any],
        seat_pref: str | None = None,
        meal_pref: str | None = None,
    ) -> dict[str, Any] | None:
        pnr = f"TTK{self._next_pnr_index:03d}"
        self._next_pnr_index += 1
        record = {
            "pnr": pnr,
            "passenger_name": passenger_name or "Guest",
            "status": "confirmed",
            "flight_id": flight["flight_id"],
            "carrier": flight.get("carrier", ""),
            "origin_city": flight["origin_city"],
            "dest_city": flight["dest_city"],
            "date": flight["date"],
            "departure_time": flight["departure_time"],
            "seat_pref": seat_pref,
            "meal_pref": meal_pref,
            "price_usd": flight.get("price_usd"),
        }
        self._records[pnr] = record
        return record

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        return self._records.get(pnr_code.strip().upper())


class _ConcurrentBackend(_TestBookingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.search_started = asyncio.Event()
        self.pnr_started = asyncio.Event()

    async def search_flights(self, **kwargs):
        self.search_started.set()
        await asyncio.wait_for(self.pnr_started.wait(), timeout=1)
        return await super().search_flights(**kwargs)

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        self.pnr_started.set()
        await asyncio.wait_for(self.search_started.wait(), timeout=1)
        return await super().get_pnr(pnr_code)


class _FailingBookingBackend(_TestBookingBackend):
    async def create_booking(self, **kwargs):
        raise RuntimeError("booking backend unavailable")

    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        raise RuntimeError("pnr backend unavailable")


class _PartiallyFailingBackend(_TestBookingBackend):
    async def get_pnr(self, pnr_code: str) -> dict[str, Any] | None:
        raise RuntimeError("pnr backend unavailable")


class _BlockingCreateBackend(_TestBookingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.create_started = asyncio.Event()

    async def create_booking(self, **kwargs):
        self.create_started.set()
        await asyncio.sleep(10)
        return await super().create_booking(**kwargs)


class _RaisingThinker:
    async def call(self, query: str, slots: dict[str, Any] | None = None, *, on_started=None) -> dict[str, Any]:
        raise RuntimeError("thinker exploded")

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        return False

    def cancel_pending_booking(self) -> bool:
        return False


class _FrameCapturingLLM:
    def __init__(self) -> None:
        self.frames = []

    async def push_frame(self, frame, direction=None) -> None:
        self.frames.append(frame)


class _InferenceCapturingLLM:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def run_inference(self, context, max_tokens=None) -> str:
        self.messages = list(context.get_messages())
        return json.dumps(
            {
                "tool": "response_hint",
                "reason": "unsupported_request",
                "action": "answer_directly",
                "context": "general",
                "response_text": "I can help with flights.",
            }
        )


class _CancellingAfterStartLLM(_FrameCapturingLLM):
    async def push_frame(self, frame, direction=None) -> None:
        self.frames.append(frame)
        if isinstance(frame, LLMFullResponseStartFrame):
            task = asyncio.current_task()
            if task is not None:
                task.cancel()
        await asyncio.sleep(0)


class _DoubleStartedThinker:
    async def call(self, query: str, slots: dict[str, Any] | None = None, *, on_started=None) -> dict[str, Any]:
        if on_started:
            event = ThinkerLifecycleEvent(marker="ThinkerStarted", call_id="call_double", query=query)
            await on_started(event)
            await on_started(event)
        await asyncio.sleep(0.05)
        return {
            "type": "tool_result",
            "tool": "flight_search",
            "status": "success",
            "data": {},
            "response_text": "Done.",
            "context": "flight_search",
        }

    def cancel_active(self, reason: str = "new_user_query") -> bool:
        return False


def _materialize_test_flight(template: dict[str, Any], travel_date: str) -> dict[str, Any]:
    return {
        "flight_id": template["flight_id"],
        "carrier": template["carrier"],
        "origin_city": _TEST_CITY_NAMES[template["origin"]],
        "dest_city": _TEST_CITY_NAMES[template["destination"]],
        "origin_airport": template["origin"],
        "dest_airport": template["destination"],
        "date": travel_date,
        "departure_time": f"{travel_date}T{template['depart_time']}:00",
        "arrival_time": f"{travel_date}T{template['arrive_time']}:00",
        "duration_minutes": template["duration_minutes"],
        "price_usd": template["price_usd"],
    }


def _make_thinker(**kwargs) -> ThinkerBackend:
    kwargs.setdefault("planner", _StructuredTestPlanner())
    kwargs.setdefault("backend", _TestBookingBackend())
    return ThinkerBackend(**kwargs)


def _filled(value: Any) -> bool:
    return value is not None and str(value).strip() != ""


def _has_route_fields(slots: dict[str, Any]) -> bool:
    return any(
        _filled(slots.get(key))
        for key in (
            "origin_city",
            "origin_airport",
            "dest_city",
            "dest_airport",
            "date",
            "sorting",
        )
    )


class FrontendBackendPipelineConfigTests(unittest.TestCase):
    def test_booking_backend_url_uses_explicit_env_override(self) -> None:
        with patch.dict(
            "os.environ",
            {"BOOKING_BACKEND_URL": "http://custom.example:8001", "APP_RUNTIME": ""},
            clear=True,
        ):
            url = frontend_backend_pipeline._booking_backend_url({"server": "http://booking-server:8001"})

        self.assertEqual(url, "http://custom.example:8001")

    def test_booking_backend_url_rewrites_docker_hostname_for_host_native(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            url = frontend_backend_pipeline._booking_backend_url({"server": "http://booking-server:8001"})

        self.assertEqual(url, "http://localhost:8001")

    def test_booking_backend_url_preserves_container_docker_hostname(self) -> None:
        with patch.dict("os.environ", {"APP_RUNTIME": "container"}, clear=True):
            url = frontend_backend_pipeline._booking_backend_url({"server": "http://booking-server:8001"})

        self.assertEqual(url, "http://booking-server:8001")

    def test_booking_backend_url_preserves_custom_catalog_url(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            url = frontend_backend_pipeline._booking_backend_url({"server": "http://booking.internal:8001"})

        self.assertEqual(url, "http://booking.internal:8001")


class FrontendBackendAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_runtime_today_uses_frontend_backend_agent_override(self) -> None:
        with patch.dict("os.environ", {"FRONTEND_BACKEND_AGENT_TODAY": "2026-04-30"}):
            self.assertEqual(runtime_today(), date(2026, 4, 30))

    def test_runtime_today_rejects_invalid_override(self) -> None:
        for value in ("04/30/2026", "20260430", "2026-W18-4", "2026-02-30"):
            with (
                self.subTest(value=value),
                patch.dict("os.environ", {"FRONTEND_BACKEND_AGENT_TODAY": value}),
                self.assertRaisesRegex(ValueError, "YYYY-MM-DD"),
            ):
                runtime_today()

    def test_spoken_time_preserves_ten_oclock_hours(self) -> None:
        self.assertEqual(spoken_time("2026-05-26T10:05:00"), "10:05 AM")
        self.assertEqual(spoken_time("2026-05-26T22:45:00"), "10:45 PM")
        self.assertEqual(spoken_time("not-a-timestamp"), "not-a-timestamp")

    def test_server_flight_options_compute_duration_minutes(self) -> None:
        daytime = _server_flight_to_option(
            {
                "flight_number": "AA2116",
                "origin": "JFK",
                "destination": "SEA",
                "departure": "2026-04-22T08:00:00",
                "arrival": "2026-04-22T13:57:00",
            },
            fallback_date="2026-05-26",
        )
        overnight = _server_flight_to_option(
            {
                "flight_number": "AA2117",
                "origin": "JFK",
                "destination": "SEA",
                "departure": "2026-04-22T23:30:00",
                "arrival": "2026-04-22T01:00:00",
            },
            fallback_date="2026-05-26",
        )
        epoch = _server_flight_to_option(
            {"flight_number": "AA2118", "origin": "JFK", "destination": "SEA", "departure": 0, "arrival": 5400},
            fallback_date="2026-05-26",
        )

        self.assertEqual(daytime["duration_minutes"], 357)
        self.assertEqual(overnight["duration_minutes"], 90)
        self.assertEqual(epoch["duration_minutes"], 90)

    async def test_tts_filter_strips_asterisks_and_keeps_base_cleanup(self) -> None:
        filtered = await NemotronSpeechTextFilter().filter("PNR **ABC123** <break> {AA123}")

        self.assertEqual(filtered, "PNR ABC123 break> AA123")

    async def test_tts_filter_expands_airline_codes_after_base_cleanup(self) -> None:
        base_filtered = await NemotronSpeechTextFilter().filter(
            "- Your PNR's status for **ABC123** on flight AA2072 from JFK to SFO."
        )
        filtered = await FrontendBackendAgentPronunciationTextFilter().filter(base_filtered)

        self.assertEqual(
            filtered,
            "Your P N R code status for A B C 1 2 3 on flight A A 2 0 7 2 from John F Kennedy to San Francisco.",
        )

    async def test_thinker_planner_uses_direct_today_for_runtime_context(self) -> None:
        llm = _InferenceCapturingLLM()
        planner = NvidiaThinkerPlanner(
            llm=llm,
            system_prompt="You are a planner.",
        )
        today = date(2026, 4, 30)
        tomorrow = today + timedelta(days=1)

        with patch.dict("os.environ", {"FRONTEND_BACKEND_AGENT_TODAY": today.isoformat()}):
            await planner.plan(query="Search flights tomorrow", slots={}, state={})

        self.assertIn(f"Today is {today.isoformat()}.", llm.messages[0]["content"])
        self.assertIn(f"Tomorrow is {tomorrow.isoformat()}.", llm.messages[0]["content"])
        user_payload = json.loads(llm.messages[1]["content"])
        self.assertEqual(
            user_payload["runtime_context"],
            {"today": today.isoformat(), "tomorrow": tomorrow.isoformat()},
        )

    async def test_thinker_started_is_internal_only_while_response_hint_is_speakable(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call("Book flight 1", slots={"flight_selected": "1"})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["action"], "req_flight_search")
        self.assertTrue(is_speakable_payload(payload))
        started = thinker.state.lifecycle_events[0]
        self.assertEqual(started.marker, "ThinkerStarted")
        self.assertFalse(started.speakable)
        self.assertNotIn("ThinkerStarted", payload["response_text"])

    def test_lifecycle_history_is_bounded(self) -> None:
        state = ThinkerSessionState()

        for index in range(MAX_LIFECYCLE_EVENTS + 3):
            state.add_event(ThinkerLifecycleEvent(marker="ThinkerStarted", call_id=str(index)))

        self.assertEqual(len(state.lifecycle_events), MAX_LIFECYCLE_EVENTS)
        self.assertEqual(state.lifecycle_events[0].call_id, "3")

    async def test_flight_search_then_selected_flight_booking_happy_path(self) -> None:
        thinker = _make_thinker()

        search = await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        self.assertEqual(search["type"], "tool_result")
        self.assertEqual(search["tool"], "flight_search")
        self.assertEqual(search["data"]["search_context"]["origin_airport"], "JFK")
        self.assertEqual(search["data"]["search_context"]["dest_airport"], "SEA")
        self.assertEqual(search["data"]["search_context"]["date"], "2026-05-26")
        self.assertEqual(len(search["data"]["flights"]), 2)
        self.assertIn("AA311", search["response_text"])
        self.assertIn("AA315", search["response_text"])
        self.assertIn("7:30 AM", search["response_text"])
        self.assertIn("G Force Airline's AA311", search["response_text"])
        self.assertNotIn("Booking Server", search["response_text"])
        self.assertEqual(search["data"]["flights"][0]["carrier"], "G Force Airline's")

        preferences = await thinker.call(
            "I want flight 1",
            slots={"flight_selected": "1"},
        )
        self.assertEqual(preferences["type"], "response_hint")
        self.assertEqual(preferences["reason"], "params_optional")
        self.assertEqual(preferences["action"], "req_params")

        confirm = await thinker.call(
            "Window seat and vegetarian meal",
            slots={"seat_pref": "window", "meal_pref": "vegetarian"},
        )
        self.assertEqual(confirm["type"], "response_hint")
        self.assertEqual(confirm["reason"], "confirm_required")
        self.assertEqual(confirm["action"], "confirm_booking")
        self.assertIn("AA311", confirm["response_text"])
        self.assertIn("G Force Airline's AA311", confirm["response_text"])
        self.assertNotIn("Booking Server", confirm["response_text"])

        booked = await thinker.call("Yes, confirm", slots={"confirmed": True})
        self.assertEqual(booked["type"], "tool_result")
        self.assertEqual(booked["tool"], "booking")
        self.assertEqual(booked["data"]["pnr"], "TTK001")
        self.assertEqual(booked["data"]["booking"]["seat_pref"], "window")
        self.assertEqual(booked["data"]["booking"]["meal_pref"], "vegetarian")

    async def test_booking_confirmation_merges_preferences_from_same_turn(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        confirm = await thinker.call(
            "Book the first one with defaults",
            slots={"flight_selected": "1", "use_defaults": True},
        )
        self.assertEqual(confirm["type"], "response_hint")
        self.assertEqual(confirm["reason"], "confirm_required")

        booked = await thinker.call(
            "Actually make it a window seat with vegetarian meal and confirm",
            slots={
                "seat_pref": "window",
                "meal_pref": "vegetarian",
                "confirmed": True,
            },
        )

        self.assertEqual(booked["type"], "tool_result")
        self.assertEqual(booked["tool"], "booking")
        self.assertEqual(booked["data"]["booking"]["seat_pref"], "window")
        self.assertEqual(booked["data"]["booking"]["meal_pref"], "vegetarian")

    async def test_booking_confirmation_updates_selected_flight_before_finalizing(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call(
            "I want flight 1",
            slots={"flight_selected": "1"},
        )
        first_confirm = await thinker.call(
            "Window seat and vegetarian meal",
            slots={"seat_pref": "window", "meal_pref": "vegetarian"},
        )
        self.assertEqual(first_confirm["summary"]["flight_id"], "AA311")

        updated_confirm = await thinker.call(
            "Actually make it the second one",
            slots={"flight_selected": "2"},
        )
        self.assertEqual(updated_confirm["type"], "response_hint")
        self.assertEqual(updated_confirm["reason"], "confirm_required")
        self.assertEqual(updated_confirm["summary"]["flight_id"], "AA315")
        self.assertIn("AA315", updated_confirm["response_text"])
        self.assertNotIn("AA311", updated_confirm["response_text"])

        booked = await thinker.call("Yes, confirm", slots={"confirmed": True})

        self.assertEqual(booked["type"], "tool_result")
        self.assertEqual(booked["tool"], "booking")
        self.assertEqual(booked["data"]["booking"]["flight_id"], "AA315")
        self.assertEqual(booked["data"]["booking"]["seat_pref"], "window")
        self.assertEqual(booked["data"]["booking"]["meal_pref"], "vegetarian")

    async def test_waiting_for_preferences_name_update_stays_in_preferences_state(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call("I want flight 1", slots={"flight_selected": "1"})

        payload = await thinker.call("The passenger name is Ava Chen", slots={"passenger_name": "Ava Chen"})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["reason"], "params_optional")
        self.assertEqual(payload["action"], "req_params")
        self.assertEqual(thinker.state.booking_draft.passenger_name, "Ava Chen")
        self.assertTrue(thinker.state.waiting_for_preferences)
        self.assertFalse(thinker.state.waiting_for_confirmation)

    async def test_flight_change_while_waiting_for_preferences_keeps_preferences_prompt(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call("I want flight 1", slots={"flight_selected": "1"})

        payload = await thinker.call("Actually make it the second flight", slots={"flight_selected": "2"})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["reason"], "params_optional")
        self.assertEqual(thinker.state.booking_draft.flight["flight_id"], "AA315")
        self.assertTrue(thinker.state.waiting_for_preferences)
        self.assertFalse(thinker.state.waiting_for_confirmation)

    async def test_confirmation_includes_user_provided_passenger_name_only(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        guest_confirm = await thinker.call(
            "Book the first one with defaults", slots={"flight_selected": "1", "use_defaults": True}
        )
        self.assertNotIn("passenger_name", guest_confirm["summary"])
        self.assertNotIn("Guest", guest_confirm["response_text"])

        thinker = _make_thinker()
        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        named_confirm = await thinker.call(
            "Book the first one for Ava Chen with defaults",
            slots={"flight_selected": "1", "passenger_name": "Ava Chen", "use_defaults": True},
        )

        self.assertEqual(named_confirm["summary"]["passenger_name"], "Ava Chen")
        self.assertIn("Ava Chen", named_confirm["response_text"])

    async def test_invalid_flight_selection_during_confirmation_clears_draft(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call(
            "Book the first one with defaults",
            slots={"flight_selected": "1", "use_defaults": True},
        )

        invalid = await thinker.call(
            "Actually make it flight 99",
            slots={"flight_selected": "99"},
        )
        self.assertEqual(invalid["type"], "response_hint")
        self.assertEqual(invalid["reason"], "params_missing")
        self.assertEqual(invalid["action"], "req_params")
        self.assertIn("could not match", invalid["response_text"])
        self.assertIsNone(thinker.state.booking_draft)
        self.assertFalse(thinker.state.waiting_for_confirmation)

        replacement_confirm = await thinker.call(
            "Book the second one with defaults",
            slots={"flight_selected": "2", "use_defaults": True},
        )

        self.assertEqual(replacement_confirm["type"], "response_hint")
        self.assertEqual(replacement_confirm["reason"], "confirm_required")
        self.assertEqual(replacement_confirm["summary"]["flight_id"], "AA315")

    async def test_cancel_backend_clears_pending_search_and_booking_context(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call("I want flight 1", slots={"flight_selected": "1"})

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="cancel_backend",
            tool_call_id="cancel_test",
            arguments={},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker)["cancel_backend"](params)

        self.assertEqual(results[-1][0]["reason"], "cancelled")
        self.assertEqual(results[-1][0]["context"], "cancel_backend")
        self.assertEqual(thinker.state.search_results, [])
        self.assertEqual(thinker.state.search_context, {})
        self.assertIsNone(thinker.state.booking_draft)

        replacement_confirm = await thinker.call(
            "Book the second one with defaults",
            slots={"flight_selected": "2", "use_defaults": True},
        )

        self.assertEqual(replacement_confirm["type"], "response_hint")
        self.assertEqual(replacement_confirm["action"], "req_flight_search")

    async def test_new_flight_search_overrides_pending_booking_confirmation(self) -> None:
        thinker = _make_thinker()

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call(
            "Book the first one with defaults",
            slots={"flight_selected": "1", "use_defaults": True},
        )

        search = await thinker.call(
            "Now search flights from San Francisco to Los Angeles on 2026-06-09",
            slots={
                "origin_airport": "SFO",
                "dest_airport": "LAX",
                "date": "2026-06-09",
            },
        )

        self.assertEqual(search["type"], "tool_result")
        self.assertEqual(search["tool"], "flight_search")
        self.assertEqual(search["data"]["search_context"]["origin_airport"], "SFO")
        self.assertEqual(search["data"]["search_context"]["dest_airport"], "LAX")
        self.assertIsNone(thinker.state.booking_draft)
        self.assertFalse(thinker.state.waiting_for_confirmation)

    def test_call_backend_normalizes_wrapped_original_args(self) -> None:
        payload = {"original_args": ('{"query": "search flights", "origin_airport": "JFK", "dest_airport": "SEA"}')}

        normalized = _normalize_arguments(payload)

        self.assertEqual(normalized["query"], "search flights")
        self.assertEqual(normalized["origin_airport"], "JFK")
        self.assertEqual(normalized["dest_airport"], "SEA")

    def test_call_backend_schema_only_requires_query_and_omits_intent(self) -> None:
        parameters = CALL_BACKEND_TOOL["function"]["parameters"]

        self.assertEqual(CALL_BACKEND_TOOL["function"]["name"], "call_backend")
        self.assertEqual(parameters["required"], ["query"])
        self.assertFalse(parameters["additionalProperties"])
        self.assertEqual(set(parameters["properties"]), {"query", "filler_text"})
        self.assertNotIn("intent", parameters["properties"])
        self.assertIn("filler_text", parameters["properties"])
        self.assertNotIn("origin_airport", parameters["properties"])
        self.assertNotIn("pnr_code", parameters["properties"])

    def test_cancel_backend_schema_accepts_no_arguments(self) -> None:
        parameters = CANCEL_BACKEND_TOOL["function"]["parameters"]

        self.assertEqual(CANCEL_BACKEND_TOOL["function"]["name"], "cancel_backend")
        self.assertEqual(parameters["required"], [])
        self.assertEqual(parameters["properties"], {})
        self.assertFalse(parameters["additionalProperties"])

    async def test_call_backend_emits_started_filler_as_talker_text_when_threshold_is_met(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "filler_text": "I need to check the live booking tools for that.",
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker, filler_threshold_seconds=0)["call_backend"](params)

        self.assertEqual(len(llm.frames), 3)
        self.assertIsInstance(llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(llm.frames[1], LLMTextFrame)
        self.assertEqual(llm.frames[1].text, "I need to check the live booking tools for that.")
        self.assertIsNone(llm.frames[1].skip_tts)
        self.assertTrue(llm.frames[1].append_to_context)
        self.assertIsInstance(llm.frames[2], LLMFullResponseEndFrame)
        self.assertEqual(results[-1][0]["type"], "tool_result")
        markers = [event.marker for event in thinker.state.lifecycle_events]
        self.assertEqual(markers, ["ThinkerStarted", "IntermediateResponse", "ThinkerCompleted"])

    async def test_call_backend_direct_response_emits_talker_text_and_suppresses_llm_rerun(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        with patch.dict("os.environ", {"FRONTEND_BACKEND_DIRECT_TOOL_RESPONSE": "true"}):
            await build_handlers(thinker)["call_backend"](params)

        self.assertEqual(len(llm.frames), 3)
        self.assertIsInstance(llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(llm.frames[1], LLMTextFrame)
        self.assertEqual(llm.frames[1].text, results[-1][0]["response_text"])
        self.assertIsInstance(llm.frames[2], LLMFullResponseEndFrame)
        self.assertEqual(results[-1][0]["type"], "tool_result")
        self.assertFalse(results[-1][1].run_llm)

    async def test_call_backend_ignores_duplicate_started_events_for_filler(self) -> None:
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "filler_text": "I need to check the live booking tools for that.",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(_DoubleStartedThinker(), filler_threshold_seconds=0.01)["call_backend"](params)

        self.assertEqual(len(llm.frames), 3)
        self.assertIsInstance(llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(llm.frames[1], LLMTextFrame)
        self.assertEqual(llm.frames[1].text, "I need to check the live booking tools for that.")
        self.assertIsInstance(llm.frames[2], LLMFullResponseEndFrame)
        self.assertEqual(results[-1][0]["type"], "tool_result")

    async def test_call_backend_suppresses_started_filler_when_thinker_finishes_before_threshold(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "filler_text": "I need to check the live booking tools for that.",
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker, filler_threshold_seconds=1.0)["call_backend"](params)

        self.assertEqual(llm.frames, [])
        self.assertEqual(results[-1][0]["type"], "tool_result")

    async def test_call_backend_omits_started_filler_when_talker_does_not_provide_one(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker)["call_backend"](params)

        self.assertEqual(llm.frames, [])
        self.assertEqual(results[-1][0]["type"], "tool_result")

    async def test_call_backend_returns_result_callback_on_backend_exception(self) -> None:
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={"query": "Search flights from New York to Seattle tomorrow"},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(_RaisingThinker())["call_backend"](params)

        self.assertEqual(results[-1][0]["type"], "response_hint")
        self.assertEqual(results[-1][0]["reason"], "tool_error")
        self.assertIn("try again", results[-1][0]["response_text"].lower())

    async def test_cancel_backend_cancels_active_call_and_suppresses_stale_result(self) -> None:
        thinker = _make_thinker(tool_delay_seconds=1.0)
        llm = _FrameCapturingLLM()
        call_results = []
        cancel_results = []

        async def call_result_callback(result, *, properties=None) -> None:
            call_results.append((result, properties))

        async def cancel_result_callback(result, *, properties=None) -> None:
            cancel_results.append((result, properties))

        handlers = build_handlers(thinker, filler_threshold_seconds=10.0)
        call_params = FunctionCallParams(
            function_name="call_backend",
            tool_call_id="call_test",
            arguments={
                "query": "Search flights from New York to Seattle tomorrow",
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=call_result_callback,
        )
        cancel_params = FunctionCallParams(
            function_name="cancel_backend",
            tool_call_id="cancel_test",
            arguments={},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=cancel_result_callback,
        )

        running_call = asyncio.create_task(handlers["call_backend"](call_params))
        await asyncio.sleep(0)
        self.assertIsNotNone(thinker.state.active_task)

        await handlers["cancel_backend"](cancel_params)
        await running_call

        self.assertEqual(cancel_results[-1][0]["reason"], "cancelled")
        self.assertEqual(cancel_results[-1][0]["response_text"], "Okay, I stopped that.")
        self.assertEqual(call_results[-1][0]["reason"], "aborted")
        self.assertEqual(call_results[-1][0]["response_text"], "")
        self.assertFalse(call_results[-1][1].run_llm)
        self.assertEqual(llm.frames, [])
        self.assertIsNone(thinker.state.active_task)
        aborted = [event for event in thinker.state.lifecycle_events if event.marker == "ThinkerAborted"]
        self.assertEqual(len(aborted), 1)

    async def test_cancel_backend_reports_nothing_pending_without_starting_thinker(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="cancel_backend",
            tool_call_id="cancel_test",
            arguments={},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker)["cancel_backend"](params)

        self.assertEqual(results[-1][0]["reason"], "nothing_to_cancel")
        self.assertEqual(results[-1][0]["context"], "cancel_backend")
        self.assertEqual(results[-1][0]["response_text"], "There is nothing pending right now.")
        self.assertEqual(thinker.state.lifecycle_events, [])

    async def test_cancel_backend_direct_response_emits_talker_text_and_suppresses_llm_rerun(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="cancel_backend",
            tool_call_id="cancel_test",
            arguments={},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        with patch.dict("os.environ", {"FRONTEND_BACKEND_DIRECT_TOOL_RESPONSE": "true"}):
            await build_handlers(thinker)["cancel_backend"](params)

        self.assertEqual(len(llm.frames), 3)
        self.assertIsInstance(llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(llm.frames[1], LLMTextFrame)
        self.assertEqual(llm.frames[1].text, "There is nothing pending right now.")
        self.assertIsInstance(llm.frames[2], LLMFullResponseEndFrame)
        self.assertEqual(results[-1][0]["context"], "cancel_backend")
        self.assertFalse(results[-1][1].run_llm)

    async def test_cancel_backend_clears_pending_booking_confirmation(self) -> None:
        thinker = _make_thinker()
        llm = _FrameCapturingLLM()
        results = []

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call(
            "Book the first one with defaults",
            slots={"flight_selected": "1", "use_defaults": True},
        )

        async def result_callback(result, *, properties=None) -> None:
            results.append((result, properties))

        params = FunctionCallParams(
            function_name="cancel_backend",
            tool_call_id="cancel_test",
            arguments={},
            llm=llm,
            pipeline_worker=None,
            context=None,
            result_callback=result_callback,
        )

        await build_handlers(thinker)["cancel_backend"](params)

        self.assertEqual(results[-1][0]["reason"], "cancelled")
        self.assertEqual(results[-1][0]["response_text"], "Okay, I stopped that.")
        self.assertIsNone(thinker.state.booking_draft)
        self.assertFalse(thinker.state.waiting_for_preferences)
        self.assertFalse(thinker.state.waiting_for_confirmation)

    async def test_emit_talker_response_ends_response_when_cancelled_after_start(self) -> None:
        llm = _CancellingAfterStartLLM()

        with self.assertRaises(asyncio.CancelledError):
            await _emit_talker_response(llm, "I need to check the live booking tools for that.")

        self.assertEqual(len(llm.frames), 2)
        self.assertIsInstance(llm.frames[0], LLMFullResponseStartFrame)
        self.assertIsInstance(llm.frames[1], LLMFullResponseEndFrame)

    def test_booking_server_materializes_route_templates_onto_requested_date(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        apply_schema(conn)
        conn.execute(
            """
            INSERT INTO flights (flight_number, origin, destination, departure, arrival, cabin)
            VALUES ('AA2116', 'JFK', 'SEA', '2026-04-22T08:00:00', '2026-04-22T13:57:00', 'economy')
            """
        )
        api = BookingAPI(conn)

        flights = api.list_alternatives("JFK", "SEA", travel_date="2026-05-26")
        booking = api.create_booking(
            passenger="Guest",
            origin="JFK",
            destination="SEA",
            flight_number="AA2116",
            departure=flights[0]["departure"],
            seat="window",
            meal="vegetarian",
        )

        self.assertEqual(flights[0]["departure"], "2026-05-26T08:00:00")
        self.assertEqual(flights[0]["arrival"], "2026-05-26T13:57:00")
        self.assertIsNotNone(booking)
        if booking is None:
            self.fail("expected booking to be created")
        self.assertEqual(booking["departure"], "2026-05-26T08:00:00")
        record = api.get_pnr(booking["pnr"])
        self.assertIsNotNone(record)
        if record is None:
            self.fail("expected created PNR to be readable")
        self.assertEqual(record["departure"], "2026-05-26T08:00:00")
        self.assertEqual(record["ancillaries"]["seat"], "WINDOW")
        self.assertEqual(record["ancillaries"]["meal"], "VGML")

    def test_booking_server_rejects_invalid_departure_override(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        apply_schema(conn)
        conn.execute(
            """
            INSERT INTO flights (flight_number, origin, destination, departure, arrival, cabin)
            VALUES ('AA2116', 'JFK', 'SEA', '2026-04-22T08:00:00', '2026-04-22T13:57:00', 'economy')
            """
        )
        conn.commit()
        api = BookingAPI(conn)

        with self.assertRaises(ValueError):
            api.create_booking(
                passenger="Guest",
                origin="JFK",
                destination="SEA",
                flight_number="AA2116",
                departure="2026-05-26Tbad",
            )

        flights = conn.execute("SELECT departure FROM flights ORDER BY departure").fetchall()
        self.assertEqual([row["departure"] for row in flights], ["2026-04-22T08:00:00"])
        pnr_count = conn.execute("SELECT COUNT(*) AS n FROM pnrs").fetchone()
        self.assertEqual(pnr_count["n"], 0)

    def test_http_booking_record_uses_passenger_name_when_available(self) -> None:
        flight = _materialize_test_flight(_TEST_FLIGHT_TEMPLATES[0], "2026-05-26")

        record = _server_booking_to_record(
            {
                "pnr": "ABC123",
                "confirmation_code": "BOOK1234",
                "flight_number": "AA311",
                "departure": "2026-05-26T07:30:00",
            },
            flight=flight,
            passenger_name="Ava Chen",
            seat_pref=None,
            meal_pref=None,
        )
        fallback = _server_booking_to_record(
            {
                "pnr": "DEF456",
                "confirmation_code": "BOOK5678",
                "flight_number": "AA311",
                "departure": "2026-05-26T07:30:00",
            },
            flight=flight,
            passenger_name="",
            seat_pref=None,
            meal_pref=None,
        )

        self.assertEqual(record["passenger_name"], "Ava Chen")
        self.assertEqual(fallback["passenger_name"], "Guest")

    async def test_booking_selection_without_prior_search_requests_flight_search_first(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call("Book flight 1", slots={"flight_selected": "1"})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["context"], "booking")
        self.assertEqual(payload["action"], "req_flight_search")
        self.assertIn("search flights before booking", payload["response_text"].lower())
        self.assertIsNone(thinker.state.booking_draft)

    async def test_pnr_status_uses_thinker_tool_result(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call("Check PNR ABC123", slots={"pnr_code": "ABC123"})

        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["tool"], "pnr_status")
        self.assertEqual(payload["data"]["booking"]["pnr"], "ABC123")
        self.assertIn("confirmed", payload["response_text"])

    async def test_pnr_status_uses_planner_canonical_pnr_digits(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call("Check PNR ABC one two three", slots={"pnr_code": "ABC123"})

        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["data"]["booking"]["pnr"], "ABC123")

    async def test_pnr_status_uses_planner_canonical_spelled_pnr(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call("Check PNR G H I seven eight nine", slots={"pnr_code": "GHI789"})

        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["data"]["booking"]["pnr"], "GHI789")

    async def test_pnr_status_backend_error_returns_response_hint(self) -> None:
        thinker = _make_thinker(backend=_FailingBookingBackend())

        payload = await thinker.call("Check PNR ABC123", slots={"pnr_code": "ABC123"})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["reason"], "tool_error")
        self.assertIn("could not check", payload["response_text"].lower())

    async def test_booking_backend_error_returns_response_hint_without_hanging(self) -> None:
        thinker = _make_thinker(backend=_FailingBookingBackend())
        thinker.state.search_results = [_materialize_test_flight(_TEST_FLIGHT_TEMPLATES[0], "2026-05-26")]
        thinker.state.search_context = {"origin_airport": "JFK", "dest_airport": "SEA", "date": "2026-05-26"}

        await thinker.call("Book the first one with defaults", slots={"flight_selected": "1", "use_defaults": True})
        payload = await thinker.call("Yes, confirm", slots={"confirmed": True})

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["reason"], "tool_error")
        self.assertIn("could not complete", payload["response_text"].lower())

    async def test_flight_search_accepts_city_names_when_planner_omits_iata_codes(self) -> None:
        thinker = _make_thinker()

        payload = await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={"origin_city": "New York", "dest_city": "Seattle", "date": "2026-05-26"},
        )

        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["tool"], "flight_search")
        self.assertEqual(payload["data"]["search_context"]["origin_airport"], "JFK")
        self.assertEqual(payload["data"]["search_context"]["dest_airport"], "SEA")

    async def test_thinker_runs_independent_tool_calls_in_parallel(self) -> None:
        backend = _ConcurrentBackend()
        planner = _StaticPlanner(
            {
                "tool_calls": [
                    {
                        "tool": "flight_search",
                        "params": {
                            "origin_airport": "JFK",
                            "dest_airport": "SEA",
                            "date": "2026-05-26",
                        },
                    },
                    {"tool": "pnr_status", "params": {"pnr_code": "ABC123"}},
                ]
            }
        )
        thinker = ThinkerBackend(backend=backend, planner=planner)

        payload = await asyncio.wait_for(
            thinker.call("Search New York to Seattle and check PNR ABC123"),
            timeout=1,
        )

        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["tool"], "multi_tool")
        self.assertEqual(len(payload["data"]["results"]), 2)
        self.assertEqual(
            {result.get("tool") for result in payload["data"]["results"]},
            {"flight_search", "pnr_status"},
        )
        self.assertTrue(backend.search_started.is_set())
        self.assertTrue(backend.pnr_started.is_set())

    async def test_parallel_tool_calls_merge_successes_when_one_tool_fails(self) -> None:
        planner = _StaticPlanner(
            {
                "tool_calls": [
                    {
                        "tool": "flight_search",
                        "params": {
                            "origin_airport": "JFK",
                            "dest_airport": "SEA",
                            "date": "2026-05-26",
                        },
                    },
                    {"tool": "pnr_status", "params": {"pnr_code": "ABC123"}},
                ]
            }
        )
        thinker = ThinkerBackend(backend=_PartiallyFailingBackend(), planner=planner)

        payload = await thinker.call("Search New York to Seattle and check PNR ABC123")

        self.assertEqual(payload["type"], "response_hint")
        self.assertEqual(payload["context"], "multi_tool")
        self.assertEqual(len(payload["data"]["results"]), 2)
        self.assertEqual(payload["data"]["results"][0]["tool"], "flight_search")
        self.assertEqual(payload["data"]["results"][1]["reason"], "tool_error")

    async def test_abort_records_internal_marker_and_does_not_return_speakable_payload(self) -> None:
        thinker = _make_thinker(tool_delay_seconds=1.0)

        task = asyncio.create_task(
            thinker.call(
                "Search flights from New York to Seattle tomorrow",
                slots={
                    "origin_airport": "JFK",
                    "dest_airport": "SEA",
                    "date": "2026-05-26",
                },
            )
        )
        await asyncio.sleep(0)
        cancelled = thinker.cancel_active("new_user_query")

        self.assertTrue(cancelled)
        with self.assertRaises(asyncio.CancelledError):
            await task
        aborted = [event for event in thinker.state.lifecycle_events if event.marker == "ThinkerAborted"]
        self.assertEqual(len(aborted), 1)
        self.assertFalse(aborted[0].speakable)

    async def test_new_call_waits_for_previous_cancellation_before_started_callback(self) -> None:
        thinker = _make_thinker(tool_delay_seconds=0.1)
        first_call = asyncio.create_task(
            thinker.call(
                "Search flights from New York to Seattle tomorrow",
                slots={
                    "origin_airport": "JFK",
                    "dest_airport": "SEA",
                    "date": "2026-05-26",
                },
            )
        )
        await asyncio.sleep(0)
        self.assertIsNotNone(thinker.state.active_task)

        started_call_ids = []
        observed_active_call_ids = []

        async def on_started(event: ThinkerLifecycleEvent) -> None:
            started_call_ids.append(event.call_id)
            await asyncio.sleep(0)
            observed_active_call_ids.append(thinker.state.active_call_id)

        payload = await thinker.call(
            "Search flights from San Francisco to Los Angeles tomorrow",
            slots={
                "origin_airport": "SFO",
                "dest_airport": "LAX",
                "date": "2026-05-26",
            },
            on_started=on_started,
        )

        with self.assertRaises(asyncio.CancelledError):
            await first_call
        self.assertEqual(observed_active_call_ids, started_call_ids)
        self.assertEqual(payload["type"], "tool_result")
        self.assertEqual(payload["tool"], "flight_search")

    async def test_current_call_cancellation_not_swallowed_while_waiting_for_previous_task(self) -> None:
        thinker = _make_thinker()
        previous_cancellation_started = asyncio.Event()

        async def slow_previous_task() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                previous_cancellation_started.set()
                await asyncio.sleep(10)
                raise

        previous_task = asyncio.create_task(slow_previous_task())
        thinker.state.active_task = previous_task
        thinker.state.active_call_id = "previous"

        current_call = asyncio.create_task(
            thinker.call(
                "Search flights from New York to Seattle tomorrow",
                slots={
                    "origin_airport": "JFK",
                    "dest_airport": "SEA",
                    "date": "2026-05-26",
                },
            )
        )
        await previous_cancellation_started.wait()
        current_call.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await current_call
        self.assertEqual(thinker.state.lifecycle_events, [])
        self.assertIs(thinker.state.active_task, previous_task)
        previous_task.cancel()
        with suppress(asyncio.CancelledError):
            await previous_task

    async def test_booking_finalization_cancellation_clears_pending_draft(self) -> None:
        backend = _BlockingCreateBackend()
        thinker = _make_thinker(backend=backend)

        await thinker.call(
            "Search flights from New York to Seattle tomorrow",
            slots={
                "origin_airport": "JFK",
                "dest_airport": "SEA",
                "date": "2026-05-26",
            },
        )
        await thinker.call("Book the first one with defaults", slots={"flight_selected": "1", "use_defaults": True})
        confirm_task = asyncio.create_task(thinker.call("Yes, confirm", slots={"confirmed": True}))
        await backend.create_started.wait()
        confirm_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await confirm_task
        self.assertIsNone(thinker.state.booking_draft)
        self.assertFalse(thinker.state.waiting_for_preferences)
        self.assertFalse(thinker.state.waiting_for_confirmation)


if __name__ == "__main__":
    unittest.main()
