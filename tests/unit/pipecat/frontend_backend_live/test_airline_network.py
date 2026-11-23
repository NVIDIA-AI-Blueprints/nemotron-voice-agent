# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""Demo Air's small world: the timetable, the sample trips made from it, and the story of a cancelled flight."""

import json
import re
import sys
import unittest
from datetime import UTC, date, datetime
from pathlib import Path

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline import (
    airports,
    schedule,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.schemas import (
    AIRLINE_TOOLS,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.speech import (
    spoken_time,
)
from tests.unit.pipecat.airline_helpers import (
    K7_FLIGHT,
    K7_LEAVES,
    M4_DAY,
    M4_FLIGHT,
    M4_LEAVES,
    R9_FLIGHT,
    R9_LEAVES,
    R9_SAYS,
    DeskCase,
    desk,
)

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts"))
from export_airline_demo import render as render_demo  # noqa: E402

REAL_CARRIERS = re.compile(
    r"\b(american|delta|united|southwest|jetblue|alaska|spirit|frontier|lufthansa|emirates|british airways|"
    r"air canada|qantas|air france|klm|ryanair|easyjet|indigo)\b",
    re.IGNORECASE,
)


class TimetableTests(unittest.TestCase):
    def setUp(self):
        self.table = schedule.timetable()
        self.every = airports.all_airports()

    def test_every_pair_of_the_twelve_cities_has_nonstop_flights_both_ways(self):
        codes = sorted(self.every)
        self.assertEqual(len(schedule.routes()), 12 * 11)
        for origin in codes:
            for destination in codes:
                if origin == destination:
                    continue
                with self.subTest(f"{origin}-{destination}"):
                    flights = schedule.sample_flights(self.every[origin], self.every[destination], date(2026, 11, 9))
                    expected = schedule.frequency(self.every[origin], self.every[destination])
                    self.assertTrue(3 <= expected <= 5)
                    self.assertEqual(len(flights), expected)
        self.assertEqual(len(self.table), 528)

    def test_flight_numbers_are_unique_three_digits_even_toward_the_east_and_odd_toward_the_west(self):
        numbers = [int(number.split(" ")[1]) for number in self.table]
        self.assertEqual(len(set(numbers)), len(numbers))
        self.assertTrue(all(100 <= number <= 999 for number in numbers))
        for number, departure in self.table.items():
            origin, destination = self.every[departure.origin], self.every[departure.destination]
            with self.subTest(number):
                self.assertRegex(number, r"^DA \d{3}$")
                self.assertEqual(int(number.split(" ")[1]) % 2 == 0, destination.lon > origin.lon)
                self.assertEqual(departure.eastbound, destination.lon > origin.lon)

    def test_a_pair_of_cities_has_its_own_block_of_ten_numbers_in_both_directions(self):
        blocks = {}
        for number, departure in self.table.items():
            blocks.setdefault(frozenset((departure.origin, departure.destination)), set()).add(int(number[3:]) // 10)
        self.assertEqual(len(blocks), 66)
        self.assertTrue(all(len(block) == 1 for block in blocks.values()))
        self.assertEqual(len({next(iter(block)) for block in blocks.values()}), 66)

    def test_the_numbers_follow_the_clock_and_the_direction(self):
        day = date(2026, 11, 9)
        for origin, destination, numbers in (
            ("DEN", "BOS", ["DA 210", "DA 212", "DA 214", "DA 216"]),
            ("BOS", "DEN", ["DA 211", "DA 213", "DA 215", "DA 217"]),
        ):
            with self.subTest(origin):
                flights = schedule.sample_flights(self.every[origin], self.every[destination], day)
                self.assertEqual([flight["flight_number"] for flight in flights], numbers)

    def test_a_flight_leaves_at_the_same_local_time_every_day(self):
        boston, denver = self.every["BOS"], self.every["DEN"]
        across_the_clock_change = [date(2026, 10, 31), date(2026, 11, 2), date(2026, 12, 20)]
        runs = [schedule.sample_flights(boston, denver, day) for day in across_the_clock_change]
        for flights in zip(*runs, strict=True):
            self.assertEqual({flight["flight_number"] for flight in flights}, {flights[0]["flight_number"]})
            self.assertEqual({flight["depart_local"][11:16] for flight in flights}, {flights[0]["depart_local"][11:16]})
        # The offset moves with the clock change, the local time does not.
        self.assertEqual(
            [flight["depart_local"][-6:] for flight in (run[0] for run in runs)], ["-04:00", "-05:00", "-05:00"]
        )

    def test_the_times_are_deterministic_and_local_to_each_airport(self):
        first = schedule.sample_flights(self.every["SEA"], self.every["MIA"], date(2026, 11, 9))
        again = schedule.sample_flights(self.every["SEA"], self.every["MIA"], date(2026, 11, 9))
        self.assertEqual(first, again)
        for departure in self.table.values():
            with self.subTest(departure.number):
                self.assertTrue(5 * 60 + 30 <= departure.minute <= 20 * 60)
        for flight in first:
            leaves, lands = (datetime.fromisoformat(flight[key]) for key in ("depart_local", "arrive_local"))
            self.assertEqual(
                (leaves.utcoffset().total_seconds(), lands.utcoffset().total_seconds()), (-8 * 3600, -5 * 3600)
            )
            self.assertEqual((lands - leaves).total_seconds() / 60, flight["duration_min"])

    def test_phoenix_keeps_standard_time_in_the_summer(self):
        summer = date(2026, 7, 15)
        phoenix = schedule.sample_flights(self.every["PHX"], self.every["DEN"], summer)[0]
        denver = schedule.sample_flights(self.every["DEN"], self.every["PHX"], summer)[0]
        self.assertEqual(phoenix["depart_local"][-6:], "-07:00")
        self.assertEqual(denver["depart_local"][-6:], "-06:00")

    def test_a_flight_toward_the_east_is_a_little_shorter_than_the_way_back(self):
        for first, second in (("JFK", "LAX"), ("BOS", "DEN"), ("SEA", "MIA")):
            west_to_east = schedule.sample_flights(self.every[first], self.every[second], date(2026, 11, 9))[0]
            east_to_west = schedule.sample_flights(self.every[second], self.every[first], date(2026, 11, 9))[0]
            durations = sorted((west_to_east["duration_min"], east_to_west["duration_min"]))
            with self.subTest(first):
                self.assertLess(durations[0], durations[1])
        for departure in self.table.values():
            self.assertTrue(45 <= departure.block <= 6 * 60 + 30, departure.number)

    def test_a_known_number_has_its_route_and_an_unknown_one_has_none(self):
        day = date(2026, 11, 9)
        known = schedule.scheduled("DA 213", day)
        self.assertEqual((known["origin"], known["destination"]), ("BOS", "DEN"))
        self.assertEqual(known["depart_local"][:16], "2026-11-09T09:55")
        matching = [
            f
            for f in schedule.sample_flights(self.every["BOS"], self.every["DEN"], day)
            if f["flight_number"] == "DA 213"
        ]
        self.assertEqual(matching[0]["depart_local"], known["depart_local"])
        for number in ("DA 12", "DA 999", "UA 1534", ""):
            with self.subTest(number):
                self.assertIsNone(schedule.scheduled(number, day))

    def test_a_late_flight_has_a_gate_in_the_terminal_the_airline_uses_at_that_airport(self):
        terminals = {}
        for flight in schedule.sample_flights(self.every["ORD"], self.every["SEA"], date(2026, 11, 9)):
            terminals.setdefault("ORD", set()).add(flight["origin"]["terminal"])
            terminals.setdefault("SEA", set()).add(flight["destination"]["terminal"])
        self.assertTrue(all(len(found) == 1 for found in terminals.values()))


class SampleTripTests(DeskCase):
    """The accounts that every session starts with, made from the timetable at the fixed clock."""

    def test_jane_doe_has_two_trips_made_from_timetable_flights(self):
        k7 = self.tools.store.booking("K7Q2ZP").legs[0]
        m4 = self.tools.store.booking("M4X8RT").legs[0]
        self.assertEqual((k7.flight_number, k7.origin, k7.destination), (K7_FLIGHT, "BOS", "DEN"))
        self.assertEqual((k7.seat, k7.meal, k7.delay_min, k7.cancelled), ("18A", "VGML", 45, False))
        self.assertEqual(datetime.fromisoformat(k7.depart_local).strftime("%Y-%m-%d %H:%M"), "2026-11-05 09:55")
        self.assertEqual(spoken_time(k7.depart_local), K7_LEAVES)
        self.assertEqual(spoken_time(m4.depart_local), M4_LEAVES)
        self.assertEqual((m4.flight_number, m4.origin, m4.destination, m4.seat), (M4_FLIGHT, "JFK", "LAX", "5A"))
        self.assertEqual(m4.depart_local[:16], f"{M4_DAY}T16:00")
        for leg in (k7, m4):
            number = schedule.scheduled(leg.flight_number, date.fromisoformat(leg.depart_local[:10]))
            self.assertEqual((number["depart_local"], number["arrive_local"]), (leg.depart_local, leg.arrive_local))

    def test_marcus_lee_has_one_trip_whose_flight_is_cancelled(self):
        leg = self.tools.store.booking("R9T4WN").legs[0]
        self.assertEqual((leg.flight_number, leg.origin, leg.destination), (R9_FLIGHT, "ORD", "SEA"))
        self.assertTrue(leg.cancelled)
        self.assertEqual(self.tools.store.accounts["marcus_lee_002"].name, "Marcus Lee")
        self.assertEqual(self.tools.store.booking("R9T4WN").user_id, "marcus_lee_002")
        self.assertEqual(
            [item["user_id"] for item in self.tools.store.sample_accounts()], ["jane_doe_001", "marcus_lee_002"]
        )

    def test_the_next_flight_of_a_sample_trip_leaves_at_least_ninety_minutes_from_now(self):
        for hour in (4, 12, 18, 23):
            now = datetime(2026, 11, 4, hour, 0, tzinfo=UTC)
            leg = desk(now).store.booking("R9T4WN").legs[0]
            with self.subTest(hour):
                self.assertGreaterEqual((datetime.fromisoformat(leg.depart_local) - now).total_seconds(), 90 * 60)
        # Late in the evening the day's flights are gone, so the trip is the first flight of tomorrow.
        late = desk(datetime(2026, 11, 5, 3, 0, tzinfo=UTC)).store.booking("R9T4WN").legs[0]
        self.assertEqual(late.depart_local[:10], "2026-11-05")


class DisruptionTests(DeskCase):
    """Marcus Lee's flight was cancelled: the desk says so everywhere, and rebooks it at no charge."""

    async def test_signing_in_says_the_flight_has_been_cancelled(self):
        result = await self.sign_in("marcus_lee_002")
        self.assertEqual((result["status"], result["account"]["tier"], result["trip_count"]), ("ok", "Silver", 1))
        self.assertEqual(
            result["trips"][0]["trip_spoken"], "Chicago to Seattle on Wednesday, November 4th, cancelled by Demo Air"
        )
        self.assertEqual(
            result["heads_up"],
            f"{R9_SAYS} from Chicago to Seattle on Wednesday, November 4th has been cancelled.",
        )

    async def test_the_trip_list_marks_the_leg_and_the_booking_that_needs_rebooking(self):
        await self.sign_in("marcus_lee_002")
        booking = (await self.call("get_existing_bookings"))["bookings"][0]
        self.assertTrue(booking["needs_rebooking"])
        self.assertEqual(
            (booking["flights"][0]["cancelled"], booking["flights"][0]["origin"], booking["flights"][0]["seat"]),
            (True, "ORD", "12C"),
        )

    async def test_the_status_of_the_cancelled_flight_is_cancelled_with_its_cause_and_no_gate(self):
        await self.sign_in("marcus_lee_002")
        status = await self.call("get_flight_status", flight_number=R9_FLIGHT)
        self.assertEqual(
            (status["status"], status["state"], status["cause"]), ("ok", "cancelled", "an aircraft maintenance issue")
        )
        self.assertIn(
            "is cancelled on Wednesday, November 4th because of an aircraft maintenance issue", status["summary"]
        )
        self.assertIsNone(status["departure"]["gate"])
        self.assertIsNone(status["departure"]["estimated"])
        self.assertEqual(status["departure"]["scheduled_spoken"], R9_LEAVES)
        # The same number runs on other days.
        other = await self.call("get_flight_status", flight_number=R9_FLIGHT, date="2026-11-06")
        self.assertEqual((other["state"], other["departure"]["scheduled_spoken"]), ("scheduled", R9_LEAVES))
        self.assertNotIn("cause", other)

    async def test_a_search_never_offers_the_cancelled_flight(self):
        await self.sign_in("marcus_lee_002")
        today = (await self.call("flights_search", origin="ORD", destination="SEA", date="2026-11-04"))["flights"]
        self.assertTrue(today)
        self.assertNotIn(R9_FLIGHT, [flight["flight"] for flight in today])
        # The same number runs on other days.
        later = (await self.call("flights_search", origin="ORD", destination="SEA", date="2026-11-06"))["flights"]
        self.assertIn(R9_FLIGHT, [flight["flight"] for flight in later])

    async def test_the_cancelled_flight_is_rebooked_at_no_charge_and_keeps_the_fare_paid(self):
        await self.sign_in("marcus_lee_002")
        paid = self.tools.store.booking("R9T4WN").total
        found = await self.call("flights_search", origin="Chicago", destination="Seattle", date="2026-11-04")
        self.assertEqual(found["status"], "ok")
        read_back = await self.call("modify_existing_booking", flight="1")
        self.assertEqual((read_back["status"], read_back["booking_id"]), ("needs_confirmation", "R9T4WN"))
        self.assertEqual(read_back["fare_difference_usd"], 0)
        self.assertEqual(read_back["no_charge_reason"], "Demo Air cancelled the old flight")
        self.assertIn(
            "Demo Air cancelled the old flight, so there is no charge for the change",
            read_back["confirmation_summary"],
        )
        self.assertNotIn("fare difference", read_back["confirmation_summary"])
        self.assertTrue(self.tools.store.booking("R9T4WN").legs[0].cancelled)
        done = await self.call("modify_existing_booking", flight="1", confirmed=True)
        booking = done["booking"]
        self.assertEqual((done["status"], done["fare_difference_usd"]), ("confirmed", 0))
        self.assertEqual(booking["flights"][0]["flight"], found["flights"][0]["flight"])
        self.assertNotIn("cancelled", booking["flights"][0])
        self.assertNotIn("needs_rebooking", booking)
        self.assertEqual(booking["total_usd"], paid)
        self.assertEqual(self.tools.state.last_change, {"kind": "modified", "booking_id": "R9T4WN"})

    async def test_cancelling_the_booking_of_a_cancelled_flight_refunds_it_in_full(self):
        await self.sign_in("marcus_lee_002")
        read_back = await self.call("cancel_booking", booking_id="R9T4WN")
        total = self.tools.store.booking("R9T4WN").total
        self.assertEqual(
            read_back["refund_rule"], f"Demo Air cancelled the flight, so the full ${total:,.0f} would be refunded"
        )
        done = await self.call("cancel_booking", booking_id="R9T4WN", confirmed=True)
        self.assertEqual((done["status"], done["booking"]["status"]), ("confirmed", "cancelled"))

    async def test_the_other_accounts_trips_are_not_cancelled(self):
        result = await self.sign_in()
        self.assertNotIn("cancelled", json.dumps(result["trips"]))
        self.assertIn("delay of about 45 minutes", result["heads_up"])
        self.assertEqual([leg.flight_number for leg in self.tools.store.booking("K7Q2ZP").legs], [K7_FLIGHT])


class DemoStepsTests(unittest.TestCase):
    """The client's recorded call comes from the real tools, and its copy is current."""

    DEMO_COPY = ROOT / "clients" / "pipecat" / "tests" / "airline-recorded-call.ts"

    def test_the_file_in_the_client_is_current(self):
        self.assertEqual(
            self.DEMO_COPY.read_text(encoding="utf-8"),
            render_demo(),
            "run scripts/export_airline_demo.py to refresh clients/pipecat/tests/airline-recorded-call.ts",
        )

    def test_the_steps_relay_every_tool_call_whole_and_in_order(self):
        text = self.DEMO_COPY.read_text(encoding="utf-8")
        steps = json.loads(text[text.index("= [") + 2 :].rstrip().rstrip(";"))
        seqs = [event["seq"] for step in steps for event in step["events"]]
        self.assertEqual(seqs, list(range(1, len(seqs) + 1)))
        calls = [
            event["event"]["item"]
            for step in steps
            for event in step["events"]
            if event["kind"] == "response.output_item.done"
        ]
        self.assertEqual(
            [call["name"] for call in calls],
            [
                "flights_search",
                "new_booking",
                "sign_in",
                "new_booking",
                "new_booking",
                "get_flight_status",
                "get_ontime_trend",
                "flights_search",
            ],
        )
        for call in calls:
            with self.subTest(call["name"]):
                self.assertEqual((call["server_label"], call["status"]), ("airline", "completed"))
                self.assertIn("status", json.loads(call["output"]))
        self.assertEqual(json.loads(calls[-1]["output"])["status"], "not_served")


class NoRealAirlinesTests(unittest.TestCase):
    """The airline and its flights are made up: the tool pack names no real carrier."""

    def test_the_tool_schemas_name_no_real_carrier(self):
        self.assertIsNone(REAL_CARRIERS.search(json.dumps(AIRLINE_TOOLS)))

    def test_the_timetable_has_one_carrier(self):
        self.assertEqual({number.split(" ")[0] for number in schedule.timetable()}, {"DA"})
        flights = schedule.sample_flights(airports.get_airport("DEN"), airports.get_airport("SEA"), date(2026, 11, 9))
        self.assertEqual({flight["airline"]["name"] for flight in flights}, {"Demo Air"})


if __name__ == "__main__":
    unittest.main()
