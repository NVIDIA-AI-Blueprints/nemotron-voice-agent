# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The airline desk's tools as a server tool source: schemas, search and filters, status, sign-in and bookings."""

import json
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import yaml

from nemotron_voice_agent.pipecat.examples.frontend_backend_live import factory
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.app import tool_sources
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools import TOOL_SOURCES
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline import (
    airports,
    carriers,
    filters,
    schedule,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline import (
    tools as airline_tools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.schemas import (
    AIRLINE_TOOLS,
    CABINS,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.speech import (
    spoken_amount,
    spoken_code,
    spoken_date,
    spoken_day,
    spoken_duration,
    spoken_time,
    spoken_user_id,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.store import Leg
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.airline.tools import (
    TOOL_NAMES,
    AirlineTools,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.tools.cafe.schemas import (
    CAFE_TOOLS,
)
from tests.unit.pipecat.airline_helpers import (
    BUSY_DAY,
    CODE,
    DATA_NOTICE,
    FIVE_A_DAY,
    FOUR_A_DAY,
    K7_DELAYED,
    K7_FLIGHT,
    K7_LEAVES,
    K7_SAYS,
    M4_DAY,
    M4_FLIGHT,
    M4_LEAVES,
    NOW,
    SEARCH,
    TOMORROW,
    DeskCase,
    desk,
    flight_on,
)

# The bookings of a new session: Jane Doe's two trips, and Marcus Lee's one, whose flight was cancelled.
SEEDED = 3


class SchemaTests(unittest.TestCase):
    def test_the_schemas_are_the_eleven_tools_and_each_is_a_method(self):
        names = [tool["name"] for tool in AIRLINE_TOOLS]
        self.assertEqual(
            names,
            [
                "flights_search",
                "where_we_fly",
                "get_flight_status",
                "get_ontime_trend",
                "sign_in",
                "get_existing_bookings",
                "new_booking",
                "modify_existing_booking",
                "cancel_booking",
                "select_seat_and_meal",
                "upgrade_class",
            ],
        )
        self.assertEqual(len(set(names)), len(names))
        self.assertEqual(TOOL_NAMES, frozenset(names))
        self.assertFalse(TOOL_NAMES & {tool["name"] for tool in CAFE_TOOLS})
        for tool in AIRLINE_TOOLS:
            with self.subTest(tool["name"]):
                self.assertTrue(callable(getattr(AirlineTools, tool["name"])))
                self.assertEqual(tool["type"], "function")
                self.assertGreater(len(tool["description"]), 40)
                parameters = tool["parameters"]
                self.assertEqual((parameters["type"], parameters["additionalProperties"]), ("object", False))
                self.assertLessEqual(set(parameters["required"]), set(parameters["properties"]))
                for name, spec in parameters["properties"].items():
                    self.assertTrue(spec["description"], name)
                    self.assertIn(spec["type"], {"string", "boolean"}, name)
                    if "enum" in spec:
                        self.assertTrue(set(spec["enum"]) <= {*CABINS, "price", "duration", "departure_time"})
        json.dumps(AIRLINE_TOOLS)

    def test_the_required_parameters_and_the_change_tools_confirmation(self):
        by_name = {tool["name"]: tool for tool in AIRLINE_TOOLS}
        required = {name: tool["parameters"]["required"] for name, tool in by_name.items()}
        self.assertEqual(required["flights_search"], ["origin", "destination", "date"])
        self.assertEqual(required["new_booking"], ["flight"])
        self.assertEqual(required["upgrade_class"], ["cabin"])
        self.assertEqual(required["sign_in"], [])
        for name in (
            "new_booking",
            "modify_existing_booking",
            "cancel_booking",
            "select_seat_and_meal",
            "upgrade_class",
        ):
            with self.subTest(name):
                self.assertEqual(by_name[name]["parameters"]["properties"]["confirmed"]["type"], "boolean")
                self.assertIn("confirmed", by_name[name]["description"])
        self.assertEqual(
            set(by_name["flights_search"]["parameters"]["properties"]),
            {
                "origin",
                "destination",
                "date",
                "cabin",
                "sort",
                "time_of_day",
                "depart_after",
                "depart_before",
            },
        )

    def test_the_tool_list_is_a_copy_with_todays_date_in_the_date_parameters(self):
        tools = desk().tools()
        self.assertEqual([tool["name"] for tool in tools], [tool["name"] for tool in AIRLINE_TOOLS])
        dated = {
            tool["name"]: tool["parameters"]["properties"]["date"]["description"]
            for tool in tools
            if "date" in tool["parameters"]["properties"]
        }
        self.assertEqual(set(dated), {"flights_search", "get_flight_status"})
        for text in dated.values():
            self.assertTrue(text.endswith("Today is 2026-11-04 (Wednesday)."), text)
        tools[0]["parameters"]["properties"]["origin"]["description"] = "changed"
        tools[0]["parameters"]["required"].clear()
        again = desk(datetime(2026, 12, 25, 9, 0, tzinfo=UTC)).tools()
        self.assertEqual(
            again[0]["parameters"]["properties"]["origin"]["description"],
            AIRLINE_TOOLS[0]["parameters"]["properties"]["origin"]["description"],
        )
        self.assertEqual(again[0]["parameters"]["required"], ["origin", "destination", "date"])
        self.assertIn("Today is 2026-12-25 (Friday).", again[0]["parameters"]["properties"]["date"]["description"])
        self.assertNotIn("Today is", AIRLINE_TOOLS[0]["parameters"]["properties"]["date"]["description"])

    def test_the_clock_may_be_naive_and_the_default_is_now(self):
        self.assertEqual(AirlineTools(clock=lambda: datetime(2026, 11, 4, 8, 0)).now().tzinfo, UTC)
        self.assertLess(abs(AirlineTools().now() - datetime.now(UTC)), timedelta(seconds=5))

    def test_the_source_is_registered_and_takes_no_settings(self):
        self.assertIs(TOOL_SOURCES["airline"], AirlineTools)
        self.assertIsInstance(AirlineTools.from_settings({}), AirlineTools)
        for block in ({"currency": "EUR"}, None, ["x"]):
            with (
                self.subTest(block=block),
                self.assertRaisesRegex(ValueError, "backend.tools.airline takes no settings"),
            ):
                AirlineTools.from_settings(block)
        made = tool_sources({"airline": {}})()
        self.assertEqual([type(source) for source in made], [AirlineTools])
        self.assertIsNot(tool_sources({"airline": {}})()[0], made[0])

    def test_the_tools_catalog_of_the_ui_lists_the_schemas_of_both_sources(self):
        path = Path(factory.__file__).with_name("tools.yaml")
        catalog = yaml.safe_load(path.read_text(encoding="utf-8"))
        schemas = {tool["name"]: tool for tool in [*CAFE_TOOLS, *AIRLINE_TOOLS]}
        self.assertEqual(list(catalog), list(schemas))
        for name, entry in catalog.items():
            with self.subTest(name):
                self.assertEqual(entry["type"], "function")
                function, schema = entry["function"], schemas[name]
                self.assertEqual(
                    function, {"name": name, "description": schema["description"], "parameters": schema["parameters"]}
                )


class SpeechTests(unittest.TestCase):
    def test_dates_times_prices_and_codes_are_said_as_people_say_them(self):
        today = date(2026, 11, 4)
        self.assertEqual(spoken_date("2026-11-05", today), "November 5th")
        self.assertEqual(spoken_date(date(2026, 11, 22), today), "November 22nd")
        self.assertEqual(spoken_date("2026-11-13", today), "November 13th")
        self.assertEqual(spoken_date("2027-01-01", today), "January 1st, 2027")
        self.assertEqual(spoken_day("2026-11-05T07:15:00-05:00", today), "Thursday, November 5th")
        self.assertEqual(spoken_time("2026-11-05T07:15:00-05:00"), "7:15 AM")
        self.assertEqual(spoken_time("2026-11-05T00:05:00+00:00"), "12:05 AM")
        self.assertEqual(spoken_time("2026-11-05T13:00:00+00:00"), "1:00 PM")
        self.assertEqual((spoken_time(None), spoken_time("later")), ("", "later"))
        self.assertEqual(spoken_amount(1234.4), "$1,234")
        self.assertEqual(
            [spoken_duration(m) for m in (305, 45, 120, 61, 0, None)],
            ["5 hours 5 minutes", "45 minutes", "2 hours", "1 hour 1 minute", "", ""],
        )
        self.assertEqual(spoken_code("K7Q2ZP"), "K 7 Q 2 Z P")
        self.assertEqual(spoken_user_id("jane_doe_001"), "jane doe zero zero one")


class PlaceTests(unittest.TestCase):
    def test_a_city_an_airport_name_an_alias_or_a_code_resolves(self):
        for text, code in (
            ("Boston", "BOS"),
            ("boston", "BOS"),
            ("BOS", "BOS"),
            ("bos", "BOS"),
            ("Logan International Airport", "BOS"),
            ("New York", "JFK"),
            ("NYC", "JFK"),
            ("Chicago O'Hare", "ORD"),
            ("O'Hare", "ORD"),
            ("Dallas", "DFW"),
            ("Dallas Fort Worth", "DFW"),
            ("Las Vegas", "LAS"),
            ("Vegas", "LAS"),
            ("the Bay Area", "SFO"),
            ("LA", "LAX"),
            ("Denver Colorado", "DEN"),
            ("Seattle, Washington", "SEA"),
            ("Sea-Tac", "SEA"),
        ):
            with self.subTest(text):
                self.assertEqual(airports.resolve_airport(text).airport.iata, code)

    def test_a_place_that_is_not_served_resolves_to_a_place_with_its_closest_served_city(self):
        for text, label, nearest, miles in (
            ("Boise", "Boise, Idaho", "SEA", 399),
            ("Newark", "Newark, New Jersey", "JFK", 21),
            ("EWR", "Newark, New Jersey", "JFK", 21),
            ("San Diego", "San Diego, California", "LAX", 111),
            ("Orlando", "Orlando, Florida", "MIA", 193),
            ("Washington DC", "Washington, D.C.", "JFK", 213),
            ("London", "London, United Kingdom", "BOS", 3257),
        ):
            with self.subTest(text):
                match = airports.resolve_airport(text)
                self.assertIsNone(match.airport)
                self.assertEqual(match.unserved.label, label)
                airport, distance = airports.nearest_served(match.unserved)
                self.assertEqual(airport.iata, nearest)
                self.assertAlmostEqual(distance, miles, delta=4)

    def test_what_is_not_known_is_neither_served_nor_unserved(self):
        for text in ("Atlantis", "", None, "xx", "Mordor"):
            with self.subTest(text):
                match = airports.resolve_airport(text)
                self.assertEqual((match.airport, match.unserved), (None, None))

    def test_the_network_is_twelve_us_airports_in_five_time_zones(self):
        every = airports.all_airports()
        self.assertEqual(
            sorted(every), sorted(["ATL", "BOS", "DEN", "DFW", "JFK", "LAS", "LAX", "MIA", "ORD", "PHX", "SEA", "SFO"])
        )
        self.assertEqual({airport.country for airport in every.values()}, {"US"})
        self.assertEqual(
            {airport.tz for airport in every.values()},
            {"America/New_York", "America/Chicago", "America/Denver", "America/Phoenix", "America/Los_Angeles"},
        )
        for airport in every.values():
            with self.subTest(airport.iata):
                self.assertRegex(airport.iata, "^[A-Z]{3}$")
                self.assertTrue(24 <= airport.lat <= 49 and -125 <= airport.lon <= -70)
                self.assertIsNotNone(airport.zone.utcoffset(NOW))
        self.assertEqual(airports.great_circle_km(every["BOS"], every["BOS"]), 0)
        self.assertTrue(2700 < airports.great_circle_km(every["BOS"], every["DEN"]) < 3000)
        self.assertEqual(airports.get_airport(" jfk ").iata, "JFK")
        self.assertIsNone(airports.get_airport("ZZZ"))

    def test_the_unserved_places_are_not_served_and_each_has_a_position(self):
        places = airports.all_places()
        self.assertGreater(len(places), 60)
        self.assertFalse(set(places) & set(airports.all_airports()))
        for code, place in places.items():
            with self.subTest(code):
                self.assertTrue(-90 <= place.lat <= 90 and -180 <= place.lon <= 180)
                airport, miles = airports.nearest_served(place)
                self.assertIn(airport.iata, airports.all_airports())
                self.assertGreaterEqual(miles, 0)


class CarrierTests(unittest.TestCase):
    def test_flight_numbers_are_read_however_they_are_written(self):
        for text, number in (
            ("DA214", "DA 214"),
            ("da 214", "DA 214"),
            ("DA-214", "DA 214"),
            ("Demo Air 214", "DA 214"),
            ("Demo Air flight 214", "DA 214"),
            ("flight 214", "DA 214"),
            ("214", "DA 214"),
            ("DA 0214", "DA 214"),
            ("UA 1534", "UA 1534"),
        ):
            with self.subTest(text):
                self.assertEqual(carriers.normalize_flight_number(text), number)
        for text in ("United", "hello", "", None, "DL 99999", "5"):
            with self.subTest(text):
                self.assertIsNone(carriers.normalize_flight_number(text))
        self.assertTrue(carriers.is_demo_air("DA 214"))
        self.assertFalse(carriers.is_demo_air("UA 1534"))

    def test_the_one_airline_is_demo_air_and_a_flight_is_said_as_an_announcer_says_it(self):
        self.assertEqual((carriers.CODE, carriers.NAME), ("DA", "Demo Air"))
        self.assertEqual(carriers.airline_name("da"), "Demo Air")
        self.assertIsNone(carriers.airline_name("UA"))
        for number, spoken in (
            ("DA 100", "one hundred"),
            ("DA 101", "one oh one"),
            ("DA 114", "one fourteen"),
            ("DA 213", "two thirteen"),
            ("DA 305", "three oh five"),
            ("DA 720", "seven twenty"),
            ("DA 769", "seven sixty-nine"),
        ):
            with self.subTest(number):
                self.assertEqual(carriers.spoken_flight(number), f"Demo Air flight {spoken}")
        self.assertEqual(carriers.spoken_flight("UA 1534", "United Airlines"), "United Airlines flight 1534")
        self.assertEqual(carriers.spoken_flight("UA 1534", with_code=True), "UA 1534")


class TimeFilterTests(unittest.TestCase):
    """``FlightFilters`` on flights built by their local departure time, so the boundaries are exact."""

    def kept(self, filter_: filters.FlightFilters, *clocks: str, code: str = "DA") -> list[str]:
        return [clock for clock in clocks if filter_.matches(flight_on(clock, code))]

    def read(self, **arguments) -> filters.FlightFilters:
        arguments = dict(arguments)
        filters.normalize_time_arguments(arguments)
        return filters.FlightFilters.from_arguments(arguments)

    def test_a_part_of_the_day_is_half_open(self):
        day = ["04:59", "05:00", "11:59", "12:00", "17:59", "18:00", "20:59", "21:00", "23:59"]
        self.assertEqual(self.kept(self.read(time_of_day="morning"), *day), ["05:00", "11:59"])
        self.assertEqual(self.kept(self.read(time_of_day="afternoon"), *day), ["12:00", "17:59"])
        self.assertEqual(self.kept(self.read(time_of_day="evening"), *day), ["18:00", "20:59", "21:00", "23:59"])
        self.assertEqual(self.kept(self.read(time_of_day="overnight"), *day), ["21:00", "23:59"])
        self.assertEqual(self.kept(self.read(time_of_day="red-eye"), *day), ["21:00", "23:59"])

    def test_depart_after_includes_its_minute_and_depart_before_leaves_it_out(self):
        day = ["11:59", "12:00", "12:01", "15:00", "23:59"]
        self.assertEqual(self.kept(self.read(depart_after="12:00"), *day), ["12:00", "12:01", "15:00", "23:59"])
        self.assertEqual(self.kept(self.read(depart_before="12:00"), "11:59", "12:00", "12:01"), ["11:59"])
        self.assertEqual(self.kept(self.read(depart_after="12:00", depart_before="15:00"), *day), ["12:00", "12:01"])
        self.assertEqual(self.kept(self.read(depart_before="24:00"), *day), day)
        self.assertEqual(self.kept(self.read(depart_before="00:00"), *day), day)

    def test_a_window_that_runs_past_midnight_runs_to_the_end_of_the_day(self):
        window = self.read(depart_after="22:00", depart_before="02:00")
        self.assertEqual(self.kept(window, "01:00", "21:59", "22:00", "23:59"), ["22:00", "23:59"])
        self.assertEqual(window.window_phrase(), "leaving after 10:00 PM ")

    def test_spoken_times_are_read_into_bounds_and_a_part_of_the_day_keeps_both(self):
        self.assertEqual(self.read(depart_after="3 pm").after, 15 * 60)
        self.assertEqual(self.read(time_of_day="after 3 pm").after, 15 * 60)
        self.assertIsNone(self.read(time_of_day="after 3 pm").time_of_day)
        self.assertEqual(self.read(time_of_day="before noon").before, 12 * 60)
        self.assertEqual(self.read(time_of_day="no later than 6:30 pm").before, 18 * 60 + 30)
        self.assertEqual(self.read(time_of_day="by midnight").before, 24 * 60)
        evening = self.read(time_of_day="evening after 7 pm")
        self.assertEqual((evening.time_of_day, evening.after), ("evening", 19 * 60))
        self.assertEqual(self.read(time_of_day="after 7 tonight").after, 19 * 60)
        self.assertFalse(self.read(time_of_day="any time").active)
        self.assertEqual(filters.parse_clock("3:30 p.m."), 15 * 60 + 30)
        self.assertEqual(filters.parse_clock("noon"), 12 * 60)
        for unreadable in ("3", "3:00", "soon", "25:00"):
            self.assertIsNone(filters.parse_clock(unreadable), unreadable)

    def test_an_unreadable_filter_is_refused_not_dropped(self):
        for arguments in ({"depart_after": "3"}, {"depart_before": "soonish"}, {"time_of_day": "teatime"}):
            with self.subTest(arguments), self.assertRaises(ValueError):
                self.read(**arguments)

    def test_a_filter_is_active_only_when_one_is_set(self):
        self.assertTrue(self.read(time_of_day="morning").active)
        self.assertTrue(self.read(depart_after="12:00").active)
        self.assertFalse(filters.FlightFilters().active)
        self.assertEqual(
            self.read(time_of_day="afternoon", depart_after="15:00").as_data(),
            {"time_of_day": "afternoon", "depart_after": "15:00"},
        )

    def test_the_filters_are_described_in_words(self):
        window = self.read(time_of_day="afternoon", depart_after="15:00")
        self.assertEqual(window.description(), "afternoon leaving after 3:00 PM")
        self.assertEqual(self.read(depart_before="12:00").description(), "leaving before 12:00 PM")
        self.assertEqual(
            self.read(depart_after="09:00", depart_before="11:30").description(), "leaving between 9:00 AM and 11:30 AM"
        )
        self.assertEqual(self.read().description(), "")


class SearchTests(DeskCase):
    async def test_a_search_gives_up_to_five_nonstop_flights_with_spoken_fields(self):
        result = await self.call("flights_search", **FIVE_A_DAY)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["count"], 5)
        self.assertEqual([flight["option"] for flight in result["flights"]], [1, 2, 3, 4, 5])
        self.assertEqual(
            (result["route_spoken"], result["date"], result["date_spoken"]),
            ("New York to Los Angeles", BUSY_DAY, "November 9th"),
        )
        self.assertEqual((result["weekday"], result["cabin"], result["data_source"]), ("Monday", "economy", "sample"))
        times = [flight["depart_local"] for flight in result["flights"]]
        self.assertEqual(times, sorted(times))
        first = result["flights"][0]
        self.assertRegex(first["flight"], r"^DA \d{3}$")
        self.assertEqual((first["origin"], first["destination"]), ("JFK", "LAX"))
        self.assertEqual(first["flight_spoken"], carriers.spoken_flight(first["flight"]))
        self.assertEqual(first["depart_spoken"], spoken_time(first["depart_local"]))
        self.assertEqual(first["price_spoken"], spoken_amount(first["price_usd"]))
        self.assertEqual(first["fare_label"], "demo fare")
        self.assertRegex(first["duration_spoken"], r"minute|hour")
        self.assertEqual((first["airline"], first["airline_code"]), ("Demo Air", "DA"))
        self.assertEqual(result["data_notice"], DATA_NOTICE)
        self.assertNotIn("filters", result)

    async def test_the_data_notice_comes_once_per_session_on_the_first_result(self):
        first = await self.call("flights_search", **SEARCH)
        second = await self.call("flights_search", **{**SEARCH, "destination": "Chicago"})
        status = await self.call("get_flight_status", flight_number=K7_FLIGHT)
        self.assertEqual(first["data_notice"], DATA_NOTICE)
        self.assertNotIn("data_notice", second)
        self.assertNotIn("data_notice", status)
        other = desk()
        self.assertEqual(
            (await self.call("get_flight_status", other, flight_number=K7_FLIGHT))["data_notice"], DATA_NOTICE
        )
        self.assertNotIn("data_notice", await self.call("get_ontime_trend", other, flight_number=K7_FLIGHT))

    async def test_the_same_query_gives_the_same_flights_in_any_session(self):
        one = await self.call("flights_search", **SEARCH)
        two = await self.call("flights_search", desk(), **SEARCH)
        self.assertEqual(one["flights"], two["flights"])
        other_day = await self.call("flights_search", **{**SEARCH, "date": "2026-11-06"})
        other_route = await self.call("flights_search", **{**SEARCH, "destination": "Chicago"})
        self.assertNotEqual(one["flights"], other_day["flights"])
        self.assertNotEqual(one["flights"], other_route["flights"])

    async def test_places_are_resolved_by_the_tool(self):
        for origin, destination in (("Boston", "Denver"), ("BOS", "DEN"), ("logan airport", "Denver International")):
            with self.subTest(origin=origin):
                result = await self.call("flights_search", origin=origin, destination=destination, date=TOMORROW)
                self.assertEqual((result["origin"]["code"], result["destination"]["code"]), ("BOS", "DEN"))
        self.assertEqual(
            (await self.call("flights_search", origin="New York", destination="LA", date=TOMORROW))["route_spoken"],
            "New York to Los Angeles",
        )

    async def test_a_city_that_is_not_served_is_answered_with_the_closest_one_that_is(self):
        near = await self.call("flights_search", **{**SEARCH, "origin": "Boise"})
        self.assertEqual((near["status"], near["reason"], near["field"]), ("not_served", "not_served", "origin"))
        self.assertEqual(near["place"], "Boise, Idaho")
        self.assertEqual((near["nearest"]["code"], near["nearest"]["city"]), ("SEA", "Seattle"))
        self.assertTrue(380 <= near["distance_miles"] <= 420)
        self.assertEqual(
            near["message"],
            f"Demo Air doesn't fly to Boise, Idaho. The closest city we serve is Seattle, about "
            f"{near['distance_miles']} miles away.",
        )
        self.assertEqual(near["question"], "Would you like to fly from Seattle instead?")
        self.assertNotIn("served", near)
        # The answer is not a search: nothing is kept to book from.
        self.assertIsNone(self.tools.state.search)
        close = await self.call("flights_search", **{**SEARCH, "destination": "Newark"})
        self.assertEqual(
            (close["field"], close["nearest"]["code"], close["question"]),
            ("destination", "JFK", "Would you like to fly to New York instead?"),
        )

    async def test_a_place_that_is_far_from_every_served_city_is_not_answered_with_one(self):
        far = await self.call("flights_search", **{**SEARCH, "destination": "London"})
        self.assertEqual(
            (far["status"], far["field"], far["place"]), ("not_served", "destination", "London, United Kingdom")
        )
        self.assertGreater(far["distance_miles"], airline_tools.NEAR_MILES)
        self.assertIn("or anywhere near it", far["message"])
        self.assertIn("twelve U.S. cities", far["message"])
        self.assertEqual(len(far["served"]), 12)
        self.assertEqual(far["question"], "Would you like to travel between two of those cities?")
        for city in ("Atlanta", "Boston", "Seattle", "San Francisco"):
            self.assertIn(city, far["message"])

    async def test_the_nearest_city_is_flown_when_the_caller_says_yes(self):
        await self.call("flights_search", **{**SEARCH, "origin": "Boise"})
        found = await self.call("flights_search", **{**SEARCH, "origin": "Seattle"})
        self.assertEqual((found["status"], found["origin"]["code"]), ("ok", "SEA"))

    async def test_where_we_fly_lists_the_twelve_cities_and_the_scope(self):
        result = await self.call("where_we_fly")
        self.assertEqual((result["status"], result["airline"], result["city_count"]), ("ok", "Demo Air", 12))
        self.assertEqual({city["code"] for city in result["cities"]}, set(airports.all_airports()))
        self.assertEqual([city["city"] for city in result["cities"]], sorted(city["city"] for city in result["cities"]))
        self.assertTrue(all(city["state"] and city["airport"] for city in result["cities"]))
        self.assertIn("Every pair of these cities has nonstop flights", result["routes"])
        self.assertIn("60 days ahead", result["routes"])
        self.assertEqual(result["data_notice"], DATA_NOTICE)
        self.assertNotIn("data_notice", await self.call("where_we_fly"))

    async def test_missing_unknown_and_unreadable_inputs_are_asked_about(self):
        for arguments, reason, field in (
            ({"origin": ""}, "missing_place", "origin"),
            ({"destination": None}, "missing_place", "destination"),
            ({"origin": "Atlantis"}, "unknown_airport", "origin"),
            ({"destination": "Boston"}, "same_airport", "destination"),
            ({"date": ""}, "missing_date", "date"),
            ({"date": "next Friday"}, "missing_date", "date"),
            ({"cabin": "steerage"}, "unknown_cabin", "cabin"),
            ({"time_of_day": "teatime"}, "invalid_time", "time_of_day"),
            ({"depart_after": "3"}, "invalid_time", "time_of_day"),
        ):
            with self.subTest(arguments):
                result = await self.call("flights_search", **{**SEARCH, **arguments})
                self.assertEqual(
                    (result["status"], result["reason"], result["field"]), ("needs_clarification", reason, field)
                )
                self.assertTrue(result["question"])

    async def test_a_past_date_is_an_error_that_asks_for_a_future_one(self):
        past = await self.call("flights_search", **{**SEARCH, "date": "2026-11-03"})
        self.assertEqual((past["status"], past["reason"]), ("error", "past_date"))
        self.assertIn("already passed", past["error"])
        self.assertIn("future date", past["error"])
        self.assertIn("2026-11-04", past["error"])
        far = await self.call("flights_search", **{**SEARCH, "date": "2027-12-25"})
        self.assertEqual((far["status"], far["reason"]), ("error", "too_far_ahead"))
        self.assertIn("60 days ahead", far["error"])
        today = await self.call("flights_search", origin="Boston", destination="Chicago", date="2026-11-04")
        self.assertIn(today["status"], {"ok", "no_results"})
        # A later clock moves the same date into the past; the desk's clock, not the machine's, decides.
        later = desk(datetime(2026, 11, 6, 9, 0, tzinfo=UTC))
        self.assertEqual((await self.call("flights_search", later, **SEARCH))["reason"], "past_date")

    async def test_the_desks_day_is_the_us_day_not_the_utc_date(self):
        # 2 AM UTC on November 5 is 8 PM on November 4 in Central time: an evening caller's today is still the 4th.
        evening = desk(datetime(2026, 11, 5, 2, 0, tzinfo=UTC))
        self.assertEqual((evening.today, evening.store.today), (date(2026, 11, 4), date(2026, 11, 4)))
        tonight = await self.call("flights_search", evening, origin="Boston", destination="Chicago", date="2026-11-04")
        self.assertNotEqual(tonight.get("reason"), "past_date")

    async def test_the_timetable_runs_sixty_days_ahead_and_no_further(self):
        last = (NOW.date() + timedelta(days=60)).isoformat()
        beyond = (NOW.date() + timedelta(days=61)).isoformat()
        self.assertEqual((await self.call("flights_search", **{**SEARCH, "date": last}))["status"], "ok")
        refused = await self.call("flights_search", **{**SEARCH, "date": beyond})
        self.assertEqual((refused["status"], refused["reason"]), ("error", "too_far_ahead"))
        self.assertIn(last, refused["error"])

    async def test_flights_that_have_left_today_are_not_offered(self):
        route = {"origin": "JFK", "destination": "BOS", "date": "2026-11-05"}
        morning = desk(datetime(2026, 11, 5, 4, 0, tzinfo=UTC))
        evening = desk(datetime(2026, 11, 5, 23, 30, tzinfo=UTC))
        every = await self.call("flights_search", morning, **route)
        late = await self.call("flights_search", evening, **route)
        self.assertEqual(every["count"], 4)
        self.assertEqual(late["status"], "ok")
        self.assertEqual(late["count"], 1)
        for flight in late["flights"]:
            self.assertGreater(
                datetime.fromisoformat(flight["depart_local"]), datetime(2026, 11, 5, 23, 30, tzinfo=UTC)
            )

    async def test_filters_narrow_the_days_flights_before_the_best_are_kept(self):
        everything = schedule.sample_flights(
            airports.get_airport("JFK"), airports.get_airport("LAX"), date.fromisoformat(BUSY_DAY)
        )
        self.assertEqual(len(everything), 5)
        # No route has more than five flights a day, so the cut is lowered to see that the filter comes first.
        with patch.object(airline_tools, "MAX_RESULTS", 2):
            unfiltered = await self.call("flights_search", **FIVE_A_DAY)
            evening = await self.call("flights_search", **FIVE_A_DAY, depart_after="12:00")
        self.assertEqual([f["flight"] for f in unfiltered["flights"]], [f["flight_number"] for f in everything[:2]])
        wanted = [f["flight_number"] for f in everything if f["depart_local"][11:16] >= "12:00"]
        self.assertEqual(len(wanted), 3)
        self.assertEqual([f["flight"] for f in evening["flights"]], wanted[:2])
        self.assertEqual(evening["filters"], {"depart_after": "12:00"})
        self.assertEqual(evening["unfiltered_count"], 5)

    async def test_each_filter_keeps_only_what_it_names(self):
        route = FIVE_A_DAY
        for arguments, keep in (
            ({"time_of_day": "afternoon"}, lambda f: "12:00" <= f["depart_local"][11:16] < "18:00"),
            ({"time_of_day": "morning"}, lambda f: "05:00" <= f["depart_local"][11:16] < "12:00"),
            ({"time_of_day": "overnight"}, lambda f: f["depart_local"][11:16] >= "21:00"),
            ({"depart_before": "13:30"}, lambda f: f["depart_local"][11:16] < "13:30"),
            ({"depart_after": "13:30"}, lambda f: f["depart_local"][11:16] >= "13:30"),
            ({"time_of_day": "after 3 pm"}, lambda f: f["depart_local"][11:16] >= "15:00"),
            (
                {"time_of_day": "afternoon", "depart_after": "15:00"},
                lambda f: "15:00" <= f["depart_local"][11:16] < "18:00",
            ),
        ):
            with self.subTest(arguments):
                every = schedule.sample_flights(
                    airports.get_airport("JFK"), airports.get_airport("LAX"), date.fromisoformat(BUSY_DAY)
                )
                expected = [f["flight_number"] for f in every if keep(f)][:5]
                result = await self.call("flights_search", **route, **arguments)
                if expected:
                    self.assertEqual((result["status"], [f["flight"] for f in result["flights"]]), ("ok", expected))
                else:
                    self.assertEqual(result["status"], "no_results")

    async def test_nothing_matching_is_reported_honestly_with_no_other_flights(self):
        for arguments, mentions in (
            ({"depart_before": "05:00"}, "leaving before 5:00 AM"),
            ({"depart_after": "22:00"}, "leaving after 10:00 PM"),
            ({"time_of_day": "overnight"}, "overnight nonstop flights"),
        ):
            with self.subTest(arguments):
                result = await self.call("flights_search", **FIVE_A_DAY, **arguments)
                self.assertEqual((result["status"], result["reason"]), ("no_results", "filters"))
                self.assertNotIn("flights", result)
                self.assertIn(mentions, result["message"])
                self.assertIn("The nonstop flights that day leave between", result["message"])
                self.assertEqual(result["unfiltered_count"], 5)
                self.assertEqual(len(result["departures_that_day"]), 5)
        # The latest search is the empty one: a booking cannot pick a flight from the earlier results.
        await self.sign_in()
        self.assertEqual((await self.call("new_booking", flight="1"))["status"], "needs_search")

    async def test_a_day_with_every_flight_gone_says_so_and_offers_none(self):
        with patch.object(airline_tools, "sample_flights", return_value=[]):
            result = await self.call("flights_search", **SEARCH)
        self.assertEqual((result["status"], result["reason"]), ("no_results", "no_flights_left"))
        self.assertIn("No Demo Air flights are left from Boston to Denver", result["message"])
        self.assertNotIn("flights", result)

    async def test_the_results_are_sorted_as_asked_and_priced_by_cabin(self):
        route = FOUR_A_DAY
        by_price = (await self.call("flights_search", **route, sort="price"))["flights"]
        self.assertEqual([f["price_usd"] for f in by_price], sorted(f["price_usd"] for f in by_price))
        by_time = (await self.call("flights_search", **route, sort="duration"))["flights"]
        self.assertEqual([f["duration_min"] for f in by_time], sorted(f["duration_min"] for f in by_time))
        economy = (await self.call("flights_search", **SEARCH))["flights"][0]
        business = (await self.call("flights_search", **SEARCH, cabin="business"))["flights"][0]
        self.assertEqual((business["cabin"], business["flight"]), ("business", economy["flight"]))
        self.assertGreater(business["price_usd"], 2 * economy["price_usd"])
        self.assertEqual((await self.call("flights_search", **SEARCH, cabin="coach"))["cabin"], "economy")

    async def test_a_long_flight_arrives_in_the_local_time_of_its_destination(self):
        result = await self.call("flights_search", origin="Boston", destination="San Francisco", date="2026-11-10")
        self.assertEqual(result["status"], "ok")
        flight = result["flights"][0]
        leaves, lands = datetime.fromisoformat(flight["depart_local"]), datetime.fromisoformat(flight["arrive_local"])
        self.assertEqual((leaves.utcoffset(), lands.utcoffset()), (timedelta(hours=-5), timedelta(hours=-8)))
        self.assertEqual((lands - leaves).total_seconds() / 60, flight["duration_min"])

    async def test_a_late_flight_toward_the_east_lands_the_next_day(self):
        result = await self.call("flights_search", origin="Los Angeles", destination="New York", date=BUSY_DAY)
        last = result["flights"][-1]
        self.assertTrue(last["arrives_next_day"])
        self.assertGreater(last["arrive_local"][:10], last["depart_local"][:10])


class StatusTests(DeskCase):
    async def test_a_known_flight_has_a_sample_status_with_times_gate_and_terminal(self):
        result = await self.call("get_flight_status", flight_number="Demo Air 213")
        self.assertEqual((result["status"], result["flight"], result["state"]), ("ok", K7_FLIGHT, "delayed"))
        self.assertEqual(
            (result["delay_min"], result["route_spoken"], result["date"]), (45, "Boston to Denver", TOMORROW)
        )
        self.assertEqual(
            (result["departure"]["scheduled_spoken"], result["departure"]["estimated_spoken"]),
            (K7_LEAVES, K7_DELAYED),
        )
        self.assertTrue(result["departure"]["gate"] and result["departure"]["terminal"])
        self.assertIn("delayed by about 45 minutes", result["summary"])
        self.assertIn(f"{K7_DELAYED} instead of {K7_LEAVES}", result["summary"])
        self.assertIn(f"because of {result['cause']}", result["summary"])
        self.assertIn(result["cause"], schedule._DELAY_REASONS)
        self.assertEqual(result["airline"], "Demo Air")
        self.assertEqual(result["data_source"], "sample")
        again = await self.call("get_flight_status", desk(), flight_number=K7_FLIGHT)
        self.assertEqual(
            {k: v for k, v in again.items() if k != "data_notice"},
            {k: v for k, v in result.items() if k != "data_notice"},
        )

    async def test_the_state_follows_the_clock_from_delayed_to_landed(self):
        now = [NOW]
        tools = AirlineTools(clock=lambda: now[0], seed=7)
        states = []
        # The flight leaves Boston at 9:55 AM (14:55 UTC), 45 minutes late, and lands in Denver about 19:50 UTC.
        for moment in (
            NOW,
            datetime(2026, 11, 5, 14, 30, tzinfo=UTC),
            datetime(2026, 11, 5, 15, 10, tzinfo=UTC),
            datetime(2026, 11, 5, 17, 0, tzinfo=UTC),
            datetime(2026, 11, 5, 21, 0, tzinfo=UTC),
        ):
            now[0] = moment
            result = await self.call("get_flight_status", tools, flight_number=K7_FLIGHT)
            states.append((result["state"], result["departure"]["actual"] is not None))
        self.assertEqual(
            states,
            [("delayed", False), ("delayed", False), ("boarding", False), ("en_route", True), ("landed", True)],
        )
        on_time = await self.call("get_flight_status", flight_number=M4_FLIGHT, date=M4_DAY)
        self.assertEqual((on_time["state"], on_time["delay_min"]), ("scheduled", 0))
        self.assertNotIn("cause", on_time)
        self.assertIn(f"on schedule to leave at {M4_LEAVES}", on_time["summary"])

    async def test_a_flight_is_boarding_forty_minutes_before_it_leaves_with_its_gate(self):
        tools = AirlineTools(clock=lambda: datetime(2026, 11, 28, 20, 45, tzinfo=UTC), seed=7)
        result = await self.call("get_flight_status", tools, flight_number=M4_FLIGHT, date=M4_DAY)
        self.assertEqual(result["state"], "boarding")
        self.assertIn("is boarding now at gate", result["summary"])
        self.assertIn(f"leaves at {M4_LEAVES}", result["summary"])

    async def test_a_flight_from_the_search_is_known_and_a_date_moves_it(self):
        found = await self.call("flights_search", **SEARCH)
        number = found["flights"][1]["flight"]
        status = await self.call("get_flight_status", flight_number=number)
        self.assertEqual((status["status"], status["date"]), ("ok", TOMORROW))
        self.assertEqual(status["departure"]["scheduled_spoken"], found["flights"][1]["depart_spoken"])
        later = await self.call("get_flight_status", flight_number=number, date="2026-11-12")
        self.assertEqual(
            (later["date"], later["departure"]["scheduled_spoken"]),
            ("2026-11-12", found["flights"][1]["depart_spoken"]),
        )

    async def test_a_timetable_flight_that_nobody_booked_has_its_stable_route_and_time(self):
        first = await self.call("get_flight_status", flight_number="DA 100", date="2026-11-10")
        second = await self.call("get_flight_status", desk(), flight_number="DA 100", date="2026-11-12")
        self.assertEqual((first["status"], second["status"]), ("ok", "ok"))
        self.assertEqual(first["route_spoken"], second["route_spoken"])
        self.assertEqual(first["route_spoken"], "Atlanta to Boston")
        self.assertEqual(first["departure"]["scheduled_spoken"], second["departure"]["scheduled_spoken"])

    async def test_a_number_the_airline_does_not_fly_has_no_status(self):
        nobody = await self.call("get_flight_status", flight_number="DA 12")
        self.assertEqual((nobody["status"], nobody["reason"]), ("no_data", "unknown_flight"))
        self.assertIn("Demo Air flight DA 12", nobody["message"])
        other = await self.call("get_flight_status", flight_number="UA 1534")
        self.assertEqual((other["status"], other["reason"]), ("no_data", "not_demo_air"))
        self.assertIn("whose numbers start with DA", other["message"])
        trend = await self.call("get_ontime_trend", flight_number="UA 1534")
        self.assertEqual(
            (trend["status"], trend["reason"], trend["field"]),
            ("needs_clarification", "not_demo_air", "flight_number"),
        )

    async def test_a_missing_number_or_a_bad_date_is_asked_about(self):
        for arguments, field in (
            ({}, "flight_number"),
            ({"flight_number": "hello"}, "flight_number"),
            ({"flight_number": K7_FLIGHT, "date": "soon"}, "date"),
        ):
            with self.subTest(arguments):
                result = await self.call("get_flight_status", **arguments)
                self.assertEqual((result["status"], result["field"]), ("needs_clarification", field))
        yesterday = await self.call("get_flight_status", flight_number=K7_FLIGHT, date="2026-11-03")
        self.assertEqual((yesterday["status"], yesterday["state"]), ("ok", "landed"))
        self.assertEqual(
            (await self.call("get_flight_status", flight_number=K7_FLIGHT, date="2028-01-01"))["reason"],
            "too_far_ahead",
        )

    async def test_the_ontime_trend_is_a_stable_ninety_day_record(self):
        result = await self.call("get_ontime_trend", flight_number=K7_FLIGHT)
        self.assertEqual((result["status"], result["window_days"], result["data_source"]), ("ok", 90, "sample"))
        self.assertEqual(result["on_time_pct"] + result["late_pct"] + result["very_late_pct"], 100)
        self.assertTrue(0 < result["avg_delay_min"] < 60 and 0 < result["flights_counted"] <= 90)
        self.assertIn(f"on time {result['on_time_pct']} percent", result["summary"])
        again = await self.call("get_ontime_trend", desk(), flight_number="da213")
        self.assertEqual(
            {k: v for k, v in again.items() if k != "data_notice"},
            {k: v for k, v in result.items() if k != "data_notice"},
        )
        other = await self.call("get_ontime_trend", flight_number="DA 100")
        self.assertNotEqual(other["on_time_pct"], result["on_time_pct"])
        self.assertEqual((await self.call("get_ontime_trend", flight_number="nope"))["status"], "needs_clarification")


class AccountTests(DeskCase):
    async def test_the_sample_account_signs_in_however_its_id_is_said(self):
        for said in ("jane_doe_001", "jane doe zero zero one", "Jane Doe 001", "Jane Doe"):
            with self.subTest(said):
                tools = desk()
                result = await self.call("sign_in", tools, user_id=said)
                self.assertEqual(
                    (result["status"], result["account"]["user_id"], result["created"]), ("ok", "jane_doe_001", False)
                )
                self.assertEqual(result["account"]["user_id_spoken"], "jane doe zero zero one")
                self.assertEqual(
                    (result["account"]["name"], result["account"]["tier"], result["trip_count"]),
                    ("Jane Doe", "Gold", 2),
                )
                self.assertEqual({trip["booking_id"] for trip in result["trips"]}, {"K7Q2ZP", "M4X8RT"})
                first = next(trip for trip in result["trips"] if trip["booking_id"] == "K7Q2ZP")
                self.assertEqual(first["trip_spoken"], "Boston to Denver on Thursday, November 5th")
                self.assertEqual((first["flights"][0]["flight"], first["flights"][0]["origin"]), (K7_FLIGHT, "BOS"))
                self.assertIn("delay of about 45 minutes", result["heads_up"])

    async def test_an_unknown_id_asks_to_sign_in_and_names_the_sample_account(self):
        result = await self.call("sign_in", user_id="nobody_007")
        self.assertEqual((result["status"], result["reason"]), ("needs_sign_in", "sign_in_required"))
        self.assertEqual(result["sample_account"], {"user_id": "jane_doe_001", "spoken": "jane doe zero zero one"})
        self.assertIn("jane doe zero zero one", result["question"])
        self.assertIsNone(self.tools.state.account_id)
        bare = await self.call("sign_in")
        self.assertEqual(bare["status"], "needs_sign_in")

    async def test_a_new_demo_account_is_created_in_the_callers_name_once(self):
        result = await self.call("sign_in", new_account_name="alex morgan")
        self.assertEqual((result["status"], result["created"]), ("ok", True))
        self.assertEqual(
            result["account"],
            {
                "user_id": "alex_morgan_003",
                "name": "Alex Morgan",
                "tier": "Member",
                "user_id_spoken": "alex morgan zero zero three",
            },
        )
        self.assertEqual((result["trip_count"], result["trips"]), (0, []))
        self.assertIn("no trips yet", result["summary"])
        again = await self.call("sign_in", new_account_name="Alex Morgan")
        self.assertEqual((again["created"], again["account"]["user_id"]), (False, "alex_morgan_003"))
        by_id = await self.call("sign_in", user_id="alex_morgan_003")
        self.assertEqual(by_id["account"]["name"], "Alex Morgan")
        other = await self.call("sign_in", new_account_name="Ana María O'Neil")
        self.assertTrue(other["created"] and other["account"]["user_id"].startswith("ana_maria_oneil_"))
        for bad in ("12345", "x", "Robert'); DROP TABLE", "a" * 80):
            with self.subTest(bad):
                refused = await self.call("sign_in", new_account_name=bad)
                self.assertEqual((refused["status"], refused["reason"]), ("needs_clarification", "invalid_name"))

    async def test_the_number_of_new_accounts_in_a_session_is_limited(self):
        for index in range(20):
            made = await self.call(
                "sign_in", new_account_name=f"{'abcdefghijklmnopqrstuvwxyz'[index]}b{'x' * 3} Person"
            )
            self.assertEqual(made["status"], "ok")
        refused = await self.call("sign_in", new_account_name="Last Person")
        self.assertEqual(refused["reason"], "invalid_name")

    async def test_the_trips_need_a_sign_in_and_list_every_booking_with_codes_and_words(self):
        asked = await self.call("get_existing_bookings")
        self.assertEqual((asked["status"], asked["sample_account"]["user_id"]), ("needs_sign_in", "jane_doe_001"))
        listed = await self.call("get_existing_bookings", user_id="jane doe zero zero one")
        self.assertEqual((listed["status"], listed["active_count"], listed["cancelled_count"]), ("ok", 2, 0))
        first, second = listed["bookings"]
        self.assertEqual((first["booking_id"], first["booking_id_spoken"]), ("K7Q2ZP", "K 7 Q 2 Z P"))
        self.assertEqual((first["cabin"], first["cabin_spoken"], first["status"]), ("economy", "economy", "confirmed"))
        self.assertEqual(first["trip_spoken"], "Boston to Denver on Thursday, November 5th")
        self.assertEqual((first["flights"][0]["flight"], first["flights"][0]["seat"]), (K7_FLIGHT, "18A"))
        self.assertEqual((first["flights"][0]["origin"], first["flights"][0]["destination"]), ("BOS", "DEN"))
        self.assertEqual(first["flights"][0]["delay_min"], 45)
        self.assertNotIn("needs_rebooking", first)
        self.assertEqual((first["flights"][0]["meal"], first["flights"][0]["meal_spoken"]), ("VGML", "vegan meal"))
        self.assertEqual(
            (second["booking_id"], second["cabin"], second["flights"][0]["seat"]), ("M4X8RT", "business", "5A")
        )
        self.assertEqual((await self.call("get_existing_bookings", user_id="nobody_007"))["status"], "needs_sign_in")
        mine = await self.call("get_existing_bookings")
        self.assertEqual(mine["status"], "ok")

    async def test_a_new_account_has_no_trips(self):
        await self.call("sign_in", new_account_name="Sam Rivera")
        listed = await self.call("get_existing_bookings")
        self.assertEqual((listed["status"], listed["active_count"], listed["bookings"]), ("ok", 0, []))


class BookingTests(DeskCase):
    async def test_a_booking_needs_a_sign_in_and_then_a_search(self):
        needs = await self.call("new_booking", flight="1")
        self.assertEqual((needs["status"], needs["reason"]), ("needs_sign_in", "sign_in_required"))
        await self.sign_in()
        no_search = await self.call("new_booking", flight="1")
        self.assertEqual((no_search["status"], no_search["reason"]), ("needs_search", "search_first"))
        await self.call("flights_search", **SEARCH)
        which = await self.call("new_booking", flight="the purple one")
        self.assertEqual((which["status"], which["reason"]), ("needs_clarification", "which_flight"))
        self.assertEqual(len(which["options"]), 4)
        self.assertEqual(len(self.bookings()), SEEDED)

    async def test_the_server_reads_a_booking_back_and_changes_nothing_until_a_matching_yes(self):
        await self.sign_in()
        searched = await self.call("flights_search", **SEARCH)
        before = len(self.bookings())
        read_back = await self.call("new_booking", flight="2")
        self.assertEqual(
            (read_back["status"], read_back["pending"], read_back["change"]), ("needs_confirmation", True, "booked")
        )
        chosen = searched["flights"][1]
        self.assertEqual(read_back["flight"]["flight"], chosen["flight"])
        self.assertIn("Nothing is booked yet", read_back["confirmation_summary"])
        self.assertTrue(read_back["confirmation_summary"].endswith("Shall I go ahead?"))
        self.assertIn(spoken_amount(chosen["price_usd"]), read_back["confirmation_summary"])
        self.assertEqual(len(self.bookings()), before)
        self.assertEqual(self.tools.state.pending.tool, "new_booking")
        done = await self.call("new_booking", flight="2", confirmed=True)
        self.assertEqual((done["status"], done["change"]), ("confirmed", "booked"))
        booking = done["booking"]
        self.assertRegex(booking["booking_id"], CODE)
        self.assertEqual(booking["booking_id_spoken"], spoken_code(booking["booking_id"]))
        self.assertEqual(
            (booking["passengers"], booking["cabin"], booking["total_usd"]),
            (["Jane Doe"], "economy", chosen["price_usd"]),
        )
        self.assertEqual(booking["flights"][0]["flight"], chosen["flight"])
        self.assertIn(spoken_code(booking["booking_id"]), done["summary"])
        self.assertEqual(len(self.bookings()), before + 1)
        self.assertIsNone(self.tools.state.pending)
        self.assertEqual(self.tools.state.last_change, {"kind": "booked", "booking_id": booking["booking_id"]})
        # The yes was used: asking again to confirm the same booking reads it back instead of booking twice.
        again = await self.call("new_booking", flight="2", confirmed=True)
        self.assertEqual(again["status"], "needs_confirmation")
        self.assertEqual(len(self.bookings()), before + 1)

    async def test_confirmed_with_no_matching_read_back_reads_it_back_and_says_so(self):
        for confirmed in (True, "true", "True"):
            with self.subTest(confirmed=confirmed):
                self.setUp()
                await self.sign_in()
                await self.call("flights_search", **SEARCH)
                result = await self.call("new_booking", flight="1", confirmed=confirmed)
                self.assertEqual((result["status"], result["pending"]), ("needs_confirmation", True))
                self.assertIn("confirmed was not taken", result["note"])
                self.assertEqual(len(self.bookings()), SEEDED)
        # Only the boolean true, or the word, counts as a yes; nothing else books, read-back after read-back.
        self.setUp()
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        for confirmed in (False, "yes", "false", 1, None, [], {}):
            with self.subTest(confirmed=confirmed):
                result = await self.call("new_booking", flight="1", confirmed=confirmed)
                self.assertEqual(result["status"], "needs_confirmation")
                self.assertNotIn("note", result)
        self.assertEqual(len(self.bookings()), SEEDED)

    async def test_a_changed_parameter_reads_the_new_change_back(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        before = len(self.bookings())
        await self.call("new_booking", flight="1")
        for changed in ({"flight": "2"}, {"cabin": "business"}, {"passengers": "Jane Doe, Sam Doe"}):
            with self.subTest(changed):
                await self.call("new_booking", flight="1")
                moved = await self.call("new_booking", **{"flight": "1", **changed, "confirmed": True})
                self.assertEqual((moved["status"], moved["pending"]), ("needs_confirmation", True))
                self.assertIn("confirmed was not taken", moved["note"])
        self.assertEqual(len(self.bookings()), before)
        # The read-back that matches is the one that books.
        await self.call("new_booking", flight="1", cabin="business", passengers="Jane Doe and Sam Doe")
        done = await self.call(
            "new_booking", flight="1", cabin="business", passengers="Jane Doe and Sam Doe", confirmed=True
        )
        self.assertEqual(done["status"], "confirmed")
        self.assertEqual(
            (done["booking"]["passengers"], done["booking"]["cabin"]), (["Jane Doe", "Sam Doe"], "business")
        )
        self.assertEqual(len(self.bookings()), before + 1)

    async def test_a_yes_for_another_tool_or_after_a_different_search_does_not_book(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        await self.call("new_booking", flight="1")
        await self.call("cancel_booking", booking_id="K7Q2ZP")
        self.assertEqual(self.tools.state.pending.tool, "cancel_booking")
        wrong = await self.call("new_booking", flight="1", confirmed=True)
        self.assertEqual(wrong["status"], "needs_confirmation")
        self.assertEqual(self.tools.store.booking("K7Q2ZP").status, "confirmed")
        await self.call("new_booking", flight="1")
        await self.call("flights_search", **{**SEARCH, "date": "2026-11-06"})
        self.assertIsNone(self.tools.state.pending)
        after = await self.call("new_booking", flight="1", confirmed=True)
        self.assertEqual(after["status"], "needs_confirmation")
        self.assertEqual(len(self.bookings()), SEEDED)

    async def test_running_the_same_search_again_keeps_the_read_back(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        await self.call("new_booking", flight="1")
        await self.call("flights_search", **SEARCH)
        self.assertEqual((await self.call("new_booking", flight="1", confirmed=True))["status"], "confirmed")

    async def test_the_same_call_id_returns_its_first_result_and_changes_nothing_twice(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        await self.call("new_booking", flight="1")
        arguments = json.dumps({"flight": "1", "confirmed": True})
        first = await self.tools.execute("replayed", "new_booking", arguments)
        again = await self.tools.execute("replayed", "new_booking", arguments)
        self.assertEqual((json.loads(first)["status"], again), ("confirmed", first))
        self.assertEqual(len(self.bookings()), SEEDED + 1)
        # A replay under another name or arguments is still the first call.
        other = await self.tools.execute("replayed", "cancel_booking", "{}")
        self.assertEqual(other, first)

    async def test_a_flight_is_named_by_option_ordinal_flight_number_time_or_price(self):
        await self.sign_in()
        found = (await self.call("flights_search", **FIVE_A_DAY))["flights"]
        searched = self.tools.state.search.results
        cases = {
            "1": 0,
            "option 3": 2,
            "#4": 3,
            "second": 1,
            "the fifth one": 4,
            "last": 4,
            found[1]["flight"]: 1,
            found[1]["flight"].replace(" ", "").lower(): 1,
            found[2]["flight"].split()[1]: 2,
            "cheapest": min(range(5), key=lambda i: found[i]["price_usd"]),
            "earliest": 0,
            found[2]["depart_spoken"]: 2,
            found[4]["depart_local"][11:16]: 4,
        }
        for named, index in cases.items():
            with self.subTest(named):
                read_back = await self.call("new_booking", flight=named)
                self.assertEqual(read_back["flight"]["flight"], searched[index]["flight_number"])
        self.assertEqual((await self.call("new_booking", flight=2))["flight"]["flight"], found[1]["flight"])
        self.assertEqual((await self.call("new_booking", flight="9"))["reason"], "which_flight")
        # The airline's name names no flight: every flight is Demo Air's.
        self.assertEqual((await self.call("new_booking", flight="Demo Air"))["reason"], "which_flight")

    async def test_the_passengers_default_to_the_account_holder_and_are_cleaned(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        for given, names in (
            (None, ["Jane Doe"]),
            ("me, sam rivera and Ana", ["Jane Doe", "Sam Rivera", "Ana"]),
            ("sam, sam", ["Sam"]),
            (["x one", "y two"], ["X One", "Y Two"]),
            (",,", ["Jane Doe"]),
            (", ".join(f"p{i}" for i in range(9)), [f"P{i}" for i in range(6)]),
        ):
            with self.subTest(given):
                result = await self.call(
                    "new_booking", flight="1", **({"passengers": given} if given is not None else {})
                )
                self.assertEqual(result["passengers"], names)

    async def test_the_cabin_of_a_booking_defaults_to_the_search_and_prices_each_passenger(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH, cabin="premium economy")
        await self.call("new_booking", flight="1", passengers="Jane Doe, Sam Doe")
        done = await self.call("new_booking", flight="1", passengers="Jane Doe, Sam Doe", confirmed=True)
        booking = done["booking"]
        self.assertEqual(booking["cabin"], "premium_economy")
        flight = self.tools.state.search.results[0]
        self.assertEqual(booking["total_usd"], 2 * flight["fare"]["amount"])

    async def test_a_modification_moves_the_booking_and_keeps_the_meal(self):
        await self.sign_in()
        await self.call("flights_search", **{**SEARCH, "date": "2026-11-06"})
        target = self.tools.state.search.results[0]
        read_back = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP")
        self.assertEqual((read_back["status"], read_back["change"]), ("needs_confirmation", "modified"))
        self.assertEqual(read_back["new_flight"]["flight"], target["flight_number"])
        self.assertIn(f"move booking K7Q2ZP from {K7_SAYS}", read_back["confirmation_summary"])
        self.assertIn("The seat is released; the meal stays", read_back["confirmation_summary"])
        self.assertRegex(
            read_back["confirmation_summary"],
            r"fare difference is \$\d+ more|fare is \$\d+ less|no fare difference",
        )
        booking = self.tools.store.booking("K7Q2ZP")
        self.assertEqual((booking.legs[0].flight_number, booking.legs[0].seat), (K7_FLIGHT, "18A"))
        done = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP", confirmed=True)
        self.assertEqual((done["status"], done["change"]), ("confirmed", "modified"))
        leg = done["booking"]["flights"][0]
        self.assertEqual((leg["flight"], leg["seat"], leg["meal"]), (target["flight_number"], None, "VGML"))
        self.assertEqual(done["fare_difference_usd"], read_back["fare_difference_usd"])
        self.assertEqual(self.tools.state.last_change, {"kind": "modified", "booking_id": "K7Q2ZP"})

    async def test_a_modification_needs_a_search_a_matching_route_and_an_active_booking(self):
        await self.sign_in()
        self.assertEqual(
            (await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP"))["status"], "needs_search"
        )
        await self.call("flights_search", origin="Boston", destination="Chicago", date=TOMORROW)
        # A flight to Chicago replaces the whole one-flight trip.
        self.assertEqual(
            (await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP"))["status"],
            "needs_confirmation",
        )
        await self.call("cancel_booking", booking_id="K7Q2ZP")
        await self.call("cancel_booking", booking_id="K7Q2ZP", confirmed=True)
        cancelled = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP")
        self.assertEqual((cancelled["status"], cancelled["reason"]), ("error", "booking_cancelled"))

    async def test_a_change_to_another_route_names_both_routes_in_the_read_back(self):
        await self.sign_in()
        before = [leg.flight_number for leg in self.tools.store.booking("K7Q2ZP").legs]
        await self.call("flights_search", origin="New York", destination="Los Angeles", date=TOMORROW)
        other = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP")
        self.assertEqual(other["status"], "needs_confirmation")
        self.assertIn(", Boston to Denver, on", other["confirmation_summary"])
        self.assertIn(", New York to Los Angeles, on", other["confirmation_summary"])
        self.assertEqual([leg.flight_number for leg in self.tools.store.booking("K7Q2ZP").legs], before)
        # A change on the same route names no route: the flights and times say it all.
        await self.call("flights_search", **SEARCH)
        same = await self.call("modify_existing_booking", flight="2", booking_id="K7Q2ZP")
        self.assertNotIn("Boston to Denver", same["confirmation_summary"])

    async def test_first_class_in_a_choice_names_a_cabin_not_the_first_flight(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        results = self.tools.state.search.results
        second = spoken_time(results[1]["depart_local"]).lower()
        self.assertEqual(airline_tools._chosen_index(results, f"the {second} one in first class"), 1)
        self.assertEqual(airline_tools._chosen_index(results, "the first one"), 0)

    async def test_a_round_trip_changes_only_the_leg_whose_route_matches(self):
        await self.sign_in()
        store = self.tools.store
        out = store.booking("K7Q2ZP").legs[0]
        back = schedule.sample_flights(airports.get_airport("DEN"), airports.get_airport("BOS"), date(2026, 11, 12))[0]
        store.booking("K7Q2ZP").legs.append(Leg.from_flight(self.tools._priced(back, "economy"), 200.0))
        await self.call("flights_search", origin="Denver", destination="Boston", date="2026-11-13")
        await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP")
        done = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP", confirmed=True)
        legs = done["booking"]["flights"]
        self.assertEqual((legs[0]["flight"], len(legs)), (out.flight_number, 2))
        self.assertEqual(legs[1]["flight"], self.tools.state.search.results[0]["flight_number"])
        await self.call("flights_search", origin="Chicago", destination="Miami", date=TOMORROW)
        mismatch = await self.call("modify_existing_booking", flight="1", booking_id="K7Q2ZP")
        self.assertEqual((mismatch["status"], mismatch["reason"]), ("needs_clarification", "route_mismatch"))

    async def test_a_cancellation_states_the_refund_rule_of_its_fare(self):
        await self.sign_in()
        economy = await self.call("cancel_booking", booking_id="K7Q2ZP")
        self.assertEqual((economy["status"], economy["change"]), ("needs_confirmation", "cancelled"))
        self.assertIn("non-refundable", economy["refund_rule"])
        self.assertIn("travel credit", economy["refund_rule"])
        self.assertIn("Nothing is cancelled yet", economy["confirmation_summary"])
        self.assertEqual(self.tools.store.booking("K7Q2ZP").status, "confirmed")
        business = await self.call("cancel_booking", booking_id="M4X8RT")
        self.assertIn("fully refundable", business["refund_rule"])
        done = await self.call("cancel_booking", booking_id="M4X8RT", confirmed=True)
        self.assertEqual((done["status"], done["booking"]["status"]), ("confirmed", "cancelled"))
        self.assertEqual(self.tools.store.booking("M4X8RT").status, "cancelled")
        self.assertEqual(self.tools.state.last_change, {"kind": "cancelled", "booking_id": "M4X8RT"})
        again = await self.call("cancel_booking", booking_id="M4X8RT")
        self.assertEqual((again["status"], again["reason"]), ("error", "booking_cancelled"))
        listed = await self.call("get_existing_bookings")
        self.assertEqual((listed["active_count"], listed["cancelled_count"]), (1, 1))

    async def test_a_booking_made_in_the_session_is_refunded_in_full(self):
        booking = await self.booked()
        read_back = await self.call("cancel_booking")
        self.assertEqual(read_back["booking_id"], booking["booking_id"])
        self.assertIn("full", read_back["refund_rule"])
        premium = self.tools.store.booking("M4X8RT")
        premium.cabin = "premium_economy"
        self.assertIn("minus a $100 fee", (await self.call("cancel_booking", booking_id="M4X8RT"))["refund_rule"])

    async def test_a_booking_code_alone_finds_the_account(self):
        for said in ("K7Q2ZP", "k7q2zp", "K 7 Q 2 Z P", "k seven q two z p", "K-7-Q-2-Z-P"):
            with self.subTest(said):
                tools = desk()
                self.assertIsNone(tools.state.account_id)
                result = await self.call("cancel_booking", tools, booking_id=said)
                self.assertEqual((result["status"], result["booking_id"]), ("needs_confirmation", "K7Q2ZP"))
                self.assertEqual(tools.state.account_id, "jane_doe_001")
        for tool, arguments in (
            ("select_seat_and_meal", {"booking_id": "M4X8RT", "seat": "window"}),
            ("upgrade_class", {"booking_id": "K7Q2ZP", "cabin": "business"}),
        ):
            with self.subTest(tool):
                self.assertEqual((await self.call(tool, desk(), **arguments))["status"], "needs_confirmation")
        modify = desk()
        await self.call("flights_search", modify, **SEARCH)
        self.assertEqual(
            (await self.call("modify_existing_booking", modify, flight="1", booking_id="K7Q2ZP"))["status"],
            "needs_confirmation",
        )

    async def test_an_unknown_code_is_not_found_and_another_accounts_code_is_refused(self):
        unknown = await self.call("cancel_booking", booking_id="ZZZZZZ")
        self.assertEqual(
            (unknown["status"], unknown["reason"], unknown["field"]), ("not_found", "unknown_booking", "booking_id")
        )
        await self.call("sign_in", new_account_name="Sam Rivera")
        other = await self.call("cancel_booking", booking_id="K7Q2ZP")
        self.assertEqual((other["status"], other["reason"]), ("error", "other_account"))
        self.assertEqual(self.tools.state.account_id, "sam_rivera_003")

    async def test_without_a_code_the_desk_uses_the_booking_just_made_else_the_only_active_one_else_asks(self):
        needs = await self.call("select_seat_and_meal", seat="window")
        self.assertEqual((needs["status"], needs["reason"]), ("needs_sign_in", "sign_in_required"))
        await self.sign_in()
        which = await self.call("cancel_booking")
        self.assertEqual(
            (which["status"], which["reason"], which["field"]), ("needs_booking_id", "which_booking", "booking_id")
        )
        self.assertEqual([item["booking_id"] for item in which["bookings"]], ["K7Q2ZP", "M4X8RT"])
        self.assertIn("Boston to Denver on Thursday, November 5th, code K7Q2ZP", which["question"])
        made = await self.booked()
        again = await self.call("select_seat_and_meal", seat="aisle")
        self.assertEqual(again["booking_id"], made["booking_id"])
        await self.call("cancel_booking", booking_id="K7Q2ZP")
        await self.call("cancel_booking", booking_id="K7Q2ZP", confirmed=True)
        await self.call("cancel_booking", booking_id=made["booking_id"])
        await self.call("cancel_booking", booking_id=made["booking_id"], confirmed=True)
        only = await self.call("upgrade_class", cabin="first")
        self.assertEqual(only["booking_id"], "M4X8RT")
        await self.call("cancel_booking", booking_id="M4X8RT")
        await self.call("cancel_booking", booking_id="M4X8RT", confirmed=True)
        none = await self.call("cancel_booking")
        self.assertEqual((none["status"], none["reason"]), ("no_data", "no_bookings"))

    async def test_a_seat_by_number_or_kind_and_a_meal_by_words_or_code(self):
        await self.sign_in()
        window = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="window")
        self.assertEqual((window["status"], window["change"]), ("needs_confirmation", "seat_meal"))
        seat = window["seats"][K7_FLIGHT]
        self.assertRegex(seat, r"^\d+[AF]$")
        # Row 15 is an exit row, and the summary says so.
        self.assertRegex(window["confirmation_summary"], rf"seat {seat}, a window seat(, an exit row)? on {K7_SAYS}")
        self.assertEqual(self.tools.store.booking("K7Q2ZP").legs[0].seat, "18A")
        done = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="window", confirmed=True)
        self.assertEqual((done["status"], done["booking"]["flights"][0]["seat"]), ("confirmed", seat))
        self.assertEqual(done["booking"]["flights"][0]["meal"], "VGML")
        for meal, code in (
            ("kosher", "KSML"),
            ("Vegetarian", "VLML"),
            ("gluten free", "GFML"),
            ("MOML", "MOML"),
            ("asian vegetarian", "AVML"),
        ):
            with self.subTest(meal):
                await self.call("select_seat_and_meal", booking_id="K7Q2ZP", meal=meal)
                changed = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", meal=meal, confirmed=True)
                self.assertEqual(changed["booking"]["flights"][0]["meal"], code)
                self.assertEqual(changed["booking"]["flights"][0]["seat"], seat)
        await self.call("select_seat_and_meal", booking_id="K7Q2ZP", meal="standard")
        cleared = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", meal="standard", confirmed=True)
        self.assertIsNone(cleared["booking"]["flights"][0]["meal"])

    async def test_an_explicit_seat_an_exit_row_and_a_seat_that_is_taken(self):
        await self.sign_in()
        keep = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="18A")
        self.assertEqual(keep["seats"], {K7_FLIGHT: "18A"})
        exit_row = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="exit row")
        self.assertIn(int(exit_row["seats"][K7_FLIGHT][:-1]), {15, 28})
        self.assertIn("an exit row", exit_row["confirmation_summary"])
        business = await self.call("select_seat_and_meal", booking_id="M4X8RT", seat="exit row and window")
        self.assertEqual(business["status"], "needs_confirmation")
        wrong_row = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="3A")
        self.assertEqual((wrong_row["status"], wrong_row["reason"]), ("needs_clarification", "seat_unavailable"))
        self.assertIn("Row 3 isn't in economy", wrong_row["question"])
        unclear = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", seat="somewhere nice")
        self.assertEqual(unclear["reason"], "seat_unclear")
        # A seat that another booking of the session holds on the same flight is taken.
        made = await self.booked()
        flight = self.tools.store.booking(made["booking_id"]).legs[0]
        held_by = self.tools.store.booking("K7Q2ZP")
        held_by.legs[0].flight_number, held_by.legs[0].depart_local = flight.flight_number, flight.depart_local
        held_by.legs[0].seat = "20C"
        taken = await self.call("select_seat_and_meal", booking_id=made["booking_id"], seat="20C")
        self.assertEqual((taken["status"], taken["reason"]), ("needs_clarification", "seat_unavailable"))
        self.assertIn("20C is taken", taken["question"])

    async def test_a_seat_or_meal_must_be_named_and_the_meal_known(self):
        await self.sign_in()
        nothing = await self.call("select_seat_and_meal", booking_id="K7Q2ZP")
        self.assertEqual((nothing["status"], nothing["reason"]), ("needs_clarification", "seat_or_meal"))
        unknown = await self.call("select_seat_and_meal", booking_id="K7Q2ZP", meal="moon dust")
        self.assertEqual((unknown["status"], unknown["reason"]), ("needs_clarification", "unknown_meal"))

    async def test_an_upgrade_states_the_price_difference_and_releases_the_seats(self):
        await self.sign_in()
        read_back = await self.call("upgrade_class", booking_id="K7Q2ZP", cabin="business")
        self.assertEqual(
            (read_back["status"], read_back["change"], read_back["cabin"]),
            ("needs_confirmation", "upgraded", "business"),
        )
        self.assertGreater(read_back["fare_difference_usd"], 0)
        self.assertIn("upgrade booking K7Q2ZP from economy to business", read_back["confirmation_summary"])
        self.assertIn("Seats are released", read_back["confirmation_summary"])
        before = self.tools.store.booking("K7Q2ZP").total
        done = await self.call("upgrade_class", booking_id="K7Q2ZP", cabin="business", confirmed=True)
        self.assertEqual((done["status"], done["booking"]["cabin"]), ("confirmed", "business"))
        self.assertIsNone(done["booking"]["flights"][0]["seat"])
        self.assertAlmostEqual(done["booking"]["total_usd"] - before, read_back["fare_difference_usd"])
        down = await self.call("upgrade_class", booking_id="K7Q2ZP", cabin="economy")
        self.assertIn("move booking K7Q2ZP from business to economy", down["confirmation_summary"])
        self.assertLess(down["fare_difference_usd"], 0)
        self.assertRegex(down["confirmation_summary"], r"The fare is \$\d+ less")
        self.assertNotIn("demo", down["confirmation_summary"])

    async def test_an_upgrade_to_the_same_or_an_unknown_cabin_is_not_run(self):
        await self.sign_in()
        same = await self.call("upgrade_class", booking_id="K7Q2ZP", cabin="economy")
        self.assertEqual((same["status"], same["reason"]), ("error", "same_cabin"))
        for cabin in ("steerage", "", None):
            with self.subTest(cabin):
                result = await self.call("upgrade_class", booking_id="K7Q2ZP", cabin=cabin)
                self.assertEqual((result["status"], result["reason"]), ("needs_clarification", "which_cabin"))
        self.assertEqual(
            (await self.call("upgrade_class", booking_id="K7Q2ZP", cabin="Business Class"))["cabin"], "business"
        )

    async def test_every_change_tool_reads_back_first_whatever_confirmed_says(self):
        await self.sign_in()
        await self.call("flights_search", **SEARCH)
        calls = {
            "new_booking": {"flight": "1"},
            "modify_existing_booking": {"flight": "1", "booking_id": "K7Q2ZP"},
            "cancel_booking": {"booking_id": "K7Q2ZP"},
            "select_seat_and_meal": {"booking_id": "K7Q2ZP", "seat": "aisle"},
            "upgrade_class": {"booking_id": "K7Q2ZP", "cabin": "business"},
        }
        snapshot = json.dumps([booking.view() for booking in self.bookings()], sort_keys=True)
        for name, arguments in calls.items():
            with self.subTest(name):
                result = await self.call(name, **arguments, confirmed=True)
                self.assertEqual((result["status"], result["pending"]), ("needs_confirmation", True))
                self.assertTrue(result["confirmation_summary"])
        self.assertEqual(json.dumps([booking.view() for booking in self.bookings()], sort_keys=True), snapshot)

    async def test_booking_codes_look_like_codes_and_repeat_for_a_seed(self):
        codes = set()
        for seed in range(30):
            tools = AirlineTools(clock=lambda: NOW, seed=seed)
            code = tools.store.new_booking_id()
            self.assertRegex(code, CODE)
            codes.add(code)
        self.assertGreater(len(codes), 25)
        self.assertEqual(desk().store.new_booking_id(), desk().store.new_booking_id())
        self.assertNotIn(AirlineTools().store.new_booking_id(), {"K7Q2ZP", "M4X8RT", "R9T4WN"})


class ErrorTests(DeskCase):
    async def test_an_unknown_tool_or_unusable_arguments_are_an_error_result(self):
        unknown = json.loads(await self.tools.execute("u", "book_hotel", "{}"))
        self.assertEqual((unknown["status"], unknown["reason"]), ("error", "unknown_tool"))
        self.assertIn("book_hotel", unknown["error"])
        for index, arguments in enumerate(("not json", "[1]", '"text"', "null", "42")):
            with self.subTest(arguments):
                result = json.loads(await self.tools.execute(f"bad{index}", "flights_search", arguments))
                self.assertEqual((result["status"], result["reason"]), ("error", "bad_arguments"))
        blank = json.loads(await self.tools.execute("blank", "sign_in", ""))
        self.assertEqual(blank["status"], "needs_sign_in")
        wrong_types = json.loads(
            await self.tools.execute("w", "flights_search", json.dumps({"origin": 5, "destination": ["x"], "date": 3}))
        )
        self.assertEqual(wrong_types["status"], "needs_clarification")

    async def test_a_bug_in_a_tool_ends_as_an_error_result(self):
        class Broken(AirlineTools):
            def sign_in(self, args):
                raise RuntimeError("boom")

        result = json.loads(await Broken(clock=lambda: NOW).execute("b", "sign_in", "{}"))
        self.assertEqual((result["status"], result["reason"]), ("error", "tool_failed"))
        self.assertIn("boom", result["error"])

    async def test_every_result_is_compact_json_with_a_status(self):
        calls = [
            ("flights_search", SEARCH),
            ("where_we_fly", {}),
            ("get_flight_status", {"flight_number": K7_FLIGHT}),
            ("get_ontime_trend", {"flight_number": K7_FLIGHT}),
            ("sign_in", {"user_id": "jane_doe_001"}),
            ("get_existing_bookings", {}),
            ("new_booking", {"flight": "1"}),
            ("cancel_booking", {"booking_id": "K7Q2ZP"}),
        ]
        for name, arguments in calls:
            with self.subTest(name):
                text = await self.tools.execute(f"id_{name}", name, json.dumps(arguments))
                self.assertIsInstance(text, str)
                result = json.loads(text)
                self.assertIn("status", result)
                self.assertLess(len(text), 8000)
                self.assertNotIn("\n", text)


if __name__ == "__main__":
    unittest.main()
