# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The desk's network: the twelve airports Demo Air serves, and the places it does not serve.

The desk is a demo with a deliberately small world. A search for any other place does not fail quietly: the desk says
which place it does not serve and offers the closest served city (:func:`nearest_served`). Airport facts (codes,
names, positions) are public geographic data, as published by OurAirports (public domain); time zones are IANA names.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, tzinfo
from functools import cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

MILES_PER_KM = 0.621371

# One row per served airport: IATA code, city, state, airport name, IANA time zone, latitude, longitude, size. The size
# (3 large, 2 medium) sets how often the airline flies a route (``schedule.frequency``).
_SERVED = """
ATL|Atlanta|GA|Hartsfield-Jackson Atlanta International|America/New_York|33.64|-84.43|3
BOS|Boston|MA|Logan International|America/New_York|42.36|-71.01|2
DEN|Denver|CO|Denver International|America/Denver|39.86|-104.67|3
DFW|Dallas|TX|Dallas Fort Worth International|America/Chicago|32.90|-97.04|3
JFK|New York|NY|John F. Kennedy International|America/New_York|40.64|-73.78|3
LAS|Las Vegas|NV|Harry Reid International|America/Los_Angeles|36.08|-115.15|2
LAX|Los Angeles|CA|Los Angeles International|America/Los_Angeles|33.94|-118.41|3
MIA|Miami|FL|Miami International|America/New_York|25.79|-80.29|2
ORD|Chicago|IL|O'Hare International|America/Chicago|41.98|-87.90|3
PHX|Phoenix|AZ|Phoenix Sky Harbor International|America/Phoenix|33.43|-112.01|2
SEA|Seattle|WA|Seattle-Tacoma International|America/Los_Angeles|47.45|-122.31|2
SFO|San Francisco|CA|San Francisco International|America/Los_Angeles|37.62|-122.38|2
"""

# Places that the airline does not serve, which a caller may still ask for: name, state or country, airport code,
# latitude, longitude. Each gets the nearest served city. Only these and the served cities are recognised.
_UNSERVED = """
Albuquerque|New Mexico|ABQ|35.04|-106.61
Anchorage|Alaska|ANC|61.17|-150.00
Austin|Texas|AUS|30.19|-97.67
Baltimore|Maryland|BWI|39.18|-76.67
Birmingham|Alabama|BHM|33.56|-86.75
Boise|Idaho|BOI|43.56|-116.22
Buffalo|New York|BUF|42.94|-78.73
Burbank|California|BUR|34.20|-118.36
Charleston|South Carolina|CHS|32.90|-80.04
Charlotte|North Carolina|CLT|35.21|-80.94
Cincinnati|Ohio|CVG|39.05|-84.66
Cleveland|Ohio|CLE|41.41|-81.85
Colorado Springs|Colorado|COS|38.81|-104.70
Columbus|Ohio|CMH|39.99|-82.89
Detroit|Michigan|DTW|42.21|-83.35
El Paso|Texas|ELP|31.81|-106.38
Fort Lauderdale|Florida|FLL|26.07|-80.15
Hartford|Connecticut|BDL|41.94|-72.68
Honolulu|Hawaii|HNL|21.32|-157.92
Houston|Texas|IAH|29.98|-95.34
Indianapolis|Indiana|IND|39.72|-86.29
Jacksonville|Florida|JAX|30.49|-81.69
Kansas City|Missouri|MCI|39.30|-94.71
LaGuardia|New York|LGA|40.78|-73.87
Long Beach|California|LGB|33.82|-118.15
Louisville|Kentucky|SDF|38.17|-85.74
Memphis|Tennessee|MEM|35.04|-89.98
Milwaukee|Wisconsin|MKE|42.95|-87.90
Minneapolis|Minnesota|MSP|44.88|-93.22
Nashville|Tennessee|BNA|36.12|-86.68
New Orleans|Louisiana|MSY|29.99|-90.26
Newark|New Jersey|EWR|40.69|-74.17
Oakland|California|OAK|37.72|-122.22
Oklahoma City|Oklahoma|OKC|35.39|-97.60
Omaha|Nebraska|OMA|41.30|-95.89
Orlando|Florida|MCO|28.43|-81.31
Palm Springs|California|PSP|33.83|-116.51
Philadelphia|Pennsylvania|PHL|39.87|-75.24
Pittsburgh|Pennsylvania|PIT|40.49|-80.23
Portland|Oregon|PDX|45.59|-122.60
Providence|Rhode Island|PVD|41.72|-71.43
Raleigh|North Carolina|RDU|35.88|-78.79
Reno|Nevada|RNO|39.50|-119.77
Sacramento|California|SMF|38.70|-121.59
Salt Lake City|Utah|SLC|40.79|-111.98
San Antonio|Texas|SAT|29.53|-98.47
San Diego|California|SAN|32.73|-117.19
San Jose|California|SJC|37.36|-121.93
Spokane|Washington|GEG|47.62|-117.53
St. Louis|Missouri|STL|38.75|-90.37
Tampa|Florida|TPA|27.98|-82.53
Tucson|Arizona|TUS|32.12|-110.94
Washington|D.C.|DCA|38.85|-77.04
Amsterdam|Netherlands|AMS|52.31|4.76
Cancun|Mexico|CUN|21.04|-86.87
Delhi|India|DEL|28.56|77.10
Dubai|United Arab Emirates|DXB|25.25|55.36
Frankfurt|Germany|FRA|50.03|8.57
London|United Kingdom|LHR|51.47|-0.45
Madrid|Spain|MAD|40.47|-3.57
Mexico City|Mexico|MEX|19.44|-99.07
Montreal|Canada|YUL|45.47|-73.74
Mumbai|India|BOM|19.09|72.87
Paris|France|CDG|49.01|2.55
Rome|Italy|FCO|41.80|12.25
Singapore|Singapore|SIN|1.36|103.99
Sydney|Australia|SYD|-33.95|151.18
Tokyo|Japan|HND|35.55|139.78
Toronto|Canada|YYZ|43.68|-79.63
Vancouver|Canada|YVR|49.19|-123.18
"""

# Short and alternative names of served cities and of unserved ones, by their normalized spelling, to an airport code.
_ALIASES = {
    "new york city": "JFK",
    "nyc": "JFK",
    "manhattan": "JFK",
    "kennedy": "JFK",
    "chi": "ORD",
    "chicago o hare": "ORD",
    "o hare": "ORD",
    "ohare": "ORD",
    "la": "LAX",
    "l a": "LAX",
    "sf": "SFO",
    "the bay area": "SFO",
    "bay area": "SFO",
    "dallas fort worth": "DFW",
    "fort worth": "DFW",
    "dallas forth worth": "DFW",
    "vegas": "LAS",
    "sky harbor": "PHX",
    "sea tac": "SEA",
    "seatac": "SEA",
    "logan": "BOS",
    "hartsfield": "ATL",
    # Unserved places that go by another name.
    "washington dc": "DCA",
    "washington d c": "DCA",
    "dc": "DCA",
    "dulles": "DCA",
    "bombay": "BOM",
    "new delhi": "DEL",
    "saint louis": "STL",
    "st louis": "STL",
    "salt lake": "SLC",
    "fort lauderdale hollywood": "FLL",
    "nola": "MSY",
}
_NOISE_WORDS = re.compile(r"\b(?:international|intl|airport|airfield|the|city of|in|to|from)\b")


@dataclass(frozen=True, slots=True)
class Airport:
    """One served airport."""

    iata: str
    city: str
    state: str
    name: str
    country: str
    tz: str
    lat: float
    lon: float
    size: int

    @property
    def place(self) -> str:
        """The airport's city, with its state: "Seattle, WA"."""
        return f"{self.city}, {self.state}"

    @property
    def zone(self) -> tzinfo:
        """The airport's time zone; UTC when the system has no time zone data for it."""
        return _zone(self.tz)

    def ref(self) -> dict[str, str]:
        """Return the airport as the ``{"code", "city"}`` reference that flights and bookings carry."""
        return {"code": self.iata, "city": self.city}


@dataclass(frozen=True, slots=True)
class Place:
    """A place that the airline does not serve."""

    name: str
    region: str
    iata: str
    lat: float
    lon: float

    @property
    def label(self) -> str:
        """The place as it is said: "Boise, Idaho"; a place that goes by one name, such as "Singapore", alone."""
        return self.name if self.region == self.name else f"{self.name}, {self.region}"


@dataclass(frozen=True, slots=True)
class AirportMatch:
    """The result of resolving a place: a served airport, an unserved place, or neither."""

    airport: Airport | None = None
    unserved: Place | None = None


@cache
def _zone(name: str) -> tzinfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return UTC


def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", folded)).strip()


@cache
def all_airports() -> dict[str, Airport]:
    """Return every served airport, by IATA code."""
    airports = {}
    for row in _SERVED.strip().splitlines():
        iata, city, state, name, tz, lat, lon, size = row.split("|")
        airports[iata] = Airport(iata, city, state, name, "US", tz, float(lat), float(lon), int(size))
    return airports


@cache
def all_places() -> dict[str, Place]:
    """Return every unserved place the desk knows, by airport code."""
    places = {}
    for row in _UNSERVED.strip().splitlines():
        name, region, iata, lat, lon = row.split("|")
        places[iata] = Place(name, region, iata, float(lat), float(lon))
    return places


@cache
def _names() -> dict[str, tuple[str, str]]:
    """Return every spoken name, as ``(kind, code)``: kind is "served" or "unserved"."""
    names: dict[str, tuple[str, str]] = {}
    for place in all_places().values():
        names[_normalize(place.name)] = ("unserved", place.iata)
    for airport in all_airports().values():
        names[_normalize(airport.city)] = ("served", airport.iata)
    for alias, code in _ALIASES.items():
        names[alias] = ("served", code) if code in all_airports() else ("unserved", code)
    return names


def get_airport(code: str | None) -> Airport | None:
    """Return the served airport for an IATA code, or None."""
    return all_airports().get((code or "").strip().upper())


def _found(kind: str, code: str) -> AirportMatch:
    return AirportMatch(airport=all_airports()[code]) if kind == "served" else AirportMatch(unserved=all_places()[code])


def resolve_airport(text: str | None) -> AirportMatch:
    """Resolve a city, alias, airport name or IATA code to a served airport, or to an unserved place, or to neither.

    A trailing state or country is ignored ("Denver Colorado"). A place that is neither served nor in the desk's list
    of known places returns an empty match.
    """
    raw = (text or "").strip()
    if not raw:
        return AirportMatch()
    if re.fullmatch(r"[A-Za-z]{3}", raw):
        code = raw.upper()
        if code in all_airports():
            return _found("served", code)
        if code in all_places():
            return _found("unserved", code)
    names = _names()
    query = _normalize(raw)
    stripped = _normalize(_NOISE_WORDS.sub(" ", query))
    for candidate in dict.fromkeys((query, stripped)):
        words = candidate.split()
        for split in range(len(words), 0, -1):
            hit = names.get(" ".join(words[:split]))
            if hit:
                return _found(*hit)
    if len(stripped) >= 4:
        for airport in all_airports().values():
            if re.search(rf"\b{re.escape(stripped)}\b", _normalize(airport.name)):
                return _found("served", airport.iata)
    return AirportMatch()


def great_circle_km(origin: Airport, destination: Airport) -> float:
    """Return the great-circle distance between two airports in kilometres."""
    return _distance_km(origin.lat, origin.lon, destination.lat, destination.lon)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_served(place: Place) -> tuple[Airport, int]:
    """Return the served airport closest to an unserved place, and its distance in whole miles."""
    airport = min(all_airports().values(), key=lambda item: _distance_km(place.lat, place.lon, item.lat, item.lon))
    return airport, round(_distance_km(place.lat, place.lon, airport.lat, airport.lon) * MILES_PER_KM)


def local_datetime(airport: Airport, day: date, clock: time) -> datetime:
    """Return an aware local datetime at ``airport``."""
    return datetime.combine(day, clock, tzinfo=airport.zone)
