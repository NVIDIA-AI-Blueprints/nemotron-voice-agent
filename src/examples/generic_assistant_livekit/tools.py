# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plain async helpers behind the agent's tools. No LiveKit imports, so they are easy to unit-test."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "heavy freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "heavy freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "light rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "light snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}


def current_datetime(timezone: str = "") -> dict[str, str]:
    """Return the current date and time in an IANA timezone (default: the machine's local timezone)."""
    try:
        now = datetime.now(ZoneInfo(timezone)) if timezone else datetime.now().astimezone()
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(f"Unknown timezone '{timezone}'. Use an IANA name such as 'Europe/Paris'.") from None
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p").lstrip("0"),
        "timezone": timezone or str(now.tzinfo),
        "iso": now.isoformat(timespec="seconds"),
    }


def describe_weather(code: int) -> str:
    """Return the words for a WMO weather code."""
    return WEATHER_CODES.get(code, f"weather code {code}")


async def _get_json(session, url: str, params: dict) -> dict:
    async with session.get(url, params=params) as resp:
        resp.raise_for_status()
        return await resp.json()


async def current_weather(location: str, fahrenheit: bool = False) -> dict[str, object]:
    """Return current weather from Open-Meteo (free, no API key): geocode the name, then fetch conditions."""
    import aiohttp  # imported here so the unit tests do not need a network stack

    unit = "fahrenheit" if fahrenheit else "celsius"
    timeout = aiohttp.ClientTimeout(total=8)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        geo = await _get_json(session, GEOCODE_URL, {"name": location, "count": 1, "language": "en"})
        results = geo.get("results") or []
        if not results:
            raise ValueError(f"Could not find a place called '{location}'.")
        place = results[0]
        data = await _get_json(
            session,
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,weather_code",
                "temperature_unit": unit,
                "wind_speed_unit": "mph" if fahrenheit else "kmh",
            },
        )
    cur = data["current"]
    deg = "°F" if fahrenheit else "°C"
    name = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    return {
        "location": name,
        "conditions": describe_weather(cur["weather_code"]),
        "temperature": f"{cur['temperature_2m']}{deg}",
        "feels_like": f"{cur['apparent_temperature']}{deg}",
        "humidity": f"{cur['relative_humidity_2m']}%",
        "wind": f"{cur['wind_speed_10m']} {'mph' if fahrenheit else 'km/h'}",
    }
