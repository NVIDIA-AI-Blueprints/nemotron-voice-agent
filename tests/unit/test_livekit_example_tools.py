# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio

import pytest

from examples.generic_assistant_livekit import tools


def test_current_datetime_in_timezone():
    out = tools.current_datetime("Asia/Tokyo")
    assert out["timezone"] == "Asia/Tokyo"
    assert out["iso"].endswith("+09:00")


def test_current_datetime_default_is_local():
    assert tools.current_datetime()["iso"]


def test_current_datetime_rejects_bad_timezone():
    with pytest.raises(ValueError, match="Unknown timezone"):
        tools.current_datetime("Mars/Olympus")


def test_weather_code_description():
    assert tools.describe_weather(0) == "clear sky"
    assert tools.describe_weather(63) == "rain"
    assert "1234" in tools.describe_weather(1234)


def test_weather_formats_open_meteo_response(monkeypatch):
    pytest.importorskip("aiohttp")
    calls = []

    async def fake_get_json(session, url, params):
        calls.append((url, params))
        if url == tools.GEOCODE_URL:
            return {
                "results": [
                    {"name": "Paris", "admin1": "Île-de-France", "country": "France", "latitude": 1, "longitude": 2}
                ]
            }
        return {
            "current": {
                "temperature_2m": 21.5,
                "apparent_temperature": 20.0,
                "relative_humidity_2m": 55,
                "wind_speed_10m": 12.3,
                "weather_code": 2,
            }
        }

    monkeypatch.setattr(tools, "_get_json", fake_get_json)
    out = asyncio.run(tools.current_weather("Paris", fahrenheit=False))
    assert out["location"] == "Paris, Île-de-France, France"
    assert out["conditions"] == "partly cloudy"
    assert out["temperature"] == "21.5°C"
    assert out["wind"] == "12.3 km/h"
    assert calls[1][1]["temperature_unit"] == "celsius"


def test_weather_unknown_place(monkeypatch):
    pytest.importorskip("aiohttp")

    async def fake_get_json(session, url, params):
        return {}

    monkeypatch.setattr(tools, "_get_json", fake_get_json)
    with pytest.raises(ValueError, match="Could not find"):
        asyncio.run(tools.current_weather("Nowhereville"))
