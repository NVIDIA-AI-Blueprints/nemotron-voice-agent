# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

from examples.generic import tool_handlers


def _run(handler, arguments: dict) -> dict:
    results: list[dict] = []

    async def result_callback(result):
        results.append(result)

    asyncio.run(handler(SimpleNamespace(arguments=arguments, result_callback=result_callback)))
    assert len(results) == 1
    return results[0]


@pytest.fixture
def offline():
    with patch.object(httpx.AsyncClient, "get", side_effect=httpx.ConnectError("offline")):
        yield


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"from_currency": "USD", "to_currency": "EUR"}, "amount is required"),
        ({"amount": "lots", "from_currency": "USD", "to_currency": "EUR"}, "amount must be numeric"),
        ({"amount": 100, "to_currency": "EUR"}, "from_currency and to_currency are required"),
        ({"amount": 100, "from_currency": "S", "to_currency": "euros"}, "Unsupported currency code: S, EUROS"),
    ],
)
def test_convert_currency_rejects_missing_or_unknown_arguments(offline, arguments, error):
    assert _run(tool_handlers.handle_convert_currency, arguments) == {"error": error}


def test_convert_currency_static_fallback(offline):
    result = _run(tool_handlers.handle_convert_currency, {"amount": 100, "from_currency": "usd", "to_currency": "EUR"})
    assert result["converted_amount"] == 92.0
    assert result["source"] == "static_fallback"
    assert "mock" in result["note"]


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"weight_kg": 70}, "weight_kg and height_m are required"),
        ({"weight_kg": 70, "height_m": 0}, "height_m must be greater than zero"),
        ({"weight_kg": 0, "height_m": 1.75}, "weight_kg must be greater than zero"),
        ({"weight_kg": "heavy", "height_m": 1.75}, "weight_kg and height_m must be numeric"),
    ],
)
def test_calculate_bmi_rejects_missing_or_invalid_arguments(arguments, error):
    assert _run(tool_handlers.handle_calculate_bmi, arguments) == {"error": error}


def test_calculate_bmi():
    result = _run(tool_handlers.handle_calculate_bmi, {"weight_kg": 70, "height_m": 1.75})
    assert result["bmi"] == 22.86
    assert result["category"] == "Normal weight"
