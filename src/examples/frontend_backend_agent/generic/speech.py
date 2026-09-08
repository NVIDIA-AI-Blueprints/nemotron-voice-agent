# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Grounded success speech for generic-domain ToolSpecs."""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_agent.generic.result_formatters import _speech_text


def weather(arguments: dict[str, Any], data: dict[str, Any]) -> str:
    """Speak current conditions from the trusted service payload."""
    del arguments
    city = _speech_text(data.get("city"), max_length=120) or "that location"
    condition = _speech_text(data.get("condition"), max_length=100) or "reported conditions"
    unit = _speech_text(data.get("temperature_unit"), max_length=2) or "C"
    text = f"In {city}, it's {data.get('temperature')} degrees {unit} with {condition.lower()}"
    if data.get("feels_like") is not None:
        text += f", and it feels like {data.get('feels_like')} degrees {unit}"
    return f"{text}."


def stock(arguments: dict[str, Any], data: dict[str, Any]) -> str:
    """Speak an exact current quote from the trusted service payload."""
    del arguments
    company = _speech_text(data.get("company"), max_length=120) or "That company"
    symbol = _speech_text(data.get("symbol"), max_length=16)
    ticker_label = f", ticker {symbol}," if symbol else ""
    currency = _speech_text(data.get("currency"), max_length=8) or "USD"
    return f"{company}{ticker_label} is trading at {data.get('price')} {currency}."


def search(arguments: dict[str, Any], data: dict[str, Any]) -> str:
    """Speak only the bounded answer returned by the search service."""
    del arguments
    return _speech_text(data.get("answer")) or "I couldn't find a verified answer for that."


def bmi(arguments: dict[str, Any], data: dict[str, Any]) -> str:
    """Speak BMI with the required screening disclaimer."""
    del arguments
    return (
        f"Your BMI is {data.get('bmi')}, which falls in the {data.get('category')} category. "
        "BMI is a screening measure, not a diagnosis."
    )


def random_number(arguments: dict[str, Any], data: dict[str, Any]) -> str:
    """Speak the generated value and its inclusive range."""
    del arguments
    return f"Your random number between {data.get('min')} and {data.get('max')} is {data.get('result')}."
