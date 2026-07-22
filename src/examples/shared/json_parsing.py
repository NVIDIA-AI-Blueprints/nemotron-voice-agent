# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared, defensive JSON extraction for model responses."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object from model text, tolerating code fences.

    Returns an empty dict when no parseable object is present, so callers can fall
    back to treating the raw text as plain output.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return {}
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    while start != -1:
        try:
            payload, _ = decoder.raw_decode(cleaned[start:])
        except (ValueError, TypeError):
            start = cleaned.find("{", start + 1)
            continue
        if isinstance(payload, dict):
            return payload
        start = cleaned.find("{", start + 1)
    return {}
