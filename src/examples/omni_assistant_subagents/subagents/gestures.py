# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared visual-control gesture schema for the omni-assistant-subagents webcam stream.

The webcam worker scores a small, conservative set of deliberate hand gestures
per frame (greet / stop / continue) into a ``visual_control`` object; the
``ProactiveGestureController`` gates those into proactive actions. Keeping the
schema here lets the producer (webcam worker) and the consumer (gesture
controller) agree on one normalized shape without coupling to each other.
"""

from __future__ import annotations

from typing import Any

GESTURE_NONE = "none"
GESTURE_GREET = "greet"
GESTURE_STOP = "stop"
GESTURE_CONTINUE = "continue"
GESTURE_DOWN = "down"
VISUAL_CONTROL_INTENTS = frozenset({GESTURE_NONE, GESTURE_GREET, GESTURE_STOP, GESTURE_CONTINUE, GESTURE_DOWN})


def normalize_visual_control(payload: Any) -> dict[str, Any]:
    """Normalize a model-scored visual-control gesture into a stable schema.

    Whitelists ``intent`` to the supported gestures, clamps ``confidence`` to
    ``[0, 1]``, and trims ``reason``, so consumers can trust the shape.
    """
    record = payload if isinstance(payload, dict) else {}
    intent = str(record.get("intent") or GESTURE_NONE).strip().lower()
    if intent not in VISUAL_CONTROL_INTENTS:
        intent = GESTURE_NONE
    try:
        confidence = float(record.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(record.get("reason") or "").strip()
    return {
        "intent": intent,
        "confidence": min(1.0, max(0.0, confidence)),
        "reason": reason,
    }
