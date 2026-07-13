# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Conservative gating of webcam gestures into proactive behaviors.

The webcam worker scores a deliberate gesture per frame (greet / stop /
continue / thumbs-down). This controller decides — conservatively — when a gesture should
actually drive an action, so the assistant feels responsive without being
trigger-happy:

- edge-detection: act only on a FRESH gesture, never every frame it is held;
- per-intent confidence gates: ignore low-confidence guesses;
- a cooldown: never fire twice in quick succession;
- context gates: greet only when idle, barge-in only while the assistant is
  speaking, and treat a thumbs-up as "resume" only if a barge-in happened
  recently (otherwise as positive feedback).

The gesture *classification* is the model's (the webcam VLM); the gating here is
objective orchestration; and the *wording* of any spoken reply is the Speaker's.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.omni_assistant_subagents.subagents.gestures import (
    GESTURE_CONTINUE,
    GESTURE_DOWN,
    GESTURE_GREET,
    GESTURE_NONE,
    GESTURE_STOP,
    normalize_visual_control,
)

_GESTURE_CONFIDENCE = {
    GESTURE_GREET: 0.85,
    GESTURE_STOP: 0.75,
    GESTURE_CONTINUE: 0.7,
    GESTURE_DOWN: 0.7,
}
_COOLDOWN_SECONDS = 5.0
_RESUME_WINDOW_SECONDS = 30.0


class ProactiveGestureController:
    """Own the edge-detection, gating, and dispatch of webcam-driven proactive actions."""

    def __init__(
        self,
        *,
        queue_frame: Callable[[Any], Awaitable[None]],
        greet: Callable[[], Awaitable[None]],
        barge_in: Callable[[], Awaitable[None]],
        resume_or_compliment: Callable[[bool], Awaitable[None]],
        acknowledge_feedback: Callable[[], Awaitable[None]],
        is_assistant_speaking: Callable[[], bool],
        is_user_speaking: Callable[[], bool],
    ) -> None:
        """Initialize gating thresholds and the action/state callbacks.

        ``resume_or_compliment(resume)`` is called for a thumbs-up: ``resume`` is
        ``True`` when a barge-in happened within the resume window (continue the
        prior thought), ``False`` otherwise (acknowledge positive feedback).
        ``acknowledge_feedback`` is called for a thumbs-down (the user is not impressed):
        the Speaker briefly takes it as feedback and carries on — it does NOT re-answer.
        """
        self._queue_frame = queue_frame
        self._greet = greet
        self._barge_in = barge_in
        self._resume_or_compliment = resume_or_compliment
        self._acknowledge_feedback = acknowledge_feedback
        self._is_assistant_speaking = is_assistant_speaking
        self._is_user_speaking = is_user_speaking
        self._confidence = dict(_GESTURE_CONFIDENCE)
        self._cooldown_secs = _COOLDOWN_SECONDS
        self._resume_window_secs = _RESUME_WINDOW_SECONDS
        self._last_intent = GESTURE_NONE
        self._last_action_at = float("-inf")
        self._last_interrupted_at = float("-inf")

    async def handle(self, visual_control: Any, *, frame: dict[str, Any]) -> None:
        """Gate one scored gesture from the webcam stream into a proactive action."""
        control = normalize_visual_control(visual_control)
        intent = control["intent"]
        confidence = control["confidence"]

        if intent == GESTURE_NONE:
            self._last_intent = GESTURE_NONE
            return
        if intent == self._last_intent:
            return
        now = time.monotonic()
        if intent != GESTURE_STOP and now - self._last_action_at < self._cooldown_secs:
            logger.debug(f"Gesture {intent!r} ignored: within cooldown")
            return
        if confidence < self._confidence[intent]:
            logger.debug(f"Gesture {intent!r} ignored: confidence {confidence:.2f} below threshold")
            return

        acted = False
        if intent == GESTURE_GREET:
            acted = await self._on_greet(control, frame, now)
        elif intent == GESTURE_STOP:
            acted = await self._on_stop(control, frame, now)
        elif intent == GESTURE_CONTINUE:
            acted = await self._on_continue(control, frame, now)
        elif intent == GESTURE_DOWN:
            acted = await self._on_down(control, frame, now)

        if acted:
            self._last_intent = intent

    async def _on_greet(self, control: dict[str, Any], frame: dict[str, Any], now: float) -> bool:
        """Greet back — but only when idle, never over the user's or our own turn.

        Returns True when we actually greeted (so the caller consumes the gesture),
        False when it was gated out (so a later frame can still fire it).
        """
        if self._is_assistant_speaking() or self._is_user_speaking():
            logger.info("Proactive greet skipped: a turn is already active")
            return False
        await self._greet()
        self._last_action_at = now
        logger.info(f"Proactive greet: confidence={control['confidence']:.2f}")
        await self._emit_action("greet", "greeted", control, frame)
        return True

    async def _on_stop(self, control: dict[str, Any], frame: dict[str, Any], now: float) -> bool:
        """Barge in on a clear stop sign — only meaningful while the assistant is speaking."""
        if not self._is_assistant_speaking():
            logger.debug("Stop gesture ignored: the assistant is not speaking")
            return False
        await self._barge_in()
        self._last_action_at = now
        self._last_interrupted_at = now
        logger.info(f"Proactive barge-in: confidence={control['confidence']:.2f}")
        await self._emit_action("stop", "barged_in", control, frame)
        return True

    async def _on_continue(self, control: dict[str, Any], frame: dict[str, Any], now: float) -> bool:
        """Thumbs-up: resume if we were just barged-in, otherwise take it as a compliment."""
        if self._is_assistant_speaking() or self._is_user_speaking():
            logger.info("Proactive continue skipped: a turn is already active")
            return False
        resume = (now - self._last_interrupted_at) <= self._resume_window_secs
        await self._resume_or_compliment(resume)
        self._last_action_at = now
        if resume:
            self._last_interrupted_at = float("-inf")
        logger.info(f"Proactive continue: resume={resume}, confidence={control['confidence']:.2f}")
        await self._emit_action("continue", "resumed" if resume else "acknowledged", control, frame)
        return True

    async def _on_down(self, control: dict[str, Any], frame: dict[str, Any], now: float) -> bool:
        """Thumbs-down: the user is not impressed — take it as feedback (no retry). Idle only."""
        if self._is_assistant_speaking() or self._is_user_speaking():
            logger.info("Proactive feedback ack skipped: a turn is already active")
            return False
        await self._acknowledge_feedback()
        self._last_action_at = now
        logger.info(f"Proactive feedback ack: confidence={control['confidence']:.2f}")
        await self._emit_action("down", "noted", control, frame)
        return True

    async def _emit_action(self, action: str, state: str, control: dict[str, Any], frame: dict[str, Any]) -> None:
        """Surface the proactive action to the client UI (reuses webcam-control-update)."""
        await self._queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "webcam-control-update",
                    "action": action,
                    "state": state,
                    "visual_control": control,
                    "frame": frame,
                }
            )
        )
