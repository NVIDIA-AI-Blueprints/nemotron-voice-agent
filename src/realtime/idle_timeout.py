# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Connection-local OpenAI Realtime server-VAD idle timeout coordination."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from fractions import Fraction
from typing import Any

from loguru import logger

from realtime.controller import RealtimeSessionController

IdleTurnHook = Callable[[int, int, int, int, str], Awaitable[None]]
IdleErrorHook = Callable[[Exception], Awaitable[None]]
SleepHook = Callable[[float], Awaitable[None]]


class RealtimeServerVADIdleTimeout:
    """Arm the native idle turn only after a played assistant audio response."""

    def __init__(
        self,
        *,
        controller: RealtimeSessionController,
        trigger_idle_turn: IdleTurnHook,
        report_error: IdleErrorHook,
        sleep: SleepHook = asyncio.sleep,
    ) -> None:
        """Bind timer state to one canonical Realtime connection."""
        self._controller = controller
        self._trigger_idle_turn = trigger_idle_turn
        self._report_error = report_error
        self._sleep = sleep
        self._input_audio_ms = Fraction(0)
        self._generation = 0
        self._generation_changed = asyncio.Event()
        self._timer: asyncio.Task[None] | None = None
        self._preceding_assistant_item_id: str | None = None
        self._closed = False

    @property
    def input_audio_position_ms(self) -> int:
        """Return the rounded offset of valid audio written by the client."""
        return round(self._input_audio_ms)

    @property
    def armed(self) -> bool:
        """Return whether a post-response timeout is currently pending."""
        return self._timer is not None and not self._timer.done()

    def record_input_audio(self, *, sample_count: int, sample_rate: int) -> None:
        """Advance the public input-buffer clock for one validated audio append."""
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
            raise ValueError("sample_count must be a non-negative integer")
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")
        self._input_audio_ms += Fraction(sample_count * 1000, sample_rate)

    def observe_published_events(self, events: Sequence[dict[str, Any]]) -> None:
        """Update timer ownership only after the corresponding events reach the wire."""
        if self._closed:
            return
        terminal_response: dict[str, Any] | None = None
        for event in events:
            event_type = event.get("type")
            if event_type in {"input_audio_buffer.speech_started", "response.created"}:
                self.cancel()
            elif event_type in {"conversation.item.deleted", "conversation.item.truncated"}:
                if event.get("item_id") == self._preceding_assistant_item_id:
                    self.cancel()
            elif event_type == "conversation.item.added":
                item = event.get("item")
                if isinstance(item, dict) and item.get("role") == "user":
                    self.cancel()
            elif event_type == "response.done":
                response = event.get("response")
                if isinstance(response, dict):
                    terminal_response = response
        if terminal_response is not None:
            self._arm_after_response(terminal_response)

    def cancel(self) -> None:
        """Cancel a pending timeout after real user or response activity."""
        self._advance_generation()
        self._preceding_assistant_item_id = None
        timer = self._timer
        self._timer = None
        if timer is not None and not timer.done():
            try:
                current = asyncio.current_task()
            except RuntimeError:
                current = None
            if timer is not current:
                timer.cancel()

    def _advance_generation(self) -> None:
        changed = self._generation_changed
        self._generation += 1
        self._generation_changed = asyncio.Event()
        changed.set()

    async def wait_until_stale(self, generation: int) -> None:
        """Wait until another activity retires one captured timer generation."""
        changed = self._generation_changed
        if self._closed or generation != self._generation:
            return
        await changed.wait()

    def close(self) -> None:
        """Permanently release the timer when its WebSocket closes."""
        if self._closed:
            return
        self._closed = True
        self.cancel()

    def _idle_timeout_ms(self) -> int | None:
        config = self._controller.turn_detection_config
        if not isinstance(config, dict) or config.get("type") != "server_vad":
            return None
        value = config.get("idle_timeout_ms")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    def can_trigger(
        self,
        *,
        generation: int,
        timeout_ms: int,
        preceding_assistant_item_id: str,
    ) -> bool:
        """Return whether a timer still owns the right to create its idle turn.

        The transport calls this while holding both its connection-state and
        wire-emission locks, immediately before mutating conversation state.
        That makes a previously published speech event win over a stale timer.
        """
        return (
            not self._closed
            and generation == self._generation
            and not self._controller.response_in_progress
            and self._idle_timeout_ms() == timeout_ms
            and preceding_assistant_item_id == self._preceding_assistant_item_id
        )

    def _arm_after_response(self, response: dict[str, Any]) -> None:
        self.cancel()
        timeout_ms = self._idle_timeout_ms()
        if timeout_ms is None or response.get("status") not in {"completed", "incomplete"}:
            return
        if self._controller.response_audio_duration_ms(response) is None:
            return
        assistant_item_id = next(
            (
                item.get("id")
                for item in response.get("output", [])
                if isinstance(item, dict)
                and item.get("type") == "message"
                and item.get("role") == "assistant"
                and isinstance(item.get("id"), str)
                and item.get("id")
            ),
            None,
        )
        if assistant_item_id is None:
            return
        self._preceding_assistant_item_id = assistant_item_id
        self._advance_generation()
        generation = self._generation
        # FastAPIWebsocketOutputTransport paces every audio chunk and the
        # lifecycle publishes response.done only after TTSStoppedFrame crosses
        # that drained output boundary. Playback has therefore already ended;
        # adding the response's audio duration here would count it twice.
        audio_start_ms = self.input_audio_position_ms
        self._timer = asyncio.create_task(
            self._wait_and_trigger(
                generation=generation,
                timeout_ms=timeout_ms,
                audio_start_ms=audio_start_ms,
                preceding_assistant_item_id=assistant_item_id,
            ),
            name=f"realtime-server-vad-idle-{self._controller.id}",
        )

    async def _wait_and_trigger(
        self,
        *,
        generation: int,
        timeout_ms: int,
        audio_start_ms: int,
        preceding_assistant_item_id: str,
    ) -> None:
        timer = asyncio.current_task()
        try:
            await self._sleep(timeout_ms / 1000)
            if self._closed or generation != self._generation:
                return
            if self._controller.response_in_progress or self._idle_timeout_ms() != timeout_ms:
                return
            await self._trigger_idle_turn(
                audio_start_ms,
                self.input_audio_position_ms,
                generation,
                timeout_ms,
                preceding_assistant_item_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Realtime server-VAD idle timeout failed")
            await self._report_error(exc)
        finally:
            if self._timer is timer:
                self._timer = None
