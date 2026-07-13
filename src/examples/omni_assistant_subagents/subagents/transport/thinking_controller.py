# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deliberate-thinking dispatch and follow-up emission.

When the Speaker stalls ("let me think") because it is stuck or the request is
hard, this controller dispatches the reasoning-ON Thinker over the conversation
so far (after the turn finishes, mirroring the media-analysis flow) and speaks the
Thinker's answer as a follow-up turn. The bridging line the user hears first — the
model's own stall, or a filler the Speaker substitutes for a suppressed repeat —
is spoken upstream by the Speaker service; ``reason`` here only tells the Thinker
why it was called so it can recover appropriately.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.bus.messages import BusJobResponseMessage
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from examples.omni_assistant_subagents.subagents.thinker import THINKING_TASK_NAME, ThinkerWorker

_MAX_THINKER_TURNS = 40
_MAX_THINKER_CONVERSATION_CHARS = 8000


class ThinkingController:
    """Own queued deliberate-thinking work and its spoken follow-up output."""

    def __init__(
        self,
        *,
        context: LLMContext,
        request_job: Callable[..., Awaitable[str]],
        queue_frame: Callable[[Any], Awaitable[None]],
        followup_delay_secs: float,
    ) -> None:
        """Initialize deliberate-thinking dispatch state for one session."""
        self._context = context
        self._request_job = request_job
        self._queue_frame = queue_frame
        self._followup_delay_secs = followup_delay_secs
        self._pending: dict[str, str] | None = None
        self._active = False
        self._generation = 0
        self._active_task_id = ""

    def queue(self, transcript: str, *, effort: str, reason: str = "") -> None:
        """Queue a thinking pass to run after the Speaker's turn completes.

        ``reason`` is empty when the model spoke its own stall; a non-empty reason
        (e.g. ``repetition``) is passed to the Thinker as recovery context. The
        bridging line the user hears is spoken by the Speaker service, not here.
        """
        cleaned = transcript.strip()
        if not cleaned:
            return
        self._pending = {"transcript": cleaned, "effort": effort, "reason": reason}
        logger.info(
            f"Queued deliberate thinking: effort={effort}, reason={reason or 'needs_thinking'}, "
            f"transcript_chars={len(cleaned)}"
        )

    def clear_pending(self) -> None:
        """Drop any queued thinking (e.g. when the stall was interrupted)."""
        self._pending = None
        self._generation += 1
        self._active = False
        self._active_task_id = ""

    async def start_pending(self) -> None:
        """Dispatch the queued thinking pass after the stall turn has completed."""
        pending = self._pending
        self._pending = None
        if not pending or self._active:
            return
        generation = self._generation
        self._active = True
        conversation = self._render_conversation()
        try:
            task_id = await self._request_job(
                ThinkerWorker.AGENT_NAME,
                name=THINKING_TASK_NAME,
                payload={"conversation": conversation, **pending},
                timeout=120.0,
            )
        except Exception as exc:
            self._active = False
            logger.warning(f"Failed to dispatch deliberate thinking: {exc}")
            return
        if generation != self._generation:
            return
        self._active_task_id = task_id
        logger.info(
            f"Deliberate thinking dispatched: task_id={task_id}, effort={pending['effort']}, "
            f"reason={pending.get('reason') or 'needs_thinking'}"
        )
        await self._emit_update(task_id=task_id, status="running", detail="Thinking it through...")

    async def handle_job_response(self, message: BusJobResponseMessage) -> bool:
        """Speak the Thinker's answer as a follow-up turn. Return whether handled."""
        if message.job_id != self._active_task_id:
            return True
        generation = self._generation
        self._active = False
        self._active_task_id = ""
        response = message.response or {}
        answer = str(response.get("response") or "").strip()
        reasoning = str(response.get("reasoning") or "").strip()
        agent = str(getattr(message, "source", "") or ThinkerWorker.AGENT_NAME)
        if not answer:
            await self._emit_update(task_id=message.job_id, status="error", detail="Thinking failed.", agent=agent)
            return True
        await asyncio.sleep(self._followup_delay_secs)
        if generation != self._generation:
            return True
        await self._queue_frame(LLMFullResponseStartFrame())
        await self._queue_frame(LLMTextFrame(text=answer))
        await self._queue_frame(LLMFullResponseEndFrame())
        await self._emit_update(task_id=message.job_id, status="done", detail=answer, agent=agent, reasoning=reasoning)
        return True

    def _render_conversation(self) -> str:
        """Render the recent conversation (plus pinned subagent context) as plain text.

        The Thinker gets the recent turns (bounded to keep the reasoning prompt small)
        plus the pinned state, so it can reason about source provenance — e.g. that an
        uploaded file was analyzed earlier and is stale versus the live webcam now.
        """
        messages = [m for m in self._context.get_messages() if isinstance(m, dict)]
        pinned: list[str] = []
        turns: list[str] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = message.get("content")
            if not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            if role == "system":
                if message is not messages[0]:
                    pinned.append(text)
            elif role in ("user", "assistant"):
                turns.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
        turns = turns[-_MAX_THINKER_TURNS:]
        conversation = "\n".join(turns)
        if len(conversation) > _MAX_THINKER_CONVERSATION_CHARS:
            conversation = conversation[-_MAX_THINKER_CONVERSATION_CHARS:]
        parts = []
        if pinned:
            parts.append("Known context:\n" + "\n".join(pinned))
        parts.append(conversation)
        return "\n\n".join(parts).strip()

    async def _emit_update(
        self, *, task_id: str, status: str, detail: str, agent: str = ThinkerWorker.AGENT_NAME, reasoning: str = ""
    ) -> None:
        """Emit thinking progress as a client-visible bus update.

        The worker streams reasoning/answer deltas during inference; this finalizes the
        card with the authoritative full reasoning and answer when the pass completes.
        """
        await self._queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "agent-task-update",
                    "task_id": task_id,
                    "agent": agent,
                    "status": status,
                    "stage": "thinking" if status == "running" else "complete",
                    "detail": detail,
                    "reasoning": reasoning,
                    "response": detail if status == "done" else "",
                }
            )
        )
