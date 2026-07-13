# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Deliberate-reasoning Thinker worker subagent.

The Speaker runs reasoning-OFF for low latency. When it is stuck or the question
is genuinely hard, it stalls with a brief "let me think" line and hands the turn
to this Thinker, which runs the same Omni model **reasoning-ON** (with a budget
scaled to the situation's complexity) over the conversation so far and returns the
best spoken answer. Mirrors the media-analyzer worker's async job pattern: it
streams its reasoning and answer tokens to the client as ``agent-task-update``
bus frames (so the UI shows them live, just like the analyzer), and the transport
speaks the final result as a follow-up turn.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.bus.messages import BusFrameMessage, BusJobRequestMessage
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.workers.base_worker import BaseWorker

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniService,
    NvidiaOmniSettings,
    text_message_part,
)
from utils import parse_env_float, parse_env_int

THINKING_TASK_NAME = "think"

_REASONING_BUDGETS = {"low": 512, "medium": 1024, "high": 2048}
_DEFAULT_EFFORT = "medium"

_REASON_NOTES = {
    "repetition": (
        "IMPORTANT: the fast assistant has been repeating almost the same answer and the user is frustrated by "
        "the repetition. Work out what the user actually wants now — they may be doubting the answer, want it "
        "explained or justified, or want you to move on — and give a genuinely different, fresh reply that breaks "
        "the loop. Do not restate the previous answer in the same words."
    ),
}

_SYSTEM_PROMPT = (
    "You are the deliberate reasoning agent behind a friendly NVIDIA voice assistant. Reason privately, then "
    "reply with ONLY the complete final spoken answer in plain prose addressed to 'you', with no formatting."
)


class ThinkerWorker(BaseWorker):
    """Worker that re-answers a hard/stuck turn with reasoning enabled."""

    AGENT_NAME = "omni_thinker"

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        extra_params: dict[str, Any] | None = None,
        system_prompt: str = "",
    ) -> None:
        """Configure the reasoning-ON Omni client for deliberate thinking."""
        super().__init__(name or self.AGENT_NAME, active=True)
        self._base_url = base_url
        self._model_id = model_id
        self._system_prompt = system_prompt.strip() or _SYSTEM_PROMPT
        self._temperature = parse_env_float("THINKER_TEMPERATURE", 0.6, min_value=0.0)
        self._max_tokens = min(parse_env_int("THINKER_MAX_TOKENS", 16384, min_value=1024), 32768)
        omni_extra = dict(extra_params or {})
        extra_body = dict(omni_extra.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            **dict(extra_body.get("chat_template_kwargs") or {}),
            "enable_thinking": True,
        }
        omni_extra["extra_body"] = extra_body
        self._omni = NvidiaOmniService(
            api_key=api_key,
            base_url=base_url,
            extra=omni_extra,
            settings=NvidiaOmniSettings(
                model=model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                input_modalities=("text",),
                stream=True,
            ),
        )

    @job(name=THINKING_TASK_NAME)
    async def think(self, message: BusJobRequestMessage) -> None:
        """Reason over the conversation; stream reasoning/answer tokens, return the answer."""
        payload = message.payload or {}
        requester = message.source
        task_id = message.job_id
        transcript = str(payload.get("transcript") or "").strip()
        conversation = str(payload.get("conversation") or "").strip()
        reason = str(payload.get("reason") or "").strip()
        effort = str(payload.get("effort") or _DEFAULT_EFFORT).strip().lower()
        reasoning_budget = min(
            _REASONING_BUDGETS.get(effort, _REASONING_BUDGETS[_DEFAULT_EFFORT]),
            self._max_tokens - 1,
        )

        await self._emit_update(
            target=requester, task_id=task_id, status="running", stage="started", detail="Thinking it through..."
        )
        answer = ""
        reasoning = ""
        try:
            answer, reasoning = await self._think(
                conversation,
                transcript,
                reason,
                reasoning_budget,
                requester=requester,
                task_id=task_id,
            )
        except Exception as exc:
            logger.exception(f"Thinker Omni request failed: {exc}")
            answer = ""

        await self.send_job_response(
            message.job_id,
            {"response": answer, "reasoning": reasoning},
        )

    async def _think(
        self,
        conversation: str,
        transcript: str,
        reason: str,
        reasoning_budget: int,
        *,
        requester: str,
        task_id: str,
    ) -> tuple[str, str]:
        """Call the reasoning-ON Omni endpoint, streaming reasoning and answer tokens to the client."""
        recovery_note = _REASON_NOTES.get(reason, "")
        user_text = (
            f"Conversation so far:\n{conversation}\n\n"
            f"{recovery_note}\n"
            f"The user's latest request: {transcript}\n"
            "Reason carefully about what the user wants and where earlier replies fell short, then give the "
            "complete spoken answer in the length and format the user requested."
        )
        context = LLMContext(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": [text_message_part(user_text)]},
            ]
        )
        logger.info(
            f"Thinker Omni request: base_url={self._base_url}, model={self._model_id}, "
            f"max_tokens={self._max_tokens}, reasoning_budget={reasoning_budget}, transcript_chars={len(transcript)}"
        )

        async def on_reasoning_delta(reasoning_delta: str) -> None:
            await self._emit_update(
                target=requester,
                task_id=task_id,
                status="running",
                stage="reasoning",
                detail="Thinking it through...",
                reasoning_delta=reasoning_delta,
            )

        async def on_text_delta(text_delta: str) -> None:
            await self._emit_update(
                target=requester,
                task_id=task_id,
                status="running",
                stage="answering",
                response_delta=text_delta,
            )

        result = await self._omni.run_multimodal_inference(
            context,
            max_tokens=self._max_tokens,
            reasoning_budget=reasoning_budget,
            temperature=self._temperature,
            stream=True,
            on_reasoning_delta=on_reasoning_delta,
            on_text_delta=on_text_delta,
        )
        answer = result.text.strip()
        reasoning = (result.reasoning or "").strip()
        logger.info(
            f"Thinker Omni answer: answer_chars={len(answer)}, finish_reason={result.finish_reason or 'unknown'}, "
            f"reasoning_chars={len(reasoning)}"
        )
        return answer, reasoning

    async def _emit_update(
        self,
        *,
        target: str,
        task_id: str,
        status: str,
        stage: str,
        detail: str = "",
        reasoning_delta: str = "",
        response_delta: str = "",
    ) -> None:
        """Emit a client-visible task update over the bus (streamed reasoning/answer tokens)."""
        await self.bus.send(
            BusFrameMessage(
                source=self.name,
                target=target,
                direction=FrameDirection.DOWNSTREAM,
                frame=RTVIServerMessageFrame(
                    data={
                        "type": "agent-task-update",
                        "task_id": task_id,
                        "agent": self.name,
                        "status": status,
                        "stage": stage,
                        "detail": detail,
                        "reasoning_delta": reasoning_delta,
                        "response_delta": response_delta,
                    }
                ),
            )
        )
