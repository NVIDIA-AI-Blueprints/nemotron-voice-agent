# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Nemotron Omni media analyzer worker subagent."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pipecat.bus.messages import BusFrameMessage, BusJobRequestMessage
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.workers.base_worker import BaseWorker

from attachment_store import Attachment, get_attachment
from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniService,
    NvidiaOmniSettings,
    media_message_part,
    text_message_part,
)
from examples.shared.json_parsing import extract_json_object
from utils import parse_env_float, parse_env_int

MEDIA_ANALYSIS_TASK_NAME = "analyze_media"

MEDIA_ANALYSIS_RUNNING_PREFIX = "An uploaded media analysis task is running asynchronously."
SPEAKER_STATE_PREFIXES: tuple[str, ...] = (MEDIA_ANALYSIS_RUNNING_PREFIX,)

_SYSTEM_PROMPT = (
    "You are a careful uploaded-media analysis worker. Reply with ONE JSON object and nothing else. "
    'For a fresh analysis reply {"tts": "<two or three short spoken sentences answering the user, plain '
    'prose for TTS>", "analysis": "<a thorough, detailed, plain-text description capturing every element, '
    'label, text, number, and relationship you can see>"}. When the user message includes an EXISTING '
    'ANALYSIS and asks what to add, reply {"tts": "<short spoken answer>", "append_patch": "<only the new '
    'details to add that are not already in the existing analysis; empty string if nothing new>"}. Only '
    "describe what is clearly supported by the media; if uncertain, say so. The tts field is plain spoken "
    "prose with no markdown, bullets, asterisks, parentheses, slashes, or code formatting."
)


class MediaAnalyzerWorker(BaseWorker):
    """Worker that analyzes uploaded media with Nemotron Omni and reports over the bus."""

    AGENT_NAME = "omni_media_analyzer"

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        extra_params: dict[str, Any] | None = None,
        system_prompt: str = "",
        reasoning: str = "on",
    ) -> None:
        """Configure the OpenAI-compatible client and analyzer defaults.

        ``reasoning`` (from ``subagents.yaml``) selects the model thinking mode:
        ``on`` enables reasoning tokens; anything else runs reasoning-off.
        """
        super().__init__(name or self.AGENT_NAME, active=True)
        self._base_url = base_url
        self._model_id = model_id
        self._system_prompt = system_prompt.strip() or _SYSTEM_PROMPT
        self._max_tokens = parse_env_int("MEDIA_ANALYZER_MAX_TOKENS", 8192, min_value=256)
        self._temperature = parse_env_float("MEDIA_ANALYZER_TEMPERATURE", 0.2, min_value=0.0)
        omni_extra = dict(extra_params or {})
        extra_body = dict(omni_extra.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            **dict(extra_body.get("chat_template_kwargs") or {}),
            "enable_thinking": reasoning == "on",
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
                input_modalities=("image", "audio", "video", "text"),
                stream=True,
            ),
        )

    @job(name=MEDIA_ANALYSIS_TASK_NAME)
    async def analyze_media(self, message: BusJobRequestMessage) -> None:
        """Analyze one uploaded attachment (fresh full analysis or an extend patch)."""
        payload = message.payload or {}
        requester = message.source
        attachment = payload.get("attachment") if isinstance(payload.get("attachment"), dict) else {}
        transcript = str(payload.get("transcript") or "").strip()
        prompt = str(payload.get("analysis_prompt") or "").strip()
        prior_analysis = str(payload.get("prior_analysis") or "").strip()
        query = prompt or transcript
        session_id = str(payload.get("session_id") or "").strip()
        attachment_id = str(attachment.get("id") or "").strip()
        tts = analysis = append_patch = reasoning = ""

        await self._emit_update(
            target=requester,
            task_id=message.job_id,
            status="running",
            stage="started",
            detail=f"Analyzing {attachment.get('kind', 'media')} attachment...",
            attachment=attachment,
            query=query,
        )

        stored_attachment = get_attachment(session_id, attachment_id)
        if stored_attachment is None:
            tts = "I could not access the uploaded media for analysis."
        else:
            try:
                text, reasoning = await self._analyze_attachment(
                    stored_attachment,
                    query,
                    prior_analysis=prior_analysis,
                    requester=requester,
                    task_id=message.job_id,
                    attachment_metadata=attachment,
                )
                tts, analysis, append_patch = _parse_analyzer_result(text, is_patch=bool(prior_analysis))
            except Exception as exc:
                logger.exception(f"Media analyzer Omni request failed: {exc}")
                tts = "I could not analyze the uploaded media because the analyzer request failed."

        await self.send_job_response(
            message.job_id,
            {
                "tts": tts,
                "analysis": analysis,
                "append_patch": append_patch,
                "is_patch": bool(prior_analysis),
                "reasoning": reasoning,
                "query": query,
                "transcript": transcript,
                "attachment": attachment,
            },
        )

    async def _analyze_attachment(
        self,
        attachment: Attachment,
        prompt: str,
        *,
        prior_analysis: str,
        requester: str,
        task_id: str,
        attachment_metadata: dict,
    ) -> tuple[str, str]:
        """Call the multimodal Omni endpoint for one attachment; return (raw_text, reasoning)."""
        context = LLMContext(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        media_message_part(
                            attachment.data, modality=attachment.kind, mime_type=attachment.content_type
                        ),
                        text_message_part(_build_user_prompt(prompt, prior_analysis)),
                    ],
                },
            ]
        )
        logger.info(
            "Media analyzer Omni request: "
            f"base_url={self._base_url}, model={self._model_id}, kind={attachment.kind}, "
            f"bytes={len(attachment.data)}, mode={'patch' if prior_analysis else 'full'}"
        )
        reasoning = ""

        async def on_reasoning_delta(reasoning_delta: str) -> None:
            nonlocal reasoning
            reasoning += reasoning_delta
            await self._emit_update(
                target=requester,
                task_id=task_id,
                status="running",
                stage="reasoning",
                detail="Reasoning about the uploaded media...",
                attachment=attachment_metadata,
                reasoning_delta=reasoning_delta,
            )

        result = await self._omni.run_multimodal_inference(
            context,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=True,
            on_reasoning_delta=on_reasoning_delta,
        )
        text = result.text.strip()
        reasoning = (result.reasoning or reasoning).strip()
        logger.info(f"Media analyzer Omni answer: answer_chars={len(text)}")
        return text, reasoning

    async def _emit_update(
        self,
        *,
        target: str,
        task_id: str,
        status: str,
        stage: str,
        detail: str,
        attachment: dict,
        query: str = "",
        reasoning_delta: str = "",
        response_delta: str = "",
        reasoning: str = "",
        response: str = "",
    ) -> None:
        """Emit semantic worker progress as a client-visible bus update."""
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
                        "attachment": attachment,
                        "query": query,
                        "reasoning_delta": reasoning_delta,
                        "response_delta": response_delta,
                        "reasoning": reasoning,
                        "response": response,
                    }
                ),
            )
        )


def _build_user_prompt(question: str, prior_analysis: str) -> str:
    """Build the analyzer user prompt — fresh full analysis, or an extend-with-patch request."""
    if prior_analysis:
        return (
            f"Existing analysis of this media:\n{prior_analysis}\n\n"
            f"The user now asks: {question}\n"
            "Answer their question in 'tts'. In 'append_patch' put ONLY new details to add that are not "
            "already in the existing analysis above; use an empty string if there is nothing new to add."
        )
    return (
        f"{question}\n\n"
        "Give 'tts' (a short spoken answer for the user) and 'analysis' (a thorough, detailed description "
        "capturing everything visible in the media)."
    )


def _parse_analyzer_result(text: str, *, is_patch: bool) -> tuple[str, str, str]:
    """Parse the analyzer's JSON answer into ``(tts, analysis, append_patch)``, defensively.

    Falls back to the raw text so a malformed/empty answer never dead-ends as an
    unusable description.
    """
    data = extract_json_object(text)
    tts = str(data.get("tts") or "").strip()
    analysis = str(data.get("analysis") or "").strip()
    append_patch = str(data.get("append_patch") or "").strip()
    raw = text.strip()
    if not tts:
        tts = raw or "I could not produce a description of the uploaded media."
    if is_patch:
        return tts, "", append_patch
    return tts, (analysis or raw or tts), ""
