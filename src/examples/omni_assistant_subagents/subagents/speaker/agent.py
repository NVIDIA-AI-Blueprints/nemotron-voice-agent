# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""User-facing Speaker Omni agent."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniMultimodalService,
    NvidiaOmniSettings,
    NvidiaOmniTurnResult,
)
from utils import parse_env_float, parse_env_int


class SubagentsSpeakerOmniService(NvidiaOmniMultimodalService):
    """Subagents wrapper around :class:`NvidiaOmniMultimodalService`.

    Speaker Omni produces a strict JSON response per turn. This subclass
    parses the structured control fields (``selected_input_source``,
    ``media_analysis_action``, ``media_analysis_prompt``) and forwards them
    to the transport agent via the provided handler callback so analyzer work
    runs after the spoken acknowledgement instead of inside the response stream.
    """

    def __init__(
        self,
        *,
        audio_response_instruction: str,
        media_analysis_prompt_handler: Callable[[str, str, str, str], Awaitable[None]] | None = None,
        uploaded_attachment_available: Callable[[], bool] | None = None,
        **kwargs,
    ) -> None:
        """Configure the wrapper with the JSON response contract from ``prompts.yaml``.

        ``audio_response_instruction`` must come from
        ``agent_prompts.SpeakerAgent.audio_response_instruction`` and define
        the per-turn JSON schema Speaker Omni produces.
        """
        super().__init__(**kwargs)
        self._media_analysis_prompt_handler = media_analysis_prompt_handler
        self._uploaded_attachment_available = uploaded_attachment_available
        self._audio_response_instruction_content = audio_response_instruction.strip()
        if not self._audio_response_instruction_content:
            raise ValueError("SpeakerAgent audio_response_instruction must be provided from prompts.yaml")

    def _audio_response_instruction(self) -> str:
        return self._audio_response_instruction_content

    def _parse_turn_result(self, raw_content: str, *, parse_json: bool) -> NvidiaOmniTurnResult:
        result = super()._parse_turn_result(raw_content, parse_json=parse_json)
        response = _clean_spoken_response_artifacts(result.response)
        selected_input_source = _normalize_selected_input_source(result.payload.get("selected_input_source"))
        media_action = _normalize_media_analysis_action(result.payload.get("media_analysis_action"))
        if self._is_missing_uploaded_attachment_route(selected_input_source, media_action):
            payload = dict(result.payload)
            payload["selected_input_source"] = "none"
            payload["media_analysis_action"] = "none"
            payload["media_analysis_prompt"] = ""
            response = _missing_uploaded_attachment_response(result.transcript)
            payload["response"] = response
            return NvidiaOmniTurnResult(
                transcript=result.transcript,
                response=response,
                raw_content=result.raw_content,
                payload=payload,
            )
        if response == result.response:
            return result
        payload = dict(result.payload)
        payload["response"] = response
        return NvidiaOmniTurnResult(
            transcript=result.transcript,
            response=response,
            raw_content=result.raw_content,
            payload=payload,
        )

    async def _emit_assistant_text(self, text: str, *, response_started: bool) -> bool:
        cleaned = _clean_spoken_response_artifacts(text)
        if not cleaned:
            return response_started
        return await super()._emit_assistant_text(cleaned, response_started=response_started)

    def _structured_response_control_fields(self) -> tuple[str, ...]:
        return ("selected_input_source", "media_analysis_action")

    def _should_emit_streamed_structured_response(self, field_values: Mapping[str, str]) -> bool:
        selected_input_source = _normalize_selected_input_source(field_values.get("selected_input_source"))
        media_action = _normalize_media_analysis_action(field_values.get("media_analysis_action"))
        if self._is_missing_uploaded_attachment_route(selected_input_source, media_action):
            logger.info(
                "Speaker Omni suppressed streamed attachment acknowledgement "
                "because no uploaded attachment is available"
            )
            return False
        return True

    def _is_missing_uploaded_attachment_route(self, selected_input_source: str, media_action: str) -> bool:
        if selected_input_source == "live_webcam":
            return False
        if selected_input_source != "uploaded_attachment" and media_action not in {"new", "rerun"}:
            return False
        if self._uploaded_attachment_available is None:
            return False
        return not self._uploaded_attachment_available()

    async def _on_turn_result(self, result: NvidiaOmniTurnResult) -> None:
        transcript = result.transcript.strip()
        response = _clean_spoken_response_artifacts(result.response)
        user_text = transcript or response or result.raw_content.strip()
        if not user_text:
            return

        selected_input_source = _normalize_selected_input_source(result.payload.get("selected_input_source"))
        media_prompt = str(result.payload.get("media_analysis_prompt", "")).strip()
        media_action = _normalize_media_analysis_action(result.payload.get("media_analysis_action"))
        should_analyze_media = selected_input_source == "uploaded_attachment" and (
            bool(media_prompt) or media_action in {"new", "rerun"}
        )
        if selected_input_source != "uploaded_attachment" and (media_prompt or media_action in {"new", "rerun"}):
            logger.info(
                "Speaker Omni ignored media analysis trigger because selected_input_source="
                f"{selected_input_source!r}: action={media_action}, transcript={transcript!r}"
            )
        if should_analyze_media and self._media_analysis_prompt_handler:
            media_prompt = media_prompt or transcript or response
            media_action = "new" if media_action == "none" else media_action
            try:
                logger.info(
                    "Speaker Omni queued media analysis trigger: "
                    f"source={selected_input_source}, action={media_action}, "
                    f"has_model_prompt={bool(result.payload.get('media_analysis_prompt'))}, "
                    f"transcript={transcript!r}, response={response!r}"
                )
                await self._media_analysis_prompt_handler(
                    user_text,
                    media_prompt,
                    media_action,
                    selected_input_source,
                )
            except Exception as exc:
                logger.warning(f"Speaker Omni media-analysis prompt handler failed: {exc}")


def _clean_spoken_response_artifacts(text: str) -> str:
    """Remove worker-only prompt fragments if the model leaks them into speech."""
    cleaned = text.strip()
    cleaned = cleaned.replace("Answer only with the final user-facing result.", "")
    cleaned = cleaned.replace("Answer only with the final user-facing result", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _normalize_media_analysis_action(value: Any) -> str:
    action = str(value or "none").strip().lower()
    return action if action in {"none", "new", "rerun"} else "none"


def _normalize_selected_input_source(value: Any) -> str:
    source = str(value or "none").strip().lower()
    return source if source in {"none", "live_webcam", "uploaded_attachment"} else "none"


def _missing_uploaded_attachment_response(transcript: str) -> str:
    if "video" in transcript.lower():
        return "Please upload or attach the video first, then I can take a look."
    return "Please upload or attach the media first, then I can take a look."


class SpeakerOmniAgent(PipelineWorker):
    """Main conversational agent backed by the upstream-style Omni service.

    A bus-bridged ``PipelineWorker`` (``bridged=()`` accepts frames from all
    bridges): it receives user frames teed from the transport worker's
    ``BusBridgeProcessor`` and is the only worker that emits spoken responses.
    """

    AGENT_NAME = "speaker_omni"

    def __init__(
        self,
        name: str | None = None,
        *,
        context: LLMContext,
        api_key: str,
        base_url: str,
        model_id: str,
        audio_response_instruction: str,
        extra_params: dict[str, Any] | None = None,
        media_analysis_prompt_handler: Callable[[str, str, str, str], Awaitable[None]] | None = None,
        uploaded_attachment_available: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the bridged Speaker Omni agent."""
        omni = SubagentsSpeakerOmniService(
            api_key=api_key,
            base_url=base_url,
            context=context,
            extra=dict(extra_params or {}),
            settings=NvidiaOmniSettings(
                model=model_id,
                max_tokens=parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64),
                temperature=parse_env_float("OMNI_TEMPERATURE", 0.7, min_value=0.0),
                top_p=parse_env_float("OMNI_TOP_P", 0.95, min_value=0.0),
                response_format={"type": "json_object"},
                emit_transcriptions=True,
                min_user_audio_secs=parse_env_float("OMNI_MIN_USER_AUDIO_SECS", 0.3, min_value=0.0),
            ),
            media_analysis_prompt_handler=media_analysis_prompt_handler,
            uploaded_attachment_available=uploaded_attachment_available,
            audio_response_instruction=audio_response_instruction,
        )
        super().__init__(
            Pipeline([omni]),
            name=name or self.AGENT_NAME,
            active=True,
            bridged=(),
            enable_rtvi=False,
        )
