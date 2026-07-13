# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Uploaded-media analysis dispatch and follow-up emission."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger
from pipecat.bus.messages import BusJobResponseMessage
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from attachment_store import latest_user_attachment, remove_attachment
from examples.omni_assistant_subagents.subagents.media_analyzer import (
    MEDIA_ANALYSIS_RUNNING_PREFIX,
    MEDIA_ANALYSIS_TASK_NAME,
    MediaAnalyzerWorker,
)
from examples.omni_assistant_subagents.subagents.transport.speaker_context import SpeakerContextManager
from examples.omni_assistant_subagents.subagents.transport.subagent_state_board import SubagentStateBoard

_PENDING_ANALYSIS = "PENDING — a freshly uploaded attachment is waiting and has not been analyzed yet"


class MediaAnalysisController:
    """Own queued uploaded-media analysis and client-visible follow-up output."""

    def __init__(
        self,
        *,
        session_id: str,
        speaker_context: SpeakerContextManager,
        board: SubagentStateBoard,
        request_job: Callable[..., Awaitable[str]],
        queue_frame: Callable[[Any], Awaitable[None]],
        followup_delay_secs: float,
    ) -> None:
        """Initialize uploaded-media dispatch state for one session."""
        self._session_id = session_id
        self._speaker_context = speaker_context
        self._board = board
        self._request_job = request_job
        self._queue_frame = queue_frame
        self._followup_delay_secs = followup_delay_secs
        self._pending_transcript = ""
        self._pending_prompt = ""
        self._pending_action = "none"
        self._pending_attachment_id = ""
        self._active_attachment_id = ""
        self._analyzed_attachment_id = ""
        self._capture_task_ids: set[str] = set()

    def has_uploaded_attachment(self) -> bool:
        """Return whether this session currently has a user-uploaded attachment."""
        return latest_user_attachment(self._session_id) is not None

    def is_attachment_pending(self) -> bool:
        """Whether a freshly uploaded attachment is still waiting to be analyzed.

        This is the ONLY state in which the uploaded attachment is the Speaker's visual
        focus: once analyzed it becomes past context and the live webcam is the default
        again. The Speaker offers the attachment as a routing target only while pending,
        so it can never re-route a "what do you see now" turn to a stale, already-seen file.
        """
        attachment = latest_user_attachment(self._session_id)
        return attachment is not None and attachment.id != self._analyzed_attachment_id

    def mark_attachment_pending(self) -> None:
        """Mark the board pending when a fresh, unanalyzed attachment is uploaded.

        Surfaces the plain fact that a new attachment is waiting (the user's most recent
        visual share), so the Speaker — per the prompt's source-priority principle — treats
        it as the current focus until analyzed; afterwards the live webcam is the default
        again. No-op if there is no attachment or it is already the analyzed one.
        """
        attachment = latest_user_attachment(self._session_id)
        if attachment is None or attachment.id == self._analyzed_attachment_id:
            return
        self._board.set_findings(MediaAnalyzerWorker.AGENT_NAME, _PENDING_ANALYSIS)
        logger.info("Marked uploaded attachment as pending analysis on the subagents board")

    def _clear_pending(self) -> None:
        self._pending_transcript = ""
        self._pending_prompt = ""
        self._pending_action = "none"
        self._pending_attachment_id = ""

    async def queue_prompt(
        self,
        transcript: str,
        prompt: str,
        action: str,
        selected_input_source: str,
    ) -> None:
        """Queue Speaker Omni's hidden analyzer prompt after its acknowledgement closes."""
        cleaned_transcript = transcript.strip()
        cleaned_prompt = prompt.strip()
        if not (cleaned_prompt or cleaned_transcript):
            return
        if selected_input_source != "uploaded_attachment":
            logger.info(
                "Ignoring LLM-selected media analysis because selected_input_source is not uploaded_attachment: "
                f"source={selected_input_source!r}, transcript_chars={len(cleaned_transcript)}"
            )
            self._clear_pending()
            return
        media_action = action if action in {"new", "rerun"} else "new"
        attachment = latest_user_attachment(self._session_id)
        if attachment is None:
            logger.info(
                "Ignoring LLM-selected media analysis because no attachment is available: "
                f"transcript_chars={len(cleaned_transcript)}"
            )
            self._clear_pending()
            return
        attachment_id = attachment.id
        if attachment_id == self._active_attachment_id:
            logger.info(
                "Ignoring media analysis prompt because analysis is already active: "
                f"attachment_id={attachment_id}, action={media_action}, transcript_chars={len(cleaned_transcript)}"
            )
            self._clear_pending()
            return
        self._pending_transcript = cleaned_transcript
        self._pending_prompt = cleaned_prompt
        self._pending_action = media_action
        self._pending_attachment_id = attachment_id
        logger.info(
            f"Queued LLM-selected media analysis after ack: action={media_action}, "
            f"transcript_chars={len(cleaned_transcript)}"
        )

    async def start_pending(self) -> None:
        """Dispatch the LLM-selected analyzer after the ack turn has completed."""
        transcript = self._pending_transcript
        prompt = self._pending_prompt
        action = self._pending_action
        attachment_id = self._pending_attachment_id
        self._clear_pending()
        if not transcript:
            logger.debug("Bot stopped speaking; no pending media analysis queued")
            return
        attachment = latest_user_attachment(self._session_id)
        if attachment is None or attachment.id != attachment_id:
            logger.info("LLM-selected media analysis attachment is no longer current")
            return
        attachment_metadata = attachment.metadata()
        if attachment_id == self._active_attachment_id:
            logger.info(
                "Skipping duplicate media analysis dispatch because analysis is already active: "
                f"attachment_id={attachment_id}, action={action}, transcript_chars={len(transcript)}"
            )
            return

        self._active_attachment_id = attachment_id
        prior_analysis = (
            self._board.get_findings(MediaAnalyzerWorker.AGENT_NAME)
            if action != "rerun" and attachment_id == self._analyzed_attachment_id
            else ""
        )
        try:
            task_id = await self._request_job(
                MediaAnalyzerWorker.AGENT_NAME,
                name=MEDIA_ANALYSIS_TASK_NAME,
                payload={
                    "transcript": transcript,
                    "session_id": self._session_id,
                    "attachment": attachment_metadata,
                    "analysis_prompt": prompt,
                    "prior_analysis": prior_analysis,
                    "analysis_action": action,
                },
                timeout=120.0,
            )
        except Exception as exc:
            self._active_attachment_id = ""
            logger.warning(f"Failed to dispatch media analysis: {exc}")
            return
        logger.info(
            "Media analysis dispatched: "
            f"task_id={task_id}, action={action}, mode={'patch' if prior_analysis else 'full'}, "
            f"transcript_chars={len(transcript)}"
        )
        self._speaker_context.set_pinned_state(
            MEDIA_ANALYSIS_RUNNING_PREFIX,
            f"{MEDIA_ANALYSIS_RUNNING_PREFIX} "
            "If the user asks for status before it completes, say it is still running.",
        )

    async def clear_pending_on_interruption(self) -> None:
        """Cancel pending post-ack media work when the ack was interrupted by speech."""
        if not self._pending_transcript:
            return
        logger.info(
            "Clearing pending media analysis because the assistant ack was interrupted: "
            f"transcript_chars={len(self._pending_transcript)}"
        )
        self._clear_pending()

    async def analyze_capture(self, attachment_metadata: dict[str, Any], query: str) -> None:
        """Dispatch a one-shot focused analysis of an agent-captured high-res webcam frame.

        Reuses the media-analyzer worker and the spoken-followup path, but the result is
        NOT pinned to the board: this is a transient focused look at the live scene, not
        the uploaded-media analysis the board tracks.
        """
        cleaned_query = query.strip() or "Describe what is visible in this image in detail."
        attachment_id = str(attachment_metadata.get("id") or "")
        try:
            task_id = await self._request_job(
                MediaAnalyzerWorker.AGENT_NAME,
                name=MEDIA_ANALYSIS_TASK_NAME,
                payload={
                    "transcript": cleaned_query,
                    "session_id": self._session_id,
                    "attachment": attachment_metadata,
                    "analysis_prompt": cleaned_query,
                    "prior_analysis": "",
                    "analysis_action": "new",
                },
                timeout=120.0,
            )
        except Exception as exc:
            if attachment_id:
                remove_attachment(self._session_id, attachment_id)
            logger.warning(f"Failed to dispatch high-res capture analysis: {exc}")
            return
        self._capture_task_ids.add(task_id)
        logger.info(f"High-res capture analysis dispatched: task_id={task_id}, query_chars={len(cleaned_query)}")

    async def handle_job_response(self, message: BusJobResponseMessage) -> bool:
        """Emit uploaded-media job results. Return whether the response was handled."""
        response = message.response or {}
        tts = str(response.get("tts") or "").strip()
        analysis = str(response.get("analysis") or "").strip()
        append_patch = str(response.get("append_patch") or "").strip()
        is_patch = bool(response.get("is_patch"))
        reasoning = str(response.get("reasoning") or "").strip()
        query = str(response.get("query") or "").strip()
        attachment = response.get("attachment") if isinstance(response.get("attachment"), dict) else {}
        is_capture = message.job_id in self._capture_task_ids
        self._capture_task_ids.discard(message.job_id)
        attachment_id = str(attachment.get("id") or "")
        if not tts:
            if is_capture and attachment_id:
                remove_attachment(self._session_id, attachment_id)
            return False

        if not is_capture:
            latest = latest_user_attachment(self._session_id)
            if latest is None or latest.id != attachment_id:
                logger.info(
                    f"Skipping board update for a superseded attachment analysis: attachment_id={attachment_id}"
                )
            elif is_patch and append_patch:
                self._board.append_findings(MediaAnalyzerWorker.AGENT_NAME, append_patch)
            elif not is_patch and analysis:
                self._board.set_findings(MediaAnalyzerWorker.AGENT_NAME, analysis)
                self._analyzed_attachment_id = attachment_id

        try:
            await self._emit_analysis_response(
                tts,
                task_id=message.job_id,
                agent=str(getattr(message, "source", "") or MediaAnalyzerWorker.AGENT_NAME),
                attachment=attachment,
                reasoning=reasoning,
                query=query,
                clear_running=not is_capture,
            )
        finally:
            if is_capture and attachment_id:
                remove_attachment(self._session_id, attachment_id)
        if attachment_id and self._active_attachment_id == attachment_id:
            self._active_attachment_id = ""
        return True

    async def _emit_analysis_response(
        self,
        tts: str,
        *,
        task_id: str,
        agent: str,
        attachment: dict[str, Any],
        reasoning: str,
        query: str,
        clear_running: bool,
    ) -> None:
        """Speak the analyzer's TTS answer as its own assistant turn.

        The detailed analysis already lives in the pinned board and therefore in
        the context snapshot.
        """
        spoken = tts.strip()
        if clear_running:
            self._speaker_context.set_pinned_state(MEDIA_ANALYSIS_RUNNING_PREFIX, "")
        await asyncio.sleep(self._followup_delay_secs)
        await self._queue_frame(LLMFullResponseStartFrame())
        await self._queue_frame(LLMTextFrame(text=spoken))
        await self._queue_frame(LLMFullResponseEndFrame())
        await self._queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "agent-task-update",
                    "task_id": task_id,
                    "agent": agent,
                    "status": "done",
                    "stage": "complete",
                    "detail": spoken,
                    "attachment": attachment,
                    "query": query,
                    "reasoning": reasoning,
                    "response": spoken,
                    "spoken_response": spoken,
                }
            )
        )
