# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Browser webcam vision worker subagent.

Uses the Nemotron 3 Nano Omni model card's recommended NATIVE video path: the
model understands motion from ONE continuous video (Conv3D + EVS), so this worker
concatenates the most recent seconds of webcam frames into a single mp4 and sends
it as one ``video_url`` — NOT frame-by-frame images (which the card notes cause
hallucinated temporal relations). Reasoning-off keeps each summary sub-second.

When gesture detection is enabled (the default), the reply also carries a small
``visual_control`` object scoring deliberate hand gestures (greet / stop /
continue / down); video makes the wave-vs-still judgement more reliable.
"""

from __future__ import annotations

import asyncio
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import imageio_ffmpeg
from loguru import logger
from pipecat.bus.messages import BusJobRequestMessage
from pipecat.pipeline.job_decorator import job
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.workers.base_worker import BaseWorker

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniService,
    NvidiaOmniSettings,
    text_message_part,
    video_message_part,
)
from examples.omni_assistant_subagents.subagents.gestures import normalize_visual_control
from examples.shared.json_parsing import extract_json_object
from webcam_frame_store import WebcamFrame, recent_webcam_frames

WEBCAM_SUMMARY_TASK_NAME = "summarize_webcam_frame"

WEBCAM_CONTEXT_PREFIX = "Live webcam state:"
WEBCAM_FIRST_SIGHT_PREFIX = "First live webcam sighting:"
SPEAKER_STATE_PREFIXES: tuple[str, ...] = (WEBCAM_CONTEXT_PREFIX, WEBCAM_FIRST_SIGHT_PREFIX)

_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
_SUMMARY_MAX_TOKENS = 128
_WINDOW_SECONDS = 8.0
_MAX_FRAMES = 32
_ENCODE_FPS = 2
_TEMPERATURE = 0.2
_FFMPEG_TIMEOUT_SECONDS = 15


def _steering_preamble(conversation: str) -> str:
    """Build the self-steering block prepended to the per-frame prompt.

    The worker steers itself: by default it prioritizes the person's current activity —
    what they are holding, showing, or doing — and, when a recent conversation is given,
    it emphasizes whichever visible things and details that conversation is about. Both
    are priorities, not strict filters, and neither changes the output format or the
    gesture-scoring rules that follow.
    """
    parts: list[str] = []
    if conversation:
        parts.append(
            f"RECENT CONVERSATION (context only; the person cannot be heard in this silent video):\n{conversation}\n"
        )
    parts.append(
        "Lead with what the person is actively holding, showing, or doing right now, and describe ALL the items "
        "they are presenting or interacting with, not just one. When the conversation above is about something "
        "visible, emphasize that; otherwise simply prioritize their current activity. Report ONLY what is genuinely "
        "visible in this video: never state a detail the conversation implies but the video does not actually show, "
        "and if a relevant detail is not visually clear, say so rather than guessing. Still obey the output format, "
        "the rule against reading fine printed text, and the gesture rules below exactly.\n"
    )
    return "".join(parts)


class WebcamAgent(BaseWorker):
    """Worker that summarizes the recent webcam window as ONE video via Nemotron Omni."""

    AGENT_NAME = "omni_webcam"

    def __init__(
        self,
        name: str | None = None,
        *,
        api_key: str,
        base_url: str,
        model_id: str,
        extra_params: dict[str, Any] | None = None,
        reasoning: str = "off",
        gesture_system_prompt: str = "",
        gesture_prompt: str = "",
    ) -> None:
        """Initialize the webcam vision worker.

        ``reasoning`` (from ``subagents.yaml``) selects the model thinking mode;
        reasoning-off (the default) is what keeps each summary sub-second.
        """
        super().__init__(name or self.AGENT_NAME, active=True)
        self._base_url = base_url
        self._model_id = model_id
        enable_thinking = reasoning == "on"
        self._max_tokens = _SUMMARY_MAX_TOKENS
        self._window_seconds = _WINDOW_SECONDS
        self._max_frames = _MAX_FRAMES
        self._encode_fps = _ENCODE_FPS
        self._temperature = _TEMPERATURE
        self._system_prompt = gesture_system_prompt.strip()
        self._prompt = gesture_prompt.strip()
        if not self._system_prompt or not self._prompt:
            raise ValueError("WebcamAgent system and user prompts must be provided from prompts.yaml")
        omni_extra = dict(extra_params or {})
        extra_body = dict(omni_extra.get("extra_body") or {})
        extra_body["chat_template_kwargs"] = {
            **dict(extra_body.get("chat_template_kwargs") or {}),
            "enable_thinking": enable_thinking,
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
                input_modalities=("video", "text"),
                stream=False,
            ),
        )

    @job(name=WEBCAM_SUMMARY_TASK_NAME)
    async def summarize_webcam_frame(self, message: BusJobRequestMessage) -> None:
        """Summarize the recent webcam window (ONE continuous video) and score gestures."""
        payload = message.payload or {}
        frame_metadata = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
        session_id = str(payload.get("session_id") or "").strip()
        conversation_context = str(payload.get("conversation_context") or "").strip()
        try:
            window_seconds = float(payload.get("window_seconds") or self._window_seconds)
        except (TypeError, ValueError):
            window_seconds = self._window_seconds
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            window_seconds = self._window_seconds

        observation = ""
        focus = ""
        visual_control = normalize_visual_control({})
        frames = recent_webcam_frames(session_id, max_seconds=window_seconds, max_count=self._max_frames)
        if frames:
            try:
                mp4 = await asyncio.to_thread(self._encode_mp4, frames)
                if mp4:
                    observation, visual_control, focus = await self._describe(
                        mp4, len(frames), window_seconds, conversation_context
                    )
            except Exception as exc:
                logger.exception(f"Webcam video summary failed: {exc}")
                observation = ""
                visual_control = normalize_visual_control({})

        await self.send_job_response(
            message.job_id,
            {
                "mode": "summary",
                "observation": observation,
                "focus": focus,
                "visual_control": visual_control,
                "frame": frame_metadata,
            },
        )

    def _encode_mp4(self, frames: list[WebcamFrame]) -> bytes | None:
        """Concatenate recent JPEG frames into ONE continuous mp4 (blocking; run in a thread)."""
        imgs = list(frames)
        if not imgs:
            return None
        if len(imgs) < 2:
            imgs = imgs + [imgs[-1]]
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(imgs):
                (Path(d) / f"{i:05d}.jpg").write_bytes(f.data)
            out = Path(d) / "out.mp4"
            subprocess.run(
                [
                    _FFMPEG,
                    "-y",
                    "-framerate",
                    str(self._encode_fps),
                    "-i",
                    str(Path(d) / "%05d.jpg"),
                    "-vf",
                    "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(out),
                ],
                check=True,
                capture_output=True,
                timeout=_FFMPEG_TIMEOUT_SECONDS,
            )
            return out.read_bytes()

    async def _describe(
        self, mp4: bytes, n_frames: int, window_seconds: float, conversation: str = ""
    ) -> tuple[str, dict[str, Any], str]:
        """Describe the recent-window video and score gestures.

        The recent ``conversation`` is prepended so the worker self-steers onto the
        details that matter now while defaulting to the person's current activity.
        """
        steering = _steering_preamble(conversation)
        prompt = f"{steering}\n{self._prompt}" if steering else self._prompt
        content = [video_message_part(mp4), text_message_part(prompt)]
        context = LLMContext(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": content},
            ]
        )
        logger.debug(
            f"Webcam video Omni request: base_url={self._base_url}, model={self._model_id}, "
            f"mp4_bytes={len(mp4)}, frames={n_frames}, window_s={window_seconds}, "
            f"conversation_chars={len(conversation)}"
        )
        result = await self._omni.run_multimodal_inference(
            context,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            stream=False,
        )
        raw = result.text.strip()
        payload = extract_json_object(raw)
        if not payload:
            logger.warning("Ignoring malformed WebcamAgent response")
            return "", normalize_visual_control({}), ""
        observation = str(payload.get("observation") or "").strip()
        focus = str(payload.get("focus") or "").strip()
        return observation, normalize_visual_control(payload.get("visual_control")), focus
