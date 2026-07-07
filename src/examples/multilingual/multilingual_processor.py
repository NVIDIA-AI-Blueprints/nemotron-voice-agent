# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Multilingual support for the cascaded pipeline.

The session is locked to a single language for the whole connection: the ASR,
the TTS voice, and the LLM all operate in that one language. The LLM replies
with plain spoken text (no JSON, no metadata), so responses flow straight to
TTS and into a clean chat history.

``PerTurnReminderProcessor`` re-states the language contract on every user turn
at request time only, so the stored context stays clean.
"""

import copy

from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

import config_store
from examples.shared.prewarm import build_session_languages, load_voice_map


class PerTurnReminderProcessor(FrameProcessor):
    """Append a language reminder to the last user message at request time only.

    Sits between the user aggregator and the LLM. When an ``LLMContextFrame``
    passes through, it forwards a *copy* of the context whose last user message
    carries the reminder, leaving the shared/stored context untouched so chat
    history (and summaries) never contain the reminder text.
    """

    def __init__(self, reminder: str):
        """Build the processor with the reminder text to inject per turn."""
        super().__init__()
        self._reminder = (reminder or "").strip()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward frames; inject the reminder into outbound context frames."""
        await super().process_frame(frame, direction)
        if self._reminder and isinstance(frame, LLMContextFrame):
            await self.push_frame(self._reminded_frame(frame), direction)
            return
        await self.push_frame(frame, direction)

    def _reminded_frame(self, frame: LLMContextFrame) -> LLMContextFrame:
        source = frame.context
        messages = self._append_reminder(list(source.get_messages()))
        new_context = LLMContext(messages, tools=source.tools, tool_choice=source.tool_choice)
        return LLMContextFrame(context=new_context)

    def _append_reminder(self, messages: list) -> list:
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, dict) and msg.get("role") == "user":
                messages[i] = self._augment(msg)
                return messages
        messages.append({"role": "user", "content": self._reminder})
        return messages

    def _augment(self, msg: dict) -> dict:
        content = msg.get("content")
        if isinstance(content, str):
            return {**msg, "content": f"{content}\n\n{self._reminder}"}
        if isinstance(content, list):
            return {**msg, "content": [*content, {"type": "text", "text": self._reminder}]}
        return {**msg, "content": self._reminder}


def with_reasoning(extra: dict, enabled: bool) -> dict:
    """Return a copy of OpenAI extra params with reasoning toggled.

    Sets ``extra_body.chat_template_kwargs.enable_thinking``. Used to enable
    reasoning for the out-of-band summarizer (better summary faithfulness)
    while spoken turns stay reasoning-off for low latency.
    """
    merged = copy.deepcopy(extra) if extra else {}
    extra_body = merged.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        return merged
    ctk = extra_body.setdefault("chat_template_kwargs", {})
    if isinstance(ctk, dict):
        ctk["enable_thinking"] = enabled
    return merged


_LANGUAGE_NAMES: dict[str, str] = {
    "de": "German (Deutsch)",
    "en": "English",
    "es": "Spanish (español)",
    "fr": "French (français)",
    "hi": "Hindi (हिन्दी)",
    "it": "Italian (italiano)",
    "ja": "Japanese (日本語)",
    "vi": "Vietnamese (Tiếng Việt)",
    "zh": "Chinese (中文)",
}


def describe_language(code: str) -> str:
    """Return a human-readable language name for a BCP-47 code (falls back to the code)."""
    if not code:
        return ""
    base = code.split("-")[0].strip().lower()
    return _LANGUAGE_NAMES.get(base, code)


_ONE_LANGUAGE_RULE = (
    "Your ENTIRE response must be in ONE single language only. Never mix two languages "
    "and never switch language mid-sentence."
)


def build_reminder(fixed_language: str) -> str:
    """Build the concise per-turn language reminder for the fixed session language."""
    name = describe_language(fixed_language)
    return f"Reminder: reply only in {name}, using its standard native script. {_ONE_LANGUAGE_RULE}"


def get_lang_codes(
    *,
    asr_server: str = "",
    asr_model: str = "",
    asr_function_id: str = "",
    tts_server: str = "",
    tts_voice_id: str = "",
) -> str:
    """Comma-separated language codes shared by the ASR and TTS services (informational)."""
    if asr_server or tts_server:
        languages = build_session_languages(
            asr_server,
            asr_model,
            asr_function_id,
            tts_server,
            tts_voice_id,
        ).get("languages", [])
        if languages:
            return ", ".join(languages)

    tts_config = config_store.get("tts", {})
    languages = tts_config.get("languages", []) if isinstance(tts_config, dict) else []
    if languages:
        return ", ".join(languages)
    return ", ".join(sorted(load_voice_map()))


FIXED_SESSION_LANGUAGE_ADDON_KEY = "fixed_session_language_addon"
FIXED_SESSION_GREETING_TRIGGER = "[session_start]"
