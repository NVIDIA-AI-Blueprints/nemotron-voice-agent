# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Multilingual support for the cascaded pipeline.

Parses the LLM's ``Language: <code> Text: <reply> MetaData: <info>`` format in
a streaming aggregator, switches the TTS voice on the detected language, and
exposes the lang/meta segments to the assistant context (so chat history
keeps the structured response) while skipping them on the TTS side.

Designed to be plugged into a Pipecat ``LLMTextProcessor`` together with a
TTS service configured with ``skip_aggregator_types=SKIP_TTS_AGGREGATIONS``.
"""

import re
from collections.abc import AsyncIterator, Awaitable, Callable

from loguru import logger
from pipecat.frames.frames import (
    AggregatedTextFrame,
    Frame,
    LLMTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator

import config_store
from examples.shared.prewarm import build_session_languages, load_voice_map
from utils import normalize_lang_code

LANG_TYPE = "lang"
META_TYPE = "meta"
SKIP_TTS_AGGREGATIONS: list[str] = [LANG_TYPE, META_TYPE]

_LANG_PREFIX = "Language:"
_TEXT_MARKER = "Text:"
_META_MARKER = "MetaData:"
_TEXT_MARKER_RE = re.compile(rf"\s*{re.escape(_TEXT_MARKER)}\s*", re.IGNORECASE)
_META_MARKER_RE = re.compile(rf"\s*{re.escape(_META_MARKER)}\s*", re.IGNORECASE)

_STATE_HEADER = 0
_STATE_TEXT = 1
_STATE_META = 2

_LanguageHandler = Callable[[str], Awaitable[None]]
_LanguageSwitchNotifier = Callable[[str, str], Awaitable[None]]


class MultilingualTextAggregator(BaseTextAggregator):
    """Streaming aggregator for the ``Language: ... Text: ... MetaData: ...`` format.

    Yields three aggregation types:

    * ``"lang"`` — the full ``Language: <code> Text:`` header.
    * ``"sentence"`` — each complete sentence of the spoken reply.
    * ``"meta"`` — the trailing ``MetaData: <info>`` block.

    ``on_language`` fires the moment ``Language: <code>`` is fully buffered,
    independent of (and before) the ``Text:`` marker.
    """

    def __init__(self, *, on_language: _LanguageHandler | None = None):
        """Build the aggregator with an optional language-switch callback."""
        super().__init__(aggregation_type=AggregationType.SENTENCE)
        self._on_language = on_language
        self._state = _STATE_HEADER
        self._buf = ""
        self._sentence_aggregator = SimpleTextAggregator()
        self._lang_handler_fired = False
        self._pending_meta = ""

    @property
    def text(self) -> Aggregation:
        """Return the currently buffered text as a sentence aggregation."""
        return Aggregation(text=self._buf, type=AggregationType.SENTENCE.value)

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        """Consume streaming LLM text and yield typed aggregations."""
        self._buf += text
        await self._maybe_fire_language_handler()

        while True:
            if self._state == _STATE_HEADER:
                text_match = _TEXT_MARKER_RE.search(self._buf)
                if text_match:
                    header = self._buf[: text_match.start()].strip()
                    self._buf = self._buf[text_match.end() :]
                    self._state = _STATE_TEXT
                    if header:
                        yield Aggregation(text=f"{header} Text:", type=LANG_TYPE)
                    continue

                return

            if self._state == _STATE_TEXT:
                meta_match = _META_MARKER_RE.search(self._buf)
                if meta_match:
                    pending = self._buf[: meta_match.start()]
                    self._buf = self._buf[meta_match.end() :]
                    if pending:
                        async for agg in self._sentence_aggregator.aggregate(pending):
                            yield agg
                    trailing = await self._sentence_aggregator.flush()
                    if trailing and trailing.text.strip():
                        yield trailing
                    self._state = _STATE_META
                    continue

                holdback = _partial_field_marker_suffix_len(self._buf, _META_MARKER)
                pending, self._buf = (self._buf[:-holdback], self._buf[-holdback:]) if holdback else (self._buf, "")
                if pending:
                    async for agg in self._sentence_aggregator.aggregate(pending):
                        yield agg
                return

            return

    async def flush(self) -> Aggregation | None:
        """Yield any pending text once the LLM response ends."""
        if self._pending_meta:
            text = self._pending_meta
            self._pending_meta = ""
            return Aggregation(text=f"MetaData: {text}", type=META_TYPE)

        if self._state == _STATE_META:
            text = self._buf.strip()
            self._buf = ""
            return Aggregation(text=f"MetaData: {text}", type=META_TYPE) if text else None

        if self._state == _STATE_TEXT:
            meta_match = _META_MARKER_RE.search(self._buf)
            if meta_match:
                pending = self._buf[: meta_match.start()]
                self._buf = self._buf[meta_match.end() :]
                spoken_text = pending.strip()
                meta_text = self._buf.strip()
                self._buf = ""
                self._state = _STATE_META
                if spoken_text:
                    self._pending_meta = meta_text
                    return Aggregation(text=spoken_text, type=AggregationType.SENTENCE.value)
                return Aggregation(text=f"MetaData: {meta_text}", type=META_TYPE) if meta_text else None

            pending, self._buf = self._buf, ""
            if pending:
                async for _ in self._sentence_aggregator.aggregate(pending):
                    pass
            tail = await self._sentence_aggregator.flush()
            return tail if tail and tail.text.strip() else None

        text = self._buf.strip()
        self._buf = ""
        if not text:
            return None
        logger.warning("Multilingual: LLM response missing 'Text:' marker; dropping raw output")
        return None

    def set_on_language(self, handler: _LanguageHandler | None) -> None:
        """Wire (or rewire) the language-switch handler post-construction."""
        self._on_language = handler

    async def handle_interruption(self):
        """Drop all buffered state when the user interrupts the bot."""
        await self.reset()

    async def reset(self):
        """Clear buffered state and reset the streaming parse to the header."""
        self._state = _STATE_HEADER
        self._buf = ""
        self._lang_handler_fired = False
        self._pending_meta = ""
        await self._sentence_aggregator.reset()

    async def _maybe_fire_language_handler(self) -> None:
        if self._lang_handler_fired or self._on_language is None:
            return
        code = _detect_language_code(self._buf)
        if not code:
            return
        self._lang_handler_fired = True
        try:
            await self._on_language(code)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Multilingual: language handler failed: {exc}")


def _detect_language_code(buf: str) -> str:
    """Return ``<code>`` once ``Language: <code>`` is fully buffered (code + ws)."""
    stripped = buf.lstrip()
    if not stripped.lower().startswith(_LANG_PREFIX.lower()):
        return ""
    rest = stripped[len(_LANG_PREFIX) :].lstrip()
    for i, ch in enumerate(rest):
        if ch.isspace():
            return rest[:i].strip()
    return ""


def _partial_field_marker_suffix_len(buf: str, marker: str) -> int:
    """Length of trailing text that may become a field marker on the next chunk."""
    trailing_whitespace = len(buf) - len(buf.rstrip())
    if trailing_whitespace:
        return trailing_whitespace

    lower_buf = buf.lower()
    lower_marker = marker.lower()
    for k in range(min(len(buf), len(marker) - 1), 0, -1):
        if lower_buf[-k:] == lower_marker[:k]:
            return k
    return 0


def get_lang_codes(
    *,
    asr_server: str = "",
    asr_model: str = "",
    asr_function_id: str = "",
    tts_server: str = "",
    tts_voice_id: str = "",
) -> str:
    """Comma-separated language codes for prompt ``{lang_codes}`` injection."""
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
        return ""

    tts_config = config_store.get("tts", {})
    languages = tts_config.get("languages", []) if isinstance(tts_config, dict) else []
    if languages:
        return ", ".join(languages)
    return ", ".join(sorted(load_voice_map()))


FIXED_SESSION_LANGUAGE_ADDON_KEY = "fixed_session_language_addon"
AUTO_DETECT_LANGUAGE_ADDON_KEY = "auto_detect_language_addon"
FIXED_SESSION_GREETING_TRIGGER = "[session_start]"


def make_language_handler(
    tts: NvidiaTTSService,
    task: PipelineWorker,
    *,
    on_language_switched: _LanguageSwitchNotifier | None = None,
    fixed_language: str = "",
) -> _LanguageHandler:
    """Build the language handler that queues a TTSUpdateSettingsFrame on language change."""
    locked = normalize_lang_code(fixed_language) if fixed_language else ""
    current = ""

    async def handle(code: str) -> None:
        nonlocal current
        normalized = normalize_lang_code(locked or code)
        if not normalized:
            logger.warning(f"Multilingual: empty language code in LLM response: {code!r}")
            return
        if current.lower() == normalized.lower():
            return
        voice_map = load_voice_map()
        voice_id = voice_map.get(normalized.lower())
        if not voice_id:
            logger.warning(f"Multilingual: language '{normalized}' not in voice catalog ({sorted(voice_map.keys())})")
            return
        current = normalized.lower()
        try:
            await task.queue_frame(
                TTSUpdateSettingsFrame(
                    delta=NvidiaTTSSettings(voice=voice_id, language=normalized),
                    service=tts,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Multilingual: TTS settings update failed: {exc}")
            return
        logger.info(f"Multilingual: TTS → language={normalized}, voice={voice_id}")
        if on_language_switched:
            try:
                await on_language_switched(normalized, voice_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Multilingual: RTVI language notification failed: {exc}")

    return handle


class RTVISpokenTextEmitter(FrameProcessor):
    """Mirror spoken-sentence aggregations as ``LLMTextFrame``s for RTVI.

    Sits between ``LLMTextProcessor`` and TTS so RTVI's ``BotLlmText`` event
    sees clean spoken text from a non-llm source. Mirrored frames carry
    ``skip_tts=True`` (TTS already speaks via the AggregatedTextFrame path)
    and ``append_to_context=False`` (the assistant aggregator already captures
    the AggregatedTextFrame, so we must not double-add).
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward each frame; mirror spoken aggregations as ``LLMTextFrame``."""
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)
        if not isinstance(frame, AggregatedTextFrame):
            return
        if frame.aggregated_by in SKIP_TTS_AGGREGATIONS:
            return
        text = (frame.text or "").strip()
        if not text:
            return
        mirror = LLMTextFrame(text=text)
        mirror.skip_tts = True
        mirror.append_to_context = False
        await self.push_frame(mirror, direction)
