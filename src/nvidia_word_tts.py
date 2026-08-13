# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA Magpie WordTTS service.

Local draft of a Pipecat child class: does **not** modify upstream
:class:`~pipecat.services.nvidia.tts.NvidiaTTSService`. Later, move this module
into ``pipecat.services.nvidia.tts`` alongside the parent.

Child of :class:`NvidiaTTSService` for the same spoken-word context commit path
used by Cartesia / ElevenLabs / Rime.

* ``push_text_frames=False`` — commits come from ``add_word_timestamps`` (and
  Pipecat ``force_complete`` when Magpie meta is absent)
* Default ``text_aggregation_mode=TOKEN`` — main differentiator vs parent:
  LLM tokens stream into Magpie as they arrive (no sentence buffering)
* Default ``synthesis_mode=stitched`` — one ``SynthesizeOnline`` stream for the
  whole LLM turn; ``TTSStoppedFrame`` / bot-stopped only after that stream ends
* Requests word timings via ``SynthesizeSpeechRequest.enable_word_time_offsets``
  when the installed Riva proto exposes it, and also via
  ``custom_configuration["enable_word_time_offsets"]=true`` for NIM builds that
  map that Triton input through custom config
* Prefer ``response.meta.words`` (``WordTiming``: word, start_time/end_time ms);
  fall back to ``processed_text`` + ``predicted_durations``
* Context commits use **Magpie meta word strings** with
  ``includes_inter_frame_spaces=False`` (insert a space between every timed
  token). Magpie strips leading spaces from ``meta.words``; until that is
  fixed upstream, space injection is the interim for readable context /
  interruption. Magpie PTS remains the interruption source of truth;
  unspoken slot remainder is not force-completed into context.

Stock :class:`NvidiaTTSService` is unchanged (``push_text_frames=True``,
sentence aggregation by default, no timestamp commits).

Proto contract (riva-speech streaming + word timestamps, e.g. !2703)::

    request.enable_word_time_offsets = true
    # once per LLM response, on an empty final SynthesizeOnline message:
    request.custom_configuration["riva_end_stream"] = "true"
    request.custom_configuration["is_last_request"] = "true"
    response.meta.words[].{word, start_time, end_time}  # milliseconds
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncGenerator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
)
from pipecat.services.nvidia.tts import (
    NvidiaTTSService,
    NvidiaTTSSettings,
    NvidiaTTSSynthesisMode,
    _SynthesisStreamState,
)
from pipecat.services.settings import _NotGiven
from pipecat.services.tts_service import TTSService, TextAggregationMode
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import riva.client.proto.riva_tts_pb2 as rtts
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(f"Missing module: {e}") from e

_TOKEN_RE = re.compile(r"\S+|\s+")
_ENABLE_WORD_TIME_OFFSETS_KEY = "enable_word_time_offsets"
# Magpie/Riva end-of-turn flush (riva-speech !2703). Empty final SynthesizeOnline
# request with these keys triggers Triton ``is_last_request`` so meta.words can land.
_RIVA_END_STREAM_KEY = "riva_end_stream"
_IS_LAST_REQUEST_KEY = "is_last_request"
_END_OF_TURN = object()


class _MagpieWordCommitSequencer(AggregatedFrameSequencer):
    """Commit Magpie ``meta.words`` with inter-frame space injection.

    WordTTS path (interim until Magpie preserves input token spacing):
    * ``includes_inter_frame_spaces=False`` — insert a space between Magpie
      tokens (``I``+``am`` → ``I am``). Subword splits may still show gaps
      (``Nem``+``otron`` → ``Nem otron``).
    * Do **not** rewrite commits to LLM token spans — those bypass Magpie PTS
      alignment and can dump unplayed text into context on barge-in.
    * When Magpie meta was seen, ``force_complete`` does **not** emit unspoken
      remainder (timed words only). When meta was missing, remainder commit is
      allowed as a fallback so context/UI are not empty.
    """

    # Set by NvidiaWordTTSService before force_complete when meta.words never arrived.
    commit_unspoken_remainder: bool = False

    def process_word(
        self,
        word: str,
        pts: int,
        context_id: str | None,
        includes_inter_frame_spaces: bool = False,
    ) -> list[Frame]:
        """Commit Magpie ``word`` text with IFS=False (inject spaces)."""
        _ = includes_inter_frame_spaces
        self._current_magpie_word = word
        return super().process_word(word, pts, context_id, False)

    def _build_word_frame(
        self,
        text: str,
        pts: int,
        context_id: str | None,
        raw_text: str | None = None,
        suppress_in_context: bool = False,
        includes_inter_frame_spaces: bool = False,
    ) -> Frame:
        """Build commit frame from Magpie surface text; force IFS=False + log."""
        _ = includes_inter_frame_spaces
        frame = super()._build_word_frame(
            text,
            pts,
            context_id,
            raw_text=raw_text,
            suppress_in_context=suppress_in_context,
            includes_inter_frame_spaces=False,
        )
        if isinstance(frame, TTSTextFrame) and frame.append_to_context:
            magpie = getattr(self, "_current_magpie_word", None)
            logger.debug(
                f"{self._name}: context commit magpie={magpie!r} "
                f"text={frame.text!r} raw={frame.raw_text!r} "
                f"ifs={frame.includes_inter_frame_spaces}"
            )
        return frame

    def force_complete(self, last_word_pts: int) -> list[Frame]:
        """Finish slots; optionally commit unspoken remainder if meta was missing."""
        self._current_magpie_word = None
        if self.commit_unspoken_remainder:
            logger.debug(
                f"{self._name}: force-complete with unspoken remainder "
                f"(Magpie meta missing fallback)"
            )
            # Parent emits remaining TTS/LLM text; our _build_word_frame keeps IFS=False.
            return AggregatedFrameSequencer.force_complete(self, last_word_pts)

        for slot in self._slots:
            if slot.spoken and not slot.complete:
                if slot.tracker:
                    remaining = slot.tracker.get_remaining_tts_text(strip=False)
                    if remaining:
                        logger.debug(
                            f"{self._name}: skip force-complete remainder "
                            f"{remaining!r} (Magpie-timed commits only)"
                        )
                slot.complete = True
        return self.flush(last_word_pts=last_word_pts)


@dataclass
class NvidiaWordTTSSettings(NvidiaTTSSettings):
    """Settings for :class:`NvidiaWordTTSService`.

    Defaults to Magpie ``stitched`` streaming (required for the shared
    ``SynthesizeOnline`` response path that surfaces ``meta.words``).
    """

    synthesis_mode: NvidiaTTSSynthesisMode | _NotGiven = field(
        default_factory=lambda: NvidiaTTSSynthesisMode.STITCHED
    )


@dataclass
class _WordTimingState:
    """Per-audio-context bookkeeping for incremental Magpie meta."""

    emitted_tokens: int = 0


class NvidiaWordTTSService(NvidiaTTSService):
    """NVIDIA TTS WordTTS path: spoken commits + Magpie/Riva word timestamps.

    Use this instead of :class:`NvidiaTTSService` when you need barge-in-accurate
    assistant context (heard words only). When Magpie does not yet return meta,
    Pipecat still force-completes finished spoken aggregation slots on clean
    context end; incomplete slots are dropped on interruption.
    """

    Settings = NvidiaWordTTSSettings
    _settings: NvidiaWordTTSSettings

    def __init__(
        self,
        *,
        enable_word_time_offsets: bool = True,
        custom_configuration: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Create Magpie WordTTS.

        Args:
            enable_word_time_offsets: Request per-word timings from Magpie/Riva.
                When the installed client proto has
                ``SynthesizeSpeechRequest.enable_word_time_offsets``, that field
                is set. Always also sets the matching ``custom_configuration``
                key for NIM builds that forward it to Triton.
            custom_configuration: Extra Riva ``custom_configuration`` entries
                merged into every synthesize request.
            **kwargs: Forwarded to :class:`NvidiaTTSService` (``api_key``,
                ``server``, ``settings``, ``sample_rate``, filters, etc.).

        Notes:
            ``push_text_frames`` is always forced to ``False``. Default
            aggregation is ``TOKEN``. Default synthesis mode is ``stitched`` via
            :class:`NvidiaWordTTSSettings`. Stop frames are emitted only when the
            Magpie ``SynthesizeOnline`` stream ends (after
            ``LLMFullResponseEndFrame`` flush), not on the parent's idle timeout.
        """
        kwargs["push_text_frames"] = False
        kwargs.setdefault("text_aggregation_mode", TextAggregationMode.TOKEN)

        # Parent hardcodes synthesis_mode=PER_SENTENCE when constructing Settings;
        # start from WordTTS stitched defaults, then apply any caller settings delta.
        word_settings = NvidiaWordTTSSettings(
            synthesis_mode=NvidiaTTSSynthesisMode.STITCHED
        )
        if kwargs.get("settings") is not None:
            word_settings.apply_update(kwargs["settings"])
        kwargs["settings"] = word_settings

        super().__init__(**kwargs)

        # Parent hardcodes push_stop_frames=True with a 3s audio-queue idle timeout,
        # which can fire mid-turn while Magpie waits for the next LLM token. Disable
        # that and emit TTSStoppedFrame only when SynthesizeOnline actually ends.
        self._push_stop_frames = False

        # Commit Magpie meta words (no space insert); skip unspoken force-complete.
        # Stock NvidiaTTSService keeps the default AggregatedFrameSequencer.
        self._aggregated_frame_sequencer = _MagpieWordCommitSequencer(name=str(self))

        cfg = dict(custom_configuration or {})
        if enable_word_time_offsets:
            cfg.setdefault(_ENABLE_WORD_TIME_OFFSETS_KEY, "true")
        self._enable_word_time_offsets = enable_word_time_offsets
        self._custom_configuration = cfg
        self._word_states: dict[str, _WordTimingState] = {}
        self._meta_seen = False
        self._turn_meta_seen = False
        self._warned_missing_meta = False

        logger.debug(
            f"{self}: WordTTS "
            f"aggregation={self._text_aggregation_mode} "
            f"push_text_frames={self._push_text_frames} "
            f"push_stop_frames={self._push_stop_frames} "
            f"synthesis_mode={self._settings.synthesis_mode} "
            f"enable_word_time_offsets={self._enable_word_time_offsets} "
            f"commit_sequencer=MagpieWord"
        )

    def _word_state(self, context_id: str) -> _WordTimingState:
        state = self._word_states.get(context_id)
        if state is None:
            state = _WordTimingState()
            self._word_states[context_id] = state
        return state

    def _clear_word_state(self, context_id: str | None = None) -> None:
        if context_id is None:
            self._word_states.clear()
        else:
            self._word_states.pop(context_id, None)

    def _build_base_request(self) -> rtts.SynthesizeSpeechRequest:
        """Build Magpie request including word-timestamp enable flags."""
        req = super()._build_base_request()
        if self._enable_word_time_offsets and hasattr(req, _ENABLE_WORD_TIME_OFFSETS_KEY):
            setattr(req, _ENABLE_WORD_TIME_OFFSETS_KEY, True)
        for key, value in self._custom_configuration.items():
            req.custom_configuration[key] = str(value)
        # Normal token chunks must not carry a stale end-of-turn flush.
        self._set_end_of_turn_flags(req, False)
        return req

    def _set_end_of_turn_flags(self, req: rtts.SynthesizeSpeechRequest, enabled: bool) -> None:
        """Set or clear once-per-LLM-response Magpie flush flags on ``req``."""
        if enabled:
            req.custom_configuration[_RIVA_END_STREAM_KEY] = "true"
            req.custom_configuration[_IS_LAST_REQUEST_KEY] = "true"
        else:
            req.custom_configuration.pop(_RIVA_END_STREAM_KEY, None)
            req.custom_configuration.pop(_IS_LAST_REQUEST_KEY, None)

    def _synthesis_handler(self, state: _SynthesisStreamState):
        """SynthesizeOnline with an explicit end-of-turn flush before stream close.

        Parent closes the client stream with ``None`` only. Magpie word timings are
        emitted on an empty final request that sets ``is_last_request`` /
        ``riva_end_stream`` (once per LLM response). Flags are cleared afterward so
        the next turn starts clean.
        """
        event_loop = self.get_event_loop()
        base_req = self._build_base_request()

        def request_generator():
            while True:
                if state.stop_event.is_set():
                    break
                item = state.text_queue.get()
                if state.stop_event.is_set():
                    break
                if item is _END_OF_TURN:
                    self._set_end_of_turn_flags(base_req, True)
                    base_req.text = ""
                    logger.debug(f"{self}: Magpie end-of-turn flush (is_last_request)")
                    yield base_req
                    self._set_end_of_turn_flags(base_req, False)
                    break
                if item is None:
                    break
                self._set_end_of_turn_flags(base_req, False)
                base_req.text = item
                yield base_req

        try:
            call = self._service.stub.SynthesizeOnline(
                request_generator(),
                metadata=self._service.auth.get_auth_metadata(),
            )
            state.rpc_call = call

            for resp in call:
                if state.stop_event.is_set():
                    break
                asyncio.run_coroutine_threadsafe(state.response_queue.put(resp), event_loop)
        except Exception as e:
            if not state.stop_event.is_set():
                logger.error(f"{self} gRPC synthesis stream error: {e}")
                asyncio.run_coroutine_threadsafe(state.response_queue.put(e), event_loop)
        finally:
            state.rpc_call = None
            asyncio.run_coroutine_threadsafe(state.response_queue.put(None), event_loop)

    async def flush_audio(self, context_id: str | None = None):
        """Flush Magpie with end-of-turn flags, then close the synthesis stream.

        Called from Pipecat on ``LLMFullResponseEndFrame``. Sends one empty
        ``SynthesizeOnline`` request with ``is_last_request`` / ``riva_end_stream``
        so Magpie can return ``meta.words``, then ends the client stream.
        """
        state = self._stream_state
        if state is not None:
            state.text_queue.put(_END_OF_TURN)
        # Skip parent NvidiaTTSService.flush_audio (it only queues ``None``).
        await TTSService.flush_audio(self, context_id)

    async def on_audio_context_interrupted(self, context_id: str):
        """Clear word-timing state when playback is interrupted."""
        self._clear_word_state(context_id)
        await super().on_audio_context_interrupted(context_id)

    async def on_audio_context_completed(self, context_id: str):
        """Clear word-timing state when a context finishes cleanly."""
        if not self._meta_seen and not self._warned_missing_meta:
            self._warned_missing_meta = True
            logger.debug(
                f"{self}: Magpie gRPC meta empty; spoken slots rely on "
                f"force_complete ({self._text_aggregation_mode.value}) until "
                "meta.words is available"
            )
        self._clear_word_state(context_id)
        await super().on_audio_context_completed(context_id)

    async def _apply_force_complete(self):
        """Force-complete slots; allow remainder only when Magpie meta was missing."""
        seq = self._aggregated_frame_sequencer
        if isinstance(seq, _MagpieWordCommitSequencer):
            seq.commit_unspoken_remainder = not self._turn_meta_seen
        await super()._apply_force_complete()

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """Chunk text without stripping whitespace (TOKEN / stitched pacing)."""
        if text == "":
            return []
        max_len = self._MAX_CHUNK_LEN
        if len(text) <= max_len:
            return [text]
        return [text[i : i + max_len] for i in range(0, len(text), max_len)]

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send every LLM token to Magpie as-is (including spaces / punctuation).

        Parent ``NvidiaTTSService`` strips text and drops whitespace/punct-only
        chunks — appropriate for sentence aggregation on older Magpie builds.
        WordTTS targets Magpie RC3 with TOKEN streaming: every LLM piece must
        go through unchanged (spaces and punctuation-only tokens included).
        """
        if text == "":
            return

        logger.trace(f"{self}: TTS text -> Magpie: {text!r}")

        try:
            assert self._service is not None, "TTS service not initialized"
            if self._settings.synthesis_mode == NvidiaTTSSynthesisMode.PER_SENTENCE:
                # Unary per-sentence path does not use ``_process_responses``;
                # word timestamps require STITCHED streaming.
                async for frame in super()._run_tts_per_sentence(text, context_id):
                    yield frame
                return

            if not self.audio_context_available(context_id):
                await self.create_audio_context(context_id)
                await self.start_ttfb_metrics()
                yield TTSStartedFrame(context_id=context_id)
                self._start_synthesis_stream(context_id)
                self._turn_meta_seen = False
                logger.trace(f"{self}: Started synthesis stream for context {context_id}")

            state = self._stream_state
            if state is None:
                raise RuntimeError("Synthesis stream not started")

            for chunk in self._split_text_into_chunks(text):
                if chunk != "":
                    state.text_queue.put(chunk)

            await self.start_tts_usage_metrics(text)
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"{self} error: {e}")

    async def _process_responses(self, state: _SynthesisStreamState):
        """Consume Magpie audio and emit ``meta.words`` timings when present.

        ``TTSStoppedFrame`` is appended only when the ``SynthesizeOnline`` stream
        ends (after turn flush), so bot-stopped happens after the full LLM turn
        has been synthesized — not on mid-turn audio-queue idle timeouts.
        """
        while True:
            item = await state.response_queue.get()
            if item is None:
                if self.audio_context_available(state.context_id):
                    await self.append_to_audio_context(
                        state.context_id,
                        TTSStoppedFrame(context_id=state.context_id),
                    )
                    await self.remove_audio_context(state.context_id)
                break
            if isinstance(item, Exception):
                if self._stream_state is state and not state.stop_event.is_set():
                    await self.push_error(f"{self} synthesis error: {item}")
                break
            if self._stream_state is not state:
                continue

            await self.stop_ttfb_metrics()
            audio = getattr(item, "audio", b"") or b""
            if audio:
                frame = TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=state.context_id,
                )
                await self.append_to_audio_context(state.context_id, frame)

            await self._maybe_emit_meta_timestamps(item, state.context_id)

        if self._stream_state is state and not state.stop_event.is_set():
            self._stream_state = None

    async def _maybe_emit_meta_timestamps(self, response: Any, context_id: str) -> None:
        """Commit Magpie/Riva word timings into the spoken-slot sequencer."""
        if not hasattr(response, "HasField") or not response.HasField("meta"):
            return

        meta = response.meta
        state = self._word_state(context_id)
        words = list(getattr(meta, "words", []) or [])

        if words:
            # Primary contract (riva-speech WordTiming / meta.words).
            word_times, _next_t, total = word_times_from_meta_words(
                words,
                skip_tokens=state.emitted_tokens,
            )
        else:
            # Legacy / transitional meta: processed_text + predicted_durations.
            processed = (
                getattr(meta, "processed_text", "") or getattr(meta, "text", "") or ""
            ).strip()
            durations = list(getattr(meta, "predicted_durations", []) or [])
            if not processed or not durations:
                return
            word_times, _next_t, total = word_times_from_magpie_meta(
                processed,
                durations,
                skip_tokens=state.emitted_tokens,
            )

        if not word_times:
            return

        if not self._meta_seen:
            self._meta_seen = True
            logger.debug(f"{self}: Magpie gRPC meta word timings available")
        self._turn_meta_seen = True

        state.emitted_tokens = total
        # Interim: inject spaces between Magpie tokens (meta strips leading spaces).
        sample = [(w, round(t, 3)) for w, t in word_times[:12]]
        logger.debug(
            f"{self}: Magpie meta words n={len(word_times)} "
            f"sample={sample}{'…' if len(word_times) > 12 else ''} "
            f"(commit Magpie words, ifs=False, insert spaces)"
        )
        await self.add_word_timestamps(
            word_times,
            context_id,
            includes_inter_frame_spaces=False,
        )


def _split_processed_tokens(processed_text: str) -> list[str]:
    return [tok for tok in processed_text.split() if tok]


def _looks_like_frame_durations(durations: Sequence[float], *, sample_hint_hz: float = 80.0) -> bool:
    if not durations:
        return False
    total = float(sum(durations))
    if total > 60.0 and all(d >= 1.0 for d in durations):
        return True
    intish = sum(1 for d in durations if abs(d - round(d)) < 1e-3)
    return intish >= max(1, int(0.8 * len(durations))) and total > sample_hint_hz


def normalize_durations_to_seconds(
    durations: Sequence[float],
    *,
    frame_rate_hz: float = 80.0,
) -> list[float]:
    """Convert Magpie predicted durations to seconds when they look like frame counts."""
    values = [float(d) for d in durations]
    if _looks_like_frame_durations(values):
        return [d / frame_rate_hz for d in values]
    return values


def word_times_from_magpie_meta(
    processed_text: str,
    predicted_durations: Sequence[float],
    *,
    start_time: float = 0.0,
    skip_tokens: int = 0,
    frame_rate_hz: float = 80.0,
) -> tuple[list[tuple[str, float]], float, int]:
    """Build Pipecat ``(word, start_s)`` pairs from legacy Magpie meta fields."""
    text = (processed_text or "").strip()
    if not text or not predicted_durations:
        return [], start_time, 0

    durations = normalize_durations_to_seconds(predicted_durations, frame_rate_hz=frame_rate_hz)
    tokens = _split_processed_tokens(text)

    if len(durations) == len(text) and len(tokens) != len(durations):
        return _word_times_from_char_durations(
            text,
            durations,
            start_time=start_time,
            skip_tokens=skip_tokens,
        )

    count = min(len(tokens), len(durations))
    tokens = tokens[:count]
    durations = durations[:count]

    word_times: list[tuple[str, float]] = []
    t = start_time
    for index, (token, dur) in enumerate(zip(tokens, durations, strict=True)):
        if index < skip_tokens:
            t += dur
            continue
        word_times.append((token, t))
        t += max(0.0, dur)

    return word_times, t, len(tokens)


def _word_times_from_char_durations(
    text: str,
    durations: Sequence[float],
    *,
    start_time: float,
    skip_tokens: int,
) -> tuple[list[tuple[str, float]], float, int]:
    pieces = _TOKEN_RE.findall(text)
    word_times: list[tuple[str, float]] = []
    t = start_time
    token_index = 0
    cursor = 0

    for piece in pieces:
        piece_len = len(piece)
        piece_dur = float(sum(durations[cursor : cursor + piece_len]))
        cursor += piece_len
        if piece.isspace():
            t += max(0.0, piece_dur)
            continue
        if token_index >= skip_tokens:
            word_times.append((piece, t))
        t += max(0.0, piece_dur)
        token_index += 1

    return word_times, t, token_index


def word_times_from_meta_words(
    words: Sequence[Any],
    *,
    skip_tokens: int = 0,
) -> tuple[list[tuple[str, float]], float, int]:
    """Build Pipecat word times from Riva ``meta.words`` (``WordTiming``).

    Each entry is a proto message or mapping with ``word`` and
    ``start_time`` / ``end_time`` in milliseconds.
    """
    if not words:
        return [], 0.0, 0

    word_times: list[tuple[str, float]] = []
    next_t = 0.0
    total = 0
    for index, entry in enumerate(words):
        if isinstance(entry, Mapping):
            token = str(entry.get("word", "") or "")
            start_ms = float(entry.get("start_time", 0) or 0)
            end_ms = float(entry.get("end_time", start_ms) or start_ms)
        else:
            token = str(getattr(entry, "word", "") or "")
            start_ms = float(getattr(entry, "start_time", 0) or 0)
            end_ms = float(getattr(entry, "end_time", start_ms) or start_ms)
        if not token:
            continue
        start_s = max(0.0, start_ms / 1000.0)
        next_t = max(next_t, end_ms / 1000.0)
        if index >= skip_tokens:
            word_times.append((token, start_s))
        total += 1
    return word_times, next_t, total
