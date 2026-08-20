# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""NVIDIA Magpie WordTTS service.

Compatibility:
    Word-level input streaming and word timestamps require Magpie TTS
    Multilingual NIM 1.10.0 or newer.

Local draft of a Pipecat child class: does **not** modify upstream
:class:`~pipecat.services.nvidia.tts.NvidiaTTSService`. Later, move this module
into ``pipecat.services.nvidia.tts`` alongside the parent.

Child of :class:`NvidiaTTSService` for the same spoken-word context commit path
used by Cartesia / ElevenLabs / Rime.

* ``push_text_frames=False`` — commits come from ``add_word_timestamps`` only.
  Unspoken remainder is never force-completed (interrupt before any timestamps
  commits nothing).
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

Proto contract (Magpie TTS Multilingual NIM 1.10.0+)::

    request.enable_word_time_offsets = true
    request.custom_configuration["max_chunk_threshold"] = "100"
    # once per LLM response, empty final SynthesizeOnline message:
    request.custom_configuration["riva_end_stream"] = "true"
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
)
from pipecat.services.nvidia.tts import (
    NvidiaTTSService,
    NvidiaTTSSettings,
    NvidiaTTSSynthesisMode,
    _SynthesisStreamState,
)
from pipecat.services.settings import _NotGiven
from pipecat.services.tts_service import TextAggregationMode, TTSService
from pipecat.utils.context.aggregated_frame_sequencer import AggregatedFrameSequencer
from pipecat.utils.tracing.service_decorators import traced_tts

try:
    import riva.client.proto.riva_tts_pb2 as rtts
except ModuleNotFoundError as e:  # pragma: no cover
    raise ImportError(f"Missing module: {e}") from e

_TOKEN_RE = re.compile(r"\S+|\s+")
_ENABLE_WORD_TIME_OFFSETS_KEY = "enable_word_time_offsets"
# Magpie streaming flush: force a chunk (and timestamp segment) when EOS is
# not seen by this many buffered characters.
# https://docs.nvidia.com/nim/speech/latest/tts/customization.html#streaming-text-flush-controls
_MAX_CHUNK_THRESHOLD_KEY = "max_chunk_threshold"
_MAX_CHUNK_THRESHOLD_VALUE = "100"
# Empty final SynthesizeOnline message; Riva maps this to the backend last-request flag.
_RIVA_END_STREAM_KEY = "riva_end_stream"
_END_OF_TURN = object()
# Treat a new batch as per-sentence relative if its first start is at/near 0.
_RELATIVE_RESTART_EPS_S = 0.02


class _MagpieWordCommitSequencer(AggregatedFrameSequencer):
    """Commit Magpie ``meta.words`` with inter-frame space injection.

    WordTTS path (interim until Magpie preserves input token spacing):
    * ``includes_inter_frame_spaces=False`` — insert a space between Magpie
      tokens (``I``+``am`` → ``I am``). Subword splits may still show gaps
      (``Nem``+``otron`` → ``Nem otron``).
    * Do **not** rewrite commits to LLM token spans — those bypass Magpie PTS
      alignment and can dump unplayed text into context on barge-in.
    * ``force_complete`` never emits unspoken remainder. Commits are timed
      words only; barge-in before any ``meta.words`` commits nothing.
    """

    def process_word(
        self,
        word: str,
        pts: int,
        context_id: str | None,
        includes_inter_frame_spaces: bool = False,
    ) -> list[Frame]:
        """Commit Magpie ``word`` text with IFS=False (inject spaces)."""
        return super().process_word(word, pts, context_id, False)

    def force_complete(self, context_id: str, last_word_pts: int) -> list[Frame]:
        """Finish a context's slots without committing unspoken remainder."""
        for slot in self._slots:
            if slot.spoken and not slot.complete and slot.context_id == context_id:
                slot.complete = True
        frames = self.flush(last_word_pts=last_word_pts)
        self._context_append_to_context.pop(context_id, None)
        self._streaming_contexts.pop(context_id, None)
        return frames


@dataclass
class NvidiaWordTTSSettings(NvidiaTTSSettings):
    """Settings for :class:`NvidiaWordTTSService`.

    Defaults to Magpie ``stitched`` streaming (required for the shared
    ``SynthesizeOnline`` response path that surfaces ``meta.words``).
    """

    synthesis_mode: NvidiaTTSSynthesisMode | _NotGiven = field(default_factory=lambda: NvidiaTTSSynthesisMode.STITCHED)


@dataclass(frozen=True)
class TimedWord:
    """One Magpie-timed token in stream-absolute seconds."""

    word: str
    start_s: float
    end_s: float


@dataclass
class _WordTimingState:
    """Per-audio-context bookkeeping for incremental Magpie meta."""

    accepted: list[TimedWord] = field(default_factory=list)


class NvidiaWordTTSService(NvidiaTTSService):
    """NVIDIA TTS WordTTS path: spoken commits + Magpie/Riva word timestamps.

    .. note::
       Word-level input streaming and word timestamps require Magpie TTS
       Multilingual NIM 1.10.0 or newer.

    Use this instead of :class:`NvidiaTTSService` when you need barge-in-accurate
    assistant context (heard words only). Unspoken remainder is never
    force-completed: interrupt before any timestamps commits nothing.
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
        word_settings = NvidiaWordTTSSettings(synthesis_mode=NvidiaTTSSynthesisMode.STITCHED)
        if kwargs.get("settings") is not None:
            word_settings.apply_update(kwargs["settings"])
        kwargs["settings"] = word_settings

        super().__init__(**kwargs)

        # Parent hardcodes push_stop_frames=True with a 3s audio-queue idle timeout,
        # which can fire mid-turn while Magpie waits for the next LLM token. Disable
        # that and emit TTSStoppedFrame only when SynthesizeOnline actually ends.
        self._push_stop_frames = False

        # Magpie meta tokens have no leading spaces; IFS=False inserts them.
        # Stock NvidiaTTSService keeps the default AggregatedFrameSequencer.
        self._aggregated_frame_sequencer = _MagpieWordCommitSequencer(
            name=str(self),
            streaming=self._text_aggregation_mode == TextAggregationMode.TOKEN,
        )

        cfg = dict(custom_configuration or {})
        if enable_word_time_offsets:
            cfg.setdefault(_ENABLE_WORD_TIME_OFFSETS_KEY, "true")
        cfg.setdefault(_MAX_CHUNK_THRESHOLD_KEY, _MAX_CHUNK_THRESHOLD_VALUE)
        self._enable_word_time_offsets = enable_word_time_offsets
        self._custom_configuration = cfg
        self._word_states: dict[str, _WordTimingState] = {}
        self._meta_seen = False
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
        """Set or clear the once-per-LLM-response Magpie ``riva_end_stream`` flag."""
        if enabled:
            req.custom_configuration[_RIVA_END_STREAM_KEY] = "true"
        else:
            req.custom_configuration.pop(_RIVA_END_STREAM_KEY, None)

    def _synthesis_handler(self, state: _SynthesisStreamState):
        """SynthesizeOnline with an explicit end-of-turn flush before stream close.

        Parent closes the client stream with ``None`` only. Magpie word timings are
        emitted on an empty final request that sets ``riva_end_stream`` (once per
        LLM response). The flag is cleared afterward so the next turn starts clean.
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
                    logger.debug(f"{self}: Magpie end-of-turn flush (riva_end_stream)")
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
        ``SynthesizeOnline`` request with ``riva_end_stream`` so Magpie can return
        ``meta.words``, then ends the client stream.
        """
        state = self._stream_state
        if state is not None and (context_id is None or state.context_id == context_id):
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
                f"{self}: Magpie gRPC meta empty this turn; no timed commits "
                f"(aggregation={self._text_aggregation_mode.value})"
            )
        self._clear_word_state(context_id)
        await super().on_audio_context_completed(context_id)

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
        Magpie TTS Multilingual NIM 1.10.0+ supports TOKEN streaming: every
        LLM piece must go through unchanged (spaces and punctuation-only
        tokens included).
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
                logger.trace(f"{self}: Started synthesis stream for context {context_id}")

            state = self._stream_state
            if state is None:
                raise RuntimeError("Synthesis stream not started")
            if state.context_id != context_id:
                raise RuntimeError(
                    f"Synthesis stream context mismatch: active={state.context_id}, requested={context_id}"
                )

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
        try:
            while True:
                item = await state.response_queue.get()
                if item is None:
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
        finally:
            # A superseded response task must not stop/remove the newer stream's
            # context or clear its state.
            if self._stream_state is state and not state.stop_event.is_set():
                if self.audio_context_available(state.context_id):
                    await self.append_to_audio_context(
                        state.context_id,
                        TTSStoppedFrame(context_id=state.context_id),
                    )
                    if self._stream_state is state:
                        await self.remove_audio_context(state.context_id)
                if self._stream_state is state:
                    self._stream_state = None

    async def _maybe_emit_meta_timestamps(self, response: Any, context_id: str) -> None:
        """Ingest Magpie/Riva word timings and register only new words."""
        if not hasattr(response, "HasField") or not response.HasField("meta"):
            return
        if not self.audio_context_available(context_id):
            logger.debug(f"{self}: dropping late Magpie meta; context {context_id} gone")
            return

        meta = response.meta
        words = getattr(meta, "words", None)
        incoming: list[TimedWord]
        if words:
            incoming = parse_meta_word_entries(words)
        else:
            processed = (getattr(meta, "processed_text", "") or getattr(meta, "text", "") or "").strip()
            durations = list(getattr(meta, "predicted_durations", []) or [])
            if not processed or not durations:
                return
            pairs, next_t, _total = word_times_from_magpie_meta(processed, durations)
            incoming = timed_words_from_pairs(pairs, next_t)

        if not incoming:
            return

        state = self._word_state(context_id)
        new_words = new_words_from_meta_batch(incoming, state.accepted)
        if not new_words:
            return

        if not self._meta_seen:
            self._meta_seen = True
            logger.debug(f"{self}: Magpie gRPC meta word timings available")

        state.accepted.extend(new_words)
        word_times = [(w.word, w.start_s) for w in new_words]
        sample = [(w, round(t, 3)) for w, t in word_times[:12]]
        logger.debug(
            f"{self}: Magpie meta new={len(word_times)} accepted={len(state.accepted)} "
            f"sample={sample}{'…' if len(word_times) > 12 else ''} "
            f"(commit Magpie words, ifs=False, insert spaces)"
        )
        await self.add_word_timestamps(
            word_times,
            context_id,
            includes_inter_frame_spaces=False,
        )


def _meta_word_fields(entry: Any) -> tuple[str, float, float]:
    """Return ``(word, start_ms, end_ms)`` from a proto or mapping ``WordTiming``."""
    if isinstance(entry, Mapping):
        token = str(entry.get("word", "") or "")
        start_ms = float(entry.get("start_time", 0) or 0)
        end_ms = float(entry.get("end_time", start_ms) or start_ms)
    else:
        token = str(getattr(entry, "word", "") or "")
        start_ms = float(getattr(entry, "start_time", 0) or 0)
        end_ms = float(getattr(entry, "end_time", start_ms) or start_ms)
    return token, start_ms, end_ms


def parse_meta_word_entries(words: Sequence[Any]) -> list[TimedWord]:
    """Parse Riva ``meta.words`` entries into stream-local ``TimedWord`` values."""
    parsed: list[TimedWord] = []
    for entry in words:
        token, start_ms, end_ms = _meta_word_fields(entry)
        if not token:
            continue
        start_s = max(0.0, start_ms / 1000.0)
        parsed.append(TimedWord(token, start_s, max(start_s, end_ms / 1000.0)))
    return parsed


def timed_words_from_pairs(
    pairs: Sequence[tuple[str, float]],
    next_t: float,
) -> list[TimedWord]:
    """Build ``TimedWord`` rows from ``(word, start_s)`` pairs."""
    out: list[TimedWord] = []
    for index, (word, start_s) in enumerate(pairs):
        if not word:
            continue
        end_s = pairs[index + 1][1] if index + 1 < len(pairs) else max(next_t, start_s)
        out.append(TimedWord(word, start_s, max(start_s, end_s)))
    return out


def new_words_from_meta_batch(
    incoming: Sequence[TimedWord],
    already_emitted: Sequence[TimedWord],
    *,
    relative_restart_eps: float = _RELATIVE_RESTART_EPS_S,
) -> list[TimedWord]:
    """Return only new words from a Magpie timestamp batch.

    Cumulative payloads (full turn so far) yield the unmatched suffix. Because
    a single repeated token is indistinguishable from a one-token cumulative
    prefix, cumulative classification requires an exact prefix of at least two
    previously accepted ``TimedWord`` entries (word and timestamps).
    Incremental sentence payloads yield the whole batch. If an incremental
    batch restarts ``start_s`` near 0, times are offset by the previous
    sentence end so PTS stays stream-absolute.
    """
    if not incoming:
        return []

    prefix_len = len(already_emitted)
    is_cumulative = prefix_len >= 2 and list(incoming[:prefix_len]) == list(already_emitted)

    if is_cumulative:
        return list(incoming[prefix_len:])

    new_words = list(incoming)
    if already_emitted and new_words and new_words[0].start_s <= relative_restart_eps:
        offset = already_emitted[-1].end_s
        new_words = [TimedWord(item.word, item.start_s + offset, item.end_s + offset) for item in new_words]
    return new_words


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
    if _looks_like_frame_durations(values, sample_hint_hz=frame_rate_hz):
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
