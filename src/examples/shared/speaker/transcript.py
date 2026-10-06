# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn speaker-tagged ASR results into per-speaker turns."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame

from utils import parse_env_bool

# Halves of one word split across two ASR finals arrive less than 1 s apart.
WORD_CONTINUATION_MAX_GAP_S = 1.0
_CUT_WORD_PUNCTUATION = re.compile(r"[.,?!;:]+$")


def extract_speaker_id(result) -> int | None:
    """Return the dominant speaker_tag from a Riva ASR result, or None if unavailable."""
    try:
        words = result.alternatives[0].words
        if not words:
            return None
        tags = [w.speaker_tag for w in words]
        return Counter(tags).most_common(1)[0][0]
    except (IndexError, AttributeError, TypeError):
        return None


def split_transcript_by_speaker(result, fallback_text: str = "") -> list[tuple[int | None, str]]:
    """Split an ASR result into speaker runs, falling back to transcript text."""
    try:
        words = result.alternatives[0].words
    except (IndexError, AttributeError, TypeError):
        words = []

    if not words:
        text = fallback_text.strip()
        if not text:
            return []
        return [(extract_speaker_id(result), text)]

    runs: list[tuple[int | None, list[str]]] = []
    for word in words:
        text = getattr(word, "word", None) or getattr(word, "text", None)
        if not text:
            continue
        tag = getattr(word, "speaker_tag", None)
        if runs and (tag is None or tag == runs[-1][0]):
            runs[-1][1].append(str(text))
        else:
            runs.append((tag, [str(text)]))

    out = [(speaker_id, " ".join(parts).strip()) for speaker_id, parts in runs if parts]
    if not out and fallback_text.strip():
        return [(extract_speaker_id(result), fallback_text.strip())]
    return out


def last_word_end_s(result) -> float | None:
    """Return when the result's last word ends, in ASR stream seconds."""
    try:
        words = result.alternatives[0].words
        return words[-1].end_time / 1000.0 if words else None
    except (IndexError, AttributeError, TypeError):
        return None


def continues_previous_word(result, previous_end_s: float | None) -> bool:
    """A final with no leading space continues the previous word when the halves are at most 1 s apart."""
    if previous_end_s is None:
        return False
    try:
        alternative = result.alternatives[0]
        transcript = alternative.transcript
        words = alternative.words
    except (IndexError, AttributeError, TypeError):
        return False
    if not transcript or transcript[0].isspace() or not words:
        return False
    gap = words[0].start_time / 1000.0 - previous_end_s
    return 0.0 <= gap <= WORD_CONTINUATION_MAX_GAP_S


def reassign_first_word(runs: list[tuple[int | None, str]], speaker_id: int | None) -> list[tuple[int | None, str]]:
    """Give the first word of a result to ``speaker_id``; later words keep their own speakers."""
    if not runs:
        return runs
    tag, text = runs[0]
    first, _, rest = text.partition(" ")
    reassigned = [(speaker_id, first)]
    if rest:
        reassigned.append((tag, rest))
    reassigned.extend(runs[1:])
    merged: list[tuple[int | None, str]] = []
    for speaker, words in reassigned:
        if merged and merged[-1][0] == speaker:
            merged[-1] = (speaker, f"{merged[-1][1]} {words}")
        else:
            merged.append((speaker, words))
    return merged


def join_cut_word(previous: str, piece: str) -> str:
    """Join the second half of a cut word to the first, dropping punctuation the ASR added at the cut."""
    return _CUT_WORD_PUNCTUATION.sub("", previous.rstrip()) + piece


def extract_word_span(result) -> tuple[float, float] | None:
    """Return the result's Riva word-time span in seconds."""
    try:
        words = result.alternatives[0].words
        if not words:
            return None
        start = min(w.start_time for w in words) / 1000.0
        end = max(w.end_time for w in words) / 1000.0
    except (IndexError, AttributeError, TypeError, ValueError):
        return None
    if end <= start:
        return None
    return start, end


@dataclass
class TranscriptFragment:
    """One final ASR result inside a user turn, with who said it and when."""

    speaker_id: int | None
    text: str
    start_s: float | None = None
    end_s: float | None = None


@dataclass
class SpeakerRun:
    """Consecutive fragments of a turn that the ASR attributed to one speaker."""

    speaker_id: int | None
    text: str
    start_s: float | None = None
    end_s: float | None = None


def group_fragments_by_speaker(fragments: Sequence[TranscriptFragment]) -> list[SpeakerRun]:
    """Collapse fragments into runs, keeping untagged audio out of a tagged speaker's run."""
    runs: list[SpeakerRun] = []
    for frag in fragments:
        text = frag.text.strip()
        if not text:
            continue
        extend = runs and frag.speaker_id == runs[-1].speaker_id
        if extend:
            run = runs[-1]
            run.text = f"{run.text} {text}"
            if frag.start_s is not None:
                run.start_s = frag.start_s if run.start_s is None else min(run.start_s, frag.start_s)
            if frag.end_s is not None:
                run.end_s = frag.end_s if run.end_s is None else max(run.end_s, frag.end_s)
        else:
            runs.append(SpeakerRun(frag.speaker_id, text, frag.start_s, frag.end_s))
    return runs


def finalize_speaker_runs(
    runs: Sequence[SpeakerRun],
    speaker_id: int | None,
    transcript: str,
) -> list[SpeakerRun]:
    """Use aggregator text for one run and fragment boundaries for many runs."""
    if not runs:
        text = transcript.strip()
        return [SpeakerRun(speaker_id, text)] if text else []
    if len(runs) == 1:
        run = runs[0]
        return [
            SpeakerRun(
                run.speaker_id if run.speaker_id is not None else speaker_id,
                transcript.strip() or run.text,
                run.start_s,
                run.end_s,
            )
        ]
    return list(runs)


class SpeakerDiarizationProcessor(FrameProcessor):
    """Latch one speaker or label all speaker runs for LLM context."""

    def __init__(
        self,
        *,
        multi_speaker_support: bool,
        incremental_handler: Callable[[list[SpeakerRun], int, str], Awaitable[None]] | None = None,
        interim_handler: Callable[[str, str], Awaitable[None]] | None = None,
        **kwargs,
    ):
        """Configure single-speaker isolation or multi-speaker attribution."""
        super().__init__(**kwargs)
        self.multi_speaker_support = multi_speaker_support
        self.incremental_handler = incremental_handler
        self.interim_handler = interim_handler
        self.turn_sequence = 0
        self.primary_speaker_id: int | None = None
        self.last_speaker_id: int | None = None
        self.last_final_speaker_id: int | None = None
        self.has_final_text = False
        self.turn_fragments: list[TranscriptFragment] = []
        self._previous_final_end_s: float | None = None
        self._previous_final_speaker_id: int | None = None
        self._interim_text = ""
        self._speaker_numbers: dict[int, int] = {}

    def take_turn_fragments(self) -> list[TranscriptFragment]:
        """Return the fragments recorded for the current turn and clear them, with the cut-word state."""
        fragments = self.turn_fragments
        self.turn_fragments = []
        self._previous_final_end_s = None
        self._previous_final_speaker_id = None
        return fragments

    def _start_transcript_turn(self) -> None:
        """Start a turn when its first accepted final transcript arrives."""
        self.turn_sequence += 1
        self.last_speaker_id = None
        self.last_final_speaker_id = None
        self.has_final_text = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Route tagged final transcripts and pass unrelated frames through."""
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.primary_speaker_id = None
            self.last_speaker_id = None
            self.last_final_speaker_id = None
            self.has_final_text = False
            self.turn_fragments = []
            self._previous_final_end_s = None
            self._previous_final_speaker_id = None
            self._interim_text = ""
            self._speaker_numbers = {}
        elif isinstance(frame, UserStartedSpeakingFrame) and not self.turn_fragments:
            # Transcription-based turn start is broadcast upstream after the
            # first final passes this processor. Do not erase that final.
            self.last_speaker_id = None
            self.last_final_speaker_id = None
            self.has_final_text = False
        if isinstance(frame, (InterimTranscriptionFrame, TranscriptionFrame)) and frame.result is not None:
            is_final = isinstance(frame, TranscriptionFrame)
            await self._publish_interim("" if is_final else frame.text.strip(), frame.timestamp)
            mapped_speaker = getattr(frame, "nvidia_speaker_tag", None)
            runs = (
                [(mapped_speaker, frame.text)]
                if mapped_speaker is not None
                else self._number_speakers(split_transcript_by_speaker(frame.result, frame.text))
            )
            if not runs:
                await self.push_frame(frame, direction)
                return

            continues_word = False
            continues_speaker: int | None = None
            if is_final and mapped_speaker is None:
                continues_word = continues_previous_word(frame.result, self._previous_final_end_s)
                if continues_word:
                    runs = reassign_first_word(runs, self._previous_final_speaker_id)
                continues_speaker = self._previous_final_speaker_id
                self._previous_final_end_s = last_word_end_s(frame.result)
                self._previous_final_speaker_id = runs[-1][0]

            if is_final and self.primary_speaker_id is None:
                self.primary_speaker_id = next((speaker_id for speaker_id, _ in runs if speaker_id is not None), None)

            if not self.multi_speaker_support and self.primary_speaker_id is not None:
                # Untagged interims still reach Pipecat's turn stop, which needs them to see ongoing speech.
                runs = [
                    (speaker_id, text)
                    for speaker_id, text in runs
                    if speaker_id == self.primary_speaker_id or (speaker_id is None and not is_final)
                ]
                if not runs:
                    return

            # The joined piece survives the single-speaker filter only when the previous word was kept.
            glue_first = continues_word and runs[0][0] == continues_speaker
            if is_final:
                if not self.turn_fragments:
                    self._start_transcript_turn()
                span = extract_word_span(frame.result)
                fragment_runs = runs
                if glue_first and self.turn_fragments and self.turn_fragments[-1].speaker_id == runs[0][0]:
                    previous = self.turn_fragments[-1]
                    piece, _, rest = runs[0][1].partition(" ")
                    previous.text = join_cut_word(previous.text, piece)
                    if span:
                        previous.end_s = span[1] if previous.end_s is None else max(previous.end_s, span[1])
                    fragment_runs = ([(runs[0][0], rest)] if rest else []) + runs[1:]
                for speaker_id, text in fragment_runs:
                    if speaker_id is not None:
                        self.last_speaker_id = speaker_id
                    self.turn_fragments.append(
                        TranscriptFragment(
                            speaker_id=speaker_id,
                            text=text,
                            start_s=span[0] if span else None,
                            end_s=span[1] if span else None,
                        )
                    )
                if self.incremental_handler:
                    await self.incremental_handler(
                        group_fragments_by_speaker(self.turn_fragments),
                        self.turn_sequence,
                        frame.timestamp,
                    )

            if self.multi_speaker_support and mapped_speaker is None and len(runs) > 1:
                for run_index, (speaker_id, text) in enumerate(runs):
                    copied = _copy_transcription_frame(frame, text=text)
                    copied.nvidia_speaker_tag = speaker_id
                    self._prefix_final_for_llm(copied, speaker_id)
                    _set_llm_spacing(copied, glued=glue_first and run_index == 0)
                    await self.push_frame(copied, direction)
                return
            frame.text = " ".join(text for _, text in runs)
            speaker_id = runs[0][0] if len(runs) == 1 else mapped_speaker
            self._prefix_final_for_llm(frame, speaker_id)
            _set_llm_spacing(frame, glued=glue_first)
        await self.push_frame(frame, direction)

    def _number_speakers(self, runs: list[tuple[int | None, str]]) -> list[tuple[int | None, str]]:
        """Number ASR tags by first appearance, since some ASRs count speakers from 0 and others from 1."""
        return [
            (None if tag is None else self._speaker_numbers.setdefault(tag, len(self._speaker_numbers)), text)
            for tag, text in runs
        ]

    async def _publish_interim(self, text: str, timestamp: str) -> None:
        """Send the in-progress text once per change; a final clears it."""
        if not self.interim_handler or text == self._interim_text:
            return
        self._interim_text = text
        await self.interim_handler(text, timestamp)

    def _prefix_final_for_llm(self, frame, speaker_id: int | None) -> None:
        if not self.multi_speaker_support or not isinstance(frame, TranscriptionFrame):
            return
        starts_run = not self.has_final_text or (speaker_id is not None and speaker_id != self.last_final_speaker_id)
        if starts_run and speaker_id is not None:
            frame.text = f"Speaker {speaker_id + 1}: {frame.text}"
        self.has_final_text = True
        if speaker_id is not None:
            self.last_final_speaker_id = speaker_id


def _set_llm_spacing(frame, *, glued: bool) -> None:
    """Carry the word boundary into the user aggregator, which otherwise puts a space between finals."""
    if not isinstance(frame, TranscriptionFrame):
        return
    text = frame.text.lstrip()
    frame.text = text if glued else f" {text}"
    frame.includes_inter_frame_spaces = True


def _copy_transcription_frame(
    frame: InterimTranscriptionFrame | TranscriptionFrame,
    *,
    text: str,
) -> InterimTranscriptionFrame | TranscriptionFrame:
    """Copy a transcript frame so LLM-only text cannot leak back into RTVI."""
    common = {
        "text": text,
        "user_id": frame.user_id,
        "timestamp": frame.timestamp,
        "language": frame.language,
        "result": frame.result,
    }
    if isinstance(frame, TranscriptionFrame):
        copied = TranscriptionFrame(**common, finalized=frame.finalized)
    else:
        copied = InterimTranscriptionFrame(**common)
    copied.nvidia_speaker_tag = getattr(frame, "nvidia_speaker_tag", None)
    return copied


def progress_payloads(runs: list[SpeakerRun], turn_id: int, timestamp: str) -> list[dict[str, Any]]:
    """Build the live speaker snapshots sent before a user turn ends."""
    return [
        {
            "type": "user-turn-progress",
            "timestamp": timestamp,
            "transcript": run.text,
            "speaker_labeled": True,
            "turn_id": turn_id,
            "run_index": run_index,
            **_speaker_fields(run.speaker_id),
        }
        for run_index, run in enumerate(runs)
    ]


def interim_payload(text: str, timestamp: str) -> dict[str, Any]:
    """Build the in-progress text shown before the ASR assigns a speaker; empty text clears it."""
    return {
        "type": "user-turn-identifying",
        "timestamp": timestamp,
        "transcript": text,
    }


def finalized_turn_payloads(processor: SpeakerDiarizationProcessor | None, message) -> list[dict[str, Any]]:
    """Build the ``user-turn-finalized`` payloads for one finished user turn.

    Without a speaker processor this is the single plain payload. With one, it is one
    speaker-labeled payload per speaker run.
    """
    if processor is None:
        return [_final_payload(message, getattr(message, "content", None))]

    transcript = getattr(message, "content", None) or ""
    fragments = processor.take_turn_fragments()
    # A close with no fragments of its own is an empty follow-up turn. Its
    # aggregator text must not reuse the turn_id that was just emitted.
    if not fragments:
        processor.turn_sequence += 1
    runs = finalize_speaker_runs(
        group_fragments_by_speaker(fragments),
        processor.last_speaker_id,
        "" if processor.multi_speaker_support else transcript,
    )
    if not runs:
        return [_final_payload(message, transcript, speaker_labeled=True, turn_id=processor.turn_sequence, run_index=0)]

    return [
        _final_payload(
            message,
            run.text,
            speaker_labeled=True,
            turn_id=processor.turn_sequence,
            run_index=run_index,
            **_speaker_fields(run.speaker_id),
        )
        for run_index, run in enumerate(runs)
    ]


async def send_rtvi_payloads(task, payloads: list[dict[str, Any]]) -> None:
    """Queue each payload on the pipeline task as an RTVI server message for the browser."""
    for payload in payloads:
        await task.queue_frame(RTVIServerMessageFrame(data=payload))


def forward_speaker_events(processor: SpeakerDiarizationProcessor, task) -> None:
    """Send the processor's live speaker runs and in-progress text to the browser."""

    async def on_runs(runs: list[SpeakerRun], turn_id: int, timestamp: str) -> None:
        await send_rtvi_payloads(task, progress_payloads(runs, turn_id, timestamp))

    async def on_interim(text: str, timestamp: str) -> None:
        await send_rtvi_payloads(task, [interim_payload(text, timestamp)])

    processor.incremental_handler = on_runs
    processor.interim_handler = on_interim


def speaker_diarization_requested(body: dict) -> bool:
    """Return whether this session asked for diarization, defaulting to ``ENABLE_SPEAKER_DIARIZATION``."""
    requested = body.get("asr_speaker_diarization")
    if requested is None:
        return parse_env_bool("ENABLE_SPEAKER_DIARIZATION")
    return requested is True or str(requested).strip().lower() == "true"


def resolve_asr_speaker_support(body: dict, default_asr: dict | None = None) -> tuple[bool, int | None]:
    """Return whether the session's ASR supports speaker diarization, and its speaker cap.

    A session that names no ASR (no ``asr_id``) uses the catalog default, so ``default_asr`` supplies both
    values. Otherwise the session body carries the values hydrated from the catalog entry it chose, and a
    custom ASR carries none.
    """
    if body.get("asr_id"):
        supported = body.get("asr_speaker_diarization_supported", False)
        max_speakers = body.get("asr_speaker_diarization_max_speakers")
    else:
        supported = (default_asr or {}).get("speaker_diarization_supported", False)
        max_speakers = (default_asr or {}).get("speaker_diarization_max_speakers")
    diarization_supported = supported if isinstance(supported, bool) else str(supported).strip().lower() == "true"
    try:
        speaker_cap = int(max_speakers) if max_speakers else None
    except (TypeError, ValueError):
        speaker_cap = None
    return diarization_supported, speaker_cap


def speaker_diarization_enabled(
    body: dict, *, default_asr: dict | None = None, asr_model: str = "", asr_server: str = ""
) -> bool:
    """Return whether this session asked for diarization on an ASR that supports it."""
    diarization_requested = speaker_diarization_requested(body)
    diarization_supported, _ = resolve_asr_speaker_support(body, default_asr)
    if diarization_requested and not diarization_supported:
        logger.warning(f"ASR {asr_model or asr_server!r} does not support speaker diarization; ignoring request")
    return diarization_requested and diarization_supported


def _speaker_fields(speaker_id: int | None) -> dict[str, Any]:
    return {
        "speaker_id": speaker_id,
        "speaker_display_name": f"Speaker {speaker_id + 1}" if speaker_id is not None else "Unknown",
    }


def _final_payload(message, transcript: str | None, **speaker_extras: Any) -> dict[str, Any]:
    return {
        "type": "user-turn-finalized",
        "timestamp": getattr(message, "timestamp", None),
        "transcript": transcript,
        "user_id": getattr(message, "user_id", None),
        **speaker_extras,
    }
