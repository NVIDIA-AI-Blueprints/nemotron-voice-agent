# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-offset turn detection using the generic assistant's Pipecat models.

Upload packet boundaries and arrival times never reach this component. It consumes
one sealed PCM source, feeds Silero exactly one analysis window at a time, and
stores Smart Turn's pre-speech audio at integer sample offsets. Monotonic time is
used exclusively to measure model work, never to decide a conversational event.

The two mixins deliberately isolate the small compatibility surface with pinned
Pipecat 1.12.0. Model loading is lazy so scripted runtime tests need no model files.
"""

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Protocol

from .models import RATE, AudioSource, DetectionTrace, SpeechInterval, UserTurn


class VADPort(Protocol):
    """The subset of Pipecat's VAD analyzer needed by a finite source detector."""

    params: Any

    def set_sample_rate(self, sample_rate: int) -> None:
        """Configure the model and reset its window counters."""
        ...

    def num_frames_required(self) -> int:
        """Return the inference window size in samples."""
        ...

    async def analyze_audio(self, buffer: bytes) -> Any:
        """Return a state with Pipecat VADState's names."""
        ...

    async def cleanup(self) -> None:
        """Release session-owned inference resources."""
        ...


class TurnAnalyzerPort(Protocol):
    """The generic Smart Turn public interface, independent of ASR readiness."""

    def set_sample_rate(self, sample_rate: int) -> None:
        """Configure inference and source duration accounting."""
        ...

    def update_vad_start_secs(self, vad_start_secs: float) -> None:
        """Retain the generic pre-speech compensation for VAD confirmation."""
        ...

    def append_audio(self, buffer: bytes, is_speech: bool) -> Any:
        """Accumulate source PCM and return a possible sample-silence endpoint."""
        ...

    async def analyze_end_of_turn(self) -> tuple[Any, Any]:
        """Run the selected model on source audio through the current offset."""
        ...

    def clear(self) -> None:
        """Reset turn-local state while preserving the source cursor."""
        ...

    async def cleanup(self) -> None:
        """Release session-owned inference resources."""
        ...


class SampleTimedSileroMixin:
    """Replace Silero's periodic wall-clock reset with source sample positions."""

    def __init__(self, *args, **kwargs):
        """Initialize a source-local reset schedule before loading the model."""
        self._offline_source_end_sample = 0
        self._offline_next_reset_sample: int | None = None
        self._offline_model_reset_offsets: list[int] = []
        super().__init__(*args, **kwargs)

    @property
    def model_reset_offsets(self) -> tuple[int, ...]:
        """Return actual window boundaries at which recurrent state was reset."""
        return tuple(self._offline_model_reset_offsets)

    def set_source_window_end(self, end_sample: int) -> None:
        """Set the actual source extent, excluding any final inference padding."""
        if end_sample < self._offline_source_end_sample:
            raise ValueError("Silero source positions must be monotonic")
        self._offline_source_end_sample = end_sample

    def voice_confidence(self, buffer: bytes) -> float:
        """Run the existing model, then reset on five seconds of source audio."""
        import numpy as np

        audio = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
        confidence = float(np.asarray(self._model(audio, self.sample_rate)).reshape(-1)[0])
        period_samples = 5 * self.sample_rate
        if self._offline_next_reset_sample is None:
            self._offline_next_reset_sample = period_samples
        if self._offline_source_end_sample >= self._offline_next_reset_sample:
            self._model.reset_states()
            self._offline_model_reset_offsets.append(self._offline_source_end_sample)
            while self._offline_next_reset_sample <= self._offline_source_end_sample:
                self._offline_next_reset_sample += period_samples
        return confidence


class SampleTimedSmartTurnMixin:
    """Keep Smart Turn buffering and silence accounting at exact sample offsets."""

    def __init__(self, *args, **kwargs):
        """Initialize source and turn cursors before configuring the base analyzer."""
        self._offline_sample_cursor = 0
        self._offline_speech_start_sample: int | None = None
        self._offline_silence_samples = 0
        super().__init__(*args, **kwargs)

    def append_audio(self, buffer: bytes, is_speech: bool) -> Any:
        """Append an immutable PCM view and advance using its actual sample count."""
        import numpy as np
        from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState

        audio = np.frombuffer(buffer, dtype=np.int16)
        start_sample = self._offline_sample_cursor
        self._offline_sample_cursor += len(audio)
        self._audio_buffer.append((start_sample, audio))

        if is_speech:
            self._offline_silence_samples = 0
            self._silence_ms = 0
            self._speech_triggered = True
            if self._offline_speech_start_sample is None:
                self._offline_speech_start_sample = start_sample
        elif self._speech_triggered:
            self._offline_silence_samples += len(audio)
            self._silence_ms = self._offline_silence_samples * 1000 / self.sample_rate
            stop_samples = round(self._params.stop_secs * self.sample_rate)
            if self._offline_silence_samples >= stop_samples:
                self._clear(EndOfTurnState.COMPLETE)
                return EndOfTurnState.COMPLETE
        else:
            keep_samples = round(
                (self._params.pre_speech_ms / 1000 + self._params.stop_secs + self._params.max_duration_secs)
                * self.sample_rate
            )
            minimum_sample = max(0, self._offline_sample_cursor - keep_samples)
            self._audio_buffer = self._source_chunks_since(self._audio_buffer, minimum_sample)

        return EndOfTurnState.INCOMPLETE

    @staticmethod
    def _source_chunks_since(audio_buffer, minimum_sample: int):
        """Trim at a sample boundary, including the first chunk's partial prefix."""
        result = []
        for start_sample, audio in audio_buffer:
            cut_samples = max(0, minimum_sample - start_sample)
            if cut_samples < len(audio):
                result.append((start_sample + cut_samples, audio[cut_samples:]))
        return result

    def _process_speech_segment(self, audio_buffer) -> tuple[Any, Any]:
        """Select source audio exactly, then reuse upstream model work and metrics."""
        if self._offline_speech_start_sample is None:
            return super()._process_speech_segment([])
        pre_samples = round((self._params.pre_speech_ms / 1000 + self._vad_start_secs) * self.sample_rate)
        minimum_sample = max(0, self._offline_speech_start_sample - pre_samples)
        selected = self._source_chunks_since(audio_buffer, minimum_sample)

        # BaseSmartTurn's prediction and metrics remain unchanged. It receives
        # already selected chunks; its timestamp-based selector always starts at
        # zero, and cannot read system time to choose or trim source audio.
        selected_for_prediction = [(0.0, audio) for _, audio in selected]
        return super()._process_speech_segment(selected_for_prediction)

    def _clear(self, turn_state) -> None:
        """Clear turn-local sample counters while retaining the session position."""
        self._offline_speech_start_sample = None
        self._offline_silence_samples = 0
        super()._clear(turn_state)


@dataclass
class _PendingTurn:
    """Mutable accumulator local to one chronological detection pass."""

    start_sample: int
    end_sample: int


class AudioOffsetDetector:
    """Run one finite session with Silero and optional generic Smart Turn policy.

    Speech intervals retain the first source window in each confirmed run. Their
    confirmation offset is the end of the configured VAD start window. Turn
    endpoints include configured silence; ASR availability is owned by the
    recognizer and scheduler, not by this source-only component.
    """

    def __init__(self, *, vad_analyzer: VADPort, turn_analyzer: TurnAnalyzerPort | None = None):
        """Inject session-owned analyzers; omit Smart Turn for Silero-only mode."""
        self._vad = vad_analyzer
        self._turn = turn_analyzer
        self._used = False

    @property
    def configuration(self) -> dict:
        """Expose effective model thresholds for paired-evaluation fingerprints."""
        vad = self._vad.params
        turn = self._turn.params if self._turn is not None else None
        return {
            "policy": "smart_turn" if turn is not None else "silero",
            "vad": vad.model_dump() if hasattr(vad, "model_dump") else vars(vad),
            "smart_turn": (turn.model_dump() if hasattr(turn, "model_dump") else vars(turn)) if turn else None,
            "window_samples": 512,
            "model_reset_samples": 5 * RATE,
        }

    async def detect(self, source: AudioSource) -> DetectionTrace:
        """Return ordered source decisions and explicitly flush at source EOF."""
        if self._used:
            raise RuntimeError("Create a new detector for each offline session")
        self._used = True
        try:
            return await self._detect_source(source)
        finally:
            await self._vad.cleanup()
            if self._turn is not None:
                await self._turn.cleanup()

    async def _detect_source(self, source: AudioSource) -> DetectionTrace:
        """Consume fixed analysis windows without losing within-upload transitions."""
        self._vad.set_sample_rate(source.sample_rate)
        window_samples = self._vad.num_frames_required()
        if window_samples != 512:
            raise ValueError("Offline detection requires Silero's 512-sample 16 kHz windows")
        if self._turn is not None:
            self._turn.set_sample_rate(source.sample_rate)
            self._turn.update_vad_start_secs(self._vad.params.start_secs)

        intervals: list[SpeechInterval] = []
        turns: list[UserTurn] = []
        events: list[dict[str, Any]] = []
        pending: _PendingTurn | None = None
        speech_start: int | None = None
        confirmed_start: int | None = None
        tentative_start: int | None = None
        tentative_stop: int | None = None
        previous_state = "QUIET"
        total_vad_ns = 0
        total_turn_ns = 0

        def close_speech(end_sample: int, decision_sample: int, reason: str) -> None:
            nonlocal speech_start, confirmed_start, tentative_stop, pending
            if speech_start is None or confirmed_start is None:
                return
            end_sample = max(speech_start, end_sample)
            intervals.append(SpeechInterval(speech_start, end_sample, confirmed_start))
            if pending is not None:
                pending.end_sample = end_sample
            events.append(
                {"type": "speech_end", "sample": end_sample, "decision_sample": decision_sample, "reason": reason}
            )
            speech_start = None
            confirmed_start = None
            tentative_stop = None

        def finish_turn(endpoint: int, compute_ns: int, reason: str) -> None:
            nonlocal pending
            if pending is None:
                return
            turn_id = f"turn-{len(turns) + 1}"
            turns.append(UserTurn(turn_id, pending.start_sample, pending.end_sample, endpoint, compute_ns))
            events.append(
                {
                    "type": "turn_endpoint",
                    "turn_id": turn_id,
                    "sample": endpoint,
                    "speech_end_sample": pending.end_sample,
                    "compute_ns": compute_ns,
                    "reason": reason,
                }
            )
            pending = None
            if self._turn is not None:
                self._turn.clear()

        for start_sample in range(0, source.sample_count, window_samples):
            end_sample = min(start_sample + window_samples, source.sample_count)
            audio = source.pcm[start_sample * 2 : end_sample * 2]
            inference_audio = audio.ljust(window_samples * 2, b"\0")
            set_window_end = getattr(self._vad, "set_source_window_end", None)
            if set_window_end is not None:
                set_window_end(end_sample)
            before_vad = perf_counter_ns()
            state = (await self._vad.analyze_audio(inference_audio)).name
            vad_compute_ns = perf_counter_ns() - before_vad
            total_vad_ns += vad_compute_ns
            stopped = False

            if end_sample - start_sample < window_samples:
                # Padding lets the model inspect the final real samples, but is
                # not source duration and cannot satisfy a confirmation window.
                if state == "SPEAKING" and speech_start is None and self._vad.params.start_secs > 0:
                    candidate = tentative_start if tentative_start is not None else start_sample
                    needed = max(1, round(self._vad.params.start_secs * source.sample_rate / window_samples))
                    if end_sample < candidate + needed * window_samples:
                        state = "STARTING"
                elif state == "QUIET" and speech_start is not None and self._vad.params.stop_secs > 0:
                    candidate = tentative_stop if tentative_stop is not None else start_sample
                    needed = max(1, round(self._vad.params.stop_secs * source.sample_rate / window_samples))
                    if end_sample < candidate + needed * window_samples:
                        state = "STOPPING"

            if state == "STARTING" and previous_state != "STARTING":
                tentative_start = start_sample
            elif state == "SPEAKING":
                if speech_start is None:
                    speech_start = tentative_start if tentative_start is not None else start_sample
                    confirmed_start = end_sample
                    if pending is None:
                        pending = _PendingTurn(speech_start, end_sample)
                    events.append({"type": "speech_start", "sample": speech_start, "confirmed_sample": confirmed_start})
                tentative_stop = None
            elif state == "STOPPING" and previous_state != "STOPPING":
                tentative_stop = start_sample
            elif state == "QUIET":
                if speech_start is not None:
                    close_speech(
                        tentative_stop if tentative_stop is not None else start_sample,
                        end_sample,
                        "vad_silence",
                    )
                    stopped = True
                tentative_start = None

            if pending is not None and speech_start is not None:
                pending.end_sample = tentative_stop if tentative_stop is not None else end_sample

            if self._turn is None:
                if stopped:
                    finish_turn(end_sample, vad_compute_ns, "vad_silence")
            else:
                # Include this window before predicting so an endpoint's model
                # input is explicitly complete through that endpoint offset.
                before_turn = perf_counter_ns()
                turn_state = self._turn.append_audio(audio, speech_start is not None)
                turn_compute_ns = perf_counter_ns() - before_turn
                if turn_state.name == "COMPLETE":
                    finish_turn(end_sample, vad_compute_ns + turn_compute_ns, "smart_turn_silence")
                elif stopped:
                    before_turn = perf_counter_ns()
                    turn_state, _ = await self._turn.analyze_end_of_turn()
                    turn_compute_ns += perf_counter_ns() - before_turn
                    if turn_state.name == "COMPLETE":
                        finish_turn(end_sample, vad_compute_ns + turn_compute_ns, "smart_turn_prediction")
                total_turn_ns += turn_compute_ns

            previous_state = state

        if speech_start is not None:
            close_speech(
                tentative_stop if tentative_stop is not None else source.sample_count,
                source.sample_count,
                "source_eof",
            )
        if pending is not None:
            finish_turn(source.sample_count, 0, "source_eof")
        events.append(
            {
                "type": "detector_complete",
                "sample": source.sample_count,
                "vad_compute_ns": total_vad_ns,
                "turn_compute_ns": total_turn_ns,
                "model_reset_samples": list(getattr(self._vad, "model_reset_offsets", ())),
            }
        )
        return DetectionTrace(tuple(intervals), tuple(turns), tuple(events))


def create_generic_detector() -> AudioOffsetDetector:
    """Load the generic assistant's selected Silero/Smart Turn configuration."""
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    from examples.shared.pipeline_utils import SMART_TURN_FALLBACK_SECS
    from utils import parse_env_bool, parse_env_float

    class SourceSilero(SampleTimedSileroMixin, SileroVADAnalyzer):
        """Pinned model with source-driven recurrent resets."""

    class SourceSmartTurn(SampleTimedSmartTurnMixin, LocalSmartTurnAnalyzerV3):
        """Pinned endpoint model with sample-indexed buffering."""

    silero_only = parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False)
    vad_stop_secs = parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0) if silero_only else 0.2
    vad = SourceSilero(sample_rate=RATE, params=VADParams(stop_secs=vad_stop_secs))
    turn_stop_secs = parse_env_float("SMART_TURN_STOP_SECS", SMART_TURN_FALLBACK_SECS, min_value=0.0)
    turn = None if silero_only else SourceSmartTurn(sample_rate=RATE, params=SmartTurnParams(stop_secs=turn_stop_secs))
    return AudioOffsetDetector(vad_analyzer=vad, turn_analyzer=turn)
