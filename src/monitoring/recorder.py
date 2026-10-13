# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-session recorder: timeline, metrics, turns, LLM inputs, audio, images.

Wiring (see ``examples/multilingual/pipeline.py``)::

    recorder = SessionRecorder.create(session_id=..., example=..., config=snapshot, llm=llm)
    pipeline = Pipeline([..., transport.output(), *recorder.processors(), ...])
    worker = PipelineWorker(..., observers=[..., *recorder.observers()])
    recorder.attach(worker, user_aggregator, assistant_aggregator, latency_observer)
    await recorder.start()           # before the runner starts
    ...                              # on client ready: await recorder.on_session_started()
    await recorder.close("...")      # in a finally block after the runner returns

Hot-path rule: every hook only copies references into a dict and hands it to
``RecordWriter``; serialization, hashing, encoding and I/O happen in the writer
thread. Per-turn metrics (``monitoring.turn_metrics``) are derived there too, when
a turn ends and when the session closes.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import uuid
import wave
from collections import deque
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallsStartedFrame,
    InputImageRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserImageRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.services.llm_service import LLMService

from monitoring import media as media_utils
from monitoring import system_metrics, turn_metrics
from monitoring.config import MonitoringConfig, load_monitoring_config
from monitoring.store import ArtifactStore, SessionStore, open_stores
from monitoring.writer import RecordWriter
from utils import PROJECT_ROOT

RECORDING_SAMPLE_RATE = 16_000
_STEREO_CHUNK_SECS = 10
_KEYFRAME_MAX_WIDTH = 640
_VIDEO_CHUNK_SECS = 30


@cache
def _stores_for(config: MonitoringConfig) -> tuple[SessionStore, ArtifactStore]:
    return open_stores(config)


# Metric samples pipecat emits with a zero value (one per service at start-up): they
# are not measurements and would drag percentiles down.
_DROP_ZERO_METRICS = frozenset({"ttfb", "processing"})
_KEPT_TURNS = 4


def git_revision(root: Path = PROJECT_ROOT) -> str | None:
    """Best-effort git sha of the running code (env ``GIT_SHA`` wins).

    Handles a detached HEAD, loose and packed refs, and git worktrees. Returns
    None when ``root`` has no ``.git`` (for example in the app container, which
    only mounts ``src``): set ``GIT_SHA`` there.
    """
    if os.getenv("GIT_SHA"):
        return os.getenv("GIT_SHA")
    try:
        git_dir = root / ".git"
        if git_dir.is_file():  # worktree: ".git" is a "gitdir: <path>" pointer
            git_dir = (root / git_dir.read_text().strip().removeprefix("gitdir: ")).resolve()
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        common = git_dir
        if (git_dir / "commondir").is_file():
            common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
        for base in (git_dir, common):
            if (base / ref).is_file():
                return (base / ref).read_text().strip()
        for line in (common / "packed-refs").read_text().splitlines():
            sha, _, name = line.partition(" ")
            if name == ref:
                return sha
    except OSError:
        pass
    return None


def _metric_base_name(data: Any) -> str:
    name = type(data).__name__.removesuffix("MetricsData") or type(data).__name__
    return re.sub(r"(?<!^)(?=[A-Z][a-z])", "_", name).lower()


def flatten_metrics(data: Any) -> list[tuple[str, float]]:
    """Flatten a pipecat ``MetricsData`` into ``(name, value)`` numeric pairs."""
    base = _metric_base_name(data)
    out: list[tuple[str, float]] = []
    for key, value in data.model_dump().items():
        if key in ("processor", "model"):
            continue
        name = base if key == "value" else f"{base}.{key}"
        if isinstance(value, int | float):  # bool included: stored as 0.0 / 1.0
            out.append((name, float(value)))
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, int | float) and not isinstance(sub_value, bool):
                    out.append((f"{base}.{sub_key}", float(sub_value)))
    return out


class _VideoChunkWriter:
    """Chunked MP4 writer for ``RECORD_VIDEO=full`` (writer-thread only, dev use)."""

    def __init__(self, artifacts: ArtifactStore, prefix: str, fps: float):
        self._artifacts = artifacts
        self._prefix = prefix
        self._fps = max(1, round(fps))
        self._container = None
        self._stream = None
        self._key = ""
        self._opened_at = 0.0
        self._index = 0

    def add(self, image: bytes, size: tuple[int, int], fmt: str, ts: float) -> list[tuple[str, dict]]:
        import av
        import numpy as np

        rows = []
        if self._container is not None and ts - self._opened_at >= _VIDEO_CHUNK_SECS:
            rows = self.close()
        if self._container is None:
            self._key = f"{self._prefix}video/{self._index:04d}.mp4"
            self._index += 1
            self._container = av.open(str(self._artifacts.local_path(self._key)), mode="w")
            self._stream = self._container.add_stream("mpeg4", rate=self._fps)
            self._stream.width, self._stream.height = size[0] - size[0] % 2, size[1] - size[1] % 2
            self._stream.pix_fmt = "yuv420p"
            self._opened_at = ts
        array = np.frombuffer(image, dtype=np.uint8).reshape(size[1], size[0], -1)[:, :, :3]
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        frame = frame.reformat(width=self._stream.width, height=self._stream.height)
        for packet in self._stream.encode(frame):
            self._container.mux(packet)
        return rows

    def close(self) -> list[tuple[str, dict]]:
        if self._container is None:
            return []
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._artifacts.commit(self._key)
        self._container = None
        return [("__video_chunk__", {"artifact_key": self._key, "ts": self._opened_at})]


class _StereoWavWriter:
    """Appends stereo chunks to ``conversation.wav`` (writer-thread only)."""

    def __init__(self, artifacts: ArtifactStore, key: str):
        self._artifacts = artifacts
        self.key = key
        self._wav: wave.Wave_write | None = None
        self.frames = 0
        self.sample_rate = 0
        self.num_channels = 0

    def append(self, audio: bytes, sample_rate: int, num_channels: int) -> None:
        if self._wav is None:
            # Kept open across writer batches; closed in ``close``.
            self._wav = wave.open(str(self._artifacts.local_path(self.key)), "wb")  # noqa: SIM115
            self._wav.setnchannels(num_channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
            self.sample_rate, self.num_channels = sample_rate, num_channels
        self._wav.writeframes(audio)
        self.frames += len(audio) // (2 * num_channels)

    def close(self) -> bool:
        if self._wav is None:
            return False
        self._wav.close()
        self._wav = None
        self._artifacts.commit(self.key)
        return True


class _RecorderObserver(BaseObserver):
    """Frame observer feeding the recorder; runs in pipecat's per-observer task."""

    def __init__(self, recorder: SessionRecorder):
        super().__init__(name="SessionRecorderObserver")
        self._rec = recorder
        self._seen_ids: set[int] = set()
        self._seen_order: deque[int] = deque()

    def _first_sight(self, frame) -> bool:
        # Frames are reported at every hop; system frames may also be broadcast
        # as an upstream/downstream sibling pair. Only handle each once.
        if frame.id in self._seen_ids or frame.broadcast_sibling_id in self._seen_ids:
            return False
        self._seen_ids.add(frame.id)
        self._seen_order.append(frame.id)
        if len(self._seen_order) > 2048:
            self._seen_ids.discard(self._seen_order.popleft())
        return True

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame
        rec = self._rec
        if isinstance(frame, InputImageRawFrame):
            # Checked first: video can stream at 30 fps through every hop.
            if self._first_sight(frame):
                rec.on_image_frame(frame)
            return
        if isinstance(frame, MetricsFrame):
            if self._first_sight(frame):
                rec.on_metrics(frame)
        elif isinstance(frame, LLMContextFrame):
            if rec.is_main_llm(data.destination) and self._first_sight(frame):
                rec.on_llm_call_started(frame.context, data.destination)
        elif isinstance(frame, LLMTextFrame):
            if rec.is_main_llm(data.source):
                rec.on_llm_text(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if rec.is_main_llm(data.source):
                rec.on_llm_call_finished()
        elif isinstance(frame, FunctionCallsStartedFrame):
            if self._first_sight(frame):
                rec.on_function_calls(frame)
        elif isinstance(frame, FunctionCallResultFrame):
            if self._first_sight(frame):
                rec.on_function_result(frame)
        elif isinstance(frame, FunctionCallInProgressFrame):
            if self._first_sight(frame):
                # Broadcast right before the handler runs; RTVI forwards the call to the
                # client on this frame, so this is the server send time.
                rec.event("function_call_in_progress", name=frame.function_name, tool_call_id=frame.tool_call_id)
        elif isinstance(frame, FunctionCallCancelFrame):
            if self._first_sight(frame):
                rec.event("function_call_cancelled", name=frame.function_name, tool_call_id=frame.tool_call_id)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            if self._first_sight(frame):
                rec.on_vad_stopped(frame)
        elif isinstance(
            frame,
            UserStartedSpeakingFrame
            | UserStoppedSpeakingFrame
            | BotStartedSpeakingFrame
            | BotStoppedSpeakingFrame
            | InterruptionFrame,
        ):
            if self._first_sight(frame):
                rec.on_speech_event(frame)
        elif isinstance(frame, TranscriptionFrame):
            if self._first_sight(frame):
                rec.event(
                    "asr_final",
                    processor=data.source.name,
                    text=frame.text,
                    language=str(frame.language) if frame.language else None,
                )
        elif isinstance(frame, ErrorFrame) and self._first_sight(frame):
            rec.event("error", processor=data.source.name, error=str(frame.error), fatal=frame.fatal)


class SessionRecorder:
    """Records one conversation into the session and artifact stores."""

    def __init__(
        self,
        *,
        session_id: str,
        example: str,
        config: dict[str, Any],
        settings: MonitoringConfig,
        store: SessionStore,
        artifacts: ArtifactStore,
        llm: LLMService | None = None,
        language: str | None = None,
    ):
        """Prepare the recorder; call ``start`` before the pipeline runs."""
        self.session_id = session_id
        self.example = example
        self.settings = settings
        self._config = config
        self._store = store
        self._artifacts = artifacts
        self._llm = llm
        self._language = language
        self.started_at = time.time()
        self.prefix = f"sessions/{datetime.fromtimestamp(self.started_at, UTC):%Y-%m-%d}/{session_id}/"
        self.writer = RecordWriter(session_id, store, artifacts)

        # Turn 0 holds the greeting (bot speech before the first user turn).
        self.turn_idx = 0
        self._bot_turn_idx = 0
        self._turn: dict[str, Any] = {"idx": 0}
        # Recent turns by index: an interrupted assistant turn reports its text after
        # the next user turn has started, and must still land on its own turn.
        self._turns: dict[int, dict[str, Any]] = {0: self._turn}
        self._assistant_turn_idx: int | None = None
        self._sampling = False
        self._audio_segments: dict[tuple[int, str], int] = {}
        self._llm_call: dict[str, Any] | None = None
        self._last_keyframe_at = 0.0
        self._closed = False

        self._observer = _RecorderObserver(self)
        self._stereo = _StereoWavWriter(artifacts, f"{self.prefix}conversation.wav")
        self._video: _VideoChunkWriter | None = None
        if settings.record_video == "full":
            try:
                import av  # noqa: F401

                self._video = _VideoChunkWriter(artifacts, self.prefix, settings.video_fps)
            except ImportError:
                logger.warning("RECORD_VIDEO=full needs PyAV; falling back to keyframes")
        self._audio = self._build_audio_processor()

    # ----------------------------------------------------------------- set up
    @classmethod
    def create(
        cls,
        *,
        session_id: str | None,
        example: str,
        config: dict[str, Any],
        llm: LLMService | None = None,
        language: str | None = None,
        settings: MonitoringConfig | None = None,
    ) -> SessionRecorder | None:
        """Return a recorder, or None when ``MONITORING_ENABLED`` is false or storage fails."""
        settings = settings or load_monitoring_config()
        if not settings.enabled:
            return None
        try:
            store, artifacts = _stores_for(settings)
        except Exception as exc:
            logger.opt(exception=exc).error("Monitoring storage unavailable; recording disabled")
            return None
        return cls(
            session_id=session_id or uuid.uuid4().hex[:12],
            example=example,
            config=config,
            settings=settings,
            store=store,
            artifacts=artifacts,
            llm=llm,
            language=language,
        )

    def observers(self) -> list[BaseObserver]:
        """Observers to add to the ``PipelineWorker``."""
        return [self._observer]

    def processors(self) -> list[AudioBufferProcessor]:
        """Processors to insert right after ``transport.output()``."""
        return [self._audio] if self._audio else []

    def attach(self, worker, user_aggregator, assistant_aggregator, latency_observer=None) -> None:
        """Register the aggregator / latency / turn event handlers."""

        @user_aggregator.event_handler("on_user_turn_started")
        async def _on_user_turn_started(aggregator, strategy):
            self._start_user_turn()

        @user_aggregator.event_handler("on_user_turn_stopped")
        async def _on_user_turn_stopped(aggregator, strategy, message):
            text = getattr(message, "content", None) or ""
            previous = self._turn.get("user_text")
            self._turn["user_text"] = f"{previous} {text}".strip() if previous else text
            # When the turn is released to the LLM (after turn detection and ASR), not
            # when the user stopped speaking: see ``on_vad_stopped``.
            self._turn["user_stopped_at"] = time.time()
            self._save_turn()

        @assistant_aggregator.event_handler("on_assistant_turn_started")
        async def _on_assistant_turn_started(aggregator):
            # Start of the LLM response, not of the bot audio: see ``on_speech_event``.
            self._assistant_turn_idx = self.turn_idx
            self._turn.setdefault("bot_started_at", time.time())

        @assistant_aggregator.event_handler("on_assistant_turn_stopped")
        async def _on_assistant_turn_stopped(aggregator, message):
            # On a barge-in the next user turn starts before this fires: write to the
            # turn the response belongs to, not to the current one.
            idx, self._assistant_turn_idx = self._assistant_turn_idx, None
            turn = self._turns.get(self.turn_idx if idx is None else idx, self._turn)
            text = getattr(message, "content", None) or ""
            previous = turn.get("bot_text")
            turn["bot_text"] = f"{previous} {text}".strip() if previous else text
            turn["interrupted"] = bool(turn.get("interrupted")) or bool(getattr(message, "interrupted", False))
            turn["bot_stopped_at"] = time.time()
            self._save_turn(turn)

        turn_tracker = getattr(worker, "turn_tracking_observer", None)
        if turn_tracker is not None:

            @turn_tracker.event_handler("on_turn_ended")
            async def _on_turn_ended(observer, turn_number, duration, was_interrupted):
                self.event("turn_ended", turn_number=turn_number, duration=duration, interrupted=was_interrupted)
                self.metric("turn_duration", duration, processor="TurnTrackingObserver")

        if latency_observer is not None:

            @latency_observer.event_handler("on_latency_measured")
            async def _on_latency(observer, latency):
                self.metric("user_bot_latency", latency, processor="UserBotLatencyObserver")

            @latency_observer.event_handler("on_first_bot_speech_latency")
            async def _on_first_speech(observer, latency):
                self.metric("first_bot_speech_latency", latency, processor="UserBotLatencyObserver")

            @latency_observer.event_handler("on_latency_breakdown")
            async def _on_breakdown(observer, breakdown):
                self.event("latency_breakdown", processor="UserBotLatencyObserver", **breakdown.model_dump())
                if breakdown.user_turn_secs is not None:
                    self.metric("user_turn_secs", breakdown.user_turn_secs, processor="UserBotLatencyObserver")

    async def start(self) -> None:
        """Create the session row and start the background writer."""
        row = {
            "id": self.session_id,
            "example": self.example,
            "started_at": self.started_at,
            "last_seen_at": self.started_at,
            "ended_at": None,
            "end_reason": None,
            "config": media_utils.to_jsonable(
                {**self._config, "git_sha": git_revision(), "recorder_version": turn_metrics.RECORDER_VERSION}
            ),
            "artifact_prefix": self.prefix,
        }
        await asyncio.to_thread(self._store.create_session, row)
        self.writer.start()
        self._save_turn()
        if self.settings.record_system_metrics:
            await asyncio.to_thread(system_metrics.sampler().acquire, self._store)
            self._sampling = True
        logger.info(f"Recording session {self.session_id} → {self.settings.data_dir}")

    async def on_session_started(self) -> None:
        """Call once the client is ready (starts audio capture)."""
        if self._audio:
            await self._audio.start_recording()

    async def stop_audio(self) -> None:
        """Flush buffered audio; call before cancelling the worker."""
        if self._audio:
            await self._audio.stop_recording()

    async def close(self, reason: str = "completed") -> None:
        """Flush everything and mark the session ended. Idempotent."""
        if self._closed:
            return
        self._closed = True
        await self.stop_audio()
        self.on_llm_call_finished()
        self._save_turn()
        self.writer.put_task(self._finalize_artifacts)
        # Whole-session pass: same code as the backfill, so live and backfilled rows agree.
        session_id = self.session_id
        self.writer.put_post_commit(lambda store: turn_metrics.recompute_session(store, session_id))
        await self.writer.close()
        if self._sampling:
            self._sampling = False
            await asyncio.to_thread(system_metrics.sampler().release)
        await asyncio.to_thread(self._store.end_session, self.session_id, time.time(), reason)
        if self.writer.dropped:
            logger.warning(f"Recorder dropped {self.writer.dropped} item(s) for session {self.session_id}")
        logger.info(f"Session {self.session_id} recorded (max writer queue depth {self.writer.max_depth})")

    # ------------------------------------------------------------ primitives
    def is_main_llm(self, processor) -> bool:
        """Return whether ``processor`` is the conversation LLM (not e.g. the summarizer)."""
        if self._llm is not None:
            return processor is self._llm
        return isinstance(processor, LLMService)

    def event(self, kind: str, *, processor: str | None = None, **data: Any) -> None:
        """Queue a timeline event for the current turn."""
        self.writer.put_row(
            "events",
            {
                "session_id": self.session_id,
                "turn_idx": self.turn_idx,
                "ts": time.time(),
                "kind": kind,
                "processor": processor,
                "data": media_utils.to_jsonable(data) if data else None,
            },
        )

    def metric(self, name: str, value: float, *, processor: str | None = None, model: str | None = None) -> None:
        """Queue a numeric metric sample for the current turn."""
        self.writer.put_row(
            "metrics",
            {
                "session_id": self.session_id,
                "turn_idx": self.turn_idx,
                "ts": time.time(),
                "processor": processor,
                "model": model,
                "name": name,
                "value": float(value),
            },
        )

    # ------------------------------------------------------------------ turns
    def _start_user_turn(self) -> None:
        self._save_turn()
        finished = self.turn_idx
        self.turn_idx += 1
        self._turn = {"idx": self.turn_idx, "user_started_at": time.time()}
        self._turns[self.turn_idx] = self._turn
        self._turns.pop(self.turn_idx - _KEPT_TURNS, None)
        self._save_turn()
        if finished > 0:
            # Turn end: derive its metrics row once everything queued so far is stored.
            session_id = self.session_id
            self.writer.put_post_commit(
                lambda store: turn_metrics.recompute_session(store, session_id, only_turn=finished)
            )

    def _save_turn(self, turn: dict[str, Any] | None = None) -> None:
        turn = self._turn if turn is None else turn
        # ``barge_in`` is not listed: it is written by ``monitoring.turn_metrics``.
        self.writer.put_row(
            "turns",
            {
                "session_id": self.session_id,
                "idx": turn["idx"],
                "user_text": turn.get("user_text"),
                "user_started_at": turn.get("user_started_at"),
                "user_stopped_at": turn.get("user_stopped_at"),
                "user_speech_stopped_at": turn.get("user_speech_stopped_at"),
                "bot_text": turn.get("bot_text"),
                "bot_started_at": turn.get("bot_started_at"),
                "bot_speech_started_at": turn.get("bot_speech_started_at"),
                "bot_stopped_at": turn.get("bot_stopped_at"),
                "interrupted": bool(turn.get("interrupted")),
                "language": self._language,
            },
        )

    # ---------------------------------------------------------------- frames
    def on_metrics(self, frame: MetricsFrame) -> None:
        """Persist a ``MetricsFrame`` and fill in the pending LLM call's TTFB/usage."""
        for data in frame.data:
            pairs = [(n, v) for n, v in flatten_metrics(data) if v != 0 or n not in _DROP_ZERO_METRICS]
            for name, value in pairs:
                self.metric(name, value, processor=data.processor, model=data.model)
            call = self._llm_call
            if call is not None and self._llm is not None and data.processor == self._llm.name:
                values = dict(pairs)
                if "ttfb" in values and call.get("ttfb") is None:
                    call["ttfb"] = values["ttfb"]
                if "llm_usage.prompt_tokens" in values:
                    call["prompt_tokens"] = int(values["llm_usage.prompt_tokens"])
                    call["completion_tokens"] = int(values.get("llm_usage.completion_tokens", 0))
                if data.model and not call.get("model"):
                    call["model"] = data.model

    def on_speech_event(self, frame) -> None:
        """Record VAD / bot speaking / interruption events."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_turn_idx = self.turn_idx
            if "bot_speech_started_at" not in self._turn:
                self._turn["bot_speech_started_at"] = time.time()  # first bot audio of the turn
                self._save_turn()
        elif isinstance(frame, InterruptionFrame) and self._llm_call is not None:
            # The interruption cancels the completion: close the call now. The TTFB
            # sample pipecat emits while stopping its metrics is the time until the
            # interruption, not a first token, and must not be attached to the call.
            self._llm_call["interrupted"] = True
            self.on_llm_call_finished()
        kind = re.sub(r"(?<!^)(?=[A-Z])", "_", type(frame).__name__.removesuffix("Frame")).lower()
        self.event(kind)

    def on_vad_stopped(self, frame: VADUserStoppedSpeakingFrame) -> None:
        """Record when the user really stopped speaking (VAD decision minus its silence window).

        The VAD can stop several times within one turn; the last stop before the bot
        answers is the one the latency is measured from.
        """
        if "bot_speech_started_at" not in self._turn:
            self._turn["user_speech_stopped_at"] = frame.timestamp - frame.stop_secs

    def on_llm_call_started(self, context, llm) -> None:
        """Snapshot the LLM input when a context reaches the conversation LLM."""
        self.on_llm_call_finished()
        # Shallow copies only: the context keeps mutating while we serialize later.
        messages = [dict(m) if isinstance(m, dict) else m for m in context.get_messages()]
        self._llm_call = {
            "turn_idx": self.turn_idx,
            "started_at": time.time(),
            "processor": llm.name,
            "model": getattr(getattr(llm, "_settings", None), "model", None),
            "messages": messages,
            "tools": context.tools,
            "output": [],
            "function_calls": [],
        }

    def on_llm_text(self, text: str) -> None:
        """Accumulate streamed LLM output."""
        if self._llm_call is not None:
            self._llm_call["output"].append(text)

    def on_function_calls(self, frame: FunctionCallsStartedFrame) -> None:
        """Record the function calls requested by the LLM."""
        calls = [
            {"name": c.function_name, "tool_call_id": c.tool_call_id, "arguments": c.arguments}
            for c in frame.function_calls
        ]
        if self._llm_call is not None:
            self._llm_call["function_calls"].extend(calls)
        self.event("function_calls", calls=calls)

    def on_function_result(self, frame: FunctionCallResultFrame) -> None:
        """Record a function result, extracting any inline media it returns."""
        turn_idx, ts = self.turn_idx, time.time()
        name, call_id, result = frame.function_name, frame.tool_call_id, frame.result

        def persist(artifacts: ArtifactStore) -> list[tuple[str, dict]]:
            stripped, found = media_utils.extract_inline_media(result)
            rows = self._media_rows(artifacts, found, turn_idx, ts, source=f"tool:{name}", ref=call_id)
            rows.append(
                (
                    "events",
                    {
                        "session_id": self.session_id,
                        "turn_idx": turn_idx,
                        "ts": ts,
                        "kind": "function_result",
                        "processor": None,
                        "data": {"name": name, "tool_call_id": call_id, "result": stripped},
                    },
                )
            )
            return rows

        self.writer.put_task(persist)

    def on_llm_call_finished(self) -> None:
        """Queue the pending LLM call for persistence."""
        call, self._llm_call = self._llm_call, None
        if call is None:
            return
        call["ended_at"] = time.time()

        def persist(artifacts: ArtifactStore) -> list[tuple[str, dict]]:
            messages, found = media_utils.extract_inline_media(call["messages"])
            rows = self._media_rows(artifacts, found, call["turn_idx"], call["started_at"], source="context")
            images = [m for m in found if m.mime.startswith("image/")]
            pixels = 0
            for image in {m.sha256: m for m in images}.values():
                width, height = media_utils.image_dimensions(image.data)
                pixels += (width or 0) * (height or 0)
            tools = call["tools"]
            tool_names = [
                getattr(t, "name", None) or getattr(t, "__name__", str(t))
                for t in getattr(tools, "standard_tools", None) or []
            ]
            rows.append(
                (
                    "llm_calls",
                    {
                        "session_id": self.session_id,
                        "turn_idx": call["turn_idx"],
                        "started_at": call["started_at"],
                        "ended_at": call["ended_at"],
                        "processor": call["processor"],
                        "model": call["model"],
                        "messages": messages,
                        "tools": tool_names,
                        "output_text": "".join(call["output"]),
                        "function_calls": media_utils.to_jsonable(call["function_calls"]),
                        "ttfb": call.get("ttfb"),
                        "prompt_tokens": call.get("prompt_tokens"),
                        "completion_tokens": call.get("completion_tokens"),
                        "n_images": len(images),
                        "image_pixels": pixels,
                        "interrupted": bool(call.get("interrupted")),
                    },
                )
            )
            return rows

        self.writer.put_task(persist)

    def on_image_frame(self, frame: InputImageRawFrame) -> None:
        """Record an image frame: explicit user images always, video frames per ``RECORD_VIDEO``."""
        is_user_image = isinstance(frame, UserImageRawFrame)
        mode = self.settings.record_video
        now = time.time()
        if not is_user_image:
            if mode == "off":
                return
            if self._video is None and now - self._last_keyframe_at < 1.0 / self.settings.video_fps:
                return
            self._last_keyframe_at = now
        turn_idx = self.turn_idx
        image, size, fmt = frame.image, frame.size, frame.format or "RGB"

        if self._video is not None and not is_user_image:
            video = self._video

            def persist_video(artifacts: ArtifactStore) -> list[tuple[str, dict]]:
                return self._video_rows(video.add(image, size, fmt, now), turn_idx)

            self.writer.put_task(persist_video)
            return

        modality = "image" if is_user_image else "video_keyframe"
        source = "user_image" if is_user_image else f"camera:{frame.transport_source or 'default'}"
        max_width = None if is_user_image else _KEYFRAME_MAX_WIDTH

        def persist(artifacts: ArtifactStore) -> list[tuple[str, dict]]:
            data, mime, width, height = media_utils.encode_raw_image(image, size, fmt, max_width=max_width)
            found = [media_utils.ExtractedMedia(media_utils.sha256(data), mime, data)]
            return self._media_rows(artifacts, found, turn_idx, now, source=source, modality=modality)

        self.writer.put_task(persist)

    # ------------------------------------------------------------------ audio
    def _build_audio_processor(self) -> AudioBufferProcessor | None:
        turns, stereo = self.settings.record_audio_turns, self.settings.record_audio_stereo
        if not turns and not stereo:
            return None
        processor = AudioBufferProcessor(
            sample_rate=RECORDING_SAMPLE_RATE,
            num_channels=2 if stereo else 1,
            buffer_size=RECORDING_SAMPLE_RATE * 2 * _STEREO_CHUNK_SECS if stereo else 0,
            enable_turn_audio=turns,
        )

        if turns:

            @processor.event_handler("on_user_turn_audio_data")
            async def _on_user_audio(proc, audio: bytes, sample_rate: int, num_channels: int):
                self._queue_turn_audio("user", self.turn_idx, audio, sample_rate)

            @processor.event_handler("on_bot_turn_audio_data")
            async def _on_bot_audio(proc, audio: bytes, sample_rate: int, num_channels: int):
                self._queue_turn_audio("bot", self._bot_turn_idx, audio, sample_rate)

        if stereo:

            @processor.event_handler("on_audio_data")
            async def _on_audio(proc, audio: bytes, sample_rate: int, num_channels: int):
                if audio:
                    stereo_writer = self._stereo
                    self.writer.put_task(lambda artifacts: stereo_writer.append(audio, sample_rate, num_channels) or [])

        return processor

    def _queue_turn_audio(self, role: str, turn_idx: int, audio: bytes, sample_rate: int) -> None:
        if not audio:
            return
        segment = self._audio_segments.get((turn_idx, role), 0)
        self._audio_segments[(turn_idx, role)] = segment + 1
        key = f"{self.prefix}turns/{turn_idx:03d}_{role}_{segment:02d}.wav"
        ts = time.time()
        duration = len(audio) / (2 * sample_rate)

        def persist(artifacts: ArtifactStore) -> list[tuple[str, dict]]:
            wav = media_utils.pcm16_to_wav(audio, sample_rate)
            artifacts.put(key, wav)
            return [
                (
                    "media",
                    {
                        "session_id": self.session_id,
                        "turn_idx": turn_idx,
                        "ts": ts - duration,
                        "modality": f"audio_{role}",
                        "source": "pipeline",
                        "ref": f"segment:{segment}",
                        "sha256": media_utils.sha256(audio),
                        "mime": "audio/wav",
                        "sample_rate": sample_rate,
                        "duration_secs": duration,
                        "artifact_key": key,
                    },
                )
            ]

        self.writer.put_task(persist)

    # --------------------------------------------------------------- helpers
    def _media_rows(
        self,
        artifacts: ArtifactStore,
        found: list[media_utils.ExtractedMedia],
        turn_idx: int,
        ts: float,
        *,
        source: str,
        ref: str | None = None,
        modality: str | None = None,
    ) -> list[tuple[str, dict]]:
        rows = []
        for item in {m.sha256: m for m in found}.values():
            key = f"{self.prefix}media/{item.sha256}.{media_utils.extension_for(item.mime)}"
            if not artifacts.exists(key):
                artifacts.put(key, item.data)
            width = height = None
            if item.mime.startswith("image/"):
                width, height = media_utils.image_dimensions(item.data)
            rows.append(
                (
                    "media",
                    {
                        "session_id": self.session_id,
                        "turn_idx": turn_idx,
                        "ts": ts,
                        "modality": modality or item.mime.split("/")[0],
                        "source": source,
                        "ref": ref,
                        "sha256": item.sha256,
                        "mime": item.mime,
                        "width": width,
                        "height": height,
                        "artifact_key": key,
                    },
                )
            )
        return rows

    def _video_rows(self, chunks: list[tuple[str, dict]], turn_idx: int) -> list[tuple[str, dict]]:
        rows = []
        for _, chunk in chunks:
            rows.append(
                (
                    "media",
                    {
                        "session_id": self.session_id,
                        "turn_idx": turn_idx,
                        "ts": chunk["ts"],
                        "modality": "video_chunk",
                        "source": "camera",
                        "sha256": media_utils.sha256(chunk["artifact_key"].encode()),
                        "mime": "video/mp4",
                        "artifact_key": chunk["artifact_key"],
                    },
                )
            )
        return rows

    def _finalize_artifacts(self, artifacts: ArtifactStore) -> list[tuple[str, dict]]:
        rows: list[tuple[str, dict]] = []
        if self._video is not None:
            rows.extend(self._video_rows(self._video.close(), self.turn_idx))
        if self._stereo.close():
            rows.append(
                (
                    "media",
                    {
                        "session_id": self.session_id,
                        "turn_idx": None,
                        "ts": self.started_at,
                        "modality": "audio_conversation",
                        "source": "pipeline",
                        "sha256": media_utils.sha256(self._stereo.key.encode()),
                        "mime": "audio/wav",
                        "sample_rate": self._stereo.sample_rate,
                        "duration_secs": self._stereo.frames / max(self._stereo.sample_rate, 1),
                        "artifact_key": self._stereo.key,
                    },
                )
            )
        return rows


def artifact_path(settings: MonitoringConfig, key: str) -> Path:
    """Local path of an artifact key (local store only; handy for scripts)."""
    return settings.artifacts_dir / key
