# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned options for explicit-EOF NVIDIA stitched TTS context completion.

This compatibility adapter is deliberately limited to Pipecat 1.12.0. That
version forwards ``stop_frame_timeout_s=None`` to ``asyncio.wait_for``, which
waits for an actual queue item without an idle deadline. Pipecat does not
document ``None`` for this constructor argument; the behavior must become an
upstream contract before this adapter can be considered generally supported.

Native NVIDIA request generation, RPC responses, queue serialization, empty
result errors, interruption and cleanup are inherited unchanged. Generic
Assistant's live and offline WebSocket compositions select this policy for
stitched synthesis. Offline response completion waits for worker EOF and native
audio-context drain under its execution watchdog. The response consumer also has
a 600-second provider-inactivity failure bound,
refreshed by actual provider audio. It fails stalled work, not conversational
silence, and closes terminal-error contexts so later requests can drain.
WebRTC and per-sentence synthesis retain their existing policy.
"""

import asyncio
import math
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import version

from pipecat.frames.frames import ErrorFrame, TTSAudioRawFrame
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings, NvidiaTTSSynthesisMode

PINNED_PIPECAT_VERSION = "1.12.0"
EXPLICIT_EOF_COMPLETION_POLICY = "explicit_provider_eof_pipecat_1_12_compat"
TTS_PROVIDER_IDLE_TIMEOUT_SECONDS = 600.0


@dataclass
class _ResponseCompletion:
    owner: object
    context_id: str
    deadline: asyncio.Timeout
    closure_requested: bool = False


_RESPONSE_COMPLETION: ContextVar[_ResponseCompletion | None] = ContextVar("nva_tts_response_completion", default=None)


def explicit_eof_tts_options(synthesis_mode: str | NvidiaTTSSynthesisMode) -> dict[str, None]:
    """Validate the compatibility contract and return native constructor options.

    Callers select this policy explicitly. Returning options instead of a new
    service type lets ordinary, traced and capturing NVIDIA services preserve
    their existing inheritance and all native provider/queue operations.
    """
    if version("pipecat-ai") != PINNED_PIPECAT_VERSION:
        raise RuntimeError("Explicit EOF TTS candidate requires exactly Pipecat 1.12.0")
    if synthesis_mode != NvidiaTTSSynthesisMode.STITCHED:
        raise ValueError("Explicit EOF TTS candidate requires stitched synthesis")
    return {"stop_frame_timeout_s": None}


class ExplicitEOFCompletionMixin:
    """Close only a finished native response context and bound provider inactivity.

    The inherited NVIDIA service still consumes every response and owns its RPC,
    PCM queue, errors and serialization. The compatibility boundary observes
    public context appends/removal and wraps the pinned response consumer only.
    The operational deadline fails stalled provider work; it never ends user
    speech, adds silence, or turns a missing EOF into successful completion.
    """

    def __init__(self, *args, provider_idle_timeout_s=TTS_PROVIDER_IDLE_TIMEOUT_SECONDS, **kwargs):
        """Keep idle-mode services native and validate the explicit-EOF failure bound."""
        self._explicit_eof_enabled = "stop_frame_timeout_s" in kwargs and kwargs["stop_frame_timeout_s"] is None
        if self._explicit_eof_enabled:
            settings = kwargs.get("settings")
            explicit_eof_tts_options(settings.synthesis_mode if settings else "stitched")
        if (
            isinstance(provider_idle_timeout_s, bool)
            or not isinstance(provider_idle_timeout_s, int | float)
            or not math.isfinite(provider_idle_timeout_s)
            or provider_idle_timeout_s <= 0
        ):
            raise ValueError("TTS provider inactivity deadline must be finite and positive")
        self._tts_provider_idle_timeout_s = float(provider_idle_timeout_s)
        super().__init__(*args, **kwargs)

    async def append_to_audio_context(self, context_id, frame):
        """Refresh provider activity only after this response appends nonempty PCM."""
        scope = _RESPONSE_COMPLETION.get()
        result = await super().append_to_audio_context(context_id, frame)
        if (
            scope is not None
            and scope.owner is self
            and scope.context_id == context_id
            and isinstance(frame, TTSAudioRawFrame)
            and bool(frame.audio)
            and not scope.closure_requested
            and self.audio_context_available(context_id)
            and not scope.deadline.expired()
        ):
            scope.deadline.reschedule(asyncio.get_running_loop().time() + self._tts_provider_idle_timeout_s)
        return result

    async def remove_audio_context(self, context_id):
        """Observe native EOF requests without changing their serialized order."""
        scope = _RESPONSE_COMPLETION.get()
        if scope is not None and scope.owner is self and scope.context_id == context_id:
            scope.closure_requested = True
        return await super().remove_audio_context(context_id)

    async def _process_responses(self, state):
        if not self._explicit_eof_enabled:
            return await super()._process_responses(state)
        scope = _ResponseCompletion(self, state.context_id, asyncio.timeout(self._tts_provider_idle_timeout_s))
        token = _RESPONSE_COMPLETION.set(scope)
        cancelled = False
        timed_out = False
        try:
            try:
                async with scope.deadline:
                    return await super()._process_responses(state)
            except TimeoutError as error:
                if not scope.deadline.expired():
                    raise
                # Use the pinned native cancellation primitives. The synthesis
                # worker owns the blocking iterator and will unwind on cancel.
                timed_out = True
                state.stop_event.set()
                self._cancel_stream_call(state)
                self._signal_synthesis_close(state)
                try:
                    await self.push_error_frame(
                        ErrorFrame(error="TTS provider exceeded its operational inactivity deadline", exception=error)
                    )
                finally:
                    if self._stream_state is state:
                        self._stream_state = None
        except asyncio.CancelledError:
            # This includes interruption while timeout/error publication awaits.
            # Native interruption/reset owns the context and its cancellation.
            cancelled = True
            raise
        finally:
            try:
                # Pipecat1.12 closes on EOF, but not on its terminal exception
                # branch. Complete that exact old context without touching a new
                # active stream, and never enqueue duplicate EOF sentinels.
                if (
                    not cancelled
                    and (timed_out or not state.stop_event.is_set())
                    and not scope.closure_requested
                    and self.audio_context_available(state.context_id)
                ):
                    await self.remove_audio_context(state.context_id)
            finally:
                _RESPONSE_COMPLETION.reset(token)


class ExplicitEOFNvidiaTTSService(ExplicitEOFCompletionMixin, NvidiaTTSService):
    """Delegate native stitched synthesis while waiting for explicit context EOF."""

    def __init__(self, *, settings: NvidiaTTSSettings, **kwargs):
        """Reject unsupported versions, synthesis modes and conflicting deadlines."""
        options = explicit_eof_tts_options(settings.synthesis_mode)
        if kwargs.pop("stop_frame_timeout_s", None) is not None:
            raise ValueError("Explicit EOF TTS candidate cannot use an audio idle deadline")
        # Pinned, undocumented compatibility behavior; no native processing loop
        # is copied or replaced. The mixin bounds provider inactivity separately from media deadlines.
        super().__init__(settings=settings, **options, **kwargs)
