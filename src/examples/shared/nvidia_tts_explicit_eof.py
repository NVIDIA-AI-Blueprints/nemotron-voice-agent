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
audio-context drain under its execution watchdog. Live sessions retain native
session/evaluator cancellation bounds; this helper adds no per-generation
watchdog. WebRTC and per-sentence synthesis retain their existing policy.
"""

from importlib.metadata import version

from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings, NvidiaTTSSynthesisMode

PINNED_PIPECAT_VERSION = "1.12.0"
EXPLICIT_EOF_COMPLETION_POLICY = "explicit_provider_eof_pipecat_1_12_compat"


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


class ExplicitEOFNvidiaTTSService(NvidiaTTSService):
    """Delegate native stitched synthesis while waiting for explicit context EOF."""

    def __init__(self, *, settings: NvidiaTTSSettings, **kwargs):
        """Reject unsupported versions, synthesis modes and conflicting deadlines."""
        options = explicit_eof_tts_options(settings.synthesis_mode)
        if kwargs.pop("stop_frame_timeout_s", None) is not None:
            raise ValueError("Explicit EOF TTS candidate cannot use an audio idle deadline")
        # Pinned, undocumented compatibility behavior; no native processing loop
        # is copied or replaced. A separate owner must enforce compute failures.
        super().__init__(settings=settings, **options, **kwargs)
