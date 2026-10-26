# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exception hierarchy for the voice Frontend/Backend Agent prototype."""

from __future__ import annotations


class VoiceAgentError(Exception):
    """Base class for every error raised by this package."""


class VoiceConfigError(VoiceAgentError):
    """The voice configuration is invalid or references something that does not exist."""


class WireProtocolError(VoiceAgentError):
    """A client event violates the Realtime protocol.

    Always non-fatal: the session answers with an ``error`` event and keeps running.
    """

    def __init__(self, message: str, *, code: str = "invalid_value", param: str | None = None) -> None:
        """Record the Realtime error ``code`` and the offending ``param``."""
        super().__init__(message)
        self.code = code
        self.param = param


class SpeechServiceError(VoiceAgentError):
    """An ASR or TTS endpoint failed or is unreachable."""


class HistoryRepairError(VoiceAgentError):
    """The stored history does not have the shape or text a barge-in repair expects."""


class ProbeStateError(VoiceAgentError):
    """A barge-in probe cannot run or be carried out (no delegated turn, or the state changed since)."""
