# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live session protocol: the strict startup schema, the voices, and the rejection error.

This sits next to, and independent of, the Realtime protocol in ``realtime``. The ``model`` field is a client-chosen
label; the server's own services decide which models run.
"""

from live.protocol.config import (
    Audio,
    AudioFormat,
    AudioOutput,
    BackendConfig,
    Client,
    Delegation,
    Permissions,
    SessionConfig,
    StrictModel,
)
from live.protocol.errors import ProtocolError, approx_tokens, uid
from live.protocol.voices import LIVE_VOICES, TTS_VOICES

__all__ = [
    "LIVE_VOICES",
    "TTS_VOICES",
    "Audio",
    "AudioFormat",
    "AudioOutput",
    "BackendConfig",
    "Client",
    "Delegation",
    "Permissions",
    "ProtocolError",
    "SessionConfig",
    "StrictModel",
    "approx_tokens",
    "uid",
]
