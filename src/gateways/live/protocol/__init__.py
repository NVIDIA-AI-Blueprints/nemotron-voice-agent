# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The live session protocol: the strict startup schema, the voices, and the rejection error.

This sits next to, and independent of, the Realtime protocol in ``gateways.realtime``. The ``model`` field is a
client-chosen label; the server's own services decide which models run.
"""

from gateways.live.protocol.config import (
    DELEGATION_CONFIGS,
    Audio,
    AudioFormat,
    AudioOutput,
    BackendConfig,
    Client,
    ClientDelegationConfig,
    Permissions,
    ResponsesDelegationConfig,
    SessionConfig,
    StrictModel,
)
from gateways.live.protocol.errors import ProtocolError, validation_error
from gateways.live.protocol.voices import LIVE_VOICES, TTS_VOICES

__all__ = [
    "DELEGATION_CONFIGS",
    "LIVE_VOICES",
    "TTS_VOICES",
    "Audio",
    "AudioFormat",
    "AudioOutput",
    "BackendConfig",
    "Client",
    "ClientDelegationConfig",
    "Permissions",
    "ProtocolError",
    "ResponsesDelegationConfig",
    "SessionConfig",
    "StrictModel",
    "validation_error",
]
