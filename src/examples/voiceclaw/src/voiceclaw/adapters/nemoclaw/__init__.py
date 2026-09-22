# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Optional adapters for NemoClaw runtime surfaces."""

from voiceclaw.adapters.nemoclaw.committed_turn import (
    EndpointPolicy,
    NemoClawBackendFailure,
    NemoClawCommittedTurnAdapter,
    NemoClawCommittedTurnError,
    NemoClawEndpointError,
    NemoClawProtocolError,
    NemoClawRequestRejected,
    NemoClawRequestValidationError,
    NemoClawTransportError,
)

__all__ = [
    "EndpointPolicy",
    "NemoClawBackendFailure",
    "NemoClawCommittedTurnAdapter",
    "NemoClawCommittedTurnError",
    "NemoClawEndpointError",
    "NemoClawProtocolError",
    "NemoClawRequestRejected",
    "NemoClawRequestValidationError",
    "NemoClawTransportError",
]
