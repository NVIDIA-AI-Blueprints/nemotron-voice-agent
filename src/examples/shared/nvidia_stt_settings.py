# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve deployment-owned NVIDIA streaming ASR endpointing settings."""

import inspect
import os
import re

from pipecat.services.nvidia.stt import NvidiaSTTService

STANDARD_STT_STOP_HISTORY = inspect.signature(NvidiaSTTService).parameters["stop_history"].default


def resolve_stt_stop_history_ms() -> int:
    """Use one validated history window for ordinary and finite streaming ASR."""
    raw = os.environ.get("ASR_STOP_HISTORY_MS")
    if raw is None:
        return STANDARD_STT_STOP_HISTORY
    if not re.fullmatch(r"[0-9]+", raw) or not 1 <= int(raw) <= 10000:
        raise ValueError("ASR_STOP_HISTORY_MS must be an integer between 1 and 10000")
    return int(raw)
