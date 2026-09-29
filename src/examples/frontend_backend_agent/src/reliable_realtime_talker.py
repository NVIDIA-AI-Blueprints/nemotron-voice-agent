# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime-only Talker service composition for the Frontend/Backend Agent."""

from examples.frontend_backend_agent.src.reliable_talker import ReliableNvidiaLLMService
from examples.shared.nvidia_llm import NvidiaLLMService as RealtimeNvidiaLLMService


class ReliableRealtimeNvidiaLLMService(ReliableNvidiaLLMService, RealtimeNvidiaLLMService):
    """Apply the bounded Talker guards to the protocol-aware Realtime service.

    This module keeps ordinary RTVI imports outside the Realtime protocol stack.
    """
