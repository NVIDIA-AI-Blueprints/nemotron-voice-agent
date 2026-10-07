# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voice Frontend/Backend Agent prototype: an OpenAI Realtime (GA) server around the text agent.

ASR and TTS (nemo-speech over Riva gRPC) wrap the text prototype's
``FrontendBackendAgent``, served as ``WS /v1/realtime`` so tau3-bench's audio-native
``openai`` client can drive it. See ``README.md`` and
``misc/prototypes/voice/prototype-plan.md``.

``build_app`` is imported lazily so the wire, engine and agent layers can be used
without importing FastAPI.
"""

from __future__ import annotations

from typing import Any

from examples.frontend_backend_verdict.voice.config import VoiceConfig, load_voice_config

__all__ = ["VoiceConfig", "build_app", "load_voice_config"]


def __getattr__(name: str) -> Any:
    """Lazily expose :func:`server.build_app`."""
    if name == "build_app":
        from examples.frontend_backend_verdict.voice.server import build_app

        return build_app
    raise AttributeError(name)
