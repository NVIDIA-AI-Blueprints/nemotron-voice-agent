# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Offline doubles for the voice-layer tests: the base config, an in-memory transport, τ²-shaped payloads.

Named ``_voice_fakes`` because pytest imports test helpers by basename.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from examples.frontend_delegation_agent.core.audio.formats import AudioFormat
from examples.frontend_delegation_agent.core.audio.pcm import silence, tone
from examples.frontend_delegation_agent.core.voice_config import VoiceConfig, load_voice_config

REPO_ROOT = Path(__file__).resolve().parents[4]
PACKAGE_DIR = REPO_ROOT / "src" / "examples" / "frontend_delegation_agent"
BASE_CONFIG = PACKAGE_DIR / "config" / "voice_agent.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
PCMU = AudioFormat("audio/pcmu", 8000)
PCM16K = AudioFormat("audio/pcm", 16000)

_CONFIG_CACHE: dict[str, VoiceConfig] = {}


def base_config() -> VoiceConfig:
    if "base" not in _CONFIG_CACHE:
        _CONFIG_CACHE["base"] = load_voice_config(BASE_CONFIG)
    return _CONFIG_CACHE["base"]


class MemoryTransport:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.changed = asyncio.Event()

    async def send_text(self, text: str) -> None:
        self.events.append(json.loads(text))
        self.changed.set()


# -- tau2-shaped payloads ---------------------------------------------------------------


def tau2_session_update(domain: str = "mock") -> dict[str, Any]:
    return json.loads((FIXTURES / "tau2_session_update" / f"{domain}.json").read_text(encoding="utf-8"))


def pcmu_speech(ms: int, frequency: float = 300.0) -> bytes:
    return PCMU.encode(tone(ms, 8000, frequency=frequency))


def pcmu_silence(ms: int) -> bytes:
    return PCMU.encode(silence(ms, 8000))
