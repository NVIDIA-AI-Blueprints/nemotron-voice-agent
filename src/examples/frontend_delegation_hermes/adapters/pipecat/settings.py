# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat-head settings (``config/adapters/pipecat*.yaml``), kept out of the shared delegation config.

The shared :class:`DelegationConfig` (and its ``config_hash``) never sees these keys,
so the τ³ head's fingerprint is unaffected by anything set here. Every key that is
``null`` in the file keeps Pipecat's own default.

Selection: ``--pipecat-settings`` or ``FDH_PIPECAT_SETTINGS`` (a path, or a name
relative to the example's ``config/`` directory), default ``adapters/pipecat.yaml``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_SETTINGS = "adapters/pipecat.yaml"
SETTINGS_ENV = "FDH_PIPECAT_SETTINGS"


@dataclass(frozen=True, slots=True)
class VadSettings:
    """``VADParams`` overrides plus the ``VADProcessor`` idle timeout (``None`` keeps Pipecat's default).

    ``audio_idle_timeout`` is wall-clock: when no input audio arrives for that long while the
    user is speaking, Pipecat forces speech-stopped. ``0`` disables it, which a client that
    pauses its input between ticks (τ²) needs, so the end of a turn is measured in audio only.
    """

    confidence: float | None = None
    start_secs: float | None = None
    stop_secs: float | None = None
    min_volume: float | None = None
    audio_idle_timeout: float | None = None

    def overrides(self) -> dict[str, float]:
        """The ``VADParams`` keys that are set."""
        return {key: value for key, value in asdict(self).items() if value is not None and key != "audio_idle_timeout"}

    def processor_kwargs(self) -> dict[str, float]:
        """The ``VADProcessor`` keyword arguments that are set."""
        return {} if self.audio_idle_timeout is None else {"audio_idle_timeout": self.audio_idle_timeout}


@dataclass(frozen=True, slots=True)
class TurnSettings:
    """User-turn settings: Pipecat's default start strategies, speech-timeout stop, no Smart Turn."""

    user_speech_timeout: float | None = None
    user_turn_stop_timeout: float | None = None


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """Transport audio settings."""

    in_sample_rate: int = 16000
    out_sample_rate: int = 24000
    out_10ms_chunks: int = 10
    write_timeout_s: float = 120.0


@dataclass(frozen=True, slots=True)
class TtsSettings:
    """TTS service settings (``per_sentence``: one request per sentence, the output adapter's contract)."""

    synthesis_mode: str = "per_sentence"


@dataclass(frozen=True, slots=True)
class PipecatHeadSettings:
    """Everything the Pipecat head reads besides the shared delegation config."""

    vad: VadSettings = field(default_factory=VadSettings)
    turn: TurnSettings = field(default_factory=TurnSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    source: str = ""

    def summary(self) -> dict[str, Any]:
        """Logged in ``/health`` and ``session_start`` so a run shows which settings it used."""
        return {
            "settings_file": self.source,
            "vad": {**self.vad.overrides(), **self.vad.processor_kwargs()},
            "turn": {key: value for key, value in asdict(self.turn).items() if value is not None},
            "tts_synthesis_mode": self.tts.synthesis_mode,
            "smart_turn": False,
        }


class PipecatSettingsError(ValueError):
    """A settings file is missing or malformed."""


def resolve_settings_path(value: str | None = None) -> Path:
    """The settings file: ``value``, else ``FDH_PIPECAT_SETTINGS``, else the default (relative to ``config/``)."""
    raw = value or os.environ.get(SETTINGS_ENV, "") or DEFAULT_SETTINGS
    path = Path(raw)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / raw
    return path


def load_pipecat_settings(value: str | None = None) -> PipecatHeadSettings:
    """Load and validate a settings file."""
    path = resolve_settings_path(value)
    if not path.is_file():
        raise PipecatSettingsError(f"Pipecat head settings not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise PipecatSettingsError(f"{path}: expected a mapping")
    unknown = set(data) - {"vad", "turn", "audio", "tts"}
    if unknown:
        raise PipecatSettingsError(f"{path}: unknown keys {sorted(unknown)}")
    try:
        return PipecatHeadSettings(
            vad=VadSettings(**(data.get("vad") or {})),
            turn=TurnSettings(**(data.get("turn") or {})),
            audio=AudioSettings(**(data.get("audio") or {})),
            tts=TtsSettings(**(data.get("tts") or {})),
            source=str(path),
        )
    except TypeError as exc:
        raise PipecatSettingsError(f"{path}: {exc}") from exc
