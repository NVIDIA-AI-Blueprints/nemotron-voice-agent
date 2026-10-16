# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pipecat VAD, user-turn, transport-audio and TTS settings of the adapter.

Ported from FDH ``adapters/pipecat/settings.py`` (``a0ef50c``). Each profile names its settings
file in ``core/config/profiles.py`` (``eval`` -> ``settings/eval.yaml``, the Tau evaluation
settings; ``default`` -> ``settings/default.yaml``, the browser settings), and the runtime passes it
here explicitly. No environment variable selects it, so nothing outside the profile can change the
evaluation timing. With no argument the loader returns ``settings/eval.yaml``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

SETTINGS_DIR = Path(__file__).resolve().parent / "settings"
DEFAULT_SETTINGS = "eval.yaml"


@dataclass(frozen=True, slots=True)
class VadSettings:
    """``VADParams`` overrides plus the ``VADProcessor`` idle timeout (``None`` keeps Pipecat's default).

    ``audio_idle_timeout`` is wall-clock: when no input audio arrives for that long while the
    user is speaking, Pipecat forces speech-stopped. ``0`` disables it, which a client that
    pauses its input between ticks (Tau) needs, so the end of a turn is measured in audio only.
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
    """User-turn settings: Pipecat's default start strategies and a speech-timeout or Smart Turn stop.

    ``stt_ttfs_p99`` is the STT service's ``ttfs_p99_latency``: after the VAD stop, the turn
    waits up to ``stt_ttfs_p99 - stop_secs`` seconds for a finalized transcript. A finalized
    transcript (the force-EOU final) ends the wait at once, so a generous value costs nothing
    when ASR answers, and keeps a slow final from being split into a second user turn (v1
    always waited for its ASR final). ``None`` keeps Pipecat's NVIDIA default.

    ``wait_for_transcript`` is the stop strategy's option of that name. ``False`` ends a turn
    with no transcript (noise, a cough) after that wait, with an empty transcript, as v1 did
    at its ASR final; Pipecat's default (``True``) holds it open until ``user_turn_stop_timeout``.

    ``smart_turn: true`` ends turns with the repository's shared Smart Turn stop strategy
    (``examples/shared/pipeline_utils.py``, ``build_smart_turn_stop_strategies``) instead of the
    speech-timeout stop; ``user_speech_timeout`` then does not apply. Unset or ``false`` (the
    evaluation settings) keeps the speech-timeout stop.
    """

    user_speech_timeout: float | None = None
    user_turn_stop_timeout: float | None = None
    stt_ttfs_p99: float | None = None
    wait_for_transcript: bool | None = None
    smart_turn: bool | None = None


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """Transport audio settings.

    ``out_sample_rate`` ``None`` runs the output transport at the profile's TTS rate, so the
    agent's audio is resampled once, to the client's format in the serializer, as in v1.
    """

    in_sample_rate: int = 16000
    out_sample_rate: int | None = None
    out_10ms_chunks: int = 10
    write_timeout_s: float = 120.0


@dataclass(frozen=True, slots=True)
class TtsSettings:
    """TTS service settings (``per_sentence``: one request per sentence, as v1's output path)."""

    synthesis_mode: str = "per_sentence"


@dataclass(frozen=True, slots=True)
class PipecatSettings:
    """Everything the Pipecat adapter reads besides the FBV profile."""

    vad: VadSettings = field(default_factory=VadSettings)
    turn: TurnSettings = field(default_factory=TurnSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    tts: TtsSettings = field(default_factory=TtsSettings)
    source: str = ""

    def summary(self) -> dict[str, Any]:
        """Logged in ``session_start`` so a run shows which settings it used."""
        return {
            "settings_file": Path(self.source).name if self.source else "",
            "vad": {**self.vad.overrides(), **self.vad.processor_kwargs()},
            "turn": {
                key: value for key, value in asdict(self.turn).items() if value is not None and key != "smart_turn"
            },
            "tts_synthesis_mode": self.tts.synthesis_mode,
            "smart_turn": bool(self.turn.smart_turn),
        }


class PipecatSettingsError(ValueError):
    """A settings file is missing or malformed."""


def resolve_settings_path(value: str | None = None) -> Path:
    """The settings file: ``value`` (absolute, or relative to ``settings/``), else ``settings/eval.yaml``."""
    raw = value or DEFAULT_SETTINGS
    path = Path(raw)
    if not path.is_absolute() and not path.exists():
        path = SETTINGS_DIR / raw
    return path


def load_pipecat_settings(value: str | None = None) -> PipecatSettings:
    """Load and validate a settings file."""
    path = resolve_settings_path(value)
    if not path.is_file():
        raise PipecatSettingsError(f"Pipecat adapter settings not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise PipecatSettingsError(f"{path}: expected a mapping")
    unknown = set(data) - {"vad", "turn", "audio", "tts"}
    if unknown:
        raise PipecatSettingsError(f"{path}: unknown keys {sorted(unknown)}")
    try:
        return PipecatSettings(
            vad=VadSettings(**(data.get("vad") or {})),
            turn=TurnSettings(**(data.get("turn") or {})),
            audio=AudioSettings(**(data.get("audio") or {})),
            tts=TtsSettings(**(data.get("tts") or {})),
            source=str(path),
        )
    except TypeError as exc:
        raise PipecatSettingsError(f"{path}: {exc}") from exc
