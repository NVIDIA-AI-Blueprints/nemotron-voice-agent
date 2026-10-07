# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Realtime session object: GA schema, patch merging, public echo.

``session.update`` patches are validated and merged here, then resolved into an
immutable :class:`SessionSettings` the engine consumes. GA only (plan section
7.4): a patch carrying beta-only fields or beta format strings is rejected as a
whole with an error naming each field and its GA replacement, because applying
the rest would silently fall back to defaults. Unknown keys (``reasoning``,
future GA fields) are tolerated and echoed.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat, parse_format
from examples.frontend_backend_verdict.voice.errors import WireProtocolError

#: Beta-only top-level session fields and their GA replacements.
BETA_FIELDS: dict[str, str] = {
    "modalities": "output_modalities",
    "input_audio_format": "audio.input.format",
    "output_audio_format": "audio.output.format",
    "voice": "audio.output.voice",
    "turn_detection": "audio.input.turn_detection",
    "input_audio_transcription": "audio.input.transcription",
    "input_audio_noise_reduction": "audio.input.noise_reduction",
    "temperature": "(removed in GA; the LLMs take theirs from YAML)",
    "max_response_output_tokens": "max_output_tokens",
}
#: Beta format strings and their GA format objects.
BETA_FORMATS: dict[str, str] = {
    "pcm16": "{'type': 'audio/pcm', 'rate': 24000}",
    "g711_ulaw": "{'type': 'audio/pcmu'}",
    "g711_alaw": "{'type': 'audio/pcma'}",
}
#: ``semantic_vad`` eagerness -> silence duration used by the silence-based approximation (section 7.5).
SEMANTIC_EAGERNESS_SILENCE_MS: dict[str, int] = {"high": 300, "medium": 500, "auto": 500, "low": 800}


@dataclass(frozen=True, slots=True)
class TurnDetectionSettings:
    """Effective turn detection for one session."""

    mode: str  # "server_vad" | "manual"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = True
    interrupt_response: bool = True
    requested_type: str | None = "server_vad"
    eagerness: str | None = None

    @property
    def vad_enabled(self) -> bool:
        """Whether the server detects turns itself."""
        return self.mode == "server_vad"

    def effective(self) -> dict[str, Any]:
        """The server_vad parameters actually running."""
        return {
            "type": "server_vad",
            "threshold": self.threshold,
            "prefix_padding_ms": self.prefix_padding_ms,
            "silence_duration_ms": self.silence_duration_ms,
        }


@dataclass(frozen=True, slots=True)
class SessionDefaults:
    """Server-side defaults for everything a client may leave unset."""

    input_format: AudioFormat
    output_format: AudioFormat
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int
    honor_client_values: bool = True
    voice: str = "alloy"


@dataclass(frozen=True, slots=True)
class SessionSettings:
    """Immutable per-session settings resolved from the session view."""

    input_format: AudioFormat
    output_format: AudioFormat
    output_modality: str
    voice: str
    turn_detection: TurnDetectionSettings
    tools: tuple[dict[str, Any], ...] = ()
    tool_choice: Any = "auto"
    instructions: str = ""


@dataclass(frozen=True, slots=True)
class ApplyResult:
    """Outcome of one accepted ``session.update``."""

    settings: SessionSettings
    warnings: tuple[str, ...] = ()
    tools_changed: bool = False
    instructions_changed: bool = False
    formats_changed: bool = False
    semantic_vad: bool = False


@dataclass(frozen=True, slots=True)
class SessionPreview:
    """A validated ``session.update`` not applied yet: :meth:`commit` applies it (transactional updates)."""

    result: ApplyResult
    commit: Callable[[], ApplyResult]


@dataclass(slots=True)
class _State:
    input_format: AudioFormat
    output_format: AudioFormat
    output_modalities: list[str]
    voice: str
    turn_detection: TurnDetectionSettings
    turn_detection_echo: dict[str, Any] | None
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = "auto"
    instructions: str = ""
    transcription: dict[str, Any] | None = None
    noise_reduction: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _find_beta_fields(patch: dict[str, Any]) -> list[str]:
    problems = [f"session.{name} -> use {replacement}" for name, replacement in BETA_FIELDS.items() if name in patch]
    audio = patch.get("audio")
    if isinstance(audio, dict):
        for side in ("input", "output"):
            block = audio.get(side)
            if isinstance(block, dict) and isinstance(block.get("format"), str) and block["format"] in BETA_FORMATS:
                problems.append(
                    f"session.audio.{side}.format {block['format']!r} -> use {BETA_FORMATS[block['format']]}"
                )
    return problems


def _require(condition: bool, message: str, param: str) -> None:
    if not condition:
        raise WireProtocolError(message, code="invalid_value", param=param)


def _validate_tools(tools: Any) -> list[dict[str, Any]]:
    _require(isinstance(tools, list), "tools must be a list", "session.tools")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        param = f"session.tools[{index}]"
        _require(isinstance(tool, dict), "each tool must be an object", param)
        _require(tool.get("type", "function") == "function", "only function tools are supported", f"{param}.type")
        name = tool.get("name")
        _require(isinstance(name, str) and bool(name.strip()), "tool name must be a non-empty string", f"{param}.name")
        _require(name not in seen, f"duplicate tool name {name!r}", f"{param}.name")
        _require(isinstance(tool.get("description", ""), str), "description must be a string", f"{param}.description")
        parameters = tool.get("parameters", {"type": "object", "properties": {}})
        _require(isinstance(parameters, dict), "parameters must be a JSON schema object", f"{param}.parameters")
        seen.add(name)
        validated.append(copy.deepcopy(tool))
    return validated


class RealtimeSessionView:
    """The session object a client sees, and the settings it implies."""

    def __init__(self, *, session_id: str, model: str, defaults: SessionDefaults) -> None:
        """Start from the server defaults."""
        self.session_id = session_id
        self.model = model
        self._defaults = defaults
        turn = TurnDetectionSettings(
            mode="server_vad",
            threshold=defaults.threshold,
            prefix_padding_ms=defaults.prefix_padding_ms,
            silence_duration_ms=defaults.silence_duration_ms,
        )
        self._state = _State(
            input_format=defaults.input_format,
            output_format=defaults.output_format,
            output_modalities=["audio"],
            voice=defaults.voice,
            turn_detection=turn,
            turn_detection_echo=self._server_vad_echo(turn),
        )

    # -- public ------------------------------------------------------------

    @property
    def settings(self) -> SessionSettings:
        """The current immutable settings."""
        return self._settings_of(self._state)

    @staticmethod
    def _settings_of(state: _State) -> SessionSettings:
        return SessionSettings(
            input_format=state.input_format,
            output_format=state.output_format,
            output_modality=state.output_modalities[0],
            voice=state.voice,
            turn_detection=state.turn_detection,
            tools=tuple(copy.deepcopy(state.tools)),
            tool_choice=copy.deepcopy(state.tool_choice),
            instructions=state.instructions,
        )

    def public(self) -> dict[str, Any]:
        """The GA session object echoed in ``session.created`` / ``session.updated``."""
        state = self._state
        audio_input: dict[str, Any] = {
            "format": state.input_format.to_wire(),
            "transcription": copy.deepcopy(state.transcription),
            "noise_reduction": copy.deepcopy(state.noise_reduction),
            "turn_detection": copy.deepcopy(state.turn_detection_echo),
        }
        body: dict[str, Any] = {
            **copy.deepcopy(state.extra),
            "type": "realtime",
            "object": "realtime.session",
            "id": self.session_id,
            "model": self.model,
            "output_modalities": list(state.output_modalities),
            "instructions": state.instructions,
            "tools": copy.deepcopy(state.tools),
            "tool_choice": copy.deepcopy(state.tool_choice),
            "max_output_tokens": "inf",
            "audio": {
                "input": audio_input,
                "output": {"format": state.output_format.to_wire(), "voice": state.voice, "speed": 1.0},
            },
        }
        return body

    def apply(self, patch: dict[str, Any]) -> ApplyResult:
        """Validate and merge one ``session.update`` patch; raises on rejection (nothing applied)."""
        return self.preview(patch).commit()

    def preview(self, patch: dict[str, Any]) -> SessionPreview:
        """Validate ``patch`` against a copy of the state; nothing changes until ``commit()``.

        Raises on rejection. ``commit()`` fails if another update was applied in between.
        """
        beta = _find_beta_fields(patch)
        if beta:
            raise WireProtocolError(
                "beta Realtime session fields are not supported; this server speaks the GA schema only: "
                + "; ".join(beta),
                code="invalid_value",
                param="session",
            )
        session_type = patch.get("type", "realtime")
        _require(session_type == "realtime", f"session.type {session_type!r} is not supported", "session.type")

        new = copy.deepcopy(self._state)
        warnings: list[str] = []
        semantic = False
        if "output_modalities" in patch:
            modalities = patch["output_modalities"]
            _require(
                isinstance(modalities, list) and len(modalities) == 1 and modalities[0] in ("audio", "text"),
                "output_modalities must be ['audio'] or ['text']",
                "session.output_modalities",
            )
            new.output_modalities = list(modalities)
        if "instructions" in patch:
            _require(isinstance(patch["instructions"], str), "instructions must be a string", "session.instructions")
            new.instructions = patch["instructions"]
        if "tools" in patch:
            new.tools = _validate_tools(patch["tools"])
        if "tool_choice" in patch:
            new.tool_choice = patch["tool_choice"]
            if patch["tool_choice"] != "auto":
                warnings.append(f"tool_choice {patch['tool_choice']!r} is treated as 'auto'")
        audio = patch.get("audio")
        if audio is not None:
            _require(isinstance(audio, dict), "audio must be an object", "session.audio")
            semantic = self._apply_audio(audio, new, warnings)
        for key, value in patch.items():
            if key not in ("type", "output_modalities", "instructions", "tools", "tool_choice", "audio"):
                new.extra[key] = copy.deepcopy(value)
                if key not in ("model", "reasoning", "include", "tracing", "truncation", "prompt"):
                    warnings.append(f"session.{key} is not used by this server")

        old = self._state
        result = ApplyResult(
            settings=self._settings_of(new),
            warnings=tuple(warnings),
            tools_changed=old.tools != new.tools,
            instructions_changed=old.instructions != new.instructions,
            formats_changed=(old.input_format, old.output_format) != (new.input_format, new.output_format),
            semantic_vad=semantic,
        )

        def commit() -> ApplyResult:
            if self._state is not old:
                raise RuntimeError("the session changed since this preview was made")
            self._state = new
            return result

        return SessionPreview(result=result, commit=commit)

    # -- helpers -----------------------------------------------------------

    def _apply_audio(self, audio: dict[str, Any], new: _State, warnings: list[str]) -> bool:
        semantic = False
        audio_in = audio.get("input")
        if audio_in is not None:
            _require(isinstance(audio_in, dict), "audio.input must be an object", "session.audio.input")
            if audio_in.get("format") is not None:
                new.input_format = self._format(audio_in["format"], "session.audio.input.format")
            if "transcription" in audio_in:
                new.transcription = copy.deepcopy(audio_in["transcription"])
            if "noise_reduction" in audio_in:
                new.noise_reduction = copy.deepcopy(audio_in["noise_reduction"])
            if "turn_detection" in audio_in:
                semantic = self._apply_turn_detection(audio_in["turn_detection"], new, warnings)
        audio_out = audio.get("output")
        if audio_out is not None:
            _require(isinstance(audio_out, dict), "audio.output must be an object", "session.audio.output")
            if audio_out.get("format") is not None:
                new.output_format = self._format(audio_out["format"], "session.audio.output.format")
            if "voice" in audio_out:
                voice = audio_out["voice"]
                if isinstance(voice, dict):
                    voice = voice.get("id", "")
                _require(isinstance(voice, str), "voice must be a string", "session.audio.output.voice")
                new.voice = voice
        return semantic

    @staticmethod
    def _format(value: Any, param: str) -> AudioFormat:
        try:
            return parse_format(value, param=param)
        except ValueError as exc:
            raise WireProtocolError(str(exc), code="invalid_value", param=param) from exc

    def _apply_turn_detection(self, value: Any, new: _State, warnings: list[str]) -> bool:
        param = "session.audio.input.turn_detection"
        defaults = self._defaults
        if value is None:
            new.turn_detection = TurnDetectionSettings(
                mode="manual",
                threshold=defaults.threshold,
                prefix_padding_ms=defaults.prefix_padding_ms,
                silence_duration_ms=defaults.silence_duration_ms,
                create_response=False,
                interrupt_response=False,
                requested_type=None,
            )
            new.turn_detection_echo = None
            return False
        _require(isinstance(value, dict), "turn_detection must be an object or null", param)
        kind = value.get("type", "server_vad")
        create_response = value.get("create_response", True)
        interrupt_response = value.get("interrupt_response", True)
        _require(isinstance(create_response, bool), "create_response must be a boolean", f"{param}.create_response")
        _require(
            isinstance(interrupt_response, bool), "interrupt_response must be a boolean", f"{param}.interrupt_response"
        )
        if kind == "server_vad":
            threshold, prefix, silence = defaults.threshold, defaults.prefix_padding_ms, defaults.silence_duration_ms
            if defaults.honor_client_values:
                threshold = value.get("threshold", threshold)
                prefix = value.get("prefix_padding_ms", prefix)
                silence = value.get("silence_duration_ms", silence)
            _require(
                isinstance(threshold, int | float) and 0.0 <= float(threshold) <= 1.0,
                "threshold must be in [0, 1]",
                f"{param}.threshold",
            )
            _require(
                isinstance(prefix, int) and not isinstance(prefix, bool) and prefix >= 0,
                "prefix_padding_ms must be a non-negative integer",
                f"{param}.prefix_padding_ms",
            )
            _require(
                isinstance(silence, int) and not isinstance(silence, bool) and silence > 0,
                "silence_duration_ms must be a positive integer",
                f"{param}.silence_duration_ms",
            )
            if value.get("idle_timeout_ms") is not None:
                warnings.append("turn_detection.idle_timeout_ms is ignored")
            turn = TurnDetectionSettings(
                mode="server_vad",
                threshold=float(threshold),
                prefix_padding_ms=int(prefix),
                silence_duration_ms=int(silence),
                create_response=create_response,
                interrupt_response=interrupt_response,
            )
            new.turn_detection = turn
            new.turn_detection_echo = self._server_vad_echo(turn)
            return False
        if kind == "semantic_vad":
            eagerness = value.get("eagerness", "auto")
            _require(
                eagerness in SEMANTIC_EAGERNESS_SILENCE_MS,
                f"eagerness must be one of {sorted(SEMANTIC_EAGERNESS_SILENCE_MS)}",
                f"{param}.eagerness",
            )
            turn = TurnDetectionSettings(
                mode="server_vad",
                threshold=defaults.threshold,
                prefix_padding_ms=defaults.prefix_padding_ms,
                silence_duration_ms=SEMANTIC_EAGERNESS_SILENCE_MS[eagerness],
                create_response=create_response,
                interrupt_response=interrupt_response,
                requested_type="semantic_vad",
                eagerness=eagerness,
            )
            new.turn_detection = turn
            echo = copy.deepcopy(value)
            echo.setdefault("type", "semantic_vad")
            echo.setdefault("eagerness", eagerness)
            echo.setdefault("create_response", create_response)
            echo.setdefault("interrupt_response", interrupt_response)
            echo["x_nvidia_effective"] = turn.effective()
            new.turn_detection_echo = echo
            warnings.append(
                f"semantic_vad is approximated by silence-based server_vad: eagerness {eagerness} -> "
                f"silence_duration_ms {turn.silence_duration_ms}; turns end on silence, not on semantic completeness"
            )
            return True
        raise WireProtocolError(f"turn_detection.type {kind!r} is not supported", code="invalid_value", param=param)

    @staticmethod
    def _server_vad_echo(turn: TurnDetectionSettings) -> dict[str, Any]:
        return {
            "type": "server_vad",
            "threshold": turn.threshold,
            "prefix_padding_ms": turn.prefix_padding_ms,
            "silence_duration_ms": turn.silence_duration_ms,
            "idle_timeout_ms": None,
            "create_response": turn.create_response,
            "interrupt_response": turn.interrupt_response,
        }
