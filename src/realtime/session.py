# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict per-connection state for the canonical OpenAI Realtime session."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Any

from realtime.protocol import (
    RealtimeProtocolError,
    immutable_field,
    invalid_type,
    invalid_value,
    new_realtime_id,
    unsupported_capability,
)
from realtime.tool_schema import compile_tool_arguments_validator, validate_tool_collection_bounds

_SESSION_FIELDS = frozenset(
    {
        "type",
        "audio",
        "include",
        "instructions",
        "max_output_tokens",
        "model",
        "output_modalities",
        "parallel_tool_calls",
        "prompt",
        "reasoning",
        "tool_choice",
        "tools",
        "tracing",
        "truncation",
    }
)
_AUDIO_FIELDS = frozenset({"input", "output"})
_AUDIO_INPUT_FIELDS = frozenset({"format", "noise_reduction", "transcription", "turn_detection"})
_AUDIO_OUTPUT_FIELDS = frozenset({"format", "speed", "voice"})
_FUNCTION_TOOL_FIELDS = frozenset({"type", "name", "description", "parameters"})
_MCP_TOOL_FIELDS = frozenset(
    {
        "type",
        "server_label",
        "server_url",
        "connector_id",
        "tunnel_id",
        "authorization",
        "headers",
        "allowed_tools",
        "require_approval",
        "server_description",
        "allowed_callers",
        "defer_loading",
    }
)
_MCP_CONNECTOR_IDS = frozenset(
    {
        "connector_dropbox",
        "connector_gmail",
        "connector_googlecalendar",
        "connector_googledrive",
        "connector_microsoftteams",
        "connector_outlookcalendar",
        "connector_outlookemail",
        "connector_sharepoint",
    }
)
_MCP_CALLERS = frozenset({"direct", "programmatic"})
_MCP_TOOL_FILTER_FIELDS = frozenset({"read_only", "tool_names"})
_MCP_APPROVAL_FILTER_FIELDS = frozenset({"always", "never"})
_PROMPT_FIELDS = frozenset({"id", "variables", "version"})
_TRUNCATION_FIELDS = frozenset({"type", "retention_ratio", "token_limits"})
_TRUNCATION_TOKEN_LIMIT_FIELDS = frozenset({"post_instructions"})
_SERVER_VAD_OPTIONS = frozenset(
    {
        "type",
        "threshold",
        "prefix_padding_ms",
        "silence_duration_ms",
        "create_response",
        "interrupt_response",
        "idle_timeout_ms",
    }
)
_SEMANTIC_VAD_OPTIONS = frozenset(
    {
        "type",
        "eagerness",
        "create_response",
        "interrupt_response",
    }
)
_TURN_DETECTION_OPTIONS = _SERVER_VAD_OPTIONS | _SEMANTIC_VAD_OPTIONS

_DEFAULT_SERVER_VAD_THRESHOLD = 0.5
_DEFAULT_SERVER_VAD_PREFIX_PADDING_MS = 300
_DEFAULT_SERVER_VAD_SILENCE_DURATION_MS = 500


def _default_turn_detection_config(
    typ: str,
    *,
    supported_options: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Return the canonical defaults for one supported VAD mode."""
    if typ == "server_vad":
        config: dict[str, Any] = {
            "type": "server_vad",
            "threshold": _DEFAULT_SERVER_VAD_THRESHOLD,
            "prefix_padding_ms": _DEFAULT_SERVER_VAD_PREFIX_PADDING_MS,
            "silence_duration_ms": _DEFAULT_SERVER_VAD_SILENCE_DURATION_MS,
            "create_response": True,
            "interrupt_response": True,
            "idle_timeout_ms": None,
        }
    elif typ == "semantic_vad":
        config = {
            "type": "semantic_vad",
            "eagerness": "auto",
            "create_response": True,
            "interrupt_response": True,
        }
    else:
        config = {"type": typ}
    if supported_options is None:
        return config
    return {name: value for name, value in config.items() if name in supported_options}


@dataclass(frozen=True, slots=True)
class AudioFormatCapability:
    """One audio format that the configured transport/backend can honor."""

    type: str
    rate: int | None = None


_PCM24 = AudioFormatCapability("audio/pcm", 24000)


@dataclass(frozen=True, slots=True)
class RealtimeSessionCapabilities:
    """Explicit capability contract used while validating session updates.

    Empty capability sets reject the associated optional feature. ``voices``
    set to ``None`` means any voice ID is accepted; production gateways should
    pass the exact trusted catalog entries they can honor.
    """

    input_formats: frozenset[AudioFormatCapability] = field(default_factory=lambda: frozenset({_PCM24}))
    output_formats: frozenset[AudioFormatCapability] = field(default_factory=lambda: frozenset({_PCM24}))
    voices: frozenset[str] | None = None
    supports_custom_voices: bool = False
    supports_output_speed: bool = False
    turn_detection_types: frozenset[str] = field(default_factory=lambda: frozenset({"server_vad"}))
    turn_detection_options: frozenset[str] = field(default_factory=lambda: _TURN_DETECTION_OPTIONS)
    turn_detection_create_response_values: frozenset[bool] = field(default_factory=lambda: frozenset({False, True}))
    turn_detection_interrupt_response_values: frozenset[bool] = field(default_factory=lambda: frozenset({False, True}))
    default_turn_detection_type: str = "server_vad"
    supports_manual_input: bool = False
    noise_reduction_types: frozenset[str] = field(default_factory=frozenset)
    input_transcription_models: frozenset[str] = field(default_factory=frozenset)
    input_transcription_language_aliases: tuple[tuple[str, str], ...] = ()
    supports_input_transcription_disable: bool = False
    supports_empty_instructions: bool = True
    function_tools: bool = True
    mcp_tools: bool = False
    trusted_function_tools: frozenset[str] = field(default_factory=frozenset)
    parallel_tool_calls: bool = True
    sequential_tool_calls: bool = False
    prompt: bool = False
    reasoning: bool = False
    include_fields: frozenset[str] = field(default_factory=frozenset)
    tracing: bool = False
    truncation: bool = False


def _merge_object(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a validated partial object without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_object(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _reject_unknown_fields(value: dict[str, Any], allowed: frozenset[str], *, param: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        field_name = unknown[0]
        dotted = f"{param}.{field_name}" if param else field_name
        raise RealtimeProtocolError(
            message=f"Unknown parameter: {dotted}",
            code="unknown_parameter",
            param=dotted,
        )


def _require_object(value: Any, *, param: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid_type(f"{param} must be an object", param=param)
    return value


def _require_string(value: Any, *, param: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise invalid_type(f"{param} must be a string", param=param)
    if not allow_empty and not value:
        raise invalid_value(f"{param} must not be empty", param=param)
    return value


def _validate_format(
    value: Any,
    *,
    param: str,
    supported: frozenset[AudioFormatCapability],
) -> dict[str, Any]:
    fmt = _require_object(value, param=param)
    typ = _require_string(fmt.get("type"), param=f"{param}.type")
    allowed = frozenset({"type", "rate"}) if typ == "audio/pcm" else frozenset({"type"})
    _reject_unknown_fields(fmt, allowed, param=param)

    rate: int | None = None
    if typ == "audio/pcm":
        # OpenAI Realtime schema makes the PCM rate optional, but the
        # effective wire rate is always 24 kHz when it is omitted. Normalize
        # that default before capability matching and publication so every
        # downstream consumer sees the exact effective format.
        rate = fmt.get("rate", 24000)
        if isinstance(rate, bool) or not isinstance(rate, int):
            raise invalid_type(f"{param}.rate must be an integer", param=f"{param}.rate")
    capability = AudioFormatCapability(typ, rate)
    if capability not in supported:
        rendered = f"{typ} at {rate} Hz" if rate is not None else typ
        raise unsupported_capability(f"Audio format {rendered} is not available", param=param)
    normalized = copy.deepcopy(fmt)
    if typ == "audio/pcm":
        normalized["rate"] = rate
    return normalized


def _normalize_voice(value: Any, *, param: str, capabilities: RealtimeSessionCapabilities) -> str | dict[str, str]:
    if isinstance(value, str):
        voice = _require_string(value, param=param)
        if capabilities.voices is not None and voice not in capabilities.voices:
            raise unsupported_capability(f"Voice {voice!r} is not available", param=param)
        return voice
    if isinstance(value, dict):
        _reject_unknown_fields(value, frozenset({"id"}), param=param)
        voice_id = _require_string(value.get("id"), param=f"{param}.id")
        if not capabilities.supports_custom_voices:
            raise unsupported_capability("Custom voice references are not available", param=param)
        return {"id": voice_id}
    raise invalid_type(f"{param} must be a voice ID string or custom voice object", param=param)


def _normalize_transcription_language(
    value: Any,
    *,
    param: str,
    capabilities: RealtimeSessionCapabilities,
) -> str:
    language = _require_string(value, param=param)
    aliases = {alias.lower(): canonical for alias, canonical in capabilities.input_transcription_language_aliases}
    canonical = aliases.get(language.lower())
    if canonical is None:
        raise unsupported_capability(
            f"Input transcription language {language!r} is not available",
            param=param,
        )
    return canonical


def _validate_turn_detection(
    value: Any,
    *,
    current: dict[str, Any] | None,
    capabilities: RealtimeSessionCapabilities,
) -> dict[str, Any] | None:
    param = "session.audio.input.turn_detection"
    if value is None:
        if not capabilities.supports_manual_input:
            raise unsupported_capability("Manual input commit mode is not available", param=param)
        return None
    patch = _require_object(value, param=param)
    typ = _require_string(patch.get("type"), param=f"{param}.type")
    if typ not in capabilities.turn_detection_types:
        raise unsupported_capability(f"Turn detection type {typ!r} is not available", param=f"{param}.type")
    canonical_options = _SERVER_VAD_OPTIONS if typ == "server_vad" else _SEMANTIC_VAD_OPTIONS
    _reject_unknown_fields(patch, canonical_options, param=param)
    if isinstance(current, dict) and current.get("type") == typ:
        config = _merge_object(current, patch)
    else:
        config = _merge_object(
            _default_turn_detection_config(
                typ,
                supported_options=capabilities.turn_detection_options,
            ),
            patch,
        )
    if typ == "server_vad" and "idle_timeout_ms" in config:
        idle_timeout_ms = config["idle_timeout_ms"]
        if idle_timeout_ms is not None:
            if isinstance(idle_timeout_ms, bool) or not isinstance(idle_timeout_ms, int):
                raise invalid_type(
                    f"{param}.idle_timeout_ms must be an integer or null",
                    param=f"{param}.idle_timeout_ms",
                )
            if idle_timeout_ms <= 0:
                raise invalid_value(
                    f"{param}.idle_timeout_ms must be a positive integer",
                    param=f"{param}.idle_timeout_ms",
                )
    unsupported = sorted(set(config) - capabilities.turn_detection_options)
    if unsupported:
        name = unsupported[0]
        raise unsupported_capability(
            f"Turn detection option {name!r} is not available on this endpoint",
            param=f"{param}.{name}",
        )

    for name in ("create_response", "interrupt_response"):
        if name in config and not isinstance(config[name], bool):
            raise invalid_type(f"{param}.{name} must be a boolean", param=f"{param}.{name}")
    if (
        "create_response" in config
        and config["create_response"] not in capabilities.turn_detection_create_response_values
    ):
        raise unsupported_capability(
            "Disabling automatic responses is not available on this endpoint",
            param=f"{param}.create_response",
        )
    if (
        "interrupt_response" in config
        and config["interrupt_response"] not in capabilities.turn_detection_interrupt_response_values
    ):
        raise unsupported_capability(
            "Disabling automatic response interruption is not available on this endpoint",
            param=f"{param}.interrupt_response",
        )

    if typ == "server_vad":
        if "threshold" in config:
            threshold = config["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, int | float):
                raise invalid_type(f"{param}.threshold must be a number", param=f"{param}.threshold")
            if (isinstance(threshold, float) and not math.isfinite(threshold)) or threshold < 0 or threshold > 1:
                raise invalid_value(
                    f"{param}.threshold must be between 0 and 1",
                    param=f"{param}.threshold",
                )
        for name in ("prefix_padding_ms", "silence_duration_ms"):
            if name not in config:
                continue
            duration_ms = config[name]
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
                raise invalid_type(f"{param}.{name} must be an integer", param=f"{param}.{name}")
            if duration_ms < 0:
                raise invalid_value(f"{param}.{name} must be nonnegative", param=f"{param}.{name}")
    elif "eagerness" in config:
        eagerness = config["eagerness"]
        if not isinstance(eagerness, str):
            raise invalid_type(f"{param}.eagerness must be a string", param=f"{param}.eagerness")
        if eagerness != "auto":
            raise unsupported_capability(
                "Only semantic VAD eagerness 'auto' is available on this endpoint",
                param=f"{param}.eagerness",
            )
    return copy.deepcopy(config)


def _validate_audio_input(
    value: Any,
    *,
    current: dict[str, Any],
    capabilities: RealtimeSessionCapabilities,
) -> dict[str, Any]:
    param = "session.audio.input"
    patch = _require_object(value, param=param)
    _reject_unknown_fields(patch, _AUDIO_INPUT_FIELDS, param=param)
    candidate = _merge_object(current, patch)

    if "format" in patch:
        candidate["format"] = _validate_format(
            patch["format"],
            param=f"{param}.format",
            supported=capabilities.input_formats,
        )
    if "noise_reduction" in patch:
        noise_reduction = patch["noise_reduction"]
        if noise_reduction is not None:
            config = _require_object(noise_reduction, param=f"{param}.noise_reduction")
            _reject_unknown_fields(config, frozenset({"type"}), param=f"{param}.noise_reduction")
            typ = _require_string(config.get("type"), param=f"{param}.noise_reduction.type")
            if typ not in capabilities.noise_reduction_types:
                raise unsupported_capability(
                    f"Input noise reduction type {typ!r} is not available",
                    param=f"{param}.noise_reduction.type",
                )
            candidate["noise_reduction"] = copy.deepcopy(config)
        else:
            candidate["noise_reduction"] = None
    if "transcription" in patch:
        transcription = patch["transcription"]
        if transcription is not None:
            config = _require_object(transcription, param=f"{param}.transcription")
            _reject_unknown_fields(
                config,
                frozenset({"delay", "keywords", "language", "languages", "model", "prompt"}),
                param=f"{param}.transcription",
            )
            model_param = f"{param}.transcription.model"
            current_config = current.get("transcription")
            merged_config = candidate.get("transcription")
            if not isinstance(merged_config, dict):
                raise invalid_type(f"{param}.transcription must be an object", param=f"{param}.transcription")
            model = _require_string(merged_config.get("model"), param=model_param)
            current_model = current_config.get("model") if isinstance(current_config, dict) else None
            if model != current_model and model not in capabilities.input_transcription_models:
                raise unsupported_capability(
                    f"Input transcription model {model!r} is not available",
                    param=model_param,
                )
            normalized = copy.deepcopy(merged_config)
            normalized["model"] = model
            if "language" in config:
                normalized["language"] = _normalize_transcription_language(
                    config["language"],
                    param=f"{param}.transcription.language",
                    capabilities=capabilities,
                )
            if "prompt" in config:
                _require_string(
                    config["prompt"],
                    param=f"{param}.transcription.prompt",
                    allow_empty=True,
                )
                if not isinstance(current_config, dict) or config["prompt"] != current_config.get("prompt"):
                    raise unsupported_capability(
                        "Input transcription prompts are not available on this backend",
                        param=f"{param}.transcription.prompt",
                    )
                normalized["prompt"] = config["prompt"]
            if "delay" in config:
                delay = config["delay"]
                if not isinstance(delay, str):
                    raise invalid_type(
                        f"{param}.transcription.delay must be a string",
                        param=f"{param}.transcription.delay",
                    )
                if delay not in {"minimal", "low", "medium", "high", "xhigh"}:
                    raise invalid_value(
                        f"{param}.transcription.delay is invalid",
                        param=f"{param}.transcription.delay",
                    )
                raise unsupported_capability(
                    "Input transcription delay is not configurable on this backend",
                    param=f"{param}.transcription.delay",
                )
            for name in ("keywords", "languages"):
                if name not in config:
                    continue
                values = config[name]
                if not isinstance(values, list):
                    raise invalid_type(
                        f"{param}.transcription.{name} must be an array",
                        param=f"{param}.transcription.{name}",
                    )
                for index, entry in enumerate(values):
                    if not isinstance(entry, str) or not entry:
                        raise invalid_value(
                            f"{param}.transcription.{name}[{index}] must be a non-empty string",
                            param=f"{param}.transcription.{name}[{index}]",
                        )
                raise unsupported_capability(
                    f"Input transcription {name} are not configurable on this backend",
                    param=f"{param}.transcription.{name}",
                )
            candidate["transcription"] = normalized
        else:
            if current.get("transcription") is not None and not capabilities.supports_input_transcription_disable:
                raise unsupported_capability(
                    "Input transcription cannot be disabled on this backend",
                    param=f"{param}.transcription",
                )
            candidate["transcription"] = None
    if "turn_detection" in patch:
        candidate["turn_detection"] = _validate_turn_detection(
            patch["turn_detection"],
            current=current.get("turn_detection") if isinstance(current.get("turn_detection"), dict) else None,
            capabilities=capabilities,
        )
    return candidate


def _validate_speed(value: Any, *, param: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise invalid_type(f"{param} must be a number", param=param)
    speed = float(value)
    if not math.isfinite(speed) or speed < 0.25 or speed > 1.5:
        raise invalid_value(f"{param} must be between 0.25 and 1.5", param=param)
    return speed


def _validate_max_output_tokens(value: Any, *, param: str) -> int | str:
    """Validate the canonical per-response output-token ceiling."""
    if value == "inf":
        return "inf"
    if isinstance(value, bool) or not isinstance(value, int):
        raise invalid_type(f"{param} must be an integer or 'inf'", param=param)
    if value < 1 or value > 4096:
        raise invalid_value(f"{param} must be between 1 and 4096", param=param)
    return value


def _validate_truncation(value: Any, *, param: str = "session.truncation") -> str | dict[str, Any]:
    """Validate the native Realtime context-truncation strategy."""
    if isinstance(value, str):
        if value not in {"auto", "disabled"}:
            raise invalid_value(f"{param} must be 'auto', 'disabled', or a retention-ratio object", param=param)
        return value

    config = _require_object(value, param=param)
    _reject_unknown_fields(config, _TRUNCATION_FIELDS, param=param)
    if config.get("type") != "retention_ratio":
        raise invalid_value(f"{param}.type must be 'retention_ratio'", param=f"{param}.type")
    ratio = config.get("retention_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, int | float):
        raise invalid_type(f"{param}.retention_ratio must be a number", param=f"{param}.retention_ratio")
    ratio = float(ratio)
    if not math.isfinite(ratio) or ratio < 0.0 or ratio > 1.0:
        raise invalid_value(
            f"{param}.retention_ratio must be between 0.0 and 1.0",
            param=f"{param}.retention_ratio",
        )

    normalized: dict[str, Any] = {"type": "retention_ratio", "retention_ratio": ratio}
    if "token_limits" in config:
        limits = _require_object(config["token_limits"], param=f"{param}.token_limits")
        _reject_unknown_fields(limits, _TRUNCATION_TOKEN_LIMIT_FIELDS, param=f"{param}.token_limits")
        normalized_limits: dict[str, Any] = {}
        if "post_instructions" in limits:
            post_instructions = limits["post_instructions"]
            if isinstance(post_instructions, bool) or not isinstance(post_instructions, int):
                raise invalid_type(
                    f"{param}.token_limits.post_instructions must be an integer",
                    param=f"{param}.token_limits.post_instructions",
                )
            if post_instructions < 1:
                raise invalid_value(
                    f"{param}.token_limits.post_instructions must be positive",
                    param=f"{param}.token_limits.post_instructions",
                )
            normalized_limits["post_instructions"] = post_instructions
        normalized["token_limits"] = normalized_limits
    return normalized


def _validate_audio_output(
    value: Any,
    *,
    current: dict[str, Any],
    capabilities: RealtimeSessionCapabilities,
    response_in_progress: bool,
    output_audio_started: bool,
    locked_output_voice: str | dict[str, str] | None = None,
    param: str = "session.audio.output",
    allowed_fields: frozenset[str] = _AUDIO_OUTPUT_FIELDS,
) -> dict[str, Any]:
    patch = _require_object(value, param=param)
    _reject_unknown_fields(patch, allowed_fields, param=param)
    candidate = _merge_object(current, patch)

    if "format" in patch:
        candidate["format"] = _validate_format(
            patch["format"],
            param=f"{param}.format",
            supported=capabilities.output_formats,
        )
    if "voice" in patch:
        voice = _normalize_voice(patch["voice"], param=f"{param}.voice", capabilities=capabilities)
        if output_audio_started and voice != locked_output_voice:
            raise immutable_field(
                f"{param}.voice cannot change after output audio has started",
                param=f"{param}.voice",
            )
        candidate["voice"] = voice
    if "speed" in patch:
        speed = _validate_speed(patch["speed"], param=f"{param}.speed")
        if speed != current.get("speed") and not capabilities.supports_output_speed:
            raise unsupported_capability(
                "Output speech speed is not configurable on this backend",
                param=f"{param}.speed",
            )
        if speed != current.get("speed") and response_in_progress:
            raise immutable_field(
                "session.audio.output.speed can change only between responses",
                param=f"{param}.speed",
            )
        candidate["speed"] = speed
    return candidate


def validate_session_tools_bounds(value: Any, *, param: str = "session.tools") -> None:
    """Reject an oversized tools array before copying or compiling its schemas."""
    if not isinstance(value, list):
        raise invalid_type(f"{param} must be an array", param=param)
    try:
        validate_tool_collection_bounds(value, param=param)
    except ValueError as exc:
        raise RealtimeProtocolError(
            message=str(exc),
            code="invalid_tool_schema",
            param=param,
        ) from exc


def _validate_mcp_tool_names(value: Any, *, param: str) -> list[str]:
    if not isinstance(value, list):
        raise invalid_type(f"{param} must be an array of strings", param=param)
    names: list[str] = []
    seen: set[str] = set()
    for index, raw_name in enumerate(value):
        name = _require_string(raw_name, param=f"{param}[{index}]")
        if name in seen:
            raise invalid_value(f"Duplicate MCP tool name {name!r}", param=f"{param}[{index}]")
        seen.add(name)
        names.append(name)
    return names


def _validate_mcp_tool_filter(value: Any, *, param: str) -> dict[str, Any]:
    tool_filter = _require_object(value, param=param)
    _reject_unknown_fields(tool_filter, _MCP_TOOL_FILTER_FIELDS, param=param)
    normalized: dict[str, Any] = {}
    if "read_only" in tool_filter:
        read_only = tool_filter["read_only"]
        if not isinstance(read_only, bool):
            raise invalid_type(f"{param}.read_only must be a boolean", param=f"{param}.read_only")
        normalized["read_only"] = read_only
    if "tool_names" in tool_filter:
        normalized["tool_names"] = _validate_mcp_tool_names(
            tool_filter["tool_names"],
            param=f"{param}.tool_names",
        )
    return normalized


def _validate_mcp_allowed_tools(value: Any, *, param: str) -> list[str] | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return _validate_mcp_tool_names(value, param=param)
    return _validate_mcp_tool_filter(value, param=param)


def _validate_mcp_require_approval(value: Any, *, param: str) -> str | dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value not in {"always", "never"}:
            raise invalid_value(f"{param} must be always, never, or an approval filter", param=param)
        return value
    approval_filter = _require_object(value, param=param)
    _reject_unknown_fields(approval_filter, _MCP_APPROVAL_FILTER_FIELDS, param=param)
    if not approval_filter:
        raise invalid_value(f"{param} must include an always or never filter", param=param)
    normalized = {
        rule: _validate_mcp_tool_filter(rule_filter, param=f"{param}.{rule}")
        for rule, rule_filter in approval_filter.items()
    }
    always = normalized.get("always")
    never = normalized.get("never")
    if always is not None and never is not None:
        always_names = set(always.get("tool_names", []))
        never_names = set(never.get("tool_names", []))
        overlapping_names = sorted(always_names & never_names)
        read_only_overlaps = (
            "read_only" not in always or "read_only" not in never or always["read_only"] == never["read_only"]
        )
        if overlapping_names and read_only_overlaps:
            raise invalid_value(
                f"MCP tool {overlapping_names[0]!r} cannot match both always and never approval filters",
                param=f"{param}.never.tool_names",
            )
    return normalized


def _validate_mcp_tool(tool: dict[str, Any], *, param: str) -> dict[str, Any]:
    _reject_unknown_fields(tool, _MCP_TOOL_FIELDS, param=param)
    _require_string(tool.get("server_label"), param=f"{param}.server_label")

    endpoint_fields = [field_name for field_name in ("server_url", "connector_id", "tunnel_id") if field_name in tool]
    if len(endpoint_fields) > 1:
        raise invalid_value(
            f"{param} must provide exactly one of server_url, connector_id, or tunnel_id",
            param=param,
        )
    if not endpoint_fields:
        reference_fields = {"type", "server_label"}
        if set(tool) != reference_fields:
            extra_field = sorted(set(tool) - reference_fields)[0]
            raise invalid_value(
                f"{param} without an endpoint must contain only type and server_label",
                param=f"{param}.{extra_field}",
            )
        return copy.deepcopy(tool)

    endpoint_field = endpoint_fields[0]
    endpoint_value = _require_string(tool[endpoint_field], param=f"{param}.{endpoint_field}")
    if endpoint_field == "connector_id" and endpoint_value not in _MCP_CONNECTOR_IDS:
        raise invalid_value(
            f"MCP connector {endpoint_value!r} is not supported",
            param=f"{param}.connector_id",
        )

    normalized = copy.deepcopy(tool)
    if "authorization" in tool:
        normalized["authorization"] = _require_string(
            tool["authorization"],
            param=f"{param}.authorization",
        )
    if "headers" in tool and tool["headers"] is not None:
        headers = _require_object(tool["headers"], param=f"{param}.headers")
        normalized_headers: dict[str, str] = {}
        canonical_header_names: set[str] = set()
        for header_name, header_value in headers.items():
            name_param = f"{param}.headers.{header_name}"
            name = _require_string(header_name, param=name_param)
            if not isinstance(header_value, str):
                raise invalid_type(f"{name_param} must be a string", param=name_param)
            canonical_name = name.casefold()
            if canonical_name in canonical_header_names:
                raise invalid_value(
                    f"Duplicate MCP header name {name!r}",
                    param=name_param,
                )
            canonical_header_names.add(canonical_name)
            normalized_headers[name] = header_value
        if tool.get("authorization") is not None and "authorization" in canonical_header_names:
            raise invalid_value(
                "MCP authorization cannot be provided in both authorization and headers",
                param=f"{param}.headers",
            )
        normalized["headers"] = normalized_headers
    if "allowed_tools" in tool:
        normalized["allowed_tools"] = _validate_mcp_allowed_tools(
            tool["allowed_tools"],
            param=f"{param}.allowed_tools",
        )
    if "require_approval" in tool:
        normalized["require_approval"] = _validate_mcp_require_approval(
            tool["require_approval"],
            param=f"{param}.require_approval",
        )
    if "server_description" in tool:
        normalized["server_description"] = _require_string(
            tool["server_description"],
            param=f"{param}.server_description",
            allow_empty=True,
        )
    if "allowed_callers" in tool and tool["allowed_callers"] is not None:
        allowed_callers = tool["allowed_callers"]
        if not isinstance(allowed_callers, list):
            raise invalid_type(f"{param}.allowed_callers must be an array", param=f"{param}.allowed_callers")
        normalized_callers: list[str] = []
        seen_callers: set[str] = set()
        for index, raw_caller in enumerate(allowed_callers):
            caller_param = f"{param}.allowed_callers[{index}]"
            caller = _require_string(raw_caller, param=caller_param)
            if caller not in _MCP_CALLERS:
                raise invalid_value(
                    f"{caller_param} must be direct or programmatic",
                    param=caller_param,
                )
            if caller in seen_callers:
                raise invalid_value(f"Duplicate MCP caller {caller!r}", param=caller_param)
            seen_callers.add(caller)
            normalized_callers.append(caller)
        normalized["allowed_callers"] = normalized_callers
    if "defer_loading" in tool and not isinstance(tool["defer_loading"], bool):
        raise invalid_type(f"{param}.defer_loading must be a boolean", param=f"{param}.defer_loading")
    return normalized


def _validate_tools(
    value: Any,
    *,
    capabilities: RealtimeSessionCapabilities,
    bounds_validated: bool = False,
    param: str = "session.tools",
) -> list[dict[str, Any]]:
    if not bounds_validated:
        validate_session_tools_bounds(value, param=param)
    tools: list[dict[str, Any]] = []
    function_names: set[str] = set()
    mcp_server_labels: set[str] = set()
    for index, raw_tool in enumerate(value):
        tool_param = f"{param}[{index}]"
        tool = _require_object(raw_tool, param=tool_param)
        typ = _require_string(tool.get("type"), param=f"{tool_param}.type")
        if typ == "mcp":
            if not capabilities.mcp_tools:
                raise unsupported_capability(f"Tool type {typ!r} is not available", param=f"{tool_param}.type")
            normalized = _validate_mcp_tool(tool, param=tool_param)
            server_label = normalized["server_label"]
            if server_label in mcp_server_labels:
                raise invalid_value(
                    f"Duplicate MCP server label {server_label!r}",
                    param=f"{tool_param}.server_label",
                )
            mcp_server_labels.add(server_label)
            tools.append(normalized)
            continue
        if typ != "function" or not capabilities.function_tools:
            raise unsupported_capability(f"Tool type {typ!r} is not available", param=f"{tool_param}.type")
        _reject_unknown_fields(tool, _FUNCTION_TOOL_FIELDS, param=tool_param)
        name = _require_string(tool.get("name"), param=f"{tool_param}.name")
        if name in function_names:
            raise invalid_value(f"Duplicate tool name {name!r}", param=f"{tool_param}.name")
        function_names.add(name)
        if "description" in tool:
            _require_string(tool["description"], param=f"{tool_param}.description", allow_empty=True)
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            raise invalid_type(f"{tool_param}.parameters must be an object", param=f"{tool_param}.parameters")
        try:
            compile_tool_arguments_validator(parameters)
        except ValueError as exc:
            raise RealtimeProtocolError(
                message=str(exc),
                code="invalid_tool_schema",
                param=f"{tool_param}.parameters",
            ) from exc
        normalized = copy.deepcopy(tool)
        normalized.setdefault("parameters", {})
        tools.append(normalized)
    return tools


def _validate_tool_choice(
    value: Any,
    *,
    tools: list[dict[str, Any]],
    param: str = "session.tool_choice",
) -> str | dict[str, Any]:
    available_function_names = {tool["name"] for tool in tools if tool["type"] == "function"}
    available_mcp_labels = {tool["server_label"] for tool in tools if tool["type"] == "mcp"}
    if isinstance(value, str):
        if value not in {"auto", "none", "required"}:
            raise invalid_value(f"{param} must be auto, none, required, or a forced tool object", param=param)
        if value == "required" and not tools:
            raise unsupported_capability(
                "Required tool choice is not available because this pipeline exposes no tools",
                param=param,
            )
        return value
    choice = _require_object(value, param=param)
    typ = _require_string(choice.get("type"), param=f"{param}.type")
    if typ == "function":
        _reject_unknown_fields(choice, frozenset({"type", "name"}), param=param)
        name = _require_string(choice.get("name"), param=f"{param}.name")
        if name not in available_function_names:
            raise invalid_value(f"Forced tool {name!r} is not available for this response", param=f"{param}.name")
        return {"type": "function", "name": name}
    if typ == "mcp":
        _reject_unknown_fields(choice, frozenset({"type", "server_label", "name"}), param=param)
        server_label = _require_string(choice.get("server_label"), param=f"{param}.server_label")
        if server_label not in available_mcp_labels:
            raise invalid_value(
                f"Forced MCP server {server_label!r} is not available for this response",
                param=f"{param}.server_label",
            )
        normalized: dict[str, Any] = {"type": "mcp", "server_label": server_label}
        if "name" in choice:
            if choice["name"] is None:
                normalized["name"] = None
            else:
                normalized["name"] = _require_string(choice["name"], param=f"{param}.name")
        return normalized
    raise unsupported_capability(f"Forced tool type {typ!r} is not available", param=f"{param}.type")


def project_tool_choice_to_pipeline(value: str | dict[str, Any]) -> str | dict[str, Any]:
    """Convert a canonical Realtime tool choice to the LLM-context shape."""
    if isinstance(value, dict):
        if value.get("type") == "mcp":
            return copy.deepcopy(value)
        return {
            "type": "function",
            "function": {"name": value["name"]},
        }
    return value


def _validate_prompt(value: Any) -> dict[str, Any] | None:
    param = "session.prompt"
    if value is None:
        return None
    prompt = _require_object(value, param=param)
    _reject_unknown_fields(prompt, _PROMPT_FIELDS, param=param)
    _require_string(prompt.get("id"), param=f"{param}.id")
    if "version" in prompt:
        _require_string(prompt["version"], param=f"{param}.version")
    if "variables" in prompt and not isinstance(prompt["variables"], dict):
        raise invalid_type(f"{param}.variables must be an object", param=f"{param}.variables")
    return copy.deepcopy(prompt)


class CanonicalRealtimeSession:
    """Own and atomically update one canonical Realtime session."""

    def __init__(
        self,
        *,
        model: str,
        voice: str,
        instructions: str = "",
        max_output_tokens: int | str = "inf",
        input_transcription_model: str | None = None,
        trusted_tools: list[dict[str, Any]] | None = None,
        capabilities: RealtimeSessionCapabilities | None = None,
    ) -> None:
        """Create a session using explicit server-selected model and voice IDs."""
        self.capabilities = capabilities or RealtimeSessionCapabilities()
        if _PCM24 not in self.capabilities.input_formats or _PCM24 not in self.capabilities.output_formats:
            raise ValueError("Canonical Realtime sessions require 24 kHz PCM as the default input and output format")
        default_turn_detection_type = self.capabilities.default_turn_detection_type
        if (
            default_turn_detection_type not in self.capabilities.turn_detection_types
            or "type" not in self.capabilities.turn_detection_options
        ):
            raise ValueError("Canonical Realtime sessions require a supported non-manual default turn detection mode")
        model = _require_string(model, param="session.model")
        normalized_voice = _normalize_voice(
            voice,
            param="session.audio.output.voice",
            capabilities=self.capabilities,
        )
        _require_string(instructions, param="session.instructions", allow_empty=True)
        max_output_tokens = _validate_max_output_tokens(
            max_output_tokens,
            param="session.max_output_tokens",
        )
        transcription = None
        if input_transcription_model is not None:
            transcription = {
                "model": _require_string(
                    input_transcription_model,
                    param="session.audio.input.transcription.model",
                )
            }
        trusted_tool_view = _validate_tools(
            trusted_tools or [],
            capabilities=replace(self.capabilities, function_tools=True, mcp_tools=False),
        )
        untrusted_names = {
            tool["name"] for tool in trusted_tool_view if tool["name"] not in self.capabilities.trusted_function_tools
        }
        if untrusted_names:
            name = sorted(untrusted_names)[0]
            raise ValueError(f"Trusted tool schema {name!r} has no trusted runtime owner")
        self._trusted_tools = {tool["name"]: copy.deepcopy(tool) for tool in trusted_tool_view}

        self.id = new_realtime_id("sess")
        self._output_audio_started = False
        self._locked_output_voice: str | dict[str, str] | None = None
        self._active_response_id: str | None = None
        self._view: dict[str, Any] = {
            "id": self.id,
            "object": "realtime.session",
            "type": "realtime",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "noise_reduction": None,
                    "transcription": transcription,
                    "turn_detection": _default_turn_detection_config(
                        default_turn_detection_type,
                        supported_options=self.capabilities.turn_detection_options,
                    ),
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "speed": 1.0,
                    "voice": normalized_voice,
                },
            },
            "include": [],
            "instructions": instructions,
            "max_output_tokens": max_output_tokens,
            "model": model,
            "output_modalities": ["audio"],
            "parallel_tool_calls": True,
            "prompt": None,
            "tool_choice": "auto",
            "tools": trusted_tool_view,
        }
        if self.capabilities.truncation:
            self._view["truncation"] = "auto"

    @property
    def response_in_progress(self) -> bool:
        """Whether a response currently owns the session output slot."""
        return self._active_response_id is not None

    @property
    def output_audio_started(self) -> bool:
        """Whether any response has emitted audio and permanently locked voice."""
        return self._output_audio_started

    def public_view(self) -> dict[str, Any]:
        """Return a detached snapshot suitable for session.created/updated."""
        return copy.deepcopy(self._view)

    def restore_update_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore an internally captured view after a failed multi-owner update."""
        restored = copy.deepcopy(snapshot)
        if restored.get("id") != self.id or restored.get("object") != "realtime.session":
            raise ValueError("Realtime session rollback snapshot does not belong to this session")
        self._view = restored

    def validate_response_tools(self, value: Any) -> list[dict[str, Any]]:
        """Validate a response-local function-tool replacement without mutating the session."""
        validate_session_tools_bounds(value, param="response.tools")
        selected_tools = _validate_tools(
            value,
            capabilities=self.capabilities,
            bounds_validated=True,
            param="response.tools",
        )
        self._validate_tool_ownership(selected_tools, param="response.tools")
        return selected_tools

    def validate_response_tool_choice(
        self,
        value: Any,
        *,
        tools: list[dict[str, Any]],
    ) -> str | dict[str, Any]:
        """Validate a response-local choice against that response's effective tools."""
        return _validate_tool_choice(value, tools=tools, param="response.tool_choice")

    def validate_response_parallel_tool_calls(self, value: Any) -> bool:
        """Validate a response-local parallel-call policy without mutating the session."""
        if not isinstance(value, bool):
            raise invalid_type(
                "response.parallel_tool_calls must be a boolean",
                param="response.parallel_tool_calls",
            )
        if value and not self.capabilities.parallel_tool_calls:
            raise unsupported_capability(
                "Parallel function calls are not available on this backend",
                param="response.parallel_tool_calls",
            )
        if not value and not self.capabilities.sequential_tool_calls:
            raise unsupported_capability(
                "Sequential-only function calls are not available on this backend",
                param="response.parallel_tool_calls",
            )
        return value

    def validate_response_instructions(self, value: Any) -> str:
        """Validate one response's instruction overlay without mutating the session."""
        return _require_string(
            value,
            param="response.instructions",
            allow_empty=True,
        )

    def validate_response_max_output_tokens(self, value: Any) -> int | str:
        """Validate one response's token limit without mutating the session."""
        return _validate_max_output_tokens(
            value,
            param="response.max_output_tokens",
        )

    def validate_response_audio_output(self, value: Any) -> dict[str, Any]:
        """Validate and materialize a response-local audio output snapshot."""
        audio = _require_object(value, param="response.audio")
        _reject_unknown_fields(audio, frozenset({"output"}), param="response.audio")
        current = self._view["audio"]["output"]
        if "output" not in audio:
            candidate = copy.deepcopy(current)
        else:
            candidate = _validate_audio_output(
                audio["output"],
                current=current,
                capabilities=self.capabilities,
                response_in_progress=False,
                output_audio_started=self._output_audio_started,
                locked_output_voice=self._locked_output_voice,
                param="response.audio.output",
                allowed_fields=frozenset({"format", "voice"}),
            )
        self.validate_effective_response_voice(candidate.get("voice"))
        return {name: copy.deepcopy(candidate[name]) for name in ("format", "voice")}

    def validate_effective_response_voice(self, value: Any) -> str | dict[str, str]:
        """Validate the effective voice and enforce the session-wide audio lock."""
        voice = _normalize_voice(
            value,
            param="response.audio.output.voice",
            capabilities=self.capabilities,
        )
        if self._output_audio_started and voice != self._locked_output_voice:
            raise immutable_field(
                "response.audio.output.voice cannot change after output audio has started",
                param="response.audio.output.voice",
            )
        return voice

    def _validate_tool_ownership(self, tools: list[dict[str, Any]], *, param: str) -> None:
        """Validate exact trusted/client ownership for one effective function-tool set."""
        for tool in tools:
            if tool["type"] != "function":
                continue
            name = tool["name"]
            trusted = self._trusted_tools.get(name)
            if trusted is not None and tool != trusted:
                raise RealtimeProtocolError(
                    message=f"Client tool {name!r} conflicts with a trusted backend tool",
                    code="tool_name_conflict",
                    param=param,
                )

    def bind_output_voice(self, voice: str) -> None:
        """Bind a server-resolved voice before the pipeline owns the session."""
        self._bind_output_voice(voice, allow_discovered=False)

    def bind_discovered_output_voice(self, voice: str) -> None:
        """Bind a voice returned by trusted discovery for the selected TTS route."""
        self._bind_output_voice(voice, allow_discovered=True)

    def bind_discovered_output_voices(self, voices: frozenset[str]) -> None:
        """Publish the exact voice catalog discovered for the trusted TTS route."""
        if (
            not isinstance(voices, frozenset)
            or not voices
            or any(not isinstance(voice, str) or not voice for voice in voices)
        ):
            raise ValueError("A discovered Realtime voice catalog must contain non-empty voice IDs")
        current_voice = self._view["audio"]["output"]["voice"]
        if not isinstance(current_voice, str) or current_voice not in voices:
            raise ValueError("The discovered Realtime voice catalog does not contain the configured voice")
        self.capabilities = replace(self.capabilities, voices=voices)

    def _bind_output_voice(self, voice: str, *, allow_discovered: bool) -> None:
        """Bind an output voice, optionally extending the trusted route allowlist."""
        if self._active_response_id is not None or self._output_audio_started:
            raise RealtimeProtocolError(
                message="The output voice cannot change after response output starts",
                code="immutable_field",
                param="session.audio.output.voice",
            )
        capabilities = self.capabilities
        if allow_discovered:
            discovered_voice = _require_string(voice, param="session.audio.output.voice")
            if capabilities.voices is not None and discovered_voice not in capabilities.voices:
                capabilities = replace(
                    capabilities,
                    voices=frozenset({*capabilities.voices, discovered_voice}),
                )
        normalized_voice = _normalize_voice(
            voice,
            param="session.audio.output.voice",
            capabilities=capabilities,
        )
        self.capabilities = capabilities
        self._view["audio"]["output"]["voice"] = normalized_voice

    def preflight_update_without_output_voice_catalog(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate an update except for membership in the deferred voice catalog."""
        trial = copy.deepcopy(self)
        trial.capabilities = replace(trial.capabilities, voices=None)
        return trial.apply_update(patch)

    def apply_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Validate and atomically apply a canonical ``session.update`` patch."""
        if not isinstance(patch, dict):
            raise invalid_type("session must be an object", param="session")
        _reject_unknown_fields(patch, _SESSION_FIELDS, param="session")
        if "tools" in patch:
            validate_session_tools_bounds(patch["tools"])
        candidate = self.public_view()

        if "type" in patch and patch["type"] != "realtime":
            raise invalid_value("session.type must be 'realtime'", param="session.type")
        if "model" in patch:
            model = _require_string(patch["model"], param="session.model")
            if model != candidate["model"]:
                raise immutable_field("session.model cannot change after connection", param="session.model")
        if "output_modalities" in patch:
            modalities = patch["output_modalities"]
            if modalities not in (["audio"], ["text"]):
                raise invalid_value(
                    "session.output_modalities must be exactly ['audio'] or ['text']",
                    param="session.output_modalities",
                )
            candidate["output_modalities"] = copy.deepcopy(modalities)
        if "instructions" in patch:
            instructions = _require_string(
                patch["instructions"],
                param="session.instructions",
                allow_empty=True,
            )
            if not instructions and candidate["instructions"] and not self.capabilities.supports_empty_instructions:
                raise unsupported_capability(
                    "The trusted pipeline prompt cannot be cleared on this endpoint",
                    param="session.instructions",
                )
            candidate["instructions"] = instructions
        if "max_output_tokens" in patch:
            candidate["max_output_tokens"] = _validate_max_output_tokens(
                patch["max_output_tokens"],
                param="session.max_output_tokens",
            )
        if "audio" in patch:
            audio_patch = _require_object(patch["audio"], param="session.audio")
            _reject_unknown_fields(audio_patch, _AUDIO_FIELDS, param="session.audio")
            audio = copy.deepcopy(candidate["audio"])
            if "input" in audio_patch:
                audio["input"] = _validate_audio_input(
                    audio_patch["input"],
                    current=audio["input"],
                    capabilities=self.capabilities,
                )
            if "output" in audio_patch:
                audio["output"] = _validate_audio_output(
                    audio_patch["output"],
                    current=audio["output"],
                    capabilities=self.capabilities,
                    response_in_progress=self.response_in_progress,
                    output_audio_started=self.output_audio_started,
                    locked_output_voice=self._locked_output_voice,
                )
            candidate["audio"] = audio
        if "tools" in patch:
            requested_tool_types = {
                tool.get("type")
                for tool in patch["tools"]
                if isinstance(tool, dict) and isinstance(tool.get("type"), str)
            }
            if "function" in requested_tool_types and not self.capabilities.function_tools:
                raise unsupported_capability(
                    "Client-defined session function tools are not available on this endpoint",
                    param="session.tools",
                )
            if "mcp" in requested_tool_types and not self.capabilities.mcp_tools:
                raise unsupported_capability(
                    "MCP session tools are not available on this endpoint",
                    param="session.tools",
                )
            selected_tools = _validate_tools(
                patch["tools"],
                capabilities=self.capabilities,
                bounds_validated=True,
            )
            self._validate_tool_ownership(selected_tools, param="session.tools")
            candidate["tools"] = selected_tools
        if "parallel_tool_calls" in patch:
            if not isinstance(patch["parallel_tool_calls"], bool):
                raise invalid_type(
                    "session.parallel_tool_calls must be a boolean",
                    param="session.parallel_tool_calls",
                )
            if patch["parallel_tool_calls"] and not self.capabilities.parallel_tool_calls:
                raise unsupported_capability(
                    "Parallel function calls are not available on this backend",
                    param="session.parallel_tool_calls",
                )
            if not patch["parallel_tool_calls"] and not self.capabilities.sequential_tool_calls:
                raise unsupported_capability(
                    "Sequential-only function calls are not available on this backend",
                    param="session.parallel_tool_calls",
                )
            candidate["parallel_tool_calls"] = patch["parallel_tool_calls"]
        if "tool_choice" in patch:
            candidate["tool_choice"] = _validate_tool_choice(
                patch["tool_choice"],
                tools=candidate["tools"],
            )
        elif "tools" in patch and isinstance(candidate["tool_choice"], dict):
            # Revalidate an existing forced choice against the replacement tool list.
            candidate["tool_choice"] = _validate_tool_choice(
                candidate["tool_choice"],
                tools=candidate["tools"],
            )
        if "prompt" in patch:
            prompt = _validate_prompt(patch["prompt"])
            if prompt != candidate.get("prompt") and not self.capabilities.prompt:
                raise unsupported_capability(
                    "Stored prompt references are not available on this backend; use session.instructions",
                    param="session.prompt",
                )
            candidate["prompt"] = prompt
        if "reasoning" in patch:
            if not self.capabilities.reasoning:
                raise unsupported_capability(
                    "Realtime reasoning controls are not available on this backend",
                    param="session.reasoning",
                )
            reasoning = _require_object(patch["reasoning"], param="session.reasoning")
            _reject_unknown_fields(reasoning, frozenset({"effort"}), param="session.reasoning")
            _require_string(reasoning.get("effort"), param="session.reasoning.effort")
            candidate["reasoning"] = copy.deepcopy(reasoning)
        if "include" in patch:
            include = patch["include"]
            if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
                raise invalid_type("session.include must be an array of strings", param="session.include")
            unsupported = [item for item in include if item not in self.capabilities.include_fields]
            if unsupported:
                raise unsupported_capability(
                    f"Included field {unsupported[0]!r} is not available",
                    param="session.include",
                )
            candidate["include"] = list(include)
        if "tracing" in patch and (patch["tracing"] is not None or candidate.get("tracing") is not None):
            if not self.capabilities.tracing:
                raise unsupported_capability("session.tracing is not available", param="session.tracing")
            candidate["tracing"] = copy.deepcopy(patch["tracing"])
        if "truncation" in patch:
            if not self.capabilities.truncation:
                raise unsupported_capability("session.truncation is not available", param="session.truncation")
            candidate["truncation"] = _validate_truncation(patch["truncation"])

        self._view = candidate
        return self.public_view()

    def begin_response(self, response_id: str) -> None:
        """Reserve the single active response slot."""
        _require_string(response_id, param="response.id")
        if self._active_response_id is not None:
            raise RealtimeProtocolError(
                message=f"Response {self._active_response_id} is already in progress",
                code="response_in_progress",
                param="response",
            )
        self._active_response_id = response_id

    def finish_response(self, response_id: str) -> None:
        """Release the active response slot, rejecting stale completions."""
        if self._active_response_id != response_id:
            raise RealtimeProtocolError(
                message=f"Response {response_id} does not own the active response slot",
                code="response_not_found",
                param="response.id",
            )
        self._active_response_id = None

    def mark_output_audio_started(
        self,
        response_id: str,
        *,
        voice: str | dict[str, str] | None = None,
    ) -> None:
        """Permanently lock the voice when the active response emits audio."""
        if self._active_response_id != response_id:
            raise RealtimeProtocolError(
                message=f"Response {response_id} does not own the active response slot",
                code="response_not_found",
                param="response.id",
            )
        effective_voice = self._view["audio"]["output"]["voice"] if voice is None else voice
        normalized_voice = self.validate_effective_response_voice(effective_voice)
        if self._locked_output_voice is None:
            self._locked_output_voice = copy.deepcopy(normalized_voice)
        self._output_audio_started = True

    def response_defaults(self) -> dict[str, Any]:
        """Snapshot the session fields inherited by a newly created response."""
        defaults = {
            "instructions": self._view["instructions"],
            "output_modalities": copy.deepcopy(self._view["output_modalities"]),
            "max_output_tokens": self._view["max_output_tokens"],
            "parallel_tool_calls": self._view["parallel_tool_calls"],
        }
        if self._view["output_modalities"] == ["audio"]:
            output = self._view["audio"]["output"]
            defaults["audio"] = {"output": {name: copy.deepcopy(output[name]) for name in ("format", "voice")}}
        return defaults
