# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``voice_agent.yaml``: loading, path resolution, profile merging, validation.

The loading rules are exact (plan section 6.1):

1. Every file is read and ``${VAR}`` / ``${VAR:-default}``-interpolated on its own.
2. Input paths (``[in-path]`` keys) are resolved against the directory of the
   file that wrote them, *before* any merge. Output paths resolve against the
   working directory.
3. A profile names its base with ``extends``; chains resolve depth-first, and a
   cycle is an error.
4. One deep-merge rule serves both profile -> base and ``agent.overrides`` ->
   the text ``agent.yaml``: mappings merge key by key, scalars and lists replace,
   ``null`` resets to the default, and a mapping/non-mapping change is an error.
5. The text agent config is built with the text prototype's own
   ``build_config(raw, source_dir=<its directory>)``.
6. Validation runs once, on the merged result.
"""

from __future__ import annotations

import copy
import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_backend_verdict.text.config import Config, build_config, interpolate_env
from examples.frontend_backend_verdict.text.errors import ConfigError
from examples.frontend_backend_verdict.text.prompts import load_catalog, render
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.agent.write_gate import ToolGateSettings, WriteGateSettings
from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat, parse_format
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from examples.frontend_backend_verdict.voice.normalization import NormalizationSettings
from examples.frontend_backend_verdict.voice.normalization.arguments import (
    GUARD_SCOPES,
    ON_INVALID,
    ArgumentRule,
    ResultHintSettings,
    RetryGuardSettings,
    ToolArgumentSettings,
)
from examples.frontend_backend_verdict.voice.normalization.rules import RULESETS
from examples.frontend_backend_verdict.voice.normalization.transcript import (
    CASES,
    SpelledRunSettings,
    TranscriptSettings,
)
from examples.frontend_backend_verdict.voice.speech.catalog import SpeechEndpoint, build_endpoint, load_catalog_entry

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config" / "voice_agent.yaml"

#: Keys whose values are input paths, resolved against the declaring file (step 2).
IN_PATH_KEYS: tuple[str, ...] = (
    "extends",
    "agent.config",
    "agent.overrides.prompts.path",
    "asr.catalog.example_dir",
    "tts.catalog.example_dir",
    "instructions.fallback_file",
)
#: Keys whose values are output paths, resolved against the working directory.
OUT_PATH_KEYS: tuple[str, ...] = ("filler.log_path", "logging.event_log")
#: Mappings whose keys are free-form (not checked against the schema).
_FREE_FORM: frozenset[str] = frozenset(
    {"agent.overrides", "tts.voice_map", "normalization.transcript.separator_words", "write_gate.tools"}
)

_SPEECH_DEFAULTS: dict[str, Any] = {
    "source": "catalog",
    "catalog": {"example_dir": "", "services": "local", "platform": "singlegpu", "key": "", "server_override": ""},
    "inline": {
        "server": "localhost:50051",
        "model": "",
        "function_id": "",
        "voice_id": "",
        "language_code": "en-US",
        "use_ssl": False,
    },
    "max_concurrent_streams": 0,
}

#: The complete schema with its defaults. A key absent here is unknown (except in free-form mappings).
DEFAULTS: dict[str, Any] = {
    "extends": "",
    "server": {
        "host": "0.0.0.0",
        "port": 8765,
        "path": "/v1/realtime",
        "require_bearer": False,
        "bearer_token": "",
        "max_sessions": 8,
        "session_update_timeout_s": 30.0,
        "warmup": True,
        "ws_ping_interval_s": 20.0,
        "ws_ping_timeout_s": 20.0,
    },
    "agent": {"config": "", "overrides": {}},
    "protocol": {
        "greeting": {"enabled": False, "text": ""},
        "seed_history_with_client_greeting": "",
        "auto_response": True,
        "resume_on": "response_create",
        "emit_input_transcription": True,
        "emit_usage": True,
    },
    "audio": {
        "engine_rate": 16000,
        "default_input_format": {"type": "audio/pcm", "rate": 24000},
        "default_output_format": {"type": "audio/pcm", "rate": 24000},
        "output_chunk_ms": 100,
        "pace_output": False,
        "pace_lead_ms": 300,
    },
    "turn_detection": {
        "vad": "silero",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 500,
        "min_speech_ms": 120,
        "min_transcript_chars": 1,
        "honor_client_values": True,
    },
    "barge_in": {
        "enabled": True,
        "history": "truncate_heard",
        "interruption_marker": " [interrupted by the user]",
        "while_thinking": "cancel_and_merge",
        "frontend_verdict": {"note_key": "frontend_task_in_progress", "timeout_ms": 4000, "same_query_guard": True},
    },
    "asr": {**copy.deepcopy(_SPEECH_DEFAULTS), "interim_results": True},
    "tts": {
        **copy.deepcopy(_SPEECH_DEFAULTS),
        "voice_map": {},
        "sample_rate": 22050,
        "normalize_text": True,
        "sentence_split": True,
    },
    "filler": {"mode": "log_only", "speak_after_ms": 300, "log_path": ""},
    "tools": {"source": "client", "config_tools": "", "result_timeout_s": 120.0},
    "instructions": {
        "apply_to": ["backend"],
        "placement": "policy_slot",
        "fallback_file": "",
        "cascade_addendum_key": "cascade_voice_addendum",
        "strip_patterns": [],
        "frontend_capabilities": "from_tools",
    },
    "logging": {"level": "INFO", "event_log": "", "log_wire": False, "redact_content": False},
    "normalization": {
        "transcript": {
            "enabled": False,
            "separator_words": {"underscore": "_"},
            "ruleset": "en",
            "number_words": True,
            "compound_numbers": True,
            "extra_filler_words": [],
            "extra_stop_words": [],
            "max_part_tokens": 6,
            "case": "keep",
            "frontend_note_key": "",
            "spelled_runs": {"enabled": False, "min_tokens": 3, "case": "keep"},
        },
        "tool_arguments": {
            "enabled": False,
            "rules": [],
            "invalid_message_key": "tool_argument_invalid",
            "escalate_invalid_message_key": "",
            "max_local_rounds": 3,
            "retry_guard": {
                "enabled": False,
                "scope": "rules",
                "permanent_failure_pattern": "",
                "message_key": "tool_call_already_failed",
            },
            "result_hints": {
                "enabled": False,
                "tools": [],
                "failure_pattern": "",
                "count_local_invalid": True,
                "max_hints": 3,
                "message_key": "lookup_not_found_hint",
                "escalate_message_key": "",
            },
        },
    },
    "write_gate": {
        "enabled": False,
        "default": "non_read",
        "exempt": [],
        "include": [],
        "tools": {},
        "max_summary_chars": 600,
        "max_unconfirmed_reissues": 2,
    },
}

#: Keys of one ``write_gate.tools.<name>`` entry.
_WRITE_GATE_TOOL_KEYS = frozenset({"not_consequential", "labels"})

#: Keys of one ``normalization.tool_arguments.rules`` entry, with their defaults.
_ARGUMENT_RULE_DEFAULTS: dict[str, Any] = {
    "tool": "",
    "argument": "",
    "label": "",
    "format_hint": "",
    "spoken_form": False,
    "strip": "",
    "collapse_separators": "",
    "case": "keep",
    "pattern": "",
    "on_invalid": "answer_locally",
    "restore_prefix": "",
    "keep_pattern": "",
}

_ENUMS: dict[str, tuple[str, ...]] = {
    "protocol.resume_on": ("response_create", "last_function_output"),
    "turn_detection.vad": ("silero", "energy"),
    "barge_in.history": ("truncate_heard", "keep_full"),
    "barge_in.while_thinking": ("cancel_and_merge", "ignore", "frontend_verdict"),
    "asr.source": ("catalog", "inline"),
    "tts.source": ("catalog", "inline"),
    "asr.catalog.services": ("local", "cloud"),
    "tts.catalog.services": ("local", "cloud"),
    "filler.mode": ("log_only", "speak"),
    "tools.source": ("client", "config"),
    "instructions.placement": ("policy_slot", "replace_prompt", "append"),
    "instructions.frontend_capabilities": ("from_tools", "static", "none"),
    "normalization.transcript.case": CASES,
    "normalization.transcript.spelled_runs.case": CASES,
    "normalization.tool_arguments.retry_guard.scope": GUARD_SCOPES,
    "write_gate.default": ("non_read", "none"),
}


# -- value types ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """WebSocket server settings."""

    host: str
    port: int
    path: str
    require_bearer: bool
    bearer_token: str = field(repr=False)
    max_sessions: int
    session_update_timeout_s: float
    warmup: bool
    ws_ping_interval_s: float = 20.0  # uvicorn WebSocket keepalive ping; 0 disables it
    ws_ping_timeout_s: float = 20.0


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """Realtime protocol behaviour."""

    greeting_enabled: bool
    greeting_text: str
    seed_history_with_client_greeting: str
    auto_response: bool
    resume_on: str
    emit_input_transcription: bool
    emit_usage: bool


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Internal audio settings and per-session defaults."""

    engine_rate: int
    default_input_format: AudioFormat
    default_output_format: AudioFormat
    output_chunk_ms: int
    pace_output: bool
    pace_lead_ms: int


@dataclass(frozen=True, slots=True)
class TurnDetectionConfig:
    """VAD and endpointing defaults (``session.update`` may override per session)."""

    vad: str
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int
    min_speech_ms: int
    min_transcript_chars: int
    honor_client_values: bool


@dataclass(frozen=True, slots=True)
class FrontendVerdictConfig:
    """``barge_in.frontend_verdict``: the frontend decides whether speech continues the running request."""

    note_key: str = "frontend_task_in_progress"
    timeout_ms: int = 4000
    #: A probe delegation with the running request's query is ``continue``, whatever its ``task``.
    same_query_guard: bool = True


@dataclass(frozen=True, slots=True)
class BargeInConfig:
    """What happens when the user talks over the agent."""

    enabled: bool
    history: str
    interruption_marker: str
    while_thinking: str
    frontend_verdict: FrontendVerdictConfig = field(default_factory=FrontendVerdictConfig)

    @property
    def uses_frontend_verdict(self) -> bool:
        """Whether speech during a delegated turn is judged by the frontend (``frontend_verdict``)."""
        return self.enabled and self.while_thinking == "frontend_verdict"


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    """One resolved speech service plus its per-kind options."""

    endpoint: SpeechEndpoint
    max_concurrent_streams: int
    interim_results: bool = True
    voice_map: dict[str, str] = field(default_factory=dict)
    sample_rate: int = 22050
    normalize_text: bool = True
    sentence_split: bool = True


@dataclass(frozen=True, slots=True)
class FillerConfig:
    """Filler handling in paired mode."""

    mode: str
    speak_after_ms: int
    log_path: str


@dataclass(frozen=True, slots=True)
class ToolsConfig:
    """Where the backend's tools come from."""

    source: str
    config_tools: str
    result_timeout_s: float
    config_tool_specs: tuple[ToolSpec, ...] = ()


@dataclass(frozen=True, slots=True)
class InstructionsConfig:
    """How ``session.update.instructions`` reaches the prompts."""

    apply_to: tuple[str, ...]
    placement: str
    fallback_file: str
    fallback_text: str
    cascade_addendum_key: str
    strip_patterns: tuple[str, ...]
    frontend_capabilities: str


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Logging and event-log settings."""

    level: str
    event_log: str
    log_wire: bool
    redact_content: bool


@dataclass(frozen=True, slots=True)
class VoiceConfig:
    """The fully resolved, immutable voice configuration."""

    server: ServerConfig
    agent: Config
    agent_config_path: Path
    protocol: ProtocolConfig
    audio: AudioConfig
    turn_detection: TurnDetectionConfig
    barge_in: BargeInConfig
    asr: SpeechConfig
    tts: SpeechConfig
    filler: FillerConfig
    tools: ToolsConfig
    instructions: InstructionsConfig
    logging: LoggingConfig
    normalization: NormalizationSettings = field(default_factory=NormalizationSettings)
    write_gate: WriteGateSettings = field(default_factory=WriteGateSettings)
    source_files: tuple[Path, ...] = ()
    resolved_in_paths: dict[str, str] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


# -- loading ----------------------------------------------------------------------


def _get(mapping: dict[str, Any], dotted: str) -> Any:
    node: Any = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _set(mapping: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = mapping
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _read_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VoiceConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise VoiceConfigError(f"{path}: invalid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise VoiceConfigError(f"{path}: config root must be a mapping")
    return interpolate_env(raw)


def _resolve_in_paths(raw: dict[str, Any], source: Path) -> None:
    for key in IN_PATH_KEYS:
        value = _get(raw, key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value.strip()).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        resolved = path.resolve()
        if not resolved.exists():
            raise VoiceConfigError(
                f"{key}: {value!r} declared in {source} resolves to {resolved}, which does not exist"
            )
        _set(raw, key, str(resolved))


def deep_merge(
    base: Any,
    override: Any,
    *,
    source: str,
    path: str = "",
    defaults: Any = None,
) -> Any:
    """Merge ``override`` onto ``base`` with the rule table of plan section 6.1 step 4.

    ``defaults`` is the schema default at ``path`` (used when ``override`` is ``null``).
    """
    if override is None:
        return copy.deepcopy(defaults)
    if isinstance(override, dict):
        if base is None:
            base = {}
        if not isinstance(base, dict):
            raise VoiceConfigError(f"{source}: {path or '<root>'} changes type from a value to a mapping")
        merged = dict(base)
        for key, value in override.items():
            child = f"{path}.{key}" if path else str(key)
            child_defaults = defaults.get(key) if isinstance(defaults, dict) else None
            if value is None:
                # null is not a stored value: it resets the key to its schema default.
                if child_defaults is None:
                    merged.pop(key, None)
                else:
                    merged[key] = copy.deepcopy(child_defaults)
                continue
            merged[key] = deep_merge(merged.get(key), value, source=source, path=child, defaults=child_defaults)
        return merged
    if isinstance(base, dict):
        raise VoiceConfigError(f"{source}: {path} changes type from a mapping to {type(override).__name__}")
    return copy.deepcopy(override)


def _load_chain(path: Path, seen: tuple[Path, ...]) -> tuple[dict[str, Any], tuple[Path, ...]]:
    path = path.expanduser().resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise VoiceConfigError(f"extends cycle: {chain}")
    raw = _read_file(path)
    _resolve_in_paths(raw, path)
    parent = raw.pop("extends", "") or ""
    if parent:
        merged, files = _load_chain(Path(parent), (*seen, path))
    else:
        merged, files = copy.deepcopy(DEFAULTS), ()
    merged.pop("extends", None)
    return deep_merge(merged, raw, source=str(path), defaults=DEFAULTS), (*files, path)


def _check_unknown(merged: dict[str, Any], schema: dict[str, Any], files: Sequence[Path], prefix: str = "") -> None:
    for key, value in merged.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            raise VoiceConfigError(f"unknown key {dotted!r} (loaded from {', '.join(str(f) for f in files)})")
        if dotted in _FREE_FORM:
            if not isinstance(value, dict):
                raise VoiceConfigError(f"{dotted} must be a mapping")
            continue
        if isinstance(schema[key], dict):
            if not isinstance(value, dict):
                raise VoiceConfigError(f"{dotted} must be a mapping")
            _check_unknown(value, schema[key], files, dotted)


class _Reader:
    """Typed accessors over the merged mapping, with key-naming errors."""

    def __init__(self, merged: dict[str, Any], files: Sequence[Path]) -> None:
        self._merged = merged
        self._where = ", ".join(str(item) for item in files)

    def _fail(self, key: str, problem: str) -> VoiceConfigError:
        return VoiceConfigError(f"{key}: {problem} (loaded from {self._where})")

    def raw(self, key: str) -> Any:
        value = _get(self._merged, key)
        return copy.deepcopy(_get(DEFAULTS, key)) if value is None else value

    def str(self, key: str) -> str:
        value = self.raw(key)
        if not isinstance(value, str | int | float) or isinstance(value, bool):
            raise self._fail(key, f"must be a string, got {type(value).__name__}")
        return str(value)

    def int(self, key: str, *, minimum: int | None = None) -> int:
        value = self.raw(key)
        try:
            if isinstance(value, bool):
                raise TypeError
            number = int(value)
        except (TypeError, ValueError):
            raise self._fail(key, f"must be an integer, got {value!r}") from None
        if minimum is not None and number < minimum:
            raise self._fail(key, f"must be >= {minimum}, got {number}")
        return number

    def float(self, key: str, *, minimum: float | None = None) -> float:
        value = self.raw(key)
        try:
            if isinstance(value, bool):
                raise TypeError
            number = float(value)
        except (TypeError, ValueError):
            raise self._fail(key, f"must be a number, got {value!r}") from None
        if minimum is not None and number < minimum:
            raise self._fail(key, f"must be >= {minimum}, got {number}")
        return number

    def bool(self, key: str) -> bool:
        value = self.raw(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0", "yes", "no"):
            return value.strip().lower() in ("true", "1", "yes")
        raise self._fail(key, f"must be a boolean, got {value!r}")

    def enum(self, key: str) -> str:
        value = self.str(key).strip()
        allowed = _ENUMS[key]
        if value not in allowed:
            raise self._fail(key, f"must be one of {allowed}, got {value!r}")
        return value

    def str_list(self, key: str) -> tuple[str, ...]:
        value = self.raw(key)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise self._fail(key, "must be a list of strings")
        return tuple(value)

    def mapping(self, key: str) -> dict[str, Any]:
        value = self.raw(key)
        if not isinstance(value, dict):
            raise self._fail(key, "must be a mapping")
        return value


def _audio_format(reader: _Reader, key: str) -> AudioFormat:
    value = dict(reader.mapping(key))
    if value.get("type") in ("audio/pcmu", "audio/pcma"):
        value.pop("rate", None)
    try:
        return parse_format(value, param=key)
    except ValueError as exc:
        raise VoiceConfigError(str(exc)) from exc


def _speech(reader: _Reader, kind: str) -> SpeechConfig:
    source = reader.enum(f"{kind}.source")
    if source == "catalog":
        example_dir = reader.str(f"{kind}.catalog.example_dir")
        if not example_dir:
            raise VoiceConfigError(f"{kind}.catalog.example_dir is required when {kind}.source is catalog")
        services = reader.enum(f"{kind}.catalog.services")
        platform = reader.str(f"{kind}.catalog.platform").strip()
        key = reader.str(f"{kind}.catalog.key").strip()
        entry = load_catalog_entry(kind, example_dir=Path(example_dir), services=services, platform=platform, key=key)
        endpoint = build_endpoint(
            kind,
            entry,
            source=f"catalog {example_dir}/services.{services}.yaml [{platform or '-'}] {key}",
            server_override=reader.str(f"{kind}.catalog.server_override").strip(),
        )
    else:
        inline = reader.mapping(f"{kind}.inline")
        endpoint = build_endpoint(kind, inline, source="inline", use_ssl=bool(inline.get("use_ssl")))
    common: dict[str, Any] = {
        "endpoint": endpoint,
        "max_concurrent_streams": reader.int(f"{kind}.max_concurrent_streams", minimum=0),
    }
    if kind == "asr":
        return SpeechConfig(**common, interim_results=reader.bool("asr.interim_results"))
    voice_map = {str(k): str(v or "") for k, v in reader.mapping("tts.voice_map").items()}
    return SpeechConfig(
        **common,
        voice_map=voice_map,
        sample_rate=reader.int("tts.sample_rate", minimum=8000),
        normalize_text=reader.bool("tts.normalize_text"),
        sentence_split=reader.bool("tts.sentence_split"),
    )


def import_reference(reference: str) -> Any:
    """Import ``module:attr`` and return the attribute."""
    module_name, _, attr = reference.partition(":")
    if not module_name or not attr:
        raise VoiceConfigError(f"expected 'module:attr', got {reference!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise VoiceConfigError(f"cannot import {module_name!r}: {exc}") from exc
    if not hasattr(module, attr):
        raise VoiceConfigError(f"{module_name!r} has no attribute {attr!r}")
    return getattr(module, attr)


def _tools(reader: _Reader) -> ToolsConfig:
    raw_source = reader.str("tools.source").strip()
    if raw_source == "merge":
        raise VoiceConfigError(
            "tools.source: merge is not supported; mixed internal/external execution needs a text-prototype "
            "seam that is deferred (plan section 18.1). Use client or config."
        )
    source = reader.enum("tools.source")
    reference = reader.str("tools.config_tools").strip()
    specs: tuple[ToolSpec, ...] = ()
    if source == "config":
        if not reference:
            raise VoiceConfigError("tools.config_tools is required when tools.source is config")
        loaded = import_reference(reference)
        if not isinstance(loaded, list | tuple) or not all(isinstance(item, ToolSpec) for item in loaded):
            raise VoiceConfigError(f"tools.config_tools {reference!r} must be a list of ToolSpec")
        specs = tuple(loaded)
    return ToolsConfig(
        source=source,
        config_tools=reference,
        result_timeout_s=reader.float("tools.result_timeout_s", minimum=0.1),
        config_tool_specs=specs,
    )


def _write_gate(reader: _Reader) -> WriteGateSettings:
    """``write_gate``; tool names and schema paths are checked per session, when the tools are known."""
    tools: dict[str, ToolGateSettings] = {}
    for name, raw in reader.mapping("write_gate.tools").items():
        key = f"write_gate.tools.{name}"
        if not isinstance(raw, dict) or set(raw) - _WRITE_GATE_TOOL_KEYS:
            raise VoiceConfigError(f"{key}: must be a mapping with keys {sorted(_WRITE_GATE_TOOL_KEYS)}")
        paths = raw.get("not_consequential", [])
        labels = raw.get("labels", {})
        if not isinstance(paths, list) or not all(isinstance(item, str) and item for item in paths):
            raise VoiceConfigError(f"{key}.not_consequential: must be a list of paths")
        if not isinstance(labels, dict) or not all(isinstance(v, str) and v for v in labels.values()):
            raise VoiceConfigError(f"{key}.labels: must map paths to nonempty labels")
        tools[str(name)] = ToolGateSettings(
            not_consequential=tuple(paths), labels={str(path): label for path, label in labels.items()}
        )
    return WriteGateSettings(
        enabled=reader.bool("write_gate.enabled"),
        default=reader.enum("write_gate.default"),
        exempt=reader.str_list("write_gate.exempt"),
        include=reader.str_list("write_gate.include"),
        tools=tools,
        max_summary_chars=reader.int("write_gate.max_summary_chars", minimum=1),
        max_unconfirmed_reissues=reader.int("write_gate.max_unconfirmed_reissues", minimum=1),
    )


def _check_write_gate(config: VoiceConfig) -> None:
    if config.write_gate.enabled and not config.agent.frontend_enabled:
        raise VoiceConfigError(
            "write_gate.enabled needs the frontend (agent frontend enabled): it judges the caller's confirmation"
        )


def _instructions(reader: _Reader) -> InstructionsConfig:
    apply_to = reader.str_list("instructions.apply_to")
    for target in apply_to:
        if target not in ("backend", "frontend"):
            raise VoiceConfigError(f"instructions.apply_to entries must be backend or frontend, got {target!r}")
    fallback_file = reader.str("instructions.fallback_file").strip()
    fallback_text = Path(fallback_file).read_text(encoding="utf-8") if fallback_file else ""
    return InstructionsConfig(
        apply_to=apply_to,
        placement=reader.enum("instructions.placement"),
        fallback_file=fallback_file,
        fallback_text=fallback_text,
        cascade_addendum_key=reader.str("instructions.cascade_addendum_key").strip(),
        strip_patterns=reader.str_list("instructions.strip_patterns"),
        frontend_capabilities=reader.enum("instructions.frontend_capabilities"),
    )


def _regex(key: str, pattern: str) -> str:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise VoiceConfigError(f"{key}: invalid regular expression {pattern!r}: {exc}") from None
    return pattern


def _argument_rule(index: int, raw: Any) -> ArgumentRule:
    key = f"normalization.tool_arguments.rules[{index}]"
    if not isinstance(raw, dict):
        raise VoiceConfigError(f"{key} must be a mapping")
    if unknown := sorted(set(raw) - set(_ARGUMENT_RULE_DEFAULTS)):
        raise VoiceConfigError(f"{key}: unknown keys {unknown}")
    values = {**_ARGUMENT_RULE_DEFAULTS, **{k: v for k, v in raw.items() if v is not None}}
    for name, default in _ARGUMENT_RULE_DEFAULTS.items():
        if not isinstance(values[name], type(default)):
            raise VoiceConfigError(f"{key}.{name} must be a {type(default).__name__}, got {values[name]!r}")
    for name in ("tool", "argument"):
        values[name] = values[name].strip()
        if not values[name]:
            raise VoiceConfigError(f"{key}.{name} is required")
    if values["case"] not in CASES:
        raise VoiceConfigError(f"{key}.case must be one of {CASES}, got {values['case']!r}")
    if values["on_invalid"] not in ON_INVALID:
        raise VoiceConfigError(f"{key}.on_invalid must be one of {ON_INVALID}, got {values['on_invalid']!r}")
    _regex(f"{key}.pattern", values["pattern"])
    _regex(f"{key}.keep_pattern", values["keep_pattern"])
    if any(ch.isalnum() for ch in values["restore_prefix"]):
        raise VoiceConfigError(f"{key}.restore_prefix may contain symbols only, got {values['restore_prefix']!r}")
    return ArgumentRule(**values)


def _normalization(reader: _Reader, tools: ToolsConfig, warnings: list[str]) -> NormalizationSettings:
    prefix = "normalization.transcript"
    separators = reader.mapping(f"{prefix}.separator_words")
    if not all(isinstance(k, str) and isinstance(v, str) and k.strip() and v for k, v in separators.items()):
        raise VoiceConfigError(f"{prefix}.separator_words must map non-empty words to non-empty strings")
    ruleset = reader.str(f"{prefix}.ruleset").strip()
    if ruleset not in RULESETS:
        raise VoiceConfigError(f"{prefix}.ruleset: unknown ruleset {ruleset!r}; known: {sorted(RULESETS)}")
    transcript = TranscriptSettings(
        enabled=reader.bool(f"{prefix}.enabled"),
        separator_words={k.strip().lower(): v for k, v in separators.items()},
        ruleset=ruleset,
        number_words=reader.bool(f"{prefix}.number_words"),
        compound_numbers=reader.bool(f"{prefix}.compound_numbers"),
        extra_filler_words=reader.str_list(f"{prefix}.extra_filler_words"),
        extra_stop_words=reader.str_list(f"{prefix}.extra_stop_words"),
        max_part_tokens=reader.int(f"{prefix}.max_part_tokens", minimum=1),
        case=reader.enum(f"{prefix}.case"),
        frontend_note_key=reader.str(f"{prefix}.frontend_note_key").strip(),
        spelled_runs=SpelledRunSettings(
            enabled=reader.bool(f"{prefix}.spelled_runs.enabled"),
            min_tokens=reader.int(f"{prefix}.spelled_runs.min_tokens", minimum=2),
            case=reader.enum(f"{prefix}.spelled_runs.case"),
        ),
    )
    prefix = "normalization.tool_arguments"
    raw_rules = reader.raw(f"{prefix}.rules")
    schema_rules = raw_rules == "auto"  # derived per session from the tool schemas (normalization/schema_rules.py)
    if schema_rules:
        raw_rules = []
    if not isinstance(raw_rules, list):
        raise VoiceConfigError(f"{prefix}.rules must be a list or 'auto'")
    raw_hint_tools = reader.raw(f"{prefix}.result_hints.tools")
    auto_hint_tools = raw_hint_tools == "auto"  # every read tool of the session
    guard = RetryGuardSettings(
        enabled=reader.bool(f"{prefix}.retry_guard.enabled"),
        scope=reader.enum(f"{prefix}.retry_guard.scope"),
        permanent_failure_pattern=_regex(
            f"{prefix}.retry_guard.permanent_failure_pattern",
            reader.str(f"{prefix}.retry_guard.permanent_failure_pattern").strip(),
        ),
        message_key=reader.str(f"{prefix}.retry_guard.message_key").strip(),
    )
    hints = ResultHintSettings(
        enabled=reader.bool(f"{prefix}.result_hints.enabled"),
        tools=()
        if auto_hint_tools
        else tuple(t.strip() for t in reader.str_list(f"{prefix}.result_hints.tools") if t.strip()),
        auto_tools=auto_hint_tools,
        failure_pattern=_regex(
            f"{prefix}.result_hints.failure_pattern", reader.str(f"{prefix}.result_hints.failure_pattern").strip()
        ),
        count_local_invalid=reader.bool(f"{prefix}.result_hints.count_local_invalid"),
        max_hints=reader.int(f"{prefix}.result_hints.max_hints", minimum=1),
        message_key=reader.str(f"{prefix}.result_hints.message_key").strip(),
        escalate_message_key=reader.str(f"{prefix}.result_hints.escalate_message_key").strip(),
    )
    arguments = ToolArgumentSettings(
        enabled=reader.bool(f"{prefix}.enabled"),
        rules=tuple(_argument_rule(index, raw) for index, raw in enumerate(raw_rules)),
        schema_rules=schema_rules,
        invalid_message_key=reader.str(f"{prefix}.invalid_message_key").strip(),
        escalate_invalid_message_key=reader.str(f"{prefix}.escalate_invalid_message_key").strip(),
        max_local_rounds=reader.int(f"{prefix}.max_local_rounds", minimum=1),
        retry_guard=guard,
        result_hints=hints,
    )
    if hints.enabled:
        if not arguments.enabled:
            raise VoiceConfigError(f"{prefix}.result_hints.enabled requires {prefix}.enabled")
        if not hints.tools and not hints.auto_tools:
            raise VoiceConfigError(f"{prefix}.result_hints.tools is required when result hints are enabled")
        if not hints.failure_pattern:
            raise VoiceConfigError(f"{prefix}.result_hints.failure_pattern is required when result hints are enabled")
        if not hints.message_key:
            raise VoiceConfigError(f"{prefix}.result_hints.message_key is required when result hints are enabled")
    if arguments.enabled:
        if tools.source != "client":
            raise VoiceConfigError(
                f"{prefix}.enabled requires tools.source: client (the hook screens calls handed to the client; "
                "with tools.source: config the text agent executes tools itself)"
            )
        if guard.enabled and not guard.permanent_failure_pattern:
            raise VoiceConfigError(
                f"{prefix}.retry_guard.permanent_failure_pattern is required when the retry guard is enabled"
            )
        if (
            not arguments.rules
            and not arguments.schema_rules
            and not (guard.enabled and guard.scope == "all")
            and not hints.enabled
        ):
            warnings.append(f"{prefix}.enabled has no effect without rules (or retry_guard.scope: all)")
    return NormalizationSettings(transcript=transcript, tool_arguments=arguments)


def _check_normalization_prompts(config: VoiceConfig) -> None:
    settings = config.normalization
    keys: list[tuple[str, str]] = []
    if settings.transcript.enabled and settings.transcript.frontend_note_key:
        keys.append(("normalization.transcript.frontend_note_key", settings.transcript.frontend_note_key))
    if settings.tool_arguments.enabled:
        keys.append(("normalization.tool_arguments.invalid_message_key", settings.tool_arguments.invalid_message_key))
        if settings.tool_arguments.escalate_invalid_message_key:
            keys.append(
                (
                    "normalization.tool_arguments.escalate_invalid_message_key",
                    settings.tool_arguments.escalate_invalid_message_key,
                )
            )
        if settings.tool_arguments.retry_guard.enabled:
            keys.append(
                (
                    "normalization.tool_arguments.retry_guard.message_key",
                    settings.tool_arguments.retry_guard.message_key,
                )
            )
        hints = settings.tool_arguments.result_hints
        if hints.enabled:
            keys.append(("normalization.tool_arguments.result_hints.message_key", hints.message_key))
            if hints.escalate_message_key:
                keys.append(
                    ("normalization.tool_arguments.result_hints.escalate_message_key", hints.escalate_message_key)
                )
    if not keys:
        return
    catalog = load_catalog(config.agent.prompts_path, config.agent.prompts.inline)
    for name, key in keys:
        try:
            catalog.get(key)
        except ConfigError as exc:
            raise VoiceConfigError(f"{name}: prompt {key!r} is not in the catalog ({exc})") from None


def prompt_context(config: VoiceConfig) -> dict[str, Any]:
    """The Jinja variables of the prompt templates (``in_progress`` is added per frontend call)."""
    return {
        "barge_in": {
            "while_thinking": config.barge_in.while_thinking,
            "note_key": config.barge_in.frontend_verdict.note_key,
        },
        "filler": {"mode": config.filler.mode},
    }


#: A representative ``in_progress`` for the load-time template check.
SAMPLE_IN_PROGRESS: dict[str, Any] = {
    "query": "Check the status of the user's request.",
    "filler": "One moment.",
    "filler_heard": True,
    "new_words": "okay",
}


def _check_prompt_templates(config: VoiceConfig) -> None:
    """Render the frontend (and backend) templates once, so a broken template fails at load time."""
    agent = config.agent
    catalog = load_catalog(agent.prompts_path, agent.prompts.inline)
    where = ", ".join(str(item) for item in config.source_files)
    base = prompt_context(config)
    samples: list[tuple[str, dict[str, Any]]] = [(agent.backend.prompt_key, base)]
    if agent.frontend_enabled:
        samples.append((agent.frontend.prompt_key, {**base, "in_progress": None}))
        if config.barge_in.while_thinking == "frontend_verdict":
            samples.append((agent.frontend.prompt_key, {**base, "in_progress": SAMPLE_IN_PROGRESS}))
    for key, context in samples:
        try:
            render(catalog.get(key), agent, context, catalog=catalog, key=key)
        except ConfigError as exc:
            raise VoiceConfigError(f"{exc} (loaded from {where})") from None


def _check_frontend_verdict(config: VoiceConfig) -> None:
    if config.barge_in.while_thinking != "frontend_verdict":
        return
    where = ", ".join(str(item) for item in config.source_files)
    if not config.agent.frontend_enabled:
        raise VoiceConfigError(
            "barge_in.while_thinking: frontend_verdict needs the frontend; it is not available with "
            f"agent.mode: backend_only (loaded from {where})"
        )
    key = config.barge_in.frontend_verdict.note_key
    if not key:
        raise VoiceConfigError(f"barge_in.frontend_verdict.note_key must not be empty (loaded from {where})")
    catalog = load_catalog(config.agent.prompts_path, config.agent.prompts.inline)
    try:
        catalog.get(key)
    except ConfigError as exc:
        raise VoiceConfigError(
            f"barge_in.frontend_verdict.note_key: prompt {key!r} is not in the catalog ({exc}) (loaded from {where})"
        ) from None


def _text_agent_config(reader: _Reader, tools: ToolsConfig) -> tuple[Config, Path]:
    config_path = reader.str("agent.config").strip()
    if not config_path:
        raise VoiceConfigError("agent.config is required (path to the text prototype's agent.yaml)")
    path = Path(config_path)
    raw = _read_file(path)
    overrides = copy.deepcopy(reader.mapping("agent.overrides"))
    tools_overrides = overrides.setdefault("backend", {}).setdefault("tools", {})
    if not isinstance(tools_overrides, dict):
        raise VoiceConfigError("agent.overrides.backend.tools must be a mapping")
    # Exactly one execution mode per session, chosen by tools.source (plan section 10).
    tools_overrides["execution"] = "external" if tools.source == "client" else "internal"
    merged = deep_merge(raw, overrides, source=f"agent.overrides -> {path}")
    try:
        return build_config(merged, source_dir=path.parent), path
    except ConfigError as exc:
        raise VoiceConfigError(f"text agent config ({path} + agent.overrides): {exc}") from exc


def build_voice_config(merged: dict[str, Any], files: Sequence[Path] = ()) -> VoiceConfig:
    """Validate a merged mapping and build a :class:`VoiceConfig`."""
    _check_unknown(merged, DEFAULTS, files)
    reader = _Reader(merged, files)
    warnings: list[str] = []

    tools = _tools(reader)
    text_config, text_path = _text_agent_config(reader, tools)
    filler_mode = reader.enum("filler.mode")
    if filler_mode == "speak" and not text_config.frontend_enabled:
        warnings.append("filler.mode: speak has no effect with agent.mode: backend_only (there is no filler)")

    config = VoiceConfig(
        server=ServerConfig(
            host=reader.str("server.host"),
            port=reader.int("server.port", minimum=0),
            path=reader.str("server.path"),
            require_bearer=reader.bool("server.require_bearer"),
            bearer_token=reader.str("server.bearer_token"),
            max_sessions=reader.int("server.max_sessions", minimum=1),
            session_update_timeout_s=reader.float("server.session_update_timeout_s", minimum=0.1),
            warmup=reader.bool("server.warmup"),
            ws_ping_interval_s=reader.float("server.ws_ping_interval_s", minimum=0.0),
            ws_ping_timeout_s=reader.float("server.ws_ping_timeout_s", minimum=0.1),
        ),
        agent=text_config,
        agent_config_path=text_path,
        protocol=ProtocolConfig(
            greeting_enabled=reader.bool("protocol.greeting.enabled"),
            greeting_text=reader.str("protocol.greeting.text"),
            seed_history_with_client_greeting=reader.str("protocol.seed_history_with_client_greeting"),
            auto_response=reader.bool("protocol.auto_response"),
            resume_on=reader.enum("protocol.resume_on"),
            emit_input_transcription=reader.bool("protocol.emit_input_transcription"),
            emit_usage=reader.bool("protocol.emit_usage"),
        ),
        audio=AudioConfig(
            engine_rate=reader.int("audio.engine_rate", minimum=8000),
            default_input_format=_audio_format(reader, "audio.default_input_format"),
            default_output_format=_audio_format(reader, "audio.default_output_format"),
            output_chunk_ms=reader.int("audio.output_chunk_ms", minimum=10),
            pace_output=reader.bool("audio.pace_output"),
            pace_lead_ms=reader.int("audio.pace_lead_ms", minimum=0),
        ),
        turn_detection=TurnDetectionConfig(
            vad=reader.enum("turn_detection.vad"),
            threshold=reader.float("turn_detection.threshold", minimum=0.0),
            prefix_padding_ms=reader.int("turn_detection.prefix_padding_ms", minimum=0),
            silence_duration_ms=reader.int("turn_detection.silence_duration_ms", minimum=1),
            min_speech_ms=reader.int("turn_detection.min_speech_ms", minimum=0),
            min_transcript_chars=reader.int("turn_detection.min_transcript_chars", minimum=0),
            honor_client_values=reader.bool("turn_detection.honor_client_values"),
        ),
        barge_in=BargeInConfig(
            enabled=reader.bool("barge_in.enabled"),
            history=reader.enum("barge_in.history"),
            interruption_marker=reader.str("barge_in.interruption_marker"),
            while_thinking=reader.enum("barge_in.while_thinking"),
            frontend_verdict=FrontendVerdictConfig(
                note_key=reader.str("barge_in.frontend_verdict.note_key").strip(),
                timeout_ms=reader.int("barge_in.frontend_verdict.timeout_ms", minimum=1),
                same_query_guard=reader.bool("barge_in.frontend_verdict.same_query_guard"),
            ),
        ),
        asr=_speech(reader, "asr"),
        tts=_speech(reader, "tts"),
        filler=FillerConfig(
            mode=filler_mode,
            speak_after_ms=reader.int("filler.speak_after_ms", minimum=0),
            log_path=reader.str("filler.log_path").strip(),
        ),
        tools=tools,
        instructions=_instructions(reader),
        logging=LoggingConfig(
            level=reader.str("logging.level").upper(),
            event_log=reader.str("logging.event_log").strip(),
            log_wire=reader.bool("logging.log_wire"),
            redact_content=reader.bool("logging.redact_content"),
        ),
        normalization=_normalization(reader, tools, warnings),
        write_gate=_write_gate(reader),
        source_files=tuple(files),
        resolved_in_paths={key: str(_get(merged, key) or "") for key in IN_PATH_KEYS if key != "extends"},
        warnings=tuple(warnings),
    )
    _check_normalization_prompts(config)
    _check_frontend_verdict(config)
    _check_write_gate(config)
    _check_prompt_templates(config)
    transcript = config.normalization.transcript
    language = RULESETS[transcript.ruleset].language
    if transcript.enabled and not config.asr.endpoint.language_code.lower().startswith(language):
        config = replace(
            config,
            warnings=(
                *config.warnings,
                f"normalization.transcript.ruleset {transcript.ruleset!r} is for {language!r}, but the ASR "
                f"language_code is {config.asr.endpoint.language_code!r}; its word lists will not match",
            ),
        )
    return config


def load_voice_config(path: str | Path = DEFAULT_CONFIG_PATH) -> VoiceConfig:
    """Load a voice config (the base file or a profile) and validate it."""
    merged, files = _load_chain(Path(path), ())
    return build_voice_config(merged, files)
