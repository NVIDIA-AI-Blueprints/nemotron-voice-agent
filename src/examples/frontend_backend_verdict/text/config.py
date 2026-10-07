# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""YAML configuration for the text Frontend/Backend Agent prototype.

``agent.yaml`` is the single entry point. It references a prompt catalog by
path (matching the repository's ``prompts.yaml`` / ``services.*.yaml``
convention) and may override any prompt inline. ``${VAR}`` and
``${VAR:-default}`` are resolved from the environment so no credential is ever
written to the file.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_backend_verdict.text.errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

MODE_FRONTEND_BACKEND = "frontend_backend"
MODE_BACKEND_ONLY = "backend_only"
_MODES = (MODE_FRONTEND_BACKEND, MODE_BACKEND_ONLY)

_EXECUTION_MODES = ("internal", "external")
_INCOMPLETE_POLICIES = ("error", "synthesize_error_result")
_PENDING_POLICIES = ("error", "discard_pending")
_VIOLATION_POLICIES = ("fallback_text", "error")

HISTORY_FULL = "full"
HISTORY_BACKEND_TURNS = "backend_turns"
HISTORY_TRANSCRIPT = "transcript"
_HISTORY_INCLUDES = (HISTORY_FULL, HISTORY_BACKEND_TURNS, HISTORY_TRANSCRIPT)

_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS = frozenset({"false", "0", "no", "off", ""})


def interpolate_env(value: Any) -> Any:
    """Recursively resolve ``${VAR}`` / ``${VAR:-default}`` inside ``value``."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), m.group(2) if m.group(2) is not None else ""), value)
    if isinstance(value, dict):
        return {key: interpolate_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate_env(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Connection and sampling settings for one LLM role."""

    model: str
    base_url: str = "https://inference-api.nvidia.com/v1"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DelegationConfig:
    """Frontend delegation-contract enforcement settings."""

    max_repair_attempts: int = 1
    on_contract_violation: str = "fallback_text"
    fallback_text: str = "Sorry, I could not process that. Could you rephrase?"


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    """Sliding-window size, counted in logical groups."""

    max_groups: int = 20


@dataclass(frozen=True, slots=True)
class FrontendConfig:
    """Frontend agent configuration."""

    prompt_key: str = "frontend"
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(model=""))
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)


@dataclass(frozen=True, slots=True)
class BackendToolsConfig:
    """How backend tool calls are executed and validated."""

    execution: str = "internal"
    max_tool_iterations: int = 8
    max_result_chars: int = 0
    parallel_execution: bool = False
    on_incomplete_results: str = "error"
    on_user_message_while_pending: str = "error"


@dataclass(frozen=True, slots=True)
class ConversationHistoryConfig:
    """Paired mode only: what of the conversation the backend sees.

    ``include`` is meaningful only when ``enabled``; :attr:`effective_include`
    is what actually runs, and is ``"off"`` for a disabled flag.
    """

    enabled: bool = False
    include: str = HISTORY_FULL
    guidance_key: str = "auto"
    request_key: str = "backend_history_request"

    @property
    def effective_include(self) -> str:
        """``include`` when enabled, else ``"off"``."""
        return self.include if self.enabled else "off"

    @property
    def keeps_backend_turns(self) -> bool:
        """Whether the backend's own turns persist across delegations."""
        return self.enabled and self.include != HISTORY_TRANSCRIPT

    @property
    def context_key(self) -> str:
        """Catalog key of the always-on context note."""
        return f"backend_history_context_{self.include}"

    @property
    def resolved_guidance_key(self) -> str:
        """Catalog key of the behavioural guidance, or ``""`` when disabled."""
        if not self.enabled or not self.guidance_key:
            return ""
        return f"backend_history_guidance_{self.include}" if self.guidance_key == "auto" else self.guidance_key


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """Backend agent configuration."""

    prompt_key: str = "backend"
    stateful: bool = False
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(model=""))
    tools: BackendToolsConfig = field(default_factory=BackendToolsConfig)
    history: HistoryConfig = field(default_factory=lambda: HistoryConfig(max_groups=40))
    conversation_history: ConversationHistoryConfig = field(default_factory=ConversationHistoryConfig)
    simulated_delay_s: float = 0.0


@dataclass(frozen=True, slots=True)
class DomainConfig:
    """Everything domain-specific, injected into prompts at render time."""

    policy: str = ""
    capabilities: tuple[str, ...] = ()
    unsupported_reply: str = "I can only help with the tasks I am set up for."


@dataclass(frozen=True, slots=True)
class AccountingConfig:
    """Optional local price table used when a provider reports no cost."""

    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    record_latency: bool = True


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Event-sink selection."""

    level: str = "INFO"
    event_sink: str = "logging"
    event_sink_path: str = ""
    log_filler_text: bool = True
    log_delegation_query: bool = True


@dataclass(frozen=True, slots=True)
class PromptsConfig:
    """Prompt catalog location plus optional inline overrides."""

    path: str = "prompts.yaml"
    inline: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Config:
    """Fully resolved prototype configuration."""

    name: str = "Assistant"
    persona: str = ""
    mode: str = MODE_FRONTEND_BACKEND
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    accounting: AccountingConfig = field(default_factory=AccountingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    source_dir: Path = field(default_factory=Path)

    @property
    def frontend_enabled(self) -> bool:
        """Whether the frontend agent participates in a turn."""
        return self.mode == MODE_FRONTEND_BACKEND

    @property
    def prompts_path(self) -> Path:
        """Absolute path of the prompt catalog, resolved against the config file."""
        path = Path(self.prompts.path)
        return path if path.is_absolute() else self.source_dir / path


def _llm_config(raw: dict[str, Any], *, role: str) -> LLMConfig:
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ConfigError(f"{role}.llm.model is required")
    return LLMConfig(
        model=model,
        base_url=str(raw.get("base_url") or "https://inference-api.nvidia.com/v1"),
        api_key=str(raw.get("api_key") or ""),
        temperature=float(raw.get("temperature", 0.0)),
        max_tokens=int(raw.get("max_tokens", 2048)),
        timeout_seconds=float(raw.get("timeout_seconds", 120.0)),
        extra_body=dict(raw.get("extra_body") or {}),
    )


def _one_of(value: Any, allowed: tuple[str, ...], *, name: str, default: str) -> str:
    resolved = str(value if value is not None else default).strip()
    if resolved not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got {resolved!r}")
    return resolved


def parse_bool(value: Any, *, name: str) -> bool:
    """Parse a YAML or environment boolean strictly (``"false"`` is false)."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
    raise ConfigError(f"{name} must be a boolean, got {value!r}")


def _conversation_history(raw: dict[str, Any]) -> ConversationHistoryConfig:
    name = "backend.conversation_history"
    return ConversationHistoryConfig(
        enabled=parse_bool(raw.get("enabled", False), name=f"{name}.enabled"),
        include=_one_of(raw.get("include"), _HISTORY_INCLUDES, name=f"{name}.include", default=HISTORY_FULL),
        guidance_key=str(raw.get("guidance_key", "auto") or "").strip(),
        request_key=str(raw.get("request_key") or "backend_history_request").strip(),
    )


def resolve_runtime_modes(
    mode: str, stateful_raw: Any, history: ConversationHistoryConfig | None = None
) -> tuple[str, bool]:
    """Reconcile ``agent.mode``, ``backend.stateful`` and ``backend.conversation_history``.

    ``stateful`` means "the backend history persists across turns". It holds
    for ``backend_only``, and for paired mode when the conversation history is
    enabled with the backend's own turns (``full`` or ``backend_turns``).
    Otherwise a paired backend is stateless per delegation. ``auto`` derives
    that; an explicit value is honoured but never allowed to contradict it.
    """
    history = history or ConversationHistoryConfig()
    resolved_mode = _one_of(mode, _MODES, name="agent.mode", default=MODE_FRONTEND_BACKEND)
    backend_only = resolved_mode == MODE_BACKEND_ONLY
    if backend_only and history.enabled:
        raise ConfigError(
            "backend.conversation_history is for agent.mode: frontend_backend; "
            "backend_only already keeps the full backend history"
        )
    derived = backend_only or history.keeps_backend_turns
    if stateful_raw in (None, "", "auto"):
        return resolved_mode, derived
    stateful = parse_bool(stateful_raw, name="backend.stateful")
    if stateful and not backend_only and not derived:
        raise ConfigError(
            "backend.stateful: true is incompatible with agent.mode: frontend_backend "
            "(a paired backend is stateless per delegation); "
            "set backend.conversation_history.enabled: true to give it the conversation history"
        )
    if stateful != derived:
        raise ConfigError(
            f"backend.stateful: {str(stateful).lower()} contradicts agent.mode: {resolved_mode} "
            f"with backend.conversation_history: {history.effective_include}; leave it on auto"
        )
    return resolved_mode, stateful


def load_config(path: str | Path) -> Config:
    """Load, interpolate, and validate ``agent.yaml``."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")
    raw = interpolate_env(raw)
    return build_config(raw, source_dir=config_path.parent)


def _simulated_delay(value: Any) -> float:
    try:
        delay = float(value or 0)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"backend.simulated_delay_s must be a number, got {value!r}") from exc
    if delay < 0:
        raise ConfigError(f"backend.simulated_delay_s must be >= 0, got {delay}")
    return delay


def build_config(raw: dict[str, Any], *, source_dir: Path | None = None) -> Config:
    """Build a :class:`Config` from an already-interpolated mapping."""
    agent_raw = dict(raw.get("agent") or {})
    backend_raw = dict(raw.get("backend") or {})
    frontend_raw = dict(raw.get("frontend") or {})
    history = _conversation_history(dict(backend_raw.get("conversation_history") or {}))
    mode, stateful = resolve_runtime_modes(
        agent_raw.get("mode", MODE_FRONTEND_BACKEND), backend_raw.get("stateful"), history
    )

    prompts_raw = dict(raw.get("prompts") or {})
    tools_raw = dict(backend_raw.get("tools") or {})
    domain_raw = dict(raw.get("domain") or {})
    accounting_raw = dict(raw.get("accounting") or {})
    logging_raw = dict(raw.get("logging") or {})
    delegation_raw = dict(frontend_raw.get("delegation") or {})

    frontend = FrontendConfig(
        prompt_key=str(frontend_raw.get("prompt_key") or "frontend"),
        llm=_llm_config(dict(frontend_raw.get("llm") or {}), role="frontend")
        if mode == MODE_FRONTEND_BACKEND
        else LLMConfig(model="unused"),
        delegation=DelegationConfig(
            max_repair_attempts=int(delegation_raw.get("max_repair_attempts", 1)),
            on_contract_violation=_one_of(
                delegation_raw.get("on_contract_violation"),
                _VIOLATION_POLICIES,
                name="frontend.delegation.on_contract_violation",
                default="fallback_text",
            ),
            fallback_text=str(
                delegation_raw.get("fallback_text") or "Sorry, I could not process that. Could you rephrase?"
            ),
        ),
        history=HistoryConfig(max_groups=int(dict(frontend_raw.get("history") or {}).get("max_groups", 20))),
    )

    backend = BackendConfig(
        prompt_key=str(backend_raw.get("prompt_key") or "backend"),
        stateful=stateful,
        llm=_llm_config(dict(backend_raw.get("llm") or {}), role="backend"),
        tools=BackendToolsConfig(
            execution=_one_of(
                tools_raw.get("execution"), _EXECUTION_MODES, name="backend.tools.execution", default="internal"
            ),
            max_tool_iterations=int(tools_raw.get("max_tool_iterations", 8)),
            max_result_chars=int(tools_raw.get("max_result_chars", 0)),
            parallel_execution=bool(tools_raw.get("parallel_execution", False)),
            on_incomplete_results=_one_of(
                tools_raw.get("on_incomplete_results"),
                _INCOMPLETE_POLICIES,
                name="backend.tools.on_incomplete_results",
                default="error",
            ),
            on_user_message_while_pending=_one_of(
                tools_raw.get("on_user_message_while_pending"),
                _PENDING_POLICIES,
                name="backend.tools.on_user_message_while_pending",
                default="error",
            ),
        ),
        history=HistoryConfig(max_groups=int(dict(backend_raw.get("history") or {}).get("max_groups", 40))),
        conversation_history=history,
        simulated_delay_s=_simulated_delay(backend_raw.get("simulated_delay_s", 0)),
    )

    return Config(
        name=str(agent_raw.get("name") or "Assistant"),
        persona=str(agent_raw.get("persona") or "").strip(),
        mode=mode,
        prompts=PromptsConfig(
            path=str(prompts_raw.get("path") or "prompts.yaml"),
            inline={str(key): str(value) for key, value in dict(prompts_raw.get("inline") or {}).items()},
        ),
        frontend=frontend,
        backend=backend,
        domain=DomainConfig(
            policy=str(domain_raw.get("policy") or ""),
            capabilities=tuple(str(item) for item in (domain_raw.get("capabilities") or [])),
            unsupported_reply=str(
                domain_raw.get("unsupported_reply") or "I can only help with the tasks I am set up for."
            ),
        ),
        accounting=AccountingConfig(
            pricing={
                str(model): {str(k): float(v) for k, v in dict(prices).items()}
                for model, prices in dict(accounting_raw.get("pricing") or {}).items()
            },
            record_latency=bool(accounting_raw.get("record_latency", True)),
        ),
        logging=LoggingConfig(
            level=str(logging_raw.get("level") or "INFO"),
            event_sink=str(logging_raw.get("event_sink") or "logging"),
            event_sink_path=str(logging_raw.get("event_sink_path") or ""),
            log_filler_text=bool(logging_raw.get("log_filler_text", True)),
            log_delegation_query=bool(logging_raw.get("log_delegation_query", True)),
        ),
        source_dir=source_dir or Path.cwd(),
    )
