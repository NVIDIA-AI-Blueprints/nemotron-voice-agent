# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend gateway configuration (plan sections 11 and 11.1).

``config/gateway.yaml`` is loaded with ``extends:`` chains (depth first, the child
wins), ``${VAR}`` / ``${VAR:-default}`` interpolation per file before YAML parsing,
deep merge of mappings, strict keys against :data:`DEFAULTS` and the cross-checks
of section 11.1. The result is a frozen :class:`GatewayConfig`.

``backend:`` holds what every backend shares (``kind``, model, timeouts, prompts); each
registered backend with settings has its own section named after its kind (``hermes:``,
``llm:``), parsed by the settings class its spec names (llm-backend-plan.md section 3).
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from examples.frontend_delegation_agent.core.backends import BACKENDS, KINDS, BackendSpec, Placement, get_spec, load
from examples.frontend_delegation_agent.core.legacy_env import legacy_env_error
from examples.frontend_delegation_agent.core.prompt_features import BACKEND_FEATURES, features

PACKAGE_DIR = Path(__file__).resolve().parents[2]  # the example root
CONFIG_DIR = PACKAGE_DIR / "config"
DEFAULT_GATEWAY_CONFIG = CONFIG_DIR / "gateway.yaml"
EMULATOR_GATEWAY_CONFIG = CONFIG_DIR / "gateway.emulator.yaml"
SRC_DIR = PACKAGE_DIR.parents[1]

WORKER_MODES = ("process_per_session", "in_process")
#: ``workers.mode`` -> the placement it needs from the backend.
MODE_PLACEMENT = {"process_per_session": Placement.WORKER_PROCESS, "in_process": Placement.IN_PROCESS}
AGENT_KINDS = KINDS
CONSTRUCT_MODES = ("eager", "lazy")

#: Settings class of every backend that has its own section, by kind.
SETTINGS_CLASSES: dict[str, Any] = {kind: load(spec.settings) for kind, spec in BACKENDS.items() if spec.settings}

DEFAULTS: dict[str, Any] = {
    "extends": None,
    "gateway": {
        "host": "127.0.0.1",
        "port": 8790,
        "max_sessions": 8,
        "log": "logs/fda_gateway_events.jsonl",
    },
    "workers": {
        "mode": "process_per_session",
        "python": "~/.cache/fda/hermes-venv-314/bin/python",
        "hermes_repo": "",
        "start_timeout_s": 15.0,
        "configure_timeout_s": 30.0,
        "stop_timeout_s": 12.0,
        "kill_grace_s": 3.0,
        "warm": 0,
        "home_root": ".cache/fda-hermes-homes",
        "keep_homes": False,
        "emulator_default_tool": False,
        "socket_dir": "/tmp/fda-workers",
        "log_dir": "logs/fda_workers",
    },
    "recovery": {"respawn": True, "max_respawns_per_session": 2},
    "backend": {
        "kind": "hermes",
        "model": "nvidia/nvidia/nemotron-3-ultra",
        "base_url": "https://inference-api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY",
        "agent_construct": "eager",
        "max_iterations": 30,
        "run_budget_seconds": 300.0,
        "run_hard_deadline_s": 330.0,
        "unwind_timeout_s": 10.0,
        "tool_result_timeout_s": 120.0,
        "steer_timeout_s": 3.0,
        "status_timeout_s": 2.0,
        # P0.3: each model request and response of a worker, written to
        # <workers.log_dir>/<session_id>.trace.jsonl (logging only; the input of the G2 decision replay).
        "backend_trace": False,
        "request_overrides": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024}
        },
        # ``{kind}`` is replaced by ``backend.kind``: every backend reads its own prompt file.
        "prompts": {"path": "prompts/generic/backend/{kind}.yaml"},
    },
    # Each backend's own section (``hermes:``, ``llm:``), from its settings class.
    **{kind: dict(cls.DEFAULTS) for kind, cls in SETTINGS_CLASSES.items()},
    # Backend prompt variants: all off = the baseline prompts.
    "prompt_features": dict(BACKEND_FEATURES),
    # Domain detection: name -> {tools_any: [tool, ...]}; the first
    # domain with one of its tools in the session's tool list wins. It selects the domain note of
    # prompts/<profile>/backend/<kind>.yaml; no domains (or no match) = no note.
    "domains": {},
}

#: Keys whose mapping value is free-form (no strict key check below them).
FREE_FORM = {("backend", "request_overrides"), ("domains",)}

#: Path keys resolved against the declaring file (``in``) or the working directory (``out``).
IN_PATH_KEYS = {("backend", "prompts", "path")}

#: Keys that moved from ``hermes:`` to ``backend:`` (llm-backend-plan.md section 3).
SHARED_KEYS = (
    "model",
    "base_url",
    "api_key_env",
    "agent_construct",
    "max_iterations",
    "run_budget_seconds",
    "run_hard_deadline_s",
    "unwind_timeout_s",
    "tool_result_timeout_s",
    "steer_timeout_s",
    "status_timeout_s",
    "backend_trace",
    "request_overrides",
    "prompts",
)
OUT_PATH_KEYS = {
    ("gateway", "log"),
    ("workers", "python"),
    ("workers", "hermes_repo"),
    ("workers", "home_root"),
    ("workers", "socket_dir"),
    ("workers", "log_dir"),
}

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class GatewayConfigError(ValueError):
    """An invalid gateway configuration."""


@dataclass(frozen=True, slots=True)
class GatewaySection:
    """Listener and capacity."""

    host: str
    port: int
    max_sessions: int
    log: str


@dataclass(frozen=True, slots=True)
class WorkersSection:
    """Worker processes (plan section 4.2)."""

    mode: str
    python: str
    hermes_repo: str
    start_timeout_s: float
    configure_timeout_s: float
    stop_timeout_s: float
    kill_grace_s: float
    warm: int
    home_root: str
    keep_homes: bool
    emulator_default_tool: bool
    socket_dir: str
    log_dir: str


@dataclass(frozen=True, slots=True)
class RecoverySection:
    """Respawn policy after a worker death (plan section 5.3)."""

    respawn: bool
    max_respawns_per_session: int


@dataclass(frozen=True, slots=True)
class BackendSection:
    """Settings every backend shares (``backend:``); the backend's own ones are in ``backend_settings``."""

    kind: str
    model: str
    base_url: str
    api_key_env: str
    agent_construct: str
    max_iterations: int
    run_budget_seconds: float
    run_hard_deadline_s: float
    unwind_timeout_s: float
    tool_result_timeout_s: float
    steer_timeout_s: float
    status_timeout_s: float
    request_overrides: Mapping[str, Any]
    prompts_path: str
    backend_trace: bool = False

    def worker_settings(self) -> dict[str, Any]:
        """The shared part of the ``backend`` block of the worker ``configure`` message."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "agent_construct": self.agent_construct,
            "max_iterations": self.max_iterations,
            "run_budget_seconds": self.run_budget_seconds,
            "unwind_timeout_s": self.unwind_timeout_s,
            "tool_result_timeout_s": self.tool_result_timeout_s,
            "request_overrides": copy.deepcopy(dict(self.request_overrides)),
        }


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """The whole gateway configuration."""

    gateway: GatewaySection
    workers: WorkersSection
    recovery: RecoverySection
    backend: BackendSection
    #: The selected backend's own section (``HermesSettings``, ``LLMSettings``); ``None`` without one.
    backend_settings: Any = None
    files: tuple[str, ...] = field(default=())
    prompt_features: Mapping[str, bool] = field(default_factory=lambda: dict(BACKEND_FEATURES))
    #: ``(domain, tools_any)`` in detection order.
    domains: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def spec(self) -> BackendSpec:
        """The registry entry of ``backend.kind``."""
        return get_spec(self.backend.kind)

    def worker_settings(self) -> dict[str, Any]:
        """The ``backend`` block of the worker ``configure`` message: shared plus the backend's own settings."""
        settings = self.backend.worker_settings()
        if self.backend_settings is not None:
            settings.update(self.backend_settings.worker_settings())
        return settings

    def summary(self) -> dict[str, Any]:
        """Effective settings for the ``session_open`` log record (no secrets live here).

        The key names ``workers.agent_kind`` (= ``backend.kind``) and ``hermes`` (= the shared ``backend:``
        settings) are kept from before the backend registry, so log readers and the frozen eval pins
        (``tests/.../pipecat/fixtures/eval_pins.json``) see the same record for Hermes.
        """
        return {
            "max_sessions": self.gateway.max_sessions,
            "workers": {
                "mode": self.workers.mode,
                "agent_kind": self.backend.kind,
                "warm": self.workers.warm,
                "start_timeout_s": self.workers.start_timeout_s,
                "stop_timeout_s": self.workers.stop_timeout_s,
            },
            "hermes": {
                "model": self.backend.model,
                "agent_construct": self.backend.agent_construct,
                "run_budget_seconds": self.backend.run_budget_seconds,
                "run_hard_deadline_s": self.backend.run_hard_deadline_s,
                "backend_trace": self.backend.backend_trace,
            },
            "recovery": {
                "respawn": self.recovery.respawn,
                "max_respawns_per_session": self.recovery.max_respawns_per_session,
            },
            "prompt_features": dict(self.prompt_features),
            "domains": [name for name, _ in self.domains],
        }


# -- loading ------------------------------------------------------------------------


def interpolate_env(text: str, env: Mapping[str, str] | None = None) -> str:
    """Replace ``${VAR}`` and ``${VAR:-default}``; an unset variable without default is an error."""
    env = os.environ if env is None else env

    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        value = env.get(name)
        if value is not None and value != "":
            return value
        if default is not None:
            return default
        raise GatewayConfigError(f"environment variable {name} is not set and has no default")

    return _ENV.sub(replace, text)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Mappings merge recursively; scalars and lists replace; ``None`` resets to the base value."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_file(path: Path) -> dict[str, Any]:
    text = interpolate_env(path.read_text(encoding="utf-8"))
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise GatewayConfigError(f"{path}: the top level must be a mapping")
    _resolve_in_paths(data, path.parent)
    return data


def _resolve_in_paths(data: dict[str, Any], base: Path) -> None:
    for keys in IN_PATH_KEYS:
        node: Any = data
        for key in keys[:-1]:
            node = node.get(key) if isinstance(node, dict) else None
        if isinstance(node, dict) and isinstance(node.get(keys[-1]), str) and node[keys[-1]]:
            candidate = Path(node[keys[-1]]).expanduser()
            node[keys[-1]] = str(candidate if candidate.is_absolute() else (base / candidate).resolve())


def _load_chain(path: Path, seen: tuple[Path, ...] = ()) -> tuple[dict[str, Any], tuple[Path, ...]]:
    path = path.resolve()
    if path in seen:
        raise GatewayConfigError(f"extends cycle: {' -> '.join(str(p) for p in (*seen, path))}")
    data = _load_file(path)
    parent = data.pop("extends", None)
    if not parent:
        return data, (*seen, path)
    parent_path = Path(parent).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base, files = _load_chain(parent_path, (*seen, path))
    return deep_merge(base, data), files


def _check_unknown(merged: Mapping[str, Any], schema: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> None:
    for key, value in merged.items():
        dotted = (*prefix, key)
        if key not in schema:
            raise GatewayConfigError(f"unknown config key {'.'.join(dotted)!r}")
        if dotted in FREE_FORM:
            continue
        if isinstance(schema[key], dict) and isinstance(value, Mapping):
            _check_unknown(value, schema[key], dotted)


def load_gateway_config(
    path: str | Path | None = DEFAULT_GATEWAY_CONFIG, *, overrides: Mapping[str, Any] | None = None
) -> GatewayConfig:
    """Load ``path`` (``None`` = defaults only), apply ``overrides`` and validate."""
    if message := legacy_env_error():
        raise GatewayConfigError(message)
    files: tuple[Path, ...] = ()
    loaded: dict[str, Any] = {}
    if path is not None:
        loaded, files = _load_chain(Path(path))
    _check_renamed(loaded, overrides or {})
    merged = deep_merge(DEFAULTS, loaded)
    if overrides:
        merged = deep_merge(merged, overrides)
    merged.pop("extends", None)
    _check_unknown(merged, DEFAULTS)
    return build_gateway_config(merged, files)


def default_emulator_gateway_config(**overrides: Any) -> GatewayConfig:
    """In-process emulator workers, no files: the default of :class:`InProcessBackendLink` (tests)."""
    base = {
        "workers": {"mode": "in_process", "start_timeout_s": 5.0, "stop_timeout_s": 2.0},
        "backend": {
            "kind": "emulator",
            "unwind_timeout_s": 1.0,
            "prompts": {"path": str(CONFIG_DIR / "prompts" / "generic" / "backend" / "{kind}.yaml")},
        },
    }
    return load_gateway_config(None, overrides=deep_merge(base, overrides))


def _check_renamed(*sources: Mapping[str, Any]) -> None:
    """Name the new key for every key the backend registry renamed (one release; llm-backend-plan.md section 3)."""
    for source in sources:
        workers = source.get("workers") if isinstance(source.get("workers"), Mapping) else {}
        if "agent_kind" in workers:
            raise GatewayConfigError(
                "workers.agent_kind was replaced by backend.kind (environment variable FDA_BACKEND, "
                "formerly FDH_AGENT_KIND)"
            )
        if workers.get("mode") == "in_process_fake":
            raise GatewayConfigError("workers.mode: in_process_fake was renamed to in_process")
        if "fake_default_tool" in workers:
            raise GatewayConfigError("workers.fake_default_tool was renamed to workers.emulator_default_tool")
        backend = source.get("backend") if isinstance(source.get("backend"), Mapping) else {}
        if backend.get("kind") == "fake":
            raise GatewayConfigError("backend.kind: fake was renamed to emulator (FDA_BACKEND=emulator)")
        hermes = source.get("hermes") if isinstance(source.get("hermes"), Mapping) else {}
        moved = sorted(key for key in hermes if key in SHARED_KEYS)
        if moved:
            raise GatewayConfigError(
                f"hermes.{moved[0]} moved to backend.{moved[0]} (every backend shares it); "
                f"move all of {', '.join('hermes.' + key for key in moved)}"
            )


def _out_path(value: str) -> str:
    if not value or ("/" not in value and not value.startswith("~")):
        return value  # empty, a command name looked up on PATH, or "self"
    candidate = Path(value).expanduser()
    return str(candidate if candidate.is_absolute() else Path.cwd() / candidate)


def _as(kind: type, value: Any, name: str) -> Any:
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false", "1", "0", "yes", "no", "on", "off"):
            return value.lower() in ("true", "1", "yes", "on")
        raise GatewayConfigError(f"{name}: expected a boolean, got {value!r}")
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise GatewayConfigError(f"{name}: expected {kind.__name__}, got {value!r}") from exc


def build_gateway_config(merged: Mapping[str, Any], files: tuple[Path, ...] = ()) -> GatewayConfig:
    """Typed, cross-checked sections from a merged mapping."""
    g, w, r, b = merged["gateway"], merged["workers"], merged["recovery"], merged["backend"]

    def pick(section: Mapping[str, Any], name: str, key: str, kind: type) -> Any:
        return _as(kind, section[key], f"{name}.{key}")

    gateway = GatewaySection(
        host=pick(g, "gateway", "host", str),
        port=pick(g, "gateway", "port", int),
        max_sessions=pick(g, "gateway", "max_sessions", int),
        log=_out_path(pick(g, "gateway", "log", str)),
    )
    kind = pick(b, "backend", "kind", str)
    if kind not in KINDS:
        raise GatewayConfigError(f"backend.kind must be one of {KINDS}, got {kind!r}")
    spec = get_spec(kind)
    workers = WorkersSection(
        mode=_mode_for(spec, pick(w, "workers", "mode", str)),
        python=_out_path(pick(w, "workers", "python", str)),
        hermes_repo=_out_path(pick(w, "workers", "hermes_repo", str)),
        start_timeout_s=pick(w, "workers", "start_timeout_s", float),
        configure_timeout_s=pick(w, "workers", "configure_timeout_s", float),
        stop_timeout_s=pick(w, "workers", "stop_timeout_s", float),
        kill_grace_s=pick(w, "workers", "kill_grace_s", float),
        warm=pick(w, "workers", "warm", int),
        home_root=_out_path(pick(w, "workers", "home_root", str)),
        keep_homes=pick(w, "workers", "keep_homes", bool),
        emulator_default_tool=pick(w, "workers", "emulator_default_tool", bool),
        socket_dir=_out_path(pick(w, "workers", "socket_dir", str)),
        log_dir=_out_path(pick(w, "workers", "log_dir", str)),
    )
    recovery = RecoverySection(
        respawn=pick(r, "recovery", "respawn", bool),
        max_respawns_per_session=pick(r, "recovery", "max_respawns_per_session", int),
    )
    prompts = b.get("prompts") or {}
    # ``{kind}`` is plain text in the path (resolved per file already), so it is filled in here.
    prompts_path = str(prompts.get("path") or "").replace("{kind}", spec.prompt_kind or kind)
    if prompts_path and not Path(prompts_path).is_absolute():
        prompts_path = str((CONFIG_DIR / prompts_path).resolve())
    backend = BackendSection(
        kind=kind,
        model=pick(b, "backend", "model", str),
        base_url=pick(b, "backend", "base_url", str),
        api_key_env=pick(b, "backend", "api_key_env", str),
        agent_construct=pick(b, "backend", "agent_construct", str),
        max_iterations=pick(b, "backend", "max_iterations", int),
        run_budget_seconds=pick(b, "backend", "run_budget_seconds", float),
        run_hard_deadline_s=pick(b, "backend", "run_hard_deadline_s", float),
        unwind_timeout_s=pick(b, "backend", "unwind_timeout_s", float),
        tool_result_timeout_s=pick(b, "backend", "tool_result_timeout_s", float),
        steer_timeout_s=pick(b, "backend", "steer_timeout_s", float),
        status_timeout_s=pick(b, "backend", "status_timeout_s", float),
        request_overrides=copy.deepcopy(dict(b.get("request_overrides") or {})),
        prompts_path=prompts_path,
        backend_trace=pick(b, "backend", "backend_trace", bool),
    )
    backend_settings = None
    if kind in SETTINGS_CLASSES:
        section = merged[kind]
        try:
            backend_settings = SETTINGS_CLASSES[kind].parse(lambda key, typ: pick(section, kind, key, typ))
        except ValueError as exc:
            if isinstance(exc, GatewayConfigError):
                raise
            raise GatewayConfigError(str(exc)) from exc
    try:
        prompt_features = dict(features(merged.get("prompt_features"), BACKEND_FEATURES))
    except ValueError as exc:
        raise GatewayConfigError(f"prompt_features: {exc}") from exc
    config = GatewayConfig(
        gateway,
        workers,
        recovery,
        backend,
        backend_settings,
        tuple(str(p) for p in files),
        prompt_features,
        _domains(merged.get("domains")),
    )
    _cross_check(config)
    return config


def _domains(raw: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, Mapping):
        raise GatewayConfigError("domains must map domain names to {tools_any: [tool, ...]}")
    out: list[tuple[str, tuple[str, ...]]] = []
    for name, spec in raw.items():
        key = f"domains.{name}"
        if not isinstance(name, str) or not name.strip():
            raise GatewayConfigError(f"{key}: the domain name must be a non-empty string")
        if not isinstance(spec, Mapping) or set(spec) != {"tools_any"}:
            raise GatewayConfigError(f"{key} must be a mapping with exactly one key, tools_any")
        tools = spec["tools_any"]
        if not isinstance(tools, list) or not tools or not all(isinstance(t, str) and t.strip() for t in tools):
            raise GatewayConfigError(f"{key}.tools_any must be a non-empty list of tool names")
        out.append((name.strip(), tuple(t.strip() for t in tools)))
    return tuple(out)


def _mode_for(spec: BackendSpec, mode: str) -> str:
    """``workers.mode``, switched to ``in_process`` for a backend that runs only in-process (``llm``)."""
    if mode not in WORKER_MODES:
        raise GatewayConfigError(f"workers.mode must be one of {WORKER_MODES}, got {mode!r}")
    if spec.supports(MODE_PLACEMENT[mode]):
        return mode
    if spec.supports(Placement.IN_PROCESS):
        return "in_process"
    raise GatewayConfigError(f"workers.mode: {mode} is not supported by backend.kind: {spec.kind}")


def _cross_check(config: GatewayConfig) -> None:
    w, h = config.workers, config.backend
    if h.agent_construct not in CONSTRUCT_MODES:
        raise GatewayConfigError(f"backend.agent_construct must be one of {CONSTRUCT_MODES}")
    if config.gateway.max_sessions < 1:
        raise GatewayConfigError("gateway.max_sessions must be at least 1")
    if w.warm < 0:
        raise GatewayConfigError("workers.warm must be >= 0")
    if h.run_hard_deadline_s <= h.run_budget_seconds:
        raise GatewayConfigError("backend.run_hard_deadline_s must be greater than backend.run_budget_seconds")
    if w.stop_timeout_s <= h.unwind_timeout_s:
        raise GatewayConfigError("workers.stop_timeout_s must be greater than backend.unwind_timeout_s")
    for name in ("start_timeout_s", "configure_timeout_s", "stop_timeout_s", "kill_grace_s"):
        if getattr(w, name) <= 0:
            raise GatewayConfigError(f"workers.{name} must be positive")
    if config.recovery.max_respawns_per_session < 0:
        raise GatewayConfigError("recovery.max_respawns_per_session must be >= 0")
    if not h.prompts_path:
        raise GatewayConfigError("backend.prompts.path is required")
