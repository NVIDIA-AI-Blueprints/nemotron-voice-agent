# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Served profiles, the route's LLM endpoints, and the WebSocket keepalive check.

Two profiles sit in ``profiles/`` next to this module: ``default`` (browser UI sessions, used when
``FBA_PROFILE`` is unset) and ``eval`` (OpenAI Realtime sessions and evaluation runs). Every
Realtime session runs with ``eval``; ``FBA_PROFILE`` decides whether browser sessions are served
(``default``) or refused (``eval``). Each profile's Pipecat settings file is chosen here
(:data:`PIPECAT_SETTINGS`). Two values come from this repository instead of the profile YAML:

* the LLM endpoints (``model`` and ``base_url`` of each role) come from the service
  catalog entries the Realtime route selected (``llm`` = frontend, ``thinker-llm`` =
  backend); every request parameter stays the profile's (``core/config/agent.yaml``);
* the WebSocket keepalive is uvicorn's, so it is checked against the profile.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from examples.frontend_backend_agent.core.config.catalog import SpeechEndpoint, build_endpoint, load_catalog_entry
from examples.frontend_backend_agent.core.config.voice import CONFIG_DIR, VoiceConfig

PROFILES_DIR = CONFIG_DIR / "profiles"
PROFILE_ENV = "FBA_PROFILE"
#: Browser UI sessions (WebRTC, the UI's WebSocket); used when FBA_PROFILE is unset.
DEFAULT_PROFILE = "default"
#: OpenAI Realtime sessions and evaluation runs. Every Realtime session uses it, whichever
#: profile the container serves; ``FBA_PROFILE=eval`` also refuses browser UI sessions.
REALTIME_PROFILE = "eval"
#: Every profile FBA_PROFILE may select.
SERVED_PROFILES = (DEFAULT_PROFILE, REALTIME_PROFILE)
#: The Pipecat VAD/turn settings file of each profile (``adapters/pipecat/settings/``).
PIPECAT_SETTINGS = {DEFAULT_PROFILE: "default.yaml", REALTIME_PROFILE: "eval.yaml"}
#: The deployment platform the repository's Compose services set (cloud, server, singlegpu).
SPEECH_PLATFORM_ENV = "REALTIME_SERVICE_PLATFORM"
#: The example directory, which holds services.cloud.yaml and services.local.yaml.
EXAMPLE_DIR = CONFIG_DIR.parent.parent
#: The variables that replace a catalog speech server (``*.catalog.server_override`` in voice_agent.yaml).
SPEECH_SERVER_ENV = {"asr": "FBA_ASR_SERVER", "tts": "FBA_TTS_SERVER"}
#: uvicorn's defaults, used by src/server.py when UVICORN_WS_PING_* are unset.
UVICORN_DEFAULT_PING_S = 20.0


class ProfileError(RuntimeError):
    """The selected profile cannot be served by this deployment."""


def profile_name(environ: Mapping[str, str] | None = None) -> str:
    """The profile selected by ``FBA_PROFILE`` (unset: ``default``); any other value is an error."""
    env = os.environ if environ is None else environ
    name = (env.get(PROFILE_ENV) or DEFAULT_PROFILE).strip()
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    if name not in SERVED_PROFILES:
        raise ProfileError(
            f"{PROFILE_ENV}={name!r} is not a profile of this example; set {PROFILE_ENV} to "
            f"{DEFAULT_PROFILE!r} (browser UI) or {REALTIME_PROFILE!r} (evaluation runs), or leave it unset"
        )
    return name


def profile_path(name: str) -> Path:
    """The YAML file of a served profile."""
    return PROFILES_DIR / f"{name}.yaml"


def pipecat_settings_file(name: str) -> str:
    """The Pipecat settings file of a served profile (chosen here, not in the profile YAML)."""
    return PIPECAT_SETTINGS[name]


def _float_env(env: Mapping[str, str], key: str, default: float) -> float:
    """Parse ``key`` the way ``uvicorn_ws_ping_kwargs`` does: invalid values fall back, negatives clamp to 0."""
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if not math.isfinite(value):
        return default
    return max(value, 0.0)


def keepalive_mismatch(config: VoiceConfig, environ: Mapping[str, str] | None = None) -> str:
    """A message when the server's WebSocket keepalive differs from the profile's, else ``""``.

    The profile's ``server.ws_ping_interval_s`` (``0``: off) and ``server.ws_ping_timeout_s``
    are compared with what ``src/server.py`` passes to uvicorn from ``UVICORN_WS_PING_INTERVAL``
    / ``UVICORN_WS_PING_TIMEOUT``.
    """
    env = os.environ if environ is None else environ
    interval = _float_env(env, "UVICORN_WS_PING_INTERVAL", UVICORN_DEFAULT_PING_S) or None
    timeout = _float_env(env, "UVICORN_WS_PING_TIMEOUT", UVICORN_DEFAULT_PING_S)
    wanted_interval = config.server.ws_ping_interval_s or None
    wanted_timeout = config.server.ws_ping_timeout_s
    if (interval, timeout) == (wanted_interval, wanted_timeout):
        return ""
    return (
        f"WebSocket keepalive differs from the profile: server ping interval={interval} timeout={timeout}, "
        f"profile ws_ping_interval_s={config.server.ws_ping_interval_s} ws_ping_timeout_s={wanted_timeout}; "
        f"set UVICORN_WS_PING_INTERVAL={config.server.ws_ping_interval_s:g} "
        f"UVICORN_WS_PING_TIMEOUT={wanted_timeout:g}"
    )


@dataclass(frozen=True, slots=True)
class LLMEndpoints:
    """Model and base URL per LLM role, from the Realtime route's catalog selection."""

    frontend_model: str = ""
    frontend_base_url: str = ""
    backend_model: str = ""
    backend_base_url: str = ""

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> LLMEndpoints:
        """Read the hydrated catalog fields (``model_id``/``base_url``, ``thinker_*``)."""
        return cls(
            frontend_model=str(body.get("model_id") or ""),
            frontend_base_url=str(body.get("base_url") or ""),
            backend_model=str(body.get("thinker_model_id") or ""),
            backend_base_url=str(body.get("thinker_base_url") or ""),
        )


def apply_llm_endpoints(config: VoiceConfig, endpoints: LLMEndpoints) -> VoiceConfig:
    """Return ``config`` with the catalog's model and base URL per role; other LLM settings unchanged."""
    agent = config.agent

    def _llm(llm: Any, model: str, base_url: str) -> Any:
        return replace(llm, model=model or llm.model, base_url=base_url or llm.base_url)

    frontend = replace(
        agent.frontend, llm=_llm(agent.frontend.llm, endpoints.frontend_model, endpoints.frontend_base_url)
    )
    backend = replace(agent.backend, llm=_llm(agent.backend.llm, endpoints.backend_model, endpoints.backend_base_url))
    return replace(config, agent=replace(agent, frontend=frontend, backend=backend))


def speech_selection_warnings(config: VoiceConfig, body: Mapping[str, Any]) -> list[str]:
    """Differences between the route's ASR/TTS catalog selection and the profile's resolved endpoints.

    The profile resolves ASR and TTS itself (``asr``/``tts.catalog`` in the profile,
    with ``FBA_ASR_SERVER``/``FBA_TTS_SERVER`` overrides); the registry selection is shown to
    clients. A difference is logged so a deployment that changes one also changes the other.
    """
    warnings = []
    for kind, endpoint in (("asr", config.asr.endpoint), ("tts", config.tts.endpoint)):
        selected = str(body.get(f"{kind}_server") or "")
        if selected and selected != endpoint.server:
            warnings.append(
                f"{kind}: the Realtime route selected server {selected!r}, but the profile's speech endpoint is "
                f"{endpoint.server!r} ({endpoint.source}); set FBA_{kind.upper()}_SERVER to change it"
            )
    return warnings


def _platform_catalog(platform: str) -> tuple[str, str] | None:
    """The catalog file and section for a platform, or ``None`` to keep the profile's endpoints."""
    if platform == "server":
        return "local", "server"
    if platform == "cloud":
        return "cloud", ""
    return None


def _remapped(endpoint: SpeechEndpoint, *, services: str, section: str, env: Mapping[str, str]) -> SpeechEndpoint:
    if not endpoint.source.startswith("catalog ") or "] " not in endpoint.source:
        return endpoint  # an inline endpoint is the deployment's own choice
    key = endpoint.source.rsplit("] ", 1)[1]
    entry = load_catalog_entry(endpoint.kind, example_dir=EXAMPLE_DIR, services=services, platform=section, key=key)
    return build_endpoint(
        endpoint.kind,
        entry,
        source=f"catalog {EXAMPLE_DIR}/services.{services}.yaml [{section or '-'}] {key}",
        server_override=str(env.get(SPEECH_SERVER_ENV[endpoint.kind]) or "").strip(),
        environ=dict(env),
    )


def apply_speech_platform(config: VoiceConfig, environ: Mapping[str, str] | None = None) -> VoiceConfig:
    """Point ASR and TTS at the deployment platform's catalog entries (plan decision D2).

    The profiles resolve speech from the single-GPU section of ``services.local.yaml`` (the
    NeMo-Speech sidecar). On the ``server`` platform the same catalog keys are read from the
    ``server`` section of ``services.local.yaml``; on ``cloud`` from ``services.cloud.yaml``.
    For ``singlegpu``, an unset platform or any other value, ``config`` is returned unchanged (the
    same object), so evaluation containers and host-native runs keep their endpoints exactly.
    The lookup is FBA's own deterministic catalog read: no reachability probing and no host
    rewriting; ``FBA_ASR_SERVER`` / ``FBA_TTS_SERVER`` still replace the server.
    """
    env = os.environ if environ is None else environ
    target = _platform_catalog(str(env.get(SPEECH_PLATFORM_ENV) or "").strip())
    if target is None:
        return config
    services, section = target
    asr = _remapped(config.asr.endpoint, services=services, section=section, env=env)
    tts = _remapped(config.tts.endpoint, services=services, section=section, env=env)
    return replace(config, asr=replace(config.asr, endpoint=asr), tts=replace(config.tts, endpoint=tts))
