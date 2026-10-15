# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Served profiles, the route's LLM endpoints, and the WebSocket keepalive check.

The profiles sit in ``profiles/`` next to this module. Two values come from this
repository instead of the profile YAML:

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

from examples.frontend_backend_verdict.core.config.voice import CONFIG_DIR, VoiceConfig

PROFILES_DIR = CONFIG_DIR / "profiles"
PROFILE_ENV = "FBV_PROFILE"
#: The Tau profiles (plan D11), all on the Tau prompt catalog. Each extends tau3_eval.yaml, so each
#: has the 800 ms, server-owned turn detection of the registry's session defaults.
SHIPPED_PROFILES = (
    "tau3_eval_frontend_verdict_speak_history",
    "tau3_eval_frontend_verdict_speak",
    "tau3_eval_frontend_verdict",
    "tau3_eval_backend_history",
    "tau3_eval_backend_history_noguide",
    "tau3_eval_normalization",
    "tau3_eval",
    # The default arm plus the R4 write gate (consequential tool calls held until the caller confirms).
    "tau3_eval_frontend_verdict_speak_history_write_gate",
)
#: The default Tau arm (when FBV_PROFILE is unset): a missing variable never changes a Tau run's prompts.
DEFAULT_PROFILE = SHIPPED_PROFILES[0]
#: The default Tau arm with the generic prompt catalog (``core/config/prompts/generic.yaml``).
GENERIC_PROFILE = "default"
#: Every profile FBV_PROFILE may select.
SERVED_PROFILES = (*SHIPPED_PROFILES, GENERIC_PROFILE)
#: uvicorn's defaults, used by src/server.py when UVICORN_WS_PING_* are unset.
UVICORN_DEFAULT_PING_S = 20.0


class ProfileError(RuntimeError):
    """The selected profile cannot be served by this deployment."""


def profile_name(environ: Mapping[str, str] | None = None) -> str:
    """The profile selected by ``FBV_PROFILE`` (default: the frontend-verdict + history arm)."""
    env = os.environ if environ is None else environ
    name = (env.get(PROFILE_ENV) or DEFAULT_PROFILE).strip()
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    if name not in SERVED_PROFILES:
        raise ProfileError(f"{PROFILE_ENV}={name!r} is not served by this example; choose one of {SERVED_PROFILES}")
    return name


def profile_path(name: str) -> Path:
    """The YAML file of a served profile."""
    return PROFILES_DIR / f"{name}.yaml"


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
