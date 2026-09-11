# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict request parsing for the Realtime client-secret endpoint."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from realtime.protocol import RealtimeProtocolError, invalid_type, invalid_value

DEFAULT_CLIENT_SECRET_TTL_SECONDS = 600
MIN_CLIENT_SECRET_TTL_SECONDS = 10
MAX_CLIENT_SECRET_TTL_SECONDS = 7_200

TRUSTED_NVIDIA_SELECTOR_FIELDS = frozenset(
    {
        "pipeline_mode",
        "prompt_key",
        "llm_id",
        "thinker_llm_id",
        "asr_id",
        "tts_id",
        "asr_language_code",
        "tts_language_code",
    }
)


@dataclass(frozen=True, slots=True)
class RealtimeClientSecretRequest:
    """Validated client-secret request body."""

    session: dict[str, Any]
    ttl_seconds: int


def _reject_unknown(value: dict[str, Any], allowed: frozenset[str], *, param: str) -> None:
    unknown = sorted(set(value) - allowed)
    if not unknown:
        return
    dotted = f"{param}.{unknown[0]}" if param else unknown[0]
    raise RealtimeProtocolError(
        message=f"Unknown parameter: {dotted}",
        code="unknown_parameter",
        param=dotted,
    )


def parse_realtime_client_secret_request(value: Any) -> RealtimeClientSecretRequest:
    """Parse the supported OpenAI request plus trusted NVIDIA selectors."""
    if not isinstance(value, dict):
        raise invalid_type("Request body must be an object", param="body")
    _reject_unknown(value, frozenset({"expires_after", "session"}), param="")

    ttl_seconds = DEFAULT_CLIENT_SECRET_TTL_SECONDS
    if "expires_after" in value:
        expires_after = value["expires_after"]
        if not isinstance(expires_after, dict):
            raise invalid_type("expires_after must be an object", param="expires_after")
        _reject_unknown(expires_after, frozenset({"anchor", "seconds"}), param="expires_after")
        anchor = expires_after.get("anchor", "created_at")
        if anchor != "created_at":
            raise invalid_value("expires_after.anchor must be created_at", param="expires_after.anchor")
        seconds = expires_after.get("seconds", DEFAULT_CLIENT_SECRET_TTL_SECONDS)
        if isinstance(seconds, bool) or not isinstance(seconds, int):
            raise invalid_type("expires_after.seconds must be an integer", param="expires_after.seconds")
        if seconds < MIN_CLIENT_SECRET_TTL_SECONDS or seconds > MAX_CLIENT_SECRET_TTL_SECONDS:
            raise invalid_value(
                "expires_after.seconds must be between 10 and 7200",
                param="expires_after.seconds",
            )
        ttl_seconds = seconds

    raw_session = value.get("session")
    if raw_session is None:
        session: dict[str, Any] = {"type": "realtime"}
    elif not isinstance(raw_session, dict):
        raise invalid_type("session must be an object", param="session")
    else:
        session = copy.deepcopy(raw_session)
        if session.get("type") != "realtime":
            raise invalid_value("session.type must be realtime", param="session.type")

    raw_nvidia = session.get("nvidia")
    if raw_nvidia is not None:
        if not isinstance(raw_nvidia, dict):
            raise invalid_type("session.nvidia must be an object", param="session.nvidia")
        _reject_unknown(raw_nvidia, TRUSTED_NVIDIA_SELECTOR_FIELDS, param="session.nvidia")
        for key, selector in raw_nvidia.items():
            param = f"session.nvidia.{key}"
            if not isinstance(selector, str):
                raise invalid_type(f"{param} must be a string", param=param)
            if not selector.strip():
                raise invalid_value(f"{param} must not be empty", param=param)
            raw_nvidia[key] = selector.strip()
    return RealtimeClientSecretRequest(session=session, ttl_seconds=ttl_seconds)
