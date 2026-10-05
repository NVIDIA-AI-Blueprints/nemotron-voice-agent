# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""What a Frontend/Backend session runs with, published in ``session.updated``.

A recorded conversation is only comparable with another when both name the
agent build, the Thinker settings that planned it and the feature switches.
The pipeline and this module resolve the Thinker settings with the same
helpers, so the published values are the ones the session uses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from examples.frontend_backend_agent.src.flags import effective_flags

#: Default Thinker output budget when neither the catalog nor the session sets one.
DEFAULT_THINKER_MAX_TOKENS = 4096
#: Reasoning budget used when ``GENERIC_THINKER_REASONING_BUDGET`` is set but does not parse.
DEFAULT_REASONING_BUDGET = 1024


def thinker_env_overrides(domain_key: str) -> tuple[str, str]:
    """Return the raw ``GENERIC_THINKER_MAX_TOKENS`` and ``GENERIC_THINKER_REASONING_BUDGET`` for a domain.

    Both apply to the generic domain only; other domains keep their catalog values.
    """
    if domain_key != "generic":
        return "", ""
    return (
        os.getenv("GENERIC_THINKER_MAX_TOKENS", "").strip(),
        os.getenv("GENERIC_THINKER_REASONING_BUDGET", "").strip(),
    )


def reasoning_settings(extra_params: object) -> tuple[bool | None, int | None]:
    """Return ``(enable_thinking, reasoning_budget)`` from a Thinker's ``extra_params``."""
    if isinstance(extra_params, str):
        try:
            extra_params = json.loads(extra_params) if extra_params.strip() else {}
        except json.JSONDecodeError:
            return None, None
    extra_body = extra_params.get("extra_body") if isinstance(extra_params, Mapping) else None
    if not isinstance(extra_body, Mapping):
        return None, None
    template = extra_body.get("chat_template_kwargs")
    template = template if isinstance(template, Mapping) else {}
    enable = template.get("enable_thinking")
    budget = extra_body.get("reasoning_budget", template.get("reasoning_budget"))
    return (
        enable if isinstance(enable, bool) else None,
        budget if isinstance(budget, int) and not isinstance(budget, bool) else None,
    )


def session_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the agent build, effective Thinker settings and switches for one session config."""
    domain_key = str(config.get("domain_profile") or "")
    env_max_tokens, env_budget = thinker_env_overrides(domain_key)
    enable_thinking, catalog_budget = reasoning_settings(config.get("thinker_extra_params"))
    # Same precedence and fallbacks as the pipeline: the first value set wins,
    # and one that does not parse falls back to the default, not to the next.
    max_tokens = _int(env_max_tokens or config.get("thinker_max_tokens"), DEFAULT_THINKER_MAX_TOKENS)
    thinker: dict[str, Any] = {
        "llm_id": str(config.get("thinker_llm_id") or ""),
        "model_id": str(config.get("thinker_model_id") or ""),
        "max_tokens": max_tokens,
        "reasoning_budget": _int(env_budget, DEFAULT_REASONING_BUDGET) if env_budget else catalog_budget,
    }
    if enable_thinking is not None:
        thinker["reasoning"] = enable_thinking
    return {
        "agent_git_sha": os.getenv("AGENT_GIT_SHA", "").strip() or "unknown",
        "domain": domain_key,
        "thinker": thinker,
        "flags": effective_flags(),
    }


def _int(raw: object, default: int) -> int:
    if raw in (None, "") or isinstance(raw, bool):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
