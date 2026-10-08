# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a ``services.yaml`` entry and the session body into the endpoint a model role talks to."""

from __future__ import annotations

import os

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import LiveConfig, ModelEndpoint
from utils import load_service_entry, nvidia_api_key, parse_json_dict

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def resolve_endpoint(
    entry: dict, body: dict, prefix: str = "", *, fallback: ModelEndpoint | None = None
) -> ModelEndpoint:
    """Resolve one role's model endpoint.

    The session ``body`` (UI overrides) wins over the ``services.yaml`` ``entry``, which wins over ``fallback``.
    An entry can set ``api_key_env`` to the name of the environment variable that holds its key, for endpoints
    that do not use ``NVIDIA_API_KEY``. ``extra_params`` carries every sampling and reasoning setting
    (``temperature``, ``max_tokens``, ``extra_body`` ...) because ``settings.yaml`` writes them there.

    Args:
        entry: The selected catalog entry (``{}`` when unknown).
        body: The session body.
        prefix: Body key prefix: ``""`` for the frontend, ``"thinker_"`` for the backend.
        fallback: Endpoint to inherit from when neither the body nor the entry sets a field.

    Returns:
        The endpoint.
    """
    model = body.get(f"{prefix}model_id") or entry.get("model_id") or (fallback.model if fallback else DEFAULT_MODEL)
    base_url = (
        body.get(f"{prefix}base_url") or entry.get("base_url") or (fallback.base_url if fallback else DEFAULT_BASE_URL)
    )
    extra = parse_json_dict(
        body.get(f"{prefix}extra_params") or entry.get("extra_params", ""), label=f"{prefix}extra_params"
    )
    max_tokens = body.get(f"{prefix}max_tokens") or entry.get("max_tokens")
    if max_tokens not in (None, ""):
        try:
            extra["max_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {prefix}max_tokens {max_tokens!r}; ignoring")
    key_env = str(entry.get("api_key_env") or "").strip()
    return ModelEndpoint(
        model=model,
        base_url=base_url,
        api_key=(os.getenv(key_env) or "").strip() or nvidia_api_key() if key_env else nvidia_api_key(),
        extra_params=extra,
        response_format=str(entry.get("response_format") or "json_schema"),
    )


def resolve_role_endpoints(config: LiveConfig, body: dict) -> tuple[ModelEndpoint, ModelEndpoint]:
    """Resolve the frontend's and the backend's endpoints from the slots ``config.yaml`` names.

    Every host (the live engines, the repository UI bot, the console's ``describe``) resolves models here, so a role
    cannot end up on a different model depending on how the call arrived. The backend inherits what its own entry
    leaves unset from the frontend.
    """
    frontend = resolve_endpoint(load_service_entry(config.frontend.slot, ""), body)
    backend = resolve_endpoint(load_service_entry(config.backend.slot, ""), body, "thinker_", fallback=frontend)
    return frontend, backend
