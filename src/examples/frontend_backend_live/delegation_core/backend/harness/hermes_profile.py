# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""An explicit model binding through Hermes's public, per-home ProviderProfile API.

The same source is copied into the owned native home as a provider plugin. It
imports no application modules and does not patch Hermes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields
from pathlib import Path
from typing import Any

PLUGIN_NAME = "voice-agent-model-binding"
REGISTRATION_FILE = "registered.json"
REASONING_FIELDS = frozenset(
    {"reasoning_effort", "reasoning_budget", "reasoning", "thinking", "thinking_config", "think", "budget_tokens"}
)
FORBIDDEN_BODY = frozenset(
    {
        "api",
        "provider",
        "model",
        "base_url",
        "api_key",
        "api_key_env",
        "authorization",
        "headers",
        "extra_headers",
        "messages",
        "input",
        "tools",
        "tool_choice",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
    }
)


def _json_value(value: Any) -> bool:
    """Do not silently coerce Python containers or non-string object keys into JSON."""
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_json_value(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _json_value(item) for key, item in value.items())
    return False


def validated_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete non-secret route and supported authored request fields."""
    required = {"provider", "model", "base_url", "api", "api_key_env", "extra_params"}
    if set(binding) != required:
        raise ValueError("Hermes model binding needs provider, model, base_url, api, api_key_env and extra_params")
    result = deepcopy(dict(binding))
    if result["api"] != "chat-completions":
        raise ValueError("Hermes model binding requires chat-completions")
    for name in required - {"api", "extra_params"}:
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"Hermes model binding {name} must be non-empty text")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", result["api_key_env"]):
        raise ValueError("Hermes model binding api_key_env must name an environment variable")
    params = result["extra_params"]
    if not isinstance(params, Mapping) or set(params) - {"temperature", "top_p", "max_tokens", "extra_body"}:
        raise ValueError("Hermes model binding supports temperature, top_p, max_tokens and extra_body")
    for name in ("temperature", "top_p"):
        if name in params and (
            isinstance(params[name], bool)
            or not isinstance(params[name], (int, float))
            or not math.isfinite(params[name])
        ):
            raise ValueError(f"Hermes model binding {name} must be a finite number")
    if "max_tokens" in params and (type(params["max_tokens"]) is not int or params["max_tokens"] <= 0):
        raise ValueError("Hermes model binding max_tokens must be a positive integer")
    body = params.get("extra_body", {})
    if not isinstance(body, Mapping) or any(not isinstance(key, str) for key in body):
        raise ValueError("Hermes model binding extra_body must be a JSON object")
    if any(key.lower() in FORBIDDEN_BODY for key in body):
        raise ValueError("Hermes extra_body cannot override routing, credentials, messages, tools or transport fields")
    if not _json_value(result):
        raise ValueError("Hermes model binding must contain only finite JSON values")
    try:
        # Disallow non-JSON values and NaN rather than changing them for the native runtime.
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("Hermes model binding must contain only finite JSON values") from None


def install_profile(home: Path, binding: Mapping[str, Any]) -> None:
    """Install this explicit public extension only in an already owned, session-specific home."""
    document = validated_binding(binding)
    home.mkdir(parents=True, mode=0o700, exist_ok=True)
    if home.is_symlink() or home.stat().st_uid != os.getuid():
        raise ValueError("Hermes model binding home must be owned by the current user")
    os.chmod(home, 0o700)
    target = home / "plugins" / "model-providers" / PLUGIN_NAME
    target.mkdir(parents=True, mode=0o700, exist_ok=True)
    (target / REGISTRATION_FILE).unlink(missing_ok=True)
    for name, content in (
        ("binding.json", json.dumps(document, sort_keys=True, allow_nan=False)),
        ("__init__.py", Path(__file__).read_text() + "\nregister()\n"),
    ):
        path = target / name
        if path.exists():
            if path.is_symlink() or path.read_text() != content:
                raise ValueError("Hermes native home already contains a different model-binding extension")
            continue
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)


def binding_digest(binding: Mapping[str, Any]) -> str:
    """Fingerprint public model configuration without including a credential value."""
    return hashlib.sha256(json.dumps(validated_binding(binding), sort_keys=True).encode()).hexdigest()


def require_registered_profile(home: Path, binding: Mapping[str, Any]) -> None:
    """Refuse a native startup that skipped or rejected the required extension, before invoking a model."""
    receipt = home / "plugins" / "model-providers" / PLUGIN_NAME / REGISTRATION_FILE
    try:
        if receipt.is_symlink() or receipt.stat().st_uid != os.getuid():
            raise ValueError("unowned registration receipt")
        document = json.loads(receipt.read_text())
        if (
            document.get("binding_sha256") != binding_digest(binding)
            or type(document.get("pid")) is not int
            or document["pid"] <= 0
        ):
            raise ValueError("different registration receipt")
    except (OSError, ValueError):
        raise RuntimeError(
            "Hermes did not register the exact required model-binding extension; refusing model work"
        ) from None


def register() -> None:
    """Extend the selected native profile through Hermes's documented provider registry."""
    from providers import get_provider_profile, register_provider

    binding = validated_binding(json.loads(Path(__file__).with_name("binding.json").read_text()))
    base = get_provider_profile(binding["provider"])
    if base is None or base.api_mode != "chat_completions":
        raise ValueError("Hermes model binding needs an installed chat-completions provider profile")

    def checked_route(model: str | None, base_url: str | None) -> None:
        if model != binding["model"] or str(base_url or "").rstrip("/") != binding["base_url"].rstrip("/"):
            raise ValueError("Hermes model binding differs from the native model or endpoint")
        if not os.environ.get(binding["api_key_env"]):
            raise ValueError("Hermes model binding credential reference is unavailable")

    def checked_reasoning(inherited: Mapping[str, Any], context: Mapping[str, Any]) -> None:
        authored = binding["extra_params"].get("extra_body", {})
        unexpected = {
            key
            for key in REASONING_FIELDS.intersection(inherited)
            if key not in authored or inherited[key] != authored[key]
        }
        if not unexpected:
            return
        # A native operator's explicit reasoning settings still belong to the
        # original provider. An internal one-shot override is not such a setting.
        from hermes_cli.config import load_config
        from hermes_constants import resolve_reasoning_config

        configured = resolve_reasoning_config(load_config(), model=binding["model"])
        if configured is not None and context.get("reasoning_config") == configured:
            return
        raise ValueError("Hermes native provider emitted unauthored reasoning settings; refusing model request")

    class BoundProfile(type(base)):
        """Preserve provider behavior and carry the explicit Core-authored request fields."""

        def default_reasoning_config(self, model: str | None = None) -> None:
            # Omission stays omission. Explicit native reasoning configuration still
            # reaches the original provider hook unchanged; no capability is falsified.
            if model != binding["model"]:
                raise ValueError("Hermes model binding differs from the native model")
            return None

        def create_client(self, **kwargs: Any) -> Any:
            if str(kwargs.get("base_url") or "").rstrip("/") != binding["base_url"].rstrip("/"):
                raise ValueError("Hermes model binding differs from the native endpoint")
            selected_key = os.environ.get(binding["api_key_env"])
            if not selected_key or kwargs.get("api_key") != selected_key:
                raise ValueError("Hermes model binding differs from the native credential reference")
            return super().create_client(**kwargs)

        def build_extra_body(self, **context: Any) -> dict[str, Any]:
            checked_route(context.get("model"), context.get("base_url"))
            body = super().build_extra_body(**context)
            checked_reasoning(body, context)
            return {**body, **deepcopy(binding["extra_params"].get("extra_body", {}))}

        def build_api_kwargs_extras(self, **context: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            checked_route(context.get("model"), context.get("base_url"))
            body, top = super().build_api_kwargs_extras(**context)
            checked_reasoning(body, context)
            checked_reasoning(top, context)
            authored = {key: value for key, value in binding["extra_params"].items() if key != "extra_body"}
            return body, {**top, **authored}

    register_provider(
        BoundProfile(**{field.name: deepcopy(getattr(base, field.name)) for field in fields(base) if field.init})
    )
    receipt = Path(__file__).with_name(REGISTRATION_FILE)
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"binding_sha256": binding_digest(binding), "pid": os.getpid()}, stream)
