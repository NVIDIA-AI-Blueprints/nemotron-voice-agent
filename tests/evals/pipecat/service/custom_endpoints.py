# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate optional endpoint overrides for the disposable service-eval bot."""

from __future__ import annotations

import json
import os
import re
from urllib.parse import urlsplit

_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:-]*\Z")
_FUNCTION = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def _credential(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if not key or "\n" in key or "\r" in key:
        raise ValueError(f"{name} must contain a non-empty, single-line credential")
    return key


def custom_llm() -> dict | None:
    """Return validated chat-completions settings without falling back to a hosted key."""
    base_url = os.environ.get("CUSTOM_LLM_BASE_URL", "")
    model = os.environ.get("CUSTOM_LLM_MODEL_ID", "")
    if not base_url and not model:
        return None
    if not base_url or not model:
        raise ValueError("Set both CUSTOM_LLM_BASE_URL and CUSTOM_LLM_MODEL_ID")
    try:
        url = urlsplit(base_url)
        port = url.port
    except ValueError:
        raise ValueError("CUSTOM_LLM_BASE_URL is invalid") from None
    if (
        url.scheme not in {"http", "https"}
        or not url.hostname
        or port == 0
        or url.username
        or url.password
        or url.query
        or url.fragment
        or any(char.isspace() for char in base_url)
    ):
        raise ValueError("CUSTOM_LLM_BASE_URL must be an HTTP(S) base URL without credentials, query, or fragment")
    if _MODEL.fullmatch(model) is None:
        raise ValueError("CUSTOM_LLM_MODEL_ID is invalid")
    limit = os.environ.get("CUSTOM_LLM_MAX_TOKENS", "4096")
    if re.fullmatch(r"[1-9][0-9]{0,8}", limit) is None:
        raise ValueError("CUSTOM_LLM_MAX_TOKENS must be a positive integer")
    try:
        extra = json.loads(os.environ.get("CUSTOM_LLM_EXTRA_PARAMS", "{}"))
        json.dumps(extra, allow_nan=False)
    except (ValueError, TypeError):
        raise ValueError("CUSTOM_LLM_EXTRA_PARAMS must be a valid JSON object") from None
    if not isinstance(extra, dict):
        raise ValueError("CUSTOM_LLM_EXTRA_PARAMS must be a JSON object")
    reserved = {
        "api_key",
        "base_url",
        "model",
        "messages",
        "stream",
        "max_tokens",
        "headers",
        "default_headers",
        "extra_headers",
        "default_query",
        "extra_query",
    }
    body = extra.get("extra_body", {})
    if not isinstance(body, dict):
        raise ValueError("CUSTOM_LLM_EXTRA_PARAMS extra_body must be a JSON object")
    if extra.keys() & reserved or body.keys() & reserved:
        raise ValueError("Use the dedicated endpoint variables instead of routing fields in CUSTOM_LLM_EXTRA_PARAMS")
    return {
        "base_url": base_url,
        "model": model,
        "api_key": _credential("CUSTOM_LLM_API_KEY"),
        "max_tokens": int(limit),
        "extra": extra,
    }


def custom_asr() -> dict | None:
    """Return validated Riva streaming settings with explicit TLS and authentication."""
    server = os.environ.get("CUSTOM_ASR_SERVER", "")
    model = os.environ.get("CUSTOM_ASR_MODEL_ID", "")
    if not server and not model:
        return None
    if not server or not model:
        raise ValueError("Set both CUSTOM_ASR_SERVER and CUSTOM_ASR_MODEL_ID")
    if not re.fullmatch(r"[A-Za-z0-9.-]+:[1-9][0-9]{0,4}", server) or int(server.rsplit(":", 1)[1]) > 65535:
        raise ValueError("CUSTOM_ASR_SERVER must be a Riva streaming gRPC host:port, not an HTTP URL")
    if _MODEL.fullmatch(model) is None:
        raise ValueError("CUSTOM_ASR_MODEL_ID is invalid")
    use_ssl = os.environ.get("CUSTOM_ASR_USE_SSL", "true")
    if use_ssl not in {"true", "false"}:
        raise ValueError("CUSTOM_ASR_USE_SSL must be true or false")
    function = os.environ.get("CUSTOM_ASR_FUNCTION_ID", "")
    if function and (_FUNCTION.fullmatch(function) is None or use_ssl != "true"):
        raise ValueError("CUSTOM_ASR_FUNCTION_ID requires a function UUID and TLS")
    return {
        "server": server,
        "model": model,
        "api_key": _credential("CUSTOM_ASR_API_KEY"),
        "use_ssl": use_ssl == "true",
        "function_id": function,
    }


if __name__ == "__main__":
    try:
        llm, asr = custom_llm(), custom_asr()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(f"Custom cascaded endpoints: LLM={'enabled' if llm else 'disabled'}, ASR={'enabled' if asr else 'disabled'}")
    if llm or asr:
        print("Overrides apply only to Generic/Multilingual cascaded scenarios; Omni retains its configured services.")
