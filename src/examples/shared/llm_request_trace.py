# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Passive, opt-in snapshots at native LLM preparation and HTTP dispatch.

``build_chat_completion_params`` observes the parameters returned by Pipecat,
including its context adapter and configured extra-body overrides. The HTTPX
hook separately observes the SDK's serialized JSON request body. Neither is
evidence of remote model consumption. Authentication and arbitrary headers,
URL query strings and provider response bodies are never retained. The SDK's
retry header is retained only as a validated nonnegative numeric counter.

The HTTP hook uses the pinned OpenAI client's private ``_client`` member to
reach HTTPX's public ``event_hooks`` API. An unsupported client remains usable
and is reported as an observation capability gap. No provider loop, retry,
queue, timer or request parameter is replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import httpx
from openai import NotGiven, Omit
from pipecat.frames.frames import LLMContextFrame

_BODY_PARAMETERS = frozenset(
    {
        "messages",
        "model",
        "stream",
        "stream_options",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "service_tier",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
        "extra_body",
        "n",
        "stop",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "user",
        "audio",
        "modalities",
        "prediction",
        "reasoning_effort",
        "verbosity",
        "store",
        "metadata",
    }
)


@dataclass
class _Request:
    identity: str
    attempts: int = 0


@dataclass
class _Scope:
    owner: str
    context_frame_id: int | None = None
    generation: str | None = None
    request: _Request | None = None


_SCOPE: ContextVar[_Scope | None] = ContextVar("nva_llm_request_trace_scope", default=None)


def current_llm_request_trace() -> dict[str, Any]:
    """Return detached identities for inline native frame callbacks.

    A native response-start frame can precede parameter preparation, so its
    ``request_id`` can be null while its ``context_frame_id`` is already known.
    This function intentionally exposes no request text or mutable scope.
    """
    scope = _SCOPE.get()
    if scope is None:
        return {}
    return {
        "request_id": scope.request.identity if scope.request else None,
        "context_frame_id": scope.context_frame_id,
        "generation": scope.generation,
    }


def _json_snapshot(value: Any) -> Any:
    """Copy JSON-shaped values without consuming iterators or guessing strings."""
    if isinstance(value, NotGiven):
        return {"$openai_omitted": "NotGiven"}
    if isinstance(value, Omit):
        return {"$openai_omitted": "Omit"}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_snapshot(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_snapshot(item) for key, item in value.items()}
    raise TypeError("LLM observation requires JSON-shaped request parameters")


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _note_failure(trace, error: Exception) -> None:
    with suppress(Exception):
        trace._note_failure(error)


def _record(trace, event: str, **fields) -> None:
    try:
        trace.record(event, **fields)
    except Exception as error:
        _note_failure(trace, error)


class LLMRequestTraceMixin:
    """Delegate a native service while observing immutable request snapshots."""

    def __init__(self, *args, evaluation_trace=None, **kwargs):
        """Attach only the opt-in request hook after native client initialization."""
        self._evaluation_trace = evaluation_trace
        self._llm_trace_owner = str(uuid.uuid4())
        self._llm_trace_index = 0
        self._llm_trace_http_client = None
        self._llm_trace_http_hook = None
        super().__init__(*args, **kwargs)
        if evaluation_trace is not None:
            self._attach_llm_request_hook()

    def _attach_llm_request_hook(self) -> None:
        trace = self._evaluation_trace
        try:
            client = getattr(getattr(self, "_client", None), "_client", None)
            if not isinstance(client, httpx.AsyncClient):
                _record(
                    trace, "llm_request_capture_capability", serialized_http_body=False, reason="unsupported_client"
                )
                return

            async def on_request(request: httpx.Request) -> None:
                # HTTPX awaits request hooks. There is deliberately no await or
                # I/O here: the body is already serialized by the native SDK.
                self._observe_llm_http_request(request)

            client.event_hooks.setdefault("request", []).append(on_request)
            self._llm_trace_http_client = client
            self._llm_trace_http_hook = on_request
            _record(
                trace,
                "llm_request_capture_capability",
                serialized_http_body=True,
                mechanism="AsyncOpenAI._client -> HTTPX.event_hooks.request",
                pipecat_version=version("pipecat-ai"),
                openai_version=version("openai"),
                httpx_version=version("httpx"),
            )
        except Exception as error:
            _note_failure(trace, error)
            _record(
                trace, "llm_request_capture_capability", serialized_http_body=False, reason="hook_attachment_failed"
            )

    def _observe_llm_http_request(self, request: httpx.Request) -> None:
        trace = self._evaluation_trace
        try:
            if request.method != "POST" or not request.url.path.rstrip("/").endswith("/chat/completions"):
                return
            scope = _SCOPE.get()
            # A supplied HTTP client can be shared by services. Its other
            # service's scoped or untraced requests must not be retained by
            # this trace. A shared client alone does not establish ownership.
            if scope is None or scope.owner != self._llm_trace_owner:
                return
            body_bytes = request.content
            body = json.loads(body_bytes)
            if not isinstance(body, dict):
                raise TypeError("Chat completion JSON body is not an object")
            identity = current_llm_request_trace()
            attempt = None
            if scope is not None and scope.request is not None:
                scope.request.attempts += 1
                attempt = scope.request.attempts
            raw_retry = request.headers.get("x-stainless-retry-count", "")
            sdk_retry_count = (
                int(raw_retry) if len(raw_retry) <= 9 and raw_retry.isascii() and raw_retry.isdecimal() else None
            )
            _record(
                trace,
                "llm_http_request",
                **identity,
                http_attempt=attempt,
                sdk_retry_count=sdk_retry_count,
                request_body=body,
                request_body_sha256=_canonical_hash(body),
                serialized_body_sha256=hashlib.sha256(body_bytes).hexdigest(),
                serialized_body_bytes=len(body_bytes),
                capture_boundary="httpx_request_hook_before_transport_send",
                request_identity_observed=bool(scope and scope.request),
            )
        except Exception as error:
            _note_failure(trace, error)

    def build_chat_completion_params(self, params_from_context):
        """Snapshot native converted options and return its identical parameter object."""
        params = super().build_chat_completion_params(params_from_context)
        trace = self._evaluation_trace
        if trace is None:
            return params
        try:
            self._llm_trace_index += 1
            request = _Request(f"{self._llm_trace_owner}:{self._llm_trace_index}")
            scope = _SCOPE.get()
            if scope is not None and scope.owner == self._llm_trace_owner:
                scope.request = request
                identity = current_llm_request_trace()
            else:
                identity = {"request_id": request.identity, "context_frame_id": None, "generation": None}
            # Call options such as extra_headers are not request-body fields
            # and can contain authentication. Their values are never copied.
            snapshot = _json_snapshot({key: value for key, value in params.items() if key in _BODY_PARAMETERS})
            _record(
                trace,
                "llm_request_prepared",
                **identity,
                parameters=snapshot,
                parameters_sha256=_canonical_hash(snapshot),
                excluded_call_option_names=sorted(key for key in params if key not in _BODY_PARAMETERS),
                capture_boundary="after_native_parameter_conversion_before_sdk",
            )
        except Exception as error:
            _note_failure(trace, error)
        return params

    async def get_chat_completions(self, context):
        """Scope direct native calls too, without wrapping or changing the returned stream."""
        if self._evaluation_trace is None:
            return await super().get_chat_completions(context)
        scope = _SCOPE.get()
        if scope is not None and scope.owner == self._llm_trace_owner:
            return await super().get_chat_completions(context)
        token = _SCOPE.set(_Scope(self._llm_trace_owner))
        try:
            return await super().get_chat_completions(context)
        finally:
            _SCOPE.reset(token)

    async def process_frame(self, frame, direction):
        """Retain request identity through native frame emission and reset on every exit."""
        if self._evaluation_trace is None:
            return await super().process_frame(frame, direction)
        scope = _Scope(
            self._llm_trace_owner,
            context_frame_id=frame.id if isinstance(frame, LLMContextFrame) else None,
            generation=frame.metadata.get("nva_offline_response_generation"),
        )
        token = _SCOPE.set(scope)
        try:
            return await super().process_frame(frame, direction)
        finally:
            _SCOPE.reset(token)

    async def cleanup(self):
        """Remove only this session's hook after the native processor cleanup."""
        try:
            return await super().cleanup()
        finally:
            try:
                if self._llm_trace_http_client is not None and self._llm_trace_http_hook is not None:
                    hooks = self._llm_trace_http_client.event_hooks.get("request", [])
                    if self._llm_trace_http_hook in hooks:
                        hooks.remove(self._llm_trace_http_hook)
                    self._llm_trace_http_hook = None
                    self._llm_trace_http_client = None
            except Exception as error:
                if self._evaluation_trace is not None:
                    _note_failure(self._evaluation_trace, error)
