# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One streamed chat-completions call with retries and an abort that works from another thread.

The request body sent to the OpenAI-compatible endpoint (llm-backend-plan.md 6.7):
``model``, ``messages``, ``tools``, ``stream`` with usage, top-level ``reasoning_effort``, and the
``request_overrides`` merged into the body. No ``temperature``, ``max_tokens`` or ``tool_choice``.
"""

from __future__ import annotations

import contextlib
import random
import socket
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx
import openai

#: Backoff of a retried call: base * 2^(attempt-1), capped, plus up to 50% jitter.
BACKOFF_BASE_S = 2.0
BACKOFF_MAX_S = 60.0
#: A server's Retry-After is honoured up to this (a voice turn cannot wait longer).
RETRY_AFTER_MAX_S = 30.0

_RETRYABLE = (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError)


class Aborted(Exception):  # noqa: N818 - a control-flow signal, not an error
    """The call was aborted by :meth:`ChatClient.abort` (redirect or interrupt)."""


@dataclass(slots=True)
class ChatResult:
    """One assistant reply, assembled from the stream."""

    content: str = ""
    reasoning: str = ""
    #: ``[{"id", "name", "arguments"}]`` in the order emitted.
    tool_calls: list[dict[str, str]] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


class ChatClient:
    """The OpenAI SDK client of one LLM agent; one call at a time."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float,
        max_retries: int,
        reasoning_effort: str,
        request_overrides: Mapping[str, Any],
    ) -> None:
        """``timeout_s`` is per read, so a stalled stream fails after it."""
        self._new_client = lambda: openai.OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self._client: openai.OpenAI | None = None
        self.model = model
        self._max_retries = max_retries
        overrides = dict(request_overrides or {})
        self._extra_body = {**dict(overrides.pop("extra_body", None) or {}), **overrides}
        self._reasoning_effort = reasoning_effort
        self._lock = threading.Lock()
        #: The streaming response of the current call (an abandoned call never replaces it).
        self._stream: Any = None
        self._call: object | None = None
        self._abort = threading.Event()

    def body(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, Any]:
        """The keyword arguments of ``chat.completions.create`` (also what the trace records)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        if self._reasoning_effort:
            kwargs["reasoning_effort"] = self._reasoning_effort
        if self._extra_body:
            kwargs["extra_body"] = dict(self._extra_body)
        return kwargs

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        on_attempt: Callable[[int], None] | None = None,
    ) -> ChatResult:
        """Stream one reply; retries connection, timeout, 429, 5xx and mid-stream cuts; raises :class:`Aborted`."""
        self._abort.clear()
        attempt = 0
        while True:
            attempt += 1
            if on_attempt is not None:
                on_attempt(attempt)
            try:
                return self._abortable(messages, tools)
            except Aborted:
                raise
            except (*_RETRYABLE, httpx.TransportError) as exc:
                if self._abort.is_set():
                    raise Aborted from exc
                if attempt >= self._max_retries:
                    raise
                if self._abort.wait(self._delay(attempt, exc)):
                    raise Aborted from exc

    def abort(self) -> None:
        """Stop the call in flight (any thread): :meth:`complete` raises :class:`Aborted` at once.

        A streaming response is closed and its socket shut down. A request still waiting for
        its headers (prefill) cannot be cut from another thread; it is abandoned and ends on
        its own thread, its reply discarded.
        """
        self._abort.set()
        with self._lock:
            stream = self._stream
        if stream is None:
            return
        response = getattr(stream, "response", None)
        network = getattr(response, "extensions", {}).get("network_stream") if response is not None else None
        sock = network.get_extra_info("socket") if network is not None else None
        if isinstance(sock, socket.socket):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(Exception):
            stream.close()

    def close(self) -> None:
        """Release the HTTP client."""
        with self._lock:
            client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()

    # -- internals ----------------------------------------------------------------------

    def _abortable(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        """:meth:`_once` on its own thread, so an abort returns now even while the request waits for headers."""
        done = threading.Event()
        box: dict[str, Any] = {}
        call = object()
        with self._lock:
            self._call = call

        def run() -> None:
            try:
                box["result"] = self._once(messages, tools, call)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
                box["error"] = exc
            finally:
                done.set()

        threading.Thread(target=run, name="fda-llm-call", daemon=True).start()
        while not done.wait(0.02):
            if self._abort.is_set():
                raise Aborted
        if "error" in box:
            raise box["error"]
        return box["result"]

    def _once(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, call: object) -> ChatResult:
        if self._abort.is_set():
            raise Aborted
        with self._lock:
            if self._client is None:
                self._client = self._new_client()
            client = self._client
        try:
            stream = client.chat.completions.create(**self.body(messages, tools))
        except Exception as exc:
            if self._abort.is_set():
                raise Aborted from exc
            raise
        with self._lock:
            if self._call is call:
                self._stream = stream

        def stop() -> bool:  # aborted, or abandoned by a newer call: stop reading
            return self._abort.is_set() or self._call is not call

        try:
            if stop():
                raise Aborted
            return _assemble(stream, stop)
        except Exception as exc:
            if self._abort.is_set():
                raise Aborted from exc
            raise
        finally:
            with self._lock:
                if self._call is call:
                    self._stream = None
            with contextlib.suppress(Exception):
                stream.close()

    def _delay(self, attempt: int, exc: BaseException) -> float:
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        with contextlib.suppress(TypeError, ValueError):
            if header is not None:
                return min(max(float(header), 0.0), RETRY_AFTER_MAX_S)
        delay = min(BACKOFF_BASE_S * 2 ** (attempt - 1), BACKOFF_MAX_S)
        return delay + random.uniform(0, delay / 2)  # noqa: S311 - jitter, not security


def _assemble(stream: Any, stop: Callable[[], bool]) -> ChatResult:
    result = ChatResult()
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    for chunk in stream:
        if stop():
            raise Aborted
        if chunk.usage is not None:
            details = getattr(chunk.usage, "completion_tokens_details", None)
            result.usage = {
                "input_tokens": int(chunk.usage.prompt_tokens or 0),
                "output_tokens": int(chunk.usage.completion_tokens or 0),
                "reasoning_tokens": int(getattr(details, "reasoning_tokens", 0) or 0),
            }
        for choice in chunk.choices or ():
            delta = choice.delta
            if delta is not None:
                if delta.content:
                    content.append(delta.content)
                extra = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if isinstance(extra, str) and extra:
                    reasoning.append(extra)
                for call in delta.tool_calls or ():
                    slot = calls.setdefault(call.index, {"id": "", "name": "", "arguments": []})
                    if call.id:
                        slot["id"] = call.id
                    if call.function is not None:
                        if call.function.name:
                            slot["name"] += call.function.name
                        if call.function.arguments:
                            slot["arguments"].append(call.function.arguments)
            if choice.finish_reason:
                result.finish_reason = choice.finish_reason
    result.content = "".join(content)
    result.reasoning = "".join(reasoning)
    result.tool_calls = [
        {"id": slot["id"], "name": slot["name"], "arguments": "".join(slot["arguments"])}
        for _, slot in sorted(calls.items())
    ]
    return result
