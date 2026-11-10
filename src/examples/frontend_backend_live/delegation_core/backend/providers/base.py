# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend model provider: one round of Responses events from a model API, read through one template."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from openai import APIError, AsyncOpenAI

from examples.shared.delegation_contract import TERMINALS, ModelEndpoint

from .retry import TRANSIENT, StreamErrorEvent, with_retries


@dataclass(frozen=True)
class ProviderLimits:
    """How a provider's round is opened and timed (the ``backend.llm`` keys of the same names)."""

    provider_retries: int
    first_event_timeout_seconds: float
    idle_timeout_seconds: float


class BrokenStreamError(RuntimeError):
    """A round's stream broke the Responses rules; nothing was delivered or dispatched, so the round may be retried."""


def apply_thinking(payload: dict, extra_body: dict | None) -> bool | None:
    """Turn the endpoint's thinking switch off for a round whose ``reasoning.effort`` is ``none``; return the switch.

    The settings profile writes the switch, ``enable_thinking``, to ``extra_body.chat_template_kwargs``: a request's
    own copy of it is changed. ``None`` when the endpoint has no switch.
    """
    switch = (extra_body or {}).get("chat_template_kwargs")
    if not isinstance(switch, dict) or "enable_thinking" not in switch:
        return None
    if (payload.get("reasoning") or {}).get("effort") == "none":
        switch["enable_thinking"] = False
    return bool(switch["enable_thinking"])


# The shell environments a server runs itself; a shell tool without one runs its commands on the client.
HOSTED_SHELLS = frozenset({"container_auto", "container_reference"})


def _local_shell(tool: Mapping[str, Any]) -> bool:
    """Whether a shell tool runs its commands on the client: it names no hosted container."""
    environment = tool.get("environment")
    return not (isinstance(environment, dict) and environment.get("type") in HOSTED_SHELLS)


# The tools other than functions whose calls the client runs or approves, by type: the test of a tool of each type.
# The core passes the client only plain function calls, so a provider offers none of these to its model. A namespace's
# calls and programmatic calls are function calls with fields that the core does not pass on.
CLIENT_RUN: Mapping[str, Callable[[Mapping[str, Any]], bool]] = {
    **dict.fromkeys(
        (
            "custom",
            "computer",
            "computer_use_preview",
            "local_shell",
            "apply_patch",
            "namespace",
            "programmatic_tool_calling",
        ),
        lambda tool: True,
    ),
    "shell": _local_shell,
    "tool_search": lambda tool: tool.get("execution") == "client",
    "mcp": lambda tool: tool.get("require_approval") != "never",
}


def client_run(tool: Any) -> bool:
    """Whether the client runs or approves ``tool``'s calls, which the core does not pass on (:data:`CLIENT_RUN`)."""
    test = CLIENT_RUN.get(tool.get("type")) if isinstance(tool, Mapping) else None
    return test is not None and test(tool)


def make_client(endpoint: ModelEndpoint, limits: ProviderLimits) -> AsyncOpenAI:
    """Return the providers' OpenAI client, with no retries and a timeout above the template's limits as a backstop."""
    return AsyncOpenAI(
        api_key=endpoint.api_key or "not-needed",
        base_url=endpoint.base_url,
        timeout=limits.first_event_timeout_seconds + limits.idle_timeout_seconds,
        max_retries=0,
    )


class Backend:
    """A model provider: a subclass gives its ``name`` and :meth:`_events`, and joins ``BACKENDS``."""

    name: ClassVar[str] = ""
    # The errors worth another try when opening a round; a client that raises others sets its own.
    transient_errors: ClassVar[tuple[type[BaseException], ...]] = TRANSIENT

    def __init__(self, endpoint: ModelEndpoint, limits: ProviderLimits):
        """Bind the model endpoint and the round's limits; nothing connects yet."""
        self.endpoint, self.limits = endpoint, limits

    @classmethod
    def unsupported(cls, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings this provider cannot honor, as ``(param, message)``: each tool the client runs.

        A tool whose calls the client runs or approves, such as a custom tool, is left out of the model's request.
        """
        kinds = [str(tool.get("type")) for tool in settings.get("tools") or [] if client_run(tool)]
        return [
            (
                "tools",
                f"The {kind} tool needs the client to run or approve its calls; this backend passes the client only "
                "plain function calls.",
            )
            for kind in dict.fromkeys(kinds)
        ]

    async def stream(self, payload: dict, *, retries: int = 0) -> AsyncIterator[dict]:
        """Yield one round's Responses events up to its first terminal, within the first-event and silence limits.

        The round opens once unless the caller asks for ``retries``: only the opening is then tried again, and only
        after a transient error, so a refused request is never repeated.
        """
        events, event = await with_retries(
            lambda: self._open(payload),
            attempts=retries + 1,
            transient=self.transient_errors,
            role="backend",
        )
        try:
            while True:
                yield event
                if event.get("type") in TERMINALS:
                    return
                try:
                    event = await self._next(events)
                except StopAsyncIteration:
                    # No terminal: the adapter reads the round as broken.
                    return
        finally:
            await events.aclose()

    async def _open(self, payload: dict) -> tuple[AsyncIterator[dict | None], dict]:
        """Open the round once: return the provider's events and their first one, within the first-event limit."""
        events = self._events(payload)
        try:
            async with asyncio.timeout(self.limits.first_event_timeout_seconds):
                event = None
                while event is None:
                    event = await anext(events)
                return events, event
        except StopAsyncIteration:
            await events.aclose()
            raise BrokenStreamError("The backend's stream ended before its first event") from None
        except APIError as exc:
            await events.aclose()
            # A bare APIError is an error event in the stream: the server's state, not the request.
            if type(exc) is APIError:
                raise StreamErrorEvent(str(exc)) from exc
            raise
        except BaseException:
            await events.aclose()
            raise

    async def _next(self, events: AsyncIterator[dict | None]) -> dict:
        """Return the round's next event within the silence limit; a sign of life (``None``) restarts the limit."""
        while True:
            try:
                async with asyncio.timeout(self.limits.idle_timeout_seconds):
                    event = await anext(events)
            except APIError as exc:
                if type(exc) is APIError:
                    raise StreamErrorEvent(str(exc)) from exc
                raise
            if event is not None:
                return event

    def _events(self, payload: dict) -> AsyncIterator[dict | None]:
        """Yield one round's own events: ``response.created`` once the model answers, each item, then a terminal.

        ``payload`` is a Responses request with the whole history as ``input``. Yield ``None`` while the model works
        with nothing to pass on; :meth:`stream` adds the retries and the limits.
        """
        raise NotImplementedError(f"{type(self).__name__} gives no events")

    async def close(self) -> None:
        """Release the provider's connections."""
