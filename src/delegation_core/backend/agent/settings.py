# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's settings: the ``backend.llm-agent`` block and every default.

:meth:`AgentSettings.parse` checks the block when the core is built, so an unknown key, a fraction where a whole number
is due, or an unknown name fails the build. The ``worker`` block takes the backend LLM's keys (``backend.llm``) except
``conductor``, since the agent's conductor has its own block.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from delegation_core.backend.llm import LlmSettings
from delegation_core.backend.providers import BACKENDS, Backend, ChatCompletionsBackend
from delegation_core.tables import pick

BLOCK = "backend.llm-agent"
# The keys of whole numbers in the conductor's block, with their least value.
CONDUCTOR_COUNTS = {"memory_tokens": 1, "max_output_tokens_per_request": 1}


def _mapping(where: str, raw: Any, allowed: set[str]) -> Mapping[str, Any]:
    """``raw`` as a mapping whose keys are all in ``allowed``; ``ValueError`` otherwise."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where} must be a mapping")
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    return raw


def _count(where: str, value: Any, least: int) -> int:
    """A whole number of at least ``least``: a fraction or a boolean is refused."""
    if isinstance(value, bool) or not isinstance(value, int) or value < least:
        raise ValueError(f"{where} must be a whole number of at least {least}")
    return value


def _number(where: str, value: Any, *, positive: bool) -> float:
    """A number: positive, or (``positive=False``) not negative."""
    number = not isinstance(value, bool) and isinstance(value, int | float)
    if not number or value < 0 or (positive and value == 0):
        raise ValueError(f"{where} must be a {'positive' if positive else 'non-negative'} number")
    return float(value)


def _name(where: str, value: Any) -> str | None:
    """Non-empty text, or ``None``."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be non-empty text or null")
    return value


@dataclass(frozen=True)
class ConductorSettings:
    """How the agent's conductor model is called: one call per ``decide()``, each decision through a control tool."""

    provider: type[Backend] = ChatCompletionsBackend  # chat-completions | openai-responses (providers.BACKENDS)
    model: str | None = None  # a model of the catalog; None: the backend's default (the slot's)
    # A decision round that takes longer places nothing more; the backend LLM's steer rule places its requests.
    timeout_seconds: float = 20.0
    memory_tokens: int = 8000  # the conductor's own memory of earlier decision rounds, kept or dropped whole
    max_output_tokens_per_request: int = 256  # the round's output cap, for each new request it places
    temperature: float | None = 0.0  # sampling for a placement; None: the endpoint's own
    reasoning_effort: str | None = "none"  # the official reasoning.effort (none: thinking off); None: the model's own

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> ConductorSettings:
        """Validate the ``conductor`` block and turn its provider's name into the provider class.

        Raises:
            ValueError: A key it does not have, a value of the wrong kind, or a provider no table has.
        """
        where = f"{BLOCK}.conductor"
        block = _mapping(where, block, {f.name for f in fields(cls)})
        values: dict[str, Any] = {}
        if "provider" in block:
            values["provider"] = pick(BACKENDS, block["provider"], "backend provider")
        if "model" in block:
            values["model"] = _name(f"{where}.model", block["model"])
        if "timeout_seconds" in block:
            values["timeout_seconds"] = _number(f"{where}.timeout_seconds", block["timeout_seconds"], positive=True)
        for key, least in CONDUCTOR_COUNTS.items():
            if key in block:
                values[key] = _count(f"{where}.{key}", block[key], least)
        if "temperature" in block:
            temperature = block["temperature"]
            values["temperature"] = (
                None if temperature is None else _number(f"{where}.temperature", temperature, positive=False)
            )
        if "reasoning_effort" in block:
            values["reasoning_effort"] = _name(f"{where}.reasoning_effort", block["reasoning_effort"])
        return cls(**values)

    def describe(self) -> dict:
        """The settings as plain JSON, by name."""
        return {
            "provider": self.provider.name,
            "timeout_seconds": self.timeout_seconds,
            "memory_tokens": self.memory_tokens,
            "max_output_tokens_per_request": self.max_output_tokens_per_request,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class AgentSettings:
    """The backend LLM agent's settings, with every name already turned into what it names."""

    max_running: int = 3  # tasks running at once in one IM session
    conductor: ConductorSettings = field(default_factory=ConductorSettings)
    worker: LlmSettings = field(default_factory=LlmSettings)  # the backend LLM's keys, for the workers

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> AgentSettings:
        """Validate the ``backend.llm-agent`` block.

        Raises:
            ValueError: A key the agent does not have, a value of the wrong kind, a name no table has, or a
                ``conductor`` in the worker's block.
        """
        block = _mapping(BLOCK, block, {f.name for f in fields(cls)})
        values: dict[str, Any] = {}
        if "max_running" in block:
            values["max_running"] = _count(f"{BLOCK}.max_running", block["max_running"], 1)
        if "conductor" in block:
            values["conductor"] = ConductorSettings.parse(block["conductor"])
        if "worker" in block:
            worker = block["worker"]
            if not isinstance(worker, Mapping):
                raise ValueError(f"{BLOCK}.worker must be a mapping of the backend LLM's keys")
            if "conductor" in worker:
                raise ValueError(f"{BLOCK}.worker takes no conductor: the agent's conductor is {BLOCK}.conductor")
            try:
                values["worker"] = LlmSettings.parse(worker)
            except ValueError as exc:
                raise ValueError(f"{BLOCK}.worker: {exc}") from None
        return cls(**values)
