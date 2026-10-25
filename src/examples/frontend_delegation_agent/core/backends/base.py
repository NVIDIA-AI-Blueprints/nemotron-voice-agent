# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every backend implements (llm-backend-plan.md section 2).

A backend is a folder under ``core/backends/`` with a :class:`BackendSpec` in its
``__init__.py``. The spec names its classes as ``"module:attribute"`` strings, so reading
the registry imports nothing heavy: Hermes and the OpenAI SDK load only inside the worker
that runs them.

Stdlib only: the voice server (Python 3.12) and the Hermes worker (Python 3.14) both import it.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class Placement(StrEnum):
    """Where a backend's worker may run."""

    #: The gateway spawns one worker process per session (``workers.mode: process_per_session``).
    WORKER_PROCESS = "worker_process"
    #: ``WorkerCore`` runs on a thread in the caller's process (``workers.mode: in_process``).
    IN_PROCESS = "in_process"


class AgentLike(Protocol):
    """What the worker core needs from an agent (Hermes ``AIAgent``, ``LLMAgent`` or ``EmulatorAgent``)."""

    def run_conversation(
        self, user_message: str, conversation_history: list[dict[str, Any]], task_id: str
    ) -> dict[str, Any]:
        """Run one turn (blocking)."""
        ...

    def steer(self, text: str) -> bool:
        """Queue text for delivery after the current tool batch."""
        ...

    def redirect(self, text: str) -> bool:
        """Abort the in-flight model request and retry with ``text``; ``False`` outside one."""
        ...

    def get_activity_summary(self) -> dict[str, Any]:
        """What the agent is doing now."""
        ...

    def clear_interrupt(self) -> None:
        """Drop an interrupt left over from an idle period."""
        ...

    def hard_interrupt(self, message: str) -> None:
        """Stop the running turn as soon as possible."""
        ...

    def model_request_active(self) -> bool:
        """Whether a model request is in flight."""
        ...

    def usage_snapshot(self) -> dict[str, int]:
        """Cumulative token counters."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class AgentHost(Protocol):
    """Builds and owns the agent of one worker."""

    def configure(self, *, tools: list[dict[str, Any]], instructions: str, settings: dict[str, Any]) -> list[str]:
        """Register the session's tools (and build the agent when eager); returns the tool names.

        ``settings`` is the ``backend`` block of the worker ``configure`` message: the shared
        ``backend:`` values plus the backend's own section (``BackendSettings.worker_settings``).
        """
        ...

    def agent(self) -> AgentLike:
        """The agent, built on first use when construction is lazy."""
        ...

    @property
    def built(self) -> bool:
        """Whether the agent exists."""
        ...

    def is_interrupted(self) -> bool:
        """Whether the calling tool thread was interrupted by the agent."""
        ...

    def version(self) -> str:
        """Agent implementation version, for ``ready``/``configured``."""
        ...

    def close(self) -> None:
        """Release the agent and deregister tools."""
        ...


class BackendSettings(Protocol):
    """A backend's own config section (``hermes:``, ``llm:``), parsed by its ``settings`` class."""

    def worker_settings(self) -> dict[str, Any]:
        """The section's fields that go into the worker ``configure`` message."""
        ...

    def summary(self) -> dict[str, Any]:
        """The section's fields for logs and ``/health`` (no secrets)."""
        ...


@dataclass(frozen=True, slots=True)
class WorkerPrep:
    """What a backend needs around a worker process: its home directory and environment."""

    home: Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    #: Extra ``PYTHONPATH`` entries (after the example's own package).
    pythonpath: tuple[str, ...] = ()


class WorkerSetup(Protocol):
    """Per-process preparation of a ``WORKER_PROCESS`` backend (Hermes: ``HERMES_HOME``)."""

    def prepare(self, config: Any, name: str, *, soul: str) -> WorkerPrep:
        """Create what one worker process needs before it starts."""
        ...

    def remove(self, prep: WorkerPrep) -> None:
        """Delete what :meth:`prepare` created."""
        ...

    def clear_stale(self, config: Any) -> int:
        """Remove leftovers of a crashed gateway; returns how many were removed."""
        ...


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """One backend, as the registry knows it. Every ``"module:attribute"`` is imported lazily."""

    kind: str
    #: ``AgentHost`` class; constructed as ``Host(futures, on_activity, session_id=..., in_process=...)``.
    host: str
    placements: frozenset[Placement]
    #: Settings class of the backend's own config section (``kind:``); ``None`` = no section.
    settings: str | None = None
    #: Prompt keys this backend needs in ``prompts/<profile>/backend/<kind>.yaml``, besides the common ones.
    required_prompt_keys: tuple[str, ...] = ()
    #: The ``{kind}`` of its prompt file when it reuses another backend's (emulator: ``hermes``); ``None`` = own kind.
    prompt_kind: str | None = None
    #: ``WorkerSetup`` class for ``WORKER_PROCESS`` workers; ``None`` = nothing to prepare.
    worker_setup: str | None = None
    #: ``func(templates) -> dict`` adding rendered prompt text to the worker settings; ``None`` = nothing.
    worker_prompts: str | None = None

    def supports(self, placement: Placement) -> bool:
        """Whether the backend may run with ``placement``."""
        return placement in self.placements


def load(target: str) -> Any:
    """Import ``"package.module:attribute"``."""
    module, _, attribute = target.partition(":")
    if not module or not attribute:
        raise ValueError(f"expected 'module:attribute', got {target!r}")
    return getattr(importlib.import_module(module), attribute)
