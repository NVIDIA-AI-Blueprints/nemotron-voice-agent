# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The delegation core's builder: one core from its own settings, every implementation chosen by name."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS, AdapterLoader
from examples.frontend_backend_live.delegation_core.backend.control import ControlVocabulary
from examples.frontend_backend_live.delegation_core.backend.port import (
    AdapterSource,
    AppOutputStep,
    AppOutputSteps,
    EndpointLookup,
    OneShotSource,
    ToolSources,
)
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.im.session import IMSession
from examples.frontend_backend_live.delegation_core.tables import pick


def log_trace(kind: str, **fields: Any) -> None:
    """Log one trace of the core as a debug line named ``core.<kind>``."""
    logger.debug(f"core.{kind} {fields}")


class DelegationCore:
    """One delegation core in process: the IM sessions of its conversations, on the backend its settings name."""

    def __init__(
        self,
        settings: CoreSettings,
        source: AdapterSource,
        one_shot_source: OneShotSource,
        *,
        defaults: BackendTask,
        result_format: Any,
        output_steps: AppOutputSteps = lambda observe: (),
        observe: Callable[..., Any] = log_trace,
    ):
        """Hold what each IM session is built from; :func:`build_core` makes these choices."""
        self.settings, self.source, self.one_shot_source = settings, source, one_shot_source
        self.defaults, self.result_format = defaults, result_format
        self.output_steps, self.observe = output_steps, observe

    def session(self, **hooks: Callable[[IMSession], Any]) -> IMSession:
        """Return a new IM session for a new conversation, on a new adapter of the backend source.

        The session takes another adapter of the source for a later call once its adapter closed. ``hooks`` are its
        ``room`` and ``on_closed`` hooks (``IMSession``).
        """
        return IMSession(
            self.source.new(),
            self.defaults,
            self.settings.delegation,
            self.settings.texts,
            result_format=self.result_format,
            output_steps=self.output_steps(self.observe),
            observe=self.observe,
            renew=self.source.new,
            **hooks,
        )

    def one_shot_session(self) -> IMSession:
        """Return a new one-shot IM session for one stateless request, on a new adapter of the one-shot source."""
        from examples.frontend_backend_live.delegation_core.im.oneshot import OneShotSession

        return OneShotSession(
            self.one_shot_source.new_one_shot(),
            self.defaults,
            self.settings.delegation,
            self.settings.texts,
            observe=self.observe,
        )

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` if the backend cannot serve ``model``."""
        await self.source.check_model(model)

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """Return the request settings the backend cannot honor, as ``(param, message)`` pairs."""
        return self.source.unsupported(settings)

    @property
    def close_seconds(self) -> float | None:
        """How long closing one conversation may take: the close grace, else the backstop, and two stop waits.

        A stop wait comes before the grace, for queued work and work that waits for the app, and another after it.
        ``None`` when neither the grace nor the backstop bounds it.
        """
        delegation = self.settings.delegation
        grace = delegation.close_grace_seconds
        running = grace if grace is not None else delegation.work_timeout_seconds
        return None if running is None else running + 2 * delegation.stop_timeout_seconds

    def describe(self) -> dict:
        """Return what the core runs, as plain JSON: its choices, then the backend source's own."""
        backend = self.settings.backend
        return {"adapter": backend.adapter, "one_shot_adapter": backend.one_shot_adapter, **self.source.describe()}

    async def aclose(self) -> None:
        """Release what the backend sources hold, each source once."""
        for source in dict.fromkeys((self.source, self.one_shot_source)):
            await source.aclose()


def tool_sources(blocks: Mapping[str, Mapping[str, Any]]) -> ToolSources:
    """Return the maker of each IM session's server tool sources; an unknown name fails the core's build here."""
    if not blocks:
        return lambda: ()
    # The tool sources load only when the core's settings name one.
    from examples.frontend_backend_live.delegation_core.backend.tools import TOOL_SOURCES

    classes = {name: pick(TOOL_SOURCES, name, "tool source") for name in blocks}
    return lambda: tuple(source.from_settings(blocks[name]) for name, source in classes.items())


def app_output_steps(blocks: Mapping[str, Mapping[str, Any]]) -> AppOutputSteps:
    """Return the maker of each IM session's app output steps; an unknown name or a bad block fails the build here."""
    if not blocks:
        return lambda observe: ()
    # The steps load only when the core's settings name one.
    from examples.frontend_backend_live.delegation_core.backend.output_steps import APP_OUTPUT_STEPS

    classes = {name: pick(APP_OUTPUT_STEPS, name, "app output step") for name in blocks}

    def steps(observe: Callable[..., Any]) -> tuple[AppOutputStep, ...]:
        return tuple(step.from_settings(blocks[name], observe=observe) for name, step in classes.items())

    # Built once here, so a bad block fails the core's build.
    steps(log_trace)
    return steps


def build_core(
    settings: CoreSettings,
    *,
    endpoints: EndpointLookup,
    defaults: BackendTask,
    adapters: Mapping[str, AdapterLoader] = ADAPTERS,
    observe: Callable[..., Any] = log_trace,
) -> DelegationCore:
    """Build one delegation core from its own settings, every choice made once, by name.

    Raises:
        ValueError: A name no table holds, a block for no adapter, a block its adapter or step refuses, or a one-shot
            source that cannot serve one-shot sessions.
    """
    backend = settings.backend
    unknown = sorted(set(backend.blocks) - set(adapters))
    if unknown:
        raise ValueError(
            f"backend has a block for an adapter it does not know: {', '.join(unknown)}; "
            f"known: {', '.join(sorted(adapters))}"
        )
    texts = settings.texts
    vocabulary = ControlVocabulary(
        texts.control_tools, texts.conductor_prompt, texts.refusal("idle"), texts.failed_output
    )
    tools = tool_sources(backend.tools)
    steps = app_output_steps(backend.app_output_steps)
    sources = {
        name: pick(adapters, name, "backend adapter")().from_settings(
            backend.blocks.get(name, {}), endpoints=endpoints, tools=tools, vocabulary=vocabulary
        )
        for name in dict.fromkeys((backend.adapter, backend.one_shot_adapter))
    }
    one_shot_source = sources[backend.one_shot_adapter]
    if not isinstance(one_shot_source, OneShotSource):
        raise ValueError(
            f"backend.one_shot_adapter {backend.one_shot_adapter!r} cannot serve one-shot sessions: name an adapter "
            "whose source offers new_one_shot(), such as llm"
        )
    from examples.frontend_backend_live.delegation_core.im.results import RESULT_FORMATS

    name = settings.delegation.result_format
    result_format = pick(RESULT_FORMATS, name, "result format")(settings.texts.result_formats[name])
    return DelegationCore(
        settings,
        sources[backend.adapter],
        one_shot_source,
        defaults=defaults,
        result_format=result_format,
        output_steps=steps,
        observe=observe,
    )
