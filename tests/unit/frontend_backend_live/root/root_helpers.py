# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D102, D107
"""Doubles of the composition roots' tests: a frontend adapter class by the class contract, and root state."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any, ClassVar

from examples.frontend_backend_live import factory
from examples.frontend_backend_live.adapter.contract import ServiceStarting
from examples.frontend_backend_live.models.endpoints import ExampleCatalog

FRONT = {"model_id": "org/front", "base_url": "http://front/v1"}
BACK = {"model_id": "org/back", "base_url": "http://back/v1"}
CATALOG = ExampleCatalog(
    {
        "llm": [FRONT],
        "thinker-llm": [BACK, {"model_id": "org/other", "base_url": "http://o/v1"}],
        "asr": [{"server": "asr:50052"}],
        "tts": [{"server": "tts:50051"}],
    }
)


@dataclass(frozen=True)
class StubSettings:
    """A frontend adapter class's settings: the block as the root handed it."""

    block: Mapping[str, Any]
    parsed: ClassVar[list] = []  # every block parsed, in order (reset by each test that reads it)

    @classmethod
    def parse(cls, block: Mapping[str, Any]) -> StubSettings:
        cls.parsed.append(dict(block))
        return cls(dict(block))


class StubReadiness:
    """Whether the stub's services are up: ``down`` names the ones still starting."""

    def __init__(self, down: tuple[str, ...] = ()):
        self.down, self.checks = down, 0

    async def ensure_ready(self) -> None:
        self.checks += 1
        if self.down:
            raise ServiceStarting(self.down[0])


class StubAdapter:
    """A frontend adapter class by the class contract; its instances only record how they were built."""

    Settings: ClassVar[type] = StubSettings
    hosts_rtvi: ClassVar[bool] = True
    built: ClassVar[list] = []  # every from_config call's arguments, in order

    def __init__(self, *args):
        self.args = args

    @classmethod
    def from_config(cls, session, transport, connection, settings, catalog):
        adapter = cls(session, transport, connection, settings, catalog)
        cls.built.append(adapter)
        return adapter

    @classmethod
    def describe(cls, settings, catalog) -> dict:
        endpoint = catalog.model("llm")
        return {
            "frontend": {"provider": "stub-provider", "slot": "llm", "model": endpoint.model, "prompt": "talker"},
            "prompt_version": "v2",
            "instructions": {"frontend": "the talker's instructions"},
        }

    @classmethod
    def readiness(cls, settings, catalog) -> StubReadiness:
        return StubReadiness()


class NoRtviAdapter(StubAdapter):
    """An adapter class the repository UI bot cannot host (``hosts_rtvi`` false), such as a realtime model's."""

    hosts_rtvi: ClassVar[bool] = False


def reset_stubs() -> None:
    """Forget what the stubs recorded."""
    StubSettings.parsed.clear()
    StubAdapter.built.clear()


def clear_root_caches(case: unittest.TestCase) -> None:
    """Make the root read and build again, now and after the test."""
    for cached in (factory.root, factory.delegation_core, factory.responses_client, factory.sample_tools):
        cached.cache_clear()
        case.addCleanup(cached.cache_clear)


def stub_root(**changes) -> factory.Root:
    """A root state with the stub adapter class, the shipped files and :data:`CATALOG`."""
    from delegation_core.config import CoreSettings
    from examples.frontend_backend_live.config_manager.loader import CONFIG_PATH, load_live_config

    core = CoreSettings.load(CONFIG_PATH.with_name("core.yaml"))
    values = {
        "config": load_live_config(CONFIG_PATH),
        "core": core,
        "adapter": StubAdapter,
        "settings": StubSettings({"display_label": core.delegation.display_label}),
        "readiness": StubReadiness(),
        "catalog": CATALOG,
    }
    return factory.Root(**{**values, **changes})


def module(name: str, **attributes) -> dict[str, ModuleType]:
    """A stand-in module for ``patch.dict(sys.modules, ...)``."""
    stand_in = ModuleType(name)
    stand_in.__dict__.update(attributes)
    return {name: stand_in}
