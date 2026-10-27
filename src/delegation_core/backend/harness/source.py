# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's source (``backend.adapter: harness``): built once per core, it makes one adapter per IM session.

It is a ``port.AdapterSource`` but not a ``port.OneShotSource``: the source ``backend.one_shot_adapter`` names serves
stateless requests. Building it checks the block, loads the classes it names and sweeps stale work folders; it starts no
runtime process and binds no port, and the bridge starts with the first runtime process. The harness serves one model
name, the default endpoint's, and reports the request settings a runtime that owns its model calls cannot honor;
instructions and functions are fixed when a worker starts, so a later change reaches only new workers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

from delegation_core.backend.port import ControlTools, EndpointLookup, ToolSources
from delegation_core.config import ModelEndpoint

from .adapter import HarnessAdapter
from .bridge import McpBridge
from .settings import HarnessSettings
from .workdir import WorkFolder

# The tool choice a runtime that picks its own tools honors.
AUTO_CHOICE = (None, "auto")
# The service tiers it honors: none chosen.
PLAIN_TIERS = (None, "auto", "default")


class HarnessSource:
    """An external agent over its protocol: its settings, the model name it serves, and the bridge."""

    name: ClassVar[str] = "harness"

    def __init__(
        self,
        settings: HarnessSettings,
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ):
        """Hold what each IM session's adapter is made of; nothing starts."""
        self.settings, self.endpoints, self.tools, self.vocabulary = settings, endpoints, tools, vocabulary
        self.bridge = McpBridge(settings.bridge, start_timeout_seconds=settings.start_timeout_seconds)

    @classmethod
    def from_settings(
        cls,
        block: Mapping[str, Any],
        *,
        endpoints: EndpointLookup,
        tools: ToolSources,
        vocabulary: ControlTools,
    ) -> HarnessSource:
        """Validate ``backend.harness`` (``block``), load the classes it names, sweep stale work folders; start nothing.

        Raises:
            ValueError: The block has an unknown key, a value of the wrong kind, a name no table has, or a setting the
                runtime cannot run with (see :meth:`HarnessSettings.parse`).
        """
        settings = HarnessSettings.parse(block)
        WorkFolder.sweep(settings.work_root)
        return cls(settings, endpoints=endpoints, tools=tools, vocabulary=vocabulary)

    def endpoint(self) -> ModelEndpoint:
        """The configured default model endpoint, whose model name the harness serves."""
        return self.endpoints(None)

    def describe(self) -> dict:
        """What it serves, as plain JSON for ``GET /v1/live/info``: no credentials, no environment values."""
        return {
            "adapter": self.name,
            **self.settings.describe(),
            "model": self.endpoint().model,
            "bridge": self.settings.bridge.advertise,
            "tools": [tool for source in self.tools() for tool in source.tools()],
        }

    async def check_model(self, model: str) -> None:
        """Raise ``ValueError`` unless ``model`` is the one model name the harness serves."""
        served = self.endpoint().model
        if model != served:
            raise ValueError(f"The harness runs {served!r}; it cannot serve the model {model!r}")

    def unsupported(self, settings: Mapping[str, Any]) -> list[tuple[str, str]]:
        """The request settings a runtime that owns its model calls cannot honor, as ``(param, message)`` pairs."""
        problems = []
        if settings.get("max_output_tokens") is not None:
            problems.append(("max_output_tokens", "The harness's runtime sets its own output limit."))
        text = settings.get("text") or {}
        if (text.get("format") or {}).get("type", "text") != "text" or text.get("verbosity") is not None:
            problems.append(("text", "The harness's runtime answers in plain text."))
        if settings.get("tool_choice") not in AUTO_CHOICE:
            problems.append(("tool_choice", "The harness's runtime chooses its own tools (tool_choice auto)."))
        hosted = [tool.get("type") for tool in settings.get("tools") or [] if tool.get("type") != "function"]
        problems += [
            ("tools", f"The {kind} tool is not available on the harness: it bridges function tools.")
            for kind in dict.fromkeys(hosted)
        ]
        if (settings.get("reasoning") or {}).get("effort") is not None:
            problems.append(("reasoning", "The harness's runtime sets its own reasoning effort."))
        if settings.get("service_tier") not in PLAIN_TIERS:
            problems.append(("service_tier", f"service_tier {settings['service_tier']!r} is not available here."))
        return problems

    def new(self) -> HarnessAdapter:
        """The backend of a new IM session, with a fresh set of the session's tool sources."""
        return HarnessAdapter(self, tools=self.tools())

    async def aclose(self) -> None:
        """Stop the bridge (each adapter closes its own runtime processes)."""
        await self.bridge.aclose()
