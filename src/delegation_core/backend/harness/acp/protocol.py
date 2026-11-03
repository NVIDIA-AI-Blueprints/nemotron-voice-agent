# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ACP for one core (``backend.harness.protocol: acp``): a transport and an agent class, one client per process."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from delegation_core.tables import pick

from ..protocol import PermissionAnswer, RuntimeLaunch
from .agent import AcpAgent
from .agents import AGENTS
from .client import AcpClient
from .transports import TRANSPORTS, Transport


class AcpProtocol:
    """ACP over one transport, to the agent of one agent class."""

    def __init__(self, transport: type[Transport], agent: AcpAgent):
        """Reach each runtime process by ``transport`` and talk to it as ``agent`` says."""
        self.transport, self.agent = transport, agent

    @classmethod
    def from_settings(cls, block: Mapping[str, Any]) -> AcpProtocol:
        """The transport and the agent class the harness block names, each checked.

        Raises:
            ValueError: A transport or an agent no table has, a transport missing its ``command`` or ``url``, an
                ``agents`` block for an agent no table has, or an agent block its class refuses.
        """
        transport = pick(TRANSPORTS, block["transport"], "harness transport")
        transport.check(tuple(block["command"] or ()), block["url"])
        blocks = block["agents"]
        if not isinstance(blocks, Mapping):
            raise ValueError("backend.harness.agents must map agent names to their blocks")
        unknown = set(blocks) - set(AGENTS)
        if unknown:
            raise ValueError(f"backend.harness.agents names no known agent: {', '.join(sorted(map(str, unknown)))}")
        name = block["agent"]
        agent = pick(AGENTS, name, "harness agent")().from_settings(blocks.get(name) or {})
        return cls(transport, agent)

    def client(
        self,
        launch: RuntimeLaunch,
        *,
        permission: PermissionAnswer,
        start_timeout_seconds: float,
        stop_grace_seconds: float,
    ) -> AcpClient:
        """A client for one runtime process; it connects nothing until ``open()``."""
        transport = self.transport.for_launch(launch, stop_grace_seconds=stop_grace_seconds)
        return AcpClient(transport, self.agent, permission=permission, start_timeout_seconds=start_timeout_seconds)
