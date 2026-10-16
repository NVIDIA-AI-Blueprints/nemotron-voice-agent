# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM clients and the per-session agent factory, shared by every adapter.

``ServerOptions``, ``build_clients`` and ``_agent_factory``. The factory reads only ``config``, ``options``,
``clients`` and ``routing_sink`` from its state, so its annotation is the type-only
``AgentFactoryState`` protocol and ``core/`` never imports an adapter's state class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from examples.frontend_backend_verdict.core.agent.llm import OpenAIChatClient
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig, prompt_context
from examples.frontend_backend_verdict.core.engine.event_log import SessionRoutingSink
from examples.frontend_backend_verdict.core.engine.port import AgentPort
from examples.frontend_backend_verdict.core.engine.runner import (
    AgentClients,
    TextAgentRunner,
    frontend_verdict_settings,
)
from examples.frontend_backend_verdict.core.engine.scripted import ScriptedAgentPort


@dataclass(slots=True)
class ServerOptions:
    """Runtime switches that are not part of the YAML (test and gate stand-ins)."""

    stub_speech: bool = False
    stub_agent: str = ""
    stub_agent_script: str = ""
    tls: bool = False


class AgentFactoryState(Protocol):
    """What :func:`_agent_factory` reads from an adapter's process state (type-only)."""

    config: VoiceConfig
    options: ServerOptions
    clients: AgentClients | None
    routing_sink: SessionRoutingSink


def build_clients(config: VoiceConfig) -> AgentClients:
    """One OpenAI-compatible client per LLM role, shared by all sessions."""
    agent = config.agent
    frontend = OpenAIChatClient(agent.frontend.llm, accounting=agent.accounting) if agent.frontend_enabled else None
    return AgentClients(backend=OpenAIChatClient(agent.backend.llm, accounting=agent.accounting), frontend=frontend)


def _agent_factory(state: AgentFactoryState) -> Any:
    config, options = state.config, state.options
    barge_in = frontend_verdict_settings(config)

    def factory(session_id: str) -> AgentPort:
        if options.stub_agent == "scripted":
            agent: AgentPort = (
                ScriptedAgentPort.from_file(options.stub_agent_script)
                if options.stub_agent_script
                else ScriptedAgentPort(backend_only=not config.agent.frontend_enabled)
            )
            return agent
        assert state.clients is not None  # noqa: S101 - built in lifespan
        return TextAgentRunner(
            base_config=config.agent,
            tools_config=config.tools,
            instructions_config=config.instructions,
            clients=state.clients,
            sink=state.routing_sink,
            session_id=session_id,
            seed_greeting=config.protocol.seed_history_with_client_greeting,
            normalization=config.normalization,
            barge_in=barge_in,
            prompt_context=prompt_context(config),
            write_gate=config.write_gate,
        )

    return factory
