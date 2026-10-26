# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the session's instructions into a per-session text-agent ``Config``.

Resolution, first match wins: ``session.update.instructions`` (after
``strip_patterns``), then ``instructions.fallback_file``, then the agent
config's own ``domain.policy`` (``agent.yaml``). Placement for the backend:

* ``policy_slot`` (default): the text becomes ``{domain_policy}``;
* ``replace_prompt``: the text *is* the backend system prompt;
* ``append``: backend prompt, a blank line, then the text.

The frontend never has its contract replaced: when ``frontend`` is in
``apply_to`` the policy is appended to its prompt. Independently, the frontend's
``{capabilities}`` come from the session's tools by default, so a policy-less
frontend still knows what is in scope and delegates instead of refusing
(plan section 11). The service domain (``domain.name``, which selects the
backend's domain notes) is the configured one, else the first
``instructions.domain_from_tools`` rule with one of the session's tools. The
cascade addendum is appended to every prompt that received instructions. Only
frozen dataclasses are replaced; the shared base config is never mutated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from examples.frontend_backend_agent.core.agent.prompts import literal, load_catalog
from examples.frontend_backend_agent.core.config.agent import Config
from examples.frontend_backend_agent.core.config.voice import InstructionsConfig
from examples.frontend_backend_agent.core.tools.realtime_tools import capability_lines
from examples.frontend_backend_agent.core.tools.tools import ToolSpec

POLICY_HEADING = "Policy you must follow:"


@dataclass(frozen=True, slots=True)
class ResolvedInstructions:
    """Which policy text and service domain a session uses, and where they came from."""

    text: str
    source: str  # "session" | "fallback_file" | "config"
    domain_source: str = "none"  # "configured" | "detected" | "none"


def resolve_domain(base: Config, cfg: InstructionsConfig, tools: Sequence[ToolSpec]) -> tuple[str, str]:
    """The session's ``domain.name`` and its source: configured, else the first matching rule, else none."""
    if base.domain.name:
        return base.domain.name, "configured"
    names = {spec.name for spec in tools}
    for rule in cfg.domain_from_tools:
        if names.intersection(rule.tools_any):
            return rule.name, "detected"
    return "", "none"


def resolve_instructions(session_text: str, cfg: InstructionsConfig, base: Config) -> ResolvedInstructions:
    """Pick the policy text for a session."""
    text = session_text or ""
    for pattern in cfg.strip_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE)
    text = text.strip()
    if text:
        return ResolvedInstructions(text, "session")
    if cfg.fallback_text.strip():
        return ResolvedInstructions(cfg.fallback_text.strip(), "fallback_file")
    return ResolvedInstructions(base.domain.policy.strip(), "config")


def session_agent_config(
    base: Config,
    cfg: InstructionsConfig,
    *,
    session_instructions: str,
    tools: Sequence[ToolSpec],
) -> tuple[Config, ResolvedInstructions]:
    """Build the per-session text-agent config for ``session_instructions`` and ``tools``."""
    resolved = resolve_instructions(session_instructions, cfg, base)
    received = resolved.source != "config" and bool(resolved.text)
    catalog = load_catalog(base.prompts_path, base.prompts.inline)
    addendum = ""
    if received and cfg.cascade_addendum_key:
        addendum = "\n\n" + catalog.get(cfg.cascade_addendum_key).strip()

    inline = dict(base.prompts.inline)
    policy = base.domain.policy
    # The client's text is spliced into prompt templates: escape it so that braces in it (some
    # benchmark policies contain Jinja-like "{{ ... }}") render verbatim. {domain_policy} needs no
    # escaping: placeholders are substituted after the template is rendered.
    text = literal(resolved.text)
    if "backend" in cfg.apply_to and received:
        template = catalog.get(base.backend.prompt_key)
        if cfg.placement == "policy_slot":
            policy = resolved.text
            inline[base.backend.prompt_key] = template + addendum
        elif cfg.placement == "replace_prompt":
            inline[base.backend.prompt_key] = text + addendum
        else:
            inline[base.backend.prompt_key] = f"{template.rstrip()}\n\n{text}{addendum}"
    if "frontend" in cfg.apply_to and received and base.frontend_enabled:
        template = catalog.get(base.frontend.prompt_key)
        inline[base.frontend.prompt_key] = f"{template.rstrip()}\n\n{POLICY_HEADING}\n{text}{addendum}"

    if cfg.frontend_capabilities == "from_tools":
        capabilities = capability_lines(tools)
    elif cfg.frontend_capabilities == "static":
        capabilities = base.domain.capabilities
    else:
        capabilities = ()
    domain, domain_source = resolve_domain(base, cfg, tools)

    config = replace(
        base,
        domain=replace(base.domain, name=domain, policy=policy, capabilities=tuple(capabilities)),
        prompts=replace(base.prompts, inline=inline),
    )
    return config, replace(resolved, domain_source=domain_source)
