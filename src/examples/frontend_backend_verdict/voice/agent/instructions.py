# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render the session's instructions into a per-session text-agent ``Config``.

Resolution, first match wins: ``session.update.instructions`` (after
``strip_patterns``), then ``instructions.fallback_file``, then the text agent
config's own ``domain.policy``. Placement for the backend:

* ``policy_slot`` (default): the text becomes ``{domain_policy}``;
* ``replace_prompt``: the text *is* the backend system prompt;
* ``append``: backend prompt, a blank line, then the text.

The frontend never has its contract replaced: when ``frontend`` is in
``apply_to`` the policy is appended to its prompt. Independently, the frontend's
``{capabilities}`` come from the session's tools by default, so a policy-less
frontend still knows what is in scope and delegates instead of refusing
(plan section 11). The cascade addendum is appended to every prompt that
received instructions. Only frozen dataclasses are replaced; the shared base
config is never mutated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace

from examples.frontend_backend_verdict.text.config import Config
from examples.frontend_backend_verdict.text.prompts import literal, load_catalog
from examples.frontend_backend_verdict.text.tools import ToolSpec
from examples.frontend_backend_verdict.voice.agent.tools import capability_lines
from examples.frontend_backend_verdict.voice.config import InstructionsConfig

POLICY_HEADING = "Policy you must follow:"


@dataclass(frozen=True, slots=True)
class ResolvedInstructions:
    """Which policy text a session uses, and where it came from."""

    text: str
    source: str  # "session" | "fallback_file" | "config"


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
    # tau2 policies contain Jinja-like "{{ ... }}") render verbatim. {domain_policy} needs no
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

    config = replace(
        base,
        domain=replace(base.domain, policy=policy, capabilities=tuple(capabilities)),
        prompts=replace(base.prompts, inline=inline),
    )
    return config, resolved
