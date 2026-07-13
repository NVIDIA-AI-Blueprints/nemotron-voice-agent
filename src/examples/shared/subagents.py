# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared subagent registry: capability/routing config declared in YAML.

A subagent's prose (capability, routing rules, label) and config (reasoning
mode, routing token) live in an example's ``subagents.yaml`` — not in Python —
so they read like the rest of the catalog files and can be tuned without code
changes. This module is the single, framework-agnostic source of truth used by:

* the Speaker prompt (delegatable subagents' capability + routing block),
* the ``/api/subagents`` UI endpoint (all subagents, for status), and
* per-worker config such as the reasoning mode.

Examples without a ``subagents.yaml`` simply yield an empty registry, so nothing
example-specific leaks into shared code.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import yaml

#: Reasoning modes a subagent's underlying model can run in.
REASONING_MODES = ("on", "off", "on_demand")
_DEFAULT_REASONING = "off"

#: Leading text of the Speaker's pinned "subagents available" note. Used as a stable
#: marker so the note can be located and replaced in the Speaker context when subagents
#: are toggled at runtime. The note itself is rendered by ``SubagentStateBoard``.
SPEAKER_CAPABILITIES_PREFIX = "Subagents available to you this session"


@dataclass(frozen=True)
class SubagentSpec:
    """Self-contained description of one subagent (loaded from YAML).

    ``source_token`` is the ``selected_input_source`` value the Speaker emits to
    delegate to this subagent; ambient subagents (e.g. webcam eye-vision) that
    are not Speaker-routed leave it ``None``. ``reasoning`` selects the model's
    thinking mode: ``on``, ``off``, or ``on_demand``.
    """

    key: str
    label: str
    capability: str
    routing_rules: str = ""
    source_token: str | None = None
    reasoning: str = _DEFAULT_REASONING
    findings_label: str = ""

    @property
    def delegatable(self) -> bool:
        """Whether the Speaker routes to this subagent via ``selected_input_source``."""
        return bool(self.source_token)


class SubagentRegistry:
    """Ordered set of subagents wired into one example/session."""

    def __init__(self, specs: Iterable[SubagentSpec]) -> None:
        """Index specs by key, preserving declaration order."""
        self._specs: dict[str, SubagentSpec] = {spec.key: spec for spec in specs}

    def __bool__(self) -> bool:
        """True when at least one subagent is registered."""
        return bool(self._specs)

    def specs(self) -> list[SubagentSpec]:
        """All specs, in declaration order."""
        return list(self._specs.values())

    def get(self, key: str) -> SubagentSpec | None:
        """Look up a spec by key."""
        return self._specs.get(key)

    def keys(self) -> list[str]:
        """All registered keys."""
        return list(self._specs)

    def to_payload(self) -> list[dict]:
        """Serialize all subagents for the UI."""
        return [
            {
                "key": spec.key,
                "label": spec.label,
                "capability": spec.capability,
                "delegatable": spec.delegatable,
                "reasoning": spec.reasoning,
            }
            for spec in self._specs.values()
        ]


def normalize_reasoning(value: object) -> str:
    """Coerce a YAML reasoning value to one of :data:`REASONING_MODES`.

    PyYAML (YAML 1.1) parses bare ``on``/``off`` as booleans, so map those back
    to their reasoning-mode strings — this lets ``subagents.yaml`` read naturally
    (``reasoning: on``) without quoting.
    """
    if isinstance(value, bool):
        return "on" if value else "off"
    mode = str(value or _DEFAULT_REASONING).strip().lower()
    return mode if mode in REASONING_MODES else _DEFAULT_REASONING


def load_subagent_registry(yaml_path: str | Path) -> SubagentRegistry:
    """Build a registry from an example's ``subagents.yaml`` (empty if absent/blank).

    ``subagents`` is a mapping keyed by stable subagent id (matching the worker's
    ``AGENT_NAME``), mirroring the project's other catalogs (``prompts.yaml``,
    ``services.*.yaml``). Declaration order is preserved.
    """
    path = Path(yaml_path)
    if not path.is_file():
        return SubagentRegistry([])
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("subagents") if isinstance(data, dict) else None
    if not isinstance(items, dict):
        return SubagentRegistry([])
    specs: list[SubagentSpec] = []
    for raw_key, item in items.items():
        key = str(raw_key or "").strip()
        if not key or not isinstance(item, dict):
            continue
        token = item.get("source_token")
        specs.append(
            SubagentSpec(
                key=key,
                label=str(item.get("label") or key).strip(),
                capability=str(item.get("capability") or "").strip(),
                routing_rules=str(item.get("routing_rules") or "").strip(),
                source_token=(str(token).strip() or None) if token else None,
                reasoning=normalize_reasoning(item.get("reasoning")),
                findings_label=str(item.get("findings_label") or "").strip(),
            )
        )
    return SubagentRegistry(specs)
