# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend prompt catalog: a YAML mapping of Jinja templates, checked at load time."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, StrictUndefined, Template

from examples.frontend_delegation_hermes.core.prompt_features import FRONTEND_FEATURES, PromptFeatures, features

#: Keys every catalog must define, with sample variables used to check each template at load time.
REQUIRED: dict[str, dict[str, Any]] = {
    "frontend_system": {"capabilities": ["get_order: Look up an order."], "normalization_note": "note"},
    "frontend_backend_state": {
        "state": "WORKING",
        "current_request": "check order 1234",
        "elapsed_s": 3,
        "recent_activity": ["get_order(order_id=1234) -> done"],
        "backend_asked_question": False,
        "last_answer_unheard": False,
    },
    "delegate_tool_description": {},
    "delegate_param_delegate": {},
    "delegate_param_filler_text": {},
    "delegate_param_request": {},
    "contract_repair": {},
    "status_verbalize": {"summary_text": "working"},
    "status_template": {"current_tool": "get_order", "current_tool_phrase": "looking that up", "done_count": 1},
    "normalization_note": {},
}

#: Keys a catalog may define (one spoken line per non-empty line); a missing one means "no lines".
OPTIONAL: dict[str, dict[str, Any]] = {
    "status_proactive_lines": {},  # M1: the proactive status while the backend works
    "filler_alternatives": {},  # G1: replacements for a repeated filler
}


class PromptError(ValueError):
    """A missing or broken template."""


class PromptCatalog:
    """Compiled templates by key."""

    def __init__(
        self,
        templates: Mapping[str, str],
        *,
        source: str = "<inline>",
        prompt_features: Mapping[str, object] | None = None,
    ) -> None:
        """Compile and check every required (and present optional) template.

        ``prompt_features`` switches the ``{% if features.<name> %}`` variants (all off by default).
        """
        env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)  # noqa: S701 - plain text prompts
        try:
            self.features: PromptFeatures = features(prompt_features, FRONTEND_FEATURES)
        except ValueError as exc:
            raise PromptError(f"{source}: {exc}") from exc
        env.globals["features"] = self.features
        self.source = source
        self._compiled: dict[str, Template] = {}
        for key, sample in {**REQUIRED, **OPTIONAL}.items():
            if key not in templates:
                if key in OPTIONAL:
                    continue
                raise PromptError(f"{source}: prompt key {key!r} is missing")
            try:
                compiled = env.from_string(str(templates[key]))
                compiled.render(**sample)
            except Exception as exc:
                raise PromptError(f"{source}: template {key!r} does not render: {exc}") from exc
            self._compiled[key] = compiled

    def render(self, key: str, **values: Any) -> str:
        """Render ``key``; raises for unknown keys or undefined variables."""
        try:
            template = self._compiled[key]
        except KeyError as exc:
            raise PromptError(f"unknown prompt key {key!r}") from exc
        return template.render(**values).strip()

    def lines(self, key: str) -> tuple[str, ...]:
        """The non-empty lines of an optional list key (``()`` when the catalog does not define it)."""
        if key not in self._compiled:
            return ()
        return tuple(line.strip() for line in self.render(key).splitlines() if line.strip())


def load_prompts(path: str | Path, *, prompt_features: Mapping[str, object] | None = None) -> PromptCatalog:
    """Load a catalog file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PromptError(f"{path}: must be a mapping of key -> template")
    return PromptCatalog({str(k): str(v) for k, v in raw.items()}, source=str(path), prompt_features=prompt_features)
