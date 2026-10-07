# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompt catalog loading and rendering.

A prompt is rendered in two passes. First it is a Jinja 2 template: ``{% if %}``
blocks and ``{{ variables }}`` from a caller's context, and ``{% include "<key>" %}``
of another catalog entry. A prompt without Jinja syntax renders unchanged. Then
the domain placeholders (``{persona}`` ...) are substituted by literal
replacement rather than ``str.format``: the prompts contain JSON examples full
of single braces, and ``format`` would choke on them.

Text that is not a template (a client's policy spliced into a prompt) must go
through :func:`literal` first, so that braces in it render verbatim.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jinja2
import yaml

from examples.frontend_backend_verdict.text.config import Config
from examples.frontend_backend_verdict.text.errors import ConfigError

PERSONA = "{persona}"
DOMAIN_POLICY = "{domain_policy}"
CAPABILITIES = "{capabilities}"
UNSUPPORTED_REPLY = "{unsupported_reply}"
AGENT_NAME = "{agent_name}"


@dataclass(frozen=True, slots=True)
class PromptCatalog:
    """Prompt texts keyed by catalog key."""

    prompts: dict[str, str]

    def get(self, key: str) -> str:
        """Return the prompt for ``key``."""
        if key not in self.prompts:
            raise ConfigError(f"prompt key not found in catalog: {key!r}")
        return self.prompts[key]


def load_catalog(path: str | Path, inline: dict[str, str] | None = None) -> PromptCatalog:
    """Load a ``prompts.yaml`` catalog, applying inline overrides."""
    prompts: dict[str, str] = {}
    catalog_path = Path(path)
    if catalog_path.is_file():
        data = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"prompt catalog must be a mapping: {catalog_path}")
        for key, item in data.items():
            if isinstance(item, dict) and "content" in item:
                prompts[str(key)] = str(item["content"])
            elif isinstance(item, str):
                prompts[str(key)] = item
    elif not inline:
        raise ConfigError(f"prompt catalog not found: {catalog_path}")
    prompts.update({str(key): str(value) for key, value in (inline or {}).items()})
    if not prompts:
        raise ConfigError("prompt catalog is empty")
    return PromptCatalog(prompts=prompts)


_JINJA_START = re.compile(r"\{[{%#]")


def literal(text: str) -> str:
    """``text`` escaped for splicing into a template: it renders back to exactly ``text``."""
    return _JINJA_START.sub(lambda match: "{{ '" + match.group(0) + "' }}", text)


def environment(catalog: PromptCatalog | None = None) -> jinja2.Environment:
    """The Jinja environment for prompts; ``{% include %}`` resolves keys of ``catalog``."""
    return jinja2.Environment(
        loader=jinja2.DictLoader(dict(catalog.prompts)) if catalog is not None else None,
        undefined=jinja2.StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def compile_template(template: str, catalog: PromptCatalog | None = None, *, key: str = "") -> jinja2.Template:
    """Compile a prompt template (raises ``ConfigError`` naming ``key`` on a syntax error)."""
    try:
        return environment(catalog).from_string(template)
    except jinja2.TemplateError as exc:
        raise ConfigError(f"prompt {key or '<inline>'!r} is not a valid template: {exc}") from None


def render_template(
    template: jinja2.Template, config: Config, context: Mapping[str, Any] | None = None, *, key: str = ""
) -> str:
    """Render a compiled prompt template with ``context``, then substitute the domain placeholders."""
    try:
        rendered = template.render(**dict(context or {}))
    except jinja2.TemplateError as exc:
        raise ConfigError(f"prompt {key or '<inline>'!r} failed to render: {exc}") from None
    return substitute(rendered, config)


def render(
    template: str,
    config: Config,
    context: Mapping[str, Any] | None = None,
    *,
    catalog: PromptCatalog | None = None,
    key: str = "",
) -> str:
    """Render ``template`` (Jinja with ``context``, then the domain placeholders)."""
    return render_template(compile_template(template, catalog, key=key), config, context, key=key)


def substitute(template: str, config: Config) -> str:
    """Substitute the domain placeholders in ``template``."""
    capabilities = "\n".join(f"- {item}" for item in config.domain.capabilities)
    replacements = {
        PERSONA: config.persona,
        DOMAIN_POLICY: config.domain.policy.strip(),
        CAPABILITIES: capabilities or "- (no capability list configured)",
        UNSUPPORTED_REPLY: config.domain.unsupported_reply,
        AGENT_NAME: config.name,
    }
    rendered = template
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered.strip()
