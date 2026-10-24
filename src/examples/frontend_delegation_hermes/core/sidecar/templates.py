# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jinja2 rendering of the backend prompt catalog (``config/prompts/<profile>/backend/<kind>.yaml``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jinja2
import yaml

from examples.frontend_delegation_hermes.core.prompt_features import BACKEND_FEATURES, features, sha256_text

#: Keys every backend's prompt file has; a backend adds its own (``BackendSpec.required_prompt_keys``).
REQUIRED_KEYS = (
    "backend_soul",
    "backend_system",
    "context_user",
    "context_frontend",
    "context_status",
    "context_controller",
    "context_delivery_note",
    "context_requeued_request",
    "context_side_effect",
    "context_block",
    "backend_delegation_message",
    "backend_steer",
    "backend_error_spoken",
    "backend_unavailable",
)

#: The one mapping-valued key: domain name -> note template (tau3-identity-fixes-plan.md section 3.2).
DOMAIN_NOTES_KEY = "backend_domain_notes"


class TemplateError(ValueError):
    """A missing or broken template."""


class BackendTemplates:
    """``render(key, **vars)`` over a YAML catalog of Jinja templates (undefined variables fail)."""

    def __init__(
        self,
        path: str | Path,
        *,
        prompt_features: Mapping[str, object] | None = None,
        required_keys: Sequence[str] = (),
    ) -> None:
        """Load and compile every template; missing required keys fail now.

        ``prompt_features`` switches the catalog's ``{% if features.<name> %}`` variants (all off by default).
        ``required_keys`` are the backend's own keys, required besides :data:`REQUIRED_KEYS`.
        """
        self.path = Path(path)
        raw = self.path.read_text(encoding="utf-8")
        try:
            self.features = features(prompt_features, BACKEND_FEATURES)
        except ValueError as exc:
            raise TemplateError(str(exc)) from exc
        #: Fingerprint of the catalog file (tau3-failure-fixes-plan.md section 1, rule 5).
        self.catalog_sha256 = sha256_text(raw)
        data = yaml.safe_load(raw) or {}
        if not isinstance(data, dict):
            raise TemplateError(f"{self.path}: expected a mapping of templates")
        missing = [key for key in (*REQUIRED_KEYS, *required_keys) if key not in data]
        if missing:
            raise TemplateError(f"{self.path}: missing template(s) {missing}")
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False, keep_trailing_newline=False)  # noqa: S701 - plain text prompts
        env.globals["features"] = self.features
        notes = data.pop(DOMAIN_NOTES_KEY, None) or {}
        self._sources = {key: str(value) for key, value in data.items()}
        if not isinstance(notes, dict):
            raise TemplateError(f"{self.path}: {DOMAIN_NOTES_KEY} must map domain names to templates")
        try:
            self._templates = {key: env.from_string(str(value)) for key, value in data.items()}
            self._domain_notes = {str(domain): env.from_string(str(value)) for domain, value in notes.items()}
        except jinja2.TemplateError as exc:
            raise TemplateError(f"{self.path}: {exc}") from exc

    @property
    def domains_with_notes(self) -> tuple[str, ...]:
        """Domains that have a note template (rendered or not, depending on its feature)."""
        return tuple(self._domain_notes)

    def domain_note(self, domain: str) -> str:
        """The rendered note of ``domain``; empty for no domain, no note, or a note whose feature is off."""
        template = self._domain_notes.get(domain) if domain else None
        if template is None:
            return ""
        try:
            return template.render().strip()
        except jinja2.TemplateError as exc:
            raise TemplateError(f"{DOMAIN_NOTES_KEY}.{domain}: {exc}") from exc

    def source(self, key: str) -> str:
        """The raw template text of ``key`` (for a backend that renders it itself, per call)."""
        if key not in self._sources:
            raise TemplateError(f"unknown template {key!r}")
        return self._sources[key]

    def render(self, key: str, **variables: Any) -> str:
        """Render ``key``; raises :class:`TemplateError` for unknown keys or undefined variables."""
        template = self._templates.get(key)
        if template is None:
            raise TemplateError(f"unknown template {key!r}")
        try:
            return template.render(**variables).strip()
        except jinja2.TemplateError as exc:
            raise TemplateError(f"{key}: {exc}") from exc
