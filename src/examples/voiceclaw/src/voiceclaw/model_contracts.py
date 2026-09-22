# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Strict, immutable model-facing contract catalog for VoiceClaw."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

MODEL_CONTRACT_SCHEMA = "voiceclaw.model-contracts.v1"
PACKAGED_MODEL_CONTRACTS = "model_contracts.v1.yaml"
_MAX_CATALOG_BYTES = 512 * 1024
_MAX_RENDERED_CONTRACT_CHARACTERS = 512 * 1024
_MAX_FAILURE_TITLE_CHARACTERS = 160
_MAX_FAILURE_TEXT_CHARACTERS = 2_000
_MAX_FAILURE_CODE_OVERRIDES = 256
_TOKEN = re.compile(r"\$\{([a-z][a-z0-9_]*)\}")
_PLACEHOLDER = re.compile(r"\$\{([^}]*)\}")
_FAILURE_CODE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_INSTRUCTION_TEMPLATE_TOKENS = {
    "server_policy": frozenset({"content"}),
    "untrusted_session": frozenset({"content"}),
    "untrusted_response": frozenset({"content"}),
    "response_context": frozenset({"context_json"}),
    "application_response_turn": frozenset(),
    "dynamic_projection": frozenset({"projection"}),
    "bootstrap_projection": frozenset({"projection"}),
}
_RESULT_TEMPLATE_TOKENS = frozenset({"user_goal_json", "result_schema", "speech_budget_bytes", "display_budget_bytes"})


class ModelContractError(ValueError):
    """The selected model-contract catalog is malformed or incompatible."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML safe loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise ModelContractError("model-contract mapping keys must be scalar values") from error
        if duplicate:
            raise ModelContractError(f"duplicate model-contract key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class FailureCopy:
    """One resolved, operator-owned failure presentation."""

    title: str
    display: str
    speech: str


@dataclass(frozen=True, slots=True)
class ModelContractCatalog:
    """Validated frontend and response-only result policy."""

    schema_version: str
    profile: str
    digest: str
    static_instructions: str
    instruction_templates: Mapping[str, str]
    result_envelope_template: str
    failure_copy_title: str
    default_failure_copy: FailureCopy
    failure_copy_overrides: Mapping[str, FailureCopy]

    def failure_copy(self, code: str) -> FailureCopy:
        """Resolve a stable machine failure code to bounded display and speech copy."""
        if not isinstance(code, str) or _FAILURE_CODE.fullmatch(code) is None:
            raise ModelContractError("failure code must match [a-z][a-z0-9_]{0,127}")
        return self.failure_copy_overrides.get(code, self.default_failure_copy)

    def render_instruction(self, name: str, **values: object) -> str:
        """Render one trusted instruction-boundary template."""
        try:
            template = self.instruction_templates[name]
            expected = _INSTRUCTION_TEMPLATE_TOKENS[name]
        except KeyError as error:
            raise ModelContractError(f"unknown instruction template: {name}") from error
        return _render(template, expected=expected, values=values)

    def render_result_envelope(
        self,
        *,
        user_goal_json: str,
        result_schema: str,
        speech_budget_bytes: int,
        display_budget_bytes: int,
    ) -> str:
        """Render the backend result-channel contract with validated scalar values."""
        return _render(
            self.result_envelope_template,
            expected=_RESULT_TEMPLATE_TOKENS,
            values={
                "user_goal_json": user_goal_json,
                "result_schema": result_schema,
                "speech_budget_bytes": speech_budget_bytes,
                "display_budget_bytes": display_budget_bytes,
            },
        )


def load_model_contract_catalog(
    path: str | Path | None = None,
    *,
    profile: str = "default",
) -> ModelContractCatalog:
    """Load a packaged catalog or one explicit operator-owned override file."""
    if not isinstance(profile, str) or not profile.strip() or "\x00" in profile:
        raise ModelContractError("model-contract profile must be a non-empty string")
    selected_profile = profile.strip()
    if path is None:
        resource = files("voiceclaw.resources").joinpath(PACKAGED_MODEL_CONTRACTS)
        try:
            text = resource.read_text(encoding="utf-8")
        except OSError as error:
            raise ModelContractError("cannot read packaged model-contract catalog") from error
        source = f"packaged:{PACKAGED_MODEL_CONTRACTS}"
    else:
        source_path = Path(path)
        try:
            if source_path.stat().st_size > _MAX_CATALOG_BYTES:
                raise ModelContractError("model-contract catalog exceeds the size limit")
            text = source_path.read_text(encoding="utf-8")
        except OSError as error:
            raise ModelContractError(f"cannot read model-contract catalog: {source_path}") from error
        source = str(source_path)
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ModelContractError(f"model-contract catalog is not valid UTF-8: {source}") from error
    if not encoded or len(encoded) > _MAX_CATALOG_BYTES or b"\x00" in encoded:
        raise ModelContractError("model-contract catalog is empty, oversized, or contains NUL bytes")
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except ModelContractError:
        raise
    except yaml.YAMLError as error:
        raise ModelContractError(f"invalid YAML in model-contract catalog: {source}") from error
    return _parse_catalog(_mapping(raw, "catalog"), profile=selected_profile)


def _parse_catalog(raw: Mapping[str, Any], *, profile: str) -> ModelContractCatalog:
    _exact_keys(raw, {"schema_version", "profiles"}, "catalog")
    schema_version = _text(raw.get("schema_version"), "schema_version")
    if schema_version != MODEL_CONTRACT_SCHEMA:
        raise ModelContractError(f"schema_version must be {MODEL_CONTRACT_SCHEMA}")
    profiles = _mapping(raw.get("profiles"), "profiles")
    try:
        profile_raw = _mapping(profiles[profile], f"profiles.{profile}")
    except KeyError as error:
        raise ModelContractError(f"unknown model-contract profile: {profile}") from error
    _exact_keys(profile_raw, {"frontend", "backend"}, f"profiles.{profile}")

    frontend = _mapping(profile_raw.get("frontend"), f"profiles.{profile}.frontend")
    _exact_keys(
        frontend,
        {"static_instructions", "instruction_templates", "failure_copy"},
        f"profiles.{profile}.frontend",
    )
    static_instructions = _text(frontend.get("static_instructions"), "frontend.static_instructions")
    _validate_tokens(static_instructions, expected=frozenset(), path="frontend.static_instructions")
    instruction_templates = _template_mapping(
        frontend.get("instruction_templates"),
        expected=_INSTRUCTION_TEMPLATE_TOKENS,
        path="frontend.instruction_templates",
    )
    failure_copy_title, default_failure_copy, failure_copy_overrides = _failure_copy_catalog(
        frontend.get("failure_copy"),
        path="frontend.failure_copy",
    )

    backend = _mapping(profile_raw.get("backend"), f"profiles.{profile}.backend")
    _exact_keys(backend, {"result_envelope"}, f"profiles.{profile}.backend")
    result_envelope = _text(backend.get("result_envelope"), "backend.result_envelope")
    _validate_tokens(result_envelope, expected=_RESULT_TEMPLATE_TOKENS, path="backend.result_envelope")

    canonical = json.dumps(
        {"schema_version": schema_version, "profile": profile, "contracts": profile_raw},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return ModelContractCatalog(
        schema_version=schema_version,
        profile=profile,
        digest=digest,
        static_instructions=static_instructions,
        instruction_templates=MappingProxyType(dict(instruction_templates)),
        result_envelope_template=result_envelope,
        failure_copy_title=failure_copy_title,
        default_failure_copy=default_failure_copy,
        failure_copy_overrides=MappingProxyType(dict(failure_copy_overrides)),
    )


def _failure_copy_catalog(
    value: object,
    *,
    path: str,
) -> tuple[str, FailureCopy, dict[str, FailureCopy]]:
    raw = _mapping(value, path)
    _exact_keys(raw, {"title", "default", "codes"}, path)
    title = _failure_text(
        raw.get("title"),
        f"{path}.title",
        maximum=_MAX_FAILURE_TITLE_CHARACTERS,
    )
    default = _failure_copy_entry(raw.get("default"), title=title, path=f"{path}.default")
    codes = _mapping(raw.get("codes"), f"{path}.codes")
    if len(codes) > _MAX_FAILURE_CODE_OVERRIDES:
        raise ModelContractError(f"{path}.codes exceeds the entry limit")
    overrides: dict[str, FailureCopy] = {}
    for code, entry in codes.items():
        if _FAILURE_CODE.fullmatch(code) is None:
            raise ModelContractError(f"{path}.codes key {code!r} must match [a-z][a-z0-9_]{{0,127}}")
        overrides[code] = _failure_copy_entry(entry, title=title, path=f"{path}.codes.{code}")
    return title, default, overrides


def _failure_copy_entry(value: object, *, title: str, path: str) -> FailureCopy:
    raw = _mapping(value, path)
    _exact_keys(raw, {"display", "speech"}, path)
    return FailureCopy(
        title=title,
        display=_failure_text(raw.get("display"), f"{path}.display", maximum=_MAX_FAILURE_TEXT_CHARACTERS),
        speech=_failure_text(raw.get("speech"), f"{path}.speech", maximum=_MAX_FAILURE_TEXT_CHARACTERS),
    )


def _failure_text(value: object, path: str, *, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelContractError(f"{path} must be non-empty text")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ModelContractError(f"{path} exceeds the size limit")
    if not normalized.isprintable():
        raise ModelContractError(f"{path} must not contain control characters")
    _validate_tokens(normalized, expected=frozenset(), path=path)
    return normalized


def _template_mapping(
    value: object,
    *,
    expected: Mapping[str, frozenset[str]],
    path: str,
) -> dict[str, str]:
    raw = _mapping(value, path)
    _exact_keys(raw, set(expected), path)
    result: dict[str, str] = {}
    for name, expected_tokens in expected.items():
        template = _text(raw[name], f"{path}.{name}")
        _validate_tokens(template, expected=expected_tokens, path=f"{path}.{name}")
        result[name] = template
    return result


def _render(template: str, *, expected: frozenset[str], values: Mapping[str, object]) -> str:
    if set(values) != set(expected):
        raise ModelContractError("model-contract template values do not match its declared placeholders")
    rendered_values: dict[str, str] = {}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ModelContractError(f"model-contract value {name} must be text or an integer")
        rendered = str(value)
        if "\x00" in rendered:
            raise ModelContractError(f"model-contract value {name} contains a NUL byte")
        rendered_values[name] = rendered
    rendered = _TOKEN.sub(lambda match: rendered_values[match.group(1)], template)
    if len(rendered) > _MAX_RENDERED_CONTRACT_CHARACTERS:
        raise ModelContractError("rendered model contract exceeds the size limit")
    return rendered


def _validate_tokens(template: str, *, expected: frozenset[str], path: str) -> None:
    if "${" in _PLACEHOLDER.sub("", template):
        raise ModelContractError(f"{path} contains a malformed placeholder")
    placeholders = _PLACEHOLDER.findall(template)
    if any(_TOKEN.fullmatch(f"${{{name}}}") is None for name in placeholders):
        raise ModelContractError(f"{path} contains an unsupported placeholder")
    actual = frozenset(placeholders)
    if actual != expected:
        raise ModelContractError(f"{path} placeholders must be: {', '.join(sorted(expected)) or '<none>'}")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelContractError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ModelContractError(f"{path} keys must be strings")
    return value


def _text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ModelContractError(f"{path} must be non-empty text without NUL bytes")
    normalized = value.strip()
    if len(normalized) > _MAX_RENDERED_CONTRACT_CHARACTERS:
        raise ModelContractError(f"{path} exceeds the size limit")
    return normalized


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise ModelContractError(f"{path} has invalid keys: {'; '.join(details)}")


__all__ = [
    "FailureCopy",
    "MODEL_CONTRACT_SCHEMA",
    "ModelContractCatalog",
    "ModelContractError",
    "load_model_contract_catalog",
]
