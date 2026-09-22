# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Validated, provider-neutral interaction profiles for VoiceClaw."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

INTERACTION_PROFILE_SCHEMA = "voiceclaw.interaction-profiles.v2"
PACKAGED_INTERACTION_PROFILES = "interaction_profiles.v2.yaml"
# Match the public goal limit without coupling the provider-neutral schema to a runtime port.
MAX_DELEGATED_GOAL_BYTES = 48 * 1024
MAX_DELEGATED_GOAL_CHARACTERS = MAX_DELEGATED_GOAL_BYTES // 4

_MAX_CATALOG_BYTES = 256 * 1024
_MAX_DESCRIPTION_CHARACTERS = 16 * 1024
_MAX_PROPERTY_DESCRIPTION_CHARACTERS = 4 * 1024
_MAX_PROFILES = 64
_PROFILE_NAME = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")
_CANONICAL_TOOLS = frozenset(
    {
        "conversation.respond",
        "work.delegate",
        "work.answer_agent",
        "work.cancel",
        "work.status",
    }
)


class InteractionProfileError(ValueError):
    """The interaction-profile catalog or a prose override is invalid."""


class SessionScope(StrEnum):
    """Identity scope used to correlate backend interaction state."""

    NONE = "none"
    ATTACHMENT = "attachment"
    TARGET = "target"


class WorkCardinality(StrEnum):
    """How active Work is addressed within an interaction scope."""

    MANY = "many"
    SINGLE = "single"
    PER_TARGET = "per_target"


class ArgumentBinding(StrEnum):
    """Core-owned model argument shape for a canonical tool."""

    NONE = "none"
    GOAL_REQUIRED = "goal_required"
    WORK_ID_REQUIRED = "work_id_required"
    WORK_ID_OPTIONAL = "work_id_optional"
    QUERY_ID_REQUIRED = "query_id_required"
    TARGET_REQUIRED = "target_required"
    TARGET_OPTIONAL = "target_optional"
    TARGET_GOAL_REQUIRED = "target_goal_required"


class InteractionOperation(StrEnum):
    """Canonical backend operation referenced by an interaction tool."""

    SUBMIT = "work.submit"
    STATUS = "work.status"
    CANCEL = "work.cancel"
    STEER = "work.steer"
    ANSWER_QUERY = "work.answer_query"
    RESPOND_PERMISSION = "work.respond_permission"


_ARGUMENTS_BY_BINDING: Mapping[ArgumentBinding, tuple[str, ...]] = MappingProxyType(
    {
        ArgumentBinding.NONE: (),
        ArgumentBinding.GOAL_REQUIRED: ("goal",),
        ArgumentBinding.WORK_ID_REQUIRED: ("work_id",),
        ArgumentBinding.WORK_ID_OPTIONAL: ("work_id",),
        ArgumentBinding.QUERY_ID_REQUIRED: ("query_id",),
        ArgumentBinding.TARGET_REQUIRED: ("agent",),
        ArgumentBinding.TARGET_OPTIONAL: ("agent",),
        ArgumentBinding.TARGET_GOAL_REQUIRED: ("agent", "goal"),
    }
)
_REQUIRED_ARGUMENTS_BY_BINDING: Mapping[ArgumentBinding, tuple[str, ...]] = MappingProxyType(
    {
        ArgumentBinding.NONE: (),
        ArgumentBinding.GOAL_REQUIRED: ("goal",),
        ArgumentBinding.WORK_ID_REQUIRED: ("work_id",),
        ArgumentBinding.WORK_ID_OPTIONAL: (),
        ArgumentBinding.QUERY_ID_REQUIRED: ("query_id",),
        ArgumentBinding.TARGET_REQUIRED: ("agent",),
        ArgumentBinding.TARGET_OPTIONAL: (),
        ArgumentBinding.TARGET_GOAL_REQUIRED: ("agent", "goal"),
    }
)
_ARGUMENT_LIMITS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        "goal": (1, MAX_DELEGATED_GOAL_CHARACTERS),
        "work_id": (1, 512),
        "query_id": (1, 512),
        "agent": (1, 256),
    }
)
_ENUM_ARGUMENTS = frozenset({"work_id", "query_id", "agent"})

_QUERY_OPERATIONS = frozenset({InteractionOperation.ANSWER_QUERY, InteractionOperation.RESPOND_PERMISSION})


def _profile_name(value: object, path: str) -> str:
    """Return one bounded operator-selected profile identifier."""
    if not isinstance(value, str) or len(value) > 64 or _PROFILE_NAME.fullmatch(value) is None:
        raise InteractionProfileError(f"{path} must be 1..64 lowercase letters, digits, hyphens, or underscores")
    return value


def _validate_scope_cardinality(
    name: str,
    scope: SessionScope,
    cardinality: WorkCardinality,
) -> None:
    """Reject combinations for which core state gating has no safe meaning."""
    valid = (
        (scope is SessionScope.NONE and cardinality is WorkCardinality.MANY)
        or (scope is SessionScope.ATTACHMENT and cardinality in {WorkCardinality.SINGLE, WorkCardinality.MANY})
        or (scope is SessionScope.TARGET and cardinality is WorkCardinality.PER_TARGET)
    )
    if not valid:
        raise InteractionProfileError(
            f"profile {name} has incompatible session_scope={scope.value} and work_cardinality={cardinality.value}"
        )


def _expected_tool_semantics(
    tool_name: str,
    scope: SessionScope,
    cardinality: WorkCardinality,
) -> tuple[frozenset[InteractionOperation], ArgumentBinding]:
    """Return the one core-owned semantic shape safe for this normalized tool."""
    if tool_name == "conversation.respond":
        return frozenset(), ArgumentBinding.NONE
    if tool_name == "work.delegate":
        if scope is SessionScope.NONE:
            return frozenset({InteractionOperation.SUBMIT}), ArgumentBinding.GOAL_REQUIRED
        if scope is SessionScope.TARGET:
            return (
                frozenset({InteractionOperation.SUBMIT, InteractionOperation.STEER}),
                ArgumentBinding.TARGET_GOAL_REQUIRED,
            )
        if cardinality is WorkCardinality.SINGLE:
            return (
                frozenset({InteractionOperation.SUBMIT, InteractionOperation.STEER}),
                ArgumentBinding.GOAL_REQUIRED,
            )
        return (
            frozenset({InteractionOperation.SUBMIT, InteractionOperation.STEER}),
            ArgumentBinding.GOAL_REQUIRED,
        )
    if tool_name == "work.answer_agent":
        if scope is SessionScope.NONE:
            raise InteractionProfileError("work.answer_agent requires a session-scoped interaction profile")
        return _QUERY_OPERATIONS, ArgumentBinding.QUERY_ID_REQUIRED
    if tool_name == "work.cancel":
        if scope is SessionScope.TARGET:
            binding = ArgumentBinding.TARGET_REQUIRED
        elif cardinality is WorkCardinality.SINGLE:
            binding = ArgumentBinding.NONE
        else:
            binding = ArgumentBinding.WORK_ID_REQUIRED
        return frozenset({InteractionOperation.CANCEL}), binding
    if tool_name == "work.status":
        if scope is SessionScope.TARGET:
            binding = ArgumentBinding.TARGET_OPTIONAL
        elif cardinality is WorkCardinality.SINGLE:
            binding = ArgumentBinding.NONE
        else:
            binding = ArgumentBinding.WORK_ID_OPTIONAL
        return frozenset({InteractionOperation.STATUS}), binding
    raise InteractionProfileError(f"unsupported canonical tool: {tool_name!r}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects ambiguous duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as error:
            raise InteractionProfileError("interaction-profile mapping keys must be scalar values") from error
        if duplicate:
            raise InteractionProfileError(f"duplicate interaction-profile key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class InteractionToolSpec:
    """One canonical tool's immutable routing semantics and configurable prose."""

    name: str
    operations: frozenset[InteractionOperation]
    argument_binding: ArgumentBinding
    description: str
    property_descriptions: Mapping[str, str]

    def __post_init__(self) -> None:
        """Normalize and validate a tool independently of its profile."""
        if self.name not in _CANONICAL_TOOLS:
            raise InteractionProfileError(f"unsupported canonical tool: {self.name!r}")
        try:
            operations = frozenset(InteractionOperation(operation) for operation in self.operations)
            binding = ArgumentBinding(self.argument_binding)
        except (TypeError, ValueError) as error:
            raise InteractionProfileError(
                f"{self.name} contains an unsupported operation or argument binding"
            ) from error
        description = _prose(self.description, f"tools.{self.name}.description", _MAX_DESCRIPTION_CHARACTERS)
        properties = _mapping(self.property_descriptions, f"tools.{self.name}.property_descriptions")
        expected_properties = set(_ARGUMENTS_BY_BINDING[binding])
        _exact_keys(properties, expected_properties, f"tools.{self.name}.property_descriptions")
        parsed_properties = {
            key: _prose(
                value,
                f"tools.{self.name}.property_descriptions.{key}",
                _MAX_PROPERTY_DESCRIPTION_CHARACTERS,
            )
            for key, value in properties.items()
        }
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "argument_binding", binding)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "property_descriptions", MappingProxyType(parsed_properties))

    @property
    def arguments(self) -> tuple[str, ...]:
        """Return the closed canonical argument list in schema order."""
        return _ARGUMENTS_BY_BINDING[self.argument_binding]

    @property
    def required_arguments(self) -> tuple[str, ...]:
        """Return the canonical arguments required by this binding."""
        return _REQUIRED_ARGUMENTS_BY_BINDING[self.argument_binding]

    def authorized_operations(
        self,
        backend_operations: Iterable[str],
    ) -> frozenset[InteractionOperation]:
        """Intersect this presentation profile with authoritative backend evidence."""
        try:
            authorized = frozenset(InteractionOperation(operation) for operation in backend_operations)
        except (TypeError, ValueError) as error:
            raise InteractionProfileError("backend operations contain an unsupported value") from error
        return self.operations & authorized

    def render_input_schema(
        self,
        *,
        enum_values: Mapping[str, Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        """Render a closed JSON Schema, optionally narrowing identifier arguments."""
        narrowed = {} if enum_values is None else dict(_mapping(enum_values, "enum_values"))
        unknown = sorted(set(narrowed) - set(self.arguments))
        if unknown:
            raise InteractionProfileError(f"enum_values contains unknown arguments: {', '.join(unknown)}")
        if set(narrowed) - _ENUM_ARGUMENTS:
            name = sorted(set(narrowed) - _ENUM_ARGUMENTS)[0]
            raise InteractionProfileError(f"argument {name} cannot be narrowed with enum values")

        properties: dict[str, Any] = {}
        for name in self.arguments:
            minimum, maximum = _ARGUMENT_LIMITS[name]
            property_schema: dict[str, Any] = {
                "type": "string",
                "minLength": minimum,
                "maxLength": maximum,
                "description": self.property_descriptions[name],
            }
            if name in narrowed:
                values = narrowed[name]
                if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                    raise InteractionProfileError(f"enum_values.{name} must be a sequence of strings")
                normalized = tuple(values)
                if (
                    not normalized
                    or len(normalized) != len(set(normalized))
                    or any(
                        not isinstance(value, str) or not value or "\x00" in value or len(value) > maximum
                        for value in normalized
                    )
                ):
                    raise InteractionProfileError(f"enum_values.{name} must contain unique bounded strings")
                property_schema["enum"] = list(normalized)
            properties[name] = property_schema
        return {
            "type": "object",
            "properties": properties,
            "required": list(self.required_arguments),
            "additionalProperties": False,
        }


@dataclass(frozen=True, slots=True)
class InteractionProfile:
    """One named composition of core-owned semantics and operator-owned prose."""

    name: str
    session_scope: SessionScope
    work_cardinality: WorkCardinality
    tools: Mapping[str, InteractionToolSpec]

    def __post_init__(self) -> None:
        """Validate a profile by normalized semantics rather than its name."""
        name = _profile_name(self.name, "profile name")
        try:
            scope = SessionScope(self.session_scope)
            cardinality = WorkCardinality(self.work_cardinality)
        except (TypeError, ValueError) as error:
            raise InteractionProfileError(f"profile {name} has an unsupported scope or cardinality") from error
        _validate_scope_cardinality(name, scope, cardinality)
        tools = dict(_mapping(self.tools, f"profiles.{name}.tools"))
        unknown_tools = sorted(set(tools) - _CANONICAL_TOOLS)
        if unknown_tools:
            raise InteractionProfileError(
                f"profiles.{name}.tools contains unsupported canonical tools: {', '.join(unknown_tools)}"
            )
        if "conversation.respond" not in tools:
            raise InteractionProfileError(f"profiles.{name}.tools must define conversation.respond")
        for tool_name, tool in tools.items():
            if not isinstance(tool, InteractionToolSpec) or tool.name != tool_name:
                raise InteractionProfileError(f"profiles.{name}.tools.{tool_name} is invalid")
            expected_operations, expected_binding = _expected_tool_semantics(tool_name, scope, cardinality)
            if tool.operations != expected_operations or tool.argument_binding is not expected_binding:
                raise InteractionProfileError(
                    f"profiles.{name}.tools.{tool_name} must use operations "
                    f"{sorted(operation.value for operation in expected_operations)} and "
                    f"argument_binding={expected_binding.value}"
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "session_scope", scope)
        object.__setattr__(self, "work_cardinality", cardinality)
        object.__setattr__(self, "tools", MappingProxyType(tools))

    def tool(self, logical_name: str) -> InteractionToolSpec:
        """Return one canonical tool or fail with a profile-specific error."""
        try:
            return self.tools[logical_name]
        except KeyError as error:
            raise InteractionProfileError(f"profile {self.name} does not define tool {logical_name!r}") from error

    @property
    def resolved_digest(self) -> str:
        """Identify the exact normalized semantics and prose used by the model."""
        tool_names = sorted(
            self.tools,
            key=lambda tool_name: (tool_name != "conversation.respond", tool_name),
        )
        canonical = {
            "name": self.name,
            "session_scope": self.session_scope.value,
            "work_cardinality": self.work_cardinality.value,
            "tools": [
                {
                    "name": tool_name,
                    "operations": sorted(operation.value for operation in self.tools[tool_name].operations),
                    "argument_binding": self.tools[tool_name].argument_binding.value,
                    "description": self.tools[tool_name].description,
                    "property_descriptions": dict(self.tools[tool_name].property_descriptions),
                }
                for tool_name in tool_names
            ],
        }
        payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def with_prose_overrides(self, overrides: Mapping[str, Any]) -> InteractionProfile:
        """Apply a validated per-backend prose-only overlay."""
        raw_overrides = _mapping(overrides, "prose_overrides")
        unknown_tools = sorted(set(raw_overrides) - set(self.tools))
        if unknown_tools:
            raise InteractionProfileError(f"prose_overrides contains unknown tools: {', '.join(unknown_tools)}")
        updated = dict(self.tools)
        for tool_name, raw_override in raw_overrides.items():
            path = f"prose_overrides.{tool_name}"
            override = _mapping(raw_override, path)
            _allowed_keys(override, {"description", "property_descriptions"}, path)
            if not override:
                raise InteractionProfileError(f"{path} must change at least one prose field")
            current = self.tools[tool_name]
            description = current.description
            if "description" in override:
                description = _prose(override["description"], f"{path}.description", _MAX_DESCRIPTION_CHARACTERS)
            properties = dict(current.property_descriptions)
            if "property_descriptions" in override:
                property_overrides = _mapping(override["property_descriptions"], f"{path}.property_descriptions")
                unknown_properties = sorted(set(property_overrides) - set(current.arguments))
                if unknown_properties:
                    raise InteractionProfileError(
                        f"{path}.property_descriptions contains unknown arguments: {', '.join(unknown_properties)}"
                    )
                if not property_overrides:
                    raise InteractionProfileError(f"{path}.property_descriptions must not be empty")
                for property_name, value in property_overrides.items():
                    properties[property_name] = _prose(
                        value,
                        f"{path}.property_descriptions.{property_name}",
                        _MAX_PROPERTY_DESCRIPTION_CHARACTERS,
                    )
            updated[tool_name] = replace(
                current,
                description=description,
                property_descriptions=properties,
            )
        return replace(self, tools=updated)


@dataclass(frozen=True, slots=True)
class InteractionProfileCatalog:
    """One immutable, versioned collection of normalized interaction profiles."""

    schema_version: str
    digest: str
    profiles: Mapping[str, InteractionProfile]

    def __post_init__(self) -> None:
        """Validate bounded catalog identity and detach the profile mapping."""
        if self.schema_version != INTERACTION_PROFILE_SCHEMA:
            raise InteractionProfileError(f"schema_version must be {INTERACTION_PROFILE_SCHEMA}")
        profiles = dict(_mapping(self.profiles, "profiles"))
        if not profiles or len(profiles) > _MAX_PROFILES:
            raise InteractionProfileError(f"profiles must define between 1 and {_MAX_PROFILES} profiles")
        for name in profiles:
            _profile_name(name, "profiles key")
        invalid = any(
            not isinstance(profile, InteractionProfile) or profile.name != name for name, profile in profiles.items()
        )
        if invalid:
            raise InteractionProfileError("profiles contains an invalid interaction profile")
        if not isinstance(self.digest, str) or not self.digest.startswith("sha256:") or len(self.digest) != 71:
            raise InteractionProfileError("interaction-profile digest must be a SHA-256 digest")
        object.__setattr__(self, "profiles", MappingProxyType(profiles))

    def resolve(
        self,
        name: str,
        *,
        prose_overrides: Mapping[str, Any] | None = None,
    ) -> InteractionProfile:
        """Select one profile and apply an optional per-backend prose overlay."""
        try:
            profile = self.profiles[name]
        except (KeyError, TypeError) as error:
            raise InteractionProfileError(f"unknown interaction profile: {name!r}") from error
        if prose_overrides is None:
            return profile
        return profile.with_prose_overrides(prose_overrides)


def load_interaction_profile_catalog(path: str | Path | None = None) -> InteractionProfileCatalog:
    """Load packaged examples or one trusted operator-owned replacement."""
    if path is None:
        resource = files("voiceclaw.resources").joinpath(PACKAGED_INTERACTION_PROFILES)
        try:
            text = resource.read_text(encoding="utf-8")
        except OSError as error:
            raise InteractionProfileError("cannot read packaged interaction-profile catalog") from error
        source = f"packaged:{PACKAGED_INTERACTION_PROFILES}"
    else:
        source_path = Path(path)
        try:
            if source_path.stat().st_size > _MAX_CATALOG_BYTES:
                raise InteractionProfileError("interaction-profile catalog exceeds the size limit")
            text = source_path.read_text(encoding="utf-8")
        except OSError as error:
            raise InteractionProfileError(f"cannot read interaction-profile catalog: {source_path}") from error
        source = str(source_path)
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise InteractionProfileError(f"interaction-profile catalog is not valid UTF-8: {source}") from error
    if not encoded or len(encoded) > _MAX_CATALOG_BYTES or b"\x00" in encoded:
        raise InteractionProfileError("interaction-profile catalog is empty, oversized, or contains NUL bytes")
    try:
        raw = yaml.load(text, Loader=_UniqueKeyLoader)
    except InteractionProfileError:
        raise
    except yaml.YAMLError as error:
        raise InteractionProfileError(f"invalid YAML in interaction-profile catalog: {source}") from error
    return parse_interaction_profile_catalog(_mapping(raw, "catalog"))


def parse_interaction_profile_catalog(raw: Mapping[str, Any]) -> InteractionProfileCatalog:
    """Validate one already-decoded normalized interaction-profile catalog."""
    _exact_keys(raw, {"schema_version", "profiles"}, "catalog")
    schema_version = _text(raw.get("schema_version"), "schema_version")
    if schema_version != INTERACTION_PROFILE_SCHEMA:
        raise InteractionProfileError(f"schema_version must be {INTERACTION_PROFILE_SCHEMA}")
    profiles_raw = _mapping(raw.get("profiles"), "profiles")
    if not profiles_raw or len(profiles_raw) > _MAX_PROFILES:
        raise InteractionProfileError(f"profiles must define between 1 and {_MAX_PROFILES} profiles")
    profiles: dict[str, InteractionProfile] = {}
    for raw_profile_name, raw_profile in profiles_raw.items():
        profile_name = _profile_name(raw_profile_name, "profiles key")
        path = f"profiles.{profile_name}"
        profile = _mapping(raw_profile, path)
        _exact_keys(profile, {"session_scope", "work_cardinality", "tools"}, path)
        tools_raw = _mapping(profile.get("tools"), f"{path}.tools")
        unknown_tools = sorted(set(tools_raw) - _CANONICAL_TOOLS)
        if unknown_tools:
            raise InteractionProfileError(
                f"{path}.tools contains unsupported canonical tools: {', '.join(unknown_tools)}"
            )
        if "conversation.respond" not in tools_raw:
            raise InteractionProfileError(f"{path}.tools must define conversation.respond")
        tools: dict[str, InteractionToolSpec] = {}
        for tool_name, raw_tool in tools_raw.items():
            tool_path = f"{path}.tools.{tool_name}"
            tool = _mapping(raw_tool, tool_path)
            _exact_keys(
                tool,
                {"operations", "argument_binding", "description", "property_descriptions"},
                tool_path,
            )
            operations_raw = tool.get("operations")
            if not isinstance(operations_raw, list):
                raise InteractionProfileError(f"{tool_path}.operations must be a list")
            try:
                operations = frozenset(InteractionOperation(operation) for operation in operations_raw)
            except (TypeError, ValueError) as error:
                raise InteractionProfileError(f"{tool_path}.operations contains an unsupported operation") from error
            if len(operations) != len(operations_raw):
                raise InteractionProfileError(f"{tool_path}.operations contains duplicates")
            try:
                binding = ArgumentBinding(tool.get("argument_binding"))
            except (TypeError, ValueError) as error:
                raise InteractionProfileError(f"{tool_path}.argument_binding is unsupported") from error
            tools[tool_name] = InteractionToolSpec(
                name=tool_name,
                operations=operations,
                argument_binding=binding,
                description=tool.get("description"),
                property_descriptions=_mapping(
                    tool.get("property_descriptions"),
                    f"{tool_path}.property_descriptions",
                ),
            )
        try:
            session_scope = SessionScope(profile.get("session_scope"))
            work_cardinality = WorkCardinality(profile.get("work_cardinality"))
        except (TypeError, ValueError) as error:
            raise InteractionProfileError(f"{path} has an unsupported scope or cardinality") from error
        profiles[profile_name] = InteractionProfile(
            name=profile_name,
            session_scope=session_scope,
            work_cardinality=work_cardinality,
            tools=tools,
        )
    canonical = json.dumps(raw, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return InteractionProfileCatalog(
        schema_version=schema_version,
        digest=f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        profiles=profiles,
    )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InteractionProfileError(f"{path} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise InteractionProfileError(f"{path} keys must be strings")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise InteractionProfileError(f"{path} must be non-empty text without NUL bytes")
    return value


def _prose(value: Any, path: str, maximum: int) -> str:
    text = _text(value, path).strip()
    if not text:
        raise InteractionProfileError(f"{path} must contain non-whitespace text")
    if len(text) > maximum:
        raise InteractionProfileError(f"{path} exceeds the size limit")
    return text


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    details = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        details.append(f"missing {', '.join(missing)}")
    if unknown:
        details.append(f"unknown {', '.join(unknown)}")
    raise InteractionProfileError(f"{path} has invalid keys: {'; '.join(details)}")


def _allowed_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InteractionProfileError(f"{path} has unknown keys: {', '.join(unknown)}")


__all__ = [
    "INTERACTION_PROFILE_SCHEMA",
    "MAX_DELEGATED_GOAL_BYTES",
    "MAX_DELEGATED_GOAL_CHARACTERS",
    "ArgumentBinding",
    "InteractionOperation",
    "InteractionProfile",
    "InteractionProfileCatalog",
    "InteractionProfileError",
    "InteractionToolSpec",
    "SessionScope",
    "WorkCardinality",
    "load_interaction_profile_catalog",
    "parse_interaction_profile_catalog",
]
