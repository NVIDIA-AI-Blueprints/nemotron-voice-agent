# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Self-contained JSON Schema validation for function-tool arguments."""

from __future__ import annotations

import contextvars
import copy
import functools
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import regex
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import extend, validator_for
from loguru import logger
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

_SCHEMA_REFERENCE_KEYWORDS = frozenset({"$dynamicRef", "$recursiveRef", "$ref"})
_MAX_VALIDATION_PATH_PARTS = 8
_MAX_VALIDATION_PATH_PART_LENGTH = 48
_MAX_SCHEMA_BYTES = 32 * 1024
_MAX_SCHEMA_NODES = 1024
_MAX_SCHEMA_DEPTH = 24
_MAX_SCHEMA_COLLECTION_ITEMS = 128
_MAX_ARGUMENT_BYTES = 64 * 1024
_MAX_ARGUMENT_NODES = 2048
_MAX_ARGUMENT_DEPTH = 24
_MAX_ARGUMENT_COLLECTION_ITEMS = 256
_MAX_SCHEMA_REGEXES = 64
_MAX_SCHEMA_REGEX_BYTES = 512
_MAX_VALIDATION_OPERATIONS = 4096
_MAX_VALIDATION_SECONDS = 0.010
_MAX_REGEX_SECONDS = 0.005

# Admission limits for the complete client-supplied ``session.tools`` value.
# The count matches the conventional maximum exposed by Realtime clients; the
# aggregate budget bounds work that happens before individual 32 KiB schemas
# are compiled while leaving ample room for ordinary tool descriptions.
MAX_SESSION_TOOL_COUNT = 128
MAX_SESSION_TOOLS_JSON_BYTES = 256 * 1024


class _JSONValueError(ValueError):
    """A value is not composed exclusively of finite JSON data."""


class _JSONBudgetExceeded(ValueError):
    """A JSON value exceeds a configured structural or serialized limit."""


class _ValidationBudgetExceeded(RuntimeError):
    """A schema evaluation exhausted its connection-local execution budget."""


@dataclass(slots=True)
class _ValidationBudget:
    deadline: float
    operations_remaining: int = _MAX_VALIDATION_OPERATIONS

    def consume(self) -> None:
        self.operations_remaining -= 1
        if self.operations_remaining < 0 or time.monotonic() >= self.deadline:
            raise _ValidationBudgetExceeded("JSON Schema evaluation budget exceeded")

    def regex_timeout(self) -> float:
        self.consume()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise _ValidationBudgetExceeded("JSON Schema evaluation budget exceeded")
        return min(_MAX_REGEX_SECONDS, remaining)


_ACTIVE_VALIDATION_BUDGET: contextvars.ContextVar[_ValidationBudget | None] = contextvars.ContextVar(
    "realtime_tool_validation_budget",
    default=None,
)


@dataclass(frozen=True, slots=True)
class ToolArgumentValidationFailure:
    """A bounded failure safe to return in a terminal tool result."""

    code: str
    message: str


def validate_tool_collection_bounds(tools: list[object], *, param: str = "session.tools") -> None:
    """Bound a complete tool collection without materializing its JSON copy."""
    if len(tools) > MAX_SESSION_TOOL_COUNT:
        raise ValueError(f"{param} supports at most {MAX_SESSION_TOOL_COUNT} tools")

    encoder = json.JSONEncoder(
        allow_nan=False,
        check_circular=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    serialized_bytes = 0
    try:
        for chunk in encoder.iterencode(tools):
            serialized_bytes += len(chunk.encode("utf-8"))
            if serialized_bytes > MAX_SESSION_TOOLS_JSON_BYTES:
                raise _JSONBudgetExceeded
    except _JSONBudgetExceeded as exc:
        raise ValueError(f"{param} exceeds the {MAX_SESSION_TOOLS_JSON_BYTES}-byte aggregate JSON limit") from exc
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError(f"{param} must contain only finite JSON values") from exc


def _validate_bounded_json(
    root: object,
    *,
    max_bytes: int,
    max_nodes: int,
    max_depth: int,
    max_collection_items: int,
) -> None:
    """Reject non-JSON values and structurally expensive JSON documents."""
    pending = [(root, 0)]
    visited: set[int] = set()
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise _JSONBudgetExceeded
        if value is None or type(value) in {bool, int, str}:
            continue
        if type(value) is float:
            if not math.isfinite(value):
                raise _JSONValueError
            continue
        if type(value) is dict:
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if len(value) > max_collection_items or any(type(key) is not str for key in value):
                raise _JSONBudgetExceeded
            nodes += len(value)
            if nodes > max_nodes:
                raise _JSONBudgetExceeded
            pending.extend((item, depth + 1) for item in value.values())
            continue
        if type(value) is list:
            identity = id(value)
            if identity in visited:
                continue
            visited.add(identity)
            if len(value) > max_collection_items:
                raise _JSONBudgetExceeded
            pending.extend((item, depth + 1) for item in value)
            continue
        raise _JSONValueError

    try:
        serialized = json.dumps(
            root,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise _JSONValueError from exc
    if len(serialized.encode("utf-8")) > max_bytes:
        raise _JSONBudgetExceeded


def _consume_validation_operation() -> None:
    budget = _ACTIVE_VALIDATION_BUDGET.get()
    if budget is not None:
        budget.consume()


@functools.lru_cache(maxsize=512)
def _compile_bounded_regex(pattern: str) -> regex.Pattern:
    return regex.compile(pattern)


def _bounded_regex_search(pattern: str, value: str) -> bool:
    """Search with a per-expression and whole-validation deadline."""
    budget = _ACTIVE_VALIDATION_BUDGET.get()
    timeout = budget.regex_timeout() if budget is not None else _MAX_REGEX_SECONDS
    try:
        return (
            _compile_bounded_regex(pattern).search(
                value,
                concurrent=True,
                timeout=timeout,
            )
            is not None
        )
    except TimeoutError as exc:
        raise _ValidationBudgetExceeded("JSON Schema regular-expression budget exceeded") from exc


def _validate_pattern(validator: Validator, pattern: str, instance: object, schema: object):
    if validator.is_type(instance, "string") and not _bounded_regex_search(pattern, instance):
        yield ValidationError("String does not match the declared pattern")


def _validate_pattern_properties(
    validator: Validator,
    pattern_properties: Mapping[str, Any],
    instance: object,
    schema: object,
):
    if not validator.is_type(instance, "object"):
        return
    for pattern, subschema in pattern_properties.items():
        for key, value in instance.items():
            if _bounded_regex_search(pattern, key):
                yield from validator.descend(
                    value,
                    subschema,
                    path=key,
                    schema_path=pattern,
                )


def _validate_additional_properties(
    validator: Validator,
    additional_properties: object,
    instance: object,
    schema: Mapping[str, Any],
):
    """Validate extras without jsonschema's unbounded stdlib-regex helper."""
    if not validator.is_type(instance, "object"):
        return
    properties = schema.get("properties", {})
    patterns = schema.get("patternProperties", {})
    extras: list[str] = []
    for key in instance:
        if key in properties or any(_bounded_regex_search(pattern, key) for pattern in patterns):
            continue
        extras.append(key)

    if validator.is_type(additional_properties, "object"):
        for key in extras:
            yield from validator.descend(instance[key], additional_properties, path=key)
    elif not additional_properties and extras:
        yield ValidationError("Additional properties are not allowed")


def _budgeted_keyword(keyword_validator):
    """Share one operation/deadline budget through every validator descent."""

    @functools.wraps(keyword_validator)
    def _wrapped(validator: Validator, keyword_value: object, instance: object, schema: object):
        _consume_validation_operation()
        errors = keyword_validator(validator, keyword_value, instance, schema)
        if errors is None:
            return
        for error in errors:
            _consume_validation_operation()
            yield error

    return _wrapped


@functools.cache
def _bounded_validator_class(validator_class: type) -> type:
    keyword_validators = dict(validator_class.VALIDATORS)
    keyword_validators.update(
        {
            "additionalProperties": _validate_additional_properties,
            "pattern": _validate_pattern,
            "patternProperties": _validate_pattern_properties,
        }
    )
    return extend(
        validator_class,
        validators={name: _budgeted_keyword(function) for name, function in keyword_validators.items()},
    )


def _validate_schema_resources(schema: Mapping[str, Any], *, validator_class: type) -> None:
    """Resolve references and preflight every schema-bearing resource."""
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    uri = resource.id() or ""
    registry = Registry().with_resource(uri, resource).crawl()
    pending = [(resource, registry.resolver_with_root(resource), True)]
    regex_count = 0
    has_pattern_properties = False
    has_unevaluated_properties = False
    while pending:
        current_resource, resolver, is_root = pending.pop()
        contents = current_resource.contents
        if isinstance(contents, Mapping):
            if not is_root and "$schema" in contents:
                raise ValueError("Nested JSON Schema dialect declarations are not supported")
            if contents.get("format") == "regex":
                # jsonschema's built-in checker compiles the instance with
                # stdlib ``re`` and offers no execution deadline.
                raise ValueError("The JSON Schema 'regex' format is not supported for function tools")
            pattern = contents.get("pattern")
            if isinstance(pattern, str):
                regex_count += 1
                if len(pattern.encode("utf-8")) > _MAX_SCHEMA_REGEX_BYTES:
                    raise ValueError("Function tool regular expressions exceed the supported size limit")
                try:
                    _compile_bounded_regex(pattern)
                except regex.error as exc:
                    raise ValueError("Function tool parameters contain an invalid regular expression") from exc
            pattern_properties = contents.get("patternProperties")
            if isinstance(pattern_properties, Mapping):
                has_pattern_properties = True
                for property_pattern in pattern_properties:
                    regex_count += 1
                    if len(property_pattern.encode("utf-8")) > _MAX_SCHEMA_REGEX_BYTES:
                        raise ValueError("Function tool regular expressions exceed the supported size limit")
                    try:
                        _compile_bounded_regex(property_pattern)
                    except regex.error as exc:
                        raise ValueError("Function tool parameters contain an invalid regular expression") from exc
            if "unevaluatedProperties" in contents:
                has_unevaluated_properties = True
            for keyword in _SCHEMA_REFERENCE_KEYWORDS:
                reference = contents.get(keyword)
                if isinstance(reference, str) and reference and not reference.startswith("#"):
                    raise ValueError("Function tool parameters must not use external JSON Schema references")
                if isinstance(reference, str):
                    try:
                        resolver.lookup(reference)
                    except Exception as exc:
                        raise ValueError(
                            "Function tool parameters contain an unresolved JSON Schema reference"
                        ) from exc
        pending.extend(
            (subresource, resolver.in_subresource(subresource), False)
            for subresource in current_resource.subresources()
        )
    if regex_count > _MAX_SCHEMA_REGEXES:
        raise ValueError("Function tool parameters contain too many regular expressions")
    if "unevaluatedProperties" in validator_class.VALIDATORS and has_pattern_properties and has_unevaluated_properties:
        raise ValueError("Function tool parameters cannot combine patternProperties with unevaluatedProperties")


def compile_tool_arguments_validator(parameters: Mapping[str, Any]) -> Validator:
    """Compile one self-contained JSON Schema without dialect fallback."""
    try:
        schema = copy.deepcopy(dict(parameters))
        _validate_bounded_json(
            schema,
            max_bytes=_MAX_SCHEMA_BYTES,
            max_nodes=_MAX_SCHEMA_NODES,
            max_depth=_MAX_SCHEMA_DEPTH,
            max_collection_items=_MAX_SCHEMA_COLLECTION_ITEMS,
        )
    except _JSONBudgetExceeded as exc:
        raise ValueError("Function tool parameters exceed the supported JSON Schema complexity limits") from exc
    except (_JSONValueError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Function tool parameters must be a finite JSON Schema object") from exc

    if "$schema" in schema:
        validator_class = validator_for(schema, default=None)
        if validator_class is None:
            raise ValueError("Function tool parameters declare an unsupported JSON Schema dialect")
    else:
        validator_class = Draft202012Validator
    try:
        validator_class.check_schema(schema)
    except (SchemaError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("Function tool parameters are not a valid JSON Schema") from exc
    try:
        _validate_schema_resources(schema, validator_class=validator_class)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Function tool parameters could not be resolved as a self-contained JSON Schema") from exc

    # ``jsonschema.Validator.evolve`` consults a nested/root ``$schema`` and
    # would select the stock, unbounded class after a local-reference descent.
    # The dialect has already been selected and checked above, so remove only
    # this private validator copy's root declaration.
    schema.pop("$schema", None)
    bounded_class = _bounded_validator_class(validator_class)
    return bounded_class(schema, format_checker=validator_class.FORMAT_CHECKER)


def _validation_path(error: ValidationError) -> str:
    """Render a bounded instance path without including argument values."""
    path = "$"
    parts = list(error.absolute_path)
    for part in parts[:_MAX_VALIDATION_PATH_PARTS]:
        if isinstance(part, int):
            path += f"[{part}]"
            continue
        text = str(part)
        if (
            not text
            or len(text) > _MAX_VALIDATION_PATH_PART_LENGTH
            or any(not (character.isalnum() or character in {"_", "-"}) for character in text)
        ):
            text = "<property>"
        path += f".{text}"
    if len(parts) > _MAX_VALIDATION_PATH_PARTS:
        path += ".<nested>"
    return path


def tool_argument_validation_failure(
    validator: Validator,
    arguments: object,
) -> ToolArgumentValidationFailure | None:
    """Return a safe failure when arguments do not satisfy a compiled schema."""
    if type(arguments) is not dict:
        return ToolArgumentValidationFailure(
            code="invalid_tool_arguments",
            message="Tool arguments must be a JSON object",
        )
    try:
        _validate_bounded_json(
            arguments,
            max_bytes=_MAX_ARGUMENT_BYTES,
            max_nodes=_MAX_ARGUMENT_NODES,
            max_depth=_MAX_ARGUMENT_DEPTH,
            max_collection_items=_MAX_ARGUMENT_COLLECTION_ITEMS,
        )
    except _JSONBudgetExceeded:
        return ToolArgumentValidationFailure(
            code="invalid_tool_arguments",
            message="Tool arguments exceed the supported JSON complexity limits",
        )
    except (_JSONValueError, TypeError, ValueError, OverflowError, RecursionError):
        return ToolArgumentValidationFailure(
            code="invalid_tool_arguments",
            message="Tool arguments must contain only finite JSON values",
        )

    token = _ACTIVE_VALIDATION_BUDGET.set(_ValidationBudget(deadline=time.monotonic() + _MAX_VALIDATION_SECONDS))
    try:
        error = next(validator.iter_errors(arguments), None)
    except Exception as exc:
        # Never attach a traceback here: validator frames can contain the full
        # client argument object. The type is sufficient operational evidence.
        logger.warning(
            "Tool argument JSON Schema evaluation failed error_type={}",
            type(exc).__name__,
        )
        return ToolArgumentValidationFailure(
            code="tool_schema_error",
            message="Tool arguments could not be checked against the declared JSON Schema",
        )
    finally:
        _ACTIVE_VALIDATION_BUDGET.reset(token)
    if error is None:
        return None

    keyword = str(error.validator or "constraint")
    if len(keyword) > 48 or any(not (character.isalnum() or character in {"_", "-"}) for character in keyword):
        keyword = "constraint"
    return ToolArgumentValidationFailure(
        code="invalid_tool_arguments",
        message=(
            f"Tool arguments do not match the declared JSON Schema at {_validation_path(error)} (constraint: {keyword})"
        ),
    )
