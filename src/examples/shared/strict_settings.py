# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict settings models: a frozen base that refuses unknown keys, and their field types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

import yaml
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, StrictStr, with_config


class StrictSettings(BaseModel):
    """Base of a settings model: frozen, and an unknown key is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# A whole number of at least 1: a fraction or a boolean is refused.
WholeNumber = Annotated[int, Field(strict=True, ge=1)]
# A duration in seconds above 0; a whole number is a number of seconds as well.
Seconds = Annotated[float, Field(strict=True, gt=0)]


def _not_blank(text: str) -> str:
    if not text.strip():
        raise ValueError("must be non-empty text")
    return text


Text = Annotated[StrictStr, AfterValidator(_not_blank)]


def _from_yaml(schema: Any) -> Any:
    """Read a tool's schema written as YAML text; return any other value as it is."""
    if not isinstance(schema, str):
        return schema
    try:
        return yaml.safe_load(schema)
    except yaml.YAMLError as exc:
        raise ValueError(f"is not a schema in YAML: {exc}") from None


def _an_object(parameters: dict[str, Any]) -> dict[str, Any]:
    if parameters.get("type") != "object":
        raise ValueError("must be a JSON schema of type object")
    return parameters


@with_config(ConfigDict(extra="forbid"))
class ToolSchema(TypedDict):
    """One tool's schema, as a mapping or as its YAML text: a description, and its parameters."""

    description: Text
    parameters: Annotated[dict[str, Any], AfterValidator(_an_object)]


Tool = Annotated[ToolSchema, BeforeValidator(_from_yaml)]


def with_blocks(raw: Mapping[str, Any], own: set[str]) -> dict[str, Any]:
    """Return ``raw`` with every key not in ``own`` gathered under ``blocks``, one adapter's block each."""
    blocks = {str(key): block for key, block in raw.items() if key not in own}
    return {**{key: value for key, value in raw.items() if key in own}, "blocks": blocks}


def placed(texts: Mapping[str, Any], what: str) -> dict[str, Any]:
    """Return ``texts`` as blocks, each text at the place its dotted name gives; two at one place fail."""
    blocks: dict[str, Any] = {}
    for name, text in texts.items():
        *path, key = str(name).split(".")
        block = blocks
        for part in path:
            block = block.setdefault(part, {})
            if not isinstance(block, dict):
                break
        if not isinstance(block, dict) or key in block:
            raise ValueError(f"{what} {name} names a place that another text holds")
        block[key] = text
    return blocks
