# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Server-owned Realtime tools and protected session projection."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from voiceclaw.domain.capabilities import SemanticTool
from voiceclaw.interaction_profiles import (
    MAX_DELEGATED_GOAL_CHARACTERS,
    InteractionProfile,
    load_interaction_profile_catalog,
)
from voiceclaw.model_contracts import ModelContractCatalog, load_model_contract_catalog

_MAX_ESCAPED_JSON_BYTES_PER_CHARACTER = 12
_MAX_NON_GOAL_ARGUMENT_CHARACTERS = 512
_MAX_JSON_STRUCTURE_BYTES = 4096
_MAX_ARGUMENT_BYTES = (
    MAX_DELEGATED_GOAL_CHARACTERS + _MAX_NON_GOAL_ARGUMENT_CHARACTERS
) * _MAX_ESCAPED_JSON_BYTES_PER_CHARACTER + _MAX_JSON_STRUCTURE_BYTES
_MAX_INSTRUCTIONS_CHARACTERS = 64_000


class ProtectedTool(StrEnum):
    """Server-owned routing and Work tools hidden behind the public facade."""

    CONVERSATION_RESPOND = "voiceclaw_conversation_respond"
    WORK_DELEGATE = "voiceclaw_work_delegate"
    WORK_ANSWER_AGENT = "voiceclaw_work_answer_agent"
    WORK_CANCEL = "voiceclaw_work_cancel"
    WORK_STATUS = "voiceclaw_work_status"

    @property
    def logical_name(self) -> str:
        """Return the protocol-neutral Work tool name."""
        return {
            self.CONVERSATION_RESPOND: "conversation.respond",
            self.WORK_DELEGATE: "work.delegate",
            self.WORK_ANSWER_AGENT: "work.answer_agent",
            self.WORK_CANCEL: "work.cancel",
            self.WORK_STATUS: "work.status",
        }[self]

    @classmethod
    def from_logical_name(cls, name: str) -> ProtectedTool:
        """Resolve one logical Work tool to its wire-safe Realtime name."""
        for tool in cls:
            if tool.logical_name == name:
                return tool
        raise ValueError(name)


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    """Validated protected function call emitted by a frontend model."""

    call_id: str
    name: ProtectedTool
    arguments: dict[str, Any]
    finalized_user_text: str | None = None

    @property
    def logical_name(self) -> str:
        """Return the Interaction Manager tool name."""
        return self.name.logical_name


class VoiceClawToolRegistry:
    """Project, protect, and validate VoiceClaw-owned frontend tools."""

    def __init__(
        self,
        *,
        tools: Iterable[SemanticTool] = (),
        include_direct_route: bool = False,
        contracts: ModelContractCatalog | None = None,
        interaction_profile: InteractionProfile | None = None,
    ) -> None:
        """Render only the semantic tools authorized by the Interaction Manager."""
        self._contracts = contracts or load_model_contract_catalog()
        self._interaction_profile = interaction_profile or load_interaction_profile_catalog().resolve("stateless")
        definitions: dict[ProtectedTool, SemanticTool] = {}
        try:
            for semantic in tools:
                if not isinstance(semantic, SemanticTool):
                    raise TypeError("tools must contain SemanticTool values")
                wire = ProtectedTool.from_logical_name(semantic.name)
                if wire in definitions:
                    raise ValueError(f"duplicate semantic tool: {semantic.name}")
                definitions[wire] = semantic
        except (TypeError, ValueError) as error:
            raise ValueError("tools contains an invalid semantic tool") from error
        if include_direct_route and definitions:
            direct = self._interaction_profile.tool("conversation.respond")
            definitions[ProtectedTool.CONVERSATION_RESPOND] = SemanticTool(
                name="conversation.respond",
                description=direct.description,
                input_schema=direct.render_input_schema(),
            )
        self._definitions = definitions
        self._enabled = frozenset(definitions)

    @property
    def enabled(self) -> frozenset[ProtectedTool]:
        """Return the immutable set of server-owned tools."""
        return self._enabled

    def schemas(self) -> tuple[dict[str, Any], ...]:
        """Render semantic definitions as OpenAI Realtime function schemas."""
        schemas: list[dict[str, Any]] = []
        for wire in ProtectedTool:
            semantic = self._definitions.get(wire)
            if semantic is None:
                continue
            schemas.append(
                {
                    "type": "function",
                    "name": wire.value,
                    "description": semantic.description,
                    "parameters": copy.deepcopy(dict(semantic.input_schema)),
                }
            )
        return tuple(schemas)

    def merge_session_update(
        self,
        event: dict[str, Any],
        *,
        static_instructions: str | None = None,
        dynamic_projection: str = "",
    ) -> dict[str, Any]:
        """Merge protected schemas and instructions into a client session update."""
        if event.get("type") != "session.update" or not isinstance(event.get("session"), dict):
            raise ValueError("session.update requires a session object")
        session = copy.deepcopy(event["session"])
        client_tools = session.get("tools", [])
        if not isinstance(client_tools, list):
            raise ValueError("session.tools must be a list")
        # Names remain server-owned even when an adapter is disabled. Otherwise
        # a browser could define the disabled name and have it treated as a
        # client-owned tool in the same protocol namespace.
        for tool in client_tools:
            if not isinstance(tool, dict):
                raise ValueError("session.tools entries must be objects")
            name = tool.get("name")
            if isinstance(name, str) and name.startswith("voiceclaw_"):
                raise ValueError(f"client cannot define protected tool {name!r}")
        client_instructions = session.get("instructions", "")
        if client_instructions is None:
            client_instructions = ""
        if not isinstance(client_instructions, str):
            raise ValueError("session.instructions must be a string or null")
        policy = self._contracts.static_instructions if static_instructions is None else static_instructions
        sections = [policy.strip()]
        if client_instructions.strip():
            sections.append(
                self._contracts.render_instruction("untrusted_session", content=client_instructions.strip())
            )
        if dynamic_projection.strip():
            sections.append(
                self._contracts.render_instruction("bootstrap_projection", projection=dynamic_projection.strip())
            )
        instructions = "\n\n".join(section for section in sections if section)
        if not instructions or len(instructions) > _MAX_INSTRUCTIONS_CHARACTERS:
            raise ValueError("merged session instructions are empty or too large")
        session["instructions"] = instructions
        session["tools"] = [*client_tools, *copy.deepcopy(self.schemas())]
        if self._enabled:
            session["tool_choice"] = "auto"
        return {**copy.deepcopy(event), "session": session}

    def parse_call(self, item: object) -> ParsedToolCall | None:
        """Return a validated protected call, or ``None`` for other output items."""
        if not isinstance(item, dict) or item.get("type") != "function_call":
            return None
        name_raw = item.get("name")
        if not isinstance(name_raw, str):
            return None
        try:
            name = ProtectedTool(name_raw)
        except ValueError:
            return None
        if name not in self._enabled:
            raise ValueError(f"protected tool {name.value!r} is not enabled")
        call_id = item.get("call_id")
        arguments_raw = item.get("arguments")
        if not isinstance(call_id, str) or not call_id.strip() or len(call_id) > 512:
            raise ValueError("protected tool call_id is invalid")
        try:
            arguments_size = len(arguments_raw.encode("utf-8")) if isinstance(arguments_raw, str) else 0
        except UnicodeEncodeError as error:
            raise ValueError("protected tool arguments are invalid") from error
        if not isinstance(arguments_raw, str) or arguments_size > _MAX_ARGUMENT_BYTES:
            raise ValueError("protected tool arguments are invalid")
        try:
            arguments = json.loads(arguments_raw or "{}")
        except json.JSONDecodeError as error:
            raise ValueError("protected tool arguments are not valid JSON") from error
        if not isinstance(arguments, dict):
            raise ValueError("protected tool arguments must be an object")
        arguments = self._validate_arguments(arguments, self._definitions[name].input_schema)
        return ParsedToolCall(call_id=call_id.strip(), name=name, arguments=arguments)

    @classmethod
    def _validate_arguments(
        cls,
        arguments: dict[str, Any],
        schema: Mapping[str, Any],
    ) -> dict[str, Any]:
        if schema.get("type") != "object" or not isinstance(schema.get("properties"), Mapping):
            raise ValueError("protected tool schema is invalid")
        properties = schema["properties"]
        required_raw = schema.get("required", [])
        if not isinstance(required_raw, list) or not all(isinstance(name, str) for name in required_raw):
            raise ValueError("protected tool schema is invalid")
        required = set(required_raw)
        unknown = set(arguments) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError("protected tool arguments contain unknown properties")
        missing = required - set(arguments)
        if missing:
            raise ValueError("protected tool arguments are missing required properties")
        validated: dict[str, Any] = {}
        for key, value in arguments.items():
            property_schema = properties.get(key)
            if property_schema is None:
                validated[key] = copy.deepcopy(value)
                continue
            if not isinstance(property_schema, Mapping):
                raise ValueError("protected tool schema is invalid")
            validated[key] = cls._validate_value(value, property_schema)
        return validated

    @classmethod
    def _validate_value(cls, value: Any, schema: Mapping[str, Any]) -> Any:
        expected = schema.get("type")
        if expected == "string":
            if not isinstance(value, str):
                raise ValueError("protected tool argument must be a string")
            normalized = value.strip()
            minimum = schema.get("minLength", 0)
            maximum = schema.get("maxLength", 512)
            if (
                isinstance(minimum, bool)
                or not isinstance(minimum, int)
                or minimum < 0
                or isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or maximum < minimum
            ):
                raise ValueError("protected tool schema is invalid")
            try:
                normalized.encode("utf-8")
            except UnicodeEncodeError as error:
                raise ValueError("protected tool string argument is invalid") from error
            if "\x00" in normalized or len(normalized) < minimum or len(normalized) > maximum:
                raise ValueError("protected tool string argument is invalid")
            allowed = schema.get("enum")
            if allowed is not None and normalized not in allowed:
                raise ValueError("protected tool string argument is invalid")
            return normalized
        if expected == "object":
            if not isinstance(value, dict):
                raise ValueError("protected tool argument must be an object")
            return cls._validate_arguments(value, schema)
        raise ValueError("protected tool schema uses an unsupported argument type")
