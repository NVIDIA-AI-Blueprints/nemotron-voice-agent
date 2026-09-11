# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Ordered conversation journal and independent Realtime response lifecycles."""

from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass
from typing import Any

from realtime.protocol import (
    RealtimeProtocolError,
    build_server_event,
    immutable_field,
    invalid_type,
    invalid_value,
    new_realtime_id,
)
from realtime.session import CanonicalRealtimeSession

_APPEND = object()
_MCP_ITEM_TYPES = frozenset(
    {
        "mcp_list_tools",
        "mcp_call",
        "mcp_approval_request",
        "mcp_approval_response",
    }
)
_ITEM_TYPES = frozenset({"message", "function_call", "function_call_output", *_MCP_ITEM_TYPES})
_ITEM_TERMINAL_STATUSES = frozenset({"completed", "incomplete"})
_RESPONSE_TERMINAL_STATUSES = frozenset({"completed", "cancelled", "failed", "incomplete"})
_MESSAGE_CONTENT_TYPES = {
    "system": frozenset({"input_text"}),
    "user": frozenset({"input_audio", "input_text"}),
    "assistant": frozenset({"output_audio", "output_text"}),
}

# A Realtime connection is intentionally finite-lived. Keep its canonical
# conversation useful for long sessions while making every retained collection
# and payload byte budget explicit. These are protocol safety bounds, not model
# context heuristics: the model provider remains responsible for token limits.
MAX_CONVERSATION_ITEMS = 4_096
MAX_CONVERSATION_ITEM_BYTES = 1_048_576
MAX_LIVE_CONVERSATION_BYTES = 16_777_216
MAX_JOURNAL_ENTRIES = 4_096
MAX_RESPONSE_RECORDS = 1_024
MAX_ITEM_ID_LENGTH = 512

_COMMON_ITEM_FIELDS = frozenset({"id", "object", "type", "status"})
_MCP_LIST_TOOL_FIELDS = frozenset({"name", "input_schema", "description", "annotations"})


@dataclass(frozen=True, slots=True)
class ConversationJournalSnapshot:
    """Detached journal state used by a surrounding Realtime transaction."""

    conversation_id: str
    order: tuple[str, ...]
    items: dict[str, dict[str, Any]]
    known_item_ids: frozenset[str]
    item_sizes: dict[str, int]
    live_item_bytes: int
    entries: tuple[JournalEntry, ...]
    next_sequence: int


def _wire_item(item: dict[str, Any]) -> dict[str, Any]:
    """Return the exact public item shape for the Realtime wire."""
    wire = copy.deepcopy(item)
    if wire.get("type") in _MCP_ITEM_TYPES:
        # OpenAI's native MCP items carry lifecycle through their dedicated
        # events; unlike message/function items they do not expose these fields.
        wire.pop("object", None)
        wire.pop("status", None)
    return wire


def _serialized_size(value: dict[str, Any]) -> int:
    """Return the deterministic UTF-8 JSON footprint retained by the journal."""
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _journal_item_snapshot(item: dict[str, Any] | None) -> dict[str, Any] | None:
    """Retain lifecycle metadata without duplicating conversation payloads."""
    if item is None:
        return None
    return {
        field_name: copy.deepcopy(item[field_name])
        for field_name in (
            "id",
            "object",
            "type",
            "status",
            "role",
            "call_id",
            "name",
            "server_label",
            "approval_request_id",
        )
        if field_name in item
    }


def _response_history_snapshot(response: dict[str, Any]) -> dict[str, Any]:
    """Retain response lifecycle metadata without copying output bodies."""
    snapshot = copy.deepcopy(response)
    snapshot["output"] = [_journal_item_snapshot(item) for item in response.get("output", []) if isinstance(item, dict)]
    return snapshot


def _require_non_empty_string(value: Any, *, param: str) -> str:
    if not isinstance(value, str):
        raise invalid_type(f"{param} must be a string", param=param)
    if not value:
        raise invalid_value(f"{param} must be a non-empty string", param=param)
    return value


def _validate_mcp_error(value: Any, *, param: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid_type(f"{param} must be an object", param=param)
    normalized = copy.deepcopy(value)
    typ = normalized.get("type")
    if typ == "tool_execution_error":
        allowed_fields = frozenset({"type", "message"})
    elif typ in {"protocol_error", "http_error"}:
        allowed_fields = frozenset({"type", "code", "message"})
    else:
        raise invalid_value(
            f"{param}.type must be protocol_error, tool_execution_error, or http_error",
            param=f"{param}.type",
        )
    unknown = sorted(set(normalized) - allowed_fields)
    if unknown:
        raise RealtimeProtocolError(
            message=f"Unknown parameter: {param}.{unknown[0]}",
            code="unknown_parameter",
            param=f"{param}.{unknown[0]}",
        )
    if not isinstance(normalized.get("message"), str):
        raise invalid_type(f"{param}.message must be a string", param=f"{param}.message")
    if typ in {"protocol_error", "http_error"}:
        code = normalized.get("code")
        if isinstance(code, bool) or not isinstance(code, int):
            raise invalid_type(f"{param}.code must be an integer", param=f"{param}.code")
    return normalized


def _validate_mcp_list_tools_item(normalized: dict[str, Any]) -> None:
    _require_non_empty_string(normalized.get("server_label"), param="item.server_label")
    raw_tools = normalized.get("tools")
    if not isinstance(raw_tools, list):
        raise invalid_type("item.tools must be an array", param="item.tools")
    tools: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, raw_tool in enumerate(raw_tools):
        tool_param = f"item.tools[{index}]"
        if not isinstance(raw_tool, dict):
            raise invalid_type(f"{tool_param} must be an object", param=tool_param)
        tool = copy.deepcopy(raw_tool)
        unknown = sorted(set(tool) - _MCP_LIST_TOOL_FIELDS)
        if unknown:
            raise RealtimeProtocolError(
                message=f"Unknown parameter: {tool_param}.{unknown[0]}",
                code="unknown_parameter",
                param=f"{tool_param}.{unknown[0]}",
            )
        name = _require_non_empty_string(tool.get("name"), param=f"{tool_param}.name")
        if name in names:
            raise invalid_value(f"Duplicate MCP tool name {name!r}", param=f"{tool_param}.name")
        names.add(name)
        if not isinstance(tool.get("input_schema"), dict):
            raise invalid_type(f"{tool_param}.input_schema must be an object", param=f"{tool_param}.input_schema")
        if "description" in tool and tool["description"] is not None and not isinstance(tool["description"], str):
            raise invalid_type(f"{tool_param}.description must be a string or null", param=f"{tool_param}.description")
        if "annotations" in tool and tool["annotations"] is not None and not isinstance(tool["annotations"], dict):
            raise invalid_type(f"{tool_param}.annotations must be an object or null", param=f"{tool_param}.annotations")
        tools.append(tool)
    normalized["tools"] = tools


def _validate_mcp_item(normalized: dict[str, Any], *, typ: str) -> None:
    if typ == "mcp_list_tools":
        _validate_mcp_list_tools_item(normalized)
        return

    _require_non_empty_string(normalized.get("id"), param="item.id")
    if typ == "mcp_approval_response":
        _require_non_empty_string(
            normalized.get("approval_request_id"),
            param="item.approval_request_id",
        )
        if not isinstance(normalized.get("approve"), bool):
            raise invalid_type("item.approve must be a boolean", param="item.approve")
        if "reason" in normalized and normalized["reason"] is not None and not isinstance(normalized["reason"], str):
            raise invalid_type("item.reason must be a string or null", param="item.reason")
        return

    for field_name in ("server_label", "name"):
        _require_non_empty_string(normalized.get(field_name), param=f"item.{field_name}")
    if not isinstance(normalized.get("arguments"), str):
        raise invalid_type(f"{typ}.arguments must be a string", param="item.arguments")
    if normalized.get("status") in _ITEM_TERMINAL_STATUSES and not normalized["arguments"]:
        raise invalid_value(f"terminal {typ}.arguments must not be empty", param="item.arguments")
    if typ == "mcp_approval_request":
        return

    approval_request_id = normalized.get("approval_request_id")
    if approval_request_id is not None:
        _require_non_empty_string(approval_request_id, param="item.approval_request_id")
    output = normalized.get("output")
    if output is not None and not isinstance(output, str):
        raise invalid_type("item.output must be a string or null", param="item.output")
    error = normalized.get("error")
    if error is not None:
        normalized["error"] = _validate_mcp_error(error, param="item.error")


@dataclass(frozen=True, slots=True)
class JournalEntry:
    """One immutable operation in the conversation journal."""

    sequence: int
    operation: str
    item_id: str
    previous_item_id: str | None
    item: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class ResponseRecord:
    """One immutable response lifecycle snapshot."""

    generation: int
    response_id: str
    status: str
    response: dict[str, Any]


def _validate_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise invalid_type("conversation item must be an object", param="item")
    normalized = copy.deepcopy(item)
    typ = normalized.get("type")
    if not isinstance(typ, str):
        raise invalid_type("item.type must be a string", param="item.type")
    if typ not in _ITEM_TYPES:
        raise RealtimeProtocolError(
            message=f"Conversation item type {typ!r} is not available",
            code="unsupported_capability",
            param="item.type",
        )
    allowed_fields = {
        "message": _COMMON_ITEM_FIELDS | {"role", "content"},
        "function_call": _COMMON_ITEM_FIELDS | {"call_id", "name", "arguments"},
        "function_call_output": _COMMON_ITEM_FIELDS | {"call_id", "output"},
        "mcp_list_tools": _COMMON_ITEM_FIELDS | {"server_label", "tools"},
        "mcp_call": _COMMON_ITEM_FIELDS
        | {"arguments", "name", "server_label", "approval_request_id", "error", "output"},
        "mcp_approval_request": _COMMON_ITEM_FIELDS | {"arguments", "name", "server_label"},
        "mcp_approval_response": _COMMON_ITEM_FIELDS | {"approval_request_id", "approve", "reason"},
    }[typ]
    unknown = sorted(set(normalized) - allowed_fields)
    if unknown:
        raise RealtimeProtocolError(
            message=f"Unknown parameter: item.{unknown[0]}",
            code="unknown_parameter",
            param=f"item.{unknown[0]}",
        )
    if "object" in normalized:
        if not isinstance(normalized["object"], str):
            raise invalid_type("item.object must be a string", param="item.object")
        if normalized["object"] != "realtime.item":
            raise invalid_value("item.object must be 'realtime.item'", param="item.object")

    if typ == "message":
        role = normalized.get("role")
        if role not in {"system", "user", "assistant"}:
            raise invalid_value("message role must be system, user, or assistant", param="item.role")
        content = normalized.get("content", [])
        if not isinstance(content, list):
            raise invalid_type("message content must be an array", param="item.content")
        normalized_content: list[dict[str, Any]] = []
        for index, raw_part in enumerate(content):
            part_param = f"item.content[{index}]"
            if not isinstance(raw_part, dict):
                raise invalid_type(f"{part_param} must be an object", param=part_param)
            part = copy.deepcopy(raw_part)
            content_type = part.get("type")
            if content_type not in _MESSAGE_CONTENT_TYPES[role]:
                raise invalid_value(
                    f"Content type {content_type!r} is not valid for role {role!r}",
                    param=f"{part_param}.type",
                )
            allowed_part_fields = {
                "input_text": frozenset({"type", "text"}),
                "input_audio": frozenset({"type", "audio", "transcript"}),
                "output_text": frozenset({"type", "text"}),
                "output_audio": frozenset({"type", "audio", "transcript"}),
            }[content_type]
            unknown_part_fields = sorted(set(part) - allowed_part_fields)
            if unknown_part_fields:
                field_name = unknown_part_fields[0]
                raise RealtimeProtocolError(
                    message=f"Unknown parameter: {part_param}.{field_name}",
                    code="unknown_parameter",
                    param=f"{part_param}.{field_name}",
                )
            text_field = "text" if content_type.endswith("text") else "transcript"
            if text_field in part and part[text_field] is not None and not isinstance(part[text_field], str):
                raise invalid_type(
                    f"{part_param}.{text_field} must be a string",
                    param=f"{part_param}.{text_field}",
                )
            if "audio" in part and not isinstance(part["audio"], str):
                raise invalid_type(f"{part_param}.audio must be a base64 string", param=f"{part_param}.audio")
            normalized_content.append(part)
        normalized["content"] = normalized_content
    elif typ == "function_call":
        for field_name in ("call_id", "name"):
            if not isinstance(normalized.get(field_name), str) or not normalized[field_name]:
                raise invalid_value(
                    f"function_call.{field_name} must be a non-empty string",
                    param=f"item.{field_name}",
                )
        if not isinstance(normalized.get("arguments"), str):
            raise invalid_type("function_call.arguments must be a string", param="item.arguments")
        if normalized.get("status") in _ITEM_TERMINAL_STATUSES and not normalized["arguments"]:
            raise invalid_value(
                "terminal function_call.arguments must not be empty",
                param="item.arguments",
            )
    elif typ == "function_call_output":
        call_id = normalized.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise invalid_value("function_call_output.call_id must be a non-empty string", param="item.call_id")
        if not isinstance(normalized.get("output"), str):
            raise invalid_type("function_call_output.output must be a string", param="item.output")
    else:
        _validate_mcp_item(normalized, typ=typ)

    item_id = normalized.get("id")
    if item_id is not None:
        if not isinstance(item_id, str):
            raise invalid_type("item.id must be a string", param="item.id")
        if not item_id:
            raise invalid_value("item.id must be a non-empty string", param="item.id")
        if len(item_id) > MAX_ITEM_ID_LENGTH:
            raise invalid_value(
                f"item.id must contain at most {MAX_ITEM_ID_LENGTH} characters",
                param="item.id",
            )
    status = normalized.get("status")
    if status is not None:
        if not isinstance(status, str):
            raise invalid_type("item.status must be a string", param="item.status")
        if status not in {"in_progress", "completed", "incomplete"}:
            raise invalid_value(
                "item.status must be in_progress, completed, or incomplete",
                param="item.status",
            )
    normalized.setdefault("object", "realtime.item")
    return normalized


class ConversationJournal:
    """Maintain current item order backed by an append-only operation journal."""

    def __init__(self) -> None:
        """Create an empty default conversation."""
        self.id = new_realtime_id("conv")
        self._order: list[str] = []
        self._items: dict[str, dict[str, Any]] = {}
        self._known_item_ids: set[str] = set()
        self._item_sizes: dict[str, int] = {}
        self._live_item_bytes = 0
        self._entries: deque[JournalEntry] = deque(maxlen=MAX_JOURNAL_ENTRIES)
        self._next_sequence = 1

    @staticmethod
    def validate_item(item: Any) -> dict[str, Any]:
        """Return a detached, canonical item after applying journal validation."""
        return _validate_item(item)

    def created_event(self) -> dict[str, Any]:
        """Return the canonical initial ``conversation.created`` event."""
        return build_server_event(
            "conversation.created",
            conversation={"id": self.id, "object": "realtime.conversation"},
        )

    @property
    def tail_id(self) -> str | None:
        """Return the last live item ID."""
        return self._order[-1] if self._order else None

    def ordered_item_ids(self) -> tuple[str, ...]:
        """Return the current conversation order."""
        return tuple(self._order)

    def entries(self) -> tuple[JournalEntry, ...]:
        """Return the immutable operation history."""
        return tuple(copy.deepcopy(self._entries))

    def snapshot_state(self) -> ConversationJournalSnapshot:
        """Capture every mutable field changed by conversation operations."""
        return ConversationJournalSnapshot(
            conversation_id=self.id,
            order=tuple(self._order),
            items=copy.deepcopy(self._items),
            known_item_ids=frozenset(self._known_item_ids),
            item_sizes=copy.deepcopy(self._item_sizes),
            live_item_bytes=self._live_item_bytes,
            entries=tuple(copy.deepcopy(self._entries)),
            next_sequence=self._next_sequence,
        )

    def restore_state(self, snapshot: ConversationJournalSnapshot) -> None:
        """Restore a trusted snapshot without replacing this journal object."""
        if not isinstance(snapshot, ConversationJournalSnapshot):
            raise TypeError("Conversation rollback requires a journal snapshot")
        if snapshot.conversation_id != self.id:
            raise ValueError("Conversation rollback snapshot belongs to another journal")

        order = list(snapshot.order)
        items = copy.deepcopy(snapshot.items)
        known_item_ids = set(snapshot.known_item_ids)
        item_sizes = copy.deepcopy(snapshot.item_sizes)
        entries = deque(copy.deepcopy(snapshot.entries), maxlen=MAX_JOURNAL_ENTRIES)

        self._order = order
        self._items = items
        self._known_item_ids = known_item_ids
        self._item_sizes = item_sizes
        self._live_item_bytes = snapshot.live_item_bytes
        self._entries = entries
        self._next_sequence = snapshot.next_sequence

    def item(self, item_id: str) -> dict[str, Any]:
        """Return a detached current item snapshot."""
        try:
            return copy.deepcopy(self._items[item_id])
        except KeyError as exc:
            raise RealtimeProtocolError(
                message=f"Conversation item {item_id!r} was not found",
                code="item_not_found",
                param="item_id",
            ) from exc

    def wire_item(self, item_id: str) -> dict[str, Any]:
        """Return a detached item normalized to its public wire schema."""
        return _wire_item(self.item(item_id))

    def add_item(
        self,
        item: dict[str, Any],
        *,
        previous_item_id: str | None | object = _APPEND,
    ) -> dict[str, Any]:
        """Add an item, preserving explicit insertion order and stable IDs."""
        normalized = _validate_item(item)
        item_id = normalized.get("id") or new_realtime_id("item")
        if item_id in self._known_item_ids:
            raise RealtimeProtocolError(
                message=f"Conversation item ID {item_id!r} already exists",
                code="duplicate_item_id",
                param="item.id",
            )
        normalized["id"] = item_id
        if len(self._known_item_ids) >= MAX_CONVERSATION_ITEMS:
            raise RealtimeProtocolError(
                message=("This Realtime conversation reached its item limit; start a new session to continue"),
                code="conversation_limit_exceeded",
                param="item",
            )
        item_size = self._validate_retained_size(normalized)

        if previous_item_id is _APPEND:
            predecessor = self.tail_id
            insert_at = len(self._order)
        elif previous_item_id is None:
            predecessor = None
            insert_at = 0
        elif not isinstance(previous_item_id, str) or not previous_item_id:
            raise invalid_value("previous_item_id must be a non-empty string or null", param="previous_item_id")
        else:
            if previous_item_id not in self._items:
                raise RealtimeProtocolError(
                    message=f"Previous conversation item {previous_item_id!r} was not found",
                    code="invalid_previous_item_id",
                    param="previous_item_id",
                )
            predecessor = previous_item_id
            insert_at = self._order.index(previous_item_id) + 1

        self._items[item_id] = copy.deepcopy(normalized)
        self._known_item_ids.add(item_id)
        self._item_sizes[item_id] = item_size
        self._live_item_bytes += item_size
        self._order.insert(insert_at, item_id)
        self._record("added", item_id, predecessor, normalized)
        return build_server_event(
            "conversation.item.added",
            previous_item_id=predecessor,
            item=_wire_item(normalized),
        )

    def complete_item(
        self,
        item_id: str,
        *,
        status: str = "completed",
        item_patch: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Commit the terminal item snapshot and emit ``conversation.item.done``."""
        if status not in _ITEM_TERMINAL_STATUSES:
            raise invalid_value("terminal item status must be completed or incomplete", param="item.status")
        current = self.item(item_id)
        if current.get("status") in _ITEM_TERMINAL_STATUSES:
            raise RealtimeProtocolError(
                message=f"Conversation item {item_id!r} is already terminal",
                code="item_already_done",
                param="item_id",
            )
        if item_patch is not None:
            if not isinstance(item_patch, dict):
                raise invalid_type("item_patch must be an object", param="item")
            for field_name in (
                "id",
                "object",
                "type",
                "role",
                "call_id",
                "name",
                "server_label",
            ):
                if field_name in item_patch and item_patch[field_name] != current.get(field_name):
                    raise immutable_field(
                        f"item.{field_name} cannot change",
                        param=f"item.{field_name}",
                    )
            if "approval_request_id" in item_patch and current.get("approval_request_id") not in {
                None,
                item_patch["approval_request_id"],
            }:
                raise immutable_field(
                    "item.approval_request_id cannot change",
                    param="item.approval_request_id",
                )
            current.update(copy.deepcopy(item_patch))
        current["status"] = status
        current = _validate_item(current)
        item_size = self._validate_retained_size(current, replacing_item_id=item_id)
        self._items[item_id] = copy.deepcopy(current)
        self._replace_item_size(item_id, item_size)
        predecessor = self._predecessor(item_id)
        self._record("done", item_id, predecessor, current)
        return build_server_event(
            "conversation.item.done",
            previous_item_id=predecessor,
            item=_wire_item(current),
        )

    def terminal_item_done_event(self, item_id: str) -> dict[str, Any]:
        """Record ``done`` for an item that was atomically added as terminal.

        Client-created text messages and function outputs arrive as complete
        items rather than streamed model output. OpenAI therefore publishes
        their ``conversation.item.added`` snapshot with a terminal status and
        follows it with the matching ``conversation.item.done`` lifecycle
        event. Keep that atomic path separate from :meth:`complete_item`, whose
        job is to transition an in-progress item exactly once.
        """
        current = self.item(item_id)
        if current.get("status") not in _ITEM_TERMINAL_STATUSES:
            raise RealtimeProtocolError(
                message=f"Conversation item {item_id!r} is not terminal",
                code="item_not_done",
                param="item_id",
            )
        predecessor = self._predecessor(item_id)
        self._record("done", item_id, predecessor, current)
        return build_server_event(
            "conversation.item.done",
            previous_item_id=predecessor,
            item=_wire_item(current),
        )

    def patch_item(self, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update a live snapshot without changing order or terminal status.

        This is used for asynchronous metadata such as an input-audio
        transcript that can arrive after the item itself has become terminal.
        The mutation is retained in the append-only journal and intentionally
        has no standalone wire event; the protocol-specific transcript event
        carries the update to the client.
        """
        if not isinstance(patch, dict):
            raise invalid_type("item patch must be an object", param="item")
        current = self.item(item_id)
        for field_name in (
            "id",
            "object",
            "type",
            "role",
            "call_id",
            "name",
            "server_label",
            "status",
        ):
            if field_name in patch and patch[field_name] != current.get(field_name):
                raise immutable_field(
                    f"item.{field_name} cannot change",
                    param=f"item.{field_name}",
                )
        if "approval_request_id" in patch and current.get("approval_request_id") not in {
            None,
            patch["approval_request_id"],
        }:
            raise immutable_field(
                "item.approval_request_id cannot change",
                param="item.approval_request_id",
            )
        current.update(copy.deepcopy(patch))
        current = _validate_item(current)
        item_size = self._validate_retained_size(current, replacing_item_id=item_id)
        self._items[item_id] = copy.deepcopy(current)
        self._replace_item_size(item_id, item_size)
        self._record("updated", item_id, self._predecessor(item_id), current)
        return copy.deepcopy(current)

    def delete_item(self, item_id: str) -> dict[str, Any]:
        """Remove a live item while retaining its immutable journal history."""
        current = self.item(item_id)
        predecessor = self._predecessor(item_id)
        self._order.remove(item_id)
        del self._items[item_id]
        self._live_item_bytes -= self._item_sizes.pop(item_id)
        self._record("deleted", item_id, predecessor, current)
        return build_server_event("conversation.item.deleted", item_id=item_id)

    def _validate_retained_size(
        self,
        item: dict[str, Any],
        *,
        replacing_item_id: str | None = None,
    ) -> int:
        """Validate one canonical item and the aggregate live-byte budget."""
        try:
            item_size = _serialized_size(item)
        except (TypeError, ValueError) as exc:
            raise invalid_value("conversation item must contain JSON-compatible values", param="item") from exc
        if item_size > MAX_CONVERSATION_ITEM_BYTES:
            raise RealtimeProtocolError(
                message=(f"Conversation item exceeds the maximum retained size of {MAX_CONVERSATION_ITEM_BYTES} bytes"),
                code="conversation_item_too_large",
                param="item",
            )
        replaced_size = self._item_sizes.get(replacing_item_id, 0) if replacing_item_id is not None else 0
        next_live_bytes = self._live_item_bytes - replaced_size + item_size
        if next_live_bytes > MAX_LIVE_CONVERSATION_BYTES:
            raise RealtimeProtocolError(
                message=(
                    "This Realtime conversation reached its retained payload limit; delete items or start a new session"
                ),
                code="conversation_limit_exceeded",
                param="item",
            )
        return item_size

    def _replace_item_size(self, item_id: str, item_size: int) -> None:
        """Commit one already-validated item-size replacement."""
        previous_size = self._item_sizes[item_id]
        self._item_sizes[item_id] = item_size
        self._live_item_bytes += item_size - previous_size

    def _predecessor(self, item_id: str) -> str | None:
        index = self._order.index(item_id)
        return self._order[index - 1] if index else None

    def _record(
        self,
        operation: str,
        item_id: str,
        previous_item_id: str | None,
        item: dict[str, Any] | None,
    ) -> None:
        self._entries.append(
            JournalEntry(
                sequence=self._next_sequence,
                operation=operation,
                item_id=item_id,
                previous_item_id=previous_item_id,
                item=_journal_item_snapshot(item),
            )
        )
        self._next_sequence += 1


class ResponseLifecycleLedger:
    """Coordinate distinct, serialized response lifecycles for one session."""

    def __init__(self, *, session: CanonicalRealtimeSession, conversation: ConversationJournal) -> None:
        """Bind response state to one session and its default conversation."""
        self._session = session
        self._conversation = conversation
        self._active: dict[str, Any] | None = None
        self._active_output_ids: list[str] = []
        self._records: deque[ResponseRecord] = deque(maxlen=MAX_RESPONSE_RECORDS)
        self._generation = 0

    @property
    def active_response_id(self) -> str | None:
        """Return the current response ID, if any."""
        return self._active["id"] if self._active is not None else None

    def records(self) -> tuple[ResponseRecord, ...]:
        """Return immutable created/terminal response snapshots."""
        return tuple(copy.deepcopy(self._records))

    @staticmethod
    def validate_metadata(metadata: Any) -> dict[str, str] | None:
        """Validate and freeze response metadata before any external work."""
        if metadata is None:
            return None
        if not isinstance(metadata, dict):
            raise invalid_type("response.metadata must be an object or null", param="response.metadata")
        if len(metadata) > 16:
            raise invalid_value("response.metadata supports at most 16 entries", param="response.metadata")
        normalized: dict[str, str] = {}
        for key, value in metadata.items():
            if not isinstance(key, str) or len(key) > 64:
                raise invalid_value(
                    "response.metadata keys must be strings of at most 64 characters",
                    param="response.metadata",
                )
            if not isinstance(value, str) or len(value) > 512:
                raise invalid_value(
                    "response.metadata values must be strings of at most 512 characters",
                    param=f"response.metadata.{key}",
                )
            normalized[key] = value
        return normalized

    def start(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        instructions: str | None = None,
        max_output_tokens: int | str | None = None,
        parallel_tool_calls: bool | None = None,
        output_modalities: list[str] | None = None,
        audio_output: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a response with an optional detached inference snapshot.

        ``instructions`` and ``parallel_tool_calls`` affect inference but are
        not fields on the Realtime Response resource, so they are deliberately
        absent from the emitted ``response.created`` and ``response.done`` bodies.
        """
        metadata = self.validate_metadata(metadata)
        response_id = new_realtime_id("resp")
        self._session.begin_response(response_id)
        self._generation += 1
        response_defaults = self._session.response_defaults()
        response_defaults.pop("instructions", None)
        response_defaults.pop("parallel_tool_calls", None)
        if max_output_tokens is not None:
            response_defaults["max_output_tokens"] = max_output_tokens
        if output_modalities is not None:
            response_defaults["output_modalities"] = copy.deepcopy(output_modalities)
        if response_defaults["output_modalities"] == ["audio"]:
            effective_audio = audio_output
            if effective_audio is None:
                effective_audio = self._session.public_view()["audio"]["output"]
            response_defaults["audio"] = {"output": copy.deepcopy(effective_audio)}
        else:
            response_defaults.pop("audio", None)
        response: dict[str, Any] = {
            "id": response_id,
            "object": "realtime.response",
            "status": "in_progress",
            "status_details": None,
            "output": [],
            "conversation_id": self._conversation.id,
            **response_defaults,
            "usage": None,
            "metadata": copy.deepcopy(metadata),
        }
        self._active = response
        self._active_output_ids = []
        self._records.append(
            ResponseRecord(
                generation=self._generation,
                response_id=response_id,
                status="in_progress",
                response=_response_history_snapshot(response),
            )
        )
        return build_server_event("response.created", response=copy.deepcopy(response))

    def add_output_item(self, item_id: str) -> None:
        """Associate a conversation output item with the active response once."""
        self._require_active()
        self._conversation.item(item_id)
        if item_id in self._active_output_ids:
            raise RealtimeProtocolError(
                message=f"Output item {item_id!r} is already attached to the active response",
                code="duplicate_output_item",
                param="item_id",
            )
        self._active_output_ids.append(item_id)

    def mark_output_audio_started(self, *, voice: str | dict[str, str] | None = None) -> None:
        """Lock the session voice when the active response emits its first audio."""
        response = self._require_active()
        self._session.mark_output_audio_started(response["id"], voice=voice)

    def finish(
        self,
        *,
        status: str,
        status_details: dict[str, Any] | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish the active response exactly once and emit ``response.done``."""
        if status not in _RESPONSE_TERMINAL_STATUSES:
            raise invalid_value(
                "response status must be completed, cancelled, failed, or incomplete",
                param="response.status",
            )
        if status == "completed" and status_details is not None:
            raise invalid_value(
                "completed responses must not have status_details",
                param="response.status_details",
            )
        if status != "completed" and not isinstance(status_details, dict):
            raise invalid_type(
                "non-completed responses require status_details",
                param="response.status_details",
            )
        if usage is not None and not isinstance(usage, dict):
            raise invalid_type("response.usage must be an object or null", param="response.usage")

        active = self._require_active()
        response_id = active["id"]
        terminal = copy.deepcopy(active)
        internal_output = [self._conversation.item(item_id) for item_id in self._active_output_ids]
        unfinished = [
            item["id"]
            for item in internal_output
            if item.get("status") == "in_progress" and item.get("type") != "mcp_call"
        ]
        if unfinished:
            raise RealtimeProtocolError(
                message=f"Output item {unfinished[0]!r} is still in progress",
                code="output_item_in_progress",
                param="response.output",
            )
        terminal["status"] = status
        terminal["status_details"] = copy.deepcopy(status_details)
        terminal["output"] = [_wire_item(item) for item in internal_output]
        terminal["usage"] = copy.deepcopy(usage)
        self._session.finish_response(response_id)
        self._records.append(
            ResponseRecord(
                generation=self._generation,
                response_id=response_id,
                status=status,
                response=_response_history_snapshot(terminal),
            )
        )
        self._active = None
        self._active_output_ids = []
        return build_server_event("response.done", response=terminal)

    def _require_active(self) -> dict[str, Any]:
        if self._active is None:
            raise RealtimeProtocolError(
                message="No response is currently in progress",
                code="response_not_found",
                param="response",
            )
        return self._active
