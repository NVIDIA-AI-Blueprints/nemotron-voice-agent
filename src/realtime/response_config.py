# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Validation and model-context projection for ``response.create`` overrides."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from realtime.conversation import ConversationJournal
from realtime.protocol import RealtimeProtocolError, invalid_type, invalid_value, unsupported_capability

_INLINE_CONTEXT_ITEM_TYPES = frozenset({"message", "function_call", "function_call_output"})
_REFERENCE_FIELDS = frozenset({"id", "type"})


@dataclass(frozen=True, slots=True)
class RealtimeResponseInput:
    """A detached custom response context retaining item references until use."""

    items: tuple[dict[str, Any], ...]


def validate_response_conversation(value: Any) -> str:
    """Validate the default-conversation selector supported by this gateway."""
    if not isinstance(value, str):
        raise invalid_type("response.conversation must be a string", param="response.conversation")
    if value == "auto":
        return value
    if value == "none":
        raise unsupported_capability(
            "Out-of-band responses require isolated response storage and are not available on this cascaded pipeline",
            param="response.conversation",
        )
    raise unsupported_capability(
        "This Realtime connection supports only its default conversation using response.conversation='auto'",
        param="response.conversation",
    )


def validate_response_output_modalities(value: Any) -> list[str]:
    """Validate the canonical one-modality Realtime response contract."""
    if not isinstance(value, list):
        raise invalid_type("response.output_modalities must be an array", param="response.output_modalities")
    if value not in (["audio"], ["text"]):
        raise invalid_value(
            "response.output_modalities must be exactly ['audio'] or ['text']",
            param="response.output_modalities",
        )
    return copy.deepcopy(value)


def validate_response_input(value: Any) -> RealtimeResponseInput:
    """Validate custom response context without mutating the default conversation."""
    if not isinstance(value, list):
        raise invalid_type("response.input must be an array", param="response.input")

    normalized: list[dict[str, Any]] = []
    for index, raw_item in enumerate(value):
        param = f"response.input[{index}]"
        if not isinstance(raw_item, dict):
            raise invalid_type(f"{param} must be an object", param=param)
        item = copy.deepcopy(raw_item)
        if item.get("type") == "item_reference":
            unknown = sorted(set(item) - _REFERENCE_FIELDS)
            if unknown:
                field = unknown[0]
                raise RealtimeProtocolError(
                    message=f"Unknown parameter: {param}.{field}",
                    code="unknown_parameter",
                    param=f"{param}.{field}",
                )
            item_id = item.get("id")
            if not isinstance(item_id, str):
                raise invalid_type(f"{param}.id must be a string", param=f"{param}.id")
            if not item_id:
                raise invalid_value(f"{param}.id must be a non-empty string", param=f"{param}.id")
            normalized.append(item)
            continue

        typ = item.get("type")
        if typ not in _INLINE_CONTEXT_ITEM_TYPES:
            raise unsupported_capability(
                (f"Custom response context item type {typ!r} cannot be represented by the cascaded text-model context"),
                param=f"{param}.type",
            )
        try:
            canonical = ConversationJournal.validate_item(item)
        except RealtimeProtocolError as exc:
            raise _rebase_item_error(exc, param=param) from exc
        if typ == "message":
            _validate_inline_text_message(canonical, param=param)
        normalized.append(canonical)

    return RealtimeResponseInput(items=tuple(normalized))


def project_response_input_messages(
    response_input: RealtimeResponseInput,
    *,
    resolve_item: Callable[[str], dict[str, Any]],
    resolve_context_message: Callable[[str], dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Project custom items into one isolated OpenAI-compatible LLM context.

    Item references prefer the exact object already owned by Pipecat's canonical
    context. A journal projection is used only for item kinds whose context can
    be reproduced losslessly from their public Realtime representation. Audio
    transcripts are deliberately not used as substitutes for audio input.
    """
    messages: list[dict[str, Any]] = []
    for index, item in enumerate(response_input.items):
        param = f"response.input[{index}]"
        if item.get("type") == "item_reference":
            item_id = item["id"]
            if resolve_context_message is not None:
                exact = resolve_context_message(item_id)
                if exact is not None:
                    messages.append(copy.deepcopy(exact))
                    continue
            try:
                referenced = resolve_item(item_id)
            except RealtimeProtocolError as exc:
                if exc.code == "item_not_found":
                    raise RealtimeProtocolError(
                        message=f"Referenced conversation item {item_id!r} was not found",
                        code="item_not_found",
                        param=f"{param}.id",
                    ) from exc
                raise
            messages.append(_item_to_context_message(referenced, param=param))
            continue
        messages.append(_item_to_context_message(item, param=param))
    return messages


def _validate_inline_text_message(item: dict[str, Any], *, param: str) -> None:
    content = item.get("content")
    if not isinstance(content, list):
        # The journal validator will provide the canonical type error.
        return
    for index, part in enumerate(content):
        part_param = f"{param}.content[{index}]"
        if isinstance(part, dict) and part.get("type") in {"input_audio", "output_audio"}:
            raise unsupported_capability(
                "Custom response input supports text messages and function-call context on this cascaded pipeline",
                param=f"{part_param}.type",
            )


def _item_to_context_message(item: dict[str, Any], *, param: str) -> dict[str, Any]:
    typ = item.get("type")
    if typ == "message":
        role = item.get("role")
        if role not in {"system", "user", "assistant"}:
            raise invalid_value(f"{param}.role is invalid", param=f"{param}.role")
        text = _message_text(item, param=param)
        return {"role": role, "content": text}
    if typ == "function_call":
        call_id = item.get("call_id")
        name = item.get("name")
        arguments = item.get("arguments")
        if not all(isinstance(value, str) and value for value in (call_id, name)) or not isinstance(arguments, str):
            raise unsupported_capability(
                "Referenced function calls must retain call_id, name, and string arguments",
                param=param,
            )
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    if typ == "function_call_output":
        call_id = item.get("call_id")
        output = item.get("output")
        if not isinstance(call_id, str) or not call_id or not isinstance(output, str):
            raise unsupported_capability(
                "Referenced function outputs must retain call_id and string output",
                param=param,
            )
        return {"role": "tool", "tool_call_id": call_id, "content": output}
    raise unsupported_capability(
        f"Referenced item type {typ!r} cannot be represented by the cascaded text-model context",
        param=param,
    )


def _message_text(item: dict[str, Any], *, param: str) -> str:
    content = item.get("content")
    if not isinstance(content, list) or not content:
        raise invalid_value(f"{param}.content must contain text", param=f"{param}.content")
    parts: list[str] = []
    for index, part in enumerate(content):
        part_param = f"{param}.content[{index}]"
        if not isinstance(part, dict):
            raise invalid_type(f"{part_param} must be an object", param=part_param)
        typ = part.get("type")
        text = part.get("text")
        if typ not in {"input_text", "output_text"} or not isinstance(text, str):
            raise unsupported_capability(
                "Referenced audio requires its exact model context and cannot be replaced by a transcript",
                param=part_param,
            )
        parts.append(text)
    return "".join(parts)


def _rebase_item_error(exc: RealtimeProtocolError, *, param: str) -> RealtimeProtocolError:
    source_param = exc.param
    if source_param == "item":
        target_param = param
    elif isinstance(source_param, str) and source_param.startswith("item."):
        target_param = f"{param}.{source_param.removeprefix('item.')}"
    else:
        target_param = source_param
    message = exc.message
    if isinstance(source_param, str) and source_param in message and target_param is not None:
        message = message.replace(source_param, target_param)
    return RealtimeProtocolError(
        message=message,
        code=exc.code,
        param=target_param,
        error_type=exc.error_type,
    )
