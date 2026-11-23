# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Responses API's items: whether one is a message, a message's text, and the input item a request sends of it."""

from __future__ import annotations

import json

from openai.types.responses import ResponseInputFileContentParam, ResponseInputImageContentParam
from pydantic import TypeAdapter, ValidationError

TEXT_PART_TYPES = frozenset({"input_text", "output_text", "text"})
_MEDIA_PARTS = {
    "input_image": (TypeAdapter(ResponseInputImageContentParam), ("file_id", "image_url")),
    "input_file": (TypeAdapter(ResponseInputFileContentParam), ("file_data", "file_id", "file_url")),
}
_OUTPUT_PART_TYPES = TEXT_PART_TYPES | frozenset(_MEDIA_PARTS)


class WorkerData(dict):
    """A Core-authored worker data message with the usual wire shape and internal provenance."""

    def __init__(self, item: dict, *, starts_request: bool = True, **fields: object) -> None:
        """Keep context-window metadata alongside the wire item."""
        super().__init__(item, **fields)
        self.starts_request = starts_request


def is_message(item: dict) -> bool:
    """Return whether an input item is a message (a typed one may leave its ``type`` out)."""
    return item.get("type", "message") == "message"


def is_output_content(output: object) -> bool:
    """Recognize an array of content parts. Other arrays carry JSON data."""
    return (
        isinstance(output, list)
        and bool(output)
        and all(
            isinstance(part, dict) and isinstance(kind := part.get("type"), str) and kind in _OUTPUT_PART_TYPES
            for part in output
        )
    )


def validate_output_content(output: str | list) -> None:
    """Check recognized content parts before accepting a tool result."""
    if not is_output_content(output):
        return
    for part in output:
        kind = part["type"]
        if kind in TEXT_PART_TYPES:
            if not isinstance(part.get("text"), str):
                raise ValueError(f"The {kind} content must have string text")
            continue
        adapter, references = _MEDIA_PARTS[kind]
        try:
            adapter.validate_python(part, strict=True)
        except ValidationError:
            raise ValueError(f"The {kind} content has invalid fields") from None
        if not any(part.get(ref) and part[ref].strip() for ref in references):
            raise ValueError(f"The {kind} content needs a file or URL reference")


def message_text(item: dict) -> str:
    """Return the text of a message item: its ``output_text`` parts joined, or a plain string content whole."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") == "output_text"
    )


def input_item(item: dict) -> dict:
    """Return a history item as a Responses input item.

    A message other than an output message (one with a status) drops its ``id``.
    """
    if is_message(item) and not (item.get("role") == "assistant" and "status" in item):
        return {key: value for key, value in item.items() if key != "id"}
    return item


def call_arguments(call: dict, *, preserve_invalid: bool = False) -> str | None:
    """Return sorted JSON arguments, optionally distinguishing invalid input for identity binding."""
    raw = call.get("arguments") or "{}"
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return json.dumps(("raw", raw), sort_keys=True) if preserve_invalid else None
    return json.dumps(("json", parsed) if preserve_invalid else parsed, sort_keys=True)
