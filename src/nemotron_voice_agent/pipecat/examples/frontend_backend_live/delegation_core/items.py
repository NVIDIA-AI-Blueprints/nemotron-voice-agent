# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Responses API's items: whether one is a message, a message's text, and the input item a request sends of it."""

from __future__ import annotations


def is_message(item: dict) -> bool:
    """Return whether an input item is a message (a typed one may leave its ``type`` out)."""
    return item.get("type", "message") == "message"


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
