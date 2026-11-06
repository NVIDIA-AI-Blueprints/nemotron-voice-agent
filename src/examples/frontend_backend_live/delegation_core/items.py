# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Responses API's message items: their text."""

from __future__ import annotations


def message_text(item: dict) -> str:
    """The text of a message item: its ``output_text`` parts, joined (a plain string content is taken whole)."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") == "output_text"
    )
