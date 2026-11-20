# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a LiveKit pipeline error into a short, readable summary for the UI. Standard library only."""

from __future__ import annotations

from typing import Any

MAX_MESSAGE_CHARS = 300
ELLIPSIS = " … "


def shorten(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Cut `text` to `limit` characters in the middle. The start names the error and the end holds its cause."""
    if len(text) <= limit:
        return text
    head = (limit - len(ELLIPSIS)) // 3
    tail = limit - len(ELLIPSIS) - head
    return f"{text[:head]}{ELLIPSIS}{text[-tail:]}"


def describe_error(error: object) -> dict[str, Any]:
    """Return `{"message", "status", "recoverable"}` for an LLM, ASR, or TTS error.

    LiveKit wraps the underlying API error and marks it recoverable when it retries the call, for example after a
    rate limit (HTTP 429) or a refused connection. The UI shows a recoverable error as a note instead of a failure.
    """
    inner = getattr(error, "error", error)
    status = getattr(inner, "status_code", None)
    text = " ".join(str(inner).split())
    return {
        "message": shorten(f"{type(inner).__name__}: {text}"),
        "status": status if isinstance(status, int) else None,
        "recoverable": bool(getattr(error, "recoverable", False)),
    }
