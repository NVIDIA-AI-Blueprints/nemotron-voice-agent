# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a LiveKit pipeline error into a short, readable summary for the UI. Standard library only."""

from __future__ import annotations

from typing import Any


def describe_error(error: object) -> dict[str, Any]:
    """Return `{"message", "status", "recoverable"}` for an LLM, ASR, or TTS error.

    LiveKit wraps the underlying API error and marks it recoverable when it retries the call. A recoverable error is
    usually a short hiccup, such as a rate limit (HTTP 429), so the UI shows it as a quiet note instead of a failure.
    """
    inner = getattr(error, "error", error)
    status = getattr(inner, "status_code", None)
    text = " ".join(str(inner).split())
    return {
        "message": f"{type(inner).__name__}: {text}"[:200],
        "status": status if isinstance(status, int) else None,
        "recoverable": bool(getattr(error, "recoverable", False)),
    }
