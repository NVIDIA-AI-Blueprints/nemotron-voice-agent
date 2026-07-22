# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Opt-in debugging snapshot of the Speaker's LLM context.

Off by default; set ``OMNI_CONVERSATION_LOG_ENABLED=1`` to write a plain-text
snapshot of the faithful Speaker context plus per-turn detail (timestamps and each
turn's full raw output) to ``OMNI_CONVERSATION_LOG`` (default ``conversation.log``).
Set ``OMNI_CONVERSATION_LOG_COLOR=1`` to dim the detail with ANSI grey.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

_ENABLED = os.environ.get("OMNI_CONVERSATION_LOG_ENABLED", "") == "1"
_PATH = Path(os.environ.get("OMNI_CONVERSATION_LOG", "conversation.log"))
_lock = threading.Lock()
_USE_COLOR = os.environ.get("OMNI_CONVERSATION_LOG_COLOR", "") == "1"
_GREY = "\033[90m"
_RESET = "\033[0m"

_session_id = "-"
_session_started = ""
_user_details: dict[str, list[dict]] = {}
_assistant_details: dict[str, list[dict]] = {}


def log_session_start(session_id: str | None = None) -> None:
    """Begin a FRESH snapshot for a new session (clears the previous one)."""
    if not _ENABLED:
        return
    global _session_id, _session_started, _user_details, _assistant_details
    _session_id = session_id or "-"
    _session_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _user_details = {}
    _assistant_details = {}
    log_context([])


def record_user_turn(content: str) -> None:
    """Record a user turn's timestamp, keyed by its transcript."""
    if not _ENABLED:
        return
    _user_details.setdefault(_key(content), []).append({"ts": datetime.now().strftime("%H:%M:%S")})


def record_assistant_turn(content: str, full: str = "", source: str = "") -> None:
    """Record an assistant turn's timestamp and full raw output, keyed by its spoken text.

    ``source`` names the producer when it is not the Speaker (e.g. ``"Thinker"``),
    so the greyed detail block can attribute the turn.
    """
    if not _ENABLED:
        return
    _assistant_details.setdefault(_key(content), []).append(
        {"ts": datetime.now().strftime("%H:%M:%S"), "full": (full or "").strip(), "source": source.strip()}
    )


def log_context(messages: list) -> None:
    """Rewrite the log: faithful context (white) + greyed per-turn detail (content-matched)."""
    if not _ENABLED:
        return
    parts = [f"===== session {_session_id} @ {_session_started} (snapshot {datetime.now():%H:%M:%S}) =====\n"]
    seen: dict[tuple[str, str], int] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "?")
        content = _format_content(message.get("content"))
        if role in ("user", "assistant"):
            store = _user_details if role == "user" else _assistant_details
            key = _key(content)
            occurrence = seen.get((role, key), 0)
            seen[(role, key)] = occurrence + 1
            metas = store.get(key, [])
            meta = metas[occurrence] if occurrence < len(metas) else None
            parts.append(f"\n{_stamp(meta)}[{role}]\n{content}\n")
            if role == "assistant":
                parts.append(_full_detail(meta, spoken=content))
        else:
            parts.append(f"\n[{role}]\n{content}\n")
    _write("".join(parts))


def _key(text: str) -> str:
    """Normalize spoken text (collapse whitespace) for robust content matching."""
    return " ".join((text or "").split())


def _stamp(meta: dict | None) -> str:
    """Greyed ``[HH:MM:SS] `` prefix for a turn (empty when unknown)."""
    return _dim(f"[{meta['ts']}] ") if meta and meta.get("ts") else ""


def _full_detail(meta: dict | None, *, spoken: str) -> str:
    """Greyed block with the turn's full raw output, when it adds information."""
    full = (meta or {}).get("full", "")
    if not full:
        return ""
    rendered = _format_content(full)
    if not rendered.strip() or rendered.strip() == spoken.strip():
        return ""
    source = (meta or {}).get("source", "")
    label = (
        f"{source} full response (filtered out of the spoken line):"
        if source
        else "full response (filtered out of the spoken line):"
    )
    return _dim(f"{label}\n{rendered}") + "\n"


def _dim(text: str) -> str:
    """Wrap text in ANSI dim/grey when colour is enabled."""
    return f"{_GREY}{text}{_RESET}" if _USE_COLOR else text


def _format_content(content: object) -> str:
    """Render content: pretty-print JSON, summarize multimodal parts."""
    if isinstance(content, str):
        raw = content.strip()
        try:
            return json.dumps(json.loads(raw), indent=2, ensure_ascii=False)
        except (ValueError, TypeError):
            return raw
    if isinstance(content, list):
        rendered: list[str] = []
        for part in content:
            if isinstance(part, dict):
                kind = str(part.get("type") or "part")
                rendered.append(str(part.get("text", "")) if kind == "text" else f"<{kind}>")
            else:
                rendered.append(str(part))
        return "\n".join(rendered)
    return str(content)


def _write(text: str) -> None:
    """Rewrite the log file with the current snapshot, best-effort."""
    try:
        with _lock:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(text)
    except Exception:
        pass
