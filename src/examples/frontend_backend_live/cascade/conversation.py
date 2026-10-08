# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A small view of the conversation the session reads and writes, independent of Pipecat's context type."""

from __future__ import annotations

from typing import Protocol

from examples.frontend_backend_live.common.history import BACKEND_RESULT_PREFIX, CLIENT_CONTEXT_PREFIX


class Conversation(Protocol):
    """The shared message list. Pipecat's ``LLMContext`` satisfies this through ``ContextConversation``."""

    def messages(self) -> list[dict]:
        """Return the messages in order (system, user and assistant roles)."""

    def add(self, message: dict) -> None:
        """Append a message."""

    def trim(self, limit: int) -> None:
        """Keep the leading system messages and only the latest ``limit`` others."""


def trim_messages(messages: list[dict], preserve: int, limit: int) -> list[dict]:
    """Keep the first ``preserve`` messages (the prompt) and the latest ``limit`` of the rest."""
    if limit < 1 or len(messages) <= preserve + limit:
        return messages
    return messages[:preserve] + messages[preserve:][-limit:]


class ContextConversation:
    """Adapts a Pipecat ``LLMContext`` to :class:`Conversation`."""

    def __init__(self, context):
        """Wrap ``context``."""
        self.context = context

    def messages(self) -> list[dict]:
        """Return the context's messages."""
        return list(self.context.get_messages())

    def add(self, message: dict) -> None:
        """Append ``message`` to the context."""
        self.context.add_message(message)

    def trim(self, limit: int) -> None:
        """Keep the leading system messages and the latest ``limit`` others, so a long call stays bounded."""
        messages = self.context.get_messages()
        preserve = next(
            (i for i, m in enumerate(messages) if m.get("role") not in ("system", "developer")), len(messages)
        )
        trimmed = trim_messages(messages, preserve, limit)
        if trimmed is not messages:
            self.context.set_messages(trimmed)


def message_text(message: dict) -> str:
    """Return a message's text, joining content parts when the content is a list."""
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))


def is_reference_item(message: dict) -> bool:
    """Return True for a user-role item that carries a backend result or client context, not caller speech."""
    text = message_text(message)
    return message.get("role") == "user" and text.startswith((BACKEND_RESULT_PREFIX, CLIENT_CONTEXT_PREFIX))


def split_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Split a context into the talker's instructions and its plain-text voice history.

    Leading system messages are the instructions; later system or developer messages join them as context,
    exactly as the frontend does. Everything else is history with string content.
    """
    instructions: list[str] = []
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        text = message_text(message)
        if role in ("system", "developer"):
            instructions.append(text)
        elif role in ("user", "assistant"):
            history.append({"role": role, "content": text})
    return "\n\n".join(instructions), history


def last_caller_text(messages: list[dict]) -> str:
    """Return the newest user message that is caller speech, skipping reference items."""
    for message in reversed(messages):
        if message.get("role") == "user" and not is_reference_item(message):
            return message_text(message)
    return ""


def last_assistant_text(messages: list[dict]) -> str | None:
    """Return the newest assistant message, or None before the assistant has spoken."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message_text(message)
    return None
