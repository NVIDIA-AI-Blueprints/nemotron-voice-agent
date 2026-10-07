# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What of the conversation a paired backend sees (``backend.conversation_history``).

With the flag off, a paired backend gets the frontend's query and nothing else,
exactly as before. With it on, the per-turn request may also carry the user's
own words, and the backend history may persist across delegations; see
``misc/prototypes/frontend-backend-agent-backend-history-plan.md``.

The earlier turns come from the *frontend* history: it already holds every
exchange that crossed the agent boundary (a seeded greeting, direct answers,
and answers repaired to what the user heard), so no extra session state is
needed. Everything here is a pure function over immutable types.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_verdict.text.config import (
    HISTORY_BACKEND_TURNS,
    HISTORY_TRANSCRIPT,
    Config,
)
from examples.frontend_backend_verdict.text.delegation import CALL_BACKEND
from examples.frontend_backend_verdict.text.errors import ConfigError
from examples.frontend_backend_verdict.text.history import History
from examples.frontend_backend_verdict.text.messages import Message
from examples.frontend_backend_verdict.text.prompts import PromptCatalog, render

EARLIER_TURNS = "{earlier_turns}"
USER_MESSAGE = "{user_message}"
QUERY = "{query}"

HEADING_SINCE_LAST_REPLY = "Conversation since your last reply (oldest first):"
HEADING_SO_FAR = "Conversation so far (oldest first):"

#: Closes a backend group whose tool calls were discarded before their results arrived.
DISCARDED_RESULT_TEXT = "No result: the user spoke before this call finished; its outcome is unknown."
DISCARDED_TURN_TEXT = "[This turn was interrupted before it finished.]"

_SPEAKER = {"user": "User", "assistant": "Assistant"}


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """The first backend input of one delegated turn, and where it came from."""

    text: str
    history: History
    earlier_turns: int


def _is_spoken(message: Message) -> bool:
    return message.role in _SPEAKER and not message.tool_calls and message.content is not None


def _delegates(group: Sequence[Message]) -> bool:
    return any(call.name == CALL_BACKEND for message in group for call in message.tool_calls)


def conversation_turns(frontend_history: History) -> tuple[Message, ...]:
    """Every user text and spoken assistant text, oldest first (delegation plumbing skipped)."""
    return tuple(message for message in frontend_history.messages if _is_spoken(message))


def undelegated_turns(frontend_history: History) -> tuple[Message, ...]:
    """The spoken turns after the last delegated group: what the backend has not seen yet."""
    groups = frontend_history.groups()
    start = 0
    for index, group in enumerate(groups):
        if _delegates(group):
            start = index + 1
    return tuple(message for group in groups[start:] for message in group if _is_spoken(message))


def render_request(template: str, *, query: str, user_message: str, turns: Sequence[Message], heading: str) -> str:
    """Fill the request template; the earlier-turns section is omitted when there are none.

    Substitution is literal, like :func:`~examples.frontend_backend_verdict.text.prompts.render`.
    """
    section = ""
    if turns:
        lines = [heading, *(f"{_SPEAKER[message.role]}: {message.content}" for message in turns)]
        section = "\n".join(lines) + "\n\n"
    rendered = template.strip().replace(EARLIER_TURNS, section)
    return rendered.replace(USER_MESSAGE, user_message.strip()).replace(QUERY, query.strip()).strip()


def build_request(
    config: Config,
    *,
    template: str,
    query: str,
    user_message: str,
    frontend_history: History,
    backend_history: History,
) -> BackendRequest:
    """Build the first backend input of a delegated turn for the configured ``include``."""
    history_cfg = config.backend.conversation_history
    if not history_cfg.enabled:
        return BackendRequest(text=query, history=History(), earlier_turns=0)
    history = backend_history if history_cfg.keeps_backend_turns else History()
    if history_cfg.include == HISTORY_BACKEND_TURNS:
        return BackendRequest(text=query, history=history, earlier_turns=0)
    if history_cfg.include == HISTORY_TRANSCRIPT:
        turns, heading = conversation_turns(frontend_history), HEADING_SO_FAR
    else:
        turns, heading = undelegated_turns(frontend_history), HEADING_SINCE_LAST_REPLY
    text = render_request(template, query=query, user_message=user_message, turns=turns, heading=heading)
    return BackendRequest(text=text, history=history, earlier_turns=len(turns))


def close_discarded_turn(history: History, outstanding: Sequence[str]) -> History:
    """Close a suspended group so its (possibly executed) tool calls stay visible and paired."""
    results = tuple(Message.tool(call_id, DISCARDED_RESULT_TEXT) for call_id in outstanding)
    return history.extend(results).append(Message.assistant(DISCARDED_TURN_TEXT))


def backend_system_prompt(catalog: PromptCatalog, config: Config, context: Mapping[str, Any] | None = None) -> str:
    """The backend system prompt: base render, then the history context note and guidance.

    ``context`` holds the Jinja variables of the templates (see ``prompts.render``).
    """

    def rendered(key: str) -> str:
        return render(catalog.get(key), config, context, catalog=catalog, key=key)

    prompt = rendered(config.backend.prompt_key)
    history_cfg = config.backend.conversation_history
    if not history_cfg.enabled:
        return prompt
    parts = [prompt, rendered(history_cfg.context_key)]
    if guidance_key := history_cfg.resolved_guidance_key:
        parts.append(rendered(guidance_key))
    return "\n\n".join(part for part in parts if part)


def request_template(catalog: PromptCatalog, config: Config) -> str:
    """The per-turn request template, or ``""`` when the request is the bare query."""
    history_cfg = config.backend.conversation_history
    if not history_cfg.enabled or history_cfg.include == HISTORY_BACKEND_TURNS:
        return ""
    template = catalog.get(history_cfg.request_key)
    for placeholder in (USER_MESSAGE, QUERY):
        if placeholder not in template:
            raise ConfigError(f"prompt {history_cfg.request_key!r} must contain {placeholder}")
    return template
