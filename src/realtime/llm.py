# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVIDIA LLM service for Realtime sessions that offer client-executed tools."""

from __future__ import annotations

from typing import Any

from pipecat.processors.aggregators import async_tool_messages
from pipecat.services.nvidia.llm import NvidiaLLMService


def fold_async_tool_messages(messages: list[Any]) -> list[Any]:
    """Return ``messages`` with Pipecat's async-tool developer messages folded away.

    A result that reaches the context after the conversation moved on is added
    as a ``developer`` message, while the call's ``tool`` message keeps its
    "still running" placeholder. Put each final result into its tool message and
    drop the developer messages, so the provider only sees standard tool results.
    """
    final_results: dict[str, str] = {}
    folded_any = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "developer":
            continue
        payload = async_tool_messages.parse_message(message)
        if payload is None:
            continue
        folded_any = True
        if payload.kind == "final" and payload.result is not None:
            final_results[payload.tool_call_id] = payload.result
    if not folded_any:
        return messages

    result: list[Any] = []
    for message in messages:
        if isinstance(message, dict):
            payload = async_tool_messages.parse_message(message)
            if payload is not None and message.get("role") == "developer":
                continue
            if (
                payload is not None
                and payload.kind == "started"
                and message.get("role") == "tool"
                and payload.tool_call_id in final_results
            ):
                message = {**message, "content": final_results[payload.tool_call_id]}
        result.append(message)
    return result


class RealtimeNvidiaLLMService(NvidiaLLMService):
    """``NvidiaLLMService`` whose client tool calls survive barge-in.

    Client tool calls are registered with ``cancel_on_interruption=False`` so a
    user turn cannot cancel a call the client is still running. Pipecat treats
    such calls as async tools: it adds standing async-tool guidance to the
    system instruction, and a result that arrives after the conversation moved
    on becomes a ``developer`` message. A Realtime client answers its calls
    before it requests the next response, so this service leaves that guidance
    out and folds those results into the tool message. Provider requests keep
    the system, user, assistant, and tool roles that NIM and vLLM templates use.
    """

    def _has_async_tools(self) -> bool:
        return False

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Build the chat request with async-tool messages folded into tool results."""
        params = super().build_chat_completion_params(params_from_context)
        messages = params.get("messages")
        if isinstance(messages, list):
            params["messages"] = fold_async_tool_messages(messages)
        return params
