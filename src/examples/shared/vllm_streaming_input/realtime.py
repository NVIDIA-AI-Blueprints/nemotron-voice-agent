# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI Realtime text sessions mapped to chat completions, as NVIDIA Dynamo serves them.

The server keeps a realtime session and conversation, renders them as a chat
completion request, and streams the answer back as realtime events. Beyond the
OpenAI Realtime API, it takes ``input_text.append``, ``input_text.clear``, and
``input_text.commit`` for incremental text input, and a
``chat_completion_options`` session field, also set through
``session.chat_completion_options.update``, for chat completion fields that
realtime sessions lack.
"""

import uuid
from typing import Any

#: Chat completion request fields a session may set through ``chat_completion_options``.
CHAT_COMPLETION_OPTIONS = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "seed",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "thinking_token_budget",
        "chat_template_kwargs",
        "response_format",
    }
)
REASONING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def event(event_type: str, **data: Any) -> dict[str, Any]:
    """Return a server event of ``event_type``."""
    return {"type": event_type, "event_id": f"event_{uuid.uuid4().hex}", **data}


def error_event(code: str, message: str, client_event: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return an ``invalid_request_error`` event answering ``client_event``."""
    error = {"type": "invalid_request_error", "code": code, "message": message}
    return event("error", error={**error, "event_id": (client_event or {}).get("event_id")})


def new_session(model: str) -> dict[str, Any]:
    """Return the session a new connection starts with."""
    return {
        "type": "realtime",
        "object": "realtime.session",
        "id": f"sess_{uuid.uuid4().hex}",
        "model": model,
        "output_modalities": ["text"],
        "instructions": "",
        "max_output_tokens": "inf",
        "truncation": "disabled",
    }


def validate_session(session: dict[str, Any]) -> None:
    """Raise ``ValueError`` when the session asks for something this server cannot do."""
    if session.get("type") != "realtime":
        raise ValueError("session.type must be 'realtime'")
    if session.get("output_modalities") not in (None, ["text"]):
        raise ValueError("only text output is supported")
    if session.get("audio") is not None:
        raise ValueError("audio input and output are not supported")
    if session.get("truncation") not in (None, "disabled"):
        raise ValueError("automatic truncation is not supported; use truncation='disabled'")
    if not isinstance(session.get("instructions", ""), str):
        raise ValueError("session.instructions must be a string")
    max_tokens = session.get("max_output_tokens", "inf")
    if max_tokens != "inf" and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1):
        raise ValueError("max_output_tokens must be a positive integer or 'inf'")
    for tool in session.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not tool.get("name"):
            raise ValueError("each tool must be a function with a name")
    if (session.get("reasoning") or {}).get("effort") not in (None, *REASONING_EFFORTS):
        raise ValueError(f"reasoning.effort must be one of {', '.join(REASONING_EFFORTS)}")
    options = session.get("chat_completion_options") or {}
    if not isinstance(options, dict):
        raise ValueError("chat_completion_options must be an object")
    if unsupported := set(options) - CHAT_COMPLETION_OPTIONS:
        raise ValueError(f"chat_completion_options.{sorted(unsupported)[0]} is not supported")


def normalize_item(value: Any) -> dict[str, Any]:
    """Validate a client conversation item and return it as the server stores it."""
    if not isinstance(value, dict):
        raise ValueError("item must be an object")
    item = {"id": value.get("id") or f"item_{uuid.uuid4().hex}", "object": "realtime.item", "status": "completed"}
    if value.get("type") == "function_call":
        if not value.get("call_id") or not value.get("name") or not isinstance(value.get("arguments", ""), str):
            raise ValueError("function_call needs call_id, name, and string arguments")
        return {
            **item,
            "type": "function_call",
            "call_id": value["call_id"],
            "name": value["name"],
            "arguments": value.get("arguments", ""),
        }
    if value.get("type") == "function_call_output":
        if not value.get("call_id") or not isinstance(value.get("output"), str):
            raise ValueError("function_call_output needs call_id and a string output")
        return {**item, "type": "function_call_output", "call_id": value["call_id"], "output": value["output"]}
    if value.get("type") != "message" or value.get("role") not in ("system", "user", "assistant"):
        raise ValueError(
            "only system, user, and assistant messages, function_call, and function_call_output items are supported"
        )
    content_type = "output_text" if value["role"] == "assistant" else "input_text"
    content = value.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("message content must be a non-empty array")
    if any(
        not isinstance(part, dict) or part.get("type") != content_type or not isinstance(part.get("text"), str)
        for part in content
    ):
        raise ValueError(f"{value['role']} message content must contain only {content_type}")
    return {
        **item,
        "type": "message",
        "role": value["role"],
        "content": [{"type": content_type, "text": part["text"]} for part in content],
    }


def user_item(text: str) -> dict[str, Any]:
    """Return the conversation item of a committed user text turn."""
    return normalize_item({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})


def chat_messages(items: list[dict[str, Any]], instructions: str) -> list[dict[str, Any]]:
    """Render conversation items as chat messages.

    Function calls join the assistant message they follow, as one chat
    completion turn holds both the spoken text and its tool calls.
    """
    messages: list[dict[str, Any]] = [{"role": "system", "content": instructions}] if instructions else []
    for item in items:
        if item["type"] == "function_call":
            if not messages or messages[-1]["role"] != "assistant":
                messages.append({"role": "assistant", "content": ""})
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            messages[-1].setdefault("tool_calls", []).append(call)
        elif item["type"] == "function_call_output":
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
        else:
            messages.append({"role": item["role"], "content": "".join(part["text"] for part in item["content"])})
    return messages


def chat_request_fields(session: dict[str, Any]) -> dict[str, Any]:
    """Map the session's options to chat completion request fields.

    ``reasoning.effort`` also sets the chat template's ``enable_thinking``,
    unless ``chat_completion_options.chat_template_kwargs`` already sets it.
    Realtime has no ``none`` effort, so ``minimal`` turns thinking off.
    """
    fields = dict(session.get("chat_completion_options") or {})
    if tools := session.get("tools"):
        fields["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                },
            }
            for tool in tools
        ]
        tool_choice = session.get("tool_choice", "auto")
        if isinstance(tool_choice, dict):
            tool_choice = {"type": "function", "function": {"name": tool_choice["name"]}}
        fields["tool_choice"] = tool_choice
        if (parallel := session.get("parallel_tool_calls")) is not None:
            fields["parallel_tool_calls"] = parallel
    if (max_tokens := session.get("max_output_tokens", "inf")) != "inf":
        fields["max_completion_tokens"] = max_tokens
    if effort := (session.get("reasoning") or {}).get("effort"):
        fields["reasoning_effort"] = effort
        fields["chat_template_kwargs"] = {
            "enable_thinking": effort != "minimal",
            **fields.get("chat_template_kwargs", {}),
        }
    return fields


class ResponseEvents:
    """Turn one chat completion stream into the events and items of a realtime response."""

    def __init__(self, previous_item_id: str | None) -> None:
        """Start a response whose items follow ``previous_item_id`` in the conversation."""
        self.response_id = f"resp_{uuid.uuid4().hex}"
        self.text = ""
        self.usage: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self._item_id = f"item_{uuid.uuid4().hex}"
        self._previous_item_id = previous_item_id
        self._calls: dict[int, dict[str, Any]] = {}

    def started(self) -> list[dict[str, Any]]:
        """Return the events that announce the response and its message item."""
        item = self._message("in_progress")
        return [
            event("response.created", response=self._response("in_progress", [])),
            event("response.output_item.added", response_id=self.response_id, output_index=0, item=item),
            event("conversation.item.added", previous_item_id=self._previous_item_id, item=item),
            self._part_event("response.content_part.added", ""),
        ]

    def chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the events for one chat completion chunk."""
        self.usage = chunk.get("usage") or self.usage
        events = []
        for choice in chunk.get("choices") or []:
            self.finish_reason = choice.get("finish_reason") or self.finish_reason
            delta = choice.get("delta") or {}
            for call in delta.get("tool_calls") or []:
                events.extend(self._call_events(call))
            if text := delta.get("content"):
                self.text += text
                events.append(self._text_event("response.output_text.delta", delta=text))
        return events

    def finished(
        self, status: str, status_details: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the closing events and the items the response adds to the conversation."""
        item_status = "completed" if status == "completed" else "incomplete"
        message = self._message(item_status)
        events = [
            self._text_event("response.output_text.done", text=self.text),
            self._part_event("response.content_part.done", self.text),
            event("response.output_item.done", response_id=self.response_id, output_index=0, item=message),
            event("conversation.item.done", previous_item_id=self._previous_item_id, item=message),
        ]
        items, previous = [message], message["id"]
        for index, call in enumerate(self._calls.values(), start=1):
            call["status"] = item_status
            ids = {
                "response_id": self.response_id,
                "item_id": call["id"],
                "output_index": index,
                "call_id": call["call_id"],
            }
            events += [
                event("response.function_call_arguments.done", **ids, name=call["name"], arguments=call["arguments"]),
                event("response.output_item.done", response_id=self.response_id, output_index=index, item=dict(call)),
                event("conversation.item.done", previous_item_id=previous, item=dict(call)),
            ]
            items.append(dict(call))
            previous = call["id"]
        response = self._response(status, items)
        if status_details is not None:
            response["status_details"] = status_details
        if self.usage:
            response["usage"] = {
                "input_tokens": self.usage.get("prompt_tokens", 0),
                "output_tokens": self.usage.get("completion_tokens", 0),
                "total_tokens": self.usage.get("total_tokens", 0),
            }
        events.append(event("response.done", response=response))
        return events, items

    def _call_events(self, call: dict[str, Any]) -> list[dict[str, Any]]:
        function = call.get("function") or {}
        index = call.get("index", len(self._calls))
        events = []
        if index not in self._calls:
            previous = list(self._calls.values())[-1]["id"] if self._calls else self._item_id
            item = {
                "id": f"item_{uuid.uuid4().hex}",
                "object": "realtime.item",
                "type": "function_call",
                "status": "in_progress",
                "call_id": call.get("id") or f"call_{uuid.uuid4().hex}",
                "name": function.get("name") or "",
                "arguments": "",
            }
            self._calls[index] = item
            output_index = len(self._calls)
            events += [
                event(
                    "response.output_item.added",
                    response_id=self.response_id,
                    output_index=output_index,
                    item=dict(item),
                ),
                event("conversation.item.added", previous_item_id=previous, item=dict(item)),
            ]
        item = self._calls[index]
        item["name"] = item["name"] or function.get("name") or ""
        if arguments := function.get("arguments"):
            item["arguments"] += arguments
            output_index = list(self._calls).index(index) + 1
            events.append(
                event(
                    "response.function_call_arguments.delta",
                    response_id=self.response_id,
                    item_id=item["id"],
                    output_index=output_index,
                    call_id=item["call_id"],
                    delta=arguments,
                )
            )
        return events

    def _message(self, status: str) -> dict[str, Any]:
        return {
            "id": self._item_id,
            "object": "realtime.item",
            "type": "message",
            "status": status,
            "role": "assistant",
            "content": [{"type": "output_text", "text": self.text}] if self.text else [],
        }

    def _response(self, status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": self.response_id,
            "object": "realtime.response",
            "status": status,
            "output": output,
            "output_modalities": ["text"],
        }

    def _text_event(self, event_type: str, **data: Any) -> dict[str, Any]:
        return event(
            event_type, response_id=self.response_id, item_id=self._item_id, output_index=0, content_index=0, **data
        )

    def _part_event(self, event_type: str, text: str) -> dict[str, Any]:
        return self._text_event(event_type, part={"type": "text", "text": text})
