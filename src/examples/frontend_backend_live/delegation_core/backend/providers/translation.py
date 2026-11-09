# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion between the Responses vocabulary and OpenAI-compatible chat completions, with no I/O."""

import json
import time
from dataclasses import dataclass, field

from examples.shared.ids import uid

from .base import BrokenStreamError

TEXT_PART_TYPES = {"input_text", "output_text", "text"}
PART_TYPES = TEXT_PART_TYPES | {"input_image", "input_file"}
ROLE_MAP = {"developer": "system", "system": "system", "user": "user", "assistant": "assistant"}


def item_text(content) -> str:
    """Return the text of a Responses message ``content`` (a string or a list of text parts)."""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
    )


def _is_parts(output) -> bool:
    """Return whether a tool output is content parts, one or more of the ``PART_TYPES``; any other array is data."""
    if not isinstance(output, list) or not output:
        return False
    return all(isinstance(part, dict) and part.get("type") in PART_TYPES for part in output)


def user_content(content) -> str | list[dict]:
    """Return a user message's chat content: its text, or its text and image parts when it carries images."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        part_type = part.get("type") if isinstance(part, dict) else None
        if part_type in TEXT_PART_TYPES:
            parts.append({"type": "text", "text": part.get("text", "")})
        elif part_type == "input_image" and part.get("image_url"):
            image = {"url": part["image_url"]}
            if part.get("detail"):
                image["detail"] = part["detail"]
            parts.append({"type": "image_url", "image_url": image})
    if all(part["type"] == "text" for part in parts):
        return "\n".join(part["text"] for part in parts)
    return parts


def _joined(first: str | list[dict], second: str | list[dict]) -> str | list[dict]:
    """Return two user contents as one: text joined by a line break, or the parts of both when either has an image."""
    if isinstance(first, str) and isinstance(second, str):
        return first + "\n" + second
    parts = [
        [{"type": "text", "text": content}] if isinstance(content, str) else content for content in (first, second)
    ]
    return parts[0] + parts[1]


def responses_items_to_messages(instructions: str | None, items: list[dict]) -> list[dict]:
    """Convert Responses input items to chat messages.

    Adjacent user messages are merged, since several chat templates reject them.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    previous_was_assistant_text = False
    for item in items:
        kind = item.get("type", "message" if "role" in item else None)
        if kind == "message":
            role = ROLE_MAP.get(item.get("role"), "user")
            if role == "user":
                content = user_content(item.get("content"))
            else:
                content = item_text(item.get("content"))
            if role == "user" and messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = _joined(messages[-1]["content"], content)
            else:
                messages.append({"role": role, "content": content})
            previous_was_assistant_text = role == "assistant"
        elif kind == "function_call":
            call = {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item.get("arguments") or "{}"},
            }
            last = messages[-1] if messages else None
            if last and last["role"] == "assistant" and ("tool_calls" in last or previous_was_assistant_text):
                last.setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": "", "tool_calls": [call]})
            previous_was_assistant_text = False
        elif kind == "function_call_output":
            if "output" not in item:
                raise ValueError(f"The result for call {item['call_id']} has no output")
            output = item["output"]
            content = item_text(output) if isinstance(output, str) or _is_parts(output) else json.dumps(output)
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": content})
            previous_was_assistant_text = False
        # Reasoning items and other types carry nothing a chat model can consume.
    return messages


def to_chat_tools(tools: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Return the function tools in the chat form, and the hosted tool types left out, which have no chat form."""
    converted, dropped = [], []
    for tool in tools or []:
        if tool.get("type") != "function":
            dropped.append(str(tool.get("type")))
            continue
        function = {
            "name": tool["name"],
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted, dropped


def to_chat_tool_choice(choice) -> str | dict:
    """Convert a Responses ``tool_choice`` to the chat form; one with no chat form becomes ``auto``, not a failure."""
    if isinstance(choice, dict):
        name = choice.get("name")
        named = choice.get("type") == "function" and isinstance(name, str) and bool(name)
        return {"type": "function", "function": {"name": name}} if named else "auto"
    return choice if choice in ("auto", "none", "required") else "auto"


def to_chat_response_format(text_format) -> dict | None:
    """Convert a Responses ``text.format`` to the chat ``response_format``; ``None`` for plain text."""
    if not isinstance(text_format, dict):
        return None
    format_type = text_format.get("type")
    if format_type == "json_object":
        return {"type": "json_object"}
    if format_type != "json_schema":
        return None
    schema = {key: text_format[key] for key in ("name", "schema", "strict", "description") if key in text_format}
    return {"type": "json_schema", "json_schema": schema}


@dataclass
class _Call:
    index: int
    # Our own id, never the provider's: a provider that reuses an id cannot break or lose a tool answer.
    id: str = field(default_factory=lambda: uid("call"))
    name: str = ""
    arguments: str = ""
    output_index: int | None = None
    item_id: str = field(default_factory=lambda: uid("fc"))
    sent_arguments: int = 0


# The finish reasons of a reply cut short: its calls may be cut too, so none is emitted to run.
CUT_SHORT = {"length": "max_output_tokens", "content_filter": "content_filter"}


class ChatResponseTranslator:
    """Chat-completion chunks in, Responses stream events out."""

    def __init__(self, model: str, *, single_call: bool):
        """Start a translation for ``model``."""
        self.model, self.single_call = model, single_call
        self.response_id = uid("resp")
        self.created_at = int(time.time())
        self.sequence = 0
        self.next_output_index = 0
        self.message_index: int | None = None
        self.message_id = uid("msg")
        self.live_text = ""
        self.calls: dict[int, _Call] = {}
        self.first_call_whole = False
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.output: list[dict] = []

    def _event(self, kind: str, **fields) -> dict:
        self.sequence += 1
        return {"type": kind, "sequence_number": self.sequence, **fields}

    def _response(self, status: str, output=None, **extra) -> dict:
        response = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output if output is not None else [],
            **extra,
        }
        if self.usage:
            response["usage"] = self.usage
        return response

    def start(self) -> list[dict]:
        """Return the opening ``response.created`` / ``response.in_progress`` events."""
        snapshot = self._response("in_progress")
        return [
            self._event("response.created", response=snapshot),
            self._event("response.in_progress", response=snapshot),
        ]

    def _start_message(self) -> list[dict]:
        self.message_index = self.next_output_index
        self.next_output_index += 1
        return [
            self._event(
                "response.output_item.added",
                output_index=self.message_index,
                item={
                    "id": self.message_id,
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            ),
            self._event(
                "response.content_part.added",
                item_id=self.message_id,
                output_index=self.message_index,
                content_index=0,
                part={"type": "output_text", "text": "", "annotations": [], "logprobs": []},
            ),
        ]

    def _text_delta(self, text: str) -> dict:
        return self._event(
            "response.output_text.delta",
            item_id=self.message_id,
            output_index=self.message_index,
            content_index=0,
            delta=text,
            logprobs=[],
        )

    def feed(self, chunk) -> list[dict]:
        """Translate one chat-completion chunk into zero or more Responses events; none once the reply has ended."""
        if self.first_call_whole:
            return []
        events: list[dict] = []
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = {
                "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        if not chunk.choices:
            return events
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            if self.message_index is None:
                events += self._start_message()
            self.live_text += delta.content
            events.append(self._text_delta(delta.content))
        for fragment in delta.tool_calls or []:
            if self.single_call and self.calls and fragment.index not in self.calls:
                # Parallel calls are off: end the reply with the first call, whole, before the output cap can cut it.
                self.finish_reason, self.first_call_whole = "tool_calls", True
                return events
            call = self.calls.setdefault(fragment.index, _Call(fragment.index))
            function = fragment.function
            if function and function.name:
                call.name += function.name
            if function and function.arguments:
                call.arguments += function.arguments
            if call.output_index is None and call.name:
                call.output_index = self.next_output_index
                self.next_output_index += 1
                events.append(
                    self._event(
                        "response.output_item.added",
                        output_index=call.output_index,
                        item={
                            "id": call.item_id,
                            "type": "function_call",
                            "status": "in_progress",
                            "call_id": call.id,
                            "name": call.name,
                            "arguments": "",
                        },
                    )
                )
            if call.output_index is not None and len(call.arguments) > call.sent_arguments:
                events.append(
                    self._event(
                        "response.function_call_arguments.delta",
                        item_id=call.item_id,
                        output_index=call.output_index,
                        delta=call.arguments[call.sent_arguments :],
                    )
                )
                call.sent_arguments = len(call.arguments)
        if choice.finish_reason:
            self.finish_reason = choice.finish_reason
        return events

    def finish(self) -> list[dict]:
        """Close open items and emit the terminal event; call it once.

        A reply cut short ends ``response.incomplete`` without its calls, and a stream that ends inside a call raises.
        """
        # A call without a function name cannot be run, so it has no output index and is left out.
        calls = [call for call in sorted(self.calls.values(), key=lambda c: c.index) if call.output_index is not None]
        if calls and self.finish_reason is None:
            raise BrokenStreamError("The backend's stream ended inside a tool call, with no finish reason")
        if self.finish_reason in CUT_SHORT:
            calls = []
        events: list[dict] = []
        if self.message_index is not None:
            answer = self.live_text
            part = {"type": "output_text", "text": answer, "annotations": [], "logprobs": []}
            message = {
                "id": self.message_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
            events += [
                self._event(
                    "response.output_text.done",
                    item_id=self.message_id,
                    output_index=self.message_index,
                    content_index=0,
                    text=answer,
                    logprobs=[],
                ),
                self._event(
                    "response.content_part.done",
                    item_id=self.message_id,
                    output_index=self.message_index,
                    content_index=0,
                    part=part,
                ),
                self._event("response.output_item.done", output_index=self.message_index, item=message),
            ]
            self.output.append(message)
        for call in calls:
            call.arguments = call.arguments or "{}"
            item = {
                "id": call.item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
            events += [
                self._event(
                    "response.function_call_arguments.done",
                    item_id=call.item_id,
                    output_index=call.output_index,
                    name=call.name,
                    arguments=call.arguments,
                ),
                self._event("response.output_item.done", output_index=call.output_index, item=item),
            ]
            self.output.append(item)
        if self.finish_reason in CUT_SHORT:
            details = {"reason": CUT_SHORT[self.finish_reason]}
            events.append(
                self._event(
                    "response.incomplete",
                    response=self._response("incomplete", self.output, incomplete_details=details),
                )
            )
        else:
            events.append(self._event("response.completed", response=self._response("completed", self.output)))
        return events
