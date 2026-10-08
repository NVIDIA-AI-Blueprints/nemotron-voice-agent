# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Responses-style backend over OpenAI-compatible chat completions.

The delegate worker and live-session clients speak the Responses event vocabulary
(`response.created`, `response.output_item.done`, `response.function_call_arguments.*`,
`response.output_text.delta`, `response.completed`). Most hosted models are served as
chat completions, so this module converts in both directions:

* `responses_items_to_messages` / `to_chat_tools`: a stateless request from the
  delegate's accumulated Responses items.
* `ChatResponseTranslator`: chat-completion stream chunks -> Responses stream events.

It performs no I/O. Hidden reasoning (`reasoning_content`, `<think>` spans) is
counted for diagnostics and never emitted to the client or the talker.
"""

import re
import time
from dataclasses import dataclass, field

from ...common.ids import uid

TEXT_PART_TYPES = {"input_text", "output_text", "text"}
ROLE_MAP = {"developer": "system", "system": "system", "user": "user", "assistant": "assistant"}


def item_text(content) -> str:
    """Return the text of a Responses message ``content`` (a string or a list of text parts)."""
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content or [] if isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES
    )


def responses_items_to_messages(instructions: str | None, items: list[dict]) -> list[dict]:
    """Convert Responses input items to chat messages.

    Consecutive function calls become one assistant message with `tool_calls`; each
    `function_call_output` becomes a `tool` message addressed by its original call id.
    Adjacent user messages are merged because several chat templates reject them.
    """
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    previous_was_assistant_text = False
    for item in items:
        kind = item.get("type", "message" if "role" in item else None)
        if kind == "message":
            role = ROLE_MAP.get(item.get("role"), "user")
            text = item_text(item.get("content"))
            if role == "user" and messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n" + text
            else:
                messages.append({"role": role, "content": text})
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
            messages.append({"role": "tool", "tool_call_id": item["call_id"], "content": item["output"]})
            previous_was_assistant_text = False
        # reasoning items and other types carry nothing a chat model can consume
    return messages


def to_chat_tools(tools: list[dict] | None) -> tuple[list[dict], list[str]]:
    """Function tools only; hosted tools (web_search) have no chat-completions equivalent."""
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


def to_chat_tool_choice(choice):
    """Convert a Responses ``tool_choice`` to the chat-completions form."""
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "function", "function": {"name": choice["name"]}}
    return choice if choice in {"auto", "none", "required"} else "auto"


THINK_SPAN = re.compile(r"<think>.*?</think>", re.S)


def split_reasoning(text: str) -> tuple[str, str]:
    """Return (answer, reasoning). Handles templates that open `<think>` in the prompt."""
    if "</think>" in text:
        reasoning, answer = text.rsplit("</think>", 1)
        return answer.strip(), reasoning.replace("<think>", "").strip()
    if "<think>" in text:  # reasoning was cut off before it closed; no answer exists
        before, _, reasoning = text.partition("<think>")
        return before.strip(), reasoning.strip()
    return text, ""


def strip_reasoning(text: str) -> str:
    """Return ``text`` without hidden reasoning (``<think>`` spans)."""
    answer, _ = split_reasoning(THINK_SPAN.sub("", text))
    return answer


def _extra(obj, name: str):
    """Fields such as `reasoning_content` are extensions, present as SDK extras."""
    value = getattr(obj, name, None)
    if value is None:
        value = (getattr(obj, "model_extra", None) or {}).get(name)
    return value


@dataclass
class _Call:
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""
    output_index: int | None = None
    item_id: str = field(default_factory=lambda: uid("fc"))
    sent_arguments: int = 0


class ChatResponseTranslator:
    """Chat-completion chunks in, Responses stream events out.

    `buffer_text` holds answer text until the end. Use it when the model may emit
    `<think>` reasoning inline: the visible answer is only known once it finishes.
    """

    def __init__(self, model: str, buffer_text: bool = False, response_id: str | None = None):
        """Start a translation for ``model``; ``buffer_text`` holds answer text until the stream ends."""
        self.model, self.buffer_text = model, buffer_text
        self.response_id = response_id or uid("resp")
        self.created_at = int(time.time())
        self.sequence = 0
        self.next_output_index = 0
        self.text_parts: list[str] = []
        self.message_index: int | None = None
        self.message_id = uid("msg")
        self.live_text = ""
        self.calls: dict[int, _Call] = {}
        self.finish_reason: str | None = None
        self.usage: dict | None = None
        self.reasoning_chars = 0
        self.saw_reasoning_field = False
        self.output: list[dict] = []

    # ------------------------------------------------------------------ events
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

    # ------------------------------------------------------------------- input
    def feed(self, chunk) -> list[dict]:
        """Translate one chat-completion chunk into zero or more Responses events."""
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
        reasoning = _extra(delta, "reasoning_content") or _extra(delta, "reasoning")
        if reasoning:
            self.reasoning_chars += len(reasoning)
            self.saw_reasoning_field = True
        if delta.content:
            self.text_parts.append(delta.content)
            if not self.buffer_text:
                if self.message_index is None:
                    events += self._start_message()
                self.live_text += delta.content
                events.append(self._text_delta(delta.content))
        for fragment in delta.tool_calls or []:
            call = self.calls.setdefault(fragment.index, _Call(fragment.index))
            function = fragment.function
            if fragment.id:
                call.id = fragment.id
            if function and function.name:
                call.name += function.name
            if function and function.arguments:
                call.arguments += function.arguments
            if call.output_index is None and call.name:
                call.output_index = self.next_output_index
                self.next_output_index += 1
                call.id = call.id or uid("call")
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

    # ------------------------------------------------------------------ finish
    def finish(self) -> list[dict]:
        """Close open items and emit the terminal event. Safe to call exactly once."""
        events: list[dict] = []
        raw_text = "".join(self.text_parts)
        answer = self.answer_text
        if self.buffer_text and answer.strip():
            events += self._start_message()
            events.append(self._text_delta(answer))
        elif self.buffer_text:
            self.reasoning_chars += len(raw_text)
        if self.message_index is not None:
            if not self.buffer_text:
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
        for call in sorted(self.calls.values(), key=lambda c: c.index):
            if call.output_index is None:  # a call without a function name cannot be run
                continue
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
        if self.finish_reason in {"length", "content_filter"}:
            reason = "max_output_tokens" if self.finish_reason == "length" else "content_filter"
            events.append(
                self._event(
                    "response.incomplete",
                    response=self._response("incomplete", self.output, incomplete_details={"reason": reason}),
                )
            )
        else:
            events.append(self._event("response.completed", response=self._response("completed", self.output)))
        return events

    @property
    def answer_text(self) -> str:
        """The visible answer so far, without hidden reasoning."""
        text = "".join(self.text_parts)
        if not self.buffer_text:
            return text
        # The template opens `<think>` in the prompt, so a reply cut off by the token limit
        # before `</think>` is reasoning only. Unless the server already split reasoning
        # into its own field, nothing in it is safe to show.
        if self.finish_reason == "length" and "</think>" not in text and not self.saw_reasoning_field:
            return ""
        return strip_reasoning(text)


def thinking_enabled(extra: dict) -> bool:
    """Return True when the request enables the model's reasoning mode."""
    kwargs = (extra.get("extra_body") or {}).get("chat_template_kwargs") or {}
    return bool(kwargs.get("enable_thinking"))
