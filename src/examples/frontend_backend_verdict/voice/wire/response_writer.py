# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One response's event lifecycle, with its ordering guarantees enforced.

Guarantees (plan section 7.3), enforced here rather than trusted to callers:

* ``response.done`` is emitted exactly once per ``response.created``;
* at most one output item is open at a time, so items never interleave;
* every assistant utterance gets a fresh ``item_id``;
* a message item that is closed early is closed as ``incomplete`` with the
  transcript sent so far, never with text the client did not receive.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from examples.frontend_backend_verdict.voice.wire import server_events as ev
from examples.frontend_backend_verdict.voice.wire.ids import new_item_id, new_response_id

Emit = Callable[[dict[str, Any]], None]


class ConversationOrder:
    """Tracks the last conversation item id, for ``previous_item_id`` fields."""

    def __init__(self) -> None:
        """Start with an empty conversation."""
        self.last_item_id: str | None = None
        self.item_ids: list[str] = []

    def append(self, item_id: str) -> str | None:
        """Record ``item_id`` as the newest item and return the previous one."""
        previous = self.last_item_id
        self.last_item_id = item_id
        self.item_ids.append(item_id)
        return previous


class MessageItemWriter:
    """An assistant message item inside a response."""

    def __init__(self, response: ResponseWriter, output_index: int, *, modality: str, kind: str) -> None:
        """Allocate a fresh item id; nothing is emitted until :meth:`start`."""
        self._response = response
        self.item_id = new_item_id()
        self.output_index = output_index
        self.modality = modality
        self.kind = kind
        self.text = ""
        self.started = False
        self.closed = False

    def _part(self) -> dict[str, Any]:
        if self.modality == "audio":
            return {"type": "audio", "transcript": self.text}
        return {"type": "text", "text": self.text}

    def start(self) -> None:
        """Open the item and its content part."""
        if self.started:
            return
        self.started = True
        response = self._response
        item = ev.assistant_message_item(self.item_id, status="in_progress")
        previous = response.conversation.append(self.item_id)
        response.emit(ev.output_item_added(response.response_id, self.output_index, item))
        response.emit(ev.item_added(item, previous))
        response.emit(
            ev.content_part_added(
                response.response_id,
                self.item_id,
                self.output_index,
                {"type": "audio", "transcript": ""} if self.modality == "audio" else {"type": "text", "text": ""},
            )
        )

    def transcript_delta(self, text: str) -> None:
        """Send the transcript (audio modality) or text (text modality) for the next sentence."""
        if not text:
            return
        self.start()
        response = self._response
        self.text += text
        if self.modality == "audio":
            response.emit(ev.output_audio_transcript_delta(response.response_id, self.item_id, self.output_index, text))
        else:
            response.emit(ev.output_text_delta(response.response_id, self.item_id, self.output_index, text))

    def audio_delta(self, audio_b64: str) -> None:
        """Send one audio chunk."""
        self.start()
        response = self._response
        response.emit(ev.output_audio_delta(response.response_id, self.item_id, self.output_index, audio_b64))

    def close(self, *, status: str = "completed") -> dict[str, Any]:
        """Close the item and return its final item object."""
        self.start()
        response = self._response
        final = ev.assistant_message_item(self.item_id, status=status, modality=self.modality, transcript=self.text)
        if self.closed:
            return final
        self.closed = True
        rid, iid, index = response.response_id, self.item_id, self.output_index
        if self.modality == "audio":
            response.emit(ev.output_audio_done(rid, iid, index))
            response.emit(ev.output_audio_transcript_done(rid, iid, index, self.text))
        else:
            response.emit(ev.output_text_done(rid, iid, index, self.text))
        response.emit(ev.content_part_done(rid, iid, index, self._part()))
        response.emit(ev.output_item_done(rid, index, final))
        response.emit(ev.item_done(final, None))
        response.output.append(final)
        response.open_item = None
        return final


class ResponseWriter:
    """Emits one response's events in protocol order."""

    def __init__(self, emit: Emit, conversation: ConversationOrder, *, modality: str = "audio") -> None:
        """Bind to the session's emitter; call :meth:`open` to emit ``response.created``."""
        self.emit = emit
        self.conversation = conversation
        self.modality = modality
        self.response_id = new_response_id()
        self.output: list[dict[str, Any]] = []
        self.open_item: MessageItemWriter | None = None
        self.opened = False
        self.done = False
        self.status: str | None = None
        self.function_call_ids: list[str] = []

    def open(self) -> None:
        """Emit ``response.created`` (idempotent)."""
        if self.opened:
            return
        self.opened = True
        self.emit(
            ev.response_created(ev.response_object(self.response_id, status="in_progress", modality=self.modality))
        )

    def begin_message(self, *, kind: str = "answer") -> MessageItemWriter:
        """Start a new assistant message item (``kind`` is internal: answer, filler, greeting)."""
        self._require_open()
        if self.open_item is not None:
            raise RuntimeError("an output item is already open; items never interleave")
        item = MessageItemWriter(self, len(self.output), modality=self.modality, kind=kind)
        self.open_item = item
        return item

    def function_call(self, *, call_id: str, name: str, arguments: str) -> dict[str, Any]:
        """Emit one complete function-call item."""
        self._require_open()
        if self.open_item is not None:
            raise RuntimeError("close the open message item before emitting function calls")
        item_id = new_item_id()
        index = len(self.output)
        in_progress = ev.function_call_item(item_id, call_id=call_id, name=name, arguments="", status="in_progress")
        previous = self.conversation.append(item_id)
        self.emit(ev.output_item_added(self.response_id, index, in_progress))
        self.emit(ev.item_added(in_progress, previous))
        self.emit(ev.function_call_arguments_delta(self.response_id, item_id, index, call_id, arguments))
        self.emit(
            ev.function_call_arguments_done(
                self.response_id, item_id, index, call_id=call_id, name=name, arguments=arguments
            )
        )
        final = ev.function_call_item(item_id, call_id=call_id, name=name, arguments=arguments, status="completed")
        self.emit(ev.output_item_done(self.response_id, index, final))
        self.emit(ev.item_done(final, None))
        self.output.append(final)
        self.function_call_ids.append(call_id)
        return final

    def finish(
        self,
        status: str = "completed",
        *,
        usage: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> None:
        """Close any open item and emit ``response.done`` exactly once."""
        if self.done:
            return
        self.open()
        if self.open_item is not None:
            self.open_item.close(status="incomplete" if status != "completed" else "completed")
        self.done = True
        self.status = status
        details: dict[str, Any] | None = None
        if status == "cancelled":
            details = {"type": "cancelled", "reason": reason or "client_cancelled"}
        elif status in ("failed", "incomplete"):
            details = {"type": status, "reason": reason, "error": None}
        self.emit(
            ev.response_done(
                ev.response_object(
                    self.response_id,
                    status=status,
                    output=self.output,
                    usage=usage or ev.usage_object(),
                    status_details=details,
                    modality=self.modality,
                )
            )
        )

    def _require_open(self) -> None:
        if self.done:
            raise RuntimeError("response is already done")
        self.open()
