# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-speaker replies: the JSON reply format and keeping the silence reply out of text-to-speech."""

from __future__ import annotations

import contextvars
import re
from collections.abc import AsyncIterator

from loguru import logger
from openai.types.chat import ChatCompletionChunk
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    SystemFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from examples.shared.json_stream import JsonStringFieldStreamer

SILENCE_REPLY = "..."
_SILENCE_TEXT = re.compile(r"^[.…\s]*$")
_NOT_ADDRESSED = re.compile(r'(?<!\\)"addressed_to_assistant"\s*:\s*false')
_SPEAKER_LABEL = re.compile(r"Speaker \d+: ")

ADDRESSED_REPLY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "reply",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"addressed_to_assistant": {"type": "boolean"}, "reply": {"type": "string"}},
            "required": ["addressed_to_assistant", "reply"],
            "additionalProperties": False,
        },
    },
}

_out_of_band = contextvars.ContextVar("addressed_reply_out_of_band", default=False)


class AddressedReplyMixin:
    """Ask spoken turns for the multi-speaker JSON reply and stream only its ``reply`` text.

    One-shot requests such as chat-history summaries keep plain text.
    """

    def build_chat_completion_params(self, params_from_context) -> dict:
        """Add the JSON reply format to every request except one-shot inference."""
        params = super().build_chat_completion_params(params_from_context)
        if "messages" in params:
            params["messages"] = join_back_to_back_lines(params["messages"])
        if not _out_of_band.get():
            params["response_format"] = ADDRESSED_REPLY_FORMAT
        return params

    async def run_inference(self, *args, **kwargs) -> str | None:
        """Run one-shot inference without the JSON reply format."""
        token = _out_of_band.set(True)
        try:
            return await super().run_inference(*args, **kwargs)
        finally:
            _out_of_band.reset(token)

    async def get_chat_completions(self, context: LLMContext) -> AsyncIterator[ChatCompletionChunk]:
        """Stream the reply field's text in place of the JSON object."""
        return reply_field_chunks(await super().get_chat_completions(context))


def join_back_to_back_lines(messages: list) -> list:
    """Join speaker lines that follow each other with no reply between them.

    A turn interrupted before its reply leaves two user lines in a row, and the
    model then sometimes speaks them back instead of answering "...".
    """
    joined: list = []
    for message in messages:
        previous = joined[-1] if joined else None
        text = message.get("content")
        if (
            previous is not None
            and previous.get("role") == message.get("role") == "user"
            and isinstance(previous.get("content"), str)
            and previous["content"].startswith("Speaker ")
            and isinstance(text, str)
            and text.startswith("Speaker ")
        ):
            labels = _SPEAKER_LABEL.findall(previous["content"])
            if labels and text.startswith(labels[-1]):
                text = text[len(labels[-1]) :]
            joined[-1] = {**previous, "content": f"{previous['content']} {text}"}
        else:
            joined.append(message)
    return joined


async def reply_field_chunks(stream: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[ChatCompletionChunk]:
    """Rewrite chunk content to the decoded ``reply`` field, or "..." when the model flagged it as not addressed.

    A reply that is not a JSON object passes through.
    """
    reply = JsonStringFieldStreamer("reply")
    raw = ""
    is_json: bool | None = None
    spoke = silenced = False
    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None or not delta.content:
            yield chunk
            continue
        raw += delta.content
        if is_json is None:
            if not raw.strip():
                delta.content = None
                yield chunk
                continue
            is_json = raw.lstrip().startswith("{")
            delta.content = raw
        if is_json:
            text = reply.feed(delta.content)
            # A reply the model flagged as not for the assistant becomes "..." even if it wrote words.
            if not spoke and not silenced and _NOT_ADDRESSED.search(raw):
                silenced, text = True, SILENCE_REPLY
            elif silenced:
                text = ""
            spoke = spoke or bool(text)
            delta.content = text or None
        yield chunk
    logger.debug(f"Multi-speaker reply: {raw}")


def is_silence_reply(text: str) -> bool:
    """Return whether reply text is only the "..." silence sentinel."""
    return bool(text.strip()) and bool(_SILENCE_TEXT.match(text))


class SilentReplyFilter(FrameProcessor):
    """Mark an all-dots reply skip-TTS, since Pipecat's TTS service fails after three silent contexts in a row."""

    def __init__(self, **kwargs):
        """Start with no reply in progress."""
        super().__init__(**kwargs)
        self._held: list[Frame] = []
        self._holding = False
        self._text = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Hold a reply until it is known to be speech or the silence sentinel."""
        await super().process_frame(frame, direction)

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InterruptionFrame):
            self._reset()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            await self._release(direction)
            self._holding = True
            self._held = [frame]
            return

        if not self._holding or isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._held.append(frame)
            self._text += frame.text
            if not _SILENCE_TEXT.match(self._text):
                await self._release(direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            self._held.append(frame)
            if is_silence_reply(self._text):
                for held in self._held:
                    held.skip_tts = True
            await self._release(direction)
            return

        await self._release(direction)
        await self.push_frame(frame, direction)

    async def _release(self, direction: FrameDirection) -> None:
        held = self._held
        self._reset()
        for frame in held:
            await self.push_frame(frame, direction)

    def _reset(self) -> None:
        self._held = []
        self._holding = False
        self._text = ""
