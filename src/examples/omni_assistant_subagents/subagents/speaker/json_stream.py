# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Incremental decoding of a single string field out of streamed JSON."""

from __future__ import annotations

_JSON_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}


class JsonStringFieldStreamer:
    """Decode one top-level JSON string field as its text arrives.

    Lets a caller start using a field, such as speaking a response, long before
    the surrounding object is complete. :attr:`done` is set at the field's
    closing quote, after which :meth:`feed` returns nothing.
    """

    def __init__(self, field_name: str) -> None:
        """Create a streamer for the given top-level field name."""
        self._needle = f'"{field_name}"'
        self._state = "search"
        self._buffer = ""
        self._escaped = False
        self._unicode_remaining = 0
        self._unicode_buffer = ""
        self.done = False

    def feed(self, text: str) -> str:
        """Consume a content delta and return newly decoded field characters."""
        if self.done or not text:
            return ""
        if self._state != "in_string":
            self._buffer += text
            emitted = self._advance_to_string()
            if self._state != "in_string":
                return ""
            text = emitted
        return self._consume_string_chars(text)

    def _advance_to_string(self) -> str:
        while True:
            if self._state == "search":
                idx = self._buffer.find(self._needle)
                if idx < 0:
                    self._buffer = self._buffer[-len(self._needle) :]
                    return ""
                self._buffer = self._buffer[idx + len(self._needle) :]
                self._state = "colon"
            if self._state == "colon":
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    return ""
                if stripped[0] != ":":
                    self._state = "search"
                    self._buffer = stripped
                    continue
                self._buffer = stripped[1:]
                self._state = "quote"
            if self._state == "quote":
                stripped = self._buffer.lstrip()
                if not stripped:
                    self._buffer = ""
                    return ""
                if stripped[0] != '"':
                    self._state = "search"
                    self._buffer = stripped
                    continue
                self._state = "in_string"
                emitted = stripped[1:]
                self._buffer = ""
                return emitted

    def _consume_string_chars(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            if self._unicode_remaining:
                self._unicode_buffer += ch
                self._unicode_remaining -= 1
                if self._unicode_remaining == 0:
                    try:
                        out.append(chr(int(self._unicode_buffer, 16)))
                    except ValueError:
                        out.append(f"\\u{self._unicode_buffer}")
                    self._unicode_buffer = ""
                continue
            if self._escaped:
                self._escaped = False
                if ch == "u":
                    self._unicode_remaining = 4
                    self._unicode_buffer = ""
                else:
                    out.append(_JSON_ESCAPES.get(ch, ch))
                continue
            if ch == "\\":
                self._escaped = True
                continue
            if ch == '"':
                self.done = True
                break
            out.append(ch)
        return "".join(out)
