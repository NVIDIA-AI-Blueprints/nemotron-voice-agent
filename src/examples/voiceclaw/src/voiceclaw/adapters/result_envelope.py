# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Streaming codec for the provider-neutral VoiceClaw result envelope."""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum, auto

from voiceclaw.domain.response_only import RESULT_ENVELOPE_SCHEMA, ResponseOnlyResultEnvelope
from voiceclaw.model_contracts import ModelContractCatalog, ModelContractError, load_model_contract_catalog

_FIELD_ORDER = ("schema", "speech", "display")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
MAX_RESULT_DISPLAY_BYTES = 128_000
MAX_RESULT_SPEECH_BYTES = 4_096
# Each decoded byte can require six ASCII bytes as a JSON ``\u00XX`` escape.
MAX_RESULT_ENVELOPE_BYTES = (MAX_RESULT_DISPLAY_BYTES + MAX_RESULT_SPEECH_BYTES) * 6 + 1024


class ResultEnvelopeProtocolError(ValueError):
    """A backend result or result-contract request violated the envelope protocol."""

    code = "invalid_result_envelope"


class ResultEnvelopeLimitError(ResultEnvelopeProtocolError):
    """The encoded envelope or one decoded channel exceeded its declared bound."""

    code = "result_envelope_limit"


class _State(Enum):
    BEFORE_OBJECT = auto()
    BEFORE_KEY = auto()
    IN_KEY = auto()
    AFTER_KEY = auto()
    BEFORE_VALUE = auto()
    IN_STRING = auto()
    IN_ESCAPE = auto()
    IN_UNICODE = auto()
    IN_NULL = auto()
    AFTER_VALUE = auto()
    AFTER_OBJECT = auto()


def build_result_envelope_prompt(
    user_goal: str,
    *,
    display_budget_bytes: int = MAX_RESULT_DISPLAY_BYTES,
    contracts: ModelContractCatalog | None = None,
) -> str:
    """Wrap one exact user goal with the versioned result-channel contract.

    ``display_budget_bytes`` is the configured capacity of the selected
    backend, not a VoiceClaw content heuristic.  The protocol-wide parser can
    accept the larger global limit while a constrained backend is asked to
    produce a complete result that fits its actual completion budget.
    """
    if not isinstance(user_goal, str) or not user_goal.strip() or "\x00" in user_goal:
        raise ResultEnvelopeProtocolError
    if (
        isinstance(display_budget_bytes, bool)
        or not isinstance(display_budget_bytes, int)
        or not 1 <= display_budget_bytes <= MAX_RESULT_DISPLAY_BYTES
    ):
        raise ResultEnvelopeProtocolError
    try:
        goal_json = json.dumps(user_goal, ensure_ascii=False)
        goal_json.encode("utf-8")
    except (TypeError, UnicodeEncodeError) as exc:
        raise ResultEnvelopeProtocolError from exc
    try:
        return (contracts or load_model_contract_catalog()).render_result_envelope(
            user_goal_json=goal_json,
            result_schema=RESULT_ENVELOPE_SCHEMA,
            speech_budget_bytes=MAX_RESULT_SPEECH_BYTES,
            display_budget_bytes=display_budget_bytes,
        )
    except ModelContractError as exc:
        raise ResultEnvelopeProtocolError from exc


def parse_result_envelope(text: str) -> ResponseOnlyResultEnvelope:
    """Parse one streamed result envelope without accepting plain-text fallback."""
    parser = ResultEnvelopeStreamParser(lambda _delta: None)
    parser.push(text)
    return parser.finish()


class ResultEnvelopeStreamParser:
    """Decode selected fields from one JSON object across arbitrary chunks.

    ``display`` is emitted incrementally as decoded Markdown. Those deltas are
    provisional until :meth:`finish` validates the complete object. ``speech``
    contains factual presentation material; it is captured when its value
    closes and returned only with the validated terminal envelope.

    Duplicate, missing, reordered, or unknown members fail closed. The codec
    does not repair model output or apply backend-specific compatibility rules.
    """

    def __init__(
        self,
        on_display_delta: Callable[[str], None],
        *,
        maximum_bytes: int = MAX_RESULT_ENVELOPE_BYTES,
        maximum_display_bytes: int = MAX_RESULT_DISPLAY_BYTES,
        maximum_speech_bytes: int = MAX_RESULT_SPEECH_BYTES,
    ) -> None:
        """Create a bounded parser with one synchronous display callback."""
        if not callable(on_display_delta):
            raise TypeError("on_display_delta must be callable")
        if isinstance(maximum_bytes, bool) or not isinstance(maximum_bytes, int) or maximum_bytes < 1:
            raise ValueError("maximum_bytes must be a positive integer")
        for name, value in (
            ("maximum_display_bytes", maximum_display_bytes),
            ("maximum_speech_bytes", maximum_speech_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._on_display_delta = on_display_delta
        self._maximum_bytes = maximum_bytes
        self._maximum_display_bytes = maximum_display_bytes
        self._maximum_speech_bytes = maximum_speech_bytes
        self._bytes = 0
        self._raw_parts: list[str] = []
        self._state = _State.BEFORE_OBJECT
        self._members: list[tuple[str, str | None]] = []
        self._canonical: dict[str, str | None] = {}
        self._current_key: str | None = None
        self._string_chars: list[str] = []
        self._string_bytes = 0
        self._unicode_digits = ""
        self._pending_high_surrogate: int | None = None
        self._null_offset = 0
        self._finished = False
        self._display_scratch: list[str] = []

    def push(self, chunk: str) -> None:
        """Consume one arbitrary text fragment and emit decoded display deltas."""
        if self._finished or not isinstance(chunk, str) or "\x00" in chunk:
            raise ResultEnvelopeProtocolError
        try:
            encoded = chunk.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ResultEnvelopeProtocolError from exc
        self._bytes += len(encoded)
        if self._bytes > self._maximum_bytes:
            raise ResultEnvelopeLimitError
        self._raw_parts.append(chunk)
        self._display_scratch = []
        try:
            for character in chunk:
                self._consume(character)
        except ResultEnvelopeProtocolError:
            self._display_scratch = []
            raise
        if self._display_scratch:
            self._on_display_delta("".join(self._display_scratch))
            self._display_scratch = []

    def finish(self) -> ResponseOnlyResultEnvelope:
        """Validate the terminal JSON and return its independent channels."""
        if self._finished:
            raise ResultEnvelopeProtocolError
        self._finished = True
        if self._state is not _State.AFTER_OBJECT or not self._members:
            raise ResultEnvelopeProtocolError
        if len(self._members) != len(_FIELD_ORDER):
            raise ResultEnvelopeProtocolError

        raw = "".join(self._raw_parts)
        try:
            parsed = json.loads(raw, object_pairs_hook=lambda pairs: list(pairs))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ResultEnvelopeProtocolError from exc
        if not isinstance(parsed, list) or parsed != self._members:
            raise ResultEnvelopeProtocolError

        try:
            return ResponseOnlyResultEnvelope(
                schema=self._required_string("schema"),
                speech=self._optional_string("speech"),
                display=self._required_string("display"),
            )
        except (TypeError, UnicodeEncodeError, ValueError) as exc:
            raise ResultEnvelopeProtocolError from exc

    def _consume(self, character: str) -> None:
        if self._state is _State.BEFORE_OBJECT:
            if character.isspace():
                return
            if character != "{":
                raise ResultEnvelopeProtocolError
            self._state = _State.BEFORE_KEY
            return

        if self._state is _State.BEFORE_KEY:
            if character.isspace():
                return
            if character != '"':
                raise ResultEnvelopeProtocolError
            self._string_chars = []
            self._state = _State.IN_KEY
            return

        if self._state is _State.IN_KEY:
            if character == '"':
                key = "".join(self._string_chars)
                if len(self._members) >= len(_FIELD_ORDER):
                    raise ResultEnvelopeProtocolError
                expected = _FIELD_ORDER[len(self._members)]
                if key != expected:
                    raise ResultEnvelopeProtocolError
                self._current_key = key
                self._state = _State.AFTER_KEY
                return
            if character == "\\" or ord(character) < 0x20:
                raise ResultEnvelopeProtocolError
            self._string_chars.append(character)
            return

        if self._state is _State.AFTER_KEY:
            if character.isspace():
                return
            if character != ":":
                raise ResultEnvelopeProtocolError
            self._state = _State.BEFORE_VALUE
            return

        if self._state is _State.BEFORE_VALUE:
            if character.isspace():
                return
            if character == '"':
                self._string_chars = []
                self._string_bytes = 0
                self._pending_high_surrogate = None
                self._state = _State.IN_STRING
                return
            if self._current_key == "speech" and character == "n":
                self._null_offset = 1
                self._state = _State.IN_NULL
                return
            raise ResultEnvelopeProtocolError

        if self._state is _State.IN_STRING:
            if self._pending_high_surrogate is not None and character != "\\":
                raise ResultEnvelopeProtocolError
            if character == "\\":
                self._state = _State.IN_ESCAPE
                return
            if character == '"':
                if self._pending_high_surrogate is not None:
                    raise ResultEnvelopeProtocolError
                self._finish_value("".join(self._string_chars))
                return
            if ord(character) < 0x20:
                raise ResultEnvelopeProtocolError
            self._append_value_character(character)
            return

        if self._state is _State.IN_ESCAPE:
            if self._pending_high_surrogate is not None and character != "u":
                raise ResultEnvelopeProtocolError
            escapes = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if character == "u":
                self._unicode_digits = ""
                self._state = _State.IN_UNICODE
                return
            decoded = escapes.get(character)
            if decoded is None:
                raise ResultEnvelopeProtocolError
            self._append_value_character(decoded)
            self._state = _State.IN_STRING
            return

        if self._state is _State.IN_UNICODE:
            if character not in _HEX_DIGITS:
                raise ResultEnvelopeProtocolError
            self._unicode_digits += character
            if len(self._unicode_digits) < 4:
                return
            code_unit = int(self._unicode_digits, 16)
            self._unicode_digits = ""
            if self._pending_high_surrogate is not None:
                if not 0xDC00 <= code_unit <= 0xDFFF:
                    raise ResultEnvelopeProtocolError
                high = self._pending_high_surrogate
                self._pending_high_surrogate = None
                code_point = 0x10000 + ((high - 0xD800) << 10) + (code_unit - 0xDC00)
                self._append_value_character(chr(code_point))
            elif 0xD800 <= code_unit <= 0xDBFF:
                self._pending_high_surrogate = code_unit
            elif 0xDC00 <= code_unit <= 0xDFFF:
                raise ResultEnvelopeProtocolError
            else:
                self._append_value_character(chr(code_unit))
            self._state = _State.IN_STRING
            return

        if self._state is _State.IN_NULL:
            literal = "null"
            if self._null_offset >= len(literal) or character != literal[self._null_offset]:
                raise ResultEnvelopeProtocolError
            self._null_offset += 1
            if self._null_offset == len(literal):
                self._finish_value(None)
            return

        if self._state is _State.AFTER_VALUE:
            if character.isspace():
                return
            if character == ",":
                self._state = _State.BEFORE_KEY
                return
            if character == "}":
                if len(self._members) != len(_FIELD_ORDER):
                    raise ResultEnvelopeProtocolError
                self._state = _State.AFTER_OBJECT
                return
            raise ResultEnvelopeProtocolError

        if self._state is _State.AFTER_OBJECT:
            if not character.isspace():
                raise ResultEnvelopeProtocolError
            return

        raise ResultEnvelopeProtocolError

    def _append_value_character(self, character: str) -> None:
        if character == "\x00":
            raise ResultEnvelopeProtocolError
        self._string_bytes += len(character.encode("utf-8"))
        if self._current_key == "schema":
            maximum = len(RESULT_ENVELOPE_SCHEMA)
        elif self._current_key == "speech":
            maximum = self._maximum_speech_bytes
        elif self._current_key == "display":
            maximum = self._maximum_display_bytes
        else:
            raise ResultEnvelopeProtocolError
        if self._string_bytes > maximum:
            raise ResultEnvelopeLimitError
        self._string_chars.append(character)
        if self._current_key == "display":
            self._display_scratch.append(character)

    def _finish_value(self, value: str | None) -> None:
        key = self._current_key
        if key is None:
            raise ResultEnvelopeProtocolError
        if key != "speech" and value is None:
            raise ResultEnvelopeProtocolError
        if key in self._canonical:
            raise ResultEnvelopeProtocolError
        self._canonical[key] = value
        if key == "schema" and value != RESULT_ENVELOPE_SCHEMA:
            raise ResultEnvelopeProtocolError
        self._members.append((key, value))
        self._current_key = None
        self._string_chars = []
        self._state = _State.AFTER_VALUE

    def _required_string(self, key: str) -> str:
        value = self._canonical.get(key)
        if not isinstance(value, str):
            raise ResultEnvelopeProtocolError
        return value

    def _optional_string(self, key: str) -> str | None:
        value = self._canonical.get(key)
        if value is not None and not isinstance(value, str):
            raise ResultEnvelopeProtocolError
        return value


__all__ = [
    "MAX_RESULT_DISPLAY_BYTES",
    "MAX_RESULT_ENVELOPE_BYTES",
    "MAX_RESULT_SPEECH_BYTES",
    "ResultEnvelopeLimitError",
    "ResultEnvelopeProtocolError",
    "ResultEnvelopeStreamParser",
    "build_result_envelope_prompt",
    "parse_result_envelope",
]
