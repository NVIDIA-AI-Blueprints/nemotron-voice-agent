# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from __future__ import annotations

import json

import pytest

from voiceclaw.adapters.result_envelope import (
    MAX_RESULT_DISPLAY_BYTES,
    MAX_RESULT_SPEECH_BYTES,
    ResultEnvelopeLimitError,
    ResultEnvelopeProtocolError,
    ResultEnvelopeStreamParser,
    build_result_envelope_prompt,
    parse_result_envelope,
)
from voiceclaw.domain.response_only import (
    REALTIME_PROJECTION_SCHEMA,
    RESULT_ENVELOPE_SCHEMA,
    RUNTIME_PROJECTION_SCHEMA,
    ResponseOnlyRequestState,
    ResponseOnlyResultEnvelope,
    ResponseOnlyUpdateKind,
)
from voiceclaw.ports.turns import CommittedTurnDisplayDelta


def _wire(*, speech: str | None = "The tests passed.", display: str = "## Result\n\n`npm test` passed.") -> str:
    return json.dumps(
        {"schema": RESULT_ENVELOPE_SCHEMA, "speech": speech, "display": display},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def test_provider_neutral_protocol_symbols_are_stable() -> None:
    assert REALTIME_PROJECTION_SCHEMA == "voiceclaw.projection.v1"
    assert RUNTIME_PROJECTION_SCHEMA == "voiceclaw.runtime.v4"
    assert ResponseOnlyRequestState.WAITING_FOR_RESPONSE.active
    assert ResponseOnlyRequestState.SUCCEEDED.terminal
    assert ResponseOnlyUpdateKind.RESULT_DISPLAY == "result_display"


def test_typed_display_delta_preserves_whitespace_markdown() -> None:
    event = CommittedTurnDisplayDelta(
        backend_session_id="session-1",
        turn_id="turn-1",
        response_id="response-1",
        sequence=0,
        delta="\n  ",
    )

    assert event.delta == "\n  "


def test_prompt_preserves_exact_goal_and_declares_only_result_schema() -> None:
    goal = '  Preserve "quotes", newlines\n,and Unicode ☃ exactly.  '

    prompt = build_result_envelope_prompt(goal)

    encoded_goal = prompt.split("user_goal_json=", 1)[1].splitlines()[0]
    assert json.loads(encoded_goal) == goal
    assert RESULT_ENVELOPE_SCHEMA in prompt
    assert prompt.index('"schema"') < prompt.index('"speech"') < prompt.index('"display"')
    assert f"at most {MAX_RESULT_DISPLAY_BYTES} UTF-8 bytes" in prompt
    assert "Fit the complete answer within that display budget." in prompt
    assert "Markdown" in prompt
    assert "complete JSON object on one physical line" in prompt
    assert "two characters backslash+n" in prompt
    assert "Never place an actual line break" in prompt
    assert '"display":"# Findings\\n\\nDetailed result."' in prompt
    assert 'After the closing quote of "display", emit the object-closing brace.' in prompt
    assert "Do not use a Markdown code fence" in prompt
    assert prompt.index("user_goal_json=") < prompt.index("Return exactly one JSON object")
    assert prompt.endswith(
        "Return the object directly; do not create files or run tools merely to format or validate this envelope."
    )
    assert "conductor" not in prompt.lower()
    assert "route" not in prompt.lower()


def test_prompt_uses_backend_display_capacity_without_changing_protocol_limit() -> None:
    prompt = build_result_envelope_prompt("Return a result", display_budget_bytes=8192)

    assert "at most 8192 UTF-8 bytes" in prompt
    assert f"at most {MAX_RESULT_DISPLAY_BYTES} UTF-8 bytes" not in prompt


@pytest.mark.parametrize("value", [True, 0, -1, MAX_RESULT_DISPLAY_BYTES + 1, "8192"])
def test_prompt_rejects_invalid_backend_display_capacity(value: object) -> None:
    with pytest.raises(ResultEnvelopeProtocolError):
        build_result_envelope_prompt("Return a result", display_budget_bytes=value)  # type: ignore[arg-type]


def test_terminal_parser_preserves_independent_speech_and_markdown_channels() -> None:
    envelope = parse_result_envelope(_wire())

    assert envelope.schema == RESULT_ENVELOPE_SCHEMA
    assert envelope.speech == "The tests passed."
    assert envelope.display == "## Result\n\n`npm test` passed."


def test_terminal_parser_allows_null_speech_without_deriving_it_from_display() -> None:
    envelope = parse_result_envelope(_wire(speech=None))

    assert envelope.speech is None
    assert envelope.display == "## Result\n\n`npm test` passed."


@pytest.mark.parametrize(
    "text",
    [
        "plain text",
        "",
        "[]",
        '{"schema":"voiceclaw.result.v0","speech":null,"display":"ok"}',
        '{"schema":"voiceclaw.result.v1","display":"ok","speech":null}',
        '{"schema":"voiceclaw.result.v1","speech":null}',
        '{"schema":"voiceclaw.result.v1","speech":null,"extra":"x","display":"ok"}',
        '{"schema":"voiceclaw.result.v1","speech":1,"display":"ok"}',
        '{"schema":"voiceclaw.result.v1","speech":"","display":"ok"}',
        '{"schema":"voiceclaw.result.v1","speech":null,"display":""}',
        '{"schema":"voiceclaw.result.v1","speech":null,"display":"  "}',
        '{"schema":"voiceclaw.result.v1","speech":null,"display":null}',
        '{"schema":"voiceclaw.result.v1","speech":null,"display":"ok"} trailing',
        '```json\n{"schema":"voiceclaw.result.v1","speech":null,"display":"ok"}\n```',
        '{"schema":"voiceclaw.result.v1","speech":null,"display":"ok"}\n'
        '{"schema":"voiceclaw.result.v1","speech":null,"display":"ok"}',
    ],
)
def test_terminal_parser_rejects_fallback_ambiguous_or_invalid_results(text: str) -> None:
    with pytest.raises(ResultEnvelopeProtocolError):
        parse_result_envelope(text)


def test_stream_parser_emits_only_decoded_display_content() -> None:
    display = '## Result\n\nA quote: "yes". Snowman: ☃. Emoji: 🚀\n\t'
    wire = _wire(speech='Ready, "now". ☃', display=display)
    deltas: list[str] = []
    parser = ResultEnvelopeStreamParser(deltas.append)

    display_value_offset = wire.index('"display":') + len('"display":')
    parser.push(wire[:display_value_offset])
    assert deltas == []
    parser.push(wire[display_value_offset:])

    envelope = parser.finish()
    assert "".join(deltas) == display
    assert envelope.display == display
    assert envelope.speech == 'Ready, "now". ☃'


def test_stream_parser_accepts_every_wire_split_position() -> None:
    speech = "Ready, ☃."
    display = '# Result\n\nA quote: "yes" and rocket 🚀.'
    wire = _wire(speech=speech, display=display)

    for split in range(len(wire) + 1):
        deltas: list[str] = []
        parser = ResultEnvelopeStreamParser(deltas.append)
        parser.push(wire[:split])
        parser.push(wire[split:])

        envelope = parser.finish()
        assert "".join(deltas) == display
        assert envelope == ResponseOnlyResultEnvelope(speech=speech, display=display)


def test_stream_parser_handles_every_json_escape_and_surrogate_pair() -> None:
    display = 'quote=" slash=/ backslash=\\ controls=\b\f\n\r\t unicode=A rocket=🚀'
    wire = (
        '{"schema":"voiceclaw.result.v1","speech":null,"display":'
        '"quote=\\" slash=\\/ backslash=\\\\ controls=\\b\\f\\n\\r\\t '
        'unicode=\\u0041 rocket=\\ud83d\\ude80"}'
    )
    deltas: list[str] = []
    parser = ResultEnvelopeStreamParser(deltas.append)

    for character in wire:
        parser.push(character)

    assert parser.finish().display == display
    assert "".join(deltas) == display


@pytest.mark.parametrize(
    "escaped",
    [r"\ud83d", r"\ude80", r"\ud83d\u0041", r"\u12xz", r"\q"],
)
def test_stream_parser_rejects_invalid_escapes_and_surrogates(escaped: str) -> None:
    wire = f'{{"schema":"voiceclaw.result.v1","speech":null,"display":"{escaped}"}}'
    parser = ResultEnvelopeStreamParser(lambda _delta: None)

    with pytest.raises(ResultEnvelopeProtocolError):
        parser.push(wire)


def test_identical_complete_member_replay_is_rejected() -> None:
    first = '"schema":"voiceclaw.result.v1","speech":"Ready.","display":"# Result"'
    wire = "{" + first + "," + first + "}"

    with pytest.raises(ResultEnvelopeProtocolError):
        parse_result_envelope(wire)


def test_more_than_one_complete_member_replay_is_rejected() -> None:
    members = '"schema":"voiceclaw.result.v1","speech":null,"display":"ok"'

    with pytest.raises(ResultEnvelopeProtocolError):
        parse_result_envelope("{" + ",".join((members, members, members)) + "}")


def test_exact_maximum_channels_fit_envelope_bound() -> None:
    values = {
        "schema": RESULT_ENVELOPE_SCHEMA,
        "speech": "\x01" * MAX_RESULT_SPEECH_BYTES,
        "display": "\x01" * MAX_RESULT_DISPLAY_BYTES,
    }
    result = parse_result_envelope(json.dumps(values, separators=(",", ":")))

    assert len(result.speech or "") == MAX_RESULT_SPEECH_BYTES
    assert len(result.display) == MAX_RESULT_DISPLAY_BYTES


@pytest.mark.parametrize(
    "suffix",
    [
        '"schema":"voiceclaw.result.v1"',
        '"schema":"voiceclaw.result.v0","speech":"Ready.","display":"# Result"',
        '"schema":"voiceclaw.result.v1","speech":"Changed.","display":"# Result"',
        '"schema":"voiceclaw.result.v1","speech":"Ready.","display":"Changed"',
    ],
)
def test_partial_or_conflicting_member_replay_fails_closed(suffix: str) -> None:
    first = '"schema":"voiceclaw.result.v1","speech":"Ready.","display":"# Result"'

    with pytest.raises(ResultEnvelopeProtocolError):
        parse_result_envelope("{" + first + "," + suffix + "}")


def test_streamed_display_remains_provisional_until_terminal_validation() -> None:
    deltas: list[str] = []
    parser = ResultEnvelopeStreamParser(deltas.append)

    parser.push('{"schema":"voiceclaw.result.v1","speech":"Do not play.","display":"provisional')

    assert "".join(deltas) == "provisional"
    with pytest.raises(ResultEnvelopeProtocolError):
        parser.finish()


def test_stream_parser_rejects_use_after_finish() -> None:
    parser = ResultEnvelopeStreamParser(lambda _delta: None)
    parser.push(_wire())
    parser.finish()

    with pytest.raises(ResultEnvelopeProtocolError):
        parser.push(" ")
    with pytest.raises(ResultEnvelopeProtocolError):
        parser.finish()


def test_stream_parser_accepts_exact_and_rejects_one_over_utf8_bound() -> None:
    wire = _wire(speech=None, display="☃")
    wire_bytes = len(wire.encode("utf-8"))
    exact = ResultEnvelopeStreamParser(lambda _delta: None, maximum_bytes=wire_bytes)
    exact.push(wire)

    assert exact.finish().display == "☃"

    one_over = ResultEnvelopeStreamParser(lambda _delta: None, maximum_bytes=wire_bytes - 1)
    with pytest.raises(ResultEnvelopeProtocolError):
        one_over.push(wire)


@pytest.mark.parametrize(
    ("channel", "maximum"),
    [
        ("display", MAX_RESULT_DISPLAY_BYTES),
        ("speech", MAX_RESULT_SPEECH_BYTES),
    ],
)
def test_decoded_channel_accepts_exact_and_rejects_one_over_bound(channel: str, maximum: int) -> None:
    exact = {
        "schema": RESULT_ENVELOPE_SCHEMA,
        "speech": "x" * maximum if channel == "speech" else None,
        "display": "x" * maximum if channel == "display" else "ok",
    }
    one_over = {**exact, channel: "x" * (maximum + 1)}

    assert parse_result_envelope(json.dumps(exact, separators=(",", ":")))
    with pytest.raises(ResultEnvelopeLimitError):
        parse_result_envelope(json.dumps(one_over, separators=(",", ":")))
