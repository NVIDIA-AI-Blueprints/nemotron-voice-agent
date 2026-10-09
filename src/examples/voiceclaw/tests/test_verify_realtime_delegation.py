# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused protocol tests for the public Realtime delegation verifier."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import wave
from pathlib import Path

import pytest

from voiceclaw.domain import (
    ResponseOnlyRequestState,
    ResponseOnlySpeechSource,
    ResponseOnlyTargetAvailability,
)


def _load_verifier():
    path = Path(__file__).parents[1] / "scripts" / "verify-realtime-delegation.py"
    spec = importlib.util.spec_from_file_location("voiceclaw_realtime_delegation_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_VERIFIER = _load_verifier()


def _receipt_metadata(*, receipt_id: str = "receipt-vc-1", presentation_id: str = "presentation-vc-1"):
    return {
        "voiceclaw_presentation_id": presentation_id,
        "voiceclaw_playback_receipt": "conversation.item.truncate.v1",
        "voiceclaw_playback_receipt_id": receipt_id,
        "voiceclaw_playback_receipt_required": "true",
    }


def _ready_receipt_tracker():
    tracker = _VERIFIER._PlaybackReceiptTracker()
    metadata = _receipt_metadata()
    tracker.register_response("response-public", metadata)
    tracker.validate_terminal_metadata("response-public", metadata)
    tracker.observe_audio_delta(
        {
            "response_id": "response-public",
            "item_id": "item-public",
            "content_index": 0,
            "delta": "AAAA",
        }
    )
    tracker.observe_audio_done(
        {
            "response_id": "response-public",
            "item_id": "item-public",
            "content_index": 0,
        }
    )
    return tracker


@pytest.mark.parametrize("url", ["ws://127.0.0.1:18790/v1/realtime", "ws://[::1]:18790/v1/realtime"])
def test_plaintext_realtime_verification_is_explicitly_limited_to_literal_loopback(url: str) -> None:
    """The artifact flag allows only literal loopback WebSockets."""
    assert _VERIFIER._validate_url(url, allow_loopback_ws=True) == url


@pytest.mark.parametrize(
    "url",
    [
        "ws://localhost:18790/v1/realtime",
        "ws://192.168.1.20:18790/v1/realtime",
        "ws://user:secret@127.0.0.1:18790/v1/realtime",
        "http://127.0.0.1:18790/v1/realtime",
    ],
)
def test_plaintext_realtime_verification_rejects_nonliteral_or_credentialed_urls(url: str) -> None:
    """The artifact flag cannot authorize DNS, remote, or credentialed URLs."""
    with pytest.raises(ValueError):
        _VERIFIER._validate_url(url, allow_loopback_ws=True)


def test_plaintext_realtime_verification_requires_an_explicit_flag() -> None:
    """Plaintext verification remains opt-in even for literal loopback."""
    with pytest.raises(ValueError, match="allow-loopback-ws"):
        _VERIFIER._validate_url("ws://127.0.0.1:18790/v1/realtime")


def test_request_summary_is_projection_content_not_request_correlation() -> None:
    """The display summary must not become stable request identity."""
    projection = {
        "schema": "voiceclaw.projection.v2",
        "session_id": "session-1",
        "kind": "backend_turn",
        "phase": "locally_queued",
        "title": "Request queued locally",
        "request_summary": "Standalone delegated goal",
        "local_request_id": "request-1",
    }

    assert _VERIFIER._projection_correlation(projection) == {"local_request_id": "request-1"}


def test_zero_playback_receipt_uses_server_token_and_waits_for_matching_ack() -> None:
    """The generated standard event remains pending until its exact acknowledgement."""
    tracker = _ready_receipt_tracker()

    event = tracker.zero_playback_event("response-public")

    assert event == {
        "event_id": "receipt-vc-1",
        "type": "conversation.item.truncate",
        "item_id": "item-public",
        "content_index": 0,
        "audio_end_ms": 0,
    }
    assert tracker.all_acknowledged({"response-public"}) is False
    assert (
        tracker.acknowledge(
            {
                "type": "conversation.item.truncated",
                "item_id": "item-public",
                "content_index": 0,
                "audio_end_ms": 0,
            }
        )
        == "response-public"
    )
    assert tracker.all_acknowledged({"response-public"}) is True
    assert tracker.acknowledged_count == 1


def test_send_preserves_the_server_receipt_token_as_standard_event_id() -> None:
    """The opaque server token must authorize the truncate as its event identity."""

    class _WebSocket:
        sent: str | None = None

        async def send(self, message: str) -> None:
            self.sent = message

    websocket = _WebSocket()
    asyncio.run(_VERIFIER._send(websocket, _ready_receipt_tracker().zero_playback_event("response-public")))

    assert websocket.sent is not None
    assert json.loads(websocket.sent)["event_id"] == "receipt-vc-1"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {**_receipt_metadata(), "voiceclaw_playback_receipt": "custom.receipt.v1"},
        {**_receipt_metadata(), "voiceclaw_playback_receipt_required": True},
        {key: value for key, value in _receipt_metadata().items() if key != "voiceclaw_playback_receipt_id"},
        _receipt_metadata(receipt_id=""),
        _receipt_metadata(receipt_id="receipt\ninvalid"),
        _receipt_metadata(receipt_id="r" * 513),
        {key: value for key, value in _receipt_metadata().items() if key != "voiceclaw_presentation_id"},
    ],
)
def test_playback_receipt_rejects_missing_or_malformed_metadata(metadata: object) -> None:
    """Every server-owned receipt field is mandatory and strictly typed."""
    with pytest.raises(RuntimeError, match="playback receipt"):
        _VERIFIER._PlaybackReceiptTracker().register_response("response-public", metadata)


def test_playback_receipt_rejects_duplicate_response_or_reused_token() -> None:
    """Receipt authority cannot be duplicated, reused, or changed at completion."""
    tracker = _VERIFIER._PlaybackReceiptTracker()
    metadata = _receipt_metadata()
    tracker.register_response("response-one", metadata)
    tracker.validate_terminal_metadata("response-one", metadata)

    with pytest.raises(RuntimeError, match="duplicate playback receipt metadata"):
        tracker.register_response("response-one", metadata)
    with pytest.raises(RuntimeError, match="reused a playback receipt token"):
        tracker.register_response("response-two", metadata)
    with pytest.raises(RuntimeError, match="changed its playback receipt metadata"):
        tracker.validate_terminal_metadata("response-one", _receipt_metadata(receipt_id="receipt-vc-changed"))


@pytest.mark.parametrize(
    "acknowledgement",
    [
        {"item_id": "wrong-item", "content_index": 0, "audio_end_ms": 0},
        {"item_id": "item-public", "content_index": 1, "audio_end_ms": 0},
        {"item_id": "item-public", "content_index": 0, "audio_end_ms": 1},
    ],
)
def test_playback_receipt_rejects_mismatched_acknowledgement(acknowledgement: dict[str, object]) -> None:
    """Only the exact public item, content part, and zero boundary can close a receipt."""
    tracker = _ready_receipt_tracker()
    tracker.zero_playback_event("response-public")

    with pytest.raises(RuntimeError, match="playback|zero-playback"):
        tracker.acknowledge(acknowledgement)


def test_playback_receipt_rejects_duplicate_acknowledgement() -> None:
    """One standard truncation acknowledgement can close a receipt only once."""
    tracker = _ready_receipt_tracker()
    tracker.zero_playback_event("response-public")
    acknowledgement = {"item_id": "item-public", "content_index": 0, "audio_end_ms": 0}
    tracker.acknowledge(acknowledgement)

    with pytest.raises(RuntimeError, match="more than once"):
        tracker.acknowledge(acknowledgement)


def test_json_decoder_rejects_duplicate_receipt_metadata_fields() -> None:
    """Duplicate raw JSON receipt keys cannot be hidden by last-value-wins parsing."""
    raw = '{"voiceclaw_playback_receipt_id":"first","voiceclaw_playback_receipt_id":"second"}'

    with pytest.raises(RuntimeError, match="duplicate playback receipt metadata"):
        json.loads(raw, object_pairs_hook=_VERIFIER._json_object_without_duplicate_receipt_metadata)


def test_display_only_result_completes_without_result_delivery_speech() -> None:
    """A null envelope speech field must not make verification wait for speech."""
    assert _VERIFIER._delivery_completion(
        speech_source=ResponseOnlySpeechSource.NONE,
        succeeded_at=20,
        acknowledgement_completed_at=18,
        result_delivery_response_id=None,
        result_delivery_completed_at=None,
        queue_depth_transitions=[(4, 1), (21, 0)],
    ) == (21, 0)


def test_display_only_result_accepts_an_acknowledgement_queue_drained_before_backend_success() -> None:
    """A display-only result creates no post-success speech transition to await."""
    assert _VERIFIER._delivery_completion(
        speech_source=ResponseOnlySpeechSource.NONE,
        succeeded_at=20,
        acknowledgement_completed_at=18,
        result_delivery_response_id=None,
        result_delivery_completed_at=None,
        queue_depth_transitions=[(4, 1), (19, 0)],
    ) == (19, 0)


def test_verifier_default_timeout_covers_the_default_backend_invocation_budget() -> None:
    """The CLI budget must not expire before the adapter's default invocation deadline."""
    arguments = _VERIFIER._parser().parse_args(["--query", "delegate this"])

    assert arguments.timeout == 420.0


def test_verifier_accepts_repeatable_content_assertions_and_an_expected_outcome() -> None:
    """CLI qualification can assert both channels without changing their casing."""
    arguments = _VERIFIER._parser().parse_args(
        [
            "--query",
            "delegate this",
            "--expect-outcome",
            "outcome_unknown",
            "--display-contains",
            "First Marker",
            "--display-contains",
            "second marker",
            "--speech-contains",
            "Spoken Marker",
        ]
    )

    assert arguments.expect_outcome == ResponseOnlyRequestState.OUTCOME_UNKNOWN.value
    assert arguments.display_contains == ["First Marker", "second marker"]
    assert arguments.speech_contains == ["Spoken Marker"]


def test_content_assertions_are_case_insensitive_and_require_every_marker() -> None:
    """Each configured marker must occur in its selected terminal channel."""
    _VERIFIER._require_content(
        "# Result\nThe ALPHA and beta facts are ready.",
        ["alpha", "BETA facts"],
        channel="display",
    )

    with pytest.raises(RuntimeError, match="omitted required content"):
        _VERIFIER._require_content("Only alpha is present.", ["alpha", "beta"], channel="display")
    with pytest.raises(RuntimeError, match="omitted required content"):
        _VERIFIER._require_content(None, ["spoken marker"], channel="speech")


def test_latency_uses_monotonic_nanoseconds_and_never_accepts_time_reversal() -> None:
    """Reported latency is an elapsed generation measurement, not wall-clock time."""
    assert _VERIFIER._latency_ms(1_000_000_000, 1_012_345_678) == 12.346
    assert _VERIFIER._latency_ms(None, 1_012_345_678) is None
    with pytest.raises(RuntimeError, match="timeline moved backwards"):
        _VERIFIER._latency_ms(20, 19)


def test_target_contract_validates_initial_binding_and_terminal_tool_withdrawal() -> None:
    """The verifier proves both ends of the one-shot capability transition."""
    projection = {
        "backend_name": "NemoClaw",
        "backend_mode": "response_only",
        "target_ref": "workspace/sandbox/agent",
    }
    initial = {
        "backend": "NemoClaw",
        "mode": "response_only",
        "target": "workspace/sandbox/agent",
        "agent_readiness": "unknown",
        "capabilities": ["work.submit"],
        "frontend_tools": ["work.delegate"],
        "durability": "none",
        "event_delivery": "response_only",
        "max_parallel_work": 1,
        "context_continuity": "unqualified",
        "target_state": "available",
    }
    terminal = {
        **initial,
        "capabilities": [],
        "frontend_tools": [],
        "target_state": "consumed",
    }

    assert (
        _VERIFIER._target_contract_payload(
            json.dumps(initial),
            projection,
            expected_state=ResponseOnlyTargetAvailability.AVAILABLE,
            expected_capabilities=frozenset({"work.submit"}),
            expected_tools=frozenset({"work.delegate"}),
        )["target_state"]
        == "available"
    )
    assert (
        _VERIFIER._target_contract_payload(
            json.dumps(terminal),
            projection,
            expected_state=ResponseOnlyTargetAvailability.CONSUMED,
            expected_capabilities=frozenset(),
            expected_tools=frozenset(),
        )["frontend_tools"]
        == []
    )


def test_shared_target_contract_keeps_delegation_available_only_when_explicitly_selected() -> None:
    """A live shared target must advertise the explicitly selected continuation contract."""
    projection = {"backend_name": "NemoClaw", "backend_mode": "response_only", "target_ref": "workspace/sandbox/agent"}
    payload = {
        "backend": "NemoClaw",
        "mode": "response_only",
        "target": "workspace/sandbox/agent",
        "agent_readiness": "unknown",
        "capabilities": ["work.submit"],
        "frontend_tools": ["work.delegate"],
        "durability": "none",
        "event_delivery": "response_only",
        "max_parallel_work": 1,
        "context_continuity": "shared_target",
        "target_state": "available",
    }
    parameters = dict(
        expected_state=ResponseOnlyTargetAvailability.AVAILABLE,
        expected_capabilities=frozenset({"work.submit"}),
        expected_tools=frozenset({"work.delegate"}),
    )
    with pytest.raises(RuntimeError, match="qualified response-only contract"):
        _VERIFIER._target_contract_payload(json.dumps(payload), projection, **parameters)
    assert (
        _VERIFIER._target_contract_payload(
            json.dumps(payload), projection, expected_continuity="shared_target", **parameters
        )
        == payload
    )


def test_target_contract_rejects_a_consumed_target_that_still_advertises_delegation() -> None:
    """A terminal one-shot target cannot leave a stale delegate action visible."""
    projection = {
        "backend_name": "NemoClaw",
        "backend_mode": "response_only",
        "target_ref": "workspace/sandbox/agent",
    }
    stale = {
        "backend": "NemoClaw",
        "mode": "response_only",
        "target": "workspace/sandbox/agent",
        "agent_readiness": "unknown",
        "capabilities": ["work.submit"],
        "frontend_tools": ["work.delegate"],
        "durability": "none",
        "event_delivery": "response_only",
        "max_parallel_work": 1,
        "context_continuity": "unqualified",
        "target_state": "consumed",
    }

    with pytest.raises(RuntimeError, match="expected response-only contract"):
        _VERIFIER._target_contract_payload(
            json.dumps(stale),
            projection,
            expected_state=ResponseOnlyTargetAvailability.CONSUMED,
            expected_capabilities=frozenset(),
            expected_tools=frozenset(),
        )


def test_failure_delivery_waits_for_generated_speech_and_queue_drain() -> None:
    """A terminal failure is complete only after its model-mediated speech drains."""
    assert (
        _VERIFIER._failure_delivery_completion(
            terminal_at=20,
            acknowledgement_completed_at=18,
            failure_delivery_completed_at=24,
            queue_depth_transitions=[(4, 1), (23, 1)],
        )
        is None
    )
    assert _VERIFIER._failure_delivery_completion(
        terminal_at=20,
        acknowledgement_completed_at=18,
        failure_delivery_completed_at=24,
        queue_depth_transitions=[(4, 1), (23, 1), (25, 0)],
    ) == (25, 0)


def test_backend_authored_speech_waits_for_delivery_and_queue_drain() -> None:
    """Advertised result speech remains part of the verified delivery lifecycle."""
    pending = _VERIFIER._delivery_completion(
        speech_source=ResponseOnlySpeechSource.BACKEND_AUTHORED,
        succeeded_at=20,
        acknowledgement_completed_at=18,
        result_delivery_response_id="response-speech",
        result_delivery_completed_at=None,
        queue_depth_transitions=[(4, 1), (21, 1)],
    )
    assert pending is None

    completed = _VERIFIER._delivery_completion(
        speech_source=ResponseOnlySpeechSource.BACKEND_AUTHORED,
        succeeded_at=20,
        acknowledgement_completed_at=18,
        result_delivery_response_id="response-speech",
        result_delivery_completed_at=24,
        queue_depth_transitions=[(4, 1), (21, 1), (25, 0)],
    )
    assert completed == (25, 0)


def test_display_only_result_rejects_an_unadvertised_speech_delivery() -> None:
    """Delivery behavior must agree with the typed terminal speech source."""
    with pytest.raises(RuntimeError, match="display-only"):
        _VERIFIER._delivery_completion(
            speech_source=ResponseOnlySpeechSource.NONE,
            succeeded_at=20,
            acknowledgement_completed_at=18,
            result_delivery_response_id="response-speech",
            result_delivery_completed_at=24,
            queue_depth_transitions=[(4, 1), (25, 0)],
        )


def test_noncompleted_result_delivery_reports_standard_terminal_details() -> None:
    """Incomplete speech should identify the bounded standard terminal reason."""
    message = _VERIFIER._noncompleted_frontend_response_message(
        {
            "status": "incomplete",
            "status_details": {"type": "incomplete", "reason": "max_output_tokens"},
        },
        speech_purpose="result_delivery",
    )

    expected = " ".join(
        (
            "the frontend result delivery did not complete",
            "(status=incomplete, type=incomplete, reason=max_output_tokens)",
        )
    )
    assert message == expected


def test_noncompleted_response_diagnostic_does_not_echo_unbounded_prose() -> None:
    """Untrusted prose must not be reflected by the verification diagnostic."""
    unsafe_reason = "secret\n" + ("x" * 1_000)
    message = _VERIFIER._noncompleted_frontend_response_message(
        {
            "status": "failed",
            "status_details": {"type": "failed", "reason": unsafe_reason},
        },
        speech_purpose="failure_delivery",
    )

    assert message == "the frontend failure delivery did not complete (status=failed, type=failed)"
    assert "secret" not in message


def _recording(path: Path, *, rate: int = 24_000, channels: int = 1, width: int = 2, frames: int = 960) -> bytes:
    pcm = b"\x01" * (frames * channels * width)
    with wave.open(str(path), "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(width)
        recording.setframerate(rate)
        recording.writeframes(pcm)
    return pcm


def _voice_arguments(tmp_path: Path, *extra: str):
    recording = tmp_path / "input.wav"
    _recording(recording)
    return _VERIFIER._parser().parse_args(["--input-audio-wav", str(recording), *extra])


def test_verifier_requires_exactly_one_text_or_recorded_voice_input(tmp_path: Path) -> None:
    """Choose one input channel without permitting a typed fallback for audio."""
    parser = _VERIFIER._parser()
    assert parser.parse_args(["--query", "Use the agent"]).input_audio_wav is None
    arguments = _voice_arguments(tmp_path, "--transcript-contains", "agent")
    assert arguments.query is None
    assert arguments.transcript_contains == ["agent"]
    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--query", "Use the agent", "--input-audio-wav", "input.wav"])


@pytest.mark.parametrize(
    "recording_options",
    [{"rate": 16_000}, {"channels": 2}, {"width": 1}, {"frames": 0}, {"frames": 24_000 * 60 + 1}],
)
def test_voice_input_rejects_unsupported_or_unbounded_wav_before_connecting(tmp_path: Path, recording_options) -> None:
    """Reject formats and durations the public PCM stream cannot represent."""
    arguments = _voice_arguments(tmp_path)
    _recording(arguments.input_audio_wav, **recording_options)
    with pytest.raises(ValueError, match="PCM16|60 seconds"):
        _VERIFIER._VoiceProbe(arguments)


def test_voice_input_rejects_truncated_pcm(tmp_path: Path) -> None:
    """The WAV header cannot substitute for the actual recorded samples."""
    arguments = _voice_arguments(tmp_path)
    data = arguments.input_audio_wav.read_bytes()
    arguments.input_audio_wav.write_bytes(data[:-2])
    with pytest.raises(ValueError, match="truncated"):
        _VERIFIER._VoiceProbe(arguments)


@pytest.mark.parametrize(
    "turn_detection", [None, {"type": "semantic_vad"}, {"type": "server_vad", "create_response": False}]
)
def test_recorded_voice_uses_audio_only_and_negotiated_response_creation(tmp_path: Path, turn_detection) -> None:
    """Preserve each supported commit/create mode and the original PCM."""
    arguments = _voice_arguments(tmp_path, "--transcript-contains", "agent")
    probe = _VERIFIER._VoiceProbe(arguments)
    events = []

    class _Socket:
        async def send(self, raw: str) -> None:
            events.append(json.loads(raw))

    async def submit() -> None:
        async with probe:
            await probe.submit(_Socket(), turn_detection)
            while not probe.input_sent:
                await asyncio.sleep(0.001)
            needs_response = probe.observe_input({"type": "input_audio_buffer.committed", "item_id": "audio-1"})
            if needs_response:
                await _VERIFIER._send(_Socket(), {"type": "response.create", "response": {}})
            probe.observe_input(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "audio-1",
                    "transcript": "Use the agent",
                }
            )
            await probe.sender

    asyncio.run(submit())
    assert not any(event["type"] == "conversation.item.create" for event in events)
    input_pcm = b"".join(
        base64.b64decode(event["audio"]) for event in events if event["type"] == "input_audio_buffer.append"
    )
    assert input_pcm.startswith(b"\x01" * 1920)
    assert sum(event["type"] == "input_audio_buffer.commit" for event in events) == (turn_detection is None)
    assert sum(event["type"] == "response.create" for event in events) == (
        turn_detection is None or turn_detection.get("create_response") is False
    )
    result = probe.finish({"status": "succeeded"})
    assert result["query"] == "Use the agent"
    assert result["input"]["kind"] == "recorded_audio"
    assert result["input"]["duration_ms"] == 40


@pytest.mark.parametrize(
    "transcription",
    [
        {"item_id": "different", "transcript": "Use the agent"},
        {"item_id": "audio-1", "transcript": ""},
        {"item_id": "audio-1", "transcript": "Hello there"},
    ],
)
def test_voice_success_requires_nonempty_correlated_transcription_with_expected_content(
    tmp_path: Path, transcription
) -> None:
    """An unrelated, empty, or incorrect ASR result cannot establish voice input."""
    probe = _VERIFIER._VoiceProbe(_voice_arguments(tmp_path, "--transcript-contains", "agent"))
    probe.input_sent = True
    with pytest.raises(RuntimeError, match="transcription|transcript"):
        probe.observe_input({"type": "input_audio_buffer.committed", "item_id": "audio-1"})
        probe.observe_input({"type": "conversation.item.input_audio_transcription.completed", **transcription})
        probe.finish({"status": "succeeded"})


def test_voice_probe_rejects_early_or_multiple_commits_and_transcription_failure(tmp_path: Path) -> None:
    """Reject segmented or failed recordings instead of replaying a native target."""
    probe = _VERIFIER._VoiceProbe(_voice_arguments(tmp_path))
    with pytest.raises(RuntimeError, match="before the recording"):
        probe.observe_input({"type": "input_audio_buffer.committed", "item_id": "early"})
    probe.input_sent = True
    probe.observe_input({"type": "input_audio_buffer.committed", "item_id": "audio-1"})
    with pytest.raises(RuntimeError, match="more than one"):
        probe.observe_input({"type": "input_audio_buffer.committed", "item_id": "audio-2"})
    with pytest.raises(RuntimeError, match="transcription failed"):
        probe.observe_input({"type": "conversation.item.input_audio_transcription.failed", "item_id": "audio-1"})


def test_requested_output_captures_only_result_audio_in_a_private_wav(tmp_path: Path) -> None:
    """Acknowledgement PCM must never stand in for the agent result recording."""
    output = tmp_path / "result.wav"
    arguments = _VERIFIER._parser().parse_args(["--query", "Use the agent", "--output-audio-wav", str(output)])
    probe = _VERIFIER._VoiceProbe(arguments)
    probe.observe_audio({"response_id": "ack", "delta": "AAAA"}, "result-1")
    pcm = b"\x01\x00\x02\x00" * 240
    probe.observe_audio({"response_id": "result-1", "delta": base64.b64encode(pcm).decode()}, "result-1")
    result = probe.finish({"status": "succeeded", "speech_source": "backend_authored", "result_speech_generated": True})
    with wave.open(str(output), "rb") as recording:
        assert recording.getparams()[:3] == (1, 2, 24_000)
        assert recording.readframes(recording.getnframes()) == pcm
    assert os.stat(output).st_mode & 0o777 == 0o600
    assert result["result_audio_wav"]["response_id"] == "result-1"
    assert result["result_audio_wav"]["duration_ms"] == 20


@pytest.mark.parametrize("speech_source", ["none", "backend_authored"])
def test_requested_result_recording_cannot_pass_without_result_pcm(tmp_path: Path, speech_source: str) -> None:
    """A declared speech source or display-only result cannot substitute for PCM."""
    output = tmp_path / "result.wav"
    probe = _VERIFIER._VoiceProbe(
        _VERIFIER._parser().parse_args(["--query", "Use the agent", "--output-audio-wav", str(output)])
    )
    with pytest.raises(RuntimeError, match="result audio"):
        probe.finish({"status": "succeeded", "speech_source": speech_source})
    assert not output.exists()


@pytest.mark.parametrize("delta", ["not base64!", base64.b64encode(b"\x01").decode()])
def test_result_recording_rejects_malformed_pcm(tmp_path: Path, delta: str) -> None:
    """Invalid encoding and partial samples cannot produce a successful WAV."""
    output = tmp_path / "result.wav"
    probe = _VERIFIER._VoiceProbe(
        _VERIFIER._parser().parse_args(["--query", "Use the agent", "--output-audio-wav", str(output)])
    )
    with pytest.raises(RuntimeError, match="result audio"):
        probe.observe_audio({"response_id": "result-1", "delta": delta}, "result-1")
        probe.finish({"status": "succeeded", "speech_source": "backend_authored"})
    assert not output.exists()


def test_result_recording_refuses_existing_files_before_dispatch_and_at_write(tmp_path: Path) -> None:
    """Preserve existing recordings even when a file appears during delegation."""
    output = tmp_path / "result.wav"
    arguments = _VERIFIER._parser().parse_args(["--query", "Use the agent", "--output-audio-wav", str(output)])
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        _VERIFIER._VoiceProbe(arguments)
    output.unlink()
    probe = _VERIFIER._VoiceProbe(arguments)
    probe.observe_audio({"response_id": "result-1", "delta": "AAAAAA=="}, "result-1")
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        probe.finish({"status": "succeeded", "speech_source": "backend_authored"})
    assert output.read_bytes() == b"keep"


@pytest.mark.parametrize("frontend_only", [False, True])
def test_full_verifier_requires_agent_lifecycle_after_recorded_audio(
    tmp_path: Path, monkeypatch, frontend_only: bool
) -> None:
    """Exercise the public event loop; synthetic events are protocol evidence only."""
    output = tmp_path / "result.wav"
    arguments = _voice_arguments(
        tmp_path,
        "--transcript-contains",
        "balanced",
        "--display-contains",
        "balanced",
        "--speech-contains",
        "balanced",
        "--output-audio-wav",
        str(output),
        "--timeout",
        "2",
    )
    session = {"id": "session-1", "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}}
    base = {"local_request_id": "request-1"}
    result_identity = {**base, "backend_session_id": "backend-1", "turn_id": "turn-1", "response_id": "native-result-1"}
    binding = {"backend_name": "NemoClaw", "backend_mode": "response_only", "target_ref": "workspace/sandbox/main"}
    target = {
        "backend": "NemoClaw",
        "mode": "response_only",
        "target": "workspace/sandbox/main",
        "agent_readiness": "unknown",
        "capabilities": ["work.submit"],
        "frontend_tools": ["work.delegate"],
        "durability": "none",
        "event_delivery": "response_only",
        "max_parallel_work": 1,
        "context_continuity": "unqualified",
        "target_state": "available",
    }

    def metadata(kind: str, phase: str, **correlation):
        return {
            f"voiceclaw_{key}": value
            for key, value in {
                "schema": _VERIFIER.REALTIME_PROJECTION_SCHEMA,
                "session_id": "session-1",
                "kind": kind,
                "phase": phase,
                "title": phase,
                **correlation,
            }.items()
        }

    def response(event_type: str, response_id: str, meta: dict, text: str, *, audio: bool = False):
        return {
            "type": event_type,
            "response": {
                "id": response_id,
                "metadata": meta,
                "conversation_id": None,
                "status": "completed" if event_type == "response.done" else "in_progress",
                "output_modalities": ["audio" if audio else "text"],
                "output": []
                if event_type == "response.created"
                else [
                    {
                        "content": [
                            {
                                "type": "output_audio" if audio else "output_text",
                                "transcript" if audio else "text": text,
                            }
                        ]
                    }
                ],
            },
        }

    def projection(kind: str, phase: str, text: str = "", **correlation):
        return response("response.done", f"{kind}-{phase}-{text}", metadata(kind, phase, **correlation), text)

    pcm = b"\x01\x00" * 480

    def speech(response_id: str, purpose: str, text: str):
        meta = {
            **_receipt_metadata(receipt_id=f"receipt-{response_id}", presentation_id=f"presentation-{response_id}"),
            "voiceclaw_speech_purpose": purpose,
        }
        if purpose == "result_delivery":
            meta["voiceclaw_delivery"] = "true"
        return [
            response("response.created", response_id, meta, "", audio=True),
            {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "item_id": f"item-{response_id}",
                "content_index": 0,
                "delta": base64.b64encode(pcm if purpose == "result_delivery" else b"\x00\x00").decode(),
            },
            {
                "type": "response.output_audio.done",
                "response_id": response_id,
                "item_id": f"item-{response_id}",
                "content_index": 0,
            },
            response("response.done", response_id, meta, text, audio=True),
        ]

    display = "Balanced trees provide logarithmic lookups."
    delegated_events = [
        projection("delivery_queue", "queued", json.dumps({"queue_depth": 1})),
        *(
            projection("backend_turn", phase, **base)
            for phase in ("locally_queued", "dispatching", "waiting_for_response")
        ),
        *speech("ack-1", "delegation_ack", "I will ask the connected agent."),
        projection("delivery_queue", "drained", json.dumps({"queue_depth": 0})),
        response("response.created", "display-1", metadata("result_display", "display_delta", **result_identity), ""),
        {"type": "response.output_text.delta", "response_id": "display-1", "delta": display},
        {"type": "response.output_text.done", "response_id": "display-1", "text": display},
        response("response.done", "display-1", metadata("result_display", "completed", **result_identity), display),
        projection("backend_turn", "succeeded", **result_identity, speech_source="backend_authored"),
        projection(
            "backend_target",
            "verified",
            json.dumps({**target, "capabilities": [], "frontend_tools": [], "target_state": "consumed"}),
            **binding,
        ),
        projection("delivery_queue", "queued", json.dumps({"queue_depth": 1})),
        *speech("result-1", "result_delivery", "Balanced trees provide logarithmic lookups."),
        projection("delivery_queue", "drained", json.dumps({"queue_depth": 0})),
    ]

    class _Socket:
        def __init__(self):
            self.sent = []
            self.queue = asyncio.Queue()
            for event in [
                {"type": "session.created", "session": session},
                {"type": "session.updated", "session": session},
                projection("backend_target", "verified", json.dumps(target), **binding),
            ]:
                self.queue.put_nowait(event)
            self.committed = False

        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if (
                event["type"] == "input_audio_buffer.append"
                and base64.b64decode(event["audio"]) == bytes(960)
                and not self.committed
            ):
                self.committed = True
                for incoming in [
                    {"type": "input_audio_buffer.committed", "item_id": "input-1"},
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "item_id": "input-1",
                        "transcript": "Use the agent to compare balanced trees.",
                    },
                    *(
                        [response("response.created", "frontend-only", {}, "", audio=True)]
                        if frontend_only
                        else delegated_events
                    ),
                ]:
                    self.queue.put_nowait(incoming)
            if event["type"] == "conversation.item.truncate":
                self.queue.put_nowait(
                    {
                        "type": "conversation.item.truncated",
                        **{key: event[key] for key in ("item_id", "content_index", "audio_end_ms")},
                    }
                )

        async def recv(self):
            return json.dumps(await self.queue.get())

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            pass

    socket = _Socket()
    monkeypatch.setattr(_VERIFIER, "connect", lambda *_args, **_kwargs: socket)
    if frontend_only:
        with pytest.raises(RuntimeError, match="unmarked conversational response"):
            asyncio.run(_VERIFIER._verify(arguments))
        assert not output.exists()
    else:
        result = asyncio.run(_VERIFIER._verify(arguments))
        assert result["status"] == "succeeded"
        assert result["query"] == "Use the agent to compare balanced trees."
        assert result["backend_turn_phases"] == ["locally_queued", "dispatching", "waiting_for_response", "succeeded"]
        assert result["backend_response_id"] == "native-result-1"
        assert result["playback_receipts_acknowledged"] == 2
        assert result["audio_played_ms"] == 0
        assert result["result_audio_wav"]["response_id"] == "result-1"
        with wave.open(str(output), "rb") as recording:
            assert recording.readframes(recording.getnframes()) == pcm
    assert not any(
        event["type"] in {"conversation.item.create", "response.create", "input_audio_buffer.commit"}
        for event in socket.sent
    )


def test_voice_sender_failure_interrupts_receive_and_reaps_the_task(tmp_path: Path) -> None:
    """A failed input transport cannot hide behind the overall receive deadline."""
    probe = _VERIFIER._VoiceProbe(_voice_arguments(tmp_path))

    class _Socket:
        async def send(self, _raw):
            raise OSError("input transport closed")

        async def recv(self):
            await asyncio.Event().wait()

    async def run():
        with pytest.raises(OSError, match="input transport closed"):
            async with probe:
                await probe.submit(_Socket(), {"type": "semantic_vad"})
                async with asyncio.timeout(0.5):
                    await probe.receive(_Socket())

    asyncio.run(run())
    assert probe.sender.done()
