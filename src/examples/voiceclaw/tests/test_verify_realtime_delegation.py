# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Focused protocol tests for the public Realtime delegation verifier."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from voiceclaw.domain import ResponseOnlySpeechSource


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
        "schema": "voiceclaw.projection.v1",
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
