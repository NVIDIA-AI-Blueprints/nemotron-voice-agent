# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import base64

import pytest

from nemotron_voice_agent.pipecat.runtime.attachment_store import (
    clear_session_attachments,
    create_capture_request,
    latest_attachment,
)
from nemotron_voice_agent.pipecat.runtime.session_uploads import UploadError, UploadReceiver
from nemotron_voice_agent.pipecat.runtime.webcam_frame_store import clear_session_webcam_frames, latest_webcam_frame

SESSION = "upload-session"


@pytest.fixture(autouse=True)
def _clean_stores():
    yield
    clear_session_attachments(SESSION)
    clear_session_webcam_frames(SESSION)


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _send(receiver: UploadReceiver, payload: bytes, *, upload_id="u1", chunk_size=4, **start) -> dict:
    receiver.handle(
        "upload-start",
        {"upload_id": upload_id, "total_bytes": len(payload), "content_type": "image/png", "name": "a.png", **start},
    )
    for seq, offset in enumerate(range(0, len(payload), chunk_size)):
        receiver.handle(
            "upload-chunk", {"upload_id": upload_id, "seq": seq, "data": _b64(payload[offset : offset + chunk_size])}
        )
    return receiver.handle("upload-end", {"upload_id": upload_id})


def test_attachment_round_trip_reassembles_chunks() -> None:
    receiver = UploadReceiver(SESSION, {"attachments", "webcam"})
    result = _send(receiver, b"0123456789", target="attachment", kind="image")
    assert result["status"] == "stored"
    stored = latest_attachment(SESSION)
    assert stored is not None and stored.data == b"0123456789" and stored.source == "upload"


def test_webcam_frame_is_stored() -> None:
    receiver = UploadReceiver(SESSION, {"webcam"})
    _send(receiver, b"\xff\xd8jpeg", target="webcam-frame", content_type="image/jpeg")
    frame = latest_webcam_frame(SESSION)
    assert frame is not None and frame.data == b"\xff\xd8jpeg"


def test_capture_requires_a_live_request_id_and_is_single_use() -> None:
    receiver = UploadReceiver(SESSION, {"webcam"})
    with pytest.raises(UploadError, match="unknown or expired capture request"):
        _send(receiver, b"abcd", target="webcam-capture", request_id="nope")
    request_id = create_capture_request(SESSION)
    result = _send(receiver, b"abcd", upload_id="u2", target="webcam-capture", request_id=request_id)
    assert result["source"] == "capture"
    with pytest.raises(UploadError, match="unknown or expired capture request"):
        _send(receiver, b"abcd", upload_id="u3", target="webcam-capture", request_id=request_id)


def test_capability_is_enforced() -> None:
    receiver = UploadReceiver(SESSION, {"webcam"})
    with pytest.raises(UploadError, match="does not support attachments"):
        receiver.handle("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 1})


@pytest.mark.parametrize(
    ("message", "data", "match"),
    [
        ("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 0}, "positive integer"),
        ("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 101}, "exceeds"),
        ("upload-start", {"upload_id": "u", "target": "bogus", "total_bytes": 1}, "target must be"),
        ("upload-start", {"target": "attachment", "total_bytes": 1}, "upload_id is required"),
        ("upload-chunk", {"upload_id": "ghost", "seq": 0, "data": ""}, "unknown upload_id"),
        ("upload-end", "not-a-dict", "must be an object"),
    ],
)
def test_invalid_messages_are_rejected(message, data, match) -> None:
    receiver = UploadReceiver(SESSION, {"attachments"}, max_bytes=100)
    with pytest.raises(UploadError, match=match):
        receiver.handle(message, data)


def test_out_of_order_and_overflow_chunks_abort_the_upload() -> None:
    receiver = UploadReceiver(SESSION, {"attachments"})
    receiver.handle("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 4})
    with pytest.raises(UploadError, match="out of order"):
        receiver.handle("upload-chunk", {"upload_id": "u", "seq": 1, "data": _b64(b"ab")})
    with pytest.raises(UploadError, match="unknown upload_id"):
        receiver.handle("upload-end", {"upload_id": "u"})
    receiver.handle("upload-start", {"upload_id": "v", "target": "attachment", "total_bytes": 2})
    with pytest.raises(UploadError, match="more bytes"):
        receiver.handle("upload-chunk", {"upload_id": "v", "seq": 0, "data": _b64(b"abc")})


def test_incomplete_upload_is_not_stored() -> None:
    receiver = UploadReceiver(SESSION, {"attachments"})
    receiver.handle("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 8})
    receiver.handle("upload-chunk", {"upload_id": "u", "seq": 0, "data": _b64(b"abcd")})
    with pytest.raises(UploadError, match="incomplete"):
        receiver.handle("upload-end", {"upload_id": "u"})
    assert latest_attachment(SESSION) is None


def test_invalid_base64_and_concurrency_limit() -> None:
    receiver = UploadReceiver(SESSION, {"attachments"}, max_concurrent=1)
    receiver.handle("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 4})
    with pytest.raises(UploadError, match="too many concurrent"):
        receiver.handle("upload-start", {"upload_id": "w", "target": "attachment", "total_bytes": 4})
    with pytest.raises(UploadError, match="valid base64"):
        receiver.handle("upload-chunk", {"upload_id": "u", "seq": 0, "data": "***"})


def test_abort_all_drops_partial_uploads() -> None:
    receiver = UploadReceiver(SESSION, {"attachments"})
    receiver.handle("upload-start", {"upload_id": "u", "target": "attachment", "total_bytes": 4})
    receiver.abort_all()
    with pytest.raises(UploadError, match="unknown upload_id"):
        receiver.handle("upload-chunk", {"upload_id": "u", "seq": 0, "data": _b64(b"abcd")})
