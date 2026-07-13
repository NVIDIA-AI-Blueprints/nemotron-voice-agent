# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import pytest

from attachment_store import (
    clear_session_attachments,
    consume_capture_request,
    create_capture_request,
    get_attachment,
    remove_attachment,
    store_attachment,
)


def test_capture_request_is_consumed_once() -> None:
    request_id = create_capture_request("session")
    assert consume_capture_request("session", request_id)
    assert not consume_capture_request("session", request_id)


def test_new_capture_request_invalidates_previous_request() -> None:
    first = create_capture_request("session")
    second = create_capture_request("session")
    assert not consume_capture_request("session", first)
    assert consume_capture_request("session", second)


def test_clearing_session_invalidates_capture_request() -> None:
    request_id = create_capture_request("session")
    clear_session_attachments("session")
    assert not consume_capture_request("session", request_id)


def test_invalid_attachment_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="source must be upload or capture"):
        store_attachment(
            session_id="session",
            kind="image",
            name="capture.jpg",
            content_type="image/jpeg",
            data=b"image",
            source="typo",
        )


def test_remove_attachment_releases_stored_payload() -> None:
    attachment = store_attachment(
        session_id="session",
        kind="image",
        name="capture.jpg",
        content_type="image/jpeg",
        data=b"image",
        source="capture",
    )
    remove_attachment("session", attachment.id)
    assert get_attachment("session", attachment.id) is None
