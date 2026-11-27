# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunked media uploads carried over the session's own RTVI connection.

Uploads used to travel on separate HTTP routes keyed by ``session_id``. Those routes only
work when every request reaches the worker that runs the session's bot, which a shared
uvicorn socket does not guarantee. Sending the bytes as RTVI client messages on the
WebRTC data channel or WebSocket puts them in the same process as the bot, so the
in-process stores stay correct at any worker count.

Wire protocol (RTVI client messages; ``data`` shown):

``upload-start``  ``{upload_id, target, name, content_type, total_bytes, kind?, request_id?}``
``upload-chunk``  ``{upload_id, seq, data}``   ``data`` is base64, chunks arrive in order from 0
``upload-end``    ``{upload_id}``              assembles, validates, and stores the payload

``target`` is ``attachment``, ``webcam-frame`` or ``webcam-capture``. The receiver is
transport-agnostic and session-scoped; it returns a JSON-serializable result for the caller
to send back, or raises :class:`UploadError`.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

from nemotron_voice_agent.pipecat.runtime.attachment_store import consume_capture_request, store_attachment
from nemotron_voice_agent.pipecat.runtime.webcam_frame_store import store_webcam_frame

UPLOAD_MESSAGE_TYPES = frozenset({"upload-start", "upload-chunk", "upload-end"})
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_CONCURRENT_UPLOADS = 4

# target -> capability the example must declare in examples_registry.yaml
_TARGET_CAPABILITY = {"attachment": "attachments", "webcam-frame": "webcam", "webcam-capture": "webcam"}


class UploadError(ValueError):
    """An upload message was rejected; the text is safe to show the client."""


@dataclass
class _PendingUpload:
    target: str
    name: str
    content_type: str
    total_bytes: int
    kind: str
    request_id: str
    next_seq: int = 0
    received: int = 0
    chunks: list[bytes] = field(default_factory=list)


class UploadReceiver:
    """Reassembles chunked uploads for one session and hands them to the session stores."""

    def __init__(
        self,
        session_id: str,
        capabilities: Collection[str],
        *,
        max_bytes: int = MAX_UPLOAD_BYTES,
        max_concurrent: int = MAX_CONCURRENT_UPLOADS,
    ) -> None:
        """Bind the receiver to one session; ``capabilities`` come from the example's registry entry."""
        if not session_id.strip():
            raise ValueError("session_id is required")
        self._session_id = session_id.strip()
        self._capabilities = frozenset(capabilities)
        self._max_bytes = max_bytes
        self._max_concurrent = max_concurrent
        self._pending: dict[str, _PendingUpload] = {}
        self._handlers: dict[str, Callable[[dict], dict[str, Any]]] = {
            "upload-start": self._start,
            "upload-chunk": self._chunk,
            "upload-end": self._end,
        }

    @staticmethod
    def handles(message_type: str) -> bool:
        """Whether ``message_type`` belongs to the upload protocol."""
        return message_type in UPLOAD_MESSAGE_TYPES

    def handle(self, message_type: str, data: Any) -> dict[str, Any]:
        """Process one upload message and return its acknowledgement payload."""
        handler = self._handlers.get(message_type)
        if handler is None:
            raise UploadError(f"unknown upload message {message_type!r}")
        if not isinstance(data, dict):
            raise UploadError("upload message data must be an object")
        return handler(data)

    def abort_all(self) -> None:
        """Drop partial uploads, e.g. when the session ends."""
        self._pending.clear()

    # -- protocol steps ---------------------------------------------------------------

    def _start(self, data: dict) -> dict[str, Any]:
        upload_id = self._upload_id(data)
        if upload_id in self._pending:
            raise UploadError("upload_id already in use")
        if len(self._pending) >= self._max_concurrent:
            raise UploadError("too many concurrent uploads")
        target = str(data.get("target", ""))
        capability = _TARGET_CAPABILITY.get(target)
        if capability is None:
            raise UploadError("target must be attachment, webcam-frame or webcam-capture")
        if capability not in self._capabilities:
            raise UploadError(f"session does not support {capability}")
        total = data.get("total_bytes")
        if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
            raise UploadError("total_bytes must be a positive integer")
        if total > self._max_bytes:
            raise UploadError(f"upload exceeds {self._max_bytes // (1024 * 1024)}MB limit")
        request_id = str(data.get("request_id", ""))
        if target == "webcam-capture" and not request_id:
            raise UploadError("request_id is required for webcam-capture")
        self._pending[upload_id] = _PendingUpload(
            target=target,
            name=str(data.get("name", "")),
            content_type=str(data.get("content_type", "")),
            total_bytes=total,
            kind=str(data.get("kind", "image")),
            request_id=request_id,
        )
        return {"upload_id": upload_id, "status": "started"}

    def _chunk(self, data: dict) -> dict[str, Any]:
        upload_id = self._upload_id(data)
        upload = self._pending_upload(upload_id)
        seq = data.get("seq")
        if seq != upload.next_seq:
            self._pending.pop(upload_id, None)
            raise UploadError(f"chunk out of order: expected {upload.next_seq}, got {seq!r}")
        try:
            raw = base64.b64decode(str(data.get("data", "")), validate=True)
        except (binascii.Error, ValueError) as exc:
            self._pending.pop(upload_id, None)
            raise UploadError("chunk data is not valid base64") from exc
        upload.received += len(raw)
        if upload.received > upload.total_bytes:
            self._pending.pop(upload_id, None)
            raise UploadError("received more bytes than total_bytes")
        upload.chunks.append(raw)
        upload.next_seq += 1
        return {"upload_id": upload_id, "status": "chunk", "received": upload.received}

    def _end(self, data: dict) -> dict[str, Any]:
        upload_id = self._upload_id(data)
        upload = self._pending_upload(upload_id)
        del self._pending[upload_id]
        if upload.received != upload.total_bytes:
            raise UploadError(f"incomplete upload: {upload.received} of {upload.total_bytes} bytes")
        payload = b"".join(upload.chunks)
        try:
            metadata = self._store(upload, payload)
        except ValueError as exc:
            raise UploadError(str(exc)) from exc
        return {"upload_id": upload_id, "status": "stored", **metadata}

    # -- helpers ----------------------------------------------------------------------

    def _store(self, upload: _PendingUpload, payload: bytes) -> dict[str, Any]:
        if upload.target == "attachment":
            attachment = store_attachment(
                session_id=self._session_id,
                kind=upload.kind,
                name=upload.name or "attachment",
                content_type=upload.content_type,
                data=payload,
            )
            return dict(attachment.metadata())
        if upload.target == "webcam-frame":
            frame = store_webcam_frame(
                session_id=self._session_id,
                name=upload.name or "webcam-frame.jpg",
                content_type=upload.content_type,
                data=payload,
            )
            return {"id": frame.id, "sequence": frame.sequence, "bytes": len(payload)}
        if not consume_capture_request(self._session_id, upload.request_id):
            raise UploadError("unknown or expired capture request")
        attachment = store_attachment(
            session_id=self._session_id,
            kind="image",
            name=upload.name or "focused-capture.jpg",
            content_type=upload.content_type,
            data=payload,
            source="capture",
        )
        return dict(attachment.metadata())

    @staticmethod
    def _upload_id(data: dict) -> str:
        upload_id = data.get("upload_id")
        if not isinstance(upload_id, str) or not upload_id.strip():
            raise UploadError("upload_id is required")
        return upload_id

    def _pending_upload(self, upload_id: str) -> _PendingUpload:
        upload = self._pending.get(upload_id)
        if upload is None:
            raise UploadError("unknown upload_id")
        return upload
