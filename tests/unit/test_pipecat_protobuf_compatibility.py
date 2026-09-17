# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Compatibility tests for Pipecat protobuf frames used by benchmark clients."""

import json

from pipecat.frames.protobufs import frames_pb2


def test_audio_frame_round_trip_preserves_benchmark_fields() -> None:
    """Audio frames retain the fields consumed by WebSocket benchmark clients."""
    frame = frames_pb2.Frame(
        audio=frames_pb2.AudioRawFrame(
            audio=b"\x00\x01\x02\x03",
            sample_rate=16000,
            num_channels=1,
        )
    )

    parsed = frames_pb2.Frame.FromString(frame.SerializeToString())

    assert parsed.audio.audio == b"\x00\x01\x02\x03"
    assert parsed.audio.sample_rate == 16000
    assert parsed.audio.num_channels == 1


def test_message_frame_round_trip_preserves_rtvi_payload() -> None:
    """Message frames retain the JSON payload used by the scaling client."""
    payload = {"label": "rtvi-ai", "type": "client-ready"}
    frame = frames_pb2.Frame(message=frames_pb2.MessageFrame(data=json.dumps(payload)))

    parsed = frames_pb2.Frame.FromString(frame.SerializeToString())

    assert json.loads(parsed.message.data) == payload
