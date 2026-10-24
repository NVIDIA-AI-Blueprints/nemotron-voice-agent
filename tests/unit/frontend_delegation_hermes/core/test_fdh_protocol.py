# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Protocol contract: every message round-trips; version, type and field errors are rejected."""

from __future__ import annotations

import json
import unittest

from examples.frontend_delegation_hermes.core.delegation import protocol as proto

SAMPLES = {
    proto.VOICE_TO_GATEWAY: {
        "session.open": {"session_id": "s", "settings": {}},
        "session.configure": {"tools": [], "instructions": ""},
        "history.append": {"entries": [{"seq": 1, "kind": "user", "text": "hi"}]},
        "delegate": {"turn_id": 1, "request": "task", "run_input": [{"seq": 2, "text": "x"}]},
        "tool.result": {"call_id": "c", "output": "o", "epoch": 1, "local": False},
        "session.close": {},
    },
    proto.GATEWAY_TO_VOICE: {
        "session.ready": {"worker_pid": 1},
        "session.configured": {"applied": True},
        "history.ack": {"committed_upto": 3},
        "state": {"state": "IDLE", "epoch": 1},
        "action": {"turn_id": 1, "kind": "start"},
        "tool.call": {"call_id": "c", "epoch": 1, "name": "n", "arguments": "{}"},
        "tool.cancel": {"call_id": "c", "reason": "timeout"},
        "activity": {"epoch": 1, "event": "tool_started", "name": "n"},
        "answer": {"answer_id": "a", "epoch": 1, "kind": "hermes", "text": "t"},
        "backend_run_done": {"run_id": "r", "epoch": 1, "status": "ok", "history": "committed"},
        "status": {"turn_id": 1, "summary": {}},
        "worker": {"event": "ready"},
        "error": {"code": "capacity", "message": "m", "fatal": True},
    },
    proto.GATEWAY_TO_WORKER: {
        "configure": {"tools": [], "instructions": "", "backend": {}},
        "run": {"epoch": 1, "task_id": "s:1", "user_message": "u", "conversation_history": []},
        "steer": {"epoch": 1, "text": "t"},
        "redirect": {"epoch": 1, "text": "t"},
        "status": {"epoch": 1},
        "tool.result": {"call_id": "c", "output": "o"},
        "interrupt": {"hard": True},
        "close": {},
    },
    proto.WORKER_TO_GATEWAY: {
        "ready": {"pid": 1},
        "configured": {},
        "steer_result": {"epoch": 1, "accepted": True, "via": "steer"},
        "status_result": {"epoch": 1, "summary": {}},
        "tool.call": {"call_id": "c", "epoch": 1, "name": "n", "arguments": "{}"},
        "tool.cancel": {"call_id": "c", "reason": "timeout"},
        "activity": {"epoch": 1, "event": "tool_completed", "name": "n"},
        "run_outcome": {"epoch": 1, "status": "ok"},
        "closed": {},
        "error": {"phase": "construct", "message": "m"},
    },
}


class ProtocolTest(unittest.TestCase):
    def test_every_type_round_trips(self) -> None:
        for direction, types in proto.SPECS.items():
            self.assertEqual(set(types), set(SAMPLES[direction]), direction)
            for type_, fields in SAMPLES[direction].items():
                raw = proto.encode(direction, type_, **fields)
                decoded = proto.decode(direction, raw)
                self.assertEqual(decoded["type"], type_)
                self.assertEqual(decoded["v"], proto.PROTOCOL_VERSION)
                for key, value in fields.items():
                    self.assertEqual(decoded[key], value)

    def test_version_mismatch_is_rejected(self) -> None:
        raw = json.dumps({"v": 99, "type": "session.close"})
        with self.assertRaisesRegex(proto.ProtocolError, "version"):
            proto.decode(proto.VOICE_TO_GATEWAY, raw)

    def test_unknown_type_and_wrong_direction_are_rejected(self) -> None:
        with self.assertRaisesRegex(proto.ProtocolError, "unknown message type"):
            proto.encode(proto.VOICE_TO_GATEWAY, "run", epoch=1, task_id="x", user_message="", conversation_history=[])

    def test_missing_field_is_rejected(self) -> None:
        with self.assertRaisesRegex(proto.ProtocolError, "missing"):
            proto.encode(proto.VOICE_TO_GATEWAY, "delegate", turn_id=1, request="task")

    def test_non_object_and_bad_json(self) -> None:
        with self.assertRaises(proto.ProtocolError):
            proto.decode(proto.VOICE_TO_GATEWAY, "[1]")
        with self.assertRaises(proto.ProtocolError):
            proto.decode(proto.VOICE_TO_GATEWAY, "{nope")

    def test_optional_fields_pass_through(self) -> None:
        msg = proto.decode(
            proto.GATEWAY_TO_VOICE, proto.encode(proto.GATEWAY_TO_VOICE, "state", state="IDLE", epoch=1, run_id=None)
        )
        self.assertIn("run_id", msg)


if __name__ == "__main__":
    unittest.main()
