# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Settings validated in the background, after startup and when they change.

The server delegation asks its core through ``POST /v1/responses/input_tokens``. A refused setting comes back as a
nonfatal ``responses_validation_failed``; the session stays open and its settings are not rolled back.
"""

import asyncio
import base64
import time
import unittest

from live.gateway.websocket import wire_codec
from live.protocol import SessionConfig
from live.protocol.config import TOOL_TYPES
from live.session import LiveProtocolSession
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

MCP = {
    "type": "mcp",
    "server_label": "crm",
    "server_url": "https://crm.example/mcp",
    "authorization": "Bearer secret-token",
    "headers": {"X-Api-Key": "secret-key"},
}


def official_tools():
    return [{"type": kind, "name": "get_menu"} if kind == "function" else {"type": kind} for kind in sorted(TOOL_TYPES)]


class SlowCore(FakeAdapter):
    """A core whose settings check takes ``delay`` seconds, or fails with ``fault``."""

    def __init__(self, delay=0.0, fault=None):
        super().__init__()
        self.delay, self.fault = delay, fault
        self.requests: list[dict] = []

    async def input_tokens(self, params):
        await asyncio.sleep(self.delay)
        if self.fault is not None:
            raise self.fault
        return await super().input_tokens(params)

    async def create_response(self, params):
        self.requests.append(params)
        return await super().create_response(params)


class ValidationCase(unittest.IsolatedAsyncioTestCase):
    async def start(self, config=None, core=None):
        self.core = core or SlowCore()
        config = (config or responses_config()).with_wire_format()
        self.session = LiveProtocolSession(
            config, "websocket", self.core.parts(), core=self.core.core, codec=wire_codec(config)
        )
        self.primary = Collector()
        self.key = self.session.subscribe(self.primary.send, "primary")
        await self.session.start("start_1")
        await self.primary.wait_for("session.started")
        self.addAsyncCleanup(self.session.close)

    async def update(self, responses, event_id="u1", key=None):
        message = {"type": "session.update", "event_id": event_id, "session": {"delegation": {"responses": responses}}}
        await self.session.handle(message, key or self.key)

    async def settle(self):
        """Let the background checks run and their reports reach the connections."""
        for _ in range(5):
            await asyncio.sleep(0.01)
        for sub in tuple(self.session.fanout.subscribers.values()):
            await sub.queue.join()


class AfterTheAnswerTests(ValidationCase):
    async def test_an_unknown_model_gives_session_updated_then_a_nonfatal_error_and_is_kept(self):
        await self.start()
        await self.update({"model": "bad-model"})
        error = (await self.primary.wait_for("error"))["error"]
        self.assertEqual(
            (error["type"], error["code"], error["param"], error["client_event_id"]),
            (
                "invalid_request_error",
                "responses_validation_failed",
                "session.delegation.responses.model",
                "u1",
            ),
        )
        kinds = self.primary.kinds()
        self.assertLess(kinds.index("session.updated"), kinds.index("error"))
        self.assertEqual(self.session.config.delegation.responses.model, "bad-model")
        self.assertEqual(self.session.status, "active")

    async def test_the_settings_a_session_starts_with_are_checked_after_session_started(self):
        config = SessionConfig(model="live-1", delegation={"type": "responses", "responses": {"model": "bad-model"}})
        await self.start(config)
        error = (await self.primary.wait_for("error"))["error"]
        self.assertEqual(
            (error["code"], error["param"], error["client_event_id"]),
            ("responses_validation_failed", "session.delegation.responses.model", "start_1"),
        )
        self.assertEqual(self.primary.kinds()[0], "session.started")
        self.assertEqual(self.session.status, "active")

    async def test_settings_the_core_accepts_give_no_error(self):
        await self.start()
        await self.update({"model": "org/other-model", "reasoning": {"effort": "high"}})
        await self.settle()
        self.assertNotIn("error", self.primary.kinds())
        self.assertEqual([checked["model"] for checked in self.core.checked], ["org/org/model", "org/other-model"])
        self.assertEqual(self.core.checked[-1]["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertEqual(self.core.checked[-1]["input"], [])  # only the settings are checked; nothing runs
        self.assertEqual(self.core.creates, 0)


class OffTheLockTests(ValidationCase):
    async def test_a_slow_check_never_holds_the_answer_or_the_audio(self):
        await self.start(core=SlowCore(delay=0.3))
        began = time.monotonic()
        await self.update({"model": "bad-model"})
        chunk = base64.b64encode(b"\x01\x00" * 480).decode()
        await self.session.handle({"type": "session.input_audio.append", "audio": chunk}, self.key)
        self.assertLess(time.monotonic() - began, 0.1)
        self.assertEqual(len(self.core.audio), 1)
        await self.primary.wait_for("session.updated")
        self.assertNotIn("error", self.primary.kinds())
        await self.primary.wait_for("error")  # once the core answered


class ReportTests(ValidationCase):
    async def test_the_error_goes_to_the_connection_that_sent_the_update(self):
        await self.start()
        sideband = Collector()
        sideband_key = self.session.subscribe(sideband.send, "sideband")
        await self.update({"model": "bad-model"}, "u9", sideband_key)
        error = (await sideband.wait_for("error"))["error"]
        self.assertEqual(error["client_event_id"], "u9")
        await self.settle()
        self.assertNotIn("error", self.primary.kinds())

    async def test_a_core_that_cannot_check_reports_nothing(self):
        await self.start(core=SlowCore(fault=RuntimeError("the core is down")))
        await self.update({"model": "bad-model"})
        await self.settle()
        self.assertNotIn("error", self.primary.kinds())

    async def test_a_check_still_running_at_close_reports_nothing(self):
        await self.start(core=SlowCore(delay=0.2))
        await self.update({"model": "bad-model"})
        await self.session.close("close_requested")
        await asyncio.sleep(0.3)
        self.assertNotIn("error", self.primary.kinds())
        self.assertEqual(self.primary.kinds()[-1], "session.closed")

    async def test_a_client_mode_session_checks_nothing(self):
        await self.start(SessionConfig(model="live-1"))
        message = {
            "type": "session.update",
            "event_id": "u1",
            "session": {"delegation": {"acknowledgement_delay_ms": 5}},
        }
        await self.session.handle(message, self.key)
        await self.settle()
        self.assertEqual(self.core.checked, [])
        self.assertNotIn("error", self.primary.kinds())


class ToolTests(ValidationCase):
    async def test_every_official_tool_type_is_accepted_and_checked_by_the_core(self):
        await self.start()
        await self.update({"tools": official_tools()})
        await self.primary.wait_for("session.updated")
        await self.settle()
        self.assertEqual({tool["type"] for tool in self.core.checked[-1]["tools"]}, TOOL_TYPES)
        self.assertNotIn("error", self.primary.kinds())

    async def test_a_tool_type_outside_the_schema_is_refused_at_once(self):
        await self.start()
        await self.update({"tools": [{"type": "teleport"}]})
        error = (await self.primary.wait_for("error"))["error"]
        self.assertEqual(error["code"], "invalid_value")
        self.assertNotIn("session.updated", self.primary.kinds())

    async def test_mcp_credentials_reach_the_core_but_never_come_back(self):
        config = SessionConfig(
            model="live-1", delegation={"type": "responses", "responses": {"model": "org/org/model", "tools": [MCP]}}
        )
        await self.start(config)
        started = (await self.primary.wait_for("session.started"))["session"]
        [shown] = started["delegation"]["responses"]["tools"]
        self.assertEqual(shown, {"type": "mcp", "server_label": "crm", "server_url": "https://crm.example/mcp"})
        await self.update({"instructions": "Be brief."})  # the tools are left alone: the credentials stay
        updated = (await self.primary.wait_for("session.updated"))["session"]
        self.assertNotIn("authorization", updated["delegation"]["responses"]["tools"][0])
        await self.settle()
        self.assertEqual(self.core.checked[-1]["tools"], [MCP])
        self.session.delegation.listen(type("Listener", (), {"on_delegation_event": lambda *a: None})())
        await self.session.delegation.delegate([{"role": "user", "content": "Hi."}])
        self.assertEqual(self.core.requests[-1]["tools"], [MCP])


if __name__ == "__main__":
    unittest.main()
