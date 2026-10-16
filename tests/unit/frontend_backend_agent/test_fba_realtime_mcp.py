# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""5.9: opt-in MCP tools for Realtime sessions, run by the server and hidden from the client.

Plan ``misc/frontend-backend-verdict/fba-removal-design-5.8-5.9.md`` (approved). A fake MCP server
replaces the transport; the shared gateway's validation rules run as they are. Off, or in eval mode,
``mcp`` tools are refused as before (``test_fba_eval_pins.py``, probe ``mcp_tool``).
"""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from typing import Any
from unittest import mock

from _fba_core_fakes import FakeChatClient, delegate_response, tau2_session_update, text_response, tool_response
from _fba_pipecat_fakes import PipecatHarness, PipecatTestCase
from mcp.types import CallToolResult, TextContent, Tool

from examples.frontend_backend_agent.adapters.pipecat import mcp_tools
from examples.frontend_backend_agent.core.engine.runner import AgentClients

URL = "https://mcp.example.test/mcp"
ENV = {"FBA_REALTIME_MCP": "true", "REALTIME_MCP_ALLOWED_SERVER_URLS": json.dumps([URL])}
MCP_FUNCTION = "mcp__srv__lookup"


class FakeWorker:
    """Stands in for ``realtime.mcp._MCPServerWorker`` (one MCP server connection)."""

    instances: list[FakeWorker] = []
    gate: asyncio.Event | None = None

    def __init__(self, definition: dict[str, Any]) -> None:  # noqa: D107
        self.definition = definition
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.cancelled = False
        self.dead = False
        FakeWorker.instances.append(self)

    async def start(self) -> None:
        return None

    async def list_tools(self) -> list[Tool]:
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}
        return [Tool.model_validate({"name": "lookup", "description": "Look a record up.", "inputSchema": schema})]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        try:
            if FakeWorker.gate is not None:
                await FakeWorker.gate.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return CallToolResult(content=[TextContent(type="text", text=f"record {arguments.get('id')} is green")])

    def shutdown(self) -> None:
        return None


def _mcp_entry(**overrides: Any) -> dict[str, Any]:
    return {"type": "mcp", "server_label": "srv", "server_url": URL, "require_approval": "never", **overrides}


def _client_function(name: str = "get_users") -> dict[str, Any]:
    return {"type": "function", "name": name, "description": "List users.", "parameters": {"type": "object"}}


def _update(*tools: dict[str, Any]) -> dict[str, Any]:
    """The Tau fixture's ``session.update`` (mu-law audio, server VAD) with ``tools`` instead of its own."""
    update = tau2_session_update("mock")
    update["session"]["tools"] = list(tools)
    return update


class McpSessionTests(PipecatTestCase):
    async def asyncSetUp(self) -> None:
        FakeWorker.instances = []
        FakeWorker.gate = None
        for patch in (
            mock.patch.dict(os.environ, ENV),
            mock.patch.object(mcp_tools, "_MCPServerWorker", FakeWorker),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.frontend = FakeChatClient()
        self.backend = FakeChatClient()
        self.harness = PipecatHarness(clients=AgentClients(backend=self.backend, frontend=self.frontend))
        self.assertIsNotNone(self.harness.session._mcp)

    async def asyncTearDown(self) -> None:
        await self.harness.close()
        await super().asyncTearDown()

    def _errors(self) -> list[dict[str, Any]]:
        return [event["error"] for event in self.harness.of_type("error")]

    def _wire_text(self) -> str:
        return json.dumps(self.harness.events)

    def _assert_chain_is_visible(self) -> None:
        seen: set[str] = set()
        for event in self.harness.events:
            if event["type"] == "conversation.item.added":
                previous = event.get("previous_item_id")
                self.assertTrue(previous is None or previous in seen, f"dangling previous_item_id {previous}")
                seen.add(event["item"]["id"])

    async def test_update_echoes_the_client_entries(self) -> None:
        await self.harness.start(_update(_mcp_entry(), _client_function()))
        self.assertEqual(self._errors(), [])
        tools = self.harness.of_type("session.updated")[-1]["session"]["tools"]
        self.assertEqual([t.get("name") or t.get("server_label") for t in tools], ["get_users", "srv"])
        self.assertEqual(tools[1]["type"], "mcp")
        self.assertIn(MCP_FUNCTION, self.harness.session.view.settings.tools[1]["name"])

    async def test_refusals(self) -> None:
        await self.harness.start()
        cases = [
            (_mcp_entry(require_approval="always"), "unsupported_capability"),
            ({k: v for k, v in _mcp_entry().items() if k != "require_approval"}, "unsupported_capability"),
            (_mcp_entry(server_url="https://elsewhere.test/mcp"), "mcp_server_not_allowed"),
            ({"type": "mcp", "server_label": "nope", "require_approval": "never"}, "mcp_server_not_defined"),
            (_mcp_entry(connector_id="c1", server_url=None), "unsupported_capability"),
        ]
        for entry, code in cases:
            with self.subTest(code=code, entry=entry):
                before = len(self._errors())
                await self.harness.send(_update(entry))
                self.assertEqual(self._errors()[before]["code"], code)
        before = len(self._errors())
        await self.harness.send(_update(_client_function(MCP_FUNCTION)))
        self.assertEqual(self._errors()[before]["code"], "invalid_value")

    async def test_an_mcp_only_call_is_hidden_and_answered(self) -> None:
        await self.harness.start(_update(_mcp_entry(), _client_function()))
        self.frontend.queue(delegate_response("Look up record 42."))
        self.backend.queue(tool_response((MCP_FUNCTION, {"id": "42"}), ids=["call_m"]), text_response("It is green."))
        await self.harness.speak()
        await self.harness.wait_for("response.done", 2)
        self.assertEqual(FakeWorker.instances[-1].calls, [("lookup", {"id": "42"})])
        wire = self._wire_text()
        self.assertNotIn("call_m", wire)
        self.assertNotIn(MCP_FUNCTION, wire)
        self.assertEqual(self.harness.of_type("response.done")[0]["response"]["output"], [])
        tool_messages = [m for m in self.backend.calls[1]["messages"] if m.role == "tool"]
        self.assertIn("record 42 is green", tool_messages[0].content)
        transcript = "".join(e["delta"] for e in self.harness.of_type("response.output_audio_transcript.delta"))
        self.assertEqual(transcript, "It is green.")
        self.assertEqual(self._errors(), [])
        self._assert_chain_is_visible()

    async def test_mixed_calls_show_only_the_client_call(self) -> None:
        await self.harness.start(_update(_mcp_entry(), _client_function()))
        self.frontend.queue(delegate_response("Look up record 7 and the users."))
        self.backend.queue(
            tool_response((MCP_FUNCTION, {"id": "7"}), ("get_users", {}), ids=["call_m", "call_c"]),
            text_response("Done."),
        )
        await self.harness.speak()
        done = await self.harness.wait_for("response.function_call_arguments.done")
        self.assertEqual((done["call_id"], done["name"]), ("call_c", "get_users"))
        first = await self.harness.wait_for("response.done")
        self.assertEqual([item["call_id"] for item in first["response"]["output"]], ["call_c"])
        self.assertEqual(
            [
                e["output_index"]
                for e in self.harness.of_type("response.output_item.added")
                if e["item"]["type"] == "function_call"
            ],
            [0],
        )
        await self.harness.send(
            {
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "call_c", "output": "[]"},
            }
        )
        await self.harness.send({"type": "response.create"})
        await self.harness.wait_for("response.done", 2)
        contents = sorted(m.content for m in self.backend.calls[1]["messages"] if m.role == "tool")
        self.assertEqual(len(contents), 2)
        self.assertTrue(any("record 7 is green" in c for c in contents))
        self.assertNotIn("call_m", self._wire_text())
        self.assertEqual(self._errors(), [])
        self._assert_chain_is_visible()

    async def test_cancel_stops_a_running_call(self) -> None:
        FakeWorker.gate = asyncio.Event()
        await self.harness.start(_update(_mcp_entry()))
        self.frontend.queue(delegate_response("Look up record 9."))
        self.backend.queue(tool_response((MCP_FUNCTION, {"id": "9"}), ids=["call_m"]), text_response("Stopped."))
        await self.harness.speak()
        await self.harness.wait_for("response.done")
        for _ in range(100):
            if FakeWorker.instances[-1].calls:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(self.harness.session._mcp.running, 1)
        await self.harness.send({"type": "response.cancel"})
        await self.harness.wait_for("response.done", 2)
        self.assertTrue(FakeWorker.instances[-1].cancelled)
        self.assertEqual(self.harness.session._mcp.running, 0)
        tool_messages = [m for m in self.backend.calls[1]["messages"] if m.role == "tool"]
        self.assertIn("cancelled", tool_messages[0].content)


class McpOffTests(unittest.TestCase):
    def test_ui_sessions_never_use_mcp(self) -> None:
        from examples.frontend_backend_agent.adapters.pipecat import ui_session  # noqa: PLC0415

        self.assertFalse(ui_session.UiSession.mcp_allowed(mock.Mock()))

    def test_switch_needs_the_allowlist_and_is_off_in_eval_mode(self) -> None:
        from examples.frontend_backend_agent.adapters.pipecat import switches  # noqa: PLC0415

        self.assertFalse(switches.realtime_switch("FBA_REALTIME_MCP", {}))
        self.assertFalse(switches.realtime_switch("FBA_REALTIME_MCP", {**ENV, "FBA_PROFILE": "eval"}))
        self.assertTrue(switches.realtime_switch("FBA_REALTIME_MCP", ENV))
        with mock.patch.dict(os.environ, {"REALTIME_MCP_ALLOWED_SERVER_URLS": ""}):
            self.assertFalse(mcp_tools.configured())


if __name__ == "__main__":
    unittest.main()
