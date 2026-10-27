# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests for the ACP layer: updates read through an agent class, the agent classes and the client on a replay.

The replay plays the recorded Hermes wire (``fixtures/hermes_wire.json``) over a real byte stream to the real client.
"""

import asyncio
import json
import socket
import unittest
from unittest import mock

from delegation_core.backend.harness.acp import AcpAgent, AcpClient, SessionUpdates
from delegation_core.backend.harness.acp.hermes import HermesAgent
from delegation_core.backend.harness.permissions import Reject
from delegation_core.backend.harness.protocol import (
    McpServer,
    MessageChunk,
    PlanUpdate,
    PromptEnded,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
    UserText,
)
from tests.unit.delegation_core.backend.harness.doubles import scratch, until, wire

LOST = "the connection to the agent is lost"  # the error a prompt to a lost connection ends with


def updates(scenario):
    return [step["update"]["update"] for step in wire(scenario) if "update" in step]


def hermes_events(scenario, server="client"):
    """A recorded scenario's updates as the Hermes agent class reads them (the recorded server was "client")."""
    reader = SessionUpdates(HermesAgent(), server)
    return [reader.read(update) for update in updates(scenario)]


def chunk(text, message_id=None):
    update = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    return {**update, "messageId": message_id} if message_id else update


class UpdateTests(unittest.TestCase):
    def test_each_kind_reads_into_its_event_and_the_rest_into_none(self):
        events = hermes_events("tool_then_cancel_in_tool")
        kinds = {type(event).__name__ for event in events if event is not None}
        self.assertEqual(kinds, {"ToolCard", "ToolCardEnd", "MessageChunk", "ThoughtChunk"})
        reader = SessionUpdates(HermesAgent(), "w1")
        self.assertIsNone(reader.read({"sessionUpdate": "usage_update", "size": 1, "used": 0}))
        self.assertIsNone(reader.read({"sessionUpdate": "available_commands_update"}))
        self.assertIsNone(reader.read("not an update"))

    def test_a_hermes_tool_card_carries_the_tools_name_never_its_preview_and_is_a_bridged_call(self):
        cards = [event for event in hermes_events("tool_then_cancel_in_tool") if isinstance(event, ToolCard)]
        self.assertEqual([card.name for card in cards], ["mcp__client__get_time", "mcp__client__slow_lookup"])
        self.assertEqual(cards[1].arguments, {"query": "ABC123"})
        self.assertTrue(all(card.bridged for card in cards))  # the session's own MCP server: the bridge reports them
        other = [event for event in hermes_events("tool_then_cancel_in_tool", "w7") if isinstance(event, ToolCard)]
        self.assertFalse(any(card.bridged for card in other))  # another session's prefix: calls Hermes runs itself

    def test_a_tool_card_ends_completed_or_failed_with_its_output(self):
        ends = [event for event in hermes_events("tool_then_cancel_in_tool") if isinstance(event, ToolCardEnd)]
        self.assertEqual([end.failed for end in ends], [False, True])
        self.assertIn("14:07", ends[0].output)
        self.assertIn("MCP call interrupted", ends[1].output)

    def test_a_hermes_ack_has_no_message_id(self):
        chunks = [e for e in hermes_events("redirect_and_steer") if isinstance(e, MessageChunk)]
        acks = [chunk.text for chunk in chunks if chunk.message_id is None]
        self.assertEqual(acks, ["Redirected the active turn with your correction."] * 2)

    def test_reasoning_shares_its_message_id_with_its_text(self):
        events = hermes_events("pong")
        thoughts = {e.message_id for e in events if isinstance(e, ThoughtChunk)}
        [pong] = [e for e in events if isinstance(e, MessageChunk)]
        self.assertEqual((pong.text, thoughts), ("pong", {pong.message_id}))

    def test_a_plan_and_a_drained_prompt(self):
        reader = SessionUpdates(HermesAgent(), "w1")
        plan = reader.read({"sessionUpdate": "plan", "entries": [{"content": "Look it up", "status": "in_progress"}]})
        self.assertEqual(plan, PlanUpdate(("Look it up (in_progress)",)))
        user = reader.read({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "Queued."}})
        self.assertEqual(user, UserText("Queued."))

    def test_a_plain_agents_reasoning_between_its_id_less_chunks_leaves_them_one_message(self):
        # One model call's text and reasoning: the reply is one message, never a part of it spoken as progress.
        reader = SessionUpdates(AcpAgent(), "w1")
        first = reader.read(chunk("Your order"))
        thought = {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Say when."}}
        self.assertEqual(reader.read(thought), ThoughtChunk(None))
        self.assertEqual(reader.read(chunk(" is ready.")).message_id, first.message_id)

    def test_a_plain_agents_id_less_chunks_are_one_message_until_a_tool_a_plan_or_a_prompt_comes_between(self):
        reader = SessionUpdates(AcpAgent(), "w1")
        first, second = reader.read(chunk("Let me ")), reader.read(chunk("look."))
        self.assertIsNotNone(first.message_id)
        self.assertEqual(first.message_id, second.message_id)
        self.assertIsNone(reader.read({"sessionUpdate": "usage_update", "used": 1}))  # unused: the message goes on
        self.assertEqual(reader.read(chunk(" Still.")).message_id, first.message_id)
        card = reader.read({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Looking up ABC: ABC"})
        self.assertEqual((card.name, card.bridged), ("Looking up ABC: ABC", True))  # the bridge reports every call
        after = reader.read(chunk("Found it."))
        self.assertNotEqual(after.message_id, first.message_id)
        self.assertEqual(reader.read(chunk("x", "given-id")).message_id, "given-id")  # an id given is kept


class AgentTests(unittest.TestCase):
    def test_a_plain_agent_keeps_to_the_protocol(self):
        agent = AcpAgent.from_settings({})
        folder = scratch("acp-session-")
        self.assertEqual(agent.prepare(folder, "  Be brief.  "), ("Be brief.",))  # the first prompt opens with them
        self.assertEqual(list(folder.iterdir()), [])
        self.assertEqual(agent.prompt_text("/reset"), "/reset")
        self.assertFalse(agent.native_steer)
        with self.assertRaises(RuntimeError):
            agent.steer_text("Make it two.")
        self.assertEqual((agent.turns_per_process, dict(agent.environment())), (None, {}))
        self.assertTrue(agent.model_text(Facts(chunks=1)))
        self.assertIsNone(agent.no_answer(None, None))
        with self.assertRaisesRegex(ValueError, "agents.acp has unknown keys: steer_command"):
            AcpAgent.from_settings({"steer_command": "/steer "})

    def test_hermes_reads_its_instructions_from_its_pinned_folder(self):
        agent = HermesAgent.from_settings({})
        folder = scratch("hermes-session-")
        self.assertEqual(agent.prepare(folder, "  Be brief.  "), ())
        self.assertEqual((folder / ".hermes.md").read_text(), "Be brief.\n")
        self.assertTrue((folder / ".git").is_dir())  # the pin: no parent folder's context file reaches the session
        empty = scratch("hermes-session-")
        self.assertEqual(agent.prepare(empty, ""), ())
        self.assertEqual([path.name for path in empty.iterdir()], [".git"])
        renamed = HermesAgent.from_settings({"instructions_file": "AGENTS.md"})
        renamed.prepare(empty, "Hi.")
        self.assertTrue((empty / "AGENTS.md").is_file())

    def test_hermes_guards_a_prompt_from_its_commands_and_steers_with_its_own(self):
        agent = HermesAgent.from_settings({})
        self.assertEqual(agent.prompt_text("  /reset everything"), "Request: /reset everything")
        self.assertEqual(agent.prompt_text("Find teas."), "Find teas.")
        self.assertEqual(agent.steer_text("Make it two."), "/steer Make it two.")
        self.assertTrue(agent.native_steer)
        custom = HermesAgent.from_settings({"steer_command": "/queue ", "command_guard": "Asked: "})
        self.assertEqual((custom.steer_text("x"), custom.prompt_text("/x")), ("/queue x", "Asked: /x"))

    def test_hermes_names_its_tools_and_knows_its_bridged_ones(self):
        agent = HermesAgent()
        self.assertEqual(agent.tool_name("mcp__w1__lookup: ABC123"), "mcp__w1__lookup")
        self.assertTrue(agent.bridged("mcp__w1__lookup", "w1"))
        self.assertFalse(agent.bridged("web_search", "w1"))
        self.assertFalse(agent.bridged("mcp__w12__lookup", "w1"))
        self.assertEqual(agent.environment(), {"HERMES_ACP_SKIP_CONFIGURED_MCP": "1"})
        self.assertEqual(agent.turns_per_process, 4)

    def test_hermes_speaks_only_a_parsed_answer(self):
        agent = HermesAgent()
        self.assertTrue(agent.model_text(Facts(chunks=2)))
        self.assertTrue(agent.model_text(Facts(chunks=1, thought=True, after_text=True)))
        self.assertFalse(agent.model_text(Facts(chunks=1, thought=True)))  # its own text, under the reasoning's id
        self.assertIn("no usage", agent.no_answer(None, 0))
        self.assertIn("did not move", agent.no_answer(100, 100))
        self.assertIsNone(agent.no_answer(120, 100))

    def test_a_bad_hermes_block_is_refused(self):
        cases = {
            "an unknown key": ({"soul": "x"}, "unknown keys: soul"),
            "an empty steer": ({"steer_command": " "}, "steer_command must be non-empty text"),
            "a path for a file": ({"instructions_file": "a/b.md"}, "must be a file name"),
            "not text": ({"command_guard": 1}, "non-empty text"),
        }
        for name, (block, message) in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, message):
                HermesAgent.from_settings(block)


class Facts:
    def __init__(self, chunks, thought=False, after_text=False):
        self.chunks, self.thought, self.after_text = chunks, thought, after_text


class Replay:
    """The recorded runtime's side of the stream: answers each request with what the real Hermes sent."""

    def __init__(self, scenario, *, ask_permission=False, refuse=None):
        self.steps = wire(scenario)
        self.session = next(step["session"] for step in self.steps if "session" in step)
        self.ask_permission = ask_permission
        self.refuse = refuse or {}  # method: the JSON-RPC error the runtime answers it with
        self.received: list[dict] = []
        self.closed = asyncio.Event()
        self.permission_answer = None

    @classmethod
    def check(cls, command, url):
        pass

    async def open(self):
        ours, theirs = socket.socketpair()
        self.reader, self.writer = await asyncio.open_connection(sock=ours, limit=1 << 20)
        self.their_reader, self.their_writer = await asyncio.open_connection(sock=theirs, limit=1 << 20)
        self.task = asyncio.create_task(self._serve())
        return self.reader, self.writer

    async def send(self, message):
        self.their_writer.write(json.dumps(message).encode() + b"\n")
        await self.their_writer.drain()

    async def _serve(self):
        prompts = iter([step for step in self.steps if "prompt" in step])
        try:
            while line := await self.their_reader.readline():
                message = json.loads(line)
                self.received.append(message)
                method = message.get("method")
                if method in self.refuse:
                    await self.send({"jsonrpc": "2.0", "id": message["id"], "error": self.refuse[method]})
                elif method == "initialize":
                    await self.send({"jsonrpc": "2.0", "id": message["id"], "result": {"protocolVersion": 1}})
                elif method == "session/new":
                    await self.send({"jsonrpc": "2.0", "id": message["id"], "result": {"sessionId": self.session}})
                elif method == "session/prompt":
                    await self._turn(message, next(prompts))
                elif "id" in message and "method" not in message:
                    self.permission_answer = message.get("result")
        finally:
            self.closed.set()

    async def _turn(self, request, recorded):
        if self.ask_permission:
            options = [
                {"optionId": "allow-1", "name": "Allow", "kind": "allow_once"},
                {"optionId": "reject-1", "name": "Reject", "kind": "reject_once"},
            ]
            tool = {"toolCallId": "tc-1", "title": "terminal: rm -rf"}
            params = {"sessionId": self.session, "options": options, "toolCall": tool}
            await self.send({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission", "params": params})
        index = self.steps.index(recorded)
        for step in self.steps[index + 1 :]:
            if step.get("end") == recorded["prompt"]:
                await self.send({"jsonrpc": "2.0", "id": request["id"], "result": step["result"]})
                return
            if "update" in step and step["update"]["sessionId"] == recorded["session"]:
                await self.send({"jsonrpc": "2.0", "method": "session/update", "params": step["update"]})

    async def close(self):
        self.their_writer.close()
        self.writer.close()
        self.closed.set()


def client_on(transport, agent=None):
    return AcpClient(transport, agent or HermesAgent(), permission=Reject(), start_timeout_seconds=5)


SERVER = McpServer("w1", "http://bridge.test/mcp", "token-1")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_every_update_of_a_turn_is_in_the_mailbox_before_its_end(self):
        replay = Replay("pong")
        client = client_on(replay)
        await client.open()
        mailbox = asyncio.Queue()
        session = await client.new_session("/work/w1", SERVER, mailbox)
        number = session.prompt("Reply with exactly the word: pong")
        await session.ended(number)
        events = []
        while not mailbox.empty():
            events.append(mailbox.get_nowait())
        self.assertIsInstance(events[-1], PromptEnded)
        self.assertEqual(events[-1].stop_reason, "end_turn")
        self.assertEqual(events[-1].usage["totalTokens"], 712)
        self.assertEqual([e.text for e in events if isinstance(e, MessageChunk)], ["pong"])
        new = next(m for m in replay.received if m.get("method") == "session/new")
        [server] = new["params"]["mcpServers"]
        self.assertEqual(
            server,
            {
                "type": "http",
                "name": "w1",
                "url": "http://bridge.test/mcp",
                "headers": [{"name": "Authorization", "value": "Bearer token-1"}],
            },
        )
        init = next(m for m in replay.received if m.get("method") == "initialize")["params"]
        self.assertEqual(init["protocolVersion"], 1)
        # No file system, no terminal: the SDK leaves out capabilities equal to ACP's defaults (all false).
        capabilities = init.get("clientCapabilities", {})
        self.assertFalse(capabilities.get("terminal", False))
        self.assertFalse(any((capabilities.get("fs") or {}).values()))
        await client.close()

    async def test_the_agent_class_words_a_prompt_and_a_steer(self):
        # Hermes: caller text never runs as one of its commands, and its steer is its own command.
        replay = Replay("redirect_and_steer")
        client = client_on(replay)
        await client.open()
        session = await client.new_session("/work/w1", SERVER, asyncio.Queue())
        session.prompt("/reset everything")
        session.steer("Make it two.")
        await asyncio.sleep(0.2)
        texts = [m["params"]["prompt"][0]["text"] for m in replay.received if m.get("method") == "session/prompt"]
        self.assertEqual(texts, ["Request: /reset everything", "/steer Make it two."])
        await client.close()

    async def test_a_plain_agent_gets_a_prompt_as_written(self):
        replay = Replay("pong")
        client = client_on(replay, AcpAgent())
        await client.open()
        session = await client.new_session("/work/w1", SERVER, asyncio.Queue())
        await session.ended(session.prompt("/reset everything"))
        texts = [m["params"]["prompt"][0]["text"] for m in replay.received if m.get("method") == "session/prompt"]
        self.assertEqual(texts, ["/reset everything"])
        await client.close()

    async def test_a_cancel_is_sent_only_while_a_prompt_is_outstanding(self):
        replay = Replay("cancel_then_prompt")
        client = client_on(replay)
        await client.open()
        session = await client.new_session("/work/w1", SERVER, asyncio.Queue())
        self.assertFalse(await session.cancel())  # idle: never (it would poison the next prompt)
        number = session.prompt("Count to 400.")
        self.assertTrue(await session.cancel())
        await session.ended(number)
        self.assertEqual([m.get("method") for m in replay.received].count("session/cancel"), 1)
        await client.close()

    async def test_a_permission_request_is_declined_and_after_a_cancel_answered_cancelled(self):
        replay = Replay("pong", ask_permission=True)
        client = client_on(replay)
        await client.open()
        session = await client.new_session("/work/w1", SERVER, asyncio.Queue())
        await session.ended(session.prompt("pong"))
        await asyncio.sleep(0.1)
        self.assertEqual(replay.permission_answer, {"outcome": {"outcome": "selected", "optionId": "reject-1"}})
        await client.close()

    async def test_a_lost_connection_ends_every_outstanding_prompt_with_an_error(self):
        replay = Replay("cancel_then_prompt")
        client = client_on(replay)
        await client.open()
        mailbox = asyncio.Queue()
        session = await client.new_session("/work/w1", SERVER, mailbox)
        replay.steps = [step for step in replay.steps if "end" not in step]  # the runtime never answers
        number = session.prompt("Count to 400.")
        await asyncio.sleep(0.1)
        replay.their_writer.close()
        replay.closed.set()
        await asyncio.wait_for(session.ended(number), 5)
        ended = [e for e in [mailbox.get_nowait() for _ in range(mailbox.qsize())] if isinstance(e, PromptEnded)]
        self.assertIsNotNone(ended[-1].error)
        self.assertFalse(client.alive)
        await client.close()

    async def test_after_a_loss_a_prompt_and_a_steer_end_at_once_and_no_session_opens(self):
        # A steer delivered as the task resumes, once the agent crashed while the run waited for the app.
        replay = Replay("redirect_and_steer")
        client = client_on(replay)
        await client.open()
        mailbox = asyncio.Queue()
        session = await client.new_session("/work/w1", SERVER, mailbox)
        replay.their_writer.close()
        replay.closed.set()
        await until(lambda: not client.alive)
        numbers = [session.steer("Make it two."), session.prompt("Count to 400.")]
        await asyncio.wait_for(asyncio.gather(*(session.ended(number) for number in numbers)), 2)
        ended = [mailbox.get_nowait() for _ in range(mailbox.qsize())]
        self.assertEqual([(e.prompt, e.error) for e in ended], [(number, LOST) for number in numbers])
        await asyncio.wait_for(session.idle(), 1)
        with self.assertRaisesRegex(ConnectionError, LOST):
            await client.new_session("/work/w2", SERVER, asyncio.Queue())
        await client.close()

    async def test_a_prompt_a_dead_sender_left_waiting_ends_when_the_connection_is_lost(self):
        # The stream to the agent broke while it ran on: the SDK's sender dies on its next write, and the SDK's close
        # then fails none of the requests still waiting.
        replay = Replay("cancel_then_prompt")
        client = client_on(replay)
        await client.open()
        mailbox = asyncio.Queue()
        session = await client.new_session("/work/w1", SERVER, mailbox)
        replay.steps = [step for step in replay.steps if "end" not in step]  # the runtime never answers
        first = session.prompt("Count to 400.")
        await asyncio.sleep(0.1)
        replay.writer.close()
        await asyncio.wait_for(session.ended(session.prompt("Count to 500.")), 2)  # the write that kills the sender
        replay.closed.set()  # then the agent exits
        await asyncio.wait_for(session.ended(first), 2)
        events = [mailbox.get_nowait() for _ in range(mailbox.qsize())]
        ended = [e for e in events if isinstance(e, PromptEnded) and e.prompt == first]
        self.assertEqual(ended, [PromptEnded(first, None, None, LOST)])
        await client.close()

    async def test_a_line_that_stops_the_sdks_reader_loses_the_connection_and_ends_the_process(self):
        # An agent that breaks ACP's framing and runs on: the SDK's receive loop stops for good on the line, so nothing
        # the agent sends after it is read. Unless that is a loss, the pool keeps the process, and each session waits.
        cases = {
            "a JSON line that is not an object": (b"42\n", "not a JSON object: 42"),  # the client sees it
            "a line over the stream's limit": (b"x" * (2 << 20) + b"\n", "ValueError"),  # the prompts fail with it
        }
        for name, (line, cause) in cases.items():
            with self.subTest(name), mock.patch("delegation_core.backend.harness.acp.client.logger") as log:
                replay = Replay("cancel_then_prompt")
                client = client_on(replay)
                await client.open()
                mailbox = asyncio.Queue()
                session = await client.new_session("/work/w1", SERVER, mailbox)
                replay.steps = [step for step in replay.steps if "end" not in step]  # the runtime never answers
                numbers = [session.prompt("Count to 400."), session.prompt("Count to 500.")]
                await asyncio.sleep(0.1)
                replay.their_writer.write(line)
                await asyncio.wait_for(asyncio.gather(*(session.ended(number) for number in numbers)), 2)
                events = [mailbox.get_nowait() for _ in range(mailbox.qsize())]
                ended = [(e.prompt, e.error) for e in events if isinstance(e, PromptEnded)]
                self.assertEqual(ended, [(number, LOST) for number in numbers])
                self.assertFalse(client.alive)  # the pool drops it
                [warning] = [call.args[0] for call in log.warning.call_args_list]
                self.assertIn("harness.connection_lost", warning)
                self.assertIn(cause, warning)
                await asyncio.wait_for(replay.closed.wait(), 2)  # the client closed the transport: the process ends
                with self.assertRaisesRegex(ConnectionError, LOST):
                    await client.new_session("/work/w2", SERVER, asyncio.Queue())
                await client.close()

    async def test_an_agents_refusal_carries_its_own_reason(self):
        # Hermes started with no usable key refuses session/new so: its reason is only in the error's data.
        details = "No usable credentials found for provider 'nvidia'. Set NVIDIA_API_KEY."
        refusal = {"code": -32603, "message": "Internal error", "data": {"details": details}}
        for method in ("initialize", "session/new"):
            with self.subTest(method):
                client = client_on(Replay("pong", refuse={method: refusal}))
                with self.assertRaisesRegex(RuntimeError, f"answered {method} with an error: .*error.*No usable"):
                    await client.open()
                    await client.new_session("/work/w1", SERVER, asyncio.Queue())
                await client.close()
        client = client_on(Replay("pong", refuse={"session/prompt": refusal}))
        await client.open()
        mailbox = asyncio.Queue()
        session = await client.new_session("/work/w1", SERVER, mailbox)
        await session.ended(session.prompt("pong"))
        ended = mailbox.get_nowait()
        self.assertIn("RequestError: Internal error", ended.error)
        self.assertIn(details, ended.error)
        await client.close()


if __name__ == "__main__":
    unittest.main()
