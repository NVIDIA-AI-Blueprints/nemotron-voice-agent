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

from examples.frontend_backend_live.delegation_core.backend.harness.acp import AcpAgent, AcpClient, SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.permissions import Reject
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    McpServer,
    MessageChunk,
    PlanUpdate,
    PromptEnded,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
    UserText,
)
from tests.unit.delegation_core.backend.harness.doubles import hermes, scratch, until, wire

LOST = "the connection to the agent is lost"  # the error a prompt to a lost connection ends with
SERVER = McpServer("w1", "http://bridge.test/mcp", "token-1")


def hermes_events(scenario, server="client"):
    """A recorded scenario's updates as the Hermes agent class reads them (the recorded server was "client")."""
    reader = SessionUpdates(hermes(), server)
    return [reader.read(step["update"]["update"]) for step in wire(scenario) if "update" in step]


def of(events, kind):
    return [event for event in events if isinstance(event, kind)]


def chunk(text, message_id=None):
    update = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    return {**update, "messageId": message_id} if message_id else update


def drain(mailbox):
    return [mailbox.get_nowait() for _ in range(mailbox.qsize())]


class UpdateTests(unittest.TestCase):
    def test_the_recorded_hermes_wire_reads_into_tool_cards_message_chunks_and_thoughts(self):
        events = hermes_events("tool_then_cancel_in_tool")
        kinds = {type(event).__name__ for event in events if event is not None}
        self.assertEqual(kinds, {"ToolCard", "ToolCardEnd", "MessageChunk", "ThoughtChunk"})
        cards = of(events, ToolCard)  # the tool's name, never its preview
        self.assertEqual([card.name for card in cards], ["mcp__client__get_time", "mcp__client__slow_lookup"])
        self.assertEqual(cards[1].arguments, {"query": "ABC123"})
        self.assertTrue(all(card.bridged for card in cards))  # the session's own MCP server: the bridge reports them
        self.assertFalse(any(card.bridged for card in of(hermes_events("tool_then_cancel_in_tool", "w7"), ToolCard)))
        ends = of(events, ToolCardEnd)
        self.assertEqual([end.failed for end in ends], [False, True])
        self.assertIn("14:07", ends[0].output)
        self.assertIn("MCP call interrupted", ends[1].output)
        acks = [chunk.text for chunk in of(hermes_events("redirect_and_steer"), MessageChunk) if not chunk.message_id]
        self.assertEqual(acks, ["Redirected the active turn with your correction."] * 2)  # an ack has no message id
        pong = hermes_events("pong")
        [answer] = of(pong, MessageChunk)  # reasoning shares its message id with its text
        self.assertEqual(
            (answer.text, {thought.message_id for thought in of(pong, ThoughtChunk)}), ("pong", {answer.message_id})
        )

    def test_a_tool_cards_output_is_its_raw_text_else_its_content_else_its_raw_output_and_a_plan_is_progress(self):
        reader = SessionUpdates(hermes(), "w1")
        ended = {"sessionUpdate": "tool_call_update", "toolCallId": "t1", "status": "completed"}
        content = {"content": [{"type": "content", "content": {"type": "text", "text": "3 teas"}}]}
        for update, output in (
            ({**ended, "rawOutput": "three teas", **content}, "three teas"),
            ({**ended, "rawOutput": {"teas": 3}, **content}, "3 teas"),
            ({**ended, "rawOutput": {"teas": 3}}, '{"teas": 3}'),
            (ended, ""),
        ):
            self.assertEqual(reader.read(update).output, output)
        plan = reader.read({"sessionUpdate": "plan", "entries": [{"content": "Look it up", "status": "in_progress"}]})
        user = reader.read({"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "Queued."}})
        self.assertEqual((plan, user), (PlanUpdate(("Look it up (in_progress)",)), UserText("Queued.")))
        self.assertEqual([reader.read(unread) for unread in ({"sessionUpdate": "usage_update"}, "x")], [None, None])

    def test_a_plain_agents_id_less_chunks_are_one_message_until_a_tool_a_plan_or_a_prompt_comes_between(self):
        # One model call's text and reasoning: the reply is one message, never a part of it spoken as progress.
        reader = SessionUpdates(AcpAgent(), "w1")
        first = reader.read(chunk("Let me "))
        self.assertIsNotNone(first.message_id)
        thought = {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Say when."}}
        self.assertEqual(reader.read(thought), ThoughtChunk(None))
        self.assertIsNone(reader.read({"sessionUpdate": "usage_update", "used": 1}))
        self.assertEqual({reader.read(chunk(text)).message_id for text in ("look.", " Still.")}, {first.message_id})
        card = reader.read({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Looking up ABC: ABC"})
        self.assertEqual((card.name, card.bridged), ("Looking up ABC: ABC", True))  # the bridge reports every call
        self.assertNotEqual(reader.read(chunk("Found it.")).message_id, first.message_id)
        self.assertEqual(reader.read(chunk("x", "given-id")).message_id, "given-id")  # an id given is kept


class AgentTests(unittest.TestCase):
    def test_hermes_pins_its_folder_guards_a_steer_as_a_prompt_and_knows_its_bridged_tools_by_name(self):
        agent, folder = hermes(), scratch("hermes-session-")
        self.assertEqual(agent.prepare(folder, ""), ())  # no instructions: only the pin, so no parent's file is read
        self.assertEqual([path.name for path in folder.iterdir()], [".git"])
        hermes(instructions_file="AGENTS.md").prepare(folder, "Hi.")
        self.assertEqual((folder / "AGENTS.md").read_text(), "Hi.\n")
        # On an idle session Hermes runs a steer's text as a prompt of its own: it gets a prompt's guard.
        self.assertEqual(agent.steer_text("/reset the order to tea"), "/steer Request: /reset the order to tea")
        custom = HermesAgent.from_settings({"steer_command": "/queue ", "command_guard": "Asked: "})
        self.assertEqual((custom.steer_text("x"), custom.prompt_text("/x")), ("/queue x", "Asked: /x"))
        self.assertEqual(agent.tool_name("mcp__w1__lookup: ABC123"), "mcp__w1__lookup")  # never the card's preview
        bridged = [agent.bridged(name, "w1") for name in ("mcp__w1__lookup", "web_search", "mcp__w12__lookup")]
        self.assertEqual(bridged, [True, False, False])


class Replay:
    """The recorded runtime's side of the stream: answers each request with what the real Hermes sent."""

    def __init__(self, scenario, *, answered=True, ask_permission=False, refuse=None):
        self.steps = [step for step in wire(scenario) if answered or "end" not in step]  # the runtime never answers
        self.session = next(step["session"] for step in self.steps if "session" in step)
        self.ask_permission = ask_permission
        self.refuse = refuse or {}  # method: the JSON-RPC error the runtime answers it with
        self.received: list[dict] = []
        self.closed = asyncio.Event()
        self.permission_answer = None

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
        results = {"initialize": {"protocolVersion": 1}, "session/new": {"sessionId": self.session}}
        try:
            while line := await self.their_reader.readline():
                message = json.loads(line)
                self.received.append(message)
                method = message.get("method")
                if method in self.refuse:
                    await self.send({"jsonrpc": "2.0", "id": message["id"], "error": self.refuse[method]})
                elif method in results:
                    await self.send({"jsonrpc": "2.0", "id": message["id"], "result": results[method]})
                elif method == "session/prompt":
                    await self._turn(message, next(prompts))
                elif "id" in message and "method" not in message:
                    self.permission_answer = message.get("result")
        except ConnectionError:
            pass  # the client closed the stream
        finally:
            self.closed.set()

    async def _turn(self, request, recorded):
        if self.ask_permission:
            options = [
                {"optionId": "allow-1", "name": "Allow", "kind": "allow_once"},
                {"optionId": "reject-1", "name": "Reject", "kind": "reject_once"},
            ]
            params = {"sessionId": self.session, "options": options, "toolCall": {"toolCallId": "tc-1", "title": "rm"}}
            await self.send({"jsonrpc": "2.0", "id": 900, "method": "session/request_permission", "params": params})
        for step in self.steps[self.steps.index(recorded) + 1 :]:
            if step.get("end") == recorded["prompt"]:
                await self.send({"jsonrpc": "2.0", "id": request["id"], "result": step["result"]})
                return
            if "update" in step and step["update"]["sessionId"] == recorded["session"]:
                await self.send({"jsonrpc": "2.0", "method": "session/update", "params": step["update"]})

    async def close(self):
        self.their_writer.close()
        self.writer.close()
        self.closed.set()

    def sent(self, method):
        return [message for message in self.received if message.get("method") == method]


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def connect(self, replay):
        self.replay, self.mailbox = replay, asyncio.Queue()
        self.client = AcpClient(replay, hermes(), permission=Reject(), start_timeout_seconds=5)
        self.addAsyncCleanup(self.client.close)
        await self.client.open()
        return await self.client.new_session("/work/w1", SERVER, self.mailbox)

    async def test_every_update_of_a_turn_is_in_the_mailbox_before_its_end(self):
        session = await self.connect(Replay("pong"))
        await session.ended(session.prompt("Reply with exactly the word: pong"))
        events = drain(self.mailbox)
        self.assertEqual((events[-1].stop_reason, events[-1].usage["totalTokens"]), ("end_turn", 712))
        self.assertEqual([event.text for event in of(events, MessageChunk)], ["pong"])
        [server] = self.replay.sent("session/new")[0]["params"]["mcpServers"]
        bearer = [{"name": "Authorization", "value": "Bearer token-1"}]
        self.assertEqual(server, {"type": "http", "name": "w1", "url": "http://bridge.test/mcp", "headers": bearer})
        init = self.replay.sent("initialize")[0]["params"]
        # No file system, no terminal: the SDK leaves out capabilities equal to ACP's defaults (all false).
        capabilities = init.get("clientCapabilities", {})
        self.assertEqual((init["protocolVersion"], capabilities.get("terminal", False)), (1, False))
        self.assertFalse(any((capabilities.get("fs") or {}).values()))

    async def test_a_cancel_goes_only_to_a_running_turn_whose_permission_requests_are_answered_cancelled(self):
        # The session takes the next prompt once its cancelled turn has ended: that turn's requests get the
        # settings' answer again.
        session = await self.connect(Replay("cancel_then_prompt", answered=False, ask_permission=True))
        self.assertFalse(await session.cancel())  # idle: never (it would poison the next prompt)

        async def answered(prompt, stop):
            await until(lambda: self.replay.permission_answer is not None)
            answer, self.replay.permission_answer = self.replay.permission_answer, None
            request = self.replay.sent("session/prompt")[-1]
            await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": stop}})
            await session.ended(prompt)
            return answer

        first = session.prompt("/count to 400")
        self.assertTrue(await session.cancel())
        self.assertEqual(await answered(first, "cancelled"), {"outcome": {"outcome": "cancelled"}})
        second = session.steer("Make it two.")
        self.assertEqual(
            await answered(second, "end_turn"), {"outcome": {"outcome": "selected", "optionId": "reject-1"}}
        )
        self.assertEqual(len(self.replay.sent("session/cancel")), 1)
        # The agent class words them: caller text never runs as one of Hermes's commands; its steer is its own.
        texts = [message["params"]["prompt"][0]["text"] for message in self.replay.sent("session/prompt")]
        self.assertEqual(texts, ["Request: /count to 400", "/steer Make it two."])

    async def test_a_lost_connection_ends_every_waiting_and_later_prompt_and_no_session_opens(self):
        async def agent_exits(replay, session):
            replay.their_writer.close()
            replay.closed.set()

        async def sender_dies(replay, session):
            # The stream to the agent broke while it ran on: the SDK's sender dies on its next write, and the SDK's
            # close then fails none of the requests still waiting.
            replay.writer.close()
            await asyncio.wait_for(session.ended(session.prompt("Count to 500.")), 2)
            replay.closed.set()  # then the agent exits

        def a_line(line):
            # An agent that breaks ACP's framing and runs on: the SDK's receive loop stops for good on the line, so
            # nothing the agent sends after it is read. Unless that is a loss, the pool keeps the process.
            async def write(replay, session):
                replay.their_writer.write(line)

            return write

        losses = {
            "the agent exits": agent_exits,
            "our sender dies": sender_dies,
            "a JSON line that is not an object": a_line(b"42\n"),
            "a line over the stream's limit": a_line(b"x" * (2 << 20) + b"\n"),
        }
        for name, lose in losses.items():
            with self.subTest(name):
                session = await self.connect(Replay("cancel_then_prompt", answered=False))
                waiting = session.prompt("Count to 400.")
                await asyncio.sleep(0.1)
                await lose(self.replay, session)
                await until(lambda: not self.client.alive)  # the pool drops it
                later = [session.steer("Make it two."), session.prompt("Count to 600.")]  # a steer as a task resumes
                await asyncio.wait_for(asyncio.gather(*(session.ended(prompt) for prompt in (waiting, *later))), 2)
                ended = {event.prompt: event.error for event in drain(self.mailbox) if isinstance(event, PromptEnded)}
                self.assertEqual([ended[prompt] for prompt in (waiting, *later)], [LOST] * 3)
                await asyncio.wait_for(self.replay.closed.wait(), 2)  # the client closed the transport: it ends
                with self.assertRaisesRegex(ConnectionError, LOST):
                    await self.client.new_session("/work/w2", SERVER, asyncio.Queue())

    async def test_an_agents_refusal_carries_its_own_reason(self):
        # Hermes started with no usable key refuses session/new so: its reason is only in the error's data.
        details = "No usable credentials found for provider 'nvidia'. Set NVIDIA_API_KEY."
        refusal = {"code": -32603, "message": "Internal error", "data": {"details": details}}
        for method in ("initialize", "session/new"):
            with (
                self.subTest(method),
                self.assertRaisesRegex(RuntimeError, f"answered {method} with an error: .*No usable"),
            ):
                await self.connect(Replay("pong", refuse={method: refusal}))
        session = await self.connect(Replay("pong", refuse={"session/prompt": refusal}))
        await session.ended(session.prompt("pong"))
        error = self.mailbox.get_nowait().error
        self.assertIn("RequestError: Internal error", error)
        self.assertIn(details, error)


if __name__ == "__main__":
    unittest.main()
