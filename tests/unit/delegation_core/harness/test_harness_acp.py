# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""The ACP layer: updates read through an agent class, the agent classes, and the client on a replay."""

import asyncio
import json
import socket
import unittest
from pathlib import Path

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.harness.acp import AcpAgent, AcpClient, SessionUpdates
from examples.frontend_backend_live.delegation_core.backend.harness.acp.agent import close_session
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes import HermesAgent
from examples.frontend_backend_live.delegation_core.backend.harness.acp.hermes.evidence import PRIVATE_ORIGIN
from examples.frontend_backend_live.delegation_core.backend.harness.permissions import Reject
from examples.frontend_backend_live.delegation_core.backend.harness.protocol import (
    McpServer,
    MessageChunk,
    PlanUpdate,
    PromptEnded,
    RuntimeLaunch,
    ThoughtChunk,
    ToolCard,
    ToolCardEnd,
    UserText,
)
from examples.frontend_backend_live.delegation_core.backend.harness.worker import WorkerRun
from tests.unit.delegation_core.harness.doubles import FakeHost, FakeTaskContext, hermes, plain, scratch, until, wire
from tests.unit.delegation_core.helpers import STEERING, finished_items, read_all

LOST = "the connection to the agent is lost"
SERVER = McpServer("w1", "http://bridge.test/mcp", "token-1")
# What an agent that closes sessions advertises in its initialize answer.
CLOSE = {"sessionCapabilities": {"close": {}}}


def hermes_events(scenario, server="client"):
    """Read a recorded scenario's updates as the Hermes agent class does (the recorded server was "client")."""
    reader = SessionUpdates(hermes(), server)
    return [reader.read(step["update"]["update"]) for step in wire(scenario) if "update" in step]


def of(events, kind):
    return [event for event in events if isinstance(event, kind)]


def chunk(text, message_id=None):
    update = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}
    return {**update, "messageId": message_id} if message_id else update


def drain(mailbox):
    return [mailbox.get_nowait() for _ in range(mailbox.qsize())]


def modern_tool_wire():
    """Read synthetic wire serialized by pinned Hermes 0.22.7, including its editor server mapping."""
    return json.loads((Path(__file__).parent / "fixtures" / "hermes_022_tool_cards.json").read_text())


class UpdateTests(unittest.TestCase):
    def test_hermes_excludes_private_origin_references_from_its_own_observed_calls(self):
        name = "mcp__editor_w1_abcdef12__get_menu"
        ref = f"nva1_{'a' * 32}"
        original = {"query": "lunch", PRIVATE_ORIGIN: ref}
        args = {"query": "lunch"}
        batch = {"calls": [{"name": name, "arguments": original}]}
        readers = (
            (SessionUpdates(hermes(), "w1"), name, original, args),
            (SessionUpdates(hermes(), "w1"), "tool_call", batch, {"calls": [{"name": name, "arguments": args}]}),
            (SessionUpdates(hermes(), "w10"), name, original, original),
            (SessionUpdates(AcpAgent(), "w1"), name, original, original),
        )
        for reader, title, raw, expected in readers:
            with self.subTest(title=title, agent=reader.agent.name, server=reader.server):
                event = reader.read(
                    {"sessionUpdate": "tool_call", "toolCallId": "native", "title": title, "rawInput": raw}
                )
                self.assertEqual(event.arguments, expected)
                self.assertEqual(original[PRIVATE_ORIGIN], ref)

    def test_modern_editor_direct_and_pure_mcp_batch_cards_belong_to_the_exact_assigned_server(self):
        recorded = modern_tool_wire()
        for kind, exchange in recorded["cases"].items():
            with self.subTest(kind=kind):
                reader = SessionUpdates(hermes(), recorded["assigned_server"])
                card = reader.read(exchange["start"])
                ended = reader.read(exchange["end"])
                self.assertIsInstance(card, ToolCard)
                self.assertTrue(card.bridged)
                self.assertEqual(card.call_id, ended.call_id)
                self.assertEqual(ended.output, "synthetic menu result")
                self.assertFalse(SessionUpdates(hermes(), "w10").read(exchange["start"]).bridged)
        # The old direct prefix and plain ACP's existing policy remain unchanged.
        self.assertTrue(hermes().bridged_call("mcp__w1__get_menu", {}, "w1"))
        self.assertTrue(AcpAgent().bridged_call("tool_call", None, "w1"))

    def test_modern_mcp_classification_does_not_hide_foreign_mixed_native_or_malformed_wrappers(self):
        own = modern_tool_wire()["cases"]["batch"]["start"]["rawInput"]["calls"][0]
        foreign = {**own, "name": own["name"].replace("editor_w1_", "editor_w10_")}
        native = {"name": "terminal", "arguments": {"command": "date"}}
        shapes = (
            None,
            {},
            {"calls": []},
            {"calls": [own], "extra": "metadata"},
            {"calls": [own, foreign]},
            {"calls": [own, native]},
            {"calls": ["not a call"]},
            {"calls": [{"name": own["name"], "arguments": "{}"}]},
            {"calls": [{"name": own["name"]}]},
            {"calls": [{**own, "extra": True}]},
        )
        for arguments in shapes:
            with self.subTest(arguments=arguments):
                card = SessionUpdates(hermes(), "w1").read(
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": "outer",
                        "title": "tool_call: mcp__w1__get_menu",  # The title is not target evidence.
                        "rawInput": arguments,
                    }
                )
                self.assertFalse(card.bridged)
        for name in (
            "mcp__editor_w10_abcdef12__get_menu",
            "mcp__editor_w1evil_abcdef12__get_menu",
            "mcp__editor_w1_ABCDEF12__get_menu",
            "mcp__editor_w1_abcdef1__get_menu",
            "other_mcp__editor_w1_abcdef12__get_menu",
            "mcp__editor_w1_abcdef12__",
        ):
            with self.subTest(name=name):
                self.assertFalse(hermes().bridged(name, "w1"))

    def test_the_recorded_hermes_wire_reads_into_events(self):
        events = hermes_events("tool_then_cancel_in_tool")
        kinds = {type(event).__name__ for event in events if event is not None}
        self.assertEqual(kinds, {"ToolCard", "ToolCardEnd", "MessageChunk", "ThoughtChunk"})
        cards = of(events, ToolCard)
        self.assertEqual([card.name for card in cards], ["mcp__client__get_time", "mcp__client__slow_lookup"])
        self.assertEqual(cards[1].arguments, {"query": "ABC123"})
        self.assertTrue(all(card.bridged for card in cards))
        self.assertFalse(any(card.bridged for card in of(hermes_events("tool_then_cancel_in_tool", "w7"), ToolCard)))
        ends = of(events, ToolCardEnd)
        self.assertEqual([end.failed for end in ends], [False, True])
        self.assertIn("14:07", ends[0].output)
        self.assertIn("MCP call interrupted", ends[1].output)
        acks = [chunk.text for chunk in of(hermes_events("redirect_and_steer"), MessageChunk) if not chunk.message_id]
        self.assertEqual(acks, ["Redirected the active turn with your correction."] * 2)
        pong = hermes_events("pong")
        [answer] = of(pong, MessageChunk)
        self.assertEqual(
            (answer.text, {thought.message_id for thought in of(pong, ThoughtChunk)}), ("pong", {answer.message_id})
        )

    def test_a_cards_output_is_its_raw_text_else_its_content(self):
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

    def test_a_plain_agents_id_less_chunks_are_one_message(self):
        reader = SessionUpdates(AcpAgent(), "w1")
        first = reader.read(chunk("Let me "))
        self.assertIsNotNone(first.message_id)
        thought = {"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Say when."}}
        self.assertEqual(reader.read(thought), ThoughtChunk(None))
        self.assertIsNone(reader.read({"sessionUpdate": "usage_update", "used": 1}))
        self.assertEqual({reader.read(chunk(text)).message_id for text in ("look.", " Still.")}, {first.message_id})
        card = reader.read({"sessionUpdate": "tool_call", "toolCallId": "t1", "title": "Looking up ABC: ABC"})
        self.assertEqual((card.name, card.bridged), ("Looking up ABC: ABC", True))
        self.assertNotEqual(reader.read(chunk("Found it.")).message_id, first.message_id)
        self.assertEqual(reader.read(chunk("x", "given-id")).message_id, "given-id")
        reader.begin_prompt()
        self.assertNotEqual(reader.read(chunk("A later invocation.")).message_id, first.message_id)

    def test_hermes_pins_its_folder_and_guards_a_steer(self):
        agent, folder = hermes(), scratch("hermes-session-")
        # No instructions: only the pin, so that no parent's file is read.
        self.assertEqual(agent.prepare(folder, ""), ())
        self.assertEqual([path.name for path in folder.iterdir()], [".git"])
        hermes(instructions_file="AGENTS.md").prepare(folder, "Hi.")
        self.assertEqual((folder / "AGENTS.md").read_text(), "Hi.\n")
        # On an idle session Hermes runs a steer's text as a prompt of its own: it gets a prompt's guard.
        self.assertEqual(agent.prompt_text("/reset the order to tea"), "Request: /reset the order to tea")
        custom = HermesAgent.from_settings({"command_guard": "Asked: "})
        self.assertEqual(custom.prompt_text("/x"), "Asked: /x")
        self.assertEqual(custom.capabilities.delivery, "cancel_reprompt")
        self.assertEqual(agent.tool_name("mcp__w1__lookup: ABC123"), "mcp__w1__lookup")
        bridged = [agent.bridged(name, "w1") for name in ("mcp__w1__lookup", "web_search", "mcp__w12__lookup")]
        self.assertEqual(bridged, [True, False, False])


class Replay:
    """The recorded runtime's side of the stream: answers each request with what the real Hermes sent.

    With ``capabilities``, its initialize answer advertises them, as an agent other than Hermes may.
    """

    def __init__(self, scenario, *, answered=True, ask_permission=False, refuse=None, capabilities=None):
        self.answered = answered
        self.steps = [step for step in wire(scenario) if answered or "end" not in step]
        self.session = next(step["session"] for step in self.steps if "session" in step)
        self.ask_permission = ask_permission
        self.refuse = refuse or {}
        self.capabilities = capabilities
        self.session_meta = None
        self.received: list[dict] = []
        self.closed = asyncio.Event()
        self.permission_answer = None
        self.runtime_stopped = False

    async def open(self):
        ours, theirs = socket.socketpair()
        self.reader, self.writer = await asyncio.open_connection(sock=ours, limit=1 << 20)
        self.their_reader, self.their_writer = await asyncio.open_connection(sock=theirs, limit=1 << 20)
        self.task = asyncio.create_task(self._serve())
        return self.reader, self.writer

    async def send(self, message):
        self.their_writer.write(json.dumps(message).encode() + b"\n")
        await self.their_writer.drain()

    async def send_record(self, writer, data, submitted):
        writer.write(data)
        submitted()
        await writer.drain()

    async def _serve(self):
        prompts = iter([step for step in self.steps if "prompt" in step])
        initialized = {"protocolVersion": 1, **({"agentCapabilities": self.capabilities} if self.capabilities else {})}
        session = {"sessionId": self.session}
        if self.session_meta is not None:
            session["_meta"] = self.session_meta
        results = {"initialize": initialized, "session/new": session, "session/close": {}}
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
            # The client closed the stream.
            pass
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
            if "cancel" in step and self.answered:
                await self._client_cancel()
            if "update" in step and step["update"]["sessionId"] == recorded["session"]:
                await self.send({"jsonrpc": "2.0", "method": "session/update", "params": step["update"]})

    async def _client_cancel(self):
        """Read on until the client's ``session/cancel`` comes: ``_serve`` reads nothing while a turn plays."""
        while not self.sent("session/cancel") and (line := await self.their_reader.readline()):
            self.received.append(json.loads(line))

    async def close(self):
        self.their_writer.close()
        self.writer.close()
        self.closed.set()
        self.runtime_stopped = True

    async def wait_closed(self):
        await self.closed.wait()

    def sent(self, method):
        return [message for message in self.received if message.get("method") == method]

    def closes(self):
        """Return the session of each ``session/close`` it read, in order."""
        return [message["params"]["sessionId"] for message in self.sent("session/close")]


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def connect(self, replay, agent=None, *, launch=None):
        self.replay, self.mailbox = replay, asyncio.Queue()
        self.client = AcpClient(
            replay, agent or hermes(), permission=Reject(), start_timeout_seconds=5, stop_grace_seconds=5, launch=launch
        )
        self.addAsyncCleanup(self.client.close)
        await self.client.open()
        return await self.client.new_session(Path("/work/w1"), SERVER, self.mailbox)

    async def test_another_agent_can_bind_steering_from_its_own_session_metadata(self):
        bindings = []

        class OtherAgent(AcpAgent):
            def bind_steering(self, launch, workspace, response, server, timeout):
                bindings.append((launch, workspace, response.session_id, response.field_meta, server, timeout))
                return None

        replay = Replay("pong")
        replay.session_meta = {"example.agent": {"handle": "invocation-scope"}}
        launch = RuntimeLaunch(("example-agent",), {}, None, Path("/work"))
        session = await self.connect(replay, OtherAgent(), launch=launch)
        self.assertEqual(bindings, [(launch, Path("/work/w1"), session.id, replay.session_meta, SERVER.name, 5)])

    async def test_every_update_of_a_turn_comes_before_its_end(self):
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

    async def test_late_and_unattributed_updates_do_not_change_the_next_prompts_answer(self):
        replay = Replay("cancel_then_prompt", answered=False)
        replay.steps = [step for step in replay.steps if "prompt" in step]
        session = await self.connect(replay, AcpAgent())

        async def send_update(update):
            await replay.send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {"sessionId": session.id, "update": update},
                }
            )

        async def end(prompt):
            request = replay.sent("session/prompt")[-1]
            await replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": "end_turn"}})
            await session.ended(prompt)

        start = {"sessionUpdate": "tool_call", "title": "Look up the booking"}
        complete = {"sessionUpdate": "tool_call_update", "status": "completed", "rawOutput": "Found the booking"}
        first = session.prompt("Check the booking.")
        await until(lambda: len(replay.sent("session/prompt")) == 1)
        await send_update({**start, "toolCallId": "old-tool"})
        await end(first)
        await send_update(chunk("An unowned update.", "unowned"))
        await until(lambda: len(session.unattributed) == 1)

        second = session.prompt("Move it to eight.")
        await until(lambda: len(replay.sent("session/prompt")) == 2)
        await send_update({**start, "toolCallId": "new-tool"})
        await send_update({**complete, "toolCallId": "new-tool"})
        await send_update(chunk("The booking "))
        await send_update({**complete, "toolCallId": "old-tool"})
        await send_update({**complete, "toolCallId": "unknown-tool"})
        await send_update(chunk("More text from the unowned update.", "unowned"))
        await send_update(chunk("is updated."))
        await end(second)

        events = drain(self.mailbox)
        self.assertEqual(
            [(event.call_id, event.prompt) for event in of(events, ToolCardEnd)],
            [("new-tool", second), ("old-tool", first)],
        )
        self.assertEqual([event.text for event in of(events, MessageChunk)], ["The booking ", "is updated."])
        self.assertEqual({event.prompt for event in of(events, MessageChunk)}, {second})
        self.assertEqual(len({event.message_id for event in of(events, MessageChunk)}), 1)
        self.assertEqual([event.call_id for event in of(session.unattributed, ToolCardEnd)], ["unknown-tool"])

    async def test_a_fenced_prompt_never_reaches_the_agent_and_does_not_poison_the_next_prompt(self):
        session = await self.connect(Replay("pong"))
        refused = session.prompt("Do not dispatch this.", may_dispatch=lambda: False)
        await session.ended(refused)
        self.assertEqual(self.replay.sent("session/prompt"), [])
        [terminal] = of(drain(self.mailbox), PromptEnded)
        self.assertEqual(terminal.dispatch, "not_dispatched")
        self.assertFalse(session.prompt_state(refused).dispatched)
        self.assertTrue(session.quiescent)
        self.assertTrue(self.client.alive)
        accepted = session.prompt("pong")
        await session.ended(accepted)
        self.assertEqual(len(self.replay.sent("session/prompt")), 1)

    async def test_retirement_before_native_dispatch_fences_the_scheduled_prompt(self):
        session = await self.connect(Replay("pong"))
        pending = session.prompt("Do not dispatch this.")
        session.retire()
        await session.ended(pending)
        self.assertEqual(self.replay.sent("session/prompt"), [])
        [terminal] = of(drain(self.mailbox), PromptEnded)
        self.assertEqual(terminal.dispatch, "not_dispatched")
        self.assertFalse(self.client.agent.completion(terminal).model_invoked)

    async def test_cancel_waiting_to_write_does_not_escape_its_original_prompt(self):
        session = await self.connect(Replay("cancel_then_prompt", answered=False))
        prompt = session.prompt("First turn.")
        await until(lambda: self.replay.sent("session/prompt"))
        request = self.replay.sent("session/prompt")[0]
        sender = self.client.conn._conn._sender
        async with sender._writing:
            cancel = asyncio.create_task(session.cancel())
            await until(lambda: session.cancelled)
            await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": "end_turn"}})
            await session.ended(prompt)
        receipt = await cancel

        next_prompt = session.prompt("Continue on this session.")
        await until(lambda: len(self.replay.sent("session/prompt")) == 2)
        request = self.replay.sent("session/prompt")[1]
        await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": "end_turn"}})
        await session.ended(next_prompt)
        self.assertEqual(self.replay.sent("session/cancel"), [])
        self.assertEqual(receipt, "not_dispatched")

    async def test_hermes_viewer_exit_does_not_claim_daemon_or_workspace_cleanup(self):
        session = await self.connect(Replay("pong"))
        await self.client.close()
        receipt = await self.client.wait_closed()
        self.assertTrue(session.retired)
        self.assertEqual((receipt.scope, receipt.confirmed, receipt.workspace_safe), ("connection", True, False))

    async def test_a_cancel_goes_only_to_a_running_turn(self):
        session = await self.connect(Replay("cancel_then_prompt", answered=False, ask_permission=True))
        # Never while idle: it would poison the next prompt.
        self.assertEqual(await session.cancel(), "not_dispatched")

        async def answered(prompt, stop):
            await until(lambda: self.replay.permission_answer is not None)
            answer, self.replay.permission_answer = self.replay.permission_answer, None
            request = self.replay.sent("session/prompt")[-1]
            await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": stop}})
            await session.ended(prompt)
            return answer

        first = session.prompt("/count to 400")
        await until(lambda: self.replay.sent("session/prompt"))
        self.assertEqual(await session.cancel(), "dispatched")
        assert await answered(first, "cancelled") == {"outcome": {"outcome": "selected", "optionId": "reject-1"}}
        second = session.prompt("Make it two.")
        assert await answered(second, "end_turn") == {"outcome": {"outcome": "selected", "optionId": "reject-1"}}
        self.assertEqual(len(self.replay.sent("session/cancel")), 1)
        texts = [message["params"]["prompt"][0]["text"] for message in self.replay.sent("session/prompt")]
        self.assertEqual(texts, ["Request: /count to 400", "Make it two."])

    async def test_a_retired_sessions_running_turn_is_still_cancelled(self):
        session = await self.connect(Replay("cancel_then_prompt", answered=False))
        prompt = session.prompt("/count to 400")
        await until(lambda: self.replay.sent("session/prompt"))
        session.retire()
        self.assertEqual(await session.cancel(), "dispatched")
        await until(lambda: self.replay.sent("session/prompt") and self.replay.sent("session/cancel"))
        request = self.replay.sent("session/prompt")[-1]
        await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "result": {"stopReason": "cancelled"}})
        await asyncio.wait_for(session.ended(prompt), 5)

    async def test_a_lost_connection_ends_every_prompt(self):
        async def agent_exits(replay, session):
            replay.their_writer.close()
            replay.closed.set()

        async def sender_dies(replay, session):
            # The SDK's sender dies on its next write, and its close then fails none of the requests still waiting.
            replay.writer.close()
            await asyncio.wait_for(session.ended(session.prompt("Count to 500.")), 2)
            replay.closed.set()

        def a_line(line):
            # The SDK's receive loop stops for good on such a line: unless that is a loss, the pool keeps the process.
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
                await until(lambda: not self.client.alive)
                later = [session.prompt("Make it two."), session.prompt("Count to 600.")]
                await asyncio.wait_for(asyncio.gather(*(session.ended(prompt) for prompt in (waiting, *later))), 2)
                ended = {event.prompt: event.error for event in drain(self.mailbox) if isinstance(event, PromptEnded)}
                self.assertEqual([ended[prompt] for prompt in (waiting, *later)], [LOST] * 3)
                await asyncio.wait_for(self.replay.closed.wait(), 2)
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
        error = next(event.error for event in drain(self.mailbox) if isinstance(event, PromptEnded))
        self.assertIn("RequestError: Internal error", error)
        self.assertIn(details, error)

    async def test_an_rpc_error_after_prompt_dispatch_does_not_prove_the_native_turn_ended(self):
        refusal = {
            "code": -32603,
            "message": "Internal error",
            "data": {"details": "Gateway disconnected; turn outcome is unknown."},
        }
        for agent in (AcpAgent(), hermes()):
            with self.subTest(agent=agent.name):
                session = await self.connect(Replay("pong", answered=False), agent=agent)
                prompt = session.prompt("Check the requested appointment.")
                await until(lambda: self.replay.sent("session/prompt"))
                [request] = self.replay.sent("session/prompt")
                await self.replay.send({"jsonrpc": "2.0", "id": request["id"], "error": refusal})
                await session.ended(prompt)
                terminal = session.prompt_state(prompt).terminal
                self.assertEqual(terminal.dispatch, "dispatched")
                self.assertIn("turn outcome is unknown", terminal.error)
                self.assertFalse(terminal.terminal_confirmed)
                self.assertFalse(session.quiescent)
                self.assertEqual(agent.completion(terminal).outcome, "unknown")
                self.assertIsNone(agent.completion(terminal).model_invoked)

                followup = session.prompt("Make it eight instead.")
                await session.ended(followup)
                denied = session.prompt_state(followup).terminal
                self.assertEqual(denied.dispatch, "not_dispatched")
                self.assertEqual(self.replay.sent("session/prompt"), [request])
                self.assertFalse(session.quiescent)

    async def test_a_cancelled_turns_batch_goes_to_the_same_session(self):
        host, replay = FakeHost(plain()), Replay("cancel_then_prompt")
        client = AcpClient(
            replay, host.settings.agent, permission=Reject(), start_timeout_seconds=5, stop_grace_seconds=5
        )
        host.runtime.client = client
        self.addAsyncCleanup(client.close)
        await client.open()
        task = FakeTaskContext(declared=())
        run = WorkerRun(task, host)
        events = run.events()
        self.assertEqual((await anext(events))["type"], "response.created")
        steering = task.steer_with(STEERING)
        run.steer()
        rest = await read_all(events, timeout=5)
        prompts, cancels = replay.sent("session/prompt"), replay.sent("session/cancel")
        self.assertEqual(len(replay.sent("session/new")), 1)
        self.assertEqual([sent["params"]["sessionId"] for sent in (*cancels, *prompts)], [replay.session] * 3)
        self.assertEqual(prompts[1]["params"]["prompt"][0]["text"], host.renderer.steering(steering))
        answers = [(item["phase"], item["content"][0]["text"]) for item in finished_items(rest)]
        self.assertEqual((answers, task.settled), ([("final_answer", "ok")], [("steer_1", "applied")]))
        self.assertEqual([session.id for session in host.idle()], [replay.session])

    async def test_a_retired_session_is_closed_once_where_its_agent_class_says(self):
        # By default where the agent advertises close. Hermes advertises none: a close it got would free nothing, and it
        # keeps each session until it exits.
        class Closes(AcpAgent):
            def close_request(self, advertised):
                return close_session

        class Keeps(AcpAgent):
            def close_request(self, advertised):
                return None

        for name, agent, capabilities, closes in (
            ("advertised", None, CLOSE, 1),
            ("not advertised", None, None, 0),
            ("a class that closes, none advertised", Closes(), None, 1),
            ("a class that keeps, close advertised", Keeps(), CLOSE, 0),
        ):
            with self.subTest(name):
                session = await self.connect(Replay("pong", capabilities=capabilities), agent)
                self.assertEqual(session.retire(), closes == 1)
                # Sent in the background, so a retire never waits for the agent.
                self.assertEqual((self.replay.closes(), session.retire()), ([], False))
                await self.client.close()
                self.assertEqual(self.replay.closes(), [session.id] * closes)

    async def test_a_failed_close_is_logged_and_the_connection_goes_on(self):
        # An agent may answer the close of a session it does not have with -32002: the core counts that session as
        # closed.
        for code, message, line in (
            (-32603, "Internal error", "harness.session_close_failed"),
            (-32002, "Resource not found", "harness.session_closed"),
        ):
            with self.subTest(code=code):
                refusal = {"code": code, "message": message}
                session = await self.connect(Replay("pong", capabilities=CLOSE, refuse={"session/close": refusal}))
                logged = []
                self.addCleanup(logger.remove, logger.add(logged.append, level="DEBUG", format="{message}"))
                session.retire()
                await until(lambda: self.replay.closes() and not self.client.closes)
                assert [each.split()[0] for each in logged if each.startswith("harness.session_close")] == [line]
                self.assertTrue(self.client.alive)

    async def test_the_clients_close_closes_each_session_before_the_stream_ends(self):
        session = await self.connect(Replay("pong", capabilities=CLOSE))
        await self.client.close()
        self.assertEqual((self.replay.closes(), self.client.closes), ([session.id], set()))
