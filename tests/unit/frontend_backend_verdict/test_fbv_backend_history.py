# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Paired backend conversation history (``backend.conversation_history``)."""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import pytest
from _fbv_fakes import (
    FakeChatClient,
    delegate_response,
    echo_tool,
    make_agent,
    make_config,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.text import events
from examples.frontend_backend_verdict.text.backend_context import (
    DISCARDED_RESULT_TEXT,
    DISCARDED_TURN_TEXT,
    undelegated_turns,
)
from examples.frontend_backend_verdict.text.config import parse_bool
from examples.frontend_backend_verdict.text.errors import ConfigError
from examples.frontend_backend_verdict.text.history import History, validate_tool_pairing
from examples.frontend_backend_verdict.text.messages import Message, ToolResult
from examples.frontend_backend_verdict.text.session import SessionState

FULL = {"enabled": True, "include": "full"}


def _contexts(sink: Any) -> list[dict[str, Any]]:
    return [event.data for event in sink.events if event.kind == events.BACKEND_CONTEXT]


def _roles(messages: list[Message]) -> list[str]:
    return [message.role for message in messages]


class FlagOffTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_requests_are_unchanged_when_off(self) -> None:
        """B1: every delegation is [system, user(query), ...], exactly as before the flag existed."""
        frontend = FakeChatClient([delegate_response("request one"), delegate_response("request two")])
        backend = FakeChatClient(
            [tool_response(("lookup", {"value": "a"}), ids=["t1"]), text_response("answer one"), text_response("two")]
        )
        agent, sink = make_agent(frontend=frontend, backend=backend, tools=[echo_tool()])
        session = agent.new_session()
        _, session = await agent.send("first", session)
        _, session = await agent.send("second", session)

        system = Message.system("You are the backend. Be helpful.")
        assert backend.calls[0]["messages"] == [system, Message.user("request one")]
        assert backend.calls[2]["messages"] == [system, Message.user("request two")]
        assert not session.backend_history
        assert not agent.config.backend.stateful
        assert _contexts(sink) == [
            {
                "enabled": False,
                "include": "off",
                "guidance": "",
                "history_groups": 0,
                "history_messages": 0,
                "earlier_turns": 0,
                "request_chars": len("request one"),
            },
            {
                "enabled": False,
                "include": "off",
                "guidance": "",
                "history_groups": 0,
                "history_messages": 0,
                "earlier_turns": 0,
                "request_chars": len("request two"),
            },
        ]


class FullModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_second_delegation_sees_the_first_turn_and_the_users_words(self) -> None:
        """B2."""
        frontend = FakeChatClient([delegate_response("look up a"), delegate_response("now change a")])
        backend = FakeChatClient(
            [tool_response(("lookup", {"value": "a"}), ids=["t1"]), text_response("found a"), text_response("done")]
        )
        agent, sink = make_agent(frontend=frontend, backend=backend, tools=[echo_tool()], conversation_history=FULL)
        assert agent.config.backend.stateful
        session = agent.new_session()
        _, session = await agent.send("find a please", session)
        turn, session = await agent.send("change it", session)
        assert turn.final_text == "done"

        second = backend.calls[2]["messages"]
        assert second[0].content == "You are the backend. Be helpful.\n\nCONTEXT full\n\nGUIDANCE full"
        assert _roles(second) == ["system", "user", "assistant", "tool", "assistant", "user"]
        assert second[1].content == "USER SAID:\nfind a please\n\nFRONTEND ASKS:\nlook up a"
        assert second[2].tool_calls[0].id == "t1"
        assert second[4].content == "found a"
        assert second[5].content == "USER SAID:\nchange it\n\nFRONTEND ASKS:\nnow change a"
        assert session.backend_history.messages[-1] == Message.assistant("done")
        validate_tool_pairing(session.backend_history.messages)
        assert _contexts(sink)[1] | {"request_chars": 0} == {
            "enabled": True,
            "include": "full",
            "guidance": "backend_history_guidance_full",
            "history_groups": 1,
            "history_messages": 4,
            "earlier_turns": 0,
            "request_chars": 0,
        }

    async def test_direct_answers_and_the_greeting_reach_the_backend_once(self) -> None:
        """B5: undelegated turns appear in the next request, and never again."""
        frontend = FakeChatClient(
            [text_response("You're welcome."), delegate_response("do x"), delegate_response("do y")]
        )
        backend = FakeChatClient([text_response("x done"), text_response("y done")])
        agent, _ = make_agent(frontend=frontend, backend=backend, conversation_history=FULL)
        session = agent.new_session()
        session = replace(session, frontend_history=History((Message.assistant("Hi! How can I help?"),)))
        _, session = await agent.send("thanks", session)
        _, session = await agent.send("please do x", session)
        _, session = await agent.send("and y", session)

        first_request = backend.calls[0]["messages"][-1].content
        assert first_request.startswith(
            "Conversation since your last reply (oldest first):\n"
            "Assistant: Hi! How can I help?\nUser: thanks\nAssistant: You're welcome.\n\nUSER SAID:\nplease do x"
        )
        second_request = backend.calls[1]["messages"][-1].content
        assert second_request == "USER SAID:\nand y\n\nFRONTEND ASKS:\ndo y"

    async def test_guidance_can_be_disabled_but_the_context_note_stays(self) -> None:
        backend = FakeChatClient([text_response("ok")])
        agent, sink = make_agent(
            frontend=FakeChatClient([delegate_response("q")]),
            backend=backend,
            conversation_history={**FULL, "guidance_key": ""},
        )
        await agent.send("hi", agent.new_session())
        assert backend.calls[0]["messages"][0].content == "You are the backend. Be helpful.\n\nCONTEXT full"
        assert _contexts(sink)[0]["guidance"] == ""

    async def test_pruning_keeps_whole_groups(self) -> None:
        frontend = FakeChatClient([delegate_response(f"q{index}") for index in range(3)])
        backend = FakeChatClient(
            [
                tool_response(("lookup", {}), ids=["a"]),
                text_response("one"),
                tool_response(("lookup", {}), ids=["b"]),
                text_response("two"),
                text_response("three"),
            ]
        )
        agent, _ = make_agent(
            frontend=frontend,
            backend=backend,
            tools=[echo_tool()],
            conversation_history=FULL,
            backend_history={"max_groups": 1},
        )
        session = agent.new_session()
        for text in ("u0", "u1", "u2"):
            _, session = await agent.send(text, session)
        assert len(session.backend_history.groups()) == 1
        assert session.backend_history.messages[-1].content == "three"
        # the third call saw exactly one earlier group, with its tool pair intact
        assert _roles(backend.calls[4]["messages"]) == ["system", "user", "assistant", "tool", "assistant", "user"]
        validate_tool_pairing(backend.calls[4]["messages"])


class OtherIncludeTests(unittest.IsolatedAsyncioTestCase):
    async def test_backend_turns_keeps_history_and_sends_the_bare_query(self) -> None:
        """B3."""
        frontend = FakeChatClient([delegate_response("q1"), delegate_response("q2")])
        backend = FakeChatClient([text_response("a1"), text_response("a2")])
        agent, _ = make_agent(
            frontend=frontend, backend=backend, conversation_history={"enabled": True, "include": "backend_turns"}
        )
        session = agent.new_session()
        _, session = await agent.send("u1", session)
        _, session = await agent.send("u2", session)
        assert backend.calls[1]["messages"][1:] == [Message.user("q1"), Message.assistant("a1"), Message.user("q2")]
        assert backend.calls[1]["messages"][0].content.endswith("CONTEXT backend_turns\n\nGUIDANCE backend_turns")

    async def test_transcript_is_stateless_but_carries_the_conversation(self) -> None:
        """B4."""
        frontend = FakeChatClient([delegate_response("q1"), delegate_response("q2")])
        backend = FakeChatClient([text_response("a1"), text_response("a2")])
        agent, _ = make_agent(
            frontend=frontend, backend=backend, conversation_history={"enabled": True, "include": "transcript"}
        )
        assert not agent.config.backend.stateful
        session = agent.new_session()
        _, session = await agent.send("u1", session)
        _, session = await agent.send("u2", session)
        second = backend.calls[1]["messages"]
        assert _roles(second) == ["system", "user"]
        assert second[1].content == (
            "Conversation so far (oldest first):\nUser: u1\nAssistant: a1\n\nUSER SAID:\nu2\n\nFRONTEND ASKS:\nq2"
        )
        assert not session.backend_history


class ExternalToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_suspend_and_resume_across_turns(self) -> None:
        """B9."""
        frontend = FakeChatClient([delegate_response("q1"), delegate_response("q2")])
        backend = FakeChatClient(
            [
                tool_response(("lookup", {}), ids=["x1"]),
                text_response("a1"),
                tool_response(("lookup", {}), ids=["x2"]),
                text_response("a2"),
            ]
        )
        agent, _ = make_agent(
            frontend=frontend, backend=backend, conversation_history=FULL, tools_config={"execution": "external"}
        )
        session = agent.new_session()
        _, session = await agent.send("u1", session)
        _, session = await agent.send_tool_results([ToolResult("x1", "r1")], session)
        turn, session = await agent.send("u2", session)
        assert turn.is_tool_call and session.pending is not None
        assert _roles(list(session.pending.backend_history.messages)) == [
            "user", "assistant", "tool", "assistant", "user", "assistant",
        ]  # fmt: skip
        turn, session = await agent.send_tool_results([ToolResult("x2", "r2")], session)
        assert turn.final_text == "a2"
        assert len(session.backend_history.groups()) == 2
        validate_tool_pairing(session.backend_history.messages)

    async def test_discarded_pending_turn_is_closed_in_the_backend_history(self) -> None:
        """§4.4: executed-but-unanswered calls stay visible and paired."""
        frontend = FakeChatClient([delegate_response("write it"), delegate_response("status?")])
        backend = FakeChatClient([tool_response(("update", {"v": 1}), ids=["w1"]), text_response("unknown yet")])
        agent, sink = make_agent(
            frontend=frontend,
            backend=backend,
            conversation_history=FULL,
            tools_config={"execution": "external", "on_user_message_while_pending": "discard_pending"},
        )
        session = agent.new_session()
        _, session = await agent.send("write it please", session)
        _, session = await agent.send("did it work?", session)

        seen = backend.calls[1]["messages"]
        assert _roles(seen) == ["system", "user", "assistant", "tool", "assistant", "user"]
        assert seen[3] == Message.tool("w1", DISCARDED_RESULT_TEXT)
        assert seen[4] == Message.assistant(DISCARDED_TURN_TEXT)
        validate_tool_pairing(session.backend_history.messages)
        discarded = [event.data for event in sink.events if event.kind == events.PENDING_DISCARDED]
        assert discarded == [{"outstanding": ["w1"], "closed_in_backend_history": True}]

    async def test_discard_without_history_drops_the_turn_as_before(self) -> None:
        frontend = FakeChatClient([delegate_response("write it"), delegate_response("status?")])
        backend = FakeChatClient([tool_response(("update", {}), ids=["w1"]), text_response("?")])
        agent, _ = make_agent(
            frontend=frontend,
            backend=backend,
            tools_config={"execution": "external", "on_user_message_while_pending": "discard_pending"},
        )
        session = agent.new_session()
        _, session = await agent.send("a", session)
        _, session = await agent.send("b", session)
        assert _roles(backend.calls[1]["messages"]) == ["system", "user"]
        assert not session.backend_history


class SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_round_trip_keeps_the_paired_backend_history(self) -> None:
        frontend = FakeChatClient([delegate_response("q")])
        backend = FakeChatClient([tool_response(("lookup", {}), ids=["t"]), text_response("a")])
        agent, _ = make_agent(frontend=frontend, backend=backend, tools=[echo_tool()], conversation_history=FULL)
        _, session = await agent.send("u", agent.new_session())
        restored = SessionState.from_dict(session.to_dict())
        assert restored.backend_history == session.backend_history
        assert restored.frontend_history == session.frontend_history

    def test_replay_is_lossy_for_the_backend_but_feeds_the_first_request(self) -> None:
        replayed = SessionState.from_message_history(
            [Message.assistant("Hi!"), Message.user("hello"), Message.assistant("How can I help?")],
            backend_only=False,
        )
        assert not replayed.backend_history
        assert [m.content for m in undelegated_turns(replayed.frontend_history)] == ["Hi!", "hello", "How can I help?"]


class ConfigTests(unittest.TestCase):
    def test_default_is_off(self) -> None:
        config = make_config()
        assert not config.backend.conversation_history.enabled
        assert config.backend.conversation_history.effective_include == "off"
        assert config.backend.conversation_history.resolved_guidance_key == ""

    def test_environment_strings_parse_strictly(self) -> None:
        for text, expected in (("false", False), ("False", False), ("", False), ("0", False), ("true", True),
                               (" ON ", True), ("1", True)):  # fmt: skip
            assert parse_bool(text, name="x") is expected
        with pytest.raises(ConfigError, match="boolean"):
            parse_bool("maybe", name="x")
        assert not make_config(conversation_history={"enabled": "false"}).backend.conversation_history.enabled

    def test_stateful_is_derived_per_include(self) -> None:
        for include, stateful in (("full", True), ("backend_turns", True), ("transcript", False)):
            config = make_config(conversation_history={"enabled": True, "include": include})
            assert config.backend.stateful is stateful

    def test_invalid_include_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="include"):
            make_config(conversation_history={"enabled": True, "include": "everything"})

    def test_backend_only_rejects_the_flag(self) -> None:
        with pytest.raises(ConfigError, match="backend_only already keeps"):
            make_config(mode="backend_only", conversation_history={"enabled": True})

    def test_explicit_stateful_in_paired_mode_points_to_the_flag(self) -> None:
        with pytest.raises(ConfigError, match="conversation_history.enabled"):
            make_config(backend={"llm": {"model": "m"}, "stateful": True})

    def test_request_template_must_carry_its_placeholders(self) -> None:
        config = make_config(conversation_history=FULL)
        inline = {**config.prompts.inline, "backend_history_request": "no placeholders"}
        broken = replace(config, prompts=replace(config.prompts, inline=inline))
        with pytest.raises(ConfigError, match="must contain"):
            make_agent(config=broken)
