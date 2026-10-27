# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Both delegation parts are a ``live.ports.DelegationPart``, picked from ``DELEGATION_PARTS`` by delegation type.

An omitted, ``null`` or ``{}`` delegation builds the client part, and the frontend adapter gets only the narrow
``DelegationPort``.
"""

import inspect
import unittest

from live.client_delegation import ClientDelegation
from live.ports import DelegationPart, DelegationPort, DelegationView
from live.protocol import ProtocolError, SessionConfig
from live.server_delegation import ServerDelegation
from live.session import LiveProtocolSession
from live.session.session import DELEGATION_PARTS, DelegationPortView, DelegationSideView
from tests.unit.live_helpers import Collector, FakeAdapter, responses_config

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]


def declared(protocol) -> set[str]:
    """The members a port declares: its methods and its annotated attributes, its port bases' included."""
    names = set()
    for cls in protocol.__mro__:
        if cls.__module__ == "live.ports":
            names |= {name for name in vars(cls) if not name.startswith("_")} | set(cls.__annotations__)
    return names


def public(view) -> set[str]:
    """The public names of an object."""
    return {name for name in dir(view) if not name.startswith("_")}


class View:
    """The session's delegation view, as a part reads it."""

    id, input_ms, responses, acknowledgement_delay_ms = "live_test", 0, None, 0

    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, kind, **fields):
        self.events.append((kind, fields))


class InterfaceTests(unittest.TestCase):
    def test_both_parts_carry_every_member_of_the_delegation_part(self):
        members = declared(DelegationPart)
        self.assertLessEqual(declared(DelegationPort), members)
        for part in (ClientDelegation(), ServerDelegation(FakeAdapter())):
            with self.subTest(part=type(part).__name__):
                self.assertEqual(sorted(members - public(part)), [])
                for name in ("open", "close", "update", "item_create", "response_create", "history", "delegate"):
                    self.assertTrue(inspect.iscoroutinefunction(getattr(part, name)), name)

    def test_the_table_has_a_part_for_each_delegation_type_and_only_a_responses_part_asks_for_the_core(self):
        self.assertEqual(set(DELEGATION_PARTS), {"client", "responses"})
        asked = []

        def core():
            asked.append(True)
            return FakeAdapter()

        self.assertIsInstance(DELEGATION_PARTS["client"](core), ClientDelegation)
        self.assertEqual(asked, [])  # a client-mode session builds no core
        self.assertIsInstance(DELEGATION_PARTS["responses"](core), ServerDelegation)
        self.assertEqual(asked, [True])

    def test_an_omitted_null_or_empty_delegation_builds_the_client_part(self):
        for delegation in (None, {}, {"type": "client"}):
            with self.subTest(delegation=delegation):
                fields = {} if delegation is None else {"delegation": delegation}
                session = LiveProtocolSession(
                    SessionConfig(model="live-1", **fields), "websocket", FakeAdapter().parts()
                )
                self.assertIsInstance(session.delegation, ClientDelegation)
        engine = FakeAdapter()
        session = LiveProtocolSession(responses_config(), "websocket", engine.parts(), core=engine.core)
        self.assertIsInstance(session.delegation, ServerDelegation)

    def test_a_responses_session_on_a_server_with_no_core_is_refused(self):
        with self.assertRaises(ProtocolError) as refused:
            LiveProtocolSession(responses_config(), "websocket", FakeAdapter().parts())
        self.assertEqual(refused.exception.param, "session.delegation.type")

    def test_the_adapter_gets_only_the_narrow_port_and_the_part_only_its_view(self):
        engine = FakeAdapter()
        session = LiveProtocolSession(responses_config(), "websocket", engine.parts(), core=engine.core)
        self.assertEqual(public(session.delegation_port), declared(DelegationPort))
        self.assertIsInstance(session.delegation_port, DelegationPortView)
        self.assertEqual(public(session.delegation_view), declared(DelegationView))
        self.assertIsInstance(session.delegation_view, DelegationSideView)
        self.assertNotIn("config", public(session.delegation_view))  # only what the part reads of the settings


class AdapterViewTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_adapter_starts_with_the_port_view_which_reads_the_settings_when_asked(self):
        engine = FakeAdapter()
        config = SessionConfig(model="live-1", delegation={"type": "client", "acknowledgement_delay_ms": 300})
        session = LiveProtocolSession(config, "websocket", engine.parts(), core=engine.core)
        connection = session.subscribe(Collector().send, "primary")
        await session.start()
        self.addAsyncCleanup(session.close)
        self.assertIs(engine.delegation, session.delegation_port)
        self.assertEqual((engine.delegation.target, engine.delegation.reports_events), ("client", False))
        self.assertEqual(engine.delegation.acknowledgement_delay_ms, 300)
        update = {"delegation": {"type": "client", "acknowledgement_delay_ms": 0}}
        await session.handle({"type": "session.update", "event_id": "u1", "session": update}, connection)
        self.assertEqual(engine.delegation.acknowledgement_delay_ms, 0)  # applies to later delegations


class UpdateCaseTests(unittest.IsolatedAsyncioTestCase):
    """Each ``session.update`` case, through the part the session holds."""

    async def start(self, config):
        self.engine = FakeAdapter()
        self.session = LiveProtocolSession(config, "websocket", self.engine.parts(), core=self.engine.core)
        self.primary = Collector()
        self.connection = self.session.subscribe(self.primary.send, "primary")
        await self.session.start("s")
        await self.primary.wait_for("session.started")
        self.addAsyncCleanup(self.session.close)

    async def update(self, patch, event_id="u"):
        before = len(self.primary.events)
        await self.session.handle({"type": "session.update", "event_id": event_id, "session": patch}, self.connection)
        for subscriber in tuple(self.session.fanout.subscribers.values()):
            await subscriber.queue.join()
        return self.primary.events[before:]

    async def test_a_client_session_takes_its_acknowledgement_delay_and_null_or_empty_as_no_ops(self):
        await self.start(SessionConfig(model="live-1"))
        for patch in ({}, {"delegation": None}, {"delegation": {}}, {"delegation": {"acknowledgement_delay_ms": 250}}):
            with self.subTest(patch=patch):
                [updated] = await self.update(patch)
                self.assertEqual((updated["type"], updated["client_event_id"]), ("session.updated", "u"))
        self.assertEqual(self.session.config.delegation.acknowledgement_delay_ms, 250)
        self.assertEqual(updated["session"]["delegation"], {"type": "client", "acknowledgement_delay_ms": 250})

    async def test_a_client_session_refuses_a_type_change_backend_settings_and_triggers_each_with_its_param(self):
        await self.start(SessionConfig(model="live-1", delegation={"type": "client"}))
        cases = (
            ({"delegation": {"type": "responses"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"model": "m"}}}, "unknown_parameter", "session.delegation.responses"),
            ({"delegation": {"decisions": {}}}, "unknown_parameter", "session.delegation.decisions"),
            ({"triggers": [{"event_type": "session.input_transcript.delta"}]}, "invalid_value", "session.triggers"),
            ({"instructions": "Be terse."}, "immutable_field_update", "session.instructions"),
            ({"voice": "marin"}, "unknown_parameter", "session.voice"),
        )
        for patch, code, param in cases:
            with self.subTest(patch=patch):
                [error] = await self.update(patch, "bad")
                self.assertEqual(
                    (error["type"], error["error"]["code"], error["error"]["param"], error["error"]["client_event_id"]),
                    ("error", code, param, "bad"),
                )
        self.assertEqual(self.session.config, SessionConfig(model="live-1"))

    async def test_a_responses_session_merges_its_settings_and_refuses_null_and_a_type_change(self):
        await self.start(responses_config())
        [updated] = await self.update({"delegation": {"responses": {"reasoning": {"effort": "high"}}}})
        self.assertEqual(
            updated["session"]["delegation"]["responses"]["reasoning"], {"effort": "high", "summary": "auto"}
        )
        [updated] = await self.update({"delegation": {"type": "responses", "acknowledgement_delay_ms": 400}})
        self.assertEqual(self.session.config.delegation.acknowledgement_delay_ms, 400)
        for patch, code, param in (
            ({"delegation": None}, "immutable_field_update", "session.delegation"),
            ({"delegation": {"type": "client"}}, "immutable_field_update", "session.delegation.type"),
            ({"delegation": {"responses": {"bogus": 1}}}, "unknown_parameter", "session.delegation.responses.bogus"),
        ):
            with self.subTest(patch=patch):
                [error] = await self.update(patch)
                self.assertEqual((error["error"]["code"], error["error"]["param"]), (code, param))


class ClientCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_client_part_refuses_the_responses_only_commands_with_todays_code(self):
        part = ClientDelegation()
        await part.open(View())
        for command in (part.item_create({"type": "function_call_output"}, "i1"), part.response_create("r1")):
            with self.assertRaises(ProtocolError) as refused:
                await command
            self.assertEqual(
                (refused.exception.code, str(refused.exception)),
                ("invalid_value", "This command requires Responses delegation"),
            )
        self.assertIsNone(await part.history())
        self.assertFalse(ServerDelegation(FakeAdapter()).knows("item_1"))


class ListenFirstTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_server_delegation_refuses_a_delegation_before_listen_and_sends_nothing(self):
        core = FakeAdapter()
        part = ServerDelegation(core)
        await part.open(View())
        self.addAsyncCleanup(part.close)
        with self.assertRaisesRegex(RuntimeError, "listen"):
            await part.delegate(HI)
        self.assertEqual(core.creates, 0)


if __name__ == "__main__":
    unittest.main()
