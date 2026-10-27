# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D105, D107

"""The gateway's server delegation on a real delegation core, with no frontend and no Pipecat.

The core is the one ``build_core`` makes from the shipped ``core.yaml``, reached through its in-process client, with
its backend LLM on a scripted provider. The part's own rules run on fake cores in ``tests/unit/live``.
"""

import asyncio
import unittest

from delegation_core.im.input import BackendTask, settings_for
from live.protocol import SessionConfig
from live.server_delegation.passthrough import ServerDelegation, client_settings
from tests.unit.cascade_helpers import MENU_TOOL, _delegation, core_session
from tests.unit.contracts.backend import ScriptedBackend, call_round, text_round
from tests.unit.core_helpers import answer_text

HI = [{"role": "user", "content": [{"type": "input_text", "text": "Hi."}]}]


def typed(text):
    """A typed user message, as ``response.item.create`` carries it."""
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def output(call_id, text="{}"):
    """The client's output for the call ``call_id``."""
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def responses(model="test/model", instructions=None):
    """A session's stored settings: its ``delegation.responses`` block, which declares the client's ``get_menu``."""
    block = {"model": model, "tools": [MENU_TOOL], **({"instructions": instructions} if instructions else {})}
    return SessionConfig.model_validate({"model": "live-1", "delegation": {"type": "responses", "responses": block}})


class Provider(ScriptedBackend):
    """The backend LLM's scripted provider: it plays its rounds, and counts how often its IM session releases it."""

    def __init__(self, *rounds):
        super().__init__(list(rounds))
        self.closes = 0

    async def close(self):
        self.closes += 1


class RecordingView:
    """The session's delegation view (``live.ports.DelegationView``): it notes each event relayed and each refusal."""

    id, input_ms, acknowledgement_delay_ms = "live_test", 0, 0

    def __init__(self, config=None):
        self.responses = (config or responses()).delegation.responses
        self.events: list[str] = []
        self.relayed: list[str] = []
        self.calls: list[str] = []
        self.refusals: list[tuple] = []

    async def emit(self, kind, **fields):
        self.events.append(kind)
        if kind == "response.event":
            event = fields["event"]
            self.relayed.append(event["type"])
            if event["type"] == "response.output_item.done" and event["item"].get("type") == "function_call":
                self.calls.append(event["item"]["call_id"])

    async def emit_paced(self, kind, **fields):
        await self.emit(kind, **fields)

    def report_refusal(self, message, code, client_event_id, param=None):
        self.refusals.append((code, message, client_event_id))


class RecordingFrontend:
    """The frontend's ``DelegationEvents``: each event it is handed, the answers read from them, and each end."""

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.results: list[str] = []
        self.ended: list[str] = []

    def on_delegation_event(self, delegation_id, event):
        self.events.append((delegation_id, event["type"]))
        text = answer_text(event)
        if text is not None:
            self.results.append(text)
        elif event["type"] in ("response.completed", "response.failed", "response.incomplete"):
            self.ended.append(delegation_id)

    def on_delegation_lost(self, delegation_id):
        self.ended.append(delegation_id)


class FailingFrontend(RecordingFrontend):
    """The frontend's ``DelegationEvents``, which fails on the first event it is handed."""

    def on_delegation_event(self, delegation_id, event):
        super().on_delegation_event(delegation_id, event)
        if len(self.events) == 1:
            raise RuntimeError("the frontend failed")


class SideCase(unittest.IsolatedAsyncioTestCase):
    async def side(self, provider, view=None, frontend=None, **delegation):
        """A live session's server delegation on a core whose backend LLM runs on ``provider``, opened and listening."""
        self.provider = provider
        self.view = view if view is not None else RecordingView()
        self.frontend = frontend or RecordingFrontend()
        side = _delegation(provider, **delegation)
        self.sent: list[dict] = []
        create = side.client.create_response

        async def noted(params):
            self.sent.append(params)
            return await create(params)

        side.client.create_response = noted
        side.listen(self.frontend)
        await side.open(self.view)
        self.addAsyncCleanup(side.close)
        return side

    async def until(self, predicate, timeout=3.0):
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------------------------- close
class StopHandingTests(SideCase):
    async def test_after_stop_handing_nothing_is_handed_to_the_frontend_and_the_client_still_gets_every_event(self):
        # Close step 2: the frontend is handed nothing more. The client still gets what the core writes, so the
        # ends the core writes at close step 4 reach it before session.closed.
        for stopped in (False, True):
            with self.subTest(stop_handing=stopped):
                gate = asyncio.Event()
                side = await self.side(Provider([gate, *text_round("Done.")]))
                await side.delegate(HI)
                await self.until(lambda: self.provider.payloads)  # the round runs, held at the gate
                if stopped:
                    side.stop_handing()
                gate.set()
                await self.until(lambda: "response.completed" in self.view.relayed)
                await side.close()
                self.assertEqual(self.frontend.results, [] if stopped else ["Done."])

    async def test_a_delegation_the_close_ends_reaches_the_client_and_never_the_frontend(self):
        # The close (step 4, after step 2) deletes the conversation, which ends the running delegation's response in the
        # core: its reader relays that end to the client, and the frontend, which is closed, gets nothing.
        side = await self.side(Provider([asyncio.Event(), *text_round("never")]))
        await side.delegate(HI)
        await self.until(lambda: self.provider.payloads)
        await side.close()
        self.assertEqual(self.view.relayed[-1], "response.incomplete")
        self.assertEqual(self.frontend.ended, [])


class OpenAndCloseTests(SideCase):
    async def test_a_part_closed_before_it_opens_starts_nothing(self):
        # Close is safe before open, and a session closed while starting closes both: no conversation is opened.
        side = _delegation(Provider())
        await side.close()
        await side.open(RecordingView())
        self.assertEqual((side.conversation, side.routes, side.relay), (None, None, None))
        self.assertEqual(side.client.service.sessions, {})

    async def test_a_second_close_releases_nothing_again(self):
        side = await self.side(Provider(text_round("Done.")))
        await side.delegate(HI)
        await self.until(lambda: self.frontend.results)
        await side.close()
        await side.close()
        self.assertEqual(self.provider.closes, 1)  # the IM session's provider, released with its conversation

    async def test_a_delegation_before_open_or_after_close_raises_and_sends_no_request(self):
        # Every request names the session's conversation, which exists from open until close.
        side = _delegation(Provider())
        side.listen(RecordingFrontend())
        with self.assertRaisesRegex(RuntimeError, "^The server delegation is not open$"):
            await side.delegate(HI)
        await side.open(RecordingView())
        await side.close()
        with self.assertRaisesRegex(RuntimeError, "^Session is closing$"):
            await side.delegate(HI)
        self.assertEqual(side.readers, set())


# ------------------------------------------------------------------------------------- events and requests
class FirstEventTests(SideCase):
    async def test_no_event_of_a_delegation_is_handed_before_delegate_returns(self):
        # delegate() returns D once the core admitted the request, and only then does the part read D's
        # events, so the frontend notes D before its first event, which would otherwise read as a typed run.
        side = await self.side(Provider(text_round("Done.")))
        delegation_id = await side.delegate(HI)
        self.assertEqual(self.frontend.events, [])
        await self.until(lambda: self.frontend.results)
        self.assertEqual(self.frontend.events[0], (delegation_id, "response.created"))
        self.assertEqual({handed for handed, _ in self.frontend.events}, {delegation_id})
        self.assertEqual(self.view.events[0], "session.delegation.created")


class RelayTests(SideCase):
    async def test_a_failed_delegation_reaches_the_client_with_its_response_failed(self):
        # The relay drops nothing: a failure ends the delegation's response response.failed, and the client gets it.
        side = await self.side(Provider([RuntimeError("one")], [RuntimeError("two")]))  # the round and its one retry
        delegation_id = await side.delegate(HI)
        await self.until(lambda: delegation_id in self.frontend.ended)
        self.assertEqual(
            self.view.relayed, ["response.created", "response.queued", "response.in_progress", "response.failed"]
        )

    async def test_a_frontend_that_fails_on_an_event_still_gets_the_later_ones_and_the_client_gets_them_all(self):
        side = await self.side(Provider(text_round("Done.")), frontend=FailingFrontend())
        await side.delegate(HI)
        await self.until(lambda: self.frontend.results)
        await self.until(lambda: "response.completed" in self.view.relayed)
        self.assertEqual(self.view.relayed, [kind for _, kind in self.frontend.events])
        self.assertEqual(self.frontend.results, ["Done."])


class RequestTests(SideCase):
    async def test_each_request_names_its_live_session_and_its_delegation_in_its_metadata(self):
        # The gateway-to-core convention: the metadata keys live_session_id and live_delegation_id, opaque to the core.
        # A part with no live session (the UI bot) names only the delegation.
        for view in (RecordingView(), None):
            with self.subTest(live_session=view is not None):
                side = _delegation(Provider(text_round("Done.")))
                sent = []
                create = side.client.create_response

                async def noted(params, create=create, sent=sent):
                    sent.append(params)
                    return await create(params)

                side.client.create_response = noted
                side.listen(RecordingFrontend())
                await side.open(view)
                delegation_id = await side.delegate(HI)
                await side.close()
                expected = {"live_delegation_id": delegation_id}
                if view is not None:
                    expected["live_session_id"] = view.id
                self.assertEqual([params["metadata"] for params in sent], [expected])

    async def test_each_request_names_the_conversation_streams_in_the_background_unstored_with_the_stored_settings(
        self,
    ):
        # The official endpoints with the gateway-to-core conventions: every request, a delegation or the client's typed
        # run, names the session's conversation, sets stream, background and store to true, true and false, and
        # carries what the client set in the session's delegation block, read when it is sent; the core fills the rest.
        view = RecordingView(responses("org/asked", "Asked."))
        side = await self.side(Provider(text_round("Open at nine."), text_round("Done.")), view=view)
        await side.item_create(typed("When do you open?"), None)
        await side.response_create("typed")  # the client's typed run
        await self.until(lambda: self.frontend.results)
        await side.delegate(HI)
        await self.until(lambda: len(self.frontend.results) == 2)
        fields = ("conversation", "stream", "background", "store", "model", "instructions")
        expected = (side.conversation, True, True, False, "org/asked", "Asked.")
        self.assertEqual([tuple(params[name] for name in fields) for params in self.sent], [expected, expected])
        self.assertEqual([params["input"] for params in self.sent], [[typed("When do you open?")], HI])
        self.assertEqual(self.sent[0]["metadata"]["live_typed_run"], "true")
        self.assertNotIn("live_typed_run", self.sent[1]["metadata"])

    async def test_a_session_update_changes_the_backend_settings_from_the_next_request(self):
        # Each request carries the session's stored settings of its time, so a session.update applies to the
        # requests sent after it; work already sent keeps the settings it was sent with.
        gate = asyncio.Event()
        view = RecordingView(responses("new/model", "New."))
        side = await self.side(Provider([gate, *text_round("first")], text_round("second")), view=view)
        await side.delegate(HI)
        await self.until(lambda: self.provider.payloads)
        view.responses = responses("newer/model", "Newer.").delegation.responses  # a session.update stores new ones
        await side.delegate([typed("And more.")])
        gate.set()
        await self.until(lambda: len(self.sent) == 2)
        self.assertEqual(
            [(params["model"], params["instructions"]) for params in self.sent],
            [("new/model", "New."), ("newer/model", "Newer.")],
        )


# ----------------------------------------------------------------------------- the client's outputs and input
class ContinuationTests(SideCase):
    """The client's outputs come back in a continuation request of their own."""

    async def answer_calls(self, side, event_id="continue"):
        for call_id in self.view.calls:
            await side.item_create(output(call_id), None)
        self.view.calls = []
        await side.response_create(event_id)

    async def test_two_client_tool_rounds_in_one_delegation_answer(self):
        # Each round's calls end their response, and the client's outputs continue the work in a request of their own.
        side = await self.side(
            Provider(call_round("get_menu", {}, "c1"), call_round("get_menu", {}, "c2"), text_round("Done."))
        )
        await side.delegate(HI)
        for _ in range(2):
            await self.until(lambda: self.view.relayed.count("response.completed") > len(self.sent) - 1)
            await self.answer_calls(side)
        await self.until(lambda: self.frontend.results)
        self.assertEqual((self.frontend.results, self.view.refusals, len(self.provider.payloads)), (["Done."], [], 3))

    async def test_outputs_sent_before_response_completed_are_accepted(self):
        # A route opens when its call is relayed, so the client may return the output before the round's
        # response.completed; the response.create after it continues the work.
        side = await self.side(Provider(call_round("get_menu", {}, "c1"), text_round("Done.")))
        await side.delegate(HI)
        await self.until(lambda: self.view.calls)
        for call_id in self.view.calls:
            await side.item_create(output(call_id), None)
        await self.until(lambda: "response.completed" in self.view.relayed)
        await side.response_create("continue")
        await self.until(lambda: self.frontend.results)
        self.assertEqual(self.frontend.results, ["Done."])
        self.assertEqual(self.provider.payloads[1]["input"][-1]["type"], "function_call_output")

    async def test_text_or_nothing_while_outputs_are_owed_is_refused_and_kept(self):
        # A response.create that would continue nothing while a function result is owed is refused
        # with the official code; the typed message stays pending and goes with the outputs once they come.
        side = await self.side(Provider(call_round("get_menu", {}, "c1"), text_round("Done.")))
        await side.delegate(HI)
        await self.until(lambda: "response.completed" in self.view.relayed)
        await side.item_create(typed("Only green tea."), None)
        with self.assertRaises(Exception) as refused:
            await side.response_create("create")
        self.assertEqual(getattr(refused.exception, "code", None), "function_call_outputs_required")
        self.assertEqual(core_session(side).ledger.works["W1"].state, "waiting_client")
        await self.answer_calls(side)
        await self.until(lambda: self.frontend.results)
        self.assertEqual(self.provider.payloads[1]["input"][-1], typed("Only green tea."))

    async def test_at_close_every_route_closes(self):
        side = await self.side(Provider(call_round("get_menu", {}, "c1"), text_round("Done.")))
        await side.delegate(HI)
        await self.until(lambda: self.view.calls)
        self.assertEqual(list(side.routes.routes), self.view.calls)
        await side.item_create(typed("Anything else?"), None)
        await side.close()
        self.assertEqual((side.routes.routes, side.routes.messages), ({}, []))


class LateOutputTests(SideCase):
    async def test_outputs_after_the_cores_deadline_get_an_empty_completed_response(self):
        # The delegation's call ran out at the core's client-tool deadline (core.yaml client_tool_timeout_seconds,
        # 0.05 s here). The client then returns the output: the core answers its continuation with an empty
        # response.completed, and nothing stays pending.
        side = await self.side(Provider(call_round("get_menu", {}, "c1")), client_tool_timeout_seconds=0.05)
        delegation_id = await side.delegate(HI)
        await self.until(lambda: self.view.calls)
        await self.until(lambda: not core_session(side).busy)  # the deadline passed: the work ended
        for call_id in self.view.calls:
            await side.item_create(output(call_id), None)
        await side.response_create("late")
        await self.until(lambda: self.view.relayed.count("response.completed") == 2)
        kinds = [kind for handed, kind in self.frontend.events if handed == delegation_id]
        self.assertEqual(
            kinds[-4:], ["response.created", "response.queued", "response.in_progress", "response.completed"]
        )
        self.assertEqual(self.view.refusals, [])


class TypedInputTests(SideCase):
    async def test_typed_input_while_a_delegation_runs_is_placed_by_the_core(self):
        # The core places every request: the steer conductor joins a typed run to the open Work, and the client is
        # told of no refusal.
        gate = asyncio.Event()
        side = await self.side(Provider([gate, *text_round("Done.")], text_round("Done too.")))
        await side.delegate(HI)
        await self.until(lambda: self.provider.payloads)
        await side.item_create(typed("When do you open?"), None)
        await side.response_create("typed")
        gate.set()
        await self.until(lambda: len(self.sent) == 2 and not core_session(side).busy)
        self.assertEqual(self.view.refusals, [])
        self.assertEqual(self.sent[1]["input"], [typed("When do you open?")])


class TwoSessionsOnOneCoreTests(unittest.IsolatedAsyncioTestCase):
    """Two live sessions' server delegations on one delegation core, as the gateway's one core serves them.

    Each session opens its own conversation, which the core serves with its own IM session on its own provider: their
    work runs side by side, each session's events and history are its own, and deleting one conversation leaves the
    other's work running.
    """

    async def test_two_sessions_on_one_core_keep_their_own_conversation_work_events_and_history(self):
        gates = [asyncio.Event(), asyncio.Event()]
        providers = [Provider([gates[0], *text_round("Done.")]), Provider([gates[1], *text_round("Done.")])]
        made = iter(providers)
        first_side = _delegation(lambda: next(made))
        client = first_side.client
        sides, views, frontends = [first_side, ServerDelegation(client)], [], []
        for number, side in enumerate(sides, 1):
            view, frontend = RecordingView(), RecordingFrontend()
            view.id = f"live_{number}"
            side.listen(frontend)
            await side.open(view)
            self.addAsyncCleanup(side.close)
            views.append(view)
            frontends.append(frontend)
        conversations = [side.conversation for side in sides]
        self.assertEqual(len(set(conversations)), 2)
        self.assertEqual(sorted(client.service.sessions), sorted(conversations))
        asks = [[typed(f"Session {n}.")] for n in (1, 2)]
        first, second = [await side.delegate(ask) for side, ask in zip(sides, asks, strict=True)]
        ims = [client.service.sessions[conversation] for conversation in conversations]
        self.assertIsNot(ims[0], ims[1])
        async with asyncio.timeout(3):  # both Works run at once, each on its own provider
            while not all(provider.payloads for provider in providers):
                await asyncio.sleep(0.005)
        gates[1].set()  # only the second session's backend answers
        async with asyncio.timeout(3):
            while not frontends[1].results:
                await asyncio.sleep(0.005)
        self.assertEqual((frontends[0].results, frontends[1].results), ([], ["Done."]))
        self.assertEqual({handed for handed, _ in frontends[1].events}, {second})
        self.assertEqual({handed for handed, _ in frontends[0].events}, {first})
        self.assertEqual(views[1].events[0], "session.delegation.created")
        self.assertEqual(ims[1].history[0], asks[1][0])  # each conversation holds only its own session's request
        await sides[1].close()  # the second session closes: only its conversation is deleted
        self.assertEqual(list(client.service.sessions), [conversations[0]])
        self.assertEqual((providers[1].closes, providers[0].closes), (1, 0))
        gates[0].set()
        async with asyncio.timeout(3):
            while not frontends[0].results:
                await asyncio.sleep(0.005)
        self.assertEqual(frontends[0].results, ["Done."])


# ------------------------------------------------------------------------- the reader order, on a networked core
def call_item(call_id):
    """A completed call of the client's ``get_menu``."""
    return {"type": "function_call", "call_id": call_id, "name": "get_menu", "arguments": "{}", "status": "completed"}


class LateTerminalCore:
    """A core whose first response's terminal comes late, as over a network; its continuation's events come at once."""

    def __init__(self):
        self.gate = asyncio.Event()  # the first response's terminal is on its way until the test sets it
        self.requests = 0

    async def create_conversation(self, metadata):
        return {"id": "conv_1", "object": "conversation", "created_at": 0, "metadata": metadata}

    async def delete_conversation(self, conversation_id):
        return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

    async def create_response(self, params):
        self.requests += 1
        return self.first() if self.requests == 1 else self.continuation()

    async def aclose(self):
        """The client's own close: it holds nothing to release."""

    async def first(self):
        yield {"type": "response.created", "response": {"id": "resp_1"}}
        yield {"type": "response.output_item.done", "item": call_item("call_1")}
        await self.gate.wait()
        yield {"type": "response.completed", "response": {"id": "resp_1", "output": [call_item("call_1")]}}

    async def continuation(self):
        yield {"type": "response.created", "response": {"id": "resp_2"}}
        yield {"type": "response.output_item.done", "item": call_item("call_2")}
        yield {"type": "response.completed", "response": {"id": "resp_2", "output": [call_item("call_2")]}}


class ReaderOrderTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_continuations_events_never_overtake_the_end_of_the_response_it_continues(self):
        # A delegation has a reader for each of its responses. Over a network the waiting response's terminal may still
        # be on its way when the client's continuation is accepted; the next reader relays nothing before the previous
        # one has ended, so the client and the frontend see each response whole and in order.
        core, view, frontend = LateTerminalCore(), RecordingView(), RecordingFrontend()
        side = ServerDelegation(core)
        side.listen(frontend)
        await side.open(view)
        self.addAsyncCleanup(side.close)
        await side.delegate(HI)
        async with asyncio.timeout(2):  # the call is relayed, so the client can answer it
            while "call_1" not in side.routes.routes:
                await asyncio.sleep(0.005)
        await side.item_create(output("call_1"), None)
        await side.response_create("continue")  # queued: the continuation's stream opens once the sender sends it
        await asyncio.sleep(0.05)  # time enough for the continuation's events to overtake, were they not held
        core.gate.set()
        async with asyncio.timeout(2):
            while len(view.relayed) < 6:
                await asyncio.sleep(0.005)
        each_response = ["response.created", "response.output_item.done", "response.completed"]
        self.assertEqual(view.relayed, each_response * 2)
        self.assertEqual([kind for _, kind in frontend.events], each_response * 2)


# --------------------------------------------------------------------------------- the request's settings
class BackendSettingsTests(unittest.TestCase):
    """What the client set reaches the request (the gateway's ``client_settings``), and the core fills the rest."""

    def task(self, **block):
        config = SessionConfig.model_validate(
            {"model": "live-1", "delegation": {"type": "responses", "responses": {"model": "m", **block}}}
        )
        return settings_for(client_settings(config), BackendTask("fallback", "static", []))

    def test_what_the_client_set_reaches_the_backend_request(self):
        task = self.task(
            reasoning={"effort": "high"},
            service_tier="priority",
            tool_choice={"type": "function", "name": "get_menu"},
            tools=[{"type": "function", "name": "get_menu"}],
        )
        payload = task.payload()
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertEqual(payload["service_tier"], "priority")
        self.assertEqual(payload["tool_choice"], {"type": "function", "name": "get_menu"})

    def test_settings_the_client_left_at_their_defaults_are_not_sent(self):
        payload = self.task().payload()
        for key in ("reasoning", "service_tier", "parallel_tool_calls", "text"):
            self.assertNotIn(key, payload)


if __name__ == "__main__":
    unittest.main()
