# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Public SDK, provider requests and public Live events stay separate from host event attribution."""

import asyncio
import json
from copy import deepcopy
from dataclasses import replace

import httpx
from openai import AsyncOpenAI

from examples.frontend_backend_live.delegation_core.api.local import LocalResponsesClient
from examples.frontend_backend_live.delegation_core.api.service import ResponsesService
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.backend.adapters import ADAPTERS
from examples.frontend_backend_live.delegation_core.backend.llm import LlmSettings, LlmSource
from examples.frontend_backend_live.delegation_core.backend.providers import (
    ChatCompletionsBackend,
    OpenAIResponsesBackend,
)
from examples.frontend_backend_live.delegation_core.im.input import BackendTask
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.shared.delegation_contract import ModelEndpoint
from gateways.live.server_delegation.passthrough import sanitize_response_event
from tests.unit.core_helpers import chunk, tool_fragment
from tests.unit.delegation_core.helpers import (
    CORE_SETTINGS,
    MENU,
    IMCase,
    app_outputs,
    call_round,
    read_all,
    text_round,
)

PRIVATE = "_nva_work"
KINDS = {"chat-completions": ChatCompletionsBackend, "openai-responses": OpenAIResponsesBackend}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sse(events):
    return (
        b"".join(
            b"event: " + event.get("type", "message").encode() + b"\ndata: " + encoded(event) + b"\n\n"
            for event in events
        )
        + b"data: [DONE]\n\n"
    )


async def capture_single_flow(kind):
    """Run real Core and provider code through the public SDK, with all HTTP handled in memory."""
    requests = []
    if kind == "chat-completions":
        rounds = [
            [
                chunk(tool_calls=[tool_fragment(0, "call-menu", "get_menu", "{}")]).model_dump(mode="json"),
                chunk(finish="tool_calls").model_dump(mode="json"),
            ],
            [chunk("Green tea is available.", finish="stop").model_dump(mode="json")],
        ]
    else:
        rounds = [call_round("get_menu", {}, "call-menu"), text_round("Green tea is available.")]

    def receive(request):
        requests.append(json.loads(request.content))
        if len(requests) > len(rounds):
            raise AssertionError("An ordinary two-round request was repeated")
        expected = "/v1/chat/completions" if kind == "chat-completions" else "/v1/responses"
        assert request.url.path == expected
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=sse(rounds[len(requests) - 1])
        )

    client = AsyncOpenAI(
        api_key="local-dummy-not-a-credential",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive)),
    )
    endpoint = ModelEndpoint(
        model="reference/model",
        base_url="https://provider.invalid/v1",
        api_key="local-dummy-not-a-credential",
        api=kind,
        extra_params={
            "temperature": 0,
            "max_tokens": 4096,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        },
    )

    class Source(LlmSource):
        @classmethod
        def from_settings(cls, block, *, endpoints, tools, vocabulary):
            settings = LlmSettings.model_validate(block)
            return cls(
                settings,
                endpoints=endpoints,
                tools=tools,
                make_provider=lambda bound: KINDS[kind](bound, settings.limits, client),
            )

    core = build_core(
        CORE_SETTINGS,
        endpoints=lambda model: endpoint,
        defaults=BackendTask("reference/model", "Look up the exact menu.", [MENU]),
        adapters={**ADAPTERS, "llm": lambda: Source},
    )
    service = ResponsesService(core.session, stateless_session=core.one_shot_session, backend=core)
    local = LocalResponsesClient(service, close_seconds=core.close_seconds, close_core=core.aclose)
    try:
        conversation = (await local.create_conversation({}))["id"]
        assert service.sessions[conversation].backend.max_running == 1
        params = {
            "conversation": conversation,
            "metadata": {"live_delegation_id": "D1"},
            "stream": True,
            "input": [{"id": "caller-msg", "type": "message", "role": "user", "content": "Which teas?"}],
        }
        first = await read_all(await local.create_response(params))
        second = await read_all(await local.create_response({**params, "input": app_outputs(first, '{"tea":"green"}')}))
        history = (await local.list_conversation_items(conversation))["data"]
        events = [*first, *second]
        return {
            "requests": requests,
            "history": history,
            "public": [sanitize_response_event(e) for e in events],
        }, events
    finally:
        await local.aclose()
        await client.close()


def receipt_voice(**options):
    """Build the real frontend receipt handler with an empty context and no running pipeline."""
    from pipecat.processors.aggregators.llm_context import LLMContext

    from examples.frontend_backend_live.voice import Voice
    from tests.unit.frontend_helpers import CONFIG, PROMPTS, Port

    voice = Voice(
        config=CONFIG,
        prompts=PROMPTS.cascade,
        lines=PROMPTS.lines,
        json_rules=PROMPTS.cascade.json_rules,
        empty_turn_note=PROMPTS.cascade.empty_turn_note,
        delegation=Port(),
        item_text=message_text,
        **options,
    )
    voice.context = LLMContext(voice.seed([{"role": "system", "content": "Assist the caller."}]))
    return voice


class PublicBoundaryTests(IMCase):
    async def test_superseded_held_final_closes_its_canonical_source_without_false_failure(self):
        from examples.frontend_backend_live.delegation_core.backend.port import (
            CompleteDecision,
            Start,
            Steer,
            WorkTarget,
        )
        from examples.frontend_backend_live.spoken import LineKind
        from tests.unit.delegation_core.helpers import (
            ScriptedAdapter,
            ScriptedConductor,
            answer,
            ask,
            until,
            wait,
        )

        old_text, new_text = "Old proposal, intentionally superseded.", "Updated proposal from the continuation."
        may_answer, may_decide, deciding, may_finish = (asyncio.Event() for _ in range(4))
        reader_events, frames = [], {"D1": [], "D2": []}

        async def route(given, place):
            source = given.arrivals[0].delegation_id
            if source == "D1":
                await place(CompleteDecision((Start((source,)),)))
            else:
                deciding.set()
                await may_decide.wait()
                await place(CompleteDecision((Steer((source,), (WorkTarget("W1", ("W1.1",)),)),)))

        adapter = ScriptedAdapter(
            [wait(may_answer), answer(old_text)],
            [wait(may_finish), answer(new_text)],
            max_running=3,
            conductor=ScriptedConductor(route),
        )
        voice = receipt_voice(observe=lambda kind, **fields: reader_events.append((kind, fields)))
        self.addAsyncCleanup(voice.close)
        im = self.session(adapter)

        async def receive(source, stream):
            async for event in stream:
                frames[source].append(deepcopy(event))
                voice.reader.on_delegation_event(source, event)

        pumps = []
        try:
            voice.worknote.confirm(voice.worknote.reserve("Find the choices."), "D1")
            first = im.admit(ask("Find the choices."), delegation_id="D1")
            pumps.append(asyncio.create_task(receive("D1", first.stream)))
            await until(lambda: bool(adapter.runs), "the original worker")
            old_work = im.works["W1"]
            voice.worknote.confirm(voice.worknote.reserve("Choose option one."), "D2")
            second = im.admit(ask("Choose option one."), delegation_id="D2")
            pumps.append(asyncio.create_task(receive("D2", second.stream)))
            await asyncio.wait_for(deciding.wait(), 2)
            may_answer.set()
            await until(
                lambda: bool(old_work.held) and im.ledger.tasks["W1.1"].ended,
                "the finished task's actual held final",
            )
            self.assertEqual(im.ledger.tasks["W1.1"].answer_publication, "pending")
            may_decide.set()
            await until(
                lambda: (
                    len(adapter.runs) == 2
                    and pumps[0].done()
                    and (view := voice.reader.ownership.view("D2")) is not None
                    and view.pending
                ),
                "accepted continuation, old terminal, and observed successor responsibility",
            )
            self.assertEqual(im.ledger.tasks["W1.1"].successor_task_id, "W2.1")
            self.assertEqual(im.ledger.tasks["W1.1"].answer_publication, "superseded")
            self.assertEqual(im.ledger.delegations["D1"].state, "answered")
            terminal = frames["D1"][-1]
            self.assertEqual(terminal["type"], "response.completed")
            proof = terminal[PRIVATE]
            self.assertTrue(proof["obligations"])
            self.assertTrue(
                all(
                    row["state"] == "settled"
                    and row["terminal_outcome"] == "answered"
                    and not (row["delivery_pending"] or row["publication_pending"])
                    for row in proof["obligations"]
                )
            )
            self.assertEqual(len({row["branch_ref"] for row in proof["obligations"]}), 1)
            self.assertNotIn("D1", voice.reader.open)
            self.assertFalse(voice.reader.ownership.is_composite("D1"))
            self.assertEqual(
                [fields for kind, fields in reader_events if kind == "delegation.ended"],
                [{"delegation_id": "D1", "how": "answered"}],
            )
            self.assertFalse(any(kind == "delegation.taker" for kind, _ in reader_events))
            self.assertIn("D2", voice.reader.open)
            self.assertTrue(voice.reader.ownership.view("D2").pending)
            self.assertFalse(pumps[1].done(), "the successor still owns its independent computation")
            self.assertFalse(any(getattr(item, "answers", False) for item in voice.owed))
            self.assertFalse(any(getattr(item, "kind", None) == LineKind.NOTICE for item in voice.owed))
            self.assertTrue(self.traced("im.answer.superseded"))
            self.assertFalse(self.traced("im.decision.rejected"))

            may_finish.set()
            await asyncio.wait_for(asyncio.gather(*pumps), 2)
            self.assertTrue(all(PRIVATE in event for events in frames.values() for event in events))
            [result] = [item.said for item in voice.owed if getattr(item, "answers", False)]
            self.assertEqual((result.delegation_id, result.speak), ("D2", new_text))
            published = next(
                event
                for event in frames["D2"]
                if event["type"] == "response.output_item.done" and event["item"].get("phase") == "final_answer"
            )
            self.assertEqual(result.item_id, published["item"]["id"])
            self.assertEqual(result.attribution.branch_ref, published[PRIVATE]["item_attribution"]["branch_ref"])
            self.assertEqual(voice.worknote.entries["D2"].ready, 1, "a Core terminal is not caller hearing")
            self.assertFalse(voice.spoken.presentations)
            self.assertFalse(any(getattr(item, "kind", None) == LineKind.NOTICE for item in voice.owed))
        finally:
            may_answer.set()
            may_decide.set()
            may_finish.set()
            for pump in pumps:
                if not pump.done():
                    pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)

    async def settlement_replay(self, *, forwarded):
        from examples.shared.work_provenance import Attribution, Binding, OwnerSettlement
        from gateways.live.server_delegation import ServerDelegation
        from tests.unit.frontend_backend_live.test_delegations import (
            Reading,
            failed,
            mcp_call,
            mcp_started,
        )
        from tests.unit.frontend_backend_live.test_work_ownership import edge, response_opened, snapshot, tagged

        reading = Reading("D1", "D2")
        self.addCleanup(reading.reader.close)
        gateway = ServerDelegation(object())
        gateway.retain_attachments()
        gateway.listen(reading.reader)
        original = snapshot("D1", "r1", bindings=[Binding("A", "D1", 0)], obligations=[edge("A", owner="D1")])
        attributed = replace(original, item_attribution=Attribution("A", "D1", 0))
        for event in (
            *response_opened(original),
            tagged(mcp_started("fixture", "lookup", "old-call"), attributed),
        ):
            gateway.mailbox.record("D1", event).forwarded = True
        failure = replace(edge("A", owner="D1", state="settled"), terminal_outcome="failed", stop_outcome="unconfirmed")
        newer = snapshot(
            "D2",
            "r2",
            bindings=[Binding("B", "D2", 0)],
            obligations=[edge("B", owner="D2")],
            settlements=(OwnerSettlement("D1", 2, "failed", (failure,)),),
        )
        for event in response_opened(newer):
            gateway.mailbox.record("D2", event).forwarded = forwarded
        gateway.mailbox.record(
            "D1", tagged(mcp_call("fixture", "lookup", "old-call", output="late backend fact"), attributed)
        )
        gateway.mailbox.record("D1", tagged(failed("r1"), replace(original, owner_revision=2, obligations=(failure,))))
        before = deepcopy([entry.event for entry in gateway.mailbox.entries])
        await gateway.resume_attachment()
        self.assertEqual([entry.event for entry in gateway.mailbox.entries], before)
        self.assertIn("D1", reading.reader.ended)
        self.assertIn("D2", reading.reader.open)
        self.assertEqual(sum(kind == "delegation.backend_step" for kind, _ in reading.events), 1)
        self.assertFalse(any(kind == "delegation.provenance_refused" for kind, _ in reading.events))
        return reading

    async def test_settlement_reports_its_notice_only_if_it_was_never_forwarded(self):
        for forwarded in (True, False):
            with self.subTest(forwarded=forwarded):
                reading = await self.settlement_replay(forwarded=forwarded)
                self.assertEqual(len(reading.notices()), 0 if forwarded else 1)

    async def test_ordinary_single_result_writer_keeps_the_result_separate_from_instructions(self):
        from tests.unit.frontend_helpers import PROMPTS

        _, events = await capture_single_flow("chat-completions")
        voice = receipt_voice(clock=lambda: 1.0)
        self.addAsyncCleanup(voice.close)
        voice.worknote.confirm(voice.worknote.reserve("Which teas?"), "D1")
        for event in events:
            voice.reader.on_delegation_event("D1", event)
        [current] = voice._take()
        self.assertEqual(current.said.speak, "Green tea is available.")
        content, note = voice._update([current])
        request = voice.layout.relay("\n\n".join(voice.instructions), content, note)
        messages = request.messages(voice.conversation())
        self.assertNotIn(PRIVATE, json.dumps(messages))
        self.assertNotIn("additional_targets", json.dumps(messages))
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        prompts = PROMPTS.cascade
        self.assertIn("Assist the caller.", messages[0]["content"])
        result = "Green tea is available."
        self.assertNotIn(result, messages[0]["content"])
        self.assertNotIn("Assist the caller.", messages[-1]["content"])
        update = prompts.update_label + result
        before, after = prompts.writer_notes.newer.strip("\n").split("{requests}")
        payload = messages[-1]["content"].removeprefix(before)
        requests, end = json.JSONDecoder().raw_decode(payload)
        self.assertEqual(current.said.delegation_id, "D1")
        self.assertEqual(
            requests,
            [
                {
                    "branch_ref": current.said.attribution.branch_ref,
                    "delegation_id": "D1",
                    "relationship": "same_branch",
                    "caller_requests": ["Which teas?"],
                }
            ],
        )
        self.assertEqual(payload[end:], after + "\n\n" + update)
        self.assertFalse(request.calls)

    def test_replay_preingestion_does_not_disguise_a_fresh_canonical_request_as_a_duplicate_frame(self):
        from examples.shared.work_provenance import Attribution, Binding
        from tests.unit.frontend_backend_live.test_delegations import Reading, completed, created, in_progress, message
        from tests.unit.frontend_backend_live.test_work_ownership import edge, response_opened, snapshot, tagged

        reading = Reading("transport")
        original = snapshot(
            "old-core",
            "old-response",
            bindings=[Binding("A", "old-core", 0)],
            obligations=[edge("A", owner="old-core")],
        )
        reading.hand("transport", *response_opened(original))
        reading.hand(
            "transport",
            tagged(
                message("Old result.", item_id="old-item"),
                replace(original, item_attribution=Attribution("A", "old-core", 0)),
            ),
            tagged(
                completed("old-response"),
                replace(original, owner_revision=2, obligations=(edge("A", owner="old-core", state="settled"),)),
            ),
        )
        self.assertIn("transport", reading.reader.ended)
        fresh = snapshot(
            "fresh-core",
            "fresh-response",
            bindings=[Binding("B", "fresh-core", 0)],
            obligations=[edge("B", owner="fresh-core")],
        )
        event = tagged(created("fresh-response"), fresh)
        reading.reader.restore_work_provenance("transport", deepcopy(event))
        reading.hand("transport", event, tagged(in_progress("fresh-response"), fresh))
        reading.hand(
            "transport",
            tagged(
                message("Fresh result.", item_id="fresh-item"),
                replace(fresh, item_attribution=Attribution("B", "fresh-core", 0)),
            ),
        )
        self.assertEqual(reading.calls("ready"), [("transport", "Old result."), ("transport", "Fresh result.")])

    async def test_default_single_flow_preserves_settings_history_and_public_protocol_with_current_work_tools(self):
        for kind in KINDS:
            with self.subTest(provider=kind):
                captured, events = await capture_single_flow(kind)
                self.assertEqual(len(captured["requests"]), 2)
                self.assertTrue(any(PRIVATE in event for event in events))
                self.assertNotIn(PRIVATE, json.dumps(captured))
                self.assertNotIn("additional_targets", json.dumps(captured["requests"]))
                for request in captured["requests"]:
                    self.assertEqual(
                        (request["model"], request["temperature"], request["stream"]), ("reference/model", 0, True)
                    )
                    self.assertEqual(request["chat_template_kwargs"], {"enable_thinking": True})
                    maximum = "max_tokens" if kind == "chat-completions" else "max_output_tokens"
                    self.assertEqual(request[maximum], 4096)
                    tools = [tool.get("function", tool) for tool in request["tools"]]
                    self.assertEqual(
                        [tool["name"] for tool in tools],
                        ["get_menu", "cancel_work", "set_work_state", "caller_message", "resolve_questions"],
                    )
                    self.assertEqual(tools[0]["parameters"], MENU["parameters"])
                    self.assertEqual(set(tools[2]["parameters"]["required"]), {"state"})
                    self.assertEqual(set(tools[3]["parameters"]["required"]), {"kind", "message"})
                    self.assertEqual(set(tools[4]["parameters"]["required"]), {"resolutions"})
                    if kind == "openai-responses":
                        self.assertFalse(request["store"])
                history = captured["history"]
                self.assertEqual(
                    [item["type"] for item in history], ["message", "function_call", "function_call_output", "message"]
                )
                self.assertEqual((history[0]["role"], history[0]["content"]), ("user", "Which teas?"))
                self.assertEqual((history[1]["name"], history[1]["arguments"]), ("get_menu", "{}"))
                self.assertEqual(history[2]["call_id"], history[1]["call_id"])
                self.assertEqual(history[2]["output"], '{"tea":"green"}')
                self.assertEqual(message_text(history[-1]), "Green tea is available.")
                public = captured["public"]
                terminals = [
                    event
                    for event in public
                    if event["type"] in ("response.completed", "response.failed", "response.incomplete")
                ]
                self.assertEqual([event["type"] for event in terminals], ["response.completed"] * 2)
                # Live public lifecycle snapshots intentionally redact payloads; complete items remain on their
                # own public events. The underlying Core SDK snapshots still carry the actual response output.
                for terminal in terminals:
                    self.assertEqual(terminal["response"]["output"], [])
                    self.assertIsNone(terminal["response"]["instructions"])
                    self.assertEqual(terminal["response"]["tools"], [])
                done = [event["item"] for event in public if event["type"] == "response.output_item.done"]
                self.assertEqual([item["type"] for item in done], ["function_call", "message"])
                self.assertEqual((done[0]["name"], done[0]["call_id"]), ("get_menu", history[1]["call_id"]))
                self.assertEqual(message_text(done[1]), "Green tea is available.")
                raw_terminals = [event for event in events if event["type"] == "response.completed"]
                self.assertEqual([item["type"] for item in raw_terminals[0]["response"]["output"]], ["function_call"])
                self.assertEqual(
                    [message_text(item) for item in raw_terminals[1]["response"]["output"]], ["Green tea is available."]
                )

    async def test_public_openai_client_preserves_private_core_envelopes_without_changing_public_sse(self):
        _, events = await capture_single_flow("chat-completions")

        def receive(request):
            self.assertEqual(request.url.path, "/v1/responses")
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(events))

        async with AsyncOpenAI(
            api_key="local-dummy-not-a-credential",
            base_url="https://core.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(receive)),
        ) as client:
            stream = await client.responses.create(model="reference/model", input="Which teas?", stream=True)
            parsed = [event.model_dump(mode="json", exclude_none=True) async for event in stream]
            await stream.close()
        self.assertEqual([row["type"] for row in parsed], [row["type"] for row in events])
        for raw, decoded in zip(events, parsed, strict=True):
            self.assertEqual(decoded[PRIVATE], raw[PRIVATE])
            bare = deepcopy(raw)
            bare.pop(PRIVATE)
            self.assertEqual(encoded(sanitize_response_event(raw)), encoded(sanitize_response_event(bare)))
            self.assertNotIn(PRIVATE, sanitize_response_event(decoded))
