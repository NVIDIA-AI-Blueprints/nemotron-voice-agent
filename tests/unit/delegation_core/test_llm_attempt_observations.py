# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101,D102,D103,D107

"""Logical worker attempts are observable independently of public response terminals."""

import asyncio
from copy import deepcopy
from unittest import IsolatedAsyncioTestCase

from examples.shared.delegation_contract import ModelEndpoint
from tests.unit.delegation_core.helpers import (
    ENDPOINT,
    FAST,
    FakeTaskContext,
    IMCase,
    ScriptedProvider,
    adapter,
    ask,
    drive,
    gated,
    read_all,
    text_round,
)
from tests.unit.delegation_core.test_agent import SETTINGS
from tests.unit.delegation_core.test_agent import source as agent_source

START = "backend.provider_attempt.started"
END = "backend.provider_attempt.ended"


def observations(task, kind):
    return [fields for event, fields in task.traces if event == kind]


async def turns():
    for _ in range(30):
        await asyncio.sleep(0)


class LogicalAttemptIMTests(IMCase):
    async def measure_steer(self, after_created):
        gate = asyncio.Event()
        old = text_round("Never publish the old answer.", response_id="old_attempt")
        blocked = gated(gate, old) if after_created else [gate, *old]
        workers = ScriptedProvider(blocked, text_round("The updated answer.", response_id="new_attempt"))
        backend = agent_source(
            ScriptedProvider(), workers, settings=SETTINGS.model_copy(update={"max_running": 1})
        ).new()
        im = self.session(backend)
        first = im.admit(ask("Find my options."), delegation_id="D1")
        await self.until(lambda: len(workers.payloads) == 1, "old logical stream entered")
        if after_created:
            await self.until(lambda: im.works["W1"].tasks[0]._round_identity is not None, "created consumed")
        else:
            self.assertIsNone(im.works["W1"].tasks[0]._round_identity)
        await self.request(im, "Use the updated requirements.", "D2")
        await read_all(first.stream)
        await self.turns()
        self.assertEqual(len(workers.payloads), 2)
        starts, ends = self.traced(START), self.traced(END)
        self.assertEqual(len(starts), 2, "Every real logical attempt needs an opening observation")
        self.assertEqual(len(ends), 2, "Both attempts need reader-end observations, even before created")
        self.assertEqual(len({row["observation_id"] for row in starts}), 2)
        self.assertEqual({row["observation_id"] for row in starts}, {row["observation_id"] for row in ends})
        indexed = {row["observation_id"]: row for row in ends}
        old_end = indexed[starts[0]["observation_id"]]
        self.assertEqual(starts[0]["delegation_id"], "D1")
        self.assertEqual(old_end["cut_by"], "steer")
        self.assertIsNone(old_end["usage"])
        self.assertTrue(old_end["usage_missing"])
        self.assertIsNone(old_end["cost"])
        self.assertEqual(old_end["reader_entered"], True)
        self.assertEqual(old_end["created_observed"], after_created)
        self.assertEqual(old_end["response_id"], "old_attempt" if after_created else None)
        self.assertEqual(old_end["carrier_scope"], "event_time")
        self.assertTrue(all(row["conversation_ref"] == im.conversation_ref for row in starts + ends))
        self.assertTrue(all(row["worker_implementation"] == "llm" for row in ends))
        self.assertEqual([row["round"] for row in starts], [1, 1])
        self.assertEqual([row["attempt"] for row in starts], [1, 1])
        return self.traced("backend.round.usage")

    async def test_steering_observes_both_attempts_at_either_response_boundary(self):
        for after_created in (False, True):
            with self.subTest(after_created=after_created):
                self.traces.clear()
                mirrors = await self.measure_steer(after_created)
                terminals = ["response.incomplete"] if after_created else []
                self.assertEqual([row["terminal"] for row in mirrors], [*terminals, "response.completed"])
                self.assertTrue(all(row["backend"] == "llm-agent" for row in mirrors))


class LogicalAttemptRunTests(IsolatedAsyncioTestCase):
    async def test_stop_after_provider_entry_preserves_pre_and_post_created_boundaries(self):
        for closing in (False, True):
            for after_created in (False, True):
                with self.subTest(closing=closing, after_created=after_created):
                    await self.check_cut(closing, after_created)

    async def check_cut(self, closing, after_created):
        gate = asyncio.Event()
        events = text_round("Never publish.")
        provider = ScriptedProvider(gated(gate, events) if after_created else [gate, *events])
        task = FakeTaskContext()
        run = adapter(provider).start(task)
        stream = run.events()
        reading = None
        if after_created:
            self.assertEqual((await anext(stream))["type"], "response.created")
        else:
            reading = asyncio.create_task(anext(stream, None))
            while not provider.payloads:
                await asyncio.sleep(0)
        if closing and reading is not None:
            reading.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await reading
        if not closing:
            self.assertEqual(await run.stop(), "stopped")
            if reading is not None:
                self.assertIsNone(await reading)
        await stream.aclose()
        await run.stop()
        await turns()
        [ended] = observations(task, END)
        self.assertTrue(ended["reader_entered"])
        self.assertEqual(ended["created_observed"], after_created)
        self.assertEqual(ended["cut_by"], "close" if closing else "stop")
        self.assertIsNone(ended["usage"])
        self.assertIsNone(ended["provider_terminal"])

    async def test_synchronous_stream_construction_error_closes_the_observation(self):
        class OpeningError(ScriptedProvider):
            def stream(self, payload, **kwargs):
                raise ValueError("stream construction failed")

        task = FakeTaskContext()
        with self.assertRaises(ValueError):
            await drive(adapter(OpeningError()).start(task), task)
        self.assertEqual(len(observations(task, START)), 1)
        [ended] = observations(task, END)
        self.assertFalse(ended["reader_entered"])
        self.assertEqual(ended["error_type"], "ValueError")
        self.assertIsNone(ended["response_id"])

    async def test_cancel_before_reader_first_step_is_observed_once_without_provider_entry(self):
        provider, task = ScriptedProvider(text_round("Never.")), FakeTaskContext()
        run = adapter(provider).start(task)
        run._start_attempt(0)
        self.assertEqual(await run.stop(), "stopped")
        await turns()
        self.assertEqual(provider.payloads, [])
        self.assertEqual(len(observations(task, START)), 1)
        [ended] = observations(task, END)
        self.assertFalse(ended["reader_entered"])
        self.assertFalse(ended["created_observed"])
        self.assertEqual(ended["cut_by"], "stop")
        self.assertIsNone(ended["usage"])
        self.assertIsNone(ended["response_id"])
        await run.stop()
        await turns()
        self.assertEqual(len(observations(task, END)), 1)

    async def test_terminal_queued_before_consumer_drain_keeps_reported_evidence(self):
        events = text_round("Done.", response_id="actually_observed")
        events[-1]["response"].update(model="reported/alias", usage={"output_tokens": 0})
        provider, task = ScriptedProvider(events), FakeTaskContext()
        run = adapter(provider).start(task)
        attempt = run._start_attempt(0)
        await attempt.reader
        await turns()
        self.assertIsNone(attempt.created)
        self.assertIsNone(attempt.terminal)
        [ended] = observations(task, END)
        self.assertEqual(
            (ended["provider_terminal"], ended["response_id"], ended["response_model"]),
            ("response.completed", "actually_observed", "reported/alias"),
        )
        self.assertEqual(ended["reported_model"], "reported/alias")
        self.assertEqual(ended["usage"], {"output_tokens": 0})
        self.assertFalse(ended["usage_missing"])
        before = deepcopy(ended)
        run.steer()
        await run.stop()
        attempt._observed_end(attempt.reader)
        await turns()
        self.assertEqual(observations(task, END), [before])

    async def test_end_waits_for_the_actual_reader_generator_to_close(self):
        closing, release = asyncio.Event(), asyncio.Event()
        closed = []

        class HeldClose(ScriptedProvider):
            async def stream(self, payload, **kwargs):
                try:
                    yield text_round("Never.")[0]
                    await asyncio.Event().wait()
                finally:
                    closing.set()
                    await release.wait()
                    closed.append(True)

        provider, task = HeldClose(), FakeTaskContext()
        backend = adapter(provider)
        run = backend.start(task)
        stream = run.events()
        self.assertEqual((await anext(stream))["type"], "response.created")
        attempt = run._attempt
        attempt.cut("stop")
        await closing.wait()
        self.assertEqual(observations(task, END), [])
        release.set()
        await asyncio.gather(attempt.reader, return_exceptions=True)
        await turns()
        self.assertEqual(closed, [True])
        self.assertEqual(len(observations(task, END)), 1)
        await stream.aclose()
        await backend.close()
        self.assertEqual(provider.closed, 1)

    async def test_chat_translator_model_is_not_labeled_provider_reported(self):
        from examples.frontend_backend_live.delegation_core.backend.providers import ChatCompletionsBackend
        from tests.unit.core_helpers import FakeChat, chunk

        reported = chunk("Done.")
        reported.model = "actual/chunk-alias-not-exposed-by-the-translator"
        provider = ChatCompletionsBackend(
            ModelEndpoint("configured/model", "http://unit.invalid"),
            FAST.limits,
            FakeChat([reported, chunk(finish="stop")]),
        )
        task = FakeTaskContext()
        await drive(adapter(provider).start(task), task)
        await turns()
        [ended] = observations(task, END)
        self.assertEqual(ended["requested_model"], "test/model")
        self.assertEqual(ended["configured_model"], "configured/model")
        self.assertEqual(ended["response_model"], "test/model")
        self.assertEqual(ended["response_model_source"], "translated_request")
        self.assertIsNone(ended["reported_model"])
        self.assertEqual(ended["response_id_source"], "translated_response")

    async def test_provider_responses_model_and_partial_usage_are_preserved(self):
        from examples.frontend_backend_live.delegation_core.backend.providers import OpenAIResponsesBackend
        from tests.unit.delegation_core.test_providers import FakeResponses

        events = text_round("Done.", response_id="actual_response_alias")
        events[-1]["response"].update(model="provider/alias", usage={"input_tokens": 0})
        provider = OpenAIResponsesBackend(ENDPOINT, FAST.limits, FakeResponses(events))
        task = FakeTaskContext()
        await drive(adapter(provider).start(task), task)
        await turns()
        [ended] = observations(task, END)
        self.assertEqual(ended["reported_model"], "provider/alias")
        self.assertEqual(ended["response_model_source"], "provider_event")
        self.assertEqual(ended["usage"], {"input_tokens": 0})
        self.assertNotIn("output_tokens", ended["usage"])

    async def test_callback_errors_do_not_change_events_or_escape_into_the_loop(self):
        class BrokenObserver(FakeTaskContext):
            def trace(self, kind, **fields):
                if kind in (START, END):
                    raise asyncio.CancelledError("observer is not caller cancellation")
                return super().trace(kind, **fields)

        task = BrokenObserver()
        run = adapter(ScriptedProvider(text_round("Done."))).start(task)
        errors = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        try:
            events = await drive(run, task)
            await turns()
        finally:
            loop.set_exception_handler(previous)
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(errors, [])
