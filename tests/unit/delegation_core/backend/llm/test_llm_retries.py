# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Tests for retries and timeouts: stalls are caught by silence, and nothing runs twice."""

import unittest
from unittest.mock import patch

import httpx
from openai import APIConnectionError, BadRequestError, RateLimitError

from delegation_core.backend.llm import LlmSettings
from delegation_core.backend.providers import BrokenStreamError, ChatCompletionsBackend, ProviderLimits
from delegation_core.config import ModelEndpoint
from tests.unit.contracts.backend import a_round, check_events, message_events, text_round
from tests.unit.contracts.base import collect
from tests.unit.delegation_core.backend.llm.doubles import (
    FakeChat,
    IMTask,
    ScriptedProvider,
    Sleep,
    adapter,
    chunk,
    kinds,
)

REQUEST = httpx.Request("POST", "http://localhost:1/v1/chat/completions")
QUICK = ProviderLimits(provider_retries=1, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2)
# The frontend-backend-live example's thinker slot thinks by default (examples_registry.yaml: enable_thinking: true).
THINKING = ModelEndpoint(
    "test/model",
    "http://localhost:1/v1",
    extra_params={"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
)
CHAT_ROUND = {"model": "test/model", "instructions": "Be brief.", "input": []}


def thinking_chat(*steps):
    """The chat provider on a thinking endpoint, its client streaming ``steps`` (chunks and pauses)."""
    return ChatCompletionsBackend(THINKING, QUICK, FakeChat(list(steps)))


def bad_request():
    return BadRequestError("bad request", response=httpx.Response(400, request=REQUEST), body=None)


def too_many():
    return RateLimitError("slow down", response=httpx.Response(429, request=REQUEST), body=None)


def dropped():
    return APIConnectionError(request=REQUEST)


def stalled(text="Done."):
    round_ = text_round(text)
    return [round_[0], Sleep(1.0), *round_[1:]]


def failed_round():
    created = {"type": "response.created", "response": {"id": "resp_f", "status": "in_progress"}}
    error = {"code": "server_error", "message": "The model fell over."}
    return [created, {"type": "response.failed", "response": {"id": "resp_f", "status": "failed", "error": error}}]


def cut_round():
    events = a_round(message_events("Half an ans"))
    events[-1] = {
        "type": "response.incomplete",
        "response": {"id": "resp_1", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    }
    return events


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_400_is_not_retried(self):
        provider = ScriptedProvider([bad_request()], text_round("Done."), limits=QUICK)
        with self.assertRaises(BadRequestError):
            await collect(provider.stream({}))
        self.assertEqual(provider.opened, 1)

    async def test_a_dropped_connection_is_tried_again_when_opening(self):
        provider = ScriptedProvider([dropped()], text_round("Done."), limits=QUICK)
        with patch("delegation_core.backend.providers.retry.retry_delay", return_value=0):
            events = await collect(provider.stream({}))
        self.assertEqual((provider.opened, events[-1]["type"]), (2, "response.completed"))

    async def test_the_retries_cover_only_connecting_not_a_slow_first_event(self):
        provider = ScriptedProvider([Sleep(1.0), *text_round("Late.")], text_round("Done."), limits=QUICK)
        with self.assertRaises(TimeoutError):
            await collect(provider.stream({}))
        self.assertEqual(provider.opened, 1)

    async def test_a_silence_past_the_gap_limit_ends_the_round(self):
        provider = ScriptedProvider(stalled(), limits=QUICK)
        seen = []
        with self.assertRaises(TimeoutError):
            async for event in provider.stream({}):
                seen.append(event["type"])
        self.assertEqual(seen, ["response.created"])

    async def test_a_slow_but_steady_round_is_not_cut(self):
        steady = []
        for event in text_round("Done."):
            steady += [Sleep(0.1), event]
        events = await collect(ScriptedProvider(steady, limits=QUICK).stream({}))  # 0.4 s in all, no 0.2 s of silence
        self.assertEqual(events[-1]["type"], "response.completed")

    async def test_a_thinking_round_that_streams_steadily_is_not_cut(self):
        # Each chunk counts as life, so 0.4 s without an event but never 0.2 s without a chunk is not a stall.
        steady = []
        for piece in ("<think>Let me", " think.</think>", "A latte", " is $4.50."):
            steady += [Sleep(0.1), chunk(piece)]
        events = await collect(thinking_chat(*steady, chunk(finish="stop")).stream(CHAT_ROUND))
        self.assertEqual(events[-1]["type"], "response.completed")
        [message] = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        self.assertEqual(message["content"][0]["text"], "A latte is $4.50.")

    async def test_a_thinking_round_that_stalls_is_still_cut(self):
        stalled_chat = thinking_chat(
            chunk("<think>Let me"), Sleep(1.0), chunk(" think.</think>Done."), chunk(finish="stop")
        )
        with self.assertRaises(TimeoutError):
            await collect(stalled_chat.stream(CHAT_ROUND))

    async def test_reading_stops_at_the_first_terminal(self):
        provider = ScriptedProvider([*text_round("Done."), RuntimeError("after the terminal")], limits=QUICK)
        events = await collect(provider.stream({}))
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertFalse(provider.read_after_terminal)

    async def test_a_stream_with_no_event_is_broken(self):
        with self.assertRaises(BrokenStreamError):
            await collect(ScriptedProvider([], limits=ProviderLimits(provider_retries=0)).stream({}))


class RunRetryTests(unittest.IsolatedAsyncioTestCase):
    def settings(self, **changes):
        return LlmSettings(provider_retries=0, first_event_timeout_seconds=0.2, idle_timeout_seconds=0.2, **changes)

    async def run_task(self, *rounds):
        provider = ScriptedProvider(*rounds, limits=self.settings().limits)
        task, events, error = IMTask(), [], None
        try:
            async for event in adapter(provider, settings=self.settings()).start(task).events():
                task.see(event)
                events.append(event)
        except Exception as exc:
            error = exc
        return provider, events, error

    async def test_a_400_is_not_retried_and_reaches_the_im_unchanged(self):
        provider, events, error = await self.run_task([bad_request()], text_round("Done."))
        self.assertIsInstance(error, BadRequestError)
        self.assertEqual((provider.opened, events), (1, []))

    async def test_a_stall_past_the_gap_limit_is_retried_once(self):
        provider, events, error = await self.run_task(stalled("Old."), text_round("Done.", "resp_2"))
        self.assertIsNone(error)
        self.assertEqual(provider.opened, 2)
        self.assertEqual(kinds(events)[:3], ["response.created", "response.failed", "response.created"])
        self.assertEqual(check_events(events), [])

    async def test_a_second_stall_fails_the_task(self):
        provider, events, error = await self.run_task(stalled(), stalled(), text_round("Never."))
        self.assertIsInstance(error, TimeoutError)
        self.assertEqual(provider.opened, 2)

    async def test_a_failed_round_or_a_broken_stream_is_retried_once(self):
        for first in (failed_round(), text_round("x")[:-1], [*text_round("x")[:2], *text_round("y")]):
            with self.subTest(first=kinds(first)[-1]):
                provider, events, error = await self.run_task(first, text_round("Done.", "resp_2"))
                self.assertIsNone(error)
                self.assertEqual(provider.opened, 2)
                self.assertEqual(kinds(events)[1], "response.failed")
                self.assertEqual(check_events(events), [])

    async def test_a_round_cut_short_is_passed_on_never_retried_and_ends_the_task(self):
        provider, events, error = await self.run_task(cut_round(), text_round("Done."))
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("max_output_tokens", str(error))
        self.assertEqual(provider.opened, 1)
        self.assertEqual(kinds(events), ["response.created", "response.incomplete"])
        self.assertEqual(events[-1]["response"]["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(events[-1]["response"]["output"], [])  # the cut message is not delivered

    async def test_a_tail_error_does_not_rerun_the_round(self):
        provider, events, error = await self.run_task(
            [*text_round("Done."), RuntimeError("tail")], text_round("Again.")
        )
        self.assertIsNone(error)
        self.assertEqual((provider.opened, kinds(events).count("response.created")), (1, 1))

    async def test_a_rate_limit_after_its_retry_reaches_the_im_as_a_429(self):
        provider, events, error = await self.run_task([too_many()], [too_many()], text_round("Done."))
        self.assertIsInstance(error, RateLimitError)
        self.assertEqual((error.status_code, provider.opened), (429, 2))

    async def test_a_dropped_connection_mid_round_is_retried_once(self):
        round_ = text_round("Old.")
        provider, events, error = await self.run_task([round_[0], dropped()], text_round("Done.", "resp_2"))
        self.assertIsNone(error)
        self.assertEqual(provider.opened, 2)
        self.assertEqual(events[1]["response"]["error"]["code"], "server_error")


if __name__ == "__main__":
    unittest.main()
