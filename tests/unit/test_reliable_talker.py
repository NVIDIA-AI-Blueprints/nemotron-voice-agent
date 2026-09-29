# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Tests for bounded Frontend/Backend Talker liveness recovery."""

from __future__ import annotations

import json
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace

from pipecat.processors.aggregators import async_tool_messages
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.frontend_backend_agent.src.reliable_talker import (
    CACHED_RESPONSE_CORRECTION,
    EMPTY_RESPONSE_CORRECTION,
    EMPTY_RESPONSE_FALLBACK,
    INTERNAL_MECHANICS_CORRECTION,
    INTERNAL_MECHANICS_FALLBACK,
    REPEAT_SUBJECT_CORRECTION,
    TOOL_RESULT_CORRECTION,
    WEATHER_GROUNDING_CORRECTION,
    ReliableNvidiaLLMService,
    _weather_grounding_missing,
)


def _chunk(*, content: str | None = None, tool_calls: list | None = None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def _tool_chunk(query: str):
    function = SimpleNamespace(name="call_backend", arguments=json.dumps({"query": query}))
    return _chunk(tool_calls=[SimpleNamespace(index=0, function=function)])


async def _stream(chunks: list) -> AsyncIterator:
    for chunk in chunks:
        yield chunk


class _ScriptedTalker(ReliableNvidiaLLMService):
    def __init__(self, responses: list[list]) -> None:
        self._responses = list(responses)
        self.contexts: list[LLMContext] = []
        self.fallbacks: list[str] = []

    async def _start_completion_stream(self, context: LLMContext) -> AsyncIterator:
        self.contexts.append(context)
        return _stream(self._responses.pop(0))

    async def _push_llm_text(self, text: str) -> None:
        self.fallbacks.append(text)


async def _collect(talker: _ScriptedTalker, context: LLMContext) -> list:
    stream = await talker.get_chat_completions(context)
    return [chunk async for chunk in stream]


class ReliableTalkerTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _finished_result_context(
        *,
        response_text: str = "The current check is temporarily unavailable.",
        newer_user_text: str | None = None,
    ) -> LLMContext:
        result = {
            "type": "tool_result",
            "tool": "get_weather",
            "status": "unavailable",
            "response_text": response_text,
        }
        messages = [
            {"role": "user", "content": "Check Pune weather."},
            async_tool_messages.build_final_result_message("call-weather", json.dumps(result)),
        ]
        if newer_user_text is not None:
            messages.append({"role": "user", "content": newer_user_text})
        return LLMContext(messages, tools=[], tool_choice="auto")

    @staticmethod
    def _successful_weather_context() -> LLMContext:
        result = {
            "type": "tool_result",
            "tool": "get_weather",
            "status": "success",
            "response_text": "In Pune, it is 29 degrees C with clear skies and 48 percent humidity.",
            "data": {
                "city": "Pune",
                "temperature": 29,
                "temperature_unit": "C",
                "condition": "Clear",
                "humidity_percent": 48,
                "wind_kph": 9.5,
            },
        }
        messages = [
            {"role": "user", "content": "Check Pune weather."},
            async_tool_messages.build_final_result_message("call-weather", json.dumps(result)),
        ]
        return LLMContext(messages, tools=[], tool_choice="auto")

    async def test_visible_response_does_not_retry(self) -> None:
        talker = _ScriptedTalker([[_chunk(content="Hello there.")]])

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Hello"}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_native_tool_call_does_not_retry(self) -> None:
        native_call = SimpleNamespace(index=0)
        talker = _ScriptedTalker([[_chunk(tool_calls=[native_call])]])

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Weather now"}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_internal_mechanics_are_blocked_before_direct_speech(self) -> None:
        talker = _ScriptedTalker(
            [
                [_chunk(content="I choose call_backend using my delegate contract.")],
                [_chunk(content="The backend Thinker then makes a tool call.")],
            ]
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "How do you analyze queries?"}]))

        self.assertEqual(chunks, [])
        self.assertEqual(talker.contexts[1].get_messages()[-1]["content"], INTERNAL_MECHANICS_CORRECTION)
        self.assertEqual(talker.fallbacks, [INTERNAL_MECHANICS_FALLBACK])

    async def test_empty_response_retries_once_with_ephemeral_correction(self) -> None:
        original_messages = [{"role": "user", "content": "Repeat that weather."}]
        context = LLMContext(original_messages.copy(), tools=[], tool_choice="auto")
        talker = _ScriptedTalker([[_chunk(content="  ")], [_chunk(content="It is sunny.")]])

        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].choices[0].delta.content, "It is sunny.")
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(context.get_messages(), original_messages)
        self.assertEqual(talker.contexts[1].get_messages()[-1]["role"], "system")
        self.assertEqual(talker.contexts[1].get_messages()[-1]["content"], EMPTY_RESPONSE_CORRECTION)
        self.assertIs(talker.contexts[1].tools, context.tools)
        self.assertEqual(talker.contexts[1].tool_choice, "auto")
        self.assertEqual(talker.fallbacks, [])

    async def test_two_empty_responses_emit_fallback_without_fabricating_tool_call(self) -> None:
        talker = _ScriptedTalker([[], [_chunk(content=None)]])

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "How about London?"}]))

        self.assertEqual(chunks, [])
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.fallbacks, [EMPTY_RESPONSE_FALLBACK])
        self.assertFalse(any(getattr(chunk.choices[0].delta, "tool_calls", None) for chunk in chunks))

    async def test_finished_result_redelegation_retries_as_text_only(self) -> None:
        context = self._finished_result_context()
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Check Pune weather again.")],
                [_chunk(content="I could not complete the current weather check.")],
            ]
        )

        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].choices[0].delta.content, "I could not complete the current weather check.")
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.contexts[1].get_messages()[-1]["content"], TOOL_RESULT_CORRECTION)
        self.assertEqual(talker.fallbacks, [])

    async def test_repeated_invalid_finished_result_uses_trusted_text(self) -> None:
        trusted = "The current weather check timed out. Please try again."
        context = self._finished_result_context(response_text=trusted)
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Check Pune weather again.")],
                [_tool_chunk("Check Pune weather again.")],
            ]
        )

        chunks = await _collect(talker, context)

        self.assertEqual(chunks, [])
        self.assertEqual(talker.fallbacks, [trusted])

    async def test_dynamic_weather_preserves_city_temperature_and_unit(self) -> None:
        context = self._successful_weather_context()
        talker = _ScriptedTalker([[_chunk(content="Pune is clear at 29 degrees Celsius today.")]])
        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(talker.fallbacks, [])

    def test_dynamic_weather_preserves_temperature_sign(self) -> None:
        payload = {
            "type": "tool_result",
            "tool": "get_weather",
            "status": "success",
            "data": {"city": "Oslo", "temperature": -5, "temperature_unit": "C"},
        }

        self.assertFalse(_weather_grounding_missing(payload, "Oslo is at -5 degrees Celsius."))
        self.assertFalse(_weather_grounding_missing(payload, "Oslo is at minus 5 degrees Celsius."))
        self.assertTrue(_weather_grounding_missing(payload, "Oslo is at 5 degrees Celsius."))

    async def test_dynamic_weather_falls_back_after_two_ungrounded_answers(self) -> None:
        context = self._successful_weather_context()
        trusted = json.loads(async_tool_messages.parse_message(context.get_messages()[-1]).result)["response_text"]
        talker = _ScriptedTalker(
            [
                [_chunk(content="It is sunny there today.")],
                [_chunk(content="Pune has clear skies today.")],
            ]
        )

        chunks = await _collect(talker, context)

        self.assertEqual(chunks, [])
        self.assertEqual(talker.contexts[1].get_messages()[-1]["content"], WEATHER_GROUNDING_CORRECTION)
        self.assertEqual(talker.fallbacks, [trusted])

    async def test_newer_user_turn_can_delegate_after_finished_result(self) -> None:
        context = self._finished_result_context(newer_user_text="How about London?")
        talker = _ScriptedTalker([[_tool_chunk("Get the current weather in London.")]])

        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_cached_backend_replay_is_withheld_and_retried_as_native_tool_call(self) -> None:
        native_call = SimpleNamespace(index=0)
        original_messages = [{"role": "user", "content": "Repeat that weather."}]
        context = LLMContext(original_messages.copy(), tools=[], tool_choice="auto")
        talker = _ScriptedTalker(
            [
                [
                    _chunk(content="The current weather in Cairo is 26.8 degrees C with clear skies, "),
                    _chunk(content="feeling like 28.2 degrees C."),
                ],
                [_chunk(tool_calls=[native_call])],
            ]
        )
        talker.remember_backend_response("In Cairo, it's 26.8 degrees C with clear, and it feels like 28.2 degrees C.")

        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].choices[0].delta.tool_calls, [native_call])
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(context.get_messages(), original_messages)
        self.assertEqual(talker.contexts[1].get_messages()[-1]["content"], CACHED_RESPONSE_CORRECTION)
        self.assertEqual(talker.fallbacks, [])

    async def test_short_clarification_is_not_misclassified_as_cached_replay(self) -> None:
        talker = _ScriptedTalker([[_chunk(content="It was 26.8 degrees.")]])
        talker.remember_backend_response("In Cairo, it's 26.8 degrees C with clear, and it feels like 28.2 degrees C.")

        chunks = await _collect(
            talker,
            LLMContext([{"role": "user", "content": "What was the first temperature?"}]),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)

    async def test_repeated_cached_replay_fails_closed_without_leaking_stale_speech(self) -> None:
        replay = "In Cairo, it's 26.8 degrees C with clear, and it feels like 28.2 degrees C."
        talker = _ScriptedTalker([[_chunk(content=replay)], [_chunk(content=replay)]])
        talker.remember_backend_response(replay)

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(chunks, [])
        self.assertEqual(talker.fallbacks, [EMPTY_RESPONSE_FALLBACK])

    async def test_explicit_repeat_retries_tool_call_that_changes_structured_subject(self) -> None:
        context = LLMContext([{"role": "user", "content": "Repeat that weather."}], tools=[], tool_choice="auto")
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current weather in Pune again.")],
                [_tool_chunk("Get the current weather in Toronto again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "Toronto", "units": "celsius"}},
                "response_text": "In Toronto, it is 24.5 degrees C.",
            }
        )

        chunks = await _collect(talker, context)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            chunks[0].choices[0].delta.tool_calls[0].function.arguments,
            json.dumps({"query": "Get the current weather in Toronto again."}),
        )
        self.assertEqual(len(talker.contexts), 2)
        correction = talker.contexts[1].get_messages()[-1]["content"]
        self.assertEqual(correction, REPEAT_SUBJECT_CORRECTION.format(values='["Toronto"]'))
        self.assertEqual(context.get_messages(), [{"role": "user", "content": "Repeat that weather."}])
        self.assertEqual(talker.fallbacks, [])

    async def test_weather_repeat_prefers_canonical_result_city_over_composite_input(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current weather in Lagos again.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {
                    "arguments": {"city": "Lagos Nigeria", "units": "celsius"},
                    "result": {"status": "success", "city": "Lagos", "country": "Nigeria"},
                },
                "response_text": "In Lagos, it is 25.8 degrees C.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_weather_repeat_accepts_weatherapi_japanese_city_suffix(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current weather in Osaka again.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {
                    "arguments": {"city": "Osaka Japan", "units": "celsius"},
                    "result": {"status": "success", "city": "Osaka-Shi", "country": "Japan"},
                },
                "response_text": "In Osaka-Shi, it is 27.9 degrees C.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_weather_repeat_still_rejects_drift_from_canonical_result_city(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current weather in Accra again.")],
                [_tool_chunk("Get the current weather in Lagos again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {
                    "arguments": {"city": "Lagos Nigeria", "units": "celsius"},
                    "result": {"status": "success", "city": "Lagos", "country": "Nigeria"},
                },
                "response_text": "In Lagos, it is 25.8 degrees C.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(
            talker.contexts[1].get_messages()[-1]["content"],
            REPEAT_SUBJECT_CORRECTION.format(values='["Lagos"]'),
        )
        self.assertEqual(talker.fallbacks, [])

    async def test_repeated_subject_drift_fails_closed_without_executing_wrong_tool_call(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current weather in Pune again.")],
                [_tool_chunk("Get the current weather in Mumbai again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "Toronto", "units": "celsius"}},
                "response_text": "In Toronto, it is 24.5 degrees C.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(chunks, [])
        self.assertEqual(talker.fallbacks, [EMPTY_RESPONSE_FALLBACK])

    async def test_non_repeat_turn_is_not_rewritten_by_subject_guard(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current weather in London.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "Toronto", "units": "celsius"}},
                "response_text": "In Toronto, it is 24.5 degrees C.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "How about London?"}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_failed_result_clears_stale_repeat_subject_guard(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current weather in Hyderabad again.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "Chennai", "units": "celsius"}},
                "response_text": "In Chennai, it is 29 degrees C.",
            }
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "not_found",
                "data": {"arguments": {"city": "Hyderbod", "units": "celsius"}},
                "response_text": "I could not find current weather for Hyderbod.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_stock_repeat_preserves_actual_company_name_argument(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current stock price for Tesla again.")],
                [_tool_chunk("Get the current stock price for NVIDIA again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_stock_price",
                "status": "success",
                "data": {"arguments": {"company_name": "NVIDIA"}},
                "response_text": "NVIDIA is trading at 180 dollars.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that stock price."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.fallbacks, [])

    async def test_explicit_stock_subject_uses_matching_history_after_newer_weather_result(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current stock price for NVIDIA again.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_stock_price",
                "status": "success",
                "data": {"arguments": {"company_name": "NVIDIA"}},
                "response_text": "NVIDIA is trading at 180 dollars.",
            }
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "London", "units": "celsius"}},
                "response_text": "In London, it is 20 degrees C.",
            }
        )

        chunks = await _collect(
            talker,
            LLMContext([{"role": "user", "content": "Repeat the NVIDIA stock price now."}]),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_explicit_stock_subject_survives_failed_quote_and_newer_weather(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current stock price for NVIDIA again.")],
                [_tool_chunk("Get the current stock price for NVIDIA again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_stock_price",
                "status": "unavailable",
                "data": {"arguments": {"company_name": "NVIDIA"}},
                "response_text": "I wasn't able to get the stock price right now.",
            }
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "London", "units": "celsius"}},
                "response_text": "In London, it is 20 degrees C.",
            }
        )

        chunks = await _collect(
            talker,
            LLMContext([{"role": "user", "content": "Repeat the NVIDIA stock price now."}]),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_implicit_stock_repeat_uses_latest_stock_not_newer_weather(self) -> None:
        talker = _ScriptedTalker([[_tool_chunk("Get the current stock price for NVIDIA again.")]])
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_stock_price",
                "status": "success",
                "data": {"arguments": {"company_name": "NVIDIA"}},
                "response_text": "NVIDIA is trading at 180 dollars.",
            }
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "London", "units": "celsius"}},
                "response_text": "In London, it is 20 degrees C.",
            }
        )

        chunks = await _collect(
            talker,
            LLMContext([{"role": "user", "content": "Repeat that stock price again."}]),
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 1)
        self.assertEqual(talker.fallbacks, [])

    async def test_failed_stock_result_does_not_clear_weather_repeat_history(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Get the current weather in Pune again.")],
                [_tool_chunk("Get the current weather in Toronto again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_weather",
                "status": "success",
                "data": {"arguments": {"city": "Toronto", "units": "celsius"}},
                "response_text": "In Toronto, it is 24 degrees C.",
            }
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "get_stock_price",
                "status": "unavailable",
                "data": {"arguments": {"company_name": "NVIDIA"}},
                "response_text": "I wasn't able to get the stock price right now.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Repeat that weather."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.fallbacks, [])

    async def test_multi_argument_repeat_requires_every_prior_value(self) -> None:
        talker = _ScriptedTalker(
            [
                [_tool_chunk("Calculate BMI for 70 kilograms and 1.8 metres again.")],
                [_tool_chunk("Calculate BMI for 70 kilograms and 1.75 metres again.")],
            ]
        )
        talker.remember_backend_result(
            {
                "type": "tool_result",
                "tool": "calculate_bmi",
                "status": "success",
                "data": {"arguments": {"weight_kg": 70.0, "height_m": 1.75}},
                "response_text": "Your BMI is 22.9.",
            }
        )

        chunks = await _collect(talker, LLMContext([{"role": "user", "content": "Calculate that BMI again."}]))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(len(talker.contexts), 2)
        self.assertEqual(talker.fallbacks, [])


if __name__ == "__main__":
    unittest.main()
