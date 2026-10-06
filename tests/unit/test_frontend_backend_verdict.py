# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D101, D102

"""Tests for the Frontend/Backend frontend verdict and backend-run ownership."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from typing import Any
from unittest.mock import patch

from pipecat.services.llm_service import FunctionCallParams

from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.airline.tools import TOOLS_SCHEMA as AIRLINE_TOOLS_SCHEMA
from examples.frontend_backend_agent.generic.backend import GenericThinkerBackend
from examples.frontend_backend_agent.generic.client_tools import build_client_tool_specs
from examples.frontend_backend_agent.generic.tools import TOOLS_SCHEMA as GENERIC_TOOLS_SCHEMA
from examples.frontend_backend_agent.src.delegation import DelegationRun, current_run_id
from examples.frontend_backend_agent.src.frontend_verdict import (
    ACK_PROGRESS_WORDS,
    CORRECTION_CUE_WORDS,
    decide,
    pure_ack_or_progress,
    same_request,
    with_task_field,
)
from examples.frontend_backend_agent.src.history import ConversationTranscript
from examples.frontend_backend_agent.src.stage_metrics import StageMetricsCoordinator
from examples.frontend_backend_agent.src.tool_handlers import build_handlers
from examples.shared.tool_runtime import tool_parameter_schema

RUNNING = "Get the current weather in Pune and Mumbai."


class DecideTests(unittest.TestCase):
    """Code keeps the running request only for a pure acknowledgement with the query kept."""

    def test_acknowledgements_and_progress_checks_continue_whatever_the_model_said(self) -> None:
        for utterance in ("Okay.", "Did you find anything?", "Hold on a second.", "mm-hmm", "Are you still there?"):
            for task in ("continue", "new", None, "bogus"):
                with self.subTest(utterance=utterance, task=task):
                    verdict = decide(task, RUNNING, RUNNING, utterance)
                    self.assertEqual(verdict.decision, "continue")
                    self.assertEqual(verdict.reason, "model" if task == "continue" else "acknowledgement")

    def test_explicit_repeats_start_a_new_request(self) -> None:
        for utterance in ("Check again.", "Refresh that.", "Can you repeat the check?", "One more time please."):
            with self.subTest(utterance=utterance):
                verdict = decide("continue", RUNNING, RUNNING, utterance)
                self.assertEqual((verdict.decision, verdict.reason), ("new", "substantive"))

    def test_the_model_continues_a_restatement_that_kept_the_query(self) -> None:
        verdict = decide("continue", RUNNING, RUNNING, "Mumbai please")

        self.assertEqual((verdict.decision, verdict.reason), ("continue", "same_query"))

    def test_without_the_model_a_restatement_is_substantive(self) -> None:
        verdict = decide(None, RUNNING, RUNNING, "Mumbai please")

        self.assertEqual((verdict.decision, verdict.reason), ("new", "substantive"))

    def test_an_affirmative_answers_a_pending_consent_question(self) -> None:
        verdict = decide("continue", RUNNING, RUNNING, "yes", consent_pending=True)

        self.assertEqual((verdict.decision, verdict.reason), ("new", "consent_answer"))
        self.assertEqual(decide("continue", RUNNING, RUNNING, "yes").decision, "continue")

    def test_non_numeric_corrections_with_the_old_query_copied_are_new(self) -> None:
        for utterance in (
            "Not Pune, Mumbai.",
            "Actually the other one.",
            "No, my daughter's account.",
            "Okay, but not that one.",
        ):
            with self.subTest(utterance=utterance):
                self.assertEqual(decide("continue", RUNNING, RUNNING, utterance).decision, "new")

    def test_numbers_and_codes_are_new(self) -> None:
        self.assertEqual(decide("continue", RUNNING, RUNNING, "order 1152").decision, "new")
        self.assertEqual(decide(None, RUNNING, RUNNING, "okay 3JA7XV").decision, "new")

    def test_an_acknowledgement_with_a_changed_query_is_new(self) -> None:
        verdict = decide("continue", "Get the current weather in Delhi.", RUNNING, "okay")

        self.assertEqual((verdict.decision, verdict.reason), ("new", "query_changed"))

    def test_empty_unknown_and_non_english_utterances_are_new(self) -> None:
        for utterance in ("", "   ", "d'accord", "vale gracias", "okay thanks bye now"):
            with self.subTest(utterance=utterance):
                self.assertEqual(decide("continue", RUNNING, RUNNING, utterance).decision, "new")

    def test_a_restatement_with_detail_stays_new(self) -> None:
        verdict = decide("new", RUNNING, RUNNING, "the weather in Pune please")

        self.assertEqual(verdict.decision, "new")

    def test_word_lists_never_overlap_and_contain_no_digits(self) -> None:
        self.assertFalse(ACK_PROGRESS_WORDS & CORRECTION_CUE_WORDS)
        self.assertFalse(any(character.isdigit() for word in ACK_PROGRESS_WORDS for character in word))
        self.assertFalse(pure_ack_or_progress("okay again"))

    def test_same_request_ignores_case_punctuation_and_spacing(self) -> None:
        self.assertTrue(same_request("Get the weather in Pune.", "get  the weather in pune"))
        self.assertFalse(same_request("Get the weather in Pune.", "Get the weather in Mumbai."))


class TaskFieldSchemaTests(unittest.TestCase):
    def test_task_is_optional_enum_on_call_backend_only_for_both_domains(self) -> None:
        for schema in (GENERIC_TOOLS_SCHEMA, AIRLINE_TOOLS_SCHEMA):
            with self.subTest(schema=schema):
                original = tool_parameter_schema(schema, "call_backend")
                updated = with_task_field(schema)
                parameters = tool_parameter_schema(updated, "call_backend")

                self.assertEqual(parameters["properties"]["task"]["enum"], ["continue", "new"])
                self.assertEqual(parameters["required"], original["required"])
                self.assertNotIn("task", original["properties"])
                self.assertNotIn("task", tool_parameter_schema(schema, "call_backend")["properties"])
                self.assertEqual(
                    tool_parameter_schema(updated, "cancel_backend"),
                    tool_parameter_schema(schema, "cancel_backend"),
                )


class DelegationRunTests(unittest.IsolatedAsyncioTestCase):
    async def test_adoption_supersedes_the_previous_caller_at_once(self) -> None:
        release = asyncio.Event()

        async def work(_progress) -> dict[str, Any]:
            await release.wait()
            return {"answer": 1}

        run = DelegationRun("run", "query", work)
        first = run.attach()
        first_waiter = asyncio.create_task(run.wait(first))
        await asyncio.sleep(0)
        second = run.adopt(None)

        with self.assertRaises(asyncio.CancelledError):
            await first_waiter
        self.assertFalse(run.task.done())
        release.set()
        self.assertEqual(await run.wait(second), {"answer": 1})
        self.assertTrue(run.owns(second))

    async def test_cancelling_a_displaced_caller_never_touches_the_task(self) -> None:
        release = asyncio.Event()

        async def work(_progress) -> dict[str, Any]:
            await release.wait()
            return {"answer": 2}

        run = DelegationRun("run", "query", work)
        first = run.attach()
        waiter = asyncio.create_task(run.wait(first))
        await asyncio.sleep(0)
        second = run.adopt(None)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.assertFalse(run.task.done())
        release.set()
        self.assertEqual(await run.wait(second), {"answer": 2})

    async def test_cancelling_the_owner_cancels_the_run(self) -> None:
        async def work(_progress) -> dict[str, Any]:
            await asyncio.Event().wait()
            return {}

        run = DelegationRun("run", "query", work)
        owner = run.attach()
        waiter = asyncio.create_task(run.wait(owner))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        await asyncio.sleep(0)
        self.assertTrue(run.task.cancelled())

    async def test_progress_reaches_only_the_current_owner_and_run_id_is_bound(self) -> None:
        events: list[tuple[str, str]] = []
        step = asyncio.Event()
        seen_run_ids: list[str | None] = []

        async def work(progress) -> dict[str, Any]:
            seen_run_ids.append(current_run_id())
            await step.wait()
            await progress("tick")
            return {}

        async def first_listener(event) -> None:
            events.append(("first", event))

        async def second_listener(event) -> None:
            events.append(("second", event))

        run = DelegationRun("run-7", "query", work, first_listener)
        run.attach()
        await asyncio.sleep(0)
        future = run.adopt(second_listener)
        step.set()
        await run.wait(future)

        self.assertEqual(events, [("second", "tick")])
        self.assertEqual(seen_run_ids, ["run-7"])
        self.assertIsNone(current_run_id())


class _GatedGenericPlanner:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.queries: list[str] = []

    async def plan(self, *, query: str, state: dict) -> dict:
        if state["planning_round"] > 1:
            return {"complete": True}
        self.queries.append(query)
        await self.release.wait()
        return {"tool": "lookup", "params": {"record_id": query.split()[-1].rstrip(".")}}


def _generic_backend(planner) -> GenericThinkerBackend:
    async def execute(calls, _timeout):
        return [{"ok": True, "record": "found"} for _call in calls]

    client_tools = build_client_tool_specs(
        (
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a record by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"record_id": {"type": "string"}},
                    "required": ["record_id"],
                    "additionalProperties": False,
                },
            },
        )
    )
    return GenericThinkerBackend(
        planner=planner,
        tools={},
        enabled_tools=("lookup",),
        client_tools=client_tools,
        client_tool_executor=execute,
        overall_timeout_seconds=5,
        planner_timeout_seconds=5,
        max_planning_rounds=2,
    )


class GenericContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_adopts_the_running_plan_and_delivers_once(self) -> None:
        planner = _GatedGenericPlanner()
        backend = _generic_backend(planner)
        events: list[str] = []

        async def on_started(event) -> None:
            events.append(event.marker)

        first = asyncio.create_task(backend.call("Look up record R1.", on_started=on_started))
        await asyncio.sleep(0.01)
        self.assertEqual(backend.running_query(), "Look up record R1.")
        second = asyncio.create_task(backend.call("Look up record R1.", on_started=on_started, continue_active=True))
        with self.assertRaises(asyncio.CancelledError):
            await first
        planner.release.set()
        payload = await second

        self.assertEqual(payload["tool"], "lookup")
        self.assertEqual(planner.queries, ["Look up record R1."])
        self.assertIn("ThinkerContinued", events)
        self.assertIsNone(backend.state.active_task)
        self.assertIsNone(backend.state.active_call_id)
        self.assertIsNone(backend.running_query())

    async def test_a_chain_of_continues_hands_the_result_to_the_last_caller(self) -> None:
        planner = _GatedGenericPlanner()
        backend = _generic_backend(planner)

        first = asyncio.create_task(backend.call("Look up record R1."))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(backend.call("Look up record R1.", continue_active=True))
        await asyncio.sleep(0.01)
        third = asyncio.create_task(backend.call("Look up record R1.", continue_active=True))
        for superseded in (first, second):
            with self.assertRaises(asyncio.CancelledError):
                await superseded
        planner.release.set()

        self.assertEqual((await third)["tool"], "lookup")
        self.assertEqual(len(planner.queries), 1)

    async def test_cancel_active_ends_the_adopted_run(self) -> None:
        planner = _GatedGenericPlanner()
        backend = _generic_backend(planner)

        first = asyncio.create_task(backend.call("Look up record R1."))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(backend.call("Look up record R1.", continue_active=True))
        await asyncio.sleep(0.01)
        self.assertTrue(backend.cancel_active("user_cancelled"))
        for caller in (first, second):
            with self.assertRaises(asyncio.CancelledError):
                await caller
        self.assertIsNone(backend.state.active_task)

    async def test_new_keeps_the_existing_detach_behaviour(self) -> None:
        planner = _GatedGenericPlanner()
        backend = _generic_backend(planner)

        first = asyncio.create_task(backend.call("Look up record R1."))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(backend.call("Look up record R2."))
        await asyncio.sleep(0.01)
        planner.release.set()

        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual((await second)["tool"], "lookup")
        self.assertEqual(planner.queries, ["Look up record R1.", "Look up record R2."])

    async def test_continue_without_a_running_task_starts_a_normal_run(self) -> None:
        planner = _GatedGenericPlanner()
        planner.release.set()
        backend = _generic_backend(planner)

        payload = await backend.call("Look up record R1.", continue_active=True)

        self.assertEqual(payload["tool"], "lookup")
        self.assertEqual(len(planner.queries), 1)


class _GatedAirlinePlanner:
    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.queries: list[str] = []

    async def plan(self, *, query: str, slots: dict, state: dict) -> dict:
        self.queries.append(query)
        await self.release.wait()
        return {
            "tool": "response_hint",
            "reason": "params_missing",
            "action": "req_params",
            "context": "flight_search",
            "response_text": "Where are you flying from?",
        }


class AirlineContinuationTests(unittest.IsolatedAsyncioTestCase):
    async def test_continue_adopts_instead_of_cancel_and_restart(self) -> None:
        planner = _GatedAirlinePlanner()
        thinker = ThinkerBackend(planner=planner, backend=object())

        first = asyncio.create_task(thinker.call("Find flights to Seattle."))
        await asyncio.sleep(0.01)
        self.assertEqual(thinker.running_query(), "Find flights to Seattle.")
        second = asyncio.create_task(thinker.call("Find flights to Seattle.", continue_active=True))
        with self.assertRaises(asyncio.CancelledError):
            await first
        planner.release.set()

        self.assertEqual((await second)["response_text"], "Where are you flying from?")
        self.assertEqual(planner.queries, ["Find flights to Seattle."])
        self.assertIsNone(thinker.state.active_task)

    async def test_new_still_cancels_and_restarts(self) -> None:
        planner = _GatedAirlinePlanner()
        thinker = ThinkerBackend(planner=planner, backend=object())

        first = asyncio.create_task(thinker.call("Find flights to Seattle."))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(thinker.call("Find flights to Denver."))
        await asyncio.sleep(0.01)
        planner.release.set()

        with self.assertRaises(asyncio.CancelledError):
            await first
        await second
        self.assertEqual(planner.queries, ["Find flights to Seattle.", "Find flights to Denver."])


class _RecordingLLM:
    def __init__(self) -> None:
        self.frames: list = []

    async def push_frame(self, frame, direction=None) -> None:
        self.frames.append(frame)


class _FakeThinker:
    supports_task_continuation = True
    tool_result_mode_default = "talker"

    def __init__(self, running: str | None) -> None:
        self.running = running
        self.calls: list[dict[str, Any]] = []

    def running_query(self) -> str | None:
        return self.running

    async def call(self, query, slots=None, *, on_started=None, continue_active=False):
        self.calls.append({"query": query, "slots": slots, "continue_active": continue_active})
        if on_started is not None:
            from examples.frontend_backend_agent.src.protocol import ThinkerLifecycleEvent

            marker = "ThinkerContinued" if continue_active else "ThinkerStarted"
            await on_started(ThinkerLifecycleEvent(marker=marker, call_id="run-1", query=query))
            await asyncio.sleep(0.05)
        return {"type": "tool_result", "tool": "lookup", "status": "success", "response_text": "Done.", "data": {}}

    def cancel_active(self, reason="new_user_query") -> bool:
        return False

    def cancel_pending_work(self) -> bool:
        return False


def _params(arguments: dict[str, Any], results: list) -> FunctionCallParams:
    async def result_callback(result, *, properties=None) -> None:
        results.append(result)

    return FunctionCallParams(
        function_name="call_backend",
        tool_call_id="tool-1",
        arguments=arguments,
        llm=_RecordingLLM(),
        pipeline_worker=None,
        context=None,
        result_callback=result_callback,
    )


class HandlerVerdictTests(unittest.IsolatedAsyncioTestCase):
    async def _run(
        self,
        thinker: _FakeThinker,
        arguments: dict[str, Any],
        utterance: str,
        *,
        frontend_verdict: bool = True,
        emitted: list[str] | None = None,
    ) -> list:
        transcript = ConversationTranscript()
        transcript.record_user(utterance)
        emitted = emitted if emitted is not None else []

        async def emit(text: str) -> bool:
            emitted.append(text)
            return True

        results: list = []
        handlers = build_handlers(
            thinker,
            filler_policy="talker_authored",
            allow_talker_frames=False,
            realtime_filler_emitter=emit,
            frontend_verdict=frontend_verdict,
            transcript=transcript,
        )
        await handlers["call_backend"](_params(arguments, results))
        return results

    async def test_acknowledgement_continues_and_task_never_reaches_slots(self) -> None:
        thinker = _FakeThinker(running="Check the renewal status for the user.")

        await self._run(
            thinker,
            {"query": "Check the renewal status for the user.", "task": "continue"},
            "okay",
        )

        self.assertEqual(thinker.calls[0]["continue_active"], True)
        self.assertNotIn("task", thinker.calls[0]["slots"])

    async def test_substantive_speech_takes_the_existing_new_path(self) -> None:
        thinker = _FakeThinker(running="Check the renewal status for the user.")

        await self._run(
            thinker,
            {"query": "Check the renewal status for the user.", "task": "continue"},
            "not the renewal, the booking",
        )

        self.assertEqual(thinker.calls[0]["continue_active"], False)

    async def test_flag_off_never_continues(self) -> None:
        thinker = _FakeThinker(running="Check the renewal status for the user.")

        await self._run(
            thinker,
            {"query": "Check the renewal status for the user.", "task": "continue"},
            "okay",
            frontend_verdict=False,
        )

        self.assertEqual(thinker.calls[0]["continue_active"], False)
        self.assertNotIn("task", thinker.calls[0]["slots"])

    async def test_continue_speaks_an_accepted_progress_phrase_at_once(self) -> None:
        thinker = _FakeThinker(running="Check the renewal status for the user.")
        emitted: list[str] = []

        with patch.dict(os.environ, {"FRONTEND_BACKEND_TALKER_FILLER_MODE": "emit"}):
            await self._run(
                thinker,
                {
                    "query": "Check the renewal status for the user.",
                    "filler_text": "Still checking that renewal status.",
                    "task": "continue",
                },
                "are you still there",
                emitted=emitted,
            )

        self.assertEqual(emitted, ["Still checking that renewal status."])

    async def test_continue_stays_silent_without_a_phrase_or_with_filler_off(self) -> None:
        for environment, arguments in (
            ({"FRONTEND_BACKEND_TALKER_FILLER_MODE": "emit"}, {}),
            (
                {"FRONTEND_BACKEND_TALKER_FILLER_MODE": "off"},
                {"filler_text": "Still checking that renewal status."},
            ),
        ):
            with self.subTest(environment=environment):
                thinker = _FakeThinker(running="Check the renewal status for the user.")
                emitted: list[str] = []
                with patch.dict(os.environ, environment):
                    await self._run(
                        thinker,
                        {"query": "Check the renewal status for the user.", "task": "continue", **arguments},
                        "okay",
                        emitted=emitted,
                    )
                self.assertEqual(emitted, [])

    async def test_no_running_task_ignores_the_task_field(self) -> None:
        thinker = _FakeThinker(running=None)

        await self._run(thinker, {"query": "Check the renewal status.", "task": "continue"}, "okay")

        self.assertEqual(thinker.calls[0]["continue_active"], False)


class StageMetricsCorrelationTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_adopters_binding_survives_the_displaced_cleanup(self) -> None:
        async def emit(_frame) -> None:
            return None

        metrics = StageMetricsCoordinator(emit)
        await metrics.bind_backend_call("tool-old", "run-1")
        await metrics.bind_backend_call("tool-new", "run-1")
        await metrics.cleanup_tool_call("tool-old")

        self.assertIn("run-1", metrics._backend_turns)
        await metrics.cleanup_tool_call("tool-new")
        self.assertNotIn("run-1", metrics._backend_turns)

    async def test_one_to_one_cleanup_is_unchanged(self) -> None:
        async def emit(_frame) -> None:
            return None

        metrics = StageMetricsCoordinator(emit)
        await metrics.bind_backend_call("tool-1", "run-1")
        await metrics.cleanup_tool_call("tool-1")

        self.assertNotIn("run-1", metrics._backend_turns)


class VerdictPromptTests(unittest.TestCase):
    def test_few_shots_close_every_call_and_never_leave_work_running(self) -> None:
        from examples.frontend_backend_agent import pipeline

        messages = pipeline._load_prompt_few_shots("frontend_verdict_talker")
        call_ids = [call["id"] for message in messages for call in message.get("tool_calls") or ()]
        finished = {
            json.loads(message["content"])["tool_call_id"]
            for message in messages
            if message["role"] == "developer" and json.loads(message["content"])["status"] == "finished"
        }
        tasks = [
            json.loads(call["function"]["arguments"]).get("task")
            for message in messages
            for call in message.get("tool_calls") or ()
        ]

        self.assertEqual(set(call_ids), finished)
        self.assertEqual(tasks, [None, "continue", "new"])
        self.assertIn("overrides any rule", pipeline._load_required_catalog_prompt("frontend_verdict_talker"))


if __name__ == "__main__":
    unittest.main()
