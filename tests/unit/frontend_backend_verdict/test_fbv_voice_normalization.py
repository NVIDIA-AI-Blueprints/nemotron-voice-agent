# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Normalization in a session: transcript hook, argument hook, local answers, retry guard, config, redaction."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from dataclasses import fields, replace
from io import StringIO
from pathlib import Path
from typing import Any
from unittest import mock

from _fbv_voice_fakes import (
    PACKAGE_DIR,
    FakeChatClient,
    SessionHarness,
    base_config,
    delegate_response,
    tau2_update_with,
    text_response,
    tool_response,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients
from examples.frontend_backend_verdict.voice.agent.sinks import EventLog
from examples.frontend_backend_verdict.voice.cli.normalization_replay import replay
from examples.frontend_backend_verdict.voice.config import load_voice_config
from examples.frontend_backend_verdict.voice.errors import VoiceConfigError
from examples.frontend_backend_verdict.voice.speech.stubs import StubRecognizer

PROFILES = PACKAGE_DIR / "config" / "profiles"
SPOKEN = "my user ID is Jordan underscore Lee underscore eight two six four"
WRITTEN = "my user ID is jordan_lee_8264"
NOT_FOUND = "Error: User jordan_lee_8264 not found"
TOOLS = [
    {
        "type": "function",
        "name": "get_user_details",
        "description": "Get the details of a user.",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}, "required": ["user_id"]},
    },
    {
        "type": "function",
        "name": "get_reservation_details",
        "description": "Get the details of a reservation.",
        "parameters": {"type": "object", "properties": {"reservation_id": {"type": "string"}}},
    },
]


def _env() -> Any:
    return mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})


with _env():
    PROFILE = load_voice_config(PROFILES / "tau3_eval_normalization.yaml")
    BACKEND_ONLY = load_voice_config(PROFILES / "backend_only.yaml")


def _config(base: Any = None, *, max_local_rounds: int | None = None) -> Any:
    normalization = PROFILE.normalization
    if max_local_rounds is not None:
        arguments = replace(normalization.tool_arguments, max_local_rounds=max_local_rounds)
        normalization = replace(normalization, tool_arguments=arguments)
    return replace(base or base_config(), normalization=normalization)


def lookup(user_id: str, call_id: str = "c1") -> Any:
    return tool_response(("get_user_details", {"user_id": user_id}), ids=[call_id])


def output_event(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


class _Base(unittest.IsolatedAsyncioTestCase):
    async def start(
        self, *, config: Any = None, transcripts: list[str] | None = None, event_log: EventLog | None = None
    ):
        self.frontend = FakeChatClient()
        self.backend = FakeChatClient()
        cfg = config or _config()
        self.harness = SessionHarness(
            config=cfg,
            clients=AgentClients(backend=self.backend, frontend=self.frontend if cfg.agent.frontend_enabled else None),
            recognizer=StubRecognizer(transcripts),
            event_log=event_log,
        )
        await self.harness.start(tau2_update_with(TOOLS, "Policy text."))
        return self.harness

    async def asyncTearDown(self) -> None:
        await self.harness.close()

    def wire_calls(self) -> list[dict[str, Any]]:
        return self.harness.of_type("response.function_call_arguments.done")

    def tool_messages(self, call_index: int) -> list[Any]:
        return [m for m in self.backend.calls[call_index]["messages"] if m.role == "tool"]


class TranscriptHookTests(_Base):
    async def test_the_agent_gets_written_ids_and_the_wire_keeps_the_raw_transcript(self) -> None:
        harness = await self.start(transcripts=[SPOKEN])
        self.frontend.queue(text_response("Thanks."))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.frontend.calls[0]["messages"][-1].content, WRITTEN)
        completed = harness.of_type("conversation.item.input_audio_transcription.completed")
        self.assertEqual(completed[0]["transcript"], SPOKEN)
        self.assertIn("Identifiers in user turns:", self.frontend.calls[0]["messages"][0].content)
        users = [m.content for m in harness.runners[0].state.frontend_history.messages if m.role == "user"]
        self.assertEqual(users, [WRITTEN])

    async def test_off_by_default(self) -> None:
        harness = await self.start(config=base_config(), transcripts=[SPOKEN])
        self.frontend.queue(text_response("Thanks."))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.frontend.calls[0]["messages"][-1].content, SPOKEN)
        self.assertNotIn("Identifiers in user turns:", self.frontend.calls[0]["messages"][0].content)


class ArgumentHookTests(_Base):
    async def test_a_rewritten_call_goes_out_canonical_and_is_stored_canonical(self) -> None:
        harness = await self.start()
        self.frontend.queue(delegate_response("Look up Jordan_Lee_8264."))
        self.backend.queue(lookup("Jordan_Lee_8264"))
        await harness.speak()
        await harness.wait_for("response.done")
        (wire,) = self.wire_calls()
        self.assertEqual(json.loads(wire["arguments"]), {"user_id": "jordan_lee_8264"})
        stored = harness.runners[0].state.pending.backend_history.messages[-1].tool_calls[0]
        self.assertEqual(stored.arguments_json, wire["arguments"])

    async def test_an_invalid_id_is_answered_locally_and_never_reaches_the_wire(self) -> None:
        harness = await self.start()
        self.frontend.queue(delegate_response("Look up YA."))
        self.backend.queue(lookup("YA"), text_response("Please say your complete user ID."))
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual(self.wire_calls(), [])
        (result,) = self.tool_messages(1)
        self.assertIn('"ya" is not a complete user ID', result.content)
        self.assertIsNone(harness.runners[0].state.pending)
        usage = harness.of_type("response.done")[0]["response"]["usage"]
        self.assertEqual(usage["input_tokens"], 30)  # frontend + both backend steps

    async def test_a_mixed_batch_sends_one_call_and_merges_the_local_result_on_resume(self) -> None:
        harness = await self.start()
        self.frontend.queue(delegate_response("Look up YA and H9Z1C."))
        self.backend.queue(
            tool_response(
                ("get_user_details", {"user_id": "YA"}),
                ("get_reservation_details", {"reservation_id": "H9Z1C"}),
                ids=["c1", "c2"],
            ),
            text_response("Found the reservation."),
        )
        await harness.speak()
        await harness.wait_for("response.done")
        self.assertEqual([call["call_id"] for call in self.wire_calls()], ["c2"])
        await harness.send(output_event("c2", '{"reservation_id": "H9Z1C"}'))
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        results = {m.tool_call_id: m.content for m in self.tool_messages(1)}
        self.assertIn("not a complete user ID", results["c1"])
        self.assertEqual(results["c2"], '{"reservation_id": "H9Z1C"}')

    async def test_a_permanently_failed_id_is_not_sent_again(self) -> None:
        harness = await self.start()
        self.frontend.queue(
            delegate_response("Look up jordan_lee_8264."), delegate_response("Look up JORDAN_LEE_8264.")
        )
        self.backend.queue(
            lookup("jordan_lee_8264", "c1"),
            text_response("I could not find that ID."),
            lookup("JORDAN_LEE_8264", "c2"),
            text_response("Is that j, o, r, d, a, n, underscore, l, e, e?"),
        )
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(output_event("c1", NOT_FOUND))
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        await harness.idle()
        await harness.speak()
        await harness.wait_for("response.done", 3)
        self.assertEqual([call["call_id"] for call in self.wire_calls()], ["c1"])
        (result,) = [m for m in self.tool_messages(3) if m.tool_call_id == "c2"]
        self.assertIn("already failed in this conversation", result.content)

    async def test_a_transient_error_does_not_block_a_retry(self) -> None:
        harness = await self.start()
        self.frontend.queue(delegate_response("Look up jordan_lee_8264."), delegate_response("Try again."))
        self.backend.queue(
            lookup("jordan_lee_8264", "c1"),
            text_response("The system is busy."),
            lookup("jordan_lee_8264", "c2"),
        )
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(output_event("c1", "Error: service unavailable, try again"))
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        await harness.idle()
        await harness.speak()
        await harness.wait_for("response.done", 3)
        self.assertEqual([call["call_id"] for call in self.wire_calls()], ["c1", "c2"])

    async def test_after_max_local_rounds_the_canonical_call_goes_out(self) -> None:
        harness = await self.start(config=_config(max_local_rounds=1))
        self.frontend.queue(delegate_response("Look up YA."))
        self.backend.queue(lookup("YA", "c1"), lookup("Y A", "c2"))
        await harness.speak()
        await harness.wait_for("response.done")
        (wire,) = self.wire_calls()
        self.assertEqual((wire["call_id"], json.loads(wire["arguments"])), ("c2", {"user_id": "ya"}))
        stored = harness.runners[0].state.pending.backend_history.messages[-1].tool_calls[0]
        self.assertEqual(stored.arguments_json, wire["arguments"])

    async def test_barge_in_during_a_local_round_leaves_the_session_unchanged(self) -> None:
        harness = await self.start()
        self.frontend.queue(delegate_response("Look up YA."), delegate_response("Merged."))
        self.backend.queue(lookup("YA", "c1"), text_response("never"), text_response("Merged answer."))
        self.backend.gates[1] = asyncio.Event()  # the step after the local answer hangs
        await harness.speak()
        for _ in range(100):
            if len(self.backend.calls) == 2:
                break
            await asyncio.sleep(0.005)
        self.assertEqual(len(self.backend.calls), 2)  # the local answer is being processed
        self.assertIn("not a complete user ID", self.tool_messages(1)[0].content)
        runner = harness.runners[0]
        runner_state = runner.state
        self.assertEqual(harness.session.turns.state, "THINKING")
        self.frontend.gates[1] = asyncio.Event()  # hold the merged turn so the state can be inspected
        await harness.speak()
        for _ in range(100):
            if len(self.frontend.calls) == 2:
                break
            await asyncio.sleep(0.005)
        self.assertEqual(len(self.frontend.calls), 2)  # the merged turn started
        self.assertIs(runner.state, runner_state)
        self.assertEqual(runner._failed, frozenset())
        self.assertEqual(runner._held, {})
        self.frontend.gates[1].set()

    async def test_backend_only_mode_rewrites_the_same_way(self) -> None:
        harness = await self.start(config=_config(BACKEND_ONLY))
        self.backend.queue(lookup("Jordan_Lee_8264"))
        await harness.speak()
        await harness.wait_for("response.done")
        (wire,) = self.wire_calls()
        self.assertEqual(json.loads(wire["arguments"]), {"user_id": "jordan_lee_8264"})
        stored = harness.runners[0].state.pending.backend_history.messages[-1].tool_calls[0]
        self.assertEqual(stored.arguments_json, wire["arguments"])


class RedactionTests(_Base):
    async def test_redacted_events_keep_structure_but_no_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            harness = await self.start(
                transcripts=["It's Zed underscore Quux underscore four three nine seven"],
                event_log=EventLog(str(path), redact_content=True),
            )
            self.frontend.queue(delegate_response("Look up Zed_Quux_4397."))
            self.backend.queue(
                lookup("Zed_Quux_4397", "c1"), text_response("Not found."), lookup("ZED_QUUX_4397", "c2")
            )
            self.backend.queue(text_response("Please confirm."))
            await harness.speak()
            await harness.wait_for("response.done")
            await harness.send(output_event("c1", "Error: User zed_quux_4397 not found"))
            await harness.send({"type": "response.create"})
            await harness.wait_for("response.done", 2)
            self.frontend.queue(delegate_response("Again."))
            await harness.idle()
            await harness.speak()
            await harness.wait_for("response.done", 3)
            await harness.close()
            text = path.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines()]
        kinds = {record["kind"] for record in records}
        self.assertLessEqual({"transcript_normalized", "argument_normalized", "call_answered_locally"}, kinds)
        self.assertNotIn("quux", text.lower())
        local = next(record for record in records if record["kind"] == "call_answered_locally")
        self.assertEqual(
            (local["tool"], local["argument"], local["reason"]), ("get_user_details", "user_id", "already_failed")
        )
        start = next(record for record in records if record["kind"] == "session_start")
        self.assertEqual(
            start["normalization"],
            {"transcript": True, "tool_arguments": ["get_user_details.user_id"], "retry_guard": True},
        )

    async def asyncTearDown(self) -> None:
        pass  # closed inside the test, before the file is read


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = _env()
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def load(self, body: str) -> Any:
        path = Path(self._tmp.name) / "profile.yaml"
        path.write_text(f"extends: {PACKAGE_DIR / 'config' / 'voice_agent.yaml'}\n{body}", encoding="utf-8")
        return load_voice_config(path)

    def test_unknown_ruleset(self) -> None:
        with self.assertRaisesRegex(VoiceConfigError, "unknown ruleset"):
            self.load("normalization: {transcript: {ruleset: xx}}\n")

    def test_tool_arguments_need_client_tools(self) -> None:
        body = (
            "tools: {source: config, config_tools: 'examples.frontend_backend_verdict.text.demo_tools:TOOLS'}\n"
            "normalization: {tool_arguments: {enabled: true, rules: [{tool: t, argument: a}]}}\n"
        )
        with self.assertRaisesRegex(VoiceConfigError, "requires tools.source: client"):
            self.load(body)

    def test_retry_guard_needs_a_pattern(self) -> None:
        body = "normalization: {tool_arguments: {enabled: true, retry_guard: {enabled: true, scope: all}}}\n"
        with self.assertRaisesRegex(VoiceConfigError, "permanent_failure_pattern is required"):
            self.load(body)

    def test_bad_rules(self) -> None:
        for rule, message in (
            ("{tool: t}", "argument is required"),
            ("{tool: t, argument: a, colour: red}", "unknown keys"),
            ("{tool: t, argument: a, pattern: '('}", "invalid regular expression"),
            ("{tool: t, argument: a, case: title}", "case must be one of"),
        ):
            with self.subTest(rule=rule), self.assertRaisesRegex(VoiceConfigError, message):
                self.load(f"normalization: {{tool_arguments: {{enabled: true, rules: [{rule}]}}}}\n")

    def test_missing_prompt_key(self) -> None:
        with self.assertRaisesRegex(VoiceConfigError, "not in the catalog"):
            self.load("normalization: {transcript: {enabled: true, frontend_note_key: nope}}\n")

    def test_non_english_asr_warns(self) -> None:
        body = (
            "asr: {source: inline, inline: {server: 'localhost:50051', language_code: de-DE}}\n"
            "normalization: {transcript: {enabled: true}}\n"
        )
        config = self.load(body)
        self.assertTrue(any("ruleset 'en'" in warning for warning in config.warnings))

    def test_normalization_profile_equals_tau3_eval(self) -> None:
        # Normalization is on in tau3_eval.yaml itself; the normalization profile is kept as an alias.
        base = load_voice_config(PROFILES / "tau3_eval.yaml")
        for field in fields(base):
            if field.name not in ("source_files", "resolved_in_paths", "warnings"):
                self.assertEqual(getattr(base, field.name), getattr(PROFILE, field.name), field.name)
        self.assertTrue(base.normalization.transcript.enabled)
        self.assertTrue(base.normalization.tool_arguments.enabled)
        self.assertTrue(base.normalization.tool_arguments.retry_guard.enabled)

    def test_normalization_on_in_every_tau3_arm_and_off_in_the_base(self) -> None:
        for name in (
            "backend_only.yaml",
            "tau3_eval_backend_history.yaml",
            "tau3_eval_backend_history_noguide.yaml",
            "tau3_eval_frontend_verdict.yaml",
            "tau3_eval_frontend_verdict_speak.yaml",
        ):
            with self.subTest(profile=name):
                config = load_voice_config(PROFILES / name)
                self.assertTrue(config.normalization.transcript.enabled)
                self.assertEqual(config.normalization.transcript.case, "lower")
                self.assertTrue(config.normalization.tool_arguments.enabled)
                self.assertEqual(
                    [(r.tool, r.argument) for r in config.normalization.tool_arguments.rules],
                    [("get_user_details", "user_id")],
                )
        base = load_voice_config(PROFILES.parent / "voice_agent.yaml")
        self.assertFalse(base.normalization.transcript.enabled)
        self.assertFalse(base.normalization.tool_arguments.enabled)


class ReplayCliTests(unittest.TestCase):
    def test_replay_rewrites_screens_and_guards_conservatively(self) -> None:
        def calls(call_id: str, user_id: str) -> dict[str, Any]:
            return {
                "kind": "backend_tool_calls",
                "calls": [{"id": call_id, "name": "get_user_details", "arguments": {"user_id": user_id}}],
            }

        records = [
            {"kind": "asr_final", "transcript": SPOKEN},
            calls("c1", "Jordan_Lee_8264"),
            {"kind": "tool_output_in", "call_id": "c1", "output": "Error: User Jordan_Lee_8264 not found"},
            calls("c2", "jordan_lee_8264"),  # the recorded failure was for another string: not blocked
            {"kind": "tool_output_in", "call_id": "c2", "output": NOT_FOUND},
            calls("c3", "JORDAN_LEE_8264"),  # same canonical arguments as c2, which failed: blocked
            calls("c4", "YA"),
        ]
        out = StringIO()
        counts = replay(PROFILE, {"s1": records}, out)
        rows = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(rows[0]["normalized"], WRITTEN)
        self.assertEqual([row["decision"] for row in rows[1:]], ["sent", "sent", "already_failed", "invalid"])
        self.assertEqual(rows[1]["canonical"], {"user_id": "jordan_lee_8264"})
        self.assertEqual((counts["asr_rewritten"], counts["calls_rewritten"]), (1, 3))
