# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Failure endings carry official codes and reasons, and the core never words an apology."""

import asyncio
import re
from pathlib import Path

from delegation_core.backend.port import Start
from tests.unit.delegation_core.helpers import (
    IMCase,
    ScriptedAdapter,
    ScriptStep,
    answer,
    ask,
    call_item,
    cut,
    empty,
    fail,
    failed_attempt,
    finished_items,
    own_call,
    planned,
    read_all,
    wait,
)

CORE = Path(__file__).resolve().parents[3] / "src" / "delegation_core"


class RateLimited(Exception):
    status_code = 429


class FailedWithCode(ScriptStep):
    """An attempt the backend ended failed with its own official code, then the run gives up."""

    async def play(self, run):
        error = {"error": {"code": "vector_store_timeout", "message": "The store timed out."}}
        for event in run.round_events([], terminal=("failed", error)):
            yield event
        raise RuntimeError("The backend gave up")


class FailureTests(IMCase):
    async def ending(self, *script, **kwargs) -> dict:
        im = self.session(ScriptedAdapter(list(script)), **kwargs)
        self.im = im
        events = await read_all(im.admit(ask("Find teas."), delegation_id="D1").stream)
        return events[-1]

    async def test_a_raising_run_fails_with_server_error_and_the_detail(self):
        terminal = await self.ending(fail(RuntimeError("provider exploded")))
        self.assertEqual(terminal["type"], "response.failed")
        self.assertEqual(terminal["response"]["error"]["code"], "server_error")
        self.assertIn("provider exploded", terminal["response"]["error"]["message"])
        self.assertEqual(self.im.ledger.delegations["D1"].state, "failed")

    async def test_an_upstream_429_fails_with_rate_limit_exceeded(self):
        terminal = await self.ending(fail(RateLimited("slow down")))
        self.assertEqual(terminal["response"]["error"]["code"], "rate_limit_exceeded")

    async def test_an_upstream_429_after_a_failed_attempt_still_fails_with_rate_limit_exceeded(self):
        # The backend LLM's run closes a failed attempt with its own server_error terminal, then retries; a 429 that
        # ends the retry is what failed the task.
        terminal = await self.ending(failed_attempt("server_error"), fail(RateLimited("slow down")))
        self.assertEqual(terminal["response"]["error"]["code"], "rate_limit_exceeded")

    async def test_the_backends_own_code_is_kept(self):
        terminal = await self.ending(FailedWithCode())
        self.assertEqual(terminal["response"]["error"]["code"], "vector_store_timeout")

    async def test_a_cut_round_ends_incomplete_with_its_official_reason(self):
        for reason in ("max_output_tokens", "content_filter"):
            with self.subTest(reason=reason):
                terminal = await self.ending(cut(reason))
                self.assertEqual(terminal["type"], "response.incomplete")
                self.assertEqual(terminal["response"]["incomplete_details"], {"reason": reason})
                self.assertEqual(terminal["response"]["status"], "incomplete")

    async def test_an_empty_answer_fails_and_is_unclear_once_something_acted(self):
        terminal = await self.ending(own_call("book_table"), empty())
        self.assertEqual(terminal["type"], "response.failed")
        self.assertEqual(terminal["response"]["error"]["code"], "server_error")
        work = self.im.ledger.works["W1"]
        self.assertEqual((work.state, self.im.ledger.tasks["W1.1"].outcome), ("failed", "unclear"))

    async def test_an_empty_answer_before_acting_changed_nothing(self):
        await self.ending(empty())
        self.assertEqual(self.im.ledger.tasks["W1.1"].outcome, "none")

    async def test_a_failure_closes_an_open_mcp_call_incomplete(self):
        class Dies(ScriptStep):
            async def play(self, run):
                call = call_item("lookup", {}, "call_x")
                for event in run.round_events([call]):
                    yield event
                run.task.route(call)
                run.task.tool_started(call, "cafe")
                raise RuntimeError("the tool's host died")

        terminal = await self.ending(Dies())
        (item,) = terminal["response"]["output"]
        self.assertEqual((item["type"], item["status"]), ("mcp_call", "incomplete"))
        self.assertEqual(self.im.ledger.actions["call_x"].state, "unclear")

    async def test_one_task_failing_beside_an_answer_ends_the_response_failed_after_the_answer(self):
        gate = asyncio.Event()

        async def split(given, place):
            await place(Start(("D1",), ("teas", "hours")))

        adapter = ScriptedAdapter([answer("Teas.")], [wait(gate), fail()], conductor=planned(split), max_running=2)
        im = self.session(adapter)
        stream = im.admit(ask("Teas and hours."), delegation_id="D1").stream
        await self.turns()
        gate.set()
        events = await read_all(stream)
        self.assertEqual([item["content"][0]["text"] for item in finished_items(events)], ["Teas."])
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_the_core_words_no_apology(self):
        for path in CORE.rglob("*.py"):
            with self.subTest(path=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIsNone(re.search(r"\bsorry\b", source, re.IGNORECASE))
                self.assertNotIn("apology", source.lower() if path.parent.name == "im" else "")
