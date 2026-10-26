# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""What the LLM backend sends to the model (llm-backend-plan.md sections 5 and 6).

Checked against recorded sessions and backend traces: the system message bytes, the tool schemas the model saw
(per benchmark domain), the tool-name repairs and the wire shape of the history.
"""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from examples.frontend_delegation_agent.core.backends.llm import HARNESS_KEYS
from examples.frontend_delegation_agent.core.backends.llm import tools as rules
from examples.frontend_delegation_agent.core.backends.llm.client import Aborted, ChatClient
from examples.frontend_delegation_agent.core.backends.llm.host import conversation_started, system_message
from examples.frontend_delegation_agent.core.backends.llm.wire import build, strip_think
from examples.frontend_delegation_agent.core.prompt_features import tools_canonical_sha256
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates

HERE = Path(__file__).resolve().parent
SESSION_UPDATES = HERE.parent / "voice" / "fixtures" / "benchmark_session_update"
PROMPTS = HERE.parents[3] / "src" / "examples" / "frontend_delegation_agent" / "config" / "prompts"
#: tools_canonical_sha256 of the tools in the recorded backend traces (benchmark runs, 2026-10-16).
RECORDED_TOOL_HASHES = {"airline": "2fdd5b549d9d15b4", "retail": "300feeeb65e4de4c", "telecom": "e6dc47d93703b063"}


def _session_tools(domain: str) -> list[dict]:
    tools = json.loads((SESSION_UPDATES / f"{domain}.json").read_text())["session"]["tools"]
    # What the controller sends in ``configure``.
    return [
        {"name": t["name"], "description": t.get("description") or "", "parameters": t.get("parameters") or {}}
        for t in tools
    ]


def _templates(profile: str = "customer_service") -> BackendTemplates:
    return BackendTemplates(PROMPTS / profile / "backend" / "llm.yaml", required_keys=HARNESS_KEYS)


class SystemMessageTest(unittest.TestCase):
    def test_matches_the_recorded_system_message(self) -> None:
        recorded = json.loads((HERE / "fixtures" / "llm_system_prompt.json").read_text())
        harness = _templates().source("harness_system")
        import jinja2

        rendered = jinja2.Template(harness, undefined=jinja2.StrictUndefined).render(
            started=recorded["started"], model=recorded["model"]
        )
        self.assertEqual(system_message(recorded["soul"], rendered, recorded["instructions"]), recorded["expected"])

    def test_every_profile_has_the_harness_text(self) -> None:
        for profile in ("customer_service", "generic", "general_qa"):
            with self.subTest(profile=profile):
                templates = _templates(profile)
                for key in HARNESS_KEYS:
                    self.assertTrue(templates.source(key))
                hermes = BackendTemplates(PROMPTS / profile / "backend" / "hermes.yaml")
                self.assertEqual(templates.render("backend_soul"), hermes.render("backend_soul"))

    def test_start_date_format(self) -> None:
        moment = datetime(2026, 10, 4, 9, 0, tzinfo=UTC)
        with mock.patch.dict("os.environ", {"TZ": "UTC"}):
            self.assertEqual(conversation_started(moment), "Sunday, October 04, 2026 (UTC, UTC+00:00)")
        ist = datetime(2026, 10, 8, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30), "IST"))
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(conversation_started(ist), "Thursday, October 08, 2026 (IST, UTC+05:30)")

    def test_no_instructions_leaves_the_cached_part(self) -> None:
        self.assertEqual(system_message(" soul \n", "\nharness\n", "  "), "soul\n\nharness")


class RequestTest(unittest.TestCase):
    def test_tool_schemas_match_the_recorded_traces(self) -> None:
        for domain, expected in RECORDED_TOOL_HASHES.items():
            with self.subTest(domain=domain):
                self.assertEqual(tools_canonical_sha256(rules.model_tools(_session_tools(domain))), expected)

    def test_empty_description_becomes_the_name(self) -> None:
        (tool,) = rules.model_tools([{"name": "ping", "description": "", "parameters": {}}])
        self.assertEqual(tool["function"]["description"], "ping")
        # No parameters: an empty object schema.
        self.assertEqual(tool["function"]["parameters"], {"type": "object", "properties": {}, "required": []})

    def test_request_body(self) -> None:
        client = ChatClient(
            base_url="http://x/v1",
            api_key="k",
            model="m",
            timeout_s=5,
            max_retries=3,
            reasoning_effort="medium",
            request_overrides={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024}
            },
        )
        body = client.body([{"role": "user", "content": "hi"}], [{"type": "function"}])
        self.assertEqual(body["reasoning_effort"], "medium")
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(
            body["extra_body"], {"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 1024}
        )
        for absent in ("temperature", "max_tokens", "tool_choice", "parallel_tool_calls"):
            self.assertNotIn(absent, body)
        self.assertNotIn("tools", client.body([], None))

    def test_a_pending_abort_survives_the_clear(self) -> None:
        client = ChatClient(
            base_url="http://x/v1",
            api_key="k",
            model="m",
            timeout_s=5,
            max_retries=3,
            reasoning_effort="",
            request_overrides={},
        )
        attempts: list[int] = []
        with self.assertRaises(Aborted):
            client.complete([], None, on_attempt=attempts.append, should_abort=lambda: True)
        self.assertEqual(attempts, [])  # no request was started


class ToolRulesTest(unittest.TestCase):
    TELECOM = [t["name"] for t in _session_tools("telecom")]

    def test_name_repair_matches_the_traces(self) -> None:
        self.assertEqual(rules.repair_name("toggle_roaming", self.TELECOM), "enable_roaming")
        for invented in ("run_speed_test", "toggle_data_saver_mode", "check_wifi_status"):
            with self.subTest(name=invented):
                self.assertIsNone(rules.repair_name(invented, self.TELECOM))

    def test_name_repair_rules(self) -> None:
        cases = {
            "GetCustomerByPhone": "get_customer_by_phone",
            "get-customer-by-phone": "get_customer_by_phone",
            "Get Customer By Phone": "get_customer_by_phone",
            "get_customer_by_phone_tool": "get_customer_by_phone",
            'get_customer_by_phone" parameter="phone': "get_customer_by_phone",
            "": None,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(rules.repair_name(name, self.TELECOM), expected)

    def test_arguments(self) -> None:
        self.assertEqual(rules.parse_arguments(""), ({}, ""))
        self.assertEqual(rules.parse_arguments({"a": 1}), ({"a": 1}, ""))
        self.assertEqual(rules.parse_arguments('{"b": 2}'), ({"b": 2}, ""))
        self.assertIsNone(rules.parse_arguments("{bad")[0])
        self.assertIsNone(rules.parse_arguments("[1]")[0])
        self.assertEqual(rules.canonical_arguments({"b": 1, "a": "é"}), '{"a":"\\u00e9","b":1}')

    def test_failure_warning_thresholds(self) -> None:
        templates = _templates()
        warnings = rules.FailureWarnings(templates.render)
        error = '{"error": "not found"}'
        self.assertEqual(warnings.after("get_order", {"id": 1}, error), error)  # 1st exact failure
        second = warnings.after("get_order", {"id": 1}, error)
        self.assertIn(
            "[Tool loop warning: repeated_exact_failure_warning; count=2; get_order has failed 2 times", second
        )
        self.assertTrue(second.startswith(error + "\n\n"))
        warnings = rules.FailureWarnings(templates.render)
        warnings.after("get_order", {"id": 1}, "Error: x")
        warnings.after("get_order", {"id": 2}, "Error: x")
        third = warnings.after("get_order", {"id": 3}, "Error: x")
        self.assertIn("same_tool_failure_warning; count=3; get_order has failed 3 times this turn.", third)
        self.assertEqual(
            warnings.after("get_order", {"id": 4}, '{"status": "ok"}'), '{"status": "ok"}'
        )  # success resets
        self.assertEqual(warnings.after("get_order", {"id": 5}, "Error: x"), "Error: x")
        self.assertFalse(rules.is_failure("the order failed to ship"))  # only "error"/"failed" keys count


class WireTest(unittest.TestCase):
    def test_build(self) -> None:
        rows = [
            {"role": "user", "content": " hi "},
            {"role": "assistant", "content": ""},  # a redirect placeholder
            {"role": "user", "content": "correction"},
            {"role": "user", "content": "more"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c", "content": "ok", "name": "f"},
            {"role": "assistant", "content": "done", "reasoning": "secret"},
        ]
        wire = build("SYS", rows, "[response interrupted]")
        self.assertEqual(wire[0], {"role": "system", "content": "SYS"})
        self.assertEqual(wire[1], {"role": "user", "content": "hi"})
        self.assertEqual(wire[2], {"role": "assistant", "content": "[response interrupted]"})
        self.assertEqual(wire[3], {"role": "user", "content": "correction\n\nmore"})
        self.assertEqual(wire[4]["content"], "")
        self.assertEqual(wire[5], {"role": "tool", "tool_call_id": "c", "content": "ok"})
        self.assertEqual(wire[6], {"role": "assistant", "content": "done"})

    def test_strip_think(self) -> None:
        self.assertEqual(strip_think("<think>plan</think>Answer"), "Answer")
        self.assertEqual(strip_think("<think>never closed"), "")
        self.assertEqual(strip_think("leftover</think> Answer"), " Answer")


if __name__ == "__main__":
    unittest.main()
