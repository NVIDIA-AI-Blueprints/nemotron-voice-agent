# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D100, D101, D102, D103, D107

"""Policy follow-up verification (``llm.verify_policy_follow_up_steps``, ``core/backends/llm/follow_up.py``).

The rule extraction and matching on invented, cross-domain policies and tool lists, the per-session
tracker, the agent loop against a scripted OpenAI-compatible server, the settings and the gateway files.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from _fda_backend_fakes import Harness
from _fda_llm_server import Reply, ScriptedOpenAI, tool

from examples.frontend_delegation_agent.core.backends.llm import HARNESS_KEYS
from examples.frontend_delegation_agent.core.backends.llm.follow_up import (
    MAX_FIRES,
    FollowUpRule,
    FollowUpTracker,
    build_rules,
    extract_candidates,
    stem,
    stems_equal,
    succeeded,
    tool_tokens,
    write_tools,
)
from examples.frontend_delegation_agent.core.backends.llm.host import LLMAgentHost
from examples.frontend_delegation_agent.core.backends.llm.settings import LLMSettings, worker_prompts
from examples.frontend_delegation_agent.core.backends.llm.tools import model_tools
from examples.frontend_delegation_agent.core.sidecar.gateway_config import CONFIG_DIR, load_gateway_config
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates
from examples.frontend_delegation_agent.core.worker.worker_core import WorkerCore

ROOT = Path(__file__).resolve().parents[4]
PROMPTS = ROOT / "src" / "examples" / "frontend_delegation_agent" / "config" / "prompts"
TEMPLATES = BackendTemplates(PROMPTS / "customer_service" / "backend" / "llm.yaml", required_keys=HARNESS_KEYS)


def tools(*names: str, reads: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Tool dicts; names in ``reads`` get a description that opens with "Look up" (a read)."""
    return [
        {
            "name": name,
            "description": "Look up records." if name in reads else "Change a record.",
            "parameters": {"type": "object", "properties": {}},
        }
        for name in names
    ]


CLINIC_POLICY = """\
# Appointments
- After rescheduling an appointment, always move the lab order and the reminder to the new appointment.
  Always ask the patient which reminder channel they prefer before setting one (do not assume).
- Confirm the date with the patient before any change.
"""
CLINIC_TOOLS = tools(
    "find_appointment_slots",
    "reschedule_appointment",
    "move_lab_order",
    "set_reminder",
    reads=("find_appointment_slots",),
)


def opened(policy: str, tool_list: list[dict[str, Any]], **kwargs: Any) -> dict[str, tuple[str, ...]]:
    return {rule.trigger: rule.tools for rule in build_rules(policy, tool_list, **kwargs)}


class WordsTest(unittest.TestCase):
    def test_stem(self) -> None:
        cases = {
            "rescheduling": "reschedul",
            "reschedule": "reschedul",
            "opened": "open",
            "case": "case",
            "shipped": "ship",
            "resetting": "reset",
            "called": "call",
            "billed": "bill",
            "passed": "pass",
            "cancelling": "cancell",
            "the": "",
            "add": "",
            "user": "",
        }
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(stem(word), expected)

    def test_prefix_equality(self) -> None:
        self.assertTrue(stems_equal("cancel", "cancellation"))
        self.assertTrue(stems_equal("cancell", "cancel"))
        self.assertFalse(stems_equal("add", "address"))  # shorter than four characters
        self.assertFalse(stems_equal("move", "moving"))

    def test_tool_tokens(self) -> None:
        for name in ("crm.update_case", "crm__update_case", "updateCase", "update-case", "tools/update_case"):
            with self.subTest(name=name):
                self.assertEqual(tool_tokens(name), ("update", "case"))

    def test_write_tools(self) -> None:
        tool_list = [
            {"name": "getOrder", "description": "Fetch."},
            {"name": "crm.get_case", "description": "Fetch."},
            {"name": "fetch_invoice", "description": "Fetch.", "annotations": {"readOnlyHint": True}},
            {"name": "crm.update_case", "description": "Change it."},
            {"name": "issue_refund", "description": "Look up and refund."},  # a read by its description
        ]
        self.assertEqual(write_tools(tool_list), ("crm.update_case",))

    def test_succeeded(self) -> None:
        self.assertTrue(succeeded('{"status": "moved"}'))
        self.assertTrue(succeeded("Done."))
        self.assertFalse(succeeded("Error: not found"))
        self.assertFalse(succeeded('{"error": "nope"}'))
        self.assertFalse(succeeded('{"success": false}'))
        self.assertFalse(succeeded('{"ok": false, "reason": "locked"}'))


class ExtractionTest(unittest.TestCase):
    def test_rule_keeps_its_line_and_the_list_items_below(self) -> None:
        policy = (
            "Billing:\nAfter closing an account, always do the following:\n- refund the balance\n- cancel the card\n\n"
            "Other."
        )
        [candidate] = extract_candidates(policy)
        self.assertEqual(candidate.trigger, "closing an account")
        self.assertEqual(
            candidate.text,
            "After closing an account, always do the following:\n- refund the balance\n- cancel the card",
        )

    def test_an_obligation_needs_a_marker(self) -> None:
        self.assertEqual(extract_candidates("After you resume the service, the customer will have to restart."), [])
        self.assertEqual(extract_candidates("After confirmation, the status will be changed to paid."), [])

    def test_generic_triggers_give_no_rule(self) -> None:
        for sentence in (
            "After completing any change, all notices must be sent.",
            "After making any update, always log it.",
            "After you complete any change to it, you must notify the owner.",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(extract_candidates(sentence), [])
        [kept] = extract_candidates("After any change to a reservation, always email the confirmation.")
        self.assertIn("reservation", kept.stems)

    def test_trigger_ends_at_a_marker_or_a_subject(self) -> None:
        [no_comma] = extract_candidates("After cancelling an order always restock the items.")
        self.assertEqual(no_comma.trigger, "cancelling an order")
        [subject] = extract_candidates("After a dependent is added you must notify the plan administrator.")
        self.assertEqual(subject.trigger, "a dependent is added")

    def test_after_x_colon_with_a_bullet_gives_no_rule(self) -> None:
        self.assertEqual(extract_candidates("After privileges are restored:\n- the owner must be notified"), [])

    def test_long_text_is_capped(self) -> None:
        [candidate] = extract_candidates("After closing an account, always " + "x " * 400)
        self.assertLessEqual(len(candidate.text), 600)


class MatchingTest(unittest.TestCase):
    def test_trigger_opens_its_write_tool_only(self) -> None:
        self.assertEqual(
            opened(CLINIC_POLICY, CLINIC_TOOLS), {"rescheduling an appointment": ("reschedule_appointment",)}
        )

    def test_a_read_never_opens_a_rule(self) -> None:
        policy = "After rescheduling, always move the lab order."
        tool_list = tools("find_rescheduling_options", "reschedule_appointment", reads=("find_rescheduling_options",))
        self.assertEqual(opened(policy, tool_list), {"rescheduling": ("reschedule_appointment",)})

    def test_a_shared_noun_needs_a_unique_owner(self) -> None:
        policy = "After a ticket is created, always assign an owner."
        tool_list = tools("create_ticket", "create_user", "close_ticket", "update_ticket")
        self.assertEqual(opened(policy, tool_list), {"a ticket is created": ("create_ticket",)})

    def test_the_follow_up_tool_never_opens_its_own_rule(self) -> None:
        self.assertEqual(opened("After updating a case, always add a note.", tools("add_case_note")), {})
        self.assertEqual(
            opened("After a dependent is added, you must add a note.", tools("add_dependent", "add_case_note")),
            {"a dependent is added": ("add_dependent",)},
        )
        self.assertEqual(opened("After verification, you must call the logging tool.", tools("log_verification")), {})

    def test_stem_forms(self) -> None:
        cases = [
            ("After cancelling an order, always restock the items.", "cancel_order"),
            ("After a cancellation, always restock the items.", "cancel"),
            ("After a patient is called back, always note the time.", "call_patient"),
            ("After a customer is billed, always email the invoice.", "bill_customer"),
        ]
        for policy, name in cases:
            with self.subTest(policy=policy):
                self.assertEqual(list(opened(policy, tools(name)).values()), [(name,)])

    def test_namespaced_and_camel_case_tools(self) -> None:
        policy = "After rescheduling an appointment, always move the lab order."
        for name in ("clinic.reschedule_appointment", "clinic__reschedule_appointment", "rescheduleAppointment"):
            with self.subTest(name=name):
                self.assertEqual(list(opened(policy, tools(name)).values()), [(name,)])

    def test_ignore_list_and_declared_rules(self) -> None:
        self.assertEqual(opened(CLINIC_POLICY, CLINIC_TOOLS, ignore_tools=["reschedule_appointment"]), {})
        declared = [
            {"tools": ["execute"], "text": "Nach jeder Ausführung immer das Protokoll speichern."},
            {"tools": ["move_lab_order"], "text": "Notify the lab.", "domain": "lab"},
        ]
        rules = build_rules("", tools("execute", "move_lab_order"), declared=declared, domain="")
        self.assertEqual([(r.tools, r.source) for r in rules], [(("execute",), "declared")])
        rules = build_rules("", tools("execute", "move_lab_order"), declared=declared, domain="lab")
        self.assertEqual([r.tools for r in rules], [("execute",), ("move_lab_order",)])

    def test_cap_counts_matching_rules_only(self) -> None:
        descriptive = "\n".join(f"After step {i} finishes, the supervisor must review the queue." for i in range(6))
        policy = descriptive + "\nAfter rescheduling an appointment, always move the lab order."
        self.assertEqual(list(opened(policy, CLINIC_TOOLS)), ["rescheduling an appointment"])

    def test_identical_texts_are_kept_once(self) -> None:
        policy = "After rescheduling an appointment, always move the lab order. After rescheduling, always confirm."
        self.assertEqual(len(build_rules(policy, CLINIC_TOOLS)), 1)

    def test_no_policy_no_rules(self) -> None:
        self.assertEqual(build_rules("", CLINIC_TOOLS), ())
        self.assertEqual(build_rules("Be brief and polite.", CLINIC_TOOLS), ())


class TrackerTest(unittest.TestCase):
    RULE = FollowUpRule("rescheduling", "After rescheduling, always move the lab order.", ("reschedule_appointment",))
    WRITES = ("reschedule_appointment", "move_lab_order", "set_reminder")

    def tracker(self) -> FollowUpTracker:
        tracker = FollowUpTracker([self.RULE], self.WRITES)
        tracker.start_run()
        return tracker

    def test_a_matching_write_makes_a_check_due_once(self) -> None:
        tracker = self.tracker()
        self.assertFalse(tracker.due())
        tracker.record("reschedule_appointment", {"id": "A1"}, '{"status": "moved"}')
        self.assertTrue(tracker.due())
        check = tracker.fire()
        self.assertEqual(check.changes, ('reschedule_appointment({"id":"A1"})',))
        self.assertFalse(tracker.due())
        tracker.finish("Done.", completed=True)
        self.assertEqual(tracker.open_count(), 0)

    def test_failed_reads_and_other_writes_open_nothing(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "Error: slot taken")
        tracker.record("find_appointment_slots", {}, "[]")
        tracker.record("set_reminder", {}, "ok")
        self.assertFalse(tracker.due())

    def test_a_question_keeps_the_rule_for_one_more_check(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "ok")
        tracker.fire()
        tracker.finish("Which reminder channel do you prefer?", completed=True)
        self.assertEqual(tracker.open_count(), 1)
        tracker.start_run()  # the caller answers; this run makes a write
        self.assertFalse(tracker.due())
        tracker.record("set_reminder", {"channel": "sms"}, "ok")
        self.assertTrue(tracker.due())
        tracker.fire()
        tracker.finish("Anything else?", completed=True)
        self.assertEqual(tracker.open_count(), 0)  # checked twice: closed

    def test_a_kept_rule_closes_on_a_run_without_a_write(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "ok")
        tracker.fire()
        tracker.finish("Which channel?", completed=True)
        tracker.start_run()
        self.assertFalse(tracker.due())
        tracker.finish("Goodbye.", completed=True)
        self.assertEqual(tracker.open_count(), 0)

    def test_a_write_after_the_check_closes_even_on_a_question(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "ok")
        tracker.fire()
        tracker.record("move_lab_order", {}, "ok")
        tracker.finish("Done. Anything else?", completed=True)
        self.assertEqual(tracker.open_count(), 0)

    def test_an_interrupted_run_checks_at_the_end_of_the_next_run(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "ok")
        tracker.finish(None, completed=False)
        tracker.start_run()
        self.assertTrue(tracker.due())  # no new write needed
        self.assertEqual(tracker.fire().changes, ("reschedule_appointment({})",))

    def test_at_most_max_fires_checks(self) -> None:
        tracker = self.tracker()
        tracker.record("reschedule_appointment", {}, "ok")
        for _ in range(MAX_FIRES):
            tracker.fire()
            tracker.finish(None, completed=False)
            tracker.start_run()
        self.assertFalse(tracker.due())
        self.assertEqual(tracker.open_count(), 0)


class Bridge:
    def __init__(self, results: dict[str, str] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, args: dict[str, Any], task_id: str) -> str:
        self.calls.append((name, args))
        return self.results.get(name, '{"status": "ok"}')


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ScriptedOpenAI().start()
        self.addCleanup(self.server.stop)
        patcher = mock.patch.dict(os.environ, {"FDA_FAKE_KEY": "sk-fake"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def agent(self, *, policy: str = CLINIC_POLICY, bridge: Bridge | None = None, **settings: Any) -> Any:
        self.bridge = bridge or Bridge()
        host = LLMAgentHost(self.bridge, lambda *_: None, session_id="s1")
        host.set_event_sink(lambda event, fields: self.events.append((event, fields)))
        host.configure(
            tools=CLINIC_TOOLS,
            instructions="RENDERED BACKEND PROMPT",
            settings={
                "model": "fake-model",
                "base_url": self.server.base_url,
                "api_key_env": "FDA_FAKE_KEY",
                "max_iterations": 30,
                "request_overrides": {},
                "reasoning_effort": "medium",
                "request_timeout_s": 5.0,
                "max_retries": 3,
                "client_instructions": policy,
                **worker_prompts(TEMPLATES),
                **settings,
            },
        )
        return host.agent()

    def test_matching_write_sends_one_check_before_the_answer(self) -> None:
        self.server.queue(
            Reply(tool_calls=[tool("reschedule_appointment", id="A1")]),
            Reply(content="Your appointment is moved."),
            Reply(tool_calls=[tool("move_lab_order", id="A1")]),
            Reply(content="Moved, and the lab order moved with it."),
        )
        result = self.agent().run_conversation("move my appointment", [], "s1:1")
        self.assertEqual(result["final_response"], "Moved, and the lab order moved with it.")
        self.assertEqual([c[0] for c in self.bridge.calls], ["reschedule_appointment", "move_lab_order"])
        rows = result["messages"]
        self.assertEqual(rows[3], {"role": "assistant", "content": "[draft answer withheld for a policy check]"})
        self.assertTrue(rows[4]["content"].startswith("[System check before you answer. Not from the user.]"))
        self.assertIn('reschedule_appointment({"id":"A1"})', rows[4]["content"])
        self.assertIn("- After rescheduling an appointment, always move the lab order", rows[4]["content"])
        self.assertNotIn("Your appointment is moved.", str(rows))  # the withheld draft is not kept
        names = [event for event, _ in self.events]
        self.assertEqual(names, ["follow_up_obligations", "follow_up_check"])
        self.assertEqual(self.events[1][1]["triggers"], ["rescheduling an appointment"])

    def test_empty_reply_to_the_check_returns_the_draft(self) -> None:
        self.server.queue(
            Reply(tool_calls=[tool("reschedule_appointment")]),
            Reply(content="Your appointment is moved."),
            Reply(content=""),
        )
        result = self.agent().run_conversation("move it", [], "s1:1")
        self.assertEqual(result["final_response"], "Your appointment is moved.")

    def test_failed_write_read_or_unrelated_write_sends_no_check(self) -> None:
        for name, result in (
            ("reschedule_appointment", "Error: slot taken"),
            ("find_appointment_slots", "[]"),
            ("set_reminder", "ok"),
        ):
            with self.subTest(name=name):
                self.server.queue(Reply(tool_calls=[tool(name)]), Reply(content="Done."))
                before = len(self.server.requests)
                outcome = self.agent(bridge=Bridge({name: result})).run_conversation("q", [], "s1:1")
                self.assertEqual(outcome["final_response"], "Done.")
                self.assertEqual(len(self.server.requests) - before, 2)

    def test_no_rule_sends_the_same_requests_as_switched_off(self) -> None:
        wires = []
        for enabled in (True, False):
            self.server.requests.clear()
            self.server.queue(Reply(tool_calls=[tool("reschedule_appointment")]), Reply(content="Moved."))
            agent = self.agent(policy="Be brief.", verify_policy_follow_up_steps=enabled)
            agent.run_conversation("move it", [], "s1:1")
            wires.append([request["messages"] for request in self.server.requests])
        self.assertEqual(self.without_call_ids(wires[0]), self.without_call_ids(wires[1]))
        self.assertEqual(self.events, [])  # no rule: no event at all

    @staticmethod
    def without_call_ids(wire: Any) -> str:
        """The requests with the scripted server's per-call ids (time-based) blanked."""
        return re.sub(r'"(?:id|tool_call_id)": "call_[^"]*"', '"id": ""', json.dumps(wire))

    def test_switched_off_sends_no_check(self) -> None:
        self.server.queue(Reply(tool_calls=[tool("reschedule_appointment")]), Reply(content="Moved."))
        result = self.agent(verify_policy_follow_up_steps=False).run_conversation("move it", [], "s1:1")
        self.assertEqual(result["final_response"], "Moved.")
        self.assertEqual(self.events, [])

    def test_rules_come_from_the_client_policy_only(self) -> None:
        self.agent(policy="")  # the rendered prompt ("instructions") is never read
        self.assertEqual(self.events, [])

    def test_annotations_never_reach_the_model(self) -> None:
        annotated = [{**t, "annotations": {"readOnlyHint": True}} for t in CLINIC_TOOLS]
        self.assertEqual(model_tools(annotated), model_tools(CLINIC_TOOLS))


class WiringTest(unittest.IsolatedAsyncioTestCase):
    async def test_llm_configure_carries_the_client_policy_and_annotations(self) -> None:
        h = Harness(backend_kind="llm")
        await h.controller.open({"update_after_start": "error"})
        annotated = {"name": "get_order", "description": "Look up.", "annotations": {"readOnlyHint": True}}
        await h.controller.configure([annotated], "POLICY")
        configure = h.worker.of_type("configure")[0]
        self.assertEqual(configure["backend"]["client_instructions"], "POLICY")
        self.assertIn("session_domain", configure["backend"])
        self.assertNotEqual(configure["instructions"], "POLICY")  # the rendered prompt is separate
        self.assertEqual(configure["tools"][0]["annotations"], {"readOnlyHint": True})

    async def test_other_backends_do_not_get_the_client_policy(self) -> None:
        h = Harness(backend_kind="hermes")
        await h.start()
        self.assertNotIn("client_instructions", h.worker.of_type("configure")[0]["backend"])

    async def test_event_messages_go_to_the_event_log_only(self) -> None:
        h = Harness(backend_kind="llm")
        await h.start()
        emitted = len(h.emitted)
        h.worker.reply({"type": "event", "event": "follow_up_check", "session_id": "s1", "triggers": ["x"]})
        self.assertIn(("follow_up_check", {"session_id": "s1", "triggers": ["x"]}), h.logs)
        self.assertEqual(len(h.emitted), emitted)  # nothing reaches the voice server

    async def test_worker_core_forwards_host_events(self) -> None:
        sent: list[dict[str, Any]] = []

        class Host:
            built = False

            def __init__(self, futures: Any, on_activity: Any) -> None:
                self.sink: Any = None

            def set_event_sink(self, sink: Any) -> None:
                self.sink = sink

        core = WorkerCore(session_id="s1", host_factory=Host, emit=sent.append)
        core.host.sink("follow_up_obligations", {"rules": []})
        self.assertEqual(sent, [{"type": "event", "event": "follow_up_obligations", "session_id": "s1", "rules": []}])


class SettingsTest(unittest.TestCase):
    def parse(self, **values: Any) -> LLMSettings:
        merged = {**LLMSettings.DEFAULTS, **values}
        return LLMSettings.parse(lambda key, kind: kind(merged[key]))

    def test_defaults(self) -> None:
        settings = self.parse()
        self.assertTrue(settings.verify_policy_follow_up_steps)
        self.assertEqual(settings.worker_settings()["follow_up_rules"], [])
        self.assertNotIn("follow_up_rules", settings.summary())

    def test_rules_are_validated(self) -> None:
        good = self.parse(follow_up_rules=[{"tools": ["execute"], "text": "Save the log.", "domain": "ops"}])
        self.assertEqual(good.follow_up_rules, ({"tools": ["execute"], "text": "Save the log.", "domain": "ops"},))
        for bad in ([{"tools": [], "text": "x"}], [{"tools": ["a"]}], [{"tools": ["a"], "text": "x", "other": 1}]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.parse(follow_up_rules=bad)
        with self.assertRaises(ValueError):
            self.parse(follow_up_ignore_tools=[""])

    def test_gateway_files_enable_it_and_the_variable_turns_it_off(self) -> None:
        for name in ("gateway.yaml", "gateway.eval.yaml"):
            for value, expected in ((None, True), ("false", False)):
                env = {"FDA_BACKEND": "llm"}
                if value is not None:
                    env["FDA_VERIFY_FOLLOW_UP_STEPS"] = value
                with self.subTest(file=name, value=value), mock.patch.dict(os.environ, env):
                    if value is None:
                        os.environ.pop("FDA_VERIFY_FOLLOW_UP_STEPS", None)
                    settings = load_gateway_config(CONFIG_DIR / name).backend_settings
                    self.assertIsNotNone(settings)
                    self.assertEqual(settings.verify_policy_follow_up_steps, expected)


class NamingTest(unittest.TestCase):
    """No benchmark or evaluation-suite name in the example's code, prompts, config or tests (.py, .yaml)."""

    # Built from reversed spellings and a code point, so this file does not match itself.
    NAMES = tuple(name[::-1] for name in ("uat", "2uat", "ave"))
    PATTERN = re.compile(r"\b(?:" + "|".join(NAMES) + r")\b|" + chr(0x3C4), re.IGNORECASE)

    def test_no_benchmark_names(self) -> None:
        roots = (
            ROOT / "src" / "examples" / "frontend_delegation_agent",
            ROOT / "tests" / "unit" / "frontend_delegation_agent",
        )
        hits = [
            f"{path.relative_to(ROOT)}:{number}"
            for root in roots
            for path in sorted(root.rglob("*"))
            if path.suffix in (".py", ".yaml") and path.is_file()
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if self.PATTERN.search(line)
        ]
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
