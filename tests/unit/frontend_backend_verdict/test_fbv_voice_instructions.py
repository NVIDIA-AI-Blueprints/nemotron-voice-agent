# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Rule 13: exact prompts per tau2 domain, tool-derived capabilities, placements, the voice catalog.

Snapshots live in ``fixtures/prompt_snapshots``; regenerate them after a reviewed prompt
change with ``UPDATE_PROMPT_SNAPSHOTS=1 uv run pytest tests/unit/prototypes/voice/test_voice_instructions.py``.
"""

from __future__ import annotations

import json
import os
import re
import unittest
from dataclasses import replace
from pathlib import Path

import yaml
from _fbv_voice_fakes import (
    FIXTURES,
    PACKAGE_DIR,
    FakeChatClient,
    base_config,
    delegate_response,
    tau2_session_update,
    text_response,
)

from examples.frontend_backend_verdict.voice.agent.runner import AgentClients, TextAgentRunner
from examples.frontend_backend_verdict.voice.agent.sinks import SessionRoutingSink

DOMAINS = ("mock", "airline", "retail", "telecom")
IN_SCOPE_REQUEST = {
    "mock": "Please create a task called groceries for me.",
    "airline": "I need to change the date of my flight reservation.",
    "retail": "I want to return an item from my last order.",
    "telecom": "My mobile data stopped working this morning.",
}
SNAPSHOTS = FIXTURES / "prompt_snapshots"
PLACEHOLDERS = re.compile(r"\{(persona|domain_policy|capabilities|unsupported_reply|agent_name)\}")


def _runner(apply_to: tuple[str, ...] = ("backend",), placement: str = "policy_slot", **clients) -> TextAgentRunner:
    config = base_config()
    config = replace(config, instructions=replace(config.instructions, apply_to=apply_to, placement=placement))
    return TextAgentRunner(
        base_config=config.agent,
        tools_config=config.tools,
        instructions_config=config.instructions,
        clients=AgentClients(
            backend=clients.get("backend") or FakeChatClient(), frontend=clients.get("frontend") or FakeChatClient()
        ),
        sink=SessionRoutingSink(),
        session_id="sess_test",
    )


def _session(domain: str) -> dict:
    return tau2_session_update(domain)["session"]


class ExactPromptTests(unittest.TestCase):
    def test_snapshots_for_every_domain_and_apply_to(self) -> None:
        update = os.environ.get("UPDATE_PROMPT_SNAPSHOTS") == "1"
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            session = _session(domain)
            for apply_to in (("backend",), ("backend", "frontend")):
                runner = _runner(apply_to)
                runner.configure(tools=session["tools"], instructions=session["instructions"])
                for role, prompt in runner.rendered_prompts().items():
                    # JSON keeps the prompt byte-exact: tau2 policies carry trailing spaces,
                    # which the repository's whitespace hooks would strip from a text file.
                    name = f"{domain}__{'+'.join(apply_to)}__{role}.json"
                    with self.subTest(snapshot=name):
                        path = SNAPSHOTS / name
                        if update or not path.exists():
                            path.write_text(json.dumps({"prompt": prompt}, indent=2, ensure_ascii=False) + "\n")
                        self.assertEqual(prompt, json.loads(path.read_text(encoding="utf-8"))["prompt"])

    def test_default_frontend_gets_capabilities_not_policy(self) -> None:
        addendum = yaml.safe_load((PACKAGE_DIR / "config" / "prompts.voice.yaml").read_text())["cascade_voice_addendum"]
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                session = _session(domain)
                runner = _runner()
                runner.configure(tools=session["tools"], instructions=session["instructions"])
                prompts = runner.rendered_prompts()
                frontend, backend = prompts["frontend"], prompts["backend"]
                for tool in session["tools"]:
                    self.assertIn(f"- {tool['name']}: ", frontend)
                policy_line = next(
                    line
                    for line in session["instructions"].splitlines()
                    if len(line) > 60 and "policy" not in line.lower()
                )
                self.assertNotIn(policy_line, frontend)
                self.assertIn("When unsure, delegate", frontend)
                self.assertIn(session["instructions"].strip(), backend)
                self.assertIn(addendum["content"].strip().splitlines()[0], backend)
                for prompt in prompts.values():
                    self.assertIsNone(PLACEHOLDERS.search(prompt), f"unrendered placeholder in {domain}")

    def test_frontend_apply_to_appends_the_policy(self) -> None:
        session = _session("airline")
        runner = _runner(("backend", "frontend"))
        runner.configure(tools=session["tools"], instructions=session["instructions"])
        frontend = runner.rendered_prompts()["frontend"]
        self.assertIn("Policy you must follow:", frontend)
        self.assertIn(session["instructions"].strip(), frontend)
        self.assertIn("call_backend(query, filler_text)", frontend)  # the contract is never replaced

    def test_placements(self) -> None:
        session = _session("mock")
        replaced = _runner(placement="replace_prompt")
        replaced.configure(tools=session["tools"], instructions=session["instructions"])
        backend = replaced.rendered_prompts()["backend"]
        self.assertTrue(backend.startswith(session["instructions"].strip()[:40]))
        self.assertNotIn("You are the backend agent", backend)
        appended = _runner(placement="append")
        appended.configure(tools=session["tools"], instructions=session["instructions"])
        backend = appended.rendered_prompts()["backend"]
        self.assertIn("You are the backend agent", backend)
        self.assertLess(backend.index("You are the backend agent"), backend.index(session["instructions"].strip()[:40]))

    def test_no_instructions_falls_back_to_the_config_policy_without_addendum(self) -> None:
        runner = _runner()
        runner.configure(tools=[], instructions="")
        self.assertEqual(runner.resolved_instructions.source, "config")
        self.assertNotIn("Voice cascade notes", runner.rendered_prompts()["backend"])

    def test_base_config_is_never_mutated(self) -> None:
        before = base_config().agent
        runner = _runner()
        session = _session("retail")
        runner.configure(tools=session["tools"], instructions=session["instructions"])
        self.assertEqual(base_config().agent, before)
        self.assertEqual(before.domain.policy, "")
        self.assertEqual(before.prompts.inline, {})


class ScriptedDelegationTests(unittest.IsolatedAsyncioTestCase):
    async def test_in_scope_request_reaches_the_backend(self) -> None:
        for domain in DOMAINS:
            with self.subTest(domain=domain):
                frontend = FakeChatClient([delegate_response(IN_SCOPE_REQUEST[domain])])
                backend = FakeChatClient([text_response("Sure.")])
                runner = _runner(frontend=frontend, backend=backend)
                session = _session(domain)
                runner.configure(tools=session["tools"], instructions=session["instructions"])
                reply = await runner.respond(IN_SCOPE_REQUEST[domain])
                self.assertEqual(reply.text, "Sure.")
                self.assertEqual([t["function"]["name"] for t in frontend.calls[0]["tools"]], ["call_backend"])
                self.assertEqual(
                    sorted(t["function"]["name"] for t in backend.calls[0]["tools"]),
                    sorted(t["name"] for t in session["tools"]),
                )
                self.assertIn(IN_SCOPE_REQUEST[domain], backend.calls[0]["messages"][-1].content)


class VoiceCatalogTests(unittest.TestCase):
    def test_voice_catalog_keeps_the_text_contract_and_stays_domain_neutral(self) -> None:
        catalog = yaml.safe_load((PACKAGE_DIR / "config" / "prompts.voice.yaml").read_text())
        history_keys = {
            "backend_history_request",
            *(
                f"backend_history_{kind}_{include}"
                for kind in ("context", "guidance")
                for include in ("full", "backend_turns", "transcript")
            ),
        }
        normalization_keys = {
            "identifier_note_voice",
            "tool_argument_invalid",
            "tool_argument_invalid_readback",
            "tool_argument_invalid_spell_all",
            "tool_call_already_failed",
            "lookup_not_found_hint",
            "lookup_not_found_hint_words",
        }
        barge_in_keys = {"frontend_task_in_progress"}
        self.assertEqual(
            set(catalog),
            {"frontend", "backend", "cascade_voice_addendum", *history_keys, *normalization_keys, *barge_in_keys},
        )
        frontend, backend = catalog["frontend"]["content"], catalog["backend"]["content"]
        for placeholder in ("{persona}", "{capabilities}", "{unsupported_reply}", "{agent_name}"):
            self.assertIn(placeholder, frontend)
        self.assertIn("{domain_policy}", backend)
        self.assertIn("verbatim", backend)
        self.assertEqual(frontend.count('User: "'), 6)
        for name in ("airline", "flight", "retail", "telecom", "banking"):
            for key, item in catalog.items():
                self.assertIsNone(re.search(rf"\b{name}", item["content"].lower()), f"{name} in {key}")

    def test_text_catalog_is_not_referenced_by_the_voice_default(self) -> None:
        self.assertEqual(base_config().agent.prompts_path, Path(PACKAGE_DIR / "config" / "prompts.voice.yaml"))
