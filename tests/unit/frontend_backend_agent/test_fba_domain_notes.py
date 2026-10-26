# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Domain notes: ``backend_domain_notes``, ``domain.name`` and ``instructions.domain_from_tools``.

Also the customer-service frontend's Unclear exception and the phone-number lookup retry. Plan
``misc/frontend-backend-analysis/fba-telecom-prompt-fix-plan.md``.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import yaml
from _fba_core_fakes import (
    CONFIG_DIR,
    PACKAGE_ROOT,
    FakeChatClient,
    base_config,
    benchmark_session_update,
    benchmark_update_with,
    delegate_response,
    tool_response,
)
from _fba_golden import _clients, pinned_environment
from _fba_pipecat_fakes import (
    EnergyVADAnalyzer,
    FakeWebSocket,
    PipecatHarness,
    PipecatTestCase,
    StubRecognizer,
    StubSTT,
    ToneTTS,
)

from examples.frontend_backend_agent.adapters.pipecat import ui_session
from examples.frontend_backend_agent.adapters.pipecat.pipeline import SpeechStack
from examples.frontend_backend_agent.adapters.pipecat.runtime import PipecatRuntime
from examples.frontend_backend_agent.adapters.pipecat.settings import load_pipecat_settings
from examples.frontend_backend_agent.core.agent.factory import ServerOptions
from examples.frontend_backend_agent.core.agent.instructions import resolve_domain
from examples.frontend_backend_agent.core.agent.prompts import load_catalog, render
from examples.frontend_backend_agent.core.config.profiles import DEFAULT_PROFILE, profile_path
from examples.frontend_backend_agent.core.config.voice import SAMPLE_IN_PROGRESS, DomainRule, load_voice_config
from examples.frontend_backend_agent.core.engine.event_log import EventLog, SessionRoutingSink
from examples.frontend_backend_agent.core.engine.runner import AgentClients, TextAgentRunner
from examples.frontend_backend_agent.core.errors import VoiceConfigError
from examples.frontend_backend_agent.core.tools.tools import ToolSpec
from examples.shared.pipeline_utils import create_transport

PROFILES = CONFIG_DIR / "profiles"
PROMPTS_DIR = CONFIG_DIR / "prompts"
NOTES_HEADING = "Notes for this service:"
UNCLEAR_EXCEPTION = "an unclear turn is not Unclear mode"
INCLUDE_LINE = '    {% include "backend_domain_notes" %}\n'


def _env() -> Any:
    return mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"})


with _env():
    EVAL = load_voice_config(PROFILES / "eval.yaml")


def _load_profile(body: str, files: dict[str, str] | None = None) -> Any:
    """Load an ``eval.yaml`` child profile; ``files`` are written next to it (``{dir}`` in ``body`` names it)."""
    with tempfile.TemporaryDirectory() as tmp, _env():
        for name, text in (files or {}).items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        path = Path(tmp) / "profile.yaml"
        path.write_text(f"extends: {PROFILES / 'eval.yaml'}\n{body.replace('{dir}', tmp)}", encoding="utf-8")
        return load_voice_config(path)


def _runner(config: Any) -> TextAgentRunner:
    return TextAgentRunner(
        base_config=config.agent,
        tools_config=config.tools,
        instructions_config=config.instructions,
        clients=AgentClients(backend=FakeChatClient(), frontend=FakeChatClient()),
        sink=SessionRoutingSink(),
        session_id="sess_test",
    )


def _backend_prompt(config: Any, domain: str) -> tuple[str, TextAgentRunner]:
    session = benchmark_session_update(domain)["session"]
    runner = _runner(config)
    runner.configure(tools=session["tools"], instructions=session["instructions"])
    return runner.rendered_prompts()["backend"], runner


def _specs(*names: str) -> tuple[ToolSpec, ...]:
    return tuple(ToolSpec(name=name, description="", parameters={"type": "object", "properties": {}}) for name in names)


class EvalProfileTests(unittest.TestCase):
    def test_only_telecom_gets_the_notes(self) -> None:
        for domain in ("telecom", "retail", "airline"):
            with self.subTest(domain=domain):
                prompt, runner = _backend_prompt(EVAL, domain)
                if domain == "telecom":
                    self.assertIn(NOTES_HEADING, prompt)
                    self.assertEqual(runner.config.domain.name, "telecom")
                    self.assertEqual(runner.resolved_instructions.domain_source, "detected")
                else:
                    self.assertNotIn(NOTES_HEADING, prompt)
                    self.assertEqual(runner.config.domain.name, "")
                    self.assertEqual(runner.resolved_instructions.domain_source, "none")

    def test_the_notes_follow_the_policy_after_a_blank_line(self) -> None:
        prompt, _ = _backend_prompt(EVAL, "telecom")
        policy = benchmark_session_update("telecom")["session"]["instructions"].strip()
        self.assertIn(f"{policy}\n\n{NOTES_HEADING}", prompt)


class ResolveDomainTests(unittest.TestCase):
    RULES = replace(
        EVAL.instructions,
        domain_from_tools=(DomainRule("telecom", ("get_customer_by_phone",)), DomainRule("other", ("lookup",))),
    )

    def test_a_configured_name_wins_over_detection(self) -> None:
        base = replace(EVAL.agent, domain=replace(EVAL.agent.domain, name="retail"))
        self.assertEqual(resolve_domain(base, self.RULES, _specs("get_customer_by_phone")), ("retail", "configured"))

    def test_the_first_matching_rule_wins(self) -> None:
        tools = _specs("lookup", "get_customer_by_phone")
        self.assertEqual(resolve_domain(EVAL.agent, self.RULES, tools), ("telecom", "detected"))

    def test_no_match_renders_no_notes(self) -> None:
        self.assertEqual(resolve_domain(EVAL.agent, self.RULES, _specs("get_user_details")), ("", "none"))
        catalog = load_catalog(EVAL.agent.prompts_path)
        self.assertEqual(render(catalog.get("backend_domain_notes"), EVAL.agent, catalog=catalog), "")


class CatalogTests(unittest.TestCase):
    def test_every_profile_renders_as_before_without_a_domain(self) -> None:
        for name in ("generic", "general_qa", "customer_service"):
            with self.subTest(profile=name):
                path = PROMPTS_DIR / f"{name}.yaml"
                catalog = load_catalog(path)
                text = catalog.get("backend")
                without = load_catalog(path, {"backend": text.replace(INCLUDE_LINE, "")})
                self.assertEqual(
                    render(text, EVAL.agent, catalog=catalog),
                    render(without.get("backend"), EVAL.agent, catalog=without),
                )

    def test_a_domain_overrides_a_callers_context(self) -> None:
        catalog = load_catalog(PROMPTS_DIR / "customer_service.yaml")
        telecom = replace(EVAL.agent, domain=replace(EVAL.agent.domain, name="telecom"))
        notes = render(catalog.get("backend_domain_notes"), telecom, {"domain": {"name": "x"}}, catalog=catalog)
        self.assertIn(NOTES_HEADING, notes)

    def test_only_customer_service_has_domain_notes(self) -> None:
        telecom = replace(EVAL.agent, domain=replace(EVAL.agent.domain, name="telecom"))
        for path in (*sorted(PROMPTS_DIR.glob("*.yaml")), PACKAGE_ROOT / "prompts.yaml"):
            with self.subTest(catalog=str(path.relative_to(PACKAGE_ROOT))):
                catalog = load_catalog(path)
                notes = render(catalog.get("backend_domain_notes"), telecom, catalog=catalog)
                self.assertEqual(bool(notes), path.name == "customer_service.yaml")

    def test_unclear_turns_continue_a_request_in_progress(self) -> None:
        catalog = load_catalog(PROMPTS_DIR / "customer_service.yaml")
        for in_progress in (None, SAMPLE_IN_PROGRESS):
            with self.subTest(in_progress=bool(in_progress)):
                context = {"barge_in": {"note_key": "frontend_task_in_progress"}, "in_progress": in_progress}
                frontend = render(catalog.get("frontend"), EVAL.agent, context, catalog=catalog)
                self.assertIn("- Unclear: only if the turn has no recognisable words", frontend)
                self.assertEqual(UNCLEAR_EXCEPTION in frontend, in_progress is not None)


class LoadCheckTests(unittest.TestCase):
    def test_a_name_without_a_branch_fails_at_load(self) -> None:
        with self.assertRaisesRegex(VoiceConfigError, "domain 'telcom' has no branch"):
            _load_profile("agent: {overrides: {domain: {name: telcom}}}\n")

    def test_a_rule_without_a_branch_fails_at_load(self) -> None:
        body = "instructions: {domain_from_tools: {telcom: {tools_any: [get_customer_by_phone]}}}\n"
        with self.assertRaisesRegex(VoiceConfigError, "domain 'telcom' has no branch"):
            _load_profile(body)

    def test_a_configured_name_loads(self) -> None:
        self.assertEqual(_load_profile("agent: {overrides: {domain: {name: telecom}}}\n").agent.domain.name, "telecom")

    def test_a_catalog_with_empty_notes_skips_the_check(self) -> None:
        for profile in ("generic", "general_qa"):
            with self.subTest(profile=profile), mock.patch.dict(os.environ, {"FBA_PROMPT_PROFILE": profile}):
                config = _load_profile("")
                self.assertEqual(config.agent.prompts_path.name, f"{profile}.yaml")
                self.assertEqual([rule.name for rule in config.instructions.domain_from_tools], ["telecom"])

    def test_a_catalog_without_notes_skips_the_check(self) -> None:
        catalog = yaml.safe_load((PROMPTS_DIR / "customer_service.yaml").read_text(encoding="utf-8"))
        del catalog["backend_domain_notes"]
        catalog["backend"]["content"] = catalog["backend"]["content"].replace(INCLUDE_LINE.strip() + "\n", "")
        body = "agent: {overrides: {prompts: {path: '{dir}/custom.yaml'}}}\n"
        config = _load_profile(body, {"custom.yaml": yaml.safe_dump(catalog)})
        self.assertEqual(config.agent.prompts_path.name, "custom.yaml")

    def test_a_rule_can_be_removed_with_null(self) -> None:
        config = _load_profile("instructions: {domain_from_tools: {telecom: null}}\n")
        self.assertEqual(config.instructions.domain_from_tools, ())

    def test_bad_rules_are_refused(self) -> None:
        bad = ("{telecom: [x]}", "{telecom: {tools_any: []}}", "{telecom: {tools: [x]}}", "{1: {tools_any: [x]}}")
        for rules in bad:
            with self.subTest(rules=rules), self.assertRaisesRegex(VoiceConfigError, "domain_from_tools"):
                _load_profile(f"instructions: {{domain_from_tools: {rules}}}\n")


PHONE_TOOLS = [
    {
        "type": "function",
        "name": "get_customer_by_phone",
        "description": "Find a customer by phone number.",
        "parameters": {"type": "object", "properties": {"phone_number": {"type": "string"}}},
    }
]


def _output(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def _with_domain(config: Any, name: str) -> Any:
    return replace(config, agent=replace(config.agent, domain=replace(config.agent.domain, name=name)))


class SessionTests(PipecatTestCase):
    async def asyncSetUp(self) -> None:
        self.frontend = FakeChatClient()
        self.backend = FakeChatClient()
        self.tmp = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmp.name) / "events.jsonl"
        self.harness: PipecatHarness | None = None

    async def start(self, config: Any = None) -> PipecatHarness:
        self.harness = PipecatHarness(
            # The base config's turn timing, with the eval profile's domain rules and normalization.
            config=config or replace(base_config(), instructions=EVAL.instructions, normalization=EVAL.normalization),
            clients=AgentClients(backend=self.backend, frontend=self.frontend),
            recognizer=StubRecognizer(None),
            event_log=EventLog(str(self.log_path), redact_content=False),
        )
        await self.harness.start(benchmark_update_with(PHONE_TOOLS, "Policy text."))
        return self.harness

    async def asyncTearDown(self) -> None:
        if self.harness is not None:
            await self.harness.close()
        self.tmp.cleanup()

    async def session_updated(self) -> dict[str, Any]:
        await self.harness.close()
        records = [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]
        updated = [record for record in records if record["kind"] == "session_updated"]
        self.assertTrue(updated)
        return updated[-1]

    async def test_session_updated_records_a_detected_domain(self) -> None:
        await self.start()
        record = await self.session_updated()
        self.assertEqual((record["domain"], record["domain_source"]), ("telecom", "detected"))

    async def test_session_updated_records_a_configured_domain(self) -> None:
        await self.start(_with_domain(base_config(), "telecom"))
        record = await self.session_updated()
        self.assertEqual((record["domain"], record["domain_source"]), ("telecom", "configured"))

    async def test_a_dashed_phone_retry_is_not_guarded(self) -> None:
        """A lookup retried in the other phone format goes out: retry_guard covers get_user_details only.

        The scripted backend makes the retry, so this checks the session, not the prompt rule.
        """
        harness = await self.start()
        self.frontend.queue(delegate_response("My number is 5551232002."))
        self.backend.queue(
            tool_response(("get_customer_by_phone", {"phone_number": "5551232002"}), ids=["c1"]),
            tool_response(("get_customer_by_phone", {"phone_number": "555-123-2002"}), ids=["c2"]),
        )
        await harness.speak()
        await harness.wait_for("response.done")
        await harness.send(_output("c1", "Error: Customer with phone number 5551232002 not found"))
        await harness.send({"type": "response.create"})
        await harness.wait_for("response.done", 2)
        calls = harness.of_type("response.function_call_arguments.done")
        self.assertEqual([call["call_id"] for call in calls], ["c1", "c2"])
        self.assertEqual(json.loads(calls[1]["arguments"]), {"phone_number": "555-123-2002"})
        tool_results = [m.content for call in self.backend.calls for m in call["messages"] if m.role == "tool"]
        self.assertFalse(any("already failed" in content for content in tool_results))


class UiSessionTests(unittest.TestCase):
    def _session_updated(self, domain: str) -> dict[str, Any]:
        with pinned_environment():
            _, _, clients = _clients()
            config = _with_domain(load_voice_config(profile_path(DEFAULT_PROFILE)), domain)
            runtime = PipecatRuntime(
                config,
                profile=DEFAULT_PROFILE,
                settings=load_pipecat_settings("default.yaml"),
                options=ServerOptions(stub_speech=True),
                clients=clients,
            )
            selection = ui_session.UiSelection.from_body({})
            session = ui_session.UiSession(
                runtime=ui_session._session_runtime(runtime, ui_session.session_config(runtime, selection)),
                transport=create_transport(SimpleNamespace(websocket=FakeWebSocket(), body={"protocol": "rtvi"})),
                speech=lambda _s: SpeechStack(vad=EnergyVADAnalyzer(), stt=StubSTT(), tts=ToneTTS()),
            )
            with mock.patch.object(session.log, "write") as write:
                session.configure_agent()
        ((kind,), data) = write.call_args
        self.assertEqual(kind, "session_updated")
        return data

    def test_the_browser_tools_select_no_domain(self) -> None:
        data = self._session_updated("")
        self.assertEqual((data["domain"], data["domain_source"]), ("", "none"))

    def test_a_configured_domain_is_logged(self) -> None:
        data = self._session_updated("telecom")
        self.assertEqual((data["domain"], data["domain_source"]), ("telecom", "configured"))


if __name__ == "__main__":
    unittest.main()
