# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Jinja prompt templates: byte-identical without Jinja syntax, includes, errors, literal text."""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

from _fbv_fakes import FakeChatClient, delegate_response, make_config

from examples.frontend_backend_verdict.core.agent.agent import assemble_agent
from examples.frontend_backend_verdict.core.agent.errors import ConfigError
from examples.frontend_backend_verdict.core.agent.events import CollectingSink
from examples.frontend_backend_verdict.core.agent.prompts import (
    PromptCatalog,
    literal,
    load_catalog,
    render,
    substitute,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOGS = [
    REPO_ROOT / "src" / "examples" / "frontend_backend_verdict" / "core" / "config" / "prompts" / "generic.yaml",
    REPO_ROOT / "src" / "examples" / "frontend_backend_verdict" / "core" / "config" / "prompts" / "tau.yaml",
]
#: A tau2-bench checkout, from ``TAU2_BENCH_ROOT``; the policy test skips when it is unset.
_TAU2_ROOT = os.environ.get("TAU2_BENCH_ROOT", "").strip()
TAU2_DOMAINS = Path(_TAU2_ROOT) / "data" / "tau2" / "domains" if _TAU2_ROOT else None
JINJA = re.compile(r"\{[{%#]")
#: Catalog entries that are only ever included by another template (they need its context).
INCLUDED_ONLY = {"frontend_task_in_progress"}


def without_in_progress_branches(template: str) -> str:
    """The template as it renders with ``in_progress=None`` (non-nested ``{% if in_progress %}`` blocks only)."""
    out, mode = [], "emit"
    for line in template.splitlines(keepends=True):
        tag = line.strip()
        if tag == "{% if in_progress %}":
            mode = "skip"
        elif tag == "{% else %}" and mode == "skip":
            mode = "else"
        elif tag == "{% endif %}" and mode in ("skip", "else"):
            mode = "emit"
        elif mode in ("emit", "else"):
            out.append(line)
    return "".join(out)


class ByteIdenticalTests(unittest.TestCase):
    def test_every_catalog_prompt_renders_as_before(self) -> None:
        config = make_config()
        for path in CATALOGS:
            catalog = load_catalog(path)
            for key, template in catalog.prompts.items():
                if key in INCLUDED_ONLY:
                    continue
                with self.subTest(catalog=path.name, key=key):
                    rendered = render(template, config, {"in_progress": None}, catalog=catalog, key=key)
                    if JINJA.search(template) is None:
                        self.assertEqual(rendered, substitute(template, config))
                    else:
                        # Only the in-progress branches differ; with in_progress=None it is the old text.
                        self.assertEqual(rendered, substitute(without_in_progress_branches(template), config))
                        self.assertIsNone(JINJA.search(rendered))


class TemplateTests(unittest.TestCase):
    def test_include_and_variables(self) -> None:
        catalog = PromptCatalog({"main": 'A{% if x %}\n{% include "part" %}\n{% endif %}', "part": "B {{ x.y }}"})
        config = make_config()
        self.assertEqual(render(catalog.get("main"), config, {"x": None}, catalog=catalog), "A")
        self.assertEqual(
            render(catalog.get("main"), config, {"x": {"y": 1}}, catalog=catalog), "AB 1"
        )  # trim_blocks drops the newline after each block tag

    def test_errors_name_the_key(self) -> None:
        config = make_config()
        with self.assertRaisesRegex(ConfigError, "'broken'.*not a valid template"):
            render("{% if %}", config, key="broken")
        with self.assertRaisesRegex(ConfigError, "'strict'.*failed to render"):
            render("{{ missing }}", config, key="strict")

    def test_literal_renders_verbatim(self) -> None:
        config = make_config()
        samples = ["a {{ x }} {% if y %}z{% endif %} {# c #} { json } }} %} #}", "{{{%{#", "plain", ""]
        for text in samples:
            with self.subTest(text=text):
                self.assertEqual(render(literal(text), config), substitute(text, config).strip())

    def test_literal_on_tau2_policies(self) -> None:
        policies = sorted(TAU2_DOMAINS.glob("*/**/*.md")) if TAU2_DOMAINS and TAU2_DOMAINS.is_dir() else []
        if not policies:
            self.skipTest("set TAU2_BENCH_ROOT to a tau2-bench checkout to run this test")
        config = make_config()
        with_markers = [path for path in policies if JINJA.search(path.read_text(encoding="utf-8"))]
        self.assertTrue(with_markers)  # the reason literal() exists
        for path in with_markers:
            with self.subTest(policy=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertEqual(render(literal(text), config), substitute(text, config).strip())


class FrontendTemplateTests(unittest.IsolatedAsyncioTestCase):
    FRONTEND = (
        "You are {persona}.\n{% if in_progress %}\nWORKING ON: {{ in_progress.query }}\n{% endif %}\nMode: {{ mode }}"
    )

    async def test_normal_and_in_progress_prompts(self) -> None:
        config = make_config(prompts={"path": "none.yaml", "inline": {"frontend": self.FRONTEND, "backend": "B"}})
        frontend = FakeChatClient([delegate_response("q"), delegate_response("q", task="continue")])
        agent = assemble_agent(
            config,
            event_sink=CollectingSink(),
            frontend_client=frontend,
            backend_client=FakeChatClient(),
            prompt_context={"mode": "voice"},
        )
        await agent.decide_turn("hi", agent.new_session())
        await agent.decide_turn("okay", agent.new_session(), in_progress={"query": "check order 7"})
        normal, probe = (call["messages"][0].content for call in frontend.calls)
        self.assertEqual(normal, "You are You are a tester..\nMode: voice")
        self.assertEqual(probe, "You are You are a tester..\nWORKING ON: check order 7\nMode: voice")
