# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The guard against hardcoding: no kind compare, no foreign isinstance, no banned flag.

The rules are in ``guard.py``, beside this file. A failure names the file, the line and the site.
"""

import unittest

from tests.unit.guard.guard import (
    module_name,
    sites_in,
    walked_files,
    walked_sites,
)


class GuardTests(unittest.TestCase):
    def test_the_walked_modules_hold_no_site(self):
        problems = [f"{path}:{line} {site.kind} {site.name}: hardcoding" for path, line, site in walked_sites()]
        self.assertEqual(problems, [], "\n".join(["Hardcoding the guard does not allow:", *problems]))

    def test_the_walk_reads_the_gateway_the_example_and_the_core(self):
        # The walk reads every tree it guards, so the rules cannot pass by reading nothing.
        modules = {module_name(path).split(".")[0] for path in walked_files()}
        self.assertLessEqual({"live", "examples", "delegation_core"}, modules)


class RuleTests(unittest.TestCase):
    def kinds(self, source, module="live.session.example"):
        return [(site.kind, site.name) for _, site in sites_in(source, module)]

    def test_a_kind_compared_with_a_literal_is_a_site(self):
        source = "\n".join(
            [
                "if config.mode == 'client': pass",
                "if settings.adapter != HARNESS: pass",
                "if session.config.delegation.type in ('client', 'responses'): pass",
                "if self.delegation.target == responses_target: pass",  # a variable, not a literal
                "if mode == 'client': pass",  # a local that holds a kind
                "if event.kind == 'harness': pass",
                "if delegation.type == 'client': pass",
                "if settings.provider is Provider.CHAT: pass",
                "if kind == 'response.completed': pass",  # an official event's type: the protocol reads it
                "if transport == 'webrtc': pass",  # a local the protocol's own rules read
            ]
        )
        self.assertEqual(
            self.kinds(source),
            [
                ("compare", "mode"),
                ("compare", "adapter"),
                ("compare", "delegation.type"),
                ("compare", "mode"),
                ("compare", "kind"),
                ("compare", "delegation.type"),
                ("compare", "provider"),
            ],
        )

    def test_a_match_on_a_kind_is_a_site(self):
        source = "\n".join(
            [
                "match config.mode:",
                "    case 'client' | 'responses': pass",
                "match settings.backend_calls:",
                "    case Calls.ITEMS: pass",
                "match event['type']:",  # an official event's type
                "    case 'response.completed': pass",
            ]
        )
        self.assertEqual(self.kinds(source), [("compare", "mode"), ("compare", "backend_calls")])

    def test_a_kind_looked_up_by_its_name_or_a_kinds_own_name_is_a_site(self):
        source = "\n".join(
            [
                "if settings.names['agent'] == 'hermes': pass",
                "if agent == 'hermes': pass",  # a local that holds the agent's kind
                "if self.settings.agent.name == 'hermes': pass",
                "match agent.name:",
                "    case 'hermes': pass",
                "if block['transport'] in ('stdio', 'websocket'): pass",
                "if call.name == 'show': pass",  # a tool call's name: no kind
                "if tool['name'] == 'show': pass",
                "if self.instructions == 'Be brief.': pass",  # text, no kind
            ]
        )
        self.assertEqual(
            self.kinds(source),
            [
                ("compare", "agent"),
                ("compare", "agent"),
                ("compare", "agent.name"),
                ("compare", "agent.name"),
                ("compare", "transport"),
            ],
        )

    def test_the_table_modules_may_compare_names(self):
        self.assertEqual(self.kinds("if table.adapter == 'llm': pass", "delegation_core.tables"), [])

    def test_an_isinstance_against_another_modules_class_is_a_site(self):
        source = "\n".join(
            [
                "from delegation_core.api.service import StatelessStream",
                "from live.ports import ContextMeter",
                "from live.protocol import ProtocolError",
                "class Local: pass",
                "isinstance(x, StatelessStream)",
                "isinstance(x, ContextMeter)",  # a port type: a capability check
                "isinstance(x, ProtocolError)",  # an exception
                "isinstance(x, (Local, dict))",  # this module's class, a builtin
            ]
        )
        self.assertEqual(self.kinds(source, "delegation_core.api.local"), [("isinstance", "StatelessStream")])
        self.assertEqual(self.kinds(source, "examples.frontend_backend_live.factory"), [])  # a composition root

    def test_a_class_reached_through_its_module_or_by_issubclass_is_a_site_too(self):
        source = "\n".join(
            [
                "import live.session.session as live_session",
                "import examples.frontend_backend_live.models.chat_completions",
                "from delegation_core.api import service",
                "from live import ports",
                "isinstance(x, live_session.LiveProtocolSession)",
                "isinstance(x, examples.frontend_backend_live.models.chat_completions.ChatCompletionsFrontend)",
                "issubclass(cls, service.StatelessStream)",
                "isinstance(x, ports.ContextMeter)",  # a port type, through its module
                "isinstance(x, json.JSONDecoder)",  # an outside package
            ]
        )
        self.assertEqual(
            self.kinds(source, "delegation_core.im.session"),
            [
                ("isinstance", "LiveProtocolSession"),
                ("isinstance", "ChatCompletionsFrontend"),
                ("isinstance", "StatelessStream"),
            ],
        )

    def test_a_banned_flag_is_a_site_wherever_it_appears(self):
        source = "\n".join(
            [
                "class Settings:",
                "    one_shot: bool = False",
                "def new(*, stateless=False): pass",
                "new(server_tools=True)",
                "if loop.provider.stateful: pass",
                "if self.settings.status_from_ledger: pass",
            ]
        )
        self.assertEqual(
            sorted(self.kinds(source)),
            [
                ("flag", "one_shot"),
                ("flag", "server_tools"),
                ("flag", "stateful"),
                ("flag", "stateless"),
                ("flag", "status_from_ledger"),
            ],
        )

    def test_pick_is_allowed_in_a_root_and_in_a_build_method_only(self):
        source = "\n".join(
            [
                "class Source:",
                "    @classmethod",
                "    def from_settings(cls, block): return pick(TABLE, block['x'], 'x')",
                "    def run(self): return pick(TABLE, self.name, 'x')",
            ]
        )
        self.assertEqual(self.kinds(source, "delegation_core.backend.example"), [("pick", "pick")])
        self.assertEqual(self.kinds(source, "delegation_core.app"), [])
