# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""With every feature off, every prompt is byte-identical to agent commit 3a7e04a (plan section 1, rule 2).

Except the frontend normalization note's example value, now invented (genericization item 2).
"""

from __future__ import annotations

import unittest

from _fda_golden import load_golden, render_all
from _fda_voice_fakes import PACKAGE, tau2_session_update

from examples.frontend_delegation_agent.core.frontend.prompts import load_prompts
from examples.frontend_delegation_agent.core.sidecar.templates import BackendTemplates, TemplateError


class GoldenPromptTests(unittest.TestCase):
    def test_every_baseline_render_is_byte_identical(self) -> None:
        renders, golden = render_all(), load_golden()
        self.assertEqual(sorted(renders), sorted(golden))
        for name, text in renders.items():
            with self.subTest(name=name):
                self.assertEqual(text, golden[name])

    def test_each_backend_variant_changes_only_its_own_block(self) -> None:
        path = PACKAGE / "config" / "prompts" / "customer_service" / "backend" / "hermes.yaml"
        airline = tau2_session_update("airline")["session"]["instructions"]
        base = BackendTemplates(path).render("backend_system", instructions=airline)
        spelling = BackendTemplates(path, prompt_features={"spelling_v2": True}).render(
            "backend_system", instructions=airline
        )
        self.assertIn("spelled letters are already joined", spelling)
        self.assertNotIn("ask the user to spell an\nidentifier", spelling)
        self.assertEqual(spelling.split("<policy>")[1], base.split("<policy>")[1])
        spoken = BackendTemplates(path, prompt_features={"spoken_output": True}).render(
            "backend_system", instructions=airline
        )
        self.assertTrue(spoken.startswith(base))
        self.assertIn("Confirmation override", spoken[len(base) :])
        consent = BackendTemplates(path, prompt_features={"write_consent": True}).render(
            "backend_system", instructions=""
        )
        self.assertIn("<consent>", consent)
        self.assertEqual(
            BackendTemplates(path, prompt_features={"spoken_output": False}).render("backend_soul"),
            BackendTemplates(path).render("backend_soul"),
        )

    def test_frontend_variant_adds_the_unheard_line_only_when_on(self) -> None:
        path = PACKAGE / "config" / "prompts" / "customer_service" / "frontend.yaml"
        values = {
            "state": "IDLE",
            "current_request": "",
            "elapsed_s": None,
            "recent_activity": [],
            "backend_asked_question": False,
            "last_answer_unheard": True,
        }
        self.assertNotIn("not heard", load_prompts(path).render("frontend_backend_state", **values))
        on = load_prompts(path, prompt_features={"replay_intent": True})
        self.assertTrue(on.render("frontend_backend_state", **values).endswith("your last answer was not heard: yes"))
        system = on.render("frontend_system", capabilities=[], normalization_note="")
        self.assertIn('also use request "status"', system)

    def test_unknown_feature_names_fail(self) -> None:
        with self.assertRaises(TemplateError):
            BackendTemplates(
                PACKAGE / "config" / "prompts" / "customer_service" / "backend" / "hermes.yaml",
                prompt_features={"shorter": True},
            )


if __name__ == "__main__":
    unittest.main()
