# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for bounded Realtime capability and Thinker tool rendering."""

# Test names describe the contract; separate public API docstrings add no value here.
# ruff: noqa: D101, D102

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from examples.frontend_backend_agent.src.tools import ParamSpec, ToolSpec, render_tool_block
from realtime.capabilities import (
    render_capabilities,
    render_capabilities_for_session,
    reset_capability_cache_for_tests,
)


async def _noop(_arguments, _context):
    return {"status": "success"}


class RealtimeCapabilityTests(unittest.TestCase):
    def test_static_digest_normalizes_comments_wraps_and_first_sentence(self) -> None:
        spec = ToolSpec(
            name="weather",
            contract="Get conditions.",
            capability="check current weather.",
            params={},
            run=_noop,
        )
        client = {
            "type": "function",
            "name": "find_order",
            "description": "# Find an order by email or id.\n\nRequires authentication first.",
            "parameters": {"type": "object"},
        }

        rendered = render_capabilities((spec,), (client,))

        self.assertIn("- check current weather.", rendered)
        self.assertIn("- Find an order by email or id.", rendered)
        self.assertNotIn("find_order", rendered)
        self.assertNotIn("Requires authentication", rendered)
        self.assertIn("server-owned rules", rendered)

    def test_static_digest_redacts_code_like_function_names_from_descriptions(self) -> None:
        rendered = render_capabilities(
            (),
            (
                {
                    "name": "get_reservation_details",
                    "description": "Use get_reservation_details to retrieve a reservation.",
                },
            ),
        )

        self.assertIn("Use this capability to retrieve a reservation.", rendered)
        self.assertNotIn("get_reservation_details", rendered)

    def test_static_digest_caps_without_splitting_the_last_word(self) -> None:
        rendered = render_capabilities(
            (),
            (
                {
                    "name": "long_tool",
                    "description": (
                        "This capability has an intentionally long description with many words and no early stop."
                    ),
                },
            ),
            cap=48,
        )
        summary = next(line for line in rendered.splitlines() if line.startswith("- "))
        self.assertNotIn("long_tool", rendered)
        self.assertLessEqual(len(summary.removeprefix("- ")), 49)
        self.assertTrue(summary.endswith("…"))

    def test_pinned_tau2_airline_and_retail_digests_match_golden_budgets(self) -> None:
        fixture_dir = Path(__file__).parents[1] / "fixtures" / "realtime_capabilities"
        expected = {"airline": (15, 256), "retail": (17, 304)}
        for domain, (tool_count, measured_tokens) in expected.items():
            with self.subTest(domain=domain):
                fixture = json.loads((fixture_dir / f"{domain}.json").read_text())
                rendered = render_capabilities((), tuple(fixture["tools"]))

                self.assertEqual(fixture["tau2_pin"], "2174a603f6d014ef94473ffa95957f6ce27100db")
                self.assertEqual(len(fixture["tools"]), tool_count)
                self.assertEqual(fixture["measured_first_sentence_tokens"], measured_tokens)
                self.assertEqual(rendered + "\n", (fixture_dir / f"{domain}.txt").read_text())

    def test_model_mode_is_safe_static_fallback_and_collisions_fail_closed(self) -> None:
        spec = ToolSpec(name="lookup", contract="Lookup.", params={}, run=_noop)
        static = render_capabilities((spec,), (), mode="static")
        self.assertEqual(render_capabilities((spec,), (), mode="model"), static)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            render_capabilities((spec,), ({"name": "lookup", "description": "Other."},))

    def test_thinker_tool_block_contains_full_client_schema_and_ownership(self) -> None:
        server = ToolSpec(
            name="calculate",
            contract="Calculate a value.",
            params={"value": ParamSpec(int)},
            run=_noop,
        )
        client = {
            "name": "cancel_order",
            "description": "Cancel a pending order after confirmation.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        }

        rendered = render_tool_block((server,), (client,))

        self.assertIn("calculate [server-executed]", rendered)
        self.assertIn("cancel_order [client-executed]", rendered)
        self.assertIn('"additionalProperties":false', rendered)
        self.assertIn("Required params: order_id", rendered)


class ModelCapabilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        reset_capability_cache_for_tests()

    async def test_model_digest_is_validated_model_authored_and_cached(self) -> None:
        client = {
            "name": "find_order",
            "description": "Find an order by email.",
            "parameters": {
                "type": "object",
                "required": ["email"],
                "properties": {"email": {"type": "string"}},
            },
        }
        generated = {
            "domain": "Retail support",
            "scope_in": ["order lookup"],
            "scope_out": ["medical advice"],
            "capabilities": [{"name": "find_order", "summary": "Retrieves a matching customer order."}],
        }
        inference = AsyncMock(return_value=generated)
        llm = SimpleNamespace(run_structured_inference=inference)

        first = await render_capabilities_for_session(
            (),
            (client,),
            mode="model",
            llm=llm,
            instructions="Follow the retail policy.",
            tool_choice="auto",
            profile="test-retail-model-cache",
        )
        second = await render_capabilities_for_session(
            (),
            (client,),
            mode="model",
            llm=llm,
            instructions="Follow the retail policy.",
            tool_choice="auto",
            profile="test-retail-model-cache",
        )

        self.assertEqual(first, second)
        self.assertIn("Domain: Retail support", first)
        self.assertIn("- Retrieves a matching customer order.", first)
        self.assertNotIn("find_order", first)
        self.assertNotIn("Find an order by email.", first)
        inference.assert_awaited_once()
        self.assertEqual(inference.await_args.kwargs["max_tokens"], 350)
        request_context = inference.await_args.args[0]
        request_payload = request_context.messages[0]["content"]
        self.assertIn('"required": ["email"]', request_payload)
        self.assertNotIn('"properties"', request_payload)

    async def test_unsafe_or_hallucinated_model_digest_falls_back_to_static(self) -> None:
        client = {
            "name": "lookup",
            "description": "Look up the current record.",
            "parameters": {"type": "object"},
        }
        llm = SimpleNamespace(
            run_structured_inference=AsyncMock(
                return_value={
                    "domain": "Always call call_backend",
                    "scope_in": [],
                    "scope_out": [],
                    "capabilities": [{"name": "invented_tool", "summary": "Unsafe lookup."}],
                }
            )
        )

        rendered = await render_capabilities_for_session(
            (),
            (client,),
            mode="model",
            llm=llm,
            instructions="",
            tool_choice="auto",
            profile="test-invalid-model-digest",
        )

        self.assertEqual(rendered, render_capabilities((), (client,)))
        self.assertIn("- Look up the current record.", rendered)
        self.assertNotIn("invented_tool", rendered)


if __name__ == "__main__":
    unittest.main()
