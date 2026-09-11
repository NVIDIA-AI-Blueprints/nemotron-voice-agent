# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for Realtime-specific projections into example pipeline settings."""

# ruff: noqa: D102

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pipecat.services.settings import is_given

from examples.shared.nvidia_llm import (
    NvidiaLLMService,
    NvidiaLLMSettings,
    _apply_realtime_completion_token_limit,
    _truncate_realtime_context,
)
from examples.shared.pipeline_utils import resolve_pipeline_prompt, select_max_tokens_config
from realtime.controller import RealtimeSessionController
from realtime.frames import RealtimeResponseLLMContext
from realtime.gateway import _controller_from_runtime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RTVI_COMPACTION_HANDLERS = (
    ("src/examples/generic/pipeline.py", "apply_pinned_prompt_summary"),
    ("src/examples/multilingual/pipeline.py", "apply_pinned_prompt_summary"),
    ("src/examples/frontend_backend_agent/pipeline.py", "_apply_chat_history_sliding_window"),
)
_REALTIME_PIPELINES = (
    "src/examples/generic/pipeline.py",
    "src/examples/multilingual/pipeline.py",
    "src/examples/frontend_backend_agent/pipeline.py",
    "src/examples/omni_assistant/pipeline.py",
    "src/examples/omni_assistant_subagents/pipeline.py",
)


def _called_function_names(node: ast.AST) -> set[str]:
    return {call.func.id for call in ast.walk(node) if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)}


class RealtimePipelineConfigTests(unittest.TestCase):
    """Keep canonical Realtime values distinct from ordinary config fallback."""

    def test_max_tokens_projection_is_owned_by_the_realtime_server(self) -> None:
        cases = (
            ({"protocol": "realtime", "max_tokens": None}, True, None),
            ({"protocol": "realtime", "max_tokens": "inf"}, True, None),
            ({"protocol": "realtime", "max_tokens": 64}, True, 64),
            ({"protocol": "rtvi"}, False, 4096),
            ({"max_tokens": ""}, False, 4096),
            ({"protocol": "realtime", "max_tokens": None}, False, 4096),
        )
        for body, is_realtime, expected in cases:
            with self.subTest(body=body, is_realtime=is_realtime):
                self.assertEqual(select_max_tokens_config(body, 4096, is_realtime=is_realtime), expected)

    def test_prompt_projection_is_owned_by_the_realtime_server(self) -> None:
        body = {
            "_realtime_instructions_explicit": True,
            "prompt_key": "generic_assistant",
            "prompt_content": "",
        }
        for is_realtime, expected_key in ((True, "custom"), (False, "generic_assistant")):
            with self.subTest(is_realtime=is_realtime):
                prompt_key, prompt = resolve_pipeline_prompt(
                    _PROJECT_ROOT / "src/examples/generic/pipeline.py",
                    body,
                    is_realtime=is_realtime,
                )
                self.assertEqual(prompt_key, expected_key)
                if is_realtime:
                    self.assertEqual(prompt, "")
                else:
                    self.assertTrue(prompt)

    def test_every_realtime_pipeline_uses_the_shared_prompt_resolver(self) -> None:
        for relative_path in _REALTIME_PIPELINES:
            with self.subTest(pipeline=relative_path):
                source = (_PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                bot = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "bot")
                self.assertIn("resolve_pipeline_prompt", _called_function_names(bot))

    def test_realtime_assistant_turns_bind_identity_then_skip_rtvi_compaction(self) -> None:
        for relative_path, compaction_name in _RTVI_COMPACTION_HANDLERS:
            with self.subTest(pipeline=relative_path):
                source = (_PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
                tree = ast.parse(source)
                handlers = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.AsyncFunctionDef)
                    and node.name == "on_assistant_turn_stopped"
                    and compaction_name in _called_function_names(node)
                ]
                self.assertEqual(len(handlers), 1)
                handler = handlers[0]
                self.assertGreaterEqual(len(handler.body), 3)
                self.assertIn("bind_realtime_assistant_context_message", _called_function_names(handler.body[0]))

                guard = handler.body[1]
                self.assertIsInstance(guard, ast.If)
                assert isinstance(guard, ast.If)
                self.assertIsInstance(guard.test, ast.Name)
                assert isinstance(guard.test, ast.Name)
                self.assertEqual(guard.test.id, "is_realtime")
                self.assertEqual(len(guard.body), 1)
                self.assertIsInstance(guard.body[0], ast.Return)

                remaining = ast.Module(body=handler.body[2:], type_ignores=[])
                self.assertIn(compaction_name, _called_function_names(remaining))


class MultilingualTokenConfigurationTests(unittest.IsolatedAsyncioTestCase):
    """Keep Realtime response limits out of ordinary Multilingual sessions."""

    async def test_max_tokens_changes_only_realtime_primary_llm_settings(self) -> None:
        from examples.multilingual import pipeline

        class ConstructionObserved(Exception):
            pass

        async def capture_settings(*, realtime: bool):
            captured = []

            def stop_at_primary_llm(**kwargs):
                captured.append(kwargs["settings"])
                raise ConstructionObserved

            body = {"max_tokens": 64}
            llm_owner = pipeline
            llm_name = "PipecatNvidiaLLMService"
            if realtime:
                body["realtime_controller"] = RealtimeSessionController(
                    model="test-model",
                    voice="test-voice",
                    runtime_config={},
                )
                from examples.shared import nvidia_llm

                llm_owner = nvidia_llm
                llm_name = "NvidiaLLMService"

            with (
                patch.object(pipeline, "create_transport", return_value=object()),
                patch.object(pipeline, "load_selected_service_entry", return_value={}),
                patch.object(pipeline, "validate_llm_session_language"),
                patch.object(pipeline, "_prepare_session_language_codes", new=AsyncMock(return_value="")),
                patch.object(pipeline, "NvidiaSTTService"),
                patch.object(llm_owner, llm_name, side_effect=stop_at_primary_llm),
                self.assertRaises(ConstructionObserved),
            ):
                await pipeline.bot(SimpleNamespace(body=body))

            self.assertEqual(len(captured), 1)
            return captured[0]

        ordinary_settings = await capture_settings(realtime=False)
        realtime_settings = await capture_settings(realtime=True)

        self.assertFalse(is_given(ordinary_settings.max_tokens))
        self.assertEqual(realtime_settings.max_tokens, 64)


class RealtimeModelOutputLimitTests(unittest.IsolatedAsyncioTestCase):
    """Keep public Realtime limits and the selected model boundary consistent."""

    def test_provider_uses_model_cap_for_inf_and_clamps_client_upper_bound(self) -> None:
        cases = (
            ("inf", {"max_tokens": 64}, {"max_tokens": 2048}),
            (4096, {"max_completion_tokens": 64, "max_tokens": 32}, {"max_completion_tokens": 2048}),
            (256, {}, {"max_tokens": 256}),
        )
        for limit, params, expected in cases:
            with self.subTest(limit=limit):
                _apply_realtime_completion_token_limit(
                    params,
                    RealtimeResponseLLMContext([], max_output_tokens=limit),
                    model_max_output_tokens=2048,
                )
                self.assertEqual(params, expected)

    async def test_inf_truncation_reserves_the_same_model_cap_as_provider(self) -> None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ]
        context = RealtimeResponseLLMContext(
            copy.deepcopy(messages),
            truncation="auto",
            preserve_prompt_messages=1,
            max_output_tokens="inf",
        )

        async def count_tokens(candidate: dict) -> tuple[int, int]:
            return len(candidate["messages"]) * 1000, 4096

        truncated = await _truncate_realtime_context(
            {"messages": copy.deepcopy(messages)},
            context,
            count_tokens=count_tokens,
            model_max_output_tokens=2048,
        )

        self.assertEqual(truncated["messages"], [messages[0], messages[-1]])

    async def test_inf_truncation_fails_closed_without_a_model_cap(self) -> None:
        context = RealtimeResponseLLMContext(
            [{"role": "user", "content": "hello"}],
            truncation="auto",
            max_output_tokens="inf",
        )

        async def count_tokens(_candidate: dict) -> tuple[int, int]:
            return 10, 4096

        with self.assertRaisesRegex(RuntimeError, "trusted model output limit"):
            await _truncate_realtime_context(
                {"messages": [{"role": "user", "content": "hello"}]},
                context,
                count_tokens=count_tokens,
            )

    def test_initial_catalog_default_is_clamped_before_session_created(self) -> None:
        controller = _controller_from_runtime(
            {
                "pipeline_mode": "generic-assistant",
                "model_id": "test-model",
                "max_tokens": 4096,
                "realtime_model_max_output_tokens": 2048,
            },
            server_tools=[],
            delegate_tools=[],
        )

        self.assertEqual(controller.public_session()["max_output_tokens"], 2048)
        self.assertEqual(controller.runtime_config["max_tokens"], 4096)

    def test_service_rejects_an_invalid_trusted_model_cap(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 1 and 4096"):
            NvidiaLLMService(
                api_key="test",
                base_url="http://127.0.0.1:9/v1",
                settings=NvidiaLLMSettings(model="test-model"),
                realtime_model_max_output_tokens=4097,
            )


if __name__ == "__main__":
    unittest.main()
