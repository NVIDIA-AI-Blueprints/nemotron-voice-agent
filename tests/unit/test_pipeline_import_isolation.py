# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Import-boundary regressions for ordinary example startup."""

# ruff: noqa: D102

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PipelineImportIsolationTests(unittest.TestCase):
    """Ordinary pipeline imports must not initialize the Realtime tool stack."""

    def test_pipeline_imports_do_not_load_unrelated_protocol_tool_runtimes(self) -> None:
        cases = (
            ("examples.generic.pipeline", True),
            ("examples.multilingual.pipeline", True),
            ("examples.frontend_backend_agent.pipeline", True),
            ("examples.omni_assistant.pipeline", False),
            ("examples.omni_assistant_subagents.pipeline", False),
        )
        for module, forbid_all_realtime in cases:
            with self.subTest(module=module):
                self._assert_isolated_import(module, forbid_all_realtime=forbid_all_realtime)

    def _assert_isolated_import(self, module: str, *, forbid_all_realtime: bool) -> None:
        source = f"""
import sys
import {module}

forbidden = sorted(
    name
    for name in sys.modules
    if ({forbid_all_realtime!r} and name.startswith("realtime"))
    or name in {{
        "realtime.conversation",
        "realtime.gateway",
        "realtime.mcp",
        "realtime.response_config",
        "realtime.session",
        "realtime.tool_schema",
        "realtime.transport",
        "mcp",
    }}
    or name.startswith("mcp.")
)
if forbidden:
    raise SystemExit(f"unexpected eager imports: {{forbidden}}")
"""
        env = os.environ.copy()
        src_path = str(_PROJECT_ROOT / "src")
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
        env["EXAMPLE_SELECTION"] = "all"
        env["TRANSPORT_SELECTION"] = "all"

        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=_PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


class PipelineRuntimeIsolationTests(unittest.IsolatedAsyncioTestCase):
    """Ordinary construction must not cross a Realtime-only helper boundary."""

    async def test_rtvi_cascades_use_only_default_service_lookup(self) -> None:
        cases = (
            (
                "examples.generic.pipeline",
                {"llm_id": "ignored", "tts_id": "ignored", "asr_id": "ignored"},
                [call("llm", ""), call("tts", ""), call("asr", "")],
            ),
            (
                "examples.multilingual.pipeline",
                {"tts_id": "ignored", "asr_id": "ignored"},
                [call("llm", ""), call("tts", ""), call("asr", "")],
            ),
            (
                "examples.frontend_backend_agent.pipeline",
                {"llm_id": "ignored", "tts_id": "ignored", "asr_id": "ignored", "thinker_llm_id": "ignored"},
                [
                    call("llm", ""),
                    call("tts", ""),
                    call("asr", ""),
                    call("thinker-llm", ""),
                    call("booking-server", ""),
                ],
            ),
        )

        class ConstructionObserved(Exception):
            pass

        for module_name, body, expected_calls in cases:
            with self.subTest(module=module_name):
                pipeline = import_module(module_name)
                default_loader = Mock(return_value={})
                exact_loader = Mock()
                runner_args = SimpleNamespace(
                    body=body,
                    session_id="",
                    handle_sigint=False,
                )
                with (
                    patch.object(pipeline, "create_transport", return_value=object()),
                    patch.object(pipeline, "load_service_entry", default_loader),
                    patch.object(pipeline, "load_selected_service_entry", exact_loader),
                    patch.object(pipeline, "NvidiaSTTService", side_effect=ConstructionObserved),
                    self.assertRaises(ConstructionObserved),
                ):
                    await pipeline.bot(runner_args)

                self.assertEqual(default_loader.call_args_list, expected_calls)
                exact_loader.assert_not_called()

    async def test_omni_rtvi_does_not_call_realtime_vad_lookup(self) -> None:
        from examples.omni_assistant import pipeline

        normal_transport = object()

        class ConstructionObserved(Exception):
            pass

        observed = {}

        def stop_construction(**kwargs):
            observed.update(kwargs)
            raise ConstructionObserved

        default_loader = Mock(return_value={})
        exact_loader = Mock()
        runner_args = SimpleNamespace(body={"llm_id": "ignored", "tts_id": "ignored", "max_tokens": "invalid"})
        with (
            patch.object(pipeline, "create_transport", return_value=normal_transport),
            patch.object(pipeline, "load_service_entry", default_loader),
            patch.object(pipeline, "load_selected_service_entry", exact_loader),
            patch.dict(os.environ, {"OMNI_MAX_TOKENS": "321"}),
            patch.object(pipeline, "NvidiaOmniLLMService", side_effect=stop_construction),
            patch.object(pipeline, "realtime_vad_prefix_padding_secs") as realtime_prefix_padding,
            self.assertRaises(ConstructionObserved),
        ):
            await pipeline.bot(runner_args)

        self.assertEqual(observed["settings"].max_tokens, 321)
        self.assertEqual(default_loader.call_args_list, [call("llm", ""), call("tts", "")])
        exact_loader.assert_not_called()
        realtime_prefix_padding.assert_not_called()

    async def test_omni_subagents_rtvi_does_not_call_realtime_vad_lookup(self) -> None:
        from examples.omni_assistant_subagents import pipeline

        class ConstructionObserved(Exception):
            pass

        observed = {}

        def stop_construction(**kwargs):
            observed.update(kwargs)
            raise ConstructionObserved

        transport_agent = SimpleNamespace(
            queue_media_analysis_prompt=AsyncMock(),
            has_uploaded_attachment=lambda: False,
            is_attachment_pending=lambda: False,
            queue_thinking=AsyncMock(),
            queue_highres_capture=AsyncMock(),
            current_visual_status=lambda: "",
        )
        default_loader = Mock(return_value={})
        exact_loader = Mock()
        runner_args = SimpleNamespace(
            body={"llm_id": "ignored", "tts_id": "ignored", "max_tokens": "invalid"},
            session_id="",
            handle_sigint=False,
        )
        with (
            patch.object(pipeline, "_create_transport", return_value=object()),
            patch.object(pipeline, "load_service_entry", default_loader),
            patch.object(pipeline, "load_selected_service_entry", exact_loader),
            patch.dict(os.environ, {"OMNI_MAX_TOKENS": "321"}),
            patch.object(pipeline, "WorkerRunner", return_value=SimpleNamespace(bus=object())),
            patch.object(pipeline, "OmniTransportAgent", return_value=transport_agent),
            patch.object(pipeline, "SpeakerOmniAgent", side_effect=stop_construction),
            patch.object(pipeline, "realtime_vad_prefix_padding_secs") as realtime_prefix_padding,
            self.assertRaises(ConstructionObserved),
        ):
            await pipeline.bot(runner_args)

        self.assertEqual(observed["max_tokens"], 321)
        self.assertEqual(default_loader.call_args_list, [call("llm", ""), call("tts", "")])
        exact_loader.assert_not_called()
        realtime_prefix_padding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
