# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import json
import re
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import yaml

from examples.omni_assistant.nvidia_omni_multimodal_service import NvidiaOmniInferenceResult, NvidiaOmniTurnResult
from examples.omni_assistant_subagents.pipeline import _agent_prompt_content, _expand_fragments
from examples.omni_assistant_subagents.subagents.speaker.agent import (
    _MEDIA_FIELD_PREFIXES,
    SubagentsSpeakerOmniService,
    _lean_contract,
    _normalize_action_envelope,
    _RepeatGuard,
)
from examples.omni_assistant_subagents.subagents.thinker.agent import ThinkerWorker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_PATH = PROJECT_ROOT / "src/examples/omni_assistant_subagents/prompts.yaml"


class ActionNormalizationTests(unittest.TestCase):
    def test_missing_action_is_inferred_from_single_structural_owner(self) -> None:
        payload, recovery = _normalize_action_envelope(
            {"media_analysis_prompt": "Describe the upload"},
            transcript="What is in this image?",
            response="Taking a look.",
        )
        self.assertEqual(payload["turn_action"], "analyze_attachment")
        self.assertEqual(payload["media_analysis_prompt"], "Describe the upload")
        self.assertIn("inferred analyze_attachment", recovery)

    def test_response_only_envelope_infers_respond(self) -> None:
        payload, recovery = _normalize_action_envelope(
            {},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(payload["turn_action"], "respond")
        self.assertNotIn("needs_thinking", payload)
        self.assertIn("inferred respond", recovery)

    def test_model_emitted_control_flags_are_dropped(self) -> None:
        payload, recovery = _normalize_action_envelope(
            {"turn_action": "respond", "needs_thinking": True, "request_highres_capture": True},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(payload["turn_action"], "respond")
        self.assertNotIn("needs_thinking", payload)
        self.assertNotIn("request_highres_capture", payload)
        self.assertEqual(recovery, "")

    def test_explicit_actions_fill_safe_required_controls(self) -> None:
        media, _ = _normalize_action_envelope(
            {"turn_action": "analyze_attachment"},
            transcript="What is in this image?",
            response="I will inspect it.",
        )
        self.assertEqual(media["selected_input_source"], "uploaded_attachment")
        self.assertEqual(media["media_analysis_action"], "new")
        self.assertEqual(media["media_analysis_prompt"], "What is in this image?")

        capture, _ = _normalize_action_envelope(
            {"turn_action": "capture_highres"},
            transcript="Read the small label",
            response="Capturing it.",
        )
        self.assertEqual(capture["turn_action"], "capture_highres")
        self.assertEqual(capture["highres_query"], "Read the small label")

    def test_contradictory_owners_have_one_thinker_fallback(self) -> None:
        payload, recovery = _normalize_action_envelope(
            {
                "turn_action": "respond",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
            },
            transcript="Tell me a story",
            response="I will do that.",
        )
        self.assertEqual(payload["turn_action"], "think")
        self.assertTrue(payload["_action_fallback"])
        self.assertEqual(payload["media_analysis_action"], "none")
        self.assertIn("contradicted", recovery)

    def test_respond_never_queues_work_but_async_action_keeps_acknowledgment(self) -> None:
        direct, _ = _normalize_action_envelope(
            {"turn_action": "respond"},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(direct["turn_action"], "respond")
        self.assertNotIn("needs_thinking", direct)
        self.assertEqual(direct["media_analysis_action"], "none")

        delegated, _ = _normalize_action_envelope(
            {
                "turn_action": "analyze_attachment",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
                "media_analysis_prompt": "Describe the upload",
            },
            transcript="Describe this upload",
            response="I am taking a look now.",
        )
        self.assertEqual(delegated["turn_action"], "analyze_attachment")
        self.assertEqual(delegated["media_analysis_action"], "new")

    def test_deferred_actions_require_spoken_acknowledgment(self) -> None:
        for action in ("think", "analyze_attachment"):
            payload, recovery = _normalize_action_envelope(
                {"turn_action": action},
                transcript="Handle this request",
                response="",
            )
            self.assertEqual(payload["turn_action"], "think")
            self.assertTrue(payload["_action_fallback"])
            self.assertIn("missing its spoken response", recovery)

        capture, _ = _normalize_action_envelope(
            {"turn_action": "capture_highres"},
            transcript="Read the label",
            response="",
        )
        self.assertEqual(capture["turn_action"], "capture_highres")


class PromptAndStreamingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))
        cls.full = cls.catalog["agent_prompts"]["SpeakerAgent"]["audio_response_instruction"]["content"]

    def test_yaml_is_valid_and_action_precedes_response(self) -> None:
        lean = _lean_contract(self.full)
        self.assertLess(self.full.index("- turn_action:"), self.full.index("- response:"))
        self.assertLess(lean.index("- turn_action:"), lean.index("- response:"))

    def test_lean_is_full_minus_only_the_media_field_lines(self) -> None:
        lean = _lean_contract(self.full)
        dropped = [line for line in self.full.splitlines() if line not in lean.splitlines()]
        self.assertEqual(len(dropped), len(_MEDIA_FIELD_PREFIXES))
        for line in dropped:
            self.assertTrue(line.strip().startswith(_MEDIA_FIELD_PREFIXES))
        for prefix in _MEDIA_FIELD_PREFIXES:
            self.assertNotIn(prefix, lean)

    def test_all_actions_present_and_behavior_lives_in_system_prompt(self) -> None:
        lean = _lean_contract(self.full)
        for action in ("respond", "think", "analyze_attachment", "capture_highres", "clarify"):
            self.assertIn(action, self.full)
            self.assertIn(action, lean)
        system = _expand_fragments(self.catalog["generic_omni_assistant"]["content"], self.catalog)
        self.assertIn("ten-sentence story", system)
        self.assertIn("one, two, three, four, five", system)
        self.assertIn("What would you like help with?", system)
        self.assertIn("the camera is ON", system)

    def test_catalog_prompts_have_no_unresolved_fragments(self) -> None:
        contents = [self.catalog["generic_omni_assistant"]["content"]]
        for prompts in self.catalog["agent_prompts"].values():
            contents.extend(prompt["content"] for prompt in prompts.values())
        for content in contents:
            self.assertIsNone(re.search(r"\{\{\w+\}\}", _expand_fragments(content, self.catalog)))

    def test_streaming_waits_only_for_valid_action(self) -> None:
        service = object.__new__(SubagentsSpeakerOmniService)
        self.assertEqual(service._structured_response_control_fields(), ("turn_action",))
        self.assertFalse(service._should_emit_streamed_structured_response({}))
        self.assertFalse(service._should_emit_streamed_structured_response({"turn_action": "res"}))
        self.assertFalse(service._should_emit_streamed_structured_response({"turn_action": "delegate"}))
        self.assertTrue(service._should_emit_streamed_structured_response({"turn_action": "respond"}))
        self.assertTrue(service._should_emit_streamed_structured_response({"turn_action": "capture_highres"}))

    def test_contract_omits_derived_control_booleans(self) -> None:
        for field in ("needs_thinking", "request_highres_capture"):
            self.assertNotIn(f"- {field}:", self.full)
            self.assertNotIn(f"- {field}:", _lean_contract(self.full))

    def test_late_contradiction_is_not_a_successful_direct_result(self) -> None:
        service = object.__new__(SubagentsSpeakerOmniService)
        result = service._parse_turn_result(
            json.dumps(
                {
                    "transcript": "Count one to five",
                    "turn_action": "respond",
                    "response": "I will do that.",
                    "selected_input_source": "uploaded_attachment",
                    "media_analysis_action": "new",
                    "media_analysis_prompt": "",
                    "highres_query": "",
                    "webcam_focus": "",
                }
            ),
            parse_json=True,
        )
        self.assertEqual(result.response, "")
        self.assertTrue(result.payload["_action_fallback"])
        self.assertEqual(result.payload["turn_action"], "think")


class PromptFragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = yaml.safe_load(PROMPTS_PATH.read_text(encoding="utf-8"))

    def test_shared_fragments_exist(self) -> None:
        shared = self.catalog["shared"]
        self.assertIn("spoken_format", shared)
        self.assertIn("visual_sources", shared)

    def test_all_placeholders_resolve_with_no_leftovers(self) -> None:
        generic = _expand_fragments(self.catalog["generic_omni_assistant"]["content"], self.catalog)
        thinker = _agent_prompt_content(self.catalog, "ThinkerAgent", "thinking_system_prompt")
        media = _agent_prompt_content(self.catalog, "MediaAnalyzerAgent", "analysis_system_prompt")
        for expanded in (generic, thinker, media):
            self.assertNotIn("{{", expanded)
            self.assertNotIn("}}", expanded)
        self.assertIn("different sources", generic)
        self.assertIn("no markdown", thinker)
        self.assertIn("different sources", thinker)
        self.assertIn("no markdown", media)

    def test_contract_stays_placeholder_free(self) -> None:
        contract = _agent_prompt_content(self.catalog, "SpeakerAgent", "audio_response_instruction")
        self.assertNotIn("{{", contract)


class DispatchRegressionTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._media_analysis_prompt_handler = AsyncMock()
        service._uploaded_attachment_available = lambda: True
        service._attachment_pending = lambda: True
        service._thinking_handler = AsyncMock()
        service._highres_capture_handler = AsyncMock()
        service._repeat = _RepeatGuard()
        service._capture_cooldown = 0
        service._context = None
        service.run_inference = AsyncMock()
        service.push_frame = AsyncMock()
        return service

    @staticmethod
    def _unsafe_result() -> NvidiaOmniTurnResult:
        raw = json.dumps(
            {
                "transcript": "Count one to five",
                "turn_action": "respond",
                "response": "I will do that.",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
                "media_analysis_prompt": "",
                "highres_query": "",
                "webcam_focus": "",
            }
        )
        return NvidiaOmniTurnResult(
            transcript="Count one to five",
            response="",
            raw_content=raw,
            payload={
                "transcript": "Count one to five",
                "turn_action": "think",
                "response": "",
                "selected_input_source": "none",
                "media_analysis_action": "none",
                "media_analysis_prompt": "",
                "highres_query": "",
                "_action_fallback": True,
                "_action_recovery": "turn_action respond contradicted arguments for analyze_attachment",
            },
        )

    async def test_unsafe_envelope_gets_exactly_one_successful_speaker_correction(self) -> None:
        service = self._service()
        service.run_inference = AsyncMock(
            return_value=json.dumps(
                {
                    "transcript": "Count one to five",
                    "turn_action": "respond",
                    "response": "One, two, three, four, five.",
                    "selected_input_source": "none",
                    "media_analysis_action": "none",
                    "media_analysis_prompt": "",
                    "highres_query": "",
                    "webcam_focus": "",
                }
            )
        )
        corrected = await service._on_turn_result(self._unsafe_result())

        service.run_inference.assert_awaited_once()
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.payload["turn_action"], "respond")
        self.assertEqual(corrected.response, "One, two, three, four, five.")
        self.assertEqual(service.push_frame.await_count, 0)
        self.assertIn("one two three four five", service._repeat._recent)
        service._thinking_handler.assert_not_awaited()

    async def test_failed_correction_falls_back_to_thinker_without_retrying(self) -> None:
        service = self._service()
        service.run_inference = AsyncMock(
            return_value=json.dumps(
                {
                    "transcript": "Count one to five",
                    "turn_action": "respond",
                    "response": "I will do that.",
                    "selected_input_source": "uploaded_attachment",
                    "media_analysis_action": "new",
                    "media_analysis_prompt": "",
                    "highres_query": "",
                    "webcam_focus": "",
                }
            )
        )
        corrected = await service._on_turn_result(self._unsafe_result())

        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.payload["turn_action"], "think")
        self.assertEqual(corrected.response, "Let me think that through carefully.")
        service.run_inference.assert_awaited_once()
        self.assertEqual(service.push_frame.await_count, 0)
        service._thinking_handler.assert_awaited_once_with(
            "Count one to five",
            "medium",
            "",
        )
        service._media_analysis_prompt_handler.assert_not_awaited()
        service._highres_capture_handler.assert_not_awaited()

    async def test_think_correction_is_rejected_and_defers_to_thinker(self) -> None:
        service = self._service()
        service.run_inference = AsyncMock(
            return_value=json.dumps(
                {
                    "transcript": "Count one to five",
                    "turn_action": "think",
                    "response": "Let me think about that.",
                    "selected_input_source": "none",
                    "media_analysis_action": "none",
                    "media_analysis_prompt": "",
                    "highres_query": "",
                    "webcam_focus": "",
                }
            )
        )
        corrected = await service._on_turn_result(self._unsafe_result())

        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.payload["turn_action"], "think")
        self.assertEqual(corrected.response, "Let me think that through carefully.")
        service.run_inference.assert_awaited_once()
        service._thinking_handler.assert_awaited_once()

    async def test_legitimate_attachment_acknowledgment_dispatches_exactly_once(self) -> None:
        service = self._service()
        payload, _ = _normalize_action_envelope(
            {
                "turn_action": "analyze_attachment",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
                "media_analysis_prompt": "Describe the image",
            },
            transcript="Describe this image",
            response="I am taking a look now.",
        )
        result = NvidiaOmniTurnResult(
            transcript="Describe this image",
            response="I am taking a look now.",
            raw_content="{}",
            payload=payload,
        )
        await service._on_turn_result(result)
        service._media_analysis_prompt_handler.assert_awaited_once()
        service._thinking_handler.assert_not_awaited()
        service._highres_capture_handler.assert_not_awaited()

    async def test_highres_capture_dispatches_exactly_once(self) -> None:
        service = self._service()
        payload, _ = _normalize_action_envelope(
            {"turn_action": "capture_highres", "highres_query": "Read the label"},
            transcript="Yes, read it",
            response="",
        )
        result = NvidiaOmniTurnResult(
            transcript="Yes, read it",
            response="",
            raw_content="{}",
            payload=payload,
        )

        await service._on_turn_result(result)

        service._highres_capture_handler.assert_awaited_once_with("Read the label")
        service._media_analysis_prompt_handler.assert_not_awaited()
        service._thinking_handler.assert_not_awaited()

    async def test_respond_dispatches_no_async_work(self) -> None:
        service = self._service()
        payload, _ = _normalize_action_envelope(
            {"turn_action": "respond"},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        result = NvidiaOmniTurnResult(
            transcript="Count one to five",
            response="One, two, three, four, five.",
            raw_content="{}",
            payload=payload,
        )
        await service._on_turn_result(result)
        service._media_analysis_prompt_handler.assert_not_awaited()
        service._thinking_handler.assert_not_awaited()
        service._highres_capture_handler.assert_not_awaited()
        service.run_inference.assert_not_awaited()


class ThinkerBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_constructor_uses_total_generation_ceiling(self) -> None:
        with (
            patch("examples.omni_assistant_subagents.subagents.thinker.agent.NvidiaOmniService") as omni_service,
            patch("examples.omni_assistant_subagents.subagents.thinker.agent.parse_env_float", return_value=0.6),
            patch("examples.omni_assistant_subagents.subagents.thinker.agent.parse_env_int", return_value=16384),
        ):
            worker = ThinkerWorker(
                api_key="test",
                base_url="http://localhost:8002/v1",
                model_id="test-model",
            )

        self.assertEqual(worker._max_tokens, 16384)
        settings = omni_service.call_args.kwargs["settings"]
        self.assertEqual(settings.max_tokens, 16384)

    async def test_effort_controls_reasoning_budget_not_total_tokens(self) -> None:
        worker = object.__new__(ThinkerWorker)
        worker._base_url = "http://localhost:8002/v1"
        worker._model_id = "test-model"
        worker._system_prompt = "Complete the requested answer."
        worker._temperature = 0.6
        worker._max_tokens = 16384
        worker._omni = AsyncMock()
        worker._omni.run_multimodal_inference.return_value = NvidiaOmniInferenceResult(
            text="One, two, three, four, five.",
            reasoning="Counted the requested sequence.",
            finish_reason="stop",
        )

        answer, reasoning = await worker._think(
            "",
            "Count one to five",
            "",
            1024,
            requester="omni_transport",
            task_id="task-1",
        )

        self.assertEqual(answer, "One, two, three, four, five.")
        self.assertEqual(reasoning, "Counted the requested sequence.")
        worker._omni.run_multimodal_inference.assert_awaited_once()
        kwargs = worker._omni.run_multimodal_inference.await_args.kwargs
        self.assertEqual(kwargs["max_tokens"], 16384)
        self.assertEqual(kwargs["reasoning_budget"], 1024)
        self.assertIn("on_reasoning_delta", kwargs)


if __name__ == "__main__":
    unittest.main()
