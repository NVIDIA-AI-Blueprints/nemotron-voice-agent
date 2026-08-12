# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import yaml
from pipecat.processors.aggregators.llm_context import LLMContext

from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniInferenceResult,
    NvidiaOmniLLMService,
)
from examples.omni_assistant_subagents.pipeline import _agent_prompt_content, _expand_fragments
from examples.omni_assistant_subagents.subagents.speaker.action_envelope import (
    MEDIA_FIELD_PREFIXES,
    SpeakerTurnResult,
    lean_contract,
    normalize_action_envelope,
)
from examples.omni_assistant_subagents.subagents.speaker.agent import SubagentsSpeakerOmniService
from examples.omni_assistant_subagents.subagents.speaker.repeat_guard import BRIDGE_FILLERS, RepeatGuard
from examples.omni_assistant_subagents.subagents.thinker.agent import ThinkerWorker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_PATH = PROJECT_ROOT / "src/examples/omni_assistant_subagents/prompts.yaml"

# The turn parts an audio turn carries; a text turn carries none.
AUDIO_TURN_PARTS = [
    {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
    {"type": "text", "text": "contract"},
]


class ActionNormalizationTests(unittest.TestCase):
    def test_missing_action_is_inferred_from_single_structural_owner(self) -> None:
        payload, recovery = normalize_action_envelope(
            {"media_analysis_prompt": "Describe the upload"},
            transcript="What is in this image?",
            response="Taking a look.",
        )
        self.assertEqual(payload["turn_action"], "analyze_attachment")
        self.assertEqual(payload["media_analysis_prompt"], "Describe the upload")
        self.assertIn("inferred analyze_attachment", recovery)

    def test_response_only_envelope_infers_respond(self) -> None:
        payload, recovery = normalize_action_envelope(
            {},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(payload["turn_action"], "respond")
        self.assertNotIn("needs_thinking", payload)
        self.assertIn("inferred respond", recovery)

    def test_model_emitted_control_flags_are_dropped(self) -> None:
        payload, recovery = normalize_action_envelope(
            {"turn_action": "respond", "needs_thinking": True, "request_highres_capture": True},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(payload["turn_action"], "respond")
        self.assertNotIn("needs_thinking", payload)
        self.assertNotIn("request_highres_capture", payload)
        self.assertEqual(recovery, "")

    def test_explicit_actions_fill_safe_required_controls(self) -> None:
        media, _ = normalize_action_envelope(
            {"turn_action": "analyze_attachment"},
            transcript="What is in this image?",
            response="I will inspect it.",
        )
        self.assertEqual(media["selected_input_source"], "uploaded_attachment")
        self.assertEqual(media["media_analysis_action"], "new")
        self.assertEqual(media["media_analysis_prompt"], "What is in this image?")

        capture, _ = normalize_action_envelope(
            {"turn_action": "capture_highres"},
            transcript="Read the small label",
            response="Capturing it.",
        )
        self.assertEqual(capture["turn_action"], "capture_highres")
        self.assertEqual(capture["highres_query"], "Read the small label")

    def test_contradictory_owners_have_one_thinker_fallback(self) -> None:
        payload, recovery = normalize_action_envelope(
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
        direct, _ = normalize_action_envelope(
            {"turn_action": "respond"},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        self.assertEqual(direct["turn_action"], "respond")
        self.assertNotIn("needs_thinking", direct)
        self.assertEqual(direct["media_analysis_action"], "none")

        delegated, _ = normalize_action_envelope(
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
            payload, recovery = normalize_action_envelope(
                {"turn_action": action},
                transcript="Handle this request",
                response="",
            )
            self.assertEqual(payload["turn_action"], "think")
            self.assertTrue(payload["_action_fallback"])
            self.assertIn("missing its spoken response", recovery)

        capture, _ = normalize_action_envelope(
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
        lean = lean_contract(self.full)
        self.assertLess(self.full.index("- turn_action:"), self.full.index("- response:"))
        self.assertLess(lean.index("- turn_action:"), lean.index("- response:"))

    def test_lean_is_full_minus_only_the_media_field_lines(self) -> None:
        lean = lean_contract(self.full)
        dropped = [line for line in self.full.splitlines() if line not in lean.splitlines()]
        self.assertEqual(len(dropped), len(MEDIA_FIELD_PREFIXES))
        for line in dropped:
            self.assertTrue(line.strip().startswith(MEDIA_FIELD_PREFIXES))
        for prefix in MEDIA_FIELD_PREFIXES:
            self.assertNotIn(prefix, lean)

    def test_all_actions_present_and_behavior_lives_in_system_prompt(self) -> None:
        lean = lean_contract(self.full)
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

    def test_contract_omits_derived_control_booleans(self) -> None:
        for field in ("needs_thinking", "request_highres_capture"):
            self.assertNotIn(f"- {field}:", self.full)
            self.assertNotIn(f"- {field}:", lean_contract(self.full))

    def test_late_contradiction_is_not_a_successful_direct_result(self) -> None:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._active_turn_parts = AUDIO_TURN_PARTS
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
            )
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


class EnvelopeStreamingTests(unittest.IsolatedAsyncioTestCase):
    """The envelope parser layered over the service's reasoning-filtered stream."""

    def _service(self) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._media_analysis_prompt_handler = AsyncMock()
        service._uploaded_attachment_available = lambda: True
        service._attachment_pending = lambda: True
        service._visual_status_provider = lambda: "the camera is OFF right now"
        service._thinking_handler = AsyncMock()
        service._highres_capture_handler = AsyncMock()
        service._repeat = RepeatGuard()
        service._capture_cooldown = 0
        service._context = None
        service._active_turn_parts = AUDIO_TURN_PARTS
        service.run_inference = AsyncMock()
        service.push_frame = AsyncMock()
        self.transcripts: list[str] = []
        self.spoken: list[str] = []
        service._emit_user_transcript = AsyncMock(side_effect=lambda text: self.transcripts.append(text))
        service._push_llm_text = AsyncMock(side_effect=lambda text: self.spoken.append(text))
        return service

    @staticmethod
    def _chunks(envelope: dict, *, pieces: int = 6):
        raw = json.dumps(envelope)
        step = max(len(raw) // pieces, 1)
        return [raw[i : i + step] for i in range(0, len(raw), step)]

    async def _drain(self, service, envelope: dict) -> str:
        async def stream():
            for piece in self._chunks(envelope):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

        visible = ""
        async for chunk in service._stream_action_envelope(stream()):
            visible += chunk.choices[0].delta.content or ""
        return visible

    async def test_response_field_streams_and_transcript_is_emitted(self) -> None:
        service = self._service()
        visible = await self._drain(
            service,
            {
                "transcript": "Count one to five",
                "turn_action": "respond",
                "response": "One, two, three, four, five.",
                "selected_input_source": "none",
                "media_analysis_action": "none",
                "media_analysis_prompt": "",
                "highres_query": "",
            },
        )

        self.assertEqual(visible, "One, two, three, four, five.")
        self.assertEqual(self.transcripts, ["Count one to five"])
        # Already streamed, so the parsed envelope must not repeat it.
        self.assertEqual(self.spoken, [])

    async def test_streamed_repeat_is_replaced_before_reaching_tts(self) -> None:
        service = self._service()
        envelope = {
            "transcript": "White",
            "turn_action": "respond",
            "response": "Could you confirm what color the comb is?",
            "selected_input_source": "none",
            "media_analysis_action": "none",
            "media_analysis_prompt": "",
            "highres_query": "",
        }

        first = await self._drain(service, envelope)
        repeated = await self._drain(service, envelope)

        self.assertEqual(first, envelope["response"])
        self.assertIn(repeated, BRIDGE_FILLERS)
        self.assertNotIn(envelope["response"], repeated)
        service._thinking_handler.assert_awaited_once_with("White", "high", "repetition")

    async def test_invalid_action_withholds_streamed_text(self) -> None:
        service = self._service()
        visible = await self._drain(
            service,
            {
                "transcript": "Count one to five",
                "turn_action": "delegate",
                "response": "One, two, three, four, five.",
                "selected_input_source": "none",
                "media_analysis_action": "none",
                "media_analysis_prompt": "",
                "highres_query": "",
            },
        )

        self.assertEqual(visible, "")
        self.assertEqual(self.transcripts, ["Count one to five"])
        # Nothing reached TTS mid-stream, so the resolved envelope speaks instead.
        self.assertEqual(self.spoken, ["One, two, three, four, five."])

    async def test_reply_is_only_a_repeat_on_a_later_turn(self) -> None:
        service = self._service()
        service._repeat = RepeatGuard()
        envelope = {
            "transcript": "Count one to five",
            "turn_action": "delegate",
            "response": "One, two, three, four, five.",
            "selected_input_source": "none",
            "media_analysis_action": "none",
            "media_analysis_prompt": "",
            "highres_query": "",
        }
        await self._drain(service, envelope)
        self.assertEqual(self.spoken, ["One, two, three, four, five."])

        await self._drain(service, envelope)

        self.assertEqual(len(self.spoken), 2)
        self.assertIn(self.spoken[1], BRIDGE_FILLERS)

    async def test_streamed_deltas_reach_tts_with_word_spacing_intact(self) -> None:
        """Token deltas must not be trimmed: the space before a word rides on its delta."""
        service = self._service()
        response_pieces = ["I'm", " doing", " great,", " thank", " you!", " How", " can", " I", " help?"]
        pieces = [
            '{"transcript": "How are you?", "turn_action": "respond", "response": "',
            *response_pieces,
            '", "selected_input_source": "none", "media_analysis_action": "none", ',
            '"media_analysis_prompt": "", "highres_query": ""}',
        ]

        async def stream():
            for piece in pieces:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=piece))])

        del service._push_llm_text  # the real push path, not the collector installed by _service()
        spoken: list[str] = []
        with patch.object(NvidiaOmniLLMService, "_push_llm_text", AsyncMock(side_effect=spoken.append)):
            # Mirrors the base OpenAI service loop, which pushes each delta on its own.
            async for chunk in service._stream_action_envelope(stream()):
                content = chunk.choices[0].delta.content
                if content:
                    await service._push_llm_text(content)

        self.assertEqual("".join(spoken), "I'm doing great, thank you! How can I help?")

    async def test_envelope_json_never_leaks_into_speech(self) -> None:
        service = self._service()
        visible = await self._drain(
            service,
            {
                "transcript": "Look at the photo",
                "turn_action": "analyze_attachment",
                "response": "Sure, let me look at it.",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
                "media_analysis_prompt": "describe the photo",
                "highres_query": "",
            },
        )

        for token in ("turn_action", "selected_input_source", "{", "}"):
            self.assertNotIn(token, visible)
        self.assertEqual(visible, "Sure, let me look at it.")
        service._media_analysis_prompt_handler.assert_awaited_once()

    async def test_transcript_claimed_without_user_audio_never_enters_the_conversation(self) -> None:
        """A text turn has no speech, so a reported transcript is an echo, not something said."""
        service = self._service()
        service._active_turn_parts = None
        visible = await self._drain(
            service,
            {
                "transcript": "You are correcting your own structurally invalid Speaker output.",
                "turn_action": "respond",
                "response": "Hi there! I'm your NVIDIA voice assistant.",
                "selected_input_source": "none",
                "media_analysis_action": "none",
                "media_analysis_prompt": "",
                "highres_query": "",
            },
        )

        self.assertEqual(visible, "Hi there! I'm your NVIDIA voice assistant.")
        self.assertEqual(self.transcripts, [])

    async def test_audio_turn_still_reports_its_transcript(self) -> None:
        service = self._service()
        await self._drain(
            service,
            {
                "transcript": "Count one to five",
                "turn_action": "respond",
                "response": "One, two, three, four, five.",
                "selected_input_source": "none",
                "media_analysis_action": "none",
                "media_analysis_prompt": "",
                "highres_query": "",
            },
        )

        self.assertEqual(self.transcripts, ["Count one to five"])


class EnvelopeContractDeliveryTests(unittest.TestCase):
    """Every Speaker request must describe the envelope it is asked to produce."""

    def _service(self) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._audio_response_instruction_content = "Output one JSON object only, with these fields"
        service._visual_status_provider = None
        service._attachment_pending = None
        service._uploaded_attachment_available = None
        return service

    def _messages(self, service: SubagentsSpeakerOmniService) -> list[dict]:
        base = {"messages": [{"role": "system", "content": "identity"}]}
        with patch.object(NvidiaOmniLLMService, "build_chat_completion_params", return_value=base):
            return service.build_chat_completion_params({})["messages"]

    def test_turn_without_audio_carries_the_contract(self) -> None:
        service = self._service()
        service._active_turn_parts = None

        messages = self._messages(service)

        self.assertEqual(len(messages), 2)
        self.assertIn("Output one JSON object only", messages[-1]["content"][0]["text"])

    def test_audio_turn_is_left_alone(self) -> None:
        service = self._service()
        service._active_turn_parts = AUDIO_TURN_PARTS

        self.assertEqual(len(self._messages(service)), 1)


class LiveViewDeliveryTests(unittest.TestCase):
    """The live view has to travel with the turn, not only on the pinned board.

    A reply that already claimed to see something outweighs the board, and the model
    then repeats that claim with the camera off, so every turn restates the live view.
    """

    def _service(self, live_view: str | None) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._audio_response_instruction_content = "Output one JSON object only, with these fields"
        service._visual_status_provider = (lambda: live_view) if live_view is not None else None
        service._attachment_pending = None
        service._uploaded_attachment_available = None
        return service

    def test_camera_off_is_stated_beside_the_turn(self) -> None:
        service = self._service("the camera is OFF right now — there is nothing visible live")

        instruction = service._audio_response_instruction()

        self.assertIn("Live view right now: the camera is OFF right now", instruction)
        self.assertIn("Output one JSON object only", instruction)

    def test_live_scene_is_stated_beside_the_turn(self) -> None:
        service = self._service("a GoPro and a small tripod")

        self.assertIn("Live view right now: a GoPro and a small tripod.", service._audio_response_instruction())

    def test_turn_without_a_visual_source_carries_the_contract_alone(self) -> None:
        service = self._service(None)

        instruction = service._audio_response_instruction()

        self.assertNotIn("Live view right now", instruction)
        self.assertIn("Output one JSON object only", instruction)

    def test_unavailable_visual_source_is_not_described_as_a_view(self) -> None:
        def boom() -> str:
            raise RuntimeError("webcam controller gone")

        service = self._service(None)
        service._visual_status_provider = boom

        self.assertNotIn("Live view right now", service._audio_response_instruction())


class SpeakerHistoryOwnershipTests(unittest.IsolatedAsyncioTestCase):
    """The Speaker worker runs without an aggregator, so it owns its own history."""

    def _service(self) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._name = "SpeakerOmni"
        service._context = LLMContext([{"role": "system", "content": "identity"}])
        service._answered_transcript = ""
        service.push_frame = AsyncMock()
        return service

    async def test_the_spoken_turn_is_written_to_the_speaker_context(self) -> None:
        service = self._service()

        await service._emit_user_transcript("count one to five")

        self.assertEqual(
            [(m["role"], m["content"]) for m in service._context.get_messages()],
            [("system", "identity"), ("user", "count one to five")],
        )
        service.push_frame.assert_awaited_once()

    def test_the_speaker_is_not_announced_as_a_realtime_service(self) -> None:
        # The transport worker's assistant aggregator, which sees every frame
        # bridged out of this worker, has no user half to pair with.
        self.assertFalse(self._service().service_metadata_frame().is_realtime_service)


class DispatchRegressionTests(unittest.IsolatedAsyncioTestCase):
    def _service(self) -> SubagentsSpeakerOmniService:
        service = object.__new__(SubagentsSpeakerOmniService)
        service._media_analysis_prompt_handler = AsyncMock()
        service._uploaded_attachment_available = lambda: True
        service._attachment_pending = lambda: True
        service._visual_status_provider = lambda: "the camera is OFF right now"
        service._thinking_handler = AsyncMock()
        service._highres_capture_handler = AsyncMock()
        service._repeat = RepeatGuard()
        service._capture_cooldown = 0
        service._context = None
        service._active_turn_parts = AUDIO_TURN_PARTS
        service.run_inference = AsyncMock()
        service.push_frame = AsyncMock()
        return service

    @staticmethod
    def _unsafe_result() -> SpeakerTurnResult:
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
        return SpeakerTurnResult(
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
        corrected = await service._resolve_turn(self._unsafe_result())

        service.run_inference.assert_awaited_once()
        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.payload["turn_action"], "respond")
        self.assertEqual(corrected.response, "One, two, three, four, five.")
        self.assertEqual(service.push_frame.await_count, 0)
        # Held until the next turn starts, so it is never compared against itself.
        self.assertEqual(service._repeat._pending, "one two three four five")
        service._repeat.reset()
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
        corrected = await service._resolve_turn(self._unsafe_result())

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
        corrected = await service._resolve_turn(self._unsafe_result())

        self.assertIsNotNone(corrected)
        self.assertEqual(corrected.payload["turn_action"], "think")
        self.assertEqual(corrected.response, "Let me think that through carefully.")
        service.run_inference.assert_awaited_once()
        service._thinking_handler.assert_awaited_once()

    async def test_legitimate_attachment_acknowledgment_dispatches_exactly_once(self) -> None:
        service = self._service()
        payload, _ = normalize_action_envelope(
            {
                "turn_action": "analyze_attachment",
                "selected_input_source": "uploaded_attachment",
                "media_analysis_action": "new",
                "media_analysis_prompt": "Describe the image",
            },
            transcript="Describe this image",
            response="I am taking a look now.",
        )
        result = SpeakerTurnResult(
            transcript="Describe this image",
            response="I am taking a look now.",
            raw_content="{}",
            payload=payload,
        )
        await service._resolve_turn(result)
        service._media_analysis_prompt_handler.assert_awaited_once()
        service._thinking_handler.assert_not_awaited()
        service._highres_capture_handler.assert_not_awaited()

    async def test_highres_capture_dispatches_exactly_once(self) -> None:
        service = self._service()
        payload, _ = normalize_action_envelope(
            {"turn_action": "capture_highres", "highres_query": "Read the label"},
            transcript="Yes, read it",
            response="",
        )
        result = SpeakerTurnResult(
            transcript="Yes, read it",
            response="",
            raw_content="{}",
            payload=payload,
        )

        await service._resolve_turn(result)

        service._highres_capture_handler.assert_awaited_once_with("Read the label")
        service._media_analysis_prompt_handler.assert_not_awaited()
        service._thinking_handler.assert_not_awaited()

    async def test_respond_dispatches_no_async_work(self) -> None:
        service = self._service()
        payload, _ = normalize_action_envelope(
            {"turn_action": "respond"},
            transcript="Count one to five",
            response="One, two, three, four, five.",
        )
        result = SpeakerTurnResult(
            transcript="Count one to five",
            response="One, two, three, four, five.",
            raw_content="{}",
            payload=payload,
        )
        await service._resolve_turn(result)
        service._media_analysis_prompt_handler.assert_not_awaited()
        service._thinking_handler.assert_not_awaited()
        service._highres_capture_handler.assert_not_awaited()
        service.run_inference.assert_not_awaited()


class ThinkerBudgetTests(unittest.IsolatedAsyncioTestCase):
    def test_constructor_uses_total_generation_ceiling(self) -> None:
        with (
            patch("examples.omni_assistant_subagents.subagents.thinker.agent.NvidiaOmniLLMService") as omni_service,
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
