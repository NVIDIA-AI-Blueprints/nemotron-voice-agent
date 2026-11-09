# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one build of the frontend: the same pipeline for the repository UI bot and for a live session."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMContextAggregatorPair,
    LLMUserAggregator,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.empty_user_turn import EmptyUserTurnConfig

from examples.frontend_backend_live.media.preroll import SpeechPreroll
from examples.frontend_backend_live.media.services import speech_mode
from examples.frontend_backend_live.media.silent_lines import SilentLines
from examples.frontend_backend_live.voice import Voice

if TYPE_CHECKING:
    from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregatorParams

    from examples.frontend_backend_live.config_manager.schema import LiveConfig
    from examples.frontend_backend_live.display import DisplaySink
    from examples.frontend_backend_live.prompts import FixedLines, PromptSet
    from examples.frontend_backend_live.talker import TalkerChat
    from gateways.live.ports import DelegationPort

# A turn with no words brings no run of its own: a run still due is asked for again after this idle time.
IDLE_RUN_SECONDS = 0.5


@dataclass(frozen=True)
class FrontendEdges:
    """The host's part of the pipeline: its transport, what reads the call after the output, and its worker settings."""

    inputs: Sequence[FrameProcessor]
    outputs: Sequence[FrameProcessor]
    worker: Mapping[str, Any]
    after: Sequence[FrameProcessor] = ()

    def around(self, middle: Sequence[FrameProcessor], played: Sequence[FrameProcessor]) -> list[FrameProcessor]:
        """Return the processors: ``middle`` between the input and the output, then ``played`` and ``after``."""
        return [*self.inputs, *middle, *self.outputs, *played, *self.after]


@dataclass(frozen=True)
class FrontendSpec:
    """What a host hands the build from its own source: the call's settings, services, seed and delegation."""

    config: LiveConfig
    talker: TalkerChat
    prompts: PromptSet
    lines: FixedLines
    seed: list[dict]
    stt: FrameProcessor
    tts: FrameProcessor
    user_params: LLMUserAggregatorParams
    delegation: DelegationPort
    item_text: Callable[[dict], str]
    display: Callable[[PipelineWorker], DisplaySink] | None = None


@dataclass(frozen=True)
class Frontend:
    """The built frontend: the pipeline's worker and its processors in order, the voice, and the talker's context."""

    worker: PipelineWorker
    processors: list[FrameProcessor]
    voice: Voice
    context: LLMContext
    user_aggregator: LLMUserAggregator
    assistant_aggregator: LLMAssistantAggregator


def build_frontend(spec: FrontendSpec, edges: FrontendEdges) -> Frontend:
    """Build one call's frontend between ``edges``: its pipeline's worker and its voice; nothing has started yet."""
    mode = speech_mode(spec.tts)
    voice = Voice(
        config=spec.config,
        prompts=spec.prompts,
        lines=spec.lines,
        json_rules=spec.prompts.json_rules,
        empty_turn_note=spec.prompts.empty_turn_note,
        delegation=spec.delegation,
        item_text=spec.item_text,
        reports_sentences=mode.reports_sentences,
    )
    context = LLMContext(voice.seed(spec.seed))
    empty_turn = EmptyUserTurnConfig(interrupted_prompt=spec.prompts.empty_turn_note, idle_prompt=None)
    user_params = replace(spec.user_params, empty_user_turn=empty_turn, user_idle_timeout=IDLE_RUN_SECONDS)
    user, assistant = LLMContextAggregatorPair(context, user_params=user_params)
    voice.bind(spec.talker, context)
    middle = [spec.stt, user, spec.talker, spec.tts, *mode.text_hold(), SpeechPreroll(spec.config.tts_preroll_ms)]
    processors = [*edges.around(middle, [SilentLines(voice.spoken)]), assistant]
    worker = PipelineWorker(
        Pipeline(processors),
        **edges.worker,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        setup_timeout_secs=120.0,
    )
    if spec.display is not None:
        voice.show(spec.display(worker))
    voice.attach(user, assistant, worker)
    return Frontend(worker, processors, voice, context, user, assistant)
