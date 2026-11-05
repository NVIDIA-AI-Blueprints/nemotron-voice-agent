# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one build of the frontend: the same pipeline for the repository UI bot and for a live session.

:func:`build_frontend` builds generic's chain between a host's edges: speech-to-text, the user aggregator, the
talker (Pipecat's LLM service with the voice's decision, ``examples.frontend_backend_live.talker``), text-to-speech
with its mode's text hold, the pre-roll, the host's output, what reads each line after the output
(``examples.frontend_backend_live.media.silent_lines.SilentLines``), and the assistant aggregator, which records only
what played. The voice (``examples.frontend_backend_live.voice``) chooses each run of the talker and says what the
backend sends. Only these differ by host:

* the edges (:class:`FrontendEdges`): the UI bot's RTVI transport, with its audio recorder and its worker's RTVI and
  latency observer, or a live session's input gate and WebRTC peer or primary WebSocket, with its output transcript
  and its worker's event observer (``examples.frontend_backend_live.adapter.pipeline_host``);
* the start: the shared start handler with the bot's greeting (``examples.frontend_backend_live.pipeline``), or the
  gateway's session start (``examples.frontend_backend_live.adapter.adapter``);
* the delegation port: the bot's own core, or the session's delegation part;
* the display sink: the bot's task cards, or none.

The text-to-speech speaks in its catalog entry's synthesis mode, which this example's settings in
``examples_registry.yaml`` set to ``per_sentence``, and ``tts_preroll_ms`` sets the pre-roll in both hosts. Each host
builds its own speech services and talker from its own source: the session body, or the session's startup settings.
"""

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
    from examples.frontend_backend_live.spoken import CutMemory
    from examples.frontend_backend_live.talker import TalkerChat
    from examples.frontend_backend_live.voice import ResultWording
    from gateways.live.ports import DelegationPort

SETUP_TIMEOUT_SECONDS = 120.0  # the time each processor gets to set up, as in the repository's other examples
# How soon after the bot spoke, or after a caller turn with no words, a run still due is asked for again: a turn with
# no words brings no run of its own.
IDLE_RUN_SECONDS = 0.5


@dataclass(frozen=True)
class FrontendEdges:
    """What a host puts around the frontend: its transport, what reads the call after the output, its worker settings.

    ``worker`` holds the transport's own ``PipelineWorker`` settings: its parameters, observers, idle timeout, RTVI and
    tracing.
    """

    inputs: Sequence[FrameProcessor]  # where caller audio enters, before the speech-to-text
    outputs: Sequence[FrameProcessor]  # where reply audio leaves, after the pre-roll
    worker: Mapping[str, Any]
    after: Sequence[FrameProcessor] = ()  # what reads each played line: a live client's transcript, the audio recorder

    def around(self, middle: Sequence[FrameProcessor], played: Sequence[FrameProcessor]) -> list[FrameProcessor]:
        """The pipeline's processors: ``middle`` between the input and the output, then ``played`` and ``after``."""
        return [*self.inputs, *middle, *self.outputs, *played, *self.after]


@dataclass(frozen=True)
class FrontendSpec:
    """What a host hands the build from its own source: the call's settings, services, seed and delegation."""

    config: LiveConfig  # the validated frontend.cascade block
    talker: TalkerChat  # Pipecat's LLM service on the talker's model endpoint
    prompts: PromptSet  # the session's read of the prompt file: the prompt set
    lines: FixedLines  # and the voice's fixed lines
    seed: list[dict]  # the talker's instructions (system and developer messages), then any history
    stt: FrameProcessor
    tts: FrameProcessor
    user_params: LLMUserAggregatorParams  # the turn end's settings, with the host's welcome
    delegation: DelegationPort  # the bot's own core, or the session's delegation part
    wording: type[ResultWording]  # delegation.result_wording's class
    cut_memory: type[CutMemory]  # speech.heard_words' class
    item_text: Callable[[dict], str]  # a message item's text: the delegation core's reader, from the host's root
    # Where each delegation's display part goes, made on the pipeline's worker; nothing is shown by default.
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
    """Build one call's frontend between ``edges``: its pipeline's worker and its voice; nothing has started yet.

    A caller turn with no words that cut a line runs the talker with the prompt set's ``resume_note``
    (``speech.resume_after_noise``); with the resume off, such a turn runs nothing. A processor that cannot be used
    ends the call; the talker stays usable, since the voice answers a failed call. The host starts the voice and runs
    the worker.
    """
    mode = speech_mode(spec.tts)
    voice = Voice(
        config=spec.config,
        prompts=spec.prompts,
        lines=spec.lines,
        json_rules=spec.prompts.json_rules,
        resume_note=spec.prompts.resume_note,
        delegation=spec.delegation,
        wording=spec.wording(),
        cut_memory=spec.cut_memory.from_settings(spec.config.speech),
        item_text=spec.item_text,
        reports_sentences=mode.reports_sentences,
    )
    context = LLMContext(voice.seed(spec.seed))
    resume = EmptyUserTurnConfig(interrupted_prompt=spec.prompts.resume_note, idle_prompt=None)
    user_params = replace(
        spec.user_params,
        empty_user_turn=resume if spec.config.speech.resume_after_noise else None,
        user_idle_timeout=IDLE_RUN_SECONDS,
    )
    user, assistant = LLMContextAggregatorPair(context, user_params=user_params)
    voice.bind(spec.talker, context)
    middle = [spec.stt, user, spec.talker, spec.tts, *mode.text_hold(), SpeechPreroll(spec.config.tts_preroll_ms)]
    processors = [*edges.around(middle, [SilentLines(voice.spoken)]), assistant]
    worker = PipelineWorker(
        Pipeline(processors),
        **edges.worker,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        setup_timeout_secs=SETUP_TIMEOUT_SECONDS,
    )
    if spec.display is not None:
        voice.show(spec.display(worker))
    voice.attach(user, assistant, worker)
    return Frontend(worker, processors, voice, context, user, assistant)
