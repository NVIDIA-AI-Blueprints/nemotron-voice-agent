# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared pipeline helpers used by all cascaded pipeline variants."""

import asyncio

from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.context.llm_context_summarization import (
    DEFAULT_SUMMARIZATION_PROMPT,
    LLMContextSummarizationUtil,
)

from utils import parse_env_bool, parse_env_float, parse_env_int

# Smart Turn silence fallback default (seconds); override via SMART_TURN_STOP_SECS.
# Pipecat's stock default is 3.0s.
SMART_TURN_FALLBACK_SECS = 1.0

# Magpie TTS (nemo-speech) accepts sample_rate_hz in [8000, 22050] or 0 (auto).
# Use Magpie's native max for output. Pipecat's default out rate (24000) is rejected.
PIPELINE_AUDIO_IN_SAMPLE_RATE = 16000
PIPELINE_AUDIO_OUT_SAMPLE_RATE = 22050


def build_pipeline_params(**kwargs) -> PipelineParams:
    """Build PipelineParams with Magpie-safe audio sample rates."""
    kwargs.setdefault("audio_in_sample_rate", PIPELINE_AUDIO_IN_SAMPLE_RATE)
    kwargs.setdefault("audio_out_sample_rate", PIPELINE_AUDIO_OUT_SAMPLE_RATE)
    return PipelineParams(**kwargs)


def build_smart_turn_analyzer() -> LocalSmartTurnAnalyzerV3:
    """Return LocalSmartTurnAnalyzerV3 with the configurable silence fallback."""
    stop_secs = parse_env_float("SMART_TURN_STOP_SECS", SMART_TURN_FALLBACK_SECS, min_value=0.0)
    return LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=stop_secs))


def build_smart_turn_stop_strategies() -> list[TurnAnalyzerUserTurnStopStrategy]:
    """Return the default Smart Turn stop strategy used by cascaded pipelines."""
    return [TurnAnalyzerUserTurnStopStrategy(turn_analyzer=build_smart_turn_analyzer())]


def build_user_mute_strategies(welcome_enabled: bool) -> list[MuteUntilFirstBotCompleteUserMuteStrategy]:
    """Return the user-mute strategy, or none when there is no welcome message.

    ``MuteUntilFirstBotCompleteUserMuteStrategy`` keeps the user muted until the
    bot finishes its first turn. When the welcome message is off the bot waits
    for the user, so that first turn never happens and muting would deadlock —
    return an empty list instead.
    """
    if not welcome_enabled:
        return []
    return [MuteUntilFirstBotCompleteUserMuteStrategy()]


def runner_protocol(runner_args: RunnerArguments) -> str:
    """Return the wire protocol for this session (``rtvi`` or ``realtime``)."""
    body = runner_args.body if isinstance(getattr(runner_args, "body", None), dict) else {}
    protocol = str(body.get("protocol") or "").strip().lower()
    return protocol if protocol else "rtvi"


def with_realtime_observers(*observers, transport=None) -> list:
    """Append the Realtime lifecycle observer when the transport speaks Realtime.

    Example::

        observers=with_realtime_observers(latency_observer, transport=transport)
    """
    out = list(observers)
    if transport is None:
        return out
    from realtime.transport import realtime_lifecycle_observer

    realtime_obs = realtime_lifecycle_observer(transport)
    if realtime_obs is not None:
        out.append(realtime_obs)
    return out


def register_session_start_handlers(
    *,
    transport,
    task,
    context,
    runner_args: RunnerArguments,
    intro_prompt: str = "Please introduce yourself to the user.",
    on_start=None,
    welcome_enabled: bool = True,
) -> None:
    """Start the session using the correct signal for the wire protocol.

    RTVI/WebRTC uses ``on_client_ready``; Realtime uses ``on_client_connected``.
    Both share the same optional ``on_start`` + welcome intro path. When welcome
    is off, skip the intro and (on Realtime) open the client text gate.
    """
    from pipecat.frames.frames import LLMRunFrame

    started = False

    async def _start_session(source: str) -> None:
        nonlocal started
        if started:
            return
        started = True
        logger.info(f"Client session start via {source}")
        if on_start is not None:
            await on_start()
        if not welcome_enabled:
            logger.info("Welcome message disabled; waiting for the user to speak first")
            return
        context.add_message({"role": "user", "content": intro_prompt})
        await task.queue_frames([LLMRunFrame()])

    if runner_protocol(runner_args) == "realtime":
        if not welcome_enabled:
            serializer = getattr(transport, "_realtime_serializer", None)
            conversation = getattr(serializer, "conversation", None)
            if conversation is not None:
                conversation.open_client_text()

        @transport.event_handler("on_client_connected")
        async def _on_realtime_connected(transport_obj, client):  # noqa: ARG001
            await _start_session("realtime-transport-connected")

    else:

        @task.rtvi.event_handler("on_client_ready")
        async def _on_rtvi_ready(rtvi):  # noqa: ARG001
            await _start_session("rtvi-client-ready")


def build_user_aggregator_params(
    welcome_enabled: bool, *, vad_stop_secs: float | None = None
) -> LLMUserAggregatorParams:
    """Return user-turn configuration with an optional VAD finalization delay."""
    default_stop_secs = 0.2 if vad_stop_secs is None else max(0.0, vad_stop_secs)
    if not parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False):
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=default_stop_secs)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(stop=build_smart_turn_stop_strategies()),
        )

    stop_secs = (
        parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0) if vad_stop_secs is None else default_stop_secs
    )
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_mute_strategies=build_user_mute_strategies(welcome_enabled),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
        ),
    )


def build_context_messages(base_prompt: str, system_prompt: str = "") -> list[dict]:
    """Build initial context messages.

    Branch on whether the service defines a ``system_prompt`` (services.yaml):
      * Some models (e.g. reasoning-control variants) require the system role
        to carry only a control directive and put all instructions in the
        user message. When a non-empty ``system_prompt`` is configured, the
        prompt catalog content is placed in a separate ``user`` message.
      * Nano / Super have an empty ``system_prompt``.  Their chat template
        appends tool definitions into the system section alongside whatever
        system content is there, so keeping the assistant instructions in
        the system role is both consistent with the template and preserves
        tool-calling reliability.
    """
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_prompt},
        ]
    return [{"role": "system", "content": base_prompt}]


def find_recent_turn_start(messages: list[dict], recent_turns: int) -> int:
    """Return the first message index for the last N user turns."""
    turns_seen = 0
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if isinstance(msg, dict) and msg.get("role") == "user":
            turns_seen += 1
            if turns_seen == recent_turns:
                return index
    return 0


async def generate_history_summary(
    *,
    llm: NvidiaLLMService,
    messages_to_summarize: list[dict],
    summary_system_prompt: str = "",
) -> str:
    """Generate a concise text summary for older chat messages."""
    transcript = LLMContextSummarizationUtil.format_messages_for_summary(messages_to_summarize)
    if not transcript.strip():
        raise ValueError("no transcript content available to summarize")

    summary_context = LLMContext(
        messages=[
            {
                "role": "user",
                "content": f"{DEFAULT_SUMMARIZATION_PROMPT}\n\nConversation history:\n{transcript}",
            }
        ]
    )
    summary_coro = llm.run_inference(
        summary_context,
        max_tokens=None,
        system_instruction=summary_system_prompt or None,
    )

    summary_text = await asyncio.wait_for(summary_coro, timeout=45)

    if not summary_text or not summary_text.strip():
        raise ValueError("LLM returned an empty summary")
    return summary_text.strip()


async def apply_pinned_prompt_summary(
    *,
    context: LLMContext,
    llm: NvidiaLLMService,
    preserve_prompt_messages: int,
    recent_turns: int,
    summary_system_prompt: str = "",
) -> None:
    """Summarize old chat turns while preserving the initial prompt messages."""
    if recent_turns < 1:
        return

    messages = list(context.get_messages())
    preserve_count = max(0, preserve_prompt_messages)
    recent_start = find_recent_turn_start(messages[preserve_count:], recent_turns)
    if recent_start <= 0:
        return

    pinned_messages = messages[:preserve_count]
    chat_messages = messages[preserve_count:]
    messages_to_summarize = chat_messages[:recent_start]
    recent_messages = chat_messages[recent_start:]

    if not messages_to_summarize:
        return

    try:
        summary_text = await generate_history_summary(
            llm=llm,
            messages_to_summarize=messages_to_summarize,
            summary_system_prompt=summary_system_prompt,
        )
    except Exception as exc:
        logger.warning(f"Chat history summarization failed; keeping existing context: {exc}")
        return

    if context.get_messages() != messages:
        logger.debug("Skipped applying chat history summary because context changed during summarization")
        return

    summary_message = {
        "role": "user",
        "content": f"Conversation summary of earlier turns: {summary_text}",
    }
    context.set_messages([*pinned_messages, summary_message, *recent_messages])
    logger.info(
        "Applied pinned prompt chat summary "
        f"(preserved={preserve_count}, summarized={len(messages_to_summarize)}, "
        f"recent={len(recent_messages)}, total={len(context.get_messages())})"
    )


def create_transport(runner_args: RunnerArguments):
    """Create a transport from runner arguments (WebRTC, WebSocket, Realtime, or eval)."""
    from pipecat.runner.types import EvalRunnerArguments, SmallWebRTCRunnerArguments

    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        return SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=PIPELINE_AUDIO_IN_SAMPLE_RATE,
                audio_out_enabled=True,
                audio_out_sample_rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE,
                audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 5),
            ),
            webrtc_connection=runner_args.webrtc_connection,
        )

    if isinstance(runner_args, EvalRunnerArguments):
        from pipecat.evals.serializer import RTVIEvalSerializer
        from pipecat.evals.transport import EvalTransport, EvalTransportParams

        return EvalTransport(
            params=EvalTransportParams(
                audio_in_enabled=True,
                audio_in_sample_rate=PIPELINE_AUDIO_IN_SAMPLE_RATE,
                audio_out_enabled=True,
                audio_out_sample_rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE,
                audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
                add_wav_header=False,
                serializer=RTVIEvalSerializer(),
            ),
            host=runner_args.host,
            port=runner_args.port,
        )

    websocket = getattr(runner_args, "websocket", None)
    if websocket is None:
        raise TypeError(f"Unsupported runner args type: {type(runner_args)}")

    body = runner_args.body if isinstance(getattr(runner_args, "body", None), dict) else {}
    if runner_protocol(runner_args) == "realtime":
        from realtime.transport import create_realtime_transport

        return create_realtime_transport(
            websocket,
            session_view=body.get("realtime_session_view")
            if isinstance(body.get("realtime_session_view"), dict)
            else None,
            runtime_config=body,
        )

    from pipecat.serializers.base_serializer import FrameSerializer
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=PIPELINE_AUDIO_IN_SAMPLE_RATE,
            audio_out_enabled=True,
            audio_out_sample_rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE,
            audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(params=FrameSerializer.InputParams(ignore_rtvi_messages=False)),
        ),
    )
