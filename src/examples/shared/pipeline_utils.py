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
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy, VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.context.llm_context_summarization import (
    DEFAULT_SUMMARIZATION_PROMPT,
    LLMContextSummarizationUtil,
)

from utils import parse_env_bool, parse_env_float, parse_env_int, resolve_prompt

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


def build_user_mute_strategies(
    welcome_enabled: bool,
    *,
    transport=None,
) -> list[MuteUntilFirstBotCompleteUserMuteStrategy]:
    """Return the user-mute strategy when the welcome turn produces speech.

    ``MuteUntilFirstBotCompleteUserMuteStrategy`` keeps the user muted until the
    bot emits ``BotStoppedSpeakingFrame``. When the welcome message is off the
    bot waits for the user, so that frame never arrives. A Realtime text-output
    response skips synthesis and likewise emits no bot-speech completion. In
    either case, installing the strategy would mute microphone turns indefinitely.
    """
    if not welcome_enabled:
        return []
    if transport is not None:
        from realtime.transport import realtime_controller

        controller = realtime_controller(transport)
        if controller is not None and controller.output_kind == "text":
            return []
    return [MuteUntilFirstBotCompleteUserMuteStrategy()]


def runner_protocol(runner_args: RunnerArguments) -> str:
    """Return the server-owned wire protocol for this session.

    Client request bodies are data, not protocol authority. Only the
    ``RealtimeSessionController`` installed by the Realtime server route can
    select that protocol; arbitrary ``protocol`` strings are ignored.
    """
    body = runner_args.body if isinstance(getattr(runner_args, "body", None), dict) else {}
    controller = body.get("realtime_controller")
    if controller is None:
        return "rtvi"

    from realtime.controller import RealtimeSessionController

    return "realtime" if isinstance(controller, RealtimeSessionController) else "rtvi"


def select_max_tokens_config(body: dict, default: object, *, is_realtime: bool) -> object:
    """Select a pipeline token limit without undoing Realtime ``inf``.

    The Realtime gateway projects the native ``"inf"`` value to ``None`` so
    the provider request omits both completion-token fields.  A normal
    ``body.get(...) or default`` expression would turn that intentional null
    back into the example catalog's finite cap.  Outside Realtime, retain the
    existing empty-value fallback behavior.
    """
    if is_realtime and "max_tokens" in body:
        selected = body["max_tokens"]
        return None if selected == "inf" else selected
    return body.get("max_tokens", "") or default


def resolve_pipeline_prompt(module_file, body: dict, *, is_realtime: bool) -> tuple[str, str]:
    """Resolve exact Realtime instructions or the ordinary catalog prompt."""
    if is_realtime and body.get("_realtime_instructions_explicit") is True:
        prompt_content = body.get("prompt_content")
        if not isinstance(prompt_content, str):
            raise ValueError("Explicit Realtime instructions must be a string")
        return "custom", prompt_content
    return resolve_prompt(
        module_file,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )


def with_realtime_observers(*observers, transport=None, is_realtime: bool) -> list:
    """Append the Realtime lifecycle observer when the transport speaks Realtime.

    Example::

        observers=with_realtime_observers(
            latency_observer,
            transport=transport,
            is_realtime=is_realtime,
        )
    """
    out = list(observers)
    if not is_realtime or transport is None:
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
    Both share optional ``on_start`` setup. Realtime never creates a response
    merely because the WebSocket connected; RTVI retains its configured welcome
    turn. When no welcome applies, wait for the first client event or user turn.
    """
    from pipecat.frames.frames import LLMConfigureOutputFrame, LLMRunFrame

    started = False
    is_realtime = runner_protocol(runner_args) == "realtime"

    async def _start_session(source: str) -> None:
        nonlocal started
        if started:
            return
        started = True
        logger.info(f"Client session start via {source}")
        if is_realtime:
            from realtime.transport import realtime_controller

            controller = realtime_controller(transport)
            if controller is None:
                raise RuntimeError("Realtime transport is missing its session controller")
            await task.queue_frames([LLMConfigureOutputFrame(skip_tts=controller.output_kind == "text")])
        if on_start is not None:
            await on_start()
        if is_realtime or not welcome_enabled:
            logger.info("Waiting for the first client event or user turn")
            return
        context.add_message({"role": "user", "content": intro_prompt})
        await task.queue_frames([LLMRunFrame()])

    if is_realtime:

        @transport.event_handler("on_client_connected")
        async def _on_realtime_connected(transport_obj, client):  # noqa: ARG001
            await _start_session("realtime-transport-connected")

    else:

        @task.rtvi.event_handler("on_client_ready")
        async def _on_rtvi_ready(rtvi):  # noqa: ARG001
            await _start_session("rtvi-client-ready")


def manual_realtime_user_aggregator_params(
    welcome_enabled: bool,
    *,
    transport,
) -> LLMUserAggregatorParams | None:
    """Build external turn control for client-committed Realtime audio."""
    from realtime.transport import realtime_manual_user_turn_strategies

    strategies = realtime_manual_user_turn_strategies(transport)
    if strategies is None:
        return None
    transcription_timeout = parse_env_float(
        "REALTIME_INPUT_TRANSCRIPTION_TIMEOUT_SECONDS",
        8.0,
        min_value=1.0,
    )
    return LLMUserAggregatorParams(
        vad_analyzer=None,
        user_mute_strategies=build_user_mute_strategies(welcome_enabled, transport=transport),
        user_turn_strategies=strategies,
        user_turn_stop_timeout=transcription_timeout + 1.0,
    )


def realtime_turn_detection_config(*, transport=None) -> dict | None:
    """Return the negotiated Realtime turn configuration, if applicable."""
    if transport is None:
        return None
    from realtime.transport import realtime_controller

    controller = realtime_controller(transport)
    return controller.turn_detection_config if controller is not None else None


def uses_server_vad_turn_detection(*, transport=None) -> bool:
    """Select server VAD from the Realtime session or the supported RTVI setting."""
    config = realtime_turn_detection_config(transport=transport)
    if config is not None:
        return config.get("type") == "server_vad"
    return parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False)


def build_vad_params(defaults: VADParams, *, transport=None) -> VADParams:
    """Apply exact negotiated server-VAD analyzer controls to defaults."""
    config = realtime_turn_detection_config(transport=transport)
    if config is None or config.get("type") != "server_vad":
        return defaults
    return VADParams(
        confidence=float(config.get("threshold", defaults.confidence)),
        start_secs=defaults.start_secs,
        stop_secs=float(config.get("silence_duration_ms", defaults.stop_secs * 1000)) / 1000,
        min_volume=defaults.min_volume,
    )


def realtime_vad_prefix_padding_secs(default_secs: float, *, transport=None) -> float:
    """Use negotiated server-VAD prefix audio for fused speech buffering."""
    if transport is None:
        return default_secs
    from realtime.transport import realtime_controller

    controller = realtime_controller(transport)
    if controller is None or controller.turn_detection_type != "server_vad":
        return default_secs
    return controller.server_vad_prefix_padding_ms / 1000


def build_vad_user_turn_start_strategies(
    *,
    transport=None,
    include_transcription: bool,
) -> list:
    """Build VAD-backed start strategies with the negotiated interruption policy."""
    config = realtime_turn_detection_config(transport=transport)
    interrupt_response = config is None or config.get("interrupt_response", True) is True
    strategies = [VADUserTurnStartStrategy(enable_interruptions=interrupt_response)]
    if include_transcription:
        strategies.append(TranscriptionUserTurnStartStrategy(enable_interruptions=interrupt_response))
    return strategies


def bind_realtime_automatic_response_provenance(
    strategies: list[BaseUserTurnStopStrategy],
    *,
    transport=None,
) -> list[BaseUserTurnStopStrategy]:
    """Bind typed VAD/user-turn provenance before Pipecat registers its handler."""
    if transport is not None:
        from realtime.transport import bind_realtime_automatic_response_provenance as bind_provenance

        bind_provenance(transport, strategies)
    return strategies


def build_user_aggregator_params(
    welcome_enabled: bool,
    *,
    transport=None,
) -> LLMUserAggregatorParams:
    """Return user-turn configuration, defaulting to Pipecat smart turn."""
    if transport is not None:
        manual = manual_realtime_user_aggregator_params(
            welcome_enabled,
            transport=transport,
        )
        if manual is not None:
            return manual
    start_strategies = build_vad_user_turn_start_strategies(
        transport=transport,
        include_transcription=True,
    )
    if not uses_server_vad_turn_detection(transport=transport):
        stop_strategies = bind_realtime_automatic_response_provenance(
            build_smart_turn_stop_strategies(),
            transport=transport,
        )
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=build_vad_params(VADParams(stop_secs=0.2), transport=transport)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled, transport=transport),
            user_turn_strategies=UserTurnStrategies(
                start=start_strategies,
                stop=stop_strategies,
            ),
        )

    stop_secs = parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0)
    stop_strategies = bind_realtime_automatic_response_provenance(
        [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
        transport=transport,
    )
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=build_vad_params(VADParams(stop_secs=stop_secs), transport=transport)),
        user_mute_strategies=build_user_mute_strategies(welcome_enabled, transport=transport),
        user_turn_strategies=UserTurnStrategies(
            start=start_strategies,
            stop=stop_strategies,
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
        from realtime.controller import RealtimeSessionController
        from realtime.transport import create_realtime_transport

        controller = body.get("realtime_controller")
        if not isinstance(controller, RealtimeSessionController):
            raise TypeError("Realtime runner body requires a RealtimeSessionController")
        return create_realtime_transport(
            websocket,
            controller=controller,
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
