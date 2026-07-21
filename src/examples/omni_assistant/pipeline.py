# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Nemotron Omni cascaded pipeline using the upstream-style Omni service.

This is the current experimental pipeline for the clean
``NvidiaOmniMultimodalService`` shape:

* ``transport.input`` + VAD/user-turn processing feed audio into Omni.
* Omni replaces ASR + LLM and emits standard Pipecat frames.
* NVIDIA TTS speaks the emitted ``LLMTextFrame`` response.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMRunFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from examples.omni_assistant.audio_only_smart_turn_strategy import AudioOnlySmartTurnStopStrategy
from examples.omni_assistant.nvidia_omni_multimodal_service import (
    NvidiaOmniService,
    NvidiaOmniSettings,
)
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_service_entry,
    normalize_lang_code,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
)

load_dotenv(override=True)


def _build_user_turn_strategies() -> UserTurnStrategies:
    """Build VAD-start + Smart Turn-stop strategies for Omni audio turns."""
    return UserTurnStrategies(
        start=[VADUserTurnStartStrategy()],
        stop=[
            AudioOnlySmartTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=0.7))
            )
        ],
    )


def _build_user_turn_processor() -> UserTurnProcessor:
    """Build an external turn processor for subagent branches that need one."""
    return UserTurnProcessor(user_turn_strategies=_build_user_turn_strategies())


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Nemotron Omni cascaded pipeline for one session."""
    transport = _create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}

    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    logger.info(f"Starting Nemotron Omni cascaded pipeline (prompt={prompt_key})")

    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")

    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt_override = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        "extra_params",
    )

    # Build the conversation context up-front. With emit_transcriptions enabled,
    # Omni emits TranscriptionFrame for the user side while the assistant
    # aggregator commits LLMTextFrame output as usual.
    system_content = base_system_content
    if system_prompt_override:
        system_content = f"{base_system_content}\n\n{system_prompt_override}".strip()
    context = LLMContext([{"role": "system", "content": system_content}])

    emit_transcriptions = parse_env_bool("OMNI_EMIT_TRANSCRIPTIONS", default=True)
    omni = NvidiaOmniService(
        # Name carries "llm" so metrics consumers (UI metric-group, perf
        # benchmark) attribute Omni's TTFB/processing/token-usage metrics to the
        # LLM stage. Omni fuses ASR+LLM, so these are the pipeline's LLM metrics.
        name="NemotronOmniLLM",
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=base_url,
        context=context,
        extra=extra_params,
        settings=NvidiaOmniSettings(
            model=model_id,
            max_tokens=parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64),
            temperature=parse_env_float("OMNI_TEMPERATURE", 0.6, min_value=0.0),
            top_p=parse_env_float("OMNI_TOP_P", 0.95, min_value=0.0),
            input_modalities=("text", "audio"),
            response_format={"type": "json_object"} if emit_transcriptions else None,
            emit_transcriptions=emit_transcriptions,
            min_user_audio_secs=parse_env_float("OMNI_MIN_USER_AUDIO_SECS", 0.3, min_value=0.0),
        ),
    )

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "")
    custom_dictionary = load_ipa_dictionary()

    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode

    tts = NvidiaTTSService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        server=tts_server,
        settings=NvidiaTTSSettings(**tts_settings_kwargs),
        use_ssl=tts_ssl,
        text_filters=[NemotronSpeechTextFilter()],
        custom_dictionary=custom_dictionary,
        stop_frame_timeout_s=parse_env_float("TTS_STOP_FRAME_TIMEOUT_S", 30.0, min_value=5.0),
    )
    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}"
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams()),
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_turn_strategies=_build_user_turn_strategies(),
        ),
    )

    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            omni,
            tts,
            transport.output(),
            *([audio_recorder] if audio_recorder else []),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    latency_turn_count = 1
    latest_latency_turn_id = ""
    latest_latency_turn_label = ""
    latest_latency_ms: float | None = None

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency):
        logger.info(f"First bot speech latency: {latency:.3f}s")
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "user-bot-latency",
                    "latency": round(latency, 3),
                    "first": True,
                }
            )
        )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        nonlocal latest_latency_ms, latest_latency_turn_id, latest_latency_turn_label
        latest_latency_turn_id = f"turn-{latency_turn_count}"
        latest_latency_turn_label = f"Turn {latency_turn_count}"
        latest_latency_ms = round(latency * 1000, 3)
        logger.info(f"User->Bot latency: {latency:.3f}s")
        # Also emit the benchmark-compatible message (server_e2e) alongside the
        # UI metric-group below.
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "user-bot-latency",
                    "latency": round(latency, 3),
                    "first": False,
                }
            )
        )

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_latency_breakdown(observer, breakdown):
        nonlocal latency_turn_count, latest_latency_ms, latest_latency_turn_id, latest_latency_turn_label
        if latest_latency_ms is None:
            return
        metrics = [
            {
                "key": "total_latency_ms",
                "label": "Total Latency",
                "value": latest_latency_ms,
                "unit": "ms",
            }
        ]
        if breakdown.user_turn_secs is not None:
            metrics.append(
                {
                    "key": "user_turn_ms",
                    "label": "User Turn",
                    "value": round(breakdown.user_turn_secs * 1000, 3),
                    "unit": "ms",
                }
            )
        for index, ttfb in enumerate(breakdown.ttfb):
            processor = ttfb.processor.replace("#", "_").replace(" ", "_")
            metrics.append(
                {
                    "key": f"ttfb_{index}_{processor}",
                    "label": f"{ttfb.processor} TTFB",
                    "value": round(ttfb.duration_secs * 1000, 3),
                    "unit": "ms",
                }
            )
        if breakdown.text_aggregation is not None:
            metrics.append(
                {
                    "key": "text_aggregation_ms",
                    "label": f"{breakdown.text_aggregation.processor} Text Aggregation",
                    "value": round(breakdown.text_aggregation.duration_secs * 1000, 3),
                    "unit": "ms",
                }
            )
        if not metrics:
            return
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "metric-group",
                    "group_id": latest_latency_turn_id,
                    "group_label": latest_latency_turn_label,
                    "category": "latency",
                    "source": "UserBotLatencyObserver",
                    "metrics": metrics,
                }
            )
        )
        events = breakdown.chronological_events()
        # Benchmark-compatible breakdown message (vad_smart_turn) in addition to
        # the UI metric-group above.
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "latency-breakdown",
                    "vad_smart_turn": round(breakdown.user_turn_secs, 3)
                    if breakdown.user_turn_secs is not None
                    else None,
                    "events": events,
                }
            )
        )
        if events:
            logger.info(f"Latency breakdown: {' | '.join(events)}")
        latency_turn_count += 1
        latest_latency_ms = None
        latest_latency_turn_id = ""
        latest_latency_turn_label = ""

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[latency_observer],
        enable_tracing=IS_TRACING_ENABLED,
    )

    @user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        # Omni turn boundary is decided by smart-turn, not an ASR final frame.
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "user-turn-finalized",
                    "timestamp": getattr(message, "timestamp", None),
                    "transcript": getattr(message, "content", None),
                    "user_id": getattr(message, "user_id", None),
                }
            )
        )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_connected(rtvi):
        logger.info("Client connected")
        if audio_recorder:
            await audio_recorder.start_recording()
        context.add_message({"role": "user", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    async def _apply_set_voice(payload: dict) -> None:
        voice_id = payload.get("voice_id", "")
        language = payload.get("language", "")
        if not voice_id:
            return
        settings_kwargs: dict[str, Any] = {"voice": voice_id}
        if language:
            settings_kwargs["language"] = normalize_lang_code(language)
        await task.queue_frame(
            TTSUpdateSettingsFrame(
                delta=NvidiaTTSSettings(**settings_kwargs),
                service=tts,
            )
        )
        logger.info(f"Voice switched -> {voice_id}, language={settings_kwargs.get('language', '(unchanged)')}")

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        payload = message.data if isinstance(message.data, dict) else {}
        if message.type == "set-voice":
            await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()


def _create_transport(runner_args: RunnerArguments):
    """Create a transport from runner arguments (WebRTC, WebSocket, or eval)."""
    from pipecat.runner.types import EvalRunnerArguments, SmallWebRTCRunnerArguments

    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

        return SmallWebRTCTransport(
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
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
                audio_in_sample_rate=16000,
                audio_out_enabled=True,
                audio_out_sample_rate=16000,
                audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
                add_wav_header=False,
                serializer=RTVIEvalSerializer(),
            ),
            host=runner_args.host,
            port=runner_args.port,
        )

    from pipecat.serializers.base_serializer import FrameSerializer
    from pipecat.serializers.protobuf import ProtobufFrameSerializer
    from pipecat.transports.websocket.fastapi import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    websocket = getattr(runner_args, "websocket", None)
    if websocket is None:
        raise TypeError(f"Unsupported runner args type: {type(runner_args)}")

    return FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
            add_wav_header=False,
            serializer=ProtobufFrameSerializer(params=FrameSerializer.InputParams(ignore_rtvi_messages=False)),
        ),
    )
