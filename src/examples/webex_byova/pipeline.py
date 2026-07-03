# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Cisco Webex BYOVA cascaded pipeline.

This pipeline intentionally mirrors the generic cascaded assistant, but it
overrides transport output pacing so TTS audio frames are pushed downstream
immediately instead of being rate-limited to real-time playback.
"""

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSUpdateSettingsFrame
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
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from examples.generic.tools import TOOL_HANDLERS, build_tools_schema
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    apply_pinned_prompt_summary,
    build_context_messages,
    create_transport,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_service_entry,
    normalize_lang_code,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
    resolve_tools_available,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)
WEBEX_BYOVA_VAD_STOP_SECS = parse_env_float("SILERO_VAD_STOP_SECS", 0.8, min_value=0.0)


def _enable_burst_audio_output(transport):
    """Disable transport-side real-time pacing so TTS frames flush immediately."""
    output = transport.output()
    if hasattr(output, "_write_audio_sleep"):

        async def _write_audio_sleep_burst():
            await asyncio.sleep(0)

        # Pipecat transport outputs pace audio for browser playback by default.
        # The BYOVA adapter expects chunks as soon as they are generated.
        output._write_audio_sleep = _write_audio_sleep_burst
        logger.info("Burst audio output enabled for Webex BYOVA transport")
    else:
        logger.warning("Burst audio output requested, but transport has no _write_audio_sleep hook")
    return output


def _build_webex_byova_user_aggregator_params() -> LLMUserAggregatorParams:
    """Keep a longer VAD stop window for BYOVA turn boundary stability."""
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=WEBEX_BYOVA_VAD_STOP_SECS)),
        user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
        ),
    )


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Webex BYOVA pipeline for a single session."""
    transport = create_transport(runner_args)
    transport_output = _enable_burst_audio_output(transport)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    logger.info(f"Starting Webex BYOVA pipeline (prompt={prompt_key}, tools={list(TOOL_HANDLERS)})")
    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")
    default_asr = load_service_entry("asr", "")

    asr_server = body.get("asr_server", "") or default_asr.get("server", "grpc.nvcf.nvidia.com:443")
    asr_ssl = is_nvcf(asr_server)
    asr_kwargs: dict = {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "server": asr_server,
        "use_ssl": asr_ssl,
    }
    asr_function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    asr_model = body.get("asr_model", "") or default_asr.get("model", "")
    asr_language_code = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    if asr_function_id or asr_model:
        asr_kwargs["model_function_map"] = {
            "function_id": asr_function_id,
            "model_name": asr_model or "custom-asr",
        }
    if asr_language_code:
        asr_kwargs["settings"] = NvidiaSTTSettings(language=asr_language_code)
    stt = NvidiaSTTService(**asr_kwargs, stop_history=400)
    logger.info(
        f"ASR: server={asr_server}, ssl={asr_ssl}, function_id={asr_function_id or '(default)'}, "
        f"language={asr_language_code or '(default)'}"
    )

    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3-nano-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )

    logger.info(
        f"LLM: model={model_id}, base_url={base_url}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"extra_params={extra_params or '(none)'}"
    )

    llm_settings = NvidiaLLMSettings(model=model_id)
    if extra_params:
        llm_settings.extra = extra_params
    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=base_url,
        settings=llm_settings,
    )

    tools_available = resolve_tools_available(__file__, prompt_key)
    tools_schema, registered_tools = build_tools_schema(__file__, tools_available)
    tools_enabled = tools_schema is not None

    if tools_enabled:
        for name in registered_tools:
            llm.register_function(name, TOOL_HANDLERS[name])
            logger.info(f"Registered tool handler: {name}")
    else:
        logger.info(f"Tool calling disabled for prompt_key={prompt_key!r} (no tools_available in prompts.yaml)")

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "Magpie-Multilingual.EN-US.Aria")
    custom_dictionary = load_ipa_dictionary()

    tts = NvidiaTTSService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        server=tts_server,
        settings=NvidiaTTSSettings(voice=tts_voice),
        use_ssl=tts_ssl,
        text_filters=[NemotronSpeechTextFilter()],
        custom_dictionary=custom_dictionary,
    )

    logger.info(f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, text_filters=[NemotronSpeechTextFilter]")

    messages = build_context_messages(base_system_content, system_prompt)

    if tools_enabled:
        context = LLMContext(messages, tools=tools_schema, tool_choice="auto")
    else:
        context = LLMContext(messages)
    preserve_prompt_messages = len(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=_build_webex_byova_user_aggregator_params(),
    )
    logger.info(f"BYOVA SILERO_VAD_STOP_SECS={WEBEX_BYOVA_VAD_STOP_SECS}")
    logger.info(
        f"Chat history summarization enabled: recent_turns={CHAT_HISTORY_RECENT_TURNS}, "
        f"preserve_prompt_messages={preserve_prompt_messages}"
    )

    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport_output,
            *([audio_recorder] if audio_recorder else []),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    summary_lock = asyncio.Lock()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        async with summary_lock:
            await apply_pinned_prompt_summary(
                context=context,
                llm=llm,
                preserve_prompt_messages=preserve_prompt_messages,
                recent_turns=CHAT_HISTORY_RECENT_TURNS,
                summary_system_prompt=system_prompt,
            )

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
        logger.info(f"User→Bot latency: {latency:.3f}s")
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
    async def on_breakdown(observer, breakdown):
        events = breakdown.chronological_events()
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
        settings_kwargs: dict = {"voice": voice_id}
        if language:
            settings_kwargs["language"] = normalize_lang_code(language)
        await task.queue_frame(
            TTSUpdateSettingsFrame(
                delta=NvidiaTTSSettings(**settings_kwargs),
                service=tts,
            )
        )
        logger.info(f"Voice switched → {voice_id}, language={settings_kwargs.get('language', '(unchanged)')}")

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        payload = message.data if isinstance(message.data, dict) else {}
        if message.type == "set-voice":
            await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
