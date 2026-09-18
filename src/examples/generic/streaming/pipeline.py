# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic Pipecat voice pipeline backed by native vLLM StreamingInput."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.generic.streaming.processor import StreamingLLMProcessor
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    build_pipeline_params,
    build_smart_turn_stop_strategies,
    create_transport,
    register_session_start_handlers,
    with_realtime_observers,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_service_entry,
    normalize_lang_code,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    resolve_prompt,
)

load_dotenv(override=True)


DEFAULT_STREAMING_URL = "ws://nvidia-llm-vllm-streaming:8000/v1/streaming-session"


async def run_streaming_bot(
    runner_args: RunnerArguments,
    *,
    example_file: str,
    streaming_url: str = "",
) -> None:
    """Build one streaming-input voice session from reusable pipeline pieces."""
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    prompt_key, system_prompt = resolve_prompt(
        example_file,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    default_asr = load_service_entry("asr", "")
    default_tts = load_service_entry("tts", "")

    asr_server = body.get("asr_server", "") or default_asr.get("server", "nemo-speech:50051")
    asr_function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    asr_model = body.get("asr_model", "") or default_asr.get("model", "")
    asr_language = body.get("asr_language_code", "") or default_asr.get("language_code", "en-US")
    asr_kwargs: dict[str, Any] = {
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "server": asr_server,
        "use_ssl": is_nvcf(asr_server),
        "stop_history": 400,
        "settings": NvidiaSTTSettings(language=asr_language, interim_results=True),
    }
    if asr_function_id or asr_model:
        asr_kwargs["model_function_map"] = {
            "function_id": asr_function_id,
            "model_name": asr_model or "streaming-asr",
        }
    stt = NvidiaSTTService(**asr_kwargs)

    tts_server = body.get("tts_server", "") or default_tts.get("server", "nemo-speech:50051")
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "John")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "stitched")
    tts_function_id = body.get("tts_function_id", "") or default_tts.get("function_id", "")
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    tts_language = body.get("tts_language_code", "") or default_tts.get("language_code", "en-US")
    tts_settings = NvidiaTTSSettings(
        voice=tts_voice,
        language=normalize_lang_code(tts_language),
        synthesis_mode=tts_synthesis_mode,
    )
    tts_kwargs: dict[str, Any] = {
        "api_key": os.getenv("NVIDIA_API_KEY", ""),
        "server": tts_server,
        "use_ssl": is_nvcf(tts_server),
        "settings": tts_settings,
        # The streaming-input LLM can finish a response much faster than audio
        # playback. Keep per-session TTS work serialized until playback stops so
        # a following turn cannot overlap the previous Magpie context.
        "pause_frame_processing": True,
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": load_ipa_dictionary(),
    }
    if tts_function_id or tts_model:
        tts_kwargs["model_function_map"] = {
            "function_id": str(tts_function_id),
            "model_name": tts_model,
        }
    tts = NvidiaTTSService(**tts_kwargs)

    streaming_llm = StreamingLLMProcessor(
        url=os.getenv("STREAMING_LLM_URL", "") or streaming_url or DEFAULT_STREAMING_URL,
        system_prompt=system_prompt,
        hold_words=parse_env_int("STREAMING_HOLD_WORDS", 0, min_value=0),
        final_max_tokens=parse_env_int("STREAMING_FINAL_MAX_TOKENS", 384, min_value=1),
        history_turns=parse_env_int("STREAMING_HISTORY_TURNS", 8, min_value=0),
        prefill_next_turn=parse_env_bool("STREAMING_PREFILL_NEXT_TURN", default=True),
    )
    use_silero_turn_detection = parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False)
    if use_silero_turn_detection:
        vad_stop_secs = parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0)
        turn_stop_strategies = [SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)]
        turn_detection_label = f"Silero VAD only (stop_secs={vad_stop_secs:.3f}s)"
    else:
        vad_stop_secs = 0.2
        turn_stop_strategies = build_smart_turn_stop_strategies()
        turn_detection_label = "Silero VAD + Smart Turn"

    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=vad_stop_secs)))
    user_turn = UserTurnProcessor(
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy(enable_interruptions=True)],
            stop=turn_stop_strategies,
        )
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            user_turn,
            streaming_llm,
            tts,
            transport.output(),
        ]
    )

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency):  # noqa: ARG001
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": True})
        )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):  # noqa: ARG001
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": False})
        )

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):  # noqa: ARG001
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "latency-breakdown",
                    "vad_smart_turn": round(breakdown.user_turn_secs, 3)
                    if breakdown.user_turn_secs is not None
                    else None,
                    "events": breakdown.chronological_events(),
                }
            )
        )

    task = PipelineWorker(
        pipeline,
        params=build_pipeline_params(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=with_realtime_observers(latency_observer, transport=transport),
        enable_tracing=IS_TRACING_ENABLED,
    )

    register_session_start_handlers(
        transport=transport,
        task=task,
        context=LLMContext([{"role": "system", "content": system_prompt}]),
        runner_args=runner_args,
        welcome_enabled=welcome_enabled,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        if message.type != "set-voice" or not isinstance(message.data, dict):
            return
        voice_id = message.data.get("voice_id", "")
        if not voice_id:
            return
        settings_kwargs: dict[str, Any] = {"voice": voice_id}
        if message.data.get("language"):
            settings_kwargs["language"] = normalize_lang_code(message.data["language"])
        await task.queue_frame(TTSUpdateSettingsFrame(delta=NvidiaTTSSettings(**settings_kwargs), service=tts))

    logger.info(
        f"Starting generic StreamingInput pipeline (prompt={prompt_key}, llm={streaming_llm.name}, "
        f"asr={asr_server}, tts={tts_server}, turn_detection={turn_detection_label})"
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
