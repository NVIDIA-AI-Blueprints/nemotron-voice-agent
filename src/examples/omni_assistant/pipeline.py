# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Nemotron Omni cascaded pipeline using the upstream-style Omni service.

This is the current experimental pipeline for the clean
``NvidiaOmniLLMService`` shape:

* ``transport.input`` + VAD/user-turn processing feed audio into Omni.
* Omni replaces ASR + LLM and emits standard Pipecat frames.
* NVIDIA TTS speaks the emitted ``LLMTextFrame`` response.
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    TTSUpdateSettingsFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_turn_processor import UserTurnProcessor
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.omni_assistant.audio_only_smart_turn_strategy import AudioOnlySmartTurnStopStrategy
from examples.omni_assistant.nvidia_omni_multimodal_service import (
    MAX_FUSED_USER_AUDIO_SECS,
    NvidiaOmniLLMService,
    NvidiaOmniSettings,
)
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    build_pipeline_params,
    build_smart_turn_analyzer,
    build_user_mute_strategies,
    build_vad_params,
    build_vad_user_turn_start_strategies,
    create_transport,
    realtime_vad_prefix_padding_secs,
    register_session_start_handlers,
    resolve_pipeline_prompt,
    runner_protocol,
    select_max_tokens_config,
    with_realtime_observers,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_selected_service_entry,
    load_service_entry,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
)

load_dotenv(override=True)


def _build_user_turn_strategies(*, transport=None) -> UserTurnStrategies:
    """Build VAD-start + Smart Turn-stop strategies for Omni audio turns."""
    return UserTurnStrategies(
        start=build_vad_user_turn_start_strategies(
            transport=transport,
            include_transcription=False,
        ),
        stop=[AudioOnlySmartTurnStopStrategy(turn_analyzer=build_smart_turn_analyzer())],
    )


def _build_user_turn_processor(*, transport=None) -> UserTurnProcessor:
    """Build an external turn processor for subagent branches that need one."""
    return UserTurnProcessor(user_turn_strategies=_build_user_turn_strategies(transport=transport))


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Nemotron Omni cascaded pipeline for one session."""
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    is_realtime = runner_protocol(runner_args) == "realtime"
    if is_realtime:
        from realtime.transport import (
            bind_realtime_assistant_context_message,
            bind_realtime_context,
            bind_realtime_service_response_snapshots,
            bind_realtime_tts_service,
            configure_realtime_client_tools,
            prepare_realtime_tools,
            realtime_response_gate_processors,
        )

    welcome_enabled = not is_realtime and examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))

    prompt_key, base_system_content = resolve_pipeline_prompt(__file__, body, is_realtime=is_realtime)
    logger.info(f"Starting Nemotron Omni cascaded pipeline (prompt={prompt_key})")

    default_llm = (
        load_selected_service_entry("llm", body.get("llm_id")) if is_realtime else load_service_entry("llm", "")
    )
    default_tts = (
        load_selected_service_entry("tts", body.get("tts_id")) if is_realtime else load_service_entry("tts", "")
    )

    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt_override = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        "extra_params",
    )
    if is_realtime:
        raw_max_tokens = select_max_tokens_config(
            body,
            parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64),
            is_realtime=True,
        )
        max_tokens = None if raw_max_tokens is None else int(raw_max_tokens)
    else:
        max_tokens = parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64)

    # Build the conversation context up-front. With emit_transcriptions enabled,
    # Omni reports the user's speech as a TranscriptionFrame, which the user
    # aggregator writes here as it would an STT service's transcript, while the
    # assistant aggregator commits LLMTextFrame output as usual.
    def render_realtime_instructions(instructions: str) -> list[dict]:
        system_content = instructions
        if system_prompt_override:
            system_content = f"{instructions}\n\n{system_prompt_override}".strip()
        return [{"role": "system", "content": system_content}]

    context = LLMContext(render_realtime_instructions(base_system_content))

    emit_transcriptions = parse_env_bool("OMNI_EMIT_TRANSCRIPTIONS", default=True)
    if is_realtime and body.get("client_tools") and not emit_transcriptions:
        raise ValueError("Omni Realtime client tools require OMNI_EMIT_TRANSCRIPTIONS=true for audio-turn context")
    omni = NvidiaOmniLLMService(
        # Name carries "llm" so metrics consumers (UI metric-group, perf
        # benchmark) attribute Omni's TTFB/processing/token-usage metrics to the
        # LLM stage. Omni fuses ASR+LLM, so these are the pipeline's LLM metrics.
        name="NemotronOmniLLM",
        api_key=nvidia_api_key(),
        base_url=base_url,
        context=context,
        extra=extra_params,
        settings=NvidiaOmniSettings(
            model=model_id,
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            temperature=parse_env_float("OMNI_TEMPERATURE", 0.6, min_value=0.0),
            top_p=parse_env_float("OMNI_TOP_P", 0.95, min_value=0.0),
            input_modalities=("text", "audio"),
            emit_transcriptions=emit_transcriptions,
            min_user_audio_secs=parse_env_float("OMNI_MIN_USER_AUDIO_SECS", 0.3, min_value=0.0),
            **(
                {
                    "max_user_audio_secs": parse_env_float(
                        "OMNI_MAX_USER_AUDIO_SECS",
                        MAX_FUSED_USER_AUDIO_SECS,
                        min_value=0.0,
                    )
                }
                if is_realtime
                else {}
            ),
            pre_speech_buffer_secs=realtime_vad_prefix_padding_secs(
                0.2,
                transport=transport,
            )
            if is_realtime
            else 0.2,
        ),
        realtime_parallel_tool_calls=(body.get("parallel_tool_calls", True) if is_realtime else None),
    )
    tools_schema = None
    tool_choice = body.get("tool_choice", "auto") or "auto"
    if is_realtime:
        tools_schema = configure_realtime_client_tools(
            transport,
            omni,
            body.get("client_tools"),
        )
        tools_schema, tool_choice = await prepare_realtime_tools(transport, omni)
    if tools_schema is not None:
        context.set_tools(tools_schema)
        context.set_tool_choice(tool_choice)
    if is_realtime:
        bind_realtime_context(
            transport,
            context,
            render_instructions=render_realtime_instructions,
        )

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    tts_zero_shot_audio_prompt_file = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get(
        "zero_shot_audio_prompt_file", ""
    )
    custom_dictionary = load_ipa_dictionary()

    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    tts_kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": tts_server,
        "settings": NvidiaTTSSettings(**tts_settings_kwargs),
        "use_ssl": tts_ssl,
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": custom_dictionary,
        "stop_frame_timeout_s": parse_env_float("TTS_STOP_FRAME_TIMEOUT_S", 30.0, min_value=5.0),
    }
    if tts_function_id or tts_model:
        tts_kwargs["model_function_map"] = {
            "function_id": tts_function_id,
            "model_name": tts_model,
        }
    if tts_zero_shot_audio_prompt_file:
        tts_kwargs["zero_shot_audio_prompt_file"] = tts_zero_shot_audio_prompt_file
    tts = NvidiaTTSService(**tts_kwargs)
    if is_realtime:
        bind_realtime_tts_service(transport, tts)
        bind_realtime_service_response_snapshots(transport, omni)
    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}, "
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}"
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                params=build_vad_params(VADParams(), transport=transport if is_realtime else None)
            ),
            user_mute_strategies=build_user_mute_strategies(
                welcome_enabled,
                transport=transport if is_realtime else None,
            ),
            user_turn_strategies=_build_user_turn_strategies(
                transport=transport if is_realtime else None,
            ),
        ),
    )

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if is_realtime:
            bind_realtime_assistant_context_message(transport, message)

    audio_recorder = create_audio_recorder()

    response_gate_processors = realtime_response_gate_processors(transport) if is_realtime else []
    pipeline = Pipeline(
        [
            transport.input(),
            user_aggregator,
            *response_gate_processors,
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
        params=build_pipeline_params(
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=not is_realtime,
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=with_realtime_observers(
            latency_observer,
            transport=transport,
            is_realtime=is_realtime,
        ),
        enable_tracing=IS_TRACING_ENABLED,
        enable_rtvi=not is_realtime,
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

    async def _on_session_start() -> None:
        if audio_recorder:
            await audio_recorder.start_recording()

    register_session_start_handlers(
        transport=transport,
        task=task,
        context=context,
        runner_args=runner_args,
        intro_prompt="Please introduce yourself to the user.",
        on_start=_on_session_start,
        welcome_enabled=welcome_enabled,
    )

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

    if not is_realtime:

        @task.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            payload = message.data if isinstance(message.data, dict) else {}
            if message.type == "set-voice":
                await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
