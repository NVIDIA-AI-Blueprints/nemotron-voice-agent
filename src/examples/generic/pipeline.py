# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Generic cascaded pipeline with NVIDIA ASR, Nemotron LLM, NVIDIA TTS, and tools.

Ordinary sessions use Pipecat's NVIDIA services directly. OpenAI Realtime
sessions add their protocol-specific lifecycle and tool adapters.
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame, TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService as PipecatNvidiaLLMService
from pipecat.services.nvidia.llm import NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.generic.tools import TOOL_HANDLERS, build_tools_schema
from examples.shared.activity_check import create_activity_check_processor
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    apply_pinned_prompt_summary,
    build_context_messages,
    build_pipeline_params,
    build_user_aggregator_params,
    create_transport,
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
    parse_env_int,
    parse_json_dict,
    resolve_tools_available,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the NVIDIA cascaded pipeline for a single session."""
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    is_realtime = runner_protocol(runner_args) == "realtime"
    if is_realtime:
        from examples.shared.nvidia_llm import NvidiaLLMService as RealtimeNvidiaLLMService
        from examples.shared.tool_runtime import terminal_tool_handler, tool_parameter_schema
        from realtime.transport import (
            bind_realtime_assistant_context_message,
            bind_realtime_context,
            bind_realtime_deferred_service_responses,
            bind_realtime_tts_service,
            configure_realtime_client_tools,
            prepare_realtime_tools,
            realtime_response_gate_processors,
            realtime_tool_result_processors,
        )

    welcome_enabled = not is_realtime and examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    prompt_key, base_system_content = resolve_pipeline_prompt(__file__, body, is_realtime=is_realtime)
    logger.info(f"Starting generic cascaded pipeline (prompt={prompt_key})")
    selected_llm_id = str(body.get("llm_id", "") or "")
    default_llm = load_selected_service_entry("llm", selected_llm_id) if is_realtime else load_service_entry("llm", "")
    llm_profile = default_llm
    default_tts = (
        load_selected_service_entry("tts", body.get("tts_id")) if is_realtime else load_service_entry("tts", "")
    )
    default_asr = (
        load_selected_service_entry("asr", body.get("asr_id")) if is_realtime else load_service_entry("asr", "")
    )

    # --- ASR ---
    asr_server = body.get("asr_server", "") or default_asr.get("server", "grpc.nvcf.nvidia.com:443")
    asr_ssl = is_nvcf(asr_server)
    asr_kwargs: dict = {
        "api_key": nvidia_api_key(),
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
    if is_realtime:
        from realtime.asr import RealtimeNvidiaSTTService

        stt = RealtimeNvidiaSTTService(**asr_kwargs, stop_history=400)
    else:
        stt = NvidiaSTTService(**asr_kwargs, stop_history=400)
    logger.info(
        f"ASR: server={asr_server}, ssl={asr_ssl}, function_id={asr_function_id or '(default)'}, "
        f"language={asr_language_code or '(default)'}"
    )

    # --- LLM ---
    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )

    raw_temperature = body.get("temperature", "")
    if raw_temperature in ("", None):
        raw_temperature = default_llm.get("temperature", "")
    llm_temperature = None
    if raw_temperature not in ("", None):
        try:
            llm_temperature = float(raw_temperature)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid temperature={raw_temperature!r}")

    llm_settings = NvidiaLLMSettings(model=model_id)
    max_tokens = select_max_tokens_config(
        body,
        default_llm.get("max_tokens", ""),
        is_realtime=is_realtime,
    )
    if max_tokens not in ("", None):
        try:
            llm_settings.max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring invalid max_tokens={max_tokens!r}")
    if llm_temperature is not None:
        llm_settings.temperature = llm_temperature
    if extra_params:
        llm_settings.extra = extra_params
    logger.info(
        f"LLM: model={model_id}, base_url={base_url}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"temperature={llm_temperature if llm_temperature is not None else '(default)'}, "
        f"extra_params={extra_params or '(none)'}"
    )
    if is_realtime:
        llm = RealtimeNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=base_url,
            settings=llm_settings,
            forced_tool_call_stops=llm_profile.get("forced_tool_call_stops"),
            realtime_parallel_tool_calls=body.get("parallel_tool_calls", True),
            realtime_model_max_output_tokens=body.get("realtime_model_max_output_tokens"),
        )
    else:
        llm = PipecatNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=base_url,
            settings=llm_settings,
        )

    tools_schema = None
    registered_tools: list[str] = []
    tool_choice = body.get("tool_choice", "auto") or "auto"
    if is_realtime:
        raw_server_tools = body.get("server_tools", [])
        if not isinstance(raw_server_tools, list) or not all(isinstance(name, str) for name in raw_server_tools):
            raise ValueError("server_tools must be a list of trusted function names")
        tools_available = list(dict.fromkeys(raw_server_tools))
    else:
        tools_available = resolve_tools_available(__file__, prompt_key)
    tools_schema, registered_tools = build_tools_schema(__file__, tools_available)
    missing_trusted_tools = set(tools_available) - set(registered_tools)
    if is_realtime and missing_trusted_tools:
        raise ValueError(f"Trusted pipeline tool {sorted(missing_trusted_tools)[0]!r} has no registered handler")
    logger.info(f"Trusted pipeline tools: {registered_tools or '(none)'}")
    if tools_schema is not None:
        for name in registered_tools:
            handler = TOOL_HANDLERS[name]
            if is_realtime:
                handler = terminal_tool_handler(
                    handler,
                    parameters=tool_parameter_schema(tools_schema, name),
                    timeout_secs=30.0,
                )
                # Realtime tool outputs are durable conversation items. Barge-
                # in cancels media generation, not the accepted backend call.
                llm.register_function(name, handler, cancel_on_interruption=False)
            else:
                llm.register_function(name, handler)
            logger.info(f"Registered tool handler: {name}")
    elif not is_realtime:
        logger.info(f"Tool calling disabled for prompt_key={prompt_key!r} (no tools_available in prompts.yaml)")
    if is_realtime:
        tools_schema = configure_realtime_client_tools(
            transport,
            llm,
            body.get("client_tools"),
            trusted_tools=tools_schema,
            trusted_tool_names=registered_tools,
        )
        tools_schema, tool_choice = await prepare_realtime_tools(transport, llm)
    tools_enabled = tools_schema is not None

    # --- TTS ---
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
    tts_language_code = body.get("tts_language_code", "") or default_tts.get("language_code", "")
    if tts_language_code:
        tts_language_code = normalize_lang_code(tts_language_code)
    custom_dictionary = load_ipa_dictionary()

    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    if tts_language_code:
        tts_settings_kwargs["language"] = tts_language_code
    tts_kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": tts_server,
        "settings": NvidiaTTSSettings(**tts_settings_kwargs),
        "use_ssl": tts_ssl,
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": custom_dictionary,
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

    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}, "
        f"language={tts_language_code or '(pipecat default)'}, "
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}, "
        f"text_filters=[NemotronSpeechTextFilter]"
    )

    # --- Context ---
    def render_realtime_instructions(instructions: str) -> list[dict]:
        return build_context_messages(instructions, system_prompt)

    messages = render_realtime_instructions(base_system_content)

    if tools_enabled:
        context = LLMContext(messages, tools=tools_schema, tool_choice=tool_choice)
    else:
        context = LLMContext(messages)
    if is_realtime:
        bind_realtime_context(
            transport,
            context,
            render_instructions=render_realtime_instructions,
        )
        bind_realtime_deferred_service_responses(transport, llm)
    preserve_prompt_messages = len(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(
            welcome_enabled,
            transport=transport if is_realtime else None,
        ),
    )
    if is_realtime:
        logger.info("Chat history summarization disabled: Realtime conversation history remains canonical")
    else:
        logger.info(
            f"Chat history summarization enabled: recent_turns={CHAT_HISTORY_RECENT_TURNS}, "
            f"preserve_prompt_messages={preserve_prompt_messages}"
        )

    audio_recorder = create_audio_recorder()

    async def queue_activity_llm_run() -> None:
        await task.queue_frame(LLMRunFrame())

    activity_check = None
    if not is_realtime:
        activity_check = create_activity_check_processor(
            examples_registry.activity_check_config(body.get("pipeline_mode", "generic-assistant")),
            context=context,
            queue_llm_run=queue_activity_llm_run,
            instruction_role="user",
        )
    logger.info(f"Proactive activity checks: {'enabled' if activity_check else 'disabled'}")

    response_gate_processors = realtime_response_gate_processors(transport) if is_realtime else []
    tool_result_processors = realtime_tool_result_processors(transport) if is_realtime else []
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            *response_gate_processors,
            llm,
            *tool_result_processors,
            tts,
            transport.output(),
            *([activity_check] if activity_check else []),
            *([audio_recorder] if audio_recorder else []),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    summary_lock = asyncio.Lock()

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        if is_realtime:
            bind_realtime_assistant_context_message(transport, message)
        # Realtime item edits and deletes address this canonical context. Its
        # provider-only snapshot owns any native token-budget truncation.
        if is_realtime:
            return
        async with summary_lock:
            await apply_pinned_prompt_summary(
                context=context,
                llm=llm,
                preserve_prompt_messages=preserve_prompt_messages,
                recent_turns=CHAT_HISTORY_RECENT_TURNS,
                summary_system_prompt=system_prompt,
            )

    # Forward custom latency samples over RTVI so the benchmark can stay fully
    # client-driven and avoid server log scraping.
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
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "user-turn-finalized",
                    **({"turn_frame_id": getattr(strategy, "turn_frame_id", None)} if is_realtime else {}),
                    "timestamp": getattr(message, "timestamp", None),
                    "transcript": getattr(message, "content", None),
                    "user_id": getattr(message, "user_id", None),
                }
            )
        )

    async def _on_session_start() -> None:
        if audio_recorder:
            await audio_recorder.start_recording()
        if activity_check:
            activity_check.start()

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

    if not is_realtime:

        @task.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            payload = message.data if isinstance(message.data, dict) else {}
            if message.type == "set-voice":
                await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
