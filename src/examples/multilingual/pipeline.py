# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Multilingual cascaded pipeline: NVIDIA STT -> Nemotron LLM -> Magpie TTS.

The session is locked to a single language for the whole connection (selected in
the UI, defaulting to ``default_session_language`` in ``examples_registry.yaml``):
the ASR, the TTS voice, and the LLM all operate in that one language. The LLM
replies with plain spoken text, kept on-language by the fixed-session prompt
addon plus a per-turn reminder.
"""

import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService as PipecatNvidiaLLMService
from pipecat.services.nvidia.llm import NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.multilingual.multilingual_processor import (
    FIXED_SESSION_GREETING_TRIGGER,
    FIXED_SESSION_LANGUAGE_ADDON_KEY,
    PerTurnReminderProcessor,
    build_reminder,
    describe_language,
    get_lang_codes,
    with_reasoning,
)
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    apply_pinned_prompt_summary,
    bind_realtime_automatic_response_provenance,
    build_context_messages,
    build_pipeline_params,
    build_smart_turn_stop_strategies,
    build_user_mute_strategies,
    build_vad_params,
    build_vad_user_turn_start_strategies,
    create_transport,
    manual_realtime_user_aggregator_params,
    register_session_start_handlers,
    resolve_pipeline_prompt,
    runner_protocol,
    select_max_tokens_config,
    uses_server_vad_turn_detection,
    with_realtime_observers,
)
from examples.shared.prewarm import (
    prewarm_asr,
    prewarm_tts,
    resolve_voice_for_language,
    validate_llm_session_language,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_prompt_catalog,
    load_selected_service_entry,
    load_service_entry,
    load_service_entry_by_id,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    render_prompt_addon,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)
DEFAULT_SESSION_LANGUAGE = "de-DE"


def _is_eval_transport(runner_args: RunnerArguments) -> bool:
    """Return whether the pipeline is running through Pipecat's eval transport."""
    cli_transport = str(getattr(getattr(runner_args, "cli_args", None), "transport", "") or "").strip().lower()
    return cli_transport == "eval" or isinstance(runner_args, EvalRunnerArguments)


def _build_multilingual_user_aggregator_params(
    welcome_enabled: bool,
    *,
    transport=None,
) -> LLMUserAggregatorParams:
    """Use VAD-only turn starts so interim ASR text does not start a user turn."""
    if transport is not None:
        manual = manual_realtime_user_aggregator_params(
            welcome_enabled,
            transport=transport,
        )
        if manual is not None:
            return manual
    start_strategies = build_vad_user_turn_start_strategies(
        transport=transport,
        include_transcription=False,
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


async def _prepare_session_language_codes(
    runner_args: RunnerArguments,
    *,
    include_tts: bool = True,
    tts_server: str,
    tts_voice: str,
    tts_function_id: str,
    tts_model: str,
    asr_server: str,
    asr_model: str,
    asr_function_id: str,
    llm_supported_languages=None,
) -> str:
    """Prewarm speech services and return UI language codes when startup should probe them."""
    if _is_eval_transport(runner_args):
        logger.info("Skipping ASR/TTS service prewarm for eval transport startup")
        return ""

    prewarm_tasks = [asyncio.to_thread(prewarm_asr, asr_server, asr_model, asr_function_id)]
    if include_tts:
        prewarm_tasks.append(asyncio.to_thread(prewarm_tts, tts_server, tts_voice, tts_function_id, tts_model))
    await asyncio.gather(*prewarm_tasks)
    if not include_tts:
        return ""

    language_catalog_kwargs = {
        "asr_server": asr_server,
        "asr_model": asr_model,
        "asr_function_id": asr_function_id,
        "tts_server": tts_server,
        "tts_voice_id": tts_voice,
    }
    if llm_supported_languages is not None:
        language_catalog_kwargs["llm_supported_languages"] = llm_supported_languages
    return get_lang_codes(**language_catalog_kwargs)


def _should_prepare_tts_catalog(*, is_realtime: bool, output_modalities: object) -> bool:
    """Return whether startup may discover a voice for the default output mode."""
    return not is_realtime or output_modalities != ["text"]


def _resolve_llm_supported_languages(body: dict, default_llm: dict):
    """Return selected LLM capabilities, rejecting unknown built-in selections."""
    selected_llm_id = str(body.get("llm_id", "") or "")
    custom_llm = selected_llm_id.startswith("custom-") or (not selected_llm_id and bool(body.get("model_id")))
    if custom_llm:
        return None
    if not selected_llm_id:
        return default_llm.get("supported_languages")

    selected_llm_entry = load_service_entry_by_id("llm", selected_llm_id)
    if not selected_llm_entry:
        raise ValueError(f"Unknown built-in LLM selection: {selected_llm_id}")
    return selected_llm_entry.get("supported_languages")


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the multilingual NVIDIA cascaded pipeline for a single session."""
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    is_realtime = runner_protocol(runner_args) == "realtime"
    if is_realtime:
        from examples.shared.nvidia_llm import NvidiaLLMService as RealtimeNvidiaLLMService
        from realtime.transport import (
            bind_realtime_assistant_context_message,
            bind_realtime_context,
            bind_realtime_deferred_service_responses,
            bind_realtime_tts_service,
            configure_realtime_client_tools,
            prepare_realtime_tools,
            realtime_response_gate_processors,
        )

    prepare_tts = _should_prepare_tts_catalog(
        is_realtime=is_realtime,
        output_modalities=body.get("output_modalities"),
    )
    welcome_enabled = not is_realtime and examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    prompt_key, base_system_content = resolve_pipeline_prompt(__file__, body, is_realtime=is_realtime)
    logger.info(f"Starting multilingual cascaded pipeline (prompt={prompt_key})")
    selected_llm_id = str(body.get("llm_id", "") or "")
    default_llm = load_selected_service_entry("llm", selected_llm_id) if is_realtime else load_service_entry("llm", "")
    llm_profile = default_llm
    default_tts = (
        load_selected_service_entry("tts", body.get("tts_id")) if is_realtime else load_service_entry("tts", "")
    )
    default_asr = (
        load_selected_service_entry("asr", body.get("asr_id")) if is_realtime else load_service_entry("asr", "")
    )
    llm_supported_languages = _resolve_llm_supported_languages(body, default_llm)

    # --- ASR ---
    asr_server = body.get("asr_server", "") or default_asr.get("server", "grpc.nvcf.nvidia.com:443")
    asr_ssl = is_nvcf(asr_server)
    asr_kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": asr_server,
        "use_ssl": asr_ssl,
    }
    raw_asr_function_id = body.get("asr_function_id")
    asr_function_id = (
        str(raw_asr_function_id) if raw_asr_function_id is not None else default_asr.get("function_id", "")
    )
    asr_model = body.get("asr_model", "") or default_asr.get("model", "")
    asr_language_code = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    if not asr_language_code or asr_language_code.strip().lower() == "auto":
        registry_language = examples_registry.default_session_language(body.get("pipeline_mode", ""))
        asr_language_code = registry_language or DEFAULT_SESSION_LANGUAGE
    fixed_session_language = normalize_lang_code(asr_language_code)
    validate_llm_session_language(fixed_session_language, llm_supported_languages)
    if asr_function_id or asr_model:
        asr_kwargs["model_function_map"] = {
            "function_id": asr_function_id,
            "model_name": asr_model or "custom-asr",
        }
    if fixed_session_language:
        asr_kwargs["settings"] = NvidiaSTTSettings(language=fixed_session_language)
    if is_realtime:
        from realtime.asr import RealtimeNvidiaSTTService

        stt = RealtimeNvidiaSTTService(**asr_kwargs, stop_history=400)
    else:
        stt = NvidiaSTTService(**asr_kwargs, stop_history=400)
    logger.info(
        f"ASR: server={asr_server}, ssl={asr_ssl}, function_id={asr_function_id or '(default)'}, "
        f"language={fixed_session_language}"
    )

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    lang_codes = await _prepare_session_language_codes(
        runner_args,
        include_tts=prepare_tts,
        tts_server=tts_server,
        tts_voice=tts_voice,
        tts_function_id=tts_function_id,
        tts_model=tts_model,
        asr_server=asr_server,
        asr_model=asr_model,
        asr_function_id=asr_function_id,
        llm_supported_languages=llm_supported_languages,
    )

    # --- LLM ---
    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    base_extra = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )

    raw_temperature = body.get("temperature", "")
    if raw_temperature in ("", None):
        raw_temperature = default_llm.get("temperature", "")
    llm_temperature = float(raw_temperature) if raw_temperature not in ("", None) else None

    logger.info(
        f"LLM: model={model_id}, base_url={base_url}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"temperature={llm_temperature if llm_temperature is not None else '(default)'}, "
        f"extra_params={base_extra or '(none)'}"
    )

    llm_settings = NvidiaLLMSettings(model=model_id)
    if base_extra:
        llm_settings.extra = base_extra
    if llm_temperature is not None:
        llm_settings.temperature = llm_temperature
    if is_realtime:
        max_tokens = select_max_tokens_config(
            body,
            default_llm.get("max_tokens", ""),
            is_realtime=True,
        )
        if max_tokens not in ("", None):
            llm_settings.max_tokens = int(max_tokens)
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
    tool_choice = body.get("tool_choice", "auto") or "auto"
    if is_realtime:
        tools_schema = configure_realtime_client_tools(
            transport,
            llm,
            body.get("client_tools"),
        )
        tools_schema, tool_choice = await prepare_realtime_tools(transport, llm)

    summary_extra = with_reasoning(base_extra, True)
    summary_llm_settings = NvidiaLLMSettings(model=model_id)
    if summary_extra:
        summary_llm_settings.extra = summary_extra
    if llm_temperature is not None:
        summary_llm_settings.temperature = llm_temperature
    summary_llm = PipecatNvidiaLLMService(
        api_key=nvidia_api_key(),
        base_url=base_url,
        settings=summary_llm_settings,
    )

    # --- TTS ---
    custom_dictionary = load_ipa_dictionary()
    tts_synthesis_mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    tts_zero_shot_audio_prompt_file = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get(
        "zero_shot_audio_prompt_file", ""
    )
    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    if fixed_session_language:
        tts_settings_kwargs["language"] = fixed_session_language
        if prepare_tts:
            resolved_voice = resolve_voice_for_language(
                fixed_session_language,
                tts_voice,
                server=tts_server,
                function_id=tts_function_id,
                model=tts_model,
            )
            if resolved_voice:
                tts_voice = resolved_voice
                tts_settings_kwargs["voice"] = resolved_voice

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
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}, "
        f"lang_codes={lang_codes or '(no voices discovered)'}, "
        f"text_filters=[NemotronSpeechTextFilter]"
    )

    # --- Context ---
    prompt_catalog = load_prompt_catalog(__file__)

    def render_realtime_instructions(instructions: str) -> list[dict]:
        decorated = render_prompt_addon(
            instructions,
            prompt_catalog,
            FIXED_SESSION_LANGUAGE_ADDON_KEY,
            {"fixed_language_name": describe_language(fixed_session_language)},
        )
        return build_context_messages(decorated, system_prompt)

    messages = render_realtime_instructions(base_system_content)
    if tools_schema is not None:
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
        user_params=_build_multilingual_user_aggregator_params(
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

    reminder_processor = PerTurnReminderProcessor(
        build_reminder(fixed_session_language),
        realtime=is_realtime,
    )

    audio_recorder = create_audio_recorder()

    response_gate_processors = realtime_response_gate_processors(transport) if is_realtime else []
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            *response_gate_processors,
            reminder_processor,
            llm,
            tts,
            transport.output(),
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
                llm=summary_llm,
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

    register_session_start_handlers(
        transport=transport,
        task=task,
        context=context,
        runner_args=runner_args,
        intro_prompt=FIXED_SESSION_GREETING_TRIGGER,
        on_start=_on_session_start,
        welcome_enabled=welcome_enabled,
    )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    if not is_realtime:

        @task.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            payload = message.data if isinstance(message.data, dict) else {}
            if message.type == "set-voice":
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

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
