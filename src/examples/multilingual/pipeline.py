# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from pipecat.processors.frameworks.rtvi import RTVIFunctionCallReportLevel, RTVIObserverParams
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import EvalRunnerArguments, RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
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
from examples.multilingual.tool_handlers import ClientToolResultBridge, build_client_tool_handler
from examples.multilingual.tools import build_client_tools, perceivable_client_tools
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.person_memory import PERSON_MEMORY_ADDON_KEY, load_person_context, record_person_session
from examples.shared.pipeline_utils import (
    apply_pinned_prompt_summary,
    build_context_messages,
    build_pipeline_params,
    build_smart_turn_stop_strategies,
    build_user_mute_strategies,
    create_transport,
    register_session_start_handlers,
    with_realtime_observers,
)
from examples.shared.prewarm import (
    prewarm_asr,
    prewarm_tts,
    resolve_locale_for_language,
    resolve_voice_for_language,
    validate_llm_session_language,
)
from examples.shared.tts_text_normalizer import LanguageAwareVoiceFormatter
from examples.shared.voice_id import (
    ENROLL_TOOL_NAME,
    FACE_ID_ADDON_KEY,
    PRESENCE_UPDATE_MESSAGE,
    SPEAKER_UPDATE_MESSAGE,
    VOICE_ID_ADDON_KEY,
    SpeakerTurnProcessor,
    VoiceIdSession,
    voice_id_available,
)
from monitoring.recorder import SessionRecorder
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_prompt_catalog,
    load_service_entry,
    load_service_entry_by_id,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    render_prompt_addon,
    resolve_prompt,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)
DEFAULT_SESSION_LANGUAGE = "de-DE"
CLIENT_FUNCTION_TIMEOUT_SECS = parse_env_float("CLIENT_FUNCTION_TIMEOUT_SECS", 5.0, min_value=1.0)


def _is_eval_transport(runner_args: RunnerArguments) -> bool:
    """Return whether the pipeline is running through Pipecat's eval transport."""
    cli_transport = str(getattr(getattr(runner_args, "cli_args", None), "transport", "") or "").strip().lower()
    return cli_transport == "eval" or isinstance(runner_args, EvalRunnerArguments)


def _build_multilingual_user_aggregator_params(welcome_enabled: bool) -> LLMUserAggregatorParams:
    """Use VAD-only turn starts so interim ASR text does not start a user turn."""
    if not parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False):
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_mute_strategies=build_user_mute_strategies(welcome_enabled),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=build_smart_turn_stop_strategies(),
            ),
        )

    stop_secs = parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0)
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_mute_strategies=build_user_mute_strategies(welcome_enabled),
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
        ),
    )


async def _prepare_session_language_codes(
    runner_args: RunnerArguments,
    *,
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

    prewarm_tasks = [
        asyncio.to_thread(prewarm_asr, asr_server, asr_model, asr_function_id),
        asyncio.to_thread(prewarm_tts, tts_server, tts_voice, tts_function_id, tts_model),
    ]
    await asyncio.gather(*prewarm_tasks)
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


def _session_snapshot(**sections) -> dict:
    """Pipeline variant description stored with each recorded session (A/B key)."""
    return {
        **sections,
        "turn_detection": {
            "silero_vad_only": parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False),
            "silero_vad_stop_secs": parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0),
        },
        "chat_history_recent_turns": CHAT_HISTORY_RECENT_TURNS,
    }


def _span_attributes(snapshot: dict) -> dict[str, str]:
    """Flatten the variant-defining fields into OpenTelemetry span attributes."""
    return {
        "session.example": str(snapshot.get("example", "")),
        "session.language": str(snapshot.get("language", "")),
        "asr.model": str(snapshot["asr"].get("model", "")),
        "llm.model": str(snapshot["llm"].get("model", "")),
        "tts.model": str(snapshot["tts"].get("model", "")),
        "tts.voice": str(snapshot["tts"].get("voice", "")),
    }


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
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    logger.info(f"Starting multilingual cascaded pipeline (prompt={prompt_key})")
    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")
    default_asr = load_service_entry("asr", "")
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
    llm = NvidiaLLMService(
        api_key=nvidia_api_key(),
        base_url=base_url,
        settings=llm_settings,
    )

    summary_extra = with_reasoning(base_extra, True)
    summary_llm_settings = NvidiaLLMSettings(model=model_id)
    if summary_extra:
        summary_llm_settings.extra = summary_extra
    if llm_temperature is not None:
        summary_llm_settings.temperature = llm_temperature
    summary_llm = NvidiaLLMService(
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
        # A caller (e.g. an external client passing request_data.asr_language_code) may
        # only know the bare base language ("fr"), not the full locale the TTS voice
        # catalog is keyed by ("fr-FR"); expand it here so synthesis gets the locale the
        # engine actually expects rather than silently keeping the previous voice/language.
        tts_settings_kwargs["language"] = resolve_locale_for_language(
            fixed_session_language,
            server=tts_server,
            function_id=tts_function_id,
            model=tts_model,
        )
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
    # Verbalize numbers, currencies, times, units... for EN/FR before synthesis. The
    # NeMo-Speech.cpp French TN grammar misses several of them; other languages pass through.
    tts_text_normalization = parse_env_bool("TTS_TEXT_NORMALIZATION", True)
    if tts_text_normalization:
        tts.add_text_transformer(LanguageAwareVoiceFormatter(lambda: tts._settings.language))

    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}, "
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}, "
        f"lang_codes={lang_codes or '(no voices discovered)'}, "
        f"text_filters=[NemotronSpeechTextFilter], text_normalization={tts_text_normalization}"
    )

    # --- Context ---
    prompt_catalog = load_prompt_catalog(__file__)
    base_system_content = render_prompt_addon(
        base_system_content,
        prompt_catalog,
        FIXED_SESSION_LANGUAGE_ADDON_KEY,
        {"fixed_language_name": describe_language(fixed_session_language)},
    )
    prompt_without_person = base_system_content

    def render_pinned_prompt(person, voice_id_active: bool = False, face_id_active: bool = False) -> str:
        """Prompt catalog content for the person being talked with (re-rendered when voice ID changes it)."""
        content = prompt_without_person
        if voice_id_active:
            content = render_prompt_addon(content, prompt_catalog, VOICE_ID_ADDON_KEY, {})
        if face_id_active:
            content = render_prompt_addon(content, prompt_catalog, FACE_ID_ADDON_KEY, {})
        if person:
            content = render_prompt_addon(
                content, prompt_catalog, PERSON_MEMORY_ADDON_KEY, person.prompt_replacements()
            )
        return content

    # "Who's talking" from the client: pin what the agent remembers about them.
    person_context = await load_person_context(body.get("person_id"))
    if person_context:
        logger.info(f"Person: {person_context.person['name']} ({len(person_context.memories)} memories)")
    base_system_content = render_pinned_prompt(person_context)

    client_tools_schema, client_tool_names = build_client_tools(body.get("tools"))

    messages = build_context_messages(base_system_content, system_prompt)
    if client_tools_schema is not None:
        context = LLMContext(messages, tools=client_tools_schema, tool_choice="auto")
    else:
        context = LLMContext(messages)
    preserve_prompt_messages = len(messages)

    client_tool_bridge = ClientToolResultBridge()
    if client_tool_names:
        client_tool_handler = build_client_tool_handler(CLIENT_FUNCTION_TIMEOUT_SECS, client_tool_bridge)
        for client_tool_name in client_tool_names:
            llm.register_function(client_tool_name, client_tool_handler, cancel_on_interruption=True)
        logger.info(
            f"Registered client-declared tools: {', '.join(client_tool_names)} "
            f"(timeout={CLIENT_FUNCTION_TIMEOUT_SECS:.1f}s)"
        )
    else:
        logger.info("No client tool declarations received; running without tools")

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=_build_multilingual_user_aggregator_params(welcome_enabled),
    )
    logger.info(
        f"Chat history summarization enabled: recent_turns={CHAT_HISTORY_RECENT_TURNS}, "
        f"preserve_prompt_messages={preserve_prompt_messages}"
    )

    reminder_processor = PerTurnReminderProcessor(build_reminder(fixed_session_language))

    # Voice ID: the client streams who is speaking (``speaker-update``); it needs the
    # people store for names, memories and enrollment, so it is off without monitoring.
    voice_id_session: VoiceIdSession | None = None
    if ENROLL_TOOL_NAME in client_tool_names:
        logger.warning(f"Voice ID disabled: the client declares its own '{ENROLL_TOOL_NAME}' tool")
    elif await asyncio.to_thread(voice_id_available):
        voice_id_session = VoiceIdSession(
            context=context,
            # The prompt catalog content is the last initial message (system, or user after a control prompt).
            pinned_index=len(messages) - 1,
            render_pinned=render_pinned_prompt,
            queue_frame=lambda frame: task.queue_frame(frame),
            session_id=lambda: recorder.session_id if recorder else None,
            turn_idx=lambda: recorder.turn_idx if recorder else -1,
            initial_person=person_context,
        )
        # Runs on the server (the people store lives here). It only enters the LLM's
        # tool list once the client sends its first speaker-update.
        llm.register_function(ENROLL_TOOL_NAME, voice_id_session.enroll_speaker, cancel_on_interruption=True)
    voice_id_processors = [SpeakerTurnProcessor(voice_id_session)] if voice_id_session else []

    session_snapshot = _session_snapshot(
        example="multilingual-assistant",
        pipeline_mode=body.get("pipeline_mode", ""),
        transport=type(transport).__name__,
        language=fixed_session_language,
        prompt_key=prompt_key,
        client_tools=client_tool_names,
        client_tools_perceivable=perceivable_client_tools(body.get("tools"), client_tool_names),
        asr={"server": asr_server, "model": asr_model, "function_id": asr_function_id},
        llm={"model": model_id, "base_url": base_url, "temperature": llm_temperature, "extra": base_extra},
        tts={
            "server": tts_server,
            "model": tts_model,
            "voice": tts_voice,
            "synthesis_mode": tts_synthesis_mode,
            "language": tts_settings_kwargs.get("language"),
        },
        person=person_context.snapshot if person_context else None,
    )
    recorder = SessionRecorder.create(
        session_id=body.get("session_id") or None,
        example="multilingual-assistant",
        config=session_snapshot,
        llm=llm,
        language=fixed_session_language,
    )
    # The session recorder owns audio capture when enabled; otherwise keep the
    # legacy ENABLE_*_AUDIO_DUMP per-turn recorder.
    audio_processors = recorder.processors() if recorder else [p for p in [create_audio_recorder()] if p]
    conversation_id = recorder.session_id if recorder else (body.get("session_id") or None)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            *voice_id_processors,
            reminder_processor,
            llm,
            tts,
            transport.output(),
            *audio_processors,
            client_tool_bridge,
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
        ),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=with_realtime_observers(
            latency_observer, *(recorder.observers() if recorder else []), transport=transport
        ),
        enable_tracing=IS_TRACING_ENABLED,
        conversation_id=conversation_id,
        additional_span_attributes=_span_attributes(session_snapshot),
        rtvi_observer_params=RTVIObserverParams(
            function_call_report_level={
                **{name: RTVIFunctionCallReportLevel.FULL for name in client_tool_names},
                # Server-side tool: never reported to the client as a call to run.
                ENROLL_TOOL_NAME: RTVIFunctionCallReportLevel.DISABLED,
            }
        ),
    )

    if recorder:
        recorder.attach(task, user_aggregator, assistant_aggregator, latency_observer)

    if voice_id_session:

        @user_aggregator.event_handler("on_user_turn_started")
        async def on_user_turn_started(aggregator, strategy):
            voice_id_session.on_user_turn_started()

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

    async def _on_session_start() -> None:
        if recorder:
            await recorder.on_session_started()
        else:
            for audio_processor in audio_processors:
                await audio_processor.start_recording()

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
        if recorder:
            # Flush buffered audio while the pipeline is still alive.
            await recorder.stop_audio()
        await task.cancel()

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        payload = message.data if isinstance(message.data, dict) else {}
        if message.type == SPEAKER_UPDATE_MESSAGE:
            if voice_id_session:
                await voice_id_session.on_speaker_update(payload)
            return
        if message.type == PRESENCE_UPDATE_MESSAGE:
            if voice_id_session:
                await voice_id_session.on_presence_update(payload)
            return
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
    if recorder:
        await recorder.start()
        await record_person_session(recorder.session_id, person_context)
    end_reason = "error"
    try:
        await runner.run()
        end_reason = "completed"
    finally:
        if recorder:
            await recorder.close(end_reason)
