# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Multilingual cascaded pipeline: NVIDIA STT -> Nemotron LLM -> Magpie TTS.

Always runs in multilingual mode: the LLM emits ``Language: <code> Text: <reply>
MetaData: <info>`` and the pipeline switches the TTS voice on every detected
language change.
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
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.processors.frameworks.rtvi import RTVIObserverParams
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import config_store
from examples.multilingual.multilingual_processor import (
    AUTO_DETECT_LANGUAGE_ADDON_KEY,
    FIXED_SESSION_GREETING_TRIGGER,
    FIXED_SESSION_LANGUAGE_ADDON_KEY,
    SKIP_TTS_AGGREGATIONS,
    MultilingualTextAggregator,
    RTVISpokenTextEmitter,
    get_lang_codes,
    make_language_handler,
)
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    apply_pinned_prompt_summary,
    build_context_messages,
    create_transport,
)
from examples.shared.prewarm import prewarm_asr, prewarm_tts, resolve_voice_for_language
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_prompt_catalog,
    load_service_entry,
    normalize_lang_code,
    parse_env_bool,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    render_prompt_addon,
    resolve_prompt,
)

load_dotenv(override=True)
CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)


def _build_multilingual_user_aggregator_params() -> LLMUserAggregatorParams:
    """Use VAD-only turn starts so interim ASR text does not start a user turn."""
    if not parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False):
        return LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
            user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy()]),
        )

    stop_secs = parse_env_float("SILERO_VAD_STOP_SECS", 0.5, min_value=0.0)
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.0)],
        ),
    )


async def _build_multilingual_pipeline(
    tts_server: str,
    tts_voice: str,
    asr_server: str,
    asr_model: str,
    asr_function_id: str,
) -> tuple[MultilingualTextAggregator, LLMTextProcessor, RTVISpokenTextEmitter, str]:
    """Prewarm the voice catalog and build the multilingual processors."""
    if not config_store.get("tts"):
        await asyncio.to_thread(prewarm_tts, tts_server, tts_voice)
    await asyncio.to_thread(prewarm_asr, asr_server, asr_model, asr_function_id)
    aggregator = MultilingualTextAggregator()
    text_processor = LLMTextProcessor(text_aggregator=aggregator)
    lang_codes = get_lang_codes(
        asr_server=asr_server,
        asr_model=asr_model,
        asr_function_id=asr_function_id,
        tts_server=tts_server,
        tts_voice_id=tts_voice,
    )
    return aggregator, text_processor, RTVISpokenTextEmitter(), lang_codes


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the multilingual NVIDIA cascaded pipeline for a single session."""
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    logger.info(f"Starting multilingual cascaded pipeline (prompt={prompt_key})")
    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")
    default_asr = load_service_entry("asr", "")

    # --- ASR ---
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
    if asr_language_code and asr_language_code.strip().lower() == "auto":
        asr_language_code = ""
    fixed_session_language = normalize_lang_code(asr_language_code) if asr_language_code else ""
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
        f"language={fixed_session_language or 'auto-detect'}"
    )

    # --- LLM ---
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

    # --- TTS ---
    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "Magpie-Multilingual.EN-US.Aria")
    custom_dictionary = load_ipa_dictionary()
    if not config_store.get("tts"):
        await asyncio.to_thread(prewarm_tts, tts_server, tts_voice)

    tts_settings_kwargs: dict = {"voice": tts_voice}
    if fixed_session_language:
        tts_settings_kwargs["language"] = fixed_session_language
        resolved_voice = resolve_voice_for_language(fixed_session_language, tts_voice)
        if resolved_voice:
            tts_voice = resolved_voice
            tts_settings_kwargs["voice"] = resolved_voice

    tts = NvidiaTTSService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        server=tts_server,
        settings=NvidiaTTSSettings(**tts_settings_kwargs),
        use_ssl=tts_ssl,
        text_filters=[NemotronSpeechTextFilter()],
        custom_dictionary=custom_dictionary,
        skip_aggregator_types=list(SKIP_TTS_AGGREGATIONS),
    )

    (
        multilingual_aggregator,
        multilingual_text_processor,
        multilingual_rtvi_emitter,
        lang_codes,
    ) = await _build_multilingual_pipeline(
        tts_server,
        tts_voice,
        asr_server,
        asr_model,
        asr_function_id,
    )

    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"lang_codes={lang_codes or '(no voices discovered)'}, "
        f"text_filters=[NemotronSpeechTextFilter]"
    )

    # --- Context ---
    if fixed_session_language:
        prompt_catalog = load_prompt_catalog(__file__)
        base_system_content = base_system_content.replace("{lang_codes}", fixed_session_language)
        base_system_content = render_prompt_addon(
            base_system_content,
            prompt_catalog,
            FIXED_SESSION_LANGUAGE_ADDON_KEY,
            {"fixed_language": fixed_session_language, "lang_codes": fixed_session_language},
        )
    elif lang_codes:
        prompt_catalog = load_prompt_catalog(__file__)
        base_system_content = base_system_content.replace("{lang_codes}", lang_codes)
        base_system_content = render_prompt_addon(
            base_system_content,
            prompt_catalog,
            AUTO_DETECT_LANGUAGE_ADDON_KEY,
            {"lang_codes": lang_codes},
        )

    messages = build_context_messages(base_system_content, system_prompt)
    context = LLMContext(messages)
    preserve_prompt_messages = len(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=_build_multilingual_user_aggregator_params(),
    )
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
            multilingual_text_processor,
            multilingual_rtvi_emitter,
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
        rtvi_observer_params=RTVIObserverParams(
            ignored_sources=[llm],
            skip_aggregator_types=list(SKIP_TTS_AGGREGATIONS),
        ),
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

    async def _notify_language_switched(language: str, voice_id: str) -> None:
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "language-switched",
                    "language": language,
                    "voice_id": voice_id,
                }
            )
        )

    multilingual_aggregator.set_on_language(
        make_language_handler(
            tts,
            task,
            on_language_switched=_notify_language_switched,
            fixed_language=fixed_session_language,
        )
    )

    @task.rtvi.event_handler("on_client_ready")
    async def on_client_connected(rtvi):
        logger.info("Client connected")
        if audio_recorder:
            await audio_recorder.start_recording()
        if fixed_session_language:
            context.add_message({"role": "user", "content": FIXED_SESSION_GREETING_TRIGGER})
        else:
            # Normal greeting so the LLM auto-detects language on the first LLMRunFrame turn.
            context.add_message({"role": "user", "content": "Please introduce yourself to the user."})
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

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
