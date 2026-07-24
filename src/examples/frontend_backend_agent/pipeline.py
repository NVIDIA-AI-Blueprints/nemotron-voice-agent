# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Frontend/Backend Agent cascaded pipeline: STT -> Talker LLM -> TTS with one Thinker tool."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMRunFrame, TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.llm import NvidiaLLMService, NvidiaLLMSettings
from pipecat.services.nvidia.stt import NvidiaSTTService, NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.transports.base_transport import TransportParams
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.frontend_backend_agent.airline.backend import HTTPBookingBackend
from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.airline.tools import TOOLS_SCHEMA
from examples.frontend_backend_agent.src.planner import NvidiaThinkerPlanner
from examples.frontend_backend_agent.src.runtime_context import runtime_today
from examples.frontend_backend_agent.src.tool_handlers import build_handlers
from examples.frontend_backend_agent.src.tts_filter import apply_frontend_backend_agent_pronunciation_for_tts
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import build_user_aggregator_params
from tracing import IS_TRACING_ENABLED
from utils import (
    is_nvcf,
    load_ipa_dictionary,
    load_prompt_catalog,
    load_service_entry,
    normalize_lang_code,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
)

load_dotenv(override=True)

CHAT_HISTORY_RECENT_TURNS = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 20)
THINKER_PROMPT_KEY = "thinker"
THINKER_TOOL_DELAY_MIN_SECONDS = 0.1
THINKER_TOOL_DELAY_MAX_SECONDS = 0.5
THINKER_FILLER_THRESHOLD_SECONDS = parse_env_float("THINKER_FILLER_THRESHOLD_SECONDS", 0.3, min_value=0.0)
THINKER_TOOL_TIMEOUT_SECONDS = parse_env_float("THINKER_TOOL_TIMEOUT_SECONDS", 30.0, min_value=1.0)


def _build_context_messages(base_prompt: str, system_prompt: str = "") -> list[dict]:
    """Build initial Talker context messages."""
    today = runtime_today()
    runtime_context = (
        f"\n\nRuntime context:\n"
        f"- Today is {today.isoformat()}.\n"
        f"- Tomorrow is {(today + timedelta(days=1)).isoformat()}.\n"
        "- For travel dates without a year, choose the next upcoming occurrence relative to today.\n"
        "- Always pass travel dates to call_backend as ISO YYYY-MM-DD when the date is known."
    )
    base_prompt = f"{base_prompt}{runtime_context}"
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_prompt},
        ]
    return [{"role": "system", "content": base_prompt}]


def _apply_chat_history_sliding_window(
    context: LLMContext,
    preserve_prompt_messages: int,
    chat_history_limit: int,
) -> None:
    """Keep the prompt messages and latest conversation turns."""
    if chat_history_limit < 1:
        return
    messages = context.get_messages()
    preserve = max(0, preserve_prompt_messages)
    if len(messages) <= preserve + chat_history_limit:
        return
    context.set_messages(messages[:preserve] + messages[preserve:][-chat_history_limit:])


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Frontend/Backend Agent cascaded pipeline for one session."""
    logger.info("Starting Frontend/Backend Agent cascaded pipeline")
    transport = _create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))

    prompt_key, talker_prompt = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    thinker_prompt = _load_required_catalog_prompt(THINKER_PROMPT_KEY)
    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")
    default_asr = load_service_entry("asr", "")
    default_booking_server = load_service_entry("booking-server", "")
    default_thinker_llm = load_service_entry("thinker-llm", "")

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

    # --- Talker LLM ---
    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3-nano-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    talker_max_tokens = _parse_optional_int(body.get("max_tokens", "") or default_llm.get("max_tokens"), 2048)
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )
    llm_settings = NvidiaLLMSettings(model=model_id, max_tokens=talker_max_tokens)
    if extra_params:
        llm_settings.extra = extra_params
    talker_llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=base_url,
        settings=llm_settings,
    )
    logger.info(
        f"Talker LLM: model={model_id}, base_url={base_url}, prompt={prompt_key}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"max_tokens={talker_max_tokens}, "
        f"extra_params={extra_params or '(none)'}"
    )

    booking_backend_url = _booking_backend_url(default_booking_server)
    thinker_model_id = body.get("thinker_model_id", "") or default_thinker_llm.get("model_id", "") or model_id
    thinker_base_url = body.get("thinker_base_url", "") or default_thinker_llm.get("base_url", "") or base_url
    thinker_max_tokens = _parse_optional_int(
        body.get("thinker_max_tokens", "") or default_thinker_llm.get("max_tokens"),
        4096,
    )
    thinker_extra_params = parse_json_dict(
        body.get("thinker_extra_params", "") or default_thinker_llm.get("extra_params", ""),
        label="thinker_extra_params",
    )
    thinker_llm_settings = NvidiaLLMSettings(model=thinker_model_id, max_tokens=thinker_max_tokens)
    if thinker_extra_params:
        thinker_llm_settings.extra = thinker_extra_params
    thinker_llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=thinker_base_url,
        settings=thinker_llm_settings,
    )
    thinker_planner = NvidiaThinkerPlanner(
        llm=thinker_llm,
        system_prompt=thinker_prompt,
        max_tokens=thinker_max_tokens,
    )
    thinker = ThinkerBackend(
        backend=HTTPBookingBackend(booking_backend_url),
        planner=thinker_planner,
        tool_delay_seconds=THINKER_TOOL_DELAY_MAX_SECONDS,
        tool_delay_min_seconds=THINKER_TOOL_DELAY_MIN_SECONDS,
    )
    logger.info(f"Thinker booking backend: {booking_backend_url}")
    logger.info(
        f"Thinker LLM: model={thinker_model_id}, base_url={thinker_base_url}, "
        f"max_tokens={thinker_max_tokens}, extra_params={thinker_extra_params or '(none)'}"
    )
    logger.info(f"Thinker tool delay: {THINKER_TOOL_DELAY_MIN_SECONDS:.3f}s-{THINKER_TOOL_DELAY_MAX_SECONDS:.3f}s")
    logger.info(f"Thinker filler threshold: {THINKER_FILLER_THRESHOLD_SECONDS:.3f}s")
    logger.info(f"Thinker tool timeout: {THINKER_TOOL_TIMEOUT_SECONDS:.3f}s")
    for name, handler in build_handlers(
        thinker,
        filler_threshold_seconds=THINKER_FILLER_THRESHOLD_SECONDS,
    ).items():
        cancel_on_interruption = name != "call_backend"
        talker_llm.register_function(
            name,
            handler,
            cancel_on_interruption=cancel_on_interruption,
            timeout_secs=THINKER_TOOL_TIMEOUT_SECONDS,
        )
        logger.info(f"Registered Talker tool: {name}, cancel_on_interruption={cancel_on_interruption}")

    # --- TTS ---
    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    custom_dictionary = load_ipa_dictionary()
    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    tts_kwargs: dict = {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "server": tts_server,
        "settings": NvidiaTTSSettings(**tts_settings_kwargs),
        "use_ssl": tts_ssl,
        "text_filters": [NemotronSpeechTextFilter()],
        "text_transforms": [("*", apply_frontend_backend_agent_pronunciation_for_tts)],
        "custom_dictionary": custom_dictionary,
    }
    if tts_function_id or tts_model:
        tts_kwargs["model_function_map"] = {
            "function_id": tts_function_id,
            "model_name": tts_model,
        }
    tts = NvidiaTTSService(**tts_kwargs)
    logger.info(
        f"TTS: server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}"
    )

    # --- Context + aggregators ---
    messages = _build_context_messages(talker_prompt, system_prompt)
    context = LLMContext(messages, tools=TOOLS_SCHEMA, tool_choice="auto")
    preserve_prompt_messages = len(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(welcome_enabled),
    )
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            talker_llm,
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
            _apply_chat_history_sliding_window(context, preserve_prompt_messages, CHAT_HISTORY_RECENT_TURNS)

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency):
        logger.info(f"First bot speech latency: {latency:.3f}s")
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": True})
        )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        logger.info(f"User-to-bot latency: {latency:.3f}s")
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": False})
        )

    task = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
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
        if not welcome_enabled:
            logger.info("Welcome message disabled; waiting for the user to speak first")
            return
        context.add_message({"role": "user", "content": "Please greet the user briefly."})
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
        await task.queue_frame(TTSUpdateSettingsFrame(delta=NvidiaTTSSettings(**settings_kwargs), service=tts))
        logger.info(f"Voice switched to {voice_id}, language={settings_kwargs.get('language', '(unchanged)')}")

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        payload = message.data if isinstance(message.data, dict) else {}
        if message.type == "set-voice":
            await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()


def _create_transport(runner_args: RunnerArguments):
    """Create a transport from runner arguments."""
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


def _default_booking_backend_url() -> str:
    """Return the default booking-server URL for the current runtime."""
    if os.environ.get("APP_RUNTIME", "").strip().lower() == "container":
        return "http://booking-server:8001"
    return "http://localhost:8001"


def _booking_backend_url(default_booking_server: dict) -> str:
    """Resolve the booking-server URL, preserving explicit user overrides."""
    explicit_url = os.getenv("BOOKING_BACKEND_URL", "").strip()
    if explicit_url:
        return explicit_url

    configured_url = str(default_booking_server.get("server") or "").strip()
    runtime_default = _default_booking_backend_url()
    if runtime_default == "http://localhost:8001" and configured_url == "http://booking-server:8001":
        return runtime_default
    return configured_url or runtime_default


def _parse_optional_int(raw: object, default: int) -> int:
    """Parse optional integer config values."""
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"Invalid integer config value {raw!r}; using {default}")
        return default


def _load_required_catalog_prompt(prompt_key: str) -> str:
    """Load an internal prompt from this example's prompt catalog."""
    catalog = load_prompt_catalog(__file__)
    entry = catalog.get(prompt_key)
    if not isinstance(entry, dict):
        raise KeyError(f"Prompt {prompt_key!r} was not found in Frontend/Backend Agent prompts.yaml")
    content = str(entry.get("content") or "").strip()
    if not content:
        raise KeyError(f"Prompt {prompt_key!r} has no content in Frontend/Backend Agent prompts.yaml")
    return content
