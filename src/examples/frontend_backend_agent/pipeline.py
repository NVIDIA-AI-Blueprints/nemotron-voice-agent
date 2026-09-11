# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Frontend/Backend Agent cascaded pipeline: STT -> Talker LLM -> TTS with one Thinker tool."""

from __future__ import annotations

import asyncio
import os
from datetime import timedelta

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TTSUpdateSettingsFrame
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
from examples.frontend_backend_agent.airline.backend import HTTPBookingBackend
from examples.frontend_backend_agent.airline.thinker import ThinkerBackend
from examples.frontend_backend_agent.airline.tools import TOOLS_SCHEMA
from examples.frontend_backend_agent.src.planner import NvidiaThinkerPlanner
from examples.frontend_backend_agent.src.runtime_context import runtime_today
from examples.frontend_backend_agent.src.tool_handlers import build_handlers
from examples.frontend_backend_agent.src.tts_filter import apply_frontend_backend_agent_pronunciation_for_tts
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
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
    load_prompt_catalog,
    load_selected_service_entry,
    load_service_entry,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_float,
    parse_env_int,
    parse_json_dict,
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
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    is_realtime = runner_protocol(runner_args) == "realtime"
    if is_realtime:
        from examples.shared.nvidia_llm import NvidiaLLMService as RealtimeNvidiaLLMService
        from examples.shared.tool_runtime import select_trusted_tools, terminal_tool_handler, tool_parameter_schema
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

    prompt_key, talker_prompt = resolve_pipeline_prompt(__file__, body, is_realtime=is_realtime)
    thinker_prompt = _load_required_catalog_prompt(THINKER_PROMPT_KEY)
    selected_llm_id = str(body.get("llm_id", "") or "")
    default_llm = load_selected_service_entry("llm", selected_llm_id) if is_realtime else load_service_entry("llm", "")
    llm_profile = default_llm
    default_tts = (
        load_selected_service_entry("tts", body.get("tts_id")) if is_realtime else load_service_entry("tts", "")
    )
    default_asr = (
        load_selected_service_entry("asr", body.get("asr_id")) if is_realtime else load_service_entry("asr", "")
    )
    default_thinker_llm = (
        load_selected_service_entry("thinker-llm", body.get("thinker_llm_id"))
        if is_realtime
        else load_service_entry("thinker-llm", "")
    )
    default_booking_server = load_service_entry("booking-server", "")

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

    # --- Talker LLM ---
    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    raw_talker_max_tokens = select_max_tokens_config(
        body,
        default_llm.get("max_tokens"),
        is_realtime=is_realtime,
    )
    talker_max_tokens = (
        None if is_realtime and raw_talker_max_tokens is None else _parse_optional_int(raw_talker_max_tokens, 2048)
    )
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        label="extra_params",
    )
    llm_settings = NvidiaLLMSettings(model=model_id)
    if talker_max_tokens is not None:
        llm_settings.max_tokens = talker_max_tokens
    if extra_params:
        llm_settings.extra = extra_params
    if is_realtime:
        talker_llm = RealtimeNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=base_url,
            settings=llm_settings,
            forced_tool_call_stops=llm_profile.get("forced_tool_call_stops"),
            realtime_parallel_tool_calls=body.get("parallel_tool_calls", True),
            realtime_model_max_output_tokens=body.get("realtime_model_max_output_tokens"),
        )
    else:
        talker_llm = PipecatNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=base_url,
            settings=llm_settings,
        )
    logger.info(
        f"Talker LLM: model={model_id}, base_url={base_url}, prompt={prompt_key}, "
        f"system_prompt={'<' + system_prompt + '>' if system_prompt else '(none)'}, "
        f"max_tokens={talker_max_tokens if talker_max_tokens is not None else '(provider maximum)'}, "
        f"extra_params={extra_params or '(none)'}"
    )

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
    booking_backend_url = _booking_backend_url(default_booking_server)
    thinker_llm_settings = NvidiaLLMSettings(model=thinker_model_id, max_tokens=thinker_max_tokens)
    if thinker_extra_params:
        thinker_llm_settings.extra = thinker_extra_params
    if is_realtime:
        thinker_llm = RealtimeNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=thinker_base_url,
            settings=thinker_llm_settings,
        )
    else:
        thinker_llm = PipecatNvidiaLLMService(
            api_key=nvidia_api_key(),
            base_url=thinker_base_url,
            settings=thinker_llm_settings,
        )
    thinker_planner = NvidiaThinkerPlanner(
        llm=thinker_llm,
        system_prompt=thinker_prompt,
        max_tokens=thinker_max_tokens,
        structured_output=is_realtime,
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
    available_talker_handlers = build_handlers(
        thinker,
        filler_threshold_seconds=THINKER_FILLER_THRESHOLD_SECONDS,
        allow_talker_frames=not is_realtime,
    )
    if is_realtime:
        raw_delegate_tools = body.get("delegate_tools", [])
        if not isinstance(raw_delegate_tools, list) or not all(isinstance(name, str) for name in raw_delegate_tools):
            raise ValueError("delegate_tools must be a list of trusted function names")
        active_delegate_tools = list(dict.fromkeys(raw_delegate_tools))
        missing_delegate_handlers = set(active_delegate_tools) - set(available_talker_handlers)
        if missing_delegate_handlers:
            raise ValueError(
                f"Trusted pipeline tool {sorted(missing_delegate_handlers)[0]!r} has no registered handler"
            )
        talker_handlers = {name: available_talker_handlers[name] for name in active_delegate_tools}
        trusted_tools_schema = select_trusted_tools(TOOLS_SCHEMA, active_delegate_tools)
    else:
        talker_handlers = available_talker_handlers
        trusted_tools_schema = TOOLS_SCHEMA
    for name, original_handler in talker_handlers.items():
        if is_realtime:
            handler = terminal_tool_handler(
                original_handler,
                parameters=tool_parameter_schema(trusted_tools_schema, name),
                timeout_secs=THINKER_TOOL_TIMEOUT_SECONDS,
            )
            # A Realtime call is a durable conversation item, so barge-in
            # cancels response media but not its accepted backend result.
            cancel_on_interruption = False
            talker_llm.register_function(
                name,
                handler,
                cancel_on_interruption=cancel_on_interruption,
            )
        else:
            cancel_on_interruption = name != "call_backend"
            talker_llm.register_function(
                name,
                original_handler,
                cancel_on_interruption=cancel_on_interruption,
                timeout_secs=THINKER_TOOL_TIMEOUT_SECONDS,
            )
        logger.info(f"Registered Talker tool: {name}, cancel_on_interruption={cancel_on_interruption}")
    tools_schema = trusted_tools_schema
    tool_choice = body.get("tool_choice", "auto") or "auto"
    if is_realtime:
        tools_schema = configure_realtime_client_tools(
            transport,
            talker_llm,
            body.get("client_tools"),
            trusted_tools=trusted_tools_schema,
            trusted_tool_names=talker_handlers,
        )
        tools_schema, tool_choice = await prepare_realtime_tools(transport, talker_llm)

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
        "text_transforms": [("*", apply_frontend_backend_agent_pronunciation_for_tts)],
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
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}"
    )

    # --- Context + aggregators ---
    def render_realtime_instructions(instructions: str) -> list[dict]:
        return _build_context_messages(instructions, system_prompt)

    messages = render_realtime_instructions(talker_prompt)
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
        bind_realtime_deferred_service_responses(transport, talker_llm)
    preserve_prompt_messages = len(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=build_user_aggregator_params(
            welcome_enabled,
            transport=transport if is_realtime else None,
        ),
    )
    audio_recorder = create_audio_recorder()

    response_gate_processors = realtime_response_gate_processors(transport) if is_realtime else []
    tool_result_processors = realtime_tool_result_processors(transport) if is_realtime else []
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            *response_gate_processors,
            talker_llm,
            *tool_result_processors,
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
        intro_prompt="Please greet the user briefly.",
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
        await task.queue_frame(TTSUpdateSettingsFrame(delta=NvidiaTTSSettings(**settings_kwargs), service=tts))
        logger.info(f"Voice switched to {voice_id}, language={settings_kwargs.get('language', '(unchanged)')}")

    if not is_realtime:

        @task.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            payload = message.data if isinstance(message.data, dict) else {}
            if message.type == "set-voice":
                await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()


def _parse_optional_int(raw: object, default: int) -> int:
    """Parse optional integer config values."""
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning(f"Invalid integer config value {raw!r}; using {default}")
        return default


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
