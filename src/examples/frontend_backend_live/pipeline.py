# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live pipeline: STT -> talker/thinker/commentary processor -> TTS."""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.tts import NvidiaTTSSettings
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.conversation import ContextConversation, trim_messages
from examples.frontend_backend_live.cascade.processor import LiveTalkerProcessor
from examples.frontend_backend_live.cascade.session import CascadeSession
from examples.frontend_backend_live.cascade.speech import build_stt, build_tts
from examples.frontend_backend_live.config_manager.loader import load_live_config
from examples.frontend_backend_live.delegation.tasks import BackendTask
from examples.frontend_backend_live.models.endpoints import resolve_role_endpoints
from examples.frontend_backend_live.prompts.catalog import CHANNEL_PROMPT_KEY, catalog_prompt, role_instructions
from examples.frontend_backend_live.tool_calling.cafe.schemas import CAFE_TOOLS
from examples.frontend_backend_live.tool_calling.cafe.tools import CafeTools
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.pipeline_utils import (
    build_context_messages,
    build_pipeline_params,
    build_user_aggregator_params,
    create_transport,
    register_session_start_handlers,
    with_realtime_observers,
)
from tracing import IS_TRACING_ENABLED
from utils import (
    load_service_entry,
    normalize_lang_code,
    resolve_prompt,
)

load_dotenv(override=True)

INTRO_PROMPT = (
    "This is an initial greeting, not an order. Greet the user briefly as Remy from Bluebird Cafe and ask what "
    "they would like. Respond directly with spoken text."
)


def _apply_chat_history_sliding_window(context: LLMContext, preserve: int, limit: int) -> None:
    """Keep the prompt messages and the latest conversation messages."""
    messages = context.get_messages()
    trimmed = trim_messages(messages, preserve, limit)
    if trimmed is not messages:
        context.set_messages(trimmed)


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Frontend/Backend Live pipeline for one session."""
    logger.info("Starting Frontend/Backend Live cascaded pipeline")
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    config = load_live_config()

    prompt_key, talker_prompt = resolve_prompt(__file__, body.get("prompt_content", ""), body.get("prompt_key", ""))
    channel = catalog_prompt(CHANNEL_PROMPT_KEY)
    thinker_prompt = role_instructions(config.backend.prompt)

    # The model behind each role comes from its services slot; the session body (UI) can override any field.
    frontend_endpoint, backend_endpoint = resolve_role_endpoints(config, body)
    logger.info(
        f"Frontend: provider={config.frontend.provider}, model={frontend_endpoint.model}, prompt={prompt_key}, "
        f"extra_params={frontend_endpoint.extra_params or '(none)'}"
    )
    logger.info(
        f"Backend: provider={config.backend.provider}, model={backend_endpoint.model}, "
        f"extra_params={backend_endpoint.extra_params or '(none)'}"
    )
    stt = build_stt(body, load_service_entry("asr", ""))
    tts = build_tts(body, load_service_entry("tts", ""))

    default_llm = load_service_entry(config.frontend.slot, "")
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    messages = build_context_messages(f"{talker_prompt}\n\n{channel}", system_prompt)
    context = LLMContext(messages)
    preserve_prompt_messages = len(messages)
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context, user_params=build_user_aggregator_params(welcome_enabled)
    )

    live = LiveTalkerProcessor()
    session = CascadeSession(
        build_agent(config, frontend_endpoint, backend_endpoint),
        config,
        ContextConversation(context),
        backend_task=BackendTask(
            model=backend_endpoint.model, instructions=thinker_prompt, tools=CAFE_TOOLS, tool_choice="auto"
        ),
        tools=CafeTools(),
        say=live.speak,
    )
    live.attach(session)
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            live,
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
            _apply_chat_history_sliding_window(context, preserve_prompt_messages, config.chat_history_recent_turns)

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
        params=build_pipeline_params(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=with_realtime_observers(latency_observer, transport=transport),
        enable_tracing=IS_TRACING_ENABLED,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
        setup_timeout_secs=120.0,
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

    async def _on_session_start() -> None:
        if audio_recorder:
            await audio_recorder.start_recording()

    register_session_start_handlers(
        transport=transport,
        task=task,
        context=context,
        runner_args=runner_args,
        intro_prompt=INTRO_PROMPT,
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

    @task.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        payload = message.data if isinstance(message.data, dict) else {}
        if message.type == "set-voice":
            await _apply_set_voice(payload)

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(task)
    await runner.run()
