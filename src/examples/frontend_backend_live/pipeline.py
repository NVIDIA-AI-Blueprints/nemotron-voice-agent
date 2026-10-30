# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live pipeline: STT -> talker/writer processor -> TTS, with a delegation core of its own."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

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
from delegation_core.app import build_core
from delegation_core.config import CoreSettings
from examples.frontend_backend_live.cascade.agent import build_agent
from examples.frontend_backend_live.cascade.memory import ContextConversation
from examples.frontend_backend_live.config_manager.loader import load_live_config
from examples.frontend_backend_live.config_manager.schema import ConfigError, LiveConfig
from examples.frontend_backend_live.factory import (
    adapter_block,
    backend_defaults,
    core_client,
    endpoint_lookup,
    frontend_adapters,
    once,
    ui_bot_core,
    with_display_label,
)
from examples.frontend_backend_live.media.playout import PlayoutTracker, SpeechPlayout
from examples.frontend_backend_live.media.processor import LiveTalkerProcessor
from examples.frontend_backend_live.media.services import build_stt, build_tts, speech_mode
from examples.frontend_backend_live.media.task_cards import RtviTaskCards
from examples.frontend_backend_live.media.turns import TURN_ENDS
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts.catalog import CHANNEL_PROMPT_KEY, catalog_line, catalog_prompt
from examples.frontend_backend_live.tables import pick
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.pipeline_utils import (
    build_pipeline_params,
    create_transport,
    register_session_start_handlers,
    with_realtime_observers,
)
from live.server_delegation import ServerDelegation
from tracing import IS_TRACING_ENABLED
from utils import (
    normalize_lang_code,
    resolve_prompt,
)

load_dotenv(override=True)

# The session body's fields of the backend's model (the UI's thinker selection) carry this prefix.
BACKEND_BODY_PREFIX = "thinker_"


@dataclass(frozen=True)
class BotRoot:
    """What the UI bot reads once: the cascade's settings, its own core profile, and the example's catalog."""

    cascade: LiveConfig
    core: CoreSettings
    catalog: ExampleCatalog


@once
def bot_root() -> BotRoot:
    """Read the bot's files and the bound example's catalog, once; it blocks, so the bot calls it in a thread.

    The bot hosts the frontend in its own RTVI pipeline, so ``frontend.adapter`` must name a class with ``hosts_rtvi``.
    The core profile is the one ``ui_bot.core_config`` names.

    Raises:
        ConfigError: ``frontend.adapter`` names an adapter the UI bot cannot host, or ``config.yaml`` is malformed.
        ValueError: ``frontend.adapter`` names no frontend adapter, or the core profile is malformed.
    """
    config = load_live_config()
    adapters = frontend_adapters()
    if not pick(adapters, config.adapter, "frontend adapter").hosts_rtvi:
        hosted = ", ".join(name for name, adapter in adapters.items() if adapter.hosts_rtvi) or "none"
        raise ConfigError(
            f"frontend.adapter {config.adapter!r} cannot run in the UI bot, which hosts the frontend in its own RTVI "
            f"pipeline; the frontend adapters it can host: {hosted}"
        )
    core = ui_bot_core(config)
    return BotRoot(LiveConfig.parse(with_display_label(adapter_block(config), core)), core, ExampleCatalog.read())


def backend_overrides(body: dict) -> dict:
    """The session body's settings of the backend's model, without their prefix."""
    return {
        key.removeprefix(BACKEND_BODY_PREFIX): value
        for key, value in body.items()
        if key.startswith(BACKEND_BODY_PREFIX)
    }


def hold_client_text(transport) -> None:
    """Shut a Realtime client's text gate, which the shared start handler opens when its own welcome is off.

    The gate opens when the first response, the greeting, is done (``realtime.conversation``), as the caller's audio
    waits for the greeting's end. A transport that speaks no Realtime has no gate.
    """
    conversation = getattr(getattr(transport, "_realtime_serializer", None), "conversation", None)
    if conversation is not None:
        conversation.assistant_has_responded = False


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Frontend/Backend Live pipeline for one session.

    The cascade greets the caller with its own line when the registry's welcome switch is on, shows each delegation as
    a task card, corrects its records by what the caller heard, and closes when the pipeline has finished.
    """
    logger.info("Starting Frontend/Backend Live cascaded pipeline")
    transport = create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    state = await asyncio.to_thread(bot_root)
    cascade, catalog = state.cascade, state.catalog

    prompt_key, talker_prompt = resolve_prompt(__file__, body.get("prompt_content", ""), body.get("prompt_key", ""))
    channel = catalog_prompt(CHANNEL_PROMPT_KEY)

    # The model behind each role comes from its services slot; the session body (UI) can override any field.
    frontend_endpoint = catalog.model(cascade.slot, body)
    endpoints = endpoint_lookup(catalog, state.core.backend.slot, backend_overrides(body))
    logger.info(
        f"Frontend: provider={cascade.provider}, model={frontend_endpoint.model}, prompt={prompt_key}, "
        f"extra_params={frontend_endpoint.extra_params or '(none)'}"
    )
    logger.info(f"Backend: adapter={state.core.backend.adapter}, model={endpoints(None).model}")
    # The turn end (speech.turn_end) builds the speech-to-text and the turn detection, as in a live session.
    turn_end = pick(TURN_ENDS, cascade.speech.turn_end, "speech.turn_end")(cascade.speech)
    stt = build_stt(body, catalog.service("asr"), turn_end)
    tts = build_tts(body, catalog.service("tts"))
    # What the caller heard corrects the cascade's records, as in a live session. The text-to-speech runs in the
    # catalog's synthesis mode, and the tracker sends no transcript: the RTVI client gets Pipecat's own.
    speech_playout, tracker = SpeechPlayout(speech_mode(tts)), PlayoutTracker()

    default_llm = catalog.entry(cascade.slot)
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    # The seed is the talker's instructions as system messages, which its memory pins: the services entry's
    # system_prompt when it sets one, then the talker's prompt and the channel. A user message would be a caller turn
    # there, which compaction drops.
    seed = (system_prompt, f"{talker_prompt}\n\n{channel}")
    messages = [{"role": "system", "content": text} for text in seed if text]
    context = LLMContext(messages)  # the talker's memory: only the cascade trims it, at each caller turn
    # Only the caller's turns come from Pipecat's aggregator; the cascade writes its own lines (cascade.utterances).
    user_aggregator = LLMContextAggregatorPair(context, user_params=turn_end.user_params(welcome_enabled)).user()

    # The bot's own delegation core runs the tools its profile names (ui_bot.core_config) on the server. Its server
    # delegation has no Live sink, so each delegation's events go to the cascade only, and the core runs every request
    # on its defaults.
    core = build_core(state.core, endpoints=endpoints, defaults=backend_defaults(state.core, endpoints))
    delegation = ServerDelegation(core_client(core))
    live = LiveTalkerProcessor()
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            live,
            *speech_playout.speech_out(tts),
            transport.output(),
            tracker,
            *([audio_recorder] if audio_recorder else []),
        ]
    )

    latency_observer = UserBotLatencyObserver()

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

    # The cascade shows each delegation as a task card in the UI, the display part of its result included.
    session = speech_playout.cascade(
        tracker,
        build_agent(cascade, frontend_endpoint),
        cascade,
        ContextConversation(context),
        delegation=delegation,
        say=live.speak,
        display=RtviTaskCards(task.queue_frame),
    )
    live.attach(session)
    cascade_open = True

    async def close_cascade() -> None:
        """Close the cascade once (its presenter, its timers and the talker's model); a failed close is logged."""
        nonlocal cascade_open
        if not cascade_open:
            return
        cascade_open = False
        try:
            await session.close()
        except Exception:
            logger.exception("The cascade could not close")

    @task.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(task, frame):
        # Nothing is worded or said once the pipeline has ended: a result that arrives later stays unsaid.
        await close_cascade()

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
        if welcome_enabled:
            session.greet(catalog_line("intro"))  # said before the caller speaks, as the assistant's own line

    # The shared start handler runs with its welcome off: the greeting is the cascade's line, never a caller turn.
    # The caller's audio still waits for it when the welcome is on (the aggregator follows the same switch), and so
    # does a Realtime client's text, whose gate that handler opens.
    register_session_start_handlers(
        transport=transport,
        task=task,
        context=context,
        runner_args=runner_args,
        on_start=_on_session_start,
        welcome_enabled=False,
    )
    if welcome_enabled:
        hold_client_text(transport)

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
    try:
        await delegation.open(None)  # the conversation, whose IM session's loop starts; no view: nothing is relayed
        session.start()  # the presenter's task and the silence timer
        await runner.add_workers(task)
        await runner.run()
    finally:
        # Close once each, in a live session's order: the cascade, the server delegation (which deletes the IM session
        # and its backend adapter), then the core. A failed or cancelled close still runs the later ones.
        try:
            await close_cascade()
        finally:
            try:
                await delegation.close()
            finally:
                await core.aclose()
