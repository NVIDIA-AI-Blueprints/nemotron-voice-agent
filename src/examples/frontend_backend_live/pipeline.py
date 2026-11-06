# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Live pipeline: STT -> the talker (Pipecat's LLM service) -> TTS, with a delegation core of its own.

A session's talker prompt is the ``prompts.yaml`` entry it selects, as in every example. Its backend prompt and server
tools are those of its own core (the core's settings with ``ui_bot.core`` merged over them, in the example's
``examples_registry.yaml`` entry), and the channel context that ``ui_bot.channel`` names follows both prompts.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import TTSUpdateSettingsFrame
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.tts import NvidiaTTSSettings
from pipecat.workers.runner import WorkerRunner

import examples_registry
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.delegation_core.app import build_core
from examples.frontend_backend_live.delegation_core.config import CoreSettings
from examples.frontend_backend_live.delegation_core.items import message_text
from examples.frontend_backend_live.delegation_core.tables import pick
from examples.frontend_backend_live.factory import (
    adapter_block,
    backend_defaults,
    core_client,
    endpoint_lookup,
    example_config,
    frontend_adapters,
    once,
    ui_bot_core,
    ui_bot_instructions,
    with_display_label,
)
from examples.frontend_backend_live.frontend import FrontendEdges, FrontendSpec, build_frontend
from examples.frontend_backend_live.media.services import build_stt, build_tts
from examples.frontend_backend_live.media.task_cards import RtviTaskCards
from examples.frontend_backend_live.media.turns import SmartTurn
from examples.frontend_backend_live.models.endpoints import ExampleCatalog
from examples.frontend_backend_live.prompts import load_prompts
from examples.frontend_backend_live.talker import LLM_SERVICES, RESPONSE_FORMATS
from examples.shared.audio_recorder import create_audio_recorder
from examples.shared.pipeline_utils import (
    build_pipeline_params,
    create_transport,
    latency_metric_group,
    register_session_start_handlers,
    turn_taking_latency_secs,
    with_realtime_observers,
)
from examples.shared.speaker import finalized_turn_payloads, send_rtvi_payloads
from gateways.live.server_delegation import ServerDelegation
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
    """What the UI bot reads once: the cascade's settings, its core and channel, and the catalog."""

    cascade: LiveConfig
    core: CoreSettings
    channel: str  # ui_bot.channel: the agent_prompts.Channel entry after the talker and backend prompts
    catalog: ExampleCatalog


@once
def bot_root() -> BotRoot:
    """Read the bot's settings and the bound example's catalog, once; it blocks, so the bot calls it in a thread.

    The bot's core is the core's settings with ``ui_bot.core`` merged over them.

    Raises:
        ValueError: The example's settings are malformed, ``frontend.adapter`` names no frontend adapter, or the
            bot's core is malformed.
    """
    config = example_config()
    pick(frontend_adapters(), config.frontend.adapter, "frontend adapter")  # an unknown frontend.adapter is refused
    core = ui_bot_core(config)
    cascade = LiveConfig.model_validate(with_display_label(adapter_block(config), core))
    return BotRoot(cascade, core, config.ui_bot.channel, ExampleCatalog.read())


def backend_overrides(body: dict) -> dict:
    """The session body's settings of the backend's model, without their prefix."""
    return {
        key.removeprefix(BACKEND_BODY_PREFIX): value
        for key, value in body.items()
        if key.startswith(BACKEND_BODY_PREFIX)
    }


def hold_client_text(transport) -> None:
    """Shut a Realtime client's text gate, which the shared start handler opens when its own welcome is off.

    The gate opens when the first response, the greeting, is done (``gateways.realtime.conversation``), as the caller's
    audio waits for the greeting's end. A transport that speaks no Realtime has no gate.
    """
    conversation = getattr(getattr(transport, "_realtime_serializer", None), "conversation", None)
    if conversation is not None:
        conversation.assistant_has_responded = False


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Frontend/Backend Live pipeline for one session.

    The voice greets the caller with its own line when the registry's welcome switch is on, shows each delegation as a
    task card, and closes when the pipeline has finished; the assistant aggregator records what the caller heard.
    """
    logger.info("Starting Frontend/Backend Live cascaded pipeline")
    transport = await create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    welcome_enabled = examples_registry.welcome_message_enabled(body.get("pipeline_mode", ""))
    state = await asyncio.to_thread(bot_root)
    cascade, catalog = state.cascade, state.catalog

    # The session's talker prompt, as in every example; one read of the example's prompt file gives the rest.
    prompt_key, talker_prompt = resolve_prompt(__file__, body.get("prompt_content", ""), body.get("prompt_key", ""))
    prompts = load_prompts()
    channel = prompts.text("Channel", state.channel)
    core_settings, backend_instructions = state.core, ui_bot_instructions(prompts, channel)

    # The model behind each role comes from its services slot; the session body (UI) can override any field. The
    # cascade's settings name the talker's style by its services key, and its services entry names its level.
    frontend_endpoint = catalog.model(cascade.slot, body)
    talker_style = cascade.style_of(frontend_endpoint.service_key)
    talker = pick(LLM_SERVICES, talker_style, "talker_style")
    response_format = pick(RESPONSE_FORMATS, frontend_endpoint.response_format, "response_format")
    endpoints = endpoint_lookup(catalog, core_settings.backend.slot, backend_overrides(body))
    logger.info(
        f"Frontend: talker_style={talker_style}, model={frontend_endpoint.model}, "
        f"prompt={prompt_key}, extra_params={frontend_endpoint.extra_params or '(none)'}"
    )
    logger.info(f"Backend: adapter={core_settings.backend.adapter}, model={endpoints(None).model}")
    # The caller's turns (speech) build the turn detection, as in a live session.
    turn_end = SmartTurn(cascade.speech)
    stt = build_stt(body, catalog.service("asr"))
    tts = build_tts(body, catalog.service("tts"))

    default_llm = catalog.entry(cascade.slot)
    system_prompt = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    # The seed is the talker's instructions as system messages, which its memory pins: the services entry's
    # system_prompt when it sets one, then the talker's prompt and the channel. A user message would be a caller turn
    # there, which compaction drops.
    seed = (system_prompt, f"{talker_prompt}\n\n{channel}")
    messages = [{"role": "system", "content": text} for text in seed if text]

    # The bot's own delegation core runs the tools its core settings name on the server. Its server delegation has no
    # Live sink, so each delegation's events go to the voice only, and the core runs every request on its defaults.
    defaults = backend_defaults(core_settings, endpoints, backend_instructions)
    core = build_core(core_settings, endpoints=endpoints, defaults=defaults)
    delegation = ServerDelegation(core_client(core))
    audio_recorder = create_audio_recorder()

    latency_observer = UserBotLatencyObserver()
    latency_turn = 1
    latest_latency: float | None = None

    @latency_observer.event_handler("on_first_bot_speech_latency")
    async def on_first_bot_speech(observer, latency):
        logger.info(f"First bot speech latency: {latency:.3f}s")
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": True})
        )

    @latency_observer.event_handler("on_latency_measured")
    async def on_latency(observer, latency):
        nonlocal latest_latency
        latest_latency = latency
        logger.info(f"User-to-bot latency: {latency:.3f}s")
        await task.queue_frame(
            RTVIServerMessageFrame(data={"type": "user-bot-latency", "latency": round(latency, 3), "first": False})
        )

    @latency_observer.event_handler("on_latency_breakdown")
    async def on_breakdown(observer, breakdown):
        nonlocal latency_turn, latest_latency
        if latest_latency is not None:
            group = latency_metric_group(latency_turn, latest_latency, breakdown)
            await task.queue_frame(RTVIServerMessageFrame(data=group))
            latency_turn += 1
            latest_latency = None
        events = breakdown.turn_contribution_lines()
        await task.queue_frame(
            RTVIServerMessageFrame(
                data={
                    "type": "latency-breakdown",
                    "turn_taking_latency": round(turn_taking_latency, 3)
                    if (turn_taking_latency := turn_taking_latency_secs(breakdown)) is not None
                    else None,
                    "events": events,
                }
            )
        )
        if events:
            logger.info(f"Latency breakdown: {' | '.join(events)}")

    # The frontend's one build, between the RTVI transport's edges; the RTVI client gets Pipecat's own transcript. The
    # voice shows each delegation as a task card in the UI, the display part of its result included.
    frontend = build_frontend(
        FrontendSpec(
            config=cascade,
            talker=talker.from_config(frontend_endpoint, cascade.reliability, response_format),
            prompts=prompts.cascade,
            lines=prompts.lines,
            seed=messages,
            stt=stt,
            tts=tts,
            user_params=turn_end.user_params(welcome_enabled),
            delegation=delegation,
            item_text=message_text,
            display=lambda worker: RtviTaskCards(worker.queue_frame),
        ),
        FrontendEdges(
            inputs=[transport.input()],
            outputs=[transport.output()],
            worker={
                "params": build_pipeline_params(enable_metrics=True, enable_usage_metrics=True),
                "idle_timeout_secs": runner_args.pipeline_idle_timeout_secs,
                "observers": with_realtime_observers(latency_observer, transport=transport),
                "enable_tracing": IS_TRACING_ENABLED,
            },
            after=[audio_recorder] if audio_recorder else [],
        ),
    )
    task, voice = frontend.worker, frontend.voice

    voice_open = True

    async def close_voice() -> None:
        """Close the voice once (its tasks and timers); a failed close is logged."""
        nonlocal voice_open
        if not voice_open:
            return
        voice_open = False
        try:
            await voice.close()
        except Exception:
            logger.exception("The voice could not close")

    @task.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(task, frame):
        # Nothing is worded or said once the pipeline has ended: a result that arrives later stays unsaid.
        await close_voice()

    @frontend.user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        await send_rtvi_payloads(task, finalized_turn_payloads(None, message))

    async def _on_session_start() -> None:
        if audio_recorder:
            await audio_recorder.start_recording()
        if welcome_enabled:
            voice.greet(prompts.lines.intro)  # said before the caller speaks, as the assistant's own line

    # The shared start handler runs with its welcome off: the greeting is the voice's line, never a caller turn.
    # The caller's audio still waits for it when the welcome is on (the aggregator follows the same switch), and so
    # does a Realtime client's text, whose gate that handler opens.
    register_session_start_handlers(
        transport=transport,
        task=task,
        context=frontend.context,
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
        voice.start()  # the quiet task that offers progress lines
        await runner.add_workers(task)
        await runner.run()
    finally:
        # Close once each, in a live session's order: the voice, the server delegation (which deletes the IM session
        # and its backend adapter), then the core. A failed or cancelled close still runs the later ones.
        try:
            await close_voice()
        finally:
            try:
                await delegation.close()
            finally:
                await core.aclose()
