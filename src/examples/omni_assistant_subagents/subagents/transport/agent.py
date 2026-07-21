# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Transport owner subagent."""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.bus import BusBridgeProcessor
from pipecat.bus.bus import WorkerBus
from pipecat.bus.messages import BusCancelMessage, BusJobResponseMessage
from pipecat.frames.frames import (
    ClientConnectedFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMRunFrame,
    LLMTextFrame,
    SpeechControlParamsFrame,
    TTSUpdateSettingsFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frameworks.rtvi.frames import RTVIServerMessageFrame
from pipecat.runner.types import RunnerArguments
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.turns.user_mute.mute_until_first_bot_complete_user_mute_strategy import (
    MuteUntilFirstBotCompleteUserMuteStrategy,
)

from attachment_store import (
    clear_session_attachments,
    create_capture_request,
    latest_attachment,
    register_attachment_listener,
)
from examples.omni_assistant.pipeline import _build_user_turn_processor
from examples.omni_assistant.user_mute_processor import UserMuteProcessor
from examples.omni_assistant_subagents.media_dispatch_processor import PostAckMediaDispatchProcessor
from examples.omni_assistant_subagents.subagents.thinker import ThinkerWorker
from examples.omni_assistant_subagents.subagents.transport.media_analysis_controller import MediaAnalysisController
from examples.omni_assistant_subagents.subagents.transport.proactive_gesture_controller import (
    ProactiveGestureController,
)
from examples.omni_assistant_subagents.subagents.transport.speaker_context import (
    SpeakerContextManager,
)
from examples.omni_assistant_subagents.subagents.transport.subagent_state_board import SubagentStateBoard
from examples.omni_assistant_subagents.subagents.transport.thinking_controller import ThinkingController
from examples.omni_assistant_subagents.subagents.transport.webcam_controller import (
    WebcamController,
)
from examples.omni_assistant_subagents.subagents.webcam import WebcamAgent
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.subagents import SubagentRegistry
from tracing import IS_TRACING_ENABLED
from utils import load_ipa_dictionary, normalize_lang_code, parse_env_float
from webcam_frame_store import clear_session_webcam_frames

_ANALYZER_FOLLOWUP_TURN_DELAY_SECS = 2.6


class OmniTransportAgent(PipelineWorker):
    """Owns transport I/O and bridges user frames to Speaker Omni.

    A ``PipelineWorker`` whose pipeline carries a mid-pipeline
    ``BusBridgeProcessor`` in the LLM slot, teeing user frames onto the
    shared bus for ``SpeakerOmniAgent`` and injecting the speaker's frames
    back into the local pipeline. It also acts as the job requester
    (``request_job`` / ``on_job_response``) for the media-analyzer and
    webcam workers.
    """

    AGENT_NAME = "omni_transport"

    def __init__(
        self,
        name: str | None = None,
        *,
        bus: WorkerBus,
        transport,
        context: LLMContext,
        api_key: str,
        tts_server: str,
        tts_ssl: bool,
        tts_voice: str,
        tts_synthesis_mode: str,
        runner_args: RunnerArguments,
        session_id: str,
        subagent_registry: SubagentRegistry,
        proactive_directives: dict[str, str] | None = None,
    ) -> None:
        """Initialize the transport owner and build its bridged pipeline.

        ``bus`` is the runner's ``WorkerBus``, used only to construct the
        mid-pipeline ``BusBridgeProcessor``; the worker itself receives its
        bus from the runner via ``add_workers()``.
        """
        resolved_name = name or self.AGENT_NAME
        self._transport = transport
        self._context = context
        self._tts_server = tts_server
        self._tts_ssl = tts_ssl
        self._tts_voice = tts_voice
        self._tts_synthesis_mode = tts_synthesis_mode
        self._runner_args = runner_args
        self._session_id = session_id
        self._latency_turn_count = 1
        self._proactive_directives = proactive_directives or {}
        self._unregister_attachment_listener = None
        self._pending_capture_query = ""
        self._awaiting_capture = False
        self._capture_task: asyncio.Task[None] | None = None
        self._assistant_speaking = False
        self._user_speaking = False

        tts_settings_kwargs: dict[str, Any] = {"voice": tts_voice}
        if tts_synthesis_mode:
            tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
        self._tts = NvidiaTTSService(
            api_key=api_key,
            server=tts_server,
            settings=NvidiaTTSSettings(**tts_settings_kwargs),
            use_ssl=tts_ssl,
            text_filters=[NemotronSpeechTextFilter()],
            custom_dictionary=load_ipa_dictionary(),
            stop_frame_timeout_s=parse_env_float("TTS_STOP_FRAME_TIMEOUT_S", 30.0, min_value=5.0),
        )
        logger.info(
            f"Nemotron Omni subagents TTS: server={tts_server}, ssl={tts_ssl}, "
            f"voice={tts_voice}, synthesis_mode={tts_synthesis_mode or '(pipecat default)'}"
        )

        self._speaker_context = SpeakerContextManager(context=self._context)
        self._subagent_board = SubagentStateBoard(
            registry=subagent_registry,
            speaker_context=self._speaker_context,
        )
        self._media_analysis = MediaAnalysisController(
            session_id=self._session_id,
            speaker_context=self._speaker_context,
            board=self._subagent_board,
            request_job=self.request_job,
            queue_frame=self.queue_frame,
            followup_delay_secs=_ANALYZER_FOLLOWUP_TURN_DELAY_SECS,
        )
        self._thinking = ThinkingController(
            context=self._context,
            request_job=self.request_job,
            queue_frame=self.queue_frame,
            followup_delay_secs=_ANALYZER_FOLLOWUP_TURN_DELAY_SECS,
        )
        self._webcam_controller = WebcamController(
            session_id=self._session_id,
            board=self._subagent_board,
            request_job=self.request_job,
            queue_frame=self.queue_frame,
            conversation_provider=self._recent_conversation,
        )
        self._gestures = ProactiveGestureController(
            queue_frame=self.queue_frame,
            greet=self.proactive_greet,
            barge_in=self.barge_in,
            resume_or_compliment=self.proactive_continue,
            acknowledge_feedback=self.proactive_acknowledge_feedback,
            is_assistant_speaking=lambda: self._assistant_speaking,
            is_user_speaking=lambda: self._user_speaking,
        )
        pipeline = self._build_pipeline(bus=bus, worker_name=resolved_name)
        super().__init__(
            pipeline,
            name=resolved_name,
            active=True,
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            idle_timeout_secs=self._runner_args.pipeline_idle_timeout_secs,
            observers=[self._build_latency_observer()],
            enable_tracing=IS_TRACING_ENABLED,
            enable_rtvi=True,
        )
        self._register_client_handlers()

    def _build_pipeline(self, *, bus: WorkerBus, worker_name: str) -> Pipeline:
        """Build the transport pipeline with a bus bridge in the LLM slot."""
        assistant_aggregator = LLMAssistantAggregator(self._context)
        return Pipeline(
            [
                self._transport.input(),
                UserMuteProcessor(strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()]),
                VADProcessor(vad_analyzer=SileroVADAnalyzer(params=VADParams())),
                _build_user_turn_processor(),
                BusBridgeProcessor(
                    bus=bus,
                    worker_name=worker_name,
                    exclude_frames=(
                        ClientConnectedFrame,
                        LLMFullResponseStartFrame,
                        LLMTextFrame,
                        LLMFullResponseEndFrame,
                        RTVIServerMessageFrame,
                        SpeechControlParamsFrame,
                        UserMuteStartedFrame,
                        UserMuteStoppedFrame,
                    ),
                ),
                PostAckMediaDispatchProcessor(handler=self),
                self._tts,
                self._transport.output(),
                assistant_aggregator,
            ]
        )

    def _build_latency_observer(self) -> UserBotLatencyObserver:
        """Build the latency observer that emits per-turn metric groups over RTVI."""
        latency_observer = UserBotLatencyObserver()
        latest_latency_turn_id = ""
        latest_latency_turn_label = ""
        latest_latency_ms: float | None = None

        @latency_observer.event_handler("on_latency_measured")
        async def on_latency(observer, latency):
            nonlocal latest_latency_ms, latest_latency_turn_id, latest_latency_turn_label
            latest_latency_turn_id = f"turn-{self._latency_turn_count}"
            latest_latency_turn_label = f"Turn {self._latency_turn_count}"
            latest_latency_ms = round(latency * 1000, 3)
            logger.info(f"Nemotron Omni subagents User->Bot latency: {latency:.3f}s")

        @latency_observer.event_handler("on_latency_breakdown")
        async def on_latency_breakdown(observer, breakdown):
            nonlocal latest_latency_ms, latest_latency_turn_id, latest_latency_turn_label
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
            await self.queue_frame(
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
            if events:
                logger.info(f"Nemotron Omni subagents latency breakdown: {' | '.join(events)}")
            self._latency_turn_count += 1
            latest_latency_ms = None
            latest_latency_turn_id = ""
            latest_latency_turn_label = ""

        return latency_observer

    def _register_client_handlers(self) -> None:
        """Register RTVI client and transport event handlers on this worker."""

        @self.rtvi.event_handler("on_client_ready")
        async def on_client_connected(rtvi):
            logger.info("Nemotron Omni subagents client connected")
            self._start_attachment_state_listener()
            intro_prompt = "Please introduce yourself to the user."
            self._context.add_message({"role": "user", "content": intro_prompt})
            self._webcam_controller.start_summary_loop()
            await self.queue_frame(LLMRunFrame())

        @self._transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info("Nemotron Omni subagents client disconnected")
            self._webcam_controller.stop_summary_loop()
            self._stop_attachment_state_listener()
            clear_session_attachments(self._session_id)
            clear_session_webcam_frames(self._session_id)
            await self.send_bus_message(BusCancelMessage(source=self.name, reason="client disconnected"))

        @self.rtvi.event_handler("on_client_message")
        async def on_client_message(rtvi, message):
            payload = message.data if isinstance(message.data, dict) else {}
            if message.type == "set-voice":
                await self._apply_set_voice(payload)
            elif message.type == "webcam-state":
                await self._webcam_controller.apply_webcam_state(payload)
            elif message.type == "webcam-chunk":
                await self._webcam_controller.set_window_seconds(payload)

    async def queue_media_analysis_prompt(
        self,
        transcript: str,
        prompt: str,
        action: str,
        selected_input_source: str,
    ) -> None:
        """Queue Speaker Omni's hidden analyzer prompt after its acknowledgement closes."""
        await self._media_analysis.queue_prompt(transcript, prompt, action, selected_input_source)

    def has_uploaded_attachment(self) -> bool:
        """Return whether this session currently has an uploaded attachment."""
        return self._media_analysis.has_uploaded_attachment()

    def is_attachment_pending(self) -> bool:
        """Whether a freshly uploaded attachment is waiting to be analyzed (webcam-first gate)."""
        return self._media_analysis.is_attachment_pending()

    async def start_pending_media_analysis(self) -> None:
        """Dispatch the LLM-selected analyzer after the ack turn has completed."""
        await self._media_analysis.start_pending()

    async def queue_thinking(self, transcript: str, effort: str, reason: str) -> None:
        """Queue a reasoning-ON pass when the Speaker stalls; runs after its turn.

        ``reason`` is empty when the model spoke its own stall, or names the trigger
        (e.g. ``repetition``) so the Thinker can recover.
        """
        self._thinking.queue(transcript, effort=effort, reason=reason)

    async def start_pending_thinking(self) -> None:
        """Dispatch the queued reasoning pass after the Speaker's stall turn completes."""
        await self._thinking.start_pending()

    async def queue_highres_capture(self, query: str) -> None:
        """Ask the browser for a native-res snapshot the moment the Speaker requests one.

        Dispatched immediately, NOT gated on a spoken ack: the Speaker sometimes returns an
        empty ``response`` for the capture turn, and a speech-gated dispatch would then never
        fire (no ``BotStoppedSpeaking``), stranding the capture and freezing the flow. The
        browser snapshot is independent of TTS, so it is safe to request right away — any ack
        the model did produce plays concurrently, and the analyzer answer follows when the
        snapshot lands.
        """
        self._pending_capture_query = query.strip()
        if not self._pending_capture_query:
            return
        self._awaiting_capture = True
        request_id = create_capture_request(self._session_id)
        await self.queue_frame(
            RTVIServerMessageFrame(data={"type": "webcam-capture-request", "request_id": request_id})
        )
        logger.info(
            f"Requested a high-res webcam snapshot from the browser: query_chars={len(self._pending_capture_query)}"
        )

    def current_visual_status(self) -> str:
        """The live webcam status right now, for per-turn injection into the Speaker turn.

        Read synchronously by the Speaker service while it assembles each turn so the
        freshest live view sits next to the user's audio (the most salient position),
        not only in the older top-pinned board note.
        """
        return self._webcam_controller.current_visual_status()

    def _recent_conversation(self, max_turns: int = 6, max_chars: int = 600) -> str:
        """Render the last few user/assistant turns as plain text for the webcam analyzer.

        Only conversational turns are included, never the pinned board, so the webcam worker
        learns the current topic without a feedback loop from its own past observations. Capped
        in turns and characters to keep the continuous sub-second webcam loop fast.
        """
        turns: list[str] = []
        for message in self._context.get_messages():
            if not isinstance(message, dict):
                continue
            role = str(message.get("role") or "")
            content = message.get("content")
            if role not in ("user", "assistant") or not isinstance(content, str):
                continue
            text = content.strip()
            if text:
                turns.append(f"{'User' if role == 'user' else 'Assistant'}: {text}")
        rendered = "\n".join(turns[-max_turns:]).strip()
        return rendered[-max_chars:] if len(rendered) > max_chars else rendered

    async def barge_in(self) -> None:
        """Interrupt assistant audio in response to a visual stop sign (no new line spoken).

        The InterruptionFrame, queued at the pipeline head, flows through the bus bridge
        (cancelling the Speaker) and through TTS/output (stopping playback), exactly as a
        user voice barge-in would.
        """
        logger.info("Visual barge-in: stopping assistant speech on the user's stop signal")
        await self.queue_frame(InterruptionFrame())

    async def proactive_greet(self) -> None:
        """Greet the user back after a wave/greeting gesture."""
        await self._run_proactive_turn("proactive_greet")

    async def proactive_continue(self, resume: bool) -> None:
        """Respond to a thumbs-up: resume after a recent barge-in, else accept a compliment."""
        await self._run_proactive_turn("proactive_continue_resume" if resume else "proactive_continue_compliment")

    async def proactive_acknowledge_feedback(self) -> None:
        """Respond to a thumbs-down: the user is not impressed — take it as feedback, no retry."""
        await self._run_proactive_turn("proactive_feedback")

    async def _run_proactive_turn(self, directive_key: str) -> None:
        """Inject a gesture directive (from prompts.yaml) and run a Speaker turn."""
        directive = self._proactive_directives.get(directive_key, "").strip()
        if not directive:
            logger.warning(f"No proactive directive configured for {directive_key!r}; skipping")
            return
        self._context.add_message({"role": "user", "content": directive})
        await self.queue_frame(LLMRunFrame())

    async def on_user_voice_turn_started(self) -> None:
        """Track whether the user is currently speaking."""
        self._user_speaking = True

    async def on_user_voice_turn_stopped(self) -> None:
        """Track when the user turn reaches EOU."""
        self._user_speaking = False

    async def on_user_interrupted_assistant(self) -> None:
        """Cancel pending post-ack media/thinking work when the ack was interrupted by speech.

        The high-res capture is intentionally NOT cleared here: it is dispatched the instant the
        Speaker requests it (not deferred to the ack), so by this point the snapshot is already
        in flight and clearing its query would only strand the landed image.
        """
        await self._media_analysis.clear_pending_on_interruption()
        self._thinking.clear_pending()

    async def on_assistant_speaking_started(self) -> None:
        """Track active assistant speech for gesture context (barge-in only while speaking)."""
        self._assistant_speaking = True

    async def on_assistant_speaking_stopped(self) -> None:
        """Track when assistant speech has stopped."""
        self._assistant_speaking = False

    async def on_job_response(self, message: BusJobResponseMessage) -> None:
        """Route analyzer job results back to Speaker Omni for the spoken answer."""
        await super().on_job_response(message)
        response = message.response or {}
        source = str(getattr(message, "source", "") or "")
        mode = str(response.get("mode") or "").strip()
        if source == WebcamAgent.AGENT_NAME:
            if mode == "summary":
                accepted = await self._webcam_controller.handle_summary_response(message.job_id, response)
                if accepted:
                    frame = response.get("frame") if isinstance(response.get("frame"), dict) else {}
                    await self._gestures.handle(response.get("visual_control") or {}, frame=frame)
            elif mode:
                logger.debug(f"Ignoring unsupported webcam task response mode: {mode}")
            return
        if source == ThinkerWorker.AGENT_NAME:
            await self._thinking.handle_job_response(message)
            return
        await self._media_analysis.handle_job_response(message)

    def _start_attachment_state_listener(self) -> None:
        """Start listening for stored media.

        Mark the board pending for a fresh user upload, or run a focused analysis
        when an agent-requested high-res capture lands.
        """
        if self._unregister_attachment_listener:
            return
        loop = asyncio.get_running_loop()
        self._unregister_attachment_listener = register_attachment_listener(
            self._session_id,
            lambda: loop.call_soon_threadsafe(self._on_attachment_stored),
        )

    def _on_attachment_stored(self) -> None:
        """Handle an attachment-store callback on the event loop.

        Route an agent-captured high-res snapshot to focused analysis; otherwise
        mark a user upload pending.
        """
        if self._awaiting_capture:
            latest = latest_attachment(self._session_id)
            if latest is not None and latest.source == "capture":
                self._awaiting_capture = False
                query = self._pending_capture_query
                self._pending_capture_query = ""
                self._capture_task = asyncio.create_task(self._media_analysis.analyze_capture(latest.metadata(), query))
                return
        self._media_analysis.mark_attachment_pending()

    def _stop_attachment_state_listener(self) -> None:
        """Stop listening for attachment uploads for this session."""
        if self._unregister_attachment_listener:
            self._unregister_attachment_listener()
        self._unregister_attachment_listener = None

    async def _apply_set_voice(self, payload: dict[str, Any]) -> None:
        voice_id = payload.get("voice_id", "")
        language = payload.get("language", "")
        if not voice_id or self._tts is None:
            return
        settings_kwargs: dict[str, Any] = {"voice": voice_id}
        if language:
            settings_kwargs["language"] = normalize_lang_code(language)
        await self.queue_frame(
            TTSUpdateSettingsFrame(
                delta=NvidiaTTSSettings(**settings_kwargs),
                service=self._tts,
            )
        )
        logger.info(
            f"Nemotron Omni subagents voice switched -> {voice_id}, "
            f"language={settings_kwargs.get('language', '(unchanged)')}"
        )
