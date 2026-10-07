# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend/Backend Verdict agent: the voice prototype served on the OpenAI Realtime API.

The Realtime gateway authenticates the socket and routes
``?model=nvidia/nemotron-realtime-frontend-backend-verdict`` here before it sends any
event (this pipeline owns the whole Realtime session). The pipeline is::

    FastAPIWebsocketTransport.input()      client messages, in arrival order
      -> PrototypeSessionProcessor         the prototype RealtimeSession (copied code):
                                           wire, VAD + segmenter, Riva ASR, turn manager,
                                           frontend verdict, text agent, normalization,
                                           Riva TTS
      -> FastAPIWebsocketTransport.output() server events, in the session's order

Behaviour comes from the selected prototype profile (``FBV_PROFILE``, default
``tau3_eval_frontend_verdict_speak_history``); the LLM endpoints come from the
route's ``llm`` (frontend) and ``thinker-llm`` (backend) catalog entries.
"""

from __future__ import annotations

from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from examples.frontend_backend_verdict.bridge.runtime import runtime_for
from examples.frontend_backend_verdict.bridge.session_processor import PrototypeSessionProcessor
from examples.frontend_backend_verdict.bridge.wire import PrototypeClientMessageFrame, PrototypeWireSerializer
from examples.frontend_backend_verdict.voice.server import SHOW_FILLER_PARAM
from examples.shared.pipeline_utils import build_pipeline_params


async def bot(runner_args: RunnerArguments) -> None:
    """Serve one OpenAI Realtime connection with the prototype session."""
    body = getattr(runner_args, "body", None) or {}
    websocket = getattr(runner_args, "websocket", None)
    if body.get("protocol") != "realtime" or websocket is None:
        raise RuntimeError(
            "frontend-backend-verdict-agent serves the OpenAI Realtime WebSocket only "
            "(model nvidia/nemotron-realtime-frontend-backend-verdict on /v1/realtime)"
        )
    query = websocket.query_params
    model = query.get("model", "")
    show_silent_filler = query.get(SHOW_FILLER_PARAM, "").lower() in ("1", "true")

    runtime = runtime_for(body)
    if not await runtime.admit(websocket):
        return
    try:
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=False,
                audio_out_enabled=False,
                add_wav_header=False,
                serializer=PrototypeWireSerializer(),
                session_timeout=None,
            ),
        )
        session = PrototypeSessionProcessor(runtime=runtime, model=model, show_silent_filler=show_silent_filler)
        worker = PipelineWorker(
            Pipeline([transport.input(), session, transport.output()]),
            params=build_pipeline_params(),
            # The prototype session has no idle cutoff; the gateway's 60-minute lifetime applies.
            idle_timeout_secs=None,
            idle_timeout_frames=(PrototypeClientMessageFrame,),
            enable_rtvi=False,
            enable_turn_tracking=False,
            # Same bound as the repository's other pipelines (Pipecat warms deferred imports during setup).
            setup_timeout_secs=120.0,
        )

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _websocket) -> None:
            await session.client_disconnected()
            await worker.cancel()

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await runner.run()
    finally:
        runtime.release()
        logger.info(f"frontend/backend verdict session finished (model={model or '-'})")
