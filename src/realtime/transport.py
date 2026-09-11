# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat WebSocket transport bound to one canonical Realtime controller."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any
from weakref import WeakKeyDictionary

from fastapi import WebSocketDisconnect
from loguru import logger
from openai import NOT_GIVEN
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    LLMConfigureOutputFrame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSUpdateSettingsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.nvidia.tts import NvidiaTTSSettings
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start.external_user_turn_start_strategy import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.asyncio.task_manager import BaseTaskManager

from realtime.audio import PIPELINE_OUTPUT_PCM_RATE, PIPELINE_PCM_RATE
from realtime.client_tools import ClientToolBroker
from realtime.controller import RealtimeSessionController
from realtime.events import error_event
from realtime.frames import (
    RealtimeASREndpointSilenceFrame,
    RealtimeClientToolOutputFrame,
    RealtimeConversationAppendFrame,
    RealtimeDeferredResponseCreateFrame,
    RealtimeIdleTimeoutFrame,
    RealtimeManualUserStartedSpeakingFrame,
    RealtimeManualUserStoppedSpeakingFrame,
    RealtimeResponseContextFrame,
    RealtimeResponseCreateFrame,
    RealtimeResponseLLMContext,
    RealtimeResponseOrigin,
)
from realtime.idle_timeout import RealtimeServerVADIdleTimeout
from realtime.mcp import MCPPreparedTools, RealtimeMCPRuntime
from realtime.observer import RealtimeLifecycleObserver
from realtime.protocol import RealtimeProtocolError, new_realtime_id
from realtime.serializer import RealtimeFrameSerializer
from realtime.tool_projection import project_function_tools
from utils import parse_env_float, parse_env_int


@dataclass(slots=True)
class _RealtimeTransportContext:
    controller: RealtimeSessionController
    serializer: RealtimeFrameSerializer
    client_tool_broker: ClientToolBroker
    fail_input_audio_turn: Callable[[], Awaitable[None]]
    wire_sequencer: _RealtimeWireSequencer
    response_gate: RealtimeManualResponseGate
    mcp_runtime: RealtimeMCPRuntime
    idle_timeout: RealtimeServerVADIdleTimeout
    manual_commit_publications: _RealtimeManualCommitPublications | None = None
    observer: RealtimeLifecycleObserver | None = None


_CONTEXTS: WeakKeyDictionary[Any, _RealtimeTransportContext] = WeakKeyDictionary()


_INPUT_TRANSCRIPTION_TERMINAL_EVENTS = frozenset(
    {
        "conversation.item.input_audio_transcription.completed",
        "conversation.item.input_audio_transcription.failed",
    }
)

_DEFAULT_MANUAL_ASR_ENDPOINT_SILENCE_MS = 1000
_MAX_MANUAL_ASR_ENDPOINT_SILENCE_MS = 5000


def _response_done_id(event: dict[str, Any]) -> str | None:
    """Return a valid response ID only for one terminal response event."""
    if event.get("type") != "response.done":
        return None
    response = event.get("response")
    response_id = response.get("id") if isinstance(response, dict) else None
    return response_id if isinstance(response_id, str) and response_id else None


ClientToolOutputHandler = Callable[[RealtimeClientToolOutputFrame], Awaitable[None]]
ConversationAppendHandler = Callable[[str, dict[str, Any], list[dict[str, Any]]], Awaitable[None]]
IdleTimeoutHandler = Callable[[RealtimeIdleTimeoutFrame], Awaitable[str | None]]
DeferredResponseHandler = Callable[[RealtimeDeferredResponseCreateFrame, int], Awaitable[str | None]]
PipelineResponseOutputHandler = Callable[[dict[str, Any] | None], None]
PipelineResponseOutputResetHandler = Callable[[], None]
PipelineResponseStartedHandler = Callable[[str], Awaitable[None]]
PipelineResponseStartHandler = Callable[..., Awaitable[str | None]]
ServiceResponseSnapshot = tuple[LLMContext, tuple[Frame, ...], Callable[[], None]]
FusedServiceResponseSnapshot = tuple[
    LLMContext,
    tuple[Frame, ...],
    Callable[[], Awaitable[str | None]],
    Callable[[], None],
]
RealtimeInstructionsRenderer = Callable[[str], list[dict[str, Any]]]


class _RealtimeManualCommitPublications:
    """Acknowledge manual commits only after their event reaches the wire.

    Pipecat observers run through an asynchronous proxy queue, so awaiting the
    input transport's ``push_frame`` call is not a publication boundary. The
    WebSocket receive loop waits on these FIFO acknowledgements before it may
    deserialize a following ``response.create``.
    """

    def __init__(self) -> None:
        self._waiters: deque[asyncio.Future[None]] = deque()

    def expect(self) -> asyncio.Future[None]:
        """Register the next manual commit before its frames enter Pipecat."""
        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append(waiter)
        return waiter

    def published(self) -> None:
        """Release the oldest commit after its committed event was sent."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if waiter.done():
                continue
            waiter.set_result(None)
            return
        logger.warning("Realtime manual commit was published without a registered waiter")

    def discard(self, waiter: asyncio.Future[None]) -> None:
        """Forget a handoff that failed before its commit was published."""
        with suppress(ValueError):
            self._waiters.remove(waiter)
        if not waiter.done():
            waiter.cancel()

    def reset(self) -> None:
        """Cancel every waiter when its WebSocket session terminates."""
        while self._waiters:
            waiter = self._waiters.popleft()
            if not waiter.done():
                waiter.cancel()


class RealtimeManualResponseGate(FrameProcessor):
    """Coordinate automatic/explicit responses and isolate inference state.

    Pipecat's user aggregator normally emits an ``LLMContextFrame`` as soon as
    an externally bounded audio turn has a final transcript. In Realtime manual
    mode, or when automatic responses are disabled for server-managed turns,
    commit must create/transcribe the user item without starting model inference.
    This gate drops those context frames until a ``response.create`` marker
    arrives, then releases one context only after all earlier committed turns
    have reached the shared context. In every input mode it applies response-local
    controls to an isolated context snapshot, so concurrent wire publication
    cannot change another response's settings.
    """

    def __init__(self, *, controller: RealtimeSessionController) -> None:
        """Create an empty per-connection response gate."""
        super().__init__()
        self._controller = controller
        self._installed = False
        self._llm_context: LLMContext | None = None
        self._pending_commits = 0
        self._unclaimed_manual_commits = 0
        self._pending_automatic_response_contexts = 0
        self._waiting_response_id: str | None = None
        self._waiting_tool_choice: str | dict[str, Any] | None = None
        self._waiting_tools: list[dict[str, Any]] | None = None
        self._waiting_instructions: str | None = None
        self._waiting_max_output_tokens: int | str | None = None
        self._waiting_parallel_tool_calls: bool | None = None
        self._waiting_truncation: str | dict[str, Any] | None = None
        self._waiting_input_messages: list[dict[str, Any]] | None = None
        self._waiting_output_modalities: list[str] | None = None
        self._waiting_audio_output: dict[str, Any] | None = None
        self._running_response_id: str | None = None
        self._pending_response_marker_id: str | None = None
        self._response_preparation_id: str | None = None
        self._session_tool_preparation_id: str | None = None
        self._preparation_lock = asyncio.Lock()
        self._service_audio_response_reservations: dict[str, int] = {}
        self._session_tool_waiting_context: tuple[LLMContextFrame, RealtimeResponseOrigin] | None = None
        self._cancelled_response_marker_ids: OrderedDict[str, None] = OrderedDict()
        self._pending_deferred_response_id: str | None = None
        self._pending_deferred_response_generation: int | None = None
        self._pending_deferred_audio_response = False
        self._cancelled_deferred_response_ids: OrderedDict[str, None] = OrderedDict()
        self._client_tool_output_handler: ClientToolOutputHandler | None = None
        self._conversation_append_handler: ConversationAppendHandler | None = None
        self._idle_timeout_handler: IdleTimeoutHandler | None = None
        self._deferred_response_handler: DeferredResponseHandler | None = None
        self._pipeline_response_output_handler: PipelineResponseOutputHandler | None = None
        self._pipeline_response_output_reset_handler: PipelineResponseOutputResetHandler | None = None
        self._pipeline_response_start_handler: PipelineResponseStartHandler | None = None
        self._pending_pipeline_response_abort: Callable[[], None] | None = None
        self._pending_pipeline_response_generation: int | None = None
        self._response_slot_changed = asyncio.Event()
        self._closed = False
        self._instructions_renderer: RealtimeInstructionsRenderer | None = None
        self._canonical_prompt_messages: list[dict[str, Any]] | None = None
        self._tts_service: Any | None = None

    @property
    def installed(self) -> bool:
        """Return whether this gate was inserted on the pre-LLM path."""
        return self._installed

    def install(self) -> None:
        """Record that pipeline construction inserted this processor."""
        self._installed = True

    def bind_context(
        self,
        context: LLMContext,
        *,
        instructions_renderer: RealtimeInstructionsRenderer | None = None,
        session_instructions: str | None = None,
    ) -> None:
        """Bind the canonical context and its explicit public-prompt renderer."""
        self._llm_context = context
        if instructions_renderer is None:
            return
        if session_instructions is None:
            raise TypeError("Realtime instruction rendering requires the canonical session instructions")
        rendered = self._render_instructions(instructions_renderer, session_instructions)
        messages = list(context.get_messages())
        if messages[: len(rendered)] != rendered:
            raise ValueError("Realtime instruction renderer does not reproduce the canonical prompt prefix")
        self._instructions_renderer = instructions_renderer
        self._canonical_prompt_messages = copy.deepcopy(rendered)

    @staticmethod
    def _render_instructions(
        renderer: RealtimeInstructionsRenderer,
        instructions: str,
    ) -> list[dict[str, Any]]:
        rendered = renderer(instructions)
        if not isinstance(rendered, list) or any(not isinstance(message, dict) for message in rendered):
            raise TypeError("Realtime instruction renderer must return a list of context messages")
        return copy.deepcopy(rendered)

    def _require_instruction_binding(self, *, param: str) -> tuple[RealtimeInstructionsRenderer, LLMContext]:
        if self._instructions_renderer is None or self._canonical_prompt_messages is None or self._llm_context is None:
            raise RealtimeProtocolError(
                message="Realtime instructions are not bound to this pipeline's model context",
                code="instructions_runtime_missing",
                param=param,
                error_type="server_error",
            )
        return self._instructions_renderer, self._llm_context

    def _canonical_tail(self, context: LLMContext) -> list[Any]:
        if self._canonical_prompt_messages is None:
            raise RuntimeError("Realtime instructions do not own a canonical prompt prefix")
        messages = list(context.get_messages())
        prompt_count = len(self._canonical_prompt_messages)
        if messages[:prompt_count] != self._canonical_prompt_messages:
            raise RuntimeError("The canonical Realtime prompt prefix changed outside its instruction renderer")
        return messages[prompt_count:]

    def prepare_session_instructions(self, instructions: str) -> list[dict[str, Any]]:
        """Render a validated live update without changing canonical model state."""
        renderer, context = self._require_instruction_binding(param="session.instructions")
        self._canonical_tail(context)
        return self._render_instructions(renderer, instructions)

    def validate_response_instructions(self, instructions: str) -> None:
        """Prove a response overlay can be rendered before its lifecycle starts."""
        renderer, context = self._require_instruction_binding(param="response.instructions")
        self._canonical_tail(context)
        self._render_instructions(renderer, instructions)

    def validate_response_input_context(self) -> None:
        """Prove custom response input can retain the trusted prompt prefix."""
        _, context = self._require_instruction_binding(param="response.input")
        self._canonical_tail(context)

    def bind_tts_service(self, tts: Any) -> None:
        """Bind the exact TTS processor linked after this response gate."""
        if tts is None:
            raise TypeError("Realtime response TTS binding cannot be null")
        if self._tts_service is not None and self._tts_service is not tts:
            raise RuntimeError("Realtime response gate already has a different TTS service")
        self._tts_service = tts

    def validate_response_output(
        self,
        *,
        output_modalities: list[str],
        audio_output: dict[str, Any] | None,
        param_prefix: str = "response",
    ) -> None:
        """Validate that the effective output snapshot has an executable path."""
        if output_modalities == ["text"]:
            return
        if self._tts_service is None:
            raise RealtimeProtocolError(
                message="Audio responses require a linked TTS service",
                code="tts_runtime_missing",
                param=f"{param_prefix}.output_modalities",
                error_type="server_error",
            )
        voice = audio_output.get("voice") if isinstance(audio_output, dict) else None
        if not isinstance(voice, str) or not voice:
            raise RealtimeProtocolError(
                message="The linked NVIDIA TTS service requires a voice ID string",
                code="unsupported_capability",
                param=f"{param_prefix}.audio.output.voice",
            )

    def commit_session_instructions(
        self,
        rendered: list[dict[str, Any]],
    ) -> None:
        """Replace only the renderer-owned prompt prefix in canonical context."""
        _, context = self._require_instruction_binding(param="session.instructions")
        tail = self._canonical_tail(context)
        prompt = copy.deepcopy(rendered)
        context.set_messages([*prompt, *tail])
        self._canonical_prompt_messages = prompt

    def snapshot_session_prompt(self) -> list[dict[str, Any]]:
        """Capture the renderer-owned prompt prefix for transaction rollback."""
        self._require_instruction_binding(param="session.instructions")
        if self._canonical_prompt_messages is None:
            raise RuntimeError("Realtime instructions do not own a canonical prompt prefix")
        return copy.deepcopy(self._canonical_prompt_messages)

    def restore_session_prompt(self, prompt: list[dict[str, Any]]) -> None:
        """Restore a prompt prefix after its surrounding session transaction fails."""
        self._require_instruction_binding(param="session.instructions")
        if not isinstance(prompt, list) or any(not isinstance(message, dict) for message in prompt):
            raise TypeError("Realtime session prompt rollback requires context messages")
        self._canonical_prompt_messages = copy.deepcopy(prompt)

    def set_client_tool_output_handler(self, handler: ClientToolOutputHandler) -> None:
        """Bind the transport-owned client-output state transition."""
        self._client_tool_output_handler = handler

    def set_conversation_append_handler(self, handler: ConversationAppendHandler) -> None:
        """Bind publication after a client item enters canonical model context."""
        self._conversation_append_handler = handler

    def set_idle_timeout_handler(self, handler: IdleTimeoutHandler) -> None:
        """Bind the server-VAD idle transition at the canonical context owner."""
        self._idle_timeout_handler = handler

    def set_deferred_response_handler(self, handler: DeferredResponseHandler) -> None:
        """Bind the transport-owned Response B activation transition."""
        self._deferred_response_handler = handler

    def set_pipeline_response_output_handler(
        self,
        handler: PipelineResponseOutputHandler,
        reset_handler: PipelineResponseOutputResetHandler | None = None,
    ) -> None:
        """Bind actual wire audio encoding for unowned pipeline responses."""
        self._pipeline_response_output_handler = handler
        self._pipeline_response_output_reset_handler = reset_handler

    def set_pipeline_response_start_handler(self, handler: PipelineResponseStartHandler) -> None:
        """Bind atomic activation/publication for fused service responses."""
        self._pipeline_response_start_handler = handler

    def validate_activation_generation(self, generation: int) -> None:
        """Reject a delayed activation if an interruption already crossed this gate."""
        if generation != self._controller.interruption_generation:
            raise RealtimeProtocolError(
                message="The pending response was cancelled before it could start",
                code="response_cancelled",
                param="response",
            )

    @property
    def pending_commits(self) -> int:
        """Return committed manual turns not yet reflected in LLM context."""
        return self._pending_commits

    @property
    def has_unclaimed_manual_commit(self) -> bool:
        """Return whether a committed manual turn has no accepted response yet."""
        return self._unclaimed_manual_commits > 0

    @property
    def response_slot_occupied(self) -> bool:
        """Return whether an earlier response still owns the ordered pipeline slot."""
        return bool(
            self._service_audio_response_reservations
            or self._pending_deferred_response_id is not None
            or self._pending_response_marker_id is not None
            or self._waiting_response_id is not None
            or self._running_response_id is not None
            or self._pending_pipeline_response_abort is not None
            or self._controller.response_in_progress
            or self._controller.pipeline_response_pending
        )

    @property
    def audio_response_slot_occupied(self) -> bool:
        """Return whether an audio response owns or has frozen the output slot."""
        if self._controller.audio_response_in_progress_or_pending:
            return True
        if self._pending_deferred_audio_response:
            return True
        session_audio = self._controller.session.public_view().get("output_modalities") == ["audio"]
        return bool(self._service_audio_response_reservations and session_audio)

    def register_commit(self) -> None:
        """Expect one context update from a validated manual audio commit."""
        self._pending_commits += 1
        self._unclaimed_manual_commits += 1

    def register_automatic_response_context(self) -> None:
        """Mark the exact context a server-managed user-turn strategy will emit."""
        self._pending_automatic_response_contexts += 1

    def abort_commit(self) -> None:
        """Undo a registration when the audio handoff failed before completion."""
        if self._pending_commits <= 0:
            raise RuntimeError("No manual Realtime commit is registered")
        self._pending_commits -= 1
        if self._unclaimed_manual_commits <= 0:
            raise RuntimeError("No unclaimed manual Realtime commit is registered")
        self._unclaimed_manual_commits -= 1

    def _claim_manual_commits(self) -> None:
        """Bind every preceding unclaimed manual turn to one accepted response."""
        self._unclaimed_manual_commits = 0

    async def reserve_response_preparation(self, request_id: str, *, deferred: bool = False) -> None:
        """Reserve the response slot while response-scoped tools are prepared."""
        await self._preparation_lock.acquire()
        try:
            if self._service_audio_response_reservations:
                raise RealtimeProtocolError(
                    message="A pipeline-triggered audio response is already starting",
                    code="response_in_progress",
                    param="response",
                )
            if deferred:
                if self._session_tool_preparation_id is not None:
                    raise RuntimeError("A Realtime session tool update is being prepared")
                if self._response_preparation_id is not None:
                    raise RuntimeError("A Realtime response is being prepared")
                if self._pending_deferred_response_id is not None:
                    raise RealtimeProtocolError(
                        message="A response.create request is already waiting for ordered pipeline input",
                        code="response_in_progress",
                        param="response",
                    )
                if self._pending_response_marker_id is not None:
                    raise RuntimeError("A Realtime response marker is already pending")
            else:
                self.ensure_response_marker_available()
            self._response_preparation_id = request_id
        except BaseException:
            self._preparation_lock.release()
            raise

    def release_response_preparation(self, request_id: str) -> None:
        """Release an uncommitted preparation reservation after rejection."""
        if self._response_preparation_id == request_id:
            self._response_preparation_id = None
            self._preparation_lock.release()

    async def reserve_session_tool_preparation(self, request_id: str) -> None:
        """Prevent a new inference run while session tools are being projected."""
        await self._preparation_lock.acquire()
        try:
            if self._session_tool_preparation_id is not None:
                raise RuntimeError("A Realtime session tool update is already being prepared")
            if self._response_preparation_id is not None:
                raise RuntimeError("A Realtime response is being prepared")
            self._session_tool_preparation_id = request_id
        except BaseException:
            self._preparation_lock.release()
            raise

    async def release_session_tool_preparation(self, request_id: str) -> None:
        """Release the exact reservation and resume one deferred automatic run."""
        if self._session_tool_preparation_id != request_id:
            return
        self._session_tool_preparation_id = None
        waiting = self._session_tool_waiting_context
        self._session_tool_waiting_context = None
        frozen: ServiceResponseSnapshot | None = None
        try:
            if (
                waiting is not None
                and (
                    waiting[1] is not RealtimeResponseOrigin.AUTOMATIC_USER_TURN
                    or self._controller.automatic_response_enabled
                )
                and not self._service_audio_response_reservations
            ):
                waiting_context, origin = waiting
                frozen = self._freeze_pipeline_response(
                    waiting_context.context,
                    require_output_runtime=False,
                    origin=origin,
                )
        finally:
            self._preparation_lock.release()
        if frozen is not None:
            prepared = await self._publish_pipeline_response_snapshot(frozen)
            await self.push_frame(prepared, FrameDirection.DOWNSTREAM)

    def reserve_service_audio_response(self) -> str:
        """Claim an accepted fused-audio turn before its service task is scheduled."""
        reservation_id = new_realtime_id("run")
        self._service_audio_response_reservations[reservation_id] = self._controller.interruption_generation
        return reservation_id

    def release_service_audio_response(self, reservation_id: str) -> None:
        """Release a fused-audio claim that never reached snapshot preparation."""
        if self._service_audio_response_reservations.pop(reservation_id, None) is not None:
            self._response_slot_changed.set()

    def validate_response_preparation(self, request_id: str) -> None:
        """Recheck async response preparation immediately before its atomic commit."""
        if self._response_preparation_id != request_id:
            raise RuntimeError("Realtime response preparation ownership changed")
        if self._service_audio_response_reservations:
            raise RealtimeProtocolError(
                message="A pipeline-triggered audio response is already starting",
                code="response_in_progress",
                param="response",
            )

    def _commit_response_preparation(self, request_id: str | None) -> None:
        if request_id is None:
            if self._response_preparation_id is not None:
                raise RuntimeError("A Realtime response preparation is already pending")
            return
        if self._response_preparation_id != request_id:
            raise RuntimeError("Realtime response preparation ownership changed")
        self.validate_response_preparation(request_id)

    def register_response_marker(
        self,
        response_id: str,
        *,
        preparation_id: str | None = None,
        claim_manual_commits: bool = False,
    ) -> None:
        """Record the single response marker entering the input transport."""
        self._commit_response_preparation(preparation_id)
        if self._pending_response_marker_id is not None:
            raise RuntimeError("A Realtime response marker is already pending")
        self._pending_response_marker_id = response_id
        if claim_manual_commits:
            self._claim_manual_commits()
        if preparation_id is not None:
            self._response_preparation_id = None
            self._preparation_lock.release()

    def register_deferred_response(
        self,
        request_id: str,
        *,
        activation_generation: int,
        preparation_id: str | None = None,
        claim_manual_commits: bool = False,
        audio_response: bool = False,
    ) -> None:
        """Reserve one Response B request behind its preceding lifecycle work."""
        self._commit_response_preparation(preparation_id)
        if self._pending_deferred_response_id is not None:
            raise RealtimeProtocolError(
                message="A response.create request is already waiting for ordered pipeline input",
                code="response_in_progress",
                param="response",
            )
        if self._pending_response_marker_id is not None:
            raise RealtimeProtocolError(
                message="A response.create request is already entering the pipeline",
                code="response_in_progress",
                param="response",
            )
        self._pending_deferred_response_id = request_id
        self._pending_deferred_response_generation = activation_generation
        self._pending_deferred_audio_response = audio_response
        if claim_manual_commits:
            self._claim_manual_commits()
        if preparation_id is not None:
            self._response_preparation_id = None
            self._preparation_lock.release()

    async def wait_for_deferred_response_slot(
        self,
        request_id: str,
        *,
        activation_generation: int,
    ) -> bool:
        """Wait until the preceding response has released its published wire slot."""
        while True:
            if self._closed:
                return False

            # Clear before inspecting state so response.done publication cannot
            # be lost between the busy check and this task beginning its wait.
            self._response_slot_changed.clear()
            await self._preparation_lock.acquire()
            try:
                if self._closed or self._pending_deferred_response_id != request_id:
                    return False
                self.validate_activation_generation(activation_generation)
                busy = bool(
                    self._service_audio_response_reservations
                    or self._pending_response_marker_id is not None
                    or self._waiting_response_id is not None
                    or self._running_response_id is not None
                    or self._pending_pipeline_response_abort is not None
                    or self._controller.response_in_progress
                    or self._controller.pipeline_response_pending
                )
                if not busy:
                    return True
            finally:
                self._preparation_lock.release()
            await self._response_slot_changed.wait()

    def ensure_response_marker_available(self) -> None:
        """Validate marker ownership before mutating the response controller."""
        if self._service_audio_response_reservations:
            raise RealtimeProtocolError(
                message="A pipeline-triggered audio response is already starting",
                code="response_in_progress",
                param="response",
            )
        if self._session_tool_preparation_id is not None:
            raise RuntimeError("A Realtime session tool update is being prepared")
        if self._response_preparation_id is not None:
            raise RuntimeError("A Realtime response is being prepared")
        if self._pending_deferred_response_id is not None:
            raise RealtimeProtocolError(
                message="A response.create request is already waiting for ordered pipeline input",
                code="response_in_progress",
                param="response",
            )
        if self._pending_response_marker_id is not None:
            raise RuntimeError("A Realtime response marker is already pending")
        if self._waiting_response_id is not None:
            if self._waiting_response_id != self._controller.active_response_id:
                self._waiting_response_id = None
                self._waiting_tool_choice = None
                self._waiting_tools = None
                self._waiting_instructions = None
                self._waiting_max_output_tokens = None
                self._waiting_parallel_tool_calls = None
                self._waiting_truncation = None
                self._waiting_input_messages = None
                self._waiting_output_modalities = None
                self._waiting_audio_output = None
            else:
                raise RuntimeError("A Realtime response is still waiting for its inference context")
        if self._running_response_id is not None:
            if self._running_response_id != self._controller.active_response_id:
                self._running_response_id = None
            else:
                raise RuntimeError("A Realtime response is still running")
        if self._controller.pipeline_response_pending:
            raise RealtimeProtocolError(
                message="A pipeline-triggered response is already starting",
                code="response_in_progress",
                param="response",
            )

    def cancel_response(self, response_id: str) -> None:
        """Prevent a cancelled pre-transcript response from launching later."""
        if self._waiting_response_id == response_id:
            self._waiting_response_id = None
            self._waiting_tool_choice = None
            self._waiting_tools = None
            self._waiting_instructions = None
            self._waiting_max_output_tokens = None
            self._waiting_parallel_tool_calls = None
            self._waiting_truncation = None
            self._waiting_input_messages = None
            self._waiting_output_modalities = None
            self._waiting_audio_output = None
        elif self._pending_response_marker_id == response_id:
            self._pending_response_marker_id = None
            self._remember_cancelled_marker(response_id)
        self._cancel_deferred_response()
        self._response_slot_changed.set()

    def finish_response(self, response_id: str) -> None:
        """Retire coordination state after the terminal event reaches the wire."""
        if self._waiting_response_id == response_id:
            self._waiting_response_id = None
            self._waiting_tool_choice = None
            self._waiting_tools = None
            self._waiting_instructions = None
            self._waiting_max_output_tokens = None
            self._waiting_parallel_tool_calls = None
            self._waiting_truncation = None
            self._waiting_input_messages = None
            self._waiting_output_modalities = None
            self._waiting_audio_output = None
        if self._running_response_id == response_id:
            self._running_response_id = None
        if self._pending_response_marker_id == response_id:
            self._pending_response_marker_id = None
            self._remember_cancelled_marker(response_id)
        self._pending_pipeline_response_abort = None
        self._pending_pipeline_response_generation = None
        self._response_slot_changed.set()

    def _abort_pending_pipeline_response(self, *, older_than_generation: int | None = None) -> None:
        """Abort only the still-unstarted snapshot owned by this gate."""
        if (
            older_than_generation is not None
            and self._pending_pipeline_response_generation is not None
            and self._pending_pipeline_response_generation >= older_than_generation
        ):
            return
        abort = self._pending_pipeline_response_abort
        self._pending_pipeline_response_abort = None
        self._pending_pipeline_response_generation = None
        if abort is not None:
            abort()
            return
        self._controller.cancel_pending_pipeline_response(older_than_generation=older_than_generation)

    def _remember_cancelled_marker(self, response_id: str) -> None:
        """Retire a queued marker without retaining unbounded tombstones."""
        self._cancelled_response_marker_ids[response_id] = None
        self._cancelled_response_marker_ids.move_to_end(response_id)
        if len(self._cancelled_response_marker_ids) > 128:
            self._cancelled_response_marker_ids.popitem(last=False)

    def _cancel_deferred_response(self, *, older_than_generation: int | None = None) -> None:
        request_id = self._pending_deferred_response_id
        if request_id is None:
            return
        if (
            older_than_generation is not None
            and self._pending_deferred_response_generation is not None
            and self._pending_deferred_response_generation >= older_than_generation
        ):
            return
        self._pending_deferred_response_id = None
        self._pending_deferred_response_generation = None
        self._pending_deferred_audio_response = False
        self._cancelled_deferred_response_ids[request_id] = None
        self._cancelled_deferred_response_ids.move_to_end(request_id)
        if len(self._cancelled_deferred_response_ids) > 128:
            self._cancelled_deferred_response_ids.popitem(last=False)

    def fail_commit(self) -> None:
        """Release failed input and suppress any response queued for it."""
        self._abort_pending_pipeline_response()
        self._unclaimed_manual_commits = max(0, self._unclaimed_manual_commits - self._pending_commits)
        self._pending_commits = 0
        self._waiting_response_id = None
        self._waiting_tool_choice = None
        self._waiting_tools = None
        self._waiting_instructions = None
        self._waiting_max_output_tokens = None
        self._waiting_parallel_tool_calls = None
        self._waiting_truncation = None
        self._waiting_input_messages = None
        self._waiting_output_modalities = None
        self._waiting_audio_output = None
        self._running_response_id = None
        if self._pending_response_marker_id is not None:
            self._remember_cancelled_marker(self._pending_response_marker_id)
            self._pending_response_marker_id = None
        self._cancel_deferred_response()
        self._response_slot_changed.set()

    def reset(self) -> None:
        """Release all connection-local manual response coordination state."""
        self._closed = True
        self._abort_pending_pipeline_response()
        self._pending_commits = 0
        self._unclaimed_manual_commits = 0
        self._pending_automatic_response_contexts = 0
        self._waiting_response_id = None
        self._waiting_tool_choice = None
        self._waiting_tools = None
        self._waiting_instructions = None
        self._waiting_max_output_tokens = None
        self._waiting_parallel_tool_calls = None
        self._waiting_truncation = None
        self._waiting_input_messages = None
        self._waiting_output_modalities = None
        self._waiting_audio_output = None
        self._running_response_id = None
        self._pending_response_marker_id = None
        preparation_owned = self._response_preparation_id is not None or self._session_tool_preparation_id is not None
        self._response_preparation_id = None
        self._session_tool_preparation_id = None
        if preparation_owned and self._preparation_lock.locked():
            self._preparation_lock.release()
        self._service_audio_response_reservations.clear()
        self._session_tool_waiting_context = None
        self._cancelled_response_marker_ids.clear()
        self._pending_deferred_response_id = None
        self._pending_deferred_response_generation = None
        self._pending_deferred_audio_response = False
        self._cancelled_deferred_response_ids.clear()
        self._response_slot_changed.set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Drop commit-driven runs and release only an explicitly requested response."""
        await super().process_frame(frame, direction)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, UserStoppedSpeakingFrame):
            # A stop strategy can trigger without a non-empty aggregation. Its
            # typed inference marker then has no context to consume it; retire
            # that marker at the same semantic turn boundary so it cannot be
            # mistaken for a later tool or server-initiated context.
            self._pending_automatic_response_contexts = 0

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, InterruptionFrame):
            # Pipecat has already flushed interruptible data frames at this
            # processor. Retire the matching reservation as part of that same
            # system-frame transition so a later response is not blocked.
            interruption_generation, _ = self._controller.observe_interruption(frame.id)
            self._cancel_deferred_response(older_than_generation=interruption_generation)
            self._abort_pending_pipeline_response(older_than_generation=interruption_generation)
            self._service_audio_response_reservations = {
                reservation_id: generation
                for reservation_id, generation in self._service_audio_response_reservations.items()
                if generation >= interruption_generation
            }
            self._response_slot_changed.set()

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeConversationAppendFrame):
            if self._llm_context is None:
                raise RuntimeError("Realtime conversation append has no canonical LLM context")
            if self._conversation_append_handler is None:
                raise RuntimeError("Realtime conversation append handler is not installed")
            await self._conversation_append_handler(
                frame.item_id,
                frame.context_message,
                list(frame.events),
            )
            return

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeIdleTimeoutFrame):
            if self._idle_timeout_handler is None:
                raise RuntimeError("Realtime idle timeout handler is not installed")
            response_id = await self._idle_timeout_handler(frame)
            if response_id is None or self._closed:
                return
            frame = RealtimeResponseCreateFrame(response_id=response_id)

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeClientToolOutputFrame):
            if self._client_tool_output_handler is None:
                raise RuntimeError("Realtime client-tool output handler is not installed")
            await self._client_tool_output_handler(frame)
            return

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeDeferredResponseCreateFrame):
            if frame.request_id in self._cancelled_deferred_response_ids:
                self._cancelled_deferred_response_ids.pop(frame.request_id)
                return
            if self._closed:
                return
            if self._pending_deferred_response_id != frame.request_id:
                raise RuntimeError("Deferred Realtime response has no registered owner")
            if self._deferred_response_handler is None:
                raise RuntimeError("Deferred Realtime response handler is not installed")
            try:
                response_id = await self._deferred_response_handler(frame, frame.activation_generation)
            finally:
                still_owned = self._pending_deferred_response_id == frame.request_id
                if still_owned:
                    self._pending_deferred_response_id = None
                    self._pending_deferred_response_generation = None
                    self._pending_deferred_audio_response = False
                    self._response_slot_changed.set()
            if frame.request_id in self._cancelled_deferred_response_ids:
                self._cancelled_deferred_response_ids.pop(frame.request_id)
                return
            if self._closed:
                return
            if not still_owned:
                raise RuntimeError("Deferred Realtime response ownership changed during activation")
            if response_id is None or self._controller.active_response_id != response_id:
                return
            if self._waiting_response_id is not None or self._running_response_id is not None:
                raise RuntimeError("A Realtime response is already coordinated by the response gate")
            if self._pending_commits:
                self._waiting_response_id = response_id
                self._waiting_tool_choice = copy.deepcopy(frame.tool_choice)
                self._waiting_tools = copy.deepcopy(frame.tools)
                self._waiting_instructions = frame.instructions
                self._waiting_max_output_tokens = frame.max_output_tokens
                self._waiting_parallel_tool_calls = frame.parallel_tool_calls
                self._waiting_truncation = copy.deepcopy(frame.truncation)
                self._waiting_input_messages = copy.deepcopy(frame.input_messages)
                self._waiting_output_modalities = copy.deepcopy(frame.output_modalities)
                self._waiting_audio_output = copy.deepcopy(frame.audio_output)
                return
            if self._llm_context is None:
                raise RuntimeError("Realtime response gate has no canonical LLM context")
            self._running_response_id = response_id
            await self._push_response_run(
                self._response_context_frame(
                    response_id=response_id,
                    canonical_context=self._llm_context,
                    tool_choice=frame.tool_choice,
                    tools=frame.tools,
                    instructions=frame.instructions,
                    max_output_tokens=frame.max_output_tokens,
                    parallel_tool_calls=frame.parallel_tool_calls,
                    truncation=frame.truncation,
                    input_messages=frame.input_messages,
                ),
                output_modalities=frame.output_modalities,
                audio_output=frame.audio_output,
            )
            return

        automatic_response_context = False
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            self._llm_context = (
                frame.canonical_context if isinstance(frame, RealtimeResponseContextFrame) else frame.context
            )
            if self._pending_automatic_response_contexts:
                self._pending_automatic_response_contexts -= 1
                automatic_response_context = True

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, RealtimeResponseCreateFrame):
            if frame.response_id in self._cancelled_response_marker_ids:
                self._cancelled_response_marker_ids.pop(frame.response_id)
                return
            if self._pending_response_marker_id != frame.response_id:
                raise RuntimeError("Realtime response marker has no registered owner")
            self._pending_response_marker_id = None
            if self._controller.active_response_id != frame.response_id:
                # A transcription failure or downstream cancellation can make
                # the response terminal while its marker is still queued.
                # Never turn that stale marker into a new model generation.
                return
            if self._waiting_response_id is not None:
                raise RuntimeError("A Realtime response is already waiting for input context")
            if self._pending_commits:
                self._waiting_response_id = frame.response_id
                self._waiting_tool_choice = copy.deepcopy(frame.tool_choice)
                self._waiting_tools = copy.deepcopy(frame.tools)
                self._waiting_instructions = frame.instructions
                self._waiting_max_output_tokens = frame.max_output_tokens
                self._waiting_parallel_tool_calls = frame.parallel_tool_calls
                self._waiting_truncation = copy.deepcopy(frame.truncation)
                self._waiting_input_messages = copy.deepcopy(frame.input_messages)
                self._waiting_output_modalities = copy.deepcopy(frame.output_modalities)
                self._waiting_audio_output = copy.deepcopy(frame.audio_output)
                return
            if self._llm_context is None:
                raise RuntimeError("Realtime response gate has no canonical LLM context")
            self._running_response_id = frame.response_id
            await self._push_response_run(
                self._response_context_frame(
                    response_id=frame.response_id,
                    canonical_context=self._llm_context,
                    tool_choice=frame.tool_choice,
                    tools=frame.tools,
                    instructions=frame.instructions,
                    max_output_tokens=frame.max_output_tokens,
                    parallel_tool_calls=frame.parallel_tool_calls,
                    truncation=frame.truncation,
                    input_messages=frame.input_messages,
                ),
                output_modalities=frame.output_modalities,
                audio_output=frame.audio_output,
            )
            return

        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame) and self._pending_commits:
            self._pending_commits -= 1
            if self._waiting_response_id is None or self._pending_commits:
                return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._running_response_id is not None
        ):
            return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._session_tool_preparation_id is not None
        ):
            # The context is connection-owned and already contains every
            # committed message. Keep only its latest run trigger, then resume
            # it after the new session tool projection is installed.
            origin = (
                RealtimeResponseOrigin.AUTOMATIC_USER_TURN
                if automatic_response_context
                else RealtimeResponseOrigin.SERVICE_INITIATED
            )
            self._session_tool_waiting_context = (frame, origin)
            return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._response_preparation_id is not None
        ):
            # Preserve the canonical context above while preventing an
            # automatic run from racing response-scoped MCP discovery.
            return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._pending_response_marker_id is not None
        ):
            # A response-local marker is already ordered in the transport but
            # has not reached this gate. Suppress an earlier automatic run; the
            # marker will request a fresh context containing the same canonical
            # conversation state and bind the override to that exact run.
            return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._waiting_response_id is None
            and automatic_response_context
            and not self._controller.automatic_response_enabled
        ):
            # The aggregator has already committed this turn to the canonical
            # context. Keep that state for a later explicit response.create,
            # but do not turn the context publication into model inference.
            return

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and self._waiting_response_id is not None
        ):
            response_id = self._waiting_response_id
            tool_choice = self._waiting_tool_choice
            tools = self._waiting_tools
            instructions = self._waiting_instructions
            max_output_tokens = self._waiting_max_output_tokens
            parallel_tool_calls = self._waiting_parallel_tool_calls
            truncation = self._waiting_truncation
            input_messages = self._waiting_input_messages
            output_modalities = self._waiting_output_modalities
            audio_output = self._waiting_audio_output
            self._waiting_response_id = None
            self._waiting_tool_choice = None
            self._waiting_tools = None
            self._waiting_instructions = None
            self._waiting_max_output_tokens = None
            self._waiting_parallel_tool_calls = None
            self._waiting_truncation = None
            self._waiting_input_messages = None
            self._waiting_output_modalities = None
            self._waiting_audio_output = None
            if self._controller.active_response_id != response_id:
                return
            self._running_response_id = response_id
            await self._push_response_output_config(
                output_modalities=output_modalities,
                audio_output=audio_output,
            )
            frame = self._response_context_frame(
                response_id=response_id,
                canonical_context=frame.context,
                tool_choice=tool_choice,
                tools=tools,
                instructions=instructions,
                max_output_tokens=max_output_tokens,
                parallel_tool_calls=parallel_tool_calls,
                truncation=truncation,
                input_messages=input_messages,
                original_frame=frame,
            )

        if (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, LLMContextFrame)
            and not isinstance(frame, RealtimeResponseContextFrame)
            and self._waiting_response_id is None
        ):
            if self._service_audio_response_reservations:
                # A fused audio task owns the only raw utterance and its
                # transcription. Keep this context as canonical state, but do
                # not let its echo claim the response before that task.
                return
            origin = (
                RealtimeResponseOrigin.AUTOMATIC_USER_TURN
                if automatic_response_context
                else RealtimeResponseOrigin.SERVICE_INITIATED
            )
            frame = await self._prepare_pipeline_response_context(frame.context, origin=origin)

        await self.push_frame(frame, direction)

    async def _prepare_pipeline_response_context(
        self,
        canonical_context: LLMContext,
        *,
        origin: RealtimeResponseOrigin,
    ) -> LLMContextFrame:
        """Freeze every unowned automatic or tool-followup inference run."""
        frozen = self._freeze_pipeline_response(
            canonical_context,
            require_output_runtime=False,
            origin=origin,
        )
        return await self._publish_pipeline_response_snapshot(frozen)

    async def _publish_pipeline_response_snapshot(
        self,
        frozen: ServiceResponseSnapshot,
    ) -> LLMContextFrame:
        """Publish setup for an already-frozen unowned response."""
        context, setup_frames, abort = frozen
        try:
            for setup_frame in setup_frames:
                await self.push_frame(setup_frame, FrameDirection.DOWNSTREAM)
        except BaseException:
            abort()
            raise
        return LLMContextFrame(context=context)

    async def prepare_service_response_snapshot(
        self,
        canonical_context: LLMContext,
        reservation_id: str | None = None,
        origin: RealtimeResponseOrigin = RealtimeResponseOrigin.SERVICE_INITIATED,
    ) -> FusedServiceResponseSnapshot | None:
        """Freeze an actual LLM run that originates inside a fused service.

        Fused audio services start work from a user-stop frame, and a bridged
        service can receive tool-result contexts in the reverse direction. Both
        paths bypass this gate as a processor, so the service calls this hook at
        its last pre-inference boundary and emits the returned setup frames
        before its ``LLMFullResponseStartFrame``.
        """
        if not isinstance(origin, RealtimeResponseOrigin):
            raise TypeError("Realtime service response origin must be a RealtimeResponseOrigin")
        activation_generation = self._controller.interruption_generation
        await self._preparation_lock.acquire()
        try:
            if reservation_id is not None:
                reservation_generation = self._service_audio_response_reservations.pop(reservation_id, None)
                if reservation_generation is None:
                    return None
                activation_generation = reservation_generation
            if (
                self._pending_deferred_response_id is not None
                or self._pending_response_marker_id is not None
                or self._waiting_response_id is not None
                or self._running_response_id is not None
                or self._pending_pipeline_response_abort is not None
                or self._controller.response_in_progress
                or self._controller.pipeline_response_pending
            ):
                return None
            return self._create_service_response_snapshot(
                canonical_context,
                activation_generation=activation_generation,
                origin=origin,
            )
        finally:
            self._preparation_lock.release()

    async def prepare_deferred_service_response_snapshot(
        self,
        canonical_context: LLMContext,
    ) -> FusedServiceResponseSnapshot | None:
        """Wait for and claim a response slot for one durable service result.

        Media analysis and other long-running work can legitimately finish after
        intervening user turns. It therefore waits for the default-conversation
        response slot instead of inheriting the response active at completion.
        The interruption generation is captured only when this result freezes
        its own response; an interruption after that point cancels the snapshot
        through the normal owned-response path.
        """
        while True:
            if self._closed:
                return None

            # Clear before inspecting state so a concurrent terminal transition
            # cannot be lost between the busy check and the wait below.
            self._response_slot_changed.clear()
            await self._preparation_lock.acquire()
            try:
                if self._closed:
                    return None
                busy = (
                    bool(self._service_audio_response_reservations)
                    or self._pending_deferred_response_id is not None
                    or self._pending_response_marker_id is not None
                    or self._waiting_response_id is not None
                    or self._running_response_id is not None
                    or self._pending_pipeline_response_abort is not None
                    or self._controller.response_in_progress
                    or self._controller.pipeline_response_pending
                )
                if not busy:
                    return self._create_service_response_snapshot(
                        canonical_context,
                        activation_generation=self._controller.interruption_generation,
                        origin=RealtimeResponseOrigin.INTERNAL_TOOL_CONTINUATION,
                    )
            finally:
                self._preparation_lock.release()
            await self._response_slot_changed.wait()

    def _create_service_response_snapshot(
        self,
        canonical_context: LLMContext,
        *,
        activation_generation: int,
        origin: RealtimeResponseOrigin,
    ) -> FusedServiceResponseSnapshot:
        """Freeze one service response while its preparation owner is held."""
        owner_id = new_realtime_id("run")
        frozen = self._freeze_pipeline_response(
            canonical_context,
            require_output_runtime=True,
            owner_id=owner_id,
            activation_generation=activation_generation,
            origin=origin,
        )

        async def activate(on_started: PipelineResponseStartedHandler | None = None) -> str | None:
            self.validate_activation_generation(activation_generation)
            if self._pipeline_response_start_handler is None:
                raise RealtimeProtocolError(
                    message="Service responses require an atomic Realtime start runtime",
                    code="response_start_runtime_missing",
                    param="response",
                    error_type="server_error",
                )
            if on_started is None:
                response_id = await self._pipeline_response_start_handler(owner_id, activation_generation)
            else:
                response_id = await self._pipeline_response_start_handler(
                    owner_id,
                    activation_generation,
                    on_started,
                )
            if response_id is None:
                frozen[2]()
            else:
                if self._running_response_id is not None:
                    raise RuntimeError("A Realtime response is already coordinated by the response gate")
                self._running_response_id = response_id
                self._pending_pipeline_response_abort = None
                self._pending_pipeline_response_generation = None
            return response_id

        return frozen[0], frozen[1], activate, frozen[2]

    def _freeze_pipeline_response(
        self,
        canonical_context: LLMContext,
        *,
        require_output_runtime: bool,
        origin: RealtimeResponseOrigin,
        owner_id: str | None = None,
        activation_generation: int | None = None,
    ) -> ServiceResponseSnapshot:
        owner_id = owner_id or new_realtime_id("run")
        activation_generation = (
            self._controller.interruption_generation if activation_generation is None else activation_generation
        )
        snapshot = self._controller.prepare_pipeline_response(
            owner_id=owner_id,
            interruption_generation=activation_generation,
        )
        self._pending_pipeline_response_generation = activation_generation
        audio_output = snapshot["audio_output"]
        output_activated = False
        try:
            if require_output_runtime:
                self.validate_response_output(
                    output_modalities=snapshot["output_modalities"],
                    audio_output=audio_output,
                )
                if audio_output is not None and self._pipeline_response_output_handler is None:
                    raise RealtimeProtocolError(
                        message="Fused audio responses require a linked Realtime wire output",
                        code="response_output_runtime_missing",
                        param="response.audio.output.format",
                        error_type="server_error",
                    )
            context = RealtimeResponseLLMContext(
                copy.deepcopy(list(canonical_context.get_messages())),
                tools=(NOT_GIVEN if canonical_context.tools is NOT_GIVEN else copy.deepcopy(canonical_context.tools)),
                tool_choice=(
                    "auto"
                    if origin is RealtimeResponseOrigin.INTERNAL_TOOL_CONTINUATION
                    else (
                        NOT_GIVEN
                        if canonical_context.tool_choice is NOT_GIVEN
                        else copy.deepcopy(canonical_context.tool_choice)
                    )
                ),
                max_output_tokens=snapshot["max_output_tokens"],
                parallel_tool_calls=snapshot["parallel_tool_calls"],
                truncation=snapshot["truncation"],
                preserve_prompt_messages=len(self._canonical_prompt_messages or ()),
                run_owner_id=owner_id,
                activation_generation=activation_generation,
            )
            setup_frames = self._response_output_config_frames(
                output_modalities=snapshot["output_modalities"],
                audio_output=audio_output,
            )
            if audio_output is not None and self._pipeline_response_output_handler is not None:
                self._pipeline_response_output_handler(audio_output)
                output_activated = True
        except BaseException:
            if (
                self._controller.cancel_pending_pipeline_response(owner_id=owner_id)
                and audio_output is not None
                and self._pipeline_response_output_reset_handler is not None
            ):
                self._pipeline_response_output_reset_handler()
            raise

        def abort() -> None:
            cancelled = self._controller.cancel_pending_pipeline_response(owner_id=owner_id)
            if not cancelled and (self._controller.response_in_progress or self._controller.pipeline_response_pending):
                return
            self._pending_pipeline_response_generation = None
            if output_activated and self._pipeline_response_output_reset_handler is not None:
                self._pipeline_response_output_reset_handler()
            self._response_slot_changed.set()

        self._pending_pipeline_response_abort = abort
        return context, setup_frames, abort

    async def _push_response_run(
        self,
        frame: LLMContextFrame,
        *,
        output_modalities: list[str] | None,
        audio_output: dict[str, Any] | None,
    ) -> None:
        """Apply one response snapshot before releasing its model context."""
        await self._push_response_output_config(
            output_modalities=output_modalities,
            audio_output=audio_output,
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    async def _push_response_output_config(
        self,
        *,
        output_modalities: list[str] | None,
        audio_output: dict[str, Any] | None,
    ) -> None:
        """Project effective modality and voice in deterministic frame order."""
        for frame in self._response_output_config_frames(
            output_modalities=output_modalities,
            audio_output=audio_output,
        ):
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _response_output_config_frames(
        self,
        *,
        output_modalities: list[str] | None,
        audio_output: dict[str, Any] | None,
    ) -> tuple[Frame, ...]:
        """Build the ordered service frames for one effective output snapshot."""
        modalities = output_modalities
        if modalities is None:
            return ()
        frames: list[Frame] = [LLMConfigureOutputFrame(skip_tts=modalities == ["text"])]
        if modalities != ["audio"] or self._tts_service is None:
            return tuple(frames)
        voice = audio_output.get("voice") if isinstance(audio_output, dict) else None
        if not isinstance(voice, str) or not voice:
            raise RuntimeError("Realtime audio response has no executable NVIDIA TTS voice")
        frames.append(
            TTSUpdateSettingsFrame(
                delta=NvidiaTTSSettings(voice=voice),
                service=self._tts_service,
            )
        )
        return tuple(frames)

    def _response_context_frame(
        self,
        *,
        response_id: str,
        canonical_context: LLMContext,
        tool_choice: str | dict[str, Any] | None,
        tools: list[dict[str, Any]] | None,
        instructions: str | None,
        max_output_tokens: int | str | None,
        parallel_tool_calls: bool | None,
        truncation: str | dict[str, Any] | None,
        input_messages: list[dict[str, Any]] | None,
        original_frame: LLMContextFrame | None = None,
    ) -> LLMContextFrame:
        """Build the exact downstream context for one explicit response."""
        response_messages = list(canonical_context.get_messages())
        prompt_messages = copy.deepcopy(self._canonical_prompt_messages or [])
        if input_messages is not None:
            renderer, _ = self._require_instruction_binding(param="response.input")
            prompt = self._render_instructions(renderer, instructions) if instructions is not None else prompt_messages
            prompt_messages = prompt
            response_messages = [*prompt, *copy.deepcopy(input_messages)]
        elif instructions is not None:
            renderer, _ = self._require_instruction_binding(param="response.instructions")
            prompt_messages = self._render_instructions(renderer, instructions)
            response_messages = [
                *prompt_messages,
                *self._canonical_tail(canonical_context),
            ]
        response_context = RealtimeResponseLLMContext(
            response_messages,
            tools=canonical_context.tools if tools is None else project_function_tools(tools),
            tool_choice=(canonical_context.tool_choice if tool_choice is None else copy.deepcopy(tool_choice)),
            max_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            truncation=truncation,
            preserve_prompt_messages=len(prompt_messages),
            response_id=response_id,
        )
        response_frame = RealtimeResponseContextFrame(
            context=response_context,
            response_id=response_id,
            canonical_context=canonical_context,
        )
        if original_frame is not None:
            response_frame.pts = original_frame.pts
            response_frame.broadcast_sibling_id = original_frame.broadcast_sibling_id
            response_frame.metadata = copy.deepcopy(original_frame.metadata)
            response_frame.transport_source = original_frame.transport_source
            response_frame.transport_destination = original_frame.transport_destination
        return response_frame


class RealtimeToolResultPolicy(FrameProcessor):
    """Keep mixed Realtime tool rounds under one explicit continuation owner."""

    def __init__(self, *, controller: RealtimeSessionController) -> None:
        """Bind the policy to one WebSocket connection's canonical controller."""
        super().__init__()
        self._controller = controller
        self._client_continued_pipeline_calls: set[str] = set()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Suppress pipeline auto-run when a sibling is client-managed."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, FunctionCallsStartedFrame):
            calls = tuple(frame.function_calls or ())
            owners = tuple(
                self._controller.pipeline_tool_owner(
                    call_id=call.tool_call_id or "",
                    pipeline_name=call.function_name or "",
                )
                for call in calls
            )
            if any(owner in {"client", "mcp"} for owner in owners):
                self._client_continued_pipeline_calls.update(
                    call.tool_call_id
                    for call, owner in zip(calls, owners, strict=True)
                    if call.tool_call_id and owner in {"server", "delegate"}
                )
        elif (
            direction == FrameDirection.DOWNSTREAM
            and isinstance(frame, FunctionCallResultFrame)
            and frame.tool_call_id in self._client_continued_pipeline_calls
        ):
            properties = frame.properties or FunctionCallResultProperties()
            frame.properties = replace(properties, run_llm=False)
            if frame.properties.is_final:
                self._client_continued_pipeline_calls.discard(frame.tool_call_id)
        await self.push_frame(frame, direction)


class RealtimeManualTurnStopStrategy(BaseUserTurnStopStrategy):
    """Finalize an externally bounded turn after its final ASR text settles."""

    def __init__(
        self,
        *,
        response_gate: RealtimeManualResponseGate,
        transcript_settle_seconds: float = 0.5,
    ) -> None:
        """Bind transcript-settle tracking to one manual response gate."""
        super().__init__(enable_user_speaking_frames=False)
        self._response_gate = response_gate
        self._transcript_settle_seconds = transcript_settle_seconds
        self._text_observed = False
        self._interim_observed = False
        self._final_transcription_observed = False
        self._user_stopped = False
        self._triggered = False
        self._turn_frame_id: int | None = None
        self._changed = asyncio.Event()
        self._watcher: asyncio.Task[None] | None = None

    @property
    def turn_frame_id(self) -> int | None:
        """Return the immutable raw stop-frame token for the current turn."""
        return self._turn_frame_id

    async def setup(self, task_manager: BaseTaskManager) -> None:
        """Start the bounded transcript-settle watcher."""
        await super().setup(task_manager)
        self._watcher = task_manager.create_task(
            self._watch_transcript(),
            f"{self}::_watch_transcript",
        )

    async def cleanup(self) -> None:
        """Stop the transcript-settle watcher."""
        if self._watcher is not None:
            await self.task_manager.cancel_task(self._watcher)
            self._watcher = None
        await super().cleanup()

    async def handle_user_turn_started(self) -> None:
        """Arm state for a newly committed manual input turn."""
        self._text_observed = False
        self._interim_observed = False
        self._final_transcription_observed = False
        self._user_stopped = False
        self._triggered = False
        self._turn_frame_id = None
        self._changed.clear()

    async def handle_user_turn_stopped(self) -> None:
        """Release settled state after the controller finalizes the turn."""
        self._user_stopped = False
        self._changed.clear()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Track the explicit stop boundary and its final transcript frames."""
        if isinstance(frame, UserStoppedSpeakingFrame):
            self._user_stopped = True
            self._turn_frame_id = frame.id
            await self._maybe_finalize()
        elif isinstance(frame, TranscriptionFrame):
            self._final_transcription_observed = True
            self._interim_observed = False
            if frame.text and frame.text.strip():
                self._text_observed = True
            self._changed.set()
        elif isinstance(frame, InterimTranscriptionFrame):
            self._interim_observed = True
        return ProcessFrameResult.CONTINUE

    async def _watch_transcript(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._changed.wait(),
                    timeout=self._transcript_settle_seconds,
                )
                self._changed.clear()
            except TimeoutError:
                await self._maybe_finalize()

    async def _maybe_finalize(self) -> None:
        if (
            self._triggered
            or not self._user_stopped
            or not self._final_transcription_observed
            or self._interim_observed
        ):
            return
        self._triggered = True
        if self._text_observed:
            await self.trigger_user_turn_stopped()
        else:
            self._response_gate.fail_commit()
            await self.trigger_user_turn_stopped()


class _RealtimeWireSequencer:
    """Hold response events behind the exact input-transcript terminal event.

    Fused speech models start their internal Pipecat LLM lifecycle before the
    transcript field in the provider envelope has been decoded. Inference must
    keep running, but clients should observe the committed user transcript
    before the response it caused. This connection-local sequencer operates at
    the final wire boundary, so it also covers client-requested responses and
    every response event emitted by the serializer or lifecycle observer.
    """

    def __init__(self, controller: RealtimeSessionController) -> None:
        self._controller = controller
        self._pending_input_items: OrderedDict[str, None] = OrderedDict()
        self._response_barriers: OrderedDict[str, str] = OrderedDict()
        self._deferred_events: list[tuple[str, dict[str, Any]]] = []
        self._mcp_item_response_ids: dict[str, str] = {}

    def reset(self) -> None:
        """Release all connection-local ordering state without emitting it."""
        self._pending_input_items.clear()
        self._response_barriers.clear()
        self._deferred_events.clear()
        self._mcp_item_response_ids.clear()

    def order(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return events currently safe to publish, preserving batch order."""
        if not events:
            return []

        ready: list[dict[str, Any]] = []
        batch_response_id = self._single_batch_response_id(events)
        self._remember_mcp_response_owners()

        for event in events:
            event_type = event.get("type")
            if event_type == "input_audio_buffer.committed" and self._input_transcription_enabled():
                item_id = event.get("item_id")
                if isinstance(item_id, str) and item_id:
                    self._pending_input_items.setdefault(item_id, None)

            if event_type in _INPUT_TRANSCRIPTION_TERMINAL_EVENTS:
                ready.append(event)
                item_id = event.get("item_id")
                if isinstance(item_id, str) and item_id in self._pending_input_items:
                    self._pending_input_items.pop(item_id, None)
                    # Keep the transcript terminal and every response event it
                    # releases adjacent within this serialized wire batch.
                    ready.extend(self._release_item(item_id))
                continue

            response_id = self._event_response_id(event)
            if event_type == "response.created" and response_id is not None:
                barrier_item_id = self._claim_barrier_item()
                if barrier_item_id is not None:
                    self._response_barriers[response_id] = barrier_item_id
                    # The controller has already registered every tool call in
                    # this response batch. Capture their public MCP item IDs
                    # now that the response owns a transcript barrier.
                    self._remember_mcp_response_owners()

            if response_id is None:
                response_id = self._mcp_event_response_id(event)
            if response_id is None and self._is_batch_response_event(event):
                response_id = batch_response_id
            if response_id is not None and response_id in self._response_barriers:
                self._deferred_events.append((response_id, copy.deepcopy(event)))
            else:
                ready.append(event)

        return ready

    def _input_transcription_enabled(self) -> bool:
        session = self._controller.public_session()
        audio = session.get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        return isinstance(input_audio, dict) and input_audio.get("transcription") is not None

    def _claim_barrier_item(self) -> str | None:
        if not self._pending_input_items:
            return None
        claimed = set(self._response_barriers.values())
        for item_id in self._pending_input_items:
            if item_id not in claimed:
                return item_id
        # A tool-recovery response can belong to the same still-pending user
        # turn as Response A. Reuse its most recent exact barrier rather than
        # transferring the response to an unrelated later turn.
        if self._response_barriers:
            return next(reversed(self._response_barriers.values()))
        return next(iter(self._pending_input_items))

    def _release_item(self, item_id: str) -> list[dict[str, Any]]:
        response_ids = {
            response_id
            for response_id, barrier_item_id in self._response_barriers.items()
            if barrier_item_id == item_id
        }
        if not response_ids:
            return []
        released = [event for response_id, event in self._deferred_events if response_id in response_ids]
        self._deferred_events = [
            (response_id, event) for response_id, event in self._deferred_events if response_id not in response_ids
        ]
        for response_id in response_ids:
            self._response_barriers.pop(response_id, None)
        self._mcp_item_response_ids = {
            mcp_item_id: response_id
            for mcp_item_id, response_id in self._mcp_item_response_ids.items()
            if response_id not in response_ids
        }
        return released

    def _remember_mcp_response_owners(self) -> None:
        """Correlate native MCP item IDs while their response barrier is active."""
        if not self._response_barriers:
            return
        for call_id in self._controller.pending_tool_call_ids():
            record = self._controller.tool_call(call_id)
            if record is None or record.owner != "mcp" or record.response_id not in self._response_barriers:
                continue
            self._mcp_item_response_ids[record.item_id] = record.response_id
            if record.approval_request_id is not None:
                self._mcp_item_response_ids[record.approval_request_id] = record.response_id

    def _mcp_event_response_id(self, event: dict[str, Any]) -> str | None:
        """Resolve response-owned MCP events whose public schema omits response_id."""
        event_type = event.get("type")
        item_id: Any = None
        if isinstance(event_type, str) and event_type.startswith("response.mcp_call."):
            item_id = event.get("item_id")
        elif event_type in {"conversation.item.added", "conversation.item.done"}:
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") in {"mcp_call", "mcp_approval_request"}:
                item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return None
        return self._mcp_item_response_ids.get(item_id)

    @classmethod
    def _single_batch_response_id(cls, events: list[dict[str, Any]]) -> str | None:
        response_ids = {response_id for event in events if (response_id := cls._event_response_id(event))}
        return next(iter(response_ids)) if len(response_ids) == 1 else None

    @staticmethod
    def _event_response_id(event: dict[str, Any]) -> str | None:
        response_id = event.get("response_id")
        if isinstance(response_id, str) and response_id:
            return response_id
        if event.get("type") in {"response.created", "response.done"}:
            response = event.get("response")
            response_id = response.get("id") if isinstance(response, dict) else None
            if isinstance(response_id, str) and response_id:
                return response_id
        return None

    @staticmethod
    def _is_batch_response_event(event: dict[str, Any]) -> bool:
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error")
            return isinstance(error, dict) and error.get("type") == "server_error"
        if event_type not in {"conversation.item.added", "conversation.item.done"}:
            return False
        item = event.get("item")
        return isinstance(item, dict) and (item.get("role") == "assistant" or item.get("type") == "function_call")


def create_realtime_transport(
    websocket: Any,
    *,
    controller: RealtimeSessionController,
) -> FastAPIWebsocketTransport:
    """Build a FastAPI transport whose serializer shares the gateway controller."""
    serializer = RealtimeFrameSerializer(controller=controller)
    response_gate = RealtimeManualResponseGate(controller=controller)
    manual_commit_publications = _RealtimeManualCommitPublications() if controller.manual_input_mode else None
    client_tool_broker = ClientToolBroker(
        output_timeout_secs=parse_env_float(
            "REALTIME_CLIENT_TOOL_TIMEOUT_SECONDS",
            120.0,
            min_value=1.0,
        )
    )
    serializer.set_client_tool_broker(client_tool_broker)
    serializer.bind_response_gate(response_gate)
    emit_lock = asyncio.Lock()
    wire_sequencer = _RealtimeWireSequencer(controller)
    idle_timeout: RealtimeServerVADIdleTimeout | None = None

    async def _emit_batch_locked(events: list[dict[str, Any]]) -> bool:
        """Publish one batch while the caller owns ``emit_lock``."""
        if not events:
            return not serializer.connection_closed
        ordered_events = wire_sequencer.order(events)
        payloads = [json.dumps(event, default=str, allow_nan=False) for event in ordered_events]
        for event, payload in zip(ordered_events, payloads, strict=True):
            response_id = _response_done_id(event)
            if response_id is not None:
                serializer.prepare_response_done_publication(response_id)
            try:
                await websocket.send_text(payload)
            except (RuntimeError, WebSocketDisconnect) as exc:
                logger.debug(f"Realtime emit skipped because the WebSocket is closed: {exc}")
                serializer.notify_connection_closed()
                return False
            if event.get("type") == "input_audio_buffer.committed" and manual_commit_publications is not None:
                manual_commit_publications.published()
            if (
                event.get("type") == "conversation.item.input_audio_transcription.failed"
                and controller.manual_input_mode
            ):
                response_gate.fail_commit()
            if response_id is not None:
                serializer.notify_response_done_published(response_id)
            if idle_timeout is not None:
                idle_timeout.observe_published_events([event])
        return True

    async def _emit_batch(events: list[dict[str, Any]]) -> None:
        if not events:
            return
        async with emit_lock:
            await _emit_batch_locked(events)

    async def _emit(event: dict[str, Any]) -> None:
        await _emit_batch([event])

    async def _close_failed_connection(reason: str) -> None:
        try:
            await websocket.close(code=1011, reason=reason)
        except (RuntimeError, WebSocketDisconnect) as exc:
            logger.debug(f"Realtime failure close skipped because the WebSocket is closed: {exc}")

    serializer.set_connection_failure_handler(_close_failed_connection)

    async def _emit_batch_without_cancellation(events: list[dict[str, Any]]) -> bool:
        """Finish an owned wire batch before propagating task cancellation."""
        publication = asyncio.create_task(_emit_batch_locked(events))
        try:
            return await asyncio.shield(publication)
        except asyncio.CancelledError:
            with suppress(asyncio.CancelledError):
                await publication
            raise

    async def _start_fused_pipeline_response(
        owner_id: str,
        activation_generation: int,
        on_started: PipelineResponseStartedHandler | None = None,
    ) -> str | None:
        """Activate and publish one exact frozen response as one wire transition."""
        async with controller.response_transition_lock, emit_lock:
            response_gate.validate_activation_generation(activation_generation)
            activated = controller.start_pending_pipeline_response(
                owner_id=owner_id,
                expected_generation=activation_generation,
            )
            if activated is None:
                return None
            response_id, events = activated
            try:
                wire_available = await _emit_batch_without_cancellation(events)
                if not wire_available:
                    return None
                if on_started is not None:
                    # Queue the exact typed provider boundary before another
                    # response transition can cancel/replace this owner.
                    await on_started(response_id)
            except asyncio.CancelledError:
                if controller.active_response_id == response_id:
                    cancel_reason = controller.consume_response_cancel_reason(response_id)
                    if serializer.connection_closed:
                        controller.abandon_response(response_id, reason="connection_closed")
                        raise
                    if cancel_reason is None:
                        terminal = controller.finish_response(
                            status="failed",
                            reason="response_start_cancelled",
                        )
                    else:
                        terminal = controller.finish_response(
                            status="cancelled",
                            reason=cancel_reason,
                        )
                    await _emit_batch_without_cancellation(terminal)
                raise
            except Exception:
                if controller.active_response_id == response_id:
                    terminal = [
                        error_event(
                            "The pipeline could not start the prepared response",
                            code="response_start_failed",
                            error_type="server_error",
                        ),
                        *controller.finish_response(
                            status="failed",
                            reason="response_start_failed",
                        ),
                    ]
                    await _emit_batch_without_cancellation(terminal)
                raise
            if serializer.connection_closed:
                return None
            return response_id

    response_gate.set_pipeline_response_start_handler(_start_fused_pipeline_response)

    input_audio_turn_failed = False

    async def _fail_input_audio_turn() -> None:
        """Close once, after the observer emits the ordered ASR failure batch."""
        nonlocal input_audio_turn_failed
        if input_audio_turn_failed:
            return
        input_audio_turn_failed = True
        try:
            await websocket.close(code=1011, reason="input audio transcription failed")
        except (RuntimeError, WebSocketDisconnect) as exc:
            logger.debug(f"Realtime ASR failure close skipped because the WebSocket is closed: {exc}")

    serializer.set_emit(_emit, _emit_batch)
    mcp_runtime = RealtimeMCPRuntime(controller=controller, emit_batch=_emit_batch)
    serializer.set_mcp_runtime(mcp_runtime)
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=PIPELINE_PCM_RATE,
            audio_out_enabled=True,
            audio_out_sample_rate=PIPELINE_OUTPUT_PCM_RATE,
            audio_out_10ms_chunks=parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    input_transport = transport.input()

    async def _trigger_idle_turn(
        audio_start_ms: int,
        audio_end_ms: int,
        generation: int,
        timeout_ms: int,
        preceding_assistant_item_id: str,
    ) -> None:
        """Queue one empty-audio turn for its canonical pipeline context owner."""
        if serializer.connection_closed:
            return
        await input_transport.push_frame(
            RealtimeIdleTimeoutFrame(
                audio_start_ms=audio_start_ms,
                audio_end_ms=audio_end_ms,
                generation=generation,
                timeout_ms=timeout_ms,
                preceding_assistant_item_id=preceding_assistant_item_id,
            )
        )

    async def _report_idle_timeout_error(
        _exc: Exception,
        *,
        published_response_id: str | None = None,
    ) -> None:
        """Expose an internal timeout failure and retire its untrusted connection."""
        failure = error_event(
            "The server-VAD idle timeout turn could not be processed",
            code="idle_timeout_error",
            error_type="server_error",
        )
        if published_response_id is None:
            serializer.notify_connection_closed()
            try:
                await _emit(failure)
            finally:
                await _close_failed_connection("server-VAD idle timeout failed")
            return

        async def _transition() -> None:
            events = [failure]
            async with controller.response_transition_lock:
                if controller.active_response_id == published_response_id:
                    response_gate.cancel_response(published_response_id)
                    events.extend(
                        controller.finish_response(
                            status="failed",
                            reason="idle_timeout_error",
                        )
                    )
                await _emit_batch(events)

        try:
            await serializer.run_connection_state_transition(_transition)
        finally:
            serializer.notify_connection_closed()
            await _close_failed_connection("server-VAD idle timeout failed")

    async def _commit_idle_timeout_turn(frame: RealtimeIdleTimeoutFrame) -> str | None:
        """Commit and publish an idle turn where the pipeline owns model context."""
        response_id: str | None = None
        admitted = False
        wire_available = False
        context_failure: Exception | None = None

        async def _transition() -> None:
            nonlocal admitted, response_id, wire_available
            async with controller.response_transition_lock, emit_lock:
                if idle_timeout is None or not idle_timeout.can_trigger(
                    generation=frame.generation,
                    timeout_ms=frame.timeout_ms,
                    preceding_assistant_item_id=frame.preceding_assistant_item_id,
                ):
                    return
                if context_failure is not None:
                    raise context_failure
                if response_gate.response_slot_occupied or serializer.has_pending_conversation_appends:
                    return
                response_gate.ensure_response_marker_available()
                admitted = True
                controller_snapshot = controller.snapshot_conversation_mutation_state()
                turn = controller.start_idle_timeout_turn(
                    audio_start_ms=frame.audio_start_ms,
                    audio_end_ms=frame.audio_end_ms,
                )
                response_id = turn.response_id
                try:
                    response_gate.register_response_marker(turn.response_id)
                    serializer.commit_server_conversation_context_message(
                        turn.item_id,
                        turn.context_message,
                    )
                except BaseException:
                    response_gate.cancel_response(turn.response_id)
                    controller.restore_conversation_mutation_state(controller_snapshot)
                    raise
                try:
                    wire_available = await _emit_batch_without_cancellation(turn.events)
                except BaseException:
                    response_gate.cancel_response(turn.response_id)
                    raise

        try:
            if idle_timeout is None or not idle_timeout.can_trigger(
                generation=frame.generation,
                timeout_ms=frame.timeout_ms,
                preceding_assistant_item_id=frame.preceding_assistant_item_id,
            ):
                return None
            context_ready = asyncio.create_task(
                serializer.wait_for_conversation_context_message(frame.preceding_assistant_item_id)
            )
            timer_stale = asyncio.create_task(idle_timeout.wait_until_stale(frame.generation))
            try:
                completed, _ = await asyncio.wait(
                    (context_ready, timer_stale),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if timer_stale in completed:
                    return None
                try:
                    await context_ready
                except Exception as exc:  # noqa: BLE001
                    # A client edit mutates controller/context state before its
                    # terminal event reaches the wire.  Treat a failed context
                    # barrier as provisional until that serialized event-state
                    # transaction completes and invalidates this timer.
                    context_failure = exc
            finally:
                for pending in (context_ready, timer_stale):
                    if not pending.done():
                        pending.cancel()
                with suppress(asyncio.CancelledError):
                    await asyncio.gather(context_ready, timer_stale, return_exceptions=True)
            await serializer.run_connection_state_transition(_transition)
        except asyncio.CancelledError:
            if response_id is not None and not serializer.connection_closed:
                cleanup = asyncio.create_task(
                    _report_idle_timeout_error(
                        RuntimeError("The server-VAD idle transition was cancelled after publication"),
                        published_response_id=response_id,
                    )
                )
                try:
                    await asyncio.shield(cleanup)
                except asyncio.CancelledError:
                    with suppress(asyncio.CancelledError):
                        await cleanup
            else:
                serializer.notify_connection_closed()
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Realtime server-VAD idle context transition failed")
            await _report_idle_timeout_error(exc)
            return None
        if not admitted:
            return None
        if not wire_available or serializer.connection_closed:
            if response_id is not None:
                response_gate.cancel_response(response_id)
            if not serializer.connection_closed:
                serializer.notify_connection_closed()
            await _close_failed_connection("server-VAD idle timeout failed")
            return None
        return response_id

    idle_timeout = RealtimeServerVADIdleTimeout(
        controller=controller,
        trigger_idle_turn=_trigger_idle_turn,
        report_error=_report_idle_timeout_error,
    )
    response_gate.set_idle_timeout_handler(_commit_idle_timeout_turn)
    serializer.set_idle_timeout(idle_timeout)

    if controller.manual_input_mode:
        manual_input_sample_cursor = 0
        endpoint_silence_ms = parse_env_int(
            "REALTIME_MANUAL_ASR_ENDPOINT_SILENCE_MS",
            _DEFAULT_MANUAL_ASR_ENDPOINT_SILENCE_MS,
            min_value=100,
        )
        if endpoint_silence_ms > _MAX_MANUAL_ASR_ENDPOINT_SILENCE_MS:
            logger.warning(
                "REALTIME_MANUAL_ASR_ENDPOINT_SILENCE_MS={} exceeds the safe maximum {}; clamping",
                endpoint_silence_ms,
                _MAX_MANUAL_ASR_ENDPOINT_SILENCE_MS,
            )
            endpoint_silence_ms = _MAX_MANUAL_ASR_ENDPOINT_SILENCE_MS
        publication_timeout_secs = (
            parse_env_float(
                "REALTIME_INPUT_TRANSCRIPTION_TIMEOUT_SECONDS",
                8.0,
                min_value=1.0,
            )
            + 1.0
        )

        async def _commit_manual_input(pcm: bytes, sample_rate: int) -> None:
            """Deliver one complete manual turn in strict pipeline order."""
            nonlocal manual_input_sample_cursor
            if manual_commit_publications is None:
                raise RuntimeError("Manual commit publication coordinator is missing")
            if len(pcm) % 2:
                raise RuntimeError("Manual pipeline PCM must contain complete 16-bit samples")
            turn_start_cursor = manual_input_sample_cursor
            turn_stop_cursor = turn_start_cursor + len(pcm) // 2
            published = manual_commit_publications.expect()
            try:
                await input_transport.push_frame(
                    RealtimeManualUserStartedSpeakingFrame(
                        audio_sample_cursor=turn_start_cursor,
                        sample_rate=sample_rate,
                    )
                )
                await input_transport.push_frame(
                    InputAudioRawFrame(
                        audio=pcm,
                        sample_rate=sample_rate,
                        num_channels=1,
                    )
                )
                endpoint_samples = max(1, round(sample_rate * endpoint_silence_ms / 1000))
                await input_transport.push_frame(
                    RealtimeASREndpointSilenceFrame(
                        audio=b"\x00\x00" * endpoint_samples,
                        sample_rate=sample_rate,
                        num_channels=1,
                    )
                )
                await input_transport.push_frame(
                    RealtimeManualUserStoppedSpeakingFrame(
                        audio_sample_cursor=turn_stop_cursor,
                        sample_rate=sample_rate,
                    )
                )
                manual_input_sample_cursor = turn_stop_cursor
                await asyncio.wait_for(
                    asyncio.shield(published),
                    timeout=publication_timeout_secs,
                )
            finally:
                manual_commit_publications.discard(published)

        serializer.set_manual_input_handlers(
            commit_hook=_commit_manual_input,
        )

    @transport.output().event_handler("on_before_process_frame")
    async def _arm_output_interruption(_output, frame) -> None:
        # This synchronous processor hook runs before the output transport can
        # race its paced audio task. The serializer acknowledges the same ID
        # only after Pipecat has reset that transport queue.
        if isinstance(frame, InterruptionFrame):
            serializer.cancel_output_audio(frame.id)

    _CONTEXTS[transport] = _RealtimeTransportContext(
        controller=controller,
        serializer=serializer,
        client_tool_broker=client_tool_broker,
        fail_input_audio_turn=_fail_input_audio_turn,
        wire_sequencer=wire_sequencer,
        response_gate=response_gate,
        mcp_runtime=mcp_runtime,
        idle_timeout=idle_timeout,
        manual_commit_publications=manual_commit_publications,
    )

    @transport.event_handler("on_client_disconnected")
    async def _shutdown_realtime_on_disconnect(_transport, _client) -> None:  # noqa: ARG001
        shutdown_realtime_transport(transport)

    return transport


def realtime_controller(transport: Any) -> RealtimeSessionController | None:
    """Return the controller for a Realtime-backed transport."""
    context = _CONTEXTS.get(transport)
    return context.controller if context is not None else None


def realtime_lifecycle_observer(transport: Any) -> BaseObserver | None:
    """Create or return the connection's canonical lifecycle observer."""
    context = _CONTEXTS.get(transport)
    if context is None or context.serializer.emit is None or context.serializer.emit_batch is None:
        return None
    if context.observer is None:

        async def _wait_client_tool_context(call_id: str) -> None:
            await context.client_tool_broker.wait_context_applied(
                call_id,
                timeout=parse_env_int(
                    "REALTIME_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS",
                    10,
                    min_value=1,
                ),
            )

        context.observer = RealtimeLifecycleObserver(
            emit=context.serializer.emit,
            emit_batch=context.serializer.emit_batch,
            controller=context.controller,
            flush_output_audio=context.serializer.flush_output_audio,
            reset_output_audio=context.serializer.reset_output_audio,
            cancel_output_audio=context.serializer.cancel_output_audio,
            fail_input_audio_turn=context.fail_input_audio_turn,
            wait_client_tool_context=_wait_client_tool_context,
            mcp_runtime=context.mcp_runtime,
            input_transcription_timeout_secs=parse_env_float(
                "REALTIME_INPUT_TRANSCRIPTION_TIMEOUT_SECONDS",
                8.0,
                min_value=1.0,
            ),
        )
        context.serializer.set_on_response_cancel(context.observer.on_response_cancelled)
        context.serializer.set_output_audio_failure_handler(context.observer.on_output_audio_failure)
    return context.observer


def configure_realtime_client_tools(
    transport: Any,
    llm: LLMService,
    raw_client_tools: object,
    *,
    trusted_tools: ToolsSchema | None = None,
    trusted_tool_names: Iterable[str] = (),
) -> ToolsSchema | None:
    """Configure the connection-owned client-tool provider projection."""
    context = _CONTEXTS.get(transport)
    if context is None:
        raise RuntimeError("Client-owned tools require a Realtime transport")
    return context.mcp_runtime.client_tool_projection.configure(
        llm,
        raw_client_tools,
        client_tool_handler=context.client_tool_broker.handle,
        trusted_tools=trusted_tools,
        trusted_tool_names=trusted_tool_names,
    )


async def prepare_realtime_tools(transport: Any, llm: LLMService) -> tuple[ToolsSchema | None, Any]:
    """Prepare native session tools for the local LLM boundary."""
    context = _CONTEXTS.get(transport)
    if context is None:
        raise RuntimeError("Realtime tools require a Realtime transport")
    prepared: MCPPreparedTools = await context.mcp_runtime.prepare_session(llm)
    schema = project_function_tools(prepared.pipeline_tools) if prepared.pipeline_tools else None
    return schema, prepared.pipeline_tool_choice


def bind_realtime_context(
    transport: Any,
    llm_context: Any,
    *,
    render_instructions: RealtimeInstructionsRenderer | None = None,
) -> None:
    """Bind the shared context and its public-to-provider instruction renderer."""
    context = _CONTEXTS.get(transport)
    if context is not None:
        context.serializer.bind_context(
            llm_context,
            instructions_renderer=render_instructions,
        )
        context.client_tool_broker.bind_context(llm_context)


def bind_realtime_tts_service(transport: Any, tts: Any) -> None:
    """Bind the exact linked TTS service for response-local voice updates."""
    context = _CONTEXTS.get(transport)
    if context is not None:
        context.response_gate.bind_tts_service(tts)


def bind_realtime_service_response_snapshots(transport: Any, service: Any) -> None:
    """Bind fused/service-initiated LLM runs to the canonical response gate."""
    context = _CONTEXTS.get(transport)
    if context is None:
        return
    binder = getattr(service, "bind_realtime_response_snapshot", None)
    if not callable(binder):
        raise TypeError("Realtime service response snapshots require a compatible LLM service")
    binder(
        context.response_gate.prepare_service_response_snapshot,
        reserve_audio_response=context.response_gate.reserve_service_audio_response,
        release_audio_response=context.response_gate.release_service_audio_response,
    )


def bind_realtime_deferred_service_responses(transport: Any, service: Any) -> None:
    """Bind durable precomputed service output to an owned Realtime response."""
    context = _CONTEXTS.get(transport)
    if context is None:
        return
    binder = getattr(service, "bind_realtime_deferred_response_snapshot", None)
    if not callable(binder):
        raise TypeError("Realtime deferred responses require a compatible service")
    binder(context.response_gate.prepare_deferred_service_response_snapshot)


def bind_realtime_assistant_context_message(transport: Any, turn_message: Any) -> bool:
    """Own the assistant context object just appended by its aggregator."""
    context = _CONTEXTS.get(transport)
    if context is None or turn_message is None:
        return False
    return context.serializer.bind_latest_assistant_context_message()


def realtime_manual_user_turn_strategies(transport: Any) -> UserTurnStrategies | None:
    """Return strict external turn strategies for a manual Realtime session."""
    context = _CONTEXTS.get(transport)
    if context is None or not context.controller.manual_input_mode:
        return None
    return UserTurnStrategies(
        start=[ExternalUserTurnStartStrategy()],
        stop=[RealtimeManualTurnStopStrategy(response_gate=context.response_gate)],
    )


def bind_realtime_automatic_response_provenance(
    transport: Any,
    strategies: list[BaseUserTurnStopStrategy],
) -> None:
    """Mark only contexts emitted by server-managed user-turn inference.

    Stop-strategy events are synchronous and handlers run in registration
    order. Pipeline builders call this before constructing the user aggregator,
    so the marker is installed before Pipecat's own handler pushes its context.
    The marker distinguishes a new user turn from an internal tool continuation
    whether automatic responses are currently enabled or disabled. That keeps
    response policy tied to typed pipeline provenance rather than conversation
    contents.
    """
    context = _CONTEXTS.get(transport)
    if context is None:
        return

    def _mark_automatic_response(_strategy: BaseUserTurnStopStrategy) -> None:
        context.response_gate.register_automatic_response_context()

    for strategy in strategies:
        strategy.add_event_handler(
            "on_user_turn_inference_triggered",
            _mark_automatic_response,
        )


def realtime_response_gate_processors(transport: Any) -> list[FrameProcessor]:
    """Return response coordination processors placed immediately before the LLM."""
    context = _CONTEXTS.get(transport)
    if context is None:
        return []
    context.serializer.set_response_gate(context.response_gate)
    return [context.response_gate]


def realtime_tool_result_processors(transport: Any) -> list[FrameProcessor]:
    """Return Realtime-only result policy processors placed after the LLM."""
    context = _CONTEXTS.get(transport)
    if context is None:
        return []
    return [RealtimeToolResultPolicy(controller=context.controller)]


def shutdown_realtime_transport(transport: Any) -> None:
    """Release controller/observer state when a Realtime socket disconnects."""
    context = _CONTEXTS.pop(transport, None)
    if context is not None:
        context.serializer.notify_connection_closed()
        context.wire_sequencer.reset()
        if context.manual_commit_publications is not None:
            context.manual_commit_publications.reset()
        context.serializer.reset_output_audio()
        if context.observer is not None:
            context.observer.shutdown()
