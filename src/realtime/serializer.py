# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Pipecat serializer for the OpenAI Realtime-compatible WebSocket."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMRunFrame,
    OutputAudioRawFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

from realtime.audio import (
    DEFAULT_CLIENT_PCM_RATE,
    MAX_PENDING_INPUT_BYTES,
    PIPELINE_OUTPUT_PCM_RATE,
    PIPELINE_PCM_RATE,
    SUPPORTED_G711_FORMAT_TYPES,
    AudioResampler,
    decode_base64_audio,
    encode_base64_audio,
    extract_client_input_format_type,
    extract_client_output_format_type,
    extract_client_output_pcm_rate,
    extract_client_pcm_rate,
    max_base64_audio_chars,
    max_pending_input_bytes,
)
from realtime.client_tools import ClientToolBroker
from realtime.controller import RealtimeSessionController
from realtime.events import EmitBatchFn, EmitFn
from realtime.frames import (
    RealtimeClientToolOutputFrame,
    RealtimeConversationAppendFrame,
    RealtimeDeferredResponseCreateFrame,
    RealtimeResponseCreateFrame,
)
from realtime.idle_timeout import RealtimeServerVADIdleTimeout
from realtime.lifecycle import emit_events
from realtime.mcp import MCPPreparedTools, MCPToolPreparationPoisonedError, RealtimeMCPRuntime
from realtime.protocol import (
    MAX_AUDIO_APPEND_BYTES,
    MAX_REALTIME_EVENT_BYTES,
    RealtimeProtocolError,
    build_server_event,
    invalid_type,
    invalid_value,
    new_realtime_id,
    strict_json_loads,
    validate_client_event_id,
)
from realtime.response_config import (
    RealtimeResponseInput,
    project_response_input_messages,
    validate_response_conversation,
    validate_response_input,
    validate_response_output_modalities,
)
from realtime.session import project_tool_choice_to_pipeline, validate_session_tools_bounds
from realtime.tool_projection import project_function_tools
from utils import parse_env_int

CancelHook = Callable[[], None]
ManualInputCommitHook = Callable[[bytes, int], Awaitable[None]]
OutputAudioFailureHook = Callable[[str, RealtimeProtocolError], Awaitable[None]]
ConnectionFailureHook = Callable[[str], Awaitable[None]]
StateTransitionHook = Callable[[], Awaitable[None]]

_LIVE_SESSION_FIELDS = frozenset(
    {
        "audio",
        "instructions",
        "max_output_tokens",
        "model",
        "output_modalities",
        "parallel_tool_calls",
        "tool_choice",
        "tools",
        "truncation",
        "type",
    }
)
_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS = parse_env_int(
    "REALTIME_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS",
    10,
    min_value=1,
)
_CONTEXT_APPLY_TIMEOUT_SECS = parse_env_int(
    "REALTIME_CONTEXT_APPLY_TIMEOUT_SECONDS",
    10,
    min_value=1,
)


@dataclass(frozen=True, slots=True)
class _SessionContextState:
    """Exact Pipecat context defaults owned by one session transaction."""

    messages_changed: bool
    messages: tuple[Any, ...]
    tools_changed: bool
    tools: Any
    tool_choice: Any
    prompt: list[dict[str, Any]] | None


@dataclass(frozen=True, slots=True)
class _ConversationContextState:
    """Exact context and ownership metadata for an item mutation rollback."""

    messages: tuple[Any, ...] | None
    message_owners: dict[str, dict[str, Any]]
    applied_events: dict[str, asyncio.Event]
    cleared_item_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _PendingConversationAppend:
    """A provisional text item waiting for context and wire publication."""

    rollback_state: Any
    provisional_state: Any
    publication_done: asyncio.Event


class RealtimeFrameSerializer(FrameSerializer):
    """Convert canonical Realtime events to and from Pipecat frames."""

    def __init__(
        self,
        *,
        controller: RealtimeSessionController,
        params: FrameSerializer.InputParams | None = None,
    ) -> None:
        """Bind audio serialization and client events to one controller."""
        super().__init__(params or FrameSerializer.InputParams())
        self._controller = controller
        self._resampler = AudioResampler()
        self._pipeline_in_rate = PIPELINE_PCM_RATE
        self._pipeline_out_rate = PIPELINE_OUTPUT_PCM_RATE
        self._client_in_format = extract_client_input_format_type(controller.public_session())
        self._client_out_format = extract_client_output_format_type(controller.public_session())
        self._client_in_rate = extract_client_pcm_rate(controller.public_session())
        self._client_out_rate = extract_client_output_pcm_rate(controller.public_session())
        self._emit: EmitFn | None = None
        self._emit_batch: EmitBatchFn | None = None
        self._on_response_cancel: CancelHook | None = None
        self._output_audio_failure_handler: OutputAudioFailureHook | None = None
        self._connection_failure_handler: ConnectionFailureHook | None = None
        self._client_tool_broker: ClientToolBroker | None = None
        self._mcp_runtime: RealtimeMCPRuntime | None = None
        self._response_done_events: dict[str, asyncio.Event] = {}
        self._pending_output_interruptions: set[int] = set()
        self._acknowledged_output_interruptions: set[int] = set()
        self._output_interruption_history: deque[int] = deque()
        self._output_audio_generation = 0
        self._manual_input_audio = bytearray()
        self._manual_input_commit_hook: ManualInputCommitHook | None = None
        self._idle_timeout: RealtimeServerVADIdleTimeout | None = None
        self._response_gate: Any | None = None
        self._llm_context: Any | None = None
        self._instructions_renderer: Callable[[str], list[dict[str, Any]]] | None = None
        self._context_messages_by_item_id: dict[str, dict[str, Any]] = {}
        self._context_cleared_item_ids: set[str] = set()
        self._context_applied_events: dict[str, asyncio.Event] = {}
        self._pending_conversation_appends: dict[str, _PendingConversationAppend] = {}
        self._connection_closed = False
        self._failure_close_requested = False
        self._event_state_lock = asyncio.Lock()

    @property
    def controller(self) -> RealtimeSessionController:
        """Return the single state owner shared with the lifecycle observer."""
        return self._controller

    @property
    def emit(self) -> EmitFn | None:
        """Return the registered serialized WebSocket emitter."""
        return self._emit

    @property
    def emit_batch(self) -> EmitBatchFn | None:
        """Return the registered atomic WebSocket batch emitter."""
        return self._emit_batch

    @property
    def connection_closed(self) -> bool:
        """Return whether the owning WebSocket has terminated."""
        return self._connection_closed

    def set_emit(self, emit: EmitFn, emit_batch: EmitBatchFn) -> None:
        """Register the connection's single-event and atomic-batch emitters."""
        self._emit = emit
        self._emit_batch = emit_batch

    def set_on_response_cancel(self, hook: CancelHook | None) -> None:
        """Register a hook that drains observer text on explicit cancellation."""
        self._on_response_cancel = hook

    def set_output_audio_failure_handler(self, hook: OutputAudioFailureHook | None) -> None:
        """Route output conversion failures to the response lifecycle owner."""
        self._output_audio_failure_handler = hook

    def set_client_tool_broker(self, broker: ClientToolBroker) -> None:
        """Bind the connection-owned bridge for application-executed tools."""
        self._client_tool_broker = broker

    def set_mcp_runtime(self, runtime: RealtimeMCPRuntime) -> None:
        """Bind the connection-owned native MCP host."""
        self._mcp_runtime = runtime

    def set_idle_timeout(self, idle_timeout: RealtimeServerVADIdleTimeout) -> None:
        """Bind the connection-owned server-VAD idle timeout coordinator."""
        self._idle_timeout = idle_timeout

    def set_connection_failure_handler(self, handler: ConnectionFailureHook) -> None:
        """Bind the transport close used after an unrecoverable state divergence."""
        self._connection_failure_handler = handler

    async def run_connection_state_transition(self, transition: StateTransitionHook) -> None:
        """Serialize a server transition with concurrent client event dispatch."""
        async with self._event_state_lock:
            if self._connection_closed:
                return
            await transition()

    def bind_conversation_context_message(self, item_id: str, context_message: dict[str, Any]) -> None:
        """Own an exact context object created by a server-side conversation turn."""
        if self._llm_context is None:
            raise RuntimeError("Realtime conversation mutation has no bound LLM context")
        if item_id in self._context_messages_by_item_id:
            raise RuntimeError(f"Realtime conversation item {item_id!r} already owns a context message")
        messages = self._llm_context.get_messages()
        if sum(message is context_message for message in messages) != 1:
            raise RuntimeError("The server-created context message was not applied exactly once")
        self._context_messages_by_item_id[item_id] = context_message
        self._context_applied_events.setdefault(item_id, asyncio.Event()).set()

    def commit_server_conversation_context_message(
        self,
        item_id: str,
        context_message: dict[str, Any],
    ) -> None:
        """Append and own one server-created message at the pipeline context boundary."""
        if self._llm_context is None:
            raise RuntimeError("Realtime conversation mutation has no bound LLM context")
        snapshot = self._snapshot_conversation_context()
        try:
            self._llm_context.add_message(context_message)
            self.bind_conversation_context_message(item_id, context_message)
        except BaseException as exc:
            failures = self._restore_conversation_context(snapshot)
            if failures:
                self.notify_connection_closed()
                raise RuntimeError("Realtime conversation context could not be restored; connection retired") from exc
            raise

    async def wait_for_conversation_context_message(self, item_id: str) -> None:
        """Wait until one published message has its exact canonical context owner."""
        item = self._controller.conversation.item(item_id)
        await self._wait_message_context_if_pending(item_id, item)
        if item_id not in self._context_messages_by_item_id:
            raise RuntimeError(f"Realtime conversation item {item_id!r} has no canonical context owner")

    @property
    def has_pending_conversation_appends(self) -> bool:
        """Return whether accepted client text is waiting at the context boundary."""
        return bool(self._pending_conversation_appends)

    def bind_response_gate(self, response_gate: Any) -> None:
        """Bind response coordination without claiming pipeline placement."""
        if self._llm_context is not None:
            response_gate.bind_context(
                self._llm_context,
                instructions_renderer=self._instructions_renderer,
                session_instructions=self._controller.session.public_view()["instructions"],
            )
        response_gate.set_client_tool_output_handler(self._finalize_client_tool_output)
        response_gate.set_conversation_append_handler(self._finalize_conversation_append)
        response_gate.set_deferred_response_handler(self._activate_deferred_response)
        response_gate.set_pipeline_response_output_handler(
            self.activate_response_output_audio,
            self._restore_session_output_audio,
        )
        self._response_gate = response_gate

    def set_response_gate(self, response_gate: Any) -> None:
        """Bind a response gate that has been inserted into the pipeline."""
        response_gate.install()
        self.bind_response_gate(response_gate)

    def set_manual_input_handlers(
        self,
        *,
        commit_hook: ManualInputCommitHook,
        response_gate: Any | None = None,
    ) -> None:
        """Bind manual input delivery and its no-auto-response context gate."""
        self._manual_input_commit_hook = commit_hook
        if response_gate is not None:
            self.set_response_gate(response_gate)

    def bind_context(
        self,
        context: Any,
        *,
        instructions_renderer: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        """Bind the exact shared Pipecat context used by this connection."""
        if not all(
            hasattr(context, attribute)
            for attribute in (
                "get_messages",
                "set_messages",
                "set_tool_choice",
                "set_tools",
                "tool_choice",
                "tools",
            )
        ):
            raise TypeError("Realtime conversation mutation requires an LLMContext-compatible object")
        self._llm_context = context
        self._instructions_renderer = instructions_renderer
        if self._response_gate is not None:
            self._response_gate.bind_context(
                context,
                instructions_renderer=instructions_renderer,
                session_instructions=self._controller.session.public_view()["instructions"],
            )

    def bind_latest_assistant_context_message(self) -> bool:
        """Own the exact context object appended for the latest assistant turn."""
        if self._llm_context is None:
            return False
        item_id = self._controller.assistant_item_id or self._controller.last_assistant_item_id
        if item_id is None or item_id in self._context_messages_by_item_id:
            return item_id is not None
        try:
            item = self._controller.conversation.item(item_id)
        except RealtimeProtocolError:
            return False
        if item.get("type") != "message" or item.get("role") != "assistant":
            return False
        messages = self._llm_context.get_messages()
        if not messages:
            return False
        context_message = messages[-1]
        if not isinstance(context_message, dict) or context_message.get("role") != "assistant":
            return False
        if context_message.get("tool_calls"):
            return False
        self._context_messages_by_item_id[item_id] = context_message
        self._context_applied_events.setdefault(item_id, asyncio.Event()).set()
        return True

    def prepare_response_done_publication(self, response_id: str) -> None:
        """Arm the context barrier before a terminal response can reach its client."""
        assistant_item_id = self._controller.last_assistant_item_id
        if assistant_item_id is not None and assistant_item_id not in self._context_messages_by_item_id:
            self._context_applied_events.setdefault(assistant_item_id, asyncio.Event())

    def notify_response_done_published(self, response_id: str) -> None:
        """Release staged client outputs after response.done reaches the wire."""
        self.prepare_response_done_publication(response_id)
        self._controller.mark_response_done_published(response_id)
        self._restore_session_output_audio()
        if self._response_gate is not None:
            self._response_gate.finish_response(response_id)
        terminal_event = self._response_done_events.pop(response_id, None)
        if terminal_event is not None:
            terminal_event.set()

    def notify_connection_closed(self) -> None:
        """Release every wire-publication wait when the socket is gone."""
        if self._connection_closed:
            return
        self._connection_closed = True
        response_id = self._controller.active_response_id
        if response_id is not None:
            self._controller.abandon_response(response_id, reason="connection_closed")
        self._restore_session_output_audio()
        if self._response_gate is not None:
            self._response_gate.reset()
        if self._client_tool_broker is not None:
            self._client_tool_broker.shutdown()
        if self._mcp_runtime is not None:
            self._mcp_runtime.shutdown()
        if self._idle_timeout is not None:
            self._idle_timeout.close()
        terminal_events = tuple(self._response_done_events.values())
        self._response_done_events.clear()
        for terminal_event in terminal_events:
            terminal_event.set()
        for context_event in self._context_applied_events.values():
            context_event.set()
        for transaction in self._pending_conversation_appends.values():
            transaction.publication_done.set()
        self._context_applied_events.clear()
        self._context_messages_by_item_id.clear()
        self._context_cleared_item_ids.clear()
        self._pending_conversation_appends.clear()

    async def setup(self, frame: StartFrame) -> None:
        """Capture the pipeline sample rate supplied by Pipecat."""
        await super().setup(frame)
        if getattr(frame, "audio_in_sample_rate", None):
            self._pipeline_in_rate = int(frame.audio_in_sample_rate)
        if getattr(frame, "audio_out_sample_rate", None):
            self._pipeline_out_rate = int(frame.audio_out_sample_rate)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        """Serialize outbound audio as canonical delta events."""
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, InterruptionFrame):
            # FastAPIWebsocketOutputTransport serializes this only after its
            # media sender has cancelled the paced audio task and reset the
            # interruptible queue. That real transport boundary, rather than a
            # changing controller response ID, is what makes later PCM safe.
            self.acknowledge_output_interruption(frame.id)
            return None
        if isinstance(frame, OutputAudioRawFrame):
            await self._serialize_output_audio(frame)
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        """Validate and dispatch one canonical client event atomically."""
        async with self._event_state_lock:
            if self._connection_closed:
                return None
            return await self._deserialize_event(data)

    async def _deserialize_event(self, data: str | bytes) -> Frame | None:
        """Dispatch one client event while owning the connection state lock."""
        echo_id: str | None = None
        try:
            message = self._decode_message(data)
            event_type = message.get("type")
            event_id = validate_client_event_id(message.get("event_id"))
            echo_id = event_id
            if not isinstance(event_type, str) or not event_type:
                raise RealtimeProtocolError(
                    message="Missing event type",
                    code="missing_type",
                    param="type",
                    event_id=echo_id,
                )
            return await self._dispatch(message, event_type=event_type, event_id=echo_id)
        except RealtimeProtocolError as exc:
            event = exc.with_event_id(echo_id).to_event()
            if self._connection_closed:
                await self._publish_failure_and_close(event)
            else:
                await self._emit_event(event)
            return None
        except Exception:
            logger.exception("Unexpected Realtime client-event failure")
            event = RealtimeProtocolError(
                message="The Realtime event could not be processed",
                code="event_processing_failed",
                event_id=echo_id,
                error_type="server_error",
            ).to_event()
            if self._connection_closed:
                await self._publish_failure_and_close(event)
            else:
                await self._emit_event(event)
            return None

    @staticmethod
    def _decode_message(data: str | bytes) -> dict[str, Any]:
        if len(data) > MAX_REALTIME_EVENT_BYTES:
            raise RealtimeProtocolError(
                message="Realtime client event exceeds the WebSocket message limit",
                code="event_too_large",
            )
        if isinstance(data, bytes):
            try:
                data = data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RealtimeProtocolError(
                    message="Binary WebSocket messages must contain UTF-8 JSON",
                    code="invalid_frame",
                ) from exc
        elif len(data.encode("utf-8")) > MAX_REALTIME_EVENT_BYTES:
            raise RealtimeProtocolError(
                message="Realtime client event exceeds the WebSocket message limit",
                code="event_too_large",
            )
        try:
            message = strict_json_loads(data)
        except (json.JSONDecodeError, ValueError) as exc:
            raise RealtimeProtocolError(message="Invalid JSON", code="invalid_json") from exc
        if not isinstance(message, dict):
            raise RealtimeProtocolError(message="Event must be a JSON object", code="invalid_event")
        return message

    async def _dispatch(
        self,
        message: dict[str, Any],
        *,
        event_type: str,
        event_id: str | None,
    ) -> Frame | None:
        if event_type == "input_audio_buffer.append":
            return await self._deserialize_append(message)
        if event_type == "input_audio_buffer.commit":
            return await self._deserialize_input_audio_commit(message)
        if event_type == "input_audio_buffer.clear":
            return await self._deserialize_input_audio_clear(message)
        if event_type == "session.update":
            return await self._deserialize_session_update(message)
        if event_type == "response.create":
            return await self._deserialize_response_create(message)
        if event_type == "response.cancel":
            return await self._deserialize_response_cancel(message)
        if event_type == "conversation.item.create":
            return await self._deserialize_item_create(message)
        if event_type == "conversation.item.delete":
            await self._deserialize_item_delete(message)
            return None
        if event_type == "conversation.item.retrieve":
            await self._deserialize_item_retrieve(message)
            return None
        if event_type == "conversation.item.truncate":
            await self._deserialize_item_truncate(message)
            return None
        if event_type == "output_audio_buffer.clear":
            raise RealtimeProtocolError(
                message="output_audio_buffer.clear is available only on server-managed WebRTC output buffers",
                code="unsupported_capability",
                param="type",
                event_id=event_id,
            )
        raise RealtimeProtocolError(
            message=f"Event type {event_type!r} is not available on this endpoint",
            code="unsupported_event",
            param="type",
            event_id=event_id,
        )

    async def _ensure_output_voice_capabilities(self, *, param: str) -> None:
        """Resolve the trusted TTS voice catalog before accepting audio output."""
        try:
            await self._controller.ensure_output_voice_capabilities()
        except Exception as exc:
            raise RealtimeProtocolError(
                message="The selected Realtime audio output service is not ready",
                code="services_not_ready",
                param=param,
                error_type="server_error",
            ) from exc

    def _snapshot_session_context(
        self,
        *,
        instructions_changed: bool,
        tools_changed: bool,
    ) -> _SessionContextState | None:
        if not instructions_changed and not tools_changed:
            return None
        if self._llm_context is None:
            raise RuntimeError("Realtime session update has no bound LLM context")
        prompt = self._response_gate.snapshot_session_prompt() if instructions_changed else None
        return _SessionContextState(
            messages_changed=instructions_changed,
            messages=tuple(self._llm_context.get_messages()) if instructions_changed else (),
            tools_changed=tools_changed,
            tools=self._llm_context.tools if tools_changed else None,
            tool_choice=self._llm_context.tool_choice if tools_changed else None,
            prompt=prompt,
        )

    def _restore_session_context(self, snapshot: _SessionContextState | None) -> list[Exception]:
        if snapshot is None:
            return []
        failures: list[Exception] = []
        if snapshot.messages_changed:
            try:
                self._llm_context.set_messages(list(snapshot.messages))
            except Exception as exc:
                failures.append(exc)
            try:
                self._response_gate.restore_session_prompt(snapshot.prompt or [])
            except Exception as exc:
                failures.append(exc)
        if snapshot.tools_changed:
            try:
                self._llm_context.set_tools(snapshot.tools)
            except Exception as exc:
                failures.append(exc)
            try:
                self._llm_context.set_tool_choice(snapshot.tool_choice)
            except Exception as exc:
                failures.append(exc)
        return failures

    def _snapshot_conversation_context(self) -> _ConversationContextState:
        """Capture context objects and serializer ownership maps without cloning them."""
        messages = tuple(self._llm_context.get_messages()) if self._llm_context is not None else None
        return _ConversationContextState(
            messages=messages,
            message_owners=dict(self._context_messages_by_item_id),
            applied_events=dict(self._context_applied_events),
            cleared_item_ids=frozenset(self._context_cleared_item_ids),
        )

    def _restore_conversation_context(self, snapshot: _ConversationContextState) -> list[Exception]:
        """Restore exact message identities and ownership metadata after rejection."""
        failures: list[Exception] = []
        if snapshot.messages is not None:
            if self._llm_context is None:
                failures.append(RuntimeError("Realtime conversation rollback lost its LLM context"))
            else:
                try:
                    self._llm_context.set_messages(list(snapshot.messages))
                except Exception as exc:
                    failures.append(exc)
        self._context_messages_by_item_id = dict(snapshot.message_owners)
        self._context_applied_events = dict(snapshot.applied_events)
        self._context_cleared_item_ids = set(snapshot.cleared_item_ids)
        return failures

    def _apply_conversation_ownership(
        self,
        *,
        message_owners: dict[str, dict[str, Any]],
        applied_events: dict[str, asyncio.Event],
        cleared_item_ids: set[str],
    ) -> None:
        """Publish one fully prepared serializer-side ownership projection."""
        owners = dict(message_owners)
        events = dict(applied_events)
        cleared = set(cleared_item_ids)
        self._context_messages_by_item_id = owners
        self._context_applied_events = events
        self._context_cleared_item_ids = cleared

    def _restore_conversation_transaction(
        self,
        *,
        controller_snapshot: Any,
        context_snapshot: _ConversationContextState,
        cause: BaseException,
        expected_controller_state: Any | None = None,
    ) -> None:
        """Roll back every item owner or retire a connection with uncertain state."""
        failures: list[Exception] = []
        try:
            if (
                expected_controller_state is not None
                and self._controller.snapshot_conversation_mutation_state() != expected_controller_state
            ):
                raise RuntimeError("Realtime conversation changed after the provisional item mutation")
            self._controller.restore_conversation_mutation_state(controller_snapshot)
        except Exception as exc:
            failures.append(exc)
        failures.extend(self._restore_conversation_context(context_snapshot))
        if failures:
            logger.error(
                "Realtime conversation rollback failed owners={}",
                [type(failure).__name__ for failure in failures],
            )
            self.notify_connection_closed()
            raise RuntimeError("Realtime conversation state could not be restored; connection retired") from cause

    async def _wait_pending_conversation_publications(self) -> None:
        """Keep later journal operations behind earlier provisional item acknowledgements."""
        pending = tuple(self._pending_conversation_appends.values())
        for transaction in pending:
            try:
                await asyncio.wait_for(
                    transaction.publication_done.wait(),
                    timeout=_CONTEXT_APPLY_TIMEOUT_SECS,
                )
            except TimeoutError as exc:
                self.notify_connection_closed()
                raise RealtimeProtocolError(
                    message="A preceding conversation item did not reach its publication boundary",
                    code="conversation_context_apply_timeout",
                    param="item",
                    error_type="server_error",
                ) from exc
        if self._connection_closed:
            raise RealtimeProtocolError(
                message="The Realtime connection closed before a preceding conversation item was published",
                code="conversation_context_apply_cancelled",
                param="item",
                error_type="server_error",
            )

    async def _deserialize_session_update(self, message: dict[str, Any]) -> Frame | None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "session", "type"}))
        patch = message.get("session")
        if not isinstance(patch, dict):
            raise invalid_type("session.update requires a session object", param="session")
        if "tools" in patch:
            validate_session_tools_bounds(patch["tools"])

        self._controller.session.preflight_update_without_output_voice_catalog(patch)
        if self._controller.session_update_requires_output_voice_resolution(patch):
            await self._ensure_output_voice_capabilities(param="session.audio.output.voice")

        current = self._controller.session.public_view()
        trial_session = copy.deepcopy(self._controller.session)
        trial_session.apply_update(patch)
        candidate = trial_session.public_view()
        unsupported = [
            field
            for field in sorted(patch)
            if field not in _LIVE_SESSION_FIELDS and candidate.get(field) != current.get(field)
        ]
        if unsupported:
            field = unsupported[0]
            raise RealtimeProtocolError(
                message=(
                    f"session.{field} is fixed for the active cascaded pipeline; "
                    "set it in the initial session.update before the first turn"
                ),
                code="unsupported_live_session_update",
                param=f"session.{field}",
            )
        if candidate["audio"]["input"].get("transcription") != current["audio"]["input"].get("transcription"):
            configured = candidate["audio"]["input"].get("transcription")
            if isinstance(configured, dict):
                fixed_model = self._controller.runtime_config.get("realtime_input_transcription_model")
                fixed_language = self._controller.runtime_config.get("asr_language_code")
                if isinstance(fixed_model, str) and configured.get("model") != fixed_model:
                    raise RealtimeProtocolError(
                        message="The cascaded ASR model cannot change after the pipeline starts",
                        code="unsupported_live_session_update",
                        param="session.audio.input.transcription.model",
                    )
                if "language" in configured and configured.get("language") != fixed_language:
                    raise RealtimeProtocolError(
                        message="The cascaded ASR language cannot change after the pipeline starts",
                        code="unsupported_live_session_update",
                        param="session.audio.input.transcription.language",
                    )
        current_turn_detection = current["audio"]["input"].get("turn_detection")
        candidate_turn_detection = candidate["audio"]["input"].get("turn_detection")
        if candidate_turn_detection != current_turn_detection:
            raise RealtimeProtocolError(
                message="session.audio.input.turn_detection is fixed when the cascaded pipeline starts",
                code="unsupported_live_session_update",
                param="session.audio.input.turn_detection",
            )
        if self._manual_input_audio and candidate["audio"]["input"].get("format") != current["audio"]["input"].get(
            "format"
        ):
            raise RealtimeProtocolError(
                message="Input audio format cannot change while the manual input buffer is non-empty",
                code="input_audio_buffer_not_empty",
                param="session.audio.input.format",
            )
        tools_changed = candidate.get("tools") != current.get("tools")
        tool_choice_changed = candidate.get("tool_choice") != current.get("tool_choice")
        instructions_changed = candidate.get("instructions") != current.get("instructions")
        modalities_changed = candidate.get("output_modalities") != current.get("output_modalities")
        max_output_tokens_changed = candidate.get("max_output_tokens") != current.get("max_output_tokens")
        parallel_tool_calls_changed = candidate.get("parallel_tool_calls") != current.get("parallel_tool_calls")
        voice_changed = candidate["audio"]["output"].get("voice") != current["audio"]["output"].get("voice")
        response_defaults_changed = (
            modalities_changed or max_output_tokens_changed or parallel_tool_calls_changed or voice_changed
        )
        if response_defaults_changed:
            changed_param = (
                "session.output_modalities"
                if modalities_changed
                else "session.max_output_tokens"
                if max_output_tokens_changed
                else "session.parallel_tool_calls"
                if parallel_tool_calls_changed
                else "session.audio.output.voice"
            )
            if self._response_gate is None or not self._response_gate.installed:
                raise RealtimeProtocolError(
                    message="Live response defaults require the Realtime response gate",
                    code="response_gate_missing",
                    param=changed_param,
                    error_type="server_error",
                )
            if voice_changed or (modalities_changed and candidate.get("output_modalities") == ["audio"]):
                candidate_output = candidate["audio"]["output"]
                self._response_gate.validate_response_output(
                    output_modalities=["audio"],
                    audio_output=self._controller.session.validate_response_audio_output(
                        {"output": {name: copy.deepcopy(candidate_output[name]) for name in ("format", "voice")}}
                    ),
                    param_prefix="session",
                )
        prepared_tools = None
        rendered_instructions: list[dict[str, Any]] | None = None
        preparation_id: str | None = None
        if tools_changed or tool_choice_changed or instructions_changed:
            changed_param = (
                "session.tools"
                if tools_changed
                else "session.tool_choice"
                if tool_choice_changed
                else "session.instructions"
            )
            if self._llm_context is None:
                raise RealtimeProtocolError(
                    message="Live session updates are not bound to this Realtime pipeline",
                    code="session_runtime_missing",
                    param=changed_param,
                    error_type="server_error",
                )
            if self._response_gate is None or not self._response_gate.installed:
                raise RealtimeProtocolError(
                    message="Live session update coordination is not installed in this Realtime pipeline",
                    code="response_gate_missing",
                    param=changed_param,
                    error_type="server_error",
                )
            if (tools_changed or tool_choice_changed) and self._mcp_runtime is None:
                raise RealtimeProtocolError(
                    message="Live tool updates are not bound to this Realtime pipeline",
                    code="tool_runtime_missing",
                    param="session.tools" if tools_changed else "session.tool_choice",
                    error_type="server_error",
                )
            preparation_id = new_realtime_id("request")
            await self._response_gate.reserve_session_tool_preparation(preparation_id)
            try:
                if instructions_changed:
                    rendered_instructions = self._response_gate.prepare_session_instructions(candidate["instructions"])
                if tools_changed or tool_choice_changed:
                    prepared_tools = await self._mcp_runtime.prepare_session_update(
                        candidate.get("tools", []),
                        candidate.get("tool_choice", "auto"),
                    )
            except MCPToolPreparationPoisonedError as exc:
                self.notify_connection_closed()
                raise RealtimeProtocolError(
                    message="Realtime tool preparation could not be restored safely",
                    code="tool_preparation_state_inconsistent",
                    param="session.tools",
                    error_type="server_error",
                ) from exc
            except BaseException:
                await self._response_gate.release_session_tool_preparation(preparation_id)
                raise

        old_audio_state = (
            self._client_in_format,
            self._client_in_rate,
            self._client_out_format,
            self._client_out_rate,
            self._resampler,
            self._output_audio_generation,
        )
        prepared_tools_resolved = prepared_tools is None
        controller_snapshot = None
        context_snapshot: _SessionContextState | None = None
        transaction_committed = False
        try:
            prepared_runtime_config = copy.deepcopy(self._controller.runtime_config)
            projected_pipeline_tools = None
            if rendered_instructions is not None:
                prepared_runtime_config["prompt_content"] = candidate["instructions"]
            if prepared_tools is not None:
                projected_pipeline_tools = project_function_tools(prepared_tools.pipeline_tools)
                active_function_names = {
                    tool["name"]
                    for tool in candidate.get("tools", [])
                    if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str)
                }
                prepared_runtime_config.update(
                    {
                        "client_tools": [
                            copy.deepcopy(tool)
                            for tool in candidate.get("tools", [])
                            if isinstance(tool, dict)
                            and tool.get("type") == "function"
                            and tool.get("name") not in self._controller.server_tools
                            and tool.get("name") not in self._controller.delegate_tools
                        ],
                        "delegate_tools": sorted(self._controller.delegate_tools & active_function_names),
                        "mcp_tools": [
                            copy.deepcopy(tool)
                            for tool in candidate.get("tools", [])
                            if isinstance(tool, dict) and tool.get("type") == "mcp"
                        ],
                        "realtime_tool_choice": copy.deepcopy(candidate.get("tool_choice", "auto")),
                        "server_tools": sorted(self._controller.server_tools & active_function_names),
                        "tool_choice": copy.deepcopy(prepared_tools.pipeline_tool_choice),
                    }
                )

            async with self._controller.response_transition_lock:
                if voice_changed and self._response_gate.audio_response_slot_occupied:
                    raise RealtimeProtocolError(
                        message="session.audio.output.voice can change only between audio responses",
                        code="immutable_field",
                        param="session.audio.output.voice",
                    )

                preserve_active_output = (
                    self._controller.response_in_progress or self._controller.pipeline_response_pending
                )
                next_client_in_format = extract_client_input_format_type(candidate)
                next_client_in_rate = extract_client_pcm_rate(candidate)
                next_client_out_format = self._client_out_format
                next_client_out_rate = self._client_out_rate
                if not preserve_active_output:
                    next_client_out_format = extract_client_output_format_type(candidate)
                    next_client_out_rate = extract_client_output_pcm_rate(candidate)
                input_signature_changed = (self._client_in_format, self._client_in_rate) != (
                    next_client_in_format,
                    next_client_in_rate,
                )
                output_signature_changed = (self._client_out_format, self._client_out_rate) != (
                    next_client_out_format,
                    next_client_out_rate,
                )
                prepared_resampler = self._resampler
                if input_signature_changed or output_signature_changed:
                    prepared_resampler = self._resampler.for_format_transition(
                        reset_uplink=input_signature_changed,
                        reset_downlink=output_signature_changed,
                    )

                controller_snapshot = self._controller.snapshot_session_runtime_state()
                context_snapshot = self._snapshot_session_context(
                    instructions_changed=rendered_instructions is not None,
                    tools_changed=prepared_tools is not None,
                )
                if rendered_instructions is not None:
                    self._response_gate.commit_session_instructions(rendered_instructions)
                if prepared_tools is not None:
                    self._llm_context.set_tools(projected_pipeline_tools)
                    self._llm_context.set_tool_choice(copy.deepcopy(prepared_tools.pipeline_tool_choice))
                self._controller.apply_session_update(patch)
                if rendered_instructions is not None or prepared_tools is not None:
                    self._controller.bind_session_runtime_projection(
                        prepared_runtime_config,
                        client_tool_bindings=(
                            prepared_tools.client_tool_bindings if prepared_tools is not None else None
                        ),
                        mcp_pipeline_names=(prepared_tools.mcp_pipeline_names if prepared_tools is not None else None),
                    )
                self._client_in_format = next_client_in_format
                self._client_in_rate = next_client_in_rate
                self._client_out_format = next_client_out_format
                self._client_out_rate = next_client_out_rate
                self._resampler = prepared_resampler
                if output_signature_changed:
                    self._output_audio_generation += 1
                if prepared_tools is not None:
                    self._mcp_runtime.commit_session_update(prepared_tools)
                    prepared_tools_resolved = True
                transaction_committed = True
                try:
                    await self._emit_event(self._controller.session_updated_event())
                except BaseException:
                    self.notify_connection_closed()
                    raise
        except BaseException as transaction_error:
            if not transaction_committed:
                rollback_failures: list[Exception] = []
                (
                    self._client_in_format,
                    self._client_in_rate,
                    self._client_out_format,
                    self._client_out_rate,
                    self._resampler,
                    self._output_audio_generation,
                ) = old_audio_state
                if controller_snapshot is not None:
                    try:
                        self._controller.restore_session_runtime_state(controller_snapshot)
                    except Exception as exc:
                        rollback_failures.append(exc)
                rollback_failures.extend(self._restore_session_context(context_snapshot))
                if prepared_tools is not None and not prepared_tools_resolved:
                    try:
                        self._mcp_runtime.rollback_session_update(prepared_tools)
                        prepared_tools_resolved = True
                    except Exception as exc:
                        rollback_failures.append(exc)
                if rollback_failures:
                    logger.error(
                        "Realtime session rollback failed owners={}",
                        [type(failure).__name__ for failure in rollback_failures],
                    )
                    self.notify_connection_closed()
                    raise RuntimeError(
                        "Realtime session state could not be restored; connection retired"
                    ) from transaction_error
            raise
        finally:
            if preparation_id is not None:
                try:
                    await self._response_gate.release_session_tool_preparation(preparation_id)
                except BaseException:
                    if transaction_committed:
                        self.notify_connection_closed()
                    raise

        return None

    async def _deserialize_response_create(self, message: dict[str, Any]) -> Frame | None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "response", "type"}))
        response = message.get("response") if "response" in message else None
        metadata: dict[str, Any] | None = None
        response_tool_choice: str | dict[str, Any] | None = None
        response_tools: list[dict[str, Any]] | None = None
        response_instructions: str | None = None
        response_max_output_tokens: int | str | None = None
        response_parallel_tool_calls: bool | None = None
        response_input: RealtimeResponseInput | None = None
        response_input_messages: list[dict[str, Any]] | None = None
        response_output_modalities: list[str] | None = None
        response_audio_output: dict[str, Any] | None = None
        pipeline_response_tools: list[dict[str, Any]] | None = None
        projected_response_tool_choice: str | dict[str, Any] | None = None
        response_mcp_pipeline_names: frozenset[str] | None = None
        response_client_tool_bindings: dict[str, str] | None = None
        prepared_response_tools: MCPPreparedTools | None = None
        response_tool_preparation_admitted = False
        if response is not None:
            if not isinstance(response, dict):
                raise invalid_type("response.create.response must be an object or null", param="response")
            unsupported = sorted(
                set(response)
                - {
                    "audio",
                    "conversation",
                    "input",
                    "instructions",
                    "max_output_tokens",
                    "metadata",
                    "output_modalities",
                    "parallel_tool_calls",
                    "tool_choice",
                    "tools",
                }
            )
            if unsupported:
                field = unsupported[0]
                raise RealtimeProtocolError(
                    message=(
                        f"response.{field} overrides are not available on the cascaded pipeline; "
                        "configure the initial session instead"
                    ),
                    code="unsupported_response_override",
                    param=f"response.{field}",
                )
            raw_metadata = response.get("metadata")
            # Validate the complete metadata budget before MCP preparation can
            # perform discovery, publish items, or retain remote state.
            metadata = self._controller.responses.validate_metadata(raw_metadata)
            if "conversation" in response:
                validate_response_conversation(response["conversation"])
            if "input" in response:
                response_input = validate_response_input(response["input"])
            if "output_modalities" in response:
                response_output_modalities = validate_response_output_modalities(response["output_modalities"])
            effective_output_modalities = (
                response_output_modalities
                if response_output_modalities is not None
                else self._controller.session.public_view()["output_modalities"]
            )
            if "audio" in response and effective_output_modalities != ["audio"]:
                raise invalid_value(
                    "response.audio is valid only when output_modalities is ['audio']",
                    param="response.audio",
                )
            if "instructions" in response:
                response_instructions = self._controller.session.validate_response_instructions(
                    response["instructions"]
                )
            if "max_output_tokens" in response:
                response_max_output_tokens = self._controller.session.validate_response_max_output_tokens(
                    response["max_output_tokens"]
                )
            if "parallel_tool_calls" in response:
                response_parallel_tool_calls = self._controller.session.validate_response_parallel_tool_calls(
                    response["parallel_tool_calls"]
                )
            if "tools" in response:
                response_tools = self._controller.session.validate_response_tools(response["tools"])
            effective_tools = (
                response_tools
                if response_tools is not None
                else self._controller.session.public_view().get("tools", [])
            )
            if "tool_choice" in response:
                response_tool_choice = self._controller.session.validate_response_tool_choice(
                    response["tool_choice"],
                    tools=effective_tools,
                )
            elif response_tools is not None:
                self._controller.session.validate_response_tool_choice(
                    self._controller.session.public_view().get("tool_choice", "auto"),
                    tools=effective_tools,
                )
        session_view = self._controller.session.public_view()
        effective_response_output_modalities = (
            copy.deepcopy(response_output_modalities)
            if response_output_modalities is not None
            else copy.deepcopy(session_view["output_modalities"])
        )
        if effective_response_output_modalities == ["audio"]:
            await self._ensure_output_voice_capabilities(param="response.audio.output.voice")
            raw_response_audio = response.get("audio", {}) if isinstance(response, dict) else {}
            response_audio_output = self._controller.session.validate_response_audio_output(raw_response_audio)
        effective_response_max_output_tokens = (
            response_max_output_tokens if response_max_output_tokens is not None else session_view["max_output_tokens"]
        )
        effective_response_parallel_tool_calls = (
            response_parallel_tool_calls
            if response_parallel_tool_calls is not None
            else session_view["parallel_tool_calls"]
        )
        effective_response_truncation = copy.deepcopy(session_view.get("truncation"))
        effective_response_instructions = (
            response_instructions if response_instructions is not None else session_view["instructions"]
        )
        effective_response_tools = copy.deepcopy(
            response_tools if response_tools is not None else session_view.get("tools", [])
        )
        effective_response_tool_choice = copy.deepcopy(
            response_tool_choice if response_tool_choice is not None else session_view.get("tool_choice", "auto")
        )
        pending_calls = self._controller.pending_tool_call_ids()
        pending_context = (
            self._client_tool_broker.pending_context_call_ids() if self._client_tool_broker is not None else ()
        )
        pending_conversation_item_ids = self._pending_client_text_item_ids()
        staged_client_outputs = False
        if pending_calls and self._client_tool_broker is not None:
            staged_client_outputs = await self._client_tool_broker.outputs_staged(pending_calls)
        defer_for_client_output = staged_client_outputs or bool(pending_context)
        if pending_calls and not staged_client_outputs:
            raise RealtimeProtocolError(
                message=f"Function call {pending_calls[0]!r} requires a correlated output before response.create",
                code="tool_output_pending",
                param="response",
            )
        manual_input_mode = self._manual_input_mode()
        session_has_tools = bool(effective_response_tools)
        tool_choice_is_noop = (
            not session_has_tools and isinstance(response_tool_choice, str) and response_tool_choice in {"auto", "none"}
        )
        gate_installed = self._response_gate is not None and self._response_gate.installed
        claim_manual_commits = bool(
            manual_input_mode and gate_installed and self._response_gate.has_unclaimed_manual_commit
        )
        defer_for_manual_commit = bool(claim_manual_commits and self._response_gate.response_slot_occupied)
        defer_for_conversation_append = bool(
            gate_installed and pending_conversation_item_ids and not self._response_gate.response_slot_occupied
        )
        defer_response = defer_for_client_output or defer_for_manual_commit or defer_for_conversation_append
        response_needs_gate = (
            gate_installed
            or manual_input_mode
            or response_tools is not None
            or response_instructions is not None
            or response_max_output_tokens is not None
            or response_parallel_tool_calls is not None
            or response_input is not None
            or response_output_modalities is not None
            or (response_tool_choice is not None and not tool_choice_is_noop)
        )
        if response_needs_gate and not gate_installed:
            raise RealtimeProtocolError(
                message="Response-scoped inference coordination is not installed in this Realtime pipeline",
                code="response_gate_missing",
                param="response",
                error_type="server_error",
            )
        if self._controller.response_in_progress and not defer_response:
            active_response_id = self._controller.active_response_id
            raise RealtimeProtocolError(
                message=f"Response {active_response_id} is already in progress",
                code="response_in_progress",
                param="response",
            )
        preparation_id: str | None = None
        if response_needs_gate:
            preparation_id = new_realtime_id("request")
            await self._response_gate.reserve_response_preparation(
                preparation_id,
                deferred=defer_response,
            )

        try:
            if response_instructions is not None:
                self._response_gate.validate_response_instructions(response_instructions)
            if response_input is not None:
                self._response_gate.validate_response_input_context()
                response_input_messages = await self._freeze_response_input(response_input)
            if (
                effective_response_output_modalities == ["audio"]
                and isinstance(response, dict)
                and ("audio" in response or "output_modalities" in response)
            ):
                self._response_gate.validate_response_output(
                    output_modalities=effective_response_output_modalities,
                    audio_output=response_audio_output,
                )
            if response_needs_gate:
                if self._mcp_runtime is None:
                    if any(tool.get("type") == "mcp" for tool in effective_response_tools):
                        raise RealtimeProtocolError(
                            message="Native MCP is not bound to this Realtime transport",
                            code="mcp_runtime_missing",
                            param="response.tools",
                            error_type="server_error",
                        )
                    pipeline_response_tools = copy.deepcopy(effective_response_tools)
                    projected_response_tool_choice = project_tool_choice_to_pipeline(effective_response_tool_choice)
                else:
                    prepared_response_tools = await self._mcp_runtime.prepare_tools(
                        effective_response_tools,
                        effective_response_tool_choice,
                    )
                    pipeline_response_tools = prepared_response_tools.pipeline_tools
                    projected_response_tool_choice = prepared_response_tools.pipeline_tool_choice
                    response_mcp_pipeline_names = prepared_response_tools.mcp_pipeline_names
                    response_client_tool_bindings = prepared_response_tools.client_tool_bindings
            if preparation_id is not None:
                # MCP discovery and other response preparation can yield while
                # a fused audio turn claims the same response slot. Recheck at
                # the last pre-commit boundary so no response is journaled with
                # two physical inference owners.
                self._response_gate.validate_response_preparation(preparation_id)
            if defer_response:
                if not gate_installed:
                    raise RealtimeProtocolError(
                        message="Deferred response ordering is not installed in this Realtime pipeline",
                        code="response_gate_missing",
                        param="response",
                        error_type="server_error",
                    )
                request_id = new_realtime_id("request")
                activation_generation = self._controller.interruption_generation
                frame = RealtimeDeferredResponseCreateFrame(
                    request_id=request_id,
                    event_id=message.get("event_id"),
                    response_metadata=copy.deepcopy(metadata),
                    tool_call_ids=tuple(dict.fromkeys((*pending_calls, *pending_context))),
                    activation_generation=activation_generation,
                    conversation_item_ids=pending_conversation_item_ids,
                    tool_choice=copy.deepcopy(projected_response_tool_choice),
                    tools=copy.deepcopy(pipeline_response_tools),
                    public_tools=copy.deepcopy(effective_response_tools),
                    mcp_pipeline_names=response_mcp_pipeline_names,
                    client_tool_bindings=copy.deepcopy(response_client_tool_bindings),
                    instructions=effective_response_instructions,
                    max_output_tokens=effective_response_max_output_tokens,
                    parallel_tool_calls=effective_response_parallel_tool_calls,
                    truncation=effective_response_truncation,
                    response_input=None,
                    input_messages=copy.deepcopy(response_input_messages),
                    output_modalities=effective_response_output_modalities,
                    audio_output=copy.deepcopy(response_audio_output),
                )
                self._response_gate.register_deferred_response(
                    request_id,
                    activation_generation=activation_generation,
                    preparation_id=preparation_id,
                    claim_manual_commits=claim_manual_commits,
                    audio_response=effective_response_output_modalities == ["audio"],
                )
                response_tool_preparation_admitted = True
                preparation_id = None
                if prepared_response_tools is not None:
                    self._mcp_runtime.commit_response_tools(prepared_response_tools)
                    prepared_response_tools = None
                return frame
            async with self._controller.response_transition_lock:
                if self._controller.response_in_progress:
                    active_response_id = self._controller.active_response_id
                    raise RealtimeProtocolError(
                        message=f"Response {active_response_id} is already in progress",
                        code="response_in_progress",
                        param="response",
                    )
                events = self._controller.start_response(
                    metadata=metadata,
                    tools=copy.deepcopy(effective_response_tools),
                    mcp_pipeline_names=response_mcp_pipeline_names,
                    client_tool_bindings=response_client_tool_bindings,
                    instructions=effective_response_instructions,
                    max_output_tokens=effective_response_max_output_tokens,
                    parallel_tool_calls=effective_response_parallel_tool_calls,
                    output_modalities=effective_response_output_modalities,
                    audio_output=copy.deepcopy(response_audio_output),
                )
                response_id = self._controller.active_response_id or ""
                try:
                    self._refresh_active_response_output_audio()
                    if response_needs_gate:
                        frame = RealtimeResponseCreateFrame(
                            response_id=response_id,
                            tool_choice=copy.deepcopy(projected_response_tool_choice),
                            tools=copy.deepcopy(pipeline_response_tools),
                            mcp_pipeline_names=response_mcp_pipeline_names,
                            client_tool_bindings=copy.deepcopy(response_client_tool_bindings),
                            instructions=effective_response_instructions,
                            max_output_tokens=effective_response_max_output_tokens,
                            parallel_tool_calls=effective_response_parallel_tool_calls,
                            truncation=effective_response_truncation,
                            input_messages=copy.deepcopy(response_input_messages),
                            output_modalities=effective_response_output_modalities,
                            audio_output=copy.deepcopy(response_audio_output),
                        )
                        self._response_gate.register_response_marker(
                            response_id,
                            preparation_id=preparation_id,
                            claim_manual_commits=claim_manual_commits,
                        )
                        response_tool_preparation_admitted = True
                        preparation_id = None
                        if prepared_response_tools is not None:
                            self._mcp_runtime.commit_response_tools(prepared_response_tools)
                            prepared_response_tools = None
                    else:
                        frame = LLMRunFrame()
                    await self._emit_events(events)
                except BaseException as exc:
                    await self._terminalize_failed_response_start(
                        response_id,
                        cancelled=isinstance(exc, asyncio.CancelledError),
                    )
                    raise
                if self._connection_closed:
                    await self._terminalize_failed_response_start(response_id, cancelled=True)
                    return None
                return frame
        except MCPToolPreparationPoisonedError as cause:
            self.notify_connection_closed()
            raise RealtimeProtocolError(
                message="Realtime tool preparation could not be restored safely",
                code="tool_preparation_state_inconsistent",
                param="response.tools",
                error_type="server_error",
            ) from cause
        except BaseException as cause:
            if prepared_response_tools is not None:
                if response_tool_preparation_admitted:
                    self.notify_connection_closed()
                else:
                    try:
                        self._mcp_runtime.rollback_response_tools(prepared_response_tools)
                    except BaseException as rollback_error:
                        logger.error(
                            "Realtime response tool rollback failed error_type={}",
                            type(rollback_error).__name__,
                        )
                        self.notify_connection_closed()
                        if isinstance(cause, asyncio.CancelledError):
                            raise cause from rollback_error
                        raise RealtimeProtocolError(
                            message="Realtime tool preparation could not be restored safely",
                            code="tool_preparation_state_inconsistent",
                            param="response.tools",
                            error_type="server_error",
                        ) from rollback_error
            raise
        finally:
            if preparation_id is not None and self._response_gate is not None:
                self._response_gate.release_response_preparation(preparation_id)

    async def _deserialize_response_cancel(self, message: dict[str, Any]) -> Frame | None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "response_id", "type"}))
        response_id = message.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise invalid_type("response_id must be a string", param="response_id")
        async with self._controller.response_transition_lock:
            if not self._controller.response_in_progress:
                raise RealtimeProtocolError(
                    message="No response is currently in progress",
                    code="response_not_found",
                    param="response_id",
                )
            if response_id and response_id != self._controller.active_response_id:
                raise invalid_value("response_id does not match the active response", param="response_id")
            if self._on_response_cancel is not None:
                self._on_response_cancel()
            active_response_id = self._controller.active_response_id or ""
            if self._response_gate is not None:
                self._response_gate.cancel_response(active_response_id)
            interruption = InterruptionFrame()
            # Event acceptance is the ordering point for a following
            # commit/response.create. The same physical frame is deduplicated
            # when the gate and observer later see it in the Pipecat graph.
            self._controller.observe_interruption(
                interruption.id,
                reason="client_cancelled",
                response_id=active_response_id,
            )
            self.cancel_output_audio(interruption.id)
        return interruption

    async def _deserialize_item_create(self, message: dict[str, Any]) -> Frame | None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "item", "previous_item_id", "type"}))
        item = message.get("item")
        if not isinstance(item, dict):
            raise invalid_type("conversation.item.create requires an item object", param="item")
        previous_supplied = "previous_item_id" in message
        previous_item_id = message.get("previous_item_id")
        if previous_supplied and not isinstance(previous_item_id, str):
            raise invalid_type("previous_item_id must be a string", param="previous_item_id")

        role = item.get("role")
        is_text_item = item.get("type") == "message" and isinstance(role, str) and role in {"system", "user"}
        if not is_text_item:
            await self._wait_pending_conversation_publications()

        if item.get("type") == "mcp_approval_response":
            validated_item = self._controller.conversation.validate_item(item)
            if self._mcp_runtime is None:
                raise RealtimeProtocolError(
                    message="Native MCP approvals are not bound to this Realtime transport",
                    code="mcp_runtime_missing",
                    param="item.approval_request_id",
                    error_type="server_error",
                )
            approval_claim = self._mcp_runtime.claim_approval_response(validated_item)
            try:
                events = self._controller.create_conversation_item(
                    validated_item,
                    previous_item_id=previous_item_id,
                    previous_item_id_supplied=previous_supplied,
                )
            except BaseException:
                self._mcp_runtime.abandon_approval_response(approval_claim, cancel_call=False)
                raise
            try:
                await self._emit_events(events)
            except BaseException as exc:
                cancellation_events = self._mcp_runtime.abandon_approval_response(
                    approval_claim,
                    cancel_call=True,
                )
                if cancellation_events and not isinstance(exc, asyncio.CancelledError):
                    try:
                        await self._emit_events(cancellation_events)
                    except Exception:
                        logger.exception("Could not publish MCP cancellation after approval send failure")
                raise
            if self._connection_closed:
                self._mcp_runtime.abandon_approval_response(approval_claim, cancel_call=True)
                return None
            await self._emit_events(self._mcp_runtime.resolve_approval_response(approval_claim))
            return None

        if item.get("type") == "function_call_output":
            validated_item = self._controller.conversation.validate_item(item)
            call_id = validated_item["call_id"]
            output = validated_item["output"]
            record = self._controller.tool_call(call_id)
            if record is None:
                raise RealtimeProtocolError(
                    message=f"Function call {call_id!r} was not found",
                    code="call_not_found",
                    param="item.call_id",
                )
            if self._client_tool_broker is None:
                raise RealtimeProtocolError(
                    message="Client-owned tools are not bound to this Realtime transport",
                    code="client_tool_broker_missing",
                    param="item.call_id",
                    error_type="server_error",
                )
            if self._response_gate is None or not self._response_gate.installed:
                raise RealtimeProtocolError(
                    message="Client-tool output ordering is not installed in this Realtime pipeline",
                    code="response_gate_missing",
                    param="item.call_id",
                    error_type="server_error",
                )
            add_kwargs = {
                "call_id": call_id,
                "output": output,
                "owner": "client",
                "item_id": validated_item.get("id"),
                "previous_item_id": previous_item_id,
                "previous_item_id_supplied": previous_supplied,
            }
            # Preview every controller/journal validation before waking the
            # pipeline handler, so an invalid item cannot leak into context.
            # A fast official client sends this event when output_item.done
            # arrives, before Response A's later response.done event.
            trial = copy.deepcopy(self._controller)
            trial_record = trial.tool_call(call_id)
            if trial_record is not None and trial_record.response_status is None:
                if record.response_id != self._controller.active_response_id:
                    raise RealtimeProtocolError(
                        message=f"Function call {call_id!r} has no active originating response",
                        code="tool_call_lifecycle_invalid",
                        param="item.call_id",
                        error_type="server_error",
                    )
                trial_record.response_status = "completed"
            trial.add_function_output(**add_kwargs)
            await self._client_tool_broker.stage_output(
                call_id=call_id,
                name=record.name,
                output=output,
            )
            return RealtimeClientToolOutputFrame(
                event_id=message.get("event_id"),
                call_id=call_id,
                tool_name=record.name,
                response_id=record.response_id,
                output=output,
                item_id=validated_item.get("id"),
                previous_item_id=previous_item_id,
                previous_item_id_supplied=previous_supplied,
            )

        if item.get("type") != "message" or item.get("role") not in {"system", "user"}:
            raise RealtimeProtocolError(
                message="This cascaded pipeline accepts client-created system or user text messages",
                code="unsupported_item",
                param="item",
            )
        validated_item = self._controller.conversation.validate_item(item)
        text = _extract_item_text(validated_item)
        if text is None:
            raise invalid_value("conversation item requires input_text content", param="item.content")
        if self._response_gate is None or not self._response_gate.installed or self._llm_context is None:
            raise RealtimeProtocolError(
                message="Client conversation items require an installed canonical context owner",
                code="conversation_context_runtime_missing",
                param="item",
                error_type="server_error",
            )
        normalized = copy.deepcopy(validated_item)
        normalized["content"] = [{"type": "input_text", "text": text}]
        controller_snapshot = self._controller.snapshot_conversation_mutation_state()
        try:
            events = self._controller.create_conversation_item(
                normalized,
                previous_item_id=previous_item_id,
                previous_item_id_supplied=previous_supplied,
            )
        except BaseException as exc:
            try:
                self._controller.restore_conversation_mutation_state(controller_snapshot)
            except Exception as rollback_error:
                logger.error(
                    "Realtime conversation create rollback failed owner={}",
                    type(rollback_error).__name__,
                )
                self.notify_connection_closed()
                raise RuntimeError("Realtime conversation state could not be restored; connection retired") from exc
            raise
        item_id = events[0]["item"]["id"]
        context_message = {"role": normalized["role"], "content": text}
        self._pending_conversation_appends[item_id] = _PendingConversationAppend(
            rollback_state=controller_snapshot,
            provisional_state=self._controller.snapshot_conversation_mutation_state(),
            publication_done=asyncio.Event(),
        )
        self._context_applied_events.setdefault(item_id, asyncio.Event())
        return RealtimeConversationAppendFrame(
            item_id=item_id,
            context_message=context_message,
            events=tuple(events),
        )

    async def _deserialize_item_delete(self, message: dict[str, Any]) -> None:
        """Delete an item only when its exact Pipecat context owner is known."""
        _reject_unknown_event_fields(message, frozenset({"event_id", "item_id", "type"}))
        await self._wait_pending_conversation_publications()
        item_id = _required_item_id(message)
        item = self._controller.conversation.item(item_id)
        if self._controller.is_active_output_item(item_id):
            raise RealtimeProtocolError(
                message="An item in the active response cannot be deleted",
                code="conversation_item_in_use",
                param="item_id",
            )
        await self._wait_message_context_if_pending(item_id, item)

        context_messages = self._context_messages_after_removing_item(item_id, item)
        controller_snapshot = self._controller.snapshot_conversation_mutation_state()
        context_snapshot = self._snapshot_conversation_context()
        next_owners = dict(self._context_messages_by_item_id)
        next_owners.pop(item_id, None)
        next_applied_events = dict(self._context_applied_events)
        next_applied_events.pop(item_id, None)
        next_cleared_item_ids = set(self._context_cleared_item_ids)
        next_cleared_item_ids.discard(item_id)

        committed = False
        try:
            event = self._controller.delete_item_event(item_id)
            if context_messages is not None:
                self._llm_context.set_messages(context_messages)
            self._apply_conversation_ownership(
                message_owners=next_owners,
                applied_events=next_applied_events,
                cleared_item_ids=next_cleared_item_ids,
            )
            committed = True
            try:
                await self._emit_event(event)
            except BaseException:
                self.notify_connection_closed()
                raise
        except BaseException as exc:
            if not committed:
                self._restore_conversation_transaction(
                    controller_snapshot=controller_snapshot,
                    context_snapshot=context_snapshot,
                    cause=exc,
                )
            raise

    async def _deserialize_item_truncate(self, message: dict[str, Any]) -> None:
        """Synchronize assistant audio and Pipecat context to a playback boundary."""
        _reject_unknown_event_fields(
            message,
            frozenset({"audio_end_ms", "content_index", "event_id", "item_id", "type"}),
        )
        await self._wait_pending_conversation_publications()
        item_id = _required_item_id(message)
        content_index = message.get("content_index")
        audio_end_ms = message.get("audio_end_ms")
        if isinstance(content_index, bool) or not isinstance(content_index, int):
            raise invalid_type("content_index must be an integer", param="content_index")
        if isinstance(audio_end_ms, bool) or not isinstance(audio_end_ms, int):
            raise invalid_type("audio_end_ms must be an integer", param="audio_end_ms")
        if content_index < 0:
            raise invalid_value("content_index must be non-negative", param="content_index")
        if audio_end_ms < 0:
            raise invalid_value("audio_end_ms must be non-negative", param="audio_end_ms")
        if self._controller.is_active_output_item(item_id):
            raise RealtimeProtocolError(
                message="An item in the active response cannot be truncated",
                code="conversation_item_in_use",
                param="item_id",
            )
        item = self._controller.conversation.item(item_id)
        await self._wait_message_context_if_pending(item_id, item)

        projection = self._controller.preview_item_truncation(
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )

        replacement: dict[str, Any] | None = None
        if audio_end_ms == 0:
            context_messages = self._context_messages_after_removing_item(item_id, item)
            if context_messages is None:
                raise RealtimeProtocolError(
                    message="The selected assistant item is already absent from model context",
                    code="conversation_item_context_missing",
                    param="item_id",
                )
        else:
            context_messages, replacement = self._context_messages_after_truncating_item(
                item_id,
                item,
                context_text=projection.context_text,
                previous_context_text=projection.previous_context_text,
            )
        controller_snapshot = self._controller.snapshot_conversation_mutation_state()
        context_snapshot = self._snapshot_conversation_context()
        next_owners = dict(self._context_messages_by_item_id)
        next_cleared_item_ids = set(self._context_cleared_item_ids)
        if replacement is not None:
            next_owners[item_id] = replacement
            next_cleared_item_ids.discard(item_id)
        elif projection.context_text:
            next_cleared_item_ids.discard(item_id)
        else:
            next_owners.pop(item_id, None)
            next_cleared_item_ids.add(item_id)

        committed = False
        try:
            event = self._controller.truncate_item_event(
                item_id=item_id,
                content_index=content_index,
                audio_end_ms=audio_end_ms,
            )
            if context_messages is not None:
                self._llm_context.set_messages(context_messages)
            self._apply_conversation_ownership(
                message_owners=next_owners,
                applied_events=self._context_applied_events,
                cleared_item_ids=next_cleared_item_ids,
            )
            committed = True
            try:
                await self._emit_event(event)
            except BaseException:
                self.notify_connection_closed()
                raise
        except BaseException as exc:
            if not committed:
                self._restore_conversation_transaction(
                    controller_snapshot=controller_snapshot,
                    context_snapshot=context_snapshot,
                    cause=exc,
                )
            raise

    def _context_messages_after_truncating_item(
        self,
        item_id: str,
        item: dict[str, Any],
        *,
        context_text: str,
        previous_context_text: str,
    ) -> tuple[list[Any] | None, dict[str, Any] | None]:
        """Build an identity-safe Pipecat context projection for a truncation."""
        if item_id in self._context_cleared_item_ids:
            if context_text:
                raise RealtimeProtocolError(
                    message="The conversation item's model context has already been removed",
                    code="conversation_item_context_missing",
                    param="item_id",
                )
            return None, None
        target = self._context_messages_by_item_id.get(item_id)
        if target is None or self._llm_context is None:
            item_type = item.get("type")
            role = item.get("role")
            raise RealtimeProtocolError(
                message=(
                    f"Conversation {item_type or 'item'} with role {role!r} has no exact "
                    "Pipecat context ownership mapping"
                ),
                code="conversation_item_context_unavailable",
                param="item_id",
            )
        messages = self._llm_context.get_messages()
        matches = [index for index, context_message in enumerate(messages) if context_message is target]
        if len(matches) != 1:
            raise RealtimeProtocolError(
                message="The conversation item's exact Pipecat context message is no longer available",
                code="conversation_item_context_missing",
                param="item_id",
            )
        if target.get("content") != previous_context_text:
            raise RealtimeProtocolError(
                message="The assistant transcript no longer matches its exact Pipecat context message",
                code="conversation_item_context_mismatch",
                param="item_id",
            )
        match = matches[0]
        if not context_text:
            return [context_message for index, context_message in enumerate(messages) if index != match], None
        replacement = copy.deepcopy(target)
        replacement["content"] = context_text
        return [replacement if index == match else value for index, value in enumerate(messages)], replacement

    def _context_messages_after_removing_item(
        self,
        item_id: str,
        item: dict[str, Any],
    ) -> list[Any] | None:
        """Return a context projection with one exactly owned item removed."""
        if item_id in self._context_cleared_item_ids:
            return None
        target = self._context_messages_by_item_id.get(item_id)
        if target is None or self._llm_context is None:
            item_type = item.get("type")
            role = item.get("role")
            raise RealtimeProtocolError(
                message=(
                    f"Conversation {item_type or 'item'} with role {role!r} has no exact "
                    "Pipecat context ownership mapping"
                ),
                code="conversation_item_context_unavailable",
                param="item_id",
            )
        messages = self._llm_context.get_messages()
        matches = [index for index, context_message in enumerate(messages) if context_message is target]
        if len(matches) != 1:
            raise RealtimeProtocolError(
                message="The conversation item's exact Pipecat context message is no longer available",
                code="conversation_item_context_missing",
                param="item_id",
            )
        return [context_message for index, context_message in enumerate(messages) if index != matches[0]]

    async def _deserialize_item_retrieve(self, message: dict[str, Any]) -> None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "item_id", "type"}))
        await self._wait_pending_conversation_publications()
        item_id = _required_item_id(message)
        await self._emit_event(self._controller.retrieve_item_event(item_id))

    async def _deserialize_append(self, message: dict[str, Any]) -> Frame | None:
        _reject_unknown_event_fields(message, frozenset({"audio", "event_id", "type"}))
        audio_b64 = message.get("audio")
        if not isinstance(audio_b64, str):
            raise invalid_type("audio must be a base64 string", param="audio")
        if len(audio_b64) > max_base64_audio_chars(MAX_AUDIO_APPEND_BYTES):
            raise RealtimeProtocolError(
                message="input_audio_buffer.append exceeds the pipeline buffer limit",
                code="input_buffer_overflow",
                param="audio",
            )
        try:
            raw = decode_base64_audio(
                audio_b64,
                format_type=self._client_in_format,
            )
            if len(raw) > MAX_AUDIO_APPEND_BYTES:
                raise RealtimeProtocolError(
                    message="input_audio_buffer.append exceeds the pipeline buffer limit",
                    code="input_buffer_overflow",
                    param="audio",
                )
            if self._manual_input_mode():
                max_client_bytes = max_pending_input_bytes(self._client_in_format, self._client_in_rate)
                if len(self._manual_input_audio) + len(raw) > max_client_bytes:
                    raise RealtimeProtocolError(
                        message="Manual input audio buffer exceeds the pipeline buffer limit",
                        code="input_buffer_overflow",
                        param="audio",
                    )
                self._manual_input_audio.extend(raw)
                self._record_input_audio(raw)
                return None
            pcm = await self._resampler.to_pipeline(
                raw,
                self._client_in_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_in_rate,
                format_type=self._client_in_format,
            )
        except RealtimeProtocolError:
            raise
        except ValueError as exc:
            raise RealtimeProtocolError(
                message=str(exc),
                code="invalid_audio",
                param="audio",
            ) from exc
        self._record_input_audio(raw)
        if not pcm:
            return None
        if len(pcm) > MAX_AUDIO_APPEND_BYTES * 4:
            raise RealtimeProtocolError(
                message="Decoded input audio exceeds the pipeline buffer limit",
                code="input_buffer_overflow",
                param="audio",
            )
        return InputAudioRawFrame(audio=pcm, sample_rate=self._pipeline_in_rate, num_channels=1)

    def _record_input_audio(self, raw: bytes) -> None:
        """Advance idle-timeout offsets after one wire append is accepted."""
        if self._idle_timeout is None:
            return
        bytes_per_sample = 1 if self._client_in_format in SUPPORTED_G711_FORMAT_TYPES else 2
        self._idle_timeout.record_input_audio(
            sample_count=len(raw) // bytes_per_sample,
            sample_rate=self._client_in_rate or DEFAULT_CLIENT_PCM_RATE,
        )

    async def _deserialize_input_audio_commit(self, message: dict[str, Any]) -> None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "type"}))
        if not self._manual_input_mode():
            raise RealtimeProtocolError(
                message="Manual audio commit requires session.audio.input.turn_detection=null",
                code="unsupported_capability",
                param="type",
            )
        if not self._manual_input_audio:
            raise RealtimeProtocolError(
                message="Cannot commit an empty input audio buffer",
                code="input_audio_buffer_commit_empty",
                param="type",
            )
        if self._manual_input_commit_hook is None or self._response_gate is None:
            raise RealtimeProtocolError(
                message="Manual input delivery is not bound to the Realtime pipeline",
                code="manual_input_handler_missing",
                param="type",
                error_type="server_error",
            )
        if self._response_gate.pending_commits:
            raise RealtimeProtocolError(
                message="The previous manual audio commit is still awaiting its final transcript",
                code="input_audio_buffer_commit_pending",
                param="type",
            )

        raw = bytes(self._manual_input_audio)
        try:
            pcm = await self._resampler.complete_input_to_pipeline(
                raw,
                self._client_in_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_in_rate,
                format_type=self._client_in_format,
            )
        except ValueError as exc:
            raise RealtimeProtocolError(
                message=str(exc),
                code="invalid_audio",
                param="audio",
            ) from exc
        if not pcm:
            raise RealtimeProtocolError(
                message="Committed input audio produced no pipeline samples",
                code="invalid_audio",
                param="audio",
            )
        if len(pcm) > MAX_PENDING_INPUT_BYTES:
            raise RealtimeProtocolError(
                message="Committed input audio exceeds the pipeline buffer limit",
                code="input_buffer_overflow",
                param="audio",
            )

        self._response_gate.register_commit()
        try:
            await self._manual_input_commit_hook(pcm, self._pipeline_in_rate)
        except BaseException:
            self._response_gate.abort_commit()
            raise
        self._manual_input_audio.clear()
        return None

    async def _deserialize_input_audio_clear(self, message: dict[str, Any]) -> None:
        _reject_unknown_event_fields(message, frozenset({"event_id", "type"}))
        if not self._manual_input_mode():
            raise RealtimeProtocolError(
                message="Input audio cannot be retracted after it enters the server-VAD streaming pipeline",
                code="unsupported_capability",
                param="type",
            )
        self._manual_input_audio.clear()
        await self._emit_event(build_server_event("input_audio_buffer.cleared"))
        return None

    async def _serialize_output_audio(self, frame: OutputAudioRawFrame) -> None:
        if self._output_is_cancelled():
            return
        response_id = self._controller.active_response_id
        if response_id is None:
            logger.debug("Dropping pipeline audio with no active Realtime response")
            return
        output_generation = self._output_audio_generation
        pcm = frame.audio or b""
        if not pcm:
            return
        if self._controller.output_kind != "audio":
            logger.error("Dropping pipeline audio for a text-only Realtime session")
            return
        try:
            client_pcm = await self._resampler.from_pipeline(
                pcm,
                self._client_out_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_out_rate,
                format_type=self._client_out_format,
            )
        except ValueError as exc:
            await self._report_output_audio_conversion_failure(response_id, exc)
            return
        if (
            not client_pcm
            or self._output_is_cancelled()
            or output_generation != self._output_audio_generation
            or response_id != self._controller.active_response_id
        ):
            return
        await self._emit_events(
            self._controller.output_audio_delta_event(
                encode_base64_audio(client_pcm, format_type=self._client_out_format),
                sample_count=len(client_pcm) // (2 if self._client_out_format == "audio/pcm" else 1),
                sample_rate=self._client_out_rate or DEFAULT_CLIENT_PCM_RATE,
            )
        )

    async def flush_output_audio(self) -> None:
        """Emit SOXR's delayed samples before the response terminal events."""
        if self._output_is_cancelled():
            return
        response_id = self._controller.active_response_id
        if response_id is None:
            self._resampler.reset_downlink()
            return
        output_generation = self._output_audio_generation
        try:
            client_pcm = await self._resampler.flush_from_pipeline(
                self._client_out_rate or DEFAULT_CLIENT_PCM_RATE,
                pipeline_rate=self._pipeline_out_rate,
                format_type=self._client_out_format,
            )
        except ValueError as exc:
            await self._report_output_audio_conversion_failure(response_id, exc)
            return
        if (
            self._output_is_cancelled()
            or output_generation != self._output_audio_generation
            or response_id != self._controller.active_response_id
        ):
            return
        if client_pcm:
            await self._emit_events(
                self._controller.output_audio_delta_event(
                    encode_base64_audio(client_pcm, format_type=self._client_out_format),
                    sample_count=len(client_pcm) // (2 if self._client_out_format == "audio/pcm" else 1),
                    sample_rate=self._client_out_rate or DEFAULT_CLIENT_PCM_RATE,
                )
            )

    async def _report_output_audio_conversion_failure(self, response_id: str, exc: ValueError) -> None:
        """Report one safe, owner-correlated failure for an output codec error."""
        logger.warning(f"Realtime output audio conversion failed response_id={response_id}: {exc}")
        failure = RealtimeProtocolError(
            message="The server could not convert generated audio to the requested output format",
            code="output_audio_conversion_error",
            param="response.audio.output.format",
            error_type="server_error",
        )
        if self._output_audio_failure_handler is None:
            raise failure from exc
        await self._output_audio_failure_handler(response_id, failure)

    def reset_output_audio(self) -> None:
        """Discard delayed samples when output is interrupted or cancelled."""
        self._output_audio_generation += 1
        self._resampler.reset_downlink()

    def cancel_output_audio(self, interruption_id: int) -> None:
        """Block PCM until the output transport acknowledges its queue reset."""
        if (
            interruption_id in self._acknowledged_output_interruptions
            or interruption_id in self._pending_output_interruptions
        ):
            return
        self._pending_output_interruptions.add(interruption_id)
        self.reset_output_audio()

    def acknowledge_output_interruption(self, interruption_id: int) -> None:
        """Release one barrier only after that interruption reset the media queue."""
        self._pending_output_interruptions.discard(interruption_id)
        if interruption_id in self._acknowledged_output_interruptions:
            return
        if len(self._output_interruption_history) >= 4096:
            expired = self._output_interruption_history.popleft()
            self._acknowledged_output_interruptions.discard(expired)
        self._output_interruption_history.append(interruption_id)
        self._acknowledged_output_interruptions.add(interruption_id)

    def _output_is_cancelled(self) -> bool:
        return bool(self._pending_output_interruptions)

    def _refresh_active_response_output_audio(self) -> None:
        """Bind wire encoding to the active response's effective audio snapshot."""
        output = self._controller.active_audio_output
        if output is None:
            return
        self.activate_response_output_audio(output)

    def activate_response_output_audio(self, output: dict[str, Any] | None) -> None:
        """Bind wire encoding to a frozen explicit or pipeline response."""
        if output is None:
            return
        session = self._controller.public_session()
        audio = session.get("audio")
        if not isinstance(audio, dict):
            raise RuntimeError("Realtime session has no audio configuration")
        session["audio"] = copy.deepcopy(audio)
        session["audio"]["output"] = output
        old_signature = (self._client_out_format, self._client_out_rate)
        self._client_out_format = extract_client_output_format_type(session)
        self._client_out_rate = extract_client_output_pcm_rate(session)
        if old_signature != (self._client_out_format, self._client_out_rate):
            self._output_audio_generation += 1
            self._resampler.reset_downlink()

    def _restore_session_output_audio(self) -> None:
        """Restore canonical output encoding at the published terminal boundary."""
        old_signature = (self._client_out_format, self._client_out_rate)
        session = self._controller.public_session()
        self._client_out_format = extract_client_output_format_type(session)
        self._client_out_rate = extract_client_output_pcm_rate(session)
        if old_signature != (self._client_out_format, self._client_out_rate):
            self._output_audio_generation += 1
            self._resampler.reset_downlink()

    def _project_response_input(self, response_input: RealtimeResponseInput) -> list[dict[str, Any]]:
        """Resolve custom response items against exact connection-owned state."""
        return project_response_input_messages(
            response_input,
            resolve_item=self._controller.conversation.item,
            resolve_context_message=self._context_messages_by_item_id.get,
        )

    def _pending_client_text_item_ids(self) -> tuple[str, ...]:
        """Snapshot preceding client text items not yet in canonical context."""
        pending: list[str] = []
        for item_id, applied in self._context_applied_events.items():
            if applied.is_set():
                continue
            try:
                item = self._controller.conversation.item(item_id)
            except RealtimeProtocolError:
                continue
            if item.get("type") == "message" and item.get("role") in {"system", "user"}:
                content = item.get("content")
                if isinstance(content, list) and all(
                    isinstance(part, dict) and part.get("type") == "input_text" for part in content
                ):
                    pending.append(item_id)
        return tuple(pending)

    async def _wait_message_context_if_pending(self, item_id: str, item: dict[str, Any]) -> None:
        """Wait for an acknowledged message's downstream context commit."""
        if item.get("type") != "message" or item_id in self._context_messages_by_item_id:
            return
        applied = self._context_applied_events.get(item_id)
        if applied is None:
            return
        try:
            await asyncio.wait_for(applied.wait(), timeout=_CONTEXT_APPLY_TIMEOUT_SECS)
        except TimeoutError as exc:
            raise RealtimeProtocolError(
                message=f"Conversation item {item_id!r} did not reach model context before the server deadline",
                code="conversation_context_apply_timeout",
                param="item_id",
                error_type="server_error",
            ) from exc
        if self._connection_closed:
            raise RealtimeProtocolError(
                message="The Realtime connection closed before the conversation item reached model context",
                code="conversation_context_apply_cancelled",
                param="item_id",
                error_type="server_error",
            )

    async def _freeze_response_input(self, response_input: RealtimeResponseInput) -> list[dict[str, Any]]:
        """Resolve every reference once at response.create acceptance time."""
        for raw_item in response_input.items:
            if raw_item.get("type") != "item_reference":
                continue
            item_id = raw_item["id"]
            item = self._controller.conversation.item(item_id)
            await self._wait_message_context_if_pending(item_id, item)
        return self._project_response_input(response_input)

    def _manual_input_mode(self) -> bool:
        session = self._controller.public_session()
        audio = session.get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        return isinstance(input_audio, dict) and input_audio.get("turn_detection") is None

    async def _finalize_conversation_append(
        self,
        item_id: str,
        context_message: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> None:
        """Publish an item only after its exact model-context object is durable."""
        transaction = self._pending_conversation_appends.get(item_id)
        if transaction is None:
            self.notify_connection_closed()
            raise RuntimeError("Realtime conversation append has no pending transaction")
        if self._llm_context is None:
            self.notify_connection_closed()
            raise RuntimeError("Realtime conversation append lost its canonical LLM context")

        context_snapshot = self._snapshot_conversation_context()
        committed = False
        try:
            self._llm_context.add_message(context_message)
            self.bind_conversation_context_message(item_id, context_message)
            committed = True
            try:
                await self._emit_events(events)
            except BaseException:
                self.notify_connection_closed()
                raise
            self._pending_conversation_appends.pop(item_id, None)
            transaction.publication_done.set()
        except BaseException as exc:
            if not committed:
                try:
                    self._restore_conversation_transaction(
                        controller_snapshot=transaction.rollback_state,
                        context_snapshot=context_snapshot,
                        cause=exc,
                        expected_controller_state=transaction.provisional_state,
                    )
                finally:
                    # A later accepted item can depend on this provisional tail.
                    # Retire the connection even after exact rollback rather than
                    # letting dependent pipeline frames observe a removed item.
                    self.notify_connection_closed()
            raise

    async def _finalize_client_tool_output(self, frame: RealtimeClientToolOutputFrame) -> None:
        """Apply one staged output without occupying the WebSocket reader."""
        broker = self._client_tool_broker
        if broker is None:
            await self._emit_event(
                RealtimeProtocolError(
                    message="Client-owned tools are not bound to this Realtime transport",
                    code="client_tool_broker_missing",
                    param="item.call_id",
                    event_id=frame.event_id,
                    error_type="server_error",
                ).to_event()
            )
            return

        output_recorded = False
        released = False
        try:
            await self._wait_for_response_done_published(
                call_id=frame.call_id,
                response_id=frame.response_id,
            )
            current_record = self._controller.tool_call(frame.call_id)
            if current_record is None:
                raise RealtimeProtocolError(
                    message=f"Function call {frame.call_id!r} was not found",
                    code="call_not_found",
                    param="item.call_id",
                    error_type="server_error",
                )
            add_kwargs = {
                "call_id": frame.call_id,
                "output": frame.output,
                "owner": "client",
                "item_id": frame.item_id,
                "previous_item_id": frame.previous_item_id,
                "previous_item_id_supplied": frame.previous_item_id_supplied,
            }
            if current_record.response_status != "completed":
                await broker.discard_staged_output(
                    call_id=frame.call_id,
                    name=frame.tool_name,
                )
                # Let the controller produce the canonical terminal-response
                # protocol error for the client event that supplied the output.
                self._controller.add_function_output(**add_kwargs)
                raise RuntimeError("A non-completed response accepted a client tool output")

            events = self._controller.add_function_output(**add_kwargs)
            output_recorded = True
            await self._emit_events(events)
            await broker.release_output(
                call_id=frame.call_id,
                name=frame.tool_name,
            )
            released = True
            await broker.wait_context_applied(
                frame.call_id,
                timeout=_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS,
            )
        except asyncio.CancelledError:
            if output_recorded:
                self.notify_connection_closed()
            elif not released:
                await self._discard_staged_client_output(frame)
            if output_recorded:
                await self._retire_failed_client_tool_connection()
            raise
        except RealtimeProtocolError as exc:
            if output_recorded:
                self.notify_connection_closed()
            elif not released:
                await self._discard_staged_client_output(frame)
            try:
                await self._emit_event(exc.with_event_id(frame.event_id).to_event())
            finally:
                if output_recorded:
                    await self._retire_failed_client_tool_connection()
        except Exception:
            if output_recorded:
                self.notify_connection_closed()
            elif not released:
                await self._discard_staged_client_output(frame)
            logger.exception(f"Unexpected Realtime client-tool output failure call_id={frame.call_id}")
            try:
                await self._emit_event(
                    RealtimeProtocolError(
                        message="The client tool output could not be applied",
                        code="client_tool_output_failed",
                        param="item.call_id",
                        event_id=frame.event_id,
                        error_type="server_error",
                    ).to_event()
                )
            finally:
                if output_recorded:
                    await self._retire_failed_client_tool_connection()

    async def _retire_failed_client_tool_connection(self) -> None:
        """Close after a public tool output cannot become model context."""
        self.notify_connection_closed()
        await self._request_failure_close("client tool context failed")

    async def _publish_failure_and_close(self, event: dict[str, Any]) -> None:
        """Best-effort one terminal error before physically closing poisoned state."""
        self.notify_connection_closed()
        try:
            await self._emit_event(event)
        except Exception as exc:
            logger.debug("Realtime terminal error publication failed error_type={}", type(exc).__name__)
        finally:
            await self._request_failure_close("realtime state transition failed")

    async def _request_failure_close(self, reason: str) -> None:
        """Invoke the transport failure close at most once for this connection."""
        if self._failure_close_requested:
            return
        self._failure_close_requested = True
        if self._connection_failure_handler is not None:
            await self._connection_failure_handler(reason)

    async def _discard_staged_client_output(self, frame: RealtimeClientToolOutputFrame) -> None:
        broker = self._client_tool_broker
        if broker is None:
            return
        try:
            await broker.discard_staged_output(
                call_id=frame.call_id,
                name=frame.tool_name,
            )
        except RealtimeProtocolError:
            logger.debug(f"Client tool output cleanup raced terminal state call_id={frame.call_id}")

    async def _activate_deferred_response(
        self,
        frame: RealtimeDeferredResponseCreateFrame,
        activation_generation: int,
    ) -> str | None:
        """Start a response after its preceding response and input are durable."""
        try:
            if self._response_gate is None:
                raise RealtimeProtocolError(
                    message="Deferred response activation is not bound to its response gate",
                    code="response_gate_missing",
                    param="response",
                    error_type="server_error",
                )
            slot_available = await self._response_gate.wait_for_deferred_response_slot(
                frame.request_id,
                activation_generation=activation_generation,
            )
            if not slot_available:
                return None
            if self._client_tool_broker is not None:
                for call_id in frame.tool_call_ids:
                    await self._client_tool_broker.wait_context_applied(
                        call_id,
                        timeout=_CLIENT_TOOL_CONTEXT_TIMEOUT_SECS,
                    )
            for item_id in frame.conversation_item_ids:
                item = self._controller.conversation.item(item_id)
                await self._wait_message_context_if_pending(item_id, item)
                if item_id not in self._context_messages_by_item_id:
                    raise RealtimeProtocolError(
                        message=f"Conversation item {item_id!r} did not reach model context",
                        code="conversation_context_not_applied",
                        param="response",
                        error_type="server_error",
                    )
            for call_id in frame.tool_call_ids:
                record = self._controller.tool_call(call_id)
                if record is None:
                    raise RealtimeProtocolError(
                        message=f"Function call {call_id!r} was not found",
                        code="call_not_found",
                        param="response",
                    )
                if record.response_status != "completed" or not record.completed:
                    raise RealtimeProtocolError(
                        message=f"Function call {call_id!r} did not produce an applicable output",
                        code="tool_call_not_active",
                        param="response",
                    )
            pending_calls = self._controller.pending_tool_call_ids()
            if pending_calls:
                raise RealtimeProtocolError(
                    message=f"Function call {pending_calls[0]!r} requires a correlated output before response.create",
                    code="tool_output_pending",
                    param="response",
                )
            if self._client_tool_broker is not None:
                pending_context = self._client_tool_broker.pending_context_call_ids()
                if pending_context:
                    raise RealtimeProtocolError(
                        message=f"Function call {pending_context[0]!r} has not reached the pipeline context",
                        code="tool_output_not_applied",
                        param="response",
                    )
            async with self._controller.response_transition_lock:
                self._response_gate.validate_activation_generation(activation_generation)
                if self._controller.response_in_progress:
                    active_response_id = self._controller.active_response_id
                    raise RealtimeProtocolError(
                        message=f"Response {active_response_id} is already in progress",
                        code="response_in_progress",
                        param="response",
                    )
                events = self._controller.start_response(
                    metadata=copy.deepcopy(frame.response_metadata),
                    tools=copy.deepcopy(frame.public_tools),
                    mcp_pipeline_names=frame.mcp_pipeline_names,
                    client_tool_bindings=frame.client_tool_bindings,
                    instructions=frame.instructions,
                    max_output_tokens=frame.max_output_tokens,
                    parallel_tool_calls=frame.parallel_tool_calls,
                    output_modalities=copy.deepcopy(frame.output_modalities),
                    audio_output=copy.deepcopy(frame.audio_output),
                )
                self._refresh_active_response_output_audio()
                response_id = self._controller.active_response_id or ""
                try:
                    await self._emit_events(events)
                except BaseException as exc:
                    await self._terminalize_failed_response_start(
                        response_id,
                        cancelled=isinstance(exc, asyncio.CancelledError),
                    )
                    raise
                if self._connection_closed:
                    await self._terminalize_failed_response_start(response_id, cancelled=True)
                    return None
                return response_id
        except asyncio.CancelledError:
            raise
        except RealtimeProtocolError as exc:
            await self._emit_event(exc.with_event_id(frame.event_id).to_event())
        except Exception:
            logger.exception("Unexpected deferred Realtime response failure")
            await self._emit_event(
                RealtimeProtocolError(
                    message="The deferred Realtime response could not be started",
                    code="response_start_failed",
                    param="response",
                    event_id=frame.event_id,
                    error_type="server_error",
                ).to_event()
            )
        return None

    async def _terminalize_failed_response_start(self, response_id: str, *, cancelled: bool) -> None:
        """Close the exact response whose created batch could not finish publishing."""
        if not response_id or self._controller.active_response_id != response_id:
            return
        if self._connection_closed:
            self._controller.abandon_response(response_id, reason="connection_closed")
            self._restore_session_output_audio()
            if self._response_gate is not None:
                self._response_gate.finish_response(response_id)
            return
        events = self._controller.finish_response(
            status="failed",
            reason="response_start_cancelled" if cancelled else "response_start_failed",
        )
        if self._response_gate is not None:
            self._response_gate.finish_response(response_id)
        publication = asyncio.create_task(self._emit_events(events))
        try:
            await asyncio.shield(publication)
        except asyncio.CancelledError:
            with suppress(asyncio.CancelledError):
                await publication

    async def _emit_events(self, events: list[dict[str, Any]]) -> None:
        if self._emit_batch is None:
            logger.warning("Realtime emitter is not registered; dropping lifecycle events")
            return
        await emit_events(self._emit_batch, events)

    async def _wait_for_response_done_published(self, *, call_id: str, response_id: str) -> None:
        """Wait for the originating response terminal event's actual wire send."""
        if self._connection_closed:
            raise RealtimeProtocolError(
                message="The Realtime connection closed before Response A was published",
                code="client_tool_broker_closed",
                param="item.call_id",
                error_type="server_error",
            )
        record = self._controller.tool_call(call_id)
        if record is None:
            raise RealtimeProtocolError(
                message=f"Function call {call_id!r} was not found",
                code="call_not_found",
                param="item.call_id",
                error_type="server_error",
            )
        if record.response_done_published:
            return
        terminal_event = self._response_done_events.setdefault(response_id, asyncio.Event())
        # No await occurs between the first check and event registration, but
        # rechecking keeps this safe if notification behavior changes later.
        record = self._controller.tool_call(call_id)
        if record is not None and record.response_done_published:
            self._response_done_events.pop(response_id, None)
            return
        if self._connection_closed:
            self._response_done_events.pop(response_id, None)
            raise RealtimeProtocolError(
                message="The Realtime connection closed before Response A was published",
                code="client_tool_broker_closed",
                param="item.call_id",
                error_type="server_error",
            )
        await terminal_event.wait()
        if self._connection_closed:
            raise RealtimeProtocolError(
                message="The Realtime connection closed before Response A was published",
                code="client_tool_broker_closed",
                param="item.call_id",
                error_type="server_error",
            )

    async def _emit_event(self, event: dict[str, Any]) -> None:
        await self._emit_events([event])


def _required_item_id(message: dict[str, Any]) -> str:
    item_id = message.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise invalid_value("item_id is required", param="item_id")
    return item_id


def _reject_unknown_event_fields(message: dict[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(message) - allowed)
    if unknown:
        field = unknown[0]
        raise RealtimeProtocolError(
            message=f"Unknown parameter: {field}",
            code="unknown_parameter",
            param=field,
        )


def _extract_item_text(item: dict[str, Any]) -> str | None:
    content = item.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            return None
        if part.get("type") != "input_text" or not isinstance(part.get("text"), str):
            return None
        parts.append(part["text"])
    return "".join(parts) if parts else None
