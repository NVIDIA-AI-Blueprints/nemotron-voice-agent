# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Per-connection canonical Realtime session and lifecycle controller."""

from __future__ import annotations

import asyncio
import copy
import json
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

from realtime.conversation import ConversationJournal, ConversationJournalSnapshot, ResponseLifecycleLedger
from realtime.protocol import (
    RealtimeProtocolError,
    build_server_event,
    invalid_type,
    invalid_value,
    new_realtime_id,
)
from realtime.session import CanonicalRealtimeSession, RealtimeSessionCapabilities

ToolOwner = Literal["client", "server", "delegate", "mcp"]
OutputKind = Literal["audio", "text"]
ResponseTerminalStatus = Literal["completed", "cancelled", "failed", "incomplete"]
ResponseCancellationReason = Literal["client_cancelled", "turn_detected"]
OutputVoiceResolver = Callable[[], Awaitable[frozenset[str]]]
_DEFAULT_SERVER_VAD_PREFIX_PADDING_MS = 300


def _transcript_text(parts: list[TextPartForConcatenation]) -> str:
    """Join transcript parts with Pipecat's canonical aggregation rules."""
    return concatenate_aggregated_text(parts)


@dataclass(slots=True)
class _UserTranscriptState:
    """Ordered transcript segments plus the pipeline's explicit turn barrier."""

    parts: list[TextPartForConcatenation]
    transcript_observed: bool = False
    turn_finalized: bool = False
    producer_ended: bool = False
    producer_failure_code: str | None = None
    producer_failure_message: str | None = None

    def append(self, text: str, *, includes_inter_frame_spaces: bool) -> None:
        if not text.strip():
            return
        self.transcript_observed = True
        self.parts.append(
            TextPartForConcatenation(
                text=text,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
        )

    def finalize_turn(self) -> None:
        """Record Pipecat's semantic user-turn barrier."""
        self.turn_finalized = True

    def end_producer(self, *, code: str, message: str) -> None:
        """Record that the fused producer cannot emit another transcript frame."""
        self.producer_ended = True
        self.producer_failure_code = code
        self.producer_failure_message = message

    @property
    def ready(self) -> bool:
        """Return whether both transcript data and a turn barrier were observed."""
        return self.turn_finalized and self.transcript_observed

    @property
    def failure_ready(self) -> bool:
        """Return whether the turn ended without any possible transcript data."""
        return self.turn_finalized and self.producer_ended and not self.ready

    @property
    def text(self) -> str:
        """Return the canonical concatenation of all final segments."""
        return _transcript_text(self.parts)


@dataclass(frozen=True, slots=True)
class AssistantAudioTruncation:
    """One validated projection of assistant audio onto heard transcript text."""

    transcript: str
    context_text: str
    previous_context_text: str


@dataclass(frozen=True, slots=True)
class IdleTimeoutTurn:
    """One atomically reserved server-VAD empty-audio turn and response."""

    item_id: str
    response_id: str
    context_message: dict[str, Any]
    events: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class _InterruptionRecord:
    """Stable ownership captured when an interruption first enters the pipeline."""

    generation: int
    response_id: str | None
    reason: ResponseCancellationReason


@dataclass(frozen=True, slots=True)
class _SessionRuntimeState:
    """Detached state required to roll back one live session transaction."""

    session_view: dict[str, Any]
    runtime_config: dict[str, Any]
    client_tool_bindings: dict[str, str]
    mcp_pipeline_names: frozenset[str]
    client_tool_projection_bound: bool


@dataclass(frozen=True, slots=True)
class _ConversationMutationState:
    """Controller-owned state changed by one conversation item mutation."""

    journal: ConversationJournalSnapshot
    assistant_audio_alignments: dict[str, _AssistantAudioAlignment]


@dataclass(frozen=True, slots=True)
class _AssistantAudioCheckpoint:
    """Transcript state known to have crossed the paced WebSocket audio boundary."""

    audio_end_ms: int
    transcript: str
    context_text: str


@dataclass(slots=True)
class _AssistantAudioAlignment:
    """Wire-audio duration and Pipecat playout checkpoints for one item."""

    sample_rate: int | None = None
    emitted_samples: int = 0
    transcript: str = ""
    context_text: str = ""
    checkpoints: list[_AssistantAudioCheckpoint] = field(default_factory=list)
    truncated_end_ms: int | None = None

    @property
    def audio_end_ms(self) -> int:
        """Return the current inclusive audio duration exposed to the model."""
        if self.truncated_end_ms is not None:
            return self.truncated_end_ms
        if self.sample_rate is None:
            return 0
        return self.emitted_samples * 1000 // self.sample_rate

    def append_audio(self, *, sample_count: int, sample_rate: int) -> None:
        """Advance duration by one successfully serialized wire-audio delta."""
        if self.truncated_end_ms is not None:
            raise RuntimeError("A truncated assistant item cannot accept more output audio")
        if sample_count <= 0 or sample_rate <= 0:
            raise ValueError("Wire audio sample_count and sample_rate must be positive")
        if self.sample_rate is None:
            self.sample_rate = sample_rate
        elif self.sample_rate != sample_rate:
            raise RuntimeError("Assistant output audio rate changed inside one response")
        self.emitted_samples += sample_count

    def append_checkpoint(self, *, transcript: str, context_text: str) -> None:
        """Record text published at the current paced wire-audio cursor."""
        if self.truncated_end_ms is not None:
            raise RuntimeError("A truncated assistant item cannot accept transcript deltas")
        checkpoint = _AssistantAudioCheckpoint(
            audio_end_ms=self.audio_end_ms,
            transcript=transcript,
            context_text=context_text,
        )
        if self.checkpoints and self.checkpoints[-1].audio_end_ms == checkpoint.audio_end_ms:
            self.checkpoints[-1] = checkpoint
        else:
            self.checkpoints.append(checkpoint)
        self.transcript = transcript
        self.context_text = context_text

    def project(self, audio_end_ms: int) -> AssistantAudioTruncation:
        """Project to text that Pipecat had published by an inclusive boundary."""
        if audio_end_ms > self.audio_end_ms:
            raise invalid_value(
                (f"audio_end_ms {audio_end_ms} exceeds the assistant audio duration of {self.audio_end_ms} ms"),
                param="audio_end_ms",
            )
        transcript = ""
        context_text = ""
        for checkpoint in self.checkpoints:
            if checkpoint.audio_end_ms > audio_end_ms:
                break
            transcript = checkpoint.transcript
            context_text = checkpoint.context_text
        return AssistantAudioTruncation(
            transcript=transcript,
            context_text=context_text,
            previous_context_text=self.context_text,
        )

    def truncate(self, *, audio_end_ms: int, projection: AssistantAudioTruncation) -> None:
        """Commit a previously validated audio/text projection."""
        self.checkpoints = [checkpoint for checkpoint in self.checkpoints if checkpoint.audio_end_ms <= audio_end_ms]
        self.transcript = projection.transcript
        self.context_text = projection.context_text
        self.truncated_end_ms = audio_end_ms


@dataclass(slots=True)
class ToolCallRecord:
    """Correlated state for one model-produced function call."""

    call_id: str
    item_id: str
    response_id: str
    name: str
    arguments: str
    owner: ToolOwner
    pipeline_name: str | None = None
    server_label: str | None = None
    output_index: int | None = None
    approval_request_id: str | None = None
    mcp_execution_started: bool = False
    output_item_id: str | None = None
    completed: bool = False
    response_status: ResponseTerminalStatus | None = None
    response_done_published: bool = False

    @property
    def retired(self) -> bool:
        """Return whether Response A terminated without accepting an output."""
        # Native MCP work has its own lifecycle and may finish after any
        # terminal Response A status. Function/delegation outputs remain bound
        # to a successfully completed Response A.
        return self.owner != "mcp" and self.response_status in {"cancelled", "failed", "incomplete"}


class RealtimeSessionController:
    """Own all public state for one canonical Realtime WebSocket connection."""

    def __init__(
        self,
        *,
        model: str,
        voice: str,
        runtime_config: dict[str, Any],
        instructions: str = "",
        max_output_tokens: int | str = "inf",
        input_transcription_model: str | None = None,
        server_tools: list[str] | None = None,
        delegate_tools: list[str] | None = None,
        trusted_tool_schemas: list[dict[str, Any]] | None = None,
        capabilities: RealtimeSessionCapabilities | None = None,
        output_voice_resolver: OutputVoiceResolver | None = None,
        output_voices_resolved: bool = True,
    ) -> None:
        """Create a controller from trusted, already-sanitized server config."""
        self.session = CanonicalRealtimeSession(
            model=model,
            voice=voice,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            input_transcription_model=input_transcription_model,
            trusted_tools=trusted_tool_schemas,
            capabilities=capabilities,
        )
        self.conversation = ConversationJournal()
        self.responses = ResponseLifecycleLedger(session=self.session, conversation=self.conversation)
        self.runtime_config = copy.deepcopy(runtime_config)
        self.server_tools = frozenset(server_tools or [])
        self.delegate_tools = frozenset(delegate_tools or [])
        self._output_voice_resolver = output_voice_resolver
        self._output_voices_resolved = output_voices_resolved or output_voice_resolver is None
        self._output_voice_resolution_lock = asyncio.Lock()
        overlap = self.server_tools & self.delegate_tools
        if overlap:
            name = sorted(overlap)[0]
            raise ValueError(f"Tool {name!r} cannot be both server-owned and delegated")

        self._assistant_item_id: str | None = None
        self._last_assistant_item_id: str | None = None
        self._assistant_output_kind: OutputKind | None = None
        self._assistant_text = ""
        self._assistant_audio_transcript_parts: list[TextPartForConcatenation] = []
        self._assistant_audio_context_parts: list[TextPartForConcatenation] = []
        self._assistant_audio_alignments: dict[str, _AssistantAudioAlignment] = {}
        self._content_part_announced = False
        self._active_output_ids: list[str] = []
        self._active_response_tools: list[dict[str, Any]] | None = None
        self._active_output_modalities: list[str] | None = None
        self._active_audio_output: dict[str, Any] | None = None
        self._mcp_tool_bindings: dict[str, tuple[str, str]] = {}
        self._session_mcp_pipeline_names: frozenset[str] = frozenset()
        self._active_mcp_pipeline_names: frozenset[str] = frozenset()
        self._session_client_tool_bindings: dict[str, str] = {}
        self._active_client_tool_bindings: dict[str, str] = {}
        self._client_tool_projection_bound = False
        self._tool_calls: dict[str, ToolCallRecord] = {}
        self._input_audio_samples = 0
        self._input_sample_rate = 16000
        self._user_item_id: str | None = None
        self._user_item_announced = False
        self._user_turn_start_sample: int | None = None
        self._user_turn_stopped = False
        self._pending_user_transcript = _UserTranscriptState(parts=[])
        self._pending_user_interim = ""
        self._user_transcript_publication_enabled: bool | None = None
        self._emitted_user_interims: dict[str, str] = {}
        self._user_audio_duration_seconds: dict[str, float] = {}
        self._stopped_user_items: deque[str] = deque()
        self._stopped_user_transcripts: dict[str, _UserTranscriptState] = {}
        self._stopped_user_transcript_publication_enabled: dict[str, bool] = {}
        self._response_usage: dict[str, Any] | None = None
        self._pending_pipeline_response_config: dict[str, Any] | None = None
        self._pending_pipeline_response_owner: str | None = None
        self._pending_pipeline_response_generation: int | None = None
        # Response activation and interruption are concurrent pipeline/wire
        # transitions.  Their shared lock and generation provide one ordering
        # authority regardless of which Pipecat processor observes an
        # InterruptionFrame first.
        self.response_transition_lock = asyncio.Lock()
        self._interruption_generation = 0
        self._interruption_frame_ids: deque[int] = deque()
        self._interruption_records: dict[int, _InterruptionRecord] = {}
        self._response_cancel_reasons: dict[str, ResponseCancellationReason] = {}

    @property
    def id(self) -> str:
        """Return the public session ID."""
        return self.session.id

    @property
    def active_response_id(self) -> str | None:
        """Return the active response ID, if one exists."""
        return self.responses.active_response_id

    @property
    def response_in_progress(self) -> bool:
        """Return whether the default conversation has an active response."""
        return self.active_response_id is not None

    @property
    def pipeline_response_pending(self) -> bool:
        """Return whether an unowned LLM run has frozen its response defaults."""
        return self._pending_pipeline_response_config is not None

    @property
    def audio_response_in_progress_or_pending(self) -> bool:
        """Return whether a frozen response owner is configured to emit audio."""
        if self.response_in_progress:
            return self.output_kind == "audio"
        pending = self._pending_pipeline_response_config
        return isinstance(pending, dict) and pending.get("output_modalities") == ["audio"]

    @property
    def interruption_generation(self) -> int:
        """Return the connection-wide pipeline interruption generation."""
        return self._interruption_generation

    def observe_interruption(
        self,
        frame_id: int,
        *,
        reason: ResponseCancellationReason = "turn_detected",
        response_id: str | None = None,
    ) -> tuple[int, bool]:
        """Record one physical interruption once across every processor edge.

        Pipecat exposes the same system frame to the response gate and lifecycle
        observer at different times.  Deduplicating by frame ID lets delayed
        response activations compare against the first observed edge instead of
        racing two processor-local counters.
        """
        existing = self._interruption_records.get(frame_id)
        if existing is not None:
            return existing.generation, False
        self._interruption_generation += 1
        target_response_id = self.active_response_id if response_id is None else response_id
        record = _InterruptionRecord(
            generation=self._interruption_generation,
            response_id=target_response_id,
            reason=reason,
        )
        self._interruption_frame_ids.append(frame_id)
        self._interruption_records[frame_id] = record
        if target_response_id is not None:
            self._response_cancel_reasons.setdefault(target_response_id, reason)
        if len(self._interruption_frame_ids) > 512:
            expired = self._interruption_frame_ids.popleft()
            self._interruption_records.pop(expired, None)
        return record.generation, True

    def interruption_target(self, frame_id: int) -> tuple[str | None, ResponseCancellationReason]:
        """Return the response and reason captured for an observed interruption."""
        record = self._interruption_records.get(frame_id)
        if record is None:
            raise RuntimeError(f"Interruption frame {frame_id} has not been observed")
        return record.response_id, record.reason

    @property
    def manual_input_mode(self) -> bool:
        """Return whether the client, rather than server turn detection, commits audio turns."""
        audio = self.session.public_view().get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        return isinstance(input_audio, dict) and input_audio.get("turn_detection") is None

    @property
    def turn_detection_type(self) -> str | None:
        """Return the active native turn-detection type, or ``None`` for manual mode."""
        turn_detection = self.turn_detection_config
        if turn_detection is None:
            return None
        return turn_detection.get("type")

    @property
    def turn_detection_config(self) -> dict[str, Any] | None:
        """Return a detached copy of the initial turn-detection configuration."""
        audio = self.session.public_view().get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        turn_detection = input_audio.get("turn_detection") if isinstance(input_audio, dict) else None
        return copy.deepcopy(turn_detection) if isinstance(turn_detection, dict) else None

    @property
    def automatic_response_enabled(self) -> bool:
        """Whether a detected user turn should automatically start inference."""
        config = self.turn_detection_config
        return config is not None and config.get("create_response", True) is True

    @property
    def interrupt_response_enabled(self) -> bool:
        """Whether the start of user speech should interrupt active output."""
        config = self.turn_detection_config
        return config is not None and config.get("interrupt_response", True) is True

    @property
    def server_vad_prefix_padding_ms(self) -> int:
        """Return the negotiated server-VAD prefix used for public audio timing."""
        config = self.turn_detection_config
        if config is None or config.get("type") != "server_vad":
            return 0
        return int(config.get("prefix_padding_ms", _DEFAULT_SERVER_VAD_PREFIX_PADDING_MS))

    @property
    def assistant_item_id(self) -> str | None:
        """Return the active assistant message item ID."""
        return self._assistant_item_id

    @property
    def last_assistant_item_id(self) -> str | None:
        """Return the most recently terminal assistant message item ID."""
        return self._last_assistant_item_id

    def is_active_output_item(self, item_id: str) -> bool:
        """Return whether an item belongs to the response being generated."""
        return self.response_in_progress and item_id in self._active_output_ids

    @property
    def assistant_text(self) -> str:
        """Return accumulated text/transcript for the active assistant item."""
        return self._assistant_text

    @property
    def output_kind(self) -> OutputKind:
        """Return the output kind selected by the active response or session."""
        modalities = self._active_output_modalities
        if modalities is None:
            modalities = self.public_session()["output_modalities"]
        return "audio" if modalities == ["audio"] else "text"

    @property
    def active_audio_output(self) -> dict[str, Any] | None:
        """Return the active response's detached effective audio configuration."""
        return copy.deepcopy(self._active_audio_output)

    def public_session(self) -> dict[str, Any]:
        """Return the canonical session plus the NVIDIA catalog selection view."""
        view = self.session.public_view()
        for tool in view.get("tools", []):
            if isinstance(tool, dict) and tool.get("type") == "mcp":
                # MCP credentials remain connection-private. They are retained
                # in CanonicalRealtimeSession for the hosted MCP transport but
                # never echoed through session.created/session.updated or the
                # client-secret bootstrap response.
                tool.pop("authorization", None)
                tool.pop("headers", None)
        public_nvidia = {
            key: copy.deepcopy(value)
            for key, value in self.runtime_config.items()
            if key
            in {
                "pipeline_mode",
                "prompt_key",
                "llm_id",
                "thinker_llm_id",
                "asr_id",
                "tts_id",
                "model_id",
                "thinker_model_id",
                "asr_model",
                "tts_model",
                "asr_language_code",
                "tts_language_code",
            }
            and value not in (None, "")
        }
        if self.server_tools:
            public_nvidia["server_tools"] = sorted(self.server_tools)
        if self.delegate_tools:
            public_nvidia["delegate_tools"] = sorted(self.delegate_tools)
        if public_nvidia:
            view["nvidia"] = public_nvidia
        return view

    def created_events(self) -> list[dict[str, Any]]:
        """Return the two events that establish a canonical voice conversation."""
        return [
            build_server_event("session.created", session=self.public_session()),
            self.conversation.created_event(),
        ]

    def session_update_requires_output_voice_resolution(self, patch: dict[str, Any]) -> bool:
        """Return whether a valid session patch needs the full trusted voice catalog."""
        if self._output_voices_resolved or not isinstance(patch, dict):
            return False
        current = self.session.public_view()
        modalities = current["output_modalities"]
        if "output_modalities" in patch:
            requested_modalities = patch["output_modalities"]
            if requested_modalities not in (["audio"], ["text"]):
                return False
            modalities = requested_modalities
        if modalities == ["audio"]:
            return True

        audio = patch.get("audio")
        output = audio.get("output") if isinstance(audio, dict) else None
        requested_voice = output.get("voice") if isinstance(output, dict) else None
        current_voice = current["audio"]["output"]["voice"]
        return isinstance(requested_voice, str) and bool(requested_voice) and requested_voice != current_voice

    async def ensure_output_voice_capabilities(self) -> None:
        """Discover and publish the exact trusted voice catalog once per route."""
        if self._output_voices_resolved:
            return
        resolver = self._output_voice_resolver
        if resolver is None:
            raise RuntimeError("Realtime output voice discovery is not configured")
        async with self._output_voice_resolution_lock:
            if self._output_voices_resolved:
                return
            voices = await resolver()
            self.session.bind_discovered_output_voices(voices)
            self._output_voices_resolved = True

    def bind_prepared_runtime(self, runtime_config: dict[str, Any]) -> None:
        """Atomically bind server-prepared routing before pipeline handoff."""
        if not isinstance(runtime_config, dict):
            raise TypeError("Prepared Realtime runtime must be a dictionary")
        prepared = copy.deepcopy(runtime_config)
        voice = prepared.get("tts_voice_id")
        if isinstance(voice, str) and voice:
            # This method is reached only with server-sanitized runtime data.
            # Exact-route discovery may legitimately resolve a language voice
            # that was absent from the cold bootstrap cache.
            self.session.bind_discovered_output_voice(voice)
        self.runtime_config = prepared

    def session_updated_event(self) -> dict[str, Any]:
        """Return the current canonical session as ``session.updated``."""
        return build_server_event("session.updated", session=self.public_session())

    def apply_session_update(self, patch: dict[str, Any]) -> dict[str, Any]:
        """Atomically apply standard client-controlled session fields."""
        trial = copy.deepcopy(self.session)
        trial.apply_update(patch)
        self.session.apply_update(patch)
        return build_server_event("session.updated", session=self.public_session())

    def snapshot_session_runtime_state(self) -> _SessionRuntimeState:
        """Capture every controller-owned default changed by a live update."""
        return _SessionRuntimeState(
            session_view=self.session.public_view(),
            runtime_config=copy.deepcopy(self.runtime_config),
            client_tool_bindings=copy.deepcopy(self._session_client_tool_bindings),
            mcp_pipeline_names=frozenset(self._session_mcp_pipeline_names),
            client_tool_projection_bound=self._client_tool_projection_bound,
        )

    def restore_session_runtime_state(self, snapshot: _SessionRuntimeState) -> None:
        """Restore a trusted snapshot after another session-update owner fails."""
        if not isinstance(snapshot, _SessionRuntimeState):
            raise TypeError("Realtime session rollback requires a controller snapshot")
        session_view = copy.deepcopy(snapshot.session_view)
        runtime_config = copy.deepcopy(snapshot.runtime_config)
        client_tool_bindings = copy.deepcopy(snapshot.client_tool_bindings)
        mcp_pipeline_names = frozenset(snapshot.mcp_pipeline_names)
        self.session.restore_update_snapshot(session_view)
        self.runtime_config = runtime_config
        self._session_client_tool_bindings = client_tool_bindings
        self._session_mcp_pipeline_names = mcp_pipeline_names
        self._client_tool_projection_bound = snapshot.client_tool_projection_bound

    def client_tool_names(self) -> frozenset[str]:
        """Return names supplied through the standard session tools field."""
        tools = self.session.public_view().get("tools", [])
        return frozenset(
            tool["name"]
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str)
        ) - (self.server_tools | self.delegate_tools)

    def register_mcp_tool_binding(self, *, pipeline_name: str, server_label: str, name: str) -> bool:
        """Register an internal LLM name and report whether it was newly added."""
        if not all(isinstance(value, str) and value for value in (pipeline_name, server_label, name)):
            raise ValueError("MCP tool bindings require non-empty names")
        binding = (server_label, name)
        existing = self._mcp_tool_bindings.get(pipeline_name)
        if existing is not None and existing != binding:
            raise ValueError(f"MCP pipeline tool {pipeline_name!r} is already bound")
        self._mcp_tool_bindings[pipeline_name] = binding
        return existing is None

    def unregister_mcp_tool_binding(self, *, pipeline_name: str, server_label: str, name: str) -> None:
        """Rollback a newly registered internal MCP binding before it becomes active."""
        binding = (server_label, name)
        if self._mcp_tool_bindings.get(pipeline_name) != binding:
            raise ValueError(f"MCP pipeline tool {pipeline_name!r} does not have the expected binding")
        if pipeline_name in self._session_mcp_pipeline_names or pipeline_name in self._active_mcp_pipeline_names:
            raise RuntimeError(f"Active MCP pipeline tool {pipeline_name!r} cannot be unregistered")
        if any(
            record.pipeline_name == pipeline_name and not record.completed and not record.retired
            for record in self._tool_calls.values()
        ):
            raise RuntimeError(f"Pending MCP pipeline tool {pipeline_name!r} cannot be unregistered")
        del self._mcp_tool_bindings[pipeline_name]

    def bind_session_mcp_tools(self, pipeline_names: frozenset[str]) -> None:
        """Freeze the native MCP projection used by automatic responses."""
        unknown = set(pipeline_names) - self._mcp_tool_bindings.keys()
        if unknown:
            raise ValueError(f"MCP pipeline tool {sorted(unknown)[0]!r} has no binding")
        self._session_mcp_pipeline_names = frozenset(pipeline_names)

    def bind_session_tool_projection(
        self,
        *,
        client_tool_bindings: Mapping[str, str],
        mcp_pipeline_names: frozenset[str],
    ) -> None:
        """Atomically bind provider names for the session tool defaults."""
        unknown_mcp = set(mcp_pipeline_names) - self._mcp_tool_bindings.keys()
        if unknown_mcp:
            raise ValueError(f"MCP pipeline tool {sorted(unknown_mcp)[0]!r} has no binding")
        projected_clients = self._validate_client_tool_bindings(
            client_tool_bindings,
            tools=self.session.public_view().get("tools", []),
        )
        if set(projected_clients) & set(mcp_pipeline_names):
            raise ValueError("Client and MCP provider tool names must be disjoint")
        self._session_client_tool_bindings = projected_clients
        self._session_mcp_pipeline_names = frozenset(mcp_pipeline_names)
        self._client_tool_projection_bound = True

    def bind_session_runtime_projection(
        self,
        runtime_config: dict[str, Any],
        *,
        client_tool_bindings: Mapping[str, str] | None = None,
        mcp_pipeline_names: frozenset[str] | None = None,
    ) -> None:
        """Atomically publish prepared runtime defaults and optional tool names."""
        prepared_runtime = copy.deepcopy(runtime_config)
        projected_clients: dict[str, str] | None = None
        projected_mcp: frozenset[str] | None = None
        if (client_tool_bindings is None) != (mcp_pipeline_names is None):
            raise ValueError("Client and MCP session projections must be updated together")
        if client_tool_bindings is not None and mcp_pipeline_names is not None:
            unknown_mcp = set(mcp_pipeline_names) - self._mcp_tool_bindings.keys()
            if unknown_mcp:
                raise ValueError(f"MCP pipeline tool {sorted(unknown_mcp)[0]!r} has no binding")
            projected_clients = self._validate_client_tool_bindings(
                client_tool_bindings,
                tools=self.session.public_view().get("tools", []),
            )
            projected_mcp = frozenset(mcp_pipeline_names)
            if set(projected_clients) & set(projected_mcp):
                raise ValueError("Client and MCP provider tool names must be disjoint")

        self.runtime_config = prepared_runtime
        if projected_clients is not None and projected_mcp is not None:
            self._session_client_tool_bindings = projected_clients
            self._session_mcp_pipeline_names = projected_mcp
            self._client_tool_projection_bound = True

    def _validate_client_tool_bindings(
        self,
        bindings: Mapping[str, str],
        *,
        tools: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Validate one exact provider-to-public mapping without mutating state."""
        projected = dict(bindings)
        if any(not isinstance(name, str) or not name for pair in projected.items() for name in pair):
            raise ValueError("Client tool bindings require non-empty provider and public names")
        if len(set(projected.values())) != len(projected):
            raise ValueError("Each public client tool requires exactly one provider name")
        active_function_names = {
            tool["name"]
            for tool in tools
            if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str)
        }
        expected_client_names = active_function_names - (self.server_tools | self.delegate_tools)
        if not set(projected.values()).issubset(expected_client_names):
            raise ValueError("Client tool bindings reference an inactive or trusted public tool")
        if any(provider != public and provider in active_function_names for provider, public in projected.items()):
            raise ValueError("A client provider name cannot shadow another public tool")
        return projected

    def pending_tool_call_ids(self) -> tuple[str, ...]:
        """Return model calls that do not yet have a correlated terminal output."""
        return tuple(
            call_id for call_id, record in self._tool_calls.items() if not record.completed and not record.retired
        )

    def start_response(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        mcp_pipeline_names: frozenset[str] | None = None,
        client_tool_bindings: Mapping[str, str] | None = None,
        instructions: str | None = None,
        max_output_tokens: int | str | None = None,
        parallel_tool_calls: bool | None = None,
        truncation: str | dict[str, Any] | None = None,
        output_modalities: list[str] | None = None,
        audio_output: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Reserve a response and freeze its effective native tool set."""
        if self._pending_pipeline_response_config is not None:
            raise RealtimeProtocolError(
                message="A pipeline-triggered response is already starting",
                code="response_in_progress",
                param="response",
            )
        session_view = self.session.public_view()
        effective_modalities = (
            copy.deepcopy(session_view["output_modalities"])
            if output_modalities is None
            else copy.deepcopy(output_modalities)
        )
        effective_audio_output = None
        if effective_modalities == ["audio"]:
            effective_audio_output = (
                self.session.validate_response_audio_output({}) if audio_output is None else copy.deepcopy(audio_output)
            )
            self.session.validate_effective_response_voice(effective_audio_output.get("voice"))
        effective_tools = self.session.public_view().get("tools", []) if tools is None else tools
        if not isinstance(effective_tools, list):
            raise TypeError("Effective Realtime response tools must be a list")
        selected_mcp_names = self._session_mcp_pipeline_names if tools is None else frozenset()
        if mcp_pipeline_names is not None:
            selected_mcp_names = mcp_pipeline_names
        unknown = set(selected_mcp_names) - self._mcp_tool_bindings.keys()
        if unknown:
            raise ValueError(f"MCP pipeline tool {sorted(unknown)[0]!r} has no binding")
        selected_client_bindings = self._session_client_tool_bindings if tools is None else {}
        if client_tool_bindings is not None:
            selected_client_bindings = self._validate_client_tool_bindings(
                client_tool_bindings,
                tools=effective_tools,
            )
        elif self._client_tool_projection_bound:
            selected_client_bindings = self._validate_client_tool_bindings(
                selected_client_bindings,
                tools=effective_tools,
            )
        if set(selected_client_bindings) & set(selected_mcp_names):
            raise ValueError("Client and MCP provider tool names must be disjoint")
        event = self.responses.start(
            metadata=metadata,
            instructions=instructions,
            max_output_tokens=max_output_tokens,
            parallel_tool_calls=parallel_tool_calls,
            output_modalities=effective_modalities,
            audio_output=effective_audio_output,
        )
        self._active_response_tools = copy.deepcopy(effective_tools)
        self._active_output_modalities = effective_modalities
        self._active_audio_output = effective_audio_output
        self._active_mcp_pipeline_names = frozenset(selected_mcp_names)
        self._active_client_tool_bindings = dict(selected_client_bindings)
        self._assistant_item_id = None
        self._assistant_output_kind = None
        self._assistant_text = ""
        self._assistant_audio_transcript_parts = []
        self._assistant_audio_context_parts = []
        self._content_part_announced = False
        self._active_output_ids = []
        self._response_usage = None
        return [event]

    def prepare_pipeline_response(
        self,
        *,
        owner_id: str | None = None,
        interruption_generation: int | None = None,
    ) -> dict[str, Any]:
        """Freeze session defaults before an unowned model run enters the LLM."""
        if self.response_in_progress:
            raise RealtimeProtocolError(
                message=f"Response {self.active_response_id} is already in progress",
                code="response_in_progress",
                param="response",
            )
        if self._pending_pipeline_response_config is not None:
            raise RealtimeProtocolError(
                message="A pipeline-triggered response is already starting",
                code="response_in_progress",
                param="response",
            )
        session_view = self.session.public_view()
        output_modalities = copy.deepcopy(session_view["output_modalities"])
        audio_output = None
        if output_modalities == ["audio"]:
            audio_output = self.session.validate_response_audio_output({})
            self.session.validate_effective_response_voice(audio_output.get("voice"))
        snapshot = {
            "tools": copy.deepcopy(session_view.get("tools", [])),
            "mcp_pipeline_names": self._session_mcp_pipeline_names,
            "client_tool_bindings": self._session_client_tool_bindings,
            "max_output_tokens": session_view["max_output_tokens"],
            "parallel_tool_calls": session_view["parallel_tool_calls"],
            "truncation": copy.deepcopy(session_view.get("truncation")),
            "output_modalities": output_modalities,
            "audio_output": audio_output,
        }
        self._pending_pipeline_response_config = copy.deepcopy(snapshot)
        self._pending_pipeline_response_owner = owner_id
        self._pending_pipeline_response_generation = (
            self._interruption_generation if interruption_generation is None else interruption_generation
        )
        return snapshot

    def cancel_pending_pipeline_response(
        self,
        *,
        owner_id: str | None = None,
        older_than_generation: int | None = None,
    ) -> bool:
        """Retire an owned unstarted run and report whether it was still pending."""
        if owner_id is not None and owner_id != self._pending_pipeline_response_owner:
            return False
        if self._pending_pipeline_response_config is None:
            return False
        if (
            older_than_generation is not None
            and self._pending_pipeline_response_generation is not None
            and self._pending_pipeline_response_generation >= older_than_generation
        ):
            return False
        self._pending_pipeline_response_config = None
        self._pending_pipeline_response_owner = None
        self._pending_pipeline_response_generation = None
        return True

    def start_pending_pipeline_response(
        self,
        *,
        owner_id: str | None = None,
        expected_generation: int | None = None,
    ) -> tuple[str, list[dict[str, Any]]] | None:
        """Atomically activate the exact unstarted pipeline response owner."""
        if self._pending_pipeline_response_config is None or (
            owner_id is not None and self._pending_pipeline_response_owner != owner_id
        ):
            return None
        generation = self._pending_pipeline_response_generation
        if generation != self._interruption_generation or (
            expected_generation is not None and generation != expected_generation
        ):
            self.cancel_pending_pipeline_response(owner_id=owner_id)
            return None
        events = self.ensure_response()
        response_id = self.active_response_id
        if response_id is None:
            raise RuntimeError("A pending pipeline response did not acquire a response ID")
        return response_id, events

    def consume_response_cancel_reason(self, response_id: str) -> ResponseCancellationReason | None:
        """Consume the canonical cancellation cause owned by one response."""
        return self._response_cancel_reasons.pop(response_id, None)

    def ensure_response(self) -> list[dict[str, Any]]:
        """Start a server-triggered response only when no response exists."""
        if self.response_in_progress:
            return []
        pending = self._pending_pipeline_response_config
        self._pending_pipeline_response_config = None
        self._pending_pipeline_response_owner = None
        self._pending_pipeline_response_generation = None
        if pending is None:
            return self.start_response()
        return self.start_response(**pending)

    def ensure_assistant_message(self, kind: OutputKind | None = None) -> list[dict[str, Any]]:
        """Announce the response, assistant output item, and content part once."""
        events = self.ensure_response()
        selected = kind or self.output_kind
        if selected != self.output_kind:
            raise RealtimeProtocolError(
                message=f"The active session is configured for {self.output_kind} output",
                code="invalid_response_state",
                param="response.output",
            )
        if self._assistant_item_id is None:
            response_id = self._require_response_id()
            item_event = self.conversation.add_item(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            )
            item = item_event["item"]
            self._assistant_item_id = item["id"]
            self._assistant_output_kind = selected
            if selected == "audio":
                self._assistant_audio_alignments[self._assistant_item_id] = _AssistantAudioAlignment()
            self.responses.add_output_item(self._assistant_item_id)
            self._active_output_ids.append(self._assistant_item_id)
            output_index = self._response_output_index(self._assistant_item_id)
            events.extend(
                [
                    item_event,
                    build_server_event(
                        "response.output_item.added",
                        response_id=response_id,
                        output_index=output_index,
                        item=copy.deepcopy(item),
                    ),
                    build_server_event(
                        "response.content_part.added",
                        response_id=response_id,
                        item_id=self._assistant_item_id,
                        output_index=output_index,
                        content_index=0,
                        part={"type": "audio", "transcript": ""}
                        if selected == "audio"
                        else {"type": "text", "text": ""},
                    ),
                ]
            )
            self._content_part_announced = True
        return events

    def append_assistant_text(self, text: str) -> list[dict[str, Any]]:
        """Append one text/transcript delta to the active assistant message."""
        if not text:
            return []
        events = self.ensure_assistant_message()
        self._assistant_text += text
        response_id = self._require_response_id()
        event_type = (
            "response.output_audio_transcript.delta"
            if self._assistant_output_kind == "audio"
            else "response.output_text.delta"
        )
        output_index = self._response_output_index(self._assistant_item_id)
        events.append(
            build_server_event(
                event_type,
                response_id=response_id,
                item_id=self._assistant_item_id,
                output_index=output_index,
                content_index=0,
                delta=text,
            )
        )
        return events

    def append_assistant_audio_transcript(
        self,
        text: str,
        *,
        includes_inter_frame_spaces: bool,
        context_text: str | None = None,
    ) -> list[dict[str, Any]]:
        """Append one Pipecat-aggregated, playout-aligned transcript fragment."""
        if not text:
            return []
        events = self.ensure_assistant_message("audio")
        previous = _transcript_text(self._assistant_audio_transcript_parts)
        if previous != self._assistant_text:
            raise RuntimeError("Exact assistant text and aggregated audio transcript paths cannot be mixed")
        self._assistant_audio_transcript_parts.append(
            TextPartForConcatenation(
                text=text,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
        )
        transcript = _transcript_text(self._assistant_audio_transcript_parts)
        context_piece = context_text if context_text else text
        self._assistant_audio_context_parts.append(
            TextPartForConcatenation(
                text=context_piece,
                includes_inter_part_spaces=includes_inter_frame_spaces,
            )
        )
        model_context_text = _transcript_text(self._assistant_audio_context_parts)
        if not transcript.startswith(previous):
            raise RuntimeError("Assistant audio transcript aggregation must be append-only")
        delta = transcript[len(previous) :]
        self._assistant_text = transcript
        alignment = self._assistant_audio_alignments.get(self._assistant_item_id or "")
        if alignment is None:
            raise RuntimeError("Assistant audio transcript has no wire-audio alignment owner")
        alignment.append_checkpoint(
            transcript=transcript,
            context_text=model_context_text,
        )
        if delta:
            events.append(
                build_server_event(
                    "response.output_audio_transcript.delta",
                    response_id=self._require_response_id(),
                    item_id=self._assistant_item_id,
                    output_index=self._response_output_index(self._assistant_item_id),
                    content_index=0,
                    delta=delta,
                )
            )
        return events

    def output_audio_delta_event(
        self,
        delta: str,
        *,
        sample_count: int,
        sample_rate: int,
    ) -> list[dict[str, Any]]:
        """Create one canonical output-audio delta and advance its wire clock."""
        if not self.response_in_progress:
            raise RuntimeError("Output audio requires an active Realtime response")
        events = self.ensure_assistant_message("audio")
        alignment = self._assistant_audio_alignments.get(self._assistant_item_id or "")
        if alignment is None:
            raise RuntimeError("Assistant output audio has no alignment owner")
        alignment.append_audio(sample_count=sample_count, sample_rate=sample_rate)
        voice = self._active_audio_output.get("voice") if self._active_audio_output is not None else None
        self.responses.mark_output_audio_started(voice=voice)
        output_index = self._response_output_index(self._assistant_item_id)
        events.append(
            build_server_event(
                "response.output_audio.delta",
                response_id=self._require_response_id(),
                item_id=self._assistant_item_id,
                output_index=output_index,
                content_index=0,
                delta=delta,
            )
        )
        return events

    def response_audio_duration_ms(self, response: dict[str, Any]) -> int | None:
        """Return the exact emitted-audio duration for one public response.

        The duration comes from the same wire sample clock used for assistant
        truncation. ``None`` means that the response contains no output-audio
        part. A zero duration is retained for a declared but empty audio part.
        """
        output = response.get("output")
        if not isinstance(output, list):
            return None
        found_audio = False
        duration_ms = 0
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list) or not any(
                isinstance(part, dict) and part.get("type") == "output_audio" for part in content
            ):
                continue
            found_audio = True
            item_id = item.get("id")
            if not isinstance(item_id, str):
                continue
            alignment = self._assistant_audio_alignments.get(item_id)
            if alignment is not None:
                duration_ms += alignment.audio_end_ms
        return duration_ms if found_audio else None

    def start_function_call(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Any,
    ) -> list[dict[str, Any]]:
        """Add a model-produced function or internally projected MCP call."""
        if not call_id or not name:
            raise invalid_value("Function calls require call_id and name", param="response.output")
        arguments_json = arguments if isinstance(arguments, str) else json.dumps(arguments or {}, ensure_ascii=False)
        existing = self._tool_calls.get(call_id)
        if existing is not None:
            if existing.response_id != self.active_response_id:
                raise RealtimeProtocolError(
                    message=f"Function call ID {call_id!r} was already used by another response",
                    code="duplicate_tool_call_id",
                    param="response.output",
                )
            expected_pipeline_name = existing.pipeline_name or existing.name
            if expected_pipeline_name != name or existing.arguments != arguments_json:
                raise RealtimeProtocolError(
                    message=f"Function call {call_id!r} changed after it was announced",
                    code="tool_call_conflict",
                    param="response.output",
                )
            # Pipecat can surface the same finalized call through more than one
            # frame. An identical replay within Response A is idempotent.
            return []
        mcp_binding = self._mcp_tool_bindings.get(name)
        if mcp_binding is not None:
            if name not in self._active_mcp_pipeline_names:
                raise RealtimeProtocolError(
                    message="Model requested an inactive MCP tool",
                    code="unknown_tool",
                    param="response.output",
                )
            return self._start_mcp_call(
                call_id=call_id,
                pipeline_name=name,
                server_label=mcp_binding[0],
                name=mcp_binding[1],
                arguments=arguments_json,
            )

        pipeline_name = name
        projected_client_name = self._active_client_tool_bindings.get(pipeline_name)
        if projected_client_name is not None:
            name = projected_client_name

        effective_tools = (
            self._active_response_tools
            if self._active_response_tools is not None
            else self.session.public_view().get("tools", [])
        )
        active_tool_names = {
            tool["name"] for tool in effective_tools if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        if name not in active_tool_names:
            raise RealtimeProtocolError(
                message=f"Model requested inactive or unknown tool {name!r}",
                code="unknown_tool",
                param="response.output",
            )
        if name in self.delegate_tools:
            owner = "delegate"
        elif name in self.server_tools:
            owner = "server"
        else:
            # Any declared function without an exact trusted runtime binding is
            # owned by the Realtime client. This includes response-scoped tools,
            # which intentionally do not mutate ``session.tools``.
            owner: ToolOwner = "client"
            if self._client_tool_projection_bound and projected_client_name is None:
                raise RealtimeProtocolError(
                    message="Model requested an unprojected client tool",
                    code="unknown_tool",
                    param="response.output",
                )
        events = self.ensure_response()
        added = self.conversation.add_item(
            {
                "type": "function_call",
                "status": "in_progress",
                "call_id": call_id,
                "name": name,
                "arguments": "",
            }
        )
        item = added["item"]
        item_id = item["id"]
        response_id = self._require_response_id()
        self.responses.add_output_item(item_id)
        self._active_output_ids.append(item_id)
        self._tool_calls[call_id] = ToolCallRecord(
            call_id=call_id,
            item_id=item_id,
            response_id=response_id,
            name=name,
            arguments=arguments_json,
            owner=owner,
            pipeline_name=pipeline_name if projected_client_name is not None else None,
        )
        events.extend(
            [
                added,
                build_server_event(
                    "response.output_item.added",
                    response_id=response_id,
                    output_index=self._response_output_index(item_id),
                    item=copy.deepcopy(item),
                ),
                build_server_event(
                    "response.function_call_arguments.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=self._response_output_index(item_id),
                    call_id=call_id,
                    delta=arguments_json,
                ),
                build_server_event(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=self._response_output_index(item_id),
                    call_id=call_id,
                    name=name,
                    arguments=arguments_json,
                ),
            ]
        )
        done = self.conversation.complete_item(
            item_id,
            item_patch={"arguments": arguments_json},
        )
        events.extend(
            [
                done,
                build_server_event(
                    "response.output_item.done",
                    response_id=response_id,
                    output_index=self._response_output_index(item_id),
                    item=copy.deepcopy(done["item"]),
                ),
            ]
        )
        return events

    def _start_mcp_call(
        self,
        *,
        call_id: str,
        pipeline_name: str,
        server_label: str,
        name: str,
        arguments: str,
    ) -> list[dict[str, Any]]:
        """Add one native MCP call while retaining its private Pipecat ID."""
        events = self.ensure_response()
        added = self.conversation.add_item(
            {
                "id": new_realtime_id("item"),
                "type": "mcp_call",
                "status": "in_progress",
                "server_label": server_label,
                "name": name,
                "arguments": "",
            }
        )
        item = added["item"]
        item_id = item["id"]
        response_id = self._require_response_id()
        self.responses.add_output_item(item_id)
        self._active_output_ids.append(item_id)
        output_index = self._response_output_index(item_id)
        self._tool_calls[call_id] = ToolCallRecord(
            call_id=call_id,
            item_id=item_id,
            response_id=response_id,
            name=name,
            arguments=arguments,
            owner="mcp",
            pipeline_name=pipeline_name,
            server_label=server_label,
            output_index=output_index,
        )
        events.extend(
            [
                added,
                build_server_event(
                    "response.output_item.added",
                    response_id=response_id,
                    output_index=output_index,
                    item=copy.deepcopy(item),
                ),
                build_server_event(
                    "response.mcp_call_arguments.delta",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    delta=arguments,
                ),
                build_server_event(
                    "response.mcp_call_arguments.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=output_index,
                    arguments=arguments,
                ),
            ]
        )
        self.conversation.patch_item(item_id, {"arguments": arguments})
        return events

    def request_mcp_approval(self, call_id: str) -> tuple[str, list[dict[str, Any]]]:
        """Create a completed approval-request item for one active MCP call."""
        record = self._require_mcp_call(call_id)
        if record.approval_request_id is not None:
            raise RealtimeProtocolError(
                message=f"MCP call {record.item_id!r} already requested approval",
                code="duplicate_mcp_approval_request",
                param="item_id",
            )
        approval_id = new_realtime_id("item")
        record.approval_request_id = approval_id
        self.conversation.patch_item(record.item_id, {"approval_request_id": approval_id})
        events = self.create_conversation_item(
            {
                "id": approval_id,
                "type": "mcp_approval_request",
                "server_label": record.server_label,
                "name": record.name,
                "arguments": record.arguments,
            }
        )
        return approval_id, events

    def mark_mcp_call_in_progress(self, call_id: str) -> list[dict[str, Any]]:
        """Emit the native execution-start signal exactly once."""
        record = self._require_mcp_call(call_id)
        if record.mcp_execution_started:
            return []
        if record.retired:
            return []
        record.mcp_execution_started = True
        return [
            build_server_event(
                "response.mcp_call.in_progress",
                item_id=record.item_id,
                output_index=record.output_index,
            )
        ]

    def complete_mcp_call(
        self,
        call_id: str,
        *,
        output: str | None,
        error: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Complete one MCP output item after its originating response may end."""
        record = self._require_mcp_call(call_id)
        if record.retired:
            return []
        if record.completed:
            raise RealtimeProtocolError(
                message=f"MCP call {record.item_id!r} already completed",
                code="duplicate_tool_output",
                param="item_id",
            )
        if (output is None) == (error is None):
            raise ValueError("MCP completion requires exactly one of output or error")
        patch: dict[str, Any] = {
            "approval_request_id": record.approval_request_id,
            "output": output,
            "error": copy.deepcopy(error),
        }
        done = self.conversation.complete_item(
            record.item_id,
            status="incomplete" if error is not None else "completed",
            item_patch=patch,
        )
        record.completed = True
        terminal_type = "response.mcp_call.failed" if error is not None else "response.mcp_call.completed"
        return [
            build_server_event(
                terminal_type,
                item_id=record.item_id,
                output_index=record.output_index,
            ),
            done,
            build_server_event(
                "response.output_item.done",
                response_id=record.response_id,
                output_index=record.output_index,
                item=copy.deepcopy(done["item"]),
            ),
        ]

    def _require_mcp_call(self, call_id: str) -> ToolCallRecord:
        record = self._tool_calls.get(call_id)
        if record is None or record.owner != "mcp":
            raise RealtimeProtocolError(
                message=f"MCP call {call_id!r} was not found",
                code="call_not_found",
                param="item_id",
            )
        return record

    def add_function_output(
        self,
        *,
        call_id: str,
        output: str,
        owner: ToolOwner,
        item_id: str | None = None,
        previous_item_id: str | None = None,
        previous_item_id_supplied: bool = False,
    ) -> list[dict[str, Any]]:
        """Add exactly one correlated function_call_output conversation item."""
        record = self._tool_calls.get(call_id)
        if record is None:
            raise RealtimeProtocolError(
                message=f"Function call {call_id!r} was not found",
                code="call_not_found",
                param="item.call_id",
            )
        if record.retired:
            raise RealtimeProtocolError(
                message=(
                    f"Function call {call_id!r} belongs to a {record.response_status} response "
                    "and cannot accept an output"
                ),
                code="tool_call_not_active",
                param="item.call_id",
            )
        if record.owner != owner:
            raise RealtimeProtocolError(
                message=f"Function call {call_id!r} is owned by {record.owner}",
                code="tool_owner_mismatch",
                param="item.call_id",
            )
        if record.completed:
            raise RealtimeProtocolError(
                message=f"Function call {call_id!r} already has an output",
                code="duplicate_tool_output",
                param="item.call_id",
            )
        if record.response_status != "completed":
            raise RealtimeProtocolError(
                message=f"Function output {call_id!r} cannot be added before Response A completes",
                code="tool_output_before_response_done",
                param="item.call_id",
            )
        if not isinstance(output, str):
            raise invalid_type("function_call_output.output must be a string", param="item.output")
        self._require_tail_append(
            previous_item_id=previous_item_id,
            previous_item_id_supplied=previous_item_id_supplied,
        )
        added = self.conversation.add_item(
            {
                **({"id": item_id} if item_id else {}),
                "type": "function_call_output",
                "status": "completed",
                "call_id": call_id,
                "output": output,
            }
        )
        done = self.conversation.terminal_item_done_event(added["item"]["id"])
        record.output_item_id = added["item"]["id"]
        record.completed = True
        return [added, done]

    def finish_response(
        self,
        *,
        status: ResponseTerminalStatus,
        reason: str | None = None,
    ) -> list[dict[str, Any]]:
        """Finish active output items, then emit one terminal response.done."""
        if not self.response_in_progress:
            return []
        response_id = self._require_response_id()
        cancel_reason = self._response_cancel_reasons.get(response_id)
        if cancel_reason is not None:
            status = "cancelled"
            reason = cancel_reason
        details = _response_status_details(status=status, reason=reason)
        events: list[dict[str, Any]] = []
        if self._assistant_item_id is not None:
            item_id = self._assistant_item_id
            kind = self._assistant_output_kind or self.output_kind
            item_status = "completed" if status == "completed" else "incomplete"
            if kind == "audio":
                events.extend(
                    [
                        build_server_event(
                            "response.output_audio.done",
                            response_id=response_id,
                            item_id=item_id,
                            output_index=self._response_output_index(item_id),
                            content_index=0,
                        ),
                        build_server_event(
                            "response.output_audio_transcript.done",
                            response_id=response_id,
                            item_id=item_id,
                            output_index=self._response_output_index(item_id),
                            content_index=0,
                            transcript=self._assistant_text,
                        ),
                    ]
                )
                content = {"type": "output_audio", "transcript": self._assistant_text}
                event_part = {"type": "audio", "transcript": self._assistant_text}
            else:
                events.append(
                    build_server_event(
                        "response.output_text.done",
                        response_id=response_id,
                        item_id=item_id,
                        output_index=self._response_output_index(item_id),
                        content_index=0,
                        text=self._assistant_text,
                    )
                )
                content = {"type": "output_text", "text": self._assistant_text}
                event_part = {"type": "text", "text": self._assistant_text}
            events.append(
                build_server_event(
                    "response.content_part.done",
                    response_id=response_id,
                    item_id=item_id,
                    output_index=self._response_output_index(item_id),
                    content_index=0,
                    part=event_part,
                )
            )
            item_done = self.conversation.complete_item(
                item_id,
                status=item_status,
                item_patch={"content": [content]},
            )
            events.extend(
                [
                    item_done,
                    build_server_event(
                        "response.output_item.done",
                        response_id=response_id,
                        output_index=self._response_output_index(item_id),
                        item=copy.deepcopy(item_done["item"]),
                    ),
                ]
            )
        events.append(
            self.responses.finish(
                status=status,
                status_details=details,
                usage=copy.deepcopy(self._response_usage),
            )
        )
        self._terminalize_response_tool_calls(response_id=response_id, status=status)
        if self._assistant_item_id is not None:
            self._last_assistant_item_id = self._assistant_item_id
        self._assistant_item_id = None
        self._assistant_output_kind = None
        self._assistant_text = ""
        self._assistant_audio_transcript_parts = []
        self._content_part_announced = False
        self._active_output_ids = []
        self._active_response_tools = None
        self._active_output_modalities = None
        self._active_audio_output = None
        self._active_mcp_pipeline_names = frozenset()
        self._active_client_tool_bindings = {}
        self._response_usage = None
        self._response_cancel_reasons.pop(response_id, None)
        return events

    def abandon_response(self, response_id: str, *, reason: str) -> bool:
        """Terminalize an exact response locally when no wire event can be delivered."""
        if self.active_response_id != response_id:
            return False
        self._response_cancel_reasons.pop(response_id, None)
        self.finish_response(status="failed", reason=reason)
        return True

    def create_conversation_item(
        self,
        item: dict[str, Any],
        *,
        previous_item_id: str | None = None,
        previous_item_id_supplied: bool = False,
    ) -> list[dict[str, Any]]:
        """Add and complete a client-created conversation item."""
        self._require_tail_append(
            previous_item_id=previous_item_id,
            previous_item_id_supplied=previous_item_id_supplied,
        )
        terminal_status = item.get("status", "completed")
        if terminal_status not in {"completed", "incomplete"}:
            terminal_status = "completed"
        working_item = copy.deepcopy(item)
        working_item["status"] = terminal_status
        if previous_item_id_supplied:
            if previous_item_id == "root":
                added = self.conversation.add_item(working_item, previous_item_id=None)
            elif isinstance(previous_item_id, str) and previous_item_id:
                added = self.conversation.add_item(working_item, previous_item_id=previous_item_id)
            else:
                raise invalid_value(
                    "previous_item_id must be 'root' or an existing item ID",
                    param="previous_item_id",
                )
        else:
            added = self.conversation.add_item(working_item)
        return [
            added,
            self.conversation.terminal_item_done_event(added["item"]["id"]),
        ]

    def start_idle_timeout_turn(
        self,
        *,
        audio_start_ms: int,
        audio_end_ms: int,
    ) -> IdleTimeoutTurn:
        """Commit native empty input audio and reserve its automatic response."""
        for name, value in (("audio_start_ms", audio_start_ms), ("audio_end_ms", audio_end_ms)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if audio_end_ms < audio_start_ms:
            raise ValueError("audio_end_ms must be greater than or equal to audio_start_ms")
        config = self.turn_detection_config
        if not isinstance(config, dict) or config.get("type") != "server_vad" or config.get("idle_timeout_ms") is None:
            raise RuntimeError("A server-VAD idle timeout is not active for this session")
        if self.response_in_progress:
            raise RuntimeError("A Realtime response is already in progress")

        item_events = self.create_conversation_item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_audio", "transcript": None}],
            }
        )
        item_id = item_events[0]["item"]["id"]
        previous_item_id = item_events[0].get("previous_item_id")
        events = [
            build_server_event(
                "input_audio_buffer.timeout_triggered",
                item_id=item_id,
                audio_start_ms=audio_start_ms,
                audio_end_ms=audio_end_ms,
            ),
            build_server_event(
                "input_audio_buffer.committed",
                item_id=item_id,
                previous_item_id=previous_item_id,
            ),
            *item_events,
        ]
        if self._input_transcription_enabled():
            self.conversation.patch_item(
                item_id,
                {"content": [{"type": "input_audio", "transcript": ""}]},
            )
            events.append(
                build_server_event(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=item_id,
                    content_index=0,
                    transcript="",
                    usage={
                        "type": "duration",
                        "seconds": (audio_end_ms - audio_start_ms) / 1000,
                    },
                )
            )
        response_events = self.start_response()
        response_id = self.active_response_id
        if response_id is None:
            raise RuntimeError("The idle timeout did not reserve a Realtime response")
        events.extend(response_events)
        return IdleTimeoutTurn(
            item_id=item_id,
            response_id=response_id,
            context_message={"role": "user", "content": ""},
            events=events,
        )

    def set_response_usage(self, usage: dict[str, Any] | None) -> None:
        """Set the most recent standard token-usage projection."""
        self._response_usage = copy.deepcopy(usage)

    def set_input_audio_cursor(self, sample_count: int, sample_rate: int) -> None:
        """Snapshot the pipeline-observed input clock without reading ahead."""
        self._input_sample_rate = max(int(sample_rate), 1)
        self._input_audio_samples = max(0, int(sample_count))

    def begin_user_turn(
        self,
        *,
        new_turn: bool = False,
        audio_sample_cursor: int | None = None,
        sample_rate: int | None = None,
    ) -> tuple[str, int]:
        """Allocate an input-audio item and return its speech start position."""
        cursor = self._input_audio_samples if audio_sample_cursor is None else max(0, int(audio_sample_cursor))
        if sample_rate is not None:
            self._input_sample_rate = max(1, int(sample_rate))
        if new_turn and self._user_item_id is not None:
            self.clear_user_turn()
        if self._user_item_id is None:
            self._user_item_id = new_realtime_id("item")
        if self._user_turn_start_sample is None:
            self._user_turn_start_sample = cursor
            self._user_turn_stopped = False
            self._user_item_announced = False
            self._pending_user_transcript = _UserTranscriptState(parts=[])
            self._pending_user_interim = ""
            self._user_transcript_publication_enabled = self._input_transcription_enabled()
        return self._user_item_id, self._input_ms(self._user_turn_start_sample)

    def announce_user_item(self) -> list[dict[str, Any]]:
        """Add the active user audio item before any transcript event."""
        item_id, _ = self.begin_user_turn()
        if self._user_item_announced:
            return []
        added = self.conversation.add_item(
            {
                "id": item_id,
                "type": "message",
                "role": "user",
                "status": "in_progress",
                "content": [{"type": "input_audio", "transcript": None}],
            }
        )
        self._user_item_announced = True
        return [added]

    def _input_transcription_enabled(self) -> bool:
        """Return whether this session advertises an input transcript producer."""
        audio = self.session.public_view().get("audio")
        input_audio = audio.get("input") if isinstance(audio, dict) else None
        return isinstance(input_audio, dict) and input_audio.get("transcription") is not None

    def _input_transcription_enabled_for_item(self, item_id: str | None) -> bool:
        """Return the immutable publication policy captured for one audio turn."""
        if item_id is not None and item_id in self._stopped_user_transcript_publication_enabled:
            return self._stopped_user_transcript_publication_enabled[item_id]
        if self._user_item_id is not None and (item_id is None or item_id == self._user_item_id):
            return bool(self._user_transcript_publication_enabled)
        return self._input_transcription_enabled()

    def _complete_user_transcript(
        self,
        item_id: str,
        state: _UserTranscriptState,
    ) -> list[dict[str, Any]]:
        """Patch one committed item and emit its single terminal transcript event."""
        transcription_enabled = self._input_transcription_enabled_for_item(item_id)
        if item_id in self._stopped_user_transcripts:
            self._stopped_user_transcripts.pop(item_id, None)
            with suppress(ValueError):
                self._stopped_user_items.remove(item_id)
        self._stopped_user_transcript_publication_enabled.pop(item_id, None)
        self._emitted_user_interims.pop(item_id, None)
        if not transcription_enabled:
            self._user_audio_duration_seconds.pop(item_id, None)
            return []
        item = self.conversation.item(item_id)
        content = copy.deepcopy(item.get("content") or [])
        if content and isinstance(content[0], dict):
            content[0]["transcript"] = state.text
        self.conversation.patch_item(item_id, {"content": content})
        return [
            build_server_event(
                "conversation.item.input_audio_transcription.completed",
                item_id=item_id,
                content_index=0,
                transcript=state.text,
                usage=self._take_transcription_usage(item_id),
            )
        ]

    def _fail_user_transcript_item(
        self,
        item_id: str,
        *,
        code: str,
        message: str,
    ) -> list[dict[str, Any]]:
        """Release one transcript item and emit its single terminal failure."""
        transcription_enabled = self._input_transcription_enabled_for_item(item_id)
        self._stopped_user_transcripts.pop(item_id, None)
        with suppress(ValueError):
            self._stopped_user_items.remove(item_id)
        self._stopped_user_transcript_publication_enabled.pop(item_id, None)
        self._emitted_user_interims.pop(item_id, None)
        self._user_audio_duration_seconds.pop(item_id, None)
        if not transcription_enabled:
            return []
        return [
            build_server_event(
                "conversation.item.input_audio_transcription.failed",
                item_id=item_id,
                content_index=0,
                error={
                    "type": "transcription_error",
                    "code": code,
                    "message": message,
                    "param": None,
                },
            )
        ]

    def stop_user_turn(
        self,
        *,
        audio_sample_cursor: int | None = None,
        sample_rate: int | None = None,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Commit and semantically finalize the active user audio turn."""
        return self._commit_user_turn(
            audio_sample_cursor=audio_sample_cursor,
            sample_rate=sample_rate,
            finalize_transcript=True,
        )

    def commit_user_turn(
        self,
        *,
        audio_sample_cursor: int | None = None,
        sample_rate: int | None = None,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Commit an externally bounded audio item without finalizing its transcript."""
        return self._commit_user_turn(
            audio_sample_cursor=audio_sample_cursor,
            sample_rate=sample_rate,
            finalize_transcript=False,
        )

    def _commit_user_turn(
        self,
        *,
        audio_sample_cursor: int | None,
        sample_rate: int | None,
        finalize_transcript: bool,
    ) -> tuple[str, int, list[dict[str, Any]]]:
        """Commit one audio item, optionally treating this edge as the semantic barrier."""
        cursor = self._input_audio_samples if audio_sample_cursor is None else max(0, int(audio_sample_cursor))
        if sample_rate is not None:
            self._input_sample_rate = max(1, int(sample_rate))
        item_id, _ = self.begin_user_turn(
            audio_sample_cursor=cursor,
            sample_rate=sample_rate,
        )
        self._user_turn_stopped = True
        start_sample = (
            self._user_turn_start_sample if self._user_turn_start_sample is not None else self._input_audio_samples
        )
        cursor = max(cursor, start_sample)
        self._user_audio_duration_seconds[item_id] = max(
            0.0,
            (cursor - start_sample) / max(self._input_sample_rate, 1),
        )
        if finalize_transcript:
            # Pipecat's UserTurnController has declared the complete semantic
            # turn stopped. Provider-level ``TranscriptionFrame.finalized``
            # flags may occur several times within that turn and are
            # deliberately not used as this barrier.
            self._pending_user_transcript.finalize_turn()
        transcription_enabled = self._input_transcription_enabled_for_item(item_id)
        transcript = self._pending_user_transcript.text
        events = self.announce_user_item()
        events.append(
            self.conversation.complete_item(
                item_id,
                item_patch={
                    "content": [
                        {
                            "type": "input_audio",
                            "transcript": transcript if transcription_enabled and transcript else None,
                        }
                    ]
                },
            )
        )
        if transcription_enabled and self._pending_user_transcript.failure_ready:
            events.extend(
                self._fail_user_transcript_item(
                    item_id,
                    code=self._pending_user_transcript.producer_failure_code or "transcription_failed",
                    message=self._pending_user_transcript.producer_failure_message
                    or "Input audio transcription failed",
                )
            )
        elif transcription_enabled and not self._pending_user_transcript.ready:
            self._stopped_user_items.append(item_id)
            self._stopped_user_transcripts[item_id] = self._pending_user_transcript
            self._stopped_user_transcript_publication_enabled[item_id] = transcription_enabled
        if transcription_enabled and self._pending_user_interim and not self._pending_user_transcript.failure_ready:
            events.append(
                build_server_event(
                    "conversation.item.input_audio_transcription.delta",
                    item_id=item_id,
                    content_index=0,
                    delta=self._pending_user_interim,
                )
            )
            if (
                transcription_enabled
                and not self._pending_user_transcript.ready
                and not self._pending_user_transcript.failure_ready
            ):
                self._emitted_user_interims[item_id] = self._pending_user_interim
        if transcription_enabled and self._pending_user_transcript.ready:
            events.extend(self._complete_user_transcript(item_id, self._pending_user_transcript))
        elif not transcription_enabled:
            self._user_audio_duration_seconds.pop(item_id, None)
        self.clear_user_turn()
        return item_id, self._input_ms(cursor), events

    def finalize_user_transcript(
        self,
        *,
        item_id: str,
        transcript: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Finalize one exact committed item at its semantic pipeline barrier."""
        state = self._stopped_user_transcripts.get(item_id)
        if state is None:
            return item_id, []

        # Provider segments accumulate until this barrier. The user aggregator
        # supplies the exact text written to LLM context, so make the terminal
        # public transcript identical to that canonical context message.
        if transcript.strip() and transcript != state.text:
            state.parts = [
                TextPartForConcatenation(
                    text=transcript,
                    includes_inter_part_spaces=True,
                )
            ]
            state.transcript_observed = True
        state.finalize_turn()
        if state.failure_ready:
            return item_id, self._fail_user_transcript_item(
                item_id,
                code=state.producer_failure_code or "transcription_failed",
                message=state.producer_failure_message or "Input audio transcription failed",
            )
        return item_id, self._complete_user_transcript(item_id, state) if state.ready else []

    def set_user_transcript(
        self,
        transcript: str,
        *,
        includes_inter_frame_spaces: bool = False,
        item_id: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Append one ordered transcript segment without treating it as a turn barrier."""
        target_item_id = item_id
        if target_item_id is None:
            # Transcript frames observe an existing audio turn; they never
            # establish one. A provider can publish a delayed frame after the
            # item's terminal event, and allocating here would let that text
            # become the next user's transcript.
            return "", []
        if not self._input_transcription_enabled_for_item(target_item_id):
            return target_item_id, []
        if not transcript.strip():
            return item_id or "", []

        state = self._stopped_user_transcripts.get(target_item_id)
        if state is not None:
            state.append(
                transcript,
                includes_inter_frame_spaces=includes_inter_frame_spaces,
            )
            events = self._complete_user_transcript(target_item_id, state) if state.ready else []
            return target_item_id, events

        if target_item_id != self._user_item_id:
            # A correlated transcript may arrive after its item was already
            # terminal. Never transfer it to another user turn.
            return target_item_id, []
        self._pending_user_transcript.append(
            transcript,
            includes_inter_frame_spaces=includes_inter_frame_spaces,
        )
        self._pending_user_interim = ""
        return target_item_id, []

    def end_user_transcript_producer(
        self,
        *,
        item_id: str,
        code: str,
        message: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """End one exact fused transcript stream and fail a finalized empty turn."""
        if not isinstance(item_id, str) or not item_id:
            raise invalid_value("transcription producer item_id must be a non-empty string", param="item_id")
        if not isinstance(code, str) or not code:
            raise invalid_value("transcription producer code must be a non-empty string", param="error.code")
        if not isinstance(message, str) or not message:
            raise invalid_value("transcription producer message must be a non-empty string", param="error.message")
        state = self._stopped_user_transcripts.get(item_id)
        if state is not None:
            state.end_producer(code=code, message=message)
            events = self._fail_user_transcript_item(item_id, code=code, message=message) if state.failure_ready else []
            return item_id, events

        if item_id != self._user_item_id:
            # This exact item already completed (or was never owned here).
            # A late terminal must never attach itself to a newer active turn.
            return item_id, []
        self._pending_user_transcript.end_producer(code=code, message=message)
        return item_id, []

    def fail_user_transcription(
        self,
        *,
        code: str,
        message: str,
        item_id: str,
    ) -> list[dict[str, Any]]:
        """Fail and release one exact stopped input-audio transcript."""
        if not isinstance(code, str) or not code:
            raise invalid_value("transcription failure code must be a non-empty string", param="error.code")
        if not isinstance(message, str) or not message:
            raise invalid_value("transcription failure message must be a non-empty string", param="error.message")
        if not isinstance(item_id, str) or not item_id:
            raise invalid_value("transcription failure item_id must be a non-empty string", param="item_id")
        if item_id not in self._stopped_user_transcripts:
            return []
        return self._fail_user_transcript_item(item_id, code=code, message=message)

    def user_transcript_delta(
        self,
        hypothesis: str,
        *,
        item_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert full evolving ASR hypotheses into append-only Realtime deltas."""
        target_item_id = item_id
        if target_item_id is None:
            return []
        if not self._input_transcription_enabled_for_item(target_item_id):
            if target_item_id == self._user_item_id:
                self._pending_user_interim = ""
            return []
        if not hypothesis.strip():
            if target_item_id == self._user_item_id:
                self._pending_user_interim = ""
            return []
        if target_item_id in self._stopped_user_transcripts:
            emitted = self._emitted_user_interims.get(target_item_id, "")
            if hypothesis.startswith(emitted):
                delta = hypothesis[len(emitted) :]
                self._emitted_user_interims[target_item_id] = hypothesis
                if delta:
                    return [
                        build_server_event(
                            "conversation.item.input_audio_transcription.delta",
                            item_id=target_item_id,
                            content_index=0,
                            delta=delta,
                        )
                    ]
            return []
        if target_item_id == self._user_item_id:
            self._pending_user_interim = hypothesis
        return []

    def clear_user_turn(self) -> None:
        """Release transient ASR correlation state after a completed turn."""
        self._user_item_id = None
        self._user_item_announced = False
        self._user_turn_start_sample = None
        self._user_turn_stopped = False
        self._pending_user_transcript = _UserTranscriptState(parts=[])
        self._pending_user_interim = ""
        self._user_transcript_publication_enabled = None

    def retrieve_item_event(self, item_id: str) -> dict[str, Any]:
        """Return the complete current item snapshot."""
        item = self.conversation.item(item_id)
        content = item.get("content")
        if isinstance(content, list) and any(
            isinstance(part, dict) and part.get("type") in {"input_audio", "output_audio"} for part in content
        ):
            raise RealtimeProtocolError(
                message="Audio item retrieval is unavailable because the cascaded gateway does not retain PCM history",
                code="unsupported_capability",
                param="item_id",
            )
        return build_server_event("conversation.item.retrieved", item=self.conversation.wire_item(item_id))

    def snapshot_conversation_mutation_state(self) -> _ConversationMutationState:
        """Capture controller state changed by a conversation item mutation."""
        return _ConversationMutationState(
            journal=self.conversation.snapshot_state(),
            assistant_audio_alignments=copy.deepcopy(self._assistant_audio_alignments),
        )

    def restore_conversation_mutation_state(self, snapshot: _ConversationMutationState) -> None:
        """Restore a trusted item-mutation snapshot without replacing the journal."""
        if not isinstance(snapshot, _ConversationMutationState):
            raise TypeError("Conversation rollback requires a controller snapshot")
        alignments = copy.deepcopy(snapshot.assistant_audio_alignments)
        self.conversation.restore_state(snapshot.journal)
        self._assistant_audio_alignments = alignments

    def delete_item_event(self, item_id: str) -> dict[str, Any]:
        """Delete one conversation item."""
        event = self.conversation.delete_item(item_id)
        self._assistant_audio_alignments.pop(item_id, None)
        return event

    def preview_item_truncation(
        self,
        *,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> AssistantAudioTruncation:
        """Validate truncation and return the exact Pipecat text projection."""
        item = self.conversation.item(item_id)
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise RealtimeProtocolError(
                message="Only assistant message audio can be truncated",
                code="invalid_truncate",
                param="item_id",
            )
        if item.get("status") not in {"completed", "incomplete"}:
            raise RealtimeProtocolError(
                message="Only a terminal assistant audio item can be truncated",
                code="invalid_truncate",
                param="item_id",
            )
        content = item.get("content") or []
        if content_index < 0 or content_index >= len(content):
            raise RealtimeProtocolError(
                message="content_index does not identify an audio content part",
                code="invalid_truncate",
                param="content_index",
            )
        part = content[content_index]
        if not isinstance(part, dict) or part.get("type") != "output_audio":
            raise RealtimeProtocolError(
                message="The selected content part is not output audio",
                code="invalid_truncate",
                param="content_index",
            )
        if audio_end_ms == 0:
            alignment = self._assistant_audio_alignments.get(item_id)
            return AssistantAudioTruncation(
                transcript="",
                context_text="",
                previous_context_text=alignment.context_text if alignment is not None else "",
            )

        alignment = self._assistant_audio_alignments.get(item_id)
        if alignment is None or part.get("transcript", "") != alignment.transcript:
            raise RealtimeProtocolError(
                message="The assistant audio item has no exact Pipecat playout alignment",
                code="invalid_truncate",
                param="item_id",
            )
        return alignment.project(audio_end_ms)

    def truncate_item_event(
        self,
        *,
        item_id: str,
        content_index: int,
        audio_end_ms: int,
    ) -> dict[str, Any]:
        """Truncate an assistant audio item at a client playback boundary."""
        projection = self.preview_item_truncation(
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )
        item = self.conversation.item(item_id)
        content = copy.deepcopy(item.get("content") or [])
        content[content_index]["transcript"] = projection.transcript
        self.conversation.patch_item(item_id, {"content": content})
        alignment = self._assistant_audio_alignments.get(item_id)
        if alignment is not None:
            alignment.truncate(audio_end_ms=audio_end_ms, projection=projection)
        return build_server_event(
            "conversation.item.truncated",
            item_id=item_id,
            content_index=content_index,
            audio_end_ms=audio_end_ms,
        )

    def tool_call(self, call_id: str) -> ToolCallRecord | None:
        """Return the mutable record for one correlated tool call."""
        return self._tool_calls.get(call_id)

    def pipeline_tool_owner(self, *, call_id: str, pipeline_name: str) -> ToolOwner | None:
        """Classify a provider call without publishing or mutating its lifecycle."""
        record = self._tool_calls.get(call_id)
        if record is not None:
            return record.owner
        if pipeline_name in self._active_mcp_pipeline_names and pipeline_name in self._mcp_tool_bindings:
            return "mcp"
        if pipeline_name in self._active_client_tool_bindings:
            return "client"
        if pipeline_name in self.delegate_tools:
            return "delegate"
        if pipeline_name in self.server_tools:
            return "server"
        return None

    def mark_response_done_published(self, response_id: str) -> None:
        """Record that one terminal response event crossed the WebSocket boundary."""
        for record in self._tool_calls.values():
            if record.response_id == response_id:
                record.response_done_published = True

    def _terminalize_response_tool_calls(
        self,
        *,
        response_id: str,
        status: ResponseTerminalStatus,
    ) -> None:
        """Bind every call record to Response A's immutable terminal status."""
        for record in self._tool_calls.values():
            if record.response_id != response_id:
                continue
            if record.response_status is not None and record.response_status != status:
                raise RuntimeError(
                    f"Function call {record.call_id!r} changed terminal response status "
                    f"from {record.response_status} to {status}"
                )
            record.response_status = status

    def _input_ms(self, samples: int) -> int:
        return int(samples * 1000 / max(self._input_sample_rate, 1))

    def _take_transcription_usage(self, item_id: str) -> dict[str, Any]:
        """Return duration-billed ASR usage and release the turn accounting."""
        seconds = self._user_audio_duration_seconds.pop(item_id, 0.0)
        return {"type": "duration", "seconds": seconds}

    def _require_response_id(self) -> str:
        response_id = self.active_response_id
        if response_id is None:
            raise RealtimeProtocolError(
                message="No response is currently in progress",
                code="response_not_found",
                param="response",
            )
        return response_id

    def _response_output_index(self, item_id: str) -> int:
        try:
            return self._active_output_ids.index(item_id)
        except ValueError as exc:
            raise RealtimeProtocolError(
                message=f"Item {item_id!r} is not attached to the active response",
                code="output_item_not_found",
                param="item_id",
            ) from exc

    def _require_tail_append(
        self,
        *,
        previous_item_id: str | None,
        previous_item_id_supplied: bool,
    ) -> None:
        """Reject wire insertions that the Pipecat context cannot mirror."""
        if not previous_item_id_supplied:
            return
        tail_id = self.conversation.tail_id
        if previous_item_id == tail_id or (tail_id is None and previous_item_id == "root"):
            return
        raise RealtimeProtocolError(
            message=(
                "Inserting a conversation item before the current tail is not available "
                "until Pipecat context mutation is wired"
            ),
            code="unsupported_context_insertion",
            param="previous_item_id",
        )


def _response_status_details(
    *,
    status: ResponseTerminalStatus,
    reason: str | None,
) -> dict[str, Any] | None:
    """Build the status-specific canonical Realtime response details."""
    if status == "completed":
        return None
    if status == "cancelled":
        if reason not in {"turn_detected", "client_cancelled"}:
            raise invalid_value(
                "cancelled response reason must be turn_detected or client_cancelled",
                param="response.status_details.reason",
            )
        return {"type": "cancelled", "reason": reason}
    if status == "incomplete":
        if reason not in {"max_output_tokens", "content_filter"}:
            raise invalid_value(
                "incomplete response reason must be max_output_tokens or content_filter",
                param="response.status_details.reason",
            )
        return {"type": "incomplete", "reason": reason}

    code = reason or "response_failed"
    return {
        "type": "failed",
        "error": {
            "type": "server_error",
            "code": code,
        },
    }
