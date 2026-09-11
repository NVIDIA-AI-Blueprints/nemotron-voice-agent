# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Canonical OpenAI Realtime WebSocket gateway and pipeline handoff."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from realtime.controller import RealtimeSessionController
from realtime.protocol import (
    MAX_REALTIME_EVENT_BYTES,
    RealtimeProtocolError,
    strict_json_loads,
    validate_client_event_id,
)
from realtime.session import (
    RealtimeSessionCapabilities,
    project_tool_choice_to_pipeline,
    validate_session_tools_bounds,
)
from utils import parse_env_bool, parse_env_int

SanitizeFn = Callable[..., dict[str, Any]]
PrepareRuntimeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
EnsureReadyFn = Callable[[dict[str, Any]], Awaitable[None]]
StartBotFn = Callable[[Any, dict[str, Any], RealtimeSessionController], Awaitable[None]]
ServerToolsFn = Callable[[dict[str, Any]], list[str]]
DelegateToolsFn = Callable[[dict[str, Any]], list[str]]
ToolSchemasFn = Callable[[dict[str, Any]], list[dict[str, Any]]]
ResolveCapabilitiesFn = Callable[[dict[str, Any], bool], Awaitable[RealtimeSessionCapabilities]]


@dataclass(frozen=True, slots=True)
class RealtimeModelRoute:
    """One public Realtime model bound to trusted pipeline selectors."""

    model: str
    runtime_config: dict[str, Any]


ResolveModelRouteFn = Callable[[str | None, dict[str, Any]], RealtimeModelRoute]

_DEFAULT_INITIAL_EVENT_TIMEOUT_SECS = 60
_DEFAULT_MAX_REJECTED_EVENTS = 32
_DEFAULT_PIPELINE_MODE = "generic-assistant"
_OPENAI_REALTIME_SESSION_MAX_DURATION_SECS = 60 * 60
_MCP_ENDPOINT_FIELDS = frozenset({"server_url", "connector_id", "tunnel_id"})
_SMART_TURN_PIPELINES = frozenset({"omni-assistant", "omni-assistant-subagents"})
_CONFIGURABLE_CASCADE_PIPELINES = frozenset({"frontend-backend-agent", "generic-assistant", "multilingual-assistant"})


class _ReplayWebSocket:
    """Replay one consumed text event, then delegate to the FastAPI socket."""

    def __init__(self, websocket: WebSocket, first_text: str) -> None:
        self._websocket = websocket
        self._first_text: str | None = first_text

    async def receive(self) -> dict[str, Any]:
        if self._first_text is not None:
            text = self._first_text
            self._first_text = None
            return {"type": "websocket.receive", "text": text}
        return await self._websocket.receive()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._websocket, name)


def _select_realtime_subprotocol(websocket: WebSocket) -> str | None:
    """Negotiate only the public ``realtime`` WebSocket subprotocol."""
    protocols = websocket.headers.get("sec-websocket-protocol") or ""
    offered = {value.strip() for value in protocols.split(",") if value.strip()}
    return "realtime" if "realtime" in offered else None


async def _send_json(websocket: Any, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, default=str, allow_nan=False))


async def _initial_runtime_config(
    *,
    sanitize_session_config: SanitizeFn,
    resolve_server_tools: ServerToolsFn | None,
    resolve_delegate_tools: DelegateToolsFn | None,
    resolve_server_tool_schemas: ToolSchemasFn | None,
    resolve_delegate_tool_schemas: ToolSchemasFn | None,
    default_example_key: str,
    default_pipeline_mode: str,
    initial_runtime_config: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str], list[str], list[dict[str, Any]]]:
    base = {"pipeline_mode": default_pipeline_mode}
    if initial_runtime_config is not None:
        base.update(copy.deepcopy(initial_runtime_config))
    sanitized = sanitize_session_config(base, fallback_example_key=default_example_key)
    server_tools = resolve_server_tools(sanitized) if resolve_server_tools is not None else []
    delegate_tools = resolve_delegate_tools(sanitized) if resolve_delegate_tools is not None else []
    server_schemas = resolve_server_tool_schemas(sanitized) if resolve_server_tool_schemas is not None else []
    delegate_schemas = resolve_delegate_tool_schemas(sanitized) if resolve_delegate_tool_schemas is not None else []
    sanitized["server_tools"] = list(server_tools)
    sanitized["delegate_tools"] = list(delegate_tools)
    return sanitized, server_tools, delegate_tools, [*server_schemas, *delegate_schemas]


async def create_realtime_controller(
    *,
    sanitize_session_config: SanitizeFn,
    initial_session: dict[str, Any] | None = None,
    ensure_services_ready: EnsureReadyFn | None = None,
    prepare_initial_runtime: PrepareRuntimeFn | None = None,
    resolve_server_tools: ServerToolsFn | None = None,
    resolve_delegate_tools: DelegateToolsFn | None = None,
    resolve_server_tool_schemas: ToolSchemasFn | None = None,
    resolve_delegate_tool_schemas: ToolSchemasFn | None = None,
    resolve_session_capabilities: ResolveCapabilitiesFn | None = None,
    resolve_model_route: ResolveModelRouteFn | None = None,
    default_example_key: str = "",
    default_pipeline_mode: str = _DEFAULT_PIPELINE_MODE,
    requested_model: str | None = None,
) -> RealtimeSessionController:
    """Build a controller, applying a client-secret session before publication."""
    standard_patch: dict[str, Any] = {}
    nvidia_patch: dict[str, Any] = {}
    session_model: str | None = None
    if initial_session is not None:
        if not isinstance(initial_session, dict):
            raise RealtimeProtocolError(
                message="session must be an object",
                code="invalid_type",
                param="session",
            )
        if "tools" in initial_session:
            validate_session_tools_bounds(initial_session["tools"])
        standard_patch, nvidia_patch = _extract_nvidia_patch(initial_session)
        if "model" in standard_patch:
            raw_model = standard_patch["model"]
            if not isinstance(raw_model, str):
                raise RealtimeProtocolError(
                    message="session.model must be a string",
                    code="invalid_type",
                    param="session.model",
                )
            if not raw_model:
                raise RealtimeProtocolError(
                    message="session.model must not be empty",
                    code="invalid_value",
                    param="session.model",
                )
            session_model = raw_model

    if requested_model is not None and not isinstance(requested_model, str):
        raise RealtimeProtocolError(
            message="model must be a string",
            code="invalid_type",
            param="model",
        )
    if requested_model == "":
        requested_model = None
    if requested_model is not None and session_model is not None and requested_model != session_model:
        raise RealtimeProtocolError(
            message="The connection model does not match the client-secret session model",
            code="model_not_available",
            param="model",
        )

    selected_model = requested_model or session_model
    runtime_seed = copy.deepcopy(nvidia_patch)
    advertised_model: str | None = None
    if resolve_model_route is not None:
        route = resolve_model_route(selected_model, copy.deepcopy(nvidia_patch))
        if not isinstance(route, RealtimeModelRoute):
            raise TypeError("Realtime model resolver must return RealtimeModelRoute")
        if not isinstance(route.model, str) or not route.model:
            raise TypeError("Realtime model resolver returned an invalid public model id")
        if not isinstance(route.runtime_config, dict):
            raise TypeError("Realtime model resolver returned an invalid runtime configuration")
        advertised_model = route.model
        runtime_seed = copy.deepcopy(route.runtime_config)

    runtime_config, server_tools, delegate_tools, trusted_tool_schemas = await _initial_runtime_config(
        sanitize_session_config=sanitize_session_config,
        resolve_server_tools=resolve_server_tools,
        resolve_delegate_tools=resolve_delegate_tools,
        resolve_server_tool_schemas=resolve_server_tool_schemas,
        resolve_delegate_tool_schemas=resolve_delegate_tool_schemas,
        default_example_key=default_example_key,
        default_pipeline_mode=default_pipeline_mode,
        initial_runtime_config=runtime_seed,
    )
    resolved_capabilities = None
    output_voice_resolver = None
    if resolve_session_capabilities is not None:
        resolved_capabilities = await resolve_session_capabilities(copy.deepcopy(runtime_config), False)
        if not isinstance(resolved_capabilities, RealtimeSessionCapabilities):
            raise TypeError("Realtime capability resolver must return RealtimeSessionCapabilities")
        if resolved_capabilities.voices is None or not resolved_capabilities.voices:
            raise RuntimeError("The selected Realtime TTS route did not provide any trusted voices")

        async def output_voice_resolver() -> frozenset[str]:
            discovered = await resolve_session_capabilities(copy.deepcopy(runtime_config), True)
            if not isinstance(discovered, RealtimeSessionCapabilities):
                raise TypeError("Realtime capability resolver must return RealtimeSessionCapabilities")
            if discovered.voices is None or not discovered.voices:
                raise RuntimeError("The selected Realtime TTS route did not provide any trusted voices")
            return discovered.voices

    controller = _controller_from_runtime(
        runtime_config,
        server_tools=server_tools,
        delegate_tools=delegate_tools,
        trusted_tool_schemas=trusted_tool_schemas,
        advertised_model=advertised_model,
        requested_model=selected_model if resolve_model_route is None else None,
        resolved_capabilities=resolved_capabilities,
        output_voice_resolver=output_voice_resolver,
        output_voices_resolved=resolve_session_capabilities is None,
    )
    if initial_session is None:
        await controller.ensure_output_voice_capabilities()
        return controller

    standard_patch, sanitized = await _prepare_session_update(
        controller,
        initial_session,
        sanitize_session_config=sanitize_session_config,
        prepare_runtime=prepare_initial_runtime,
        ensure_services_ready=ensure_services_ready,
        default_example_key=default_example_key,
    )
    controller.apply_session_update(standard_patch)
    controller.bind_prepared_runtime(sanitized)
    return controller


def _controller_from_runtime(
    runtime_config: dict[str, Any],
    *,
    server_tools: list[str],
    delegate_tools: list[str],
    trusted_tool_schemas: list[dict[str, Any]] | None = None,
    advertised_model: str | None = None,
    requested_model: str | None = None,
    resolved_capabilities: RealtimeSessionCapabilities | None = None,
    output_voice_resolver: Callable[[], Awaitable[frozenset[str]]] | None = None,
    output_voices_resolved: bool = True,
) -> RealtimeSessionController:
    owned_tools = {*server_tools, *delegate_tools}
    schema_names = {
        schema.get("name")
        for schema in trusted_tool_schemas or []
        if isinstance(schema, dict) and isinstance(schema.get("name"), str)
    }
    missing_schemas = sorted(owned_tools - schema_names)
    if missing_schemas:
        raise ValueError(f"Trusted tool {missing_schemas[0]!r} has no callable schema")
    model = advertised_model or str(
        runtime_config.get("model_id") or runtime_config.get("llm_id") or "nvidia-realtime-cascade"
    )
    if requested_model is not None and requested_model != model:
        raise RealtimeProtocolError(
            message=f"Model {requested_model!r} is not available on this endpoint",
            code="model_not_available",
            param="model",
        )
    voice = str(runtime_config.get("tts_voice_id") or "default")
    instructions = str(runtime_config.get("prompt_content") or "")
    transcription_model = runtime_config.get("realtime_input_transcription_model") or runtime_config.get("asr_model")
    if not isinstance(transcription_model, str) or not transcription_model:
        transcription_model = None
    raw_model_output_limit = runtime_config.get("realtime_model_max_output_tokens")
    if raw_model_output_limit is None:
        model_output_limit = None
    elif (
        isinstance(raw_model_output_limit, bool)
        or not isinstance(raw_model_output_limit, int)
        or not 1 <= raw_model_output_limit <= 4096
    ):
        raise ValueError("Configured realtime_model_max_output_tokens must be an integer from 1 to 4096")
    else:
        model_output_limit = raw_model_output_limit
    raw_max_tokens = runtime_config.get("max_tokens")
    if raw_max_tokens in (None, "", "inf"):
        max_output_tokens: int | str = "inf"
    else:
        if isinstance(raw_max_tokens, bool):
            raise ValueError("Configured max_tokens must be an integer")
        try:
            max_output_tokens = int(raw_max_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("Configured max_tokens must be an integer") from exc
        if model_output_limit is not None:
            max_output_tokens = min(max_output_tokens, model_output_limit)
    pipeline_mode = str(runtime_config.get("pipeline_mode") or _DEFAULT_PIPELINE_MODE)
    if pipeline_mode in _SMART_TURN_PIPELINES:
        turn_detection_type = "semantic_vad"
    elif pipeline_mode in _CONFIGURABLE_CASCADE_PIPELINES:
        turn_detection_type = (
            "server_vad" if parse_env_bool("USE_SILERO_VAD_TURN_DETECTION", default=False) else "semantic_vad"
        )
    else:
        raise ValueError(f"Realtime turn detection is not defined for pipeline_mode {pipeline_mode!r}")
    manual_input_available = bool(transcription_model) and pipeline_mode in {
        "frontend-backend-agent",
        "generic-assistant",
        "multilingual-assistant",
    }
    tool_runtime_available = runtime_config.get("pipeline_mode") != "omni-assistant-subagents"
    capabilities = replace(
        resolved_capabilities or RealtimeSessionCapabilities(voices=frozenset({voice})),
        function_tools=tool_runtime_available,
        mcp_tools=tool_runtime_available,
        trusted_function_tools=frozenset({*server_tools, *delegate_tools}),
        parallel_tool_calls=True,
        sequential_tool_calls=tool_runtime_available,
        turn_detection_types=frozenset({turn_detection_type}),
        default_turn_detection_type=turn_detection_type,
        turn_detection_create_response_values=(
            frozenset({True}) if pipeline_mode in _SMART_TURN_PIPELINES else frozenset({False, True})
        ),
        turn_detection_interrupt_response_values=(
            frozenset({True}) if pipeline_mode in _SMART_TURN_PIPELINES else frozenset({False, True})
        ),
        supports_manual_input=manual_input_available,
    )
    return RealtimeSessionController(
        model=model,
        voice=voice,
        runtime_config=runtime_config,
        instructions=instructions,
        max_output_tokens=max_output_tokens,
        input_transcription_model=transcription_model,
        server_tools=server_tools,
        delegate_tools=delegate_tools,
        trusted_tool_schemas=trusted_tool_schemas,
        capabilities=capabilities,
        output_voice_resolver=output_voice_resolver,
        output_voices_resolved=output_voices_resolved,
    )


def _extract_nvidia_patch(session_patch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split the namespaced catalog view from the standard session patch."""
    standard = {key: copy.deepcopy(value) for key, value in session_patch.items() if key != "nvidia"}
    raw_nvidia = session_patch.get("nvidia")
    if raw_nvidia is None:
        return standard, {}
    if not isinstance(raw_nvidia, dict):
        raise RealtimeProtocolError(
            message="session.nvidia must be an object",
            code="invalid_type",
            param="session.nvidia",
        )
    return standard, copy.deepcopy(raw_nvidia)


def _validate_immutable_nvidia_patch(nvidia_patch: dict[str, Any], current: dict[str, Any]) -> None:
    """Reject backend routing changes after session.created advertised a model."""
    public = current.get("nvidia") if isinstance(current.get("nvidia"), dict) else {}
    for key, value in nvidia_patch.items():
        if key not in public:
            raise RealtimeProtocolError(
                message=f"Unknown parameter: session.nvidia.{key}",
                code="unknown_parameter",
                param=f"session.nvidia.{key}",
            )
        if value != public.get(key):
            raise RealtimeProtocolError(
                message=(
                    f"session.nvidia.{key} is fixed when the connection is created; "
                    "select backend routing in the connection configuration"
                ),
                code="immutable_field",
                param=f"session.nvidia.{key}",
            )


def _session_patch_to_runtime(
    session_view: dict[str, Any],
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    """Project supported standard session fields onto the cascaded pipeline."""
    runtime = copy.deepcopy(runtime_config)
    instructions = session_view.get("instructions")
    if isinstance(instructions, str):
        runtime["prompt_content"] = instructions

    max_tokens = session_view.get("max_output_tokens")
    runtime["max_tokens"] = None if max_tokens == "inf" else max_tokens
    native_tool_choice = copy.deepcopy(session_view.get("tool_choice", "auto"))
    runtime["realtime_tool_choice"] = native_tool_choice
    runtime["tool_choice"] = project_tool_choice_to_pipeline(native_tool_choice)
    configured_server_tools = {name for name in runtime.get("server_tools", []) if isinstance(name, str)}
    configured_delegate_tools = {name for name in runtime.get("delegate_tools", []) if isinstance(name, str)}
    active_function_tools = [
        copy.deepcopy(tool)
        for tool in session_view.get("tools", [])
        if (isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str))
    ]
    runtime["mcp_tools"] = [
        copy.deepcopy(tool)
        for tool in session_view.get("tools", [])
        if isinstance(tool, dict) and tool.get("type") == "mcp"
    ]
    active_names = {tool["name"] for tool in active_function_tools}
    runtime["server_tools"] = sorted(configured_server_tools & active_names)
    runtime["delegate_tools"] = sorted(configured_delegate_tools & active_names)
    trusted_names = configured_server_tools | configured_delegate_tools
    runtime["client_tools"] = [tool for tool in active_function_tools if tool["name"] not in trusted_names]
    runtime["parallel_tool_calls"] = bool(session_view.get("parallel_tool_calls", True))
    runtime["output_modalities"] = copy.deepcopy(session_view.get("output_modalities", ["audio"]))

    audio = session_view.get("audio")
    if isinstance(audio, dict):
        input_audio = audio.get("input")
        if isinstance(input_audio, dict):
            transcription = input_audio.get("transcription")
            if isinstance(transcription, dict):
                transcription_model = transcription.get("model")
                if isinstance(transcription_model, str) and transcription_model:
                    runtime["realtime_input_transcription_model"] = transcription_model
                transcription_language = transcription.get("language")
                if isinstance(transcription_language, str) and transcription_language:
                    runtime["asr_language_code"] = transcription_language
        output = audio.get("output")
        if isinstance(output, dict):
            voice = output.get("voice")
            if isinstance(voice, str) and voice:
                runtime["tts_voice_id"] = voice
    return runtime


async def _prepare_session_update(
    controller: RealtimeSessionController,
    raw_patch: dict[str, Any],
    *,
    sanitize_session_config: SanitizeFn,
    prepare_runtime: PrepareRuntimeFn | None,
    ensure_services_ready: EnsureReadyFn | None,
    default_example_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and prepare one session patch without mutating the controller."""
    if "tools" in raw_patch:
        validate_session_tools_bounds(raw_patch["tools"])
        for index, tool in enumerate(raw_patch["tools"]):
            if not isinstance(tool, dict) or tool.get("type") != "mcp":
                continue
            if not any(field in tool for field in _MCP_ENDPOINT_FIELDS):
                raise RealtimeProtocolError(
                    message=(
                        "An initial session MCP tool must define its endpoint; "
                        "server_label-only reuse is available in response.tools after discovery"
                    ),
                    code="mcp_server_not_defined",
                    param=f"session.tools[{index}].server_label",
                )
    standard_patch, nvidia_patch = _extract_nvidia_patch(raw_patch)
    _validate_immutable_nvidia_patch(nvidia_patch, controller.public_session())

    controller.session.preflight_update_without_output_voice_catalog(standard_patch)

    if controller.session_update_requires_output_voice_resolution(standard_patch):
        await controller.ensure_output_voice_capabilities()

    trial_session = copy.deepcopy(controller.session)
    trial_session.apply_update(standard_patch)
    projected = _session_patch_to_runtime(trial_session.public_view(), controller.runtime_config)
    sanitized = sanitize_session_config(projected, fallback_example_key=default_example_key)
    instructions_explicit = "instructions" in standard_patch
    if instructions_explicit:
        # Empty instructions are a valid, intentional Realtime value. The
        # ordinary pipeline sanitizer treats an empty prompt as "use the
        # catalog default", so retain this ownership bit until the Realtime
        # pipeline has rendered the exact client value.
        sanitized["_realtime_instructions_explicit"] = True
        sanitized["prompt_content"] = copy.deepcopy(projected["prompt_content"])
    for key in (
        "client_tools",
        "delegate_tools",
        "max_tokens",
        "mcp_tools",
        "output_modalities",
        "parallel_tool_calls",
        "server_tools",
        "tool_choice",
        "realtime_tool_choice",
    ):
        if key in projected:
            sanitized[key] = copy.deepcopy(projected[key])
    if prepare_runtime is not None:
        sanitized = await prepare_runtime(copy.deepcopy(sanitized))
    if not isinstance(sanitized, dict):
        raise TypeError("Prepared Realtime runtime must be a dictionary")
    if instructions_explicit:
        sanitized["_realtime_instructions_explicit"] = True
        sanitized["prompt_content"] = copy.deepcopy(projected["prompt_content"])
    prepared_voice = sanitized.get("tts_voice_id")
    if isinstance(prepared_voice, str) and prepared_voice:
        # Validate the trusted discovery result against the same trial state
        # before mutating the live controller. Client-provided voices were
        # already checked strictly by trial_session.apply_update above.
        trial_session.bind_discovered_output_voice(prepared_voice)
    if ensure_services_ready is not None:
        await ensure_services_ready(sanitized)
    return standard_patch, sanitized


async def _apply_initial_session_update(
    websocket: Any,
    controller: RealtimeSessionController,
    message: dict[str, Any],
    *,
    sanitize_session_config: SanitizeFn,
    prepare_runtime: PrepareRuntimeFn | None,
    ensure_services_ready: EnsureReadyFn | None,
    default_example_key: str,
) -> bool:
    """Validate, prepare, and atomically apply the first session update."""
    unknown_fields = sorted(set(message) - {"event_id", "session", "type"})
    client_event_id = message.get("event_id")
    echo_id = client_event_id if isinstance(client_event_id, str) else None
    if unknown_fields:
        field = unknown_fields[0]
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message=f"Unknown parameter: {field}",
                code="unknown_parameter",
                param=field,
                event_id=echo_id,
            ).to_event(),
        )
        return False
    raw_patch = message.get("session")
    if not isinstance(raw_patch, dict):
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="session.update requires a session object",
                code="invalid_type",
                param="session",
                event_id=echo_id,
            ).to_event(),
        )
        return False

    try:
        standard_patch, sanitized = await _prepare_session_update(
            controller,
            raw_patch,
            sanitize_session_config=sanitize_session_config,
            prepare_runtime=prepare_runtime,
            ensure_services_ready=ensure_services_ready,
            default_example_key=default_example_key,
        )
        controller.apply_session_update(standard_patch)
        controller.bind_prepared_runtime(sanitized)
    except RealtimeProtocolError as exc:
        await _send_json(websocket, exc.with_event_id(echo_id).to_event())
        return False
    except ValueError as exc:
        logger.warning(f"Realtime session configuration rejected: {exc}")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="The Realtime session configuration is invalid",
                code="invalid_session",
                param="session",
                event_id=echo_id,
            ).to_event(),
        )
        return False
    except RuntimeError as exc:
        logger.warning(f"Realtime service readiness failed during session update: {exc}")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="One or more required Realtime services are not ready",
                code="services_not_ready",
                event_id=echo_id,
                error_type="server_error",
            ).to_event(),
        )
        return False
    except Exception:
        logger.exception("Failed to prepare Realtime session")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="The Realtime session could not be prepared",
                code="session_update_failed",
                event_id=echo_id,
                error_type="server_error",
            ).to_event(),
        )
        return False

    await _send_json(websocket, controller.session_updated_event())
    return True


async def _ensure_runtime_ready(
    websocket: Any,
    controller: RealtimeSessionController,
    prepare_runtime: PrepareRuntimeFn | None,
    ensure_services_ready: EnsureReadyFn | None,
    *,
    event_id: str | None,
) -> bool:
    try:
        public_before = controller.public_session()
        # The public session intentionally redacts MCP credentials. Runtime
        # preparation must use the private canonical view so those credentials
        # still reach the allowlisted MCP transport.
        prepared = _session_patch_to_runtime(controller.session.public_view(), controller.runtime_config)
        if prepare_runtime is not None:
            prepared = await prepare_runtime(copy.deepcopy(prepared))
        if ensure_services_ready is not None:
            await ensure_services_ready(prepared)
        controller.bind_prepared_runtime(prepared)
    except Exception as exc:
        logger.warning(f"Realtime service readiness failed: {exc}")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="One or more required Realtime services are not ready",
                code="services_not_ready",
                event_id=event_id,
                error_type="server_error",
            ).to_event(),
        )
        return False
    if controller.public_session() != public_before:
        await _send_json(websocket, controller.session_updated_event())
    return True


async def _run_realtime_websocket(
    websocket: WebSocket,
    *,
    sanitize_session_config: SanitizeFn,
    initial_session: dict[str, Any] | None = None,
    ensure_services_ready: EnsureReadyFn | None = None,
    prepare_initial_runtime: PrepareRuntimeFn | None = None,
    start_bot: StartBotFn | None = None,
    resolve_server_tools: ServerToolsFn | None = None,
    resolve_delegate_tools: DelegateToolsFn | None = None,
    resolve_server_tool_schemas: ToolSchemasFn | None = None,
    resolve_delegate_tool_schemas: ToolSchemasFn | None = None,
    resolve_session_capabilities: ResolveCapabilitiesFn | None = None,
    resolve_model_route: ResolveModelRouteFn | None = None,
    default_example_key: str = "",
    default_pipeline_mode: str = _DEFAULT_PIPELINE_MODE,
) -> None:
    """Accept a canonical Realtime socket and lazily hand it to one pipeline.

    The gateway immediately sends ``session.created`` and
    ``conversation.created``. A client can send ``session.update`` first, but it
    is not mandatory: the first actionable event starts the configured default
    pipeline and is replayed into its transport.
    """
    subprotocol = _select_realtime_subprotocol(websocket)
    if subprotocol:
        await websocket.accept(subprotocol=subprotocol)
    else:
        await websocket.accept()

    try:
        query_params = getattr(websocket, "query_params", None)
        requested_model = query_params.get("model") if query_params is not None else None
        if requested_model == "":
            requested_model = None
        controller = await create_realtime_controller(
            sanitize_session_config=sanitize_session_config,
            initial_session=initial_session,
            resolve_server_tools=resolve_server_tools,
            resolve_delegate_tools=resolve_delegate_tools,
            resolve_server_tool_schemas=resolve_server_tool_schemas,
            resolve_delegate_tool_schemas=resolve_delegate_tool_schemas,
            resolve_session_capabilities=resolve_session_capabilities,
            resolve_model_route=resolve_model_route,
            default_example_key=default_example_key,
            default_pipeline_mode=default_pipeline_mode,
            requested_model=requested_model,
        )
    except RealtimeProtocolError as exc:
        await _send_json(websocket, exc.to_event())
        await websocket.close(code=1008, reason="invalid realtime model")
        return
    except RuntimeError as exc:
        logger.warning(f"Realtime initial service preparation failed: {exc}")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="One or more required Realtime services are not ready",
                code="services_not_ready",
                error_type="server_error",
            ).to_event(),
        )
        await websocket.close(code=1013, reason="realtime services not ready")
        return
    except Exception:
        logger.exception("Failed to initialize Realtime session")
        await _send_json(
            websocket,
            RealtimeProtocolError(
                message="The Realtime session could not be initialized",
                code="session_initialization_failed",
                error_type="server_error",
            ).to_event(),
        )
        await websocket.close(code=1011, reason="session initialization failed")
        return

    with logger.contextualize(stream_id=controller.id):
        logger.info(f"Realtime WS connected session_id={controller.id}")
        for event in controller.created_events():
            await _send_json(websocket, event)

        timeout_secs = parse_env_int(
            "REALTIME_INITIAL_EVENT_TIMEOUT_SECS",
            _DEFAULT_INITIAL_EVENT_TIMEOUT_SECS,
            min_value=5,
        )
        max_rejected = parse_env_int(
            "REALTIME_MAX_REJECTED_EVENTS",
            _DEFAULT_MAX_REJECTED_EVENTS,
            min_value=1,
        )
        rejected = 0

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(websocket.receive_text(), timeout=timeout_secs)
                except TimeoutError:
                    await websocket.close(code=1008, reason="initial event timeout")
                    return

                if len(raw.encode("utf-8")) > MAX_REALTIME_EVENT_BYTES:
                    await _send_json(
                        websocket,
                        RealtimeProtocolError(
                            message="Realtime event exceeds the WebSocket message limit",
                            code="event_too_large",
                        ).to_event(),
                    )
                    rejected += 1
                    if rejected >= max_rejected:
                        await websocket.close(code=1008, reason="too many invalid events")
                        return
                    continue

                try:
                    message = strict_json_loads(raw)
                except (json.JSONDecodeError, ValueError):
                    await _send_json(
                        websocket,
                        RealtimeProtocolError(message="Invalid JSON", code="invalid_json").to_event(),
                    )
                    rejected += 1
                    if rejected >= max_rejected:
                        await websocket.close(code=1008, reason="too many invalid events")
                        return
                    continue
                if not isinstance(message, dict):
                    await _send_json(
                        websocket,
                        RealtimeProtocolError(
                            message="Event must be a JSON object",
                            code="invalid_event",
                        ).to_event(),
                    )
                    rejected += 1
                    if rejected >= max_rejected:
                        await websocket.close(code=1008, reason="too many invalid events")
                        return
                    continue
                try:
                    event_id = validate_client_event_id(message.get("event_id"))
                except RealtimeProtocolError as exc:
                    await _send_json(
                        websocket,
                        exc.to_event(),
                    )
                    rejected += 1
                    if rejected >= max_rejected:
                        await websocket.close(code=1008, reason="too many invalid events")
                        return
                    continue
                event_type = message.get("type")
                if not isinstance(event_type, str) or not event_type:
                    await _send_json(
                        websocket,
                        RealtimeProtocolError(
                            message="Missing event type",
                            code="missing_type",
                            param="type",
                            event_id=event_id,
                        ).to_event(),
                    )
                    rejected += 1
                    if rejected >= max_rejected:
                        await websocket.close(code=1008, reason="too many invalid events")
                        return
                    continue

                if event_type == "session.update":
                    applied = await _apply_initial_session_update(
                        websocket,
                        controller,
                        message,
                        sanitize_session_config=sanitize_session_config,
                        prepare_runtime=prepare_initial_runtime,
                        ensure_services_ready=ensure_services_ready,
                        default_example_key=default_example_key,
                    )
                    if not applied:
                        rejected += 1
                        if rejected >= max_rejected:
                            await websocket.close(code=1008, reason="too many rejected session updates")
                            return
                        continue
                    bot_socket: Any = websocket
                else:
                    ready = await _ensure_runtime_ready(
                        websocket,
                        controller,
                        prepare_initial_runtime,
                        ensure_services_ready,
                        event_id=event_id,
                    )
                    if not ready:
                        rejected += 1
                        if rejected >= max_rejected:
                            await websocket.close(code=1011, reason="services not ready")
                            return
                        continue
                    bot_socket = _ReplayWebSocket(websocket, raw)

                if start_bot is None:
                    logger.info("Realtime session configured without a pipeline start callback")
                    return
                logger.info(
                    f"Realtime handoff session_id={controller.id} "
                    f"pipeline_mode={controller.runtime_config.get('pipeline_mode')}"
                )
                await start_bot(bot_socket, controller.runtime_config, controller)
                return
        except WebSocketDisconnect:
            logger.info(f"Realtime WS disconnected session_id={controller.id}")
        except Exception:
            logger.exception(f"Realtime WS error session_id={controller.id}")
            raise


async def handle_realtime_websocket(
    websocket: WebSocket,
    *,
    sanitize_session_config: SanitizeFn,
    initial_session: dict[str, Any] | None = None,
    ensure_services_ready: EnsureReadyFn | None = None,
    prepare_initial_runtime: PrepareRuntimeFn | None = None,
    start_bot: StartBotFn | None = None,
    resolve_server_tools: ServerToolsFn | None = None,
    resolve_delegate_tools: DelegateToolsFn | None = None,
    resolve_server_tool_schemas: ToolSchemasFn | None = None,
    resolve_delegate_tool_schemas: ToolSchemasFn | None = None,
    resolve_session_capabilities: ResolveCapabilitiesFn | None = None,
    resolve_model_route: ResolveModelRouteFn | None = None,
    default_example_key: str = "",
    default_pipeline_mode: str = _DEFAULT_PIPELINE_MODE,
    session_max_duration_secs: float = _OPENAI_REALTIME_SESSION_MAX_DURATION_SECS,
) -> None:
    """Run one WebSocket session within OpenAI's 60-minute lifetime.

    ``session_max_duration_secs`` exists only to exercise the deadline without
    sleeping for an hour in protocol tests. Production callers use the fixed
    OpenAI-compatible default.
    """
    if isinstance(session_max_duration_secs, bool) or session_max_duration_secs <= 0:
        raise ValueError("session_max_duration_secs must be positive")

    lifetime = asyncio.timeout(float(session_max_duration_secs))
    try:
        async with lifetime:
            await _run_realtime_websocket(
                websocket,
                sanitize_session_config=sanitize_session_config,
                initial_session=initial_session,
                ensure_services_ready=ensure_services_ready,
                prepare_initial_runtime=prepare_initial_runtime,
                start_bot=start_bot,
                resolve_server_tools=resolve_server_tools,
                resolve_delegate_tools=resolve_delegate_tools,
                resolve_server_tool_schemas=resolve_server_tool_schemas,
                resolve_delegate_tool_schemas=resolve_delegate_tool_schemas,
                resolve_session_capabilities=resolve_session_capabilities,
                resolve_model_route=resolve_model_route,
                default_example_key=default_example_key,
                default_pipeline_mode=default_pipeline_mode,
            )
    except TimeoutError:
        if not lifetime.expired():
            raise
        logger.info("Realtime WS reached the 60-minute session lifetime")
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close(code=1000, reason="realtime session expired")
