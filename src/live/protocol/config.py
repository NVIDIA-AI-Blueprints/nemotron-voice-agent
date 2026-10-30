# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strict startup schema of a live session: what a client may send in ``session`` and the limits on it.

A session's ``delegation`` is one of two classes, picked by its official ``type`` from :data:`DELEGATION_CONFIGS`:
:class:`ClientDelegationConfig` (``client``; an omitted, ``null`` or ``{}`` delegation resolves to it, as GPT-Live's
"Omitted or null selects your application") and :class:`ResponsesDelegationConfig` (``responses``). Each keeps
its own ``session.update`` rule (:meth:`ClientDelegationConfig.updated`, :meth:`ResponsesDelegationConfig.updated`), so
nothing outside the class compares the type.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from live.protocol.errors import ProtocolError, approx_tokens, validation_error
from live.protocol.voices import LIVE_VOICES, TTS_VOICES


class StrictModel(BaseModel):
    """Base for protocol models: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


# The one rate each G.711 wire format runs at.
G711_RATES = {"audio/pcmu": 8000, "audio/pcma": 8000}


class AudioFormat(StrictModel):
    """Wire audio format: 16 or 24 kHz PCM, or 8 kHz G.711; an omitted rate is the format's own (24 kHz, or 8 kHz)."""

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: Literal[8000, 16000, 24000] = 24000

    @model_validator(mode="before")
    @classmethod
    def default_rate(cls, value: Any) -> Any:
        """G.711 has one rate, so a G.711 format without one takes 8000 Hz; PCM keeps its 24 kHz default."""
        if isinstance(value, dict) and "rate" not in value:
            return {**value, "rate": G711_RATES.get(value.get("type"), 24000)}
        return value

    @model_validator(mode="after")
    def valid_rate(self):
        """Reject a rate that the type does not allow."""
        if (self.type == "audio/pcm" and self.rate not in (16000, 24000)) or (
            self.type != "audio/pcm" and self.rate != 8000
        ):
            raise ValueError("PCM requires 16/24 kHz; G.711 requires 8 kHz")
        return self


class AudioOutput(StrictModel):
    """Output audio settings: the voice, as a built-in name or a custom ``{"id": ...}``."""

    voice: str | dict[str, str] = "marin"

    @model_validator(mode="after")
    def valid_voice(self):
        """Reject a name that is neither a GPT-Live voice nor one of the speech service's own."""
        if isinstance(self.voice, str) and self.voice not in LIVE_VOICES | TTS_VOICES:
            raise ValueError("Unknown voice: use a built-in voice name or a custom voice {id}")
        if isinstance(self.voice, dict) and (set(self.voice) != {"id"} or not self.voice["id"]):
            raise ValueError("Custom voice requires a non-empty id")
        return self


class Audio(StrictModel):
    """Session audio: an optional wire format (WebSocket only) and the output voice."""

    format: AudioFormat | None = None
    output: AudioOutput = Field(default_factory=AudioOutput)


class TextConfig(StrictModel):
    """``delegation.responses.text``: only the backend's verbosity."""

    verbosity: Literal["low", "medium", "high"] | None = None


# The official tool types of ``delegation.responses.tools``: whether the backend supports one is the
# background validation's to report.
TOOL_TYPES = frozenset(
    {
        "function",
        "web_search",
        "file_search",
        "code_interpreter",
        "shell",
        "image_generation",
        "mcp",
        "custom",
        "namespace",
        "tool_search",
        "programmatic_tool_calling",
        "computer",
        "apply_patch",
    }
)
# What a tool of each type keeps out of the settings the session sends back: an MCP server's credentials, which go to
# the backend but never back to the application.
TOOL_SECRETS = {"mcp": frozenset({"authorization", "headers"})}


def without_secrets(tool: dict) -> dict:
    """``tool`` as the session sends it back: without the credentials its type keeps out (:data:`TOOL_SECRETS`)."""
    secrets = TOOL_SECRETS.get(tool.get("type"), frozenset())
    return {key: value for key, value in tool.items() if key not in secrets}


class BackendConfig(StrictModel):
    """Settings for the server-owned backend (``delegation.responses``).

    Every documented value is accepted, every official tool type included; one the backend cannot honor is reported by
    the background validation as a nonfatal ``responses_validation_failed``, and the setting is kept.
    """

    model: str = Field(min_length=1)
    instructions: str | None = None
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = "auto"
    parallel_tool_calls: bool | None = True
    reasoning: dict | None = None
    text: TextConfig | None = None
    service_tier: Literal["auto", "default", "fast_tier_temp_pilot", "flex", "priority", "ultrafast"] | None = "auto"
    max_output_tokens: int | None = Field(default=None, ge=16)

    @model_validator(mode="after")
    def official_tools(self):
        """Reject a tool that is not an object of an official type, and a function tool without a name."""
        if any(not isinstance(t, dict) or t.get("type") not in TOOL_TYPES for t in self.tools):
            raise ValueError(f"Each tool must be an object whose type is one of: {', '.join(sorted(TOOL_TYPES))}")
        if any(t["type"] == "function" and not (isinstance(t.get("name"), str) and t["name"]) for t in self.tools):
            raise ValueError("A function tool needs a name")
        if isinstance(self.tool_choice, str) and self.tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            raise ValueError("Invalid tool_choice")
        return self


def _merged(current: dict, patch: dict) -> dict:
    """``current`` with a sparse ``patch``: a nested object merges, so a new ``reasoning.effort`` keeps its summary."""
    merged = dict(current)
    for key, value in patch.items():
        merged[key] = (
            {**merged[key], **value} if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def _delegation_changes(patch: object, current_type: str) -> dict:
    """A ``session.update``'s delegation block as changes to the current delegation, whose type it may not change."""
    if not isinstance(patch, dict):
        raise ProtocolError("delegation must be an object or null", param="session.delegation")
    if patch.get("type", current_type) != current_type:
        raise ProtocolError(
            "The delegation type cannot change during a session; start a new session to change it",
            "immutable_field_update",
            "session.delegation.type",
        )
    return {key: value for key, value in patch.items() if key != "type"}


class ClientDelegationConfig(StrictModel):
    """Client delegation: the application does the delegated work, and the session announces each delegation."""

    type: Literal["client"] = "client"
    # Milliseconds before the automatic spoken acknowledgement of a delegation: 0 asks at once.
    acknowledgement_delay_ms: int = Field(default=0, ge=0)

    @property
    def responses(self) -> None:
        """No Responses backend settings: the application runs its own backend."""
        return None

    def updated(self, patch: object) -> ClientDelegationConfig:
        """This delegation after a ``session.update``'s ``delegation`` block.

        ``null`` and ``{}`` change nothing; ``acknowledgement_delay_ms`` applies to later delegations. A type change is
        ``immutable_field_update``; a ``responses`` (or any other) key is unknown here, so pydantic refuses it.

        Raises:
            ProtocolError: The block changes the type or is not an object.
            ValidationError: The block holds a key or a value this class refuses.
        """
        if patch is None:
            return self
        return type(self).model_validate({**self.model_dump(), **_delegation_changes(patch, self.type)})


class ResponsesDelegationConfig(StrictModel):
    """Responses delegation: the session's delegation core does the work with these backend settings."""

    type: Literal["responses"]
    responses: BackendConfig
    acknowledgement_delay_ms: int = Field(default=0, ge=0)

    def updated(self, patch: object) -> ResponsesDelegationConfig:
        """This delegation after a ``session.update``'s ``delegation`` block.

        The block may change ``acknowledgement_delay_ms`` and, sparsely, the ``responses`` settings: omitted settings
        keep their values, and a nested object merges. ``null`` would select client mode, which a session cannot switch
        to; a type change likewise.

        Raises:
            ProtocolError: The block is ``null``, changes the type or is not an object.
            ValidationError: The block holds a key or a value this class refuses.
        """
        if patch is None:
            raise ProtocolError(
                "A null delegation selects client mode, which a running Responses session cannot switch to",
                "immutable_field_update",
                "session.delegation",
            )
        changes = _delegation_changes(patch, self.type)
        current = self.model_dump()
        if isinstance(changes.get("responses"), dict):
            changes["responses"] = _merged(current["responses"], changes["responses"])
        return type(self).model_validate({**current, **changes})


# The delegation settings each official delegation type takes: the one place the type picks a class.
DELEGATION_CONFIGS: dict[str, type[ClientDelegationConfig | ResponsesDelegationConfig]] = {
    "client": ClientDelegationConfig,
    "responses": ResponsesDelegationConfig,
}
# What a session.update may change; every other startup setting is immutable.
UPDATABLE = frozenset({"delegation", "triggers"})


class ServerEventSelector(StrictModel):
    """One server event a WebRTC data channel may carry; a Responses event names its nested type."""

    type: str
    response_event: str | None = None

    @model_validator(mode="after")
    def nested_type(self):
        """``response_event`` is required for ``response.event`` and forbidden for every other type."""
        if (self.type == "response.event") != (self.response_event is not None):
            raise ValueError("response_event is required when type is response.event, and forbidden otherwise")
        return self


class Permissions(StrictModel):
    """Which events may cross the WebRTC data channel in each direction."""

    allowed_client_events: Literal["all"] | list[str] = "all"
    allowed_server_events: Literal["all"] | list[str | ServerEventSelector] = "all"

    @property
    def restricted(self) -> bool:
        """Whether either direction carries less than every event."""
        return self.allowed_client_events != "all" or self.allowed_server_events != "all"


class Client(StrictModel):
    """Client-side permissions (WebRTC only)."""

    data_channel: Permissions = Field(default_factory=Permissions)


# The startup limits on the session's own instructions and history, as GPT-Live's session.start sets them.
MAX_INSTRUCTION_TOKENS = 16384
MAX_INPUT_MESSAGES = 128
MAX_INPUT_TOKENS = 8192


class SessionConfig(StrictModel):
    """The immutable startup configuration of a session, with its size limits."""

    model: str = Field(min_length=1)
    instructions: str | None = None
    input: list[dict] = Field(default_factory=list)
    audio: Audio = Field(default_factory=Audio)
    # Omitted, null or {}: Live client mode, as in official GPT-Live ("Omitted or null selects your application").
    delegation: ClientDelegationConfig | ResponsesDelegationConfig = Field(default_factory=ClientDelegationConfig)
    client: Client | None = None
    store: bool = False
    # Decisions triggers: this server runs no Decisions, so only none ([] or null) is accepted.
    triggers: list | None = None

    @field_validator("triggers")
    @classmethod
    def no_triggers(cls, value: list | None) -> list | None:
        """Refuse Decisions triggers, which this server does not run."""
        if value:
            raise ValueError("Decisions triggers are not supported by this server")
        return value

    @field_validator("delegation", mode="before")
    @classmethod
    def resolved_delegation(cls, value: Any) -> Any:
        """Pick the delegation's class by its official type; an omitted type, ``null`` or ``{}`` is client mode."""
        if value is None:
            value = {}
        if not isinstance(value, dict):
            return value
        kind = value.get("type", "client")
        if kind not in DELEGATION_CONFIGS:
            raise ValueError(f"Unsupported delegation type {kind!r}: this server delegates to client or responses")
        return DELEGATION_CONFIGS[kind].model_validate(value)

    @field_validator("instructions")
    @classmethod
    def instruction_limit(cls, value: str | None) -> str | None:
        """Enforce the instructions' size limit."""
        if approx_tokens(value or "") > MAX_INSTRUCTION_TOKENS:
            raise ValueError(f"instructions exceeds {MAX_INSTRUCTION_TOKENS:,} tokens")
        return value

    @field_validator("input")
    @classmethod
    def input_limits(cls, value: list[dict]) -> list[dict]:
        """Enforce the input history's shape and size limits."""
        if len(value) > MAX_INPUT_MESSAGES:
            raise ValueError(f"input exceeds {MAX_INPUT_MESSAGES} messages")
        total = 0
        for item in value:
            parts = item.get("content", []) if isinstance(item, dict) else None
            role = item.get("role") if isinstance(item, dict) else None
            types = {"text", "output_text"} if role == "assistant" else {"input_text"}
            if (
                not isinstance(item, dict)
                or item.get("type", "message") != "message"
                or not isinstance(role, str)
                or role not in {"developer", "user", "assistant"}
                or not isinstance(parts, list)
                or len(parts) != 1
                or not isinstance(parts[0], dict)
                or parts[0].get("type") not in types
                or not isinstance(parts[0].get("text"), str)
            ):
                raise ValueError("input requires text-only messages with exactly one text part")
            total += approx_tokens(parts[0]["text"])
        if total > MAX_INPUT_TOKENS:
            raise ValueError(f"input exceeds {MAX_INPUT_TOKENS:,} tokens")
        return value

    @property
    def mode(self) -> str:
        """Return the resolved delegation mode: ``client`` or ``responses``."""
        return self.delegation.type

    def updated(self, patch: object) -> SessionConfig:
        """These settings after a ``session.update`` patch; the patch's delegation goes to its class.

        Only ``delegation`` and ``triggers`` may appear: any other startup setting is ``immutable_field_update``, an
        unknown key ``unknown_parameter``. This server runs no Decisions, so ``triggers`` is refused. An empty patch
        changes nothing.

        Raises:
            ProtocolError: The patch is refused, with its ``param``.
        """
        if not isinstance(patch, dict):
            raise ProtocolError("session must be an object", param="session")
        for key in patch:
            if key not in UPDATABLE:
                known = key in type(self).model_fields
                raise ProtocolError(
                    f"session.{key} cannot change after startup" if known else f"Unknown parameter: 'session.{key}'.",
                    "immutable_field_update" if known else "unknown_parameter",
                    f"session.{key}",
                )
        if patch.get("triggers"):
            raise ProtocolError("Decisions triggers are not supported by this server", param="session.triggers")
        if "delegation" not in patch:
            return self
        try:
            delegation = self.delegation.updated(patch["delegation"])
        except ValidationError as refused:
            raise validation_error(refused, "session.delegation") from refused
        return self.model_copy(update={"delegation": delegation})

    def public_dump(self) -> dict:
        """These settings as the session's events carry them: an MCP tool's credentials are left out.

        The settings the session keeps still hold them, so each request carries them and an update that leaves the
        tools alone keeps them; an update that replaces the tools must send them again.
        """
        data = self.model_dump(exclude_none=True)
        responses = data["delegation"].get("responses")
        if responses is not None:
            responses["tools"] = [without_secrets(tool) for tool in responses["tools"]]
        return data

    def validate_transport(self, transport: str):
        """Reject settings that the transport cannot honor."""
        if transport == "webrtc" and self.audio.format is not None:
            raise ProtocolError("WebRTC negotiates audio format in SDP", param="session.audio.format")
        if transport == "websocket" and self.client is not None:
            raise ProtocolError("client.data_channel is only supported for WebRTC sessions", param="session.client")

    def with_wire_format(self) -> SessionConfig:
        """These settings with the WebSocket wire format resolved: the client's, or 24 kHz PCM when it set none."""
        if self.audio.format is not None:
            return self
        return self.model_copy(update={"audio": self.audio.model_copy(update={"format": AudioFormat()})})
