# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The strict startup schema of a live session: what a client may send in ``session`` and the limits on it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from live.protocol.errors import ProtocolError, approx_tokens
from live.protocol.voices import LIVE_VOICES, TTS_VOICES


class StrictModel(BaseModel):
    """Base for protocol models: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


class AudioFormat(StrictModel):
    """Wire audio format: 16 or 24 kHz PCM, or 8 kHz G.711."""

    type: Literal["audio/pcm", "audio/pcmu", "audio/pcma"] = "audio/pcm"
    rate: Literal[8000, 16000, 24000] = 24000

    @model_validator(mode="after")
    def valid_rate(self):
        """Reject a rate that the type does not allow."""
        if (self.type == "audio/pcm" and self.rate not in (16000, 24000)) or (
            self.type != "audio/pcm" and self.rate != 8000
        ):
            raise ValueError("PCM requires 16/24 kHz; G.711 requires 8 kHz")
        return self


class AudioOutput(StrictModel):
    """Output audio settings: the voice, by name or as a custom ``{"id": ...}``."""

    voice: str | dict[str, str] = "marin"

    @model_validator(mode="after")
    def valid_voice(self):
        """Reject voice names that no TTS can serve."""
        if isinstance(self.voice, str) and self.voice not in LIVE_VOICES | TTS_VOICES:
            raise ValueError("Unknown voice")
        if isinstance(self.voice, dict) and (set(self.voice) != {"id"} or not self.voice["id"]):
            raise ValueError("Custom voice requires a non-empty id")
        return self


class Audio(StrictModel):
    """Session audio: an optional wire format (WebSocket only) and the output voice."""

    format: AudioFormat | None = None
    output: AudioOutput = Field(default_factory=AudioOutput)


class BackendConfig(StrictModel):
    """Settings for the server-owned backend (``delegation.responses``)."""

    model: str = Field(min_length=1)
    instructions: str | None = None
    tools: list[dict] = Field(default_factory=list)
    tool_choice: str | dict = "auto"
    parallel_tool_calls: bool = True
    reasoning: dict | None = None
    text: dict | None = None
    service_tier: Literal["auto", "default", "flex", "priority"] = "auto"
    max_output_tokens: int | None = Field(default=None, ge=16)

    @model_validator(mode="after")
    def supported_tools(self):
        """Reject tool types other than ``function`` and ``web_search``."""
        if any(not isinstance(t, dict) or t.get("type") not in {"function", "web_search"} for t in self.tools):
            raise ValueError("Live delegation supports function and web_search tools")
        if any(t["type"] == "function" and not (isinstance(t.get("name"), str) and t["name"]) for t in self.tools):
            raise ValueError("A function tool needs a name")
        if isinstance(self.tool_choice, str) and self.tool_choice not in {
            "auto",
            "none",
            "required",
        }:
            raise ValueError("Invalid tool_choice")
        return self


class Delegation(StrictModel):
    """Who owns delegated work: the ``client`` or a server-owned ``responses`` backend."""

    type: Literal["client", "responses"] = "client"
    responses: BackendConfig | None = None

    @model_validator(mode="after")
    def valid_mode(self):
        """Require backend settings exactly when the delegation type is ``responses``."""
        if (self.type == "responses") != (self.responses is not None):
            raise ValueError("Only Responses mode requires responses configuration")
        return self


class Permissions(StrictModel):
    """Which events may cross the WebRTC data channel in each direction."""

    allowed_client_events: Literal["all"] | list[str] = "all"
    allowed_server_events: Literal["all"] | list[str | dict[str, str]] = "all"


class Client(StrictModel):
    """Client-side permissions (WebRTC only)."""

    data_channel: Permissions = Field(default_factory=Permissions)


class SessionConfig(StrictModel):
    """The immutable startup configuration of a session, with its size limits."""

    model: str = Field(min_length=1)
    instructions: str | None = None
    input: list[dict] = Field(default_factory=list)
    audio: Audio = Field(default_factory=Audio)
    delegation: Delegation | None = None
    client: Client | None = None
    store: bool = False

    @model_validator(mode="after")
    def limits(self):
        """Enforce the instruction and input-history size limits."""
        if approx_tokens(self.instructions or "") > 16384:
            raise ValueError("instructions exceeds 16,384 tokens")
        if len(self.input) > 128:
            raise ValueError("input exceeds 128 messages")
        total = 0
        for item in self.input:
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
        if total > 8192:
            raise ValueError("input exceeds 8,192 tokens")
        return self

    @property
    def mode(self) -> str:
        """Return the delegation mode: ``client`` or ``responses``."""
        return self.delegation.type if self.delegation else "client"

    def validate_transport(self, transport: str):
        """Reject settings that the transport cannot honor."""
        if transport == "webrtc" and self.audio.format is not None:
            raise ProtocolError("WebRTC negotiates audio format in SDP", param="session.audio.format")
        if transport == "websocket" and self.client is not None:
            raise ProtocolError("client.data_channel is only supported for WebRTC sessions", param="session.client")
