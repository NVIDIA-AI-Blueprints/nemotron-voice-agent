# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed settings of the example: the ``config`` block of its ``examples_registry.yaml`` entry. No provider is imported.

:class:`ExampleConfig` is the whole block: the frontend adapter that serves live sessions (``frontend.adapter``) with
each adapter's own block (``frontend.<name>``), the delegation core's settings (``core``), and the UI bot's channel
context and core (``ui_bot``). The cascade's block is :class:`LiveConfig`, which the cascade adapter's class validates.
The core's settings are the core's own to validate (``examples.frontend_backend_live.delegation_core.config``), so they
are kept here as written. Each model is a pydantic model, like the Live gateway's session config: an unknown key is an
error, and every default lives in its model.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from examples.shared.delegation_contract import ModelEndpoint as SharedEndpoint


class StrictModel(BaseModel):
    """Base of the example's settings models: frozen, and an unknown key is an error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# A name: non-empty text.
Name = Annotated[str, Field(strict=True, min_length=1)]
# A positive whole number; a fraction or a boolean is refused.
WholeNumber = Annotated[int, Field(strict=True, gt=0)]
# A positive number of seconds (a whole number is one too).
Seconds = Annotated[float, Field(strict=True, gt=0)]


def merged(base: Any, changes: Any) -> Any:
    """``changes`` over ``base``: mappings merge key by key, anything else replaces (``null`` included)."""
    if not isinstance(base, Mapping) or not isinstance(changes, Mapping):
        return deepcopy(changes)
    result = {key: deepcopy(value) for key, value in base.items()}
    for key, value in changes.items():
        result[key] = merged(result[key], value) if key in result else deepcopy(value)
    return result


@dataclass(frozen=True)
class ModelEndpoint(SharedEndpoint):
    """One chat-style model endpoint, with the frontend's own reply format and entry key on top of the shared fields."""

    # Frontend only: json_schema or none, the talker's decision format; nothing steps down.
    response_format: str = "json_schema"
    # Frontend only: the services key of the catalog entry it comes from ("" for none); the cascade's talker_style names
    # the talker's style by it.
    service_key: str = ""


class ReliabilityConfig(StrictModel):
    """The talker's timeout and retries. Retries never wrap an action that could repeat."""

    # The model client's own retries of transient failures; 0: none.
    provider_retries: Annotated[int, Field(strict=True, ge=0)] = 2
    talker_attempt_timeout_seconds: Seconds = 10.0  # one talker call, from its request to the end of its decision


class DelegationConfig(StrictModel):
    """What the voice does around its delegations: the app wait and spoken progress."""

    # How long the voice waits for the app's outputs to a delegation before it says the app-wait line; null: no line.
    # It is below the core's deadline for those outputs (the core's delegation.client_tool_timeout_seconds).
    app_wait_seconds: Seconds | None = 50.0
    # The talker may word a short progress line when work has run this long without the assistant saying anything;
    # null: none. A later line for the same delegation needs a backend step that ended or a new progress note.
    progress_after_seconds: Seconds | None = 8.0
    progress_max_per_delegation: WholeNumber = 3


class MemoryConfig(StrictModel):
    """The talker's memory."""

    context_tokens: WholeNumber = 16_000  # the talker's memory budget
    # Whole turn groups are dropped once memory passes this share of the budget: above 0, at most 1.
    compact_at: Annotated[float, Field(strict=True, gt=0, le=1)] = 0.8


class SpeechConfig(StrictModel):
    """How the caller's turn starts and ends (media/turns.py)."""

    # Speech this long marks the caller as speaking; while the bot is silent, that first sound starts their turn.
    vad_start_seconds: Seconds = 0.12
    vad_stop_seconds: Seconds = 0.2  # the pause Smart Turn judges, below turn_silence_seconds
    turn_silence_seconds: Seconds = 1.2  # a pause this long ends the turn whatever Smart Turn judged
    # While the bot speaks, the words that start the caller's turn and so cut it; while it is silent, the caller's first
    # sound starts it.
    interrupt_min_words: WholeNumber = 2


class LiveConfig(StrictModel):
    """The cascade's settings: the ``frontend.cascade`` block.

    ``slot`` names the ``examples_registry.yaml`` services slot whose selected ``services.yaml`` entry is the talker's
    model, ``talker_style`` the talker's style by that entry's services key (:meth:`style_of`), and ``prompt`` the
    talker prompt of ``prompts.yaml`` (``agent_prompts.Talker``) used, as written, as the talker's instructions when a
    session supplies none or blank ones. A part left out, or null, keeps its defaults.
    ``display_label`` is not a key of the block: the root hands the cascade the delegation core's
    ``delegation.display_label`` (the server label of the core's display items), so the label is written once.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="frontend.cascade")

    slot: Name
    # The talker's style by the services key of the slot's selected entry, its service in talker.LLM_SERVICES: json
    # (the JSON decision, for a key not listed) or tools (the routing decision with function calling).
    talker_style: dict[Name, Name] = {}
    prompt: Name = "generic"
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    delegation: DelegationConfig = Field(default_factory=DelegationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    # Reply audio held before playback starts, so uneven TTS chunks do not stall it (0 sends audio at once), at most
    # 2000 ms.
    tts_preroll_ms: Annotated[int, Field(strict=True, ge=0, le=2000)] = 300
    display_label: Name = "delegation_core"

    @field_validator("talker_style", "reliability", "delegation", "memory", "speech", mode="before")
    @classmethod
    def null_is_the_defaults(cls, part: Any) -> Any:
        """A part set to null keeps its defaults."""
        return {} if part is None else part

    def style_of(self, service_key: str) -> str:
        """The talker's style for the model whose services key is ``service_key``: json unless the block names one."""
        return self.talker_style.get(service_key, "json")


class FrontendChoice(StrictModel):
    """The frontend adapter that serves each live session (``frontend.adapter``), and each adapter's own block.

    Every key besides ``adapter`` is one adapter's block (``blocks``), kept as written for its class to validate.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, title="frontend")

    adapter: Name  # a FRONTEND_ADAPTERS name
    blocks: dict[str, dict[str, Any]] = {}  # frontend.<name>: each adapter's own block, as written

    @model_validator(mode="before")
    @classmethod
    def adapter_blocks(cls, raw: Any) -> Any:
        """Every key besides ``adapter`` is one adapter's block."""
        if not isinstance(raw, Mapping):
            return raw
        blocks = {str(key): block for key, block in raw.items() if key != "adapter"}
        return {**({"adapter": raw["adapter"]} if "adapter" in raw else {}), "blocks": blocks}

    @model_validator(mode="after")
    def the_adapter_has_its_block(self) -> FrontendChoice:
        """``adapter`` names a block, and no block sets what the root hands it (the core's display label)."""
        for name, block in self.blocks.items():
            if "display_label" in block:
                raise ValueError(f"frontend.{name}.display_label comes from the core's delegation.display_label")
        if self.adapter not in self.blocks:
            raise ValueError(f"frontend.adapter names {self.adapter!r}, but there is no frontend.{self.adapter} block")
        return self

    @property
    def block(self) -> dict[str, Any]:
        """The block the chosen frontend adapter reads: ``frontend.<adapter>``."""
        return self.blocks[self.adapter]


class UiBotConfig(StrictModel):
    """The repository UI bot's own settings (``ui_bot``)."""

    channel: Name  # agent_prompts.Channel: the context after the bot's talker and backend prompts
    # What the bot's own delegation core changes from the core's settings, merged over them key by key: its backend
    # prompt and server tools, the display part of its results, and the llm adapter whatever the core's adapter is.
    core: dict[str, Any] = {}


class ExampleConfig(StrictModel):
    """Everything the example's ``config`` block sets: the frontend, the delegation core, and the UI bot."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="the example's settings")

    frontend: FrontendChoice
    core: dict[str, Any]  # the delegation core's settings, which the core validates with its texts
    ui_bot: UiBotConfig

    @property
    def ui_bot_core(self) -> dict[str, Any]:
        """The UI bot's delegation core settings: the core's, with ``ui_bot.core`` merged over them."""
        return merged(self.core, self.ui_bot.core)
