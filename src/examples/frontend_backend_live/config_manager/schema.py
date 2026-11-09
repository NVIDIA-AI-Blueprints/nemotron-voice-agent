# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed settings of the example: the ``config`` block of its ``examples_registry.yaml`` entry."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from examples.shared.delegation_contract import ModelEndpoint as SharedEndpoint
from examples.shared.strict_settings import Seconds, WholeNumber, with_blocks
from examples.shared.strict_settings import StrictSettings as StrictModel

Name = Annotated[str, Field(strict=True, min_length=1)]


def merged(base: Any, changes: Any) -> Any:
    """Return ``changes`` over ``base``: mappings merge key by key, anything else replaces, ``null`` included."""
    if not isinstance(base, Mapping) or not isinstance(changes, Mapping):
        return deepcopy(changes)
    result = {key: deepcopy(value) for key, value in base.items()}
    for key, value in changes.items():
        result[key] = merged(result[key], value) if key in result else deepcopy(value)
    return result


@dataclass(frozen=True)
class ModelEndpoint(SharedEndpoint):
    """One chat-style model endpoint, with the frontend's reply format and catalog key on top of the shared fields."""

    response_format: str = "json_schema"
    service_key: str = ""


class ReliabilityConfig(StrictModel):
    """The talker's timeout and retries; a retry never wraps an action that could repeat."""

    # The model client's own retries of transient failures; 0: none.
    provider_retries: Annotated[int, Field(strict=True, ge=0)] = 2
    # One talker call, from its request to the end of its decision.
    talker_attempt_timeout_seconds: Seconds = 10.0


class DelegationConfig(StrictModel):
    """What the voice does around its delegations: the app wait and spoken progress."""

    # The app-wait line after this long; null: none. Below the core's deadline, so it comes before the work fails.
    app_wait_seconds: Seconds | None = 50.0
    # A progress line may come after this long of silent work, a next one after a step ended or a note; null: none.
    progress_after_seconds: Seconds | None = 8.0
    # The progress lines offered for one delegation.
    progress_max_per_delegation: WholeNumber = 3


class MemoryConfig(StrictModel):
    """The talker's memory."""

    # The talker's memory budget.
    context_tokens: WholeNumber = 16_000
    # Whole turn groups are dropped once memory passes this share of the budget.
    compact_at: Annotated[float, Field(strict=True, gt=0, le=1)] = 0.8


class SpeechConfig(StrictModel):
    """How the caller's turn starts and ends (``media/turns.py``)."""

    # Speech this long marks the caller as speaking.
    vad_start_seconds: Seconds = 0.12
    # The pause Smart Turn judges, below turn_silence_seconds.
    vad_stop_seconds: Seconds = 0.2
    # A pause this long ends the turn, whatever Smart Turn judged.
    turn_silence_seconds: Seconds = 1.2
    # The words that start the caller's turn while the bot speaks.
    interrupt_min_words: WholeNumber = 2


class LiveConfig(StrictModel):
    """The cascade's settings: the ``frontend.cascade`` block; a part left out, or null, keeps its defaults."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="frontend.cascade")

    slot: Name
    # The talker's style by its model's services key: json (also for a key not listed) or tools.
    talker_style: dict[Name, Name] = {}
    # The talker prompt (agent_prompts.Talker) of a session that sets no instructions.
    prompt: Name = "generic"
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    delegation: DelegationConfig = Field(default_factory=DelegationConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    # Reply audio held before playback starts, so uneven TTS chunks do not stall it; 0: off.
    tts_preroll_ms: Annotated[int, Field(strict=True, ge=0, le=2000)] = 300
    # The root hands these two over from the core's ``delegation`` block.
    display_label: Name = "delegation_core"
    client_tool_timeout_seconds: Seconds | None = None

    @field_validator("talker_style", "reliability", "delegation", "memory", "speech", mode="before")
    @classmethod
    def null_is_the_defaults(cls, part: Any) -> Any:
        """Keep a part's defaults when it is set to null."""
        return {} if part is None else part

    def style_of(self, service_key: str) -> str:
        """Return the talker's style for the model whose services key is ``service_key``; ``json`` by default."""
        return self.talker_style.get(service_key, "json")


class FrontendChoice(StrictModel):
    """The frontend adapter that serves each live session (``frontend.adapter``), and each adapter's block."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="frontend")

    adapter: Name
    # Every other key of ``frontend``: one adapter's own block, as written.
    blocks: dict[str, dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def adapter_blocks(cls, raw: Any) -> Any:
        """Collect every key besides ``adapter`` as one adapter's block."""
        if not isinstance(raw, Mapping):
            return raw
        return with_blocks(raw, set(cls.model_fields) - {"blocks"})

    @model_validator(mode="after")
    def the_adapter_has_its_block(self) -> FrontendChoice:
        """Check that ``adapter`` names a block, and that no block sets what the root hands it."""
        for name, block in self.blocks.items():
            for key in ("display_label", "client_tool_timeout_seconds"):
                if key in block:
                    raise ValueError(f"frontend.{name}.{key} comes from the core's delegation.{key}")
        if self.adapter not in self.blocks:
            raise ValueError(f"frontend.adapter names {self.adapter!r}, but there is no frontend.{self.adapter} block")
        return self

    @property
    def block(self) -> dict[str, Any]:
        """The block the chosen frontend adapter reads: ``frontend.<adapter>``."""
        return self.blocks[self.adapter]


class UiBotConfig(StrictModel):
    """The repository UI bot's own settings (``ui_bot``)."""

    channel: Name
    # What the bot's own core changes in the core's settings, merged over them key by key.
    core: dict[str, Any] = {}


class ExampleConfig(StrictModel):
    """Everything the example's ``config`` block sets: the frontend, the delegation core, and the UI bot."""

    model_config = ConfigDict(extra="forbid", frozen=True, title="the example's settings")

    frontend: FrontendChoice
    core: dict[str, Any]
    ui_bot: UiBotConfig

    @property
    def ui_bot_core(self) -> dict[str, Any]:
        """The UI bot's delegation core settings: the core's, with ``ui_bot.core`` merged over them."""
        return merged(self.core, self.ui_bot.core)
