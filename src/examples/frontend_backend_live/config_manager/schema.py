# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed configuration of the example's own file, ``config.yaml``, and its checks. No provider is imported here.

``config.yaml`` holds the frontend's settings only. It names the frontend adapter that serves live sessions
(``frontend.adapter``), holds each adapter's own block (``frontend.<name>``), and names the UI bot's core profile
(``ui_bot.core_config``): :class:`ExampleConfig`. The cascade's block is :class:`LiveConfig`, which the cascade
adapter's class parses (``LiveConfig.parse``). The delegation core reads its own file (``core.yaml``,
``delegation_core.config``); a key that belongs elsewhere stops loading with a message that names where the setting
lives (:data:`MOVED`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any

from examples.frontend_backend_live.prompts import PROMPT_SETS
from examples.frontend_backend_live.tables import pick

DEFAULT_CONTINUERS = ("uh-huh", "uh huh", "mm-hmm", "mm hmm", "mhm", "hmm", "mm", "mmm", "ah", "oh", "i see", "got it")
# Every other number is a timeout, a limit or a count, where 0 would disable or break the feature.
ZERO_ALLOWED = frozenset({"provider_retries"})
# Settings a profile may turn off with null.
NULL_ALLOWED = frozenset({"app_wait_seconds", "progress_after_seconds"})
# The cascade's keys that sat at the top of config.yaml before the file named its frontend adapter.
CASCADE_KEYS = (
    "prompt_version",
    "reliability",
    "guards",
    "delegation",
    "normalizer",
    "memory",
    "speech",
    "tts_preroll_ms",
)

# Keys config.yaml does not accept, by their path in the file, each with where its setting lives.
MOVED = MappingProxyType(
    {
        "backend": "the delegation core's file: core.yaml backend (slot and prompt; provider is backend.llm.provider)",
        "voice_note": "core.yaml delegation.voice_note",
        "backend_context_tokens": "core.yaml delegation.context_tokens",
        "chat_history_recent_turns": "frontend.cascade.memory.context_tokens (the talker's memory is a token budget)",
        **{key: f"frontend.cascade.{key}" for key in CASCADE_KEYS},
        "frontend.provider": "frontend.cascade.provider (frontend.adapter names the frontend adapter)",
        "frontend.slot": "frontend.cascade.slot",
        "frontend.prompt": "frontend.cascade.prompt",
        "frontend.cascade.display_label": "core.yaml delegation.display_label (the root hands it to the frontend)",
        "frontend.cascade.chat_history_recent_turns": "frontend.cascade.memory.context_tokens",
        "frontend.cascade.reliability.api_timeout_seconds": (
            "frontend.cascade.reliability.client_timeout_seconds for the talker's model, and the backend LLM's "
            "first_event_timeout_seconds and idle_timeout_seconds in core.yaml backend.llm"
        ),
        "frontend.cascade.reliability.tool_timeout_seconds": (
            "core.yaml delegation.client_tool_timeout_seconds (the IM's deadline for the app's outputs), core.yaml "
            "backend.llm.server_tool_timeout_seconds, and frontend.cascade.delegation.app_wait_seconds (the voice's "
            "app-wait line)"
        ),
        "frontend.cascade.reliability.max_tool_rounds": "core.yaml backend.llm.max_tool_rounds",
        "frontend.cascade.reliability.max_pending_delegations": "core.yaml delegation.max_pending_delegations",
        "frontend.cascade.reliability.recover_failures": (
            "core.yaml backend.llm.provider_retries; the voice says its own lines (prompts.yaml lines)"
        ),
        "frontend.cascade.delegation.merge_pending": (
            "core.yaml backend.llm.conductor (steer joins open work; start_each gives each request its own)"
        ),
        "frontend.cascade.delegation.fold_before_acting": "core.yaml backend.llm.conductor",
        "frontend.cascade.delegation.status_from_ledger": (
            "nothing: the talker answers a status question from its work line, or delegates it"
        ),
        "frontend.cascade.delegation.progress_speech": "frontend.cascade.delegation.progress_after_seconds (null: off)",
        "frontend.cascade.delegation.progress_interval_seconds": (
            "nothing: a progress line comes after progress_after_seconds of quiet, at most "
            "progress_max_per_delegation times"
        ),
    }
)


class ConfigError(ValueError):
    """Raised when ``config.yaml`` is malformed."""


def section(where: str, raw: Any, cls: type) -> dict[str, Any]:
    """``raw`` (the block at ``where``) checked against the fields of ``cls``; a misplaced key names its place.

    Each value is checked against the type of the default it replaces: a whole number refuses a fraction, a duration
    refuses a negative, and only :data:`NULL_ALLOWED` settings take ``null``.

    Raises:
        ConfigError: The block is not a mapping, a key belongs elsewhere or is unknown, or a value has the wrong kind.
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{where} must be a mapping")
    for key in raw:
        moved = MOVED.get(f"{where}.{key}")
        if moved:
            raise ConfigError(f"{where}.{key} moved: set {moved}")
    allowed = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
    defaults = cls()
    return {name: _typed(f"{where}.{name}", value, getattr(defaults, name)) for name, value in raw.items()}


def _typed(where: str, value: Any, default: Any) -> Any:
    """``value`` checked against the kind of ``default``."""
    name = where.rsplit(".", 1)[-1]
    if value is None and name in NULL_ALLOWED:
        return None
    if isinstance(default, tuple):
        if not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(value)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if isinstance(default, str):
        if not isinstance(value, str) or not value:
            raise ConfigError(f"{where} must be a name")
        return value
    minimum = "non-negative" if name in ZERO_ALLOWED else "positive"
    whole = isinstance(default, int)
    number = not isinstance(value, bool) and isinstance(value, int if whole else int | float)
    if not number or value < 0 or (value == 0 and name not in ZERO_ALLOWED):
        raise ConfigError(f"{where} must be a {minimum} {'whole number' if whole else 'number'}")
    return value if whole else float(value)


@dataclass(frozen=True)
class ModelEndpoint:
    """One chat-style model endpoint, resolved from a ``services.yaml`` entry and the session body.

    ``extra_params`` are passed to the client verbatim (``temperature``, ``max_tokens``, ``extra_body``,
    ``stream_options`` ...), so settings written there by ``settings.yaml`` take effect unchanged.
    """

    model: str
    base_url: str
    api_key: str = "not-needed"
    extra_params: dict = field(default_factory=dict)
    # Frontend only: json_schema -> json_object -> none; a rejecting 400 steps down once.
    response_format: str = "json_schema"


@dataclass(frozen=True)
class ReliabilityConfig:
    """The frontend model's timeouts and retries. Retries never wrap an action that could repeat."""

    provider_retries: int = 2
    talker_attempt_timeout_seconds: float = 10.0  # one routing or phrasing attempt
    client_timeout_seconds: float = 45.0  # the frontend model's client, per request


@dataclass(frozen=True)
class GuardConfig:
    """Deterministic floors beneath the prompts."""

    repeat_guard: bool = False  # a spoken reply that nearly repeats an earlier one becomes a delegation
    # Drop acknowledgment-only turns before the talker ("okay", "mm-hmm"); shadow mode only traces them.
    turn_router: bool = False
    turn_router_shadow: bool = False
    continuers: tuple[str, ...] = DEFAULT_CONTINUERS


@dataclass(frozen=True)
class DelegationConfig:
    """What the voice does around its delegations: the app wait, how results are worded, spoken progress."""

    # How long the voice waits for the app's outputs to a delegation before it says the app-wait line; null: no line.
    app_wait_seconds: float | None = 62.0
    result_wording: str = "writer"  # how ready results are worded: writer (a model call) or as_written
    # A short progress line is spoken when work has run this long without the assistant saying anything; null: none.
    progress_after_seconds: float | None = 8.0
    progress_max_per_delegation: int = 1


@dataclass(frozen=True)
class NormalizerConfig:
    """Rewriting the caller's turn into written form before the talker sees it."""

    enabled: bool = False
    # Normalize every turn, not only turns that carry details (digits, number words, spelled letters).
    every_turn: bool = False
    # A slower normalization keeps the turn as heard.
    timeout_seconds: float = 4.0


@dataclass(frozen=True)
class MemoryConfig:
    """The talker's memory."""

    context_tokens: int = 16_000  # the talker's memory budget
    compact_at: float = 0.8  # whole turn groups are dropped once memory passes this share of the budget


@dataclass(frozen=True)
class SpeechConfig:
    """How replies are spoken."""

    tts_mode: str = "per_sentence"  # per_sentence | stitched
    start_timeout_seconds: float = 3.0  # a line whose audio has not started by then is retried or written off
    resume_after_noise: bool = True  # a line cut by a noise that was not speech resumes
    retry_silent_lines: bool = True  # a line that produced no audio is said once more


@dataclass(frozen=True)
class LiveConfig:
    """The cascade's settings: the ``frontend.cascade`` block of ``config.yaml``.

    ``provider`` names the frontend model's implementation (``models/``), ``slot`` the ``examples_registry.yaml``
    services slot whose selected ``services.yaml`` entry is that model, and ``prompt`` the ``prompts.yaml`` entry used
    as the talker's instructions when a session supplies none. ``display_label`` is not a key of the file: the root
    hands the cascade the delegation core's ``delegation.display_label`` (the server label of the core's display
    items), so the label is written once.
    """

    provider: str
    slot: str
    prompt: str = "talker"
    prompt_version: str = ""  # a name in the prompt-set table; empty: the table's default set
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    guards: GuardConfig = field(default_factory=GuardConfig)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    normalizer: NormalizerConfig = field(default_factory=NormalizerConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    # Reply audio held before playback starts, so uneven TTS chunks do not stall it (0 sends audio at once).
    tts_preroll_ms: int = 300
    display_label: str = "voiceclaw"

    @classmethod
    def parse(cls, block: Any) -> LiveConfig:
        """Validate the cascade's block (with the ``display_label`` the root put in).

        Raises:
            ConfigError: A key moved or is unknown, a value has the wrong kind, the model is not named, or
                ``prompt_version`` names no prompt set.
        """
        where = "frontend.cascade"
        if not isinstance(block, Mapping):
            raise ConfigError(f"{where} must be a mapping")
        own = {name: value for name, value in block.items() if name != "display_label"}
        for key in own:
            moved = MOVED.get(f"{where}.{key}")
            if moved:
                raise ConfigError(f"{where}.{key} moved: set {moved}")
        parts = {
            "reliability": ReliabilityConfig,
            "guards": GuardConfig,
            "delegation": DelegationConfig,
            "normalizer": NormalizerConfig,
            "memory": MemoryConfig,
            "speech": SpeechConfig,
        }
        plain = {"provider", "slot", "prompt", "prompt_version", "tts_preroll_ms"}
        unknown = set(own) - set(parts) - plain
        if unknown:
            raise ConfigError(f"{where} has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        values: dict[str, Any] = {
            name: part(**section(f"{where}.{name}", own[name], part)) for name, part in parts.items() if name in own
        }
        for name in ("provider", "slot"):
            if not isinstance(own.get(name), str) or not own[name]:
                raise ConfigError(f"{where}.{name} must name the frontend model's {name}")
            values[name] = own[name]
        if "prompt" in own:
            if not isinstance(own["prompt"], str) or not own["prompt"]:
                raise ConfigError(f"{where}.prompt must be the name of a prompts.yaml entry")
            values["prompt"] = own["prompt"]
        version = own.get("prompt_version", "")
        if version:
            try:
                pick(PROMPT_SETS, version, "prompt set")
            except ValueError as exc:
                raise ConfigError(f"{where}.prompt_version: {exc}") from None
            values["prompt_version"] = version
        if "tts_preroll_ms" in own:
            preroll = own["tts_preroll_ms"]
            if isinstance(preroll, bool) or not isinstance(preroll, int) or not 0 <= preroll <= 2000:
                raise ConfigError(f"{where}.tts_preroll_ms must be a whole number from 0 to 2000")
            values["tts_preroll_ms"] = preroll
        if "display_label" in block:
            values["display_label"] = _typed(f"{where}.display_label", block["display_label"], cls.display_label)
        return cls(**values)


@dataclass(frozen=True)
class UiBotConfig:
    """The repository UI bot's own settings (``ui_bot``)."""

    # The delegation core profile the bot builds its core from, beside config.yaml (core.ui-bot.yaml: the cafe tools,
    # and results with a display part for the task cards).
    core_config: str = "core.ui-bot.yaml"


@dataclass(frozen=True)
class ExampleConfig:
    """Everything ``config.yaml`` sets: the frontend adapter, each adapter's own block, and the UI bot's profile."""

    adapter: str  # frontend.adapter: the frontend adapter that serves each live session (a FRONTEND_ADAPTERS name)
    blocks: Mapping[str, Mapping[str, Any]]  # frontend.<name>: each adapter's own block, as written
    ui_bot: UiBotConfig = field(default_factory=UiBotConfig)

    @classmethod
    def parse(cls, data: Any) -> ExampleConfig:
        """Validate the file's top level; each adapter's block is kept as written, for its class to parse.

        Raises:
            ConfigError: The file is not a mapping, a key moved or is unknown, ``frontend.adapter`` is missing or names
                no block, a block is not a mapping, or a block sets what the root hands it.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("config.yaml must be a mapping")
        for key in data:
            moved = MOVED.get(str(key))
            if moved:
                raise ConfigError(f"{key} moved: set {moved}")
        unknown = set(data) - {"frontend", "ui_bot"}
        if unknown:
            raise ConfigError(f"config.yaml has unknown keys: {', '.join(sorted(map(str, unknown)))}")
        frontend = data.get("frontend")
        if not isinstance(frontend, Mapping):
            raise ConfigError("config.yaml needs frontend: the frontend adapter (adapter) and its block")
        for key in frontend:
            moved = MOVED.get(f"frontend.{key}")
            if moved:
                raise ConfigError(f"frontend.{key} moved: set {moved}")
        adapter = frontend.get("adapter")
        if not isinstance(adapter, str) or not adapter:
            raise ConfigError("frontend.adapter must name the frontend adapter that serves live sessions")
        blocks = {}
        for name, block in frontend.items():
            if name == "adapter":
                continue
            if not isinstance(block, Mapping):
                raise ConfigError(f"frontend.{name} must be a mapping: a frontend adapter's block")
            if "display_label" in block:
                raise ConfigError(f"frontend.{name}.display_label moved: set {MOVED['frontend.cascade.display_label']}")
            blocks[str(name)] = MappingProxyType(dict(block))
        if adapter not in blocks:
            raise ConfigError(f"frontend.adapter names {adapter!r}, but config.yaml has no frontend.{adapter} block")
        ui_bot = UiBotConfig(**section("ui_bot", data.get("ui_bot"), UiBotConfig))
        return cls(adapter=adapter, blocks=MappingProxyType(blocks), ui_bot=ui_bot)
