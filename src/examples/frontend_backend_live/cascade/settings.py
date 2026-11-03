# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's own settings: its share of the ``frontend.cascade`` block of ``config.yaml``.

::

    delegation: {app_wait_seconds: 50, result_wording: writer, progress_after_seconds: 8,
                 progress_max_per_delegation: 3}
    memory: {context_tokens: 16000, compact_at: 0.8}
    speech: {start_timeout_seconds: 3, resume_after_noise: true, retry_silent_lines: true,
             heard_words: as_recorded, estimate_chars_per_second: 15}

A block names only what differs from these defaults. Other keys, such as ``speech.tts_mode`` and ``reliability``,
belong to the frontend adapter, and ``display_label`` comes from the core's ``delegation.display_label``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from examples.frontend_backend_live.cascade.utterances import CUT_MEMORIES
from examples.frontend_backend_live.cascade.wording import RESULT_WORDINGS
from examples.frontend_backend_live.config_manager.schema import LiveConfig
from examples.frontend_backend_live.tables import pick


def _positive(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and value > 0


# How each kind of value is checked, and what the error says it must be.
CHECKS: dict[str, tuple[Callable[[Any], bool], str]] = {
    "clock": (lambda value: value is None or _positive(value), "a positive number of seconds, or null (off)"),
    "seconds": (_positive, "a positive number of seconds"),
    "count": (lambda value: _positive(value) and isinstance(value, int), "a positive whole number"),
    "share": (lambda value: _positive(value) and value <= 1, "a number above 0 and at most 1"),
    "rate": (_positive, "a positive number of characters per second"),
    "flag": (lambda value: isinstance(value, bool), "true or false"),
    "name": (lambda value: isinstance(value, str) and bool(value), "a name"),
}
# The keys of each sub-block, with the kind of value each takes.
KEYS = {
    "delegation": {
        "app_wait_seconds": "clock",
        "result_wording": "name",
        "progress_after_seconds": "clock",
        "progress_max_per_delegation": "count",
    },
    "memory": {"context_tokens": "count", "compact_at": "share"},
    "speech": {
        "start_timeout_seconds": "seconds",
        "resume_after_noise": "flag",
        "retry_silent_lines": "flag",
        "heard_words": "name",
        "estimate_chars_per_second": "rate",
    },
}


@dataclass(frozen=True)
class CascadeSettings:
    """What the cascade reads from the settings, with their defaults."""

    app_wait_seconds: float | None = 50.0  # the app's outputs are awaited this long before the app-wait line; None: off
    result_wording: str = "writer"  # how ready results become one line: a name of RESULT_WORDINGS
    progress_after_seconds: float | None = 8.0  # quiet after which the writer may word a progress line; None: off
    progress_max_per_delegation: int = 3  # progress lines offered for one delegation (joined work shares the count)
    context_tokens: int = 16_000  # the talker's memory budget
    compact_at: float = 0.8  # the share of the budget at which the oldest unpinned turns leave the memory
    start_timeout_seconds: float = 3.0  # a line that has not started playing this long after it heads the queue
    resume_after_noise: bool = True  # an empty turn, or one the talker waits on, right after a cut says the rest once
    retry_silent_lines: bool = True  # a line that produced no audio is said once more
    heard_words: str = "as_recorded"  # what the talker's memory keeps of a cut line: a name of CUT_MEMORIES
    estimate_chars_per_second: float = 15.0  # estimated: the speaking rate until a sentence of the call has played
    display_label: str = "delegation_core"  # the server label of the display item (the core's delegation.display_label)

    @classmethod
    def parse(cls, block: Mapping[str, Any], *, display_label: str | None = None) -> CascadeSettings:
        """Validate the cascade's sub-blocks (``delegation``, ``memory``, ``speech``); unknown keys are refused.

        Raises:
            ValueError: An unknown block or key, a value of the wrong kind, or an unknown result wording or cut
                memory.
        """
        if not isinstance(block, Mapping):
            raise ValueError("the cascade's settings must be a mapping")
        unknown = set(block) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown cascade settings: {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        for name, keys in KEYS.items():
            section = block.get(name) or {}
            if not isinstance(section, Mapping):
                raise ValueError(f"{name} must be a mapping")
            unknown = set(section) - set(keys)
            if unknown:
                raise ValueError(f"{name} has unknown keys: {', '.join(sorted(unknown))}")
            for key, value in section.items():
                check, must_be = CHECKS[keys[key]]
                if not check(value):
                    raise ValueError(f"{name}.{key} must be {must_be}")
                values[key] = float(value) if isinstance(value, float) else value
        pick(RESULT_WORDINGS, values.get("result_wording", cls.result_wording), "result wording")
        pick(CUT_MEMORIES, values.get("heard_words", cls.heard_words), "speech.heard_words")
        if display_label is not None:
            values["display_label"] = display_label
        return cls(**values)

    @classmethod
    def from_live_config(cls, config: LiveConfig) -> CascadeSettings:
        """The cascade's share of the ``frontend.cascade`` block (``LiveConfig``), checked by :meth:`parse`.

        Raises:
            ValueError: A value the cascade does not accept.
        """
        delegation, memory, speech = config.delegation, config.memory, config.speech
        return cls.parse(
            {
                "delegation": {
                    "app_wait_seconds": delegation.app_wait_seconds,
                    "result_wording": delegation.result_wording,
                    "progress_after_seconds": delegation.progress_after_seconds,
                    "progress_max_per_delegation": delegation.progress_max_per_delegation,
                },
                "memory": {"context_tokens": memory.context_tokens, "compact_at": memory.compact_at},
                "speech": {
                    "start_timeout_seconds": speech.start_timeout_seconds,
                    "resume_after_noise": speech.resume_after_noise,
                    "retry_silent_lines": speech.retry_silent_lines,
                    "heard_words": speech.heard_words,
                    "estimate_chars_per_second": speech.estimate_chars_per_second,
                },
            },
            display_label=config.display_label,
        )
