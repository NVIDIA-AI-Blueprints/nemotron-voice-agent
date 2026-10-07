# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The tool-argument hook: canonicalize, validate and screen backend tool calls.

Runs on calls the runner is about to hand to the client (``tools.source:
client``). Per configured rule it rewrites one string argument to its canonical
form (spoken form, stripped characters, collapsed separators, case) and checks
it against a full-match pattern. :meth:`ArgumentNormalizer.screen` then decides,
per call, whether it goes out or is answered locally:

* ``invalid``: the canonical value does not match the rule's pattern;
* ``already_failed``: the same tool with the same canonical arguments already
  produced a *permanent* failure in this session (the retry guard).

With ``escalate_invalid_message_key`` set, the second and later ``invalid`` answers of
the same tool in a session use that (firmer) message; the caller owns the counts.

Everything here is pure; the per-session failed set is owned by the runner.
See ``misc/prototypes/voice/normalization-plan.md`` section 4.3.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from examples.frontend_backend_verdict.text.messages import ToolCall, canonical_json
from examples.frontend_backend_verdict.voice.normalization.rules import apply_case, spelled_out
from examples.frontend_backend_verdict.voice.normalization.transcript import (
    SpelledRunSettings,
    TranscriptNormalizer,
    TranscriptSettings,
)

ON_INVALID = ("answer_locally", "send")
GUARD_SCOPES = ("rules", "all")
REASON_INVALID = "invalid"
REASON_ALREADY_FAILED = "already_failed"

#: A retry-guard key: the tool name and its canonical JSON arguments.
FailureKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class ArgumentRule:
    """How one string argument of one tool is canonicalized and checked."""

    tool: str
    argument: str
    label: str = ""
    format_hint: str = ""
    spoken_form: bool = False
    strip: str = ""
    collapse_separators: str = ""
    case: str = "keep"
    pattern: str = ""
    on_invalid: str = "answer_locally"
    #: Literal symbols (never letters or digits) put back in front of a value that lacks them, e.g. ``#``.
    restore_prefix: str = ""
    #: A value full-matching this is already well formed and never rewritten (idempotence).
    keep_pattern: str = ""
    #: ``config`` (YAML) or ``schema`` (derived from the session's tools; :mod:`.schema_rules`).
    source: str = "config"

    @property
    def display_label(self) -> str:
        """The label used in local messages."""
        return self.label or self.argument


@dataclass(frozen=True, slots=True)
class RetryGuardSettings:
    """``normalization.tool_arguments.retry_guard``."""

    enabled: bool = False
    scope: str = "rules"
    permanent_failure_pattern: str = ""
    message_key: str = "tool_call_already_failed"


@dataclass(frozen=True, slots=True)
class ResultHintSettings:
    """``normalization.tool_arguments.result_hints``: a note appended to a failed lookup's result.

    The runner appends ``message_key`` to the first output of ``tools`` that matches
    ``failure_pattern`` and ``escalate_message_key`` ("" = the same message) to later ones,
    at most ``max_hints`` times until a lookup of ``tools`` succeeds. With
    ``count_local_invalid`` a local ``invalid`` answer for ``tools`` counts as a miss (it
    never gets a note itself). Independent of ``rules`` and the retry guard.
    """

    enabled: bool = False
    tools: tuple[str, ...] = ()
    #: ``tools: auto``: watch every read tool of the session (selection is then by ``failure_pattern``).
    auto_tools: bool = False
    failure_pattern: str = ""
    count_local_invalid: bool = True
    max_hints: int = 3
    message_key: str = "lookup_not_found_hint"
    escalate_message_key: str = ""


@dataclass(frozen=True, slots=True)
class ToolArgumentSettings:
    """``normalization.tool_arguments``."""

    enabled: bool = False
    rules: tuple[ArgumentRule, ...] = ()
    #: ``rules: auto``: rules derived per session from the tool schemas (:mod:`.schema_rules`).
    schema_rules: bool = False
    invalid_message_key: str = "tool_argument_invalid"
    #: Message for the 2nd+ invalid answer of the same tool in a session ("" = always ``invalid_message_key``).
    escalate_invalid_message_key: str = ""
    max_local_rounds: int = 3
    retry_guard: RetryGuardSettings = field(default_factory=RetryGuardSettings)
    result_hints: ResultHintSettings = field(default_factory=ResultHintSettings)


@dataclass(frozen=True, slots=True)
class Rewrite:
    """One argument changed by canonicalization."""

    call_id: str
    tool: str
    argument: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class LocalAnswer:
    """A call the runner answers itself instead of sending it."""

    call_id: str
    tool: str
    argument: str
    value: str
    reason: str
    message: str
    message_key: str = ""


@dataclass(frozen=True, slots=True)
class Screening:
    """The outcome of screening one batch of calls."""

    calls: tuple[ToolCall, ...]
    rewrites: tuple[Rewrite, ...] = ()
    local: tuple[LocalAnswer, ...] = ()
    keys: Mapping[str, FailureKey] = field(default_factory=dict)

    @property
    def sent(self) -> tuple[ToolCall, ...]:
        """The (canonical) calls that go to the client."""
        local_ids = {answer.call_id for answer in self.local}
        return tuple(call for call in self.calls if call.id not in local_ids)


def render_message(template: str, **values: str) -> str:
    """Literal ``{name}`` substitution, like the text prototype's prompt rendering."""
    text = template.strip()
    for name, value in values.items():
        text = text.replace("{" + name + "}", value)
    return text


class ArgumentNormalizer:
    """Canonicalizes and screens tool calls for one session's settings."""

    def __init__(
        self,
        settings: ToolArgumentSettings,
        *,
        transcript: TranscriptSettings,
        invalid_template: str,
        already_failed_template: str,
        escalate_invalid_template: str = "",
    ) -> None:
        """``transcript`` supplies the separator words and ruleset for ``spoken_form`` rules."""
        self._settings = settings
        self._rules: dict[str, list[ArgumentRule]] = {}
        for rule in settings.rules:
            self._rules.setdefault(rule.tool, []).append(rule)
        # Same words as the transcript hook; the rule decides the case.
        self._spoken = TranscriptNormalizer(
            replace(transcript, enabled=True, case="keep", spelled_runs=SpelledRunSettings())
        )
        self._separators = dict(transcript.separator_words)
        self._invalid_template = invalid_template
        self._already_failed_template = already_failed_template
        self._escalate_template = escalate_invalid_template if settings.escalate_invalid_message_key else ""
        guard = settings.retry_guard
        self._failure = re.compile(guard.permanent_failure_pattern) if guard.enabled else None

    @property
    def settings(self) -> ToolArgumentSettings:
        """The settings this normalizer was built from."""
        return self._settings

    # -- values ----------------------------------------------------------------

    def canonical(self, rule: ArgumentRule, value: str) -> str:
        """The canonical form of ``value`` under ``rule`` (unchanged when it full-matches ``keep_pattern``)."""
        if rule.keep_pattern and re.fullmatch(rule.keep_pattern, value):
            return value
        text = value
        if rule.spoken_form:
            text = self._spoken.normalize(text).text
        for char in rule.strip:
            text = text.replace(char, "")
        if separator := rule.collapse_separators:
            text = re.sub(f"{re.escape(separator)}{{2,}}", separator, text).strip(separator)
        text = apply_case(text, rule.case)
        if rule.restore_prefix and text and not text.startswith(rule.restore_prefix):
            text = rule.restore_prefix + text.lstrip(rule.restore_prefix)
        return text

    @staticmethod
    def valid(rule: ArgumentRule, value: str) -> bool:
        """Whether ``value`` full-matches the rule's pattern (always true without one)."""
        return not rule.pattern or re.fullmatch(rule.pattern, value) is not None

    def is_permanent_failure(self, output: str) -> bool:
        """Whether a client output is a permanent failure (never a transient error)."""
        return self._failure is not None and self._failure.search(output) is not None

    # -- calls ------------------------------------------------------------------

    def _guarded(self, tool: str) -> bool:
        guard = self._settings.retry_guard
        return guard.enabled and (guard.scope == "all" or tool in self._rules)

    def screen(
        self,
        calls: Sequence[ToolCall],
        failed: frozenset[FailureKey],
        invalid_counts: Mapping[str, int] | None = None,
    ) -> Screening:
        """Canonicalize every call and decide which are answered locally.

        ``invalid_counts`` (tool -> earlier ``invalid`` answers this session) selects the escalated message.
        """
        counts = dict(invalid_counts or {})
        out: list[ToolCall] = []
        rewrites: list[Rewrite] = []
        local: list[LocalAnswer] = []
        keys: dict[str, FailureKey] = {}
        for call in calls:
            try:
                arguments = json.loads(call.arguments_json or "{}")
            except json.JSONDecodeError:
                arguments = None
            if not isinstance(arguments, dict):
                out.append(call)  # malformed: passed through unchanged
                continue
            invalid: tuple[ArgumentRule, str] | None = None
            for rule in self._rules.get(call.name, ()):
                before = arguments.get(rule.argument)
                if not isinstance(before, str):
                    continue
                after = self.canonical(rule, before)
                if after != before:
                    arguments[rule.argument] = after
                    rewrites.append(Rewrite(call.id, call.name, rule.argument, before, after))
                if invalid is None and rule.on_invalid == "answer_locally" and not self.valid(rule, after):
                    invalid = (rule, after)
            canonical_call = call
            if any(rewrite.call_id == call.id for rewrite in rewrites):
                canonical_call = ToolCall(id=call.id, name=call.name, arguments_json=canonical_json(arguments))
            out.append(canonical_call)
            if invalid is not None:
                rule, value = invalid
                escalate = bool(self._escalate_template) and counts.get(call.name, 0) >= 1
                counts[call.name] = counts.get(call.name, 0) + 1
                local.append(self._answer(call, rule, value, REASON_INVALID, escalate=escalate))
                continue
            if self._guarded(call.name):
                key = (call.name, canonical_json(arguments))
                keys[call.id] = key
                if key in failed:
                    local.append(self._already_failed(call, arguments))
        return Screening(calls=tuple(out), rewrites=tuple(rewrites), local=tuple(local), keys=keys)

    def _already_failed(self, call: ToolCall, arguments: Mapping[str, object]) -> LocalAnswer:
        for rule in self._rules.get(call.name, ()):
            value = arguments.get(rule.argument)
            if isinstance(value, str):
                return self._answer(call, rule, value, REASON_ALREADY_FAILED)
        rule = ArgumentRule(tool=call.name, argument="arguments")
        return self._answer(call, rule, canonical_json(dict(arguments)), REASON_ALREADY_FAILED)

    def _answer(
        self, call: ToolCall, rule: ArgumentRule, value: str, reason: str, *, escalate: bool = False
    ) -> LocalAnswer:
        settings = self._settings
        if reason == REASON_INVALID and escalate:
            template, key = self._escalate_template, settings.escalate_invalid_message_key
        elif reason == REASON_INVALID:
            template, key = self._invalid_template, settings.invalid_message_key
        else:
            template, key = self._already_failed_template, settings.retry_guard.message_key
        message = render_message(
            template,
            tool=call.name,
            label=rule.display_label,
            value=value,
            format_hint=rule.format_hint or "see the tool description",
            spelled=spelled_out(value, self._separators),
        )
        return LocalAnswer(call.id, call.name, rule.argument, value, reason, message, key)
