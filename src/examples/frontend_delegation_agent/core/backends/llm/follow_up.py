# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy follow-up verification (``llm.verify_policy_follow_up_steps``): the rules and the per-session tracker.

A policy often says what must follow a change: "After rebooking, always move the seat, the bags and
the meal to the new flight." This module finds such rules in the client's own policy text, matches
each one to the session's write tools once at configure, and tracks, per run, which rules a
successful write opened. ``LLMAgent`` then sends one check before it answers (``harness_follow_up_check``).

* **Extraction** (:func:`extract_candidates`): a sentence (a newline, ``.``, ``!`` or ``?`` ends one) with
  the word ``after`` and an obligation marker (:data:`MARKERS`). The trigger is the words after
  ``after`` up to a comma, a marker or a subject pronoun (at most :data:`MAX_TRIGGER_WORDS`); the
  rule text is the line or bullet that holds the sentence plus the list items right below it.
* **Generic triggers** ("After completing any change") name no object and give no rule: a trigger
  left with no stem once quantifiers, change nouns, generic verbs and stop words are removed.
* **Matching** (:func:`matches`): discriminative. One shared stem counts only when exactly one write
  tool of the session has it, otherwise two are needed; a tool whose verb appears only in the rule's
  follow-up part is the follow-up step itself and never opens the rule.
* **Write tools** (:func:`write_tools`): not a read by name or description (the voice side's
  heuristic, on the namespace-stripped name) and not declared ``readOnlyHint``.

Operator rules (``llm.follow_up_rules``) name their tools and skip the text heuristics. Nothing here
knows a tool, a domain or a policy by name. Stdlib only, pure apart from :class:`FollowUpTracker`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from examples.frontend_delegation_agent.core.backends.llm.tools import canonical_arguments, is_failure
from examples.frontend_delegation_agent.core.normalization.schema_rules import is_read_tool, tool_function

#: Words that make an "after ..." sentence an obligation (multi-word markers as word tuples).
MARKERS: tuple[tuple[str, ...], ...] = (
    ("always",),
    ("must",),
    ("ensure",),
    ("make", "sure"),
    ("be", "sure", "to"),
    ("remember", "to"),
)
#: Subject words that end a trigger ("After a case is opened you must ..." -> "a case is opened").
SUBJECTS: tuple[tuple[str, ...], ...] = (("you",), ("we",), ("the", "agent"))
MAX_TRIGGER_WORDS = 8
MAX_TEXT_CHARS = 600
MAX_RULES = 5
#: A check fires at most this often for one opened rule (the first check and one re-fire).
MAX_FIRES = 2
MIN_STEM = 4
STOP_WORDS = frozenset(
    {"the", "a", "an", "any", "each", "every", "all", "is", "are", "has", "have", "been", "user", "with", "from"}
    | {"that", "this"}
)
CHANGE_NOUNS = frozenset(
    {"change", "changes", "update", "updates", "modification", "modifications", "action", "actions"}
)
#: Verbs that say nothing about what changed; never part of a trigger's matching stems.
GENERIC_VERBS = frozenset(
    {
        *("complete", "completes", "completed", "completing"),
        *("make", "makes", "made", "making"),
        *("do", "does", "did", "done", "doing"),
        *("perform", "performs", "performed", "performing"),
        *("process", "processes", "processed", "processing"),
    }
)

_SUFFIXES = ("ing", "ed", "es", "s", "e")
_KEEP_DOUBLE = "aeiouslz"
_WORD = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#")
_AFTER = re.compile(r"\bafter\b", re.IGNORECASE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NAMESPACE = re.compile(r"^.*(?:\.|/|__)")


@dataclass(frozen=True, slots=True)
class FollowUpRule:
    """One rule: its trigger words, its text and the session's write tools that open it."""

    trigger: str
    text: str
    tools: tuple[str, ...]
    source: str = "policy"  # "policy" (extracted from the instructions) or "declared" (llm.follow_up_rules)

    def snapshot(self) -> dict[str, Any]:
        """For the ``follow_up_obligations`` event."""
        return {"trigger": self.trigger, "tools": list(self.tools), "source": self.source, "text": self.text[:200]}


@dataclass(frozen=True, slots=True)
class _Candidate:
    """An extracted rule before matching: trigger words, matching stems and the follow-up words."""

    trigger: str
    stems: tuple[str, ...]
    trigger_words: tuple[str, ...]
    follow_up_words: tuple[str, ...]
    text: str


# -- words ----------------------------------------------------------------------------------------


def stem(word: str) -> str:
    """The matching stem of ``word``, or ``""`` (too short or a stop word).

    The first of ``ing``, ``ed``, ``es``, ``s``, ``e`` is stripped when at least 4 characters remain;
    after ``ing`` or ``ed`` a doubled final consonant is undoubled, except ``l``, ``s`` and ``z``
    (``shipped`` -> ``ship``, ``called`` -> ``call``).
    """
    lowered = word.lower()
    for suffix in _SUFFIXES:
        if lowered.endswith(suffix) and len(lowered) - len(suffix) >= MIN_STEM:
            lowered = lowered[: -len(suffix)]
            if suffix in ("ing", "ed") and lowered[-1] == lowered[-2] and lowered[-1] not in _KEEP_DOUBLE:
                lowered = lowered[:-1]
            break
    return lowered if len(lowered) >= MIN_STEM and lowered not in STOP_WORDS else ""


def stems_equal(a: str, b: str) -> bool:
    """Equal when one is a prefix of the other and the shorter has at least 4 characters."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= MIN_STEM and long_.startswith(short)


def tool_tokens(name: str) -> tuple[str, ...]:
    """Lower-case words of a tool name, without a namespace (``crm.updateCase`` -> ``update``, ``case``)."""
    bare = _NAMESPACE.sub("", name)
    return tuple(t.lower() for t in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", bare)) if t)


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def _starts_with(words: Sequence[str], index: int, phrases: Iterable[tuple[str, ...]]) -> bool:
    return any(tuple(words[index : index + len(p)]) == p for p in phrases)


def _has_marker(words: Sequence[str]) -> bool:
    return any(_starts_with(words, i, MARKERS) for i in range(len(words)))


# -- extraction -----------------------------------------------------------------------------------


def _blocks(instructions: str) -> list[tuple[int, str]]:
    """``(line index, block text)``: each line plus the list items right below it (to a blank line or heading)."""
    lines = instructions.splitlines()
    out = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        parts = [line.strip()]
        for following in lines[index + 1 :]:
            if not following.strip() or _HEADING.match(following) or not _LIST_ITEM.match(following):
                break
            parts.append(following.strip())
        out.append((index, "\n".join(parts)))
    return out


def _candidate(sentence: str, rest: str, block: str) -> _Candidate | None:
    match = _AFTER.search(sentence)
    if match is None:
        return None
    if not _has_marker(_words(sentence)):
        return None
    after = sentence[match.end() :]
    head, _, _ = after.partition(",")
    words = _words(head)
    trigger: list[str] = []
    for index, word in enumerate(words):
        if len(trigger) >= MAX_TRIGGER_WORDS:
            break
        if _starts_with(words, index, MARKERS) or _starts_with(words, index, SUBJECTS):
            break
        trigger.append(word)
    if not any(stem(w) for w in trigger if w not in GENERIC_VERBS and w not in CHANGE_NOUNS):
        return None  # a generic trigger ("after completing any change") names nothing that changed
    stems: list[str] = []
    for word in trigger:
        value = "" if word in GENERIC_VERBS else stem(word)
        if value and value not in stems:
            stems.append(value)
    text = block if len(block) <= MAX_TEXT_CHARS else block[: MAX_TEXT_CHARS - 1].rstrip() + "…"
    return _Candidate(
        trigger=" ".join(trigger),
        stems=tuple(stems),
        trigger_words=tuple(trigger),
        follow_up_words=tuple(_words(after)[len(trigger) :] + _words(rest)),
        text=text,
    )


def extract_candidates(instructions: str) -> list[_Candidate]:
    """Every "after X, ... must/always ..." sentence of ``instructions``, in order."""
    out: list[_Candidate] = []
    for _, block in _blocks(instructions):
        first_line = block.split("\n", 1)[0]
        sentences = [s for s in _SENTENCE_END.split(first_line) if s.strip()]
        for index, sentence in enumerate(sentences):
            rest = " ".join(sentences[index + 1 :] + block.split("\n")[1:])
            candidate = _candidate(sentence, rest, block)
            if candidate is not None:
                out.append(candidate)
    return out


# -- tools ----------------------------------------------------------------------------------------


def declares_read_only_hint(tool: Mapping[str, Any]) -> bool:
    """``annotations.readOnlyHint: true`` on the tool or its ``function`` part (idempotence not required)."""
    for part in (tool, tool_function(tool)):
        annotations = part.get("annotations")
        if isinstance(annotations, Mapping) and annotations.get("readOnlyHint") is True:
            return True
    return False


def write_tools(tools: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The session's write tools: not a read by name or description, and not declared read-only."""
    out = []
    for tool in tools:
        function = tool_function(tool)
        name = str(function.get("name") or "")
        if not name:
            continue
        normalized = "_".join(tool_tokens(name))
        if is_read_tool(normalized, str(function.get("description") or "")) or declares_read_only_hint(tool):
            continue
        out.append(name)
    return tuple(out)


def _tool_stems(name: str) -> tuple[str, ...]:
    return tuple(s for s in (stem(t) for t in tool_tokens(name)) if s)


def _is_follow_up_tool(candidate: _Candidate, name: str) -> bool:
    """The tool's verb appears in the rule's follow-up part but not in its trigger (raw words, as a prefix)."""
    tokens = tool_tokens(name)
    verb = tokens[0] if tokens else ""
    if len(verb) < 3:
        return False
    in_follow_up = any(word.startswith(verb) for word in candidate.follow_up_words)
    in_trigger = any(word.startswith(verb) for word in candidate.trigger_words)
    return in_follow_up and not in_trigger


def matches(candidate: _Candidate, name: str, writes: Sequence[str]) -> bool:
    """Whether the write tool ``name`` opens ``candidate`` (discriminative stem match, not the follow-up tool)."""
    if _is_follow_up_tool(candidate, name):
        return False
    tool_stems = _tool_stems(name)
    shared = [s for s in candidate.stems if any(stems_equal(s, t) for t in tool_stems)]
    if len(shared) >= 2:
        return True
    if len(shared) == 1:
        owners = [w for w in writes if any(stems_equal(shared[0], t) for t in _tool_stems(w))]
        return owners == [name]
    return False


def build_rules(
    instructions: str,
    tools: Sequence[Mapping[str, Any]],
    *,
    declared: Sequence[Mapping[str, Any]] = (),
    ignore_tools: Iterable[str] = (),
    domain: str = "",
) -> tuple[FollowUpRule, ...]:
    """The session's rules: extracted ones that match a write tool, then the declared ones (at most 5 in all)."""
    writes = write_tools(tools)
    names = {str(tool_function(t).get("name") or "") for t in tools}
    ignored = set(ignore_tools)
    rules: list[FollowUpRule] = []
    seen: dict[str, int] = {}
    for candidate in extract_candidates(instructions):
        opened = tuple(w for w in writes if w not in ignored and matches(candidate, w, writes))
        if not opened:
            continue
        if candidate.text in seen:
            index = seen[candidate.text]
            merged = tuple(dict.fromkeys(rules[index].tools + opened))
            rules[index] = FollowUpRule(rules[index].trigger, rules[index].text, merged)
            continue
        seen[candidate.text] = len(rules)
        rules.append(FollowUpRule(candidate.trigger, candidate.text, opened))
    for entry in declared:
        scope = str(entry.get("domain") or "")
        if scope and scope != domain:
            continue
        opened = tuple(str(t) for t in entry.get("tools") or () if str(t) in names)
        if opened:
            rules.append(FollowUpRule(", ".join(opened), str(entry.get("text") or "").strip(), opened, "declared"))
    return tuple(rules[:MAX_RULES])


def succeeded(result: str) -> bool:
    """A tool result that reports success: not :func:`is_failure`, and no ``success``/``ok`` set to false."""
    if is_failure(result):
        return False
    try:
        data = json.loads(result)
    except (TypeError, ValueError):
        return True
    return not (isinstance(data, dict) and (data.get("success") is False or data.get("ok") is False))


# -- tracker --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Open:
    writes: list[str] = field(default_factory=list)
    mode: str = "fresh"  # fresh (opened this run) | carry (asked the user; needs a write) | forced (run ended early)
    fires: int = 0
    checked: bool = False


@dataclass(frozen=True, slots=True)
class Check:
    """What one check names: the changes made and the rule texts that apply."""

    changes: tuple[str, ...]
    rules: tuple[FollowUpRule, ...]


class FollowUpTracker:
    """Which rules are open across the runs of one session, and when a check is due.

    Per run: :meth:`start_run`, :meth:`record` after every tool call, :meth:`due` / :meth:`fire`
    when the run is about to answer, then :meth:`finish`. A rule opened by a successful write is
    checked once before the answer. It stays open after the check only when the answer asks the
    user something and no write followed the check; it is then checked once more at the end of the
    next run that makes a write. A run that ends early (interrupt, failure) keeps its open rules for
    the next run, which checks them at its end. A rule is checked at most :data:`MAX_FIRES` times.
    """

    def __init__(self, rules: Sequence[FollowUpRule], writes: Iterable[str]) -> None:
        """No rule is open yet; ``writes`` are the session's write tools (a successful one counts as a write)."""
        self.rules = tuple(rules)
        self._writes = frozenset(writes)
        self._open: dict[int, _Open] = {}
        self._fired = False
        self._wrote = False
        self._wrote_after_check = False

    @property
    def fired(self) -> bool:
        """Whether a check was sent in the current run."""
        return self._fired

    def start_run(self) -> None:
        """A new run: no check and no write yet."""
        self._fired = self._wrote = self._wrote_after_check = False
        for state in self._open.values():
            state.checked = False

    def record(self, name: str, args: Mapping[str, Any], result: str) -> None:
        """After a tool call: a successful write tool is a write; a matching write opens its rules."""
        if name not in self._writes or not succeeded(result):
            return
        self._wrote = True
        if self._fired:
            self._wrote_after_check = True
        opening = [i for i, rule in enumerate(self.rules) if name in rule.tools]
        if not opening:
            return
        change = f"{name}({canonical_arguments(dict(args))})"
        for index in opening:
            state = self._open.setdefault(index, _Open())
            if change not in state.writes:
                state.writes.append(change)
            if not self._fired:
                state.mode = "fresh"

    def due(self) -> bool:
        """Whether a check should be sent before this run answers."""
        return not self._fired and bool(self._due_indexes())

    def fire(self) -> Check:
        """Mark the check as sent for this run; returns what it names."""
        indexes = self._due_indexes()
        self._fired = True
        changes: list[str] = []
        for index in indexes:
            state = self._open[index]
            state.fires += 1
            state.checked = True
            changes.extend(c for c in state.writes if c not in changes)
        return Check(tuple(changes), tuple(self.rules[i] for i in indexes))

    def finish(self, answer: str | None, *, completed: bool) -> None:
        """The run ended: close or keep each open rule (see the class docstring)."""
        if not completed:
            for state in self._open.values():
                state.mode = "forced"
            self._drop_exhausted()
            return
        asks = bool(answer) and answer.rstrip().endswith("?")
        for index in list(self._open):
            state = self._open[index]
            keep = state.checked and asks and not self._wrote_after_check
            if keep:
                state.mode = "carry"
            else:
                del self._open[index]
        self._drop_exhausted()

    def open_count(self) -> int:
        """Rules open now (for tests and logs)."""
        return len(self._open)

    def _due_indexes(self) -> list[int]:
        return [
            index
            for index, state in sorted(self._open.items())
            if state.fires < MAX_FIRES and (state.mode != "carry" or self._wrote)
        ]

    def _drop_exhausted(self) -> None:
        for index in [i for i, s in self._open.items() if s.fires >= MAX_FIRES]:
            del self._open[index]
