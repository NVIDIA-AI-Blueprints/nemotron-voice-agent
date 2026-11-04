# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example's ``prompts.yaml``: the talker prompt the UI lists, and every other text a model reads.

The file's top-level entries are the talker prompts a UI bot session selects with ``prompt_key``, through
``utils.resolve_prompt`` as in every example. ``agent_prompts`` holds the other texts, which the UI shows read-only and
no session selects, in five groups:

* ``Talker``: the talker prompt of a live session that sets no instructions or blank ones, by the name that
  ``frontend.cascade.prompt`` gives;
* ``Backend``: the backend prompt of a request that sets no instructions, by the name that the core's ``backend.prompt``
  gives;
* ``Channel``: the context that the UI bot adds after its talker prompt and its backend prompt (``ui_bot.channel``);
* ``Cascade``: the prompt set, the text of the cascade's own model calls and of each delegation's input
  (:class:`PromptSet`), each entry named by its field (``work_texts.item``);
* ``Lines``: the lines that the voice says word for word (:class:`FixedLines`).

A prompt file the server was started with (``PROMPT_FILE_PATH``) replaces this whole file, as for every example, so it
holds these entries too. Each session reads the file once when it starts, so an edit takes effect at the next session.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from examples.frontend_backend_live.cascade.lines import FixedLines
from utils import load_yaml_file, resolve_prompt_catalog_path


@dataclass(frozen=True)
class WriterNotes:
    """The notes added to an update for the writer, each only when it applies."""

    told: str  # what the caller was already told about the update's delegations: "{told}"
    newer: str  # newer requests are still open and may change the result; "{requests}" if used: quoted, "; " between


@dataclass(frozen=True)
class CallHints:
    """The description of ``speech`` in each call's JSON schema, by the call's role."""

    talker: str  # the routing decision
    answer: str  # the writer's call: every update it words
    normalizer: str  # the newest caller turn, rewritten
    progress: str  # the writer's progress line: one short sentence, or empty


@dataclass(frozen=True)
class NormalizerPrompt:
    """What the normalizer reads: its instructions, and the label in front of the turn it rewrites."""

    instructions: str
    newest_turn: str


@dataclass(frozen=True)
class WorkTexts:
    """The words of one open delegation's line in the work note."""

    item: str  # the line: "{number}", "{state}", the request as "{request}", and "{update}" (empty without one)
    running: str  # the state of work that runs: "{seconds}" since it started
    ended: str  # added to that state for the tool calls of the work's latest response that ended: "{steps}"
    ended_call: str  # one of those calls: its "{name}" as words and the official "{status}" it ended with
    steps: str  # added after them for the tool calls running now: "{steps}", their names as words
    waiting: str  # the state of work not started yet
    update: str  # the work's newest progress line: "{progress}"


@dataclass(frozen=True)
class TranscriptText:
    """The words of a delegation's input: the transcript since the last request, one labeled line per entry."""

    header: str  # the line before the lines since the last request
    omitted: str  # the line in place of the oldest lines, which the transcript's size cap leaves out
    caller: str  # the label of a caller turn
    agent: str  # the label of a line the voice said
    context: str  # the label of the application's silent context
    instruction: str  # the label of an instruction append
    earlier: str  # the first line, before the instructions an accepted delegation already carried, when there are some
    unheard: str  # the last line while an answer the backend gave waits to be heard: the caller's words do not reply


@dataclass(frozen=True, kw_only=True)
class PromptSet:
    """The text a prompt set contributes to each model call of the cascade, and to each delegation's input.

    The cascade reads the work note's texts, the labels and marks of the talker's memory, and the transcript's words
    from here; the talker, the writer and the normalizer read the rest.
    """

    routing: str  # appended to the client's instructions for the routing decision
    commentary: str  # appended for commentary phrasing
    answer: str  # appended after the commentary text in every writer call but a progress line's
    progress: str  # the writer's rules for a progress line, read in place of the commentary and answer texts
    work_in_progress: str  # the open delegations, one numbered line each: "{items}"
    no_work_in_progress: str  # what the talker reads, with a server delegation, when none is open
    answer_ready_label: str  # an open delegation whose answer arrived and is being said
    typed_request_label: str  # work the frontend did not delegate: the client's own typed run
    work_texts: WorkTexts  # the words of each open delegation's line in the work note
    update_label: str  # in front of every update the writer words: backend results, the application's message, or both
    result_label: str  # in front of a backend result in the talker's memory, which marks it as reference data
    context_label: str  # in front of the application's context and its typed question in the talker's memory
    cut_mark: str  # after the words a cut line kept, in the talker's memory and in the next delegation's input
    transcript: TranscriptText  # the words of each delegation's input
    writer_notes: WriterNotes
    call_hints: CallHints
    normalizer_prompt: NormalizerPrompt
    frame_before: str  # said before the client's instructions in the routing decision
    frame_after: str  # said after them, before the routing text
    writer_frame_before: str  # said before the client's instructions in the writer's call
    writer_frame_after: str  # said after them, before the commentary or progress text
    caller_label: str  # in front of a caller turn that follows a backend result or application context
    runtime_label: str  # in front of the routing decision's runtime data, sent after the conversation
    decision_cue: str  # the routing decision's last line, after the runtime data: what to decide now
    newest_turn: str  # before the cue when caller turns are decided: "{turn}", their words

    def framed(self, instructions: str) -> str:
        """The client's instructions as the routing decision reads them: word for word, inside the set's frame."""
        return self.frame_before + instructions + self.frame_after

    def writer_framed(self, instructions: str) -> str:
        """The client's instructions as the writer reads them: word for word, inside the set's writer frame."""
        return self.writer_frame_before + instructions + self.writer_frame_after


@dataclass(frozen=True)
class Prompts:
    """One read of the example's prompt file: the prompt set and the lines, built at once, and the named prompts."""

    path: Path  # the file read: PROMPT_FILE_PATH, or the prompts.yaml beside this module
    agent_prompts: Mapping[str, Any]  # the file's agent_prompts, as read
    cascade: PromptSet
    lines: FixedLines

    def text(self, group: str, name: str) -> str:
        """Entry ``name`` of ``agent_prompts.<group>`` (``Talker``, ``Backend`` or ``Channel``), stripped.

        Raises:
            KeyError: The file has no such entry, or its text is blank; the message names the file, and the group's
                entries when the entry is missing.
        """
        entries = self.agent_prompts.get(group)
        entries = entries if isinstance(entries, Mapping) else {}
        entry = entries.get(name)
        content = entry.get("content") if isinstance(entry, Mapping) else None
        if not isinstance(content, str):
            known = ", ".join(map(str, entries)) or "none"
            raise KeyError(f"agent_prompts.{group}.{name} was not found in {self.path}; known: {known}")
        if not content.strip():
            raise KeyError(f"agent_prompts.{group}.{name} in {self.path} has no text")
        return content.strip()


def _paths(cls: type, prefix: str = "") -> list[str]:
    """Every text field of ``cls`` by its dotted path, a nested one's under its field (``work_texts.item``)."""
    hints = get_type_hints(cls)
    paths = []
    for field in fields(cls):
        kind, path = hints[field.name], f"{prefix}{field.name}"
        paths += _paths(kind, f"{path}.") if is_dataclass(kind) else [path]
    return paths


def _built(cls: type, texts: Mapping[str, str], prefix: str = ""):
    """``cls`` built from the texts named by its fields' dotted paths."""
    hints = get_type_hints(cls)
    values = {}
    for field in fields(cls):
        kind, path = hints[field.name], f"{prefix}{field.name}"
        values[field.name] = _built(kind, texts, f"{path}.") if is_dataclass(kind) else texts[path]
    return cls(**values)


def _texts(agent_prompts: Mapping[str, Any], group: str, path: Path) -> dict[str, str]:
    """Each entry's ``content`` in ``agent_prompts.<group>``, by its name, as written.

    Raises:
        KeyError: The file has no such group.
        ValueError: An entry has no text.
    """
    entries = agent_prompts.get(group)
    if not isinstance(entries, Mapping):
        raise KeyError(f"agent_prompts.{group} was not found in {path}")
    texts = {}
    for name, entry in entries.items():
        content = entry.get("content") if isinstance(entry, Mapping) else None
        if not isinstance(content, str):
            raise ValueError(f"agent_prompts.{group}.{name} in {path} needs a content")
        texts[str(name)] = content
    return texts


def load_prompts() -> Prompts:
    """Read the example's prompt file once: ``PROMPT_FILE_PATH``, or the ``prompts.yaml`` beside this module.

    The prompt set and the lines are built at once, since every session uses them; the named prompts are read when used
    (:meth:`Prompts.text`).

    Raises:
        RuntimeError: The file is missing or malformed.
        KeyError: ``agent_prompts.Cascade`` or ``agent_prompts.Lines`` is missing, or a text of the prompt set; the
            message names it and the file.
        ValueError: An entry of those groups has no text, or a line is missing or unknown.
    """
    path = resolve_prompt_catalog_path(__file__)
    groups = load_yaml_file(path, required=True).get("agent_prompts")
    groups = groups if isinstance(groups, Mapping) else {}
    cascade = _texts(groups, "Cascade", path)
    missing = [name for name in _paths(PromptSet) if name not in cascade]
    if missing:
        raise KeyError(f"agent_prompts.Cascade in {path} has no {', '.join(missing)}")
    return Prompts(path, groups, _built(PromptSet, cascade), FixedLines.parse(_texts(groups, "Lines", path)))
