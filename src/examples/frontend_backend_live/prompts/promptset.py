# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prompt set: every text the cascade's own model calls read, and the words of each delegation's input.

A set holds the talker's routing rules and the frame around the client's instructions, the writer's frame, commentary,
answer and progress rules, the work note the talker reads, the labels and notes around an update, the marks of
reference data and of a cut line in the talker's memory, the speech hint of each call's JSON schema, the normalizer's
prompt, and the words of the transcript each delegation carries to the backend. Behavior that differs between sets is
data here (the actions a set declares, its labels and frames), so no code asks which set it runs.

A new set is one module in this package that builds its ``PROMPT_SET`` from these classes, and one entry in
``PROMPT_SETS`` (``__init__.py``); ``prompt_version`` in ``config.yaml`` chooses a set by its name.

The lines the voice says word for word are not model text: they are ``lines`` in the example's ``prompts.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    name: str
    actions: tuple[str, ...]  # the routing actions the talker may take: speak, delegate and wait
    routing: str  # appended to the client's instructions for the routing decision
    commentary: str  # appended for commentary phrasing
    answer: str  # appended after the commentary text in every writer call but a progress line's
    progress: str  # the writer's rules for a progress line, read in place of the commentary and answer texts
    work_in_progress: str  # the open delegations, one numbered line each: "{items}"
    no_work_in_progress: str  # what the talker reads, with a server delegation, when none is open
    answer_ready_label: str  # an open delegation whose answer arrived and is said next
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

    def framed(self, instructions: str) -> str:
        """The client's instructions as the routing decision reads them: word for word, inside the set's frame."""
        return self.frame_before + instructions + self.frame_after

    def writer_framed(self, instructions: str) -> str:
        """The client's instructions as the writer reads them: word for word, inside the set's writer frame."""
        return self.writer_frame_before + instructions + self.writer_frame_after
