# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prompt set: every text the cascade's own model calls read, and the texts the sets share.

A set holds the talker's routing rules and the frame around the client's instructions, the writer's commentary and
answer rules, the work note the talker reads, the labels and notes around an update, the speech hint of each call's
JSON schema, and the normalizer's prompt. Each set is one module (``v1.py``, ``v2.py``, ...) that builds its
``PROMPT_SET``; ``PROMPT_SETS`` in ``__init__.py`` lists them by name. Behavior that differs between sets is data here
(the actions a set declares, its labels and frames), so no code asks which set it runs.

The shared texts below are the work note's, the typed request's label, the writer's notes and update label, the schema
hints and the normalizer's prompt. Sets v1 and v2 use them unchanged.

The lines the voice says word for word are not model text: they are ``lines`` in the example's ``prompts.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WriterNotes:
    """The notes added to an update for the writer, each only when it applies."""

    told: str  # what the caller was already told about the update's delegations: "{told}"
    must_speak: str  # the application asked for the update to be said
    newer: str  # newer requests are still open and may change the result: "{requests}", each in quotes, "; " between


@dataclass(frozen=True)
class CallHints:
    """The description of ``speech`` in each call's JSON schema, by the call's role."""

    talker: str  # the routing decision
    commentary: str  # an update the writer may leave unsaid
    answer: str  # an answer the caller waits for
    normalizer: str  # the newest caller turn, rewritten


@dataclass(frozen=True)
class NormalizerPrompt:
    """What the normalizer reads: its instructions, and the label in front of the turn it rewrites."""

    instructions: str
    newest_turn: str


@dataclass(frozen=True, kw_only=True)
class PromptSet:
    """The text a prompt set contributes to each model call of the cascade.

    The cascade reads the work note's texts and the labels from here; the talker, the writer
    and the normalizer read the rest. Empty frames and labels change nothing, so a set that has none sends the
    calls' own text unchanged.
    """

    name: str
    actions: tuple[str, ...]  # the routing actions the talker may take: speak and delegate, and wait where declared
    routing: str  # appended to the client's instructions for the routing decision
    commentary: str  # appended for commentary phrasing
    answer: str  # appended when the update is a waiting caller's answer
    work_in_progress: str  # the open delegations, one numbered line each: "{items}"
    no_work_in_progress: str  # what the talker reads, with a server delegation, when none is open
    answer_ready_label: str  # an open delegation whose answer arrived and is said next
    typed_request_label: str  # work the frontend did not delegate: the client's own typed run
    update_label: str  # in front of a backend result the writer words
    commentary_label: str  # in front of the application's commentary the writer words
    writer_notes: WriterNotes
    call_hints: CallHints
    normalizer_prompt: NormalizerPrompt
    frame_before: str = ""  # said before the client's instructions in the routing decision
    frame_after: str = ""  # said after them, before the routing text
    writer_frame_before: str = ""  # said before the client's instructions in the writer's call
    writer_frame_after: str = ""  # said after them, before the commentary text
    caller_label: str = ""  # in front of a caller turn that follows a backend result or application context

    def framed(self, instructions: str) -> str:
        """The client's instructions as the routing decision reads them: word for word, inside the set's frame."""
        return self.frame_before + instructions + self.frame_after

    def writer_framed(self, instructions: str) -> str:
        """The client's instructions as the writer reads them: word for word, inside the set's writer frame."""
        return self.writer_frame_before + instructions + self.writer_frame_after


# ------------------------------------------------------------------------------------------- the texts the sets share
WORK_IN_PROGRESS = """Work in progress (requests you delegated earlier; each answer is spoken automatically when it
arrives):
{items}
Do not delegate again to ask for progress or to repeat one of these requests. If the caller asks how it is going,
speak a short status."""

# What the talker reads, with a server delegation, when no delegation is open, so it never takes its own words for work
# under way. A client delegation gets no line.
NO_WORK_IN_PROGRESS = (
    "Work in progress: none. Nothing is being checked or done for the caller right now: only a delegation starts work."
)

# An open delegation whose answer has arrived and is said next (the work note's state; new text).
ANSWER_READY_LABEL = "answer ready, said next"

# Work the frontend did not delegate: the client's own typed run.
TYPED_REQUEST_LABEL = "a typed request"

# In front of the update the writer words, as the last item of its request.
UPDATE_LABEL = "Commentary update (reference data):\n"

WRITER_NOTES = WriterNotes(
    told='\n(you already told the caller: "{told}" -- do not repeat that, report only the result)',
    must_speak="\n(the application asked for this to be said to the caller, so speak it)",
    newer=(
        "\n(the caller has since asked something else, still in progress: {requests}. If it changes this result, "
        "say what was found, briefly, and that you are updating it; otherwise report this result normally)"
    ),
)

# `speech` means something different per call, so each role states its own schema description rather than leaving it
# to prose alone; the normalizer's matches the answer's.
CALL_HINTS = CallHints(
    talker=(
        "For delegate: a brief acknowledgment that work is starting -- never its outcome, "
        "even a likely one -- or empty when the instructions say not to speak. For speak: the "
        "complete, already-verified answer or a question; never empty."
    ),
    commentary="A concise spoken paraphrase of the update, or empty to stay silent.",
    answer="The answer to speak aloud, kept brief.",
    normalizer="The answer to speak aloud, kept brief.",
)

NORMALIZER_PROMPT = NormalizerPrompt(
    instructions="""You clean up speech-recognition transcripts for a phone assistant. You do not talk
to the caller. You rewrite only the caller's newest turn so that the details in it are written the way a computer
system stores them, using the conversation so far as context.

Rules:
1. Join spelled letters and digits into one token, keeping the caller's order. Spoken symbols become characters:
   "underscore" -> "_", "dash" or "hyphen" -> "-", and in an email "at" -> "@" and "dot" -> ".".
   "m i a underscore l i three six six eight" -> "mia_li_3668"; "H A T zero four five" -> "HAT045".
2. Numbers become digits: "zero bags" -> "0 bags", "twenty three" -> "23", "May twentieth" -> "May 20".
   "oh" inside a spoken number is zero.
3. When the caller spells a word, the spelling wins over how the word was transcribed: "Lee, that's L I" -> "Li".
4. When a value is clearly the same as one already in the conversation (an ID, a name, a code or an airport from
   earlier turns or from reference data) apart from an obvious transcription error, write it exactly as it appears
   there. Only when the match is clear; never pick between two candidates.
5. When the caller corrects a value in this turn, keep the correction exactly as stated.
6. Keep everything else word for word. Do not answer, add, drop, summarize or explain, and never invent a value
   the caller did not say. If nothing needs normalizing, return the turn unchanged.

Put the rewritten turn in "speech" with "action": "speak".""",
    newest_turn="Caller's newest turn (rewrite only this):\n",
)
