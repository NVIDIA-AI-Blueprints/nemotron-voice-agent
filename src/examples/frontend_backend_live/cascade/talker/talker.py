# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: one routing decision per caller turn.

Joins the client's instructions with the prompt set's frame and routing rules, asks the frontend model for a
``{action, speech}`` decision, and makes it one the cascade can carry out (:meth:`Talker.floored`): a turn gets a
reply, a delegation, or a silent ``wait``. A decision that fails while work is open says the holding line instead of
starting a new delegation, which would join the work in progress. The layout of the routing call's instructions is
built once, part by part (:meth:`Talker.routing_parts`), so a view of the request (``/history``) shows what is sent.
The call's runtime data, the work line, the newest turn and the decision cue, goes after the conversation
(:meth:`Talker.runtime_data`), where the model reads it last.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from examples.frontend_backend_live.models.base import CallSpec, Frontend
from examples.frontend_backend_live.models.decision import UNUSABLE, TalkerDecision
from examples.frontend_backend_live.prompts import PromptSet


@dataclass(frozen=True)
class PromptPart:
    """One part of a call's instructions: where it comes from, its text, and what joins it to the part before it."""

    source: str
    text: str
    sep: str = ""


class Talker:
    """Routes each caller turn to a spoken reply, a delegation, or a silent wait."""

    def __init__(self, frontend: Frontend, prompts: PromptSet, *, holding: str = ""):
        """Bind the frontend model and the prompt set; the call's spec comes from the set.

        Args:
            frontend: The model that decides.
            prompts: The prompt set: the call's texts, its actions and its hint.
            holding: The line said word for word when a decision fails while work is open (``lines.holding`` in
                ``prompts.yaml``); empty: such a turn is delegated, as when no work is open.
        """
        self.frontend, self.prompts, self.holding = frontend, prompts, holding
        self.spec = CallSpec("talker", prompts.actions, prompts.call_hints.talker)

    def routing_parts(self, instructions: Sequence[PromptPart]) -> list[PromptPart]:
        """The routing call's instructions, part by part in the order they are sent, each with its source.

        Joining each part's ``sep`` and ``text`` gives exactly what :meth:`decide` sends, so a view of the request
        (``/history``) shows what is sent; the provider adds its ``reply_format`` itself.
        """
        prompts, name = self.prompts, self.prompts.name
        parts = [PromptPart(f"prompt set {name}: the voice's role", prompts.frame_before)]
        parts += instructions
        parts.append(PromptPart(f"prompt set {name}: how the instructions apply", prompts.frame_after))
        parts.append(PromptPart(f"prompt set {name}: routing rules", prompts.routing, "\n"))
        return parts

    def runtime_data(self, work_line: str = "", turn: str = "") -> str:
        """The routing call's last user entry, after the conversation: the label, the work line (if any), the cue.

        ``turn``: the words of the caller turns decided, named before the cue.
        """
        prompts = self.prompts
        newest = prompts.newest_turn.format(turn=turn) + "\n" if turn else ""
        return prompts.runtime_label + "\n" + (f"{work_line}\n\n" if work_line else "") + newest + prompts.decision_cue

    async def decide(
        self,
        instructions: str,
        history: list[dict],
        work_in_progress: str = "",
        *,
        work_open: bool = False,
        turn: str = "",
    ) -> TalkerDecision:
        """Return the routing decision for ``history`` under the client's ``instructions``, after :meth:`floored`.

        ``history`` is the voice history as the talker reads it (``TalkerMemory.labeled``). ``work_in_progress`` is the
        cascade's work line (``CallerTurns.work_line``): the delegations still open, or, with a server delegation, that
        none is open; it is empty in client mode. It goes after the conversation, in the runtime data. ``work_open``
        says whether any delegation is open, so that a failed decision holds instead of delegating. ``turn`` holds the
        words of the caller turns decided: a turn decided earlier joins them in one message after a wait.
        """
        parts = self.routing_parts([PromptPart("instructions", instructions)])
        routing = "".join(part.sep + part.text for part in parts)
        # After the caller's newest turn, a line break sets the runtime data apart: the provider joins the two entries.
        joined = bool(history) and history[-1].get("role") == "user"
        runtime = {"role": "user", "content": ("\n" if joined else "") + self.runtime_data(work_in_progress, turn)}
        decision = await self.frontend.complete(routing, [*history, runtime], self.spec)
        return self.floored(decision, work_open=work_open)

    def floored(self, decision: TalkerDecision, *, work_open: bool = False) -> TalkerDecision:
        """The decision the cascade carries out: a turn that reached the talker never ends with nothing done.

        ``delegate``, ``speak`` with words, and ``wait`` stand; a ``speak`` with nothing to say becomes a ``wait``.
        Output the parser could not use counts as a failure (:meth:`after_failure`).
        """
        action = decision.action
        if decision.diagnostics.get("recovery") in UNUSABLE:
            return self.after_failure(decision.diagnostics, work_open=work_open)
        if action == "delegate" or (action == "speak" and decision.speech.strip()):
            return decision
        if action == "wait":
            return TalkerDecision("wait", "", decision.diagnostics)
        return TalkerDecision("wait", "", {**decision.diagnostics, "recovery": "empty_speak"})

    def after_failure(self, diagnostics: dict, *, work_open: bool) -> TalkerDecision:
        """What a turn becomes when the talker's decision failed or said nothing.

        While work is open, the holding line, word for word: a delegation would join the work in progress, and could
        restart it, for a turn nobody decided. Otherwise a delegation: no tool has run at the talker stage, so
        delegating can never repeat an action.
        """
        if work_open and self.holding:
            held = {**diagnostics, "recovery": "holding", "failed": diagnostics.get("recovery")}
            return TalkerDecision("speak", self.holding, held)
        return TalkerDecision("delegate", "", diagnostics)
