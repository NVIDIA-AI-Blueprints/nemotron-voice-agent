# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The talker: one routing decision per caller turn.

Joins the client's instructions with the prompt set's frame and routing rules, asks the frontend model for a
``{action, speech}`` decision, and applies the floor (:meth:`Talker.floored`): a turn gets a reply, a delegation, or,
where the prompt set declares it, a silent ``wait``. A decision that fails while work is open says the holding line
instead of starting a new delegation, which would join the work in progress. The layout of the routing call's
instructions is built once, part by part (:meth:`Talker.routing_parts`), so a view of the request (``/history``) shows
what is sent.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from examples.frontend_backend_live.cascade.memory import is_reference_item
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
    """Routes each caller turn to a spoken reply, a delegation, or a silent wait where the set declares it."""

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

    def routing_parts(self, instructions: Sequence[PromptPart], work_line: str = "") -> list[PromptPart]:
        """The routing call's instructions, part by part in the order they are sent, each with its source.

        Joining each part's ``sep`` and ``text`` gives exactly what :meth:`decide` sends, so a view of the request
        (``/history``) shows what is sent; the provider adds its ``reply_format`` itself.
        """
        prompts, name = self.prompts, self.prompts.name
        parts = []
        if prompts.frame_before:
            parts.append(PromptPart(f"prompt set {name}: the voice's role", prompts.frame_before))
        parts += instructions
        if prompts.frame_after:
            parts.append(PromptPart(f"prompt set {name}: how the instructions apply", prompts.frame_after))
        parts.append(PromptPart(f"prompt set {name}: routing rules", prompts.routing, "\n"))
        if work_line:
            parts.append(PromptPart("open delegations (runtime)", work_line, "\n"))
        return parts

    def labeled(self, history: list[dict]) -> list[dict]:
        """``history`` with the set's ``caller_label`` in front of each caller turn that follows reference data.

        A caller's "yes" right after a backend result then reads as the caller's, not as part of the result. A set
        with an empty label leaves the history as it is.
        """
        labeled, after_reference = [], False
        for item in history:
            caller = item["role"] == "user" and not is_reference_item(item)
            labeled.append(
                {**item, "content": self.prompts.caller_label + item["content"]} if caller and after_reference else item
            )
            after_reference = is_reference_item(item)
        return labeled

    async def decide(
        self, instructions: str, history: list[dict], work_in_progress: str = "", *, work_open: bool = False
    ) -> TalkerDecision:
        """Return the routing decision for ``history`` under the client's ``instructions``, after the floor.

        ``work_in_progress`` is the cascade's work line (``CallerTurns.work_line``): the delegations still open, or,
        with a server delegation, that none is open; it is empty in client mode. ``work_open`` says whether any
        delegation is open, so that a failed decision holds instead of delegating.
        """
        parts = self.routing_parts([PromptPart("instructions", instructions)], work_in_progress)
        routing = "".join(part.sep + part.text for part in parts)
        decision = await self.frontend.complete(routing, self.labeled(history), self.spec)
        return self.floored(decision, work_open=work_open)

    def floored(self, decision: TalkerDecision, *, work_open: bool = False) -> TalkerDecision:
        """The decision the cascade carries out: a turn that reached the talker never ends with nothing done.

        ``delegate``, ``speak`` with words, and a declared ``wait`` stand. A decision with nothing to say becomes a
        ``wait`` where the set declares one, and otherwise counts as a failure (:meth:`after_failure`), as does output
        the parser could not use.
        """
        action = decision.action
        if decision.diagnostics.get("recovery") in UNUSABLE:
            return self.after_failure(decision.diagnostics, work_open=work_open)
        if action == "delegate" or (action == "speak" and decision.speech.strip()):
            return decision
        if action == "wait" and action in self.prompts.actions:
            return TalkerDecision("wait", "", decision.diagnostics)
        diagnostics = {**decision.diagnostics, "recovery": "empty_speak" if action == "speak" else f"{action}_removed"}
        if "wait" in self.prompts.actions:
            return TalkerDecision("wait", "", diagnostics)
        return self.after_failure(diagnostics, work_open=work_open)

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
