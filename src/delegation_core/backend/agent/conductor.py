# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's conductor: one model call per ``decide()`` that places requests through the control tools.

Each finished tool call is placed at once, and the IM's outcome line becomes that call's output in the conductor's
memory. ``conductor.timeout_seconds`` limits the wait for the model, never a placement in progress. Requests the round
leaves unplaced (a failed or timed-out round, or no decision naming them) fall back to the ``steer`` rule
(``SteerOpenWork``). Any text the model writes is dropped.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import aclosing
from copy import deepcopy
from dataclasses import replace
from typing import Final

from loguru import logger

from delegation_core.backend.conductors import SteerOpenWork
from delegation_core.backend.port import TERMINALS, ConductorInput, ControlTools, Decision, Place, function_call_item
from delegation_core.backend.providers import Backend

from .memory import ConductorMemory, DecisionRound
from .render import round_input
from .settings import ConductorSettings

# Every model turn is a placement, and one round may place several requests.
TOOL_CHOICE: Final = "required"
PARALLEL_TOOL_CALLS: Final = True

# The Works the conductor was shown, each with its tasks' ids.
Shown = Mapping[str, frozenset[str]]


def placing(decision: Decision, shown: Shown) -> tuple[str, ...]:
    """The requests ``decision`` places: a Start's or a Steer's ``delegations``, an Answer's one; a Cancel's none.

    A decision that names a Work, or a task of one, that the conductor was not shown places none: the IM refuses it.
    """
    work_id, task_id = getattr(decision, "work_id", None), getattr(decision, "task_id", None)
    if work_id is not None and (work_id not in shown or (task_id is not None and task_id not in shown[work_id])):
        return ()
    delegations = tuple(getattr(decision, "delegations", ()))
    answered = getattr(decision, "delegation_id", None)
    return (*delegations, answered) if answered is not None else delegations


class LlmConductor:
    """The backend LLM agent's conductor (``port.Conductor``): one per IM session, with its own memory."""

    def __init__(
        self,
        provider: Callable[[], Backend],
        vocabulary: ControlTools,
        settings: ConductorSettings,
        model: str,
    ):
        """Place requests through ``vocabulary``'s tools on ``model``; nothing is called before the first decision.

        Args:
            provider: Makes the conductor's model provider, at its first decision round.
            vocabulary: The conductor's instructions and control tools (``texts.conductor_prompt``,
                ``texts.control_tools``).
            settings: The conductor's settings (``backend.llm-agent.conductor``).
            model: The model it runs on.
        """
        self.make_provider, self.vocabulary, self.settings, self.model = provider, vocabulary, settings, model
        self.provider: Backend | None = None
        self.memory = ConductorMemory(settings.memory_tokens)
        self.rule = SteerOpenWork()  # what places a request the model left: the backend LLM's steer rule

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place every new request: the model's decisions first, then the rule for any it left.

        A round that fails or passes its time limit still ends with the rule's placements, so ``decide()`` raises only
        when it is cancelled or ``place`` itself raises.
        """
        if not given.arrivals:
            return
        shown = {work.work_id: frozenset(task.task_id for task in work.tasks) for work in given.works}
        placed: set[str] = set()
        decision_round: DecisionRound | None = None
        try:
            decision_round = DecisionRound(round_input(given))
            await self._ask(given, decision_round, place, shown, placed)
        except asyncio.CancelledError:
            if decision_round is not None and decision_round.placed:  # cancelled after it placed (the close)
                self.memory.add(decision_round)  # what it placed stays remembered
            raise
        except Exception as exc:
            logger.warning(f"conductor.failed model={self.model}: {type(exc).__name__}: {exc}")
        if decision_round is not None:  # a round whose input could not be rendered has nothing to remember
            self.memory.add(decision_round)
        left = tuple(each for each in given.arrivals if each.delegation_id not in placed)
        if left:
            logger.info(f"conductor.rule_places delegations={[each.delegation_id for each in left]}")
            await self.rule.decide(replace(given, arrivals=left), place)

    async def close(self) -> None:
        """Close the provider, if a decision made one."""
        provider, self.provider = self.provider, None
        if provider is not None:
            await provider.close()

    # ------------------------------------------------------------------------------------------------ the round
    def payload(self, given: ConductorInput, decision_round: DecisionRound) -> dict:
        """The decision round's request: the instructions, the memory and this round's input, the control tools."""
        payload = {
            "model": self.model,
            "instructions": self.vocabulary.prompt,
            "input": [*self.memory.items(), deepcopy(decision_round.message)],
            "tools": self.vocabulary.conductor_tools(),
            "tool_choice": TOOL_CHOICE,
            "parallel_tool_calls": PARALLEL_TOOL_CALLS,
            "max_output_tokens": self.settings.max_output_tokens_per_request * len(given.arrivals),
        }
        if self.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.settings.reasoning_effort}
        return payload

    async def _ask(
        self, given: ConductorInput, decision_round: DecisionRound, place: Place, shown: Shown, placed: set[str]
    ) -> None:
        """One model call: each finished call is placed at once; waiting for the model is timed, placing is not."""
        payload = self.payload(given, decision_round)
        logger.debug(
            "conductor.request model={} requests={} memory_items={}",
            self.model,
            [each.delegation_id for each in given.arrivals],
            len(payload["input"]) - 1,
        )
        if self.provider is None:
            self.provider = self.make_provider()
        deadline = asyncio.get_running_loop().time() + self.settings.timeout_seconds
        async with aclosing(self.provider.stream(payload)) as events:
            while True:
                try:
                    async with asyncio.timeout_at(deadline):
                        event = await anext(events)
                except StopAsyncIteration:
                    return
                call = function_call_item(event)
                if call is not None:
                    await self._place(call, decision_round, place, shown, placed)
                elif event.get("type") in TERMINALS:
                    self._ended(event)

    async def _place(
        self, call: dict, decision_round: DecisionRound, place: Place, shown: Shown, placed: set[str]
    ) -> None:
        """Place one call's decision and keep its outcome line as the call's output; a malformed call is skipped."""
        name = call.get("name", "")
        try:
            decision = self.vocabulary.decision(name, call.get("arguments") or "{}")
        except ValueError as exc:
            logger.warning(f"conductor.malformed call={name!r}: {exc}")
            decision_round.answered(call, json.dumps({"error": str(exc)}))
            return
        decision_round.placed = True
        outcome = await place(decision)
        placed.update(placing(decision, shown))
        decision_round.answered(call, outcome)

    def _ended(self, terminal: dict) -> None:
        """Log a round that ended without completing: no call of it came (a cut round's calls are dropped)."""
        if terminal.get("type") == "response.completed":
            return
        response = terminal.get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason") or (response.get("error") or {}).get("code")
        logger.warning(f"conductor.round_ended model={self.model} ending={terminal.get('type')} reason={reason}")
