# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's conductor: one model call per ``decide()`` that places requests through the control tools.

The round asks the model for one tool call (``parallel_tool_calls: false``).
Each finished tool call is placed at once, and the IM's outcome line becomes that call's output in the conductor's
memory. ``conductor.timeout_seconds`` limits the wait for the model, never a placement in progress. Requests the round
leaves unplaced (a failed or timed-out round, a refused decision, or no decision naming them) are left to the IM, whose
``steer`` rule places them against the Works as the round left them: a request that came with one the round started
joins that Work. Any text the model writes is dropped. While no Work is open, ``conductor.when_idle`` may place the
requests with no model call (``start``); such a round keeps its input in the memory, as a failed round does. Each
model call's request and reply are logged in full at debug level.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing
from copy import deepcopy
from typing import Final

from loguru import logger

from delegation_core.backend.port import TERMINALS, ConductorInput, ControlTools, Place, function_call_item
from delegation_core.backend.providers import Backend

from .memory import ConductorMemory, DecisionRound
from .render import round_input
from .settings import ConductorSettings

# Every model turn is a placement: one call per round, which may name several requests; the IM's steer rule places the
# requests it leaves.
TOOL_CHOICE: Final = "required"
PARALLEL_TOOL_CALLS: Final = False


class LlmConductor:
    """The backend LLM agent's conductor (``port.Conductor``): one per IM session, with its own memory."""

    def __init__(
        self,
        provider: Callable[[], Backend],
        vocabulary: ControlTools,
        settings: ConductorSettings,
        model: str,
        max_running: int,
    ):
        """Place requests through ``vocabulary``'s tools on ``model``; nothing is called before the first decision.

        Args:
            provider: Makes the conductor's model provider, at its first decision round.
            vocabulary: The conductor's instructions and control tools (``texts.conductor_prompt``,
                ``texts.control_tools``).
            settings: The conductor's settings (``backend.llm-agent.conductor``).
            model: The model it runs on.
            max_running: The tasks the agent runs at once, which its instructions state.
        """
        self.make_provider, self.vocabulary, self.settings, self.model = provider, vocabulary, settings, model
        self.max_running = max_running
        self.provider: Backend | None = None
        self.memory = ConductorMemory(settings.memory_tokens)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place the new requests the model's decisions name, or the idle rule's decision; the IM places any left.

        A round that fails or passes its time limit returns, so ``decide()`` raises only when it is cancelled or
        ``place`` itself raises.
        """
        if not given.arrivals:
            return
        decision_round: DecisionRound | None = None
        try:
            decision_round = DecisionRound(round_input(given))
            decision = self.settings.when_idle.decision(given)
            if decision is None:
                await self._ask(given, decision_round, place)
            else:
                logger.debug(f"conductor.when_idle rule={self.settings.when_idle.name} decision={decision}")
                decision_round.placed = True
                await place(decision)
        except asyncio.CancelledError:
            if decision_round is not None and decision_round.placed:  # cancelled after it placed (the close)
                self.memory.add(decision_round)  # what it placed stays remembered
            raise
        except Exception as exc:
            logger.warning(f"conductor.failed model={self.model}: {type(exc).__name__}: {exc}")
        if decision_round is not None:  # a round whose input could not be rendered has nothing to remember
            self.memory.add(decision_round)

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
            "instructions": self.vocabulary.instructions(self.max_running),
            "input": [*self.memory.items(), deepcopy(decision_round.message)],
            "tools": self.vocabulary.conductor_tools(),
            "tool_choice": TOOL_CHOICE,
            "parallel_tool_calls": PARALLEL_TOOL_CALLS,
            "max_output_tokens": self.settings.max_output_tokens_per_request * len(given.arrivals),
        }
        if self.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.settings.reasoning_effort}
        return payload

    async def _ask(self, given: ConductorInput, decision_round: DecisionRound, place: Place) -> None:
        """One model call: each finished call is placed at once; waiting for the model is timed, placing is not."""
        payload = self.payload(given, decision_round)
        logger.debug(
            "conductor.request model={} requests={} memory_items={}",
            self.model,
            [each.delegation_id for each in given.arrivals],
            len(payload["input"]) - 1,
        )
        # The round's exact request, and below its reply, for reading a decision back from the logs.
        logger.debug(f"conductor request: {json.dumps(payload, ensure_ascii=False)}")
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
                    await self._place(call, decision_round, place)
                elif event.get("type") in TERMINALS:
                    output = (event.get("response") or {}).get("output") or []
                    logger.debug(f"conductor reply: {json.dumps(output, ensure_ascii=False)}")
                    self._ended(event)

    async def _place(self, call: dict, decision_round: DecisionRound, place: Place) -> None:
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
        decision_round.answered(call, outcome)

    def _ended(self, terminal: dict) -> None:
        """Log a round that ended without completing: no call of it came (a cut round's calls are dropped)."""
        if terminal.get("type") == "response.completed":
            return
        response = terminal.get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason") or (response.get("error") or {}).get("code")
        logger.warning(f"conductor.round_ended model={self.model} ending={terminal.get('type')} reason={reason}")
