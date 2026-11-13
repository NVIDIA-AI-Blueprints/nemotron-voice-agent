# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's conductor: one model call per ``decide()``, placing requests through the control tools."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing
from copy import deepcopy
from typing import Final

from loguru import logger

from examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    ControlTools,
    Place,
    function_call_item,
)
from examples.frontend_backend_live.delegation_core.backend.providers import Backend
from examples.shared.delegation_contract import TERMINALS

from .memory import ConductorMemory, DecisionRound
from .render import round_input
from .settings import ConductorSettings

# Every round is one placement call, which may name several requests.
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
        """Place requests through ``vocabulary``'s tools on ``model``; nothing is called before the first decision."""
        self.make_provider, self.vocabulary, self.settings, self.model = provider, vocabulary, settings, model
        self.max_running = max_running
        self.provider: Backend | None = None
        self.memory = ConductorMemory(settings.memory_tokens)

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Place the new requests that the model's calls name; the IM's fallback rule places any left.

        A round that fails, times out or whose ``place`` raises is logged as ``conductor.failed``; only a cancel raises.
        """
        if not given.arrivals:
            return
        decision_round: DecisionRound | None = None
        try:
            decision_round = DecisionRound(round_input(given))
            await self._ask(given, decision_round, place)
        except asyncio.CancelledError:
            if decision_round is not None and decision_round.placed:
                self.memory.add(decision_round)
            raise
        except Exception as exc:
            logger.warning(f"conductor.failed model={self.model}: {type(exc).__name__}: {exc}")
        # A failed round is remembered too: the next round shows only the conversation after it.
        if decision_round is not None:
            self.memory.add(decision_round)

    async def close(self) -> None:
        """Close the provider, if a decision made one."""
        provider, self.provider = self.provider, None
        if provider is not None:
            await provider.close()

    def payload(self, decision_round: DecisionRound) -> dict:
        """Build the round's request: the instructions, the memory, this round's input and the control tools.

        It sets no output limit: the model's catalog entry sets it, as for every call of that model.
        """
        payload = {
            "model": self.model,
            "instructions": self.vocabulary.instructions(self.max_running),
            "input": [*self.memory.items(), deepcopy(decision_round.message)],
            "tools": self.vocabulary.conductor_tools(),
            "tool_choice": TOOL_CHOICE,
            "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        }
        if self.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.settings.reasoning_effort}
        return payload

    async def _ask(self, given: ConductorInput, decision_round: DecisionRound, place: Place) -> None:
        """Make one model call, placing each finished call at once; only the wait for the model is timed."""
        payload = self.payload(decision_round)
        logger.debug(
            "conductor.request model={} requests={} memory_items={}",
            self.model,
            [each.delegation_id for each in given.arrivals],
            len(payload["input"]) - 1,
        )
        logger.debug(f"conductor request: {json.dumps(payload, ensure_ascii=False)}")
        if self.provider is None:
            self.provider = self.make_provider()
        deadline = asyncio.get_running_loop().time() + self.settings.timeout_seconds
        async with aclosing(self.provider.stream(payload, retries=self.provider.limits.provider_retries)) as events:
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
        """Log a round that ended without completing; a cut round's calls never come."""
        if terminal.get("type") == "response.completed":
            return
        response = terminal.get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason") or (response.get("error") or {}).get("code")
        logger.warning(f"conductor.round_ended model={self.model} ending={terminal.get('type')} reason={reason}")
