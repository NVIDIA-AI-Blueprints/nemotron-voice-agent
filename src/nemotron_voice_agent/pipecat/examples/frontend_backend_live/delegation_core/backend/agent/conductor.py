# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The backend LLM agent's complete-decision producer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextlib import aclosing
from copy import deepcopy
from typing import Final

from loguru import logger

from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.port import (
    ConductorInput,
    ControlTools,
    DecisionRejected,
    Place,
    function_call_item,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.providers import (
    Backend,
    BrokenStreamError,
)
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.delegation_core.backend.providers.retry import (
    StreamErrorEvent,
)
from nemotron_voice_agent.pipecat.shared.delegation_contract import CHAT_COMPLETIONS, TERMINALS

from .memory import ConductorMemory, DecisionRound
from .render import round_input
from .settings import ConductorSettings

TOOL_CHOICE: Final = "required"
PARALLEL_TOOL_CALLS: Final = False


class LlmConductor:
    """The backend LLM agent's conductor (``port.Conductor``): one per IM session, with its own memory."""

    model_placement = True

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
        self.round = 0

    async def decide(self, given: ConductorInput, place: Place) -> None:
        """Submit one complete decision or propagate failure to bounded Core recovery."""
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
            # Core supplies the current input and rejection on repair. Keep only possibly admitted failures.
            if decision_round is not None and decision_round.placed:
                self.memory.add(decision_round)
            raise
        else:
            self.memory.add(decision_round)

    async def close(self) -> None:
        """Close the provider, if a decision made one."""
        provider, self.provider = self.provider, None
        if provider is not None:
            await provider.close()

    def payload(self, given: ConductorInput, decision_round: DecisionRound) -> dict:
        """Build the round's request: the instructions, the memory, this round's input and the control tools."""
        tools = self.vocabulary.conductor_tools()
        if not given.works:
            (tool,) = tools
            actions = tool["parameters"]["properties"]["actions"]
            items = actions.get("items") if isinstance(actions, dict) else None
            if isinstance(items, dict) and isinstance(choices := items.get("oneOf"), list):
                eligible = [
                    choice
                    for choice in choices
                    if not isinstance(choice, dict) or "targets" not in (choice.get("required") or ())
                ]
                items["oneOf"] = eligible or choices
        if self.settings.strict_tool_arguments:
            tools = [{**tool, "strict": True} for tool in tools]
        payload = {
            "model": self.model,
            "instructions": self.vocabulary.instructions(self.max_running),
            "input": [*self.memory.items(), deepcopy(decision_round.message)],
            "tools": tools,
            "tool_choice": TOOL_CHOICE,
            "parallel_tool_calls": PARALLEL_TOOL_CALLS,
        }
        if self.settings.max_output_tokens_per_request is not None:
            payload["max_output_tokens"] = self.settings.max_output_tokens_per_request * len(given.arrivals)
        if self.settings.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.settings.reasoning_effort}
        return payload

    async def _ask(self, given: ConductorInput, decision_round: DecisionRound, place: Place) -> None:
        """Validate one completed model envelope before submitting any action."""
        payload = self.payload(given, decision_round)
        if self.provider is None:
            self.provider = self.make_provider()
        translated = self.provider.endpoint.api == CHAT_COMPLETIONS
        if translated:
            (tool,) = payload["tools"]
            payload["tool_choice"] = {"type": "function", "name": tool["name"]}
            del payload["parallel_tool_calls"]
        logger.debug(
            "conductor.request model={} requests={} memory_items={}",
            self.model,
            [each.delegation_id for each in given.arrivals],
            len(payload["input"]) - 1,
        )
        logger.debug(f"conductor request: {json.dumps(payload, ensure_ascii=False)}")
        self.round += 1
        round_number = self.round
        deadline = asyncio.get_running_loop().time() + self.settings.timeout_seconds
        reported: dict = {}
        terminal = None
        calls: list[dict] = []
        opened = False
        try:
            async with aclosing(self.provider.stream(payload, retries=self.provider.limits.provider_retries)) as events:
                while True:
                    try:
                        async with asyncio.timeout_at(deadline):
                            event = await anext(events)
                    except (StopAsyncIteration, BrokenStreamError) as exc:
                        raise DecisionRejected("Conductor stream ended without a completed decision") from exc
                    except StreamErrorEvent as exc:
                        if not opened:
                            raise
                        raise DecisionRejected("Conductor stream failed before a completed decision") from exc
                    except TimeoutError as exc:
                        terminal = "timeout"
                        raise DecisionRejected("Conductor stream timed out before a completed decision") from exc
                    opened = True
                    response = event.get("response") or {}
                    if isinstance(response, dict):
                        for key in ("id", "model"):
                            if response.get(key):
                                reported[key] = response[key]
                    call = function_call_item(event)
                    if call is not None:
                        calls.append(call)
                    elif event.get("type") in TERMINALS:
                        terminal = event["type"]
                        reported["usage"] = response.get("usage")
                        output = response.get("output") or []
                        logger.debug(f"conductor reply: {json.dumps(output, ensure_ascii=False)}")
                        self._ended(event)
                        if terminal != "response.completed":
                            raise DecisionRejected(f"Conductor response did not complete: {terminal}")
                        if len(calls) != 1:
                            raise DecisionRejected("Conductor response must contain one complete decision call")
                        await self._place(calls[0], decision_round, place)
                        return
        except asyncio.CancelledError:
            terminal = terminal or "cancelled"
            raise
        except Exception:
            terminal = terminal or "stream_error"
            raise
        finally:
            usage = reported.get("usage")
            logger.debug(
                "conductor.round.usage backend=llm-agent role=conductor model={} model_source={} round={} "
                "attempt={} terminal={} conversation_ref={} delegation_ids={} response_id={} usage={} usage_missing={} "
                "requested_model={} configured_model={} reported_model={} usage_authority=logical_conductor_round",
                reported.get("model") or self.model,
                ("translated_request" if translated else "provider_event")
                if reported.get("model")
                else "configured_model",
                round_number,
                given.attempt,
                terminal or "exhausted",
                given.conversation_ref,
                [row.delegation_id for row in given.arrivals],
                reported.get("id"),
                usage,
                usage is None,
                self.model,
                self.provider.endpoint.model,
                None if translated else reported.get("model"),
            )

    async def _place(self, call: dict, decision_round: DecisionRound, place: Place) -> None:
        """Submit one validated envelope and retain the Core's complete action receipt."""
        name = call.get("name", "")
        try:
            decision = self.vocabulary.decision(name, call.get("arguments") or "{}")
        except DecisionRejected as exc:
            logger.warning(f"conductor.malformed call={name!r}: {exc}")
            raise
        decision_round.placed = True
        try:
            outcome = await place(decision)
        except DecisionRejected:
            decision_round.placed = False
            raise
        except BaseException:
            decision_round.answered(call, json.dumps({"status": "unknown"}))
            raise
        decision_round.answered(call, outcome)

    def _ended(self, terminal: dict) -> None:
        """Log a round that ended without completing; a cut round's calls never come."""
        ending = terminal.get("type")
        if ending == "response.completed":
            return
        response = terminal.get("response") or {}
        reason = (response.get("incomplete_details") or {}).get("reason") or (response.get("error") or {}).get("code")
        logger.warning(f"conductor.round_ended model={self.model} ending={ending} reason={reason}")
        if ending == "response.failed" and reason in ("server_error", "rate_limit_exceeded"):
            raise DecisionRejected(f"Conductor provider failed before a completed decision: {reason}")
        if ending != "response.incomplete" or reason != "max_output_tokens":
            raise RuntimeError(f"Conductor round failed: {reason or ending}")
