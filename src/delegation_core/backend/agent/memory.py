# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's conductor's own memory: its decision rounds, each kept or dropped whole, within a token budget.

A round keeps its input and each model call with its output (the IM's outcome line, or why a malformed call was not
placed), since a chat request needs a tool message for every earlier tool call. A round whose model call failed keeps
its input alone, because the IM gives the next ``decide()`` only the conversation after it. Rounds leave oldest first
while the memory is over ``conductor.memory_tokens``; the newest round always stays.
"""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from delegation_core.ids import approx_tokens


@dataclass
class DecisionRound:
    """One ``decide()``'s exchange with the model: its input message, the answered calls, and their outputs."""

    message: dict  # the round's input: a user message (render.round_input)
    calls: list[dict] = field(default_factory=list)  # the model's calls that got an output, as input items
    outputs: list[dict] = field(default_factory=list)  # their outputs, in the same order
    placed: bool = False  # a decision of this round was handed to the IM

    def answered(self, call: dict, output: str) -> None:
        """A call the model made, and its output: both join the round."""
        call_id, name, arguments = call.get("call_id", ""), call.get("name", ""), call.get("arguments") or "{}"
        self.calls.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
        self.outputs.append({"type": "function_call_output", "call_id": call_id, "output": output})

    def items(self) -> list[dict]:
        """The round as input items: its message, its calls, then their outputs (the shape of parallel calls)."""
        return [self.message, *self.calls, *self.outputs]


class ConductorMemory:
    """The conductor's earlier decision rounds, oldest first, within ``budget`` tokens (the newest always kept)."""

    def __init__(self, budget: int):
        """Start empty, with a budget of ``budget`` tokens."""
        self.budget = budget
        self.rounds: deque[tuple[list[dict], int]] = deque()  # each round's items and its size
        self.size = 0

    def items(self) -> list[dict]:
        """Every kept round's items, oldest first, copied."""
        return [deepcopy(item) for items, _ in self.rounds for item in items]

    def add(self, decision_round: DecisionRound) -> None:
        """Keep ``decision_round``; drop the oldest rounds while the memory is over its budget."""
        items = deepcopy(decision_round.items())
        size = approx_tokens(json.dumps(items, ensure_ascii=False))
        self.rounds.append((items, size))
        self.size += size
        while len(self.rounds) > 1 and self.size > self.budget:
            _, dropped = self.rounds.popleft()
            self.size -= dropped
