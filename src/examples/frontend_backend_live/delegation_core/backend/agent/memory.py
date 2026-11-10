# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent conductor's memory: its decision rounds, each kept or dropped whole, within a token budget."""

from __future__ import annotations

import json
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from examples.shared.ids import approx_tokens


@dataclass
class DecisionRound:
    """One ``decide()``'s exchange with the model: its input message, the answered calls, and their outputs."""

    message: dict
    calls: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    placed: bool = False

    def answered(self, call: dict, output: str) -> None:
        """Add a call the model made, and its output, to the round."""
        call_id, name, arguments = call.get("call_id", ""), call.get("name", ""), call.get("arguments") or "{}"
        self.calls.append({"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments})
        self.outputs.append({"type": "function_call_output", "call_id": call_id, "output": output})

    def items(self) -> list[dict]:
        """Return the round as input items: its message, its calls, then their outputs."""
        return [self.message, *self.calls, *self.outputs]


class ConductorMemory:
    """The conductor's earlier decision rounds, oldest first, within ``budget`` tokens (the newest always kept)."""

    def __init__(self, budget: int):
        """Start empty, with a budget of ``budget`` tokens."""
        self.budget = budget
        self.rounds: deque[tuple[list[dict], int]] = deque()
        self.size = 0

    def items(self) -> list[dict]:
        """Return a copy of every kept round's items, oldest first."""
        return [deepcopy(item) for items, _ in self.rounds for item in items]

    def add(self, decision_round: DecisionRound) -> None:
        """Keep ``decision_round``, then drop the oldest rounds while the memory is over its budget.

        A round leaves whole, since a chat request needs a tool message for every earlier tool call.
        """
        items = deepcopy(decision_round.items())
        size = approx_tokens(json.dumps(items, ensure_ascii=False))
        self.rounds.append((items, size))
        self.size += size
        while len(self.rounds) > 1 and self.size > self.budget:
            _, dropped = self.rounds.popleft()
            self.size -= dropped
