# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted ``AgentPort`` for the tau3 gates and offline smoke runs.

It never calls an LLM. By default (plan section 18.0, Gate B) turn 1 calls one
client tool and then answers, turn 2 answers, and turn 3 transfers the call by
calling ``transfer_to_human_agents`` when the client offered it. A JSON script
can replace the default; each step is ``{"say": "..."}`` or
``{"call": "<tool>", "arguments": {...}, "then_say": "..."}``.

It never delegates, so it has no ``in_flight`` turn and a barge-in review
(``barge_in.while_thinking: frontend_verdict``) never opens for it.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.frontend_backend_verdict.voice.agent.port import AgentReply, InFlight, OutgoingCall, Probe
from examples.frontend_backend_verdict.voice.errors import ProbeStateError

TRANSFER_TOOL = "transfer_to_human_agents"


def default_script(tool_names: Sequence[str], required: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Gate B script: a read-only-ish tool, an answer, then a transfer."""
    steps: list[dict[str, Any]] = []
    first = next((name for name in tool_names if name != TRANSFER_TOOL and not required.get(name)), None)
    if first is None:
        first = next((name for name in tool_names if name != TRANSFER_TOOL), None)
    if first is not None:
        arguments = {param: "unknown" for param in required.get(first, ())}
        steps.append({"call": first, "arguments": arguments, "then_say": "Thanks. I have looked that up for you."})
    else:
        steps.append({"say": "Thanks. How else can I help?"})
    steps.append({"say": "Understood. Is there anything else I can help you with today?"})
    if TRANSFER_TOOL in tool_names:
        steps.append(
            {"call": TRANSFER_TOOL, "arguments": {"summary": "scripted gate run"}, "then_say": "Transferring you now."}
        )
    else:
        steps.append({"say": "Goodbye."})
    return steps


class ScriptedAgentPort:
    """Deterministic agent that follows a script, one step per user turn."""

    def __init__(self, script: list[dict[str, Any]] | None = None, *, backend_only: bool = False) -> None:
        """Use ``script``, or build the default one from the session's tools on :meth:`configure`."""
        self._explicit = script is not None
        self._script = list(script or [])
        self._backend_only = backend_only
        self._turn = 0
        self._pending: dict[str, Any] | None = None
        self.inputs: list[str] = []
        self.outputs: list[dict[str, str]] = []
        self.history: list[tuple[str, str]] = []
        self.repairs: list[tuple[str, str]] = []

    @classmethod
    def from_file(cls, path: str | Path) -> ScriptedAgentPort:
        """Load a JSON list of steps."""
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def backend_only(self) -> bool:
        """Scripted agents have no frontend, hence no filler."""
        return self._backend_only

    def configure(self, *, tools: Sequence[Mapping[str, Any]], instructions: str) -> None:
        """Derive the default script from the offered tools."""
        if self._explicit:
            return
        names = [str(tool.get("name")) for tool in tools]
        required = {
            str(tool.get("name")): tuple((tool.get("parameters") or {}).get("required") or ()) for tool in tools
        }
        self._script = default_script(names, required)

    async def respond(self, text: str) -> AgentReply:
        """Advance one step."""
        self.inputs.append(text)
        self.history.append(("user", text))
        step = self._script[min(self._turn, len(self._script) - 1)] if self._script else {"say": "OK."}
        self._turn += 1
        if "call" in step:
            self._pending = step
            call = OutgoingCall(
                call_id=f"call_{uuid.uuid4().hex[:16]}",
                name=str(step["call"]),
                arguments=json.dumps(step.get("arguments") or {}, sort_keys=True),
            )
            return AgentReply(calls=(call,))
        text_out = str(step.get("say") or "OK.")
        self.history.append(("assistant", text_out))
        return AgentReply(text=text_out)

    async def resume(self, outputs: Mapping[str, str]) -> AgentReply:
        """Answer after the tool outputs arrive."""
        self.outputs.append(dict(outputs))
        step = self._pending or {}
        self._pending = None
        text_out = str(step.get("then_say") or "Done.")
        self.history.append(("assistant", text_out))
        return AgentReply(text=text_out)

    def repair_last_answer(self, full_text: str, replacement: str) -> None:
        """Record the repair and apply it to the scripted history."""
        self.repairs.append((full_text, replacement))
        for index in range(len(self.history) - 1, -1, -1):
            if self.history[index] == ("assistant", full_text):
                self.history[index] = ("assistant", replacement)
                return

    def seed_assistant(self, text: str) -> None:
        """Record a greeting."""
        self.history.append(("assistant", text))

    @property
    def in_flight(self) -> InFlight | None:
        """Never delegates."""
        return None

    async def probe(self, text: str, *, new_words: str, filler_spoken: bool) -> Probe:
        """Unreachable: a review needs a delegated turn."""
        raise ProbeStateError("the scripted agent never delegates, so there is nothing to probe")

    async def proceed(self, probe: Probe) -> AgentReply:
        """Unreachable: see :meth:`probe`."""
        raise ProbeStateError("the scripted agent never delegates, so there is nothing to proceed with")

    def begin_staging(self) -> None:
        """Nothing is ever staged."""

    def end_staging(self, *, commit: bool) -> None:
        """Nothing is ever staged."""
