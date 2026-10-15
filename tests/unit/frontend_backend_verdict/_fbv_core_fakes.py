# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Offline doubles for the FBV tests that need only ``core/`` (permanent).

Split from ``_fbv_voice_fakes.py`` (plan ``shared-stack-plan.md`` §8.4): config paths, the fake LLM
client, the fake clock, tau2-shaped payloads and trace normalization. The session harness for the
Pipecat adapter is ``_fbv_pipecat_fakes.py``.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
from collections.abc import Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from examples.frontend_backend_verdict.core.agent.llm import ChatResponse
from examples.frontend_backend_verdict.core.agent.messages import Message, ToolCall, Usage, canonical_json
from examples.frontend_backend_verdict.core.audio.formats import AudioFormat
from examples.frontend_backend_verdict.core.audio.pcm import silence, tone
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig, load_voice_config

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO_ROOT / "src" / "examples" / "frontend_backend_verdict"
CONFIG_DIR = PACKAGE_ROOT / "core" / "config"
BASE_CONFIG = CONFIG_DIR / "voice_agent.yaml"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
#: The agent's backend_only mode as a profile (a test profile: it is not served).
BACKEND_ONLY_PROFILE = FIXTURES / "profiles" / "backend_only.yaml"
PCMU = AudioFormat("audio/pcmu", 8000)
PCM16K = AudioFormat("audio/pcm", 16000)

_CONFIG_CACHE: dict[str, VoiceConfig] = {}


def base_config() -> VoiceConfig:
    if "base" not in _CONFIG_CACHE:
        _CONFIG_CACHE["base"] = load_voice_config(BASE_CONFIG)
    return _CONFIG_CACHE["base"]


def voice_config(**sections: Any) -> VoiceConfig:
    """The shipped base config with whole sections or fields replaced.

    ``filler_mode="speak"`` style keys are expanded as ``<section>_<field>``.
    """
    config = base_config()
    names = sorted((f.name for f in fields(config)), key=len, reverse=True)
    for key, value in sections.items():
        if key == "agent":
            config = replace(config, agent=value)
            continue
        section = next(name for name in names if key.startswith(f"{name}_"))
        name = key[len(section) + 1 :]
        current = getattr(config, section)
        config = replace(config, **{section: replace(current, **{name: value})})
    return config


# -- LLM fakes ------------------------------------------------------------------------


class FakeChatClient:
    """Scripted responses; an optional gate per call blocks it until set (tests of concurrency).

    ``failures`` maps a call index to an exception raised (after its gate) instead of a response.
    """

    def __init__(self, responses: Sequence[ChatResponse] = ()) -> None:
        self.responses: list[ChatResponse] = list(responses)
        self.gates: dict[int, asyncio.Event] = {}
        self.failures: dict[int, BaseException] = {}
        self.calls: list[dict[str, Any]] = []
        self.tool_choices: list[Any] = []

    def queue(self, *responses: ChatResponse) -> FakeChatClient:
        self.responses.extend(responses)
        return self

    async def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ):
        index = len(self.calls)
        self.calls.append({"messages": list(messages), "tools": list(tools or []), "tool_choice": tool_choice})
        self.tool_choices.append(tool_choice)
        gate = self.gates.get(index)
        if gate is not None:
            await gate.wait()
        if index in self.failures:
            raise self.failures[index]
        if not self.responses:
            raise AssertionError("FakeChatClient ran out of scripted responses")
        return self.responses.pop(0)


def text_response(text: str, tokens: int = 10) -> ChatResponse:
    return ChatResponse(
        content=text, usage=Usage(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=2 * tokens), cost=0.0
    )


def tool_response(*calls: tuple[str, dict[str, Any]], ids: Sequence[str] | None = None) -> ChatResponse:
    resolved = list(ids or [f"call_{index}" for index in range(len(calls))])
    return ChatResponse(
        content=None,
        tool_calls=tuple(
            ToolCall(id=resolved[index], name=name, arguments_json=canonical_json(arguments))
            for index, (name, arguments) in enumerate(calls)
        ),
        usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        cost=0.0,
    )


def delegate_response(query: str, filler: str = "Let me check.", *, task: str | None = None) -> ChatResponse:
    arguments = {"query": query}
    if filler:
        arguments["filler_text"] = filler
    if task is not None:
        arguments["task"] = task
    return tool_response(("call_backend", arguments), ids=["fcall_1"])


# -- clock / transport ----------------------------------------------------------------


class FakeClock:
    def __init__(self, start: float = 1_790_000_000.0) -> None:
        self.now = start
        self.mono = 1000.0

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.mono

    def advance(self, seconds: float) -> None:
        self.now += seconds
        self.mono += seconds


# -- tau2-shaped payloads ---------------------------------------------------------------


def tau2_session_update(domain: str = "mock") -> dict[str, Any]:
    return json.loads((FIXTURES / "tau2_session_update" / f"{domain}.json").read_text(encoding="utf-8"))


def pcmu_speech(ms: int, frequency: float = 300.0) -> bytes:
    return PCMU.encode(tone(ms, 8000, frequency=frequency))


def pcmu_silence(ms: int) -> bytes:
    return PCMU.encode(silence(ms, 8000))


def append_event(chunk: bytes) -> str:
    return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})


def tau2_update_with(tools: list[dict[str, Any]] | None = None, instructions: str = "policy") -> dict[str, Any]:
    update = copy.deepcopy(tau2_session_update("mock"))
    if tools is not None:
        update["session"]["tools"] = tools
    update["session"]["instructions"] = instructions
    return update


ID_KINDS = (("item_", "ITEM"), ("resp_", "RESP"), ("sess_", "SESS"), ("call_", "CALL"), ("fcall_", "FCALL"))
WALL_KEYS = frozenset({"event_id", "expires_at"})


def normalize_trace(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rule 9 normalization: drop ``event_id``, ordinal placeholders for generated ids, keep audio-clock fields."""
    mapping: dict[str, str] = {}
    counters: dict[str, int] = {}

    def norm(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: norm(item) for key, item in value.items() if key not in WALL_KEYS}
        if isinstance(value, list):
            return [norm(item) for item in value]
        if isinstance(value, str):
            for prefix, label in ID_KINDS:
                if value.startswith(prefix):
                    if value not in mapping:
                        counters[label] = counters.get(label, 0) + 1
                        mapping[value] = f"<{label}{counters[label]}>"
                    return mapping[value]
        return value

    return [norm(event) for event in events]
