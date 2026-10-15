# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Golden scenarios, captures and normalization for the FBV restructure and the Pipecat adapter.

Plan: ``misc/frontend-backend-verdict/shared-stack-plan.md`` §9.1 (R0 checklist) and §10.1 (G0, G2).

Every scenario drives a harness through the same small interface (``start``, ``send``, ``speak``, ``feed``,
``idle``, ``wait_for``, ``of_type``, ``events``, ``runners``, ``close``), so the same scenarios run on v1's
session harness (G0) and later on the Pipecat adapter's harness (G2). Captures run with a pinned environment
and a ``FakeClock``; the normalization removes ids, wall-clock fields, durations and credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest import mock

from _fbv_core_fakes import (
    PCMU,
    REPO_ROOT,
    FakeChatClient,
    FakeClock,
    delegate_response,
    normalize_trace,
    pcmu_silence,
    pcmu_speech,
    tau2_session_update,
    text_response,
    tool_response,
    voice_config,
)
from _fbv_pipecat_fakes import ToneSynthesizer

from examples.frontend_backend_verdict.core.audio.resample import StreamResampler
from examples.frontend_backend_verdict.core.config.voice import VoiceConfig, load_voice_config
from examples.frontend_backend_verdict.core.engine.event_log import EventLog
from examples.frontend_backend_verdict.core.engine.runner import AgentClients
from examples.frontend_backend_verdict.core.engine.scripted import ScriptedAgentPort
from examples.frontend_backend_verdict.core.engine.write_gate import WriteGateSettings

GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "v1_golden"
TAU_DOMAINS = ("airline", "banking_knowledge", "mock", "retail", "telecom")

# -- pinned environment ---------------------------------------------------------------

#: Every variable the YAML configuration interpolates (text/config/agent.yaml, voice/config/voice_agent.yaml).
INTERPOLATED_ENV = (
    "NVIDIA_API_KEY",
    "FRONTEND_LLM_MODEL",
    "FRONTEND_LLM_BASE_URL",
    "BACKEND_LLM_MODEL",
    "BACKEND_LLM_BASE_URL",
    "FBA_BACKEND_HISTORY",
    "FBA_EVENT_LOG",
    "FBA_VOICE_PORT",
    "FBA_VOICE_TOKEN",
    "FBA_VOICE_PROMPTS",
    "FBA_ASR_SERVER",
    "FBA_TTS_SERVER",
    "FBA_FILLER_LOG",
    "FBA_VOICE_EVENT_LOG",
    "FBV_PROFILE",
    # Not interpolated, but read by the adapter: the deployment platform (speech remap, D2) and the
    # removed settings selector (must have no effect).
    "REALTIME_SERVICE_PLATFORM",
    "FBV_PIPECAT_SETTINGS",
    "UVICORN_WS_PING_INTERVAL",
    "UVICORN_WS_PING_TIMEOUT",
)
PLACEHOLDER_KEY = "fixture-placeholder"


@contextlib.contextmanager
def pinned_environment() -> Iterator[None]:
    """Clear every interpolated variable and pin the API key to a placeholder (plan §10.1 G0)."""
    cleared = {name: "" for name in INTERPOLATED_ENV}
    with mock.patch.dict(os.environ, cleared):
        for name in INTERPOLATED_ENV:
            os.environ.pop(name, None)
        os.environ["NVIDIA_API_KEY"] = PLACEHOLDER_KEY
        yield


# -- normalization --------------------------------------------------------------------

#: JSONL fields that come from a wall or monotonic clock (dropped at any depth).
CLOCK_KEYS = frozenset({"timestamp", "latency_ms", "session_start_wall", "held_ms", "wall"})
#: Record kinds whose ``*_ms`` fields are durations measured on a clock (``audio_ms`` is kept).
DURATION_KINDS = frozenset({"turn_latency", "filler_timing"})
SECRET_KEYS = frozenset({"api_key", "bearer_token"})
SECRET_MARKERS = ("nvapi-",)


def _strip(value: Any, *, durations: bool) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in CLOCK_KEYS:
                continue
            if durations and key.endswith("_ms") and key != "audio_ms":
                continue
            if key in SECRET_KEYS:
                out[key] = "<redacted>"
                continue
            out[key] = _strip(item, durations=durations)
        return out
    if isinstance(value, list):
        return [_strip(item, durations=durations) for item in value]
    return value


def normalize_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """G0 normalization of JSONL records: clock fields, durations and credentials removed, then ids."""
    stripped = [_strip(record, durations=record.get("kind") in DURATION_KINDS) for record in records]
    return normalize_trace(stripped)


def normalize_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """G0 normalization of server events (``normalize_trace`` plus credential redaction)."""
    return normalize_trace([_strip(event, durations=False) for event in events])


def assert_no_credentials(payload: Any) -> None:
    """Fail if a captured payload contains a credential-like value or a machine-specific absolute path."""
    text = json.dumps(payload)
    for marker in (*SECRET_MARKERS, str(REPO_ROOT), str(Path.home())):
        if marker in text:
            raise AssertionError(f"golden payload contains a credential or machine-specific value ({marker!r})")


def to_plain(value: Any) -> Any:
    """A JSON-safe view of dataclasses, mappings and paths (every path becomes ``<path>``)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [to_plain(item) for item in value]
        return sorted(items, key=json.dumps) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, str) and value.startswith("/") and ("/src/" in value or "/tests/" in value):
        return "<path>"
    if isinstance(value, str) and str(REPO_ROOT) in value:
        return value.replace(str(REPO_ROOT), "<repo>")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


# -- harness interface ----------------------------------------------------------------

MakeHarness = Callable[..., Any]


@dataclass
class ScenarioResult:
    events: list[dict[str, Any]]
    records: list[dict[str, Any]]
    filler: list[dict[str, Any]]


class _Run:
    """One scenario run: a fresh FakeClock and event-log file, closed and read afterwards."""

    def __init__(self, make: MakeHarness, tmp: Path) -> None:
        self._make = make
        self._log_path = tmp / "events.jsonl"
        self.harness: Any = None

    def harness_for(self, **kwargs: Any) -> Any:
        self.harness = self._make(clock=FakeClock(), event_log=EventLog(str(self._log_path)), **kwargs)
        return self.harness

    def records(self) -> list[dict[str, Any]]:
        if not self._log_path.exists():
            return []
        return [json.loads(line) for line in self._log_path.read_text(encoding="utf-8").splitlines()]

    async def until_record(self, kind: str, timeout: float = 3.0) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not any(record["kind"] == kind for record in self.records()):
            if loop.time() > deadline:
                raise AssertionError(f"no {kind} record")
            await asyncio.sleep(0.005)

    async def finish(self) -> ScenarioResult:
        harness = self.harness
        await harness.close()
        filler = list(getattr(harness, "filler_log", None).records) if hasattr(harness, "filler_log") else []
        return ScenarioResult(events=list(harness.events), records=self.records(), filler=filler)


async def _until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def _output(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def _clients() -> tuple[FakeChatClient, FakeChatClient, AgentClients]:
    frontend, backend = FakeChatClient(), FakeChatClient()
    return frontend, backend, AgentClients(backend=backend, frontend=frontend)


# -- scenarios (plan §10.1 G2) ------------------------------------------------------------


async def voice_turn(run: _Run) -> None:
    harness = run.harness_for(agent_factory=lambda _: ScriptedAgentPort([{"say": "Hello there. How can I help?"}]))
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.done")
    await harness.idle(500)


async def tool_round(run: _Run) -> None:
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Look up user 7."))
    backend.queue(tool_response(("get_users", {}), ids=["call_provider_42"]), text_response("User seven is Ann."))
    harness = run.harness_for(clients=clients)
    await harness.start(tau2_session_update("mock"))
    await harness.speak()
    await harness.wait_for("response.done")
    await harness.send(_output("call_provider_42", "Error: user not found"))
    await harness.send({"type": "response.create"})
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def parallel_tools(run: _Run) -> None:
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Do two things."))
    backend.queue(
        tool_response(("get_users", {}), ("create_task", {"title": "x"}), ids=["call_a", "call_b"]),
        text_response("Both done."),
    )
    harness = run.harness_for(clients=clients)
    await harness.start(tau2_session_update("mock"))
    await harness.speak()
    await harness.wait_for("response.done")
    await harness.send(_output("call_b", "B"))
    await harness.send({"type": "response.create"})
    await harness.send(_output("call_a", "A"))
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def filler_then_answer(run: _Run) -> None:
    gate = asyncio.Event()
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Check the thing.", filler="Let me check."))
    backend.queue(text_response("It is all set."))
    backend.gates[0] = gate
    harness = run.harness_for(config=voice_config(filler_mode="speak", filler_speak_after_ms=0), clients=clients)
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.output_audio.delta")
    gate.set()
    await harness.wait_for("response.done")
    await harness.idle(500)


def _verdict_config(**sections: Any) -> VoiceConfig:
    return voice_config(barge_in_while_thinking="frontend_verdict", **sections)


async def _running_turn(run: _Run, frontend: FakeChatClient, backend: FakeChatClient, clients: AgentClients) -> Any:
    gate = backend.gates.setdefault(0, asyncio.Event())
    harness = run.harness_for(config=_verdict_config(), clients=clients)
    await harness.start(tau2_session_update())
    await harness.speak()
    await _until(lambda: len(backend.calls) == 1)
    return harness, gate


async def review_continue(run: _Run) -> None:
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1151.", task="continue"))
    backend.queue(text_response("Order 1151 has shipped."))
    harness, gate = await _running_turn(run, frontend, backend, clients)
    await harness.speak()
    await run.until_record("barge_in_verdict")
    gate.set()
    await harness.wait_for("response.done")
    await harness.idle(500)


async def review_new(run: _Run) -> None:
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Check order 1151."), delegate_response("Check order 1152.", task="new"))
    backend.queue(text_response("Order 1152 is pending."))
    harness, _ = await _running_turn(run, frontend, backend, clients)
    await harness.speak()
    await harness.wait_for("response.done")
    await harness.idle(500)


LONG_ANSWER = (
    "This is a long answer. It has several sentences. Each one takes a while to say. The caller may interrupt."
)


def _holding() -> ToneSynthesizer:
    """Tone TTS that holds before its second sentence, so the first sentence is sent and the response stays open."""
    return ToneSynthesizer(sample_rate=16000, ms_per_char=10.0, hold_at=2)


async def barge_in_after_answer(run: _Run) -> None:
    harness = run.harness_for(
        agent_factory=lambda _: ScriptedAgentPort([{"say": LONG_ANSWER}, {"say": "Sure, go ahead."}])
    )
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.output_audio.delta")
    await harness.feed(pcmu_silence(300))
    await harness.speak()
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def barge_in_during_answer(run: _Run) -> None:
    harness = run.harness_for(
        agent_factory=lambda _: ScriptedAgentPort([{"say": LONG_ANSWER}, {"say": "Sure, go ahead."}]),
        synthesizer=_holding(),
    )
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.output_audio.delta")
    await harness.speak()
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def barge_in_filler(run: _Run) -> None:
    gate = asyncio.Event()
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Check the thing.", filler="Let me check that for you right now."))
    frontend.queue(delegate_response("Never mind, check the other thing.", filler=""))
    backend.queue(text_response("It is all set."), text_response("The other thing is fine."))
    backend.gates[0] = gate
    harness = run.harness_for(config=voice_config(filler_mode="speak", filler_speak_after_ms=0), clients=clients)
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.output_audio.delta")
    await harness.feed(pcmu_silence(200))
    await harness.speak()
    gate.set()
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def response_cancel(run: _Run) -> None:
    harness = run.harness_for(agent_factory=lambda _: ScriptedAgentPort([{"say": LONG_ANSWER}]), synthesizer=_holding())
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.output_audio.delta")
    await harness.send({"type": "response.cancel"})
    await harness.wait_for("response.done")
    await harness.idle(500)


async def truncate_known_and_unknown(run: _Run) -> None:
    harness = run.harness_for(agent_factory=lambda _: ScriptedAgentPort([{"say": "A short answer."}]))
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.done")
    item_id = harness.of_type("response.output_audio.delta")[0]["item_id"]
    await harness.send(
        {"type": "conversation.item.truncate", "item_id": item_id, "content_index": 0, "audio_end_ms": 999999}
    )
    await harness.wait_for("conversation.item.truncated")
    await harness.send(
        {"type": "conversation.item.truncate", "item_id": "item_unknown", "content_index": 0, "audio_end_ms": 100}
    )
    await harness.wait_for("error")
    await harness.idle(300)


async def agent_failure(run: _Run) -> None:
    frontend, backend, clients = _clients()
    frontend.failures[0] = RuntimeError("frontend unavailable")
    frontend.queue(delegate_response("unused"))
    harness = run.harness_for(clients=clients)
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("error")
    await harness.idle(500)


async def tts_failure(run: _Run) -> None:
    harness = run.harness_for(
        agent_factory=lambda _: ScriptedAgentPort([{"say": "This answer cannot be spoken."}]),
        synthesizer=ToneSynthesizer(sample_rate=16000, ms_per_char=10.0, fail="TTS endpoint unavailable"),
    )
    await harness.start(tau2_session_update())
    await harness.speak()
    await harness.wait_for("response.done")
    await harness.idle(500)


BAGS = {"reservation_id": "ZFA04Y", "total_baggages": 3, "nonfree_baggages": 1, "payment_id": "credit_card_7815826"}


def _bags_call(call_id: str) -> Any:
    return tool_response(("update_reservation_baggages", BAGS), ids=[call_id])


def _confirm(verdict: str) -> Any:
    arguments = {"query": "Add the bags to reservation ZFA04Y.", "filler_text": "One moment.", "confirmation": verdict}
    return tool_response(("call_backend", arguments), ids=["fcall_confirm"])


async def _write_gate_presented(run: _Run) -> tuple[Any, FakeChatClient, FakeChatClient]:
    frontend, backend, clients = _clients()
    frontend.queue(delegate_response("Add bags to reservation ZFA04Y."))
    backend.queue(_bags_call("C1"), text_response("Here is the change."))
    config = replace(voice_config(), write_gate=WriteGateSettings(enabled=True, exempt=("transfer_to_human_agents",)))
    harness = run.harness_for(config=config, clients=clients)
    await harness.start(tau2_session_update("airline"))
    await harness.speak()
    await harness.wait_for("response.done")
    return harness, frontend, backend


async def write_gate_heard(run: _Run) -> None:
    harness, frontend, backend = await _write_gate_presented(run)
    await harness.idle(4000)
    frontend.queue(_confirm("yes"))
    backend.queue(_bags_call("C2"), text_response("Done."))
    await harness.speak()
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


async def write_gate_cut(run: _Run) -> None:
    harness, frontend, backend = await _write_gate_presented(run)
    frontend.queue(delegate_response("Add the bags.", filler=""))
    backend.queue(text_response("Okay."))
    await harness.feed(pcmu_silence(300))
    await harness.feed(pcmu_speech(300) + pcmu_silence(700))
    await harness.wait_for("response.done", 2)
    await harness.idle(500)


SCENARIOS: dict[str, Callable[[_Run], Awaitable[None]]] = {
    "voice_turn": voice_turn,
    "tool_round": tool_round,
    "parallel_tools": parallel_tools,
    "filler_then_answer": filler_then_answer,
    "review_continue": review_continue,
    "review_new": review_new,
    "barge_in_after_answer": barge_in_after_answer,
    "barge_in_during_answer": barge_in_during_answer,
    "barge_in_filler": barge_in_filler,
    "response_cancel": response_cancel,
    "truncate_known_and_unknown": truncate_known_and_unknown,
    "agent_failure": agent_failure,
    "tts_failure": tts_failure,
    "write_gate_heard": write_gate_heard,
    "write_gate_cut": write_gate_cut,
}


async def run_scenario(name: str, make: MakeHarness) -> dict[str, Any]:
    """Run one scenario on a fresh harness and return its normalized capture."""
    with tempfile.TemporaryDirectory() as tmp:
        run = _Run(make, Path(tmp))
        await SCENARIOS[name](run)
        result = await run.finish()
    return {
        "events": normalize_events(result.events),
        "records": normalize_records(result.records),
        "filler": normalize_records(result.filler),
    }


# -- configuration captures (plan §10.1 G0, N4) -------------------------------------------


def merged_profile_configs(profiles: Mapping[str, Path]) -> dict[str, Any]:
    """The merged config of every served profile, with paths and credentials neutralized."""
    return {name: _strip(to_plain(load_voice_config(path)), durations=False) for name, path in profiles.items()}


async def configure_captures(make: MakeHarness, config: VoiceConfig) -> dict[str, Any]:
    """Per Tau fixture: the tool specs and resolved instructions given to the agent, and the rendered prompts."""
    out: dict[str, Any] = {}
    for domain in TAU_DOMAINS:
        _, _, clients = _clients()
        harness = make(config=config, clients=clients, clock=FakeClock())
        await harness.start(tau2_session_update(domain))
        runner = harness.runners[0]
        out[domain] = {
            "tool_specs": to_plain(runner.tool_specs),
            "resolved_instructions": to_plain(runner.resolved_instructions),
            "rendered_prompts": runner.rendered_prompts(),
        }
        await harness.close()
    return out


def pcm_capture() -> dict[str, str]:
    """Decoded and resampled PCM for a fixed μ-law sample (B6)."""
    pcmu = pcmu_speech(400) + pcmu_silence(200)
    pcm8k = PCMU.decode(pcmu)
    resampler = StreamResampler(8000, 16000)
    pcm16k = resampler.process(pcm8k) + resampler.flush()
    return {
        "pcm8k_sha256": hashlib.sha256(pcm8k).hexdigest(),
        "pcm16k_sha256": hashlib.sha256(pcm16k).hexdigest(),
    }
