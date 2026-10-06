# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Headless tau2-shaped client for smoke tests (plan section 14).

It behaves like tau3's audio-native OpenAI adapter: one GA ``session.update``
(μ-law 8 kHz, ``server_vad`` 0.5/300/500, flat tools, instructions), 160-byte
μ-law appends every 20 ms with **continuous silence**, optional wall-clock
pauses in the stream (tau2's user simulator blocks its tick loop), tool calls
answered from a JSON fixture with ``function_call_output`` + one
``response.create`` after the last output, and a ``truncate`` on barge-in. It
then checks the tau2 contract (plan section 3) on everything it received and
exits non-zero on a violation. Run from the repository root::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.voice.cli.tau2_replay \
        --url ws://localhost:8765/v1/realtime --synthetic-turns 3 --pause-s 2

With no ``--in`` WAVs it sends synthetic tone "utterances", which the energy
VAD of a ``--stub-speech`` server detects as speech.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import websockets

from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat
from examples.frontend_backend_verdict.voice.audio.pcm import tone
from examples.frontend_backend_verdict.voice.cli.audio_io import read_wav

PCMU = AudioFormat("audio/pcmu", 8000)
FRAME_BYTES = 160
SILENCE = PCMU.encode(b"\x00\x00" * FRAME_BYTES)
BETA_NAMES = frozenset(
    {
        "conversation.item.created",
        "response.audio.delta",
        "response.audio.done",
        "response.audio_transcript.delta",
        "response.audio_transcript.done",
        "response.text.delta",
        "response.text.done",
    }
)
DEFAULT_TOOLS = [
    {
        "type": "function",
        "name": "get_order",
        "description": "Look up one order by its id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "type": "function",
        "name": "transfer_to_human_agents",
        "description": "Transfer the user to a human agent.",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]},
    },
]


@dataclass(slots=True)
class ReplayReport:
    """What the replay saw, plus contract violations."""

    events: list[dict[str, Any]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    transcripts: dict[str, str] = field(default_factory=dict)
    user_transcripts: list[str] = field(default_factory=list)

    def violate(self, message: str) -> None:
        """Record one violation."""
        self.violations.append(message)


def check_contract(report: ReplayReport) -> None:
    """tau2 contract checks over the received events (plan section 3)."""
    events = report.events
    if not events or events[0].get("type") != "session.created":
        report.violate("first server frame is not session.created")
    created = sum(1 for e in events if e.get("type") == "response.created")
    done = sum(1 for e in events if e.get("type") == "response.done")
    if created != done:
        report.violate(f"{created} response.created vs {done} response.done")
    last_start = -1
    audio_seen: set[str] = set()
    transcript_seen: set[str] = set()
    for event in events:
        kind = event.get("type")
        if kind in BETA_NAMES:
            report.violate(f"beta event emitted: {kind}")
        if kind == "input_audio_buffer.speech_started":
            start = event.get("audio_start_ms")
            if not isinstance(start, int) or start < last_start:
                report.violate(f"audio_start_ms not monotonic on the input clock: {start} after {last_start}")
            last_start = start if isinstance(start, int) else last_start
        if kind == "response.output_audio_transcript.delta":
            transcript_seen.add(event.get("item_id", ""))
        if kind == "response.output_audio.delta":
            item_id = event.get("item_id")
            if not item_id:
                report.violate("output_audio.delta without item_id")
            elif item_id not in transcript_seen and item_id not in audio_seen:
                report.violate(f"audio for {item_id} arrived before any transcript delta")
            audio_seen.add(item_id or "")
        if kind == "response.function_call_arguments.done":
            for key in ("call_id", "name", "arguments"):
                if not event.get(key):
                    report.violate(f"function_call_arguments.done without top-level {key}")
        if kind == "response.done" and not isinstance(event.get("response", {}).get("usage"), dict):
            report.violate("response.done without usage")
    item_ids = [e["item"]["id"] for e in events if e.get("type") == "response.output_item.added"]
    if len(item_ids) != len(set(item_ids)):
        report.violate("an output item id was reused")


class Tau2Replay:
    """Drives one session the way tau2 does."""

    def __init__(self, ws: Any, *, tool_outputs: dict[str, str], pause_every_ms: int, pause_s: float) -> None:
        """Bind the socket and the tool fixture."""
        self._ws = ws
        self._tool_outputs = tool_outputs
        self._pause_every_ms = pause_every_ms
        self._pause_s = pause_s
        self.report = ReplayReport()
        self._pending_calls: list[tuple[str, str, str]] = []
        self._answered = asyncio.Event()
        self._last_audio_item: str | None = None
        self._stream_ms = 0

    async def send(self, event: dict[str, Any]) -> None:
        """Send one client event."""
        await self._ws.send(json.dumps(event))

    async def handshake(self, instructions: str, tools: list[dict[str, Any]]) -> None:
        """Receive session.created, send tau2's session.update, wait for session.updated."""
        first = json.loads(await self._ws.recv())
        self.report.events.append(first)
        session = {
            "type": "realtime",
            "instructions": instructions,
            "output_modalities": ["audio"],
            "tools": tools,
            "tool_choice": "auto",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
                "output": {"format": {"type": "audio/pcmu"}, "voice": "alloy"},
            },
        }
        await self.send({"type": "session.update", "session": session})
        while True:
            event = json.loads(await self._ws.recv())
            self.report.events.append(event)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(f"session configuration failed: {event['error'].get('message')}")

    async def receive(self) -> None:
        """Record events; answer tools; truncate on barge-in."""
        async for raw in self._ws:
            event = json.loads(raw)
            self.report.events.append(event)
            kind = event.get("type")
            if kind == "response.output_audio.delta":
                self._last_audio_item = event.get("item_id")
            elif kind == "response.output_audio_transcript.delta":
                item_id = event.get("item_id", "")
                self.report.transcripts[item_id] = self.report.transcripts.get(item_id, "") + event.get("delta", "")
            elif kind == "conversation.item.input_audio_transcription.completed":
                self.report.user_transcripts.append(event.get("transcript", ""))
            elif kind == "response.function_call_arguments.done":
                call = (event["call_id"], event["name"], event.get("arguments") or "{}")
                self._pending_calls.append(call)
                self.report.tool_calls.append(call)
            elif kind == "input_audio_buffer.speech_started" and self._last_audio_item:
                await self.send(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": self._last_audio_item,
                        "content_index": 0,
                        "audio_end_ms": 999999,  # tau2's value is unreliable; the server must clamp it
                    }
                )
            elif kind == "response.done":
                calls, self._pending_calls = self._pending_calls, []
                for call_id, name, _ in calls:
                    output = self._tool_outputs.get(name, f"Error: no fixture output for tool {name}")
                    await self.send(
                        {
                            "type": "conversation.item.create",
                            "item": {"type": "function_call_output", "call_id": call_id, "output": output},
                        }
                    )
                if calls:
                    await self.send({"type": "response.create"})
                else:
                    self._answered.set()

    async def _append(self, chunk: bytes) -> None:
        await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})
        self._stream_ms += 20
        if self._pause_every_ms and self._stream_ms % self._pause_every_ms == 0:
            await asyncio.sleep(self._pause_s)  # wall-clock gap: no audio is sent meanwhile
        else:
            await asyncio.sleep(0.02)

    async def stream(self, turns: list[bytes], *, wait_timeout_s: float) -> None:
        """Each turn, then continuous silence until the agent answered (or the timeout)."""
        for turn in turns:
            self._answered.clear()
            for offset in range(0, len(turn), FRAME_BYTES):
                chunk = turn[offset : offset + FRAME_BYTES]
                await self._append(chunk + SILENCE[len(chunk) :])
            deadline = time.monotonic() + wait_timeout_s
            while not self._answered.is_set() and time.monotonic() < deadline:
                await self._append(SILENCE)
            for _ in range(50):  # one second of silence between turns
                await self._append(SILENCE)


def synthetic_turns(count: int, *, speech_ms: int = 1200) -> list[bytes]:
    """Tone bursts encoded as μ-law 8 kHz (detected as speech by the energy VAD)."""
    return [PCMU.encode(tone(speech_ms, 8000, frequency=300.0 + 40.0 * index)) for index in range(count)]


async def run(args: argparse.Namespace) -> ReplayReport:
    """Replay one session and check the contract."""
    tools = json.loads(Path(args.tools).read_text(encoding="utf-8")) if args.tools else DEFAULT_TOOLS
    outputs = json.loads(Path(args.tool_outputs).read_text(encoding="utf-8")) if args.tool_outputs else {}
    turns = [read_wav(path, PCMU) for path in args.inputs] or synthetic_turns(args.synthetic_turns)
    async with websockets.connect(
        f"{args.url}?model={args.model}", additional_headers={"Authorization": "Bearer unused"}, max_size=None
    ) as ws:
        replay = Tau2Replay(ws, tool_outputs=outputs, pause_every_ms=args.pause_every_ms, pause_s=args.pause_s)
        await replay.handshake(args.instructions, tools)
        receiver = asyncio.create_task(replay.receive())
        await replay.stream(turns, wait_timeout_s=args.turn_timeout_s)
        receiver.cancel()
    check_contract(replay.report)
    return replay.report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8765/v1/realtime")
    parser.add_argument("--model", default="pine-replay")
    parser.add_argument("--in", dest="inputs", action="append", default=[], help="WAV user turn (repeatable)")
    parser.add_argument("--synthetic-turns", type=int, default=3, help="tone turns when no --in is given")
    parser.add_argument("--tools", default="", help="JSON list of flat Realtime tools (default: two demo tools)")
    parser.add_argument("--tool-outputs", default="", help="JSON {tool_name: output string}")
    parser.add_argument("--instructions", default="You are a helpful customer service agent.")
    parser.add_argument("--pause-every-ms", type=int, default=1000, help="insert a wall-clock pause every N ms")
    parser.add_argument("--pause-s", type=float, default=0.0, help="length of each wall-clock pause")
    parser.add_argument("--turn-timeout-s", type=float, default=60.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point: prints a summary; exit code 1 on any contract violation."""
    report = asyncio.run(run(_parse_args(argv)))
    types: dict[str, int] = {}
    for event in report.events:
        types[event.get("type", "?")] = types.get(event.get("type", "?"), 0) + 1
    print(json.dumps({"event_counts": types, "tool_calls": report.tool_calls}, indent=2))
    print("user transcripts:", report.user_transcripts)
    for item_id, text in report.transcripts.items():
        print(f"agent {item_id}: {text}")
    if report.violations:
        print("CONTRACT VIOLATIONS:", *report.violations, sep="\n  ")
        sys.exit(1)
    print("tau2 contract: OK")


if __name__ == "__main__":
    main()
