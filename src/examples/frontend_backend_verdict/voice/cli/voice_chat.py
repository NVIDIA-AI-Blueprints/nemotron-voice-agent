# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Stand-alone OpenAI Realtime chat client (plan section 14).

It speaks the same wire protocol as tau3 and imports nothing from ``engine/`` or
``agent/``, so it can talk to this server, to the shared ``/v1/realtime``
gateway, or to any Realtime endpoint. Run from the repository root::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.voice.cli.voice_chat \
        --url ws://localhost:8765/v1/realtime --io text \
        --tools examples.frontend_backend_verdict.text.demo_tools:TOOLS

``--tools`` registers client-executed tools: the client runs each function call
locally and answers with ``function_call_output`` items plus one
``response.create``, which is exactly tau2's flow.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import sys
from pathlib import Path
from typing import Any

import websockets

from examples.frontend_backend_verdict.text.cli.ui import clock, rich_available
from examples.frontend_backend_verdict.voice.audio.formats import AudioFormat
from examples.frontend_backend_verdict.voice.cli.audio_io import AudioIO, MicSpeakerIO, TextIO, WavFileIO
from examples.frontend_backend_verdict.voice.cli.client_tools import ClientToolbox, load_tools

FORMATS = {"pcm24k": AudioFormat("audio/pcm", 24000), "pcmu": AudioFormat("audio/pcmu", 8000)}
_QUIET_EVENTS = frozenset({"response.output_audio.delta", "input_audio_buffer.append"})


class ChatView:
    """Presentation only: Rich when available and on a terminal, plain lines otherwise."""

    def __init__(self, *, kind: str = "auto", show_events: bool = False) -> None:
        """Pick the renderer."""
        self._show_events = show_events
        self._console: Any = None
        if kind == "rich" or (kind == "auto" and rich_available() and sys.stdout.isatty()):
            from rich.console import Console

            self._console = Console(highlight=False)

    def _line(self, label: str, text: str, style: str) -> None:
        import time

        stamp = clock(time.time())
        if self._console is not None:
            self._console.print(f"[dim]{stamp}[/dim] [{style}]{label:>6}[/{style}] {text}")
        else:
            print(f"{stamp} {label:>6} {text}", flush=True)

    def user(self, text: str) -> None:
        """What the server heard."""
        self._line("you", text, "bold green")

    def agent(self, text: str, *, status: str = "completed") -> None:
        """One agent utterance (item transcript)."""
        suffix = "" if status == "completed" else f"  [{status}]"
        self._line("agent", text + suffix, "bold cyan")

    def filler(self, text: str) -> None:
        """Filler the server generated but did not speak (``filler.mode: log_only``), in gray."""
        import time

        line = f"{clock(time.time())} {'filler':>6} {text}  (silent)"
        if self._console is not None:
            from rich.text import Text

            self._console.print(Text(line, style="grey50"))
        elif sys.stdout.isatty():
            print(f"\033[90m{line}\033[0m", flush=True)
        else:
            print(line, flush=True)

    def tool(self, text: str) -> None:
        """A client-executed tool call or output."""
        self._line("tool", text, "magenta")

    def info(self, text: str) -> None:
        """Informational line."""
        self._line("info", text, "dim")

    def error(self, text: str) -> None:
        """An error event."""
        self._line("error", text, "bold red")

    def event(self, event: dict[str, Any]) -> None:
        """Raw event type (``--show-events``)."""
        if self._show_events and event.get("type") not in _QUIET_EVENTS:
            self._line("event", str(event.get("type")), "dim")


class VoiceChatClient:
    """One Realtime session driven by an ``AudioIO`` (or typed text)."""

    def __init__(self, *, ws: Any, io: AudioIO, toolbox: ClientToolbox, view: ChatView, fmt: AudioFormat) -> None:
        """Bind the connection, I/O, tools and view."""
        self._ws = ws
        self._io = io
        self._toolbox = toolbox
        self._view = view
        self._fmt = fmt
        self._calls: list[tuple[str, str, str]] = []
        self._turn_done = asyncio.Event()
        self._closed = asyncio.Event()

    async def send(self, event: dict[str, Any]) -> None:
        """Send one client event."""
        await self._ws.send(json.dumps(event))

    async def configure(self, *, instructions: str, voice: str) -> None:
        """Send ``session.update`` and wait for ``session.updated``."""
        session: dict[str, Any] = {
            "type": "realtime",
            "instructions": instructions,
            "tools": self._toolbox.schemas(),
            "tool_choice": "auto",
            "output_modalities": ["audio"] if self._io.uses_audio else ["text"],
            "audio": {
                "input": {
                    "format": self._fmt.to_wire(),
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    }
                    if self._io.uses_audio
                    else None,
                },
                "output": {"format": self._fmt.to_wire(), "voice": voice},
            },
        }
        await self.send({"type": "session.update", "session": session})
        while True:
            event = json.loads(await self._ws.recv())
            self._view.event(event)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise SystemExit(f"session.update rejected: {event['error'].get('message')}")

    async def receive(self) -> None:
        """Render events, play audio, and run client tools."""
        transcripts: dict[str, str] = {}
        async for raw in self._ws:
            event = json.loads(raw)
            kind = event.get("type")
            self._view.event(event)
            if kind == "response.output_audio.delta":
                self._io.play(base64.b64decode(event["delta"]))
            elif kind in ("response.output_audio_transcript.delta", "response.output_text.delta"):
                transcripts[event["item_id"]] = transcripts.get(event["item_id"], "") + event["delta"]
            elif kind == "response.output_item.done" and event["item"].get("type") == "message":
                item = event["item"]
                self._view.agent(transcripts.pop(item["id"], ""), status=item.get("status", "completed"))
            elif kind == "x_nvidia.filler":
                self._view.filler(event.get("text", ""))
            elif kind == "conversation.item.input_audio_transcription.completed":
                self._view.user(event.get("transcript", ""))
            elif kind == "input_audio_buffer.speech_started":
                self._io.stop_playback()
            elif kind == "response.function_call_arguments.done":
                self._calls.append((event["call_id"], event["name"], event.get("arguments") or "{}"))
                self._view.tool(f"{event['name']}({event.get('arguments')})")
            elif kind == "response.done":
                await self._response_done(event["response"])
            elif kind == "error":
                self._view.error(f"{event['error'].get('code')}: {event['error'].get('message')}")
        self._closed.set()

    async def _response_done(self, response: dict[str, Any]) -> None:
        calls, self._calls = self._calls, []
        self._io.response_done(had_calls=bool(calls))
        if response.get("status") not in ("completed", None) and not calls:
            self._view.info(f"response {response.get('status')}")
        if not calls:
            self._turn_done.set()
            return
        for call_id, name, arguments in calls:
            output = await self._toolbox.execute(call_id, name, arguments)
            self._view.tool(f"{name} -> {output}")
            await self.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call_id, "output": output},
                }
            )
        await self.send({"type": "response.create"})

    async def stream_audio(self) -> None:
        """Stream input frames until the I/O ends."""
        async for frame in self._io.frames():
            await self.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(frame).decode("ascii")})

    async def type_loop(self) -> None:
        """Typed turns: ``/quit`` or end of input (Ctrl-D) exits."""
        while not self._closed.is_set():
            try:
                line = (await asyncio.to_thread(input, "you> ")).strip()
            except EOFError:
                return
            if line in ("/quit", "/exit"):
                return
            if not line:
                continue
            self._turn_done.clear()
            await self.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": line}]},
                }
            )
            await self.send({"type": "response.create"})
            await self._turn_done.wait()


def _build_io(args: argparse.Namespace, fmt: AudioFormat) -> AudioIO:
    if args.io == "text":
        return TextIO()
    if args.io == "wav":
        if not args.inputs:
            raise SystemExit("--io wav needs at least one --in file")
        return WavFileIO(args.inputs, fmt, out_path=args.out)
    return MicSpeakerIO(fmt)


async def run(args: argparse.Namespace) -> None:
    """Connect, configure and chat."""
    fmt = FORMATS["pcm24k"] if args.io == "mic" else FORMATS[args.format]
    toolbox = ClientToolbox(load_tools(args.tools) if args.tools else ())
    instructions = Path(args.instructions_file).read_text(encoding="utf-8") if args.instructions_file else ""
    view = ChatView(kind=args.ui, show_events=args.show_events)
    io = _build_io(args, fmt)
    url = f"{args.url}?model={args.model}"
    if not args.hide_filler:
        url += "&x_nvidia_filler=1"  # the server then reports unspoken filler (x_nvidia.filler)
    headers = {"Authorization": f"Bearer {args.token}"}
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        first = json.loads(await ws.recv())
        if first.get("type") != "session.created":
            raise SystemExit(f"expected session.created, got {first.get('type')}")
        view.info(f"connected: {first['session'].get('id')} ({url})")
        client = VoiceChatClient(ws=ws, io=io, toolbox=toolbox, view=view, fmt=fmt)
        await client.configure(instructions=instructions, voice=args.voice)
        receiver = asyncio.create_task(client.receive())
        try:
            if io.uses_audio:
                await client.stream_audio()
            else:
                await client.type_loop()
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver
            await io.close()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="ws://localhost:8765/v1/realtime")
    parser.add_argument("--model", default="pine-nemotron-fba", help="sent as ?model= (the server accepts any)")
    parser.add_argument("--token", default="unused", help="bearer token (server.require_bearer)")
    parser.add_argument("--io", choices=["mic", "wav", "text"], default="text")
    parser.add_argument("--in", dest="inputs", action="append", default=[], help="WAV turn (repeatable, --io wav)")
    parser.add_argument("--out", default="", help="write the agent's audio to this WAV (--io wav)")
    parser.add_argument("--format", choices=sorted(FORMATS), default="pcm24k", help="wire format for --io wav")
    parser.add_argument("--tools", default="", help="client-executed tools, module:attr -> list[ToolSpec]")
    parser.add_argument("--instructions-file", default="", help="sent as session.update.instructions")
    parser.add_argument("--voice", default="alloy")
    parser.add_argument("--ui", choices=["auto", "rich", "plain"], default="auto")
    parser.add_argument("--show-events", action="store_true", help="print every event type (not audio)")
    parser.add_argument("--hide-filler", action="store_true", help="do not show the unspoken filler text in gray")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run(_parse_args(argv)))


if __name__ == "__main__":
    main()
