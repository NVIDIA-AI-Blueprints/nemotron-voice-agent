# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D105, D107

"""A scripted realtime-protocol server: the remote model a realtime frontend connects to."""

import asyncio
import base64
import json

from websockets.asyncio.server import serve

TONE_100MS = base64.b64encode(b"\x10\x10" * 2400).decode()


class FakeRealtimeServer:
    """Accepts one connection, records the client's events, and speaks the realtime protocol on request."""

    def __init__(self, *, answer_session_update=True, session_error=None, reply="Okay.", strict=False):
        self.events: list[dict] = []
        self.answer_session_update = answer_session_update
        self.session_error = session_error
        self.reply = reply  # what the model says whenever it is asked for a response; None for silence
        self.strict = strict  # refuse a response.create while a response is active, as the real server does
        self.active = False
        self.refused = 0
        self._tasks: set[asyncio.Task] = set()
        self.headers: dict = {}
        self.path = ""
        self.ws = None
        self._arrived = asyncio.Event()
        self._connected = asyncio.Event()
        self._response = 0
        self._server = None
        self.port = 0

    async def __aenter__(self):
        self._server = await serve(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc):
        self._server.close()
        await self._server.wait_closed()

    @property
    def url(self):
        return f"ws://127.0.0.1:{self.port}/v1/realtime"

    async def _handle(self, ws):
        self.ws, self.path = ws, ws.request.path
        self.headers = ws.request.headers
        self._connected.set()
        try:
            async for raw in ws:
                event = json.loads(raw)
                self.events.append(event)
                self._arrived.set()
                if event["type"] == "response.create" and self.strict and self.active:
                    self.refused += 1
                    code = "conversation_already_has_active_response"
                    await self.send({"type": "error", "error": {"code": code, "message": "A response is active."}})
                elif event["type"] == "response.create" and self.reply:
                    self.active = True
                    task = asyncio.create_task(self.speak(self.reply))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                if event["type"] == "session.update":
                    if self.session_error:
                        await self.send({"type": "error", "error": self.session_error})
                    elif self.answer_session_update:
                        await self.send({"type": "session.updated", "session": event["session"]})
        except Exception:  # noqa: BLE001 - the client hanging up ends the loop
            pass

    async def send(self, event):
        await self.ws.send(json.dumps(event))

    async def connected(self):
        async with asyncio.timeout(10):
            await self._connected.wait()

    async def wait_for(self, kind, *, where=None, after=0, timeout=10.0):
        async with asyncio.timeout(timeout):
            while True:
                for event in self.events[after:]:
                    if event["type"] == kind and (where is None or where(event)):
                        return event
                self._arrived.clear()
                await self._arrived.wait()

    def kinds(self):
        return [e["type"] for e in self.events]

    def items(self, role=None):
        found = [e["item"] for e in self.events if e["type"] == "conversation.item.create"]
        return [i for i in found if role is None or i.get("role") == role]

    # ---------------------------------------------------------------- scripted model behavior
    async def hear(self, text):
        """The caller speaks and the server transcribes it."""
        await self.send({"type": "input_audio_buffer.speech_started"})
        await self.send({"type": "input_audio_buffer.speech_stopped"})
        await self.send({"type": "conversation.item.input_audio_transcription.completed", "transcript": text})

    async def start_response(self):
        self._response += 1
        self.active = True
        response_id = f"resp_{self._response}"
        await self.send({"type": "response.created", "response": {"id": response_id}})
        return response_id

    async def speak(self, text, *, finish=True, response_id=None, pieces=3):
        """The model speaks: audio, then the transcript in sub-word pieces."""
        response_id = response_id or await self.start_response()
        await self.send({"type": "response.output_audio.delta", "response_id": response_id, "delta": TONE_100MS})
        step = max(len(text) // pieces, 1)
        for start in range(0, len(text), step):
            await self.send(
                {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": response_id,
                    "delta": text[start : start + step],
                }
            )
        if finish:
            await self.finish(text, response_id)
        return response_id

    async def finish(self, text, response_id):
        await self.send(
            {"type": "response.output_audio_transcript.done", "response_id": response_id, "transcript": text}
        )
        await self.send({"type": "response.output_audio.done", "response_id": response_id})
        self.active = False
        await self.send({"type": "response.done", "response": {"id": response_id, "status": "completed"}})

    async def call(self, name, arguments, call_id="call_1", *, lead_in=None):
        """The model calls a function (after an optional spoken lead-in) and ends its response."""
        response_id = await self.speak(lead_in, finish=False) if lead_in else await self.start_response()
        if lead_in:
            await self.send(
                {"type": "response.output_audio_transcript.done", "response_id": response_id, "transcript": lead_in}
            )
            await self.send({"type": "response.output_audio.done", "response_id": response_id})
        item = {"type": "function_call", "name": name, "call_id": call_id, "arguments": json.dumps(arguments)}
        await self.send({"type": "response.output_item.done", "response_id": response_id, "item": item})
        self.active = False
        await self.send({"type": "response.done", "response": {"id": response_id, "status": "completed"}})
        return response_id
