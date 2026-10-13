# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""tau3 Gate A (plan section 18.0): tau2's real provider against this server.

Runs with the **tau2 checkout's** interpreter and imports only tau2 and the
stdlib. It calls tau2's real ``OpenAIRealtimeProvider.connect()`` and
``configure_session()`` with the ``mock`` domain's tools and prompt, streams a
μ-law tone "utterance" followed by silence, parses every event with tau2's own
``parse_realtime_event``, answers the function call like tau2 does, and checks:
no exception, ``session.id`` read, no ``error`` event, the function call parsed
with ``call_id``/``name``/``arguments``, and a scored transcript received.

Server side (voice-agent repo root)::

    PYTHONPATH=src uv run python -m examples.frontend_delegation_hermes.tau_server \
        --config src/examples/frontend_delegation_hermes/config/profiles/stub_gate.yaml \
        --stub-speech --stub-backend --port 8765

Client side (tau2 checkout)::

    PINE_REALTIME_BASE_URL=ws://localhost:8765/v1/realtime PINE_API_KEY=unused \
        uv run python <voice-agent repo>/src/examples/frontend_delegation_hermes/cli/tau2_gates/gate_a_handshake.py
"""

from __future__ import annotations

import asyncio
import math
import sys


def _ulaw_tone(ms: int, rate: int = 8000, frequency: float = 300.0) -> bytes:
    """μ-law tone (G.711, same algorithm as the server's tables)."""
    out = bytearray()
    for index in range(rate * ms // 1000):
        sample = int(0.3 * 32767 * math.sin(2 * math.pi * frequency * index / rate))
        value = sample >> 2
        mask = 0xFF
        if value < 0:
            value, mask = -value, 0x7F
        value = min(value, 8159) + 0x21
        seg = next(
            (i for i, end in enumerate((0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF)) if value <= end), 8
        )
        out.append((0x7F ^ mask) if seg >= 8 else (((seg << 4) | ((value >> (seg + 1)) & 0x0F)) ^ mask))
    return bytes(out)


async def main() -> int:
    """Run the gate; returns the process exit code."""
    from tau2.agent.discrete_time_audio_native_agent import (
        AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN,
        AUDIO_NATIVE_VOICE_INSTRUCTION,
    )
    from tau2.registry import registry
    from tau2.voice.audio_native.openai.events import (
        AudioTranscriptDeltaEvent,
        ErrorEvent,
        FunctionCallArgumentsDoneEvent,
        ResponseDoneEvent,
        TimeoutEvent,
        UnknownEvent,
    )
    from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider, OpenAIVADConfig

    environment = registry.get_env_constructor("mock")()
    prompt = AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN.format(
        agent_instruction=AUDIO_NATIVE_VOICE_INSTRUCTION, domain_policy=environment.get_policy()
    )
    provider = OpenAIRealtimeProvider(model="pine-gate-a")
    await provider.connect()
    print(f"connected: session.created read (session id {getattr(provider, 'session_id', '?')})")
    await provider.configure_session(
        system_prompt=prompt, tools=environment.get_tools(), vad_config=OpenAIVADConfig(), modality="audio"
    )
    print("session.updated received")

    problems: list[str] = []
    calls: list = []
    transcript: list[str] = []
    unknown: set[str] = set()
    audio = _ulaw_tone(1200) + b"\xff" * 8000 * 2
    responses_done = 0

    async def pump() -> None:
        nonlocal responses_done
        async for event in provider.receive_events():
            if isinstance(event, TimeoutEvent):
                continue
            if isinstance(event, ErrorEvent):
                problems.append(f"error event: {event.code} {event.message}")
            elif isinstance(event, UnknownEvent):
                unknown.add(event.type)
            elif isinstance(event, FunctionCallArgumentsDoneEvent):
                calls.append(event)
            elif isinstance(event, AudioTranscriptDeltaEvent):
                transcript.append(event.delta)
            elif isinstance(event, ResponseDoneEvent):
                responses_done += 1
                if calls and responses_done == 1:
                    for call in calls:
                        await provider.send_tool_result(call.call_id, '{"ok": true}', request_response=False)
                    await provider.ws.send('{"type": "response.create"}')

    receiver = asyncio.create_task(pump())
    for offset in range(0, len(audio), 160):
        chunk = audio[offset : offset + 160]
        await provider.send_audio(chunk + b"\xff" * (160 - len(chunk)))
        await asyncio.sleep(0.02)
    for _ in range(150):  # 3 s of silence while the agent answers
        await provider.send_audio(b"\xff" * 160)
        await asyncio.sleep(0.02)
    receiver.cancel()
    await provider.disconnect()

    if not calls:
        problems.append("no function call was parsed")
    for call in calls:
        if not (call.call_id and call.name):
            problems.append(f"function call missing call_id/name: {call}")
        print(f"tool call parsed: {call.name}({call.arguments}) call_id={call.call_id}")
    if not "".join(transcript).strip():
        problems.append("no scored transcript (response.output_audio_transcript.delta) received")
    print(f"agent transcript: {''.join(transcript)!r}")
    print(f"events tau2 does not model (informational): {sorted(unknown)}")
    if problems:
        print("GATE A FAILED:", *problems, sep="\n  ")
        return 1
    print("GATE A PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
