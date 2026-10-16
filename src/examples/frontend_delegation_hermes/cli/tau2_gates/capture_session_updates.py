# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Capture tau2's exact ``session.update`` payload per domain (fixtures for the prompt tests).

Runs with the **tau2 checkout's** interpreter and imports only tau2 and the
stdlib. It builds each domain's environment, the audio-native system prompt and
tools exactly as tau2's ``DiscreteTimeAudioNativeAgent`` does, then calls tau2's
real ``OpenAIRealtimeProvider.configure_session`` against a capturing fake
socket, so the payload is byte-for-byte what tau2 would send::

    cd <tau2-bench checkout>
    SCRIPT=<voice-agent repo>/src/examples/frontend_delegation_hermes/cli/tau2_gates/capture_session_updates.py
    uv run python "$SCRIPT" \
        --out <voice-agent repo>/tests/unit/frontend_delegation_hermes/tau/fixtures/tau2_session_update
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path

DOMAINS = ("mock", "airline", "retail", "telecom", "banking_knowledge")


class _CaptureSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.state = 1  # websockets.State.OPEN

    async def send(self, text: str) -> None:
        self.sent.append(json.loads(text))

    async def recv(self) -> str:
        return json.dumps({"type": "session.updated", "session": {}})


async def _capture(domain: str) -> dict:
    from tau2.agent.discrete_time_audio_native_agent import (
        AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN,
        AUDIO_NATIVE_VOICE_INSTRUCTION,
    )
    from tau2.registry import registry
    from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider, OpenAIVADConfig

    environment = registry.get_env_constructor(domain)()
    prompt = AUDIO_NATIVE_SYSTEM_PROMPT_PLAIN.format(
        agent_instruction=AUDIO_NATIVE_VOICE_INSTRUCTION, domain_policy=environment.get_policy()
    )
    provider = OpenAIRealtimeProvider(model="pine-capture")
    socket = _CaptureSocket()
    provider.ws = socket
    type(provider).is_connected = property(lambda self: True)  # the fake socket is always open
    await provider.configure_session(
        system_prompt=prompt, tools=environment.get_tools(), vad_config=OpenAIVADConfig(), modality="audio"
    )
    return socket.sent[0]


def main() -> None:
    """Write ``<domain>.json`` per available domain plus ``PROVENANCE.json``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--domains", nargs="*", default=list(DOMAINS))
    args = parser.parse_args()
    os.environ.setdefault("PINE_REALTIME_BASE_URL", "ws://capture.invalid/v1/realtime")
    os.environ.setdefault("PINE_API_KEY", "capture")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    for domain in args.domains:
        try:
            payload = asyncio.run(_capture(domain))
        except Exception as exc:  # noqa: BLE001 - a missing optional domain is skipped, not fatal
            print(f"skip {domain}: {type(exc).__name__}: {exc}")
            continue
        (out / f"{domain}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        captured.append(domain)
        print(f"captured {domain}: {len(payload['session']['tools'])} tools")
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    provenance = {"source": "tau2 OpenAIRealtimeProvider.configure_session", "tau2_commit": commit, "domains": captured}
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
