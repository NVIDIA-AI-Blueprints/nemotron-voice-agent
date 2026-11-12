# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveKit agent worker for the generic assistant: Nemotron ASR, Nemotron LLM, and Magpie TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import yaml
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import nvidia, openai, silero

from .config import AgentConfig, load_config
from .tool_wrapper import traced_tool
from .tools import current_datetime, current_weather

log = logging.getLogger("generic-assistant-livekit")

EVENTS_TOPIC = "agent.events"  # data-channel topic the web UI listens on
PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")
PROMPT_KEY = "generic_assistant"

server = AgentServer()


def get_current_datetime(timezone: str = "") -> dict:
    """Get the current date and time.

    Args:
        timezone: Optional IANA timezone name such as "America/New_York". Leave empty for local time.
    """
    return current_datetime(timezone)


async def get_weather(location: str, unit: Literal["fahrenheit", "celsius"] = "fahrenheit") -> dict:
    """Get the current weather for a city. Reports Fahrenheit by default; both units are supported.

    Args:
        location: City name, for example "Paris" or "Austin, Texas".
        unit: "fahrenheit" (the default) or "celsius". Use "celsius" whenever the user asks for Celsius or metric.
    """
    return await current_weather(location, fahrenheit=(unit != "celsius"))


TOOLS = (get_current_datetime, get_weather)


def load_prompt() -> dict:
    """Return the `generic_assistant` entry of prompts.yaml."""
    return yaml.safe_load(PROMPTS_FILE.read_text())[PROMPT_KEY]


def build_stt(cfg: AgentConfig) -> nvidia.STT:
    """Build the streaming ASR client. Hosted endpoints use TLS, an API key, and low-latency endpointing."""
    common = dict(
        model=cfg.asr_model,
        function_id=cfg.asr_function_id,
        language_code=cfg.asr_language,
        inference_mode="streaming",
        server=cfg.asr_server,
        use_ssl=cfg.hosted,
    )
    if cfg.hosted:
        return nvidia.STT(
            api_key=os.environ["NVIDIA_API_KEY"], endpointing=nvidia.EndpointingConfig(mode="low_latency"), **common
        )
    # Endpointing runs in the self-hosted speech service, so the client leaves it unchanged.
    return nvidia.STT(api_key="", **common)


def build_tts(cfg: AgentConfig) -> nvidia.TTS:
    """Build the TTS client."""
    return nvidia.TTS(
        server=cfg.tts_server,
        use_ssl=cfg.hosted,
        voice=cfg.tts_voice,
        function_id=cfg.tts_function_id,
        language_code=cfg.tts_language,
        sample_rate=cfg.tts_sample_rate,
        inference_mode="online",
        api_key=os.environ["NVIDIA_API_KEY"] if cfg.hosted else "",
    )


def build_llm(cfg: AgentConfig) -> openai.LLM:
    """Build the OpenAI-compatible LLM client. Self-hosted servers do not check the key."""
    return openai.LLM(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=os.environ["NVIDIA_API_KEY"] if cfg.hosted else "local",
        extra_body=cfg.llm_extra_body,
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Join the room, wire the cascaded pipeline, and greet the user."""
    cfg = load_config()
    prompt = load_prompt()
    await ctx.connect()
    await ctx.wait_for_participant()

    def emit(kind: str, **data) -> None:
        payload = json.dumps({"type": kind, "ts": time.time(), **data}).encode()
        task = asyncio.ensure_future(
            ctx.room.local_participant.publish_data(payload, reliable=True, topic=EVENTS_TOPIC)
        )
        task.add_done_callback(
            lambda t: t.cancelled() or t.exception() and log.warning("event publish failed: %s", t.exception())
        )

    session = AgentSession(vad=silero.VAD.load(), stt=build_stt(cfg), llm=build_llm(cfg), tts=build_tts(cfg))

    @session.on("agent_state_changed")
    def _agent_state(ev):
        emit("agent_state", state=ev.new_state)

    @session.on("conversation_item_added")
    def _item(ev):
        # The reply was cut off because the user spoke over it: a confirmed barge-in.
        if getattr(ev.item, "role", None) == "assistant" and getattr(ev.item, "interrupted", False):
            emit("barge_in")

    @session.on("agent_false_interruption")
    def _false_interruption(ev):
        emit("false_interruption", resumed=ev.resumed)

    agent = Agent(instructions=prompt["content"], tools=[traced_tool(fn, emit) for fn in TOOLS])
    await session.start(agent=agent, room=ctx.room)
    emit("session", recipe=cfg.recipe, models=cfg.models())
    await session.generate_reply(instructions=prompt["greeting"])


if __name__ == "__main__":
    cli.run_app(server)
