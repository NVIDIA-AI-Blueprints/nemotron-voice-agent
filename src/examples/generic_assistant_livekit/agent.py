# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveKit agent worker for the generic assistant: Nemotron ASR, Nemotron LLM, and Magpie TTS."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Literal

import yaml
from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import nvidia, openai, silero

from .config import AgentConfig, load_config
from .credentials import apply_to_environ
from .tool_wrapper import traced_tool
from .tools import current_datetime, current_weather
from .voices import (
    VOICES_FILE,
    choose_voice,
    fetch_synthesis_params,
    parse_set_voice,
    parse_voices,
    with_current,
    write_voices_file,
)

log = logging.getLogger("generic-assistant-livekit")

EVENTS_TOPIC = "agent.events"  # data-channel topic the web UI listens on
CONTROL_TOPIC = "agent.control"  # data-channel topic the web UI sends commands on
PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")
PROMPT_KEY = "generic_assistant"

apply_to_environ()  # LiveKit key and secret from the state volume, unless set in the environment
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


class SwitchableTTS(nvidia.TTS):
    """The NVIDIA TTS client with a voice that can change during a session.

    The plugin copies its options each time it opens a stream, so a new voice applies from the next reply.
    """

    def set_voice(self, voice: str) -> None:
        """Use `voice` for the next synthesis request."""
        self._opts.voice = voice


def discover_voices(cfg: AgentConfig, quiet: bool = False) -> list[dict[str, str]]:
    """Return the voices the TTS server offers for the session language. Empty when the server cannot be asked."""
    try:
        params = fetch_synthesis_params(
            cfg.tts_server,
            use_ssl=cfg.hosted,
            api_key=os.environ.get("NVIDIA_API_KEY", "") if cfg.hosted else "",
            function_id=cfg.tts_function_id,
        )
        return parse_voices(params, cfg.tts_language)
    except Exception as exc:
        (log.debug if quiet else log.warning)("could not list TTS voices: %s", exc)
        return []


def build_tts(cfg: AgentConfig, voice: str | None = None) -> SwitchableTTS:
    """Build the TTS client. `voice` overrides the catalog default."""
    return SwitchableTTS(
        server=cfg.tts_server,
        use_ssl=cfg.hosted,
        voice=voice or cfg.tts_voice,
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


def publish_voices(cfg: AgentConfig, state_dir: Path, retry_s: float = 10, refresh_s: float = 300) -> None:
    """Keep the state volume's voice list current, so the UI can offer voices before a session starts.

    Runs in a thread of the worker process. It retries until the TTS server answers, because the server can still be
    loading models when the worker starts, and it removes a list left by another profile first.
    """
    path = state_dir / VOICES_FILE
    try:
        path.unlink(missing_ok=True)
        while True:
            voices = discover_voices(cfg, quiet=True)
            if voices:
                write_voices_file(path, with_current(voices, cfg.tts_voice), cfg.tts_voice)
                time.sleep(refresh_s)
            else:
                time.sleep(retry_s)
    except OSError as exc:
        log.warning("cannot publish the voice list to %s: %s", state_dir, exc)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Join the room, wire the cascaded pipeline, and greet the user."""
    cfg = load_config()
    prompt = load_prompt()
    voices_task = asyncio.ensure_future(asyncio.to_thread(discover_voices, cfg))
    await ctx.connect()
    participant = await ctx.wait_for_participant()

    def emit(kind: str, **data) -> None:
        payload = json.dumps({"type": kind, "ts": time.time(), **data}).encode()
        task = asyncio.ensure_future(
            ctx.room.local_participant.publish_data(payload, reliable=True, topic=EVENTS_TOPIC)
        )
        task.add_done_callback(
            lambda t: t.cancelled() or t.exception() and log.warning("event publish failed: %s", t.exception())
        )

    # The voice chosen in the UI before connecting arrives as a participant attribute. Only listed voices are used.
    voices = with_current(await voices_task, cfg.tts_voice)
    names = {voice["id"]: voice["name"] for voice in voices}
    chosen = choose_voice(participant.attributes.get("voice"), names, cfg.tts_voice)
    tts = build_tts(cfg, chosen)
    session = AgentSession(vad=silero.VAD.load(), stt=build_stt(cfg), llm=build_llm(cfg), tts=tts)

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
    emit("voices", voices=voices, current=tts._opts.voice)

    def on_data(packet: rtc.DataPacket) -> None:
        if packet.topic != CONTROL_TOPIC:
            return
        voice = parse_set_voice(packet.data, names)
        if voice is None:
            return
        tts.set_voice(voice)
        emit("voice_changed", voice=voice, name=names[voice])
        session.say(f"Hi, this is my {names[voice]} voice.", add_to_chat_ctx=False)

    ctx.room.on("data_received", on_data)
    await session.generate_reply(instructions=prompt["greeting"])


if __name__ == "__main__":
    # Only the main worker process runs this block, not the job processes that import the module.
    threading.Thread(
        target=publish_voices,
        args=(load_config(), Path(os.environ.get("LIVEKIT_STATE_DIR", "/state"))),
        daemon=True,
    ).start()
    cli.run_app(server)
