# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveKit agent worker: Nemotron ASR, Nemotron LLM, and Magpie TTS, with a use case chosen in the UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

from livekit import rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import nvidia, openai, silero

from .config import AgentConfig, load_config
from .credentials import apply_to_environ
from .errors import describe_error
from .scenarios import get_scenario
from .state import DEPLOYMENT_FILE, write_json_atomic
from .tool_wrapper import traced_tool
from .voices import (
    VOICES_FILE,
    choose_voice,
    fetch_synthesis_params,
    parse_set_voice,
    parse_voices,
    read_voices_file,
    with_current,
    write_voices_file,
)

log = logging.getLogger("generic-assistant-livekit")

EVENTS_TOPIC = "agent.events"  # data-channel topic the web UI listens on
CONTROL_TOPIC = "agent.control"  # data-channel topic the web UI sends commands on
HOSTED_VOICE_LIST_TIMEOUT_S = 30
LOCAL_VOICE_LIST_TIMEOUT_S = 5

apply_to_environ()  # LiveKit key and secret from the state volume, unless set in the environment
server = AgentServer()


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
    """Return the voices the TTS server offers for the session language. Empty when the server cannot be asked.

    A hosted endpoint can be cold, so it gets a longer timeout than a local sidecar.
    """
    try:
        params = fetch_synthesis_params(
            cfg.tts_server,
            use_ssl=cfg.hosted,
            api_key=os.environ.get("NVIDIA_API_KEY", "") if cfg.hosted else "",
            function_id=cfg.tts_function_id,
            timeout=HOSTED_VOICE_LIST_TIMEOUT_S if cfg.hosted else LOCAL_VOICE_LIST_TIMEOUT_S,
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


def error_kind(source: object) -> str:
    """Name the pipeline stage (LLM, STT, or TTS) that an error event came from."""
    names = {cls.__name__ for cls in type(source).__mro__}
    return next((kind for kind in ("LLM", "STT", "TTS") if kind in names), type(source).__name__)


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
    state_dir = Path(os.environ.get("LIVEKIT_STATE_DIR", "/state"))
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

    # The use case and voice chosen in the UI arrive as participant attributes. Unknown values fall back to defaults.
    scenario = get_scenario(participant.attributes.get("scenario"))
    # The voice list comes from the file that the worker publishes, so a session never waits on the TTS server.
    listed = read_voices_file(state_dir / VOICES_FILE)["voices"]
    voices = with_current(listed, cfg.tts_voice)
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

    @session.on("error")
    def _error(ev):
        # A failed LLM, ASR, or TTS call (for example a rate-limited hosted model) would otherwise look like silence.
        emit("error", source=error_kind(getattr(ev, "source", None)), **describe_error(getattr(ev, "error", ev)))

    @session.on("agent_false_interruption")
    def _false_interruption(ev):
        emit("false_interruption", resumed=ev.resumed)

    agent = Agent(
        instructions=scenario.instructions,
        tools=[traced_tool(fn, emit) for fn in scenario.make_tools()],  # fresh demo state for each session
    )
    await session.start(agent=agent, room=ctx.room)
    emit("session", recipe=cfg.recipe, models=cfg.models(), scenario=scenario.name)
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
    await session.generate_reply(instructions=scenario.greeting)


if __name__ == "__main__":
    # Only the main worker process runs this block, not the job processes that import the module.
    config = load_config()
    state_dir = Path(os.environ.get("LIVEKIT_STATE_DIR", "/state"))
    try:
        write_json_atomic(state_dir / DEPLOYMENT_FILE, config.deployment())  # the UI shows the mode and models
    except OSError as exc:
        log.warning("cannot publish the deployment to %s: %s", state_dir, exc)
    threading.Thread(target=publish_voices, args=(config, state_dir), daemon=True).start()
    cli.run_app(server)
