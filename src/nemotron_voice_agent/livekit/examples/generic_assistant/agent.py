# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LiveKit agent worker: Nemotron ASR, Nemotron LLM, and Magpie TTS, with a use case and services chosen in the UI."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from pathlib import Path

from livekit import rtc
from livekit.agents import NOT_GIVEN, Agent, AgentServer, AgentSession, JobContext, ModelSettings, cli
from livekit.plugins import nvidia, openai, silero

from nemotron_voice_agent.core.paths import REPO_ROOT
from nemotron_voice_agent.core.session_audio import Recording, SessionAudio, write_wav
from nemotron_voice_agent.livekit.runtime.credentials import apply_to_environ

from .config import AgentConfig, load_config, service_options
from .errors import describe_error
from .scenarios import custom_scenario, get_scenario
from .service_state import (
    SERVICES_FILE,
    StatusProbe,
    probe_entry,
    read_service_state,
    voices_of,
    with_status,
    write_service_state,
)
from .session import read_choices
from .state import DEPLOYMENT_FILE, write_json_atomic
from .tool_wrapper import traced_tool
from .voices import (
    choose_voice,
    fetch_synthesis_params,
    parse_set_voice,
    parse_voices,
    preferred_voice,
    with_current,
)

log = logging.getLogger("generic-assistant-livekit")

EVENTS_TOPIC = "agent.events"  # data-channel topic the web UI listens on
CONTROL_TOPIC = "agent.control"  # data-channel topic the web UI sends commands on
HOSTED_VOICE_LIST_TIMEOUT_S = 30
LOCAL_VOICE_LIST_TIMEOUT_S = 5

apply_to_environ()  # LiveKit key and secret from the state volume, unless set in the environment
server = AgentServer()


def api_key(cfg: AgentConfig, slot: str) -> str:
    """Return the NVIDIA API key for a slot on NVIDIA Cloud. Self-hosted and custom endpoints never get it."""
    return os.environ["NVIDIA_API_KEY"] if cfg.slot_hosted(slot) else ""


def build_stt(cfg: AgentConfig) -> nvidia.STT:
    """Build the streaming ASR client. Hosted endpoints use TLS, an API key, and low-latency endpointing."""
    common = dict(
        model=cfg.asr_model,
        function_id=cfg.asr_function_id,
        language_code=cfg.asr_language,
        punctuate=cfg.asr_punctuate,
        inference_mode="streaming",
        server=cfg.asr_server,
        use_ssl=cfg.slot_hosted("asr"),
    )
    if cfg.slot_hosted("asr"):
        return nvidia.STT(
            api_key=api_key(cfg, "asr"), endpointing=nvidia.EndpointingConfig(mode="low_latency"), **common
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
    hosted = cfg.slot_hosted("tts")
    try:
        params = fetch_synthesis_params(
            cfg.tts_server,
            use_ssl=hosted,
            api_key=api_key(cfg, "tts"),
            function_id=cfg.tts_function_id,
            timeout=HOSTED_VOICE_LIST_TIMEOUT_S if hosted else LOCAL_VOICE_LIST_TIMEOUT_S,
        )
        return parse_voices(params, cfg.tts_language)
    except Exception as exc:
        (log.debug if quiet else log.warning)("could not list TTS voices: %s", exc)
        return []


def build_tts(cfg: AgentConfig, voice: str | None = None) -> SwitchableTTS:
    """Build the TTS client. `voice` overrides the catalog default."""
    tts = SwitchableTTS(
        server=cfg.tts_server,
        use_ssl=cfg.slot_hosted("tts"),
        voice=voice or cfg.tts_voice,
        function_id=cfg.tts_function_id,
        language_code=cfg.tts_language,
        sample_rate=cfg.tts_sample_rate,
        inference_mode="online",
        api_key=api_key(cfg, "tts"),
    )
    if not cfg.slot_hosted("tts"):
        # The plugin reads NVIDIA_API_KEY from the environment when api_key is empty, and sends it to any server.
        tts.nvidia_api_key = None
    return tts


def build_llm(cfg: AgentConfig) -> openai.LLM:
    """Build the OpenAI-compatible LLM client with the session's settings. Self-hosted servers do not check the key."""
    return openai.LLM(
        model=cfg.llm_model,
        base_url=cfg.llm_base_url,
        api_key=api_key(cfg, "llm") or "local",
        temperature=cfg.llm_options.get("temperature", NOT_GIVEN),
        top_p=cfg.llm_options.get("top_p", NOT_GIVEN),
        max_completion_tokens=cfg.llm_options.get("max_tokens", NOT_GIVEN),
        extra_body=cfg.llm_extra_body,
    )


def error_kind(source: object) -> str:
    """Name the pipeline stage (LLM, STT, or TTS) that an error event came from."""
    names = {cls.__name__ for cls in type(source).__mro__}
    return next((kind for kind in ("LLM", "STT", "TTS") if kind in names), type(source).__name__)


def service_snapshot(
    recipe: str,
    voices: dict[str, dict],
    probe: StatusProbe = probe_entry,
    discover: Callable[[AgentConfig], list[dict[str, str]]] = discover_voices,
) -> tuple[dict, dict[str, dict]]:
    """Return the offered services with their status, and the voices of each TTS entry that runs.

    `voices` holds the lists found before. Only a TTS entry that has none yet, and is not offline or starting, is asked.
    """
    services = with_status(service_options(recipe), probe)
    voices = dict(voices)
    for entry in services["tts"]:
        if entry["id"] in voices or entry.get("status") in ("offline", "starting"):
            continue
        tts = load_config(recipe, choices={"tts": {"key": entry["id"]}})
        if found := discover(tts):
            voices[entry["id"]] = {
                "voices": with_current(found, tts.tts_voice),
                "default": preferred_voice(found, tts.tts_voice),
            }
    return services, voices


def publish_services(recipe: str, state_dir: Path, status_s: float = 10, voices_s: float = 300) -> None:
    """Keep the state volume's service list current, so the UI can offer services and voices before a session starts.

    Runs in a thread of the worker process. The status of each self-hosted service is checked every `status_s`
    seconds, because a sidecar can still be loading. The voices of each TTS entry are asked again every `voices_s`
    seconds. A list left by another profile is removed first. A failed round is logged, and the next one runs.
    """
    path = state_dir / SERVICES_FILE
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        log.warning("cannot clear the old service list in %s: %s", state_dir, exc)
    voices: dict[str, dict] = {}
    asked_at = time.monotonic()
    while True:
        if time.monotonic() - asked_at > voices_s:
            voices, asked_at = {}, time.monotonic()
        try:
            services, voices = service_snapshot(recipe, voices, discover=lambda cfg: discover_voices(cfg, quiet=True))
            write_service_state(path, services, voices)
        except Exception as exc:
            log.warning("cannot publish the service list to %s: %s", state_dir, exc)
        time.sleep(status_s)


def session_config(attributes: dict[str, str]) -> tuple[AgentConfig, str]:
    """Return the session's configuration from the token attributes, and an error when a choice could not be used."""
    try:
        return load_config(choices=read_choices(attributes.get("services"))), ""
    except ValueError as exc:
        return load_config(), f"{str(exc).rstrip('.')}. The deployment defaults are used instead."


def missing_api_key(cfg: AgentConfig, env: Mapping[str, str] = os.environ) -> list[str]:
    """Return the slots that run on NVIDIA Cloud while `NVIDIA_API_KEY` is empty."""
    if env.get("NVIDIA_API_KEY", "").strip():
        return []
    return [slot for slot in ("llm", "asr", "tts") if cfg.slot_hosted(slot)]


def audio_dump_dir(env: Mapping[str, str] = os.environ) -> Path | None:
    """Return the folder for session recordings, or None when `ENABLE_AUDIO_DUMP` is not `true`."""
    if env.get("ENABLE_AUDIO_DUMP", "").strip().lower() != "true":
        return None
    path = Path(env.get("AUDIO_DUMP_PATH") or "audio_dumps")
    return path if path.is_absolute() else REPO_ROOT / path


def save_recording(source: Path, dump_dir: Path, name: str) -> Path | None:
    """Copy LiveKit's session recording out of the job's temporary folder, which is deleted when the job ends."""
    if not source.is_file():
        log.warning("no session recording at %s", source)
        return None
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        target = dump_dir / f"{name}.ogg"
        shutil.copyfile(source, target)
    except OSError as exc:
        log.warning("cannot save the session recording to %s: %s", dump_dir, exc)
        return None
    log.info("session audio saved: %s", target)
    return target


def save_streams(recordings: Mapping[str, Recording | None], dump_dir: Path, job_id: str) -> list[Path]:
    """Write each recorded stream to `<name>_<job_id>_session.wav`, and return the files written."""
    saved = []
    for name, recording in recordings.items():
        if recording is None:
            continue
        path = dump_dir / f"{name}_{job_id}_session.wav"
        try:
            dump_dir.mkdir(parents=True, exist_ok=True)
            write_wav(path, recording)
        except OSError as exc:
            log.warning("cannot save the session audio to %s: %s", path, exc)
            continue
        log.info("session audio saved: %s", path)
        saved.append(path)
    return saved


class RecordedAgent(Agent):
    """An agent that also records the audio that its STT received and its TTS produced, for the whole session."""

    def __init__(self, *, heard: SessionAudio, spoken: SessionAudio, **kwargs):
        """Record into ``heard`` and ``spoken``, and build the agent from ``kwargs``."""
        super().__init__(**kwargs)
        self.heard = heard
        self.spoken = spoken

    async def stt_node(self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings):
        """Record each frame before LiveKit's STT node receives it."""

        async def recorded() -> AsyncIterator[rtc.AudioFrame]:
            async for frame in audio:
                self.heard.add(bytes(frame.data), frame.sample_rate, frame.num_channels)
                yield frame

        async for event in Agent.default.stt_node(self, recorded(), model_settings):
            yield event

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
        """Record each frame that LiveKit's TTS node produces."""
        async for frame in Agent.default.tts_node(self, text, model_settings):
            self.spoken.add(bytes(frame.data), frame.sample_rate, frame.num_channels)
            yield frame


def describe_session(room: str, cfg: AgentConfig, scenario_name: str, custom_prompt: str, voice: str | None) -> str:
    """Return one log line with the services, settings, use case, and voice that a session runs with."""
    models = ", ".join(f"{model['kind']}={model.get('id') or model['name']}" for model in cfg.models())
    prompt = f"custom prompt ({len(custom_prompt)} characters)" if custom_prompt else f"use case {scenario_name!r}"
    return f"session {room}: {prompt}, {models}, voice={voice or '-'}, llm settings={cfg.llm_options or {}}"


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    """Join the room, wire the cascaded pipeline, and greet the user. A failure ends the session with its reason."""
    state_dir = Path(os.environ.get("LIVEKIT_STATE_DIR", "/state"))
    await ctx.connect()
    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError:
        log.info("room %s closed before the user joined", ctx.room.name)
        return

    def emit(kind: str, **data) -> asyncio.Future:
        payload = json.dumps({"type": kind, "ts": time.time(), **data}).encode()
        task = asyncio.ensure_future(
            ctx.room.local_participant.publish_data(payload, reliable=True, topic=EVENTS_TOPIC)
        )
        task.add_done_callback(
            lambda t: t.cancelled() or t.exception() and log.warning("event publish failed: %s", t.exception())
        )
        return task

    async def end_with_error(message: str) -> None:
        """Tell the UI why the session cannot go on, then close the room, so the page does not wait on silence."""
        log.error("session %s ended: %s", ctx.room.name, message)
        await asyncio.gather(
            emit("error", source="Session", message=message, status=None, recoverable=False), return_exceptions=True
        )
        try:
            await ctx.delete_room()
        finally:
            ctx.shutdown(message)

    try:
        cfg, config_error = session_config(participant.attributes)
        prompt = participant.attributes.get("prompt", "")
        scenario = custom_scenario(prompt) if prompt else get_scenario(participant.attributes.get("scenario"))
        # The voice list comes from the file that the worker publishes, so a session never waits on the TTS server.
        published = voices_of(read_service_state(state_dir / SERVICES_FILE), cfg.keys["tts"])
        voices = with_current(published["voices"], cfg.tts_voice) if cfg.tts_voice else published["voices"]
        names = {voice["id"]: voice["name"] for voice in voices}
        default_voice = published["default"] if published["default"] in names else cfg.tts_voice  # Sofia when listed
        chosen = choose_voice(participant.attributes.get("voice"), names, default_voice)
        if missing := missing_api_key(cfg):
            await end_with_error(
                f"NVIDIA_API_KEY is not set, and the {', '.join(missing).upper()} run on NVIDIA Cloud."
            )
            return
        tts = build_tts(cfg, chosen)
        session = AgentSession(vad=silero.VAD.load(), stt=build_stt(cfg), llm=build_llm(cfg), tts=tts)
        log.info("%s", describe_session(ctx.room.name, cfg, scenario.name, prompt, chosen))
        # Holds the task that ends the room, so it is not garbage collected.
        ending: set[asyncio.Future] = set()

        @session.on("close")
        def _close(ev):
            # Without this the agent idles in the room after the session ends, and the job is recorded as failed.
            if ev.error is not None:
                kind = ev.error.type.removesuffix("_error").upper()
                message = describe_error(ev.error)["message"]
                ending.add(asyncio.ensure_future(end_with_error(f"{kind} failed, so the session ended. {message}")))
            else:
                ctx.shutdown(f"session closed: {ev.reason.value}")

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
            # A failed LLM, ASR, or TTS call (for example a rate-limited model) would otherwise look like silence.
            emit("error", source=error_kind(getattr(ev, "source", None)), **describe_error(getattr(ev, "error", ev)))

        @session.on("agent_false_interruption")
        def _false_interruption(ev):
            emit("false_interruption", resumed=ev.resumed)

        started = False

        def on_data(packet: rtc.DataPacket) -> None:
            if packet.topic != CONTROL_TOPIC:
                return
            voice = parse_set_voice(packet.data, names)
            if voice is None:
                return
            tts.set_voice(voice)
            emit("voice_changed", voice=voice, name=names[voice])
            if started:  # a change that arrives while the session is still starting applies without a sample
                session.say(f"Hi, this is my {names[voice]} voice.", add_to_chat_ctx=False)

        ctx.room.on("data_received", on_data)  # registered before the session starts, so an early change is not lost
        agent_options = {
            "instructions": scenario.instructions,
            "tools": [traced_tool(fn, emit) for fn in scenario.make_tools()],  # fresh demo state for each session
        }
        dump_dir = audio_dump_dir()
        if dump_dir:
            agent = RecordedAgent(heard=SessionAudio(), spoken=SessionAudio(), **agent_options)

            async def _save_recording(_reason: str) -> None:
                source = ctx.session_directory / "audio.ogg"
                await asyncio.to_thread(save_recording, source, dump_dir, f"session_{ctx.job.id}")
                recordings = {"asr": agent.heard.finish(), "tts": agent.spoken.finish()}
                await asyncio.to_thread(save_streams, recordings, dump_dir, ctx.job.id)

            ctx.add_shutdown_callback(_save_recording)
            agent.heard.start()
            agent.spoken.start()
        else:
            agent = Agent(**agent_options)
        record = {"audio": True, "traces": False, "logs": False, "transcript": False} if dump_dir else NOT_GIVEN
        await session.start(agent=agent, room=ctx.room, record=record)
        started = True
        emit("session", recipe=cfg.recipe, models=cfg.models(), scenario=scenario.name)
        emit("voices", voices=voices, current=tts._opts.voice)
        if config_error:
            emit("error", source="Session", message=config_error, status=None, recoverable=False)

        if scenario.greeting_text:
            session.say(scenario.greeting_text)  # exact wording, and no model round trip before the first word
        else:
            await session.generate_reply(instructions=scenario.greeting)
    except Exception as exc:
        await end_with_error(f"The session could not start: {exc}")


def main() -> None:
    """Publish the deployment and services for the UI, then run the LiveKit agent worker CLI."""
    # Only the main worker process calls this, not the job processes that import the module.
    config = load_config()
    if missing := missing_api_key(config):
        log.error(
            "NVIDIA_API_KEY is not set, but the default %s run on NVIDIA Cloud. Every such session will fail.",
            ", ".join(missing).upper(),
        )
    state_dir = Path(os.environ.get("LIVEKIT_STATE_DIR", "/state"))
    try:
        write_json_atomic(state_dir / DEPLOYMENT_FILE, config.deployment())  # the UI shows the mode and models
    except OSError as exc:
        log.warning("cannot publish the deployment to %s: %s", state_dir, exc)
    threading.Thread(target=publish_services, args=(config.recipe, state_dir), daemon=True).start()
    cli.run_app(server)


if __name__ == "__main__":
    main()
