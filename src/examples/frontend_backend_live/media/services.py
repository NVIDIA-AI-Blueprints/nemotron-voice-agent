# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The cascade's speech services: streaming speech-to-text in, text-to-speech out, and how a line is spoken.

``speech.tts_mode`` names one of :data:`SPEECH_MODES`, the two ways the text-to-speech can take a line. Each keeps a
line's text behind its audio in its own way, so that whatever follows the audio output sees a sentence's text only once
its audio has played (the playout facts, ``media/playout.py``):

* :class:`PerSentence` (the default) sends one request per sentence. Pipecat yields a sentence's audio before its text,
  so nothing needs holding.
* :class:`Stitched` sends a line as one stitched stream, which sounds smoother across sentences, but Pipecat queues
  each sentence's text before the audio arrives. :class:`LineTextHold` therefore holds a line's text until all its
  audio has been produced: a stitched line that is cut reports none of its text as heard.

A host that builds the text-to-speech itself (the repository UI bot, from the catalog) gets the mode the service
already runs in from :func:`speech_mode`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, ClassVar, Protocol

from loguru import logger
from pipecat.frames.frames import CancelFrame, EndFrame, InterruptionFrame, TTSStoppedFrame, TTSTextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor, FrameProcessorSetup
from pipecat.services.nvidia.stt import NvidiaSTTSettings
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings

from examples.frontend_backend_live.media.turns import TurnEnd
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.nvidia_force_eou_stt import NvidiaForceEouSTTService
from utils import NVCF_GRPC_SERVER, is_nvcf, load_ipa_dictionary, nvidia_api_key


def speech_server(entry: Mapping[str, Any]) -> str:
    """The gRPC address of a speech service's catalog entry: its ``server``, or NVIDIA's cloud when it names none."""
    return str(entry.get("server") or NVCF_GRPC_SERVER)


def build_stt(body: dict, default_asr: dict, turn_end: TurnEnd) -> NvidiaForceEouSTTService:
    """Build the streaming speech-to-text service from the catalog entry and body overrides.

    ``turn_end`` builds it, so that it finalizes what it heard when that turn end needs it (``media/turns.py``).
    """
    server = body.get("asr_server", "") or speech_server(default_asr)
    kwargs: dict = {"api_key": nvidia_api_key(), "server": server, "use_ssl": is_nvcf(server)}
    function_id = body.get("asr_function_id", "") or default_asr.get("function_id", "")
    model = body.get("asr_model", "") or default_asr.get("model", "")
    language = body.get("asr_language_code", "") or default_asr.get("language_code", "")
    punctuation = str(body.get("asr_automatic_punctuation", "true")).lower() != "false"
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model or "custom-asr"}
    kwargs["settings"] = NvidiaSTTSettings(automatic_punctuation=punctuation)
    if language:
        kwargs["settings"].language = language
    logger.info(f"ASR: server={server}, function_id={function_id or '(default)'}, language={language or '(default)'}")
    return turn_end.stt(**kwargs, stop_history=400)


class OffLoopSetupTTS(NvidiaTTSService):
    """The NVIDIA text-to-speech, its client, synthesis config and zero-shot prompt loaded in a worker thread.

    ``NvidiaTTSService.setup`` opens its client and fetches the server's synthesis config with blocking gRPC calls. On
    the event loop, a slow server holds every session of the process while one starts, so sessions that start together
    wait in line and can miss their start deadline. In a thread, a slow server delays only its own session.

    A live session loads them before its pipeline starts (:func:`load_speech`), so its audio begins only when it can
    speak.
    """

    async def load(self) -> None:
        """Load the client, the synthesis config and the zero-shot prompt in a worker thread, unless they are loaded.

        A load that fails closes the client it opened. A load given up (its start cancelled or timed out) runs on in its
        thread, which nothing can stop, and closes what it opened once it ends.
        """
        if self._service is not None:
            return
        loading = asyncio.ensure_future(asyncio.to_thread(self._load))
        try:
            await asyncio.shield(loading)
        except asyncio.CancelledError:
            loading.add_done_callback(self._loaded_late)
            raise

    def _loaded_late(self, loading: asyncio.Future) -> None:
        """A load given up ended in its thread: close what it opened, and read its failure, if any."""
        if not loading.cancelled():
            loading.exception()
        self._close_client()

    def _load(self) -> None:
        try:
            self._initialize_client()
            self._config = self._create_synthesis_config()
            self._load_zero_shot_audio_prompt()
        except Exception:
            self._close_client()
            raise
        logger.debug(f"Initialized NvidiaTTSService with model: {self._settings.model}")

    async def setup(self, setup: FrameProcessorSetup):
        """The parent's processor setup, then :meth:`load`."""
        await super(NvidiaTTSService, self).setup(setup)
        await self.load()


async def load_speech(tts: Any) -> None:
    """Load what a text-to-speech needs before a session starts (:meth:`OffLoopSetupTTS.load`); others need nothing."""
    load = getattr(tts, "load", None)
    if load is not None:
        await load()


def build_tts(body: dict, default_tts: dict) -> NvidiaTTSService:
    """Build the text-to-speech service from the catalog entry and body overrides.

    The service takes the pipeline's output rate when it starts: a live session's is the shared
    ``PIPELINE_AUDIO_OUT_SAMPLE_RATE``, which every Magpie deployment accepts.
    """
    server = body.get("tts_server", "") or speech_server(default_tts)
    voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    mode = body.get("tts_synthesis_mode", "") or default_tts.get("synthesis_mode", "")
    raw_function_id = body.get("tts_function_id")
    function_id = str(raw_function_id) if raw_function_id is not None else default_tts.get("function_id", "")
    model = body.get("tts_model", "") or default_tts.get("model", "")
    zero_shot = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get("zero_shot_audio_prompt_file", "")
    settings_kwargs: dict = {"voice": voice}
    if mode:
        settings_kwargs["synthesis_mode"] = mode
    kwargs: dict = {
        "api_key": nvidia_api_key(),
        "server": server,
        "settings": NvidiaTTSSettings(**settings_kwargs),
        "use_ssl": is_nvcf(server),
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": load_ipa_dictionary(),
        # A line with no audio (a rate-limited request, say) is the cascade's to retry or write off
        # (speech.start_timeout_seconds, speech.retry_silent_lines): it costs that line, never the call.
        "max_consecutive_zero_audio_contexts": 0,
    }
    if function_id or model:
        kwargs["model_function_map"] = {"function_id": function_id, "model_name": model}
    if zero_shot:
        kwargs["zero_shot_audio_prompt_file"] = zero_shot
    logger.info(
        f"TTS: server={server}, voice={voice}, model={model or '(default)'}, synthesis_mode={mode or '(default)'}"
    )
    return OffLoopSetupTTS(**kwargs)


# ------------------------------------------------------------------------------------------- the TTS modes
class SpeechMode(Protocol):
    """How the text-to-speech takes a line, and what keeps the line's text behind its audio (``speech.tts_mode``)."""

    synthesis_mode: ClassVar[str]  # the NVIDIA text-to-speech synthesis mode, set through the session body
    reports_sentences: ClassVar[bool]  # whether each sentence's text follows its own audio, so playout reports it

    def text_hold(self) -> list[FrameProcessor]:
        """New processors to place right after the text-to-speech (none when its text already follows its audio)."""


class PerSentence:
    """One synthesis request per sentence: Pipecat yields a sentence's audio, then its text, so nothing is held."""

    synthesis_mode: ClassVar[str] = "per_sentence"
    reports_sentences: ClassVar[bool] = True

    def text_hold(self) -> list[FrameProcessor]:
        """Nothing: each sentence's text already follows its audio."""
        return []


class Stitched:
    """One stitched stream per line: smoother across sentences, with each sentence's text queued before its audio."""

    synthesis_mode: ClassVar[str] = "stitched"
    reports_sentences: ClassVar[bool] = False

    def text_hold(self) -> list[FrameProcessor]:
        """A :class:`LineTextHold`, so a line's text follows all of its audio."""
        return [LineTextHold()]


SPEECH_MODES: dict[str, type[SpeechMode]] = {"per_sentence": PerSentence, "stitched": Stitched}


def speech_mode(tts: Any) -> SpeechMode:
    """The TTS mode a text-to-speech service already runs in: the one its synthesis mode setting names.

    Magpie's synthesis mode defaults to ``stitched`` (``settings.yaml``), whose text comes before its audio, so assuming
    :class:`PerSentence` there would drop every sentence from the transcript. A service that names no mode of the
    table, or has no such setting, is taken as :class:`PerSentence`, the NVIDIA service's own default.
    """
    synthesis_mode = getattr(getattr(tts, "settings", None), "synthesis_mode", None)
    return SPEECH_MODES.get(str(synthesis_mode), PerSentence)()


class LineTextHold(FrameProcessor):
    """Holds each line's text until the line's audio has all been produced, then lets it follow that audio.

    A stitched text-to-speech queues a sentence's text before its audio arrives, and its audio carries no sentence
    boundaries. Holding the line's text until the line ends (``TTSStoppedFrame``) keeps every text frame behind the
    audio it describes. A cut (``InterruptionFrame``) drops what is held, with the audio the output drops.
    """

    def __init__(self, **kwargs):
        """Start with nothing held."""
        super().__init__(**kwargs)
        self._held: list[TTSTextFrame] = []

    async def process_frame(self, frame, direction: FrameDirection) -> None:
        """Hold a line's text frames; release them just before the line ends; drop them on a cut."""
        await super().process_frame(frame, direction)
        if direction is FrameDirection.DOWNSTREAM:
            if isinstance(frame, TTSTextFrame):
                self._held.append(frame)
                return
            if isinstance(frame, (InterruptionFrame, CancelFrame)):
                self._held.clear()
            elif isinstance(frame, (TTSStoppedFrame, EndFrame)):
                held, self._held = self._held, []
                for text in held:
                    await self.push_frame(text, direction)
        await self.push_frame(frame, direction)
