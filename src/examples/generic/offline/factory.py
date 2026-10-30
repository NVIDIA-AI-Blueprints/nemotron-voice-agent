# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the generic assistant's catalogs to independently scheduled providers."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.services.nvidia.llm import NvidiaLLMSettings
from pipecat.services.nvidia.tts import NvidiaTTSSettings, NvidiaTTSSynthesisMode

from examples.shared.evaluation_trace import _observe, create_evaluation_trace
from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.nvidia_stt_settings import resolve_stt_stop_history_ms
from examples.shared.nvidia_tts_explicit_eof import (
    EXPLICIT_EOF_COMPLETION_POLICY,
    TTS_PROVIDER_IDLE_TIMEOUT_SECONDS,
    explicit_eof_tts_options,
)
from examples.shared.pipeline_utils import (
    PIPELINE_AUDIO_OUT_SAMPLE_RATE,
    build_context_messages,
)
from examples.shared.speaker import speaker_diarization_enabled
from utils import (
    is_nvcf,
    is_streaming_llm_url,
    load_ipa_dictionary,
    load_prompt_catalog,
    load_service_entry,
    load_tools_catalog,
    normalize_lang_code,
    nvidia_api_key,
    nvidia_llm_api_key,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
    resolve_tools_available,
)

from .detection import create_generic_detector
from .native_response import (
    CapturingNativeNvidiaLLMService,
    CapturingNativeNvidiaTTSService,
    NativeResponseSession,
)
from .providers import create_riva_recognizer


class GenericOfflineRuntime:
    """Own provider resources and expose a sanitized evaluation configuration."""

    def __init__(
        self,
        engine,
        client,
        recognizer,
        synthesis,
        configuration,
        *,
        provider_observations: Callable[[], dict[str, Any]] | None = None,
        evaluation_trace=None,
    ):
        """Keep session-owned provider resources out of the scheduling core."""
        self._engine = engine
        self._client = client
        self._recognizer = recognizer
        self._synthesis = synthesis
        self._provider_observations = provider_observations
        self._evaluation_trace = evaluation_trace
        self.configuration = configuration
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._released: set[str] = set()

    async def aclose(self):
        """Attempt every release and retry only operations that did not finish."""
        async with self._close_lock:
            if self._closed:
                return
            errors = []

            async def release(name, operation, *, asynchronous=False):
                if name in self._released:
                    return
                try:
                    if asynchronous:
                        await operation()
                    else:
                        operation()
                except BaseException as error:
                    errors.append(error)
                else:
                    self._released.add(name)

            await release("recognizer_cancel", self._recognizer.cancel)
            # Legacy finite-tape responders expose cancel(); the persistent
            # native session owns cancellation and cleanup through aclose().
            if hasattr(self._synthesis, "cancel"):
                await release("synthesis_cancel", self._synthesis.cancel)
            if hasattr(self._synthesis, "aclose"):
                await release("synthesis_close", self._synthesis.aclose, asynchronous=True)
            if self._client is not None:
                await release("client_close", self._client.close, asynchronous=True)
            if self._evaluation_trace is not None:
                await release(
                    "trace_close",
                    lambda: asyncio.to_thread(_observe, self._evaluation_trace, "close"),
                    asynchronous=True,
                )
            self._closed = not errors
            if errors:
                if len(errors) > 1:
                    errors[0].add_note(
                        "Further runtime cleanup failures: " + ", ".join(type(e).__name__ for e in errors[1:])
                    )
                raise errors[0]

    async def run(self, source, *, event_sink: Callable[[dict], None] | None = None):
        """Run once and release every provider connection on success or failure."""
        if self._evaluation_trace:
            _observe(self._evaluation_trace, "offline_source", source)
        failure = None
        try:
            result = (
                await self._engine.run(source)
                if event_sink is None
                else await self._engine.run(source, event_sink=event_sink)
            )
        except BaseException as error:
            failure = error
            raise
        finally:
            try:
                await self.aclose()
            except BaseException as error:
                if failure is None:
                    raise
                failure.add_note(f"Runtime cleanup also failed: {type(error).__name__}")
        # Cleanup can emit delayed native errors after the engine has completed.
        # Collect only after it drains. Optional diagnostics must never replace
        # the result or mask an execution/cleanup exception.
        if self._provider_observations is not None:
            with suppress(Exception):
                result = replace(
                    result,
                    metrics={**result.metrics, "native_provider_observations": self._provider_observations()},
                )
        return result


def build_runtime(config: dict) -> GenericOfflineRuntime:
    """Use the same generic prompt, service entries, text filter, and tool schemas."""
    from examples.generic.tool_handlers import TOOL_HANDLERS

    from .incremental import IncrementalSessionEngine

    module_file = Path(__file__).resolve().parents[1] / "pipeline.py"
    prompt_key, prompt = resolve_prompt(module_file, config.get("prompt_content", ""), config.get("prompt_key", ""))
    prompt_entry = load_prompt_catalog(module_file).get(prompt_key)
    prompt_entry = prompt_entry if isinstance(prompt_entry, dict) else {}
    if prompt_entry.get("multi_speaker_support") is True:
        raise ValueError("Offline execution currently does not support multi-speaker prompts")
    asr = load_service_entry("asr", "")
    llm = load_service_entry("llm", "")
    tts = load_service_entry("tts", "")
    if speaker_diarization_enabled(
        config,
        default_asr=asr,
        asr_model=config.get("asr_model") or asr.get("model", ""),
        asr_server=config.get("asr_server") or asr.get("server", "grpc.nvcf.nvidia.com:443"),
    ):
        raise ValueError("Offline execution currently does not support speaker diarization")
    base_url = config.get("base_url") or llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    llm_api_key = nvidia_llm_api_key(config.get("llm_id", ""), base_url=base_url)
    settings = {
        "api_key": nvidia_api_key(),
        "asr_server": config.get("asr_server") or asr.get("server", "grpc.nvcf.nvidia.com:443"),
        "asr_function_id": config.get("asr_function_id") or asr.get("function_id", ""),
        "asr_model": config.get("asr_model") or asr.get("model", ""),
        "asr_stop_history": resolve_stt_stop_history_ms(),
        "asr_language_code": config.get("asr_language_code") or asr.get("language_code", "en-US"),
        "asr_automatic_punctuation": str(
            config.get("asr_automatic_punctuation", asr.get("automatic_punctuation", "true"))
        ).lower()
        != "false",
        "tts_server": config.get("tts_server") or tts.get("server", "grpc.nvcf.nvidia.com:443"),
        "tts_function_id": (
            str(config["tts_function_id"]) if config.get("tts_function_id") is not None else tts.get("function_id", "")
        ),
        "tts_voice_id": config.get("tts_voice_id") or tts.get("voice_id", "Magpie-Multilingual.EN-US.Aria"),
        "tts_language_code": normalize_lang_code(
            config.get("tts_language_code") or tts.get("language_code") or "en-US"
        ),
        "tts_sample_rate": PIPELINE_AUDIO_OUT_SAMPLE_RATE,
        "custom_dictionary": load_ipa_dictionary(),
    }
    # Resolve the same request/catalog precedence as live; the pinned native
    # service defaults to per-sentence synthesis when both omit the setting.
    synthesis_mode = (
        config.get("tts_synthesis_mode") or tts.get("synthesis_mode") or NvidiaTTSSynthesisMode.PER_SENTENCE
    )
    tts_model = config.get("tts_model") or tts.get("model", "")
    if (
        synthesis_mode != NvidiaTTSSynthesisMode.STITCHED
        or "zeroshot" in tts_model.lower()
        or config.get("tts_zero_shot_audio_prompt_file")
    ):
        raise ValueError("Offline execution currently supports stitched Magpie synthesis without zero-shot prompts")
    explicit_tts_function = settings["tts_function_id"]
    if is_nvcf(settings["tts_server"]) and not explicit_tts_function and not tts_model:
        # Match NvidiaTTSService's default model_function_map in the generic pipeline.
        settings["tts_function_id"] = "877104f7-e885-42b9-8de8-f6e4c6303969"
    model = config.get("model_id") or llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    if is_streaming_llm_url(base_url):
        raise ValueError("Offline execution requires an HTTP chat-completions LLM endpoint; disable streaming input")
    extra_params = parse_json_dict(config.get("extra_params") or llm.get("extra_params", ""), label="extra_params")
    extra = {}
    llm_settings = NvidiaLLMSettings(model=model)
    for key, convert in (("temperature", float), ("max_tokens", int)):
        value = config.get(key)
        if key == "temperature" and value in ("", None):
            value = prompt_entry.get("temperature")
        if value in ("", None):
            value = llm.get(key)
        if value not in ("", None):
            try:
                converted = convert(value)
            except (TypeError, ValueError):
                logger.warning(f"Ignoring invalid {key}={value!r}")
            else:
                extra[key] = converted
                setattr(llm_settings, key, converted)
    # Pipecat's live chat request applies extra parameters after scalar settings.
    extra.update(extra_params)
    if extra_params:
        llm_settings.extra = extra_params
    system_prompt = config.get("system_prompt") or llm.get("system_prompt", "")
    names = resolve_tools_available(module_file, prompt_key)
    catalog = load_tools_catalog(module_file)
    tools = []
    handlers = {}
    for name in names:
        entry = catalog.get(name)
        if (
            name in TOOL_HANDLERS
            and isinstance(entry, dict)
            and isinstance(entry.get("function"), dict)
            and entry["function"].get("name") == name
        ):
            tools.append(entry)
            handlers[name] = TOOL_HANDLERS[name]
    if tools:
        raise ValueError("Incremental offline execution currently requires a prompt without tools")
    detector = create_generic_detector()
    if detector.configuration["policy"] != "silero":
        raise ValueError("Chunk-driven offline replay currently requires USE_SILERO_VAD_TURN_DETECTION=true")
    recognizer = create_riva_recognizer(settings, capture_events=True)
    messages = build_context_messages(prompt, system_prompt)
    recent_turns = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)
    evaluation_trace = create_evaluation_trace({**config, "protocol": "offline"})

    def llm_factory(capture):
        service = CapturingNativeNvidiaLLMService(
            capture=capture,
            api_key=llm_api_key,
            base_url=base_url,
            settings=llm_settings,
            **({"evaluation_trace": evaluation_trace} if evaluation_trace else {}),
        )
        if evaluation_trace:
            _observe(evaluation_trace, "bind_processor", service, role="llm")
        return service

    def tts_factory(capture):
        model_map = {}
        if explicit_tts_function or tts_model:
            model_map["model_function_map"] = {"function_id": explicit_tts_function, "model_name": tts_model}
        return CapturingNativeNvidiaTTSService(
            capture=capture,
            api_key=settings["api_key"],
            server=settings["tts_server"],
            use_ssl=is_nvcf(settings["tts_server"]),
            settings=NvidiaTTSSettings(
                voice=settings["tts_voice_id"],
                language=settings["tts_language_code"],
                synthesis_mode=synthesis_mode,
            ),
            text_filters=[NemotronSpeechTextFilter()],
            custom_dictionary=settings["custom_dictionary"],
            **explicit_eof_tts_options(synthesis_mode),
            **model_map,
        )

    responder = NativeResponseSession(llm_factory, tts_factory, sample_rate=settings["tts_sample_rate"])
    configuration = {
        "mode": "offline",
        "sample_rate": 16000,
        "pipeline": "generic-assistant",
        "asr_mode": "continuous_stream",
        "asr_availability_policy": "serial_chunk_progress_with_incremental_observed_processing",
        "timing_basis": "observed_provider_elapsed_including_transport_and_queueing",
        "automatic_introduction": False,
        "prompt_key": prompt_key,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "model_id": model,
        "llm_extra_params": extra,
        "detector": detector.configuration,
        "asr_language_code": settings["asr_language_code"],
        "asr_model": settings["asr_model"],
        "asr_automatic_punctuation": settings["asr_automatic_punctuation"],
        "asr_stop_history_ms": settings["asr_stop_history"],
        "asr_force_eou": False,
        "asr_word_timestamps": False,
        "asr_completeness_barrier": "finite_response_stream_eof",
        "asr_delivery_mode": "incremental_callback_actor",
        "provider_execution": "overlapping_continuous_ASR_and_persistent_native_LLM_TTS",
        "llm_service": "pipecat.NvidiaLLMService",
        "tts_service": "pipecat.NvidiaTTSService",
        "tts_context_completion_policy": EXPLICIT_EOF_COMPLETION_POLICY,
        "tts_provider_idle_timeout_seconds": TTS_PROVIDER_IDLE_TIMEOUT_SECONDS,
        "pronunciation_dictionary_sha256": hashlib.sha256(
            json.dumps(settings["custom_dictionary"] or {}, sort_keys=True).encode()
        ).hexdigest(),
        "base_url": base_url,
        "asr_server": settings["asr_server"],
        "tts_server": settings["tts_server"],
        "tts_voice_id": settings["tts_voice_id"],
        "tts_language_code": settings["tts_language_code"],
        "tts_synthesis_mode": synthesis_mode,
        "tts_sample_rate": settings["tts_sample_rate"],
        "output_sample_rate": 16000,
        "tts_resampling": "native_live_chunker_then_cumulative_soxr_vhq_recording",
        "asr_function_id": settings["asr_function_id"],
        "tts_function_id": settings["tts_function_id"],
        "tool_names": sorted(handlers),
        "tool_choice": config.get("tool_choice") or "auto",
        "interrupted_history": "actual_TTSTextFrame_order_through_live_assistant_aggregator",
        "turn_policy": "actual_Pipecat_user_aggregator_with_sample_clock_waits",
        "output_chunks_10ms": parse_env_int("AUDIO_OUT_10MS_CHUNKS", 10),
        "history_summarization": "unsupported_guarded_at_native_assistant_turn_stop",
        "history_recent_turns": recent_turns,
    }
    return GenericOfflineRuntime(
        IncrementalSessionEngine(
            detector,
            recognizer,
            responder,
            messages,
            output_chunks_10ms=configuration["output_chunks_10ms"],
            output_sample_rate=PIPELINE_AUDIO_OUT_SAMPLE_RATE,
            history_recent_turns=recent_turns,
            preserve_prompt_messages=len(messages),
            evaluation_trace=evaluation_trace,
        ),
        None,
        recognizer,
        responder,
        configuration,
        provider_observations=responder.get_provider_observations,
        evaluation_trace=evaluation_trace,
    )
