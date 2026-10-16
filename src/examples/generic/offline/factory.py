# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bind the generic assistant's catalogs to independently scheduled providers."""

import hashlib
import json
from pathlib import Path

from openai import AsyncOpenAI

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from examples.shared.pipeline_utils import (
    PIPELINE_AUDIO_OUT_SAMPLE_RATE,
    apply_pinned_prompt_summary,
    build_context_messages,
)
from utils import (
    is_nvcf,
    is_streaming_llm_url,
    load_ipa_dictionary,
    load_service_entry,
    load_tools_catalog,
    normalize_lang_code,
    nvidia_api_key,
    parse_env_int,
    parse_json_dict,
    resolve_prompt,
    resolve_tools_available,
)

from .detection import create_generic_detector
from .engine import OfflineSessionEngine
from .providers import OpenAITextStream, StreamingResponder, create_riva_recognizer, create_riva_synthesis


class GenericOfflineRuntime:
    """Own provider resources and expose a sanitized evaluation configuration."""

    def __init__(self, engine, client, recognizer, synthesis, configuration):
        """Keep session-owned provider resources out of the scheduling core."""
        self._engine = engine
        self._client = client
        self._recognizer = recognizer
        self._synthesis = synthesis
        self.configuration = configuration
        self._closed = False

    async def aclose(self):
        """Release resources even when model initialization finishes after cancellation."""
        if not self._closed:
            self._closed = True
            self._recognizer.cancel()
            self._synthesis.cancel()
            await self._client.close()

    async def run(self, source):
        """Run once and release every provider connection on success or failure."""
        try:
            return await self._engine.run(source)
        finally:
            await self.aclose()


def build_runtime(config: dict) -> GenericOfflineRuntime:
    """Use the same generic prompt, service entries, text filter, and tool schemas."""
    from examples.generic.tool_handlers import TOOL_HANDLERS

    module_file = Path(__file__).resolve().parents[1] / "pipeline.py"
    prompt_key, prompt = resolve_prompt(module_file, config.get("prompt_content", ""), config.get("prompt_key", ""))
    asr = load_service_entry("asr", "")
    llm = load_service_entry("llm", "")
    tts = load_service_entry("tts", "")
    settings = {
        "api_key": nvidia_api_key(),
        "asr_server": config.get("asr_server") or asr.get("server", "grpc.nvcf.nvidia.com:443"),
        "asr_function_id": config.get("asr_function_id") or asr.get("function_id", ""),
        "asr_model": config.get("asr_model") or asr.get("model", ""),
        "asr_language_code": config.get("asr_language_code") or asr.get("language_code", "en-US"),
        "asr_automatic_punctuation": str(
            config.get("asr_automatic_punctuation", asr.get("automatic_punctuation", "true"))
        ).lower()
        != "false",
        "tts_server": config.get("tts_server") or tts.get("server", "grpc.nvcf.nvidia.com:443"),
        "tts_function_id": config.get("tts_function_id") or tts.get("function_id", ""),
        "tts_voice_id": config.get("tts_voice_id") or tts.get("voice_id", "Magpie-Multilingual.EN-US.Aria"),
        "tts_language_code": normalize_lang_code(
            config.get("tts_language_code") or tts.get("language_code") or "en-US"
        ),
        "tts_sample_rate": PIPELINE_AUDIO_OUT_SAMPLE_RATE,
        "custom_dictionary": load_ipa_dictionary(),
    }
    synthesis_mode = config.get("tts_synthesis_mode") or tts.get("synthesis_mode") or "stitched"
    tts_model = config.get("tts_model") or tts.get("model", "")
    if synthesis_mode != "stitched" or "zeroshot" in tts_model.lower() or config.get("tts_zero_shot_audio_prompt_file"):
        raise ValueError("Offline execution currently supports stitched Magpie synthesis without zero-shot prompts")
    if is_nvcf(settings["tts_server"]) and not settings["tts_function_id"]:
        # Match NvidiaTTSService's default model_function_map in the generic pipeline.
        settings["tts_function_id"] = "877104f7-e885-42b9-8de8-f6e4c6303969"
    model = config.get("model_id") or llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b")
    base_url = config.get("base_url") or llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    if is_streaming_llm_url(base_url):
        raise ValueError("Offline execution requires an HTTP chat-completions LLM endpoint; disable streaming input")
    extra_params = parse_json_dict(config.get("extra_params") or llm.get("extra_params", ""), label="extra_params")
    extra = {}
    for key, convert in (("temperature", float), ("max_tokens", int)):
        value = config.get(key)
        if value in ("", None):
            value = llm.get(key)
        if value not in ("", None):
            extra[key] = convert(value)
    # Pipecat's live chat request applies extra parameters after scalar settings.
    extra.update(extra_params)
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
    detector = create_generic_detector()
    recognizer = create_riva_recognizer(settings)
    synthesis = create_riva_synthesis(settings)
    client = AsyncOpenAI(api_key=settings["api_key"] or "not-used-for-local", base_url=base_url, timeout=600)
    messages = build_context_messages(prompt, system_prompt)
    recent_turns = parse_env_int("CHAT_HISTORY_RECENT_TURNS", 10)

    class SummaryInference:
        """Expose the generic summary helper's small inference contract."""

        async def run_inference(self, context, *, max_tokens=None, system_instruction=None):
            """Generate a summary against the same NIM model and settings."""
            request_messages = context.get_messages()
            if system_instruction:
                request_messages = [{"role": "system", "content": system_instruction}, *request_messages]
            arguments = {"model": model, "messages": request_messages, **extra}
            if max_tokens is not None:
                token_limit = "max_completion_tokens" if "max_completion_tokens" in arguments else "max_tokens"
                arguments[token_limit] = max_tokens
            response = await client.chat.completions.create(**arguments)
            return response.choices[0].message.content

    async def prepare_history(history):
        from pipecat.processors.aggregators.llm_context import LLMContext

        context = LLMContext(history)
        await apply_pinned_prompt_summary(
            context=context,
            llm=SummaryInference(),
            preserve_prompt_messages=len(messages),
            recent_turns=recent_turns,
            summary_system_prompt=system_prompt,
        )
        return context.get_messages()

    text = OpenAITextStream(
        client,
        model=model,
        extra=extra,
        tools=tools,
        handlers=handlers,
        tool_choice=config.get("tool_choice") or "auto",
        history_preparer=prepare_history,
    )
    responder = StreamingResponder(text, synthesis, text_filter=NemotronSpeechTextFilter())
    configuration = {
        "mode": "offline",
        "sample_rate": 16000,
        "pipeline": "generic-assistant",
        "asr_mode": "continuous_stream",
        "asr_availability_policy": "source_dependency_plus_observed_bulk_residual",
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
        "asr_stop_history_ms": 400,
        "asr_force_eou": False,
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
        "tts_resampling": "soxr_vhq_continuous_with_eof_flush",
        "asr_function_id": settings["asr_function_id"],
        "tts_function_id": settings["tts_function_id"],
        "tool_names": sorted(handlers),
        "tool_choice": config.get("tool_choice") or "auto",
        "interrupted_history": "aligned_prefix_else_omit",
        "history_summarization": "generic_pinned_prompt_policy_before_next_request",
        "history_recent_turns": recent_turns,
    }
    return GenericOfflineRuntime(
        OfflineSessionEngine(detector, recognizer, responder, messages), client, recognizer, synthesis, configuration
    )
