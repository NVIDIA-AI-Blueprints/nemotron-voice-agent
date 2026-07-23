# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Omni Assistant Subagents multi-agent pipeline entry point.

This package owns its prompt catalog (``prompts.yaml``), service catalogs
(``services.cloud.yaml`` / ``services.local.yaml``), subagent registry, and
workers under ``examples.omni_assistant_subagents.subagents``:

* ``OmniTransportAgent`` owns transport I/O, VAD/turn detection, TTS, and
  routes user frames to ``SpeakerOmniAgent`` through a ``BusBridgeProcessor``.
* ``SpeakerOmniAgent`` wraps ``NvidiaOmniMultimodalService`` and is the only
  agent allowed to emit spoken responses.
* ``MediaAnalyzerWorker`` analyzes uploaded image/audio/video attachments.
* ``WebcamAgent`` produces rolling scene summaries from the browser webcam.
* ``ThinkerWorker`` handles on-demand reasoning escalation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

from examples.omni_assistant.pipeline import _create_transport
from examples.omni_assistant_subagents.subagents.media_analyzer import MediaAnalyzerWorker
from examples.omni_assistant_subagents.subagents.speaker import SpeakerOmniAgent
from examples.omni_assistant_subagents.subagents.thinker import ThinkerWorker
from examples.omni_assistant_subagents.subagents.transport import OmniTransportAgent
from examples.omni_assistant_subagents.subagents.webcam import WebcamAgent
from examples.shared.subagents import SubagentRegistry, load_subagent_registry
from utils import is_nvcf, load_prompt_catalog, load_service_entry, parse_json_dict, resolve_prompt

load_dotenv(override=True)

_SUBAGENTS_YAML = Path(__file__).resolve().parent / "subagents.yaml"


def subagent_registry() -> SubagentRegistry:
    """The example's subagent registry (Speaker prompt + UI both read this)."""
    return load_subagent_registry(_SUBAGENTS_YAML)


def _reasoning_for(registry: SubagentRegistry, key: str, default: str) -> str:
    """Reasoning mode declared for a subagent in YAML, or ``default`` if absent."""
    spec = registry.get(key)
    return spec.reasoning if spec else default


_FRAGMENT_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _expand_fragments(text: str, catalog: dict) -> str:
    """Expand {{name}} placeholders from the catalog's ``shared`` fragment map."""
    shared = catalog.get("shared")
    if not isinstance(shared, dict) or not text:
        return text

    def _replace(match: re.Match[str]) -> str:
        value = shared.get(match.group(1))
        return value.strip() if isinstance(value, str) else match.group(0)

    return _FRAGMENT_PATTERN.sub(_replace, text)


def _agent_prompt_content(catalog: dict, agent_name: str, prompt_name: str) -> str:
    """Read one nested agent prompt from the local prompt catalog, expanding shared fragments."""
    agent_prompts = catalog.get("agent_prompts")
    if not isinstance(agent_prompts, dict):
        return ""
    agent = agent_prompts.get(agent_name)
    if not isinstance(agent, dict):
        return ""
    prompt = agent.get(prompt_name)
    if not isinstance(prompt, dict):
        return ""
    content = prompt.get("content")
    return _expand_fragments(content.strip(), catalog) if isinstance(content, str) else ""


async def bot(runner_args: RunnerArguments) -> None:
    """Build and run the Omni Assistant Subagents pipeline for one session."""
    transport = _create_transport(runner_args)
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    body_session_id = str(body.get("session_id") or "").strip()
    runner_session_id = str(getattr(runner_args, "session_id", "") or "").strip()
    session_id = body_session_id or runner_session_id
    prompt_catalog = load_prompt_catalog(__file__)

    prompt_key, base_system_content = resolve_prompt(
        __file__,
        body.get("prompt_content", ""),
        body.get("prompt_key", ""),
    )
    base_system_content = _expand_fragments(base_system_content, prompt_catalog)
    logger.info(
        f"Starting Nemotron Omni Assistant Subagents pipeline "
        f"(prompt={prompt_key}, agents=transport,speaker,media,webcam,thinker)"
    )

    default_llm = load_service_entry("llm", "")
    default_tts = load_service_entry("tts", "")

    model_id = body.get("model_id", "") or default_llm.get("model_id", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    base_url = body.get("base_url", "") or default_llm.get("base_url", "https://integrate.api.nvidia.com/v1")
    system_prompt_override = body.get("system_prompt", "") or default_llm.get("system_prompt", "")
    extra_params = parse_json_dict(
        body.get("extra_params", "") or default_llm.get("extra_params", ""),
        "extra_params",
    )

    system_content = base_system_content
    if system_prompt_override:
        system_content = f"{base_system_content}\n\n{system_prompt_override}".strip()
    registry = subagent_registry()
    context = LLMContext([{"role": "system", "content": system_content}])

    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    api_key = os.getenv("NVIDIA_API_KEY")

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    transport_agent = OmniTransportAgent(
        bus=runner.bus,
        transport=transport,
        context=context,
        api_key=api_key,
        tts_server=tts_server,
        tts_ssl=tts_ssl,
        tts_voice=tts_voice,
        tts_synthesis_mode=tts_synthesis_mode,
        tts_function_id=tts_function_id,
        tts_model=tts_model,
        runner_args=runner_args,
        session_id=session_id,
        subagent_registry=registry,
        proactive_directives={
            key: _agent_prompt_content(prompt_catalog, "TransportAgent", key)
            for key in (
                "proactive_greet",
                "proactive_continue_resume",
                "proactive_continue_compliment",
                "proactive_feedback",
            )
        },
    )
    speaker_agent = SpeakerOmniAgent(
        context=context,
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        extra_params=extra_params,
        audio_response_instruction=_agent_prompt_content(prompt_catalog, "SpeakerAgent", "audio_response_instruction"),
        media_analysis_prompt_handler=transport_agent.queue_media_analysis_prompt,
        uploaded_attachment_available=transport_agent.has_uploaded_attachment,
        attachment_pending=transport_agent.is_attachment_pending,
        thinking_handler=transport_agent.queue_thinking,
        highres_capture_handler=transport_agent.queue_highres_capture,
        visual_status_provider=transport_agent.current_visual_status,
    )
    media_analyzer_agent = MediaAnalyzerWorker(
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        extra_params=extra_params,
        system_prompt=_agent_prompt_content(prompt_catalog, "MediaAnalyzerAgent", "analysis_system_prompt"),
        reasoning=_reasoning_for(registry, MediaAnalyzerWorker.AGENT_NAME, "on"),
    )
    webcam_agent = WebcamAgent(
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        extra_params=extra_params,
        reasoning=_reasoning_for(registry, WebcamAgent.AGENT_NAME, "off"),
        gesture_system_prompt=_agent_prompt_content(prompt_catalog, "WebcamAgent", "gesture_system_prompt"),
        gesture_prompt=_agent_prompt_content(prompt_catalog, "WebcamAgent", "gesture_prompt"),
    )
    thinker_agent = ThinkerWorker(
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        extra_params=extra_params,
        system_prompt=_agent_prompt_content(prompt_catalog, "ThinkerAgent", "thinking_system_prompt"),
    )

    await runner.add_workers(transport_agent, media_analyzer_agent, webcam_agent, thinker_agent, speaker_agent)
    await runner.run()
