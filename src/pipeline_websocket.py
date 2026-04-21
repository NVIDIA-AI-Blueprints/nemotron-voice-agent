# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Voice Agent WebSocket Pipeline.

This module sets up a real-time speech-to-speech pipeline using FastAPI WebSocket,
enabling interactive voice agents with dynamic UI features like system prompt
editing and TTS voice switching in real time.
"""

import argparse
import contextlib
import json
import os
from enum import Enum
from pathlib import Path

import uvicorn
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from nvidia_pipecat.processors.nvidia_context_aggregator import (
    NvidiaTTSResponseCacher,
    create_nvidia_context_aggregator,
)
from nvidia_pipecat.services.nvidia_llm import NvidiaLLMService
from nvidia_pipecat.services.riva_speech import NemotronASRService, NemotronTTSService
from nvidia_pipecat.utils.logging import setup_default_logging
from nvidia_pipecat.utils.riva_text_filter import RivaTextFilter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.tracing.setup import setup_tracing

load_dotenv(override=True)

setup_default_logging(level="DEBUG", exclude_warning=True)

PROMPT_FILE = Path(os.getenv("PROMPT_FILE_PATH", str(Path(__file__).parent.parent / "config" / "prompt.yaml")))
MULTILINGUAL_MODE = os.getenv("ENABLE_MULTILINGUAL", "false").lower() == "true"

IS_TRACING_ENABLED = os.getenv("ENABLE_TRACING", "").lower() == "true"

# Initialize tracing if enabled
if IS_TRACING_ENABLED:
    # Get the endpoint URL
    endpoint_url = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "localhost:4317")

    # Determine which exporter to use based on the endpoint URL
    if endpoint_url.startswith("http://") or endpoint_url.startswith("https://"):
        # HTTP exporter - use full URL with protocol
        otlp_exporter = OTLPSpanExporterHTTP(endpoint=endpoint_url)
    else:
        # gRPC exporter - endpoint should be host:port format (no protocol prefix)
        otlp_exporter = OTLPSpanExporterGRPC(endpoint=endpoint_url, insecure=True)

    # Set up tracing with the exporter
    setup_tracing(
        service_name="nemotron-voice-agent",
        exporter=otlp_exporter,
        console_export=os.getenv("OTEL_CONSOLE_EXPORT", "").lower() == "true",
    )
    logger.info("OpenTelemetry tracing initialized")


class VADProfile(Enum):
    """VAD Profile options."""

    SILERO = "Silero"  # Transport Silero VAD analyzer
    ASR = "ASR"  # ASR VAD


VAD_PROFILE = VADProfile(os.getenv("VAD_PROFILE", VADProfile.ASR))


def _load_prompts() -> dict:
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"Prompt catalog not found at {PROMPT_FILE}")
    try:
        data = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in prompt catalog {PROMPT_FILE}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Prompt catalog at {PROMPT_FILE} must be a mapping.")
    return data


PROMPTS = _load_prompts()


def _resolve_prompt(selector: str) -> list[dict[str, str]]:
    """Resolve a selector like 'model/prompt' into a list of {role, content} messages."""
    try:
        entry = PROMPTS
        for part in selector.split("/"):
            entry = entry[part]
        return [{"role": m["role"], "content": m["content"]} for m in entry["messages"]]
    except (KeyError, TypeError) as e:
        raise KeyError(f"Prompt '{selector}' not found or invalid: {e}") from e


def _inject_prompt_variables(prompt: str, **variables) -> str:
    """Inject variables into prompt placeholders like {lang_codes}."""
    try:
        return prompt.format(**variables)
    except KeyError:
        return prompt


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def run_bot(websocket: WebSocket, stream_id: str):
    """Run the voice agent bot with WebSocket connection.

    Args:
        websocket: The FastAPI WebSocket connection for audio streaming
        stream_id: The ID of the stream
    """
    # Parse AUDIO_OUT_10MS_CHUNKS with error handling
    try:
        audio_out_10ms_chunks = int(os.getenv("AUDIO_OUT_10MS_CHUNKS", "10"))
    except ValueError:
        logger.warning("Invalid AUDIO_OUT_10MS_CHUNKS, falling back to default 10")
        audio_out_10ms_chunks = 10

    # Create FastAPI WebSocket transport
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_out_10ms_chunks=audio_out_10ms_chunks,
            serializer=ProtobufFrameSerializer(),
            vad_analyzer=SileroVADAnalyzer() if VAD_PROFILE == VADProfile.SILERO else None,
        ),
    )

    # None if unset, True if "true", False if "false"
    enable_thinking = {"true": True, "false": False}.get(os.getenv("ENABLE_THINKING", "").lower())

    # Parse TEMPERATURE with error handling
    try:
        temperature = float(os.getenv("TEMPERATURE", "1.0"))
    except ValueError:
        logger.warning("Invalid TEMPERATURE, falling back to default 1.0")
        temperature = 1.0

    # Parse TOP_P with error handling
    try:
        top_p = float(os.getenv("TOP_P", "1.0"))
    except ValueError:
        logger.warning("Invalid TOP_P, falling back to default 1.0")
        top_p = 1.0

    try:
        max_tokens = int(os.getenv("MAX_TOKENS", "2048"))
    except ValueError:
        logger.warning("Invalid MAX_TOKENS, falling back to default 2048")
        max_tokens = 2048

    llm = NvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=os.getenv("NVIDIA_LLM_URL", "https://integrate.api.nvidia.com/v1"),
        model=os.getenv("NVIDIA_LLM_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
        params=BaseOpenAILLMService.InputParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **(
                {"extra": {"extra_body": {"chat_template_kwargs": {"enable_thinking": enable_thinking}}}}
                if enable_thinking is not None
                else {}
            ),
        ),
    )

    # ASR service config - add extended stop_history for multilingual mode
    stt_config = {
        "server": os.getenv("ASR_SERVER_URL", "grpc.nvcf.nvidia.com:443"),
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "language": os.getenv("ASR_LANGUAGE", "en-US"),
        "sample_rate": 16000,
        "generate_interruptions": VAD_PROFILE == VADProfile.ASR,
        "model": os.getenv("ASR_MODEL_NAME", "parakeet-1.1b-en-US-asr-streaming-silero-vad-sortformer"),
        "function_id": os.getenv("ASR_CLOUD_FUNCTION_ID", "1598d209-5e27-4d3c-8079-4751568b1081"),
    }
    if MULTILINGUAL_MODE:
        stt_config.update(stop_history=900, stop_history_eou=900)

    stt = NemotronASRService(**stt_config)

    # Load IPA dictionary with error handling
    ipa_file = os.getenv("TTS_IPA_FILE_PATH", Path(__file__).parent.parent / "config" / "ipa.json")
    try:
        with open(ipa_file, encoding="utf-8") as f:
            ipa_dict = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"IPA dictionary file not found at {ipa_file}")
        raise FileNotFoundError(f"IPA dictionary file not found at {ipa_file}") from e
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in IPA dictionary file: {e}")
        raise ValueError(f"Invalid JSON in IPA dictionary file: {e}") from e
    except Exception as e:
        logger.error(f"Error loading IPA dictionary: {e}")
        raise

    # TTS text filter only enabled when ENABLE_TTS_TEXT_FILTER=true AND language is en-US
    enable_riva_text_filter = (
        os.getenv("ENABLE_TTS_TEXT_FILTER", "true").lower() == "true"
        and os.getenv("TTS_LANGUAGE", "en-US") == "en-US"
        and os.getenv("ENABLE_MULTILINGUAL", "false").lower() == "false"
        and os.getenv("SYSTEM_PROMPT_SELECTOR", "").lower() != "llama/tts_emotion_tags"
    )

    tts = NemotronTTSService(
        server=os.getenv("TTS_SERVER_URL", "grpc.nvcf.nvidia.com:443"),
        api_key=os.getenv("NVIDIA_API_KEY"),
        voice_id=os.getenv("TTS_VOICE_ID", "Magpie-Multilingual.EN-US.Aria"),
        model=os.getenv("TTS_MODEL_NAME", "magpie_tts_ensemble-Magpie-Multilingual"),
        language=os.getenv("TTS_LANGUAGE", "en-US"),
        sample_rate=22050,
        zero_shot_audio_prompt_file=(
            Path(os.getenv("ZERO_SHOT_AUDIO_PROMPT")) if os.getenv("ZERO_SHOT_AUDIO_PROMPT") else None
        ),
        custom_dictionary=ipa_dict,
        text_filters=[RivaTextFilter()] if enable_riva_text_filter else [],
    )

    def _validated_selector(raw_value: str | None, default: str) -> str:
        selector = (raw_value or "").strip() or default
        if "/" not in selector:
            raise ValueError("SYSTEM_PROMPT_SELECTOR must be in '<model>/<prompt>' format")
        return selector

    if MULTILINGUAL_MODE:
        prompt_selector = _validated_selector(
            os.getenv("SYSTEM_PROMPT_SELECTOR"),
            "llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant",
        )
        lang_codes = ", ".join(tts.list_available_voices().keys())
        messages = _resolve_prompt(prompt_selector)
        messages = [
            {"role": msg["role"], "content": _inject_prompt_variables(msg["content"], lang_codes=lang_codes)}
            for msg in messages
        ]
        logger.info(f"Loaded multilingual prompt: {prompt_selector} with languages: {lang_codes}")
    else:
        prompt_selector = _validated_selector(
            os.getenv("SYSTEM_PROMPT_SELECTOR"),
            "nemotron-3-nano/generic_voice_assistant",
        )
        messages = _resolve_prompt(prompt_selector)
        logger.info(f"Loaded prompt: {prompt_selector}")

    # Defensive check to ensure the resolved prompt is not empty
    if not messages:
        raise ValueError(f"Resolved system prompt has no messages for selector: {prompt_selector}")

    context = LLMContext(messages)

    # Configure speculative speech processing based on environment variable
    enable_speculative_speech = os.getenv("ENABLE_SPECULATIVE_SPEECH", "true").lower() == "true"
    try:
        chat_history_limit = int(os.getenv("CHAT_HISTORY_LIMIT", "20"))
    except ValueError:
        logger.warning("Invalid CHAT_HISTORY_LIMIT, falling back to default 20")
        chat_history_limit = 20

    # Preserve all initial prompt messages from prompt.yaml
    # This ensures system and first user messages (used for prompting) are never truncated
    preserve_prompt_messages = len(messages)

    if enable_speculative_speech:
        context_aggregator = create_nvidia_context_aggregator(
            context,
            send_interims=True,
            chat_history_limit=chat_history_limit,
            preserve_prompt_messages=preserve_prompt_messages,
        )
        tts_response_cacher = NvidiaTTSResponseCacher()
    else:
        context_aggregator = create_nvidia_context_aggregator(
            context,
            send_interims=False,
            chat_history_limit=chat_history_limit,
            preserve_prompt_messages=preserve_prompt_messages,
        )
        tts_response_cacher = None

    pipeline = Pipeline(
        [
            transport.input(),  # WebSocket input from client
            stt,  # Speech-To-Text
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            *([tts_response_cacher] if tts_response_cacher else []),
            transport.output(),  # WebSocket output to client
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            send_initial_empty_metrics=True,
            start_metadata={"stream_id": stream_id},
        ),
        enable_tracing=IS_TRACING_ENABLED,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        try:
            # Kick off the conversation.
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames(
                [
                    LLMMessagesUpdateFrame(messages=messages, run_llm=True),
                ]
            )
        except Exception as e:
            logger.error(f"Error on client connected: {e}")

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)


@app.websocket("/ws/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    """WebSocket endpoint for voice agent connections.

    Args:
        websocket: The WebSocket connection
        stream_id: The ID of the stream
    """
    await websocket.accept()
    logger.info(f"Accepted WebSocket connection for stream ID {stream_id}")

    try:
        with logger.contextualize(stream_id=stream_id):
            await run_bot(websocket, stream_id)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {websocket.client}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        with contextlib.suppress(BaseException):
            await websocket.close()


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebSocket Voice Agent Pipeline")
    parser.add_argument("--host", default="0.0.0.0", help="Host for HTTP server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server (default: 7860)")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers for HTTP server (default: 1)")
    args = parser.parse_args()

    uvicorn.run("src.pipeline_websocket:app", host=args.host, port=args.port, workers=args.workers)
