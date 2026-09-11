# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""FastAPI server: serves the client UI and handles WebRTC/WebSocket signaling.

Routes:
  POST      /api/start       - WebRTC start (TTS readiness gate + session creation)
  POST      /api/offer       - WebRTC SDP offer
  PATCH     /api/offer       - WebRTC ICE candidate trickle
  WebSocket /api/ws          - WebSocket transport (RTVI / Pipecat)
  POST      /v1/realtime/client_secrets - OpenAI Realtime client-secret bootstrap
  WebSocket /v1/realtime     - OpenAI Realtime–shaped JSON (session + pipeline handoff)
  GET       /api/deployment  - Active example metadata
  GET       /api/prompts     - Prompt catalog
  GET       /api/services    - Service catalog (LLM, TTS, ASR)
  GET       /api/tts-config  - TTS voices & languages (optional ASR params for ASR∩TTS intersection)
  GET       /                - Built client UI (from client/dist/)

Pipeline / example selection lives in ``examples_registry.yaml`` (the
``selection`` and ``transports`` fields). To change which examples or
transports the UI exposes, edit the YAML. ``EXAMPLE_SELECTION`` and
``TRANSPORT_SELECTION`` env vars exist as compose-profile pinning hooks
(used by ``docker-compose.yml``) but are intentionally not surfaced in
``.env.example`` or host-native run instructions — day-to-day deployments
should configure them via YAML or compose profile.

Run:
  uv run python src/server.py                                              # use selection/transports from YAML
  PIPELINE_TLS=false uv run python src/server.py                           # plain HTTP
  uv run python src/server.py --tls-cert c --tls-key k                     # custom TLS cert/key
  uv run python src/server.py --prompt-file benchmarking_tools/scaling-perf/perf_prompts.yaml
"""

from dotenv import load_dotenv

from utils import parse_env_bool, parse_env_int

load_dotenv(override=True)

import argparse
import asyncio
import contextlib
import copy
import json
import os
import re
import sys
import time
import urllib.request
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from urllib.parse import urlparse

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

import config_store
import examples_registry
from attachment_store import consume_capture_request, store_attachment
from examples.shared.pipeline_utils import PIPELINE_AUDIO_IN_SAMPLE_RATE, PIPELINE_AUDIO_OUT_SAMPLE_RATE
from examples.shared.prewarm import (
    build_session_languages,
    get_tts_config,
    peek_cached_asr_config,
    peek_cached_tts_config,
    prewarm_tts,
    resolve_voice_for_language,
    warmup_tts_synthesis,
)
from examples.shared.subagents import load_subagent_registry
from realtime.protocol import MAX_REALTIME_EVENT_BYTES
from utils import (
    PROJECT_ROOT,
    build_services_api_response,
    default_prompt_key,
    filter_session_config,
    is_endpoint_reachable,
    is_nvcf,
    load_prompt_catalog,
    load_service_entry,
    load_service_entry_by_id,
    load_tools_catalog,
    normalize_lang_code,
    parse_endpoint,
    resolve_prompt,
    set_active_slots,
    set_service_context,
)
from webcam_frame_store import store_webcam_frame, webcam_client_config

CLIENT_DIST = PROJECT_ROOT / "client" / "dist"
_session_configs: dict[str, dict] = {}
_active_session_configs: dict[str, dict] = {}
_CONNECT_PREWARM_TIMEOUT_SECS = parse_env_int("CONNECT_PREWARM_TIMEOUT_SECS", 45)
_CONNECT_HEALTH_TIMEOUT_SECS = 5
_REALTIME_GA_MAX_OUTPUT_TOKENS = 4096
_NIM_READY_PATH = "/v1/health/ready"
_LOCAL_SERVICE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})
_SPEECH_READY_ENDPOINTS = {
    "asr": (
        (
            "nemotron-asr-streaming-english",
            "nemotron-asr-streaming-multilingual",
            "parakeet-ctc-asr",
            "parakeet-rnnt-asr",
        ),
        50052,
        50152,
        9001,
    ),
    "tts": (
        (
            "magpie-multilingual-tts-service",
            "chatterbox-tts-service",
            "magpie-zeroshot-tts-service",
        ),
        50051,
        50151,
        9000,
    ),
}
_TURN_LISTEN_PORT = 3478
_INDEX_NO_CACHE_HEADERS = {"Cache-Control": "no-store"}
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_MAX_REALTIME_CLIENT_SECRET_REQUEST_BYTES = 256 * 1024
_REALTIME_TRUNCATION_PIPELINES = frozenset({"frontend-backend-agent", "generic-assistant", "multilingual-assistant"})
_MULTI_WORKER_SESSION_CONFIG_MESSAGE = (
    "Session-config based WebRTC and WebSocket flows are disabled when "
    "UVICORN_WORKERS is greater than 1. Use a single worker, sticky routing, "
    "or shared session storage."
)


_TRANSPORT_OPTIONS: tuple[dict[str, str], ...] = (
    {"id": "webrtc", "label": "WebRTC"},
    {"id": "websocket", "label": "WebSocket"},
)


@dataclass(frozen=True)
class WorkerAppConfig:
    """App-factory config reconstructed from server CLI args."""

    host: str = "localhost"
    prompt_file: str = ""

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "WorkerAppConfig":
        """Build a config from an already-parsed argparse namespace."""
        return cls(host=args.host, prompt_file=args.prompt_file)

    @classmethod
    def from_argv(cls, argv: list[str]) -> "WorkerAppConfig":
        """Build a config by parsing a raw argv list (used by worker subprocesses)."""
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--prompt-file", type=str, default="")
        args, _ = parser.parse_known_args(argv)
        return cls.from_args(args)


def _parse_min_int(value: str, minimum: int = 1) -> int:
    """Parse an integer CLI argument and enforce a minimum value."""
    parsed = int(value)
    if parsed < minimum:
        raise argparse.ArgumentTypeError(f"must be >= {minimum}")
    return parsed


def _run_single_worker(args: argparse.Namespace, app: FastAPI, ssl_kwargs: dict) -> None:
    """Run uvicorn with a pre-built app instance."""
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ws_max_size=MAX_REALTIME_EVENT_BYTES,
        **ssl_kwargs,
    )


def _run_multi_worker(args: argparse.Namespace, workers: int, ssl_kwargs: dict) -> None:
    """Run uvicorn in multi-worker mode via the importable app factory."""
    uvicorn.run(
        "server:app_factory",
        host=args.host,
        port=args.port,
        workers=workers,
        factory=True,
        ws_max_size=MAX_REALTIME_EVENT_BYTES,
        **ssl_kwargs,
    )


def _multi_worker_mode_enabled() -> bool:
    """Return whether the server is running with more than one uvicorn worker."""
    return parse_env_int("UVICORN_WORKERS", 1, min_value=1) > 1


def _multi_worker_session_config_response() -> JSONResponse:
    """Reject routes that rely on process-local session config in multi-worker mode."""
    return JSONResponse(status_code=503, content={"info": _MULTI_WORKER_SESSION_CONFIG_MESSAGE})


def _deployment_response(active: dict, options: list[dict]) -> dict:
    """Build the metadata payload consumed by the client selector page.

    Visibility (examples, transports) is driven entirely by the YAML
    registry plus its environment overrides, so this function just packages
    what the registry has already resolved.
    """
    active_deployment = dict(active)
    active_deployment.setdefault("slots", [])
    active_deployment.setdefault("capabilities", [])
    transports = set(examples_registry.visible_transports())
    return {
        "active": active_deployment,
        "selectable": not examples_registry.is_locked(),
        "options": options,
        "transports": [option for option in _TRANSPORT_OPTIONS if option["id"] in transports],
        "audio": {
            "input_sample_rate": PIPELINE_AUDIO_IN_SAMPLE_RATE,
            "output_sample_rate": PIPELINE_AUDIO_OUT_SAMPLE_RATE,
        },
    }


def _sanitize_session_config(data: dict, fallback_example_key: str = "") -> dict:
    """Bind the catalog for the requested pipeline_mode and drop unknown slot keys."""
    if not isinstance(data, dict):
        raise ValueError("session config must be a JSON object")
    example = _bind_example_context_by_key(str(data.get("pipeline_mode", "")) or fallback_example_key)
    config = dict(data)
    if not config.get("prompt_key") and not config.get("prompt_content"):
        prompt_key = examples_registry.prompt_default_key(example["key"])
        if prompt_key:
            config["prompt_key"] = prompt_key
    return filter_session_config(config)


def _example_with_module_file(example_key: str = "") -> tuple[dict, Path]:
    """Return ``(registry_entry, module_file)`` for a registry example key."""
    selected = examples_registry.find(example_key)
    return selected, examples_registry.example_module_file(selected)


def _activate_example_catalog(module_file: Path, example: dict) -> None:
    """Use package-local service catalogs and slot filtering for a selected example."""
    module_dir = Path(module_file).resolve().parent
    for env_var, candidate in (
        ("SERVICES_CLOUD_PATH", module_dir / "services.cloud.yaml"),
        ("SERVICES_LOCAL_PATH", module_dir / "services.local.yaml"),
    ):
        os.environ[env_var] = str(candidate)
    set_active_slots(example.get("slots") or None)


def _bind_example_context(module_file: Path, example: dict) -> None:
    """Bind package-local service catalogs to the current request context."""
    set_service_context(Path(module_file).resolve().parent, example.get("slots") or None)


def _bind_example_context_by_key(example_key: str = "") -> dict:
    """Bind the package-local service catalog for one request without mutating globals."""
    example, module_file = _example_with_module_file(example_key)
    _bind_example_context(module_file, example)
    return example


def _activate_example_catalog_by_key(example_key: str = "") -> dict:
    """Activate the package-local service catalog process-wide for startup defaults."""
    example, module_file = _example_with_module_file(example_key)
    _activate_example_catalog(module_file, example)
    return example


def _resolve_config(session_id: str = "", fallback_example_key: str = "", **query_params: str) -> dict:
    """Merge stored session config with query overrides; sanitize and hydrate from YAML."""
    base = _session_configs.pop(session_id, {}) if session_id else {}
    base.update({k: v for k, v in query_params.items() if v})
    return _sanitize_session_config({k: v for k, v in base.items() if v}, fallback_example_key=fallback_example_key)


def _get_default_tts_selection() -> tuple[str, str, str, str]:
    """Return default TTS ``(server, voice_id, function_id, model)`` from the catalog."""
    default_tts = load_service_entry("tts", "")
    return (
        default_tts.get("server", "grpc.nvcf.nvidia.com:443"),
        default_tts.get("voice_id", ""),
        default_tts.get("function_id", ""),
        default_tts.get("model", ""),
    )


def _resolve_tts_selection(
    server: str = "",
    voice_id: str = "",
    function_id: str = "",
    model: str = "",
) -> tuple[str, str, str, str]:
    """Bind server + metadata as one selection.

    When ``server`` is omitted, fall back to the catalog default for server,
    voice, function_id, and model together. When ``server`` is explicit, keep
    empty function_id/model (do not borrow Magpie defaults for another NIM).
    """
    if server and voice_id:
        return server, voice_id, function_id, model

    default_server, default_voice, default_function_id, default_model = _get_default_tts_selection()
    if server:
        return server, voice_id or default_voice, function_id, model
    return (
        default_server,
        voice_id or default_voice,
        function_id or default_function_id,
        model or default_model,
    )


def _get_default_asr_selection() -> str:
    default_asr = load_service_entry("asr", "")
    return default_asr.get("server", "grpc.nvcf.nvidia.com:443")


def _get_default_asr_catalog() -> tuple[str, str, str]:
    default_asr = load_service_entry("asr", "")
    return (
        default_asr.get("server", "grpc.nvcf.nvidia.com:443"),
        default_asr.get("model", ""),
        default_asr.get("function_id", ""),
    )


def _get_default_llm_selection() -> tuple[str, str]:
    default_llm = load_service_entry("llm", "")
    return (
        default_llm.get("base_url", "https://integrate.api.nvidia.com/v1"),
        default_llm.get("model_id", "nvidia/nemotron-3.5-lightning-30b-a3b"),
    )


def _store_session_config(data: dict, fallback_example_key: str = "") -> str:
    session_id = uuid.uuid4().hex[:12]
    _session_configs[session_id] = _sanitize_session_config(data, fallback_example_key=fallback_example_key)
    return session_id


def _build_rtvi_runner_body(
    config: dict,
    *,
    session_id: str,
    request_data: object = None,
) -> dict:
    """Build an RTVI runner body without accepting client protocol authority."""
    body = dict(request_data) if isinstance(request_data, dict) else {}
    body.update(config)
    body.pop("realtime_controller", None)
    body["protocol"] = "rtvi"
    body["session_id"] = session_id
    return body


def _session_capability_error(session_id: str, capability: str) -> JSONResponse | None:
    """Return an upload rejection when session/capability validation fails."""
    cleaned_session_id = session_id.strip()
    config = _active_session_configs.get(cleaned_session_id) or _session_configs.get(cleaned_session_id)
    if not cleaned_session_id or config is None:
        return JSONResponse(status_code=404, content={"detail": "session not found"})
    example = examples_registry.metadata(examples_registry.find(config.get("pipeline_mode", "")))
    if capability not in set(example.get("capabilities") or []):
        return JSONResponse(status_code=403, content={"detail": f"session does not support {capability}"})
    return None


async def _run_blocking(func, *args, timeout: float | None = None):
    task = asyncio.to_thread(func, *args)
    if timeout is None:
        return await task
    return await asyncio.wait_for(task, timeout=timeout)


async def _read_upload_file_with_limit(file: UploadFile, *, max_bytes: int = _MAX_UPLOAD_BYTES) -> bytes:
    """Read an upload body while enforcing a hard byte limit."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Upload exceeds {max_bytes // (1024 * 1024)}MB limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _build_ice_servers(request: Request) -> list[dict]:
    """Build ICE server config for the WebRTC client.

    Coturn (compose `turn` profile) runs with long-term credentials matching
    TURN_USERNAME / TURN_PASSWORD. Set TURN_URL when TURN is hosted separately;
    otherwise derive the host from the request for same-host deployments.
    """
    username = os.environ.get("TURN_USERNAME", "").strip()
    password = os.environ.get("TURN_PASSWORD", "").strip()
    if not username or not password:
        return []

    turn_url = (os.environ.get("TURN_URL") or os.environ.get("TURN_SERVER_URL") or "").strip()
    if turn_url:
        base = turn_url.split("?", 1)[0].rstrip("/")
        if not base.startswith(("turn:", "turns:")):
            base = f"turn:{base}"
    else:
        host = request.headers.get("x-forwarded-host", request.url.hostname or "")
        host = host.split(",")[0].strip().split(":")[0]
        if not host or host in ("localhost", "127.0.0.1", "::1"):
            return []
        base = f"turn:{host}:{_TURN_LISTEN_PORT}"

    return [
        {
            "urls": [f"{base}?transport=udp", f"{base}?transport=tcp"],
            "username": username,
            "credential": password,
        }
    ]


def _should_skip_tts_prewarm(example: dict) -> bool:
    """Skip TTS warm-up for examples without a ``tts`` slot."""
    return "tts" not in (example.get("slots") or [])


def _service_host_port(server: str) -> tuple[str, int]:
    """Parse a service address into host/port, raising on invalid input."""
    parsed = parse_endpoint(server)
    if parsed is None:
        raise RuntimeError(f"Invalid service address: {server or '(empty)'}")
    return parsed


def _check_service_port(label: str, server: str) -> None:
    if not is_endpoint_reachable(server):
        raise RuntimeError(
            f"Selected {label} service is not reachable at {server}. "
            f"Check that the {label} service is running and healthy."
        )


def _http_host(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def _service_starting_message(label: str, *, ready_json: bool = False) -> str:
    expected_state = "report ready" if ready_json else "become healthy"
    return (
        f"Selected {label} service is still starting. "
        f"Wait for the service health check to {expected_state}, then try again."
    )


def _local_speech_ready_url(label: str, server: str) -> str:
    """Return the NIM HTTP readiness URL for built-in local ASR/TTS endpoints."""
    host, port = _service_host_port(server)
    normalized_host = host.strip("[]").lower()
    service_hosts, container_port, host_port, http_port = _SPEECH_READY_ENDPOINTS.get(
        label.lower(),
        ((), 0, 0, 0),
    )

    if normalized_host in service_hosts and port == container_port:
        return f"http://{normalized_host}:{http_port}{_NIM_READY_PATH}"
    if normalized_host in _LOCAL_SERVICE_HOSTS and port == host_port:
        return f"http://{_http_host(host)}:{http_port}{_NIM_READY_PATH}"

    return ""


def _local_llm_health_url(base_url: str, model_id: str) -> tuple[str, bool]:
    """Return (health_url, expects_ready_json) for built-in local LLM endpoints."""
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if not host:
        return "", False

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    scheme = parsed.scheme or "http"
    normalized_host = host.strip("[]").lower()
    http_host = _http_host(host)

    if normalized_host == "nvidia-llm" and port == 8000:
        return f"{scheme}://nvidia-llm:8000{_NIM_READY_PATH}", False
    if normalized_host == "nvidia-llm-omni" and port == 8000:
        return f"{scheme}://nvidia-llm-omni:8000{_NIM_READY_PATH}", False
    if normalized_host == "nvidia-llm-vllm" and port == 8000:
        return f"{scheme}://nvidia-llm-vllm:8000/health", False
    if normalized_host == "nvidia-llm-vllm-omni" and port == 8002:
        return f"{scheme}://nvidia-llm-vllm-omni:8002/health", False

    if normalized_host in _LOCAL_SERVICE_HOSTS and port == 18000:
        is_vllm = "30b-a3b" in model_id.lower() or "nvfp4" in model_id.lower()
        health_path = "/health" if is_vllm else _NIM_READY_PATH
        return f"{scheme}://{http_host}:18000{health_path}", False
    if normalized_host in _LOCAL_SERVICE_HOSTS and port == 18002:
        return f"{scheme}://{http_host}:18002{_NIM_READY_PATH}", False
    if normalized_host in _LOCAL_SERVICE_HOSTS and port == 8002:
        return f"{scheme}://{http_host}:8002/health", False

    return "", False


def _check_http_health(label: str, target: str, health_url: str, expects_ready_json: bool = False) -> None:
    try:
        with urllib.request.urlopen(health_url, timeout=_CONNECT_HEALTH_TIMEOUT_SECS) as response:
            body = response.read()
    except Exception as exc:
        logger.warning(f"{label} health check failed for {target} via {health_url}: {exc}")
        raise RuntimeError(_service_starting_message(label)) from exc

    if not expects_ready_json:
        return

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"{label} readiness check for {target} via {health_url} returned invalid JSON")
        raise RuntimeError(_service_starting_message(label, ready_json=True)) from exc

    if payload.get("status") != "ready" and payload.get("ready") is not True:
        logger.warning(f"{label} readiness check for {target} via {health_url} returned {payload}")
        raise RuntimeError(_service_starting_message(label, ready_json=True))


async def _run_http_readiness_check(
    label: str,
    target: str,
    health_url: str,
    expects_ready_json: bool = False,
) -> None:
    try:
        await _run_blocking(
            _check_http_health,
            label,
            target,
            health_url,
            expects_ready_json,
            timeout=_CONNECT_HEALTH_TIMEOUT_SECS + 1,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Selected {label} service is still starting. Health check timed out after {_CONNECT_HEALTH_TIMEOUT_SECS}s."
        ) from exc


async def _ensure_llm_ready_for_connection(config: dict, example: dict) -> None:
    """Run a local LLM readiness check before starting the session."""
    if "llm" not in (example.get("slots") or []):
        return

    base_url = config.get("base_url", "")
    model_id = config.get("model_id", "")
    if not base_url or not model_id:
        default_base_url, default_model_id = _get_default_llm_selection()
        base_url = base_url or default_base_url
        model_id = model_id or default_model_id
    health_url, expects_ready_json = _local_llm_health_url(base_url, model_id)
    if not health_url:
        return

    await _run_http_readiness_check("LLM", base_url, health_url, expects_ready_json)


async def _ensure_asr_ready_for_connection(config: dict, example: dict) -> None:
    """Run an ASR readiness check before starting the session (examples with an ``asr`` slot)."""
    if "asr" not in (example.get("slots") or []):
        return

    asr_server = config.get("asr_server", "") or _get_default_asr_selection()
    ready_url = _local_speech_ready_url("ASR", asr_server)
    if ready_url:
        await _run_http_readiness_check("ASR", asr_server, ready_url, expects_ready_json=True)
        return

    try:
        await _run_blocking(
            _check_service_port,
            "ASR",
            asr_server,
            timeout=_CONNECT_HEALTH_TIMEOUT_SECS + 1,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Selected ASR service is still starting. Health check timed out after {_CONNECT_HEALTH_TIMEOUT_SECS}s."
        ) from exc


async def _ensure_tts_ready_for_connection(config: dict, example: dict) -> None:
    """Warm up TTS unless the selected pipeline handles it internally."""
    if _should_skip_tts_prewarm(example):
        return

    tts_server, voice_id, tts_function_id, tts_model = _resolve_tts_selection(
        config.get("tts_server", ""),
        config.get("tts_voice_id", ""),
        config.get("tts_function_id", ""),
        config.get("tts_model", ""),
    )

    ready_url = _local_speech_ready_url("TTS", tts_server)
    if ready_url:
        await _run_http_readiness_check("TTS", tts_server, ready_url, expects_ready_json=True)
        return

    if not is_nvcf(tts_server):
        try:
            await _run_blocking(
                _check_service_port,
                "TTS",
                tts_server,
                timeout=_CONNECT_HEALTH_TIMEOUT_SECS + 1,
            )
            return
        except TimeoutError as exc:
            raise RuntimeError(
                f"Selected TTS service is still starting. Health check timed out after {_CONNECT_HEALTH_TIMEOUT_SECS}s."
            ) from exc

    try:
        is_ready = await _run_blocking(
            warmup_tts_synthesis,
            tts_server,
            voice_id,
            tts_function_id,
            tts_model,
            timeout=_CONNECT_PREWARM_TIMEOUT_SECS,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Selected TTS service is not available at {tts_server}; "
            f"health check timed out after {_CONNECT_PREWARM_TIMEOUT_SECS}s."
        ) from exc

    if not is_ready:
        raise RuntimeError(
            f"Selected TTS service is not available at {tts_server}. Check that the TTS service is running and healthy."
        )


async def _ensure_services_ready_for_connection(
    config: dict,
    example: dict,
    *,
    is_realtime: bool,
) -> None:
    """Verify selected services before transport session handoff."""
    await _ensure_llm_ready_for_connection(config, example)
    await _ensure_asr_ready_for_connection(config, example)
    # Realtime text output sets Pipecat's skip_tts mode before the first turn,
    # so an unused speech synthesizer must not block that session at handoff.
    if not (is_realtime and config.get("output_modalities") == ["text"]):
        await _ensure_tts_ready_for_connection(config, example)


def _trusted_realtime_model_max_output_tokens(entry: dict) -> int | None:
    """Return the catalog-declared Realtime output ceiling for one LLM route."""
    value = entry.get("realtime_max_output_tokens")
    if value is None:
        if entry.get("supports_tokenize") is True:
            raise RuntimeError(
                "A Realtime LLM route with supports_tokenize=true must declare realtime_max_output_tokens"
            )
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _REALTIME_GA_MAX_OUTPUT_TOKENS:
        raise RuntimeError("Realtime LLM catalog field realtime_max_output_tokens must be an integer from 1 to 4096")
    return value


def _sanitize_realtime_config(data: dict, fallback_example_key: str = "") -> dict:
    """Bind Realtime to the exact catalog defaults the pipeline will use."""
    config = _sanitize_session_config(data, fallback_example_key=fallback_example_key)
    selected = examples_registry.find(str(config.get("pipeline_mode", "")) or fallback_example_key)
    slots = set(selected.get("slots") or [])

    _, module_file = _example_with_module_file(selected["key"])
    prompt_key, prompt_content = resolve_prompt(
        module_file,
        str(config.get("prompt_content") or ""),
        str(config.get("prompt_key") or ""),
    )
    config["prompt_key"] = prompt_key
    config["prompt_content"] = prompt_content

    def _apply_selected_service(category: str, field_map: dict[str, str]) -> None:
        # Realtime model profiles already carry source- and platform-qualified
        # selectors. Hydrate from that exact entry so connection setup never
        # performs reachability-based discovery for unrelated local recipes.
        entry = _trusted_realtime_service_entry(config, category)
        for catalog_field, body_field in field_map.items():
            value = entry.get(catalog_field)
            if value not in (None, ""):
                config.setdefault(body_field, value)

    if "llm" in slots:
        _apply_selected_service(
            "llm",
            {
                "model_id": "model_id",
                "base_url": "base_url",
                "system_prompt": "system_prompt",
                "max_tokens": "max_tokens",
                "temperature": "temperature",
                "extra_params": "extra_params",
            },
        )
        llm_entry = _trusted_realtime_service_entry(config, "llm")
        model_output_limit = _trusted_realtime_model_max_output_tokens(llm_entry)
        if model_output_limit is None:
            config.pop("realtime_model_max_output_tokens", None)
        else:
            config["realtime_model_max_output_tokens"] = model_output_limit
    if "asr" in slots:
        _apply_selected_service(
            "asr",
            {
                "server": "asr_server",
                "model": "asr_model",
                "function_id": "asr_function_id",
                "language_code": "asr_language_code",
            },
        )
    if "tts" in slots:
        _apply_selected_service(
            "tts",
            {
                "server": "tts_server",
                "voice_id": "tts_voice_id",
                "function_id": "tts_function_id",
                "model": "tts_model",
                "synthesis_mode": "tts_synthesis_mode",
                "language_code": "tts_language_code",
            },
        )
    if selected["key"] in {"omni-assistant", "omni-assistant-subagents"}:
        config.setdefault(
            "max_tokens",
            min(parse_env_int("OMNI_MAX_TOKENS", 8192, min_value=64), 4096),
        )
        if selected["key"] == "omni-assistant-subagents" or parse_env_bool(
            "OMNI_EMIT_TRANSCRIPTIONS",
            default=True,
        ):
            config["realtime_input_transcription_model"] = str(config.get("model_id") or "")
    return config


def _trusted_realtime_service_entry(config: dict, category: str) -> dict:
    """Resolve the exact built-in service entry that hydrated a Realtime route."""
    selector = config.get(f"{category}_id")
    if isinstance(selector, str) and selector:
        entry = load_service_entry_by_id(category, selector)
        if not entry:
            raise RuntimeError(f"Selected Realtime {category.upper()} service {selector!r} is not trusted")
        return entry
    entry = load_service_entry(category, "")
    if not entry:
        raise RuntimeError(f"No trusted Realtime {category.upper()} service is configured")
    return entry


def _validate_realtime_service_route(
    config: dict,
    entry: dict,
    *,
    category: str,
    fields: tuple[tuple[str, str], ...],
) -> None:
    """Ensure endpoint/function/model routing still matches its trusted entry."""
    for runtime_field, catalog_field in fields:
        actual = str(config.get(runtime_field) or "")
        expected = str(entry.get(catalog_field) or "")
        if actual != expected:
            raise RuntimeError(
                f"Realtime {category.upper()} route field {runtime_field!r} does not match the selected catalog"
            )


def _transcription_language_aliases(languages: list[object]) -> tuple[tuple[str, str], ...]:
    """Build exact BCP-47 aliases plus only unambiguous ISO-639 base aliases."""
    canonical_by_key: dict[str, str] = {}
    canonical_by_base: dict[str, set[str]] = {}
    for raw_language in languages:
        if not isinstance(raw_language, str) or not raw_language.strip():
            continue
        canonical = normalize_lang_code(raw_language.strip())
        key = canonical.lower()
        canonical_by_key[key] = canonical
        canonical_by_base.setdefault(key.split("-", 1)[0], set()).add(canonical)
    for base, candidates in canonical_by_base.items():
        if len(candidates) == 1:
            canonical_by_key.setdefault(base, next(iter(candidates)))
    return tuple(sorted(canonical_by_key.items()))


async def _resolve_realtime_session_capabilities(
    config: dict,
    discover_output_voices: bool = True,
):
    """Resolve exact-route media capabilities before validating the session."""
    from realtime.session import AudioFormatCapability, RealtimeSessionCapabilities

    selected = _bind_example_context_by_key(str(config.get("pipeline_mode") or ""))
    slots = set(selected.get("slots") or [])
    function_tools = selected["key"] != "omni-assistant-subagents"
    if "llm" not in slots:
        raise RuntimeError(f"Realtime language-model inference is not configured for {selected['key']!r}")
    if "tts" not in slots:
        raise RuntimeError(f"Realtime audio output is not configured for {selected['key']!r}")

    llm_entry = _trusted_realtime_service_entry(config, "llm")
    model_output_limit = _trusted_realtime_model_max_output_tokens(llm_entry)
    _validate_realtime_service_route(
        config,
        llm_entry,
        category="llm",
        fields=(("base_url", "base_url"), ("model_id", "model_id")),
    )
    if config.get("realtime_model_max_output_tokens") != model_output_limit:
        raise RuntimeError("Realtime LLM model output limit does not match the selected catalog")

    tts_entry = _trusted_realtime_service_entry(config, "tts")
    _validate_realtime_service_route(
        config,
        tts_entry,
        category="tts",
        fields=(("tts_server", "server"), ("tts_function_id", "function_id"), ("tts_model", "model")),
    )
    tts_server = str(config.get("tts_server") or "")
    tts_voice = str(config.get("tts_voice_id") or "")
    tts_function_id = str(config.get("tts_function_id") or "")
    tts_model = str(config.get("tts_model") or "")
    if not tts_server or not tts_voice:
        raise RuntimeError("The selected Realtime TTS route is incomplete")
    voices = {tts_voice}
    if discover_output_voices:
        tts_catalog = peek_cached_tts_config(
            tts_server,
            tts_voice,
            tts_function_id,
            tts_model,
        )
        if tts_catalog is None:
            try:
                tts_catalog = await _run_blocking(
                    get_tts_config,
                    tts_server,
                    tts_voice,
                    tts_function_id,
                    tts_model,
                    timeout=_CONNECT_PREWARM_TIMEOUT_SECS,
                )
            except TimeoutError as exc:
                raise RuntimeError(f"The selected Realtime TTS voice catalog timed out for {tts_server!r}") from exc
        if not isinstance(tts_catalog, dict) or tts_catalog.get("error"):
            raise RuntimeError(f"The selected Realtime TTS voice catalog is unavailable for {tts_server!r}")
        catalog_voices = tts_catalog.get("voices")
        if not isinstance(catalog_voices, list):
            raise RuntimeError(f"The selected Realtime TTS voice catalog is invalid for {tts_server!r}")
        voices.update(
            voice["id"]
            for voice in catalog_voices
            if isinstance(voice, dict) and isinstance(voice.get("id"), str) and voice["id"]
        )

    transcription_models: frozenset[str] = frozenset()
    transcription_language_aliases: tuple[tuple[str, str], ...] = ()
    if "asr" in slots:
        asr_entry = _trusted_realtime_service_entry(config, "asr")
        _validate_realtime_service_route(
            config,
            asr_entry,
            category="asr",
            fields=(("asr_server", "server"), ("asr_function_id", "function_id"), ("asr_model", "model")),
        )
        asr_server = str(config.get("asr_server") or "")
        asr_model = str(config.get("asr_model") or "")
        asr_function_id = str(config.get("asr_function_id") or "")
        if not asr_server or not asr_model:
            raise RuntimeError("The selected Realtime ASR route is incomplete")
        transcription_models = frozenset({asr_model})
        asr_catalog = peek_cached_asr_config(asr_server, asr_model, asr_function_id)
        if isinstance(asr_catalog, dict) and not asr_catalog.get("error"):
            raw_languages = asr_catalog.get("languages")
            if isinstance(raw_languages, list):
                transcription_language_aliases = _transcription_language_aliases(raw_languages)

    return RealtimeSessionCapabilities(
        input_formats=frozenset(
            {
                AudioFormatCapability("audio/pcm", 24000),
                AudioFormatCapability("audio/pcma"),
                AudioFormatCapability("audio/pcmu"),
            }
        ),
        output_formats=frozenset(
            {
                AudioFormatCapability("audio/pcm", 24000),
                AudioFormatCapability("audio/pcma"),
                AudioFormatCapability("audio/pcmu"),
            }
        ),
        voices=frozenset(voices),
        supports_custom_voices=False,
        input_transcription_models=transcription_models,
        input_transcription_language_aliases=transcription_language_aliases,
        # A cascaded ASR service remains required to feed the LLM, but OpenAI's
        # input-transcription setting controls whether that transcript is
        # published to the client. The lifecycle observer reads the canonical
        # session on every turn, so public transcription can be disabled
        # without pretending the pipeline can run without ASR.
        supports_input_transcription_disable="asr" in slots,
        function_tools=function_tools,
        mcp_tools=function_tools,
        sequential_tool_calls=function_tools,
        truncation=(selected["key"] in _REALTIME_TRUNCATION_PIPELINES and llm_entry.get("supports_tokenize") is True),
    )


def _as_realtime_tool_schema(tool: dict) -> dict | None:
    """Project one trusted Chat Completions function into canonical Realtime shape."""
    function = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if function is None:
        return None
    name = function.get("name")
    if not isinstance(name, str) or not name:
        return None
    schema: dict = {
        "type": "function",
        "name": name,
        "parameters": function.get("parameters") if isinstance(function.get("parameters"), dict) else {},
    }
    if isinstance(function.get("description"), str):
        schema["description"] = function["description"]
    return schema


def _resolve_realtime_server_tools(config: dict, fallback_example_key: str) -> list[str]:
    _, module_file = _example_with_module_file(str(config.get("pipeline_mode", "")) or fallback_example_key)
    prompt = load_prompt_catalog(module_file).get(str(config.get("prompt_key", "")), {})
    if not isinstance(prompt, dict):
        return []
    raw_tools = prompt.get("tools_available")
    if not isinstance(raw_tools, list):
        return []
    return [name for name in raw_tools if isinstance(name, str) and name]


def _resolve_realtime_server_tool_schemas(config: dict, fallback_example_key: str) -> list[dict]:
    _, module_file = _example_with_module_file(str(config.get("pipeline_mode", "")) or fallback_example_key)
    catalog = load_tools_catalog(module_file)
    schemas: list[dict] = []
    for name in _resolve_realtime_server_tools(config, fallback_example_key):
        raw = catalog.get(name)
        schema = _as_realtime_tool_schema(raw) if isinstance(raw, dict) else None
        if schema is not None:
            schemas.append(schema)
    return schemas


def _resolve_realtime_delegate_tool_schemas(config: dict, fallback_example_key: str) -> list[dict]:
    selected = examples_registry.find(str(config.get("pipeline_mode", "")) or fallback_example_key)
    if selected["key"] != "frontend-backend-agent":
        return []
    from examples.frontend_backend_agent.airline.tools import CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL

    return [
        schema
        for tool in (CALL_BACKEND_TOOL, CANCEL_BACKEND_TOOL)
        if (schema := _as_realtime_tool_schema(tool)) is not None
    ]


def _resolve_realtime_delegate_tools(config: dict, fallback_example_key: str) -> list[str]:
    return [
        schema["name"]
        for schema in _resolve_realtime_delegate_tool_schemas(config, fallback_example_key)
        if isinstance(schema.get("name"), str)
    ]


_REALTIME_LANGUAGE_SELECTOR = re.compile(r"^[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*$")


def _validate_realtime_bootstrap_selectors(session: dict, fallback_example_key: str) -> None:
    """Reject selectors that do not resolve to the trusted registry and catalogs."""
    raw_nvidia = session.get("nvidia")
    if not isinstance(raw_nvidia, dict) or not raw_nvidia:
        return

    requested_pipeline = str(raw_nvidia.get("pipeline_mode") or fallback_example_key)
    selected = examples_registry.find(requested_pipeline)
    if requested_pipeline != selected["key"]:
        raise ValueError(f"Unknown or unavailable pipeline_mode {requested_pipeline!r}")
    _bind_example_context_by_key(selected["key"])
    slots = set(selected.get("slots") or [])

    for field, slot, category in (
        ("llm_id", "llm", "llm"),
        ("thinker_llm_id", "thinker-llm", "thinker-llm"),
        ("asr_id", "asr", "asr"),
        ("tts_id", "tts", "tts"),
    ):
        value = raw_nvidia.get(field)
        if value is None:
            continue
        if slot not in slots or not load_service_entry_by_id(category, value):
            raise ValueError(f"Unknown or unavailable {field} {value!r} for {selected['key']!r}")

    prompt_key = raw_nvidia.get("prompt_key")
    if prompt_key is not None:
        _, module_file = _example_with_module_file(selected["key"])
        prompt = load_prompt_catalog(module_file).get(prompt_key)
        if (
            not isinstance(prompt, dict)
            or not isinstance(prompt.get("content"), str)
            or prompt.get("internal") is True
            or prompt_key in examples_registry.agent_prompt_keys(selected["key"])
        ):
            raise ValueError(f"Unknown or unavailable prompt_key {prompt_key!r} for {selected['key']!r}")

    for field in ("asr_language_code", "tts_language_code"):
        language = raw_nvidia.get(field)
        if language is None:
            continue
        required_slot = "asr" if field == "asr_language_code" else "tts"
        if required_slot not in slots:
            raise ValueError(f"{field} is not available for {selected['key']!r}")
        if language.lower() != "auto" and _REALTIME_LANGUAGE_SELECTOR.fullmatch(language) is None:
            raise ValueError(f"Invalid {field} {language!r}")
        raw_nvidia[field] = "auto" if language.lower() == "auto" else normalize_lang_code(language)


def _resolve_realtime_model_route(
    requested_model: str | None,
    nvidia_overrides: dict,
    fallback_example_key: str,
):
    """Resolve one public model to an exact, catalog-bound Realtime route."""
    from realtime.gateway import RealtimeModelRoute
    from realtime.protocol import RealtimeProtocolError

    explicit_pipeline = nvidia_overrides.get("pipeline_mode")
    pipeline_hint = str(explicit_pipeline or (fallback_example_key if requested_model is None else ""))
    try:
        profile = examples_registry.resolve_realtime_model_profile(
            requested_model or "",
            pipeline_mode=pipeline_hint,
        )
    except examples_registry.RealtimeModelProfileNotAvailable as exc:
        raise RealtimeProtocolError(
            message=str(exc),
            code="model_not_available",
            param="model" if requested_model is not None else "session.nvidia.pipeline_mode",
        ) from exc

    profile_pipeline = profile["pipeline_mode"]
    requested_pipeline = nvidia_overrides.get("pipeline_mode")
    if requested_pipeline is not None and requested_pipeline != profile_pipeline:
        raise RealtimeProtocolError(
            message=(
                f"Model {profile['model']!r} is configured for pipeline {profile_pipeline!r}, "
                f"not {requested_pipeline!r}"
            ),
            code="model_not_available",
            param="model",
        )

    profile_prompt = profile["selectors"].get("prompt_key")
    requested_prompt = nvidia_overrides.get("prompt_key")
    if requested_prompt is not None and requested_prompt != profile_prompt:
        raise RealtimeProtocolError(
            message=(
                f"Model {profile['model']!r} is configured for prompt {profile_prompt!r}, "
                f"not {requested_prompt!r}; use session.instructions to customize behavior"
            ),
            code="model_not_available",
            param="session.nvidia.prompt_key",
        )

    selectors = examples_registry.materialize_realtime_profile_selectors(profile)
    selectors["pipeline_mode"] = profile_pipeline
    canonical_overrides = copy.deepcopy(nvidia_overrides)
    for selector_name in ("llm_id", "thinker_llm_id", "asr_id", "tts_id"):
        override = canonical_overrides.get(selector_name)
        if override is None:
            continue
        try:
            canonical_overrides[selector_name] = examples_registry.canonicalize_realtime_service_selector(
                profile_pipeline,
                selector_name,
                override,
            )
        except RuntimeError as exc:
            raise RealtimeProtocolError(
                message=str(exc),
                code="invalid_value",
                param=f"session.nvidia.{selector_name}",
            ) from exc
    selectors.update(canonical_overrides)
    try:
        _validate_realtime_bootstrap_selectors({"nvidia": selectors}, fallback_example_key)
    except ValueError as exc:
        raise RealtimeProtocolError(
            message=str(exc),
            code="invalid_value",
            param="session.nvidia",
        ) from exc
    return RealtimeModelRoute(model=profile["model"], runtime_config=selectors)


def _canonical_realtime_secret_session(controller) -> dict:
    """Bind a client secret to the complete validated private session template."""
    from realtime.bootstrap import TRUSTED_NVIDIA_SELECTOR_FIELDS

    # The encrypted grant must carry the same normalized/defaulted session that
    # the mint response advertises. Use the controller's private view so hosted
    # MCP credentials survive redemption, then remove response-only identity
    # fields which are regenerated for the new WebSocket session.
    canonical = controller.session.public_view()
    canonical.pop("id", None)
    canonical.pop("object", None)
    canonical["type"] = "realtime"
    selectors = {
        field: copy.deepcopy(controller.runtime_config[field])
        for field in TRUSTED_NVIDIA_SELECTOR_FIELDS
        if controller.runtime_config.get(field) not in (None, "")
    }
    canonical["nvidia"] = selectors
    return canonical


def create_app(host: str = "localhost", prompt_file: str = "", *, realtime_only: bool = False) -> FastAPI:
    """Build and return the FastAPI application with all routes."""
    if prompt_file:
        os.environ["PROMPT_FILE_PATH"] = prompt_file

    selected_example = examples_registry.find()
    fallback_example_key = selected_example["key"]
    _activate_example_catalog_by_key(fallback_example_key)
    logger.info(
        f"Active selection: {fallback_example_key} "
        f"(locked={examples_registry.is_locked()}, "
        f"examples={examples_registry.visible_example_keys()}, "
        f"transports={examples_registry.visible_transports()})"
    )

    handler = None if realtime_only else SmallWebRTCRequestHandler(host=host)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if handler is not None:
            await handler.close()

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _resolve_example(config: dict) -> dict:
        """Return the active registry entry. ``examples_registry.find()`` honors any selection lock."""
        return examples_registry.find(config.get("pipeline_mode", ""))

    async def _readiness_check_or_503(config: dict, log_label: str) -> JSONResponse | None:
        """Return a 503 response if any selected service is not ready, else ``None``."""
        try:
            await _ensure_services_ready_for_connection(
                config,
                _resolve_example(config),
                is_realtime=False,
            )
        except RuntimeError as exc:
            logger.warning(f"Rejecting {log_label} during service readiness check: {exc}")
            return JSONResponse(status_code=503, content={"info": str(exc)})
        return None

    @app.get("/api/deployment")
    async def get_deployment():
        deployment = examples_registry.metadata(examples_registry.find(fallback_example_key))
        selector_options = examples_registry.visible_options()
        return _deployment_response(deployment, selector_options)

    # ---- Session config (avoids long URLs for prompts / system_prompt) ----

    @app.post("/api/session-config")
    async def create_session_config(request: Request):
        """Store pipeline config server-side, return a short session_id."""
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        try:
            config = _sanitize_session_config(await request.json(), fallback_example_key=fallback_example_key)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if (failure := await _readiness_check_or_503(config, "session config")) is not None:
            return failure
        return {"session_id": _store_session_config(config, fallback_example_key=fallback_example_key)}

    @app.post("/api/start")
    async def start_bot(request: Request):
        """Run readiness checks before starting a WebRTC session."""
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        try:
            config = _sanitize_session_config(await request.json(), fallback_example_key=fallback_example_key)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"detail": str(exc)})
        if (failure := await _readiness_check_or_503(config, "WebRTC start")) is not None:
            return failure
        session_id = _store_session_config(config, fallback_example_key=fallback_example_key)
        return {"webrtcUrl": f"/api/offer?session_id={session_id}"}

    @app.post("/api/sessions/{session_id}/attachments")
    async def upload_attachment(
        session_id: str,
        file: Annotated[UploadFile, File()],
        kind: str = Query(default="image"),
    ):
        """Store an uploaded media attachment for a live voice session."""
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        if (failure := _session_capability_error(session_id, "attachments")) is not None:
            return failure
        try:
            data = await _read_upload_file_with_limit(file)
            attachment = store_attachment(
                session_id=session_id,
                kind=kind,
                name=file.filename or "attachment",
                content_type=file.content_type or "",
                data=data,
            )
        except ValueError as exc:
            status_code = 413 if "limit" in str(exc).lower() else 400
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
        logger.debug(
            "Stored media attachment "
            f"(session_id={session_id[:8]}..., kind={attachment.kind}, bytes={len(attachment.data)})"
        )
        return attachment.metadata()

    @app.get("/api/webcam-config")
    async def get_webcam_config():
        """Return browser webcam capture defaults for capability-driven UI."""
        return webcam_client_config()

    @app.post("/api/sessions/{session_id}/webcam/frames")
    async def upload_webcam_frame(
        session_id: str,
        file: Annotated[UploadFile, File()],
    ):
        """Store the latest browser webcam snapshot for a live voice session."""
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        if (failure := _session_capability_error(session_id, "webcam")) is not None:
            return failure
        try:
            data = await _read_upload_file_with_limit(file)
            frame = store_webcam_frame(
                session_id=session_id,
                name=file.filename or "webcam-frame.jpg",
                content_type=file.content_type or "image/jpeg",
                data=data,
            )
        except ValueError as exc:
            status_code = 413 if "limit" in str(exc).lower() else 400
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
        logger.debug(
            "Stored webcam frame "
            f"(session_id={session_id}, name={frame.name}, bytes={len(frame.data)}, sequence={frame.sequence})"
        )
        return frame.metadata()

    @app.post("/api/sessions/{session_id}/webcam/capture")
    async def upload_webcam_capture(
        session_id: str,
        file: Annotated[UploadFile, File()],
        request_id: Annotated[str, Form()],
    ):
        """Store an agent-requested high-resolution webcam snapshot for focused analysis.

        Unlike a streamed webcam frame, this lands in the attachment store (so the media
        analyzer can read it via get_attachment) tagged source="capture", so it never
        becomes the Speaker's default visual source.
        """
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        if (failure := _session_capability_error(session_id, "webcam")) is not None:
            return failure
        if not consume_capture_request(session_id, request_id):
            return JSONResponse(status_code=409, content={"detail": "capture request is missing, stale, or invalid"})
        try:
            data = await _read_upload_file_with_limit(file)
            attachment = store_attachment(
                session_id=session_id,
                kind="image",
                name=file.filename or "focused-capture.jpg",
                content_type=file.content_type or "image/jpeg",
                data=data,
                source="capture",
            )
        except ValueError as exc:
            status_code = 413 if "limit" in str(exc).lower() else 400
            return JSONResponse(status_code=status_code, content={"detail": str(exc)})
        logger.debug(f"Stored high-res webcam capture (session_id={session_id[:8]}..., bytes={len(attachment.data)})")
        return attachment.metadata()

    # ---- WebRTC signaling ----

    @app.post("/api/offer")
    async def offer(
        request: SmallWebRTCRequest,
        background_tasks: BackgroundTasks,
        session_id: str = Query(default=""),
        pipeline_mode: str = Query(default=""),
    ):
        if _multi_worker_mode_enabled():
            return _multi_worker_session_config_response()
        config = _resolve_config(session_id, fallback_example_key=fallback_example_key, pipeline_mode=pipeline_mode)
        example = _resolve_example(config)
        selected = examples_registry.find(config.get("pipeline_mode", fallback_example_key))
        bot_fn = examples_registry.resolve_bot(selected)
        if session_id:
            _active_session_configs[session_id] = dict(config)

        async def run_bot_session(runner_args: SmallWebRTCRunnerArguments) -> None:
            try:
                await bot_fn(runner_args)
            finally:
                if session_id:
                    _active_session_configs.pop(session_id, None)

        async def on_connection(connection: SmallWebRTCConnection):
            body = _build_rtvi_runner_body(
                config,
                session_id=session_id,
                request_data=request.request_data,
            )
            _bind_example_context_by_key(example["key"])
            runner_args = SmallWebRTCRunnerArguments(webrtc_connection=connection, body=body)
            runner_args.pipeline_idle_timeout_secs = parse_env_int("PIPELINE_IDLE_TIMEOUT_SECS", 600, min_value=300)
            background_tasks.add_task(run_bot_session, runner_args)

        return await handler.handle_web_request(
            request=request,
            webrtc_connection_callback=on_connection,
        )

    @app.patch("/api/offer")
    async def ice_candidate(request: SmallWebRTCPatchRequest):
        await handler.handle_patch_request(request)
        return {"status": "success"}

    # ---- WebSocket transport ----

    @app.websocket("/api/ws")
    async def websocket_endpoint(
        websocket: WebSocket,
        session_id: str = Query(default=""),
        pipeline_mode: str = Query(default=""),
    ):
        if session_id and _multi_worker_mode_enabled():
            await websocket.close(code=1013, reason=_MULTI_WORKER_SESSION_CONFIG_MESSAGE)
            return
        stream_id = session_id or "-"
        with logger.contextualize(stream_id=stream_id):
            config = _resolve_config(session_id, fallback_example_key=fallback_example_key, pipeline_mode=pipeline_mode)
            example = _resolve_example(config)
            if not session_id:
                try:
                    await _ensure_services_ready_for_connection(
                        config,
                        example,
                        is_realtime=False,
                    )
                except RuntimeError as exc:
                    logger.warning(f"Rejecting WebSocket start during service readiness check: {exc}")
                    await websocket.close(code=1011, reason=str(exc))
                    return

            await websocket.accept()
            selected = examples_registry.find(config.get("pipeline_mode", fallback_example_key))
            bot_fn = examples_registry.resolve_bot(selected)
            _bind_example_context_by_key(example["key"])
            if session_id:
                _active_session_configs[session_id] = dict(config)

            runner_args = SimpleNamespace(
                websocket=websocket,
                body=_build_rtvi_runner_body(config, session_id=session_id),
                handle_sigint=False,
                pipeline_idle_timeout_secs=parse_env_int("PIPELINE_IDLE_TIMEOUT_SECS", 600, min_value=300),
            )
            try:
                await bot_fn(runner_args)
            except Exception as e:
                logger.error(f"WebSocket session error: {e}")
            finally:
                if session_id:
                    _active_session_configs.pop(session_id, None)
                with contextlib.suppress(Exception):
                    await websocket.close()

    # ---- OpenAI Realtime bootstrap and JSON WebSocket ----

    def _realtime_http_error(
        *,
        status_code: int,
        message: str,
        code: str,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> JSONResponse:
        error: dict[str, str] = {
            "message": message,
            "type": "invalid_request_error" if status_code < 500 else "server_error",
            "code": code,
        }
        if param is not None:
            error["param"] = param
        return JSONResponse(status_code=status_code, content={"error": error}, headers=headers)

    async def _prepare_initial_realtime_runtime(config: dict) -> dict:
        """Resolve multilingual settings at the first actionable event."""
        selected = examples_registry.find(str(config.get("pipeline_mode", "")) or fallback_example_key)
        if selected["key"] != "multilingual-assistant":
            return config

        raw_language = str(config.get("asr_language_code") or "").strip()
        if not raw_language or raw_language.lower() == "auto":
            raw_language = examples_registry.default_session_language(selected["key"])
        language = normalize_lang_code(raw_language)
        config["asr_language_code"] = language
        if config.get("output_modalities") == ["text"]:
            return config

        tts_server, tts_voice, tts_function_id, tts_model = _resolve_tts_selection(
            str(config.get("tts_server") or ""),
            str(config.get("tts_voice_id") or ""),
            str(config.get("tts_function_id") or ""),
            str(config.get("tts_model") or ""),
        )
        tts_catalog = await _run_blocking(
            get_tts_config,
            tts_server,
            tts_voice,
            tts_function_id,
            tts_model,
            timeout=_CONNECT_PREWARM_TIMEOUT_SECS,
        )
        if not isinstance(tts_catalog, dict) or tts_catalog.get("error"):
            raise RuntimeError(f"The selected TTS voice catalog is unavailable for {tts_server!r}")
        resolved_voice = resolve_voice_for_language(
            language,
            tts_voice,
            server=tts_server,
            function_id=tts_function_id,
            model=tts_model,
        )
        if not resolved_voice:
            raise RuntimeError(f"No TTS voice is available for the fixed session language {language!r}")
        config["tts_voice_id"] = resolved_voice
        return config

    @app.post("/v1/realtime/client_secrets")
    async def create_realtime_client_secret(request: Request):
        """Mint a stateless client secret bound to a validated Realtime session template."""
        from realtime.auth import (
            authenticate_realtime_master_key,
            configured_realtime_api_key,
            issue_realtime_client_secret,
        )
        from realtime.bootstrap import parse_realtime_client_secret_request
        from realtime.gateway import create_realtime_controller
        from realtime.protocol import RealtimeProtocolError, strict_json_loads

        api_key = configured_realtime_api_key()
        if api_key is None:
            return _realtime_http_error(
                status_code=503,
                message="Realtime client-secret minting is not configured",
                code="realtime_auth_not_configured",
            )
        if not authenticate_realtime_master_key(request.headers, api_key=api_key):
            return _realtime_http_error(
                status_code=401,
                message="Invalid Realtime API key",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

        raw_body = bytearray()
        async for chunk in request.stream():
            raw_body.extend(chunk)
            if len(raw_body) > _MAX_REALTIME_CLIENT_SECRET_REQUEST_BYTES:
                return _realtime_http_error(
                    status_code=413,
                    message="Realtime client-secret request is too large",
                    code="request_too_large",
                )
        try:
            body = strict_json_loads(raw_body) if raw_body else {}
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return _realtime_http_error(
                status_code=400,
                message="Request body must be valid JSON",
                code="invalid_json",
                param="body",
            )

        try:
            parsed = parse_realtime_client_secret_request(body)
            controller = await create_realtime_controller(
                sanitize_session_config=lambda data, **kwargs: _sanitize_realtime_config(data, **kwargs),
                initial_session=parsed.session,
                resolve_server_tools=lambda config: _resolve_realtime_server_tools(config, fallback_example_key),
                resolve_delegate_tools=lambda config: _resolve_realtime_delegate_tools(config, fallback_example_key),
                resolve_server_tool_schemas=lambda config: _resolve_realtime_server_tool_schemas(
                    config,
                    fallback_example_key,
                ),
                resolve_delegate_tool_schemas=lambda config: _resolve_realtime_delegate_tool_schemas(
                    config,
                    fallback_example_key,
                ),
                resolve_session_capabilities=_resolve_realtime_session_capabilities,
                resolve_model_route=lambda model, selectors: _resolve_realtime_model_route(
                    model,
                    selectors,
                    fallback_example_key,
                ),
                default_example_key=fallback_example_key,
                default_pipeline_mode=fallback_example_key,
            )
            issued_at = int(time.time())
            expires_at = issued_at + parsed.ttl_seconds
            secret = issue_realtime_client_secret(
                api_key=api_key,
                session=_canonical_realtime_secret_session(controller),
                issued_at=issued_at,
                expires_at=expires_at,
            )
        except RealtimeProtocolError as exc:
            return _realtime_http_error(
                status_code=400,
                message=exc.message,
                code=exc.code,
                param=exc.param,
            )
        except ValueError as exc:
            logger.warning(f"Realtime client-secret request rejected: {exc}")
            return _realtime_http_error(
                status_code=400,
                message=str(exc),
                code="invalid_session",
                param="session",
            )
        except RuntimeError as exc:
            logger.warning(f"Realtime client-secret media capability resolution failed: {exc}")
            return _realtime_http_error(
                status_code=503,
                message="The selected Realtime media services are not available",
                code="services_not_ready",
            )
        except Exception:
            logger.exception("Failed to create Realtime client secret")
            return _realtime_http_error(
                status_code=500,
                message="The Realtime client secret could not be created",
                code="client_secret_creation_failed",
            )

        effective_session = controller.public_session()
        return JSONResponse(
            content={"value": secret, "expires_at": expires_at, "session": effective_session},
            headers={"Cache-Control": "no-store"},
        )

    @app.websocket("/v1/realtime")
    async def realtime_websocket_endpoint(websocket: WebSocket):
        """OpenAI Realtime–compatible authenticated session endpoint.

        Immediately creates the default session. An optional initial
        ``session.update`` is validated before modality-aware readiness checks
        hand the socket to the selected example bot (``protocol=realtime``).
        """
        from realtime import RealtimeSessionController, handle_realtime_websocket
        from realtime.auth import RealtimeAuthenticationError, authenticate_realtime_websocket
        from realtime.protocol import RealtimeProtocolError

        try:
            authentication = authenticate_realtime_websocket(websocket.headers)
        except RealtimeAuthenticationError:
            await websocket.close(code=1008, reason="realtime authentication failed")
            return
        initial_session = authentication.claims.session if authentication.claims is not None else None

        async def _ensure_ready(config: dict) -> None:
            await _ensure_services_ready_for_connection(
                config,
                _resolve_example(config),
                is_realtime=True,
            )

        async def _start_bot(
            ws: WebSocket,
            config: dict,
            controller: RealtimeSessionController,
        ) -> None:
            session_id = controller.id
            close_code = 1000
            close_reason = "realtime session complete"
            close_on_return = True
            try:
                selected = examples_registry.find(config.get("pipeline_mode", fallback_example_key))
                bot_fn = examples_registry.resolve_bot(selected)
                _bind_example_context_by_key(selected["key"])
                if session_id:
                    _active_session_configs[session_id] = dict(config)
                runner_args = SimpleNamespace(
                    websocket=ws,
                    body={
                        **config,
                        "protocol": "realtime",
                        "realtime_controller": controller,
                        "session_id": session_id,
                    },
                    handle_sigint=False,
                    pipeline_idle_timeout_secs=parse_env_int(
                        "PIPELINE_IDLE_TIMEOUT_SECS",
                        600,
                        min_value=300,
                    ),
                )
                await bot_fn(runner_args)
            except asyncio.CancelledError:
                # The outer Realtime lifetime owns deadline cancellation and
                # its clean expiry close. Avoid replacing that terminal reason
                # from this nested pipeline wrapper.
                close_on_return = False
                raise
            except WebSocketDisconnect:
                logger.info(f"Realtime pipeline client disconnected session_id={session_id}")
            except Exception:
                close_code = 1011
                close_reason = "realtime pipeline failed"
                logger.exception(f"Realtime pipeline session failed session_id={session_id}")
                event = RealtimeProtocolError(
                    message="The Realtime pipeline failed",
                    code="pipeline_runtime_error",
                    error_type="server_error",
                ).to_event()
                with contextlib.suppress(RuntimeError, WebSocketDisconnect):
                    await ws.send_text(json.dumps(event))
            finally:
                if session_id:
                    _active_session_configs.pop(session_id, None)
                if close_on_return:
                    with contextlib.suppress(Exception):
                        await ws.close(code=close_code, reason=close_reason)

        await handle_realtime_websocket(
            websocket,
            sanitize_session_config=_sanitize_realtime_config,
            initial_session=initial_session,
            prepare_initial_runtime=_prepare_initial_realtime_runtime,
            ensure_services_ready=_ensure_ready,
            start_bot=_start_bot,
            resolve_server_tools=lambda config: _resolve_realtime_server_tools(config, fallback_example_key),
            resolve_delegate_tools=lambda config: _resolve_realtime_delegate_tools(config, fallback_example_key),
            resolve_server_tool_schemas=lambda config: _resolve_realtime_server_tool_schemas(
                config,
                fallback_example_key,
            ),
            resolve_delegate_tool_schemas=lambda config: _resolve_realtime_delegate_tool_schemas(
                config,
                fallback_example_key,
            ),
            resolve_session_capabilities=_resolve_realtime_session_capabilities,
            resolve_model_route=lambda model, selectors: _resolve_realtime_model_route(
                model,
                selectors,
                fallback_example_key,
            ),
            default_example_key=fallback_example_key,
            default_pipeline_mode=fallback_example_key,
        )

    # ---- Prompt catalog (read-only, scoped to the active example) ----

    @app.get("/api/prompts")
    async def get_prompts(pipeline_mode: str = Query(default="")):
        example_key = pipeline_mode or fallback_example_key
        _, module_file = _example_with_module_file(example_key)
        catalog = load_prompt_catalog(module_file)
        registry_default_key = examples_registry.prompt_default_key(example_key)
        default_key = registry_default_key if registry_default_key in catalog else default_prompt_key(catalog)
        hidden_prompt_keys = examples_registry.agent_prompt_keys(example_key)
        prompts = [
            {
                "key": key,
                "description": val.get("description", ""),
                "content": val.get("content", ""),
                "default": key == default_key,
                "builtIn": True,
                "selectable": key not in hidden_prompt_keys,
                "scope": "agent" if key in hidden_prompt_keys else "session",
                "tools": [t for t in (val.get("tools_available") or []) if isinstance(t, str)],
            }
            for key, val in catalog.items()
            if isinstance(val, dict) and "content" in val and val.get("internal") is not True
        ]
        agent_prompts = catalog.get("agent_prompts")
        if isinstance(agent_prompts, dict):
            for agent_key, agent_entries in agent_prompts.items():
                if not isinstance(agent_entries, dict):
                    continue
                for prompt_key, val in agent_entries.items():
                    if not isinstance(val, dict) or "content" not in val:
                        continue
                    prompts.append(
                        {
                            "key": f"{agent_key}.{prompt_key}",
                            "description": val.get("description", ""),
                            "content": val.get("content", ""),
                            "default": False,
                            "builtIn": True,
                            "selectable": False,
                            "scope": "agent",
                            "agent": agent_key,
                            "promptName": prompt_key,
                            "tools": [],
                        }
                    )
        return prompts

    @app.get("/api/tools")
    async def get_tools(pipeline_mode: str = Query(default="")):
        _, module_file = _example_with_module_file(pipeline_mode or fallback_example_key)
        catalog = load_tools_catalog(module_file)
        tools: list[dict] = []
        for name, entry in catalog.items():
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function") if isinstance(entry.get("function"), dict) else {}
            tools.append(
                {
                    "name": name,
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        return tools

    @app.get("/api/subagents")
    async def get_subagents(pipeline_mode: str = Query(default="")):
        """Subagents the example declares in ``subagents.yaml`` (empty if none)."""
        _, module_file = _example_with_module_file(pipeline_mode or fallback_example_key)
        registry = load_subagent_registry(Path(module_file).parent / "subagents.yaml")
        return registry.to_payload()

    # ---- Service catalog (services.cloud.yaml or services.local.yaml) ----

    @app.get("/api/services")
    async def get_services(pipeline_mode: str = Query(default="")):
        _bind_example_context_by_key(pipeline_mode or fallback_example_key)
        return build_services_api_response()

    # ---- TTS config (voices & languages from the TTS service) ----

    @app.get("/api/tts-config")
    async def tts_config(
        pipeline_mode: str = Query(default=""),
        llm_id: str = Query(default=""),
        server: str = Query(default=""),
        voice_id: str = Query(default=""),
        function_id: str = Query(default=""),
        model: str = Query(default=""),
        asr_server: str = Query(default=""),
        asr_model: str = Query(default=""),
        asr_function_id: str = Query(default=""),
    ):
        _bind_example_context_by_key(pipeline_mode or fallback_example_key)
        if asr_server or asr_model or asr_function_id:
            default_asr_server, default_asr_model, default_asr_function_id = _get_default_asr_catalog()
            if llm_id.startswith("custom-"):
                llm_entry = {}
            elif llm_id:
                llm_entry = load_service_entry_by_id("llm", llm_id) or load_service_entry("llm", "")
            else:
                llm_entry = load_service_entry("llm", "")
            llm_supported_languages = llm_entry.get("supported_languages") if llm_entry else None
            tts_server, tts_voice, tts_function_id, tts_model = _resolve_tts_selection(
                server, voice_id, function_id, model
            )
            try:
                return await _run_blocking(
                    build_session_languages,
                    asr_server or default_asr_server,
                    asr_model or default_asr_model,
                    asr_function_id or default_asr_function_id,
                    tts_server,
                    tts_voice,
                    tts_function_id,
                    tts_model,
                    llm_supported_languages,
                    timeout=_CONNECT_PREWARM_TIMEOUT_SECS,
                )
            except TimeoutError:
                logger.warning(
                    "tts-config ASR/TTS/LLM intersection timed out after {}s",
                    _CONNECT_PREWARM_TIMEOUT_SECS,
                )
                return JSONResponse(
                    status_code=504,
                    content={"detail": f"Language catalog timed out after {_CONNECT_PREWARM_TIMEOUT_SECS}s"},
                )

        if server:
            tts_server, tts_voice, tts_function_id, tts_model = _resolve_tts_selection(
                server, voice_id, function_id, model
            )
            # Cache-first (sync): only prewarm_tts when this TTS routing key is cold.
            cached = peek_cached_tts_config(tts_server, tts_voice, tts_function_id, tts_model)
            if cached is not None:
                config_store.set("tts", cached)
                return cached
            return await _run_blocking(
                prewarm_tts,
                tts_server,
                tts_voice,
                tts_function_id,
                tts_model,
            )
        return config_store.get("tts", {"languages": [], "voices": []})

    # ---- WebRTC ICE servers (TURN credentials) ----

    @app.get("/api/ice-servers")
    async def ice_servers(request: Request):
        """Return ICE server config for the WebRTC client.

        Empty list when TURN is not configured (client falls back to host-only
        candidates, which is fine for local/LAN deployments).
        """
        return {"iceServers": _build_ice_servers(request)}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- Static client UI ----

    if realtime_only:
        pass
    elif CLIENT_DIST.is_dir():
        logger.info(f"Serving client UI from {CLIENT_DIST}")
        index_path = CLIENT_DIST / "index.html"

        @app.get("/")
        async def root():
            return FileResponse(index_path, headers=_INDEX_NO_CACHE_HEADERS)

        @app.get("/{path:path}")
        async def client_files(path: str = ""):
            if path.startswith("api/"):
                return
            file = CLIENT_DIST / path
            if file.is_file():
                return FileResponse(file)
            return FileResponse(index_path, headers=_INDEX_NO_CACHE_HEADERS)
    else:
        logger.warning(f"Client build not found at {CLIENT_DIST}. Run 'npm run build' in client/ to enable the UI.")

        @app.get("/")
        async def root():
            return {
                "status": "running",
                "hint": "Build the client UI: cd client && npm run build",
            }

    return app


def create_realtime_app(host: str = "localhost", prompt_file: str = "") -> FastAPI:
    """Build the standalone app with only Realtime and liveness routes exposed."""
    app = create_app(host=host, prompt_file=prompt_file, realtime_only=True)
    exposed = {"/health", "/v1/realtime/client_secrets", "/v1/realtime"}
    app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", "") in exposed]
    app.title = "Nemotron Voice Agent Realtime API"
    return app


def main():
    """Parse CLI args and start the uvicorn server."""
    parser = argparse.ArgumentParser(description="Pipeline NVIDIA Server")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="",
        help="Optional prompt catalog YAML path to use instead of the active example's local prompts.yaml",
    )
    parser.add_argument("--tls-cert", type=str, help="Path to TLS certificate file")
    parser.add_argument("--tls-key", type=str, help="Path to TLS key file")
    parser.add_argument(
        "--workers",
        type=lambda value: _parse_min_int(value, 1),
        default=None,
        help="Number of workers to use for the server",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    logger.remove()
    logger.configure(extra={"stream_id": "-"})
    logger.add(
        sys.stderr,
        level="TRACE" if args.verbose else "DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "{name}:{function}:{line} - "
            "[stream_id={extra[stream_id]}] "
            "<level>{message}</level>"
        ),
    )

    workers = args.workers if args.workers is not None else parse_env_int("UVICORN_WORKERS", 1, min_value=1)
    os.environ["UVICORN_WORKERS"] = str(workers)

    app = None
    if workers == 1:
        app = create_app(host=args.host, prompt_file=args.prompt_file)

    tls_enabled = parse_env_bool("PIPELINE_TLS", default=True)
    ssl_kwargs: dict = {}
    scheme = "http"
    if tls_enabled:
        if args.tls_cert and args.tls_key:
            ssl_kwargs = {"ssl_certfile": args.tls_cert, "ssl_keyfile": args.tls_key}
        else:
            cert_dir = Path(__file__).resolve().parent.parent / ".certs"
            from utils import ensure_self_signed_cert

            cert_file, key_file = ensure_self_signed_cert(cert_dir)
            ssl_kwargs = {"ssl_certfile": cert_file, "ssl_keyfile": key_file}
        scheme = "https"

    ui_url = f"{scheme}://{args.host}:{args.port}/"
    if CLIENT_DIST.is_dir():
        logger.info(f"Server ready -> {ui_url} (workers={workers})")
    else:
        logger.info(f"Server ready (API only) -> {ui_url} (workers={workers})")
        logger.info("Build the client: cd client && npm run build")

    if workers > 1:
        _run_multi_worker(args, workers, ssl_kwargs)
    else:
        _run_single_worker(args, app, ssl_kwargs)


def app_factory():
    """Build an app instance for uvicorn multi-worker mode."""
    config = WorkerAppConfig.from_argv(sys.argv[1:])
    return create_app(host=config.host, prompt_file=config.prompt_file)


if __name__ == "__main__":
    main()
