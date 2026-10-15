# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture what an FBV OpenAI Realtime (eval) session gets today, for ``test_fbv_eval_pins.py``.

Plan ``misc/frontend-backend-verdict/fba-removal-plan.md``, section 2.2.4. The recorded values live
in ``pins/eval_pins.json`` (recorded at ``84affa7``). To print a fresh capture and diff it:

    PYTHONPATH=src:tests/unit/frontend_backend_verdict python _fbv_eval_capture.py

The capture never touches the network: the reachability helpers fail if called, and TTS voice
discovery is stubbed.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PINS_FILE = HERE / "pins" / "eval_pins.json"
MODEL = "nvidia/nemotron-realtime-frontend-backend-verdict"
EXAMPLE = "frontend-backend-verdict-agent"
TAU_DOMAINS = ("airline", "retail", "telecom", "banking_knowledge", "mock")
PLATFORMS = ("cloud", "server", "singlegpu")
#: The Realtime master key used only inside these captures.
TEST_REALTIME_KEY = "fbv-pin-test-key"
#: Every switch the FBA-removal plan adds (section 2.3). Set "on" to prove the eval-mode hard-off.
SWITCHES_ON = {
    "FBV_REALTIME_SPEECH_SELECTORS": "true",
    "FBV_REALTIME_TURN_MODES": "server_vad,manual,semantic_vad",
    "FBV_REALTIME_MCP": "true",
    "FBV_REALTIME_TRACING": "true",
    "ENABLE_TRACING": "true",
    "ENABLE_WELCOME_MESSAGE": "false",
}
#: Session templates minted into ``ek_`` keys (plan 2.2.4, cases a-c, plus an unknown LLM id).
EK_TEMPLATES: dict[str, dict[str, Any]] = {
    "a_no_template": {"model": MODEL},
    "b_standard_fields": {
        "model": MODEL,
        "instructions": "Answer briefly.",
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look something up.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    },
    "c_nvidia_selectors": {
        "model": MODEL,
        "nvidia": {
            "llm_id": "nemotron-3.5-lightning",
            "thinker_llm_id": "nemotron-3-ultra",
            "asr_id": "nemotron-asr-streaming-english",
            "tts_id": "magpie-multilingual-tts",
        },
    },
    "d_unknown_llm_id": {"model": MODEL, "nvidia": {"llm_id": "no-such-llm"}},
    "e_unknown_thinker_llm_id": {"model": MODEL, "nvidia": {"thinker_llm_id": "no-such-llm"}},
}
#: Keys whose values are random per session or per clock read.
_VOLATILE = frozenset({"id", "event_id", "expires_at", "session_id", "call_id", "item_id", "response_id"})
_SECRET = frozenset({"api_key", "NVIDIA_API_KEY"})


def jsonable(value: Any) -> Any:
    """A stable, JSON-friendly view: dataclasses as dicts, paths as strings, volatile ids dropped."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            key = str(key)
            if key in _VOLATILE:
                continue
            out[key] = "<secret>" if key in _SECRET and item else jsonable(item)
        return out
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, frozenset | set):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Path):
        try:
            return value.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return value.as_posix()
    if isinstance(value, str):
        return value.replace(str(ROOT), "<root>")
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<{type(value).__name__}>"


@contextlib.contextmanager
def no_network() -> Iterator[None]:
    """Fail on any reachability probe; answer TTS voice discovery with an empty catalog."""
    patches = [
        mock.patch(f"utils.{name}", side_effect=AssertionError(f"network probe {name}"))
        for name in ("_filter_reachable_entries", "_first_reachable_variant", "_load_effective_services_catalog")
    ]
    import server  # noqa: PLC0415

    patches += [
        mock.patch.object(server, "peek_cached_tts_config", return_value=None),
        mock.patch.object(server, "peek_cached_asr_config", return_value=None),
        mock.patch.object(server, "get_tts_config", return_value={"voices": []}),
    ]
    for patch in patches:
        patch.start()
    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.stop()


# -- the route: what FBV's bot receives before the session is built -------------------------------


async def _controller(initial_session: dict[str, Any] | None, requested_model: str | None) -> Any:
    import server  # noqa: PLC0415
    from realtime.gateway import create_realtime_controller  # noqa: PLC0415

    return await create_realtime_controller(
        sanitize_session_config=lambda data, **kwargs: server._sanitize_realtime_config(data, **kwargs),
        initial_session=initial_session,
        resolve_server_tools=lambda config: server._resolve_realtime_server_tools(config, EXAMPLE),
        resolve_server_tool_schemas=lambda config: server._resolve_realtime_server_tool_schemas(config, EXAMPLE),
        resolve_session_capabilities=server._resolve_realtime_session_capabilities,
        resolve_model_route=lambda model, selectors: server._resolve_realtime_model_route(model, selectors, EXAMPLE),
        default_example_key=EXAMPLE,
        default_pipeline_mode=EXAMPLE,
        requested_model=requested_model,
    )


def _body(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    body = jsonable(dict(runtime_config))
    prompt = body.pop("prompt_content", None)
    if prompt is not None:
        body["prompt_content_sha256"] = hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()
    return body


def _voice_config(body: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    from examples.frontend_backend_verdict.adapters.pipecat import runtime as runtime_module  # noqa: PLC0415

    with mock.patch.dict(runtime_module._RUNTIMES, clear=True):
        runtime = runtime_module.runtime_for(body, environ)
    return {"profile": runtime.profile, "settings": runtime.settings.summary(), "config": jsonable(runtime.config)}


async def _connect(credential: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Authenticate ``credential`` as the WebSocket route does; build the controller FBV receives."""
    from realtime.auth import authenticate_realtime_websocket  # noqa: PLC0415

    auth = authenticate_realtime_websocket({"authorization": f"Bearer {credential}"}, api_key=TEST_REALTIME_KEY)
    initial = auth.claims.session if auth.claims is not None else None
    controller = await _controller(initial, MODEL if initial is None else None)
    return initial, dict(controller.runtime_config)


async def _mint(template: dict[str, Any]) -> str:
    """Mint an ``ek_`` key as ``POST /v1/realtime/client_secrets`` does."""
    import server  # noqa: PLC0415
    from realtime.auth import issue_realtime_client_secret  # noqa: PLC0415

    controller = await _controller(json.loads(json.dumps(template)), None)
    return issue_realtime_client_secret(
        api_key=TEST_REALTIME_KEY,
        session=server._canonical_realtime_secret_session(controller),
        expires_at=int(time.time()) + 600,
    )


async def capture_route(platform: str, environ: Mapping[str, str]) -> dict[str, Any]:
    """Bearer and ``ek_`` connections on one platform: the hydrated body and the eval ``VoiceConfig``."""
    from realtime.protocol import RealtimeProtocolError  # noqa: PLC0415

    out: dict[str, Any] = {}
    _, bearer = await _connect(TEST_REALTIME_KEY)
    out["bearer"] = {"body": _body(bearer), "voice_config": _voice_config(bearer, environ)}
    templates = dict(EK_TEMPLATES)
    for name, template in EK_TEMPLATES.items():
        if name != "a_no_template":
            continue
        # The selectors exactly as a minted key carries them (canonical ids), sent back in a template.
        canonical = (await _connect(await _mint(template)))[0] or {}
        selectors = {key: canonical["nvidia"][key] for key in ("llm_id", "thinker_llm_id", "asr_id", "tts_id")}
        templates["c2_canonical_selectors"] = {"model": MODEL, "nvidia": selectors}
    for name, template in templates.items():
        try:
            secret = await _mint(template)
            initial, body = await _connect(secret)
        except RealtimeProtocolError as exc:
            out[f"ek_{name}"] = {"refused": {"code": exc.code, "param": exc.param}}
            continue
        out[f"ek_{name}"] = {
            "nvidia": jsonable((initial or {}).get("nvidia")),
            "body": _body(body),
            "voice_config": _voice_config(body, environ),
        }
    return out


# -- the session: what the Realtime pipeline answers --------------------------------------------------


async def capture_session(environ: Mapping[str, str]) -> dict[str, Any]:
    """Tau ``session.update`` echoes, the tools handed to the agent, and today's commit/turn answers."""
    from _fbv_core_fakes import tau2_session_update  # noqa: PLC0415
    from _fbv_pipecat_fakes import PipecatHarness  # noqa: PLC0415

    from examples.frontend_backend_verdict.core.config.profiles import REALTIME_PROFILE, profile_path  # noqa: PLC0415
    from examples.frontend_backend_verdict.core.config.voice import load_voice_config  # noqa: PLC0415

    with mock.patch.dict(os.environ, dict(environ)):
        config = load_voice_config(profile_path(REALTIME_PROFILE))
        out: dict[str, Any] = {"tau": {}}
        for domain in TAU_DOMAINS:
            harness = PipecatHarness(config=config)
            configured: list[Any] = []
            original = harness.session.agent.configure

            def record(*args: Any, _original: Any = original, _seen: list[Any] = configured, **kwargs: Any) -> Any:
                _seen.append(jsonable(kwargs.get("tools")))
                return _original(*args, **kwargs)

            harness.session.agent.configure = record  # type: ignore[method-assign]
            try:
                await harness.start(tau2_session_update(domain))
                out["tau"][domain] = {
                    "created": jsonable(harness.of_type("session.created")[0]),
                    "updated": jsonable(harness.of_type("session.updated")[0]),
                    "agent_tools": configured,
                    "errors": jsonable(harness.of_type("error")),
                }
            finally:
                await harness.close()

        harness = PipecatHarness(config=config)
        probes = {
            "commit": {"type": "input_audio_buffer.commit"},
            "semantic_vad": {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "audio": {"input": {"turn_detection": {"type": "semantic_vad", "eagerness": "low"}}},
                },
            },
            "turn_detection_null": {
                "type": "session.update",
                "session": {"type": "realtime", "audio": {"input": {"turn_detection": None}}},
            },
            "mcp_tool": {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "tools": [{"type": "mcp", "server_label": "x", "server_url": "https://example.invalid/mcp"}],
                },
            },
        }
        out["probes"] = {}
        try:
            await harness.start(tau2_session_update("mock"))
            for name, event in probes.items():
                before = len(harness.events)
                await harness.send(event)
                await harness.settle(0.2)
                out["probes"][name] = jsonable(harness.events[before:])
        finally:
            await harness.close()
    return out


def capture_all(base_env: Mapping[str, str]) -> dict[str, Any]:
    """The whole capture under ``base_env`` (on top of the pinned environment)."""
    from _fbv_golden import pinned_environment  # noqa: PLC0415

    result: dict[str, Any] = {"route": {}}
    fixed = {"REALTIME_API_KEY": TEST_REALTIME_KEY, "UVICORN_WS_PING_INTERVAL": "0", "UVICORN_WS_PING_TIMEOUT": "20"}
    with pinned_environment(), mock.patch.dict(os.environ, {**fixed, **base_env}):
        with no_network():
            for platform in PLATFORMS:
                env = {**os.environ, "REALTIME_SERVICE_PLATFORM": platform}
                with mock.patch.dict(os.environ, {"REALTIME_SERVICE_PLATFORM": platform}):
                    result["route"][platform] = asyncio.run(capture_route(platform, env))
        result["session"] = asyncio.run(capture_session(dict(os.environ)))
    return result


if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(ROOT / "src")]
    print(json.dumps(capture_all({}), indent=1, sort_keys=True))
