# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Pins of today's FDH evaluation behaviour (``shared-stack-plan.md``, step 0).

Captured before FDH moved onto the shared stack, from the shipped eval pair
``profiles/pipecat_realtime_eval.yaml`` + ``gateway.eval.yaml`` and ``adapters/pipecat.eval.yaml``.
Every later step must keep them equal (``test_fdh_eval_pins.py``):

* the prompts and tools every eval session runs with, per τ² domain (airline, retail, telecom):
  ``frontend_prompt_sha256``, ``session_tools_sha256`` and the backend fingerprint the gateway reports
  (``backend_system_sha256``, ``backend_soul_sha256``, ``backend_catalog_sha256``, features, domain);
* the delegation, voice and gateway settings (hashed; repository paths made relative, the backend URL
  replaced and the speech catalog's directory left out, because they change with the deployment or the
  file layout, not the behaviour);
* the resolved ASR and TTS endpoints, the Pipecat settings summary and the Realtime pipeline's processors;
* ``session.created`` / ``session.updated`` and the server-event sequence of a scripted session (stub
  speech, fake backend) per domain. The ``model`` field is replaced by a placeholder (it echoes the
  requested model, which changes from ``pine-*`` to the registry id); runs of one event type are counted
  once, because their number depends on wall-clock audio pacing.

Regenerate only for an intended change of the evaluated behaviour::

    PYTHONPATH=src uv run python tests/unit/frontend_delegation_hermes/pipecat/_fdh_eval_pins.py --write
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from _pipecat_fakes import REPO_ROOT, Client, realtime_app, stub_speech, tau2_session_update  # noqa: E402

PACKAGE = REPO_ROOT / "src" / "examples" / "frontend_delegation_hermes"
CONFIG = PACKAGE / "config"
PIN_FILE = HERE / "fixtures" / "eval_pins.json"
DOMAINS = ("airline", "retail", "telecom")
EVAL_PROFILE = CONFIG / "profiles" / "pipecat_realtime_eval.yaml"
DEMO_PROFILE = CONFIG / "profiles" / "pipecat_demo.yaml"
EVAL_GATEWAY = CONFIG / "gateway.eval.yaml"
DEMO_GATEWAY = CONFIG / "gateway.yaml"
EVAL_SETTINGS = "adapters/pipecat.eval.yaml"
DEMO_SETTINGS = "adapters/pipecat.yaml"
MODEL_PLACEHOLDER = "<model>"
BACKEND_URL_PLACEHOLDER = "<backend-url>"
USER_TEXT = "Hi, I need to change something on my account, my user id is mia_li_3668."


def without_prompt_profile() -> Any:
    """An environment without ``FDH_PROMPT_PROFILE``: the pins use each config file's default profile."""
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "FDH_PROMPT_PROFILE"}, clear=True)


def _sha(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _relative(value: Any) -> Any:
    """Repository paths made relative; secrets and the catalog ``source`` dropped."""
    if isinstance(value, dict):
        return {k: _relative(v) for k, v in value.items() if k not in ("api_key", "source", "bearer_token")}
    if isinstance(value, list | tuple):
        return [_relative(v) for v in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and str(REPO_ROOT) in value:
        return value.replace(str(REPO_ROOT), "<repo>")
    return value


def _plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if callable(obj):
        return f"{getattr(obj, '__module__', '')}:{getattr(obj, '__qualname__', type(obj).__name__)}"
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple | frozenset | set):
        items = [_plain(v) for v in obj]
        return sorted(items, key=str) if isinstance(obj, frozenset | set) else items
    return obj


def _endpoint(endpoint: Any) -> dict[str, Any]:
    keys = ("kind", "server", "model", "function_id", "language_code", "voice_id", "use_ssl")
    return {key: getattr(endpoint, key) for key in keys}


def config_pins(profile: Path) -> dict[str, Any]:
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415

    config = load_delegation_config(profile)
    effective = copy.deepcopy(config.effective)
    effective["backend"]["url"] = BACKEND_URL_PLACEHOLDER
    voice = _relative(_plain(config.voice))
    voice.pop("warnings", None)
    # The catalog file's directory is not behaviour: the resolved endpoints are pinned below.
    for kind in ("asr", "tts"):
        voice["resolved_in_paths"].pop(f"{kind}.catalog.example_dir", None)
    return {
        "delegation_settings_sha": _sha(_relative(effective)),
        "voice_settings_sha": _sha(voice),
        "asr": _endpoint(config.voice.asr.endpoint),
        "tts": _endpoint(config.voice.tts.endpoint),
    }


def gateway_pins(path: Path) -> dict[str, Any]:
    from examples.frontend_delegation_hermes.core.prompt_features import (  # noqa: PLC0415
        render_backend_system,
        sha256_text,
    )
    from examples.frontend_delegation_hermes.core.sidecar import gateway_config as gc  # noqa: PLC0415
    from examples.frontend_delegation_hermes.core.sidecar.templates import BackendTemplates  # noqa: PLC0415

    config = gc.load_gateway_config(path)
    templates = BackendTemplates(config.hermes.prompts_path, prompt_features=config.prompt_features)
    out: dict[str, Any] = {
        "settings_sha": _sha(_relative(config.summary())),
        "hermes": _relative(
            {
                "model": config.hermes.model,
                "base_url": config.hermes.base_url,
                "request_overrides": config.hermes.request_overrides,
                "backend_trace": config.hermes.backend_trace,
            }
        ),
        "backend_features": dict(templates.features),
        "backend_catalog_sha256": templates.catalog_sha256,
        "backend_soul_sha256": sha256_text(templates.render("backend_soul")),
        "domains": [name for name, _ in config.domains],
        "per_domain": {},
    }
    for domain in DOMAINS:
        session = tau2_session_update(domain)["session"]
        names = {str(tool.get("name")) for tool in session["tools"]}
        found = next((name for name, tools_any in config.domains if names.intersection(tools_any)), "")
        system = render_backend_system(templates, session["instructions"], templates.domain_note(found))
        out["per_domain"][domain] = {"backend_domain": found, "backend_system_sha256": sha256_text(system)}
    return out


def pipeline_pins() -> dict[str, Any]:
    from examples.frontend_delegation_hermes.adapters.pipecat.settings import load_pipecat_settings  # noqa: PLC0415

    eval_settings = load_pipecat_settings(EVAL_SETTINGS)
    summary = eval_settings.summary()
    summary["settings_file"] = Path(summary["settings_file"]).name
    demo = load_pipecat_settings(DEMO_SETTINGS).summary()
    demo["settings_file"] = Path(demo["settings_file"]).name
    return {"eval_settings": summary, "demo_settings": demo, "eval_processors": _processor_classes(eval_settings)}


def _processor_classes(settings: Any) -> list[str]:
    """The Realtime pipeline's processors, in order, with the real speech services of the eval profile."""
    from unittest import mock  # noqa: PLC0415

    from examples.frontend_delegation_hermes.adapters.pipecat.pipeline import SessionPipeline  # noqa: PLC0415
    from examples.frontend_delegation_hermes.adapters.pipecat.services import (  # noqa: PLC0415
        build_stt,
        build_tts,
        build_vad,
    )
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415

    config = load_delegation_config(EVAL_PROFILE).voice
    speech = mock.Mock(vad=build_vad(settings), stt=build_stt(config), tts=build_tts(config, settings))
    session = mock.Mock(settings=settings, config=config, session_id="pin")
    pipeline = SessionPipeline(session, mock.Mock(), speech)
    names = [type(p).__name__ for p in pipeline.pipeline.processors]
    return [*names, f"vad_analyzer={type(speech.vad).__name__}"]


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    event = copy.deepcopy(event)
    session = event.get("session")
    if isinstance(session, dict):
        if "model" in session:
            session["model"] = MODEL_PLACEHOLDER
        if "id" in session:
            session["id"] = "<id>"
    event.pop("event_id", None)
    return event


def _collapse(types: list[str]) -> list[str]:
    out: list[str] = []
    for kind in types:
        if not out or out[-1] != kind:
            out.append(kind)
    return out


def session_pins(domain: str) -> dict[str, Any]:
    """One scripted text session per domain through today's eval pipeline (stub speech, fake backend)."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from examples.frontend_delegation_hermes.adapters.pipecat.runtime import FdhRuntime  # noqa: PLC0415
    from examples.frontend_delegation_hermes.adapters.pipecat.settings import load_pipecat_settings  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "events.jsonl"
        config = _stub_eval_config(log_path)
        runtime = FdhRuntime(config, load_pipecat_settings(EVAL_SETTINGS), profile="eval", speech_factory=stub_speech)
        app = realtime_app(runtime)
        with TestClient(app) as client, client.websocket_connect("/v1/realtime?model=pine-pin") as ws:
            session = Client(ws)
            created = session.wait_for("session.created")
            session.send(tau2_session_update(domain))
            updated = session.wait_for("session.updated")
            session.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": USER_TEXT}]},
                }
            )
            session.send({"type": "response.create"})
            call = session.wait_for("response.function_call_arguments.done")
            session.send(
                {
                    "type": "conversation.item.create",
                    "item": {"type": "function_call_output", "call_id": call["call_id"], "output": '{"ok": true}'},
                }
            )
            session.send({"type": "response.create"})
            session.wait_for("response.done", 2)
            session.settle(2.0)
            types = session.types()
        records = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    prompts = [r for r in records if r.get("type") == "frontend_prompt" or r.get("kind") == "frontend_prompt"]
    tools = [r for r in records if (r.get("type") or r.get("kind")) == "backend_configured"]
    return {
        "session_created": _normalize_event(created),
        "session_updated": _normalize_event(updated),
        "event_types": _collapse(types),
        "frontend_prompt_sha256": [_field(r, "frontend_prompt_sha256") for r in prompts],
        "session_tools_sha256": [_field(r, "session_tools_sha256") for r in tools],
    }


def _field(record: dict[str, Any], key: str) -> Any:
    if key in record:
        return record[key]
    data = record.get("data")
    return data.get(key) if isinstance(data, dict) else None


def _stub_eval_config(log_path: Path) -> Any:
    from examples.frontend_delegation_hermes.core.config import load_delegation_config  # noqa: PLC0415

    config = load_delegation_config(EVAL_PROFILE)
    voice = replace(
        config.voice,
        server=replace(config.voice.server, warmup=False),
        logging=replace(config.voice.logging, event_log=str(log_path)),
    )
    backend = replace(
        config.backend,
        link="in_process_fake",
        verbalizer_mode="template",
        gateway_config=CONFIG / "gateway.stub_gate.yaml",
    )
    return replace(config, voice=voice, backend=backend, frontend=replace(config.frontend, decider="scripted"))


def capture() -> dict[str, Any]:
    """Every pin, by name."""
    return {
        "eval_profile": config_pins(EVAL_PROFILE),
        "demo_profile": config_pins(DEMO_PROFILE),
        "eval_gateway": gateway_pins(EVAL_GATEWAY),
        "demo_gateway": gateway_pins(DEMO_GATEWAY),
        "pipeline": pipeline_pins(),
        "sessions": {domain: session_pins(domain) for domain in DOMAINS},
    }


def load_pins() -> dict[str, Any]:
    return json.loads(PIN_FILE.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="overwrite the pin file")
    args = parser.parse_args()
    with without_prompt_profile():
        pins = capture()
    text = json.dumps(pins, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if not args.write:
        print(text)
        return
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(text, encoding="utf-8")
    print(f"wrote {PIN_FILE}")


if __name__ == "__main__":
    main()
