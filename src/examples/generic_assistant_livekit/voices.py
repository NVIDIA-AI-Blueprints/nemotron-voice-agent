# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover the TTS voices a speech server offers, for the voice selector in the UI.

`parse_voices` is pure (standard library only), so the unit tests do not need the Riva client. The formats follow the
synthesis config that each server returns from `GetRivaSynthesisConfig`:

- Magpie TTS Multilingual: `voice_name=Magpie-Multilingual`, `subvoices=EN-US.Aria:0,ES-US.Isabela:1`. The voice id is
  `Magpie-Multilingual.EN-US.Aria`.
- Magpie TTS Zeroshot: `subvoices=Female:0,Male:6`, shared by every language. The id is
  `Magpie-ZeroShot-Multilingual.Female`.
- NeMo-Speech.cpp: plain names (`John,Sofia,Aria`) in `subvoices` or `voices`, and/or a `voices_by_language` JSON map.
  The id is the plain name.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .state import read_json, write_json_atomic

VOICES_FILE = "voices.json"  # written by the agent worker into the state volume, read by the UI server
PLAIN_PREFIXES = {
    "magpietts",
    "magpie",
}  # NeMo-Speech.cpp voice ids look like "magpietts.John"; the server wants "John"


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _bare_name(token: str) -> str:
    head, _, rest = token.partition(".")
    return rest if rest and "." not in rest and head.lower() in PLAIN_PREFIXES else token


def parse_voices(model_params: Iterable[Mapping[str, str]], language: str) -> list[dict[str, str]]:
    """Return the voices for `language` as `[{"id": ..., "name": ...}]`, in server order and without duplicates.

    Args:
        model_params: The `parameters` map of each model config in the synthesis config response.
        language: A language code such as `en-US`. The match ignores case.
    """
    wanted = language.lower()
    voices: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(voice_id: str, name: str) -> None:
        if voice_id and voice_id not in seen:
            seen.add(voice_id)
            voices.append({"id": voice_id, "name": name})

    for params in model_params:
        languages = {code.lower() for code in _split(params.get("language_code", ""))}
        serves_language = not languages or wanted in languages
        prefix = params.get("voice_name", "").strip()

        by_language = params.get("voices_by_language", "")
        if by_language:
            try:
                catalog: Any = json.loads(by_language)
            except (TypeError, ValueError):
                catalog = None
            if isinstance(catalog, dict):
                for code, payload in catalog.items():
                    if str(code).lower() == wanted and isinstance(payload, dict):
                        for token in payload.get("voices", []):
                            bare = _bare_name(str(token))
                            add(bare, bare.rsplit(".", 1)[-1])

        for entry in _split(params.get("subvoices", "") or params.get("voices", "")):
            short = entry.split(":", 1)[0].strip()
            if not short:
                continue
            if ":" in entry and "." in short:  # Magpie Multilingual: Language.Name:index
                lang, _, name = short.partition(".")
                if lang.lower() == wanted:
                    add(f"{prefix}.{short}" if prefix else short, name)
            elif ":" in entry:  # Magpie Zeroshot: Name:index, the same voices for every language
                if serves_language:
                    add(f"{prefix}.{short}" if prefix else short, short)
            elif serves_language:  # NeMo-Speech.cpp: plain names
                bare = _bare_name(short)
                add(bare, bare.rsplit(".", 1)[-1])
    return voices


def with_current(voices: list[dict[str, str]], current: str) -> list[dict[str, str]]:
    """Return the list with the configured voice included, so the selector can always show the active one."""
    if any(voice["id"] == current for voice in voices):
        return voices
    return [{"id": current, "name": current.rsplit(".", 1)[-1]}, *voices]


def choose_voice(requested: object, available: Iterable[str], default: str) -> str:
    """Return the requested voice when the server listed it, otherwise the default voice."""
    return requested if isinstance(requested, str) and requested in set(available) else default


def write_voices_file(path: Path, voices: list[dict[str, str]], default: str) -> None:
    """Write the voice list for the UI server."""
    write_json_atomic(path, {"voices": voices, "default": default})


def read_voices_file(path: Path) -> dict[str, Any]:
    """Return `{"voices": [...], "default": ...}`, or an empty list when the file is missing or unreadable."""
    data = read_json(path)
    try:
        voices = [v for v in data["voices"] if isinstance(v, dict) and "id" in v and "name" in v]
        return {"voices": voices, "default": str(data.get("default", ""))}
    except (KeyError, TypeError):
        return {"voices": [], "default": ""}


def parse_set_voice(data: bytes, allowed: Iterable[str]) -> str | None:
    """Return the voice id from a `set_voice` control message, or None for anything else.

    Only ids in `allowed` (the voices the server listed) are accepted, so a client cannot make the agent request an
    arbitrary voice name.
    """
    try:
        message = json.loads(data)
    except (TypeError, ValueError):
        return None
    if not isinstance(message, dict) or message.get("type") != "set_voice":
        return None
    voice = message.get("voice")
    return voice if isinstance(voice, str) and voice in set(allowed) else None


def fetch_synthesis_params(
    server: str, use_ssl: bool, api_key: str = "", function_id: str = "", timeout: float = 5.0
) -> list[dict[str, str]]:
    """Ask a Riva speech server for its synthesis config. Blocking: call it from a thread."""
    import riva.client
    from riva.client.proto import riva_tts_pb2

    metadata = []
    if api_key:
        metadata.append(["authorization", f"Bearer {api_key}"])
    if function_id:
        metadata.append(["function-id", function_id])
    auth = riva.client.Auth(uri=server, use_ssl=use_ssl, metadata_args=metadata)
    service = riva.client.SpeechSynthesisService(auth)
    response = service.stub.GetRivaSynthesisConfig(riva_tts_pb2.RivaSynthesisConfigRequest(), timeout=timeout)
    return [dict(config.parameters) for config in response.model_config]
