# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Realtime session state and mapping onto Nemotron flat session config."""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any

from loguru import logger

from realtime.audio import validate_session_audio_config
from realtime.client_tools import validate_client_tools
from utils import _SLOT_AGNOSTIC_KEYS, _SLOT_CONFIG_KEYS, SESSION_CONFIG_KEYS

DEFAULT_PIPELINE_MODE = "generic-assistant"
DEFAULT_PROMPT_KEY = "generic_assistant_without_tools"

# Flat keys owned by OpenAI top-level session fields — not accepted under ``session.nvidia``.
_OPENAI_MAPPED_FLAT_KEYS = frozenset(
    {
        "prompt_content",
        "system_prompt",
        "tts_voice_id",
        "temperature",
        "max_tokens",
        "tool_choice",
    }
)

# Endpoints / function ids — accepted under ``session.nvidia`` but redacted from public echo.
_NVIDIA_INTERNAL_KEYS = frozenset(
    {
        "base_url",
        "asr_server",
        "asr_function_id",
        "tts_server",
        "tts_function_id",
    }
)


def _nvidia_keys_from_slot_config() -> frozenset[str]:
    """Derive ``session.nvidia`` allowlist from shared flat session/slot keys."""
    keys: set[str] = set(_SLOT_AGNOSTIC_KEYS)
    for slot_keys in _SLOT_CONFIG_KEYS.values():
        keys |= set(slot_keys)
    keys -= {key for key in keys if key.startswith("thinker_")}
    keys -= _OPENAI_MAPPED_FLAT_KEYS
    keys &= set(SESSION_CONFIG_KEYS)
    return frozenset(keys)


# Client ``session.nvidia`` allowlist: catalog / routing ids only. Endpoints and
# function ids are not accepted from the client (SSRF); sanitize/hydrate fills
# them from the selected catalog entries.
_NVIDIA_SESSION_KEYS = _nvidia_keys_from_slot_config() - _NVIDIA_INTERNAL_KEYS
# Echoed on session.created/updated. ``server_tools`` is server-generated only.
_NVIDIA_PUBLIC_KEYS = _NVIDIA_SESSION_KEYS | {"server_tools"}

_DEFAULT_AUDIO = {
    "input": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "turn_detection": {"type": "server_vad"},
    },
    "output": {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "",
    },
}


def merge_session_patch(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``patch`` into a copy of ``base`` (dicts only).

    Nested dicts are merged; ``None`` values overwrite. Lists and scalars
    replace. Used for OpenAI-style partial ``session.update`` patches so
    ``audio.output.voice`` does not wipe sibling ``audio.input`` state.
    """
    out = copy.deepcopy(base)
    for key, value in patch.items():
        if value is None:
            out[key] = None
            continue
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_session_patch(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _audio_output_voice(session_patch: dict[str, Any]) -> str | None:
    """Return nested ``audio.output.voice`` when present as a non-empty string."""
    audio = session_patch.get("audio")
    if not isinstance(audio, dict):
        return None
    output = audio.get("output")
    if not isinstance(output, dict) or "voice" not in output:
        return None
    voice = output.get("voice")
    if voice is None:
        return None
    if not isinstance(voice, str):
        raise ValueError("session.audio.output.voice must be a string")
    cleaned = voice.strip()
    return cleaned or None


def _map_max_output_tokens(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ValueError("session.max_output_tokens must be an integer")
    if isinstance(raw, str) and raw.strip().lower() in {"inf", "infinite", "null"}:
        return None
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError("session.max_output_tokens must be an integer")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("session.max_output_tokens must be an integer") from exc
    if value < 1:
        raise ValueError("session.max_output_tokens must be >= 1")
    return value


def _map_temperature(raw: Any) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, bool):
        raise ValueError("session.temperature must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("session.temperature must be a number") from exc
    if value < 0.0:
        raise ValueError("session.temperature must be >= 0")
    return value


def map_tool_choice(raw: Any) -> Any:
    """Map Realtime ``tool_choice`` to Chat Completions shape for ``LLMContext``.

    Accepts ``auto`` / ``none`` / ``required``, Chat Completions
    ``{"type":"function","function":{"name":...}}``, and Realtime
    ``{"type":"function","name":...}``.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        cleaned = raw.strip()
        if cleaned in {"auto", "none", "required"}:
            return cleaned
        raise ValueError("session.tool_choice string must be auto, none, or required")
    if isinstance(raw, dict):
        typ = raw.get("type")
        if typ != "function":
            raise ValueError("session.tool_choice object must have type 'function'")
        function = raw.get("function")
        name: str | None = None
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            name = function["name"].strip() or None
        elif isinstance(raw.get("name"), str):
            # OpenAI Realtime shape: {"type":"function","name":"..."}
            name = raw["name"].strip() or None
        if not name:
            raise ValueError("session.tool_choice function name is required")
        return {"type": "function", "function": {"name": name}}
    raise ValueError("session.tool_choice must be a string or function object")


def _validate_modalities_field(raw: Any, *, field: str) -> None:
    """Require a non-empty string list that includes ``audio`` (GA or beta field name)."""
    if raw is None:
        return
    param = f"session.{field}"
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{param} must be a non-empty array")
    if not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{param} entries must be strings")
    if "audio" not in raw:
        raise ValueError(f"{param} must include 'audio' for Nemotron Voice Agent v1")


def validate_input_transcription(session_patch: dict[str, Any]) -> None:
    """Accept OpenAI transcription selectors as compatibility no-ops."""
    audio = session_patch.get("audio")
    transcription = None
    if isinstance(audio, dict):
        inp = audio.get("input")
        if isinstance(inp, dict) and "transcription" in inp:
            transcription = inp.get("transcription")
    if transcription is None and "input_audio_transcription" in session_patch:
        transcription = session_patch.get("input_audio_transcription")
    if transcription is None:
        return
    if not isinstance(transcription, dict):
        raise ValueError("session input audio transcription selector must be an object or null")
    model = transcription.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("session input audio transcription model must be a string")
    logger.info("Ignoring input transcription selector; pipeline ASR provides transcript events")


def validate_server_vad(session_patch: dict[str, Any]) -> None:
    """Require the only implemented turn mode: unconfigured ``server_vad``."""
    candidates: list[tuple[str, Any]] = []
    if "turn_detection" in session_patch:
        candidates.append(("session.turn_detection", session_patch.get("turn_detection")))
    audio = session_patch.get("audio")
    if isinstance(audio, dict):
        inp = audio.get("input")
        if isinstance(inp, dict) and "turn_detection" in inp:
            candidates.append(("session.audio.input.turn_detection", inp.get("turn_detection")))

    for param, value in candidates:
        if value is None:
            raise ValueError(
                f"{param}: manual mode (push-to-talk) is not supported. "
                'Switch the client to VAD mode with {"type":"server_vad"}'
            )
        if not isinstance(value, dict):
            raise ValueError(f'{param} must be an object. Switch the client to VAD mode with {{"type":"server_vad"}}')
        if value.get("type") != "server_vad":
            raise ValueError(f"{param}.type must be 'server_vad'; switch the client to VAD mode")
        unsupported = sorted(set(value) - {"type"})
        if unsupported:
            raise ValueError(
                f'{param} tuning is not supported; only {{"type":"server_vad"}} is accepted '
                f"(got {', '.join(unsupported)})"
            )


def map_session_update_to_flat_config(
    session_patch: dict[str, Any],
    *,
    default_pipeline_mode: str = DEFAULT_PIPELINE_MODE,
) -> dict[str, Any]:
    """Map a Realtime ``session.update`` payload to flat session-config keys.

    OpenAI top-level fields (``instructions``, ``voice``, ``temperature``, …) map onto
    the cascaded pipeline. ``session.nvidia`` holds catalog fields with no OpenAI
    equivalent. Client ``tools`` are kept separate from catalog-owned server
    tools so the pipeline can route execution by ownership.
    """
    if not isinstance(session_patch, dict):
        raise ValueError("session must be a JSON object")

    if "model" in session_patch and session_patch.get("model") not in ("", None):
        logger.info(
            f"Ignoring session.model={session_patch.get('model')!r}; using Nemotron Voice Agent cascaded pipeline"
        )

    _validate_modalities_field(session_patch.get("output_modalities"), field="output_modalities")
    _validate_modalities_field(session_patch.get("modalities"), field="modalities")
    validate_input_transcription(session_patch)
    validate_server_vad(session_patch)
    validate_session_audio_config(session_patch)

    flat: dict[str, Any] = {}
    if "tools" in session_patch:
        flat["client_tools"] = validate_client_tools(session_patch.get("tools"))

    nvidia = session_patch.get("nvidia")
    if isinstance(nvidia, dict):
        for key, value in nvidia.items():
            if key not in _NVIDIA_SESSION_KEYS:
                continue
            if value not in ("", None):
                flat[key] = value

    instructions = session_patch.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        flat["prompt_content"] = instructions.strip()

    voice = None
    if "voice" in session_patch and session_patch.get("voice") is not None:
        raw_voice = session_patch.get("voice")
        if not isinstance(raw_voice, str):
            raise ValueError("session.voice must be a string")
        voice = raw_voice.strip()
    nested_voice = _audio_output_voice(session_patch)
    if nested_voice is not None:
        voice = nested_voice
    if voice:
        flat["tts_voice_id"] = voice

    prompt = session_patch.get("prompt")
    if isinstance(prompt, dict):
        prompt_id = prompt.get("id")
        if isinstance(prompt_id, str) and prompt_id.strip():
            flat["prompt_key"] = prompt_id.strip()

    max_tokens = None
    if "max_output_tokens" in session_patch:
        max_tokens = _map_max_output_tokens(session_patch.get("max_output_tokens"))
    elif "max_response_output_tokens" in session_patch:
        max_tokens = _map_max_output_tokens(session_patch.get("max_response_output_tokens"))
    if max_tokens is not None:
        flat["max_tokens"] = max_tokens

    if "temperature" in session_patch:
        temperature = _map_temperature(session_patch.get("temperature"))
        if temperature is not None:
            flat["temperature"] = temperature

    if "tool_choice" in session_patch and session_patch.get("tool_choice") is not None:
        flat["tool_choice"] = map_tool_choice(session_patch.get("tool_choice"))

    if not flat.get("pipeline_mode"):
        flat["pipeline_mode"] = default_pipeline_mode

    if not flat.get("prompt_key") and not flat.get("prompt_content"):
        flat["prompt_key"] = DEFAULT_PROMPT_KEY

    return flat


def nvidia_public_view(nvidia: dict[str, Any]) -> dict[str, Any]:
    """Return a client-safe ``session.nvidia`` object (no internal service endpoints)."""
    return {key: value for key, value in nvidia.items() if key in _NVIDIA_PUBLIC_KEYS}


class RealtimeSession:
    """In-memory OpenAI-shaped session view plus last sanitized flat config."""

    def __init__(self, *, default_pipeline_mode: str = DEFAULT_PIPELINE_MODE) -> None:
        """Create a session with OpenAI-shaped defaults."""
        self.id = f"sess_{uuid.uuid4().hex}"
        self.default_pipeline_mode = default_pipeline_mode
        self.view: dict[str, Any] = {
            "id": self.id,
            "object": "realtime.session",
            "type": "realtime",
            "instructions": "",
            "output_modalities": ["audio"],
            "audio": copy.deepcopy(_DEFAULT_AUDIO),
            "tools": [],
            "tool_choice": "auto",
            "nvidia": {
                "pipeline_mode": default_pipeline_mode,
                "prompt_key": DEFAULT_PROMPT_KEY,
            },
        }
        self.flat_config: dict[str, Any] = {
            "pipeline_mode": default_pipeline_mode,
            "prompt_key": DEFAULT_PROMPT_KEY,
        }

    def public_session(self) -> dict[str, Any]:
        """Return the session object embedded in created/updated events."""
        return copy.deepcopy(self.view)

    def apply_update(
        self,
        session_patch: dict[str, Any],
        *,
        sanitized_flat: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge a client patch into the public view and store sanitized flat config.

        ``sanitized_flat`` is the output of the server's existing sanitize/hydrate
        path and becomes the source of truth for ``nvidia`` catalog fields.
        """
        if not isinstance(session_patch, dict):
            raise ValueError("session must be a JSON object")

        patch_without_nvidia = {k: v for k, v in session_patch.items() if k != "nvidia"}
        self.view = merge_session_patch(self.view, patch_without_nvidia)
        if any(
            key in session_patch
            for key in (
                "turn_detection",
                "input_audio_format",
                "output_audio_format",
                "input_audio_transcription",
            )
        ):
            audio = dict(self.view.get("audio") or {})
            inp = dict(audio.get("input") or {})
            output = dict(audio.get("output") or {})
            if "turn_detection" in session_patch:
                inp["turn_detection"] = copy.deepcopy(session_patch["turn_detection"])
            if "input_audio_format" in session_patch:
                inp["format"] = copy.deepcopy(session_patch["input_audio_format"])
            if "output_audio_format" in session_patch:
                output["format"] = copy.deepcopy(session_patch["output_audio_format"])
            if "input_audio_transcription" in session_patch:
                inp["transcription"] = copy.deepcopy(session_patch["input_audio_transcription"])
            audio["input"] = inp
            audio["output"] = output
            self.view["audio"] = audio
        self.view["id"] = self.id
        self.view["object"] = "realtime.session"
        self.view["type"] = "realtime"

        nvidia = dict(self.view.get("nvidia") or {})
        if isinstance(session_patch.get("nvidia"), dict):
            nvidia = merge_session_patch(nvidia, session_patch["nvidia"])

        for key in _NVIDIA_SESSION_KEYS:
            if key in sanitized_flat and sanitized_flat[key] not in ("", None):
                nvidia[key] = sanitized_flat[key]
        if isinstance(sanitized_flat.get("server_tools"), list):
            nvidia["server_tools"] = copy.deepcopy(sanitized_flat["server_tools"])

        nvidia = nvidia_public_view(nvidia)
        if not nvidia.get("pipeline_mode"):
            nvidia["pipeline_mode"] = self.default_pipeline_mode
        if not nvidia.get("prompt_key"):
            nvidia["prompt_key"] = sanitized_flat.get("prompt_key") or DEFAULT_PROMPT_KEY

        self.view["nvidia"] = nvidia

        if session_patch.get("instructions") is not None:
            self.view["instructions"] = session_patch.get("instructions") or ""

        if "tools" in session_patch:
            self.view["tools"] = copy.deepcopy(sanitized_flat.get("client_tools", []))

        if sanitized_flat.get("tts_voice_id"):
            audio = dict(self.view.get("audio") or {})
            output = dict(audio.get("output") or {})
            output["voice"] = sanitized_flat["tts_voice_id"]
            audio["output"] = output
            self.view["audio"] = audio
            self.view["voice"] = sanitized_flat["tts_voice_id"]

        self.flat_config = dict(sanitized_flat)
        return self.public_session()


# Post-handoff live keys only; unchanged agent fields may be echoed by clients.
_LIVE_SESSION_KEYS = frozenset(
    {
        "type",
        "model",  # ignored no-op (same as connect)
        "voice",
        "turn_detection",
        "input_audio_transcription",
        "tools",
        "tool_choice",
        "audio",
        "input_audio_format",
        "output_audio_format",
    }
)
_LIVE_AUDIO_KEYS = frozenset({"input", "output"})
_LIVE_AUDIO_INPUT_KEYS = frozenset({"format", "turn_detection", "transcription"})
_LIVE_AUDIO_OUTPUT_KEYS = frozenset({"format", "voice"})


def _jsonish_equal(left: Any, right: Any) -> bool:
    """Structural equality for session field comparison (order-insensitive JSON)."""
    try:
        return json.dumps(left, sort_keys=True, default=str) == json.dumps(right, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return left == right


def live_session_patch(session_patch: dict[str, Any]) -> dict[str, Any]:
    """Return only the subset of a session patch that can take effect live."""
    out: dict[str, Any] = {}
    for key in session_patch:
        if key in _LIVE_SESSION_KEYS and key not in {"audio", "turn_detection"}:
            out[key] = copy.deepcopy(session_patch[key])

    audio = session_patch.get("audio")
    if not isinstance(audio, dict):
        return out

    audio_out: dict[str, Any] = {}
    inp = audio.get("input")
    if isinstance(inp, dict):
        inp_out = {
            key: copy.deepcopy(value)
            for key, value in inp.items()
            if key in _LIVE_AUDIO_INPUT_KEYS and key != "turn_detection"
        }
        if inp_out:
            audio_out["input"] = inp_out
    out_audio = audio.get("output")
    if isinstance(out_audio, dict):
        out_out = {key: copy.deepcopy(value) for key, value in out_audio.items() if key in _LIVE_AUDIO_OUTPUT_KEYS}
        if out_out:
            audio_out["output"] = out_out
    if audio_out:
        out["audio"] = audio_out
    return out


def unsupported_live_session_fields(
    session_patch: dict[str, Any],
    current: dict[str, Any] | None = None,
) -> list[str]:
    """Return dotted paths that would change agent config we cannot apply live.

    When ``current`` is omitted, any non-live key is treated as unsupported
    (strict mode for unit tests). With ``current``, identical re-sends of the
    active session are allowed; only differing non-live values are returned.
    """
    bad: list[str] = []
    current = current or {}

    def _turn_detection(view: dict[str, Any]) -> Any:
        audio = view.get("audio")
        if isinstance(audio, dict):
            inp = audio.get("input")
            if isinstance(inp, dict) and "turn_detection" in inp:
                return inp.get("turn_detection")
        return view.get("turn_detection")

    current_turn_detection = _turn_detection(current)
    if "turn_detection" in session_patch and not _jsonish_equal(
        session_patch.get("turn_detection"),
        current_turn_detection,
    ):
        bad.append("turn_detection")

    for key, value in session_patch.items():
        if key in _LIVE_SESSION_KEYS:
            continue
        if key == "nvidia":
            if not isinstance(value, dict):
                bad.append("nvidia")
                continue
            cur_n = dict(current.get("nvidia") or {})
            for nkey, nval in value.items():
                if nkey not in _NVIDIA_SESSION_KEYS:
                    continue
                if nkey not in cur_n or not _jsonish_equal(nval, cur_n[nkey]):
                    bad.append(f"nvidia.{nkey}")
            continue
        if key not in current:
            # Newly introduced non-live field after handoff — reject (do not silent-drop).
            bad.append(key)
            continue
        if not _jsonish_equal(value, current[key]):
            bad.append(key)

    audio = session_patch.get("audio")
    if isinstance(audio, dict):
        for key in audio:
            if key not in _LIVE_AUDIO_KEYS:
                bad.append(f"audio.{key}")
        inp = audio.get("input")
        if isinstance(inp, dict):
            for key in inp:
                if key not in _LIVE_AUDIO_INPUT_KEYS:
                    bad.append(f"audio.input.{key}")
            if "turn_detection" in inp and not _jsonish_equal(
                inp.get("turn_detection"),
                current_turn_detection,
            ):
                bad.append("audio.input.turn_detection")
        out = audio.get("output")
        if isinstance(out, dict):
            for key in out:
                if key not in _LIVE_AUDIO_OUTPUT_KEYS:
                    bad.append(f"audio.output.{key}")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in bad:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
