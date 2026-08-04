# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Soft TTS voice resolution for Realtime session.update (catalog list, default fallback)."""

from __future__ import annotations

from typing import Any

from loguru import logger

from examples.shared.prewarm import get_tts_config
from utils import load_service_entry

# Keys that change which TTS catalog is listed (same cache key as GET /api/tts-config).
TTS_ROUTING_KEYS = frozenset({"tts_id", "tts_server", "tts_model", "tts_function_id"})


def _resolve_tts_from_config(config: dict[str, Any]) -> tuple[str, str, str, str]:
    default_tts = load_service_entry("tts", "")
    server = str(config.get("tts_server", "") or default_tts.get("server", "") or "")
    voice_id = str(config.get("tts_voice_id", "") or default_tts.get("voice_id", "") or "")
    function_id = str(config.get("tts_function_id", "") or default_tts.get("function_id", "") or "")
    model = str(config.get("tts_model", "") or default_tts.get("model", "") or "")
    return server, voice_id, function_id, model


def _voice_ids_from_catalog(tts_config: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for entry in tts_config.get("voices") or []:
        if not isinstance(entry, dict):
            continue
        voice_id = str(entry.get("id") or "").strip()
        if voice_id:
            ids.append(voice_id)
    return ids


def _pick_default_voice(tts_config: dict[str, Any], *, catalog_default: str) -> str:
    """Mirror RTVI VoiceSettings soft default: YAML/catalog default → defaultVoiceId → EN-US → first."""
    voices = _voice_ids_from_catalog(tts_config)
    voice_set = set(voices)
    if catalog_default and catalog_default in voice_set:
        return catalog_default
    default_voice_id = str(tts_config.get("defaultVoiceId") or "").strip()
    if default_voice_id and default_voice_id in voice_set:
        return default_voice_id
    for voice_id in voices:
        if "EN-US" in voice_id.upper():
            return voice_id
    if voices:
        return voices[0]
    return catalog_default or default_voice_id


def resolve_realtime_tts_voice(
    config: dict[str, Any],
    *,
    voice_was_set: bool = False,
    tts_routing_changed: bool = False,
) -> str | None:
    """Resolve ``tts_voice_id`` against the TTS voice catalog (same list path as RTVI UI).

    Uses :func:`get_tts_config` — cache-first voice list (same as ``GET /api/tts-config``).
    ``prewarm_tts`` runs only when this TTS routing key has never been listed
    (first use or after a model/server switch). Subsequent voice checks reuse
    the cached list. Does **not** run synthesis warmup for membership checks.

    When the requested voice is missing/empty/unknown, logs a warning and falls
    back to the catalog default. Mutates ``config["tts_voice_id"]`` when a
    fallback is applied.

    Returns:
        The resolved voice id when resolution ran, otherwise ``None``.
    """
    if not voice_was_set and not tts_routing_changed:
        return None

    default_tts = load_service_entry("tts", "")
    catalog_default = str(default_tts.get("voice_id", "") or "").strip()
    server, current_voice, function_id, model = _resolve_tts_from_config(config)
    requested = str(config.get("tts_voice_id", "") or "").strip() if voice_was_set else current_voice

    if not server:
        fallback = requested or catalog_default
        if voice_was_set and requested and requested != catalog_default:
            logger.warning(
                f"Realtime TTS voice {requested!r} cannot be checked (TTS server not configured); using {fallback!r}"
            )
        if fallback:
            config["tts_voice_id"] = fallback
        return fallback or None

    # Prefer listing with the catalog/default voice id (matches UI /api/tts-config).
    list_voice = catalog_default or requested or current_voice or "default"
    tts_config = get_tts_config(server, list_voice, function_id, model)
    voices = set(_voice_ids_from_catalog(tts_config))
    fallback = _pick_default_voice(tts_config, catalog_default=catalog_default)

    if not voices:
        # Empty catalog (TTS unreachable or no subvoices): prefer YAML/catalog default.
        resolved = fallback or catalog_default or requested
        if voice_was_set and requested and resolved and requested != resolved:
            logger.warning(
                f"Realtime TTS voice catalog empty at {server}; cannot verify {requested!r}; using {resolved!r}"
            )
        elif tts_routing_changed:
            logger.warning(
                f"Realtime TTS voice catalog empty after nvidia TTS routing change at {server}; using {resolved!r}"
            )
        if resolved:
            config["tts_voice_id"] = resolved
        return resolved or None

    if requested and requested in voices:
        config["tts_voice_id"] = requested
        logger.info(f"Realtime TTS voice accepted from catalog: voice={requested} server={server}")
        return requested

    resolved = fallback or next(iter(voices))
    if requested:
        logger.warning(
            f"Unknown TTS voice {requested!r} for server={server}; falling back to catalog default {resolved!r}"
        )
    elif tts_routing_changed:
        logger.warning(f"TTS routing changed; resolving voice against new catalog at {server} → {resolved!r}")
    config["tts_voice_id"] = resolved
    return resolved


def tts_routing_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    """Return True when nvidia/TTS routing keys differ between configs."""
    return any(str(previous.get(key, "") or "") != str(current.get(key, "") or "") for key in TTS_ROUTING_KEYS)
