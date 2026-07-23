# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Service pre-warming to avoid blocking the event loop during first connection."""

import os

from loguru import logger
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from riva.client.proto import riva_asr_pb2

import config_store
from utils import is_nvcf, normalize_lang_code


def _create_tts_service(
    server: str,
    voice_id: str,
    function_id: str = "",
    model: str = "",
):
    tts_kwargs: dict = {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "server": server,
        "settings": NvidiaTTSSettings(voice=voice_id),
        "use_ssl": is_nvcf(server),
    }
    if function_id or model:
        tts_kwargs["model_function_map"] = {
            "function_id": function_id,
            "model_name": model,
        }
    return NvidiaTTSService(**tts_kwargs)


def _tts_cache_key(server: str, function_id: str = "", model: str = "") -> str:
    return f"tts:{server}:{function_id}:{model}"


def _parse_language_codes_param(raw: str) -> list[str]:
    """Parse comma-separated Nemotron Speech language_code model parameters."""
    if not raw or raw.strip().lower() == "auto":
        return []
    return [normalize_lang_code(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_tts_config(raw_config, model_prefix: str) -> dict:
    """Parse the TTS synthesis config into a frontend-friendly structure.

    Voice IDs are returned in full form (e.g. "Magpie-Multilingual.EN-US.Aria")
    so the frontend can use them as-is without any prefix manipulation.
    """
    if not raw_config or not raw_config.model_config:
        return {"languages": [], "voices": []}

    params = dict(raw_config.model_config[0].parameters)
    languages = _parse_language_codes_param(params.get("language_code", ""))
    subvoices_raw = params.get("subvoices", "")

    voices: list[dict] = []
    seen: set[str] = set()
    for entry in subvoices_raw.split(","):
        entry = entry.strip()
        if ":" not in entry or "." not in entry:
            continue
        short_id = entry.split(":")[0]
        parts = short_id.split(".")
        if len(parts) < 2:
            continue
        if short_id in seen:
            continue
        seen.add(short_id)
        full_id = f"{model_prefix}.{short_id}" if model_prefix else short_id
        lang = parts[0]
        name = ".".join(parts[1:])
        voices.append({"id": full_id, "name": name, "language": lang})

    return {
        "languages": languages,
        "voices": sorted(voices, key=lambda v: (v["language"], v["name"])),
    }


def _parse_asr_config(raw_config) -> dict:
    """Parse ASR recognition config into a frontend-friendly structure."""
    if not raw_config or not raw_config.model_config:
        return {"languages": []}

    params = dict(raw_config.model_config[0].parameters)
    return {"languages": _parse_language_codes_param(params.get("language_code", ""))}


def _tts_language_set(tts_config: dict) -> set[str]:
    langs = {_normalize_catalog_code(code) for code in tts_config.get("languages", []) if code}
    for voice in tts_config.get("voices", []):
        lang = voice.get("language") if isinstance(voice, dict) else None
        if lang:
            langs.add(_normalize_catalog_code(lang))
    return langs


def _normalize_catalog_code(code: str) -> str:
    return normalize_lang_code(code).lower()


_EMPTY_ASR_LANGUAGE_FALLBACK = "es-US"


def intersect_session_languages(asr_config: dict, tts_config: dict) -> list[str]:
    """Languages supported by both ASR and TTS (LLM prompt uses the TTS catalog)."""
    tts_langs = _tts_language_set(tts_config)
    if not tts_langs:
        return []

    asr_langs = {_normalize_catalog_code(code) for code in (asr_config or {}).get("languages", []) if code}
    if not asr_langs:
        asr_langs = {_normalize_catalog_code(_EMPTY_ASR_LANGUAGE_FALLBACK)}

    result = asr_langs & tts_langs

    return sorted(
        (normalize_lang_code(code) for code in result),
        key=str.lower,
    )


def build_session_languages(
    asr_server: str,
    asr_model: str,
    asr_function_id: str,
    tts_server: str,
    tts_voice_id: str,
    tts_function_id: str = "",
    tts_model: str = "",
) -> dict:
    """Return the ASR∩TTS language catalog and TTS voices for session configuration."""
    tts_config = prewarm_tts(
        tts_server,
        tts_voice_id,
        tts_function_id,
        tts_model,
    )
    asr_config = prewarm_asr(asr_server, asr_model, asr_function_id)
    languages = intersect_session_languages(asr_config, tts_config)
    return {
        "languages": languages,
        "voices": tts_config.get("voices", []),
        "defaultVoiceId": tts_config.get("defaultVoiceId", tts_voice_id),
    }


def _asr_cache_key(server: str, model: str, function_id: str) -> str:
    return f"asr:{server}:{model}:{function_id}"


def prewarm_tts(
    server: str,
    voice_id: str,
    function_id: str = "",
    model: str = "",
) -> dict:
    """Pre-warm a TTS server and cache its voice/language config.

    Returns the TTS config dict (languages, voices, defaultVoiceId).
    Results are cached per server + function_id + model in config_store so
    multiple cloud NIMs on ``grpc.nvcf.nvidia.com`` do not collide.
    """
    cache_key = _tts_cache_key(server, function_id, model)
    cached = config_store.get(cache_key)
    if cached:
        logger.debug(f"TTS config for {server} already cached")
        result = dict(cached)
        if voice_id:
            result["defaultVoiceId"] = voice_id
        config_store.set("tts", result)
        return result

    logger.info(f"Pre-warming TTS on {server} (this may take 10-20s on first run)...")
    try:
        svc = _create_tts_service(server, voice_id, function_id, model)
        svc._initialize_client()
        raw_config = svc._create_synthesis_config()

        model_prefix = voice_id.split(".")[0] if "." in voice_id else ""
        tts_config = _parse_tts_config(raw_config, model_prefix)
        tts_config["defaultVoiceId"] = voice_id
        tts_config["server"] = server

        config_store.set(cache_key, tts_config)
        config_store.set("tts", tts_config)

        n_langs = len(tts_config["languages"])
        n_voices = len(tts_config["voices"])
        logger.info(f"TTS pre-warmed ({server}) — {n_langs} languages, {n_voices} voices")
        return tts_config
    except Exception as e:
        logger.warning(f"TTS pre-warm failed for {server}: {e}")
        return {
            "languages": [],
            "voices": [],
            "defaultVoiceId": voice_id,
            "server": server,
            "error": str(e),
        }


_ASR_PREWARM_RPC_TIMEOUT_SECS = float(os.getenv("ASR_PREWARM_RPC_TIMEOUT_SECS", "5"))


def _fetch_asr_config(svc: NvidiaSTTService):
    return svc._asr_service.stub.GetRivaSpeechRecognitionConfig(
        riva_asr_pb2.RivaSpeechRecognitionConfigRequest(),
        timeout=_ASR_PREWARM_RPC_TIMEOUT_SECS,
    )


def prewarm_asr(server: str, model: str = "", function_id: str = "") -> dict:
    """Pre-warm an ASR server and cache its supported language codes.

    Uses Pipecat/Riva private hooks (``_initialize_client``, ``_asr_service.stub``)
    because Nemotron Speech does not yet expose a public language-catalog API.
    Revisit when upstream adds a supported discovery path.
    """
    cache_key = _asr_cache_key(server, model, function_id)
    cached = config_store.get(cache_key)
    if cached:
        logger.debug(f"ASR config for {server} already cached")
        return cached

    logger.info(f"Pre-warming ASR on {server}...")
    try:
        asr_kwargs: dict = {
            "api_key": os.getenv("NVIDIA_API_KEY"),
            "server": server,
            "use_ssl": is_nvcf(server),
        }
        if function_id or model:
            asr_kwargs["model_function_map"] = {
                "function_id": function_id,
                "model_name": model or "custom-asr",
            }
        svc = NvidiaSTTService(**asr_kwargs)
        svc._initialize_client()
        raw_config = _fetch_asr_config(svc)
        asr_config = _parse_asr_config(raw_config)
        asr_config["server"] = server
        asr_config["config_model"] = ""
        config_store.set(cache_key, asr_config)
        logger.info(f"ASR pre-warmed ({server}) — {len(asr_config['languages']) or 'all'} languages from model config")
        return asr_config
    except Exception as e:
        logger.warning(f"ASR pre-warm failed for {server}: {e}")
        return {"languages": [], "server": server, "error": str(e)}


def warmup_tts_synthesis(
    server: str,
    voice_id: str,
    function_id: str = "",
    model: str = "",
) -> bool:
    """Run a tiny synthesis request to verify the selected TTS is responsive."""
    logger.info(f"Warming up TTS synthesis on {server}...")
    try:
        svc = _create_tts_service(server, voice_id, function_id, model)
        svc._initialize_client()

        responses = svc._service.synthesize_online(
            "Hello.",
            svc._settings.voice,
            svc._settings.language,
            sample_rate_hz=16000,
            zero_shot_audio_prompt_file=None,
            zero_shot_quality=svc._settings.quality,
            custom_dictionary={},
        )
        for _ in responses:
            break

        logger.info(f"TTS synthesis warm-up completed ({server})")
        return True
    except Exception as e:
        logger.warning(f"TTS synthesis warm-up failed ({server}): {e}")
        return False


def load_voice_map(
    server: str = "",
    function_id: str = "",
    model: str = "",
) -> dict[str, str]:
    """``{lower_lang_code: first_voice_id}`` from the prewarm cache."""
    tts_config: dict = {}
    if server or function_id or model:
        cached = config_store.get(_tts_cache_key(server, function_id, model), {})
        if isinstance(cached, dict):
            tts_config = cached
    if not tts_config:
        legacy = config_store.get("tts", {})
        tts_config = legacy if isinstance(legacy, dict) else {}
    voices = tts_config.get("voices", [])
    result: dict[str, str] = {}
    for v in voices:
        lang = (v.get("language") or "").strip()
        vid = (v.get("id") or "").strip()
        if lang and vid and lang.lower() not in result:
            result[lang.lower()] = vid
    return result


def resolve_voice_for_language(
    language_code: str,
    preferred_voice_id: str = "",
    *,
    server: str = "",
    function_id: str = "",
    model: str = "",
) -> str:
    """Pick a TTS voice id for ``language_code`` from the prewarmed catalog."""
    normalized = normalize_lang_code(language_code).lower()
    tts_config: dict = {}
    if server or function_id or model:
        cached = config_store.get(_tts_cache_key(server, function_id, model), {})
        if isinstance(cached, dict):
            tts_config = cached
    if not tts_config:
        legacy = config_store.get("tts", {})
        tts_config = legacy if isinstance(legacy, dict) else {}
    voice_map = load_voice_map(server=server, function_id=function_id, model=model)
    if preferred_voice_id:
        for voice in tts_config.get("voices", []):
            if voice.get("id") == preferred_voice_id and voice.get("language", "").lower() == normalized:
                return preferred_voice_id
    voice_id = voice_map.get(normalized)
    if voice_id:
        return voice_id
    logger.warning(f"Multilingual: no TTS voice for language {language_code!r}")
    return ""
