# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Service pre-warming to avoid blocking the event loop during first connection."""

import concurrent.futures
import json
import os
from collections.abc import Iterable

from loguru import logger
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from riva.client.proto import riva_asr_pb2

import config_store
from utils import is_nvcf, normalize_lang_code, nvidia_api_key, parse_env_float


def _create_tts_service(
    server: str,
    voice_id: str,
    function_id: str = "",
    model: str = "",
):
    tts_kwargs: dict = {
        "api_key": nvidia_api_key(),
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


def _voice_display_id(raw_id: str, prefix: str = "") -> tuple[str, str]:
    """Return (voice_id, display_name) for a catalog voice token.

    Nemo-speech.cpp lists ``magpietts.John`` while the app/catalog use ``John``.
    Riva Magpie keeps language-scoped ids like ``Magpie-Multilingual.EN-US.Aria``.
    """
    token = (raw_id or "").strip()
    if not token:
        return "", ""
    if prefix and token.lower().startswith(f"{prefix.lower()}."):
        name = token[len(prefix) + 1 :]
        # Bare speaker names (John) — keep short id. Language-scoped keep full id.
        if "." not in name:
            return name, name
        return token, name.split(".")[-1]
    if "." in token and token.split(".", 1)[0].lower() in {"magpietts", "magpie"}:
        name = token.rsplit(".", 1)[-1]
        return name, name
    name = token.rsplit(".", 1)[-1]
    return token, name


def _add_voice(
    voices: list[dict],
    seen: set[str],
    *,
    voice_id: str,
    name: str,
    language: str,
) -> None:
    lang = normalize_lang_code(language)
    key = f"{voice_id}|{lang}"
    if not voice_id or key in seen:
        return
    seen.add(key)
    voices.append({"id": voice_id, "name": name, "language": lang})


def _parse_tts_config(raw_config, model_prefix: str) -> dict:
    """Parse the TTS synthesis config into a frontend-friendly structure.

    Voice IDs are returned in full form so the frontend can use them as-is:

    - Magpie Multilingual ``subvoices`` entries look like ``EN-US.Aria:0`` and
      become ``{prefix}.EN-US.Aria``.
    - Magpie Zeroshot ``subvoices`` entries look like ``Female:0`` / ``Male:6``
      (no language in the token). The same voice IDs are valid across every
      ``language_code``, so each subvoice is expanded once per language as
      ``{prefix}.Female`` / ``{prefix}.Male``.
    - Nemo-speech.cpp Magpie lists plain names (``John,Sofia,...``) and/or a
      ``voices_by_language`` JSON map; those become short ids like ``John``.
    """
    if not raw_config or not raw_config.model_config:
        return {"languages": [], "voices": []}

    all_params = [dict(mc.parameters) for mc in raw_config.model_config]

    languages: list[str] = []
    seen_langs: set[str] = set()

    def add_language(code: str) -> str:
        normalized = normalize_lang_code(code)
        key = normalized.lower()
        if key not in seen_langs:
            seen_langs.add(key)
            languages.append(normalized)
        return normalized

    for params in all_params:
        for code in _parse_language_codes_param(params.get("language_code", "")):
            add_language(code)

    default_language = languages[0] if languages else "en-US"

    voices: list[dict] = []
    seen: set[str] = set()

    def add_config_language(code: str, config_languages: list[str], config_seen_langs: set[str]) -> str:
        normalized = add_language(code)
        key = normalized.lower()
        if key not in config_seen_langs:
            config_seen_langs.add(key)
            config_languages.append(normalized)
        return normalized

    for params in all_params:
        prefix = (model_prefix or params.get("voice_name", "") or "").strip()
        config_languages: list[str] = []
        config_seen_langs: set[str] = set()

        for code in _parse_language_codes_param(params.get("language_code", "")):
            add_config_language(code, config_languages, config_seen_langs)

        # Nemo-speech.cpp: rich per-locale catalog when present.
        voices_by_language_raw = params.get("voices_by_language", "")
        if voices_by_language_raw:
            try:
                catalog = json.loads(voices_by_language_raw)
            except (TypeError, json.JSONDecodeError):
                catalog = None
            if isinstance(catalog, dict):
                for lang_code, payload in catalog.items():
                    lang = add_config_language(str(lang_code), config_languages, config_seen_langs)
                    voice_list = payload.get("voices", []) if isinstance(payload, dict) else []
                    for raw_id in voice_list:
                        voice_id, name = _voice_display_id(str(raw_id), prefix)
                        _add_voice(voices, seen, voice_id=voice_id, name=name, language=lang)

        subvoices_raw = params.get("subvoices", "") or params.get("voices", "")
        for entry in subvoices_raw.split(","):
            entry = entry.strip()
            if not entry:
                continue

            # Riva Magpie: Name:index (index is optional for nemo-speech plain names).
            short_id = entry.split(":", 1)[0].strip() if ":" in entry else entry
            if not short_id:
                continue

            if ":" in entry and "." in short_id:
                # Magpie Multilingual: Language.VoiceName:index
                parts = short_id.split(".")
                if len(parts) < 2:
                    continue
                lang = add_config_language(parts[0], config_languages, config_seen_langs)
                name = ".".join(parts[1:])
                full_id = f"{prefix}.{short_id}" if prefix else short_id
                _add_voice(voices, seen, voice_id=full_id, name=name, language=lang)
                continue

            if ":" in entry:
                # Magpie Zeroshot: VoiceName:index (same voices for every locale)
                name = short_id
                full_id = f"{prefix}.{name}" if prefix else name
                for lang in config_languages or ["en-US"]:
                    _add_voice(voices, seen, voice_id=full_id, name=name, language=lang)
                continue

            # Nemo-speech.cpp: plain speaker names shared across locales (John,Sofia,...)
            voice_id, name = _voice_display_id(short_id, prefix)
            for lang in config_languages or ["en-US"]:
                _add_voice(voices, seen, voice_id=voice_id, name=name, language=lang)

    return {
        "languages": languages,
        "voices": sorted(voices, key=lambda v: (v["language"], v["name"])),
        "defaultLanguage": default_language,
    }


def _parse_asr_config(raw_config) -> dict:
    """Parse ASR recognition config into a frontend-friendly structure."""
    if not raw_config or not raw_config.model_config:
        return {"languages": []}

    languages: list[str] = []
    seen: set[str] = set()
    for model_config in raw_config.model_config:
        params = dict(model_config.parameters)
        for language in _parse_language_codes_param(params.get("language_code", "")):
            key = language.lower()
            if key not in seen:
                seen.add(key)
                languages.append(language)
    return {"languages": languages}


def _tts_language_set(tts_config: dict) -> set[str]:
    langs = {_normalize_catalog_code(code) for code in tts_config.get("languages", []) if code}
    for voice in tts_config.get("voices", []):
        lang = voice.get("language") if isinstance(voice, dict) else None
        if lang:
            langs.add(_normalize_catalog_code(lang))
    return langs


def _normalize_catalog_code(code: str) -> str:
    return normalize_lang_code(code).lower()


def _llm_language_set(supported_languages: Iterable[str] | str) -> set[str]:
    """Normalize LLM language capabilities to base language codes."""
    if isinstance(supported_languages, str):
        supported_languages = supported_languages.split(",")
    return {
        _normalize_catalog_code(code.strip()).split("-", 1)[0]
        for code in supported_languages
        if isinstance(code, str) and code.strip()
    }


def llm_supports_session_language(
    language_code: str,
    supported_languages: Iterable[str] | str | None,
) -> bool:
    """Return whether an LLM capability list supports a BCP-47 session locale.

    ``None`` preserves compatibility for custom LLMs whose capabilities are not
    declared. An explicit empty collection supports no session languages.
    """
    if supported_languages is None:
        return True
    language_base = _normalize_catalog_code(language_code).split("-", 1)[0]
    return language_base in _llm_language_set(supported_languages)


def validate_llm_session_language(
    language_code: str,
    supported_languages: Iterable[str] | str | None,
) -> None:
    """Reject a session locale that the selected built-in LLM cannot serve."""
    if llm_supports_session_language(language_code, supported_languages):
        return
    supported = ", ".join(sorted(_llm_language_set(supported_languages or []))) or "none"
    raise ValueError(
        f"Session language {normalize_lang_code(language_code)!r} is not supported by the selected LLM "
        f"(supported languages: {supported})"
    )


_EMPTY_ASR_LANGUAGE_FALLBACK = "es-US"


def intersect_session_languages(
    asr_config: dict,
    tts_config: dict,
    llm_supported_languages: Iterable[str] | str | None = None,
) -> list[str]:
    """Languages supported by ASR, TTS, and the selected LLM when declared."""
    tts_langs = _tts_language_set(tts_config)
    if not tts_langs:
        return []

    asr_langs = {_normalize_catalog_code(code) for code in (asr_config or {}).get("languages", []) if code}
    if not asr_langs:
        asr_langs = {_normalize_catalog_code(_EMPTY_ASR_LANGUAGE_FALLBACK)}

    result = asr_langs & tts_langs
    if llm_supported_languages is not None:
        llm_langs = _llm_language_set(llm_supported_languages)
        result = {code for code in result if code.split("-", 1)[0] in llm_langs}

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
    llm_supported_languages: Iterable[str] | str | None = None,
) -> dict:
    """Return the compatible session-language catalog and TTS voices."""
    tts_config = prewarm_tts(
        tts_server,
        tts_voice_id,
        tts_function_id,
        tts_model,
    )
    asr_config = prewarm_asr(asr_server, asr_model, asr_function_id)
    languages = intersect_session_languages(asr_config, tts_config, llm_supported_languages)
    return {
        "languages": languages,
        "voices": tts_config.get("voices", []),
        "defaultVoiceId": tts_config.get("defaultVoiceId", tts_voice_id),
    }


def _asr_cache_key(server: str, model: str, function_id: str) -> str:
    return f"asr:{server}:{model}:{function_id}"


def peek_cached_tts_config(
    server: str,
    voice_id: str = "",
    function_id: str = "",
    model: str = "",
) -> dict | None:
    """Return the cached TTS catalog for a routing key, or ``None`` on miss."""
    cached = config_store.get(_tts_cache_key(server, function_id, model))
    if not cached:
        return None
    result = dict(cached)
    if voice_id:
        result["defaultVoiceId"] = voice_id
    return result


def get_tts_config(
    server: str,
    voice_id: str = "",
    function_id: str = "",
    model: str = "",
) -> dict:
    """Return TTS languages/voices for a routing key, fetching only on cache miss.

    Same primitive as ``GET /api/tts-config``: when this ``server`` /
    ``function_id`` / ``model`` was already listed (first connect or after a TTS
    model switch), reuse the cached voice list. Call ``prewarm_tts`` only when
    that routing key has not been fetched yet.
    """
    cached = peek_cached_tts_config(server, voice_id, function_id, model)
    if cached is not None:
        # Keep legacy config_store["tts"] in sync for fallback readers.
        config_store.set("tts", cached)
        return cached
    return prewarm_tts(server, voice_id, function_id, model)


_TTS_PREWARM_RPC_TIMEOUT_SECS = parse_env_float("TTS_PREWARM_RPC_TIMEOUT_SECS", 20.0, min_value=1.0)


def prewarm_tts(
    server: str,
    voice_id: str,
    function_id: str = "",
    model: str = "",
) -> dict:
    """Fetch TTS voice/language config and cache it for the routing key.

    Returns the TTS config dict (languages, voices, defaultVoiceId).
    Results are cached per server + function_id + model in config_store so
    multiple cloud NIMs on ``grpc.nvcf.nvidia.com`` do not collide.

    Prefer :func:`get_tts_config` at call sites that only need membership /
    listing — it skips this fetch when the catalog is already cached.

    The list RPC is bounded by ``TTS_PREWARM_RPC_TIMEOUT_SECS`` (default 20s).
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

    def _fetch() -> dict:
        svc = _create_tts_service(server, voice_id, function_id, model)
        svc._initialize_client()
        raw_config = svc._create_synthesis_config()

        model_prefix = voice_id.split(".")[0] if "." in voice_id else ""
        tts_config = _parse_tts_config(raw_config, model_prefix)
        tts_config["defaultVoiceId"] = voice_id
        tts_config["server"] = server
        return tts_config

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            tts_config = pool.submit(_fetch).result(timeout=_TTS_PREWARM_RPC_TIMEOUT_SECS)

        config_store.set(cache_key, tts_config)
        config_store.set("tts", tts_config)

        n_langs = len(tts_config["languages"])
        n_voices = len(tts_config["voices"])
        logger.info(f"TTS pre-warmed ({server}) — {n_langs} languages, {n_voices} voices")
        return tts_config
    except concurrent.futures.TimeoutError:
        logger.warning(f"TTS pre-warm timed out for {server} after {_TTS_PREWARM_RPC_TIMEOUT_SECS}s")
        return {
            "languages": [],
            "voices": [],
            "defaultVoiceId": voice_id,
            "server": server,
            "error": f"TTS catalog list timed out after {_TTS_PREWARM_RPC_TIMEOUT_SECS}s",
        }
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
            "api_key": nvidia_api_key(),
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
