# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Shared utility functions."""

import ipaddress
import json
import os
import socket
import time
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml
from loguru import logger

from runtime_platform import select_runtime_platform_catalog

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILENAME = "prompts.yaml"
TOOLS_FILENAME = "tools.yaml"
_service_context: ContextVar[tuple[Path, tuple[str, ...]] | None] = ContextVar("service_context", default=None)


def _services_cloud_path() -> Path:
    context = _service_context.get()
    if context:
        return context[0] / "services.cloud.yaml"
    return Path(os.getenv("SERVICES_CLOUD_PATH", str(PROJECT_ROOT / "src/examples/generic/services.cloud.yaml")))


def _services_local_path() -> Path:
    context = _service_context.get()
    if context:
        return context[0] / "services.local.yaml"
    return Path(os.getenv("SERVICES_LOCAL_PATH", str(PROJECT_ROOT / "src/examples/generic/services.local.yaml")))


_SLOT_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "llm": frozenset({"llm_id", "model_id", "base_url", "system_prompt", "max_tokens", "extra_params"}),
    "thinker-llm": frozenset(
        {"thinker_llm_id", "thinker_model_id", "thinker_base_url", "thinker_max_tokens", "thinker_extra_params"}
    ),
    "asr": frozenset({"asr_id", "asr_server", "asr_model", "asr_function_id", "asr_language_code"}),
    "tts": frozenset(
        {
            "tts_id",
            "tts_server",
            "tts_voice_id",
            "tts_function_id",
            "tts_model",
            "tts_synthesis_mode",
        }
    ),
}
_SLOT_AGNOSTIC_KEYS: frozenset[str] = frozenset({"pipeline_mode", "prompt_key", "prompt_content"})
_active_slots: frozenset[str] | None = None
_active_slot_order: tuple[str, ...] | None = None


def set_active_slots(slots: list[str] | tuple[str, ...] | None) -> None:
    """Declare which example slots are active; ``None`` disables filtering."""
    global _active_slots, _active_slot_order
    if slots:
        _active_slot_order = tuple(slots)
        _active_slots = frozenset(_active_slot_order)
    else:
        _active_slot_order = None
        _active_slots = None


def set_service_context(example_dir: str | Path, slots: Iterable[str] | None) -> None:
    """Bind service catalogs and active slots to the current request context."""
    _service_context.set((Path(example_dir), tuple(slots) if slots is not None else ()))


def _effective_active_slots() -> frozenset[str] | None:
    context = _service_context.get()
    if context is not None:
        return frozenset(context[1]) if context[1] else None
    return _active_slots


def _effective_slot_order() -> tuple[str, ...]:
    context = _service_context.get()
    if context is not None:
        return context[1]
    return _active_slot_order or ()


def resolve_prompt_catalog_path(module_file: str | Path) -> Path:
    """Return the prompts.yaml path beside an example module (``PROMPT_FILE_PATH`` env overrides)."""
    override = os.getenv("PROMPT_FILE_PATH", "").strip()
    return Path(override) if override else Path(module_file).resolve().parent / PROMPTS_FILENAME


def load_prompt_catalog(module_file: str | Path) -> dict:
    """Load the prompt catalog beside an example module."""
    return load_yaml_file(resolve_prompt_catalog_path(module_file))


def resolve_tools_catalog_path(module_file: str | Path) -> Path:
    """Return the tools.yaml path beside an example module (``TOOLS_FILE_PATH`` env overrides)."""
    override = os.getenv("TOOLS_FILE_PATH", "").strip()
    return Path(override) if override else Path(module_file).resolve().parent / TOOLS_FILENAME


def load_tools_catalog(module_file: str | Path) -> dict:
    """Load the tools catalog beside an example module."""
    return load_yaml_file(resolve_tools_catalog_path(module_file))


def resolve_tools_available(module_file: str | Path, prompt_key: str) -> list[str]:
    """Return the list of tool names declared under a prompt's ``tools_available``.

    Returns an empty list when the prompt is missing from the example catalog
    (e.g. a custom client-supplied prompt) or has no tools declared.
    """
    if not prompt_key:
        return []
    catalog = load_prompt_catalog(module_file)
    entry = catalog.get(prompt_key)
    if not isinstance(entry, dict):
        return []
    raw = entry.get("tools_available")
    if not isinstance(raw, list):
        return []
    return [name for name in raw if isinstance(name, str)]


def default_prompt_key(catalog: dict) -> str | None:
    """First entry marked ``default: true``, else first valid entry, else ``None``."""
    first_valid = None
    for key, value in catalog.items():
        if not isinstance(value, dict) or "content" not in value:
            continue
        if value.get("default") is True:
            return key
        if first_valid is None:
            first_valid = key
    return first_valid


def resolve_prompt(
    module_file: str | Path,
    prompt_content: str = "",
    prompt_key: str = "",
) -> tuple[str, str]:
    """Resolve ``(key, content)`` from the client body or the example's prompt catalog.

    Priority: client-provided content > ``prompt_key`` > ``PROMPT_SELECTOR`` env > catalog default.
    """
    if prompt_content:
        return prompt_key or "custom", prompt_content

    catalog_path = resolve_prompt_catalog_path(module_file)
    catalog = load_yaml_file(catalog_path)
    fallback = default_prompt_key(catalog)
    if fallback is None:
        raise KeyError(f"No prompts with content in {catalog_path}")

    requested = prompt_key or os.getenv("PROMPT_SELECTOR", "").strip() or fallback
    if requested not in catalog:
        logger.warning(f"Prompt '{requested}' not found in {catalog_path}; using '{fallback}'")
        requested = fallback
    logger.info(f"Loaded prompt from {catalog_path} [{requested}]")
    return requested, catalog[requested]["content"]


def render_prompt_addon(
    base_content: str,
    catalog: dict,
    addon_key: str,
    replacements: dict[str, str],
) -> str:
    """Append a prompt-catalog block with ``{placeholder}`` substitution."""
    entry = catalog.get(addon_key, {})
    addon = entry.get("content", "") if isinstance(entry, dict) else ""
    if not isinstance(addon, str) or not addon.strip():
        return base_content
    rendered = addon
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    return f"{base_content.rstrip()}\n\n{rendered.rstrip()}\n"


def load_yaml_file(filepath: Path) -> dict:
    """Load and return the contents of a YAML file as a dict.

    Returns an empty dict if the file is absent, unreadable, malformed, or
    the parsed value is not a mapping.
    """
    if not filepath.is_file():
        return {}
    try:
        data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning(f"Failed to load YAML from {filepath}: {exc}")
        return {}
    return data if isinstance(data, dict) else {}


def is_nvcf(server: str) -> bool:
    """Auto-detect if a gRPC server is NVIDIA Cloud Functions (requires SSL)."""
    return "nvcf.nvidia.com" in server


def _normalize_services_catalog(data: object) -> dict:
    """Normalize a services catalog into ``{category: {key: entry}}``."""
    src = data if isinstance(data, dict) else {}
    return {category: dict(section) for category, section in src.items() if isinstance(section, dict)}


def _is_container_runtime() -> bool:
    """Return ``True`` when running under Compose (``APP_RUNTIME=container``)."""
    return os.getenv("APP_RUNTIME", "").strip().lower() == "container"


_HOST_RUNTIME_PORT_OVERRIDES: dict[tuple[str, int], int] = {
    ("nvidia-llm", 8000): 18000,
    ("nvidia-llm-vllm", 8000): 18000,
    ("tts-service", 50051): 50151,
    ("chatterbox-tts-service", 50051): 50151,
    ("nemotron-asr-streaming-english", 50052): 50152,
    ("nemotron-asr-streaming-multilingual", 50052): 50152,
    ("parakeet-ctc-asr", 50052): 50152,
    ("parakeet-rnnt-asr", 50052): 50152,
}
_LOCAL_SERVICE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _is_compose_service_host(host: str) -> bool:
    """Return true for Docker Compose service names, not public hosts/IPs."""
    normalized = host.strip().lower()
    if not normalized or normalized in _LOCAL_SERVICE_HOSTS:
        return False
    try:
        ipaddress.ip_address(normalized)
        return False
    except ValueError:
        pass
    return "." not in normalized


def _rewrite_endpoint_for_host_runtime(field: str, value: str) -> str:
    """Convert Compose-oriented built-ins to host-accessible endpoints."""
    if field not in {"base_url", "server"}:
        return value

    parsed = urlparse(value if "://" in value else f"//{value}")
    host = parsed.hostname
    if not host:
        return value

    normalized_host = host.lower()
    if normalized_host == "host.docker.internal" or _is_compose_service_host(normalized_host):
        port = parsed.port
        target_port = _HOST_RUNTIME_PORT_OVERRIDES.get((normalized_host, port), port)
        netloc = "localhost" if target_port is None else f"localhost:{target_port}"
        if parsed.scheme:
            return urlunparse((parsed.scheme, netloc, parsed.path, parsed.params, parsed.query, parsed.fragment))
        suffix = parsed.path
        if parsed.query:
            suffix += f"?{parsed.query}"
        if parsed.fragment:
            suffix += f"#{parsed.fragment}"
        return f"{netloc}{suffix}"
    return value


def _rewrite_local_runtime_endpoints(catalog: dict) -> dict:
    """Rewrite local built-ins only when the backend runs outside Docker."""
    if _is_container_runtime():
        return catalog

    def _rewrite_entry(entry: dict) -> dict:
        out = dict(entry)
        for field in ("base_url", "server"):
            value = out.get(field)
            if isinstance(value, str):
                out[field] = _rewrite_endpoint_for_host_runtime(field, value)
        return out

    return {
        category: (
            {key: _rewrite_entry(entry) if isinstance(entry, dict) else entry for key, entry in section.items()}
            if isinstance(section, dict)
            else section
        )
        for category, section in catalog.items()
    }


def _section_default_key(section: dict, explicit_key: str = "") -> str:
    """Return ``explicit_key`` when present, else the first key in ``section``."""
    if explicit_key and explicit_key in section:
        return explicit_key
    return next(iter(section), "")


_REACHABILITY_TIMEOUT_SECS = 2.0
_REACHABILITY_CACHE_TTL_SECS = 5.0
_reachability_cache: dict[str, tuple[float, bool]] = {}


def parse_endpoint(server: str) -> tuple[str, int] | None:
    """Parse ``server`` (host[:port] or URL) into ``(host, port)``.

    Returns ``None`` when the address cannot be parsed.
    """
    if not server:
        return None
    parsed = urlparse(server if "://" in server else f"//{server}")
    host = parsed.hostname
    if not host:
        return None
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def is_endpoint_reachable(server: str) -> bool:
    """Return whether ``server`` accepts a TCP connection.

    Cached for ``_REACHABILITY_CACHE_TTL_SECS`` seconds per address so a single
    ``/api/services`` request does not probe each endpoint multiple times.
    """
    address = parse_endpoint(server)
    if address is None:
        return False
    cache_key = f"{address[0]}:{address[1]}"
    cached = _reachability_cache.get(cache_key)
    now = time.monotonic()
    if cached and now - cached[0] < _REACHABILITY_CACHE_TTL_SECS:
        return cached[1]
    try:
        with socket.create_connection(address, timeout=_REACHABILITY_TIMEOUT_SECS):
            ok = True
    except OSError:
        ok = False
    _reachability_cache[cache_key] = (now, ok)
    return ok


def _filter_reachable_entries(catalog: dict) -> dict:
    """Drop catalog entries whose endpoint is not reachable from this runtime."""
    filtered: dict = {}
    for category, section in catalog.items():
        if not isinstance(section, dict):
            filtered[category] = section
            continue
        kept = {
            key: entry
            for key, entry in section.items()
            if not isinstance(entry, dict)
            or is_endpoint_reachable(str(entry.get("server") or entry.get("base_url") or ""))
        }
        filtered[category] = kept
    return filtered


def _entry_endpoint(entry: dict) -> str:
    return str(entry.get("server") or entry.get("base_url") or "")


def _first_reachable_variant(variants: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    for platform_name, entry in variants:
        rewritten = _rewrite_local_runtime_endpoints({"_": {"_": entry}}).get("_", {}).get("_", {})
        if isinstance(rewritten, dict) and is_endpoint_reachable(_entry_endpoint(rewritten)):
            return platform_name, entry
    return None


def _load_cloud_services_catalog() -> dict:
    """Load cloud service entries from ``services.cloud.yaml``."""
    return _normalize_services_catalog(load_yaml_file(_services_cloud_path()))


def _load_local_services_catalog() -> dict:
    """Load local service entries for the configured platform."""
    local_path = _services_local_path()
    if not local_path.is_file():
        return _normalize_services_catalog({})
    data = load_yaml_file(local_path)
    if not isinstance(data, dict):
        return _normalize_services_catalog({})
    platform_data = select_runtime_platform_catalog(data)
    if platform_data is not None:
        return _rewrite_local_runtime_endpoints(_normalize_services_catalog(platform_data))

    variants: dict[str, dict[str, list[tuple[str, dict]]]] = {}
    for platform_name, platform_data in data.items():
        if not isinstance(platform_data, dict):
            continue
        for category, section in platform_data.items():
            if not isinstance(section, dict):
                continue
            cat_variants = variants.setdefault(str(category), {})
            for key, entry in section.items():
                if isinstance(entry, dict):
                    cat_variants.setdefault(str(key), []).append((str(platform_name), dict(entry)))

    merged: dict = {}
    for category, section in variants.items():
        target = merged.setdefault(category, {})
        for key, entries in section.items():
            first_entry = entries[0][1]
            if all(entry == first_entry for _, entry in entries):
                target[key] = first_entry
                continue
            active = _first_reachable_variant(entries)
            if active is not None:
                active_platform, active_entry = active
                target[key] = active_entry
                emitted = [active_entry]
            else:
                active_platform = ""
                emitted = []
            for platform_name, entry in entries:
                if platform_name == active_platform:
                    continue
                if any(entry == seen for seen in emitted):
                    continue
                target[f"{key}-{platform_name}"] = entry
                emitted.append(entry)
    return _rewrite_local_runtime_endpoints(_normalize_services_catalog(merged))


def _load_effective_services_catalog() -> dict:
    """Return the merged catalog combining cloud and reachable local entries.

    Reachable local entries are ordered first and win on shared keys, so the
    pipeline picks the deployed local service. When no local endpoint is
    reachable, the cloud catalog takes effect.
    """
    cloud = _load_cloud_services_catalog()
    local = _filter_reachable_entries(_load_local_services_catalog())
    merged: dict = {}
    for category in tuple(local) + tuple(key for key in cloud if key not in local):
        local_section = local.get(category, {})
        cloud_section = cloud.get(category, {})
        target = merged.setdefault(category, {})
        if isinstance(local_section, dict):
            target.update(local_section)
        if isinstance(cloud_section, dict):
            for key, entry in cloud_section.items():
                target.setdefault(key, entry)
    return merged


# Whitelist of keys accepted on session-config / offer bodies. Anything else
# sent by the client is dropped before the pipeline sees it.
SESSION_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "pipeline_mode",
        "llm_id",
        "asr_id",
        "tts_id",
        "model_id",
        "base_url",
        "system_prompt",
        "max_tokens",
        "extra_params",
        "thinker_llm_id",
        "thinker_model_id",
        "thinker_base_url",
        "thinker_extra_params",
        "thinker_max_tokens",
        "prompt_key",
        "prompt_content",
        "asr_server",
        "asr_model",
        "asr_function_id",
        "asr_language_code",
        "tts_server",
        "tts_voice_id",
        "tts_function_id",
        "tts_model",
        "tts_synthesis_mode",
    }
)

# For each category, map YAML field → session-body field. YAML is the source of
# truth for built-in selections. Client-overridable fields keep a non-empty
# user value (voice picker / session language) instead of the catalog default.
_CATALOG_HYDRATION: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "llm_id",
        "llm",
        {
            "model_id": "model_id",
            "base_url": "base_url",
            "system_prompt": "system_prompt",
            "max_tokens": "max_tokens",
            "extra_params": "extra_params",
        },
    ),
    (
        "thinker_llm_id",
        "thinker-llm",
        {
            "model_id": "thinker_model_id",
            "base_url": "thinker_base_url",
            "max_tokens": "thinker_max_tokens",
            "extra_params": "thinker_extra_params",
        },
    ),
    (
        "asr_id",
        "asr",
        {
            "server": "asr_server",
            "model": "asr_model",
            "function_id": "asr_function_id",
            "language_code": "asr_language_code",
        },
    ),
    (
        "tts_id",
        "tts",
        {
            "server": "tts_server",
            "function_id": "tts_function_id",
            "model": "tts_model",
            "voice_id": "tts_voice_id",
            "synthesis_mode": "tts_synthesis_mode",
        },
    ),
)

# Body fields the client may set explicitly; catalog hydration must not overwrite them.
_CLIENT_OVERRIDABLE_BODY_FIELDS = frozenset({"asr_language_code", "tts_voice_id"})


def hydrate_config_from_catalog(config: dict) -> None:
    """Overwrite detail fields in ``config`` from YAML for built-in selections.

    Mutates ``config`` in place. Custom (user-authored) entries are left alone so
    the client-provided details continue to drive the pipeline.
    """
    for id_field, category, field_map in _CATALOG_HYDRATION:
        entry = _load_catalog_entry_by_id(category, config.get(id_field, ""))
        if not entry:
            continue
        for yaml_field, body_field in field_map.items():
            if body_field in _CLIENT_OVERRIDABLE_BODY_FIELDS:
                user_value = str(config.get(body_field, "") or "").strip()
                if user_value:
                    config[body_field] = user_value
                    continue
            value = entry.get(yaml_field, "")
            if value in ("", None):
                config.pop(body_field, None)
            elif isinstance(value, dict | list):
                config[body_field] = json.dumps(value)
            else:
                config[body_field] = value if isinstance(value, str) else str(value)


def filter_session_config(data: dict) -> dict:
    """Return a sanitized session config ready for the pipeline.

    Keeps only keys in ``SESSION_CONFIG_KEYS`` and hydrates built-in catalog
    selections from YAML (see :func:`hydrate_config_from_catalog`).
    """
    filtered = {k: v for k, v in data.items() if k in SESSION_CONFIG_KEYS and v not in ("", None)}
    active_slots = _effective_active_slots()
    if active_slots is not None:
        allowed: set[str] = set(_SLOT_AGNOSTIC_KEYS)
        for slot in active_slots:
            allowed |= _SLOT_CONFIG_KEYS.get(slot, frozenset())
        filtered = {k: v for k, v in filtered.items() if k in allowed}
    hydrate_config_from_catalog(filtered)
    return filtered


def _load_catalog_entry_by_id(category: str, entry_id: str) -> dict:
    """Look up a built-in catalog entry by category and API id.

    Supports UI ids (``<source>:<key>``) and raw catalog keys for direct
    clients. Returns ``{}`` for custom or unknown entries.
    """
    if not entry_id or entry_id.startswith("custom-"):
        return {}
    if ":" in entry_id:
        source, key = entry_id.split(":", 1)
        if not key:
            return {}
        if source == "cloud-nim":
            catalog = _load_cloud_services_catalog()
        elif source == "self-hosted":
            catalog = _load_local_services_catalog()
        else:
            return {}
    else:
        key = entry_id
        catalog = _load_effective_services_catalog()
    entry = catalog.get(category, {}).get(key)
    return dict(entry) if isinstance(entry, dict) else {}


def load_service_entry(category: str, key: str) -> dict:
    """Load a catalog entry by category and key from the effective catalog.

    Falls back to the first entry in the category when the explicit ``key`` is
    not present (or empty).
    """
    data = _load_effective_services_catalog()
    section = data.get(category, {})
    if not isinstance(section, dict) or not section:
        return {}
    default_key = _section_default_key(section, key)
    if key and default_key != key:
        logger.warning(f"Service key '{key}' not found in category '{category}', using fallback '{default_key}'")
    return dict(section[default_key]) if default_key in section else {}


def _build_services_api_entries(section: dict, category: str, source: str) -> list[dict]:
    """Convert one catalog section into API entries for a source."""
    if not isinstance(section, dict):
        return []

    selected_key = _section_default_key(section)
    ordered_items = list(section.items())
    if selected_key in section:
        ordered_items.sort(key=lambda item: item[0] != selected_key)

    return [
        {
            "id": f"{source}:{key}",
            "name": val.get("name", key),
            "builtIn": True,
            "source": source,
            **{k: v for k, v in val.items() if k != "name"},
            "selected": key == selected_key,
        }
        for key, val in ordered_items
        if isinstance(val, dict)
    ]


def _services_api_categories(*catalogs: dict) -> tuple[str, ...]:
    """Return service categories ordered by the active example's ``slots``."""
    ordered: list[str] = list(_effective_slot_order())
    for catalog in catalogs:
        ordered.extend(category for category in catalog if category not in ordered)
    return tuple(ordered)


def build_services_api_response() -> dict:
    """Build the payload for ``GET /api/services`` with cloud and reachable local entries."""
    cloud_data = _load_cloud_services_catalog()
    local_data = _filter_reachable_entries(_load_local_services_catalog())
    active_slots = _effective_active_slots()
    result: dict = {}
    for category in _services_api_categories(cloud_data, local_data):
        if active_slots is not None and category not in active_slots:
            result[category] = []
            continue
        cloud_entries = _build_services_api_entries(cloud_data.get(category, {}), category, "cloud-nim")
        local_entries = _build_services_api_entries(local_data.get(category, {}), category, "self-hosted")
        entries = local_entries + cloud_entries
        selected_seen = False
        for entry in entries:
            if entry.get("selected") and not selected_seen:
                selected_seen = True
            else:
                entry["selected"] = False
        result[category] = entries
    return result


def parse_json_dict(raw: object, label: str = "JSON") -> dict:
    """Coerce a JSON string or mapping into a dict; return {} on empty/invalid input."""
    if raw in ("", None):
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if not isinstance(raw, str):
        logger.warning(f"Invalid {label}, expected JSON string or mapping, ignoring: {raw!r}")
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        logger.warning(f"Invalid {label}, ignoring: {raw!r}")
    return {}


def parse_env_int(name: str, default: int, min_value: int | None = None) -> int:
    """Parse an integer environment variable with safe fallback and optional minimum."""
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}, falling back to default {default}")
        value = default
    if min_value is not None and value < min_value:
        logger.warning(f"{name}={value!r} is below minimum {min_value}, clamping")
        return min_value
    return value


def parse_env_float(name: str, default: float, min_value: float | None = None) -> float:
    """Parse a float environment variable with safe fallback and optional minimum."""
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"Invalid {name}={raw!r}, falling back to default {default}")
        value = default
    if min_value is not None and value < min_value:
        logger.warning(f"{name}={value!r} is below minimum {min_value}, clamping")
        return min_value
    return value


def parse_env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable, treating empty as unset."""
    raw = (os.getenv(name) or "").strip()
    return raw.lower() == "true" if raw else default


def load_ipa_dictionary() -> dict | None:
    """Load a word-to-IPA pronunciation dictionary for ``NvidiaTTSService``.

    Reads ``TTS_IPA_FILE_PATH`` and parses JSON or YAML into a flat
    ``{grapheme: ipa}`` dict. Relative paths resolve from ``PROJECT_ROOT``.
    Returns ``None`` when unset, missing, malformed, or empty so callers can
    pass the result straight into ``custom_dictionary=``.
    """
    raw_path = os.getenv("TTS_IPA_FILE_PATH", "").strip()
    if not raw_path:
        return None

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        logger.warning(f"TTS_IPA_FILE_PATH points to a missing file, ignoring: {path}")
        return None

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        logger.warning(f"Failed to load TTS IPA dictionary from {path}: {exc}")
        return None

    if not isinstance(data, dict):
        logger.warning(f"TTS IPA dictionary must be a mapping, ignoring: {path}")
        return None

    dictionary = {
        str(word).strip(): str(ipa).strip() for word, ipa in data.items() if str(word).strip() and str(ipa).strip()
    }
    if not dictionary:
        logger.warning(f"TTS IPA dictionary is empty, ignoring: {path}")
        return None

    logger.info(f"Loaded TTS IPA dictionary from {path} ({len(dictionary)} entries)")
    return dictionary


def normalize_lang_code(code: str) -> str:
    """Normalize a language code to ISO casing (for example, ``DE-DE`` -> ``de-DE``)."""
    parts = code.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return code


def ensure_self_signed_cert(cert_dir: Path) -> tuple[str, str]:
    """Generate a self-signed TLS certificate if one doesn't already exist.

    Returns (cert_path, key_path).
    """
    cert_file = cert_dir / "cert.pem"
    key_file = cert_dir / "key.pem"
    if cert_file.exists() and key_file.exists():
        return str(cert_file), str(key_file)

    import datetime

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.UTC))
        .not_valid_after(datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    cert_dir.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    logger.info(f"Generated self-signed TLS cert at {cert_dir}")
    return str(cert_file), str(key_file)
