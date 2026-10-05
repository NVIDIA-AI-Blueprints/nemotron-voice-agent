# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utility functions."""

import ipaddress
import json
import math
import os
import socket
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse, urlunparse

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_FILENAME = "prompts.yaml"
TOOLS_FILENAME = "tools.yaml"
SELF_HOSTED_RECIPES: tuple[str, ...] = ("server", "single-gpu")
SERVICE_RECIPES: tuple[str, ...] = ("auto", "cloud", *SELF_HOSTED_RECIPES)
SERVICE_CATEGORIES: tuple[str, ...] = ("llm", "asr", "tts")
ExampleServiceConfig = Mapping[str, Any]


class _ExampleBinding(NamedTuple):
    services: dict[str, tuple[str, ...]] | None
    settings: dict[str, dict[str, dict[str, object]]]
    categories: dict[str, str]
    capabilities: frozenset[str]


_NO_BINDING = _ExampleBinding(None, {}, {}, frozenset())
_service_context: ContextVar[_ExampleBinding | None] = ContextVar("service_context", default=None)
_active_binding: _ExampleBinding = _NO_BINDING


def _services_path() -> Path:
    override = os.getenv("SERVICES_PATH", "").strip()
    return Path(override) if override else PROJECT_ROOT / "services.yaml"


def service_recipe() -> str:
    """Return ``SERVICE_RECIPE``, or ``auto`` to detect the self-hosted section by reachability."""
    recipe = os.getenv("SERVICE_RECIPE", "").strip().lower() or "auto"
    if recipe not in SERVICE_RECIPES:
        raise RuntimeError(f"SERVICE_RECIPE={recipe!r} must be one of {', '.join(SERVICE_RECIPES)}")
    return recipe


_SLOT_AGNOSTIC_KEYS: frozenset[str] = frozenset({"pipeline_mode", "prompt_key", "prompt_content", "tool_choice"})


def _example_binding(example: ExampleServiceConfig | None) -> _ExampleBinding:
    """Read an example's ``services``, ``settings``, ``categories``, and ``capabilities``."""
    if not example or not example.get("services"):
        return _NO_BINDING
    return _ExampleBinding(
        {slot: tuple(keys) for slot, keys in example["services"].items()},
        example.get("settings") or {},
        example.get("categories") or {},
        frozenset(example.get("capabilities") or ()),
    )


def set_active_services(example: ExampleServiceConfig | None) -> None:
    """Declare the process-wide example service config; ``None`` disables filtering."""
    global _active_binding
    _active_binding = _example_binding(example)


def set_service_context(example: ExampleServiceConfig | None) -> None:
    """Bind one example's service config to the current request context."""
    _service_context.set(_example_binding(example))


def clear_service_context() -> None:
    """Clear the request-scoped service binding."""
    _service_context.set(None)


def _effective_binding() -> _ExampleBinding:
    context = _service_context.get()
    return context if context is not None else _active_binding


def _effective_services() -> dict[str, tuple[str, ...]] | None:
    return _effective_binding().services


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


def load_yaml_file(filepath: Path, *, required: bool = False) -> dict:
    """Load and return the contents of a YAML file as a dict.

    Returns an empty dict if the file is absent, unreadable, malformed, or
    the parsed value is not a mapping. With ``required``, raises
    ``RuntimeError`` in those cases instead.
    """
    try:
        data = yaml.safe_load(filepath.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        if required:
            raise RuntimeError(f"Failed to load YAML from {filepath}: {exc}") from exc
        if filepath.is_file():
            logger.warning(f"Failed to load YAML from {filepath}: {exc}")
        return {}
    if isinstance(data, dict):
        return data
    if required:
        raise RuntimeError(f"YAML root must be a mapping: {filepath}")
    return {}


def is_nvcf(server: str) -> bool:
    """Auto-detect if a gRPC server is NVIDIA Cloud Functions (requires SSL)."""
    return "nvcf.nvidia.com" in server


def is_streaming_llm_url(url: str) -> bool:
    """Return True when an LLM ``base_url`` speaks native StreamingInput over WebSocket."""
    return url.startswith(("ws://", "wss://"))


def _normalize_services_catalog(data: object) -> dict:
    """Normalize a services catalog into ``{category: {key: entry}}``."""
    src = data if isinstance(data, dict) else {}
    return {category: dict(section) for category, section in src.items() if isinstance(section, dict)}


def _is_container_runtime() -> bool:
    """Return ``True`` when running under Compose (``APP_RUNTIME=container``)."""
    return os.getenv("APP_RUNTIME", "").strip().lower() == "container"


class SpeechPorts(NamedTuple):
    """Container and published host ports of a speech NIM sidecar."""

    grpc: int
    host_grpc: int
    health: int
    host_health: int


LOCAL_SPEECH_PORTS: dict[str, dict[str, SpeechPorts]] = {
    "asr": {
        "nemotron-asr-streaming-english": SpeechPorts(50052, 50152, 9001, 9001),
        "nemotron-asr-streaming-multilingual": SpeechPorts(50052, 50252, 9001, 9101),
        "parakeet-ctc-asr": SpeechPorts(50052, 50352, 9001, 9201),
        "parakeet-rnnt-asr": SpeechPorts(50052, 50452, 9001, 9301),
    },
    "tts": {
        "magpie-multilingual-tts-service": SpeechPorts(50051, 50151, 9000, 9000),
        "chatterbox-tts-service": SpeechPorts(50051, 50251, 9000, 9100),
        "magpie-zeroshot-tts-service": SpeechPorts(50051, 50351, 9000, 9200),
    },
}

_HOST_RUNTIME_PORT_OVERRIDES: dict[tuple[str, int], int] = {
    ("nvidia-llm", 8000): 18000,
    ("nemotron-3-super", 8000): 18001,
    ("nvidia-llm-omni", 8000): 18002,
    ("nvidia-llm-vllm", 8000): 18000,
    **{
        (service, ports.grpc): ports.host_grpc
        for services in LOCAL_SPEECH_PORTS.values()
        for service, ports in services.items()
    },
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
    if field not in {"base_url", "server", "streaming_url"}:
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
        for field in ("base_url", "server", "streaming_url"):
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


NVCF_GRPC_SERVER = "grpc.nvcf.nvidia.com:443"
NVCF_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
CLOUD_SOURCE = "cloud-nim"
SELF_HOSTED_SOURCE = "self-hosted"


def _entry_endpoint(entry: dict) -> str:
    return str(entry.get("server") or entry.get("base_url") or "")


def _strip_nvcf(entry: dict) -> dict:
    return {key: value for key, value in entry.items() if key != "nvcf"}


def _nvcf_entry(entry: dict) -> dict | None:
    """Return the NVIDIA Cloud variant of a ``server`` entry, or ``None`` when it has no ``nvcf`` key."""
    if "nvcf" not in entry:
        return None
    overrides = entry.get("nvcf") or {}
    if not isinstance(overrides, dict):
        raise RuntimeError(f"services.yaml nvcf must be a mapping (got {overrides!r})")
    out = _strip_nvcf(entry)
    if "server" in out:
        out["server"] = NVCF_GRPC_SERVER
    if "base_url" in out:
        out["base_url"] = NVCF_LLM_BASE_URL
    out.update(overrides)
    return out


def _nvcf_catalog(server_catalog: dict) -> dict:
    return {
        category: {key: cloud for key, entry in section.items() if (cloud := _nvcf_entry(entry)) is not None}
        for category, section in server_catalog.items()
    }


_SETTING_TYPES = frozenset({"bool", "int", "float", "enum"})


def _coerce_setting(name: str, spec: dict, raw: object) -> object:
    """Validate one client value against its schema, falling back to the default."""
    default = spec.get("default")
    if raw is None:
        return default
    kind = spec.get("type")
    try:
        if kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
                return raw.strip().lower() == "true"
            raise ValueError
        if kind == "enum":
            if raw in (spec.get("options") or []):
                return raw
            raise ValueError
        if isinstance(raw, bool):
            raise ValueError
        value: int | float = int(raw) if kind == "int" else float(raw)
        if not math.isfinite(value):
            raise ValueError
    except (TypeError, ValueError):
        logger.warning(f"Ignoring invalid value for setting {name!r}: {raw!r}")
        return default
    if spec.get("min") is not None:
        value = max(value, spec["min"])
    if spec.get("max") is not None:
        value = min(value, spec["max"])
    return value


def _set_path(target: dict, path: str, value: object) -> None:
    *parents, leaf = path.split(".")
    for part in parents:
        child = target.get(part)
        if not isinstance(child, dict):
            child = target[part] = {}
        target = child
    target[leaf] = value


def _pop_path(target: dict, path: str) -> None:
    *parents, leaf = path.split(".")
    for part in parents:
        target = target.get(part)
        if not isinstance(target, dict):
            return
    target.pop(leaf, None)


def apply_service_settings(extra_params: dict, schema: Mapping[str, dict], values: Mapping[str, object]) -> dict:
    """Return ``extra_params`` with each schema setting written at its ``path``.

    Values come from ``values`` when valid, else the schema default. A setting
    whose ``requires`` setting resolves falsy is removed instead.
    """
    resolved = {name: _coerce_setting(name, spec, values.get(name)) for name, spec in schema.items()}
    out = json.loads(json.dumps(extra_params))
    for name, spec in schema.items():
        requirement = spec.get("requires")
        if resolved[name] is None or (requirement and not resolved.get(requirement)):
            _pop_path(out, spec["path"])
        else:
            _set_path(out, spec["path"], resolved[name])
    return out


def _load_settings_profiles() -> dict[str, dict]:
    """Load ``settings.yaml`` profiles, with each setting's ``path`` defaulting to its name."""
    profiles: dict[str, dict] = {}
    for profile_name, profile in load_yaml_file(_services_path().with_name("settings.yaml"), required=True).items():
        if str(profile_name).startswith("x-"):
            continue
        if not isinstance(profile, dict):
            raise RuntimeError(f"settings.yaml profile {profile_name!r} must be a mapping")
        for name, spec in profile.items():
            if not isinstance(spec, dict) or spec.get("type") not in _SETTING_TYPES:
                raise RuntimeError(
                    f"settings.yaml {profile_name}.{name} needs a type ({', '.join(sorted(_SETTING_TYPES))})"
                )
        profiles[profile_name] = {name: {**spec, "path": spec.get("path") or name} for name, spec in profile.items()}
    return profiles


_MISSING = object()


def _get_path(source: dict, path: str) -> object:
    for part in path.split("."):
        if not isinstance(source, dict) or part not in source:
            return _MISSING
        source = source[part]
    return source


def _entry_settings_schema(profile: Mapping[str, dict], extra_params: dict) -> dict[str, dict]:
    """Use a value already set at a setting's ``path`` in ``extra_params`` as that entry's default."""
    schema: dict[str, dict] = {}
    for name, spec in profile.items():
        value = _get_path(extra_params, spec["path"])
        schema[name] = spec if value is _MISSING else {**spec, "default": value}
    return schema


def _attach_settings(
    catalog: dict, profiles: Mapping[str, dict], slot_settings: Mapping[str, Mapping[str, object]]
) -> dict:
    """Replace each ``settings`` profile name with its schema and bake defaults into ``extra_params``.

    ``slot_settings`` holds the example's per-service defaults from the registry
    (``{slot: {key: {setting: value}}}``); they win over the entry's
    ``extra_params`` and the profile ``default``.
    """
    out: dict = {}
    for category, section in catalog.items():
        out[category] = {}
        for key, entry in section.items():
            overrides = slot_settings.get(category, {}).get(key, {})
            profile_name = entry.get("settings")
            if profile_name is None:
                out[category][key] = entry
                continue
            if profile_name not in profiles:
                raise RuntimeError(f"services.yaml {category}.{key} references unknown settings {profile_name!r}")
            profile = profiles[profile_name]
            field_map = _HYDRATION_FIELD_MAPS.get(category, {})
            in_extra_params = "extra_params" in field_map
            if in_extra_params:
                base = parse_json_dict(entry.get("extra_params", ""), label="extra_params")
            else:
                base = {field: value for field, value in entry.items() if field != "settings"}
                unmapped = sorted(spec["path"] for spec in profile.values() if spec["path"] not in field_map)
                if unmapped:
                    raise RuntimeError(f"Settings profile {profile_name!r} sets {unmapped}, unknown to {category}")
            schema = _entry_settings_schema(profile, base)
            unknown = sorted(set(overrides) - set(schema))
            if unknown:
                raise RuntimeError(f"Registry settings {unknown} are not in settings profile {profile_name!r}")
            schema = {
                name: {**spec, "default": overrides[name]} if name in overrides else spec
                for name, spec in schema.items()
            }
            applied = apply_service_settings(base, schema, {})
            if in_extra_params:
                out[category][key] = {
                    **entry,
                    "settings": schema,
                    "extra_params": json.dumps(applied, separators=(",", ":")),
                }
            else:
                out[category][key] = {**applied, "settings": schema}
    return out


def _select_example_services(catalog: dict, binding: _ExampleBinding) -> dict:
    """Keep only the example's slots and keys, in registry order; ``categories`` maps a slot to another category."""
    if binding.services is None:
        return catalog
    selected: dict = {}
    for slot, keys in binding.services.items():
        section = catalog.get(binding.categories.get(slot, slot), {})
        selected[slot] = {key: section[key] for key in keys if key in section}
    return selected


def service_catalog_sources(example: ExampleServiceConfig | None = None) -> list[tuple[str, dict]]:
    """Return ``(source, catalog)`` pairs for the active recipe, self-hosted first.

    ``example`` supplies ``services`` (the slots and keys to keep, in default
    order), ``categories`` (the catalog category a slot reads), and ``settings``
    (per-service setting defaults). It defaults to the example bound by
    :func:`set_service_context`.
    """
    binding = _effective_binding() if example is None else _example_binding(example)
    data = load_yaml_file(_services_path(), required=True)
    recipe = service_recipe()
    sources: list[tuple[str, dict]] = []
    if recipe == "auto":
        if self_hosted := _detect_self_hosted_catalog(data, binding):
            sources.append((SELF_HOSTED_SOURCE, self_hosted))
    elif recipe != "cloud":
        sources.append((SELF_HOSTED_SOURCE, _self_hosted_section(data, recipe, binding)))
    if nvidia_cloud_available():
        server_catalog = _normalize_services_catalog(data.get("server"))
        sources.append((CLOUD_SOURCE, _select_example_services(_nvcf_catalog(server_catalog), binding)))
    profiles = _load_settings_profiles()
    return [(source, _attach_settings(catalog, profiles, binding.settings)) for source, catalog in sources]


def _self_hosted_section(data: dict, recipe: str, binding: _ExampleBinding) -> dict:
    """Return one self-hosted section without ``nvcf`` keys, rewritten for the runtime and filtered by ``binding``."""
    section = {
        category: {key: _strip_nvcf(entry) for key, entry in entries.items() if isinstance(entry, dict)}
        for category, entries in _normalize_services_catalog(data.get(recipe)).items()
    }
    return _select_example_services(_rewrite_local_runtime_endpoints(section), binding)


def _reachable_endpoints(endpoints: Iterable[str]) -> set[str]:
    """Probe endpoints in parallel so one slow or unresolvable host does not serialize the others."""
    unique = sorted({endpoint for endpoint in endpoints if endpoint})
    if not unique:
        return set()
    with ThreadPoolExecutor(max_workers=min(len(unique), 16)) as pool:
        results = list(pool.map(is_endpoint_reachable, unique))
    return {endpoint for endpoint, ok in zip(unique, results, strict=True) if ok}


def _detect_self_hosted_catalog(data: dict, binding: _ExampleBinding) -> dict:
    """Pick the self-hosted section with the most reachable entries.

    Within that section, a slot shows only its reachable entries, or all of
    them when none is up, so defaults still resolve and the session readiness
    check reports the stopped service. Ties go to the section listed first in
    :data:`SELF_HOSTED_RECIPES`. When nothing is reachable, returns ``{}`` if
    NVIDIA Cloud is available, else the first section.
    """
    sections = {recipe: _self_hosted_section(data, recipe, binding) for recipe in SELF_HOSTED_RECIPES}
    reachable = _reachable_endpoints(
        _entry_endpoint(entry)
        for section in sections.values()
        for entries in section.values()
        for entry in entries.values()
    )
    counts = {
        recipe: sum(_entry_endpoint(entry) in reachable for entries in section.values() for entry in entries.values())
        for recipe, section in sections.items()
    }
    best = max(SELF_HOSTED_RECIPES, key=lambda recipe: counts[recipe])
    if not counts[best]:
        return {} if nvidia_cloud_available() else sections[best]
    return {
        category: {key: entry for key, entry in entries.items() if _entry_endpoint(entry) in reachable} or entries
        for category, entries in sections[best].items()
    }


def _load_effective_services_catalog() -> dict:
    """Merge the active sources so self-hosted entries win on shared keys."""
    merged: dict = {}
    for _, catalog in service_catalog_sources():
        for category, section in catalog.items():
            target = merged.setdefault(category, {})
            for key, entry in section.items():
                target.setdefault(key, entry)
    return merged


# For each category, map YAML field → session-body field, plus the body fields
# holding the client's editable settings and streaming-input choice. YAML is the
# source of truth for built-in selections. Client-overridable fields keep a
# non-empty user value (voice picker / session language) instead of the catalog default.
_CATALOG_HYDRATION: tuple[tuple[str, str, dict[str, str], str, str], ...] = (
    (
        "llm_id",
        "llm",
        {
            "model_id": "model_id",
            "base_url": "base_url",
            "system_prompt": "system_prompt",
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "extra_params": "extra_params",
        },
        "llm_settings",
        "llm_streaming",
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
        "thinker_llm_settings",
        "",
    ),
    (
        "asr_id",
        "asr",
        {
            "server": "asr_server",
            "model": "asr_model",
            "function_id": "asr_function_id",
            "language_code": "asr_language_code",
            "automatic_punctuation": "asr_automatic_punctuation",
        },
        "asr_settings",
        "",
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
            "language_code": "tts_language_code",
            "zero_shot_audio_prompt_file": "tts_zero_shot_audio_prompt_file",
        },
        "tts_settings",
        "",
    ),
)
_HYDRATION_FIELD_MAPS: dict[str, dict[str, str]] = {slot: fields for _, slot, fields, _, _ in _CATALOG_HYDRATION}
_CATALOG_ONLY_BODY_FIELDS = frozenset({"tts_zero_shot_audio_prompt_file"})
_SLOT_CONFIG_KEYS: dict[str, frozenset[str]] = {
    slot: frozenset({id_field, *fields.values(), settings_field, streaming_field} - {""}) - _CATALOG_ONLY_BODY_FIELDS
    for id_field, slot, fields, settings_field, streaming_field in _CATALOG_HYDRATION
}
SESSION_CONFIG_KEYS: frozenset[str] = _SLOT_AGNOSTIC_KEYS.union(*_SLOT_CONFIG_KEYS.values())

# Body fields the client may set explicitly; catalog hydration must not overwrite them.
_CLIENT_OVERRIDABLE_BODY_FIELDS = frozenset(
    {"asr_language_code", "tts_language_code", "tts_voice_id", "max_tokens", "temperature"}
)


def hydrate_config_from_catalog(config: dict) -> None:
    """Overwrite detail fields in ``config`` from YAML for built-in selections.

    Mutates ``config`` in place. Custom (user-authored) entries are left alone so
    the client-provided details continue to drive the pipeline.
    """
    for id_field, category, field_map, settings_field, streaming_field in _CATALOG_HYDRATION:
        user_settings = config.pop(settings_field, None) if settings_field else None
        user_streaming = config.pop(streaming_field, None) if streaming_field else None
        entry = load_service_entry_by_id(category, config.get(id_field, ""))
        if not entry:
            continue
        for yaml_field, body_field in field_map.items():
            if body_field in _CLIENT_OVERRIDABLE_BODY_FIELDS:
                raw_user_value = config.get(body_field, "")
                if isinstance(raw_user_value, bool | dict | list):
                    # Reject non-scalar overrides; fall through to catalog.
                    pass
                else:
                    user_value = str(raw_user_value).strip() if raw_user_value not in ("", None) else ""
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
        if settings_field and isinstance(entry.get("settings"), dict):
            user_values = parse_json_dict(user_settings, label=settings_field)
            if "extra_params" in field_map:
                extra_field = field_map["extra_params"]
                extra = apply_service_settings(
                    parse_json_dict(config.get(extra_field, ""), label=extra_field),
                    entry["settings"],
                    user_values,
                )
                if extra:
                    config[extra_field] = json.dumps(extra, separators=(",", ":"))
            else:
                applied = apply_service_settings({}, entry["settings"], user_values)
                for spec in entry["settings"].values():
                    value = _get_path(applied, spec["path"])
                    if value is _MISSING:
                        config.pop(field_map[spec["path"]], None)
                    else:
                        config[field_map[spec["path"]]] = str(value)
        if _streaming_requested(user_streaming) and entry.get("streaming_url"):
            config[field_map["base_url"]] = entry["streaming_url"]


def _streaming_requested(raw: object) -> bool:
    """Return whether the client asked for streaming input and the bound example supports it."""
    requested = raw is True or (isinstance(raw, str) and raw.strip().lower() == "true")
    return requested and "streaming_input" in _effective_binding().capabilities


def filter_session_config(data: dict) -> dict:
    """Return a sanitized session config ready for the pipeline.

    Keeps only keys in ``SESSION_CONFIG_KEYS`` and hydrates built-in catalog
    selections from YAML (see :func:`hydrate_config_from_catalog`).
    Client-supplied ``tts_zero_shot_audio_prompt_file`` is always dropped;
    catalog hydration may re-add a trusted path afterward.
    """
    filtered = {k: v for k, v in data.items() if k in SESSION_CONFIG_KEYS and v not in ("", None)}
    active_services = _effective_services()
    if active_services is not None:
        allowed: set[str] = set(_SLOT_AGNOSTIC_KEYS)
        for slot in active_services:
            allowed |= _SLOT_CONFIG_KEYS.get(slot, frozenset())
        filtered = {k: v for k, v in filtered.items() if k in allowed}
    # Defense in depth: never trust a client path even if it bypasses the allowlists.
    filtered.pop("tts_zero_shot_audio_prompt_file", None)
    # Custom (non-catalog) selections skip hydration, so the raw client value would
    # reach ``normalize_lang_code`` in the pipeline. Keep only usable strings.
    raw_tts_language_code = filtered.get("tts_language_code")
    if raw_tts_language_code is not None:
        if isinstance(raw_tts_language_code, str) and raw_tts_language_code.strip():
            filtered["tts_language_code"] = raw_tts_language_code.strip()
        else:
            filtered.pop("tts_language_code", None)
    hydrate_config_from_catalog(filtered)
    return filtered


def load_service_entry_by_id(category: str, entry_id: str) -> dict:
    """Look up a built-in catalog entry by category and API id.

    Supports UI ids (``<source>:<key>``) and raw catalog keys for direct
    clients. Returns ``{}`` for custom or unknown entries.
    """
    if not entry_id or entry_id.startswith("custom-"):
        return {}
    if ":" in entry_id:
        source, key = entry_id.split(":", 1)
        catalog = next((catalog for name, catalog in service_catalog_sources() if name == source), {})
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


def service_api_entry(source: str, key: str, entry: dict) -> dict:
    """Return one catalog entry in the API shape, with its ``<source>:<key>`` id."""
    return {
        "id": f"{source}:{key}",
        "key": key,
        "name": str(entry.get("name") or key),
        "builtIn": True,
        "source": source,
        **{field: value for field, value in entry.items() if field != "name"},
    }


def build_services_api_response() -> dict:
    """Build the payload for ``GET /api/services``: self-hosted entries first, then NVIDIA Cloud."""
    sources = service_catalog_sources()
    services = _effective_services()
    categories: list[str] = list(services) if services is not None else []
    for _, catalog in sources:
        categories.extend(category for category in catalog if category not in categories)
    result: dict = {}
    for category in categories:
        entries = [
            {**service_api_entry(source, key, entry), "selected": False}
            for source, catalog in sources
            for key, entry in catalog.get(category, {}).items()
        ]
        if entries:
            entries[0]["selected"] = True
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


def nvidia_cloud_available() -> bool:
    """Return whether NVIDIA Cloud catalog entries can be used.

    True only when ``NVIDIA_API_KEY`` is a real key. Empty, unset, and the
    Compose placeholder ``not-needed`` keep cloud services hidden.
    """
    key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    return bool(key) and key.casefold() != "not-needed"


def nvidia_api_key(default: str = "not-needed") -> str:
    """Return NVIDIA_API_KEY, defaulting for local OpenAI-compatible endpoints.

    Empty, whitespace-only, and unset values become ``default`` so the OpenAI
    SDK can construct a client against local vLLM/NIM without a real cloud key.
    The value is stripped to stay consistent with ``nvidia_cloud_available()``.
    """
    key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    return key or default


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
