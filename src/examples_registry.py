# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Built-in voice-agent examples loaded from the root ``examples_registry.yaml``."""

import importlib
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import yaml

from runtime_platform import select_runtime_platform_catalog
from utils import is_endpoint_reachable


class ExampleEntry(TypedDict):
    """Raw registry entry for one example."""

    label: str
    slots: list[str]
    capabilities: list[str]
    agent_prompt_keys: list[str]
    defaults: dict[str, list[str] | str]
    welcome_message: bool
    bot: str


class EnrichedExample(ExampleEntry):
    """Registry entry plus derived id/key fields (``key == id``)."""

    id: str
    key: str


class ServiceDefault(TypedDict, total=False):
    """Resolved default service entry from an example's service catalog."""

    id: str
    key: str
    name: str
    builtIn: bool
    source: str


class PromptDefault(TypedDict, total=False):
    """Resolved default prompt entry from an example's prompt catalog."""

    key: str
    description: str
    content: str
    default: bool
    builtIn: bool
    tools: list[str]


_SRC_ROOT = Path(__file__).resolve().parent
_REGISTRY_PATH = _SRC_ROOT.parent / "examples_registry.yaml"


def _load_yaml_registry() -> dict:
    """Load the registry YAML, failing loudly because startup depends on it."""
    try:
        data = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to load examples registry from {_REGISTRY_PATH}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Examples registry root must be a mapping: {_REGISTRY_PATH}")
    return data


def _split_bot_spec(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise RuntimeError(f"Example bot must be 'module.path:attr' (got {spec!r})")
    module_path, attr = spec.split(":", 1)
    if not module_path or not attr:
        raise RuntimeError(f"Example bot must be 'module.path:attr' (got {spec!r})")
    return module_path, attr


@cache
def _resolve_bot(spec: str) -> Callable[..., Any]:
    """Resolve a ``module.path:callable`` string only when the bot is used."""
    module_path, attr = _split_bot_spec(spec)
    module = importlib.import_module(module_path)
    bot = getattr(module, attr, None)
    if not callable(bot):
        raise RuntimeError(f"Example bot target is not callable: {spec!r}")
    return bot


def resolve_bot(example: EnrichedExample) -> Callable[..., Any]:
    """Return the lazily imported bot callable for an example."""
    return _resolve_bot(example["bot"])


def example_module_file(example: EnrichedExample) -> Path:
    """Return the module file path for an example's bot spec without importing it."""
    module_path, _ = _split_bot_spec(example["bot"])
    module_parts = Path(*module_path.split("."))
    module_file = (_SRC_ROOT / module_parts).with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = _SRC_ROOT / module_parts / "__init__.py"
    if package_file.is_file():
        return package_file
    raise RuntimeError(f"Example {example['key']!r} bot module was not found: {example['bot']!r}")


def _load_yaml_mapping(path: Path) -> dict:
    """Load a YAML mapping from ``path``; return empty mapping when absent."""
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to load YAML from {path}") from exc
    return data if isinstance(data, dict) else {}


def _normalize_service_catalog(data: dict) -> dict[str, dict]:
    """Normalize a service catalog into ``{category: {key: entry}}``."""
    return {str(category): dict(section) for category, section in data.items() if isinstance(section, dict)}


def _entry_endpoint(entry: dict) -> str:
    return str(entry.get("server") or entry.get("base_url") or "")


def _rewrite_entry_for_host_runtime(entry: dict) -> dict:
    """Convert Compose endpoints to host-accessible endpoints outside Docker."""
    if os.getenv("APP_RUNTIME", "").strip().lower() == "container":
        return dict(entry)
    out = dict(entry)
    for field in ("base_url", "server"):
        value = out.get(field)
        if not isinstance(value, str):
            continue
        if field == "base_url":
            out[field] = (
                value.replace("http://nvidia-llm:8000/v1", "http://localhost:18000/v1")
                .replace("http://nvidia-llm-vllm:8000/v1", "http://localhost:18000/v1")
                .replace("host.docker.internal", "localhost")
            )
        else:
            out[field] = (
                value.replace("chatterbox-tts-service:50051", "localhost:50151")
                .replace("tts-service:50051", "localhost:50151")
                .replace("nemotron-asr-streaming-english:50052", "localhost:50152")
                .replace("nemotron-asr-streaming-multilingual:50052", "localhost:50152")
                .replace("parakeet-ctc-asr:50052", "localhost:50152")
                .replace("parakeet-rnnt-asr:50052", "localhost:50152")
                .replace("nemotron-speech:50051", "localhost:50051")
                .replace("nemotron-speech-tts:50051", "localhost:50051")
                .replace("booking-server:8001", "localhost:8001")
                .replace("host.docker.internal", "localhost")
            )
    return out


def _rewrite_catalog_for_host_runtime(catalog: dict[str, dict]) -> dict[str, dict]:
    """Rewrite every local service endpoint for host-native metadata responses."""
    if os.getenv("APP_RUNTIME", "").strip().lower() == "container":
        return catalog
    return {
        category: {
            key: _rewrite_entry_for_host_runtime(entry) if isinstance(entry, dict) else entry
            for key, entry in section.items()
        }
        for category, section in catalog.items()
    }


def _first_reachable_variant(variants: list[tuple[str, dict]]) -> tuple[str, dict] | None:
    for platform_name, entry in variants:
        if is_endpoint_reachable(_entry_endpoint(_rewrite_entry_for_host_runtime(entry))):
            return platform_name, entry
    return None


def _load_local_service_catalog(example_dir: Path) -> dict[str, dict]:
    """Load platform-scoped local service entries for one example."""
    data = _load_yaml_mapping(example_dir / "services.local.yaml")
    platform_data = select_runtime_platform_catalog(data)
    if platform_data is not None:
        return _rewrite_catalog_for_host_runtime(_normalize_service_catalog(platform_data))

    variants: dict[str, dict[str, list[tuple[str, dict]]]] = {}
    for platform_name, platform_data in data.items():
        if not isinstance(platform_data, dict):
            continue
        for category, section in platform_data.items():
            if not isinstance(section, dict):
                continue
            category_variants = variants.setdefault(str(category), {})
            for key, entry in section.items():
                if not isinstance(entry, dict):
                    continue
                category_variants.setdefault(str(key), []).append((str(platform_name), dict(entry)))

    merged: dict[str, dict] = {}
    for category, section in variants.items():
        target = merged.setdefault(category, {})
        for service_key, entries in section.items():
            first_entry = entries[0][1]
            if all(entry == first_entry for _, entry in entries):
                target[service_key] = first_entry
                continue
            active = _first_reachable_variant(entries)
            if active is not None:
                active_platform, active_entry = active
                target[service_key] = active_entry
            else:
                active_platform = ""
            for platform_name, entry in entries:
                if platform_name != active_platform:
                    target[f"{service_key}-{platform_name}"] = entry
    return _rewrite_catalog_for_host_runtime(merged)


def _load_service_catalogs(example_dir: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load cloud and local service catalogs for one example directory."""
    base = Path(example_dir)
    cloud = _normalize_service_catalog(_load_yaml_mapping(base / "services.cloud.yaml"))
    local = _load_local_service_catalog(base)
    return cloud, local


def _example_dir(example: EnrichedExample) -> Path:
    """Return the package directory for an example's bot module."""
    return example_module_file(example).resolve().parent


def _service_entry_payload(source: str, key: str, entry: dict) -> ServiceDefault:
    """Match service API entry shape while preserving every catalog param."""
    return {
        "id": f"{source}:{key}",
        "key": key,
        "name": str(entry.get("name") or key),
        "builtIn": True,
        "source": source,
        **{k: v for k, v in entry.items() if k != "name"},
    }


def _first_reachable_service_entry(section: dict) -> tuple[str, dict] | None:
    """Return the first reachable entry in a normalized service section."""
    for key, entry in section.items():
        if isinstance(entry, dict) and is_endpoint_reachable(_entry_endpoint(entry)):
            return str(key), entry
    return None


def _resolve_service_default(example: EnrichedExample, category: str, service_id: str) -> ServiceDefault:
    """Resolve one default service id to its full service-catalog payload.

    Prefers the self-hosted variant when it exists and is reachable, matching
    the runtime ``/api/services`` precedence where reachable local entries are
    used for on-prem recipes. Falls back to cloud when no local endpoint is
    available, which keeps cloud-only recipe defaults usable.
    """
    cloud, local = _load_service_catalogs(str(_example_dir(example)))
    local_section = local.get(category, {})
    local_entry = local_section.get(service_id) if isinstance(local_section, dict) else None
    if isinstance(local_entry, dict) and is_endpoint_reachable(_entry_endpoint(local_entry)):
        return _service_entry_payload("self-hosted", service_id, local_entry)
    if isinstance(local_entry, dict) and isinstance(local_section, dict):
        reachable_local = _first_reachable_service_entry(local_section)
        if reachable_local is not None:
            local_key, entry = reachable_local
            return _service_entry_payload("self-hosted", local_key, entry)

    cloud_section = cloud.get(category, {})
    if isinstance(cloud_section, dict):
        cloud_entry = cloud_section.get(service_id)
        if isinstance(cloud_entry, dict):
            return _service_entry_payload("cloud-nim", service_id, cloud_entry)
        if isinstance(local_entry, dict):
            for fallback_key, fallback_entry in cloud_section.items():
                if isinstance(fallback_entry, dict):
                    return _service_entry_payload("cloud-nim", fallback_key, fallback_entry)

    if isinstance(local_entry, dict):
        return _service_entry_payload("self-hosted", service_id, local_entry)

    raise RuntimeError(
        f"Default service {service_id!r} for {example['key']} / {category!r} "
        "was not found in services.cloud.yaml or services.local.yaml"
    )


def _resolve_service_defaults(example: EnrichedExample) -> dict[str, list[ServiceDefault]]:
    """Hydrate example default service ids from the example's service catalog."""
    return {
        category: [_resolve_service_default(example, category, service_id) for service_id in service_ids]
        for category, service_ids in example["defaults"].items()
        if category not in ("prompt", "default_session_language") and isinstance(service_ids, list)
    }


def _resolve_prompt_default(example: EnrichedExample, prompt_key: str) -> PromptDefault:
    """Resolve one default prompt key to its prompt-catalog payload."""
    catalog = _load_yaml_mapping(_example_dir(example) / "prompts.yaml")
    entry = catalog.get(prompt_key)
    if not isinstance(entry, dict) or "content" not in entry:
        raise RuntimeError(f"Default prompt {prompt_key!r} for {example['key']} was not found in prompts.yaml")
    return {
        "key": prompt_key,
        "description": str(entry.get("description", "")),
        "content": str(entry.get("content", "")),
        "default": True,
        "builtIn": True,
        "tools": [tool for tool in (entry.get("tools_available") or []) if isinstance(tool, str)],
    }


def _resolve_prompt_defaults(example: EnrichedExample) -> list[PromptDefault]:
    """Hydrate default prompt ids from the example's prompt catalog."""
    return [_resolve_prompt_default(example, prompt_key) for prompt_key in example["defaults"].get("prompt", [])]


def prompt_default_key(example_key: str = "", *, ignore_lock: bool = False) -> str | None:
    """Return the configured default prompt key for an example, if any."""
    example = find(example_key, ignore_lock=ignore_lock)
    prompt_keys = example["defaults"].get("prompt", [])
    return prompt_keys[0] if prompt_keys else None


def welcome_message_enabled(example_key: str = "") -> bool:
    """Return whether an example greets the user at session start.

    Resolution order: the ``ENABLE_WELCOME_MESSAGE`` environment variable wins
    when set (a global override used by the ``generic-assistant/workstation-perf``
    compose profile), otherwise the per-example ``welcome_message`` registry value
    applies (default ``True``).
    """
    override = os.getenv("ENABLE_WELCOME_MESSAGE", "").strip()
    if override:
        return override.lower() == "true"
    return bool(find(example_key).get("welcome_message", True))


def agent_prompt_keys(example_key: str = "") -> frozenset[str]:
    """Return prompt-catalog keys that are pipeline-only (hidden from the UI selector)."""
    return frozenset(find(example_key).get("agent_prompt_keys", []))


def _load_examples(data: dict) -> dict[str, ExampleEntry]:
    raw_examples = data.get("examples")
    if not isinstance(raw_examples, dict):
        raise RuntimeError("examples_registry.yaml requires an examples mapping")

    examples: dict[str, ExampleEntry] = {}
    for example_id, entry in raw_examples.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"Example {example_id!r} must be a mapping")
        label = str(entry.get("label") or "").strip()
        bot_spec = str(entry.get("bot") or "").strip()
        slots = entry.get("slots", [])
        capabilities = entry.get("capabilities", [])
        agent_prompt_keys = entry.get("agent_prompt_keys", [])
        defaults = entry.get("defaults", {})
        if not label or not bot_spec:
            raise RuntimeError(f"Example {example_id!r} requires label and bot")
        if not isinstance(slots, list) or not all(isinstance(slot, str) for slot in slots):
            raise RuntimeError(f"Example {example_id!r} slots must be a list of strings")
        if not isinstance(capabilities, list) or not all(isinstance(capability, str) for capability in capabilities):
            raise RuntimeError(f"Example {example_id!r} capabilities must be a list of strings")
        if not isinstance(agent_prompt_keys, list) or not all(isinstance(key, str) for key in agent_prompt_keys):
            raise RuntimeError(f"Example {example_id!r} agent_prompt_keys must be a list of strings")
        welcome_message = entry.get("welcome_message", True)
        if not isinstance(welcome_message, bool):
            raise RuntimeError(f"Example {example_id!r} welcome_message must be a boolean")
        if not isinstance(defaults, dict):
            raise RuntimeError(f"Example {example_id!r} defaults must be a mapping")
        normalized_defaults: dict[str, list[str] | str] = {}
        for slot, service_ids in defaults.items():
            if slot == "default_session_language":
                if not isinstance(service_ids, str):
                    raise RuntimeError(f"Example {example_id!r} defaults[{slot!r}] must be a string")
                normalized_defaults[str(slot)] = service_ids.strip()
                continue
            if not isinstance(service_ids, list) or not all(isinstance(service_id, str) for service_id in service_ids):
                raise RuntimeError(f"Example {example_id!r} defaults[{slot!r}] must be a list of strings")
            normalized_defaults[str(slot)] = list(service_ids)
        examples[str(example_id)] = {
            "label": label,
            "slots": list(slots),
            "capabilities": list(capabilities),
            "agent_prompt_keys": list(agent_prompt_keys),
            "defaults": normalized_defaults,
            "welcome_message": welcome_message,
            "bot": bot_spec,
        }
    return examples


class Selection(NamedTuple):
    """Resolved ``selection`` field describing what the UI exposes."""

    raw: str
    locked: bool
    example_keys: tuple[str, ...]
    default_key: str


def _parse_selection(
    raw: str,
    examples: dict[str, ExampleEntry],
) -> Selection:
    """Parse the ``selection`` value into a :class:`Selection`.

    Accepted values:
      * ``all`` — every registered example (selectable).
      * ``<example>`` — lock to one example, no switching.
    """
    cleaned = (raw or "").strip()
    if not cleaned:
        raise RuntimeError("examples_registry.yaml requires a 'selection' value")

    if cleaned == "all":
        example_keys = tuple(examples.keys())
        if not example_keys:
            raise RuntimeError("selection 'all' requires at least one example")
        return Selection(cleaned, False, example_keys, example_keys[0])

    if cleaned not in examples:
        raise RuntimeError(f"selection {cleaned!r} must be 'all' or a known example id")
    return Selection(cleaned, True, (cleaned,), cleaned)


_SUPPORTED_TRANSPORTS: tuple[str, ...] = ("webrtc", "websocket")


def _parse_transports(raw: str) -> tuple[str, ...]:
    """Parse the ``transports`` value into an ordered tuple of transport ids.

    Accepted values:
      * ``all`` — every supported transport.
      * a single transport id (e.g., ``webrtc`` or ``websocket``).
    """
    cleaned = (raw or "all").strip().lower()
    if cleaned == "all":
        return _SUPPORTED_TRANSPORTS
    if cleaned in _SUPPORTED_TRANSPORTS:
        return (cleaned,)
    raise RuntimeError(f"transports {cleaned!r} must be 'all' or one of {_SUPPORTED_TRANSPORTS}")


_REGISTRY_DATA = _load_yaml_registry()
EXAMPLES = _load_examples(_REGISTRY_DATA)
_SELECTION = _parse_selection(
    os.getenv("EXAMPLE_SELECTION", "").strip() or str(_REGISTRY_DATA.get("selection") or ""),
    EXAMPLES,
)
_TRANSPORTS = _parse_transports(
    os.getenv("TRANSPORT_SELECTION", "").strip() or str(_REGISTRY_DATA.get("transports") or "all"),
)


def is_locked() -> bool:
    """Return whether the selection pins the session to a single example."""
    return _SELECTION.locked


def visible_example_keys() -> tuple[str, ...]:
    """Return the example keys exposed by the current selection."""
    return _SELECTION.example_keys


def visible_transports() -> tuple[str, ...]:
    """Return the transports exposed by the current selection."""
    return _TRANSPORTS


def _enrich(example_id: str, entry: ExampleEntry) -> EnrichedExample:
    """Project a registry entry into a flat dict with an ``id`` and wire ``key`` (both the example id)."""
    return {**entry, "id": example_id, "key": example_id}


def _lookup_by_key(key: str) -> EnrichedExample:
    """Return the :class:`EnrichedExample` for the example id ``key``; raises on miss."""
    return _enrich(key, EXAMPLES[key])


def find(value: str = "", *, ignore_lock: bool = False) -> EnrichedExample:
    """Resolve an example.

    Routing rules:
      * Locked selection wins unless ``ignore_lock`` is set.
      * When ``ignore_lock`` is set, explicit example-id matches are resolved
        against every registered example.
      * Otherwise prefer an explicit example-id match within the visible set.
      * Fall back to the default example when no explicit match is found.
    """
    if _SELECTION.locked and not ignore_lock:
        return _lookup_by_key(_SELECTION.default_key)

    cleaned = (value or "").strip().lower()
    allowed_keys = tuple(EXAMPLES) if ignore_lock else _SELECTION.example_keys
    if cleaned and cleaned in allowed_keys:
        return _lookup_by_key(cleaned)
    return _lookup_by_key(_SELECTION.default_key)


def metadata(example: EnrichedExample) -> dict:
    """Strip internal fields (``bot`` spec) for client payloads."""
    defaults = _resolve_service_defaults(example)
    prompt_defaults = _resolve_prompt_defaults(example)
    if prompt_defaults:
        defaults["prompt"] = prompt_defaults
    return {
        "id": example["id"],
        "key": example["key"],
        "label": example["label"],
        "slots": example["slots"],
        "capabilities": example["capabilities"],
        "default_session_language": str(example["defaults"].get("default_session_language") or ""),
        "defaults": defaults,
    }


def visible_options() -> list[dict]:
    """Return metadata for every example exposed by the current selection."""
    return [metadata(_lookup_by_key(key)) for key in _SELECTION.example_keys]
