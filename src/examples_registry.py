# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in voice-agent examples loaded from the root ``examples_registry.yaml``."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

from utils import (
    SERVICE_CATEGORIES,
    load_prompt_catalog,
    load_yaml_file,
    nvidia_cloud_available,
    parse_env_bool,
    service_api_entry,
    service_catalog_sources,
    service_recipe,
)


class ActivityCheckConfig(TypedDict, total=False):
    """Per-example settings for proactive inactivity checks."""

    first_warning_s: float
    second_warning_s: float
    warning_completion_timeout_s: float


class ExampleEntry(TypedDict):
    """Raw registry entry for one example."""

    label: str
    services: dict[str, list[str]]
    settings: dict[str, dict[str, dict[str, Any]]]
    categories: dict[str, str]
    capabilities: list[str]
    agent_prompt_keys: list[str]
    activity_check: ActivityCheckConfig | None
    defaults: dict[str, list[str] | str]
    welcome_message: bool
    bot: str


class EnrichedExample(ExampleEntry):
    """Registry entry plus derived id/key fields (``key == id``)."""

    id: str
    key: str


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


def _split_bot_spec(spec: str) -> tuple[str, str]:
    module_path, separator, attr = spec.partition(":")
    if not (separator and module_path and attr):
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


def _resolve_service_defaults(example: EnrichedExample) -> dict[str, list[dict]]:
    """Return the first available catalog entry per slot, self-hosted before NVIDIA Cloud."""
    sources = service_catalog_sources(example)
    defaults: dict[str, list[dict]] = {}
    for slot in example["services"]:
        found = next(
            ((source, key, entry) for source, catalog in sources for key, entry in catalog.get(slot, {}).items()),
            None,
        )
        if found is None:
            recipe = service_recipe()
            hint = (
                " Start the self-hosted sidecars, set SERVICE_RECIPE, or set a real NVIDIA_API_KEY for NVIDIA Cloud."
                if recipe in ("auto", "cloud") and not nvidia_cloud_available()
                else ""
            )
            raise RuntimeError(
                f"No {slot!r} service for {example['key']} in services.yaml for SERVICE_RECIPE={recipe!r}; "
                f"checked {example['services'][slot]}.{hint}"
            )
        defaults[slot] = [service_api_entry(*found)]
    return defaults


def _resolve_prompt_default(example: EnrichedExample, catalog: dict, prompt_key: str) -> PromptDefault:
    """Resolve one default prompt key to its prompt-catalog payload."""
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
    """Hydrate default prompt ids from the ``prompts.yaml`` beside the example's bot module."""
    prompt_keys = example["defaults"].get("prompt", [])
    if not prompt_keys:
        return []
    catalog = load_prompt_catalog(example_module_file(example))
    return [_resolve_prompt_default(example, catalog, prompt_key) for prompt_key in prompt_keys]


def prompt_default_key(example_key: str = "", *, ignore_lock: bool = False) -> str | None:
    """Return the configured default prompt key for an example, if any."""
    example = find(example_key, ignore_lock=ignore_lock)
    prompt_keys = example["defaults"].get("prompt", [])
    return prompt_keys[0] if prompt_keys else None


def _session_language(example: EnrichedExample) -> str:
    return str(example["defaults"].get("default_session_language") or "")


def default_session_language(example_key: str = "") -> str:
    """Return the registry-declared fixed session language for an example."""
    return _session_language(find(example_key))


def welcome_message_enabled(example_key: str = "") -> bool:
    """Return whether an example greets the user at session start.

    Resolution order: the ``ENABLE_WELCOME_MESSAGE`` environment variable wins
    when set (a global override used by the ``generic-assistant/server-perf``
    compose profile), otherwise the per-example ``welcome_message`` registry value
    applies (default ``True``).
    """
    return parse_env_bool("ENABLE_WELCOME_MESSAGE", default=find(example_key)["welcome_message"])


def agent_prompt_keys(example_key: str = "") -> frozenset[str]:
    """Return prompt-catalog keys that are pipeline-only (hidden from the UI selector)."""
    return frozenset(find(example_key).get("agent_prompt_keys", []))


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _parse_str_list(example_id: str, field: str, value: Any) -> list[str]:
    if not _is_str_list(value):
        raise RuntimeError(f"Example {example_id!r} {field} must be a list of strings")
    return list(value)


def _parse_services(example_id: str, entry: dict) -> dict[str, list[str]]:
    services = entry.get("services", {})
    if not isinstance(services, dict) or not all(_is_str_list(keys) and keys for keys in services.values()):
        raise RuntimeError(f"Example {example_id!r} services must map each slot to a non-empty list of keys")
    return {str(slot): list(keys) for slot, keys in services.items()}


def _parse_settings(example_id: str, entry: dict, services: dict[str, list[str]]) -> dict:
    settings = entry.get("settings", {})
    if not isinstance(settings, dict) or not all(
        slot in services
        and isinstance(per_service, dict)
        and all(key in services[slot] and isinstance(values, dict) for key, values in per_service.items())
        for slot, per_service in settings.items()
    ):
        raise RuntimeError(f"Example {example_id!r} settings must map a slot to its service keys to setting defaults")
    return {
        str(slot): {str(key): dict(values) for key, values in per_service.items()}
        for slot, per_service in settings.items()
    }


def _parse_categories(example_id: str, entry: dict, services: dict[str, list[str]]) -> dict[str, str]:
    categories = entry.get("categories", {})
    if not isinstance(categories, dict) or not all(
        slot in services and category in SERVICE_CATEGORIES for slot, category in categories.items()
    ):
        raise RuntimeError(
            f"Example {example_id!r} categories must map a slot listed under services to one of "
            f"{', '.join(SERVICE_CATEGORIES)}"
        )
    return {str(slot): str(category) for slot, category in categories.items()}


def _parse_activity_check(example_id: str, entry: dict) -> ActivityCheckConfig | None:
    activity_check = entry.get("activity_check")
    if activity_check is None:
        return None
    if not isinstance(activity_check, dict):
        raise RuntimeError(f"Example {example_id!r} activity_check must be a mapping")
    parsed: ActivityCheckConfig = {}
    for key in ("first_warning_s", "second_warning_s", "warning_completion_timeout_s"):
        value = activity_check.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise RuntimeError(f"Example {example_id!r} activity_check.{key} must be a positive number")
        parsed[key] = float(value)
    return parsed


def _parse_defaults(example_id: str, entry: dict) -> dict[str, list[str] | str]:
    defaults = entry.get("defaults", {})
    if not isinstance(defaults, dict):
        raise RuntimeError(f"Example {example_id!r} defaults must be a mapping")
    parsed: dict[str, list[str] | str] = {}
    for field, value in defaults.items():
        if field == "default_session_language":
            if not isinstance(value, str):
                raise RuntimeError(f"Example {example_id!r} defaults[{field!r}] must be a string")
            parsed[field] = value.strip()
        elif field == "prompt":
            parsed[field] = _parse_str_list(example_id, "defaults['prompt']", value)
        else:
            raise RuntimeError(
                f"Example {example_id!r} defaults[{field!r}] is not supported; list services under 'services'"
            )
    return parsed


def _parse_example(example_id: str, entry: Any) -> ExampleEntry:
    if not isinstance(entry, dict):
        raise RuntimeError(f"Example {example_id!r} must be a mapping")
    label = str(entry.get("label") or "").strip()
    bot_spec = str(entry.get("bot") or "").strip()
    if not label or not bot_spec:
        raise RuntimeError(f"Example {example_id!r} requires label and bot")
    welcome_message = entry.get("welcome_message", True)
    if not isinstance(welcome_message, bool):
        raise RuntimeError(f"Example {example_id!r} welcome_message must be a boolean")
    services = _parse_services(example_id, entry)
    return {
        "label": label,
        "services": services,
        "settings": _parse_settings(example_id, entry, services),
        "categories": _parse_categories(example_id, entry, services),
        "capabilities": _parse_str_list(example_id, "capabilities", entry.get("capabilities", [])),
        "agent_prompt_keys": _parse_str_list(example_id, "agent_prompt_keys", entry.get("agent_prompt_keys", [])),
        "activity_check": _parse_activity_check(example_id, entry),
        "defaults": _parse_defaults(example_id, entry),
        "welcome_message": welcome_message,
        "bot": bot_spec,
    }


def _load_examples(data: dict) -> dict[str, ExampleEntry]:
    raw_examples = data.get("examples")
    if not isinstance(raw_examples, dict):
        raise RuntimeError("examples_registry.yaml requires an examples mapping")
    return {str(example_id): _parse_example(str(example_id), entry) for example_id, entry in raw_examples.items()}


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


_REGISTRY_DATA = load_yaml_file(_REGISTRY_PATH, required=True)
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


def _lookup_by_key(key: str) -> EnrichedExample:
    """Return the registry entry for example id ``key`` with ``id`` and wire ``key`` set; raises on miss."""
    return {**EXAMPLES[key], "id": key, "key": key}


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
    """Return the client payload for an example, with its service and prompt defaults resolved."""
    defaults = _resolve_service_defaults(example)
    prompt_defaults = _resolve_prompt_defaults(example)
    if prompt_defaults:
        defaults["prompt"] = prompt_defaults
    return {
        "id": example["id"],
        "key": example["key"],
        "label": example["label"],
        "slots": list(example["services"]),
        "capabilities": example["capabilities"],
        "default_session_language": _session_language(example),
        "defaults": defaults,
    }


def activity_check_config(example_key: str = "") -> ActivityCheckConfig | None:
    """Return a copy of the selected example's activity-check configuration."""
    config = find(example_key)["activity_check"]
    return dict(config) if config else None


def visible_options() -> list[dict]:
    """Return metadata for every example exposed by the current selection."""
    return [metadata(_lookup_by_key(key)) for key in _SELECTION.example_keys]
