# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Built-in voice-agent examples loaded from the root ``examples_registry.yaml``."""

from __future__ import annotations

import copy
import importlib
import os
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple, TypedDict

import yaml

from utils import LOCAL_SERVICE_CATALOG_PLATFORMS, is_endpoint_reachable, public_service_entry_fields


class ExampleEntry(TypedDict):
    """Raw registry entry for one example."""

    label: str
    slots: list[str]
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


class ActivityCheckConfig(TypedDict, total=False):
    """Per-example settings for proactive inactivity checks."""

    first_warning_s: float
    second_warning_s: float
    warning_completion_timeout_s: float


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


class RealtimeModelSelectors(TypedDict, total=False):
    """Trusted catalog selectors bound by one public Realtime model id."""

    prompt_key: str
    llm_id: str
    thinker_llm_id: str
    asr_id: str
    tts_id: str


class RealtimeModelProfile(TypedDict):
    """One validated, server-owned OpenAI Realtime deployment profile."""

    id: str
    model: str
    label: str
    pipeline_mode: str
    default: bool
    selectors: RealtimeModelSelectors
    platform_overrides: dict[str, RealtimeModelSelectors]


class RealtimeModelProfileNotAvailable(ValueError):
    """An exact public Realtime model/profile lookup could not be satisfied."""

    code = "model_not_available"
    param = "model"

    def __init__(self, model: str | None = None) -> None:
        """Describe an unavailable exact public model without leaking routes."""
        self.model = model
        if isinstance(model, str) and model:
            message = f"Model {model!r} is not available on this endpoint"
        else:
            message = "No Realtime model is available on this endpoint"
        super().__init__(message)


REALTIME_REGISTRY_DEFAULT_SELECTOR = "registry-default"
REALTIME_SERVICE_PLATFORMS: tuple[str, ...] = ("cloud", *LOCAL_SERVICE_CATALOG_PLATFORMS)

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
                .replace("http://nvidia-llm-omni:8000/v1", "http://localhost:18002/v1")
                .replace("http://nvidia-llm-vllm:8000/v1", "http://localhost:18000/v1")
                .replace("http://nvidia-llm-vllm-omni:8002/v1", "http://localhost:8002/v1")
                .replace("host.docker.internal", "localhost")
            )
        else:
            out[field] = (
                value.replace("magpie-zeroshot-tts-service:50051", "localhost:50151")
                .replace("chatterbox-tts-service:50051", "localhost:50151")
                .replace("magpie-multilingual-tts-service:50051", "localhost:50151")
                .replace("nemotron-asr-streaming-english:50052", "localhost:50152")
                .replace("nemotron-asr-streaming-multilingual:50052", "localhost:50152")
                .replace("parakeet-ctc-asr:50052", "localhost:50152")
                .replace("parakeet-rnnt-asr:50052", "localhost:50152")
                .replace("nemo-speech:50051", "localhost:50051")
                .replace("nemo-speech-multilingual:50051", "localhost:50051")
                .replace("nemo-speech-tts:50051", "localhost:50051")
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
    """Load local service entries, merging recipe sections by reachability."""
    data = _load_yaml_mapping(example_dir / "services.local.yaml")
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
            # Keep the plain key resolvable even when nothing is reachable, so
            # defaults can still fall back to cloud instead of failing lookup.
            active_platform, active_entry = _first_reachable_variant(entries) or entries[0]
            target[service_key] = active_entry
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


_REALTIME_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_REALTIME_PROFILE_FIELDS = frozenset({"label", "pipeline_mode", "default", "selectors", "platform_overrides"})
_REALTIME_SELECTOR_SLOTS: dict[str, tuple[str, str | None]] = {
    "prompt_key": ("prompt", None),
    "llm_id": ("llm", "llm"),
    "thinker_llm_id": ("thinker-llm", "thinker-llm"),
    "asr_id": ("asr", "asr"),
    "tts_id": ("tts", "tts"),
}
_REALTIME_REQUIRED_SLOT_SELECTORS = {
    "llm": "llm_id",
    "thinker-llm": "thinker_llm_id",
    "asr": "asr_id",
    "tts": "tts_id",
}
_REALTIME_SERVICE_SELECTOR_NAMES = frozenset(_REALTIME_SELECTOR_SLOTS) - {"prompt_key"}


def realtime_service_platform(value: str | None = None) -> str:
    """Return the explicit Realtime deployment catalog platform.

    Compose pins this setting for every recipe. A host-native process defaults
    deterministically to the cloud catalog; endpoint reachability never affects
    this selection.
    """
    raw = os.getenv("REALTIME_SERVICE_PLATFORM", "") if value is None else value
    if not isinstance(raw, str):
        raise RuntimeError("REALTIME_SERVICE_PLATFORM must be a string")
    platform = raw.strip().lower() or "cloud"
    if platform not in REALTIME_SERVICE_PLATFORMS:
        allowed = ", ".join(REALTIME_SERVICE_PLATFORMS)
        raise RuntimeError(f"REALTIME_SERVICE_PLATFORM must be one of: {allowed}")
    return platform


def _configured_default_service_key(
    example: EnrichedExample,
    slot: str,
    *,
    model: str,
    selector_name: str,
) -> str:
    configured_defaults = example.get("defaults", {}).get(slot)
    if (
        not isinstance(configured_defaults, list)
        or not configured_defaults
        or not isinstance(configured_defaults[0], str)
        or not configured_defaults[0]
    ):
        raise RuntimeError(
            f"Realtime model {model!r} selector {selector_name!r} requires "
            f"a configured default for pipeline slot {slot!r}"
        )
    return configured_defaults[0]


def _service_selector_key_for_platform(
    selector: str,
    platform: str,
    *,
    model: str,
    selector_name: str,
) -> str:
    """Validate selector qualification and return its raw catalog key."""
    if selector.startswith("cloud-nim:"):
        key = selector.removeprefix("cloud-nim:")
        if platform != "cloud" or not key or ":" in key:
            raise RuntimeError(
                f"Realtime model {model!r} selector {selector_name!r} is not valid for platform {platform!r}"
            )
        return key

    if selector.startswith("self-hosted:"):
        remainder = selector.removeprefix("self-hosted:")
        if platform == "cloud" or not remainder:
            raise RuntimeError(
                f"Realtime model {model!r} selector {selector_name!r} is not valid for platform {platform!r}"
            )
        if ":" not in remainder:
            return remainder
        selected_platform, key = remainder.split(":", 1)
        if (
            selected_platform not in LOCAL_SERVICE_CATALOG_PLATFORMS
            or selected_platform != platform
            or not key
            or ":" in key
        ):
            raise RuntimeError(
                f"Realtime model {model!r} selector {selector_name!r} is not valid for platform {platform!r}"
            )
        return key

    if ":" in selector:
        raise RuntimeError(
            f"Realtime model {model!r} selector {selector_name!r} has an invalid catalog id {selector!r}"
        )
    return selector


def _static_service_entry_exists(example_dir: Path, category: str, platform: str, key: str) -> bool:
    """Check one exact raw catalog section without endpoint probes."""
    if platform == "cloud":
        catalog = _normalize_service_catalog(_load_yaml_mapping(example_dir / "services.cloud.yaml"))
    else:
        local = _load_yaml_mapping(example_dir / "services.local.yaml")
        platform_data = local.get(platform)
        catalog = _normalize_service_catalog(platform_data) if isinstance(platform_data, dict) else {}
    section = catalog.get(category, {})
    return isinstance(section, dict) and isinstance(section.get(key), dict)


def _materialize_service_selector(
    example: EnrichedExample,
    category: str,
    slot: str,
    selector: str,
    platform: str,
    *,
    model: str,
    selector_name: str,
) -> str:
    """Resolve one profile selector to a canonical, platform-qualified id."""
    if selector == REALTIME_REGISTRY_DEFAULT_SELECTOR:
        selector = _configured_default_service_key(
            example,
            slot,
            model=model,
            selector_name=selector_name,
        )
    key = _service_selector_key_for_platform(
        selector,
        platform,
        model=model,
        selector_name=selector_name,
    )
    if not _static_service_entry_exists(_example_dir(example), category, platform, key):
        raise RuntimeError(
            f"Realtime model {model!r} selector {selector_name!r} does not exist "
            f"in the {example['key']!r} {platform!r} service catalog"
        )
    if platform == "cloud":
        return f"cloud-nim:{key}"
    return f"self-hosted:{platform}:{key}"


def _validate_realtime_prompt_selector(example: EnrichedExample, prompt_key: str) -> None:
    """Validate a public prompt selector from the static example catalog."""
    prompt = _load_yaml_mapping(_example_dir(example) / "prompts.yaml").get(prompt_key)
    if (
        not isinstance(prompt, dict)
        or not isinstance(prompt.get("content"), str)
        or prompt.get("internal") is True
        or prompt_key in set(example.get("agent_prompt_keys", []))
    ):
        raise RuntimeError(f"Realtime model prompt selector {prompt_key!r} is not public for {example['key']!r}")


def _load_realtime_model_profiles(
    data: dict,
    examples: dict[str, ExampleEntry],
) -> dict[str, RealtimeModelProfile]:
    """Validate the static public Realtime model registry.

    Validation deliberately reads only registry, prompt, and exact platform
    service-catalog sections. Runtime endpoint reachability cannot affect
    whether a public model id exists or which service it selects.
    """
    raw_profiles = data.get("realtime_models")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise RuntimeError("examples_registry.yaml requires a non-empty realtime_models mapping")

    profiles: dict[str, RealtimeModelProfile] = {}
    default_models: dict[str, list[str]] = {}
    for raw_model, raw_profile in raw_profiles.items():
        if not isinstance(raw_model, str) or not _REALTIME_MODEL_ID.fullmatch(raw_model):
            raise RuntimeError(f"Realtime model id {raw_model!r} is invalid")
        if not isinstance(raw_profile, dict):
            raise RuntimeError(f"Realtime model {raw_model!r} must be a mapping")
        unknown_fields = sorted(set(raw_profile) - _REALTIME_PROFILE_FIELDS)
        if unknown_fields:
            raise RuntimeError(f"Realtime model {raw_model!r} has unknown field {unknown_fields[0]!r}")

        label = raw_profile.get("label")
        pipeline_mode = raw_profile.get("pipeline_mode")
        is_default = raw_profile.get("default")
        raw_selectors = raw_profile.get("selectors")
        raw_platform_overrides = raw_profile.get("platform_overrides", {})
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(f"Realtime model {raw_model!r} requires a non-empty label")
        if not isinstance(pipeline_mode, str) or pipeline_mode not in examples:
            raise RuntimeError(f"Realtime model {raw_model!r} references unknown pipeline {pipeline_mode!r}")
        if not isinstance(is_default, bool):
            raise RuntimeError(f"Realtime model {raw_model!r} default must be a boolean")
        if not isinstance(raw_selectors, dict):
            raise RuntimeError(f"Realtime model {raw_model!r} selectors must be a mapping")
        if not isinstance(raw_platform_overrides, dict):
            raise RuntimeError(f"Realtime model {raw_model!r} platform_overrides must be a mapping")

        unknown_selectors = sorted(set(raw_selectors) - _REALTIME_SELECTOR_SLOTS.keys())
        if unknown_selectors:
            raise RuntimeError(f"Realtime model {raw_model!r} has unknown selector {unknown_selectors[0]!r}")
        selectors: RealtimeModelSelectors = {}
        for selector_name, raw_value in raw_selectors.items():
            if not isinstance(raw_value, str) or not raw_value.strip() or raw_value != raw_value.strip():
                raise RuntimeError(
                    f"Realtime model {raw_model!r} selector {selector_name!r} must be a non-empty trimmed string"
                )
            selectors[selector_name] = raw_value  # type: ignore[literal-required]

        platform_overrides: dict[str, RealtimeModelSelectors] = {}
        for raw_platform, raw_overrides in raw_platform_overrides.items():
            if not isinstance(raw_platform, str) or raw_platform not in REALTIME_SERVICE_PLATFORMS:
                raise RuntimeError(f"Realtime model {raw_model!r} has unknown platform override {raw_platform!r}")
            if not isinstance(raw_overrides, dict) or not raw_overrides:
                raise RuntimeError(
                    f"Realtime model {raw_model!r} platform override {raw_platform!r} must be a non-empty mapping"
                )
            unknown_overrides = sorted(set(raw_overrides) - _REALTIME_SERVICE_SELECTOR_NAMES)
            if unknown_overrides:
                raise RuntimeError(
                    f"Realtime model {raw_model!r} platform override {raw_platform!r} "
                    f"has unknown selector {unknown_overrides[0]!r}"
                )
            normalized_overrides: RealtimeModelSelectors = {}
            for selector_name, raw_value in raw_overrides.items():
                if not isinstance(raw_value, str) or not raw_value.strip() or raw_value != raw_value.strip():
                    raise RuntimeError(
                        f"Realtime model {raw_model!r} platform override {raw_platform!r} "
                        f"selector {selector_name!r} must be a non-empty trimmed string"
                    )
                normalized_overrides[selector_name] = raw_value  # type: ignore[literal-required]
            platform_overrides[raw_platform] = normalized_overrides

        example: EnrichedExample = {
            **examples[pipeline_mode],
            "id": pipeline_mode,
            "key": pipeline_mode,
        }
        slots = set(example.get("slots", []))
        for slot, selector_name in _REALTIME_REQUIRED_SLOT_SELECTORS.items():
            if slot in slots and selector_name not in selectors:
                raise RuntimeError(
                    f"Realtime model {raw_model!r} requires selector {selector_name!r} for slot {slot!r}"
                )
        if "prompt_key" not in selectors:
            raise RuntimeError(f"Realtime model {raw_model!r} requires selector 'prompt_key'")

        for selector_name, selector_value in selectors.items():
            slot, category = _REALTIME_SELECTOR_SLOTS[selector_name]
            if slot != "prompt" and slot not in slots:
                raise RuntimeError(
                    f"Realtime model {raw_model!r} selector {selector_name!r} is incompatible "
                    f"with pipeline {pipeline_mode!r}"
                )
            if selector_name == "prompt_key":
                if selector_value == REALTIME_REGISTRY_DEFAULT_SELECTOR:
                    raise RuntimeError(f"Realtime model {raw_model!r} prompt_key must name an explicit public prompt")
                _validate_realtime_prompt_selector(example, selector_value)
                continue
            if category is None:
                raise RuntimeError(f"Realtime model {raw_model!r} selector {selector_name!r} has no catalog")
        for platform, overrides in platform_overrides.items():
            for selector_name in overrides:
                slot, _ = _REALTIME_SELECTOR_SLOTS[selector_name]
                if slot not in slots:
                    raise RuntimeError(
                        f"Realtime model {raw_model!r} platform override {platform!r} "
                        f"selector {selector_name!r} is incompatible with pipeline {pipeline_mode!r}"
                    )

        for platform in REALTIME_SERVICE_PLATFORMS:
            overrides = platform_overrides.get(platform, {})
            for selector_name, base_selector in selectors.items():
                if selector_name == "prompt_key":
                    continue
                slot, category = _REALTIME_SELECTOR_SLOTS[selector_name]
                if category is None:
                    raise RuntimeError(f"Realtime model {raw_model!r} selector {selector_name!r} has no catalog")
                _materialize_service_selector(
                    example,
                    category,
                    slot,
                    overrides.get(selector_name, base_selector),
                    platform,
                    model=raw_model,
                    selector_name=selector_name,
                )

        profile: RealtimeModelProfile = {
            "id": raw_model,
            "model": raw_model,
            "label": label.strip(),
            "pipeline_mode": pipeline_mode,
            "default": is_default,
            "selectors": selectors,
            "platform_overrides": platform_overrides,
        }
        profiles[raw_model] = profile
        if is_default:
            default_models.setdefault(pipeline_mode, []).append(raw_model)

    realtime_pipelines = {profile["pipeline_mode"] for profile in profiles.values()}
    for pipeline_mode in realtime_pipelines:
        defaults = default_models.get(pipeline_mode, [])
        if len(defaults) != 1:
            raise RuntimeError(
                f"Realtime pipeline {pipeline_mode!r} must have exactly one default model; found {len(defaults)}"
            )
    return profiles


def _service_entry_payload(source: str, key: str, entry: dict) -> ServiceDefault:
    """Match the ordinary client metadata shape without Realtime internals."""
    return {
        "id": f"{source}:{key}",
        "key": key,
        "name": str(entry.get("name") or key),
        "builtIn": True,
        "source": source,
        **{k: v for k, v in public_service_entry_fields(entry).items() if k != "name"},
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


def default_session_language(example_key: str = "") -> str:
    """Return the registry-declared fixed session language for an example."""
    return str(find(example_key)["defaults"].get("default_session_language") or "")


def welcome_message_enabled(example_key: str = "") -> bool:
    """Return whether an example greets the user at session start.

    Resolution order: the ``ENABLE_WELCOME_MESSAGE`` environment variable wins
    when set (a global override used by the ``generic-assistant/server-perf``
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
        activity_check = entry.get("activity_check")
        defaults = entry.get("defaults", {})
        if not label or not bot_spec:
            raise RuntimeError(f"Example {example_id!r} requires label and bot")
        if not isinstance(slots, list) or not all(isinstance(slot, str) for slot in slots):
            raise RuntimeError(f"Example {example_id!r} slots must be a list of strings")
        if not isinstance(capabilities, list) or not all(isinstance(capability, str) for capability in capabilities):
            raise RuntimeError(f"Example {example_id!r} capabilities must be a list of strings")
        if not isinstance(agent_prompt_keys, list) or not all(isinstance(key, str) for key in agent_prompt_keys):
            raise RuntimeError(f"Example {example_id!r} agent_prompt_keys must be a list of strings")
        if activity_check is not None and not isinstance(activity_check, dict):
            raise RuntimeError(f"Example {example_id!r} activity_check must be a mapping")
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
        normalized_activity_check: ActivityCheckConfig | None = None
        if activity_check is not None:
            normalized_activity_check = {}
            for key in ("first_warning_s", "second_warning_s", "warning_completion_timeout_s"):
                value = activity_check.get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise RuntimeError(f"Example {example_id!r} activity_check.{key} must be a positive number")
                normalized_activity_check[key] = float(value)
        examples[str(example_id)] = {
            "label": label,
            "slots": list(slots),
            "capabilities": list(capabilities),
            "agent_prompt_keys": list(agent_prompt_keys),
            "activity_check": normalized_activity_check,
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


@cache
def realtime_model_profiles() -> dict[str, RealtimeModelProfile]:
    """Load and validate Realtime profiles only when that API is used."""
    return _load_realtime_model_profiles(_REGISTRY_DATA, EXAMPLES)


def visible_realtime_model_profiles() -> list[RealtimeModelProfile]:
    """Return detached public profiles whose pipelines are currently visible."""
    visible_pipelines = set(_SELECTION.example_keys)
    return [
        copy.deepcopy(profile)
        for profile in realtime_model_profiles().values()
        if profile["pipeline_mode"] in visible_pipelines
    ]


def resolve_realtime_model_profile(
    model: str | None = None,
    *,
    pipeline_mode: str | None = None,
) -> RealtimeModelProfile:
    """Resolve one exact visible public model or a pipeline's unique default.

    This lookup never falls back from an explicit model or pipeline value. The
    exception intentionally carries only the neutral public model contract so
    an API layer can project it to an OpenAI ``model_not_available`` error.
    """
    visible_pipelines = set(_SELECTION.example_keys)
    requested_pipeline = pipeline_mode if pipeline_mode not in (None, "") else _SELECTION.default_key
    if not isinstance(requested_pipeline, str) or requested_pipeline not in visible_pipelines:
        raise RealtimeModelProfileNotAvailable(model)

    if model not in (None, ""):
        if not isinstance(model, str):
            raise RealtimeModelProfileNotAvailable(None)
        profile = realtime_model_profiles().get(model)
        if (
            profile is None
            or profile["pipeline_mode"] not in visible_pipelines
            or (pipeline_mode not in (None, "") and profile["pipeline_mode"] != requested_pipeline)
        ):
            raise RealtimeModelProfileNotAvailable(model)
        return copy.deepcopy(profile)

    matches = [
        profile
        for profile in realtime_model_profiles().values()
        if profile["pipeline_mode"] == requested_pipeline and profile["default"]
    ]
    if len(matches) != 1:
        raise RealtimeModelProfileNotAvailable(None)
    return copy.deepcopy(matches[0])


def default_realtime_model_id(pipeline_mode: str | None = None) -> str:
    """Return the unique visible default public model id for a pipeline."""
    return resolve_realtime_model_profile(pipeline_mode=pipeline_mode)["model"]


def canonicalize_realtime_service_selector(
    pipeline_mode: str,
    selector_name: str,
    selector: str,
    platform: str | None = None,
) -> str:
    """Validate one trusted override and return its exact catalog ID.

    Raw catalog keys and ``self-hosted:<key>`` service API IDs are resolved only
    within the explicit deployment platform. Already-canonical IDs remain
    stable, while source/platform mismatches and unknown entries fail closed.
    """
    selected_platform = realtime_service_platform(platform)
    if not isinstance(pipeline_mode, str) or pipeline_mode not in EXAMPLES:
        raise RuntimeError(f"Unknown Realtime pipeline {pipeline_mode!r}")
    if selector_name not in _REALTIME_SERVICE_SELECTOR_NAMES:
        raise RuntimeError(f"Unknown Realtime service selector {selector_name!r}")
    if not isinstance(selector, str) or not selector.strip() or selector != selector.strip():
        raise RuntimeError(f"Realtime service selector {selector_name!r} must be a non-empty trimmed string")

    example = _lookup_by_key(pipeline_mode)
    slot, category = _REALTIME_SELECTOR_SLOTS[selector_name]
    if slot not in set(example.get("slots", [])) or category is None:
        raise RuntimeError(
            f"Realtime service selector {selector_name!r} is incompatible with pipeline {pipeline_mode!r}"
        )
    return _materialize_service_selector(
        example,
        category,
        slot,
        selector,
        selected_platform,
        model=pipeline_mode,
        selector_name=selector_name,
    )


def materialize_realtime_profile_selectors(
    profile: RealtimeModelProfile,
    platform: str | None = None,
) -> RealtimeModelSelectors:
    """Resolve a profile to deterministic, source-qualified catalog IDs.

    ``registry-default`` means exactly the first key declared for that slot in
    ``examples_registry.yaml``. It never means the first reachable or otherwise
    available entry. Local IDs also include the recipe section so every worker
    hydrates the same raw catalog entry.
    """
    selected_platform = realtime_service_platform(platform)
    model = profile.get("model")
    pipeline_mode = profile.get("pipeline_mode")
    selectors = profile.get("selectors")
    platform_overrides = profile.get("platform_overrides", {})
    if (
        not isinstance(model, str)
        or not isinstance(pipeline_mode, str)
        or pipeline_mode not in EXAMPLES
        or not isinstance(selectors, dict)
        or not isinstance(platform_overrides, dict)
    ):
        raise RuntimeError("Realtime model profile is not valid")

    raw_overrides = platform_overrides.get(selected_platform, {})
    if not isinstance(raw_overrides, dict) or any(name not in selectors for name in raw_overrides):
        raise RuntimeError(f"Realtime model {model!r} has invalid platform overrides")

    materialized: RealtimeModelSelectors = {}
    for selector_name, base_selector in selectors.items():
        if not isinstance(base_selector, str):
            raise RuntimeError(f"Realtime model {model!r} selector {selector_name!r} is not valid")
        if selector_name == "prompt_key":
            materialized[selector_name] = base_selector  # type: ignore[literal-required]
            continue
        selected = raw_overrides.get(selector_name, base_selector)
        if not isinstance(selected, str):
            raise RuntimeError(f"Realtime model {model!r} selector {selector_name!r} is not valid")
        materialized[selector_name] = canonicalize_realtime_service_selector(  # type: ignore[literal-required]
            pipeline_mode,
            selector_name,
            selected,
            selected_platform,
        )
    return materialized


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


def activity_check_config(example_key: str = "") -> ActivityCheckConfig | None:
    """Return a copy of the selected example's activity-check configuration."""
    config = find(example_key)["activity_check"]
    return dict(config) if config else None


def visible_options() -> list[dict]:
    """Return metadata for every example exposed by the current selection."""
    return [metadata(_lookup_by_key(key)) for key in _SELECTION.example_keys]
