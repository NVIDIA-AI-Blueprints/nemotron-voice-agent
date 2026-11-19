# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the agent's ASR, LLM, and TTS endpoints from the shared service catalog.

No LiveKit imports, so the UI server and the unit tests can use this module without the agent environment. The catalog
is the repository's `services.yaml`: the same entries and the same `SERVICE_RECIPE` values (`cloud`, `server`,
`single-gpu`) that the Pipecat server uses. A session can pick another offered service, or a custom endpoint, for each
slot, and change the editable settings that `settings.yaml` defines for the entry.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from nemotron_voice_agent.core.catalog import NVCF_GRPC_SERVER, NVCF_LLM_BASE_URL, RECIPES, load_catalog, nvcf_entry
from nemotron_voice_agent.core.registry import framework_section
from nemotron_voice_agent.core.service_status import parse_endpoint
from nemotron_voice_agent.core.settings import apply_service_settings, entry_settings_schema, load_settings_profiles
from nemotron_voice_agent.livekit.runtime.credentials import truthy

REGISTRY_EXAMPLE = "generic-assistant"
MODE_LABELS = {"cloud": "Cloud", "server": "Server (NIM)", "single-gpu": "Single GPU"}
# The runtime that serves each slot in each recipe, shown in the UI.
RUNTIMES = {
    "cloud": {"asr": "NVIDIA Cloud", "llm": "NVIDIA Cloud", "tts": "NVIDIA Cloud"},
    "server": {"asr": "NVIDIA NIM", "llm": "NVIDIA NIM", "tts": "NVIDIA NIM"},
    "single-gpu": {"asr": "NeMo-Speech.cpp", "llm": "vLLM", "tts": "NeMo-Speech.cpp"},
}
CUSTOM_RUNTIME = "Custom"
SOURCES = {"cloud": "cloud-nim", "server": "self-hosted", "single-gpu": "self-hosted"}
# Voice replies are short, so reasoning is off (the settings.yaml default for the Nemotron LLM entries).
LLM_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
# The LLM settings that `agent.build_llm` passes as client arguments. Every other LLM setting goes in the request body.
LLM_ARGUMENTS = ("temperature", "top_p", "max_tokens")
# The ASR and TTS settings that the NVIDIA plugin applies, by `settings.yaml` path. The other ones are left out.
SPEECH_SETTINGS = {"asr": frozenset({"automatic_punctuation"}), "tts": frozenset()}
SINGLE_GPU_TTS_SAMPLE_RATE = 22050  # NeMo-Speech.cpp synthesizes at 22.05 kHz
DEFAULT_TTS_SAMPLE_RATE = 16000
CUSTOM_FIELDS = {
    "llm": ("name", "base_url", "model_id", "extra_params"),
    "asr": ("name", "server", "model", "function_id"),
    "tts": ("name", "server", "voice_id", "function_id"),
}
CUSTOM_REQUIRED = {"llm": ("base_url", "model_id"), "asr": ("server",), "tts": ("server",)}
CUSTOM_ENDPOINTS_ENV = "LIVEKIT_UI_CUSTOM_ENDPOINTS"


def custom_endpoints_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether sessions may use custom endpoints. Off by default, because the agent connects to any address given."""
    return truthy((os.environ if env is None else env).get(CUSTOM_ENDPOINTS_ENV))


@dataclass(frozen=True)
class AgentConfig:
    """Endpoints and options for one session."""

    recipe: str
    llm_name: str
    llm_model: str
    llm_base_url: str
    llm_extra_body: dict[str, Any]
    asr_name: str
    asr_server: str
    asr_model: str
    asr_function_id: str
    asr_language: str
    tts_name: str
    tts_server: str
    tts_voice: str
    tts_function_id: str
    tts_language: str
    tts_sample_rate: int
    llm_options: dict[str, Any] = field(default_factory=dict)
    asr_punctuate: bool = True
    keys: dict[str, str] = field(default_factory=dict)
    hosted_slots: frozenset[str] = frozenset()

    @property
    def hosted(self) -> bool:
        """Whether the deployment recipe serves its services on NVIDIA Cloud instead of local sidecars."""
        return self.recipe == "cloud"

    def slot_hosted(self, slot: str) -> bool:
        """Whether the slot's endpoint is on NVIDIA Cloud, so it needs TLS and the API key."""
        return slot in self.hosted_slots

    def models(self) -> list[dict[str, str]]:
        """Return the models in use for the UI, in pipeline order, with the runtime that serves each one."""
        runtimes = RUNTIMES[self.recipe]

        def runs(slot: str) -> str:
            return CUSTOM_RUNTIME if self.keys.get(slot) == "" else runtimes[slot]

        return [
            {"kind": "ASR", "name": self.asr_name, "runs": runs("asr"), "id": self.asr_model},
            {"kind": "LLM", "name": self.llm_name, "runs": runs("llm"), "id": self.llm_model},
            {"kind": "TTS", "name": self.tts_name, "runs": runs("tts"), "id": self.tts_voice},
        ]

    def deployment(self) -> dict[str, object]:
        """Return the deployment mode and the models in use, for the UI."""
        return {"recipe": self.recipe, "label": MODE_LABELS[self.recipe], "models": self.models()}


def registry_services(path: Path | None = None) -> dict[str, list[str]]:
    """Return the services that the registry's `livekit` section offers for this example, per slot, in UI order."""
    services = framework_section("livekit", path)["examples"][REGISTRY_EXAMPLE]["services"]
    return {slot: list(keys) for slot, keys in services.items()}


REGISTRY_SERVICES = registry_services()
DEFAULT_SERVICES = {slot: keys[0] for slot, keys in REGISTRY_SERVICES.items()}


def _section(catalog: Mapping[str, Any], recipe: str, slot: str) -> Mapping[str, Any]:
    """Return the catalog entries of one slot that the recipe can serve."""
    if recipe not in RECIPES:
        raise ValueError(f"SERVICE_RECIPE must be one of {', '.join(RECIPES)}; got '{recipe}'.")
    entries = (catalog.get("single-gpu" if recipe == "single-gpu" else "server") or {}).get(slot, {})
    if recipe == "cloud":
        entries = {key: entry for key, entry in entries.items() if "nvcf" in entry}
    return entries


def _supported(slot: str, entry: Mapping[str, Any]) -> bool:
    """Whether this example can use the entry. The NVIDIA plugin speaks Riva gRPC, not the Transcribe HTTP API."""
    return not (slot == "asr" and entry.get("asr_type") == "transcribe")


def available_services(catalog: Mapping[str, Any], recipe: str) -> dict[str, list[str]]:
    """Return the catalog keys that `LLM_SERVICE`, `ASR_SERVICE`, and `TTS_SERVICE` accept for the recipe."""
    return {
        slot: [key for key, entry in _section(catalog, recipe, slot).items() if _supported(slot, entry)]
        for slot in DEFAULT_SERVICES
    }


def offered_services(
    catalog: Mapping[str, Any], recipe: str, env: Mapping[str, str] | None = None
) -> dict[str, list[str]]:
    """Return the registry's services that the recipe can serve, per slot. The configured default comes first."""
    env = os.environ if env is None else env
    available = available_services(catalog, recipe)
    offered = {}
    for slot, keys in REGISTRY_SERVICES.items():
        default = env.get(f"{slot.upper()}_SERVICE") or DEFAULT_SERVICES[slot]
        listed = [key for key in keys if key in available[slot]]
        offered[slot] = [default, *(key for key in listed if key != default)] if default in available[slot] else listed
    return offered


def resolve_entry(catalog: Mapping[str, Any], recipe: str, slot: str, key: str) -> dict[str, Any]:
    """Return a catalog entry for the recipe. The cloud recipe applies the entry's `nvcf` fields over it."""
    entries = _section(catalog, recipe, slot)
    entry = entries.get(key)
    if entry is None or not _supported(slot, entry):
        choices = ", ".join(available_services(catalog, recipe)[slot])
        hint = " Cloud serves only entries with an nvcf key." if recipe == "cloud" else ""
        raise ValueError(
            f"{slot} service '{key}' is not available in the {recipe} recipe.{hint} Choose one of: {choices}."
        )
    if recipe != "cloud":
        return dict(entry)
    return nvcf_entry(entry)


def _extra_params(entry: Mapping[str, Any]) -> dict[str, Any]:
    raw = entry.get("extra_params")
    parsed = json.loads(raw) if isinstance(raw, str) and raw.strip() else raw or {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _settings_base(slot: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return what the settings apply to: the LLM's `extra_params`, or the fields of an ASR or TTS entry."""
    return _extra_params(entry) if slot == "llm" else {key: value for key, value in entry.items() if key != "settings"}


def settings_schema(slot: str, entry: Mapping[str, Any], profiles: Mapping[str, dict]) -> dict[str, dict]:
    """Return the editable settings of an entry that this example applies, with the entry's own values as defaults.

    A setting whose `requires` or `below` setting is left out is left out too, so the UI never shows a control that
    depends on a missing one.
    """
    profile = profiles.get(entry.get("settings") or "", {})
    schema = entry_settings_schema(profile, _settings_base(slot, entry))
    if slot != "llm":
        schema = {name: spec for name, spec in schema.items() if spec["path"] in SPEECH_SETTINGS[slot]}
    return {
        name: spec
        for name, spec in schema.items()
        if all(spec.get(link) in (None, *schema) for link in ("requires", "below"))
    }


def custom_entry(slot: str, fields: Mapping[str, str], env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return a catalog-shaped entry for a custom endpoint.

    Raises:
        ValueError: Custom endpoints are turned off, a required field is missing, or `extra_params` is not usable.
    """
    if not custom_endpoints_enabled(env):
        raise ValueError(f"custom endpoints are turned off. Set {CUSTOM_ENDPOINTS_ENV}=1 to allow them")
    entry = {name: str(fields.get(name) or "").strip() for name in CUSTOM_FIELDS[slot]}
    missing = [name for name in CUSTOM_REQUIRED[slot] if not entry[name]]
    if missing:
        raise ValueError(f"custom {slot} service needs {', '.join(missing)}")
    entry["name"] = entry["name"] or f"Custom {slot.upper()}"
    if slot == "llm" and entry["extra_params"]:
        _check_extra_params(entry["extra_params"])
    return entry


def _finite_float(text: str) -> float:
    """Parse a JSON number. `NaN`, `Infinity`, and a number too large for a float raise `ValueError`."""
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"{text} is not a finite number")
    return value


def _check_extra_params(raw: str) -> None:
    """Raise `ValueError` unless a custom LLM's `extra_params` is a JSON object that the LLM client can send."""
    try:
        params = json.loads(raw, parse_float=_finite_float, parse_constant=_finite_float)
    except ValueError:
        params = None
    if not isinstance(params, dict):
        raise ValueError("custom llm extra_params must be a JSON object with finite numbers")
    for name in ("extra_body", "chat_template_kwargs"):
        if not isinstance(params.get(name, {}), dict):
            raise ValueError(f"custom llm extra_params.{name} must be a JSON object")


def custom_on_nvidia_cloud(slot: str, entry: Mapping[str, Any]) -> bool:
    """Whether a custom endpoint is an NVIDIA Cloud function, so it gets TLS and the API key.

    An ASR or TTS endpoint is an NVIDIA Cloud function when it has a `function_id` and is NVIDIA's gRPC host. An LLM
    endpoint is on NVIDIA Cloud when it is an `https` URL on NVIDIA's API host. The API key is never sent to any other
    address, or over plain HTTP.
    """
    if slot == "llm":
        url = str(entry.get("base_url", ""))
        return urlsplit(url).scheme == "https" and parse_endpoint(url) == parse_endpoint(NVCF_LLM_BASE_URL)
    return bool(entry.get("function_id")) and parse_endpoint(str(entry.get("server", ""))) == parse_endpoint(
        NVCF_GRPC_SERVER
    )


def _llm_request(
    params: Mapping[str, Any], template_defaults: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split applied LLM params into client arguments and the request body, under `template_defaults`."""
    options = {name: params[name] for name in LLM_ARGUMENTS if name in params}
    body = {
        **{name: value for name, value in params.items() if name not in (*LLM_ARGUMENTS, "extra_body")},
        **dict(params.get("extra_body") or {}),
    }
    kwargs = {**template_defaults, **(body.get("chat_template_kwargs") or {})}
    return options, ({**body, "chat_template_kwargs": kwargs} if kwargs else body)


def load_config(
    recipe: str | None = None,
    env: Mapping[str, str] | None = None,
    catalog: Mapping[str, Any] | None = None,
    choices: Mapping[str, Mapping[str, Any]] | None = None,
    profiles: Mapping[str, dict] | None = None,
) -> AgentConfig:
    """Build the agent configuration.

    Args:
        recipe: `cloud`, `server`, or `single-gpu`. Defaults to `SERVICE_RECIPE`, then `cloud`.
        env: Environment mapping. Defaults to `os.environ`. `LLM_SERVICE`, `ASR_SERVICE`, and `TTS_SERVICE`
            pick other catalog keys.
        catalog: Parsed `services.yaml`. Defaults to the repository file.
        choices: The session's choice per slot, as `session.read_choices` returns it: `{"key": ...}` for a catalog
            entry or `{"custom": {...}}` for a custom endpoint, with optional `settings` values. A slot without a
            choice uses the environment or registry default.
        profiles: Parsed `settings.yaml` profiles. Defaults to the repository file.
    """
    env = os.environ if env is None else env
    recipe = (recipe or env.get("SERVICE_RECIPE") or "cloud").strip().lower()
    catalog = load_catalog() if catalog is None else catalog
    profiles = load_settings_profiles() if profiles is None else profiles
    choices = choices or {}
    hosted = recipe == "cloud"
    keys: dict[str, str] = {}
    entries: dict[str, dict[str, Any]] = {}
    params: dict[str, dict[str, Any]] = {}
    hosted_slots = set()
    for slot, default in DEFAULT_SERVICES.items():
        choice = choices.get(slot) or {}
        if "custom" in choice:
            entry = custom_entry(slot, choice["custom"], env)
            keys[slot] = ""
            if custom_on_nvidia_cloud(slot, entry):
                hosted_slots.add(slot)
        else:
            keys[slot] = choice.get("key") or env.get(f"{slot.upper()}_SERVICE") or default
            entry = resolve_entry(catalog, recipe, slot, keys[slot])
            if hosted:
                hosted_slots.add(slot)
        entries[slot] = entry
        schema = settings_schema(slot, entry, profiles) if keys[slot] else {}
        params[slot] = apply_service_settings(_settings_base(slot, entry), schema, choice.get("settings") or {})
    llm, asr, tts = entries["llm"], entries["asr"], entries["tts"]
    # A custom LLM gets its own parameters only, because another server can refuse the Nemotron template ones.
    llm_options, llm_body = _llm_request(params["llm"], LLM_CHAT_TEMPLATE_KWARGS if keys["llm"] else {})
    tts_hosted = "tts" in hosted_slots
    return AgentConfig(
        recipe=recipe,
        llm_name=llm["name"],
        llm_model=llm["model_id"],
        llm_base_url=llm.get("base_url") or NVCF_LLM_BASE_URL,
        llm_extra_body=llm_body,
        llm_options=llm_options,
        asr_name=asr["name"],
        asr_server=asr.get("server") or NVCF_GRPC_SERVER,
        # Catalog servers and NVIDIA Cloud functions pick their own model, and reject the self-hosted NIM names.
        asr_model=asr.get("model", "") if not keys["asr"] else "",
        asr_function_id=asr.get("function_id", ""),
        asr_language=asr.get("language_code") or "en-US",
        asr_punctuate=bool(params["asr"].get("automatic_punctuation", True)),
        tts_name=tts["name"],
        tts_server=tts.get("server") or NVCF_GRPC_SERVER,
        tts_voice=tts.get("voice_id", ""),
        tts_function_id=tts.get("function_id", ""),
        tts_language=tts.get("language_code") or "en-US",
        tts_sample_rate=SINGLE_GPU_TTS_SAMPLE_RATE
        if recipe == "single-gpu" and keys["tts"] and not tts_hosted
        else DEFAULT_TTS_SAMPLE_RATE,
        keys=keys,
        hosted_slots=frozenset(hosted_slots),
    )


def service_options(
    recipe: str,
    catalog: Mapping[str, Any] | None = None,
    profiles: Mapping[str, dict] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return the services that the UI offers for the recipe, per slot, with their editable settings.

    The first entry of each slot is the default. Entries carry the fields that the Services tab shows, and `source`
    is `cloud-nim` or `self-hosted`, the values that the Pipecat UI uses.
    """
    catalog = load_catalog() if catalog is None else catalog
    profiles = load_settings_profiles() if profiles is None else profiles
    shown = ("base_url", "model_id", "server", "model", "voice_id", "function_id", "language_code", "health_path")
    options: dict[str, list[dict[str, Any]]] = {}
    for slot, keys in offered_services(catalog, recipe, env).items():
        options[slot] = []
        for index, key in enumerate(keys):
            entry = resolve_entry(catalog, recipe, slot, key)
            options[slot].append(
                {
                    "id": key,
                    "name": entry["name"],
                    "source": SOURCES[recipe],
                    "default": index == 0,
                    **{name: entry[name] for name in shown if entry.get(name)},
                    "settings": settings_schema(slot, entry, profiles),
                }
            )
    return options


def main(argv: list[str] | None = None) -> int:
    """List the services that each recipe offers, marking the defaults: `python -m ...config [recipe]`."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    recipes = args or list(RECIPES)
    catalog = load_catalog()
    for recipe in recipes:
        try:
            services = available_services(catalog, recipe)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(recipe)
        for slot, keys in services.items():
            listed = ", ".join(f"{key} (default)" if key == DEFAULT_SERVICES[slot] else key for key in keys)
            print(f"  {slot.upper()}_SERVICE: {listed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
