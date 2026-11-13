# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the agent's ASR, LLM, and TTS endpoints from the shared service catalog.

Standard library plus PyYAML only, so the unit tests do not need LiveKit. The catalog is the repository's
`services.yaml`: the same entries and the same `SERVICE_RECIPE` values (`cloud`, `server`, `single-gpu`) that
the Pipecat server uses.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

RECIPES = ("cloud", "server", "single-gpu")
MODE_LABELS = {"cloud": "Cloud", "server": "Server (NIM)", "single-gpu": "Single GPU"}
# The runtime that serves each slot in each recipe, shown in the UI.
RUNTIMES = {
    "cloud": {"asr": "NVIDIA Cloud", "llm": "NVIDIA Cloud", "tts": "NVIDIA Cloud"},
    "server": {"asr": "NVIDIA NIM", "llm": "NVIDIA NIM", "tts": "NVIDIA NIM"},
    "single-gpu": {"asr": "NeMo-Speech.cpp", "llm": "vLLM", "tts": "NeMo-Speech.cpp"},
}
# Defaults match the `generic-assistant` example in examples_registry.yaml (first entry of each slot).
DEFAULT_SERVICES = {
    "llm": "nemotron-lightning",
    "asr": "nemotron-asr-streaming-english",
    "tts": "magpie-multilingual-tts",
}
NVCF_SERVER = "grpc.nvcf.nvidia.com:443"
NVCF_LLM_URL = "https://integrate.api.nvidia.com/v1"
# Voice replies are short, so reasoning is off (the settings.yaml default for the Nemotron LLM entries).
LLM_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
SINGLE_GPU_TTS_SAMPLE_RATE = 22050  # NeMo-Speech.cpp synthesizes at 22.05 kHz
DEFAULT_TTS_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class AgentConfig:
    """Endpoints and options for one deployment recipe."""

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

    @property
    def hosted(self) -> bool:
        """Whether the services run on NVIDIA Cloud (TLS and an API key) instead of local sidecars."""
        return self.recipe == "cloud"

    def models(self) -> list[dict[str, str]]:
        """Return the models in use for the UI, in pipeline order, with the runtime that serves each one."""
        runtimes = RUNTIMES[self.recipe]
        return [
            {"kind": "ASR", "name": self.asr_name, "runs": runtimes["asr"], "id": self.asr_model},
            {"kind": "LLM", "name": self.llm_name, "runs": runtimes["llm"], "id": self.llm_model},
            {"kind": "TTS", "name": self.tts_name, "runs": runtimes["tts"], "id": self.tts_voice},
        ]

    def deployment(self) -> dict[str, object]:
        """Return the deployment mode and the models in use, for the UI."""
        return {"recipe": self.recipe, "label": MODE_LABELS[self.recipe], "models": self.models()}


def default_catalog_path() -> Path:
    """Return `services.yaml` from `SERVICES_FILE`, or from the repository root (`/app` in the container)."""
    override = os.environ.get("SERVICES_FILE")
    return Path(override) if override else Path(__file__).resolve().parents[3] / "services.yaml"


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load the service catalog."""
    with (path or default_catalog_path()).open() as handle:
        return yaml.safe_load(handle)


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
    # Hosted endpoints replace the Compose hostnames: only the nvcf fields may set server or base_url.
    base = {k: v for k, v in entry.items() if k not in ("nvcf", "server", "base_url", "streaming_url")}
    return {**base, **(entry["nvcf"] or {}), "riva_model": entry.get("model", "")}


def _extra_body(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Merge the entry's `extra_params.extra_body` with the default chat template kwargs."""
    raw = entry.get("extra_params")
    body = dict((json.loads(raw) if isinstance(raw, str) else raw or {}).get("extra_body") or {})
    kwargs = {**LLM_CHAT_TEMPLATE_KWARGS, **(body.get("chat_template_kwargs") or {})}
    return {**body, "chat_template_kwargs": kwargs}


def load_config(
    recipe: str | None = None, env: Mapping[str, str] | None = None, catalog: Mapping[str, Any] | None = None
) -> AgentConfig:
    """Build the agent configuration.

    Args:
        recipe: `cloud`, `server`, or `single-gpu`. Defaults to `SERVICE_RECIPE`, then `cloud`.
        env: Environment mapping. Defaults to `os.environ`. `LLM_SERVICE`, `ASR_SERVICE`, and `TTS_SERVICE`
            pick other catalog keys.
        catalog: Parsed `services.yaml`. Defaults to the repository file.
    """
    env = os.environ if env is None else env
    recipe = (recipe or env.get("SERVICE_RECIPE") or "cloud").strip().lower()
    catalog = load_catalog() if catalog is None else catalog
    keys = {slot: env.get(f"{slot.upper()}_SERVICE") or default for slot, default in DEFAULT_SERVICES.items()}
    llm = resolve_entry(catalog, recipe, "llm", keys["llm"])
    asr = resolve_entry(catalog, recipe, "asr", keys["asr"])
    tts = resolve_entry(catalog, recipe, "tts", keys["tts"])
    hosted = recipe == "cloud"
    return AgentConfig(
        recipe=recipe,
        llm_name=llm["name"],
        llm_model=llm["model_id"],
        llm_base_url=llm.get("base_url", NVCF_LLM_URL) if hosted else llm["base_url"],
        llm_extra_body=_extra_body(llm),
        asr_name=asr["name"],
        asr_server=asr.get("server", NVCF_SERVER) if hosted else asr["server"],
        # Self-hosted Riva servers pick their loaded model, so only the hosted endpoint gets a model name. It is the
        # catalog entry's own model: the nvcf "model" override (nemotron-asr-streaming) is the name Pipecat's client
        # uses, and the Riva client in the LiveKit plugin gets "Model not available" for it.
        asr_model=asr["riva_model"] if hosted else "",
        asr_function_id=asr.get("function_id", ""),
        asr_language=asr.get("language_code", "en-US"),
        tts_name=tts["name"],
        tts_server=tts.get("server", NVCF_SERVER) if hosted else tts["server"],
        tts_voice=tts["voice_id"],
        tts_function_id=tts.get("function_id", ""),
        tts_language=tts.get("language_code", "en-US"),
        tts_sample_rate=SINGLE_GPU_TTS_SAMPLE_RATE if recipe == "single-gpu" else DEFAULT_TTS_SAMPLE_RATE,
    )


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
