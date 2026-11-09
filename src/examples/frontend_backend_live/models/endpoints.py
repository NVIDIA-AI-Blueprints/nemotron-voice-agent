# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example's model catalog, read once, and the endpoint each model role talks to."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.shared.delegation_contract import CHAT_COMPLETIONS
from utils import NVCF_LLM_BASE_URL, listed_services, nvidia_api_key, parse_json_dict


def resolve_endpoint(entry: dict, body: dict, *, slot: str) -> ModelEndpoint:
    """Resolve one role's model endpoint from its catalog ``entry``, with the session ``body`` winning."""
    model = body.get("model_id") or entry.get("model_id")
    if not model:
        raise ValueError(
            f"The {slot!r} slot has no model: services.yaml lists no entry for it under this example's services and "
            "SERVICE_RECIPE"
        )
    base_url = body.get("base_url") or entry.get("base_url") or NVCF_LLM_BASE_URL
    extra = parse_json_dict(body.get("extra_params") or entry.get("extra_params", ""), label="extra_params")
    max_tokens = body.get("max_tokens") or entry.get("max_tokens")
    if max_tokens not in (None, ""):
        try:
            extra["max_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            logger.warning(f"Invalid max_tokens {max_tokens!r}; ignoring")
    return ModelEndpoint(
        model=model,
        base_url=base_url,
        api_key=_api_key(entry, base_url, slot),
        extra_params=extra,
        api=_api(entry, model, base_url),
        response_format=str(entry.get("response_format") or "json_schema"),
        service_key=str(entry.get("key") or ""),
    )


def _api(entry: dict, model: str, base_url: str) -> str:
    """Return the API the entry's ``api`` names, for the entry's own model and URL only; else chat completions."""
    if (model, base_url) != (entry.get("model_id"), entry.get("base_url") or NVCF_LLM_BASE_URL):
        return CHAT_COMPLETIONS
    return str(entry.get("api") or CHAT_COMPLETIONS)


def _api_key(entry: dict, base_url: str, slot: str) -> str:
    """Return the key from the variable ``api_key_env`` names, for the entry's own URL only; else ``NVIDIA_API_KEY``."""
    name = str(entry.get("api_key_env") or "").strip()
    if not name or base_url != (entry.get("base_url") or NVCF_LLM_BASE_URL):
        return nvidia_api_key()
    key = (os.getenv(name) or "").strip()
    if not key:
        raise ValueError(
            f"{name} is empty: the {slot!r} slot's entry takes its key from it, and no other key is sent in its place"
        )
    return key


class ExampleCatalog:
    """The bound example's catalog, read once: the entries of each services slot, the default first."""

    def __init__(self, offered: Mapping[str, list[dict]]):
        """Hold ``offered``: each services slot's entries in the catalog's order, its default first."""
        self.offered = {slot: [dict(entry) for entry in entries] for slot, entries in offered.items()}

    @classmethod
    def read(cls) -> ExampleCatalog:
        """Read the catalog of the example bound to the current context, as the repository UI lists it.

        It may probe local endpoints (``SERVICE_RECIPE=auto``), so a root calls it off the event loop.
        """
        return cls(listed_services())

    def entry(self, slot: str, entry_id: str = "") -> dict:
        """Return the slot's entry whose ``id`` or key is ``entry_id``, else its default; ``{}`` when it lists none."""
        entries = self.offered.get(slot) or [{}]
        named = [entry for entry in entries if entry_id and entry_id in (entry.get("id"), entry.get("key"))]
        return dict((named or entries)[0])

    def model(self, slot: str, overrides: Mapping[str, Any] | None = None) -> ModelEndpoint:
        """Return the model endpoint of the entry ``llm_id`` selects, else the slot's default, with ``overrides``."""
        overrides = dict(overrides or {})
        return resolve_endpoint(self.entry(slot, str(overrides.get("llm_id") or "")), overrides, slot=slot)

    def service(self, slot: str) -> dict:
        """Return the service entry of ``slot`` (speech-to-text or text-to-speech)."""
        return self.entry(slot)

    def models(self, slot: str) -> list[str]:
        """Return the model ids the catalog offers for ``slot``, the default first."""
        return list(dict.fromkeys(entry["model_id"] for entry in self.offered.get(slot, []) if entry.get("model_id")))

    def named(self, slot: str, model: str) -> ModelEndpoint:
        """Return the endpoint of the entry serving ``model``, the slot's own first; ``LookupError`` if none does."""
        own = self.offered.get(slot, [])
        others = [entry for name, entries in self.offered.items() if name != slot for entry in entries]
        entry = next((entry for entry in [*own, *others] if entry.get("model_id") == model), None)
        if entry is None:
            raise LookupError(f"Model {model!r} is not in the catalog of this example and SERVICE_RECIPE")
        return resolve_endpoint(entry, {}, slot=slot)
