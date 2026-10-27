# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The example's catalog, read once, and the endpoint each model role talks to.

:class:`ExampleCatalog` is the catalog a composition root reads once (the ``ServiceCatalog`` a frontend adapter class
is built with, ``adapter/contract.py``): the entries the bound example offers for each services slot, the default
first, and the example's prompts. :func:`resolve_endpoint` turns an entry and the session body into an endpoint. The
delegation core names its default model by reference (``backend.slot``) and never reads the catalog: the root builds
its ``EndpointLookup`` from :meth:`ExampleCatalog.model` and :meth:`ExampleCatalog.named`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from loguru import logger

from examples.frontend_backend_live.config_manager.schema import ModelEndpoint
from examples.frontend_backend_live.prompts.catalog import role_instructions
from utils import NVCF_LLM_BASE_URL, build_services_api_response, nvidia_api_key, parse_json_dict


def resolve_endpoint(entry: dict, body: dict, prefix: str = "", *, slot: str) -> ModelEndpoint:
    """Resolve one role's model endpoint.

    The session ``body`` (UI overrides) wins over the ``services.yaml`` ``entry``. A base URL that neither sets is the
    NVIDIA Cloud LLM URL (``utils.NVCF_LLM_BASE_URL``). An entry can set ``api_key_env`` to the name of the environment
    variable that holds its key, for endpoints that do not use ``NVIDIA_API_KEY``. ``extra_params`` carries every
    sampling and reasoning setting (``temperature``, ``max_tokens``, ``extra_body`` ...) because ``settings.yaml``
    writes them there.

    Args:
        entry: The selected catalog entry (``{}`` when the slot lists none).
        body: The session body.
        prefix: Body key prefix: ``""`` for the frontend, ``"thinker_"`` for the backend.
        slot: The services slot the entry comes from, named in errors.

    Returns:
        The endpoint.

    Raises:
        ValueError: Neither the body nor the entry names a model (the slot lists no entry for this example and
            ``SERVICE_RECIPE``), or the entry's ``api_key_env`` names a variable that is empty.
    """
    model = body.get(f"{prefix}model_id") or entry.get("model_id")
    if not model:
        raise ValueError(
            f"The {slot!r} slot has no model: services.yaml lists no entry for it under this example's services and "
            "SERVICE_RECIPE"
        )
    base_url = body.get(f"{prefix}base_url") or entry.get("base_url") or NVCF_LLM_BASE_URL
    extra = parse_json_dict(
        body.get(f"{prefix}extra_params") or entry.get("extra_params", ""), label=f"{prefix}extra_params"
    )
    max_tokens = body.get(f"{prefix}max_tokens") or entry.get("max_tokens")
    if max_tokens not in (None, ""):
        try:
            extra["max_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            logger.warning(f"Invalid {prefix}max_tokens {max_tokens!r}; ignoring")
    return ModelEndpoint(
        model=model,
        base_url=base_url,
        api_key=_api_key(entry, slot),
        extra_params=extra,
        response_format=str(entry.get("response_format") or "json_schema"),
    )


def _api_key(entry: dict, slot: str) -> str:
    """The entry's key: from the variable its ``api_key_env`` names, and only that one; else ``NVIDIA_API_KEY``.

    Raises:
        ValueError: The named variable is empty, so the NVIDIA key is never sent to the entry's server in its place.
    """
    name = str(entry.get("api_key_env") or "").strip()
    if not name:
        return nvidia_api_key()
    key = (os.getenv(name) or "").strip()
    if not key:
        raise ValueError(
            f"{name} is empty: the {slot!r} slot's entry takes its key from it, and no other key is sent in its place"
        )
    return key


class ExampleCatalog:
    """The bound example's catalog, read once: the entries of each services slot, the default first, and its prompts.

    It is the ``ServiceCatalog`` of ``adapter/contract.py``, and the root's source of the core's ``EndpointLookup``.
    """

    def __init__(self, offered: Mapping[str, list[dict]]):
        """Hold ``offered``: for each services slot, its entries in the catalog's order, the slot's default first."""
        self.offered = {slot: [dict(entry) for entry in entries] for slot, entries in offered.items()}

    @classmethod
    def read(cls) -> ExampleCatalog:
        """Read the catalog of the example bound to the current context (``utils.set_service_context``).

        It may probe local endpoints (``SERVICE_RECIPE=auto``), so a root calls it off the event loop.
        """
        return cls(build_services_api_response())

    def entry(self, slot: str) -> dict:
        """The slot's default entry, or ``{}`` when the example's catalog lists none for it."""
        entries = self.offered.get(slot) or [{}]
        return dict(entries[0])

    def model(self, slot: str, overrides: Mapping[str, Any] | None = None) -> ModelEndpoint:
        """The model endpoint the slot names, with the session's ``overrides`` (its own fields, unprefixed).

        Raises:
            ValueError: The slot has no model (an unlisted slot), or its entry's key variable is empty.
        """
        return resolve_endpoint(self.entry(slot), dict(overrides or {}), slot=slot)

    def service(self, slot: str) -> dict:
        """The service entry a slot names (speech-to-text, text-to-speech)."""
        return self.entry(slot)

    def prompt(self, key: str) -> str:
        """A role's static instructions, named by its ``prompts.yaml`` key: the entry, then the shared channel context.

        It is what a role starts from when a session supplies no instructions (``role_instructions``).
        """
        return role_instructions(key)

    def models(self, slot: str) -> list[str]:
        """The model ids the catalog offers for a slot, the default first."""
        return list(dict.fromkeys(entry["model_id"] for entry in self.offered.get(slot, []) if entry.get("model_id")))

    def named(self, slot: str, model: str) -> ModelEndpoint:
        """The endpoint of the catalog entry whose ``model_id`` is ``model``.

        The slot's own entries come first, then the other slots', so a model keeps the settings its slot gives it.

        Raises:
            LookupError: The catalog lists no entry for ``model`` (the backend reports it, or falls back where its
                settings say so).
            ValueError: The entry's key variable is empty.
        """
        own = self.offered.get(slot, [])
        others = [entry for name, entries in self.offered.items() if name != slot for entry in entries]
        entry = next((entry for entry in [*own, *others] if entry.get("model_id") == model), None)
        if entry is None:
            raise LookupError(f"Model {model!r} is not in the catalog of this example and SERVICE_RECIPE")
        return resolve_endpoint(entry, {}, slot=slot)
