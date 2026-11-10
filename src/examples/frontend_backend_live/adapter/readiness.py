# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness checks of the services a frontend adapter needs, by the probes of the repository UI's session start."""

from __future__ import annotations

from collections.abc import Mapping

from examples.frontend_backend_live.adapter.contract import ServiceStarting
from utils import (
    _CONNECT_HEALTH_TIMEOUT_SECS,
    _REACHABILITY_TIMEOUT_SECS,
    NVCF_LLM_BASE_URL,
    _llm_health_url,
    _local_speech_ready_url,
    _probe_http,
    _probe_tcp,
    _run_blocking,
    is_nvcf,
    parse_endpoint,
)


def on_nvidia_cloud(address: str) -> bool:
    """Return whether ``address``, a gRPC ``host:port`` or a URL, is one of NVIDIA's cloud endpoints."""
    return is_nvcf(address) or parse_endpoint(address) == parse_endpoint(NVCF_LLM_BASE_URL)


def is_ready(name: str, address: str) -> bool:
    """Return whether the service of slot ``name`` at ``address`` answers its health, readiness or TCP probe."""
    model = "://" in address
    url = _llm_health_url(address) if model else _local_speech_ready_url(name, address)
    if not url:
        return _probe_tcp(address, _REACHABILITY_TIMEOUT_SECS) == "ready"
    return _probe_http(url, not model, _REACHABILITY_TIMEOUT_SECS) == "ready"


class EndpointReadiness:
    """Readiness of the deployment's services; NVIDIA's cloud endpoints, which never start with it, are skipped."""

    def __init__(self, services: Mapping[str, str]):
        """Check ``services``, each a slot name and its address."""
        self.services = {name: address for name, address in services.items() if not on_nvidia_cloud(address)}

    async def ensure_ready(self) -> None:
        """Return once every service is ready; raise :class:`ServiceStarting` for the first one that is not."""
        for name, address in self.services.items():
            try:
                ready = await _run_blocking(is_ready, name, address, timeout=_CONNECT_HEALTH_TIMEOUT_SECS + 1)
            except TimeoutError as exc:
                raise ServiceStarting(name) from exc
            if not ready:
                raise ServiceStarting(name)
