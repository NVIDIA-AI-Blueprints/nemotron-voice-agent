# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether the services a frontend adapter class needs are up, before the root builds a session's adapter.

:class:`EndpointReadiness` is the class side's ``ServiceReadiness`` (``adapter/contract.py``): it checks that each
service of the deployment accepts connections. Each check runs off the event loop and reuses NVA's reachability probe,
which keeps each answer for a few seconds, so a burst of sessions probes each service once. NVIDIA's cloud endpoints
are not probed: they do not start or stop with the deployment.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from examples.frontend_backend_live.adapter.contract import ServiceStarting
from utils import NVCF_LLM_BASE_URL, is_endpoint_reachable, is_nvcf, parse_endpoint


def on_nvidia_cloud(address: str) -> bool:
    """Whether ``address`` (a gRPC ``host:port`` or a URL) is one of NVIDIA's cloud endpoints."""
    return is_nvcf(address) or parse_endpoint(address) == parse_endpoint(NVCF_LLM_BASE_URL)


class EndpointReadiness:
    """Whether every service of the deployment that an adapter class needs accepts connections."""

    def __init__(self, services: Mapping[str, str]):
        """Check ``services``, each a name the deployment knows it by (its slot) and its address; cloud ones are not."""
        self.services = {name: address for name, address in services.items() if not on_nvidia_cloud(address)}

    async def ensure_ready(self) -> None:
        """Return once every service answers.

        Raises:
            ServiceStarting: The first service, in the given order, that does not accept a connection yet.
        """
        for name, address in self.services.items():
            if not await asyncio.to_thread(is_endpoint_reachable, address):
                raise ServiceStarting(name)
