# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Entrypoint for running the Cisco Webex BYOVA adapter process."""

from __future__ import annotations

import asyncio
import contextlib
import logging

import grpc

from cisco_webex_byova_adapter.auth import CiscoAuthInterceptor
from cisco_webex_byova_adapter.config import AdapterConfig
from cisco_webex_byova_adapter.generated import health_pb2_grpc, voicevirtualagent_pb2_grpc
from cisco_webex_byova_adapter.http_health import start_health_http_server
from cisco_webex_byova_adapter.service import HealthServicer, VoiceVirtualAgentServicer

logger = logging.getLogger(__name__)


async def serve() -> None:
    """Start the adapter gRPC and health servers."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = AdapterConfig()
    config.validate()
    interceptor = CiscoAuthInterceptor(config)
    server = grpc.aio.server(interceptors=(interceptor,))
    voicevirtualagent_pb2_grpc.add_VoiceVirtualAgentServicer_to_server(VoiceVirtualAgentServicer(config), server)
    health_pb2_grpc.add_HealthServicer_to_server(HealthServicer(), server)
    if config.tls_enabled:
        with open(config.tls_key_path, "rb") as key_file, open(config.tls_cert_path, "rb") as cert_file:
            credentials = grpc.ssl_server_credentials([(key_file.read(), cert_file.read())])
        bound_port = server.add_secure_port(config.grpc_bind_address, credentials)
        grpc_scheme = "grpc+tls"
        health_scheme = "https"
    else:
        bound_port = server.add_insecure_port(config.grpc_bind_address)
        grpc_scheme = "grpc"
        health_scheme = "http"
    if bound_port == 0:
        logger.error("Failed to bind adapter gRPC server to %s", config.grpc_bind_address)
        raise RuntimeError(f"failed to bind adapter gRPC server to {config.grpc_bind_address}")
    health_server = await start_health_http_server(config)
    try:
        await server.start()
        print(f"cisco-webex-byova-adapter listening on {grpc_scheme}://{config.grpc_bind_address}")
        print(f"health endpoint listening on {health_scheme}://{config.health_bind_address}/voiceva/v1/ping")
        await server.wait_for_termination()
    finally:
        health_server.close()
        await health_server.wait_closed()
        await server.stop(grace=1)


def main() -> None:
    """Run the adapter until interrupted."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(serve())
