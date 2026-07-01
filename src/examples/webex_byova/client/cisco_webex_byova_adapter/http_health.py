# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Minimal HTTP health endpoint for the Cisco Webex BYOVA adapter."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import ssl
from datetime import UTC, datetime

from cisco_webex_byova_adapter.config import AdapterConfig

logger = logging.getLogger(__name__)

READ_TIMEOUT_SECS = 5.0
MAX_HEADER_LINES = 100
MAX_HEADER_BYTES = 8192


def _http_response(payload: dict[str, str | int]) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    headers = [
        b"HTTP/1.1 200 OK",
        b"Content-Type: application/json",
        f"Content-Length: {len(body)}".encode("ascii"),
        b"Connection: close",
        b"",
        b"",
    ]
    return b"\r\n".join(headers) + body


async def _handle_ping(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, config: AdapterConfig) -> None:
    try:
        request_line = await asyncio.wait_for(reader.readline(), timeout=READ_TIMEOUT_SECS)
        if not request_line:
            return
        header_count = 0
        header_bytes = 0
        while True:
            header = await asyncio.wait_for(reader.readline(), timeout=READ_TIMEOUT_SECS)
            if not header or header in {b"\r\n", b"\n"}:
                break
            header_count += 1
            header_bytes += len(header)
            if header_count > MAX_HEADER_LINES or header_bytes > MAX_HEADER_BYTES:
                logger.warning(
                    "Closing health check connection after header limit exceeded lines=%d bytes=%d",
                    header_count,
                    header_bytes,
                )
                return

        parts = request_line.decode("ascii", errors="ignore").strip().split()
        method = parts[0] if len(parts) >= 1 else ""
        path = parts[1] if len(parts) >= 2 else ""
        if method != "GET" or path != "/voiceva/v1/ping":
            writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
            return

        payload = {
            "serviceName": config.health_service_name,
            "serviceType": config.health_service_type,
            "serviceState": "SERVING",
            "message": config.health_service_message,
            "lastUpdated": datetime.now(UTC).isoformat(),
        }
        writer.write(_http_response(payload))
        await writer.drain()
    except TimeoutError:
        logger.warning("Closing health check connection after read timeout")
        return
    finally:
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            writer.close()
            await writer.wait_closed()


async def start_health_http_server(config: AdapterConfig) -> asyncio.AbstractServer:
    """Start the HTTP health server used by Cisco service checks."""
    ssl_context: ssl.SSLContext | None = None
    if config.tls_enabled:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=config.tls_cert_path, keyfile=config.tls_key_path)
    return await asyncio.start_server(
        lambda reader, writer: _handle_ping(reader, writer, config),
        host=config.health_http_host,
        port=config.health_http_port,
        ssl=ssl_context,
    )
