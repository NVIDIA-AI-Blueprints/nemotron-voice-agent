# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe whether a self-hosted service is ready, starting, or not running. Standard library only."""

import http.client
import json
import socket
import urllib.error
import urllib.request
from typing import Literal
from urllib.parse import urlparse

ServiceStatus = Literal["ready", "starting", "error", "offline"]
LOCAL_SERVICE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "host.docker.internal"})


def parse_endpoint(server: str) -> tuple[str, int] | None:
    """Parse ``server`` (host[:port] or URL) into ``(host, port)``.

    Returns ``None`` when the address cannot be parsed, for example with an out-of-range port or a broken IPv6 address.
    """
    if not server:
        return None
    try:
        parsed = urlparse(server if "://" in server else f"//{server}")
        host, port = parsed.hostname, parsed.port
    except ValueError:
        return None
    if not host:
        return None
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def service_host_port(server: str) -> tuple[str, int]:
    """Parse a service address into host/port, raising on invalid input."""
    parsed = parse_endpoint(server)
    if parsed is None:
        raise RuntimeError(f"Invalid service address: {server or '(empty)'}")
    return parsed


def _unreachable_status(host: str, exc: BaseException) -> ServiceStatus:
    """An unresolvable host, or a refused local port, means the container is not running.

    A running container's published port accepts connections through docker-proxy even while
    the model loads, so only a stopped one refuses on ``localhost``.
    """
    if isinstance(exc, socket.gaierror):
        return "offline"
    if isinstance(exc, ConnectionRefusedError) and host.strip("[]").lower() in LOCAL_SERVICE_HOSTS:
        return "offline"
    return "starting"


def probe_tcp(server: str, timeout: float) -> ServiceStatus:
    """Probe a gRPC port for services without an HTTP readiness endpoint."""
    host, port = service_host_port(server)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "ready"
    except OSError as exc:
        return _unreachable_status(host, exc)


def probe_http(health_url: str, expects_ready_json: bool, timeout: float) -> ServiceStatus:
    """Probe a health URL; see ``_unreachable_status`` for when a failure means offline."""
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as response:
            body = response.read()
    except urllib.error.URLError as exc:
        return _unreachable_status(urlparse(health_url).hostname or "", exc.reason)
    except (OSError, http.client.HTTPException):
        return "starting"
    if not expects_ready_json:
        return "ready"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return "starting"
    if isinstance(payload, dict) and (payload.get("status") == "ready" or payload.get("ready") is True):
        return "ready"
    return "starting"
