# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Configuration model for the Cisco Webex BYOVA adapter."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class AdapterConfig:
    """Environment-driven runtime configuration for the adapter."""

    grpc_host: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_GRPC_HOST", "0.0.0.0")
    grpc_port: int = int(os.getenv("NEMOTRON_BYOVA_ADAPTER_GRPC_PORT", "50061"))
    nemotron_voice_agent_ws: str = os.getenv(
        "NEMOTRON_VOICE_AGENT_WS",
        os.getenv("NEMOTRON_WS_BASE", "wss://127.0.0.1:7860"),
    )
    virtual_agent_id: str = os.getenv("DEFAULT_VIRTUAL_AGENT_ID", "nemotron-generic")
    virtual_agent_name: str = os.getenv("DEFAULT_VIRTUAL_AGENT_NAME", "Nemotron Generic")
    allow_insecure_tls: bool = os.getenv("NEMOTRON_INSECURE_TLS", "true").lower() == "true"
    enable_auth: bool = os.getenv("ENABLE_CISCO_JWS_VALIDATION", "false").lower() == "true"
    expected_jwt_issuer: str = os.getenv("CISCO_JWT_ISSUER", "")
    expected_jwt_audience: str = os.getenv("CISCO_JWT_AUDIENCE", "")
    expected_jwt_subject: str = os.getenv("CISCO_JWT_SUBJECT", "")
    jwk_cache_ttl_secs: int = int(os.getenv("CISCO_JWK_CACHE_TTL_SECS", "3600"))
    output_idle_timeout_ms: int = int(os.getenv("OUTPUT_IDLE_TIMEOUT_MS", "350"))
    response_settle_timeout_secs: float = float(os.getenv("RESPONSE_SETTLE_TIMEOUT_SECS", "8.0"))
    response_idle_timeout_secs: float = float(os.getenv("RESPONSE_IDLE_TIMEOUT_SECS", "1.5"))
    # Time to wait for Nemotron's first audio chunk before ending the turn.
    first_audio_timeout_secs: float = float(os.getenv("FIRST_AUDIO_TIMEOUT_SECS", "120.0"))
    health_http_host: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_HEALTH_HOST", "0.0.0.0")
    health_http_port: int = int(os.getenv("NEMOTRON_BYOVA_ADAPTER_HEALTH_PORT", "8081"))
    tls_cert_path: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_TLS_CERT", "")
    tls_key_path: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_TLS_KEY", "")
    tls_ca_path: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_TLS_CA", "")
    health_service_name: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_HEALTH_SERVICE_NAME", "VoiceVirtualAgent")
    health_service_type: str = os.getenv("NEMOTRON_BYOVA_ADAPTER_HEALTH_SERVICE_TYPE", "BYOVA")
    health_service_message: str = os.getenv(
        "NEMOTRON_BYOVA_ADAPTER_HEALTH_MESSAGE",
        "Nemotron Webex BYOVA adapter is healthy",
    )
    default_transfer_keywords: str = os.getenv(
        "DEFAULT_TRANSFER_KEYWORDS",
        "transfer to human,transfer to person,transfer to agent",
    )
    default_transfer_metadata_json: str = os.getenv(
        "DEFAULT_TRANSFER_METADATA_JSON",
        '{"route":"live-agent","reason":"caller_requested_human"}',
    )
    default_end_session_keywords: str = os.getenv(
        "DEFAULT_END_SESSION_KEYWORDS",
        "end the call",
    )
    idle_session_timeout_secs: int = int(os.getenv("ADAPTER_IDLE_SESSION_TIMEOUT_SECS", "600"))

    @property
    def grpc_bind_address(self) -> str:
        """Return the gRPC bind address."""
        return f"{self.grpc_host}:{self.grpc_port}"

    @property
    def health_bind_address(self) -> str:
        """Return the health server bind address."""
        return f"{self.health_http_host}:{self.health_http_port}"

    @property
    def tls_enabled(self) -> bool:
        """Return whether TLS cert and key paths are both configured."""
        return bool(self.tls_cert_path.strip() and self.tls_key_path.strip())

    def validate(self) -> None:
        """Validate settings that must be complete before startup."""
        cert_configured = bool(self.tls_cert_path.strip())
        key_configured = bool(self.tls_key_path.strip())
        if cert_configured != key_configured:
            raise ValueError(
                "Invalid adapter configuration: NEMOTRON_BYOVA_ADAPTER_TLS_CERT and "
                "NEMOTRON_BYOVA_ADAPTER_TLS_KEY must be configured together"
            )

        if not self.enable_auth:
            return

        required = {
            "CISCO_JWT_ISSUER": self.expected_jwt_issuer,
            "CISCO_JWT_AUDIENCE": self.expected_jwt_audience,
        }
        missing = [name for name, value in required.items() if not value.strip()]
        if missing:
            raise ValueError(
                "Invalid adapter configuration: ENABLE_CISCO_JWS_VALIDATION=true requires " + ", ".join(missing)
            )
