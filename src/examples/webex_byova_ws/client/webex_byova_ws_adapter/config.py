"""Environment-backed adapter configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

SCHEMA_UUID = "a38a10b7-43e4-4676-a076-a7d6dce9387d"


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes"}


@dataclass(slots=True)
class AdapterConfig:
    """Runtime settings with production-safe defaults."""

    host: str = field(default_factory=lambda: os.getenv("BYOVA_WS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("BYOVA_WS_PORT", "8086")))
    backend_ws_url: str = field(
        default_factory=lambda: os.getenv("NEMOTRON_VOICE_AGENT_WS", "wss://127.0.0.1:7860/api/ws")
    )
    backend_insecure_tls: bool = field(default_factory=lambda: _bool("NEMOTRON_INSECURE_TLS", False))
    tls_cert: str = field(default_factory=lambda: os.getenv("BYOVA_TLS_CERT", ""))
    tls_key: str = field(default_factory=lambda: os.getenv("BYOVA_TLS_KEY", ""))
    allow_plaintext_local: bool = field(default_factory=lambda: _bool("BYOVA_ALLOW_PLAINTEXT_LOCAL", False))
    auth_enabled: bool = field(default_factory=lambda: _bool("BYOVA_AUTH_ENABLED", True))
    allow_auth_disabled_local: bool = field(default_factory=lambda: _bool("BYOVA_ALLOW_AUTH_DISABLED_LOCAL", False))
    valid_issuers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            issuer.strip() for issuer in os.getenv("CISCO_JWT_ISSUERS", "").split(",") if issuer.strip()
        )
    )
    audience: str = field(default_factory=lambda: os.getenv("CISCO_JWT_AUDIENCE", ""))
    subject: str = field(default_factory=lambda: os.getenv("CISCO_JWT_SUBJECT", ""))
    nonce: str = field(default_factory=lambda: os.getenv("CISCO_JWT_NONCE", ""))
    # Reading the nonce from a file lets scheduled rotation take effect without
    # restarting the adapter and dropping a live conversation.
    nonce_file: str = field(default_factory=lambda: os.getenv("CISCO_JWT_NONCE_FILE", ""))
    datasource_url: str = field(default_factory=lambda: os.getenv("CISCO_DATASOURCE_URL", ""))
    schema_uuid: str = field(default_factory=lambda: os.getenv("CISCO_DATASOURCE_SCHEMA_UUID", SCHEMA_UUID))
    org_uuid: str = field(default_factory=lambda: os.getenv("CISCO_ORG_UUID", ""))
    jwks_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CISCO_JWKS_TTL_SECONDS", "3600")))
    auth_clock_skew_seconds: int = field(default_factory=lambda: int(os.getenv("CISCO_AUTH_CLOCK_SKEW_SECONDS", "30")))
    virtual_agent_id: str = field(default_factory=lambda: os.getenv("BYOVA_VIRTUAL_AGENT_ID", "nemotron-generic"))
    virtual_agent_name: str = field(default_factory=lambda: os.getenv("BYOVA_VIRTUAL_AGENT_NAME", "Nemotron Generic"))
    virtual_agent_description: str = field(
        default_factory=lambda: os.getenv("BYOVA_VIRTUAL_AGENT_DESCRIPTION", "NVIDIA Nemotron voice agent")
    )
    queue_size: int = field(default_factory=lambda: int(os.getenv("BYOVA_QUEUE_SIZE", "128")))
    send_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("BYOVA_SEND_TIMEOUT_SECONDS", "5")))
    first_audio_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("BYOVA_FIRST_AUDIO_TIMEOUT_SECONDS", "120"))
    )
    completion_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("BYOVA_COMPLETION_TIMEOUT_SECONDS", "10"))
    )
    continuation_grace_seconds: float = field(
        default_factory=lambda: float(os.getenv("BYOVA_CONTINUATION_GRACE_SECONDS", "5"))
    )
    max_message_bytes: int = field(default_factory=lambda: int(os.getenv("BYOVA_MAX_MESSAGE_BYTES", "1048576")))

    def validate(self) -> None:
        """Reject incomplete or unsafe production configuration."""
        if not 1 <= self.port <= 65_535:
            raise ValueError("BYOVA_WS_PORT must be between 1 and 65535")
        if self.queue_size < 1 or self.max_message_bytes < 1024:
            raise ValueError("queue and message bounds must be positive")
        if (
            self.send_timeout_seconds <= 0
            or self.first_audio_timeout_seconds <= 0
            or self.completion_timeout_seconds <= 0
            or self.continuation_grace_seconds <= 0
            or self.jwks_ttl_seconds <= 0
            or self.auth_clock_skew_seconds < 0
        ):
            raise ValueError("timeouts, JWK cache lifetime, and clock skew must use valid bounds")
        if bool(self.tls_cert) != bool(self.tls_key):
            raise ValueError("BYOVA_TLS_CERT and BYOVA_TLS_KEY must be configured together")
        if not self.tls_cert and not self.allow_plaintext_local:
            raise ValueError("TLS is required; plaintext requires BYOVA_ALLOW_PLAINTEXT_LOCAL=true")
        if not self.auth_enabled:
            if not self.allow_auth_disabled_local:
                raise ValueError("disabled auth requires BYOVA_ALLOW_AUTH_DISABLED_LOCAL=true")
            return
        required = {
            "CISCO_JWT_ISSUERS": self.valid_issuers,
            "CISCO_JWT_AUDIENCE": self.audience,
            "CISCO_JWT_SUBJECT": self.subject,
            "CISCO_JWT_NONCE or CISCO_JWT_NONCE_FILE": self.nonce or self.nonce_file,
            "CISCO_DATASOURCE_URL": self.datasource_url,
            "CISCO_DATASOURCE_SCHEMA_UUID": self.schema_uuid,
            "CISCO_ORG_UUID": self.org_uuid,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError("auth enabled but required settings are missing: " + ", ".join(missing))
        if self.schema_uuid != SCHEMA_UUID:
            raise ValueError(f"CISCO_DATASOURCE_SCHEMA_UUID must be {SCHEMA_UUID}")
