# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

import pytest

from voiceclaw.realtime.auth import ClientSecretIssuer, RealtimeAuthenticationError


def test_client_secret_round_trip_and_expiry() -> None:
    issuer = ClientSecretIssuer("a-sufficiently-long-deployment-master-key")
    secret = issuer.issue(lifetime_seconds=60, now=1_000)

    issuer.verify(secret.value, now=1_030)
    with pytest.raises(RealtimeAuthenticationError):
        issuer.verify(secret.value, now=1_060)


def test_browser_auth_accepts_only_ephemeral_subprotocol_secret() -> None:
    issuer = ClientSecretIssuer("a-sufficiently-long-deployment-master-key")
    secret = issuer.issue(lifetime_seconds=60)
    issuer.authenticate_websocket({"sec-websocket-protocol": f"realtime, openai-insecure-api-key.{secret.value}"})

    with pytest.raises(RealtimeAuthenticationError, match="ephemeral"):
        issuer.authenticate_websocket({"sec-websocket-protocol": "realtime, openai-insecure-api-key.master-secret"})


def test_server_clients_may_use_master_bearer_but_wrong_key_fails() -> None:
    key = "a-sufficiently-long-deployment-master-key"
    issuer = ClientSecretIssuer(key)
    issuer.authenticate_websocket({"authorization": f"Bearer {key}"})

    with pytest.raises(RealtimeAuthenticationError):
        issuer.authenticate_websocket({"authorization": "Bearer definitely-wrong"})
