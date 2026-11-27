# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import os
from unittest.mock import patch

import pytest

from nemotron_voice_agent.pipecat.runtime import session_ticket
from nemotron_voice_agent.pipecat.runtime.session_ticket import TicketError, issue, redeem

CONFIG = {"pipeline_mode": "generic-assistant", "prompt_content": "Be brief. " * 50}


def test_round_trip_returns_config_and_session_id() -> None:
    token, ticket = issue(CONFIG)
    assert redeem(token) == ticket
    assert ticket.config == CONFIG and len(ticket.session_id) == 12


def test_each_issue_gets_a_new_session_id() -> None:
    assert issue(CONFIG)[1].session_id != issue(CONFIG)[1].session_id


def test_tampered_or_garbage_tickets_are_rejected() -> None:
    token, _ = issue(CONFIG)
    payload, signature = token.split(".")
    for bad in (payload + "A." + signature, payload + "." + signature[:-2] + "AA", "", "x", "a.b", ".sig"):
        with pytest.raises(TicketError, match="Invalid session ticket"):
            redeem(bad)


def test_ticket_expires() -> None:
    token, _ = issue(CONFIG, ttl_secs=10, now=1000)
    assert redeem(token, now=1010).config == CONFIG
    with pytest.raises(TicketError, match="expired"):
        redeem(token, now=1011)


def test_a_ticket_from_another_secret_is_rejected() -> None:
    with patch.dict(os.environ, {session_ticket.SECRET_ENV: "replica-a"}):
        token, _ = issue(CONFIG)
        assert redeem(token).config == CONFIG
    with patch.dict(os.environ, {session_ticket.SECRET_ENV: "replica-b"}), pytest.raises(TicketError, match="Invalid"):
        redeem(token)


def test_workers_inherit_the_shared_secret() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop(session_ticket.SECRET_ENV, None)
        session_ticket.share_secret_with_workers()
        first = os.environ[session_ticket.SECRET_ENV]
        session_ticket.share_secret_with_workers()  # never overwrites an operator-provided key
        assert os.environ[session_ticket.SECRET_ENV] == first


def test_oversized_config_is_refused() -> None:
    with pytest.raises(TicketError, match="too large"):
        issue({"prompt_content": os.urandom(20_000).hex()})
