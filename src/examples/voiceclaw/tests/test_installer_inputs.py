# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause
# ruff: noqa: D100, D101, D102, D103, D107

import pytest

from voiceclaw.installer_inputs import ContainerInputError, ContainerSettings, strict_json, validate_bearer

SANDBOX_ID = "893ed65e-1972-48ba-b5b7-764119c24d07"


def environment(*, no_auth=False, **overrides):
    value = {
        "VOICECLAW_GATEWAY_URL": "https://openshell.example.com:8443",
        "VOICECLAW_WORKSPACE": "explicit-workspace",
        "VOICECLAW_SANDBOX_NAME": "assistant",
        "VOICECLAW_SANDBOX_ID": SANDBOX_ID,
        "VOICECLAW_AGENT_NAME": "assistant",
        "VOICECLAW_AUTH_MODE": "oidcBearer",
        "VOICECLAW_GATEWAY_TOKEN": "fixture.bearer",
        "VOICECLAW_SPEECH_KEY": "nvapi-fixture",
    }
    if no_auth:
        value["VOICECLAW_GATEWAY_URL"] = "http://172.18.0.2:8080"
        value["VOICECLAW_AUTH_MODE"] = "none"
        del value["VOICECLAW_GATEWAY_TOKEN"]
    for name, item in overrides.items():
        if item is None:
            value.pop(name, None)
        else:
            value[name] = item
    return value


def settings(value=None):
    return ContainerSettings.from_environment(environment() if value is None else value)


def connection(value=None):
    return settings(value).connection


def test_valid_environment_settings():
    s = settings()
    c = s.connection
    assert s.port == 18790 and s.speech_key == "nvapi-fixture"
    assert (c.workspace, c.sandbox, c.sandbox_id, c.agent) == (
        "explicit-workspace",
        "assistant",
        SANDBOX_ID,
        "assistant",
    )
    assert c.authentication_mode == "oidcBearer" and c.bearer == "fixture.bearer"
    assert c.health_seconds == 12 and c.invoke_seconds == 120
    assert settings(environment(VOICECLAW_PORT="18791")).port == 18791
    assert "fixture.bearer" not in repr(s) and "nvapi-fixture" not in repr(s)


def test_no_auth_selects_plaintext_without_any_token():
    c = connection(environment(no_auth=True))
    assert c.authentication_mode == "none" and c.bearer is None
    assert c.endpoint == "http://172.18.0.2:8080"


@pytest.mark.parametrize(
    "no_auth,overrides",
    [
        (True, {"VOICECLAW_GATEWAY_TOKEN": "fixture.bearer"}),
        (True, {"VOICECLAW_GATEWAY_URL": "https://172.18.0.2:8080"}),
        (True, {"VOICECLAW_AUTH_MODE": "oidcBearer"}),
        (False, {"VOICECLAW_GATEWAY_TOKEN": None}),
        (False, {"VOICECLAW_GATEWAY_URL": "http://172.18.0.2:8080"}),
        (False, {"VOICECLAW_AUTH_MODE": "none"}),
        (False, {"VOICECLAW_AUTH_MODE": None}),
        (False, {"VOICECLAW_AUTH_MODE": "mTLS"}),
    ],
)
def test_rejects_inconsistent_authentication_and_transport(no_auth, overrides):
    with pytest.raises(ContainerInputError, match="^container-input-invalid$"):
        settings(environment(no_auth=no_auth, **overrides))


@pytest.mark.parametrize(
    "name,bad",
    [
        ("VOICECLAW_GATEWAY_URL", None),
        ("VOICECLAW_WORKSPACE", None),
        ("VOICECLAW_WORKSPACE", ""),
        ("VOICECLAW_SANDBOX_NAME", None),
        ("VOICECLAW_SANDBOX_NAME", "a/b"),
        ("VOICECLAW_SANDBOX_ID", None),
        ("VOICECLAW_SANDBOX_ID", "ACTUAL_SANDBOX_ID"),
        ("VOICECLAW_SANDBOX_ID", "00000000-0000-0000-0000-000000000000"),
        ("VOICECLAW_AGENT_NAME", None),
        ("VOICECLAW_AGENT_NAME", "../assistant"),
        ("VOICECLAW_GATEWAY_TOKEN", "bearer with spaces"),
        ("VOICECLAW_GATEWAY_TOKEN", ""),
        ("VOICECLAW_SPEECH_KEY", None),
        ("VOICECLAW_SPEECH_KEY", ""),
        ("VOICECLAW_SPEECH_KEY", "key\nvalue"),
        ("VOICECLAW_PORT", "0"),
        ("VOICECLAW_PORT", "65536"),
        ("VOICECLAW_PORT", "018790"),
        ("VOICECLAW_PORT", "port"),
    ],
)
def test_rejects_missing_or_invalid_settings_without_echoing_values(name, bad):
    with pytest.raises(ContainerInputError, match="^container-input-invalid$"):
        settings(environment(**{name: bad}))


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://8.8.8.8:8080",
        "http://100.64.0.1:8080",
        "http://198.18.0.1:8080",
        "http://169.254.0.1:8080",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:8080",
        "http://224.0.0.1:8080",
        "http://localhost:8080",
        "http://openshell.example.com:8080",
        "http://gateway:8080",
        "http://[::1]:8080",
        "http://[fd00::1]:8080",
        "http://[::ffff:172.18.0.2]:8080",
        "http://0xac120002:8080",
        "http://172.18.2:8080",
        "http://172.018.0.2:8080",
        "http://172.18.0.2",
        "http://172.18.0.2:0",
        "http://172.18.0.2:080",
        "HTTP://172.18.0.2:8080",
        "http://172.18.0.2:65536",
        "http://172.18.0.2:8080/path",
        "http://172.18.0.2:8080?token=x",
        "http://172.18.0.2:8080#x",
        "http://user:secret@172.18.0.2:8080",
        "http://172.18.0.2:8080\n",
    ],
)
def test_no_auth_rejects_nonprivate_or_ambiguous_endpoints(endpoint):
    with pytest.raises(ContainerInputError):
        connection(environment(no_auth=True, VOICECLAW_GATEWAY_URL=endpoint))


@pytest.mark.parametrize("endpoint", ["http://10.23.1.2:8080", "http://172.18.0.2:8080", "http://192.168.50.2:8080/"])
def test_no_auth_accepts_canonical_rfc1918_literals(endpoint):
    assert connection(environment(no_auth=True, VOICECLAW_GATEWAY_URL=endpoint)).endpoint == endpoint


@pytest.mark.parametrize(
    "url",
    [
        "http://openshell.example.com:8443",
        "https://user:secret@openshell.example.com:8443",
        "https://localhost:8443",
        "https://127.0.0.1:8443",
        "https://[::1]:8443",
        "https://[::ffff:127.0.0.1]:8443",
        "https://[::ffff:0.0.0.0]:8443",
        "https://0x7f000001:8443",
        "https://0x7f.0.0.1:8443",
        "https://0.0.0.0:8443",
        "https://openshell.example.com:8443/#secret",
        "https://openshell.example.com:8443?token=secret",
        "https://openshell.example.com:8443/path",
        "https://openshell.example.com:8443\n",
        "https://<host>:8443",
    ],
)
def test_rejects_unsafe_endpoint(url):
    with pytest.raises(ContainerInputError):
        connection(environment(VOICECLAW_GATEWAY_URL=url))


@pytest.mark.parametrize("raw", [b"[]", b"{} {}", b"\xff", b'{"x":NaN}', b'{"x":1,"x":1}', b" " * 65537])
def test_strict_json_rejects_ambiguous_payloads(raw):
    with pytest.raises(ContainerInputError):
        strict_json(raw, 65536)


def test_bearer_validation_is_bounded_and_header_safe():
    assert validate_bearer(b"short-valid-bearer") == "short-valid-bearer"
    for bad in (b"", b"a b", b"a\r\nx", b"a" * (16 * 1024 + 1)):
        with pytest.raises(ContainerInputError):
            validate_bearer(bad)
