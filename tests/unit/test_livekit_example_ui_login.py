# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("livekit.api")  # the SDK belongs to the example's own environment

from examples.generic_assistant_livekit import ui_server  # noqa: E402


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-for-tests-0123456789abcdef")
    monkeypatch.setenv("LIVEKIT_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(ui_server, "FAILED_LOGIN_DELAY", 0)
    ui_server._failures.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def request(base, path, body=None, cookie=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode())
    if cookie:
        req.add_header("Cookie", cookie)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310 - a loopback test server
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_without_credentials_everything_stays_open(server, monkeypatch):
    monkeypatch.delenv("LIVEKIT_UI_PASSWORD", raising=False)
    monkeypatch.delenv("LIVEKIT_UI_USERNAME", raising=False)
    assert request(server, "/scenarios")[0] == 200
    status = json.loads(request(server, "/auth/status")[2])
    assert status == {"required": False, "username": False, "authenticated": True}


def test_with_a_password_the_data_endpoints_need_a_login(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    for path in ("/scenarios", "/voices", "/deployment", "/phone", "/token"):
        assert request(server, path)[0] == 401, path
    assert request(server, "/")[0] == 200  # the page itself loads, so the login form can show
    status = json.loads(request(server, "/auth/status")[2])
    assert status == {"required": True, "username": False, "authenticated": False}


def test_a_wrong_password_is_refused_and_a_right_one_sets_a_cookie(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    assert request(server, "/login", {"password": "nope"})[0] == 401
    assert request(server, "/login", {})[0] == 401
    status, headers, _ = request(server, "/login", {"password": "correct horse"})
    assert status == 200
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert "correct horse" not in cookie  # the cookie never carries the password
    assert request(server, "/scenarios", cookie=cookie.split(";")[0])[0] == 200
    assert request(server, "/token", cookie=cookie.split(";")[0])[0] == 200


def test_forged_and_expired_cookies_are_refused(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    assert request(server, "/scenarios", cookie="lk_ui=9999999999.deadbeef")[0] == 401
    old = ui_server.make_session(now=0)  # expired long ago
    assert not ui_server.valid_session(old)
    assert request(server, "/scenarios", cookie=f"lk_ui={old}")[0] == 401


def test_repeated_wrong_passwords_are_limited(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    codes = [request(server, "/login", {"password": "x"})[0] for _ in range(ui_server.MAX_FAILURES_PER_MINUTE + 1)]
    assert codes[-1] == 429
    assert request(server, "/login", {"password": "correct horse"})[0] == 429  # even the right one waits


def test_password_ok_compares_without_errors():
    assert ui_server.password_ok("a", "a")
    assert not ui_server.password_ok("a", "b")
    assert not ui_server.password_ok(None, "a")
    assert not ui_server.password_ok("", "")  # no password configured means no login


def test_with_a_username_both_values_must_match(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_USERNAME", "alice")
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    assert json.loads(request(server, "/auth/status")[2])["username"] is True
    assert request(server, "/login", {"password": "correct horse"})[0] == 401  # no username
    assert request(server, "/login", {"username": "other", "password": "correct horse"})[0] == 401
    assert request(server, "/login", {"username": "alice", "password": "wrong"})[0] == 401
    assert request(server, "/login", {"username": "alice", "password": "correct horse"})[0] == 200


def test_a_username_without_a_password_locks_everyone_out(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_USERNAME", "alice")
    monkeypatch.delenv("LIVEKIT_UI_PASSWORD", raising=False)
    assert request(server, "/scenarios")[0] == 401  # a half-set pair never opens the page
    assert request(server, "/login", {"username": "alice", "password": ""})[0] == 401


def test_the_public_livekit_address_is_served_without_a_login(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    monkeypatch.setenv("LIVEKIT_PUBLIC_WS_URL", "wss://example.org/livekit")
    assert json.loads(request(server, "/client-config")[2]) == {"ws_url": "wss://example.org/livekit"}
    monkeypatch.delenv("LIVEKIT_PUBLIC_WS_URL")
    assert json.loads(request(server, "/client-config")[2]) == {"ws_url": ""}


def test_the_cookie_is_secure_only_behind_an_https_proxy(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")

    def login(headers):
        req = urllib.request.Request(server + "/login", data=json.dumps({"password": "correct horse"}).encode())
        for name, value in headers.items():
            req.add_header(name, value)
        with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310 - a loopback test server
            return response.headers["Set-Cookie"]

    assert "Secure" not in login({})
    assert "Secure" in login({"X-Forwarded-Proto": "https"})


def test_a_public_livekit_address_must_use_a_websocket_scheme(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_PUBLIC_WS_URL", "voice.example.org")  # no scheme
    assert json.loads(request(server, "/client-config")[2]) == {"ws_url": ""}
    monkeypatch.setenv("LIVEKIT_PUBLIC_WS_URL", "https://voice.example.org")
    assert json.loads(request(server, "/client-config")[2]) == {"ws_url": ""}
