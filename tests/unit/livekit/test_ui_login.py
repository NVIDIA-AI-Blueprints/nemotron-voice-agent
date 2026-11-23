# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

pytest.importorskip("livekit.api")  # the SDK belongs to the example's own environment

from livekit import api  # noqa: E402

from nemotron_voice_agent.livekit.examples.generic_assistant import ui_server  # noqa: E402
from nemotron_voice_agent.livekit.examples.generic_assistant.service_state import (  # noqa: E402
    SERVICES_FILE,
    write_service_state,
)


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVEKIT_API_KEY", "key")
    monkeypatch.setenv("LIVEKIT_API_SECRET", "secret-for-tests-0123456789abcdef")
    monkeypatch.setenv("LIVEKIT_STATE_DIR", str(tmp_path))
    for name in ("LIVEKIT_UI_USERNAME", "LIVEKIT_UI_PASSWORD", "LIVEKIT_PUBLIC_WS_URL", "LIVEKIT_SIP"):
        monkeypatch.delenv(name, raising=False)  # a developer's own settings must not leak into the tests
    monkeypatch.setattr(ui_server, "FAILED_LOGIN_DELAY", 0)
    public = tmp_path / "dist"
    public.mkdir()
    (public / "index.html").write_text("<!doctype html>")
    monkeypatch.setattr(ui_server, "PUBLIC", public)
    ui_server._attempts.clear()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ui_server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def request(base, path, body=None, cookie=None, headers=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode())
    if cookie:
        req.add_header("Cookie", cookie)
    for name, value in (headers or {}).items():
        req.add_header(name, value)
    try:
        with urllib.request.urlopen(req, timeout=5) as response:  # noqa: S310 - a loopback test server
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def test_without_credentials_everything_stays_open(server, monkeypatch):
    monkeypatch.delenv("LIVEKIT_UI_PASSWORD", raising=False)
    monkeypatch.delenv("LIVEKIT_UI_USERNAME", raising=False)
    assert request(server, "/prompts")[0] == 200
    status = json.loads(request(server, "/auth/status")[2])
    assert status == {"required": False, "username": False, "authenticated": True}


def test_with_a_password_the_data_endpoints_need_a_login(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    for path in sorted(ui_server.PROTECTED):
        assert request(server, path)[0] == 401, path
    assert request(server, "/token", {})[0] == 401
    assert request(server, "/")[0] == 200  # the page itself loads, so the login form can show
    status = json.loads(request(server, "/auth/status")[2])
    assert status == {"required": True, "username": False, "authenticated": False}


def test_a_wrong_password_is_refused_and_a_right_one_sets_a_cookie(server, monkeypatch, tmp_path):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    publish(tmp_path)
    assert request(server, "/login", {"password": "nope"})[0] == 401
    assert request(server, "/login", {})[0] == 401
    status, headers, _ = request(server, "/login", {"password": "correct horse"})
    assert status == 200
    cookie = headers["Set-Cookie"]
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie
    assert "correct horse" not in cookie  # the cookie never carries the password
    assert request(server, "/prompts", cookie=cookie.split(";")[0])[0] == 200
    assert request(server, "/token", {}, cookie=cookie.split(";")[0])[0] == 200


def test_forged_and_expired_cookies_are_refused(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    assert request(server, "/prompts", cookie="lk_ui=9999999999.deadbeef")[0] == 401
    old = ui_server.make_session(now=0)  # expired long ago
    assert not ui_server.valid_session(old)
    assert request(server, "/prompts", cookie=f"lk_ui={old}")[0] == 401


def test_repeated_wrong_passwords_are_limited(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    codes = [
        request(server, "/login", {"password": "x"})[0] for _ in range(ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE + 1)
    ]
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
    assert request(server, "/prompts")[0] == 401  # a half-set pair never opens the page
    assert request(server, "/login", {"username": "alice", "password": ""})[0] == 401


def test_the_public_livekit_address_is_served_without_a_login(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    monkeypatch.setenv("LIVEKIT_PUBLIC_WS_URL", "wss://example.org/livekit")
    assert json.loads(request(server, "/client-config")[2])["ws_url"] == "wss://example.org/livekit"
    monkeypatch.delenv("LIVEKIT_PUBLIC_WS_URL")
    assert json.loads(request(server, "/client-config")[2])["ws_url"] == ""


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
    assert json.loads(request(server, "/client-config")[2])["ws_url"] == ""
    monkeypatch.setenv("LIVEKIT_PUBLIC_WS_URL", "https://voice.example.org")
    assert json.loads(request(server, "/client-config")[2])["ws_url"] == ""


def test_one_client_cannot_lock_out_the_others(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    attacker = {"X-Forwarded-For": "203.0.113.7"}  # the proxy on this host appends the real client address
    for _ in range(ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE):
        request(server, "/login", {"password": "x"}, headers=attacker)
    assert request(server, "/login", {"password": "x"}, headers=attacker)[0] == 429
    friend = {"X-Forwarded-For": "198.51.100.9"}
    assert request(server, "/login", {"password": "correct horse"}, headers=friend)[0] == 200


def test_the_forwarded_header_is_trusted_only_from_a_local_proxy():
    assert ui_server.client_key("127.0.0.1", "10.0.0.1, 203.0.113.7") == "203.0.113.7"  # the proxy's own entry
    assert ui_server.client_key("127.0.0.1", None) == "127.0.0.1"
    assert ui_server.client_key("192.168.1.50", "203.0.113.7") == "192.168.1.50"  # spoofable, so ignored


def test_concurrent_attempts_cannot_pass_the_limit():
    ui_server._attempts.clear()
    results = []
    gate = threading.Barrier(40)

    def attempt():
        gate.wait()
        results.append(ui_server.reserve_attempt("client", now=1000.0))

    threads = [threading.Thread(target=attempt) for _ in range(40)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len([r for r in results if r is not None]) == ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE
    assert (
        len({r for r in results if r is not None}) == ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE
    )  # every ticket is unique


def test_a_right_password_does_not_use_up_attempts(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    for _ in range(ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE + 5):
        assert request(server, "/login", {"password": "correct horse"})[0] == 200


def raw_status(base, content_length):
    """Send a POST with a hand-written Content-Length and return the status code of the server's reply.

    urllib refuses to send an invalid length, so this uses a socket. A refused or dropped connection raises instead of
    counting as a reply.
    """
    host, port = base.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as conn:
        conn.sendall(f"POST /login HTTP/1.1\r\nHost: {host}\r\nContent-Length: {content_length}\r\n\r\n{{}}".encode())
        reply = conn.recv(4096).decode()
    return int(reply.split(" ", 2)[1])  # "HTTP/1.0 400 Bad Request"


def test_an_invalid_or_oversized_content_length_is_rejected(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    assert request(server, "/login", {"password": "x" * 2000})[0] == 413
    assert raw_status(server, "abc") == 400
    assert raw_status(server, "-5") == 400
    assert raw_status(server, "99999") == 413


def test_a_request_that_stalls_before_its_body_is_dropped(server, monkeypatch):
    monkeypatch.setenv("LIVEKIT_UI_PASSWORD", "correct horse")
    monkeypatch.setattr(ui_server.Handler, "timeout", 0.5)
    host, port = server.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as conn:
        conn.sendall(b"POST /login HTTP/1.1\r\nHost: x\r\nContent-Length: 100\r\n\r\n{}")  # 98 bytes never arrive
        assert conn.recv(4096) == b""  # the server closed the connection instead of waiting forever
    assert not ui_server._attempts.get("127.0.0.1")  # the stalled attempt was given back
    assert request(server, "/login", {"password": "correct horse"})[0] == 200  # and it still serves others


def test_releasing_a_timed_out_attempt_keeps_the_failures_that_other_requests_recorded():
    ui_server._attempts.clear()
    stalled = ui_server.reserve_attempt("client", now=0.0)  # request A, which later times out
    for _ in range(ui_server.MAX_LOGIN_ATTEMPTS_PER_MINUTE - 1):
        assert ui_server.reserve_attempt("client", now=10.0) is not None  # nine wrong passwords
    assert ui_server.reserve_attempt("client", now=12.0) is None  # the limit is reached
    ui_server.release_attempt("client", stalled)  # A times out at t=15 and gives back only its own attempt
    # A's slot is free again, but the nine real failures still count
    assert ui_server.reserve_attempt("client", now=15.0) is not None
    assert ui_server.reserve_attempt("client", now=16.0) is None
    # after A's original minute passes, the nine failures are still inside theirs: no extra attempts are allowed
    assert ui_server.reserve_attempt("client", now=60.5) is None


PUBLISHED = {
    "services": {
        "llm": [{"id": "nemotron-super", "default": True, "settings": {"temperature": {"type": "float"}}}],
        "tts": [{"id": "magpie", "default": True}],
    },
    "voices": {"magpie": {"voices": [{"id": "Aria", "name": "Aria"}], "default": "Aria"}},
}


def publish(tmp_path):
    write_service_state(tmp_path / SERVICES_FILE, PUBLISHED["services"], PUBLISHED["voices"])


def token_attributes(server, body):
    status, _, raw = request(server, "/token", body)
    assert status == 200, raw
    claims = api.TokenVerifier("key", "secret-for-tests-0123456789abcdef").verify(json.loads(raw)["token"])
    return claims.attributes


def test_the_data_endpoints_serve_what_the_worker_published(server, tmp_path):
    publish(tmp_path)
    assert json.loads(request(server, "/services")[2]) == PUBLISHED["services"]
    assert json.loads(request(server, "/voices?tts=magpie")[2]) == PUBLISHED["voices"]["magpie"]
    assert json.loads(request(server, "/voices?tts=other")[2]) == {"voices": [], "default": ""}
    prompts = json.loads(request(server, "/prompts")[2])
    assert [p["id"] for p in prompts if p["default"]] == ["generic"]
    tools = json.loads(request(server, "/tools")[2])
    assert all(tool["name"] and tool["parameters"]["type"] == "object" for tool in tools)
    assert {name for p in prompts for name in p["tools"]} <= {tool["name"] for tool in tools}


def test_before_the_worker_publishes_no_services_are_offered(server):
    assert json.loads(request(server, "/services")[2]) == {}


def test_the_token_carries_only_the_offered_choices(server, tmp_path):
    publish(tmp_path)
    body = {
        "scenario": "bank",
        "voice": "Aria",
        "prompt": "Be brief.",
        "services": {"llm": {"key": "nemotron-super", "settings": {"temperature": 0.4}}, "tts": {"key": "nope"}},
    }
    attributes = token_attributes(server, body)
    assert {k: attributes[k] for k in ("scenario", "voice", "prompt")} == {
        "scenario": "bank",
        "voice": "Aria",
        "prompt": "Be brief.",
    }
    assert json.loads(attributes["services"]) == {"llm": {"key": "nemotron-super", "settings": {"temperature": 0.4}}}


def test_a_token_request_before_the_worker_publishes_is_refused(server):
    status, _, raw = request(server, "/token", {"services": {"llm": {"key": "nemotron-super"}}})
    assert status == 503 and "still starting" in json.loads(raw)["error"]


def test_a_body_without_content_length_is_refused(server):
    host, port = server.removeprefix("http://").split(":")
    with socket.create_connection((host, int(port)), timeout=5) as conn:
        conn.sendall(b"POST /token HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n")
        assert conn.recv(4096).decode().split(" ", 2)[1] == "411"


def test_a_bad_token_request_is_refused_with_a_reason(server, tmp_path):
    publish(tmp_path)
    status, _, raw = request(server, "/token", {"services": {"tts": {"custom": {"server": "no port"}}}})
    assert status == 400 and "custom tts" in json.loads(raw)["error"]
    status, _, raw = request(server, "/token", {"services": {"llm": "nemotron-super"}})
    assert status == 400 and json.loads(raw)["error"] == "services.llm has the wrong type"
    assert request(server, "/token", [1])[0] == 400
    assert request(server, "/token", {"prompt": "x" * (ui_server.MAX_TOKEN_BODY + 1)})[0] == 413


def test_a_deeply_nested_body_is_refused(server, tmp_path):
    publish(tmp_path)
    req = urllib.request.Request(server + "/token", data=b"[" * (ui_server.MAX_TOKEN_BODY - 1))
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(req, timeout=5)  # noqa: S310 - a loopback test server
    assert error.value.code == 400


def test_custom_endpoints_are_refused_until_turned_on(server, tmp_path, monkeypatch):
    publish(tmp_path)
    custom = {"services": {"llm": {"custom": {"base_url": "http://my-llm:8000/v1", "model_id": "m"}}}}
    assert json.loads(request(server, "/client-config")[2])["custom_endpoints"] is True
    assert "custom" in json.loads(token_attributes(server, custom)["services"])["llm"]
    monkeypatch.delenv("LIVEKIT_UI_CUSTOM_ENDPOINTS")
    config = json.loads(request(server, "/client-config")[2])
    assert config["custom_endpoints"] is False
    assert config["custom_fields"]["llm"] == {
        "fields": ["name", "base_url", "model_id", "extra_params"],
        "required": ["base_url", "model_id"],
    }
    status, _, raw = request(server, "/token", custom)
    assert status == 400 and "turned off" in json.loads(raw)["error"]
