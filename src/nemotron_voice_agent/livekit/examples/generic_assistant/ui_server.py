# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the web UI and mint a LiveKit join token at /token (a fresh room per connect), with the session choices."""

from __future__ import annotations

import asyncio
import functools
import hashlib
import hmac
import itertools
import json
import os
import threading
import time
import uuid
from datetime import timedelta
from http import cookies
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from livekit import api

from nemotron_voice_agent.core.paths import REPO_ROOT
from nemotron_voice_agent.livekit.runtime.credentials import DEFAULT_STATE_DIR, apply_to_environ

from .config import CUSTOM_FIELDS, CUSTOM_REQUIRED, custom_endpoints_enabled
from .scenarios import scenario_catalog, scenario_tools
from .service_state import SERVICES_FILE, read_service_state, voices_of
from .session import session_attributes
from .state import DEPLOYMENT_FILE, read_json

PUBLIC = REPO_ROOT / "clients" / "livekit" / "dist"

# Optional login. Set LIVEKIT_UI_USERNAME and LIVEKIT_UI_PASSWORD (in .env) and the data endpoints need a signed session
# cookie. With neither set there is no login. The values stay on the server: the page posts what the user types to
# /login and never receives the expected ones.
USERNAME_ENV = "LIVEKIT_UI_USERNAME"
PASSWORD_ENV = "LIVEKIT_UI_PASSWORD"
SESSION_COOKIE = "lk_ui"
SESSION_SECONDS = 8 * 3600
FAILED_LOGIN_DELAY = 1.0  # seconds a wrong password waits, which makes guessing slow
MAX_LOGIN_ATTEMPTS_PER_MINUTE = 10  # per client
MAX_LOGIN_BODY = 1024  # bytes
MAX_TOKEN_BODY = 32 * 1024  # bytes: the session choices, including a custom prompt
_attempts: dict[str, list[tuple[float, int]]] = {}  # client -> (time, ticket) of each attempt in the last minute
_attempts_lock = threading.Lock()
_tickets = itertools.count()


def ui_password() -> str:
    """Return the configured UI password, or an empty string."""
    return os.environ.get(PASSWORD_ENV, "")


def ui_username() -> str:
    """Return the configured UI username, or an empty string when any username is accepted."""
    return os.environ.get(USERNAME_ENV, "")


def login_required() -> bool:
    """Return True when a username or a password is set. A username without a password locks everyone out, on purpose.

    Setting only half of the pair is a mistake, and failing closed is safer than serving the page without a login.
    """
    return bool(ui_username() or ui_password())


def credentials_ok(username: object, password: object) -> bool:
    """Check the typed username (when one is configured) and password. Both are always compared."""
    user_good = password_ok(username, ui_username()) if ui_username() else True
    return password_ok(password, ui_password()) and user_good


def public_ws_url() -> str:
    """Return the LiveKit address that browsers should use, or an empty string for the page's own host on port 7880."""
    value = os.environ.get("LIVEKIT_PUBLIC_WS_URL", "").strip()
    return value if value.startswith(("ws://", "wss://")) else ""  # anything else would only break the page


def password_ok(given: object, expected: str) -> bool:
    """Compare in constant time. Both sides are hashed first, so the lengths do not leak."""
    if not expected or not isinstance(given, str):
        return False
    return hmac.compare_digest(hashlib.sha256(given.encode()).digest(), hashlib.sha256(expected.encode()).digest())


def _signing_key() -> bytes:
    """Return the cookie signing key, derived from the LiveKit secret so that no extra secret is needed."""
    return hashlib.sha256(b"ui-session:" + os.environ.get("LIVEKIT_API_SECRET", "").encode()).digest()


def make_session(now: float | None = None) -> str:
    """Return a signed cookie value that carries its own expiry, so the server stores no sessions."""
    expires = str(int((time.time() if now is None else now) + SESSION_SECONDS))
    return f"{expires}.{hmac.new(_signing_key(), expires.encode(), hashlib.sha256).hexdigest()}"


def valid_session(value: str | None, now: float | None = None) -> bool:
    """Return True for an untampered, unexpired session cookie value."""
    try:
        expires, signature = (value or "").split(".", 1)
        good = hmac.compare_digest(signature, hmac.new(_signing_key(), expires.encode(), hashlib.sha256).hexdigest())
        return good and float(expires) > (time.time() if now is None else now)
    except ValueError:
        return False


def client_key(peer: str, forwarded: str | None) -> str:
    """Return who is logging in, so that one client cannot lock out the others.

    A proxy on this machine appends the real client address as the last X-Forwarded-For entry, and the page trusts only
    that entry. From any other peer the header is client-controlled, so it is ignored and the peer address is used.
    """
    if peer in ("127.0.0.1", "::1") and forwarded:
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return peer


def reserve_attempt(key: str, now: float | None = None) -> int | None:
    """Count a login attempt before the password is checked, and return its ticket.

    Return None when the client is over the limit. The check and the count happen under one lock, so concurrent
    requests cannot all pass the limit. Give the ticket to `release_attempt` to hand back exactly this attempt.
    """
    now = time.monotonic() if now is None else now
    with _attempts_lock:
        recent = [(t, ticket) for t, ticket in _attempts.get(key, []) if now - t < 60]
        if len(recent) >= MAX_LOGIN_ATTEMPTS_PER_MINUTE:
            _attempts[key] = recent
            return None
        ticket = next(_tickets)
        _attempts[key] = [*recent, (now, ticket)]
        if len(_attempts) > 1024:  # forget idle clients, so the table cannot grow without bound
            for idle in [k for k, held in _attempts.items() if not [1 for t, _ in held if now - t < 60]]:
                del _attempts[idle]
        return ticket


def release_attempt(key: str, ticket: int) -> None:
    """Give back one attempt, found by its ticket, so a failure that another request recorded stays counted."""
    with _attempts_lock:
        _attempts[key] = [(t, held) for t, held in _attempts.get(key, []) if held != ticket]


PHONE_CACHE_SECONDS = 10
_phone_cache: dict = {"at": 0.0, "value": {"enabled": False, "numbers": []}}


def phone_numbers(trunks, rules, scenario_names: dict[str, str]) -> list[dict]:
    """Return the phone numbers that a dispatch rule maps to a use case, as {number, scenario, name}. Pure."""
    by_trunk = {trunk.sip_trunk_id: trunk for trunk in trunks}
    found: dict[str, dict] = {}
    for rule in rules:
        scenario = dict(rule.attributes).get("scenario", "")
        for trunk_id in rule.trunk_ids:
            for number in getattr(by_trunk.get(trunk_id), "numbers", []):
                found[number] = {"number": number, "scenario": scenario, "name": scenario_names.get(scenario, "")}
    return sorted(found.values(), key=lambda item: item["number"])


async def _list_sip_records() -> tuple[list, list]:
    """Read the inbound trunks and dispatch rules from the LiveKit server."""
    lk = api.LiveKitAPI(
        url=os.environ.get("LIVEKIT_URL", "ws://livekit-server:7880"),
        api_key=os.environ["LIVEKIT_API_KEY"],
        api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    try:
        trunks = (await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())).items
        rules = (await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())).items
        return list(trunks), list(rules)
    finally:
        await lk.aclose()


def phone_info() -> dict:
    """Return {enabled, numbers} for the page. Empty unless the deployment runs with `--sip`.

    The numbers come from the live LiveKit records, so a number shows only while a dispatch rule maps it. The result is
    cached for a few seconds, and an unreachable server yields no numbers instead of an error.
    """
    if not os.environ.get("LIVEKIT_SIP"):
        return {"enabled": False, "numbers": []}
    now = time.monotonic()
    if now - _phone_cache["at"] < PHONE_CACHE_SECONDS:
        return _phone_cache["value"]
    try:
        trunks, rules = asyncio.run(asyncio.wait_for(_list_sip_records(), timeout=3))
        names = {item["id"]: item["name"] for item in scenario_catalog()}
        value = {"enabled": True, "numbers": phone_numbers(trunks, rules, names)}
    except Exception:  # noqa: BLE001 - the page must still load when LiveKit is slow or restarting
        value = {"enabled": True, "numbers": []}
    _phone_cache.update(at=now, value=value)
    return value


PROTECTED = {"/services", "/prompts", "/tools", "/voices", "/deployment", "/phone", "/token"}


@functools.cache
def tool_catalog() -> list[dict]:
    """Return every use-case tool with the JSON schema that LiveKit sends to the LLM, for the Tools tab."""
    from livekit.agents import llm

    schemas = [
        llm.utils.build_strict_openai_schema(llm.function_tool(fn))["function"] for fn in scenario_tools().values()
    ]
    return [
        {
            "name": schema["name"],
            "description": schema.get("description", "").strip(),
            "parameters": schema["parameters"],
        }
        for schema in schemas
    ]


def client_config() -> dict:
    """Return the LiveKit address and the custom endpoint form. Public, because the page needs it before a login."""
    return {
        "ws_url": public_ws_url(),
        "custom_endpoints": custom_endpoints_enabled(),
        "custom_fields": {
            slot: {"fields": list(fields), "required": list(CUSTOM_REQUIRED[slot])}
            for slot, fields in CUSTOM_FIELDS.items()
        },
    }


def state_path(name: str) -> Path:
    """Return a file that the agent worker publishes into the state volume."""
    return Path(os.environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR)) / name


def service_state() -> dict:
    """Return the services and voices that the agent worker publishes."""
    return read_service_state(state_path(SERVICES_FILE))


def send_json(
    handler: SimpleHTTPRequestHandler, payload: dict | list, status: int = 200, headers: dict | None = None
) -> None:
    """Send a JSON response."""
    body = json.dumps(payload).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for name, value in (headers or {}).items():
        handler.send_header(name, value)
    handler.end_headers()
    handler.wfile.write(body)


class Handler(SimpleHTTPRequestHandler):
    """The built client from `clients/livekit/dist`, plus the login, data, and token endpoints."""

    timeout = 15  # seconds a connection may stay idle, so a stalled request cannot hold a thread forever

    def __init__(self, *args, **kwargs):
        """Serve static files from the built client."""
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def end_headers(self):
        """Disable caching, so a browser that used another app on this origin never serves its stale scripts."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def authenticated(self) -> bool:
        """Return True when no password is set, or the request carries a valid session cookie."""
        if not login_required():
            return True
        jar = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        return SESSION_COOKIE in jar and valid_session(jar[SESSION_COOKIE].value)

    def body_length(self, limit: int) -> int | None:
        """Return the request's Content-Length, or None after answering a missing, bad, or too large one."""
        # This server does not read chunked bodies.
        if self.headers.get("Content-Length") is None:
            send_json(self, {"error": "Content-Length required"}, 411)
            return None
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            length = -1
        if length < 0:
            send_json(self, {"error": "bad Content-Length"}, 400)
            return None
        if length > limit:
            send_json(self, {"error": "request too large"}, 413)
            return None
        return length

    def do_POST(self):
        """Handle /login and /token."""
        path = urlsplit(self.path).path
        if path == "/login":
            return self.login()
        if path == "/token":
            return self.token()
        return self.send_error(404)

    def login(self):
        """A right username and password set the session cookie."""
        length = self.body_length(MAX_LOGIN_BODY)
        if length is None:
            return
        key = client_key(self.client_address[0], self.headers.get("X-Forwarded-For"))
        ticket = reserve_attempt(key)
        if ticket is None:
            return send_json(self, {"error": "too many attempts, wait a minute"}, 429)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            username, password = body.get("username"), body.get("password")
        except TimeoutError:  # the client promised a body and never sent it. Give the attempt back and hang up
            release_attempt(key, ticket)
            self.close_connection = True
            return
        except (ValueError, AttributeError):
            username = password = None
        if not credentials_ok(username, password):
            time.sleep(FAILED_LOGIN_DELAY)
            return send_json(self, {"error": "wrong username or password"}, 401)
        release_attempt(key, ticket)
        cookie = f"{SESSION_COOKIE}={make_session()}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}"
        if (
            self.headers.get("X-Forwarded-Proto", "").lower() == "https"
        ):  # behind an HTTPS proxy, keep it off plain HTTP
            cookie += "; Secure"
        send_json(self, {"ok": True}, headers={"Set-Cookie": cookie})

    def token(self):
        """Mint a join token for a new room, with the validated session choices as participant attributes."""
        if not self.authenticated():
            return send_json(self, {"error": "login required"}, 401)
        length = self.body_length(MAX_TOKEN_BODY)
        if length is None:
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except TimeoutError:
            self.close_connection = True
            return
        except (ValueError, RecursionError):
            body = None
        if not isinstance(body, dict):
            return send_json(self, {"error": "the body must be a JSON object"}, 400)
        state = service_state()
        if not state["services"]:
            return send_json(self, {"error": "the agent is still starting. Try again in a few seconds."}, 503)
        try:
            attributes = session_attributes(body, state)
        except ValueError as exc:
            return send_json(self, {"error": str(exc)}, 400)
        # A new room each time, so the agent is dispatched per session.
        room = f"voice-{uuid.uuid4().hex[:8]}"
        grants = api.VideoGrants(room_join=True, room=room)
        token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(f"user-{uuid.uuid4().hex[:6]}")
            .with_attributes(attributes)
            .with_grants(grants)
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )
        send_json(self, {"token": token, "room": room})

    def do_GET(self):
        """Serve the data endpoints, and static files for everything else."""
        url = urlsplit(self.path)
        if url.path == "/auth/status":
            return send_json(
                self,
                {
                    "required": login_required(),
                    "username": bool(ui_username()),
                    "authenticated": self.authenticated(),
                },
            )
        if url.path == "/client-config":
            return send_json(self, client_config())
        if url.path in PROTECTED and not self.authenticated():
            return send_json(self, {"error": "login required"}, 401)
        if url.path == "/services":
            return send_json(self, service_state()["services"])
        if url.path == "/voices":
            return send_json(self, voices_of(service_state(), parse_qs(url.query).get("tts", [""])[0]))
        if url.path == "/prompts":
            return send_json(self, scenario_catalog())
        if url.path == "/tools":
            return send_json(self, tool_catalog())
        if url.path == "/phone":
            return send_json(self, phone_info())
        if url.path == "/deployment":
            return send_json(self, read_json(state_path(DEPLOYMENT_FILE)) or {})
        return super().do_GET()


def main() -> None:
    """Run the UI server. It listens on loopback unless `LIVEKIT_UI_HOST` says otherwise.

    The container sets `LIVEKIT_UI_HOST=0.0.0.0` so that Docker can publish the port, and Compose publishes it on
    `127.0.0.1` by default. A host run therefore never exposes the unauthenticated token endpoint by accident.
    """
    apply_to_environ()
    host = os.environ.get("LIVEKIT_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("LIVEKIT_UI_PORT", "3000"))
    print(f"UI on http://{host}:{port}", flush=True)
    if ui_username() and not ui_password():
        print(f"warning: {USERNAME_ENV} is set without {PASSWORD_ENV}, so no login can succeed. Set both.", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()  # noqa: S104 - the container binds 0.0.0.0 by design


if __name__ == "__main__":
    main()
