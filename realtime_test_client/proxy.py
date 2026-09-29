#!/usr/bin/env python3
"""Minimal local test client for the Realtime gateway.

Serves a one-page web UI and relays its plain-ws connection to the gateway's
TLS WebSocket, holding the master key server-side.

Why a proxy rather than connecting the browser straight to the gateway:

  * browsers cannot set an Authorization header on a WebSocket, and
  * the browser fallback (an ``ek_`` client secret offered as a subprotocol)
    encodes the whole bound session, which for this profile exceeds the
    16,384-char cap -- ``generic_talker`` alone is 25,823 chars. The gateway
    answers "The bound Realtime session is too large for a WebSocket client
    secret".

The proxy also terminates the gateway's self-signed TLS, so the browser sees
plain http/ws on localhost and raises no certificate warnings.

    python realtime_test_client/proxy.py            # then open http://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import ssl
import urllib.parse

import websockets
from websockets.asyncio.server import serve as ws_serve

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = "nvidia/nemotron-realtime-generic-frontend-backend"


def _read_env(path: pathlib.Path, name: str) -> str:
    if not path.is_file():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


async def _relay(src, dst) -> None:
    try:
        async for message in src:
            await dst.send(message)
    except websockets.exceptions.ConnectionClosed:
        pass


async def _handle(browser, args, api_key: str) -> None:
    model = urllib.parse.quote(args.model, safe="")
    url = f"{args.upstream}/v1/realtime?model={model}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    print(f"[proxy] browser connected -> {url}")
    try:
        async with websockets.connect(
            url,
            additional_headers=headers,
            subprotocols=["realtime"],
            ssl=ctx if url.startswith("wss://") else None,
            open_timeout=120,
            max_size=64 * 1024 * 1024,
            ping_interval=None,
        ) as upstream:
            await asyncio.gather(_relay(browser, upstream), _relay(upstream, browser))
    except Exception as exc:  # surface the reason in the browser console
        print(f"[proxy] upstream error: {type(exc).__name__}: {exc}")
        await browser.close(code=1011, reason=f"{type(exc).__name__}")
    finally:
        print("[proxy] session closed")


def _http_handler(connection, request):
    """Serve the single page for ordinary GETs; let /ws upgrade."""
    if request.path.startswith("/ws"):
        return None
    body = (HERE / "index.html").read_text(encoding="utf-8")
    response = connection.respond(200, body)
    # respond() defaults to text/plain, which makes the browser render the
    # source as text. Headers is a multidict, so assignment appends -- the
    # existing value has to be removed first or the browser sees both.
    del response.headers["Content-Type"]
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    response.headers["Cache-Control"] = "no-store"
    return response


async def main() -> None:
    """Serve the test UI and relay its WebSocket to the Realtime gateway."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--upstream", default="wss://localhost:7860")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--env-file", default=str(HERE.parent / ".env"))
    args = ap.parse_args()

    api_key = os.getenv("REALTIME_API_KEY") or _read_env(pathlib.Path(args.env_file), "REALTIME_API_KEY")
    print(f"[proxy] upstream : {args.upstream}")
    print(f"[proxy] model    : {args.model}")
    print(f"[proxy] auth     : {'bearer (' + str(len(api_key)) + ' chars)' if api_key else 'none'}")
    print(f"[proxy] open     : http://localhost:{args.port}")

    async with ws_serve(
        lambda ws: _handle(ws, args, api_key),
        "localhost",
        args.port,
        process_request=_http_handler,
        max_size=64 * 1024 * 1024,
        ping_interval=None,
    ):
        await asyncio.get_running_loop().create_future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[proxy] stopped")
