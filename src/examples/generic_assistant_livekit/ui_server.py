# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the web UI and mint a LiveKit join token at /token (a fresh room per connect), with the chosen use case."""

from __future__ import annotations

import json
import os
import uuid
from datetime import timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from livekit import api

from .credentials import DEFAULT_STATE_DIR, apply_to_environ
from .scenarios import scenario_catalog
from .session import token_attributes
from .state import DEPLOYMENT_FILE, read_json
from .voices import VOICES_FILE, read_voices_file

PUBLIC = Path(__file__).with_name("ui")


def state_path(name: str) -> Path:
    """Return a file that the agent worker publishes into the state volume."""
    return Path(os.environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR)) / name


def voices_path() -> Path:
    """Return the voice list that the agent worker publishes."""
    return state_path(VOICES_FILE)


def send_json(handler: SimpleHTTPRequestHandler, payload: dict) -> None:
    """Send a JSON response."""
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(SimpleHTTPRequestHandler):
    """Static files from ui/, plus the token endpoint."""

    def __init__(self, *args, **kwargs):
        """Serve static files from the ui directory."""
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def end_headers(self):
        """Disable caching, so a browser that used another app on this origin never serves its stale scripts."""
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        """Serve /voices and /token, and static files for everything else."""
        url = urlsplit(self.path)
        if url.path == "/voices":
            return send_json(self, read_voices_file(voices_path()))
        if url.path == "/scenarios":
            return send_json(self, {"scenarios": scenario_catalog()})
        if url.path == "/deployment":
            return send_json(self, read_json(state_path(DEPLOYMENT_FILE)) or {})
        if url.path != "/token":
            return super().do_GET()
        attributes = token_attributes(parse_qs(url.query), [v["id"] for v in read_voices_file(voices_path())["voices"]])
        room = f"voice-{uuid.uuid4().hex[:8]}"  # a new room each time, so the agent dispatches per session
        grants = api.VideoGrants(room_join=True, room=room)
        token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(f"user-{uuid.uuid4().hex[:6]}")
            .with_attributes(attributes)  # the agent reads the use case and voice when it joins
            .with_grants(grants)
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )
        send_json(self, {"token": token, "room": room})


def main() -> None:
    """Run the UI server. It listens on loopback unless `LIVEKIT_UI_HOST` says otherwise.

    The container sets `LIVEKIT_UI_HOST=0.0.0.0` so that Docker can publish the port, and Compose publishes it on
    `127.0.0.1` by default. A host run therefore never exposes the unauthenticated token endpoint by accident.
    """
    apply_to_environ()
    host = os.environ.get("LIVEKIT_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("LIVEKIT_UI_PORT", "3000"))
    print(f"UI on http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()  # noqa: S104 - the container binds 0.0.0.0 by design


if __name__ == "__main__":
    main()
