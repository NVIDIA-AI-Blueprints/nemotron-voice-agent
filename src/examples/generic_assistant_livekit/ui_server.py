# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the web UI and mint a LiveKit join token at /token (a fresh room per connect)."""

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
from .voices import VOICES_FILE, choose_voice, read_voices_file

PUBLIC = Path(__file__).with_name("ui")


def voices_path() -> Path:
    """Return the voice list that the agent worker publishes into the state volume."""
    return Path(os.environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR)) / VOICES_FILE


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
        if url.path != "/token":
            return super().do_GET()
        listed = read_voices_file(voices_path())
        requested = parse_qs(url.query).get("voice", [None])[0]
        voice = choose_voice(requested, [v["id"] for v in listed["voices"]], "")
        room = f"voice-{uuid.uuid4().hex[:8]}"  # a new room each time, so the agent dispatches per session
        grants = api.VideoGrants(room_join=True, room=room)
        token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(f"user-{uuid.uuid4().hex[:6]}")
            .with_attributes({"voice": voice} if voice else {})  # the agent reads the chosen voice when it joins
            .with_grants(grants)
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )
        send_json(self, {"token": token, "room": room})


def main() -> None:
    """Run the UI server. Compose publishes the port on loopback by default."""
    apply_to_environ()
    port = int(os.environ.get("LIVEKIT_UI_PORT", "3000"))
    print(f"UI on http://0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
