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
from urllib.parse import urlsplit

from livekit import api

from .credentials import apply_to_environ

PUBLIC = Path(__file__).with_name("ui")


class Handler(SimpleHTTPRequestHandler):
    """Static files from ui/, plus the token endpoint."""

    def __init__(self, *args, **kwargs):
        """Serve static files from the ui directory."""
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def do_GET(self):
        """Route /token to the token endpoint and everything else to static files."""
        if urlsplit(self.path).path != "/token":
            return super().do_GET()
        room = f"voice-{uuid.uuid4().hex[:8]}"  # a new room each time, so the agent dispatches per session
        token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(f"user-{uuid.uuid4().hex[:6]}")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .with_ttl(timedelta(hours=1))
            .to_jwt()
        )
        body = json.dumps({"token": token, "room": room}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """Run the UI server. Compose publishes the port on loopback by default."""
    apply_to_environ()
    port = int(os.environ.get("LIVEKIT_UI_PORT", "3000"))
    print(f"UI on http://0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
