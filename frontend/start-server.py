#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Start server script for distroless container."""

import os
import signal
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Get transport mode from environment variable
transport = os.environ.get("TRANSPORT", "WEBRTC")
directory = "websocket_static" if transport == "WEBSOCKET" else "webrtc_static"

print(f"Starting {transport} UI server...", flush=True)
print(f"Serving directory: {directory}", flush=True)

# Change to the appropriate directory
os.chdir(directory)

# Create and start the HTTP server
port = 8000
server_address = ("", port)
handler = SimpleHTTPRequestHandler

httpd = HTTPServer(server_address, handler)


def shutdown_handler(signum, frame):
    """Gracefully shutdown the server on SIGTERM or SIGINT."""
    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name}, shutting down server...", flush=True)
    httpd.shutdown()
    sys.exit(0)


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

print(f"Server running at http://0.0.0.0:{port}/", flush=True)
httpd.serve_forever()
