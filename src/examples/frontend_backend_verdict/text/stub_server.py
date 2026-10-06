# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted OpenAI-compatible endpoint, for credential-free smoke tests.

Exercises the real HTTP client, the real wire format and the real turn loop
without a key and without spend. It answers exactly one scenario — "where is my
order 5512?" — by looking at what the caller sent:

* a request whose tool list contains ``call_backend`` is the frontend asking, so
  it gets a delegation call back;
* a request carrying a ``role: "tool"`` message is the backend holding a result,
  so it gets the final answer;
* anything else is the backend starting work, so it gets a ``get_order`` call.

Run it alongside the REPL::

    PYTHONPATH=src uv run python -m examples.frontend_backend_verdict.text.stub_server &
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

DEFAULT_PORT = 8799


def scripted_message(body: dict[str, Any]) -> dict[str, Any]:
    """Return the assistant message this stub replies with for ``body``."""
    tool_names = [tool["function"]["name"] for tool in body.get("tools") or []]
    messages = body.get("messages") or []
    if "call_backend" in tool_names:
        return _tool_call_message(
            "call_fe1",
            "call_backend",
            {
                "query": "The user asks for the delivery status of order 5512.",
                "filler_text": "Let me check that.",
            },
        )
    tool_payloads = [message.get("content") for message in messages if message.get("role") == "tool"]
    if tool_payloads:
        status = json.loads(tool_payloads[-1] or "{}").get("status", "unknown")
        return {"role": "assistant", "content": f"Order 5512 is {status}."}
    return _tool_call_message("call_be1", "get_order", {"order_id": "5512"})


def _tool_call_message(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        ],
    }


class StubHandler(BaseHTTPRequestHandler):
    """Serves ``POST /v1/chat/completions`` and nothing else."""

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler naming
        """Answer one chat-completion request."""
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        payload = {
            "id": "stub-completion",
            "object": "chat.completion",
            "model": body.get("model", "stub"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": scripted_message(body)}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - signature is fixed
        """Silence the default per-request logging."""


def main() -> None:
    """Serve the stub until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    print(f"stub endpoint on http://127.0.0.1:{args.port}/v1 (ctrl-c to stop)", flush=True)
    HTTPServer(("127.0.0.1", args.port), StubHandler).serve_forever()


if __name__ == "__main__":
    main()
