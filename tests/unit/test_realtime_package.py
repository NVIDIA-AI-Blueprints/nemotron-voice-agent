# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for the Realtime package import boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class RealtimePackageTests(unittest.TestCase):
    """Keep lightweight protocol types independent from the gateway runtime."""

    def test_importing_frames_does_not_load_transport_or_mcp(self) -> None:
        """Avoid injecting the full Realtime server into shared model imports."""
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        script = """
import json
import sys
import realtime.frames

print(json.dumps({
    "conversation": "realtime.conversation" in sys.modules,
    "gateway": "realtime.gateway" in sys.modules,
    "mcp": "realtime.mcp" in sys.modules,
    "response_config": "realtime.response_config" in sys.modules,
    "session": "realtime.session" in sys.modules,
    "tool_schema": "realtime.tool_schema" in sys.modules,
    "transport": "realtime.transport" in sys.modules,
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )

        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {
                "conversation": False,
                "gateway": False,
                "mcp": False,
                "response_config": False,
                "session": False,
                "tool_schema": False,
                "transport": False,
            },
        )

    def test_public_exports_load_lazily(self) -> None:
        """Load the current package entry points only when requested."""
        from realtime import RealtimeSessionController, handle_realtime_websocket

        self.assertEqual(RealtimeSessionController.__name__, "RealtimeSessionController")
        self.assertTrue(callable(handle_realtime_websocket))

    def test_importing_combined_server_does_not_load_realtime_codec_or_tool_stack(self) -> None:
        """Keep ordinary server startup independent from heavy Realtime modules."""
        repo_root = Path(__file__).resolve().parents[2]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(repo_root / "src") + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        script = """
import json
import sys
import server

print(json.dumps({
    "audio": "realtime.audio" in sys.modules,
    "gateway": "realtime.gateway" in sys.modules,
    "mcp": "realtime.mcp" in sys.modules or "mcp" in sys.modules,
    "transport": "realtime.transport" in sys.modules,
}))
"""

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(
            json.loads(completed.stdout.strip().splitlines()[-1]),
            {"audio": False, "gateway": False, "mcp": False, "transport": False},
        )


if __name__ == "__main__":
    unittest.main()
