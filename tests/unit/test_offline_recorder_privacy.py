# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Protect offline speech/context artifacts without changing their contents."""

import asyncio
import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.generic.offline.models import AudioSource, SessionResult
from examples.generic.offline.protocol import encode_packet
from examples.generic.offline.recorder import write_private_text, write_source
from examples.generic.offline.websocket import serve_offline_session


class _Socket:
    def __init__(self, *, text=False):
        self.incoming = asyncio.Queue()
        self.sent = []
        pcm = b"\x01\x00" * 4
        self.feed(
            {
                "type": "offline.session.start",
                "protocol": "nva.offline.v1",
                "sample_rate": 16000,
                "channels": 1,
                "encoding": "pcm_s16le",
                "input_samples": 4,
                "completion": "text" if text else "full_session",
            }
        )
        self.feed(encode_packet(b"NVAI", 0, 0, pcm))
        self.feed(
            {
                "type": "offline.input.complete",
                "input_samples": 4,
                "input_packets": 1,
                "sha256": hashlib.sha256(pcm).hexdigest(),
            }
        )

    def feed(self, payload):
        field = "bytes" if isinstance(payload, bytes) else "text"
        value = payload if field == "bytes" else json.dumps(payload)
        self.incoming.put_nowait({"type": "websocket.receive", field: value})

    async def accept(self):
        pass

    async def receive(self):
        return await self.incoming.get()

    async def send_json(self, payload):
        self.sent.append(payload)

    async def send_bytes(self, payload):
        pass

    async def close(self, **kwargs):
        pass


class OfflineRecordingPrivacyTests(unittest.IsolatedAsyncioTestCase):
    """Check actual success, error and text-cancellation recording boundaries."""

    def setUp(self):
        """Exercise permissive ambient permissions without leaving the umask changed."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        previous = os.umask(0)
        self.addCleanup(os.umask, previous)

    def assert_private(self, root, expected):
        """Every session artifact is private from its first published name."""
        (session,) = Path(root).iterdir()
        self.assertEqual(stat.S_IMODE(session.stat().st_mode), 0o700)
        self.assertEqual({path.name for path in session.iterdir()}, expected)
        for path in session.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        return session

    async def test_full_session_artifacts_are_private_with_original_contents(self):
        """Both initial and final manifest writes retain private permissions."""

        class Runtime:
            configuration = {"model": "synthetic"}

            async def run(self, source):
                return SessionResult((), source.sample_count, (), ({"role": "user", "content": "synthetic"},), {})

        root = Path(self.directory.name) / "full"
        socket = _Socket()
        await serve_offline_session(socket, lambda start: Runtime(), root)
        session = self.assert_private(root, {"source.wav", "full_output.wav", "events.jsonl", "manifest.json"})
        manifest = json.loads((session / "manifest.json").read_text())
        self.assertEqual(manifest["messages"], [{"role": "user", "content": "synthetic"}])
        self.assertEqual(manifest["input_samples"], 4)
        self.assertEqual(socket.sent[-1]["type"], "offline.session.complete")

    async def test_failure_record_is_private_and_keeps_original_public_error(self):
        """Provider-independent failure evidence follows the same private writer."""

        class Runtime:
            async def run(self, source):
                raise RuntimeError("synthetic private diagnostic")

        root = Path(self.directory.name) / "error"
        socket = _Socket()
        await serve_offline_session(socket, lambda start: Runtime(), root)
        session = self.assert_private(root, {"source.wav", "failure.json"})
        self.assertEqual(
            json.loads((session / "failure.json").read_text()), {"status": "error", "code": "session_failed"}
        )
        self.assertEqual(socket.sent[-1]["code"], "session_failed")

    async def test_text_cancellation_record_is_private(self):
        """Text completion cancellation never relaxes source or diagnostic modes."""
        started = asyncio.Event()

        class Runtime:
            configuration = {}

            async def run(self, source, *, event_sink):
                started.set()
                await asyncio.Event().wait()

            async def aclose(self):
                pass

        root = Path(self.directory.name) / "text"
        socket = _Socket(text=True)
        task = asyncio.create_task(serve_offline_session(socket, lambda start: Runtime(), root))
        await asyncio.wait_for(started.wait(), timeout=2)
        socket.incoming.put_nowait({"type": "websocket.disconnect"})
        await asyncio.wait_for(task, timeout=2)
        session = self.assert_private(root, {"source.wav", "cancellation.json"})
        self.assertEqual(json.loads((session / "cancellation.json").read_text())["reason"], "client_disconnected")

    def test_atomic_replacement_does_not_follow_target_symlink_or_modify_victim(self):
        """A final-name symlink is replaced, never opened as a destination."""
        root = Path(self.directory.name)
        victim = root / "victim.txt"
        victim.write_text("unchanged")
        target = root / "record.txt"
        target.symlink_to(victim)
        write_private_text(target, "new artifact")
        self.assertFalse(target.is_symlink())
        self.assertEqual(target.read_text(), "new artifact")
        self.assertEqual(victim.read_text(), "unchanged")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    def test_failed_atomic_replacement_keeps_original_and_removes_private_temporary(self):
        """A failed commit leaves the previous complete artifact intact."""
        root = Path(self.directory.name)
        target = root / "record.txt"
        write_private_text(target, "original")
        with (
            patch("examples.generic.offline.recorder.os.replace", side_effect=OSError("synthetic disk failure")),
            self.assertRaises(OSError),
        ):
            write_private_text(target, "replacement")
        self.assertEqual(target.read_text(), "original")
        self.assertEqual(list(root.iterdir()), [target])
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)

    @unittest.skipUnless(hasattr(os, "O_NOFOLLOW"), "Directory symlink protection requires O_NOFOLLOW")
    def test_recording_rejects_symlink_session_directory(self):
        """Do not follow a final directory symlink into another recording location."""
        root = Path(self.directory.name)
        victim = root / "victim"
        victim.mkdir()
        link = root / "session"
        link.symlink_to(victim, target_is_directory=True)
        with self.assertRaises(OSError):
            write_source(link, AudioSource(b"\0\0"))
        self.assertEqual(list(victim.iterdir()), [])
