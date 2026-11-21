# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Attachment-local failure callbacks cannot revoke a replacement's authority after waiting."""

import asyncio
from unittest import mock

from tests.unit.live import test_live_reconnect as fixtures


class AttachmentCallbackTests(fixtures.ReconnectCase):
    async def test_an_old_adapter_loss_delayed_before_the_lease_lock_cannot_detach_its_replacement(self):
        wire, authority = await self.start()
        session = self.session
        old_socket, old_view = session.socket, session.adapter_view
        lease = session.reconnect
        original = lease.lost
        entered, release = asyncio.Event(), asyncio.Event()
        first = True
        captured = []

        async def delayed(socket=None):
            nonlocal first
            if first:
                first = False
                captured.append(socket)
                entered.set()
                await release.wait()
            await original(socket)

        with mock.patch.object(lease, "lost", side_effect=delayed):
            before = set(session.tasks)
            old_view.request_close("connection_lost")
            callback = next(task for task in session.tasks - before if "adapter-connection-lost" in task.get_name())
            await asyncio.wait_for(entered.wait(), 2)
            await self.drop(wire)
            _, resumed = await self.resume(authority)
            replacement = session.socket
            self.assertEqual(resumed["reconnect"]["generation"], 2)
            release.set()
            await asyncio.wait_for(callback, 2)
            self.assertEqual(captured, [old_socket])
            self.assertIs(session.socket, replacement)
            self.assertTrue(session.adapter_view.is_active())
            self.assertEqual(session.attachment_state, "active")
            self.assertEqual(session.reconnect.state, "active")
            changes = [attached for _, attached in self.core.attachments]
            self.assertEqual(changes[0], False)
            self.assertTrue(all(changes[1:]))  # A canonical pending-call refresh may repeat True, never detach again.
