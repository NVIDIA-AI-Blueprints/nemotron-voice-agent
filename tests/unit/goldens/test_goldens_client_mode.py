# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

from tests.unit.goldens.harness import GoldenCase, decision

# Live client mode, asked for by name, so the golden does not depend on what an omitted block means.
CLIENT_SESSION = {
    "model": "live-1",
    "instructions": "You are a kiosk voice assistant. Delegate every request.",
    "delegation": {"type": "client"},
}


def append(kind, content, event_id, delegation_id):
    """A ``session.<kind>.append`` (thinking or commentary) for the client delegation ``delegation_id``."""
    return {"type": f"session.{kind}.append", "event_id": event_id, "delegation_id": delegation_id, "content": content}


class ClientModeGoldens(GoldenCase):
    """Live client mode: the server only announces each delegation, and the client app answers with appends."""

    async def acknowledged(self, kind, event_id):
        """Feed caller audio until the append's acknowledgment is released, and return it."""
        self.feed_silence(2)
        return await self.wait_for(f"session.{kind}.appended", where=lambda e: e["client_event_id"] == event_id)

    async def announced(self, after):
        """The next ``session.delegation.created`` from event ``after`` on: its delegation id."""
        created = await self.wait_for("session.delegation.created", after=after)
        return created["delegation"]["id"]

    async def test_g10_client_mode(self):
        """G10: client mode, two delegations answered by the client app with appends.

        The session asks for client delegation. The caller asks about today's weather; the talker delegates with a
        lead-in, and the server announces the delegation with target "client" and no response_id, and runs no
        backend. While it is unanswered, the caller asks about tomorrow: the talker's request shows no open work, and
        it delegates again with its own lead-in. The client app sends quiet progress for the first delegation as a
        thinking append (nothing is said), then the first result as commentary: the writer is told the first lead-in
        was already said, and the cascade speaks it. The app answers the second delegation the same way, with the
        second lead-in as already said. A second commentary for the first delegation is spoken with no "already told"
        note. Each append is acknowledged on the input clock. No backend request is made.
        """
        await self.serve(
            CLIENT_SESSION,
            frontend=[
                decision("delegate", "One moment."),  # the talker, on the first question
                decision("delegate", "Let me look."),  # the talker, on the second question
                decision("speak", "It is sunny and 72 degrees today."),  # the writer, first result
                decision("speak", "Tomorrow brings rain and 60 degrees."),  # the writer, second result
                decision("speak", "A correction: it is 73 degrees today."),  # the writer, the later update
            ],
            backend=[],
        )
        after = len(self.ws.from_server)
        await self.say("What is the weather today?")
        today = await self.announced(after)
        await self.wait_quiet()
        after = len(self.ws.from_server)
        await self.say("And tomorrow?")
        tomorrow = await self.announced(after)
        await self.wait_quiet()
        self.send(append("thinking", "Checking today's forecast.", "progress_today", today))
        await self.acknowledged("thinking", "progress_today")
        await self.wait_quiet()
        self.send(append("commentary", "Today: sunny, 72 degrees.", "result_today", today))
        await self.wait_quiet()
        await self.acknowledged("commentary", "result_today")
        self.send(append("commentary", "Tomorrow: rain, 60 degrees.", "result_tomorrow", tomorrow))
        await self.wait_quiet()
        await self.acknowledged("commentary", "result_tomorrow")
        self.send(append("commentary", "Correction: 73 degrees today.", "update_today", today))
        await self.wait_quiet()
        await self.acknowledged("commentary", "update_today")
        await self.wait_quiet()
        await self.close()
        self.assert_golden("G10")
