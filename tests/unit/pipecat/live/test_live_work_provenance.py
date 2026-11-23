# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D107

"""Private host ownership survives replay without entering the public Live wire."""

import json
import unittest
from copy import deepcopy
from unittest import mock

from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import ProtocolError
from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation.passthrough import (
    BackendEventRelay,
    ClientRoutes,
    ServerDelegation,
    sanitize_response_event,
)
from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation.retention import DUPLICATE, DeliveryMailbox
from nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation.sender import Request
from tests.unit.pipecat.live_helpers import Stream, response

PRIVATE = "_nva_work"
ANSWER = {
    "id": "answer-b",
    "type": "message",
    "status": "completed",
    "role": "assistant",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": "The second result is ready."}],
}
CALL = {
    "id": "item-a",
    "type": "function_call",
    "call_id": "original-call-a",
    "name": "read_options",
    "arguments": "{}",
}


def provenance(response_id="response-1"):
    """Use separate canonical-owner and outer transport identities, as legacy late frames do."""
    return {
        "version": 1,
        "conversation_ref": "opaque-conversation",
        "response_id": response_id,
        "owner_ref": "canonical-owner",
        "owner_revision": 0,
        "bindings": [{"branch_ref": "branch-b", "owner_ref": "canonical-owner", "revision": 0}],
        "obligations": [
            {
                "obligation_ref": "input-b",
                "branch_ref": "branch-b",
                "state": "active",
                "delivery_pending": False,
                "publication_pending": True,
                "terminal_outcome": None,
                "stop_outcome": None,
                "to_owner_ref": None,
            }
        ],
        "item_attribution": None,
        "owner_settlements": [],
    }


def tagged(events):
    """Copy actual response-shaped events, retaining private facts outside their output items."""
    return [{**deepcopy(event), PRIVATE: provenance()} for event in events]


class View:
    def __init__(self):
        self.events = []

    async def emit_paced(self, kind, *, before_publish=None, **fields):
        if before_publish is not None and not before_publish():
            return False
        self.events.append({"type": kind, **deepcopy(fields)})
        return True


class Core:
    def __init__(self):
        self.pending = []

    async def set_attachment(self, conversation_id, attached):
        return {"attached": attached, "pending_calls": list(self.pending)}


class Listener:
    def __init__(self):
        self.seen = []
        self.restored = []

    def on_delegation_event(self, delegation_id, event):
        self.seen.append(("event", delegation_id, deepcopy(event)))

    def on_delegation_lost(self, delegation_id):
        self.seen.append(("lost", delegation_id))

    def restore_work_provenance(self, delegation_id, event):
        self.restored.append((delegation_id, deepcopy(event)))
        self.seen.append(("provenance", delegation_id, event["type"]))
        # A consumer cannot corrupt the canonical journal while taking a private snapshot.
        event[PRIVATE]["owner_ref"] = "consumer-local-copy"

    def restore_final(self, delegation_id):
        self.seen.append(("final", delegation_id))

    def restore_terminal(self, delegation_id, event):
        self.seen.append(("terminal", delegation_id, event["type"]))


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.core, self.view, self.listener = Core(), View(), Listener()
        self.delegation = ServerDelegation(self.core)
        self.delegation.conversation = "conversation"
        self.delegation.view = self.view
        self.delegation.routes = ClientRoutes(lambda *args, **kwargs: None)
        self.delegation.relay = BackendEventRelay(self.view, self.delegation.routes)
        self.delegation.retain_attachments()
        self.delegation.listen(self.listener)

    async def test_live_read_shares_private_facts_and_only_relays_each_public_event_once(self):
        request = Request([], "transport-different-from-core", 0, False)
        request.returned.set()
        business = json.dumps({PRIVATE: {"owner_ref": "business-not-host"}, "value": 3})
        result = {**ANSWER, "content": [{"type": "output_text", "text": business}]}
        ordinary = response("response-1", result)
        created, answer, terminal = tagged(ordinary)
        repeated = {**deepcopy(answer), "sequence_number": 3}
        await self.delegation._read(request, Stream([created, answer, repeated, terminal]), None)
        self.assertEqual([row[2] for row in self.listener.seen], [created, answer, terminal])
        entries = [entry.event for entry in self.delegation.mailbox.entries]
        self.assertEqual([row["type"] for row in entries], [row["type"] for row in (created, answer, terminal)])
        self.assertEqual([row[PRIVATE] for row in entries], [row[PRIVATE] for row in (created, answer, terminal)])
        self.assertEqual([row["event"] for row in self.view.events], [sanitize_response_event(row) for row in ordinary])
        self.assertEqual(sum(row["event"].get("item") == result for row in self.view.events), 1)
        self.assertTrue(all(PRIVATE not in row["event"] for row in self.view.events))
        self.assertTrue(all(row["delegation_id"] == request.delegation_id for row in self.view.events))

        before = deepcopy(entries)
        self.listener.seen.clear()
        self.view.events.clear()
        await self.delegation.resume_attachment()
        self.assertEqual(len(self.listener.restored), 3)
        self.assertEqual([event for _, event in self.listener.restored], before)
        kinds = [row[0] for row in self.listener.seen]
        self.assertLess(kinds.index("provenance"), kinds.index("event"))
        final_index, terminal_index = kinds.index("final"), kinds.index("terminal")
        self.assertEqual(kinds[final_index - 1], "provenance")
        self.assertEqual(kinds[terminal_index - 1], "provenance")
        self.assertFalse(any(row[0] == "event" and row[2].get("item") == result for row in self.listener.seen))
        self.assertEqual([entry.event for entry in self.delegation.mailbox.entries], before)
        self.assertEqual(self.view.events, [])

    async def test_no_private_restore_callback_is_required_of_a_legacy_listener(self):
        self.listener.restore_work_provenance = None
        for event in tagged(response("response-1", ANSWER)):
            self.delegation.mailbox.record("transport", event).forwarded = True
        await self.delegation.resume_attachment()
        self.assertIn(("final", "transport"), self.listener.seen)
        self.assertEqual(self.listener.restored, [])

    async def test_never_forwarded_events_keep_their_first_normal_metadata_processing(self):
        events = tagged(response("response-1", ANSWER))
        for event in events:
            self.delegation.mailbox.record("transport", event)
        retained = deepcopy([entry.event for entry in self.delegation.mailbox.entries])
        await self.delegation.resume_attachment()
        self.assertEqual(self.listener.restored, [], "new events must not first be classified as quiet old replay")
        self.assertEqual([row[2] for row in self.listener.seen if row[0] == "event"], retained)

    async def test_a_revalidated_never_emitted_call_strips_only_its_public_copy(self):
        raw = tagged(response("response-1", CALL))[1]
        self.delegation.mailbox.record("transport", raw)
        self.delegation.relay.remember("transport", raw)
        self.core.pending = [CALL["call_id"]]
        before = deepcopy(self.delegation.mailbox.entries[0].event)
        await self.delegation.relay_pending_calls({})
        await self.delegation.relay_pending_calls({})
        self.assertEqual(len(self.view.events), 1)
        self.assertNotIn(PRIVATE, self.view.events[0]["event"])
        self.assertEqual(self.view.events[0]["event"]["item"], CALL)
        self.assertEqual(self.delegation.mailbox.entries[0].event, before)
        self.assertEqual(self.delegation.mailbox.emitted, {CALL["call_id"]})


class JournalTests(unittest.TestCase):
    def test_private_metadata_consumes_the_existing_journal_byte_budget(self):
        raw = tagged(response("response-1", ANSWER))[1]
        size = len(json.dumps(raw, ensure_ascii=False).encode())
        mailbox = DeliveryMailbox()
        with (
            mock.patch(
                "nemotron_voice_agent.pipecat.runtime.gateways.live.server_delegation.retention.MAX_BYTES", size - 1
            ),
            self.assertRaises(ProtocolError),
        ):
            mailbox.record("transport", raw)
        self.assertEqual((mailbox.entries, mailbox.bytes), ([], 0))
        self.assertTrue(mailbox.exhausted)
        mailbox.mark_emitted("after-overflow")
        self.assertNotIn("after-overflow", mailbox.emitted)
        with self.assertRaises(ProtocolError):
            mailbox.record("transport", raw)
        mailbox = DeliveryMailbox()
        entry = mailbox.record("transport", raw)
        self.assertEqual(mailbox.bytes, size)
        self.assertEqual(entry.event[PRIVATE], raw[PRIVATE])
        self.assertNotIn(PRIVATE, json.dumps(mailbox.snapshot()))
        self.assertEqual(mailbox.record("transport", deepcopy(raw)), DUPLICATE)
        self.assertEqual(len(mailbox.entries), 1)

        revised = deepcopy(raw)
        revised[PRIVATE]["owner_revision"] += 1
        self.assertEqual(mailbox.record("transport", revised), DUPLICATE)

        header = tagged(response("response-1", ANSWER))[0]
        mailbox.record("transport", header)
        revised_header = deepcopy(header)
        revised_header[PRIVATE]["owner_revision"] += 1
        updated = mailbox.record("transport", revised_header)
        self.assertEqual(updated.event[PRIVATE], revised_header[PRIVATE])

        legacy = response("response-1", ANSWER)[1]
        self.assertNotEqual(mailbox.record("legacy", legacy, response_id="one"), DUPLICATE)
        self.assertNotEqual(mailbox.record("legacy", legacy, response_id="two"), DUPLICATE)
        self.assertEqual(mailbox.record("legacy", legacy, response_id="two"), DUPLICATE)
