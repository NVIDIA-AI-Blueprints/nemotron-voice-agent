# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Definite non-admission and optional reconnect catalogs."""

import asyncio
import os
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import yaml

from nemotron_voice_agent.pipecat.examples.frontend_backend_live import factory
from nemotron_voice_agent.pipecat.examples.frontend_backend_live.config_manager.schema import (
    ExampleConfig,
    ModelEndpoint,
)
from nemotron_voice_agent.pipecat.runtime.gateways.live.protocol import ProtocolError
from tests.unit.pipecat.core_helpers import EXAMPLE_SETTINGS, PROMPTS
from tests.unit.pipecat.delegation_core.helpers import ask
from tests.unit.pipecat.live import test_live_reconnect as fixtures


class ReconnectRepairTests(fixtures.ReconnectCase):
    async def command(self, message):
        await self.session.handle(message, self.session.socket.connection)
        await self.session.fanout.drain()

    async def test_invalid_item_append_and_create_do_not_reserve_an_operation(self):
        socket, _ = await self.start()
        for kind, message in [
            ("item", {"type": "response.item.create", "item": "invalid"}),
            ("append", {"type": "session.instructions.append", "content": "no scope"}),
            ("create", {"type": "response.create", "response": {}}),
        ]:
            with self.subTest(kind=kind):
                socket.send({**message, "event_id": kind, "operation_id": kind})
                error = await socket.wait_for("error", where=lambda event, kind=kind: event["client_event_id"] == kind)
                if kind == "create":
                    self.assertEqual(error["error"]["message"], "response.create accepts only type and event_id")
                self.assertNotIn(kind, self.session.reconnect.operations)
        self.assertEqual(self.adapters[0].appended, [])
        self.assertEqual(self.core.requests, [])

    async def test_possible_append_mutation_keeps_unknown_and_exact_retry_never_delivers_twice(self):
        await self.start()
        deliveries = []

        async def fail_after_mutation(*args):
            deliveries.append(args)
            raise ProtocolError("failure after delivery", "server_error")

        message = {
            "type": "session.instructions.append",
            "operation_id": "uncertain",
            "delegation_id": None,
            "content": "Update.",
        }
        with patch.object(self.adapters[0], "append_instructions", fail_after_mutation):
            await self.command(message)
            await self.command(message)
        self.assertEqual(len(deliveries), 1)
        self.assertEqual(self.session.reconnect.operations["uncertain"][1], "unknown")

    async def test_core_refusal_after_gateway_admission_does_not_erase_recorded_receipt(self):
        await self.start()
        await self.command(
            {
                "type": "response.item.create",
                "operation_id": "typed",
                "item": ask("Do it.")[0],
            }
        )
        self.core.fail["response"] = ValueError("Core rejected after gateway admission")
        create = {"type": "response.create", "operation_id": "queued"}
        await self.command(create)
        await asyncio.sleep(0)
        await self.command(create)
        self.assertEqual(len(self.core.requests), 1)
        self.assertEqual(self.session.reconnect.operations["queued"][1], "recorded")

    async def test_frontend_fingerprint_failure_is_redacted_before_startup(self):
        with patch.object(
            fixtures.RestorableAdapter,
            "attachment_fingerprint",
            side_effect=ValueError("private-fingerprint-detail about attachment_note"),
        ):
            socket = await self.socket(
                {"type": "session.start", "session": fixtures.CONFIG, "reconnect": {"version": 1}}
            )
            error = await socket.wait_for("error")
            await asyncio.wait_for(self.runners[-1], 2)
        self.assertEqual(error["error"]["code"], "resume_unavailable")
        self.assertEqual(error["error"]["message"], "The frontend reconnect snapshot is unavailable.")
        self.assertNotIn("private-fingerprint-detail", str(error))
        self.assertNotIn("attachment_note", str(error))
        self.assertFalse(self.adapters[-1].started)
        self.assertNotIn("create", self.core.log)
        self.assertEqual(self.gateway.resume_leases, {})

        _, authority = await self.start()
        self.assertTrue(self.adapters[-1].started)
        self.assertIsNotNone(authority["token"])


class OfflineCatalog:
    def service(self, slot):
        return {"server": "dummy.invalid:1"}

    def model(self, slot, overrides=None):
        return ModelEndpoint(model="dummy", base_url="https://dummy.invalid/v1", api_key="", extra_params={})


class ReconnectCatalogRepairTests(unittest.TestCase):
    def test_missing_reconnect_only_note_keeps_live_ui_and_real_root_constructible(self):
        catalog = deepcopy(PROMPTS)
        del catalog["agent_prompts"]["Cascade"]["attachment_note"]
        config = ExampleConfig.model_validate(deepcopy(EXAMPLE_SETTINGS))
        with TemporaryDirectory() as folder:
            path = Path(folder) / "custom.yaml"
            path.write_text(yaml.safe_dump(catalog, sort_keys=False))
            factory.root.cache_clear()
            try:
                with (
                    patch.dict(os.environ, {"PROMPT_FILE_PATH": str(path)}),
                    patch.object(factory.ExampleCatalog, "read", return_value=OfflineCatalog()),
                ):
                    self.assertIsNotNone(factory.core_settings(config.core))
                    self.assertIsNotNone(factory.ui_bot_core(config))
                    self.assertIsNone(factory.root().settings.live.model_dump().get("reconnect_grace_seconds"))
            finally:
                factory.root.cache_clear()
