# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""Definite non-admission and optional reconnect catalogs."""

import asyncio
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from examples.frontend_backend_live import factory
from examples.frontend_backend_live.config_manager.schema import ExampleConfig, ModelEndpoint
from gateways.live.protocol import ProtocolError
from tests.unit.core_helpers import EXAMPLE_SETTINGS, PROMPTS
from tests.unit.delegation_core.helpers import ask
from tests.unit.live import test_live_reconnect as fixtures


class ReconnectRepairTests(fixtures.ReconnectCase):
    async def command(self, message):
        await self.session.handle(message, self.session.socket.connection)
        await self.session.fanout.drain()

    async def test_invalid_item_append_and_create_do_not_reserve_an_operation(self):
        await self.start()
        for kind, message in [
            ("item", {"type": "response.item.create", "item": "invalid"}),
            ("append", {"type": "session.instructions.append", "content": "no scope"}),
            ("create", {"type": "response.create", "response": {}}),
        ]:
            with self.subTest(kind=kind):
                await self.command({**message, "operation_id": kind})
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

    async def test_only_plain_text_placeholders_qualify_before_any_parts_start(self):
        from examples.frontend_backend_live.adapter.adapter import CascadeAdapter

        prompts = factory.load_prompts()
        endpoint = ModelEndpoint(model="dummy", base_url="https://dummy.invalid/v1", api_key="", extra_params={})
        invalid_notes = (
            None,
            "",
            "No uncertainty placeholder",
            "Unclosed {text",
            "{unknown} {text}",
            "Custom [{text:{text}}]",
            "Converted {text!r}",
            "Truncated {text:.5}",
        )
        custom_failure = object()
        for note in (*invalid_notes, custom_failure, "Escaped {{literal}} with [{text}]"):
            with self.subTest(note=note):
                allowed = note not in invalid_notes and note is not custom_failure
                parsed = replace(prompts, cascade=prompts.cascade.model_copy(update={"attachment_note": note}))

                def fingerprint(adapter, parsed=parsed, custom=note is custom_failure):
                    if custom:
                        raise ValueError("private-fingerprint-detail")
                    return CascadeAdapter.attachment_fingerprint(
                        SimpleNamespace(
                            prompts=parsed,
                            frontend_endpoint=endpoint,
                            live_config=SimpleNamespace(model_dump=lambda: {}),
                            instructions="Instructions.",
                            lookup_tools=SimpleNamespace(schemas=[]),
                            config=adapter.config,
                        )
                    )

                with patch.object(fixtures.RestorableAdapter, "attachment_fingerprint", fingerprint):
                    ws = await self.socket(
                        {"type": "session.start", "session": fixtures.CONFIG, "reconnect": {"version": 1}}
                    )
                    async with asyncio.timeout(2):
                        while not (
                            outcomes := [
                                row for row in ws.from_server if row["type"] in {"error", "session.reconnect.ready"}
                            ]
                        ):
                            await asyncio.sleep(0)
                outcome = outcomes[0]
                self.assertEqual(outcome["type"], "session.reconnect.ready" if allowed else "error")
                self.assertEqual(self.adapters[-1].started, allowed)
                if not allowed:
                    await asyncio.wait_for(self.runners[-1], 2)
                    self.assertEqual(outcome["error"]["code"], "resume_unavailable")
                    self.assertEqual(outcome["error"]["message"], "The frontend reconnect snapshot is unavailable.")
                    self.assertNotIn("private-fingerprint-detail", str(outcome))
                    self.assertNotIn("attachment_note", str(outcome))
                    self.assertNotIn("create", self.core.log)
                    self.assertEqual(self.gateway.resume_leases, {})


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
