# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103, D107

"""Tests that each model gets its own endpoint and that an unlisted model is reported or falls back."""

import unittest
from unittest.mock import patch

from delegation_core.backend.llm import LlmSettings, LlmSource
from delegation_core.config import ModelEndpoint
from tests.unit.contracts.backend import call_round, text_round
from tests.unit.delegation_core.backend.llm.doubles import IMTask, ScriptedProvider, drive

DEFAULT = ModelEndpoint("test/model", "http://default:1/v1", extra_params={"temperature": 0.2})
OTHER = ModelEndpoint("other/model", "http://other:2/v1", extra_params={"extra_body": {"reasoning": "off"}})
CATALOG = {"test/model": DEFAULT, "other/model": OTHER}


def lookup(model):
    return DEFAULT if model is None else CATALOG[model]


def built(policy="report", rounds=None):
    made = []

    def make(endpoint):
        provider = ScriptedProvider(*(rounds or (text_round("Done."), text_round("Done."))), endpoint=endpoint)
        made.append(provider)
        return provider

    source = LlmSource(
        LlmSettings.parse({"unlisted_models": policy}), endpoints=lookup, tools=lambda: [], make_provider=make
    )
    return source, made


class ModelTask(IMTask):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def request(self):
        return {**super().request(), "model": self.model}


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_models_two_endpoints_one_provider_each(self):
        source, made = built()
        session = source.new()
        for model in ("test/model", "other/model", "other/model"):
            task = ModelTask(model)
            await drive(session.start(task), task)
        self.assertEqual([provider.endpoint for provider in made], [DEFAULT, OTHER])
        self.assertEqual([len(provider.payloads) for provider in made], [1, 2])
        self.assertEqual({payload["model"] for payload in made[1].payloads}, {"other/model"})

    async def test_an_unlisted_model_is_reported(self):
        source, made = built()
        await source.check_model("other/model")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await source.check_model("nvidia/unlisted")
        task = ModelTask("nvidia/unlisted")
        with self.assertRaisesRegex(ValueError, "not in the model catalog"):
            await drive(source.new().start(task), task)
        self.assertEqual(made, [])

    async def test_an_unlisted_model_falls_back_under_the_benchmark_setting(self):
        with patch("delegation_core.backend.llm.settings.logger") as logger:
            source, made = built("fallback")
            await source.check_model("nvidia/unlisted")
            task = ModelTask("nvidia/unlisted")
            await drive(source.new().start(task), task)
        lines = [call.args[0] for call in logger.warning.call_args_list]
        self.assertEqual(
            made[0].endpoint, ModelEndpoint("nvidia/unlisted", DEFAULT.base_url, extra_params=DEFAULT.extra_params)
        )
        self.assertEqual(made[0].payloads[0]["model"], "nvidia/unlisted")
        self.assertTrue(any("nvidia/unlisted" in line for line in lines))

    async def test_a_session_on_the_default_model_needs_no_catalog_entry_by_name(self):
        source, made = built()
        task = ModelTask(None)
        await drive(source.new().start(task), task)
        self.assertEqual(made[0].endpoint, DEFAULT)

    async def test_app_calls_keep_their_models_provider_across_resume(self):
        source, made = built(rounds=(call_round("get_menu", {}), text_round("Done.", "resp_2")))
        task = ModelTask("other/model")
        run = source.new().start(task)
        await drive(run, task)
        task.give_outputs([{"type": "function_call_output", "call_id": "call_1", "output": "{}"}])
        await run.resume()
        await drive(run, task)
        self.assertEqual((len(made), made[0].endpoint, len(made[0].payloads)), (1, OTHER, 2))


if __name__ == "__main__":
    unittest.main()
