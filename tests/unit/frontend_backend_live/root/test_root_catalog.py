# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103
"""The root's catalog: each slot's endpoint, its key, and the session's choices."""

import os
import unittest
from unittest.mock import patch

from examples.frontend_backend_live import pipeline
from examples.frontend_backend_live.factory import endpoint_lookup
from examples.frontend_backend_live.models.endpoints import ExampleCatalog, resolve_endpoint
from utils import NVCF_LLM_BASE_URL

ENTRY = {"model_id": "org/model", "base_url": "http://host:8000/v1"}
CATALOG = ExampleCatalog(
    {
        "llm": [{"model_id": "org/a", "base_url": "http://talker/v1", "extra_params": '{"temperature": 0}'}],
        "thinker-llm": [
            {"model_id": "org/b", "base_url": "http://thinker/v1"},
            {"model_id": "org/a", "base_url": "http://thinker/v1", "extra_params": '{"temperature": 0.7}'},
        ],
        "asr": [{"server": "asr:50052", "model": "parakeet"}],
    }
)


class EndpointTests(unittest.TestCase):
    def test_an_entry_uses_its_named_key_variable_alone_else_the_nvidia_key(self):
        named = {**ENTRY, "api_key_env": "ACME_API_KEY"}
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-secret", "ACME_API_KEY": "acme-key"}):
            self.assertEqual(resolve_endpoint(named, {}, slot="llm").api_key, "acme-key")
            self.assertEqual(resolve_endpoint(ENTRY, {}, slot="llm").api_key, "nvapi-secret")
            for empty in ("", "   "):
                with (
                    self.subTest(value=empty),
                    patch.dict(os.environ, {"ACME_API_KEY": empty}),
                    self.assertRaisesRegex(ValueError, "ACME_API_KEY is empty: the 'llm' slot's entry"),
                ):
                    resolve_endpoint(named, {}, slot="llm")

    def test_a_slot_with_no_model_names_itself_and_the_session_body_or_the_nvidia_cloud_fill_in(self):
        for entry in ({}, {"base_url": "http://host:8000/v1"}):
            with self.subTest(entry=entry), self.assertRaisesRegex(ValueError, "'thinker-llm' slot has no model"):
                resolve_endpoint(entry, {}, "thinker_", slot="thinker-llm")
        body = {"thinker_model_id": "org/custom", "thinker_base_url": "http://c/v1"}  # needs no catalog entry
        custom = resolve_endpoint({}, body, "thinker_", slot="thinker-llm")
        self.assertEqual((custom.model, custom.base_url), ("org/custom", "http://c/v1"))
        self.assertEqual(resolve_endpoint({"model_id": "org/model"}, {}, slot="llm").base_url, NVCF_LLM_BASE_URL)

    def test_an_endpoint_carries_its_entrys_services_key_which_names_the_talkers_style(self):
        listed = {**ENTRY, "id": "cloud:nemotron-lightning", "key": "nemotron-lightning"}  # as /api/services lists it
        self.assertEqual(resolve_endpoint(listed, {}, slot="llm").service_key, "nemotron-lightning")
        self.assertEqual(resolve_endpoint(ENTRY, {}, slot="llm").service_key, "")


class CatalogTests(unittest.TestCase):
    def test_a_slot_gives_its_default_entry_and_a_named_model_keeps_the_settings_of_the_slot_that_asks(self):
        self.assertEqual(CATALOG.model("thinker-llm").model, "org/b")
        self.assertEqual(CATALOG.model("llm", {"model_id": "org/custom"}).model, "org/custom")
        self.assertEqual(
            (CATALOG.service("asr")["server"], CATALOG.models("thinker-llm")), ("asr:50052", ["org/b", "org/a"])
        )
        with self.assertRaisesRegex(ValueError, "'tts' slot has no model"):
            CATALOG.model("tts")
        self.assertEqual(CATALOG.named("thinker-llm", "org/a").extra_params, {"temperature": 0.7})
        self.assertEqual(CATALOG.named("llm", "org/a").extra_params, {"temperature": 0})
        self.assertEqual(CATALOG.named("llm", "org/b").base_url, "http://thinker/v1")  # another slot's entry
        with self.assertRaisesRegex(LookupError, "'org/z' is not in the catalog"):
            CATALOG.named("thinker-llm", "org/z")

    def test_the_sessions_backend_choice_reaches_the_cores_lookup(self):
        body = {"thinker_model_id": "org/picked", "thinker_base_url": "http://picked/v1", "model_id": "org/front"}
        overrides = pipeline.backend_overrides(body)
        self.assertEqual(overrides, {"model_id": "org/picked", "base_url": "http://picked/v1"})
        endpoints = endpoint_lookup(CATALOG, "thinker-llm", overrides)
        self.assertEqual((endpoints(None).model, endpoints(None).base_url), ("org/picked", "http://picked/v1"))
        self.assertEqual(endpoints("org/a").base_url, "http://thinker/v1")  # a listed model keeps its entry
        with self.assertRaisesRegex(LookupError, "'org/unlisted' is not in the catalog"):
            endpoints("org/unlisted")
