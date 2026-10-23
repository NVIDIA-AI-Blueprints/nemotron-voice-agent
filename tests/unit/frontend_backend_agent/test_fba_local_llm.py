# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""5.1: the optional local LLM (plan ``misc/frontend-backend-verdict/fba-removal-plan.md``).

* ``services.local.yaml`` lists ``nemotron-3.5-lightning-local`` after the Inference Hub entry,
  which stays first (the default; ``test_fba_frozen_values.py`` pins it);
* only ``docker/docker-compose.fba-local-llm.yaml`` attaches an LLM container to an FBA recipe;
* an eval-mode container (``FBA_PROFILE=eval``) refuses a Realtime session whose ``ek_`` key
  selects the local LLM; a ``default`` container accepts it.
"""

from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml
from _fba_eval_capture import MODEL, TEST_REALTIME_KEY, _connect, _mint, _voice_config, no_network
from _fba_golden import pinned_environment

from examples.frontend_backend_agent.adapters.pipecat import switches
from examples.frontend_backend_agent.core.config.profiles import ProfileError

ROOT = Path(__file__).resolve().parents[3]
LOCAL_ID = "nemotron-3.5-lightning-local"
FBA_RECIPES = ("frontend-backend-agent", "frontend-backend-agent/server")
FBA_RECIPES += ("frontend-backend-agent/single-gpu",)


def _load(path: str) -> Any:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


class CatalogTests(unittest.TestCase):
    def test_local_entry_follows_the_inference_hub_entry(self) -> None:
        local = _load("src/examples/frontend_backend_agent/services.local.yaml")
        hosts = {"server": "http://nvidia-llm:8000/v1", "singlegpu": "http://nvidia-llm-vllm:8000/v1"}
        for section, base_url in hosts.items():
            with self.subTest(section=section):
                self.assertEqual(list(local[section]["llm"]), ["nemotron-3.5-lightning", LOCAL_ID])
                self.assertEqual(local[section]["llm"][LOCAL_ID]["base_url"], base_url)
                self.assertEqual(list(local[section]["thinker-llm"]), ["nemotron-3-ultra"])
        cloud = _load("src/examples/frontend_backend_agent/services.cloud.yaml")
        self.assertNotIn(LOCAL_ID, cloud["llm"])


class ComposeTests(unittest.TestCase):
    def test_only_the_overlay_attaches_an_llm_to_fba(self) -> None:
        for path in (
            "docker/docker-compose.nemotron35-lightning.yaml",
            "docker/docker-compose.nemotron35-lightning-nim.yaml",
        ):
            for name, service in _load(path)["services"].items():
                with self.subTest(file=path, service=name):
                    self.assertFalse(set(service.get("profiles") or ()) & set(FBA_RECIPES))
        overlay = _load("docker/docker-compose.fba-local-llm.yaml")["services"]
        self.assertEqual(
            {name: service["profiles"] for name, service in overlay.items()},
            {
                "nvidia-llm-vllm-lightning": ["frontend-backend-agent/single-gpu"],
                "nvidia-llm": ["frontend-backend-agent/server"],
            },
        )


class EvalRefusalTests(unittest.TestCase):
    def test_rule(self) -> None:
        local = {"llm_id": f"self-hosted:singlegpu:{LOCAL_ID}"}
        default = {
            "llm_id": "self-hosted:singlegpu:nemotron-3.5-lightning",
            "thinker_llm_id": "cloud-nim:nemotron-3-ultra",
        }
        eval_env = {"FBA_PROFILE": "eval"}
        self.assertIsNone(switches.eval_llm_refusal(local, {}))
        self.assertIsNone(switches.eval_llm_refusal(default, eval_env))
        self.assertIsNone(switches.eval_llm_refusal({}, eval_env))
        self.assertIn(LOCAL_ID, switches.eval_llm_refusal(local, eval_env) or "")
        self.assertIsNotNone(switches.eval_llm_refusal({"thinker_llm_id": "x:other"}, eval_env))

    def test_ek_key_selecting_the_local_llm(self) -> None:
        template = {"model": MODEL, "nvidia": {"llm_id": f"self-hosted:singlegpu:{LOCAL_ID}"}}
        base = {
            "REALTIME_API_KEY": TEST_REALTIME_KEY,
            "UVICORN_WS_PING_INTERVAL": "0",
            "UVICORN_WS_PING_TIMEOUT": "20",
            "REALTIME_SERVICE_PLATFORM": "singlegpu",
        }

        async def body() -> dict[str, Any]:
            return (await _connect(await _mint(template)))[1]

        for mode in ("default", "eval"):
            env = {**base, "FBA_PROFILE": mode}
            with self.subTest(mode=mode), pinned_environment(), mock.patch.dict(os.environ, env), no_network():
                selected = asyncio.run(body())
                self.assertEqual(selected["model_id"], "nvidia/nemotron-3.5-lightning-30b-a3b")
                if mode == "eval":
                    with self.assertRaises(ProfileError):
                        _voice_config(selected, dict(os.environ))
                else:
                    config = _voice_config(selected, dict(os.environ))["config"]
                    self.assertEqual(
                        config["agent"]["frontend"]["llm"]["model"], "nvidia/nemotron-3.5-lightning-30b-a3b"
                    )


if __name__ == "__main__":
    unittest.main()
