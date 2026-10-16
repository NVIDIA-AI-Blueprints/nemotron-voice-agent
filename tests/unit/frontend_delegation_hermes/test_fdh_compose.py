# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The FDH Compose recipe keeps eval settings out of ``.env``'s reach (``shared-stack-plan.md``, step 5).

* ``docker/fdh/gateway.env`` holds exactly the gateway variables the former FDH Compose file set, with
  its values, as plain ``KEY=value`` lines (the defaults of FDH_BACKEND_TRACE and friends differ by config);
* every FDH variable of the voice service is a literal (Compose fills ``${...}`` from ``.env``);
* the voice service mentions Hermes only through ``FDH_BACKEND_URL`` and ``depends_on: fdh-hermes``;
* the eval overlay sets only ``environment`` keys on ``fdh-hermes``, each one listed in ``gateway.env``;
* the OpenShell runtime is the default, and the plain overlay overrides single fields only (Part B, step 10).
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE = REPO_ROOT / "docker-compose.yml"
EVAL_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fdh-eval.yaml"
PLAIN_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fdh-hermes-plain.yaml"
CONTROLLER = REPO_ROOT / "docker" / "fdh" / "openshell" / "controller.sh"
GATEWAY_ENV = REPO_ROOT / "docker" / "fdh" / "gateway.env"
VOICE = "frontend-delegation-hermes-agent-single-gpu"
PROFILE = "frontend-delegation-hermes-agent/single-gpu"

#: The gateway variables and values of the former docker/docker-compose.frontend-delegation-hermes.yaml.
GATEWAY_SETTINGS = {
    "FDH_GATEWAY_CONFIG": "gateway.yaml",
    "FDH_GATEWAY_MAX_SESSIONS": "8",
    "BACKEND_LLM_MODEL": "nvidia/nvidia/nemotron-3-ultra",
    "BACKEND_LLM_BASE_URL": "https://inference-api.nvidia.com/v1",
    "FDH_GATEWAY_LOG": "logs/fdh_gateway_events.pipecat.jsonl",
    "FDH_WORKER_LOG_DIR": "logs/fdh_workers_pipecat",
}
#: The voice side's FDH variables and the values the former FDH Compose file ran with.
VOICE_SETTINGS = {
    "EXAMPLE_SELECTION": "frontend-delegation-hermes-agent",
    "FDH_PROFILE": "default",
    "FDH_MAX_SESSIONS": "4",
    "FDH_BACKEND_DELAY_S": "5",
    "FDH_PROACTIVE_AFTER_S": "15",
    "FDH_EVENT_LOG": "logs/fdh_voice_events.pipecat.jsonl",
    "FDH_PROMPTS": "prompts.yaml",
    "FRONTEND_LLM_MODEL": "nvidia/nvidia/nemotron-3.5-lightning",
    "FRONTEND_LLM_BASE_URL": "https://inference-api.nvidia.com/v1",
    "FDH_BACKEND_URL": "ws://default--fdh-hermes.openshell.localhost:8080/v1/backend",
    "UVICORN_WS_PING_INTERVAL": "0",
}
OPENSHELL_HOST = "default--fdh-hermes.openshell.localhost"
OPENSHELL_SERVICES = {"hermes-openshell-init", "hermes-openshell-gateway"}


class _ComposeLoader(yaml.SafeLoader):
    """Reads Compose's ``!override`` and ``!reset`` tags as plain values."""


def _override(loader: yaml.SafeLoader, node: yaml.Node) -> object:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    return loader.construct_scalar(node)


_ComposeLoader.add_constructor("!override", _override)
_ComposeLoader.add_constructor("!reset", lambda loader, node: None)


def _gateway_env() -> dict[str, str]:
    lines = [line.strip() for line in GATEWAY_ENV.read_text(encoding="utf-8").splitlines()]
    pairs = [line for line in lines if line and not line.startswith("#")]
    for line in pairs:
        assert re.fullmatch(r"[A-Z][A-Z0-9_]*=[^\s'\"$]*", line), f"not a plain KEY=value line: {line!r}"
    return dict(line.split("=", 1) for line in pairs)


class ComposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
        cls.overlay = yaml.safe_load(EVAL_OVERLAY.read_text(encoding="utf-8"))["services"]
        cls.plain = yaml.load(PLAIN_OVERLAY.read_text(encoding="utf-8"), Loader=_ComposeLoader)["services"]

    def test_gateway_env_is_exactly_todays_gateway_settings(self) -> None:
        self.assertEqual(_gateway_env(), GATEWAY_SETTINGS)
        self.assertTrue(GATEWAY_ENV.read_text(encoding="utf-8").endswith("\n"))

    def test_voice_settings_are_literals(self) -> None:
        environment = self.services[VOICE]["environment"]
        for key, value in VOICE_SETTINGS.items():
            with self.subTest(key=key):
                self.assertEqual(environment[key], value)
        for key, value in environment.items():
            if key.startswith(("FDH_", "FRONTEND_", "BACKEND_", "FBA_", "UVICORN_")):
                with self.subTest(key=key):
                    self.assertNotIn("${", str(value), "a ${...} value can be changed by .env")

    def test_the_voice_service_knows_hermes_only_by_url_and_health(self) -> None:
        voice = self.services[VOICE]
        self.assertEqual(voice["depends_on"], {"fdh-hermes": {"condition": "service_healthy"}})
        self.assertEqual(voice["profiles"], [PROFILE])

    def test_openshell_runtime_is_the_default(self) -> None:
        hermes = self.services["fdh-hermes"]
        self.assertEqual(hermes["command"], ["bash", "/cfg/controller.sh"])
        self.assertEqual(hermes["env_file"], ["docker/fdh/gateway.env", "docker/fdh/openshell/controller.env"])
        # Only the key comes from .env, and an unset key must not break every other recipe's config.
        self.assertEqual(hermes["environment"], {"NVIDIA_API_KEY": "${NVIDIA_API_KEY:-}"})
        self.assertNotIn("ports", hermes)
        self.assertTrue(hermes["init"])
        self.assertIn(f"Host: {OPENSHELL_HOST}", hermes["healthcheck"]["test"])
        gateway = self.services["hermes-openshell-gateway"]
        self.assertEqual(gateway["networks"]["default"]["aliases"], [OPENSHELL_HOST])
        self.assertEqual(gateway["ports"], ["127.0.0.1:8080:8080"])
        self.assertEqual(gateway["command"], [])
        for name in OPENSHELL_SERVICES | {"fdh-hermes"}:
            with self.subTest(service=name):
                self.assertEqual(self.services[name]["profiles"], [PROFILE])
                self.assertNotIn(".env", self.services[name].get("env_file", []))

    def test_plain_overlay_overrides_single_fields(self) -> None:
        plain = self.plain
        self.assertEqual(set(plain), OPENSHELL_SERVICES | {"fdh-hermes", VOICE})
        for name in OPENSHELL_SERVICES:
            self.assertEqual(plain[name], {"profiles": ["fdh-openshell-off"]})
        hermes = plain["fdh-hermes"]
        self.assertEqual(hermes["image"], self.services["fdh-hermes-image"]["image"])
        self.assertEqual(hermes["env_file"], ["docker/fdh/gateway.env"])
        # Merged, not replaced: the eval overlay's keys survive in either file order.
        self.assertEqual(hermes["environment"], {"FDH_GATEWAY_HOST": "0.0.0.0"})
        self.assertIsNone(hermes["build"])
        self.assertIsNone(hermes["command"])
        self.assertEqual(hermes["volumes"], ["./logs:/app/logs"])
        self.assertEqual(hermes["stop_grace_period"], "75s")
        self.assertEqual(plain[VOICE], {"environment": {"FDH_BACKEND_URL": "ws://fdh-hermes:8790/v1/backend"}})

    def test_controller_passes_every_gateway_env_key(self) -> None:
        # The controller's loop must keep a last line with no newline (shared-stack-plan.md, step 11).
        text = GATEWAY_ENV.read_text(encoding="utf-8").rstrip("\n")
        loop = CONTROLLER.read_text(encoding="utf-8")
        start = loop.index("env_args=()")
        body = loop[start : loop.index("done < /cfg-env/gateway.env")]
        body += 'done < "$1"\nprintf "%s\\n" "${env_args[@]}"\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gateway.env"
            path.write_text(text, encoding="utf-8")
            out = subprocess.run(["bash", "-c", body, "loop", str(path)], capture_output=True, text=True, check=True)
        args = out.stdout.split()
        self.assertEqual(args[0::2], ["--env-from"] * len(GATEWAY_SETTINGS))
        self.assertEqual(args[1::2], list(GATEWAY_SETTINGS))

    def test_eval_overlay_sets_only_gateway_env_keys_on_hermes(self) -> None:
        hermes = self.overlay["fdh-hermes"]
        self.assertEqual(set(hermes), {"environment"})
        self.assertLessEqual(set(hermes["environment"]), set(GATEWAY_SETTINGS))
        self.assertEqual(hermes["environment"]["FDH_GATEWAY_CONFIG"], "gateway.eval.yaml")
        self.assertEqual(self.overlay[VOICE]["environment"]["FDH_PROFILE"], "eval")
        self.assertEqual(set(self.overlay), {VOICE, "fdh-hermes"})


if __name__ == "__main__":
    unittest.main()
