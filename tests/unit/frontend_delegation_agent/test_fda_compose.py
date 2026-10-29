# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""The FDA Compose recipe keeps eval settings out of ``.env``'s reach (``shared-stack-plan.md``, step 5).

* ``docker/fda/gateway.env`` holds exactly the gateway variables the former FDA Compose file set, with
  its values, as plain ``KEY=value`` lines (the defaults of FDA_BACKEND_TRACE and friends differ by config);
* every FDA variable of the voice service is a literal (Compose fills ``${...}`` from ``.env``), except
  ``FDA_PROMPT_PROFILE`` (the prompt profile), which both services get with the same never-empty default;
* the voice service mentions Hermes only through ``FDA_BACKEND_URL`` and ``depends_on: fda-hermes``;
* the eval overlay sets only ``environment`` keys on ``fda-hermes``, each one listed in ``gateway.env``;
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
EVAL_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fda-eval.yaml"
PLAIN_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fda-hermes-plain.yaml"
LLM_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fda-llm.yaml"
ACKNOWLEDGE_OVERLAY = REPO_ROOT / "docker" / "docker-compose.fda-acknowledge-caller.yaml"
CONTROLLER = REPO_ROOT / "docker" / "fda" / "openshell" / "controller.sh"
GATEWAY_ENV = REPO_ROOT / "docker" / "fda" / "gateway.env"
VOICE = "frontend-delegation-agent-single-gpu"
PROFILE = "frontend-delegation-agent/single-gpu"

#: The gateway variables and values of the former docker/docker-compose.frontend-delegation-hermes.yaml.
GATEWAY_SETTINGS = {
    "FDA_GATEWAY_CONFIG": "gateway.yaml",
    "FDA_PROMPT_PROFILE": "generic",
    "FDA_GATEWAY_MAX_SESSIONS": "8",
    "BACKEND_LLM_MODEL": "nvidia/nvidia/nemotron-3-ultra",
    "BACKEND_LLM_BASE_URL": "https://inference-api.nvidia.com/v1",
    "FDA_GATEWAY_LOG": "logs/fda_gateway_events.pipecat.jsonl",
    "FDA_WORKER_LOG_DIR": "logs/fda_workers_pipecat",
}
#: The voice side's FDA variables and the values the former FDA Compose file ran with.
VOICE_SETTINGS = {
    "EXAMPLE_SELECTION": "frontend-delegation-agent",
    "FDA_PROFILE": "default",
    "FDA_MAX_SESSIONS": "4",
    "FDA_BACKEND_DELAY_S": "5",
    "FDA_PROACTIVE_AFTER_S": "15",
    "FDA_EVENT_LOG": "logs/fda_voice_events.pipecat.jsonl",
    "FRONTEND_LLM_MODEL": "nvidia/nvidia/nemotron-3.5-lightning",
    "FRONTEND_LLM_BASE_URL": "https://inference-api.nvidia.com/v1",
    "FDA_BACKEND_URL": "ws://default--fda-hermes.openshell.localhost:8080/v1/backend",
    # Read only when the voice service runs the backend itself (the LLM overlay).
    "FDA_GATEWAY_CONFIG": "gateway.yaml",
    "UVICORN_WS_PING_INTERVAL": "0",
    "FDA_SILENT_FRONTEND": "false",
    "FDA_ACKNOWLEDGE_CALLER": "false",  # the acknowledge-caller overlay turns it on
    "FDA_VERIFY_FOLLOW_UP_STEPS": "true",
}
#: The one variable the shell or .env may set: the prompt profile, with the default of each mode.
PROMPT_PROFILE = "FDA_PROMPT_PROFILE"
PROMPT_PROFILE_BASE = "${FDA_PROMPT_PROFILE:-generic}"
PROMPT_PROFILE_EVAL = "${FDA_PROMPT_PROFILE:-customer_service}"
OPENSHELL_HOST = "default--fda-hermes.openshell.localhost"
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
        cls.llm = yaml.load(LLM_OVERLAY.read_text(encoding="utf-8"), Loader=_ComposeLoader)["services"]

    def test_gateway_env_is_exactly_todays_gateway_settings(self) -> None:
        self.assertEqual(_gateway_env(), GATEWAY_SETTINGS)
        self.assertTrue(GATEWAY_ENV.read_text(encoding="utf-8").endswith("\n"))

    def test_voice_settings_are_literals(self) -> None:
        environment = self.services[VOICE]["environment"]
        for key, value in VOICE_SETTINGS.items():
            with self.subTest(key=key):
                self.assertEqual(environment[key], value)
        self.assertEqual(environment[PROMPT_PROFILE], PROMPT_PROFILE_BASE)
        for key, value in environment.items():
            if key.startswith(("FDA_", "FRONTEND_", "BACKEND_", "FBA_", "UVICORN_")) and key != PROMPT_PROFILE:
                with self.subTest(key=key):
                    self.assertNotIn("${", str(value), "a ${...} value can be changed by .env")

    def test_acknowledge_overlay_sets_one_literal(self) -> None:
        overlay = yaml.safe_load(ACKNOWLEDGE_OVERLAY.read_text(encoding="utf-8"))["services"]
        self.assertEqual(overlay, {VOICE: {"environment": {"FDA_ACKNOWLEDGE_CALLER": "true"}}})

    def test_the_voice_service_knows_hermes_only_by_url_and_health(self) -> None:
        voice = self.services[VOICE]
        self.assertEqual(voice["depends_on"], {"fda-hermes": {"condition": "service_healthy"}})
        self.assertEqual(voice["profiles"], [PROFILE])

    def test_openshell_runtime_is_the_default(self) -> None:
        hermes = self.services["fda-hermes"]
        self.assertEqual(hermes["command"], ["bash", "/cfg/controller.sh"])
        self.assertEqual(hermes["env_file"], ["docker/fda/gateway.env", "docker/fda/openshell/controller.env"])
        # Only the key and the prompt profile come from .env, and an unset key must not break every other
        # recipe's config. The profile is the voice service's, so both sides use the same prompts.
        self.assertEqual(
            hermes["environment"], {"NVIDIA_API_KEY": "${NVIDIA_API_KEY:-}", PROMPT_PROFILE: PROMPT_PROFILE_BASE}
        )
        self.assertEqual(hermes["environment"][PROMPT_PROFILE], self.services[VOICE]["environment"][PROMPT_PROFILE])
        self.assertNotIn("ports", hermes)
        self.assertTrue(hermes["init"])
        self.assertIn(f"Host: {OPENSHELL_HOST}", hermes["healthcheck"]["test"])
        gateway = self.services["hermes-openshell-gateway"]
        self.assertEqual(gateway["networks"]["default"]["aliases"], [OPENSHELL_HOST])
        self.assertEqual(gateway["ports"], ["127.0.0.1:8080:8080"])
        self.assertEqual(gateway["command"], [])
        for name in OPENSHELL_SERVICES | {"fda-hermes"}:
            with self.subTest(service=name):
                self.assertEqual(self.services[name]["profiles"], [PROFILE])
                self.assertNotIn(".env", self.services[name].get("env_file", []))

    def test_plain_overlay_overrides_single_fields(self) -> None:
        plain = self.plain
        self.assertEqual(set(plain), OPENSHELL_SERVICES | {"fda-hermes", VOICE})
        for name in OPENSHELL_SERVICES:
            self.assertEqual(plain[name], {"profiles": ["fda-openshell-off"]})
        hermes = plain["fda-hermes"]
        self.assertEqual(hermes["image"], self.services["fda-hermes-image"]["image"])
        self.assertEqual(hermes["env_file"], ["docker/fda/gateway.env"])
        # Merged, not replaced: the eval overlay's keys survive in either file order.
        self.assertEqual(hermes["environment"], {"FDA_GATEWAY_HOST": "0.0.0.0"})
        self.assertIsNone(hermes["build"])
        self.assertIsNone(hermes["command"])
        self.assertEqual(hermes["volumes"], ["./logs:/app/logs"])
        self.assertEqual(hermes["stop_grace_period"], "75s")
        self.assertEqual(plain[VOICE], {"environment": {"FDA_BACKEND_URL": "ws://fda-hermes:8790/v1/backend"}})

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
        hermes = self.overlay["fda-hermes"]
        self.assertEqual(set(hermes), {"environment"})
        self.assertLessEqual(set(hermes["environment"]), set(GATEWAY_SETTINGS))
        self.assertEqual(hermes["environment"]["FDA_GATEWAY_CONFIG"], "gateway.eval.yaml")
        self.assertEqual(self.overlay[VOICE]["environment"]["FDA_PROFILE"], "eval")
        # The LLM backend (in the voice service) loads the same gateway file as fda-hermes.
        self.assertEqual(self.overlay[VOICE]["environment"]["FDA_GATEWAY_CONFIG"], "gateway.eval.yaml")
        self.assertEqual(set(self.overlay), {VOICE, "fda-hermes"})

    def test_eval_overlay_defaults_both_services_to_the_customer_service_profile(self) -> None:
        self.assertEqual(self.overlay[VOICE]["environment"][PROMPT_PROFILE], PROMPT_PROFILE_EVAL)
        self.assertEqual(self.overlay["fda-hermes"]["environment"][PROMPT_PROFILE], PROMPT_PROFILE_EVAL)
        for name, environment in (
            (VOICE, self.overlay[VOICE]["environment"]),
            ("fda-hermes", self.overlay["fda-hermes"]["environment"]),
        ):
            with self.subTest(service=name):
                fda_keys = {key for key in environment if key.startswith("FDA_")}
                literal = {key for key in fda_keys if "${" not in str(environment[key])}
                self.assertEqual(fda_keys - literal, {PROMPT_PROFILE})


class LLMOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
        cls.llm = yaml.load(LLM_OVERLAY.read_text(encoding="utf-8"), Loader=_ComposeLoader)["services"]

    def test_the_hermes_services_do_not_start(self) -> None:
        hermes = OPENSHELL_SERVICES | {"fda-hermes", "fda-hermes-image"}
        self.assertEqual(set(self.llm), hermes | {VOICE})
        for name in hermes:
            with self.subTest(service=name):
                self.assertEqual(self.llm[name], {"profiles": ["fda-hermes-off"]})
        self.assertIsNone(self.llm[VOICE]["depends_on"])  # !reset: nothing to wait for

    def test_the_voice_service_runs_the_llm_backend_with_literal_settings(self) -> None:
        environment = self.llm[VOICE]["environment"]
        self.assertEqual((environment["FDA_BACKEND"], environment["FDA_BACKEND_LINK"]), ("llm", "in_process"))
        gateway = _gateway_env()
        for key in ("BACKEND_LLM_MODEL", "BACKEND_LLM_BASE_URL", "FDA_GATEWAY_LOG", "FDA_WORKER_LOG_DIR"):
            with self.subTest(key=key):
                self.assertEqual(environment[key], gateway[key])  # the Hermes runtime's model and log paths
        for key, value in environment.items():
            with self.subTest(key=key):
                self.assertNotIn("${", str(value))
        # The gateway file comes from docker-compose.yml or the eval overlay, never from this file.
        self.assertNotIn("FDA_GATEWAY_CONFIG", environment)


if __name__ == "__main__":
    unittest.main()
