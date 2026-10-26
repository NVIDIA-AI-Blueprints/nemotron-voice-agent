# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

"""Every OpenShell part of the FDA OpenShell runtime uses the one release in ``docker/fda/openshell/VERSION``.

``shared-stack-plan.md``, Part B: the release appears in 5 places, each pinned by tag and digest (the CLI by
tag and commit): the gateway image of ``hermes-openshell-init`` and of ``hermes-openshell-gateway``, the
sandbox and supervisor images in ``gateway.toml``, and the CLI build args of ``fda-hermes``.
"""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
OPENSHELL = REPO_ROOT / "docker" / "fda" / "openshell"
REGISTRY = "ghcr.io/nvidia/openshell"


def _version() -> dict[str, str]:
    lines = [line.strip() for line in (OPENSHELL / "VERSION").read_text(encoding="utf-8").splitlines()]
    return dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))


class OpenShellPinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.version = _version()
        compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
        cls.services = compose["services"]
        cls.toml = tomllib.loads((OPENSHELL / "gateway.toml").read_text(encoding="utf-8"))

    def _image(self, name: str) -> str:
        v = self.version
        return f"{REGISTRY}/{name}:{v['OPENSHELL_IMAGE_TAG']}@{v[f'OPENSHELL_{name.upper()}_DIGEST']}"

    def test_version_file(self) -> None:
        v = self.version
        self.assertRegex(v["OPENSHELL_VERSION"], r"^v\d+\.\d+\.\d+(-pre\.\d+)?$")
        self.assertEqual(v["OPENSHELL_IMAGE_TAG"], v["OPENSHELL_VERSION"].removeprefix("v"))
        self.assertRegex(v["OPENSHELL_COMMIT"], r"^[0-9a-f]{40}$")
        for name in ("GATEWAY", "SUPERVISOR", "SANDBOX"):
            self.assertRegex(v[f"OPENSHELL_{name}_DIGEST"], r"^sha256:[0-9a-f]{64}$")

    def test_gateway_images_in_compose(self) -> None:
        for service in ("hermes-openshell-init", "hermes-openshell-gateway"):
            with self.subTest(service=service):
                self.assertEqual(self.services[service]["image"], self._image("gateway"))

    def test_images_in_gateway_toml(self) -> None:
        docker = self.toml["openshell"]["drivers"]["docker"]
        self.assertEqual(docker["sandbox_runtime_image"], self._image("sandbox"))
        self.assertEqual(docker["supervisor_image"], self._image("supervisor"))
        # The locally built FDA image must be used, never pulled.
        self.assertEqual(docker["image_pull_policy"], "if_not_present")

    def test_cli_build_args(self) -> None:
        args = self.services["fda-hermes"]["build"]["args"]
        self.assertEqual(args["OPENSHELL_VERSION"], self.version["OPENSHELL_VERSION"])
        self.assertEqual(args["OPENSHELL_COMMIT"], self.version["OPENSHELL_COMMIT"])

    def test_no_other_openshell_reference(self) -> None:
        # A sixth, unpinned place (for example a `latest` tag) would escape the checks above.
        pinned = {self._image(name) for name in ("gateway", "sandbox", "supervisor")}
        for path in (REPO_ROOT / "docker-compose.yml", OPENSHELL / "gateway.toml"):
            for ref in re.findall(rf"{re.escape(REGISTRY)}/[^\s\"']+", path.read_text(encoding="utf-8")):
                with self.subTest(path=path.name, ref=ref):
                    self.assertIn(ref, pinned)

    def test_gateway_bind_matches_the_published_port(self) -> None:
        # Supervisors (host network) dial 127.0.0.1:<bind port> through the published host port.
        bind_port = self.toml["openshell"]["gateway"]["bind_address"].rsplit(":", 1)[1]
        self.assertEqual(self.services["hermes-openshell-gateway"]["ports"], [f"127.0.0.1:{bind_port}:{bind_port}"])

    def test_env_files_are_plain_key_value_lines(self) -> None:
        for path in (REPO_ROOT / "docker" / "fda" / "gateway.env", OPENSHELL / "controller.env", OPENSHELL / "VERSION"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    with self.subTest(path=path.name, line=line):
                        self.assertRegex(line, r"^[A-Z][A-Z0-9_]*=[^\s'\"$]*$")


if __name__ == "__main__":
    unittest.main()
