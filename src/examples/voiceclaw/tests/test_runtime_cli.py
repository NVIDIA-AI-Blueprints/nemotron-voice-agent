# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D103

from pathlib import Path
from unittest.mock import patch

import pytest

from voiceclaw.runtime_cli import _parser, main


def test_runtime_cli_exposes_only_serve_and_healthcheck() -> None:
    parser = _parser()
    serve = parser.parse_args(["serve", "--config", "/var/lib/voiceclaw/config/voiceclaw.yaml", "--ui"])
    assert serve.command == "serve"
    assert serve.config == Path("/var/lib/voiceclaw/config/voiceclaw.yaml")
    assert serve.ui is True
    assert parser.parse_args(["healthcheck"]).command == "healthcheck"
    with pytest.raises(SystemExit):
        parser.parse_args(["exec", "/bin/sh"])


def test_runtime_cli_forwards_typed_arguments_to_the_supervisor() -> None:
    with patch("voiceclaw.runtime_cli.container.main") as supervisor:
        main(["serve", "--config", "/run/config.yaml", "--ui"])
    supervisor.assert_called_once_with(["--config", "/run/config.yaml", "--ui"])

    with patch("voiceclaw.runtime_cli.container.main") as supervisor:
        main(["healthcheck"])
    supervisor.assert_called_once_with(["--healthcheck"])


def test_runtime_cli_dispatches_the_marker_selected_managed_runtime(tmp_path: Path) -> None:
    marker = tmp_path / "voiceclaw-nemoclaw-managed"
    marker.touch()

    with (
        patch("voiceclaw.runtime_cli._MANAGED_RUNTIME_MARKER", marker),
        patch("voiceclaw.runtime_cli.container.main") as supervisor,
        patch("voiceclaw.runtime_cli.managed_runtime.main") as managed,
    ):
        main(["serve"])

    managed.assert_called_once_with(["serve"])
    supervisor.assert_not_called()


def test_runtime_cli_rejects_managed_configuration_and_ui_overrides(tmp_path: Path) -> None:
    marker = tmp_path / "voiceclaw-nemoclaw-managed"
    marker.touch()

    for arguments in (("serve", "--config", "/run/config.yaml"), ("serve", "--ui")):
        with (
            patch("voiceclaw.runtime_cli._MANAGED_RUNTIME_MARKER", marker),
            patch("voiceclaw.runtime_cli.container.main") as supervisor,
            patch("voiceclaw.runtime_cli.managed_runtime.main") as managed,
            pytest.raises(SystemExit, match="2"),
        ):
            main(list(arguments))
        managed.assert_not_called()
        supervisor.assert_not_called()


def test_runtime_cli_uses_managed_liveness_check_when_marker_is_present(tmp_path: Path) -> None:
    marker = tmp_path / "voiceclaw-nemoclaw-managed"
    marker.touch()

    with (
        patch("voiceclaw.runtime_cli._MANAGED_RUNTIME_MARKER", marker),
        patch("voiceclaw.runtime_cli.container.main") as supervisor,
        patch("voiceclaw.runtime_cli.managed_runtime.run_healthcheck") as healthcheck,
    ):
        main(["healthcheck"])

    healthcheck.assert_called_once_with()
    supervisor.assert_not_called()
