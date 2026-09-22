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
