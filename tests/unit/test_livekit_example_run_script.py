# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "src/examples/generic_assistant_livekit/run.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")


def run(args, tmp_path, env_text="NVIDIA_API_KEY=nvapi-test\nHF_TOKEN=hf_test\n", extra_env=None):
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    env = {k: v for k, v in os.environ.items() if k not in ("NVIDIA_API_KEY", "HF_TOKEN")}
    env.update({"DRY_RUN": "1", "ENV_FILE": str(env_file), **(extra_env or {})})
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=30)


def commands(result):
    return [line for line in result.stderr.splitlines() if line.startswith("+ docker")]


@pytest.mark.parametrize(
    ("name", "profile"),
    [
        ("cloud", "generic-assistant-livekit"),
        ("server", "generic-assistant-livekit/server"),
        ("single-gpu", "generic-assistant-livekit/single-gpu"),
    ],
)
def test_up_runs_compose_with_the_profile(tmp_path, name, profile):
    result = run(["up", name], tmp_path)
    assert result.returncode == 0, result.stderr
    (command,) = commands(result)
    assert command.endswith(f"--profile {profile} up -d")
    assert "--env-file" in command
    assert "-f " in command
    assert "docker-compose.yaml" in command


def test_up_rejects_unknown_profile(tmp_path):
    result = run(["up", "nope"], tmp_path)
    assert result.returncode != 0
    assert "unknown profile" in result.stderr


@pytest.mark.parametrize(
    ("name", "key"), [("cloud", "NVIDIA_API_KEY"), ("server", "NVIDIA_API_KEY"), ("single-gpu", "HF_TOKEN")]
)
def test_up_requires_the_profile_key(tmp_path, name, key):
    result = run(["up", name], tmp_path, env_text=f"{key}=\n")
    assert result.returncode != 0
    assert f"{key} is empty" in result.stderr


def test_up_accepts_quoted_env_values(tmp_path):
    result = run(["up", "single-gpu"], tmp_path, env_text='HF_TOKEN="hf_quoted"\n')
    assert result.returncode == 0, result.stderr


def test_up_accepts_a_key_from_the_environment(tmp_path):
    result = run(["up", "cloud"], tmp_path, env_text="", extra_env={"NVIDIA_API_KEY": "nvapi-env"})
    assert result.returncode == 0, result.stderr


def test_up_without_env_file_explains_the_fix(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in ("NVIDIA_API_KEY", "HF_TOKEN")}
    env.update({"DRY_RUN": "1", "ENV_FILE": str(tmp_path / "missing.env")})
    result = subprocess.run(["bash", str(SCRIPT), "up", "cloud"], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode != 0
    assert "no .env file" in result.stderr


def test_down_without_profile_stops_all_three(tmp_path):
    result = run(["down"], tmp_path)
    assert result.returncode == 0
    assert [c.rsplit("--profile ", 1)[1] for c in commands(result)] == [
        "generic-assistant-livekit down",
        "generic-assistant-livekit/server down",
        "generic-assistant-livekit/single-gpu down",
    ]


def test_clean_removes_volumes_and_optionally_the_image(tmp_path):
    result = run(["clean", "server", "-y", "--image"], tmp_path)
    assert result.returncode == 0
    lines = commands(result)
    assert lines[0].endswith("--profile generic-assistant-livekit/server down -v")
    assert lines[1] == "+ docker image rm generic-assistant-livekit:latest"


def test_clean_without_image_flag_keeps_the_image(tmp_path):
    result = run(["clean", "-y"], tmp_path)
    assert len(commands(result)) == 3
    assert all(c.endswith("down -v") for c in commands(result))


@pytest.mark.parametrize(
    ("args", "service"),
    [
        (["single-gpu", "llm"], "livekit-nvidia-llm-vllm"),
        (["server", "llm"], "livekit-nvidia-llm"),
        (["server", "agent"], "livekit-agent-server"),
        (["single-gpu", "speech"], "livekit-nemo-speech"),
        (["cloud", "livekit"], "livekit-server"),
    ],
)
def test_logs_resolve_short_service_names(tmp_path, args, service):
    result = run(["logs", *args], tmp_path)
    assert result.returncode == 0, result.stderr
    assert commands(result)[0].endswith(f"logs -f --tail 50 {service}")


def test_logs_reject_services_the_cloud_profile_lacks(tmp_path):
    result = run(["logs", "cloud", "asr"], tmp_path)
    assert result.returncode != 0
    assert "no local ASR" in result.stderr


def test_ps_without_profile_lists_all_three(tmp_path):
    (command,) = commands(run(["ps"], tmp_path))
    assert command.count("--profile ") == 3


def test_unknown_command_prints_usage(tmp_path):
    result = run(["explode"], tmp_path)
    assert result.returncode == 2
    assert "run.sh up" in result.stderr
