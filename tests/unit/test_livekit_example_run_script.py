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
    up, status = commands(result)
    assert up.endswith(f"--profile {profile} up -d")
    assert "--env-file" in up
    assert "-f " in up
    assert "docker-compose.yaml" in up
    assert status.endswith(f"--profile {profile} ps")


def test_up_waits_by_default(tmp_path):
    for name in ("cloud", "server", "single-gpu"):
        result = run(["up", name], tmp_path)
        assert f"Waiting up to 3600s for the {name} profile" in result.stderr
        assert "Not waiting" not in result.stderr


def test_up_explains_how_to_stop_waiting_for_model_profiles(tmp_path):
    assert "Ctrl+C stops only this wait" in run(["up", "single-gpu"], tmp_path).stderr
    assert "Ctrl+C" not in run(["up", "cloud"], tmp_path).stderr


def test_up_no_wait_skips_the_wait(tmp_path):
    result = run(["up", "server", "--no-wait"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Waiting up to" not in result.stderr
    assert "run.sh wait server" in result.stderr


def test_up_accepts_the_explicit_wait_flag(tmp_path):
    result = run(["up", "server", "--wait"], tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Waiting up to 3600s" in result.stderr


def test_up_rejects_unknown_options(tmp_path):
    result = run(["up", "cloud", "--nope"], tmp_path)
    assert result.returncode != 0
    assert "unknown option '--nope' for up" in result.stderr
    assert "--no-wait" in result.stderr


def test_wait_announces_progress_reporting(tmp_path):
    result = run(["wait", "single-gpu", "120"], tmp_path)
    assert "up to 120s" in result.stderr
    assert "progress every 15s" in result.stderr


FAKE_DOCKER = """#!/usr/bin/env bash
# Fake docker for run.sh wait: one container that reports STATUS_SEQUENCE, one health check per call.
args="$*"
counter="$FAKE_STATE/count"
case "$args" in
  "compose version") exit 0 ;;
  *"ps -q"*) echo abc123 ;;
  *"inspect"*"Health.Status"*)
    n=$(cat "$counter" 2>/dev/null || echo 0)
    echo $((n + 1)) > "$counter"
    set -- $FAKE_HEALTH
    shift $n 2>/dev/null || true
    echo "${1:-${FAKE_HEALTH##* }}" ;;
  *"inspect"*".State.Status"*) echo "${FAKE_STATE_STATUS:-running}" ;;
  *"inspect"*".Name"*) echo /livekit-nvidia-llm-vllm ;;
  *"logs --tail 1"*) echo "Loading weights took 12.3 seconds" ;;
esac
"""


def run_wait(tmp_path, health, timeout="30", state="running"):
    fake = tmp_path / "bin"
    fake.mkdir()
    docker = fake / "docker"
    docker.write_text(FAKE_DOCKER)
    docker.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=hf_test\n")
    env = {k: v for k, v in os.environ.items() if k != "DRY_RUN"}
    env.update(
        {
            "PATH": f"{fake}:{env['PATH']}",
            "ENV_FILE": str(env_file),
            "WAIT_INTERVAL": "0",
            "FAKE_STATE": str(state_dir),
            "FAKE_HEALTH": health,
            "FAKE_STATE_STATUS": state,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT), "wait", "single-gpu", timeout], capture_output=True, text=True, env=env, timeout=60
    )


def test_wait_reports_progress_until_healthy(tmp_path):
    result = run_wait(tmp_path, "starting starting healthy")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("still starting:") == 2
    assert "livekit-nvidia-llm-vllm: starting | last log: Loading weights took 12.3 seconds" in result.stderr
    assert "All healthy. Ready to connect." in result.stderr


def test_wait_labels_an_unhealthy_container_as_not_ready(tmp_path):
    result = run_wait(tmp_path, "unhealthy unhealthy healthy")
    assert result.returncode == 0, result.stderr
    assert "livekit-nvidia-llm-vllm: not ready yet | last log: Loading weights took 12.3 seconds" in result.stderr
    assert "unhealthy" not in result.stderr
    assert result.stderr.count("is not an error") == 1
    assert "All healthy. Ready to connect." in result.stderr


def test_wait_does_not_explain_starting_containers_as_not_ready(tmp_path):
    result = run_wait(tmp_path, "starting healthy")
    assert "not ready yet" not in result.stderr
    assert "is not an error" not in result.stderr


def test_wait_times_out_and_names_the_pending_container(tmp_path):
    result = run_wait(tmp_path, "starting", timeout="0")
    assert result.returncode != 0
    assert "timed out after 0s" in result.stderr
    assert "livekit-nvidia-llm-vllm: starting" in result.stderr


def test_wait_flags_a_restarting_container(tmp_path):
    result = run_wait(tmp_path, "starting healthy", timeout="1", state="restarting")
    assert result.returncode != 0
    assert "livekit-nvidia-llm-vllm is restarting" in result.stderr


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


def test_clean_removes_volumes_but_keeps_the_image_by_default(tmp_path):
    result = run(["clean", "server", "-y"], tmp_path)
    assert result.returncode == 0
    (command,) = commands(result)
    assert command.endswith("--profile generic-assistant-livekit/server down -v")
    assert "docker image rm" not in result.stderr


def test_clean_image_flag_also_removes_the_image(tmp_path):
    result = run(["clean", "server", "-y", "--image"], tmp_path)
    assert result.returncode == 0
    lines = commands(result)
    assert lines[0].endswith("--profile generic-assistant-livekit/server down -v")
    assert lines[1] == "+ docker image rm generic-assistant-livekit:latest"


def test_clean_without_a_profile_covers_all_three(tmp_path):
    result = run(["clean", "-y"], tmp_path)
    assert len(commands(result)) == 3
    assert all(c.endswith("down -v") for c in commands(result))


def test_clean_names_the_unknown_option(tmp_path):
    result = run(["clean", "--bogus"], tmp_path)
    assert result.returncode != 0
    assert "unknown option '--bogus'" in result.stderr
    assert "--keep-image is not needed" in run(["clean", "--keep-image"], tmp_path).stderr


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
