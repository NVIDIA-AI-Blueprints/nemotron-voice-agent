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


def test_up_ends_with_the_tunnel_instructions(tmp_path):
    for args in (["up", "cloud"], ["up", "single-gpu"], ["up", "server", "--no-wait"], ["wait", "single-gpu"]):
        result = run(args, tmp_path)
        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        assert "run.sh tunnel <user>@" in " ".join(lines[-4:]), args
        assert lines[-1].startswith("Keep it running"), args
        assert "ssh -N -L 3000:127.0.0.1:3000 -L 7880:127.0.0.1:7880 -L 7881:127.0.0.1:7881" in result.stderr


def test_tunnel_forwards_the_ui_and_livekit_ports(tmp_path):
    result = run(["tunnel", "lanaz@spark"], tmp_path)
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stderr.splitlines() if line.startswith("+ ssh"))
    assert "-L 3000:127.0.0.1:3000 -L 7880:127.0.0.1:7880 -L 7881:127.0.0.1:7881 lanaz@spark" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "http://localhost:3000/" in result.stderr
    assert not any(line.startswith("+ docker") for line in result.stderr.splitlines())  # a laptop needs no docker


def test_tunnel_local_ui_port_can_move(tmp_path):
    result = run(["tunnel", "lanaz@spark"], tmp_path, extra_env={"LOCAL_UI_PORT": "3001"})
    assert "-L 3001:127.0.0.1:3000" in result.stderr
    assert "http://localhost:3001/" in result.stderr


def test_tunnel_needs_a_host(tmp_path):
    result = run(["tunnel"], tmp_path)
    assert result.returncode != 0
    assert "usage: run.sh tunnel <user@host>" in result.stderr


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
# Fake docker for run.sh. Records every call in $FAKE_LOG. The shared LiveKit services always "exist"; an agent service
# exists only when it is listed in $FAKE_RUNNING. Health checks follow $FAKE_HEALTH, one answer per call.
args="$*"
[ -n "$FAKE_LOG" ] && echo "$args" >> "$FAKE_LOG"
counter="$FAKE_STATE/count"
case "$args" in
  "compose version"*) echo "${FAKE_COMPOSE_VERSION:-2.30.0}" ;;
  *"compose"*" logs "*) [ "${FAKE_REGISTERED:-1}" = 1 ] && echo '{"message": "registered worker"}' ;;
  *"ps -q "*)
    for svc in $FAKE_RUNNING; do
      case "$args" in *"ps -q $svc"*) echo "id-$svc" ;; esac
    done ;;
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
exit 0
"""

FAKE_SS = "#!/usr/bin/env bash\nexit 0\n"  # no local port is in use


def fake_env(tmp_path, env_text="NVIDIA_API_KEY=nvapi-test\nHF_TOKEN=hf_test\n", **extra):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    for name, body in (("docker", FAKE_DOCKER), ("ss", FAKE_SS)):
        path = fake / name
        path.write_text(body)
        path.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    env = {k: v for k, v in os.environ.items() if k not in ("DRY_RUN", "NVIDIA_API_KEY", "HF_TOKEN")}
    env.update(
        {
            "PATH": f"{fake}:{env['PATH']}",
            "ENV_FILE": str(env_file),
            "WAIT_INTERVAL": "0",
            "FAKE_STATE": str(state_dir),
            "FAKE_LOG": str(tmp_path / "docker.log"),
            **extra,
        }
    )
    return env


def run_real(tmp_path, args, running="", **extra):
    env = fake_env(tmp_path, FAKE_RUNNING=running, **extra)
    result = subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)
    log = tmp_path / "docker.log"
    return result, (log.read_text().splitlines() if log.exists() else [])


def downs(calls):
    return [c for c in calls if c.rstrip().endswith(" down")]


def run_wait(tmp_path, health, timeout="30", state="running", **extra):
    env = fake_env(tmp_path, FAKE_HEALTH=health, FAKE_STATE_STATUS=state, **extra)
    return subprocess.run(
        ["bash", str(SCRIPT), "wait", "single-gpu", timeout], capture_output=True, text=True, env=env, timeout=60
    )


def test_wait_reports_progress_until_healthy(tmp_path):
    result = run_wait(tmp_path, "starting starting healthy")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("still starting:") == 2
    assert "livekit-nvidia-llm-vllm: starting | last log: Loading weights took 12.3 seconds" in result.stderr
    assert "All healthy. Ready to connect." in result.stderr
    assert result.stderr.index("Ready to connect") < result.stderr.index("run.sh tunnel <user>@")


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


def test_up_does_not_stop_other_profiles_because_of_the_shared_livekit_services(tmp_path):
    # The LiveKit server and UI belong to every profile, so a profile-wide `ps` lists them whenever any profile runs.
    result, calls = run_real(tmp_path, ["up", "server", "--no-wait"], running="livekit-agent-server")
    assert result.returncode == 0, result.stderr
    assert downs(calls) == []
    assert any(c.endswith("--profile generic-assistant-livekit/server up -d") for c in calls)


def test_up_stops_only_the_profile_whose_agent_is_running(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--no-wait"], running="livekit-agent-server")
    assert result.returncode == 0, result.stderr
    assert [c.split("--profile ")[1] for c in downs(calls)] == ["generic-assistant-livekit/server down"]
    assert "Another profile (server) is running" in result.stderr
    assert calls.index(downs(calls)[0]) < next(
        i for i, c in enumerate(calls) if c.endswith("cloud up -d") or c.endswith("livekit up -d")
    )


def test_up_with_nothing_running_stops_nothing(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--no-wait"])
    assert result.returncode == 0, result.stderr
    assert downs(calls) == []


def test_old_docker_compose_is_rejected_with_a_clear_message(tmp_path):
    result, _ = run_real(tmp_path, ["up", "cloud", "--no-wait"], FAKE_COMPOSE_VERSION="2.20.5")
    assert result.returncode != 0
    assert "Docker Compose 2.20.5 is too old" in result.stderr
    assert run_real(tmp_path, ["ps"], FAKE_COMPOSE_VERSION="v2.24.0")[0].returncode == 0


def test_wait_also_requires_the_agent_to_register_with_livekit(tmp_path):
    result = run_wait(tmp_path, "healthy", timeout="1", FAKE_REGISTERED="0")
    assert result.returncode != 0
    assert "livekit-agent-single-gpu: starting, not registered with LiveKit yet" in result.stderr
    ok = run_wait(tmp_path, "healthy", timeout="5")
    assert ok.returncode == 0, ok.stderr
    assert "Ready to connect" in ok.stderr


def test_the_ui_port_comes_from_the_env_file(tmp_path):
    result = run(["up", "cloud", "--no-wait"], tmp_path, env_text="NVIDIA_API_KEY=x\nLIVEKIT_UI_PORT=4000\n")
    assert "http://localhost:4000/" in result.stderr
    tunnel = run(["tunnel", "me@host"], tmp_path, env_text="LIVEKIT_UI_PORT=4000\n")
    assert "-L 3000:127.0.0.1:4000" in tunnel.stderr
