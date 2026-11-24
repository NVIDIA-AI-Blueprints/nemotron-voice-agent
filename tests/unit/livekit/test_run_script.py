# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "src/nemotron_voice_agent/livekit/examples/generic_assistant/run.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")

# Settings that an operator may export. They would change the defaults that these tests assert.
OPERATOR_VARIABLES = {"NVIDIA_API_KEY", "HF_TOKEN", "DRY_RUN", "ENV_FILE", "LOCAL_UI_PORT", "WAIT_INTERVAL"}


def clean_env():
    return {
        key: value
        for key, value in os.environ.items()
        if key not in OPERATOR_VARIABLES and not key.startswith(("LIVEKIT_", "FAKE_"))
    }


def run(args, tmp_path, env_text="NVIDIA_API_KEY=nvapi-test\nHF_TOKEN=hf_test\n", extra_env=None):
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    env = clean_env()
    env.update({"DRY_RUN": "1", "ENV_FILE": str(env_file), **(extra_env or {})})
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=30)


SIP_PROFILE = "generic-assistant-livekit/sip"


def profiles_of(command):
    """The --profile values of a docker compose command line."""
    return re.findall(r"--profile (\S+)", command)


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
    assert up.split(" -f ")[1].split()[0].endswith("/docker-compose.yml")
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
    result = run(["tunnel", "user@spark"], tmp_path)
    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stderr.splitlines() if line.startswith("+ ssh"))
    assert "-L 3000:127.0.0.1:3000 -L 7880:127.0.0.1:7880 -L 7881:127.0.0.1:7881 user@spark" in command
    assert "ExitOnForwardFailure=yes" in command
    assert "http://localhost:3000/" in result.stderr
    assert not any(line.startswith("+ docker") for line in result.stderr.splitlines())  # a laptop needs no docker


def test_tunnel_local_ui_port_can_move(tmp_path):
    result = run(["tunnel", "user@spark"], tmp_path, extra_env={"LOCAL_UI_PORT": "3001"})
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
  *"compose"*" logs "*)
    [ -n "$FAKE_LOG_LINE" ] && echo "$FAKE_LOG_LINE"
    [ "${FAKE_REGISTERED:-1}" = 1 ] && echo '{"message": "registered worker"}' ;;
  *"config --services"*) echo "${FAKE_SERVICES:-livekit-nvidia-llm-vllm}" | tr ' ' '\n' ;;
  *"docker port "* | "port "*) echo "${FAKE_REDIS_PORT-127.0.0.1:6379}" ;;
  *"config --images"*) echo "livekit/sip:v1.23.0" ;;
  *"image inspect"*) echo arm64 ;;
  *"ps -a --format"*) for svc in $FAKE_LEFT; do echo "$svc"; done ;;
  *"up -d")
    echo "env LIVEKIT_SIP=$LIVEKIT_SIP" >> "$FAKE_LOG" ;;
  *"ps -a -q")
    [ -n "$FAKE_STACK" ] && echo abc123 ;;
  *"ps -a -q "*)
    svc="${args##*ps -a -q }"
    # The optional overlay containers exist only when $FAKE_OVERLAYS lists them.
    case "$svc" in
      livekit-sip | livekit-redis) case " $FAKE_OVERLAYS " in *" $svc "*) ;; *) exit 0 ;; esac ;; esac
    case " $FAKE_MISSING " in *" $svc "*) ;; *) echo "id-$svc" ;; esac ;;
  *"ps -q "*)
    for svc in $FAKE_RUNNING; do
      case "$args" in *"ps -q $svc"*) echo "id-$svc" ;; esac
    done ;;
  *"ps -q"*) echo abc123 ;;
  *"inspect"*)
    id="${args##* }"; svc="${id#id-}"
    case "$args" in
      *"Health.Status"*)
        case " ${FAKE_HEALTH_SERVICES:-livekit-nvidia-llm-vllm} " in
          *" $svc "*)
            n=$(cat "$counter" 2>/dev/null || echo 0)
            echo $((n + 1)) > "$counter"
            set -- $FAKE_HEALTH
            shift $n 2>/dev/null || true
            echo "${1:-${FAKE_HEALTH##* }}" ;;
          *) echo "" ;;
        esac ;;
      *".State.ExitCode"*) [ "$svc" = livekit-init ] && echo "${FAKE_INIT_EXIT:-0}" || echo 137 ;;
      *".State.Status"*)
        if [ "$svc" = livekit-init ]; then echo "${FAKE_INIT_STATE:-exited}"
        else case " $FAKE_STOPPED " in *" $svc "*) echo exited ;; *) echo "${FAKE_STATE_STATUS:-running}" ;; esac
        fi ;;
      *".Name"*) echo "/$svc" ;;
    esac ;;
  *"logs --tail 1"*) echo "Loading weights took 12.3 seconds" ;;
esac
exit 0
"""

# No local port is in use, except those in $FAKE_LISTEN (space-separated).
FAKE_SS = """#!/usr/bin/env bash
for port in $FAKE_LISTEN; do
  case "$*" in *":$port"*) echo "LISTEN 0 128 0.0.0.0:$port 0.0.0.0:*" ;; esac
done
exit 0
"""

# Reports Linux unless $FAKE_OS says otherwise, so the --sip tests behave the same on any machine.
FAKE_UNAME = '#!/usr/bin/env bash\n[ "$1" = -s ] && echo "${FAKE_OS:-Linux}" || /usr/bin/uname "$@"\n'

# Answers the public-address lookup with $FAKE_PUBLIC_IP (a failure when it is unset), and the rest with $FAKE_TOKEN.
FAKE_CURL = """#!/usr/bin/env bash
case "$*" in
  *ifconfig.me*) [ -n "${FAKE_PUBLIC_IP:-}" ] && printf '%s' "$FAKE_PUBLIC_IP" || exit 22 ;;
  *) printf '%s' "$FAKE_TOKEN" ;;
esac
"""

# A host with the LAN address 192.168.1.20/24 on eth0 behind the router 192.168.1.1.
FAKE_IP = """#!/usr/bin/env bash
case "$*" in
  *"route get"*) echo "1.1.1.1 via 192.168.1.1 dev eth0 src 192.168.1.20 uid 1000" ;;
  *"addr show"*) echo "2: eth0    inet 192.168.1.20/24 brd 192.168.1.255 scope global dynamic eth0" ;;
esac
"""


def fake_env(tmp_path, env_text="NVIDIA_API_KEY=nvapi-test\nHF_TOKEN=hf_test\n", **extra):
    fake = tmp_path / "bin"
    fake.mkdir(exist_ok=True)
    for name, body in (
        ("docker", FAKE_DOCKER),
        ("ss", FAKE_SS),
        ("curl", FAKE_CURL),
        ("uname", FAKE_UNAME),
        ("ip", FAKE_IP),
    ):
        path = fake / name
        path.write_text(body)
        path.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir(exist_ok=True)
    env_file = tmp_path / ".env"
    env_file.write_text(env_text)
    env = clean_env()
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


def run_real(tmp_path, args, running="", env_text=None, **extra):
    env = fake_env(tmp_path, **({"env_text": env_text} if env_text else {}), FAKE_RUNNING=running, **extra)
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


def test_wait_does_not_report_ready_while_a_service_has_no_container(tmp_path):
    services = "livekit-nvidia-llm-vllm livekit-nemo-speech"
    result = run_wait(tmp_path, "healthy", timeout="1", FAKE_SERVICES=services, FAKE_MISSING="livekit-nemo-speech")
    assert result.returncode != 0
    assert "livekit-nemo-speech: not created" in result.stderr
    assert "Ready to connect" not in result.stderr


SINGLE_GPU_SERVICES = (
    "livekit-init livekit-server livekit-nvidia-llm-vllm livekit-nemo-speech-asr livekit-nemo-speech "
    "livekit-agent-single-gpu"
)


def test_wait_succeeds_when_every_container_runs_and_init_has_exited(tmp_path):
    result = run_wait(tmp_path, "healthy", timeout="5", FAKE_SERVICES=SINGLE_GPU_SERVICES)
    assert result.returncode == 0, result.stderr
    assert "Ready to connect" in result.stderr


@pytest.mark.parametrize(
    "stopped", ["livekit-nemo-speech-asr", "livekit-nemo-speech", "livekit-agent-single-gpu", "livekit-server"]
)
def test_wait_flags_a_stopped_container_even_without_a_health_check(tmp_path, stopped):
    # The speech sidecar, the agent, and the LiveKit server have no health check, so only their state shows a crash.
    # The agent's log can still contain "registered worker" after it stops.
    result = run_wait(tmp_path, "healthy", timeout="1", FAKE_SERVICES=SINGLE_GPU_SERVICES, FAKE_STOPPED=stopped)
    assert result.returncode != 0
    assert f"{stopped} is exited" in result.stderr
    assert "Ready to connect" not in result.stderr


@pytest.mark.parametrize(("init_state", "init_exit"), [("exited", "1"), ("running", "0")])
def test_wait_requires_livekit_init_to_finish_successfully(tmp_path, init_state, init_exit):
    result = run_wait(
        tmp_path,
        "healthy",
        timeout="1",
        FAKE_SERVICES=SINGLE_GPU_SERVICES,
        FAKE_INIT_STATE=init_state,
        FAKE_INIT_EXIT=init_exit,
    )
    assert result.returncode != 0
    assert "livekit-init: has not finished setting up LiveKit" in result.stderr


def test_wait_lists_stopped_containers_too(tmp_path):
    # `ps -a` includes a container that exited after startup. A plain `ps` would have skipped it.
    _, calls = run_real(tmp_path, ["wait", "single-gpu", "1"], FAKE_HEALTH="healthy")
    assert any("ps -a -q " in call for call in calls)
    assert not any(call.endswith("ps -q") for call in calls)


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
    env = clean_env()
    env.update({"DRY_RUN": "1", "ENV_FILE": str(tmp_path / "missing.env")})
    result = subprocess.run(["bash", str(SCRIPT), "up", "cloud"], capture_output=True, text=True, env=env, timeout=30)
    assert result.returncode != 0
    assert "no .env file" in result.stderr


def test_down_without_profile_stops_all_three(tmp_path):
    result = run(["down"], tmp_path)
    assert result.returncode == 0
    # Each command names its recipe plus both overlays, so a stopped stack leaves no SIP or Redis container.
    assert [profiles_of(c)[0] for c in commands(result)] == [
        "generic-assistant-livekit",
        "generic-assistant-livekit/server",
        "generic-assistant-livekit/single-gpu",
    ]
    assert all(c.endswith(" down") and profiles_of(c)[1:] == [SIP_PROFILE] for c in commands(result))


def test_clean_removes_only_the_livekit_volumes_and_keeps_the_image_by_default(tmp_path):
    result = run(["clean", "server", "-y"], tmp_path)
    assert result.returncode == 0
    command, volumes = commands(result)
    assert command.endswith(" down")
    assert profiles_of(command) == ["generic-assistant-livekit/server", SIP_PROFILE]
    assert volumes == "+ docker volume rm nemotron-voice-agent_livekit_state nemotron-voice-agent_livekit_redis"
    assert "docker image rm" not in result.stderr


def test_clean_image_flag_also_removes_the_image(tmp_path):
    result = run(["clean", "server", "-y", "--image"], tmp_path)
    assert result.returncode == 0
    lines = commands(result)
    assert lines[0].endswith(" down")
    assert profiles_of(lines[0])[0] == "generic-assistant-livekit/server"
    assert lines[2] == "+ docker image rm generic-assistant-livekit:latest"


def test_clean_without_a_profile_covers_all_three(tmp_path):
    result = run(["clean", "-y"], tmp_path)
    *stops, volumes = commands(result)
    assert len(stops) == 3
    assert all(c.endswith(" down") for c in stops)
    assert volumes == "+ docker volume rm nemotron-voice-agent_livekit_state nemotron-voice-agent_livekit_redis"


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
        (["single-gpu", "asr"], "livekit-nemo-speech-asr"),
        (["single-gpu", "tts"], "livekit-nemo-speech"),
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
    recipes = ["generic-assistant-livekit", "generic-assistant-livekit/server", "generic-assistant-livekit/single-gpu"]
    assert sorted(profiles_of(command)) == sorted([*recipes, SIP_PROFILE])


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
    assert [profiles_of(c)[0] for c in downs(calls)] == ["generic-assistant-livekit/server"]
    assert "Another profile (server) is running" in result.stderr
    assert calls.index(downs(calls)[0]) < next(
        i for i, c in enumerate(calls) if c.endswith("cloud up -d") or c.endswith("livekit up -d")
    )


def test_up_with_nothing_running_stops_nothing(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--no-wait"])
    assert result.returncode == 0, result.stderr
    assert downs(calls) == []


@pytest.mark.parametrize("version", ["2.20.5", "2.23.9", "2.24.0", "2.24.3", "1.29.2"])
def test_docker_compose_older_than_2_24_4_is_rejected(tmp_path, version):
    # The !override tag in docker-compose.yaml needs Docker Compose 2.24.4.
    result, _ = run_real(tmp_path, ["up", "cloud", "--no-wait"], FAKE_COMPOSE_VERSION=version)
    assert result.returncode != 0
    assert f"Docker Compose {version} is too old. This example needs 2.24.4 or newer" in result.stderr


@pytest.mark.parametrize("version", ["2.24.4", "v2.24.4", "2.24.10", "2.25.0", "5.5.1"])
def test_docker_compose_2_24_4_or_newer_is_accepted(tmp_path, version):
    result, _ = run_real(tmp_path, ["ps"], FAKE_COMPOSE_VERSION=version)
    assert result.returncode == 0, result.stderr


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


def up_command(calls):
    return next(c for c in calls if c.endswith(" up -d"))


def test_up_without_overlays_adds_no_overlay_profiles_and_clears_the_flags(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--no-wait"], LIVEKIT_SIP="1", LIVEKIT_TURN="1")
    assert result.returncode == 0, result.stderr
    assert profiles_of(up_command(calls)) == ["generic-assistant-livekit"]
    assert "env LIVEKIT_SIP=" in calls  # a value left in the shell must not enable an overlay


def test_up_sip_adds_the_sip_profile_and_sets_the_flag(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--sip", "--no-wait"])
    assert result.returncode == 0, result.stderr
    assert profiles_of(up_command(calls)) == ["generic-assistant-livekit", SIP_PROFILE]
    assert "env LIVEKIT_SIP=1" in calls
    assert "5060" in result.stderr  # the exposure reminder


def test_up_sip_is_refused_without_linux_host_networking(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--sip", "--no-wait"], FAKE_OS="Darwin")
    assert result.returncode != 0
    assert "only on Linux" in result.stderr
    assert not any(" up -d" in c for c in calls)


def test_up_rejects_an_unknown_option_and_lists_the_overlay_flags(tmp_path):
    result = run(["up", "cloud", "--bogus"], tmp_path)
    assert result.returncode != 0
    assert "--sip, --no-wait" in result.stderr


def test_sip_setup_requires_a_deployment_started_with_sip(tmp_path):
    result, calls = run_real(tmp_path, ["sip-setup", "cloud", "--number", "+15551234567"])
    assert result.returncode != 0
    assert "up cloud --sip" in result.stderr
    assert not any(" exec " in c for c in calls)


def test_sip_setup_runs_in_the_agent_container(tmp_path):
    result, calls = run_real(
        tmp_path, ["sip-setup", "server", "--number", "+15551234567", "--scenario", "bank"], FAKE_OVERLAYS="livekit-sip"
    )
    assert result.returncode == 0, result.stderr
    (call,) = [c for c in calls if " exec " in c]
    assert (
        "exec livekit-agent-server python -m nemotron_voice_agent.livekit.examples.generic_assistant.sip_setup trunk"
        in call
    )
    assert call.endswith("--number +15551234567 --scenario bank")


def test_sip_network_prints_guidance_in_a_throwaway_container(tmp_path):
    result, calls = run_real(tmp_path, ["sip-network", "--lan-ip", "192.168.1.9", "--lan", "192.168.1.0/24"])
    assert result.returncode == 0, result.stderr
    (call,) = [c for c in calls if " run " in c]
    assert (
        "run --rm --no-deps livekit-init python -m"
        " nemotron_voice_agent.livekit.examples.generic_assistant.sip_setup network" in call
    )
    assert call.endswith("--lan-ip 192.168.1.9 --lan 192.168.1.0/24")


@pytest.mark.parametrize(("short", "service"), [("redis", "livekit-redis"), ("sip", "livekit-sip")])
def test_logs_resolve_the_overlay_service_names(tmp_path, short, service):
    result = run(["logs", "cloud", short], tmp_path)
    assert commands(result)[0].endswith(f"logs -f --tail 50 {service}")


OVERLAY_ALL = "livekit-redis livekit-sip"


def run_check(tmp_path, args=("cloud",), **extra):
    env = fake_env(tmp_path, FAKE_OVERLAYS=OVERLAY_ALL, **extra)
    return subprocess.run(["bash", str(SCRIPT), "check", *args], capture_output=True, text=True, env=env, timeout=60)


def test_check_sip_passes_for_a_healthy_deployment(tmp_path):
    result = run_check(tmp_path, FAKE_LISTEN="6379 5060", FAKE_HEALTH="healthy", FAKE_HEALTH_SERVICES="livekit-redis")
    assert result.returncode == 0, result.stderr
    assert "PASS  Redis is published on loopback only (127.0.0.1:6379)" in result.stderr
    assert "PASS  SIP listens on port 5060" in result.stderr
    assert "arm64" in result.stderr  # the image architecture is shown
    assert "Not covered: a real call" in result.stderr


@pytest.mark.parametrize("published", ["0.0.0.0:6379", "192.168.1.20:6379", "127.0.0.1:6379\n0.0.0.0:6379"])
def test_check_sip_fails_when_redis_is_published_beyond_loopback(tmp_path, published):
    result = run_check(
        tmp_path,
        FAKE_LISTEN="6379 5060",
        FAKE_HEALTH="healthy",
        FAKE_HEALTH_SERVICES="livekit-redis",
        FAKE_REDIS_PORT=published,
    )
    assert result.returncode != 0
    assert "FAIL  Redis is published beyond loopback" in result.stderr


def test_check_sip_fails_when_the_overlay_is_not_started(tmp_path):
    env = fake_env(tmp_path, FAKE_LISTEN="")
    result = subprocess.run(
        ["bash", str(SCRIPT), "check", "cloud"], capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode != 0
    assert "FAIL  livekit-sip is missing" in result.stderr


def test_check_requires_exactly_one_known_profile(tmp_path):
    for args in ((), ("cloud", "sip"), ("bogus",)):
        assert run_check(tmp_path, args).returncode != 0
    assert "usage: run.sh check" in run_check(tmp_path, ()).stderr


def test_turn_is_no_longer_an_option(tmp_path):
    result = run(["up", "cloud", "--turn"], tmp_path)
    assert result.returncode != 0
    assert "unknown option '--turn'" in result.stderr


def test_changing_the_overlay_of_a_running_stack_restarts_it_first(tmp_path):
    result, calls = run_real(tmp_path, ["up", "cloud", "--sip", "--no-wait"], FAKE_STACK="1")
    assert result.returncode == 0, result.stderr
    assert len(downs(calls)) == 1
    assert calls.index(downs(calls)[0]) < calls.index(up_command(calls))
    assert "overlays changed" in result.stderr


def test_the_same_overlay_does_not_restart_a_running_stack(tmp_path):
    result, calls = run_real(
        tmp_path, ["up", "cloud", "--sip", "--no-wait"], FAKE_STACK="1", FAKE_OVERLAYS="livekit-sip"
    )
    assert result.returncode == 0, result.stderr
    assert downs(calls) == []


def test_down_reports_leftover_containers_and_confirms_a_clean_state(tmp_path):
    clean, _ = run_real(tmp_path, ["down"])
    assert "No containers of this example remain" in clean.stderr
    left, _ = run_real(tmp_path, ["down"], FAKE_LEFT="livekit-sip livekit-redis")
    assert "still present" in left.stderr
    assert "livekit-sip" in left.stderr


def test_clean_removes_the_overlay_containers_and_checks_nothing_is_left(tmp_path):
    result, calls = run_real(tmp_path, ["clean", "-y"])
    assert result.returncode == 0, result.stderr
    cleans = downs(calls)
    assert len(cleans) == 3
    assert all(SIP_PROFILE in c for c in cleans)
    assert not any(c.endswith(" -v") for c in calls)
    assert "No containers of this example remain" in result.stderr


def test_check_sip_fails_and_shows_the_log_when_the_sip_service_cannot_read_its_config(tmp_path):
    # A crashing container used to pass the log check, because only a few words counted as errors.
    result = run_check(
        tmp_path,
        FAKE_LISTEN="6379 5060",
        FAKE_HEALTH="healthy",
        FAKE_HEALTH_SERVICES="livekit-redis",
        FAKE_LOG_LINE="open /sip/config.yaml: no such file or directory",
    )
    assert result.returncode != 0
    assert "FAIL  the SIP log shows an error" in result.stderr
    assert "open /sip/config.yaml: no such file or directory" in result.stderr  # the cause is printed


def test_check_sip_ignores_per_call_log_lines_with_failure_counters(tmp_path):
    # A rejected call logs statistics such as "packets_failed": 0, which is not a service error.
    line = (
        'INFO sip/media.go:183 call statistics {"result": "client_error", "reason": "no-dispatch", "packets_failed": 0}'
    )
    result = run_check(
        tmp_path,
        FAKE_LISTEN="6379 5060",
        FAKE_HEALTH="healthy",
        FAKE_HEALTH_SERVICES="livekit-redis",
        FAKE_LOG_LINE=line,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS  no error in the recent SIP log" in result.stderr


def run_address(tmp_path, public="", **extra):
    env = fake_env(tmp_path, FAKE_PUBLIC_IP=public, **extra)
    return subprocess.run(["bash", str(SCRIPT), "sip-address"], capture_output=True, text=True, env=env, timeout=60)


def test_sip_address_reports_the_addresses_and_a_usable_public_address(tmp_path):
    result = run_address(tmp_path, public="203.0.113.9")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LAN address:    192.168.1.20  (interface eth0)" in result.stdout
    assert "LAN network:    192.168.1.0/24" in result.stdout
    assert "Router address: 192.168.1.1" in result.stdout
    assert "Public address: 203.0.113.9" in result.stdout
    assert "OK: 203.0.113.9 looks like a public address." in result.stdout
    assert "--lan-ip 192.168.1.20 --lan 192.168.1.0/24" in result.stdout
    assert "http://192.168.1.1" in result.stdout


@pytest.mark.parametrize("public", ["100.64.1.2", "100.100.0.9", "100.127.255.1"])
def test_sip_address_fails_for_carrier_grade_nat(tmp_path, public):
    result = run_address(tmp_path, public=public)
    assert result.returncode == 1
    assert "carrier-grade NAT" in result.stdout


@pytest.mark.parametrize("public", ["192.168.1.5", "10.1.2.3", "172.16.0.4", "172.31.9.9"])
def test_sip_address_fails_for_a_private_public_address(tmp_path, public):
    result = run_address(tmp_path, public=public)
    assert result.returncode == 1
    assert "private address" in result.stdout


@pytest.mark.parametrize("public", ["100.63.0.1", "100.128.0.1", "172.15.0.1", "172.32.0.1"])
def test_sip_address_accepts_addresses_just_outside_the_reserved_ranges(tmp_path, public):
    assert run_address(tmp_path, public=public).returncode == 0


def test_sip_address_fails_when_the_public_address_cannot_be_read(tmp_path):
    result = run_address(tmp_path, public="")
    assert result.returncode == 1
    assert "could not read the public address" in result.stdout


def test_sip_address_refuses_a_non_linux_host(tmp_path):
    result = run_address(tmp_path, public="203.0.113.9", FAKE_OS="Darwin")
    assert result.returncode != 0
    assert "Linux host" in result.stderr
