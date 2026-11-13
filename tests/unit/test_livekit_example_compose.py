# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "src/examples/generic_assistant_livekit/docker-compose.yaml"
AGENTS = ("livekit-agent", "livekit-agent-server", "livekit-agent-single-gpu")


class Loader(yaml.SafeLoader):
    """Parse the compose file without Docker: keep the value that follows a custom tag such as `!override`."""


Loader.add_multi_constructor("!", lambda loader, suffix, node: loader.construct_sequence(node))


@pytest.fixture(scope="module")
def services():
    return yaml.load(COMPOSE.read_text(), Loader=Loader)["services"]


def environment(service):
    return service.get("environment") or {}


@pytest.mark.parametrize("name", AGENTS)
def test_agents_receive_only_the_variables_they_use(services, name):
    assert "env_file" not in services[name]  # not the whole .env
    assert "HF_TOKEN" not in environment(services[name])


def test_only_the_cloud_agent_receives_the_nvidia_api_key(services):
    holders = [name for name, service in services.items() if "NVIDIA_API_KEY" in environment(service)]
    assert holders == ["livekit-agent"]


def test_the_ui_and_the_livekit_server_get_no_model_credentials(services):
    for name in ("livekit-ui", "livekit-server"):
        assert not {"NVIDIA_API_KEY", "HF_TOKEN"} & set(environment(services[name]))
        assert "env_file" not in services[name]


def test_livekit_and_ui_ports_default_to_loopback(services):
    ports = services["livekit-server"]["ports"] + services["livekit-ui"]["ports"]
    assert ports and all(
        port.startswith(("${LIVEKIT_BIND:-127.0.0.1}:", "${LIVEKIT_UI_BIND:-127.0.0.1}:")) for port in ports
    )


def test_every_profile_has_exactly_one_agent_and_the_sidecars_are_not_shared_with_pipecat_profiles(services):
    for name in AGENTS:
        assert len(services[name]["profiles"]) == 1
    for name, service in services.items():
        if service.get("extends"):  # a sidecar: its profile list replaces the inherited Pipecat profiles
            assert all(profile.startswith("generic-assistant-livekit/") for profile in service["profiles"]), name


def test_agents_share_the_livekit_server_network_for_media(services):
    for name in AGENTS:
        assert services[name]["network_mode"] == "service:livekit-server"


def test_the_ui_container_binds_all_interfaces_so_docker_can_publish_the_port(services):
    # The server itself defaults to loopback. Compose publishes the container port on 127.0.0.1 by default.
    assert environment(services["livekit-ui"])["LIVEKIT_UI_HOST"] == "0.0.0.0"  # noqa: S104


def test_the_vllm_sidecar_does_not_inherit_the_repository_env_file(services):
    # `extends` would otherwise keep `env_file: ../.env` from docker/, which ignores the .env that run.sh selected.
    vllm = services["livekit-nvidia-llm-vllm"]
    assert vllm["env_file"] == []
    assert set(environment(vllm)) == {"HF_TOKEN", "VLLM_VRAM_HEADROOM_MIB", "VLLM_GPU_MEMORY_UTILIZATION"}


# optional overlay: SIP (with Redis)


SIP_PROFILE = "generic-assistant-livekit/sip"


def test_overlay_services_belong_only_to_their_own_profile(services):
    assert services["livekit-redis"]["profiles"] == [SIP_PROFILE]
    assert services["livekit-sip"]["profiles"] == [SIP_PROFILE]


def test_overlays_are_not_part_of_any_recipe_profile(services):
    recipes = {"generic-assistant-livekit", "generic-assistant-livekit/server", "generic-assistant-livekit/single-gpu"}
    for name in ("livekit-redis", "livekit-sip"):
        assert not recipes & set(services[name]["profiles"])


def test_redis_is_published_on_loopback_only(services):
    assert services["livekit-redis"]["ports"] == ["127.0.0.1:6379:6379"]


def test_the_livekit_server_waits_for_redis_only_when_the_sip_profile_is_on(services):
    redis = services["livekit-server"]["depends_on"]["livekit-redis"]
    assert redis == {"condition": "service_healthy", "required": False}


def test_sip_uses_host_networking_and_the_images_are_pinned(services):
    assert services["livekit-sip"]["network_mode"] == "host"
    assert services["livekit-sip"]["image"].startswith("livekit/sip:v")
    assert services["livekit-redis"]["image"].startswith("redis:7.")


def test_overlay_containers_read_their_config_from_the_state_volume_read_only(services):
    assert "livekit_state:/sip:ro" in services["livekit-sip"]["volumes"]  # the image reads /sip/config.yaml
    assert services["livekit-sip"]["environment"]["SIP_CONFIG_FILE"] == "/sip/config.yaml"


def test_overlay_containers_get_no_model_credentials(services):
    for name in ("livekit-sip", "livekit-redis"):
        assert not {"NVIDIA_API_KEY", "HF_TOKEN"} & set(environment(services[name]))
        assert "env_file" not in services[name]


def test_the_init_service_receives_the_overlay_switch_and_there_is_no_relay_service(services):
    assert "LIVEKIT_SIP" in environment(services["livekit-init"])
    assert not [name for name in services if "turn" in name]
