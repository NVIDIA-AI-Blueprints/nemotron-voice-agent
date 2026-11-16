# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import stat

import pytest
import yaml

from examples.generic_assistant_livekit import credentials, sip_setup, telephony


def test_sip_config_points_at_loopback_services_and_a_small_rtp_range():
    config = yaml.safe_load(telephony.sip_config_yaml("key", "secret"))
    assert config["ws_url"] == "ws://127.0.0.1:7880"
    assert config["redis"]["address"] == "127.0.0.1:6379"
    assert config["sip_port"] == 5060
    assert config["rtp_port"] == "10000-10100"
    assert config["api_key"] == "key"
    assert config["use_external_ip"] is True


def test_sip_config_quotes_values_that_would_break_yaml():
    config = yaml.safe_load(telephony.sip_config_yaml("a: b", "#x"))
    assert (config["api_key"], config["api_secret"]) == ("a: b", "#x")


def test_cidrs_are_validated_and_open_ranges_refused():
    assert telephony.parse_cidrs(["54.172.60.0/30"], "x") == ["54.172.60.0/30"]
    for bad in ("not-an-ip", "0.0.0.0/0", "::/0"):
        with pytest.raises(ValueError):
            telephony.parse_cidrs([bad], "x")


def test_firewall_commands_are_always_restricted_by_source():
    commands = telephony.firewall_commands(["203.0.113.0/24"], ["198.51.100.0/24"], "192.168.1.0/24")
    joined = "\n".join(commands)
    assert "allow from 203.0.113.0/24 to any port 5060 proto udp" in joined
    assert "allow from 203.0.113.0/24 to any port 5060 proto tcp" in joined
    assert "allow from 198.51.100.0/24 to any port 10000:10100 proto udp" in joined
    assert not [c for c in commands if c.startswith("sudo ufw allow ") and " from " not in c]
    assert commands[0] == "sudo ufw default deny incoming"


def test_the_checklist_never_suggests_forwarding_redis_or_ssh():
    text = telephony.network_checklist("192.168.1.9")
    assert "Never forward 22" in text
    assert "6379" in text.split("Never forward")[1]
    assert "192.168.1.9" in text


def livekit_api():
    # The LiveKit SDK belongs to the example's own environment, not the repository's, so these tests skip without it.
    return pytest.importorskip("livekit.api")


def test_build_requests_makes_an_idempotent_named_trunk_and_rule():
    livekit_api()
    trunk, rule, attributes = sip_setup.build_requests("+15551234567", "bank", None, ["203.0.113.0/24"])
    assert trunk.name == "inbound-+15551234567"
    assert list(trunk.numbers) == ["+15551234567"]
    assert list(trunk.allowed_addresses) == ["203.0.113.0/24"]
    assert rule.name == "call-+15551234567"
    assert rule.rule.dispatch_rule_individual.room_prefix == "call-"
    assert attributes == {"scenario": "bank"}
    assert dict(rule.attributes) == attributes


def test_changing_the_use_case_reuses_the_rule_of_the_number():
    livekit_api()
    _, bank, _ = sip_setup.build_requests("+15551234567", "bank", None, [])
    _, generic, _ = sip_setup.build_requests("+15551234567", "generic", None, [])
    assert bank.name == generic.name


def test_find_rule_matches_by_name_or_by_trunk_so_old_rule_names_are_replaced():
    livekit_api()
    from livekit import api

    old = api.SIPDispatchRuleInfo(sip_dispatch_rule_id="SDR_old", name="call-bank-+15551234567", trunk_ids=["ST_1"])
    other = api.SIPDispatchRuleInfo(sip_dispatch_rule_id="SDR_other", name="call-+15559999999", trunk_ids=["ST_2"])
    assert sip_setup.find_rule([other, old], "call-+15551234567", "ST_1").sip_dispatch_rule_id == "SDR_old"
    assert sip_setup.find_rule([other], "call-+15551234567", "ST_1") is None
    assert sip_setup.find_rule([other], "call-+15559999999", "ST_9").sip_dispatch_rule_id == "SDR_other"


def test_build_requests_carries_the_voice():
    livekit_api()
    *_, attributes = sip_setup.build_requests("+15551234567", "generic", "Magpie.EN-US.Aria", [])
    assert attributes == {"scenario": "generic", "voice": "Magpie.EN-US.Aria"}


@pytest.mark.parametrize("number", ["5551234567", "+0123456", "+1555", "+1555123456789012", "tel:+15551234567", ""])
def test_build_requests_rejects_non_e164_numbers(number):
    livekit_api()
    with pytest.raises(ValueError, match="E.164"):
        sip_setup.build_requests(number, "generic", None, [])


def test_build_requests_rejects_unknown_use_cases_and_bad_voices():
    livekit_api()
    with pytest.raises(ValueError, match="unknown use case"):
        sip_setup.build_requests("+15551234567", "banking", None, [])
    with pytest.raises(ValueError, match="voice"):
        sip_setup.build_requests("+15551234567", "generic", "a b; rm -rf", [])


def test_the_network_command_prints_commands_and_changes_nothing(capsys):
    sip_setup.main(["network", "--lan-ip", "192.168.1.9", "--lan", "192.168.1.0/24", "--signaling", "203.0.113.0/24"])
    out = capsys.readouterr().out
    assert "sudo ufw allow from 203.0.113.0/24 to any port 5060 proto udp" in out
    assert "10000:10100" in out  # media defaults to the signaling ranges


def test_the_network_command_requires_the_provider_ranges(capsys):
    with pytest.raises(SystemExit):
        sip_setup.main(["network", "--lan-ip", "192.168.1.9", "--lan", "192.168.1.0/24"])
    with pytest.raises(SystemExit, match="every address"):
        sip_setup.main(["network", "--lan-ip", "192.168.1.9", "--lan", "192.168.1.0/24", "--signaling", "0.0.0.0/0"])


# credentials: the overlay files


def test_without_the_overlay_no_redis_or_sip_files_are_written(tmp_path):
    credentials.load_or_create(tmp_path, env={})
    assert "redis" not in yaml.safe_load((tmp_path / credentials.SERVER_CONFIG_FILE).read_text())
    assert sorted(p.name for p in tmp_path.iterdir()) == ["credentials.json", "livekit.yaml"]


def test_sip_adds_redis_to_the_server_config_and_writes_the_sip_config(tmp_path):
    creds = credentials.load_or_create(tmp_path, env={"LIVEKIT_SIP": "1"})
    server = yaml.safe_load((tmp_path / credentials.SERVER_CONFIG_FILE).read_text())
    assert server["redis"]["address"] == "livekit-redis:6379"  # the server is on the Compose network
    sip = yaml.safe_load((tmp_path / telephony.SIP_CONFIG_FILE).read_text())
    assert sip["api_key"] == creds["api_key"]
    assert sip["redis"]["address"] == "127.0.0.1:6379"  # the SIP container is on the host network


def test_turning_sip_off_removes_the_stale_files_and_the_redis_setting(tmp_path):
    credentials.load_or_create(tmp_path, env={"LIVEKIT_SIP": "1"})
    credentials.load_or_create(tmp_path, env={})
    assert not (tmp_path / telephony.SIP_CONFIG_FILE).exists()
    assert "redis" not in yaml.safe_load((tmp_path / credentials.SERVER_CONFIG_FILE).read_text())


def test_the_credentials_are_private_and_the_sip_config_is_readable_by_its_container(tmp_path):
    credentials.load_or_create(tmp_path, env={"LIVEKIT_SIP": "1"})
    mode = lambda name: stat.S_IMODE((tmp_path / name).stat().st_mode)  # noqa: E731
    assert mode(credentials.CREDENTIALS_FILE) == 0o600
    assert mode(telephony.SIP_CONFIG_FILE) == 0o644  # the SIP image may not run as root


def test_phone_numbers_list_only_numbers_that_a_rule_maps():
    livekit_api()
    from livekit import api

    from examples.generic_assistant_livekit import ui_server

    mapped = api.SIPInboundTrunkInfo(sip_trunk_id="ST_1", numbers=["+15551234567"])
    unmapped = api.SIPInboundTrunkInfo(sip_trunk_id="ST_2", numbers=["+15559999999"])
    rule = api.SIPDispatchRuleInfo(trunk_ids=["ST_1"], attributes={"scenario": "bank"})
    found = ui_server.phone_numbers([mapped, unmapped], [rule], {"bank": "Bank assistant"})
    assert found == [{"number": "+15551234567", "scenario": "bank", "name": "Bank assistant"}]


def test_phone_info_is_empty_without_sip(monkeypatch):
    livekit_api()
    from examples.generic_assistant_livekit import ui_server

    monkeypatch.delenv("LIVEKIT_SIP", raising=False)
    assert ui_server.phone_info() == {"enabled": False, "numbers": []}


def test_phone_info_survives_an_unreachable_server(monkeypatch):
    livekit_api()
    from examples.generic_assistant_livekit import ui_server

    async def fail():
        raise OSError("no route")

    monkeypatch.setenv("LIVEKIT_SIP", "1")
    monkeypatch.setattr(ui_server, "_list_sip_records", fail)
    monkeypatch.setitem(ui_server._phone_cache, "at", 0.0)
    assert ui_server.phone_info() == {"enabled": True, "numbers": []}


def test_firewall_commands_keep_ssh_from_other_networks_that_the_user_lists():
    commands = telephony.firewall_commands(["203.0.113.0/24"], ["198.51.100.0/24"], "192.168.1.0/24", ["10.8.0.0/16"])
    assert "sudo ufw allow from 10.8.0.0/16 to any port 22 proto tcp   # SSH from another network" in commands
    enable = commands.index("sudo ufw enable")
    assert (
        commands.index("sudo ufw allow from 10.8.0.0/16 to any port 22 proto tcp   # SSH from another network") < enable
    )
    assert not [
        c for c in telephony.firewall_commands(["203.0.113.0/24"], [], "192.168.1.0/24") if "another network" in c
    ]


def test_the_checklist_warns_about_ssh_and_the_restart_after_an_address_change():
    text = telephony.network_checklist("192.168.1.9")
    assert "--ssh-from" in text
    assert "restart livekit-sip" in text
    assert "transport=udp" in text


def test_a_dispatch_rule_shared_with_other_trunks_is_never_replaced():
    livekit_api()
    from livekit import api

    mine = api.SIPDispatchRuleInfo(trunk_ids=["ST_1"])
    shared = api.SIPDispatchRuleInfo(trunk_ids=["ST_1", "ST_2"])
    for_everyone = api.SIPDispatchRuleInfo()  # no trunk ids: it serves every trunk
    assert not sip_setup.shared_with_others(mine, "ST_1")
    assert sip_setup.shared_with_others(shared, "ST_1")
    assert sip_setup.shared_with_others(for_everyone, "ST_1")
    assert sip_setup.shared_with_others(mine, "ST_9")  # a rule that belongs to some other trunk
