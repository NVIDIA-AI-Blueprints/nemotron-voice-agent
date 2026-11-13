# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

import json
import stat

import pytest
import yaml

from examples.generic_assistant_livekit import credentials


def test_generates_credentials_and_server_config(tmp_path):
    creds = credentials.load_or_create(tmp_path, env={})
    assert creds["api_key"].startswith("lk")
    assert len(creds["api_secret"]) >= 32
    config = yaml.safe_load((tmp_path / credentials.SERVER_CONFIG_FILE).read_text())
    assert config["keys"] == {creds["api_key"]: creds["api_secret"]}
    assert config["port"] == 7880
    assert config["rtc"]["udp_port"] == 7882


def test_files_are_private(tmp_path):
    credentials.load_or_create(tmp_path, env={})
    for name in (credentials.CREDENTIALS_FILE, credentials.SERVER_CONFIG_FILE):
        assert stat.S_IMODE((tmp_path / name).stat().st_mode) == 0o600


def test_credentials_persist_across_runs(tmp_path):
    first = credentials.load_or_create(tmp_path, env={})
    assert credentials.load_or_create(tmp_path, env={}) == first


def test_environment_overrides_stored_credentials(tmp_path):
    credentials.load_or_create(tmp_path, env={})
    mine = {"LIVEKIT_API_KEY": "mykey", "LIVEKIT_API_SECRET": "mysecret"}
    assert credentials.load_or_create(tmp_path, env=mine) == {"api_key": "mykey", "api_secret": "mysecret"}
    stored = json.loads((tmp_path / credentials.CREDENTIALS_FILE).read_text())
    assert stored["api_key"] == "mykey"


def test_empty_environment_values_are_ignored(tmp_path):
    creds = credentials.load_or_create(tmp_path, env={"LIVEKIT_API_KEY": "", "LIVEKIT_API_SECRET": ""})
    assert creds["api_key"].startswith("lk")


def test_apply_to_environ_reads_the_volume(tmp_path):
    creds = credentials.load_or_create(tmp_path, env={})
    environ = {}
    credentials.apply_to_environ(tmp_path, environ)
    assert environ == {"LIVEKIT_API_KEY": creds["api_key"], "LIVEKIT_API_SECRET": creds["api_secret"]}


def test_apply_to_environ_keeps_explicit_values(tmp_path):
    environ = {"LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s"}
    credentials.apply_to_environ(tmp_path, environ)
    assert environ == {"LIVEKIT_API_KEY": "k", "LIVEKIT_API_SECRET": "s"}


def test_apply_to_environ_fails_clearly_without_init(tmp_path):
    with pytest.raises(RuntimeError, match="livekit-init"):
        credentials.apply_to_environ(tmp_path, {})


def test_setting_only_one_of_key_and_secret_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Only LIVEKIT_API_KEY is set"):
        credentials.load_or_create(tmp_path, env={"LIVEKIT_API_KEY": "mykey"})
    with pytest.raises(ValueError, match="Only LIVEKIT_API_SECRET is set"):
        credentials.load_or_create(tmp_path, env={"LIVEKIT_API_SECRET": "mysecret"})


def test_a_damaged_credentials_file_is_replaced_with_a_new_pair(tmp_path):
    for damaged in ("{not json", json.dumps({"api_key": "only-key"}), json.dumps([1, 2]), ""):
        (tmp_path / credentials.CREDENTIALS_FILE).write_text(damaged)
        creds = credentials.load_or_create(tmp_path, env={})
        assert creds["api_key"].startswith("lk") and len(creds["api_secret"]) >= 32
        assert json.loads((tmp_path / credentials.CREDENTIALS_FILE).read_text()) == creds
