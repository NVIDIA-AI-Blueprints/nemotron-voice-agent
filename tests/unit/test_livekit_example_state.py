# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D103

from examples.generic_assistant_livekit.state import read_json, write_json_atomic


def test_round_trip_leaves_no_temporary_files(tmp_path):
    path = tmp_path / "state" / "deployment.json"
    write_json_atomic(path, {"recipe": "server"})
    assert read_json(path) == {"recipe": "server"}
    write_json_atomic(path, {"recipe": "cloud"})
    assert read_json(path) == {"recipe": "cloud"}
    assert [p.name for p in path.parent.iterdir()] == ["deployment.json"]


def test_missing_or_broken_files_read_as_none(tmp_path):
    assert read_json(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{nope")
    assert read_json(tmp_path / "bad.json") is None
