# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, D103

"""The FBV package keeps the recorded v1 baseline (``fbv-v1-baseline``, ``fe7622d``) where it must be exact.

Plan: ``misc/frontend-backend-verdict/shared-stack-plan.md`` §9.1 and §10.1. The merged profile configs,
the ``agent.configure`` arguments and rendered prompts on the five Tau fixtures (N4, now through the
Pipecat adapter) and the decoded input PCM (B6) must equal v1's.

The ``eval`` profile is v1's default Tau arm (``tau3_eval_frontend_verdict_speak_history``) with its
``extends`` chain flattened into one file (``misc/frontend-backend-verdict/ui-transport-plan.md``,
section 4.1), so it is compared with that arm's fixture entry, ignoring ``source_files`` (the list
of files the chain was read from). The other v1 arms were deleted with their fixture entries.
The scenario fixtures are compared at decision level by ``test_fbv_pipecat_wire_parity`` (G2).

With ``FBV_GOLDEN_CAPTURE=1`` the test writes these fixtures instead of comparing (each capture runs
twice and fails if the two normalized runs differ). The fixtures are v1's: capture only from the
``fbv-v1-baseline`` tag, never to absorb a change. Both modes run with a pinned environment, so no shell
variable or credential reaches them.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any

from _fbv_core_fakes import BACKEND_ONLY_PROFILE
from _fbv_golden import (
    GOLDEN_DIR,
    assert_no_credentials,
    configure_captures,
    merged_profile_configs,
    pcm_capture,
    pinned_environment,
)
from _fbv_pipecat_fakes import PipecatHarness

from examples.frontend_backend_verdict.core.config.profiles import REALTIME_PROFILE, profile_path
from examples.frontend_backend_verdict.core.config.voice import load_voice_config

CAPTURE = os.environ.get("FBV_GOLDEN_CAPTURE") == "1"
#: v1's default Tau arm: the ``eval`` profile's name in the v1 fixtures.
V1_DEFAULT_ARM = "tau3_eval_frontend_verdict_speak_history"
#: Settings only v1's own host read (its standalone server, segmenter, output pacing and Riva
#: stream pools); they were removed from the config with that host. The merged configs must
#: equal v1's apart from these keys.
REMOVED_WITH_V1_HOST: dict[str, frozenset[str]] = {
    "server": frozenset({"host", "port", "path", "require_bearer", "bearer_token"}),
    "audio": frozenset({"output_chunk_ms", "pace_output", "pace_lead_ms"}),
    "turn_detection": frozenset({"vad", "min_speech_ms"}),
    "asr": frozenset({"max_concurrent_streams"}),
    "tts": frozenset({"max_concurrent_streams"}),
    "logging": frozenset({"level", "log_wire"}),
}


def make_harness(**kwargs: Any) -> PipecatHarness:
    return PipecatHarness(**kwargs)


def served_profiles() -> dict[str, Path]:
    """The profiles with a v1 fixture entry, under their v1 names."""
    return {V1_DEFAULT_ARM: profile_path(REALTIME_PROFILE), "backend_only": BACKEND_ONLY_PROFILE}


def _without_source_files(configs: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in config.items() if key != "source_files"} for name, config in configs.items()
    }


def _without_v1_host_keys(configs: dict[str, Any]) -> dict[str, Any]:
    """v1's merged configs without :data:`REMOVED_WITH_V1_HOST` (each key must still be there) and ``source_files``."""
    out: dict[str, Any] = {}
    for name, config in configs.items():
        config = dict(config)
        for section, keys in REMOVED_WITH_V1_HOST.items():
            assert keys <= set(config[section]), f"{name}.{section}: {sorted(keys - set(config[section]))}"
            config[section] = {key: value for key, value in config[section].items() if key not in keys}
        out[name] = config
    return _without_source_files(out)


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


class GoldenTestCase(unittest.IsolatedAsyncioTestCase):
    def check(self, name: str, payload: Any, second: Any = None, *, expected_view: Any = None) -> None:
        assert_no_credentials(payload)
        path = GOLDEN_DIR / f"{name}.json"
        if CAPTURE:
            if second is not None and second != payload:
                self.fail(f"{name}: two captures differ; a nondeterministic field must be normalized first")
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(_dump(payload), encoding="utf-8")
            return
        self.assertTrue(path.exists(), f"missing golden fixture {path.name}; capture with FBV_GOLDEN_CAPTURE=1")
        expected = json.loads(path.read_text(encoding="utf-8"))
        if expected_view is not None:
            expected = expected_view(expected)
        self.assertEqual(json.loads(_dump(payload)), expected, f"{name} differs from the v1 golden fixture")


class ConfigGoldenTests(GoldenTestCase):
    async def test_merged_profile_configs(self) -> None:
        with pinned_environment():
            first = _without_source_files(merged_profile_configs(served_profiles()))
            second = _without_source_files(merged_profile_configs(served_profiles())) if CAPTURE else None
        self.check("merged_profile_configs", first, second, expected_view=_without_v1_host_keys)

    async def test_configure_and_rendered_prompts(self) -> None:
        with pinned_environment():
            config = load_voice_config(profile_path(REALTIME_PROFILE))
            first = await configure_captures(make_harness, config)
            second = await configure_captures(make_harness, config) if CAPTURE else None
        self.check("configure_and_prompts", first, second)

    async def test_pcm_decode(self) -> None:
        self.check("pcm_decode", pcm_capture(), pcm_capture() if CAPTURE else None)


if __name__ == "__main__":
    unittest.main()
