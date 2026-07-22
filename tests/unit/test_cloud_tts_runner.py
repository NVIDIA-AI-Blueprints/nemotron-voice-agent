# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import json
import subprocess
import tempfile
import unittest
from importlib import util
from pathlib import Path
from unittest.mock import call, patch

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "pipecat_evals" / "service" / "run_cloud_tts_evals.py"
_SPEC = util.spec_from_file_location("run_cloud_tts_evals", _RUNNER_PATH)
assert _SPEC and _SPEC.loader
run_cloud_tts_evals = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_cloud_tts_evals)


class CloudTtsRunnerTests(unittest.TestCase):
    def test_yaml_runner_body_resolves_repo_root_relative_attachment_path(self) -> None:
        body = run_cloud_tts_evals._runner_body_from_yaml(
            run_cloud_tts_evals.ROOT / "tests/pipecat_evals/service/runner_bodies/cloud_tts.yaml",
            "omni_uploaded_image",
        )

        self.assertEqual(
            body["eval_attachment"]["path"],
            str(
                (
                    run_cloud_tts_evals.ROOT / "src/examples/omni_assistant/images/omni-assistant-architecture.png"
                ).resolve()
            ),
        )

    def test_materialize_manifest_writes_yaml_runner_body_with_resolved_attachment_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "cloud_tts_manifest.yaml"
            bodies_dir = root / "runner_bodies"
            bodies_dir.mkdir()
            (bodies_dir / "assets").mkdir()
            body_yaml = bodies_dir / "cloud_tts.yaml"
            body_yaml.write_text(
                """
defaults:
  llm_id: default-llm
  tts_voice_id: default-voice
bodies:
  omni_uploaded_image:
    pipeline_mode: omni-assistant-subagents
    session_id: eval-omni-uploaded-image-cloud-tts
    eval_attachment:
      path: assets/diagram.png
      kind: image
      content_type: image/png
      name: diagram.png
""".lstrip(),
                encoding="utf-8",
            )
            manifest.write_text(
                """
bots_dir: ../../src
scenarios_dir: scenarios
runs_dir: ../../eval-runs
suite:
  - bot: eval_bot.py
    runner_body: runner_bodies/cloud_tts.yaml
    runner_body_key: omni_uploaded_image
    scenarios:
      - omni_uploaded_image_path
""".lstrip(),
                encoding="utf-8",
            )

            generated_manifest = run_cloud_tts_evals._materialize_manifest(manifest, root / "generated")

            generated = run_cloud_tts_evals._load_yaml_mapping(generated_manifest, "generated manifest")
            materialized_body = Path(generated["suite"][0]["runner_body"])
            body = json.loads(materialized_body.read_text(encoding="utf-8"))

        self.assertEqual(generated["bots_dir"], str((manifest.parent / "../../src").resolve()))
        self.assertEqual(generated["scenarios_dir"], str((manifest.parent / "scenarios").resolve()))
        self.assertEqual(generated["runs_dir"], str((manifest.parent / "../../eval-runs").resolve()))
        self.assertEqual(body["llm_id"], "default-llm")
        self.assertEqual(body["pipeline_mode"], "omni-assistant-subagents")
        self.assertEqual(body["tts_voice_id"], "default-voice")
        self.assertEqual(body["eval_attachment"]["path"], str((body_yaml.parent / "assets/diagram.png").resolve()))

    def test_run_scenario_returns_timeout_status_when_child_hangs(self) -> None:
        with (
            patch.object(run_cloud_tts_evals, "_materialize_manifest", return_value=Path("/tmp/generated.yaml")),
            patch.object(
                run_cloud_tts_evals.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=62),
            ) as subprocess_run,
        ):
            code = run_cloud_tts_evals._run_scenario(
                Path("manifest.yaml"),
                "generic_initial_greeting",
                name_prefix="cloud-tts",
                timeout=2,
                attempt=1,
            )

        self.assertEqual(code, 124)
        self.assertEqual(subprocess_run.call_args.kwargs["timeout"], 62)

    def test_main_retries_nonzero_scenario_once_then_passes(self) -> None:
        with (
            patch("sys.argv", ["run_cloud_tts_evals.py", "--retries", "1", "--retry-delay", "0"]),
            patch.object(run_cloud_tts_evals, "_scenario_names", return_value=["generic_initial_greeting"]),
            patch.object(run_cloud_tts_evals, "_run_scenario", side_effect=[1, 0]) as run_scenario,
            patch.object(run_cloud_tts_evals.time, "sleep") as sleep,
        ):
            code = run_cloud_tts_evals.main()

        self.assertEqual(code, 0)
        self.assertEqual(run_scenario.call_count, 2)
        sleep.assert_called_once_with(0)

    def test_main_reports_failure_after_retries_are_exhausted(self) -> None:
        with (
            patch("sys.argv", ["run_cloud_tts_evals.py", "--retries", "1", "--retry-delay", "0"]),
            patch.object(run_cloud_tts_evals, "_scenario_names", return_value=["generic_initial_greeting"]),
            patch.object(run_cloud_tts_evals, "_run_scenario", return_value=1) as run_scenario,
            patch.object(run_cloud_tts_evals.time, "sleep") as sleep,
        ):
            code = run_cloud_tts_evals.main()

        self.assertEqual(code, 1)
        self.assertEqual(run_scenario.call_count, 2)
        sleep.assert_has_calls([call(0)])

    def test_parse_args_rejects_negative_timeout(self) -> None:
        with (
            patch("sys.argv", ["run_cloud_tts_evals.py", "--timeout", "-1"]),
            self.assertRaises(SystemExit) as exc,
        ):
            run_cloud_tts_evals._parse_args()

        self.assertEqual(exc.exception.code, 2)

    def test_parse_args_rejects_negative_retries(self) -> None:
        with (
            patch("sys.argv", ["run_cloud_tts_evals.py", "--retries", "-1"]),
            self.assertRaises(SystemExit) as exc,
        ):
            run_cloud_tts_evals._parse_args()

        self.assertEqual(exc.exception.code, 2)

    def test_parse_args_rejects_negative_retry_delay(self) -> None:
        with (
            patch("sys.argv", ["run_cloud_tts_evals.py", "--retry-delay", "-1"]),
            self.assertRaises(SystemExit) as exc,
        ):
            run_cloud_tts_evals._parse_args()

        self.assertEqual(exc.exception.code, 2)

    def test_parse_args_rejects_non_finite_retry_delay(self) -> None:
        for value in ("nan", "inf"):
            with (
                self.subTest(value=value),
                patch("sys.argv", ["run_cloud_tts_evals.py", "--retry-delay", value]),
                self.assertRaises(SystemExit) as exc,
            ):
                run_cloud_tts_evals._parse_args()

            self.assertEqual(exc.exception.code, 2)

    def test_parse_args_accepts_zero_retry_values(self) -> None:
        with patch("sys.argv", ["run_cloud_tts_evals.py", "--retries", "0", "--retry-delay", "0"]):
            args = run_cloud_tts_evals._parse_args()

        self.assertEqual(args.retries, 0)
        self.assertEqual(args.retry_delay, 0)


if __name__ == "__main__":
    unittest.main()
