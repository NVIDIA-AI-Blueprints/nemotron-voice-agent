# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# ruff: noqa: D100, D101, D102

import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pipecat.runner.types import EvalRunnerArguments

import eval_bot


class EvalBotAttachmentTests(unittest.TestCase):
    def _body_for_path(self, path: Path | str) -> dict:
        return {
            "eval_attachment": {
                "path": str(path),
                "kind": "image",
                "content_type": "image/png",
                "name": "omni-assistant-architecture.png",
            }
        }

    def test_preload_eval_attachment_allows_fixture_root_file(self) -> None:
        fixture = eval_bot._EVAL_ATTACHMENT_FIXTURE_ROOT / "omni-assistant-architecture.png"

        with patch("eval_bot.store_attachment") as store_attachment:
            eval_bot._preload_eval_attachment(self._body_for_path(fixture), "eval-session")

        store_attachment.assert_called_once()
        kwargs = store_attachment.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "eval-session")
        self.assertEqual(kwargs["kind"], "image")
        self.assertEqual(kwargs["name"], fixture.name)
        self.assertEqual(kwargs["content_type"], "image/png")
        self.assertEqual(kwargs["data"], fixture.read_bytes())

    def test_preload_eval_attachment_resolves_relative_path_from_runner_body_parent(self) -> None:
        fixture = eval_bot._EVAL_ATTACHMENT_FIXTURE_ROOT / "omni-assistant-architecture.png"
        runner_args = EvalRunnerArguments()
        runner_args.cli_args = Namespace(runner_body=str(fixture.parent / "manual-body.json"))

        with (
            TemporaryDirectory() as other_cwd,
            patch("eval_bot.Path.cwd", return_value=Path(other_cwd)),
            patch("eval_bot.store_attachment") as store_attachment,
        ):
            eval_bot._preload_eval_attachment(
                self._body_for_path(fixture.name),
                "eval-session",
                runner_args,
            )

        store_attachment.assert_called_once()
        self.assertEqual(store_attachment.call_args.kwargs["data"], fixture.read_bytes())

    def test_preload_eval_attachment_resolves_relative_path_from_cwd_without_runner_body(self) -> None:
        fixture = eval_bot._EVAL_ATTACHMENT_FIXTURE_ROOT / "omni-assistant-architecture.png"

        with (
            patch("eval_bot.Path.cwd", return_value=fixture.parent),
            patch("eval_bot.store_attachment") as store_attachment,
        ):
            eval_bot._preload_eval_attachment(self._body_for_path(fixture.name), "eval-session")

        store_attachment.assert_called_once()
        self.assertEqual(store_attachment.call_args.kwargs["data"], fixture.read_bytes())

    def test_preload_eval_attachment_rejects_path_outside_fixture_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be under"):
            eval_bot._preload_eval_attachment(self._body_for_path(Path(__file__).resolve()), "eval-session")

    def test_preload_eval_attachment_rejects_non_regular_fixture_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "regular file"):
            eval_bot._preload_eval_attachment(
                self._body_for_path(eval_bot._EVAL_ATTACHMENT_FIXTURE_ROOT), "eval-session"
            )

    def test_preload_eval_attachment_rejects_oversized_file(self) -> None:
        fixture = eval_bot._EVAL_ATTACHMENT_FIXTURE_ROOT / "omni-assistant-architecture.png"

        with (
            patch("eval_bot._MAX_EVAL_ATTACHMENT_BYTES", 1),
            self.assertRaisesRegex(ValueError, "exceeds"),
        ):
            eval_bot._preload_eval_attachment(self._body_for_path(fixture), "eval-session")


if __name__ == "__main__":
    unittest.main()
