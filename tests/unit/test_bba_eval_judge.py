# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: D101, D102

"""Tests for the judge request formats of benchmarking_tools/AA-BigBenchAudio-Eval/eval.py.

The default Bedrock request must stay byte-for-byte what it was; the OpenAI-compatible request may change
only the wire format (URL, ``model`` field, response schema), never the prompt, the sampling parameters,
or the label parsing.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

EVAL_PATH = Path(__file__).resolve().parents[2] / "benchmarking_tools" / "AA-BigBenchAudio-Eval" / "eval.py"
SPEC = importlib.util.spec_from_file_location("bba_eval", EVAL_PATH)
bba_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bba_eval)

QUESTION, OFFICIAL, CANDIDATE = "How many days are in a week?", "7", "The answer is seven."
JUDGE_MODEL = "aws/anthropic/bedrock-claude-sonnet-5"


def _response(body: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = body
    resp.raise_for_status.return_value = None
    return resp


class _JudgeEndpoint(unittest.TestCase):
    """Patches the endpoint, the key, and requests.post; records each request."""

    url = "https://judge.example/v1"

    def setUp(self):
        for name, value in (("API_URL", self.url), ("API_KEY", "test-key")):
            patcher = patch.object(bba_eval, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        post = patch.object(bba_eval.requests, "post")
        self.post = post.start()
        self.addCleanup(post.stop)

    def request(self) -> tuple[str, dict, dict]:
        self.post.assert_called_once()
        args, kwargs = self.post.call_args
        return args[0], kwargs["headers"], json.loads(kwargs["data"])


class BedrockRequestTests(_JudgeEndpoint):
    url = "https://judge.example/llm/v1/model/invoke"

    def test_default_request_is_unchanged(self):
        self.post.return_value = _response({"content": [{"type": "text", "text": "CORRECT"}]})

        label = bba_eval.evaluate_sample(None, QUESTION, OFFICIAL, CANDIDATE)

        url, headers, payload = self.request()
        self.assertEqual(label, "CORRECT")
        self.assertEqual(url, self.url)
        self.assertEqual(headers, {"Authorization": "Bearer test-key", "Content-Type": "application/json"})
        self.assertEqual(
            payload,
            {
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": bba_eval.build_eval_prompt(QUESTION, OFFICIAL, CANDIDATE)}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 8,
            },
        )


class OpenAIRequestTests(_JudgeEndpoint):
    url = "https://inference-api.nvidia.com/v1/"

    def test_request_differs_only_in_wire_format(self):
        self.post.return_value = _response({"choices": [{"message": {"role": "assistant", "content": "INCORRECT"}}]})

        label = bba_eval.evaluate_sample(None, QUESTION, OFFICIAL, CANDIDATE, "openai", JUDGE_MODEL)

        url, headers, payload = self.request()
        self.assertEqual(label, "INCORRECT")
        self.assertEqual(url, "https://inference-api.nvidia.com/v1/chat/completions")
        self.assertEqual(headers, {"Authorization": "Bearer test-key", "Content-Type": "application/json"})
        self.assertEqual(
            payload,
            {
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": bba_eval.build_eval_prompt(QUESTION, OFFICIAL, CANDIDATE)}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 8,
            },
        )

    def test_list_content_is_joined(self):
        self.post.return_value = _response(
            {"choices": [{"message": {"content": [{"type": "text", "text": "COR"}, {"type": "text", "text": "RECT"}]}}]}
        )
        self.assertEqual(
            bba_eval.evaluate_sample(None, QUESTION, OFFICIAL, CANDIDATE, "openai", JUDGE_MODEL), "CORRECT"
        )

    def test_missing_model_raises(self):
        with self.assertRaisesRegex(RuntimeError, "--judge_model"):
            bba_eval.evaluate_sample(None, QUESTION, OFFICIAL, CANDIDATE, "openai", None)
        self.post.assert_not_called()

    def test_missing_inputs_skip_the_judge(self):
        self.assertEqual(
            bba_eval.evaluate_sample(None, QUESTION, OFFICIAL, "  ", "openai", JUDGE_MODEL), bba_eval.MISSING_INFO_LABEL
        )
        self.post.assert_not_called()


class ResponseParsingTests(unittest.TestCase):
    def test_bedrock_schema_is_still_parsed(self):
        self.assertEqual(
            bba_eval._extract_text_from_response({"content": [{"type": "text", "text": "CORRECT"}]}), "CORRECT"
        )

    def test_openai_schema(self):
        self.assertEqual(
            bba_eval._extract_text_from_response({"choices": [{"message": {"content": "CORRECT"}}]}), "CORRECT"
        )

    def test_unknown_schema_is_empty(self):
        self.assertEqual(bba_eval._extract_text_from_response({"choices": []}), "")


class CommandLineTests(unittest.TestCase):
    def parse(self, *argv: str):
        with patch.object(sys, "argv", ["eval.py", "--input_dir", "d", *argv]):
            return bba_eval.parse_args()

    def test_default_is_bedrock(self):
        args = self.parse()
        self.assertEqual((args.judge_api, args.judge_model), ("bedrock", None))

    def test_openai_with_model(self):
        args = self.parse("--judge_api", "openai", "--judge_model", JUDGE_MODEL)
        self.assertEqual((args.judge_api, args.judge_model), ("openai", JUDGE_MODEL))

    def test_openai_without_model_is_an_error(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            self.parse("--judge_api", "openai")
        self.assertEqual(raised.exception.code, 2)

    def test_model_without_openai_is_an_error(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as raised:
            self.parse("--judge_model", JUDGE_MODEL)
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
