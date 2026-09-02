# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for the NVIDIA Inference API Pipecat judge factory."""

from pathlib import Path

from pipecat.evals.judge import EvalJudge
from pipecat.evals.scenario import EvalScenario

from examples.shared.nvidia_inference_judge import create_nvidia_inference_judge

ROOT = Path(__file__).resolve().parents[1]


def test_nvidia_inference_judge_uses_expected_endpoint_and_model(monkeypatch) -> None:
    """The judge uses NVIDIA's OpenAI-compatible API, not OpenAI's default API."""
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "test-key")

    judge = create_nvidia_inference_judge({})

    assert str(judge._client.base_url) == "https://inference-api.nvidia.com/v1/"
    assert judge._model == "nvidia/google/gemma-4-31b-it"


def test_scenario_uses_nvidia_inference_judge_factory(monkeypatch) -> None:
    """YAML nesting keeps the factory inside ``judge.eval``."""
    monkeypatch.setenv("NVIDIA_INFERENCE_API_KEY", "test-key")

    scenario = EvalScenario.load(ROOT / "tests/pipecat_evals/service/scenarios/generic_initial_greeting.yaml")
    judge = EvalJudge.from_config(scenario.judge)

    assert type(judge._service).__name__ == "NvidiaInferenceJudgeService"
