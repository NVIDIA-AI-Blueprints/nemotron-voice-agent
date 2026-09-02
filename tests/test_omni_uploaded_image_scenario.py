# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Regression checks for the uploaded-image Pipecat eval scenario."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_uploaded_image_acknowledgement_is_semantic_not_exact_text() -> None:
    """Allow valid acknowledgement wording before the image analysis result."""
    scenario = (ROOT / "tests/pipecat_evals/service/scenarios/omni_uploaded_image_path.yaml").read_text()
    runner_body = (ROOT / "tests/pipecat_evals/service/runner_bodies/cloud_tts.yaml").read_text()

    assert 'text_contains: "Analyzing the uploaded image now."' not in scenario
    assert "acknowledges the uploaded image" in scenario
    assert "concrete findings from the uploaded architecture diagram" not in scenario
    assert "set response exactly" not in runner_body
