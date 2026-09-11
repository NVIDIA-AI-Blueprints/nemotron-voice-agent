# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-only eval entrypoint that injects NVIDIA Inference API services."""

from __future__ import annotations

from nvidia_inference_services import (
    InferenceMagpieTTSService,
    InferenceNvidiaLLMService,
    InferenceNvidiaOmniLLMService,
)
from pipecat.runner.types import RunnerArguments

from eval_bot import bot as registered_eval_bot


def _install_ci_services(pipeline_mode: str) -> None:
    """Patch only the modules loaded by this disposable eval process."""
    if pipeline_mode == "generic-assistant":
        import examples.generic.pipeline as pipeline

        pipeline.NvidiaLLMService = InferenceNvidiaLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
    elif pipeline_mode == "multilingual-assistant":
        import examples.multilingual.pipeline as pipeline

        pipeline.NvidiaLLMService = InferenceNvidiaLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
    elif pipeline_mode == "frontend-backend-agent":
        import examples.frontend_backend_agent.pipeline as pipeline

        pipeline.NvidiaLLMService = InferenceNvidiaLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
    elif pipeline_mode == "omni-assistant":
        import examples.omni_assistant.pipeline as pipeline

        pipeline.NvidiaOmniLLMService = InferenceNvidiaOmniLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
    elif pipeline_mode == "omni-assistant-subagents":
        import examples.omni_assistant_subagents.pipeline as pipeline
        import examples.omni_assistant_subagents.subagents.transport.agent as transport_agent

        configured_reasoning_for = pipeline._reasoning_for

        def ci_reasoning_for(registry, key: str, default: str) -> str:
            if key == pipeline.MediaAnalyzerWorker.AGENT_NAME:
                return "off"
            return configured_reasoning_for(registry, key, default)

        # The local NIM can return a reasoning-only stream for image analysis,
        # leaving no visible content for the eval to verify. Keep the example's
        # default intact and disable thinking only in this disposable CI process.
        pipeline._reasoning_for = ci_reasoning_for
        pipeline.nvidia_api_key = lambda: __import__("os").environ["NVIDIA_INFERENCE_API_KEY"]
        transport_agent.NvidiaTTSService = InferenceMagpieTTSService


async def bot(runner_args: RunnerArguments) -> None:
    """Install CI services and delegate to the registered example eval bot."""
    body = runner_args.body if isinstance(runner_args.body, dict) else {}
    _install_ci_services(str(body.get("pipeline_mode") or ""))
    await registered_eval_bot(runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
