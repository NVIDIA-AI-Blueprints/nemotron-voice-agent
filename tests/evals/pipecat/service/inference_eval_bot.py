# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-only eval entrypoint that injects NVIDIA Inference API services."""

from __future__ import annotations

import json

from custom_endpoints import custom_asr, custom_llm
from nvidia_inference_services import (
    InferenceMagpieTTSService,
    InferenceNvidiaLLMService,
    InferenceNvidiaOmniLLMService,
    InferenceNvidiaSTTService,
)
from pipecat.runner.types import RunnerArguments

from nemotron_voice_agent.pipecat.runtime.eval_bot import bot as registered_eval_bot


def _install_ci_services(pipeline_mode: str) -> None:
    """Patch only the modules loaded by this disposable eval process."""
    if pipeline_mode == "generic-assistant":
        import nemotron_voice_agent.pipecat.examples.generic.pipeline as pipeline

        pipeline.NvidiaLLMService = InferenceNvidiaLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
        if custom_asr():
            pipeline.NvidiaSTTService = InferenceNvidiaSTTService
    elif pipeline_mode == "multilingual-assistant":
        import nemotron_voice_agent.pipecat.examples.multilingual.pipeline as pipeline

        pipeline.NvidiaLLMService = InferenceNvidiaLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
        if custom_asr():
            pipeline.NvidiaSTTService = InferenceNvidiaSTTService
    elif pipeline_mode == "omni-assistant":
        import nemotron_voice_agent.pipecat.examples.omni_assistant.pipeline as pipeline

        pipeline.NvidiaOmniLLMService = InferenceNvidiaOmniLLMService
        pipeline.NvidiaTTSService = InferenceMagpieTTSService
    elif pipeline_mode == "omni-assistant-subagents":
        import nemotron_voice_agent.pipecat.examples.omni_assistant_subagents.pipeline as pipeline
        from nemotron_voice_agent.pipecat.examples.omni_assistant_subagents.subagents.transport import (
            agent as transport_agent,
        )

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
    body = dict(runner_args.body) if isinstance(runner_args.body, dict) else {}
    pipeline_mode = str(body.get("pipeline_mode") or "")
    if pipeline_mode in {"generic-assistant", "multilingual-assistant"}:
        llm, asr = custom_llm(), custom_asr()
        if llm:
            body.update(
                llm_id="custom-ci-llm",
                model_id=llm["model"],
                base_url=llm["base_url"],
                max_tokens=llm["max_tokens"],
                extra_params=json.dumps(llm["extra"]),
            )
            print(f"CI custom LLM: mode={pipeline_mode}, model={llm['model']}, base_url={llm['base_url']}", flush=True)
        if asr:
            body.update(
                asr_id="custom-ci-asr",
                asr_server=asr["server"],
                asr_model=asr["model"],
                asr_function_id=asr["function_id"],
            )
            print(
                f"CI custom ASR: mode={pipeline_mode}, model={asr['model']}, "
                f"server={asr['server']}, TLS={asr['use_ssl']}",
                flush=True,
            )
    runner_args.body = body
    _install_ci_services(pipeline_mode)
    await registered_eval_bot(runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
