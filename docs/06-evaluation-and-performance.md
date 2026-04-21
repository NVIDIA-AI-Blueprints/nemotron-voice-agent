# Evaluation and Performance

This guide points to tooling for benchmarking the Nemotron Voice Agent for **accuracy** and **latency/throughput**. All tooling lives in the `nvidia-pipecat` submodule. Ensure the submodule is initialized:

```bash
git submodule update --init
```

**Transport:** The BigBench and performance scripts connect to the voice agent over **WebSocket only**. If your deployment uses WebRTC (the default), switch to WebSocket before running these benchmarks. Refer to [Choose a Transport Method](how-to/choose-transport-method.md) for how to set `TRANSPORT=WEBSOCKET` in `.env` and restart the services.

---

## Overview

The following table lists the evaluation and performance tooling:

| Goal | Location | Documentation |
| --- | --- | --- |
| **Accuracy (BigBench)** | `nvidia-pipecat/tools/scripts/AA-BigBenchAudio-eval/` | [AA-BigBenchAudio-eval README](https://github.com/NVIDIA/voice-agent-examples/tree/main/tools/scripts/AA-BigBenchAudio-eval) |
| **Latency & scalability** | `nvidia-pipecat/tests/perf/` | [Performance tests README](https://github.com/NVIDIA/voice-agent-examples/tree/main/tests/perf) |

---

## BigBench Audio Benchmarking

BigBench Audio evaluates **answer correctness** on the [ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) dataset. For setup, prerequisites, and step-by-step instructions (speech-to-speech and text-inference pipelines), refer to the [AA-BigBenchAudio-eval README](https://github.com/NVIDIA/voice-agent-examples/tree/main/tools/scripts/AA-BigBenchAudio-eval).

### Reference Results

The following table shows accuracy (%) on Big Bench Audio for text-only (standalone LLM) vs speech-to-speech (LLM in voice agent pipeline):

| Model / API | Reasoning Mode | Text Only Standalone LLM (%) | LLM In Voice Agent Pipeline (%) |
| --- | --- | --- | --- |
| Nemotron 49B (`nvidia/llama-3.3-nemotron-super-49b-v1.5`) | Reasoning ON | 91.90 | 81.30 |
| Nemotron 49B | Reasoning OFF | 82.70 | 60.30 |
| Nemotron 30B (`nvidia/nemotron-3-nano`) | Reasoning ON, Budget 500 | 78.76 | 75.60 |
| Nemotron 30B | Reasoning OFF | 56.50 | 50.40 |

---

## Performance Tests

The performance tests in `nvidia-pipecat/tests/perf/` measure latency and scalability (multi-client WebSocket benchmark, TTFB analysis, glitch and reverse barge-in detection). For prerequisites, how to run the multi-client benchmark and TTFB analyzer, and troubleshooting, refer to the [Performance tests README](https://github.com/NVIDIA/voice-agent-examples/tree/main/tests/perf).

### Reference Results

**The Nemotron Voice Agent** performance benchmark shows **sub-second End-to-End (E2E) latency**. The setup uses **4× H100 GPUs** (one for Parakeet CTC 1.1B ASR, one for Magpie TTS, and two for Nemotron-3-Nano LLM) with [speculative speech processing](how-to/tune-pipeline-performance.md#speculative-speech-processing) enabled. All latencies are in seconds.

> **Note:** This benchmark uses a 4-GPU setup to measure scalability. The [minimum deployment requirement](01-getting-started.md#gpu-requirements) is 2 GPUs.

| Parallel Streams | E2E Latency | ASR Latency | TTS TTFB | LLM TTFT | LLM First-Sentence Latency |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.79 | 0.04 | 0.078 | 0.126 | 0.138 |
| 4 | 0.76 | 0.046 | 0.066 | 0.061 | 0.181 |
| 8 | 0.77 | 0.052 | 0.066 | 0.062 | 0.136 |
| 16 | 0.91 | 0.057 | 0.068 | 0.105 | 0.208 |
| 32 | 0.80 | 0.061 | 0.080 | 0.073 | 0.294 |
| 64 | 1.00 | 0.067 | 0.110 | 0.156 | 0.386 |

*E2E: End-to-End · TTFB: Time to First Byte · TTFT: Time to First Token*

For production targets and tuning guidance, refer to [Best Practices](04-best-practices.md) and [Tune Pipeline Performance](how-to/tune-pipeline-performance.md).
