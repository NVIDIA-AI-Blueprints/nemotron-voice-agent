# Evaluation and Performance

This guide provides reference benchmarks for the Nemotron Voice Agent covering **accuracy**, **full-duplex behavior**, and **latency/throughput**.

---

## Latency and Scalability

### Reference Results

The reference performance benchmark measures the Nemotron Voice Agent on a dedicated **4x B200 GPU** setup (one GPU for Nemotron ASR Streaming (English), one for Magpie Multilingual TTS, and two for the Nemotron 3.5 Lightning LLM). E2E latency stays below one second through 64 concurrent streams. All latencies are in seconds.

> **Note:** This benchmark uses a 4-GPU setup to measure scalability. Deployment options include cloud-only with no local GPUs, about 80 GB VRAM for the default all-on-one-GPU `*/server` NIM layout, or a supported one-GPU host for `*/single-gpu`. See [Configure LLM](how-to/configure-llm.md#vram--hardware-support) for the automatic VRAM plan.
>
> The current `generic-assistant/server-perf` Compose recipe pins the `vllm-nvfp4-tp2-pp1-18.0` profile for Nemotron 3.5 Lightning. Other architectures require listing and benchmarking their compatible TP2 profiles before pinning a hardware-specific winner. See the [scaling benchmark](../benchmarking_tools/scaling-perf/README.md#reproducing-the-recommended-scaling-setup).

| Parallel Streams | Server E2E Latency | ASR TTFB | LLM Processing Time | LLM TTFT | TTS TTFB |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.93 | 0.47 | 0.43 | 0.23 | 0.08 |
| 2 | 0.87 | 0.5 | 0.35 | 0.16 | 0.07 |
| 4 | 0.86 | 0.49 | 0.41 | 0.17 | 0.07 |
| 8 | 0.87 | 0.49 | 0.35 | 0.14 | 0.08 |
| 16 | 0.86 | 0.49 | 0.34 | 0.13 | 0.08 |
| 32 | 0.9 | 0.48 | 0.37 | 0.13 | 0.09 |
| 64 | 0.93 | 0.49 | 0.4 | 0.13 | 0.1 |
| Mean | 0.89 | 0.49 | 0.38 | 0.16 | 0.08 |

*E2E: End-to-End · TTFB: Time to First Byte · TTFT: Time to First Token*

> **Note:** Performance numbers may vary based on hardware configuration (both CPU and GPU). Occasionally, higher latency may be observed due to uneven load balancing across FastAPI workers. For production deployments, using a Kubernetes setup is recommended to ensure stable load distribution and scalability.

To run these latency/scaling benchmarks yourself, see [Run Scaling & Performance Tests](how-to/run-scaling-perf-tests.md). For production targets and tuning guidance, refer to [Best Practices](05-best-practices.md) and [Tune Pipeline Performance](how-to/tune-pipeline-performance.md).

---

## Accuracy: BigBench Audio Benchmarking

BigBench Audio evaluates **answer correctness** on the [ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) dataset.

### Reference Results

The following table shows accuracy (%) on Big Bench Audio for the LLM standalone (text-only) vs the LLM running in the voice agent pipeline:

> **Note:** The Nemotron 3 Nano rows are historical benchmark results and do not represent the current default model.

| Model / API | Reasoning Mode | Text Only Standalone LLM (%) | LLM In Voice Agent Pipeline (%) |
| --- | --- | --- | --- |
| Nemotron 30B (`nemotron-3-nano`) | Reasoning ON, Budget 500 | 78.76 | 75.60 |
| Nemotron 30B (`nemotron-3-nano`)| Reasoning OFF | 56.50 | 50.40 |

### How to Reproduce

Follow steps from [`benchmarking_tools/AA-BigBenchAudio-Eval/`](../benchmarking_tools/AA-BigBenchAudio-Eval/README.md) which describe the full pipeline (download → preprocess → inference → Riva transcription → LLM-judge scoring).

---

## Full-Duplex Behavior

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) (v1, v1.5) probes turn-taking behavior under interruption. It measures when the agent yields to a user barge-in, when it keeps talking through background noise, and how quickly the bot reply lands after the user finishes speaking. The repo's [`benchmarking_tools/Full-Duplex-Bench-Eval/`](../benchmarking_tools/Full-Duplex-Bench-Eval/README.md) tool acts as the inference client: it streams each dataset sample to the running voice agent over WebSocket and writes the bot's reply audio back into the per-sample folders. Scoring (TOR, P_resp, P_inter, …) is then computed by the upstream Full-Duplex-Bench tooling against those output WAVs.

This repository provides the evaluation client and workflow, but it does not publish reference Full-Duplex-Bench scores in this page. Follow the [eval README](../benchmarking_tools/Full-Duplex-Bench-Eval/README.md) for detailed steps for running the benchmark and generating scores for your deployment.
