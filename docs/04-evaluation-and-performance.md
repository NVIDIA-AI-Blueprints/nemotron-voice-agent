# Evaluation and Performance

This guide provides reference benchmarks for the Nemotron Voice Agent covering **accuracy**, **full-duplex behavior**, and **latency/throughput**.

---

## Latency and Scalability

### Reference Results

The reference performance benchmark measures the Nemotron Voice Agent on a dedicated **4x H100 GPU** setup (one GPU for Parakeet CTC 1.1B ASR, one for Magpie TTS, and two for Nemotron-3-Nano LLM). Most tested concurrency levels are below one second E2E latency, and the 64-stream run reaches 1.00 second. All latencies are in seconds.

> **Note:** This benchmark uses a 4-GPU setup to measure scalability. The [minimum deployment requirement](01-getting-started.md#docker-based-deployment) is cloud-only (no local GPUs) or 1 GPU with roughly 80 GB available VRAM for a local profile.

| Parallel Streams | E2E Latency | ASR Latency | TTS TTFB | LLM TTFT | LLM First-Sentence Latency |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.79 | 0.04 | 0.078 | 0.126 | 0.138 |
| 4 | 0.76 | 0.046 | 0.066 | 0.061 | 0.181 |
| 8 | 0.77 | 0.052 | 0.066 | 0.062 | 0.136 |
| 16 | 0.91 | 0.057 | 0.068 | 0.105 | 0.208 |
| 32 | 0.80 | 0.061 | 0.080 | 0.073 | 0.294 |
| 64 | 1.00 | 0.067 | 0.110 | 0.156 | 0.386 |

*E2E: End-to-End · TTFB: Time to First Byte · TTFT: Time to First Token*

To run these latency/scaling benchmarks yourself, see [Run Scaling & Performance Tests](how-to/run-scaling-perf-tests.md). For production targets and tuning guidance, refer to [Best Practices](05-best-practices.md) and [Tune Pipeline Performance](how-to/tune-pipeline-performance.md).

---

## Accuracy: BigBench Audio Benchmarking

BigBench Audio evaluates **answer correctness** on the [ArtificialAnalysis/big_bench_audio](https://huggingface.co/datasets/ArtificialAnalysis/big_bench_audio) dataset.

### Reference Results

The following table shows accuracy (%) on Big Bench Audio for the LLM standalone (text-only) vs the LLM running in the voice agent pipeline:

| Model / API | Reasoning Mode | Text Only Standalone LLM (%) | LLM In Voice Agent Pipeline (%) |
| --- | --- | --- | --- |
| Nemotron 49B (`llama-3.3-nemotron-super-49b-v1.5`) | Reasoning ON | 91.90 | 81.30 |
| Nemotron 49B (`llama-3.3-nemotron-super-49b-v1.5`) | Reasoning OFF | 82.70 | 60.30 |
| Nemotron 30B (`nemotron-3-nano`) | Reasoning ON, Budget 500 | 78.76 | 75.60 |
| Nemotron 30B (`nemotron-3-nano`)| Reasoning OFF | 56.50 | 50.40 |

### How to Reproduce

Follow steps from [`benchmarking_tools/AA-BigBenchAudio-Eval/`](../benchmarking_tools/AA-BigBenchAudio-Eval/README.md) which describe the full pipeline (download → preprocess → inference → Riva transcription → LLM-judge scoring).

---

## Full-Duplex Behavior

[Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) (v1, v1.5) probes turn-taking behavior under interruption. It measures when the agent yields to a user barge-in, when it keeps talking through background noise, and how quickly the bot reply lands after the user finishes speaking. The repo's [`benchmarking_tools/Full-Duplex-Bench-Eval/`](../benchmarking_tools/Full-Duplex-Bench-Eval/README.md) tool acts as the inference client: it streams each dataset sample to the running voice agent over WebSocket and writes the bot's reply audio back into the per-sample folders. Scoring (TOR, P_resp, P_inter, …) is then computed by the upstream Full-Duplex-Bench tooling against those output WAVs.

This repository provides the evaluation client and workflow, but it does not publish reference Full-Duplex-Bench scores in this page. Follow the [eval README](../benchmarking_tools/Full-Duplex-Bench-Eval/README.md) for detailed steps for running the benchmark and generating scores for your deployment.
