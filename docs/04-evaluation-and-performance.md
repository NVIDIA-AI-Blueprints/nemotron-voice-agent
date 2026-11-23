# Evaluation and Performance

This guide provides reference benchmarks for the Nemotron Voice Agent covering **accuracy**, **full-duplex behavior**, and **latency/throughput**. It applies to the Pipecat examples.

---

## Custom Endpoints In Service Evals

Use **Actions > Pipecat Service Evals > Run workflow** with the default `cloud_tts_manifest.yaml` to evaluate a custom model.
Custom endpoint overrides apply only to Generic and Multilingual cascaded scenarios.
Omni scenarios retain their configured multimodal model and services. They do not evaluate the custom model.
Automatic push and pull request runs retain the manifest defaults.

Supply endpoint settings through the following workflow inputs or repository Actions variables.
Workflow inputs take precedence over repository defaults when nonempty.

| Workflow Input | Repository Variable | Value |
| --- | --- | --- |
| `custom_llm_base_url` | `CUSTOM_LLM_BASE_URL` | OpenAI-compatible chat-completions base URL, usually ending in `/v1`. |
| `custom_llm_model_id` | `CUSTOM_LLM_MODEL_ID` | Model ID served by that endpoint. Set both LLM fields together. |
| `custom_asr_server` | `CUSTOM_ASR_SERVER` | Riva streaming gRPC `host:port`, reachable from the runner. HTTP transcription APIs are not supported. |
| `custom_asr_model_id` | `CUSTOM_ASR_MODEL_ID` | Served ASR model name. Set both ASR fields together. |
| `custom_asr_use_ssl` | `CUSTOM_ASR_USE_SSL` | `true` by default. Use `false` only for a plaintext Riva server. |
| `custom_asr_function_id` | `CUSTOM_ASR_FUNCTION_ID` | Optional NVCF function UUID, requiring TLS. Leave blank for a direct Riva endpoint. |

Store credentials in repository Actions **secrets**, never workflow inputs or runner-body files.
Use `CUSTOM_LLM_API_KEY` and `CUSTOM_ASR_API_KEY` by default.
To choose other secret names, set the Actions variables `CUSTOM_LLM_API_KEY_ENV` and `CUSTOM_ASR_API_KEY_ENV`.
For an unauthenticated endpoint, give its secret the value `not-needed`.
An enabled custom endpoint with a missing credential fails validation instead of using a hosted-service key.
The existing `NVIDIA_API_KEY` and `NVIDIA_INFERENCE_API_KEY` remain necessary for the other services and judges.

The optional Actions variable `CUSTOM_LLM_MAX_TOKENS` defaults to `4096`.
`CUSTOM_LLM_EXTRA_PARAMS` accepts a JSON object of model-specific request options, including explicit reasoning settings.
Custom LLM requests replace inherited model-specific options; they do not automatically add Nemotron thinking flags.
These controls affect only the bot, including its Multilingual summary LLM. User simulation, TTS, and judges keep their existing settings.

For an audio smoke, select `generic_audio_e2e_roundtrip` in **scenario** and supply either or both endpoint pairs.
For a complete run, leave **scenario** blank; interpret Omni results separately because they retain the native multimodal model.
Omni-only selections and manifests that bypass `inference_eval_bot.py` reject custom overrides instead of ignoring them.
The service runner validates configuration before running scenarios. Production catalogs, recipes, and pipeline code remain unchanged.

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
