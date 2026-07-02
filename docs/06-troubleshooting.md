# Troubleshooting

Known issues and fixes for **startup and deployment** of the Nemotron Voice Agent. Find your symptom in the **Error** column, apply the **Cause & fix**, and follow the **Reference** for depth.

## Containers and first run

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `docker compose up` does nothing | No profile selected. Every deployment needs exactly one recipe profile (e.g. `--profile generic-assistant`). The no-profile no-op is intentional. | [Getting Started → Docker based Deployment](01-getting-started.md#docker-based-deployment) |
| First deploy takes 30–60 minutes | Expected. Images and models download on first run. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| First voice turn slow on local recipes, later turns fast | Expected warmup while GPU LLM sidecars load. The deploy is healthy if later turns are fast. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| A local LLM / ASR / TTS is missing from the Services tab | The sidecar is not deployed or is not reachable, and the catalog filters local entries by TCP reachability. Confirm the container is healthy (`docker compose ps`) and you launched the matching `/<hardware>` profile. | [Configure Services → On-prem catalog](how-to/configure-services.md#on-prem-catalog) |
| ASR/TTS sidecar image fails to pull | Log in to `nvcr.io` with an `NVIDIA_API_KEY` that has access to the image. The active image is set in the matching compose file. | [`docker/`](../docker/) compose files |

## Local LLM will not start (self-hosted)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `No available memory for the cache blocks` | The LLM's VRAM fraction is too **low**, leaving no room for the KV cache after the weights. **Raise** `NIM_KVCACHE_PERCENT` (NIM) or `--gpu-memory-utilization` (Omni vLLM). Do not lower it. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM GPU memory](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) |
| LLM process killed / true CUDA OOM / latency degrades under load | Too much on one GPU. Put speech sidecars on a second GPU (their `device_ids`), reduce KV cache / context length, or lower batch size / precision. Confirm `NVIDIA_API_KEY` / `HF_TOKEN` so an auth failure isn't mistaken for OOM. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM GPU memory](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) |
| Startup fails CUDA-graph capture | The cache holds fewer Mamba blocks than `LLM_MAX_NUM_SEQS` sequences (Nemotron-3 Nano is a hybrid Mamba model). Lower `LLM_MAX_NUM_SEQS` (e.g. `64`–`128`). | [Configure LLM → Deployment tuning parameters](how-to/configure-llm.md#deployment-tuning-parameters) |
| `The quantization method modelopt is not supported … Minimum capability: 89. Current capability: 80` | FP8 isn't supported on Ampere (A100) or older. Switch to BF16 with `NIM_TAGS_SELECTOR=precision=bf16,tp=1` and raise `NIM_KVCACHE_PERCENT=0.9` (BF16 weights are ~2× larger). NVFP4 needs a Blackwell GPU or newer. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html) |
| LLM weights fail to download (DGX Spark / Jetson vLLM recipes) | Set `HF_TOKEN` in `.env`. The raw-vLLM recipes pull the Nemotron weights from Hugging Face, which needs a valid token. | [Getting Started](01-getting-started.md#docker-based-deployment) · [Jetson Thor](03-jetson-thor.md) |

> Full GPU sizing and precision detail lives in [Configure LLM → VRAM & hardware support](how-to/configure-llm.md#vram--hardware-support). For other self-hosted LLM NIM issues (CUDA driver-init errors 802/803, profile selection, and more), see the [NIM for LLMs troubleshooting guide](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/index.html).

## Self-hosted LLM tool calling and reasoning

Self-hosted Nemotron-3 models only. Cloud (NVCF) has the parsers enabled server-side, and the repo's `docker/docker-compose.nemotron3-*.yaml` already sets them.

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `HTTP 400: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | The 2.x LLM NIM versions do not auto-enable the parsers. For NIM, set `NIM_PASSTHROUGH_ARGS=--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3`. For raw vLLM (DGX Spark / Jetson), pass the same flags on `vllm serve`. | [Configure LLM → parser & tool calling](how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) · [`docker-compose.nemotron3-nano.yaml`](../docker/docker-compose.nemotron3-nano.yaml) |
| Reasoning is spoken by TTS / `<think>` leaks into the answer | The reasoning parser is not set. Add `--reasoning-parser nemotron_v3`, which separates reasoning from `content` and keeps reasoning-OFF working. | [Configure LLM → parser & tool calling](how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) |
| Raw vLLM: `nemotron_v3` parser not found, or `MIXED_PRECISION` not supported | The image's vLLM is too old. Use NGC `nvcr.io/nvidia/vllm:26.05.post1-py3` (vLLM ≥ 0.20 ships both). | [`docker-compose.nemotron3-nano.yaml`](../docker/docker-compose.nemotron3-nano.yaml) |

## ASR (speech-to-text)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| ASR sidecar OOMs or won't start | The ASR NIM needs **≥ 16 GB VRAM** (compute capability ≥ 8.0). Put ASR on a second GPU (its `device_ids`) or run it from the cloud. Confirm `NVIDIA_API_KEY` so an auth failure isn't mistaken for OOM. | [Configure ASR → VRAM](how-to/configure-asr.md#vram--hardware-support) · [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) |
| Wrong language detected, or TTS speaks the wrong voice (multilingual) | The detected language must be one the active **TTS** voice supports. Check the configured TTS service exposes that language code. | [Multilingual example → Troubleshooting](../src/examples/multilingual/README.md#troubleshooting) |
| `Model not found for language` | The deployed ASR model doesn't cover that `language_code`. Switch to the multilingual ASR model or pin a supported locale. | [Configure ASR → Customization](how-to/configure-asr.md#customization) · [ASR NIM troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/asr.html) |
| No transcription / no voices discovered at startup | The speech prewarm failed. Check sidecar health (`docker compose ps`) and `NVIDIA_API_KEY`. | [Multilingual example → Troubleshooting](../src/examples/multilingual/README.md#troubleshooting) |
| Mic or ASR stops accepting input after a long idle period (around 8-10 minutes) | The Pipecat pipeline idle timeout ended the session or ASR server connection timeout due to no audio. `PIPELINE_IDLE_TIMEOUT_SECS` defaults to **600 seconds**. Start a new browser session (reload the page) to reconnect with a fresh pipeline. To allow longer idle sessions, raise `PIPELINE_IDLE_TIMEOUT_SECS` in `.env` (minimum 300) and send silence buffers to avoid ASR timeout. | [`.env.example`](../.env.example) |

## TTS (text-to-speech)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Synthesis fails or produces odd audio on code / Markdown / JSON output | Characters reserved by the Magpie preprocessor (`{`, `}`, `<tag>`) reached the engine. Apply the text filter, using `NemotronSpeechMarkdownTextFilter` for Markdown-heavy output. | [Configure TTS → TTS text filter](how-to/configure-tts.md#tts-text-filter) |
| Mispronounced brand / domain terms | Add them to an IPA dictionary via `TTS_IPA_FILE_PATH`. | [Configure TTS → Pronunciation (IPA)](how-to/configure-tts.md#pronunciation-ipa) |
| Long replies are rejected or truncated | The TTS NIM caps a request at **2,000 normalized characters**. The NVIDIA Pipecat TTS service streams replies **sentence-by-sentence with a 200-character hard limit per sentence**, so the cap isn't hit in normal use. It mainly affects custom integrations that synthesize large blocks at once. Split long text into sentence / paragraph chunks. | [Configure TTS](how-to/configure-tts.md) · [TTS NIM troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/tts.html) |

## Response quality (hallucination and repetition)

These are runtime behavior issues and apply to any deployment, cloud or local.

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Bot invents facts or answers something the user did not ask (hallucination) | Two common sources. First, ASR mis-transcription feeds a wrong query to the LLM, which is worse under background noise. Improve transcription with word boosting and domain finetuning. Second, the LLM fabricates. Lower `temperature`, prefer Nemotron 3 Super or reasoning ON for hard questions, ground answers with tool calls, and instruct the prompt to say it does not know rather than guess. | [Configure ASR → Customization](how-to/configure-asr.md#customization) · [Configure LLM → request parameters](how-to/configure-llm.md#tuning-llm-request-parameters) · [Configure Prompts](how-to/configure-prompts.md) |
| Bot repeats the same words or phrases, or loops | The model is not penalizing repetition, or the context has degenerated. Raise `repetition_penalty` above `1` in the catalog entry's `extra_body` (repo default `1.05`), and avoid an over-low `temperature`. If repetition builds up over a long session, check the chat-history window and summarization so stale or duplicated turns are not fed back into the context. | [Configure LLM → request parameters](how-to/configure-llm.md#tuning-llm-request-parameters) · [Tune Pipeline Performance](how-to/tune-pipeline-performance.md) |

## Turn-taking and interruptions

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Background or random noise interrupts the bot mid-reply (false barge-in) and leaves the conversation in a confused state | Silero VAD is detecting ambient noise as the onset of user speech, which barges in and stops the TTS. When no real utterance follows, the turn is left half-finished. The most effective fix is to reduce input noise: use a wired or directional headset mic in a quieter room. If noise still trips it, raise the Silero VAD sensitivity (its confidence and minimum-volume thresholds in `VADParams`) where the pipeline builds the transport. | [Tune Pipeline Performance → Smart Turn Detection](how-to/tune-pipeline-performance.md#smart-turn-detection) · [Configure ASR → Customization](how-to/configure-asr.md#customization) |

## Cloud (NVCF)

The hosted **[build.nvidia.com](https://build.nvidia.com/)** endpoints are for **experimentation and trials only**. For production, and for the most predictable latency and throughput, **self-host the models on-prem** (local NIM / vLLM sidecar).

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `Error code: 429 - {'status': 429, 'title': 'Too Many Requests'}` | The hosted API key hit its rate limit (tied to your `NVIDIA_API_KEY` / account, not the machine). Check the current per-model rate limits on build.nvidia.com and request a higher limit if needed. For production use, self-host with a local NIM / vLLM sidecar. | [build.nvidia.com](https://build.nvidia.com/) · [Configure LLM](how-to/configure-llm.md) |

> Cloud responses for large models are also slower (Super 120B is higher latency than Nano, especially with reasoning on). High latency on its own is not a rate-limit error. Only an explicit `429` indicates rate limiting.

## Jetson Thor

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `RuntimeError: Engine core initialization failed` (vLLM) | Often low available memory from cached pages held by the kernel (the `nvidia-llm-vllm` logs show the engine-core failure). Reclaim caches with `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, then re-up the `<example>/jetson-thor` profile and re-check `free -h`. | [Jetson Thor](03-jetson-thor.md) · [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) |
| Choppy / glitchy bot speech (vLLM and Riva share one GPU) | Enable CUDA MPS + CPU pinning before bringing up the stack: set `VLLM_MPS_THREAD_PCT` / `RIVA_MPS_THREAD_PCT` and the `*_CPUSET` vars in `.env`, then `sudo bash scripts/start-mps.sh`. | [Jetson Thor](03-jetson-thor.md) (production tuning) |
| Riva models not found | If the Riva quickstart is not the repo's sibling directory, set `RIVA_MODEL_LOC` in `.env` to the absolute path of its `model_repository/`. | [Jetson Thor](03-jetson-thor.md) |

## Browser access

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Browser shows a certificate warning on first visit | The development server uses a self-signed certificate for local HTTPS. For trusted test hosts, accept the warning and continue, then grant microphone access. For shared or production access, use a valid certificate from a reverse proxy or ingress. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| Microphone / WebRTC blocked | Browsers require a secure context. Keep TLS enabled (default HTTPS). Setting `PIPELINE_TLS=false` serves plain HTTP and is intended for headless / API testing only. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| Need plain HTTP for temporary browser testing | For local debugging only, set `PIPELINE_TLS=false`, then mark the origin secure in your browser. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure` (or `edge://flags/#unsafely-treat-insecure-origin-as-secure`), enable **Insecure origins treated as secure**, add `http://<machine-ip>:7860`, relaunch the browser, and remove the origin when done. Do not use this workaround for production. | — |
| Remote client on a different network cannot connect | Deploy a TURN server. | [Enable a TURN Server](how-to/enable-turn-server.md) |
