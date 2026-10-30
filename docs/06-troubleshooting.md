# Troubleshooting

Known issues and fixes for **startup and deployment** of the Nemotron Voice Agent. Find your symptom in the **Error** column, apply the **Cause & fix**, and follow the **Reference** for depth.

## Containers and first run

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `docker compose up` does nothing | No profile selected. Every deployment needs exactly one recipe profile, for example `--profile generic-assistant`. The no-profile no-op is intentional. | [Getting Started → Docker Based Deployment](01-getting-started.md#docker-based-deployment) |
| First deploy takes 30–60 minutes | Expected. Images and models download on first run. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| First voice turn slow on local recipes, later turns fast | Expected warmup while GPU LLM sidecars load. The deploy is healthy if later turns are fast. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| A local LLM / ASR / TTS is missing from the Services tab | The key is not listed under the example's `services` in `examples_registry.yaml`, or, on host-native runs, the sidecar is not reachable when the page loads. Compose sets `SERVICE_RECIPE` per profile; host-native runs detect it from reachable sidecars. Confirm the container is healthy (`docker compose ps`), then refresh. | [Configure Services](how-to/configure-services.md#how-the-active-recipe-selects-entries) |
| NVIDIA Cloud services are missing from the Services tab | `NVIDIA_API_KEY` is unset, empty, or `not-needed`. Cloud catalog entries (`server` entries with `nvcf`) stay hidden until a real key is set. Expected on `*/single-gpu` recipes, which require `HF_TOKEN` instead. Cloud-only, `*/server`, and `generic-assistant/server-perf` require `NVIDIA_API_KEY`. | [Configure Services](how-to/configure-services.md) |
| ASR/TTS sidecar image fails to pull (`*/server` or `server-perf`) | Log in to `nvcr.io` with an `NVIDIA_API_KEY` that has access to the image. Single-GPU recipes require `HF_TOKEN` and do not use `nvcr.io` login. The active image is set in the matching compose file. | [`docker/`](../docker/) compose files |
| `mkdir: Permission denied` on `models/nemo-speech` | Docker created the bind-mount as root because the host path was missing. Do **not** use `sudo`. Re-run `bash scripts/download-nemo-speech-models.sh` as your user. It reclaims ownership. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| Download script: `Neither 'hf' nor 'uvx' is available` | Neither supported downloader is on `PATH`. Install the Hugging Face `hf` CLI or `uv` for `uvx`, and then rerun the script as your user without `sudo`. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| Hugging Face `Permission denied` under `~/.cache/huggingface` | The vLLM sidecar bind-mounts that cache and writes as root. Re-run the download script as your user (it reclaims the cache) or `chown` it back to your user. | [Getting Started](01-getting-started.md#docker-based-deployment) |

<a id="local-llm-wont-start-self-hosted"></a>

## Local LLM Does Not Start (Self-Hosted)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `No available memory for the cache blocks` | The LLM's VRAM fraction is too **low**, leaving no room for the KV cache after the weights. Raise `NIM_KVCACHE_PERCENT` for NIM. For single-GPU vLLM, inspect the startup VRAM plan. Lower `VLLM_VRAM_HEADROOM_MIB` or set `VLLM_GPU_MEMORY_UTILIZATION` only after confirming enough memory remains for speech and the system. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM GPU memory](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) |
| LLM process killed / true CUDA OOM / latency degrades under load | Too much is running on one GPU. Put speech sidecars on a second GPU (their `device_ids`), reduce KV cache or context length, or lower batch size or precision. Confirm `NVIDIA_API_KEY` or `HF_TOKEN` so an authentication failure is not mistaken for OOM. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM GPU memory](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) |
| Startup fails CUDA-graph capture | The cache holds fewer Mamba blocks than `LLM_MAX_NUM_SEQS` sequences (Nemotron 3.5 Lightning is a hybrid Mamba model). Lower `LLM_MAX_NUM_SEQS`, for example to `64`–`128`. | [Configure LLM → Deployment tuning parameters](how-to/configure-llm.md#deployment-tuning-parameters) |
| `The quantization method modelopt is not supported … Minimum capability: 89. Current capability: 80` | The GPU does not meet the FP8 compute-capability requirement. Use a supported GPU and select a compatible model profile from the NIM support matrix. NVFP4 needs a Blackwell GPU or newer. | [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) · [NIM support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html) |
| LLM weights fail to download (`*/single-gpu` vLLM recipes) | Set `HF_TOKEN` in `.env`. The raw-vLLM recipes pull the Nemotron weights from Hugging Face, which needs a valid token. | [Getting Started](01-getting-started.md#docker-based-deployment) · [Jetson Thor](03-jetson-thor.md) |

> Full GPU sizing and precision detail lives in [Configure LLM → VRAM & hardware support](how-to/configure-llm.md#vram--hardware-support). For other self-hosted LLM NIM issues (CUDA driver-init errors 802/803, profile selection, and more), see the [NIM for LLMs troubleshooting guide](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/index.html).

## Self-hosted LLM tool calling and reasoning

Self-hosted Nemotron-3 models only. Cloud (NVCF) has the parsers enabled server-side, and the repo's `docker/docker-compose.nemotron3-*.yaml` already sets them.

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `HTTP 400: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser` | The 2.x LLM NIM versions do not enable the parsers automatically. The stock Compose files already include the required NIM and raw-vLLM flags. Confirm that you are using the matching 2.2.0 Compose file and image, and then pull and recreate the service. If you maintain a custom Compose file, copy the complete parser arguments from the stock service instead of setting `NIM_PASSTHROUGH_ARGS` only in `.env`. | [Configure LLM → Parser and Tool Calling](how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) · [`docker-compose.nemotron35-lightning.yaml`](../docker/docker-compose.nemotron35-lightning.yaml) |
| Reasoning is spoken by TTS / `<think>` leaks into the answer | The reasoning parser is not set. Add `--reasoning-parser nemotron_v3`, which separates reasoning from `content` and keeps reasoning off working. | [Configure LLM → Parser and Tool Calling](how-to/configure-llm.md#reasoning-parser--tool-calling-self-hosted) |
| Raw vLLM: `nemotron_v3` parser not found, or `MIXED_PRECISION` not supported | The image's vLLM is too old. Cascaded single-GPU recipes pin `vllm/vllm-openai:v0.29.0`. If you still hit the error after a clean pull, bump that pin. | [`docker-compose.nemotron35-lightning.yaml`](../docker/docker-compose.nemotron35-lightning.yaml) |

## ASR (speech-to-text)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| ASR sidecar OOMs or does not start | The ASR NIM needs **≥ 16 GB VRAM** (compute capability ≥ 8.0). Put ASR on a second GPU (its `device_ids`) or run it from the cloud. For a Server recipe, confirm `NVIDIA_API_KEY` in `.env` and the NGC login so an authentication failure is not mistaken for OOM. Single-GPU recipes use the local NeMo-Speech.cpp sidecar instead of the ASR NIM. | [Configure ASR → VRAM](how-to/configure-asr.md#vram--hardware-support) · [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html) |
| Session language is unavailable, or startup rejects it (multilingual) | The locale must be supported by the active **ASR**, **TTS**, and built-in **LLM**. Select a locale shown in Voice Settings; a stale or bypassed unsupported request is rejected before the pipeline starts. | [Multilingual example → Troubleshooting](../src/examples/multilingual/README.md#troubleshooting) · [Configure LLM](how-to/configure-llm.md#multilingual-session-languages) |
| `Model not found for language` | The deployed ASR model does not cover that `language_code`. Switch to the multilingual ASR model or pin a supported locale. | [Configure ASR → Customization](how-to/configure-asr.md#customization) · [ASR NIM troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/asr.html) |
| No transcription / no voices discovered at startup | The speech prewarm failed. For Cloud, confirm `NVIDIA_API_KEY` in `.env`. For Server, also confirm NGC login and sidecar health with `docker compose ps`. For Single-GPU, confirm that the NeMo-Speech.cpp sidecar is healthy and `models/nemo-speech` contains the downloaded weights. | [Multilingual Example → Troubleshooting](../src/examples/multilingual/README.md#troubleshooting) |
| Mic or ASR stops accepting input after a long idle period (around 8-10 minutes) | The Pipecat pipeline idle timeout ended the session or ASR server connection timeout due to no audio. `PIPELINE_IDLE_TIMEOUT_SECS` defaults to **600 seconds**. Start a new browser session (reload the page) to reconnect with a fresh pipeline. To allow longer idle sessions, raise `PIPELINE_IDLE_TIMEOUT_SECS` in `.env` (minimum 300) and send silence buffers to avoid ASR timeout. | [`.env.example`](../.env.example) |

## TTS (text-to-speech)

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Synthesis fails or produces odd audio on code / Markdown / JSON output | Characters reserved by the Magpie preprocessor (`{`, `}`, `<tag>`) reached the engine. Apply the text filter, using `NemotronSpeechMarkdownTextFilter` for Markdown-heavy output. | [Configure TTS → TTS text filter](how-to/configure-tts.md#tts-text-filter) |
| Mispronounced brand / domain terms | Add them to an IPA dictionary using `TTS_IPA_FILE_PATH`. | [Configure TTS → Pronunciation (IPA)](how-to/configure-tts.md#pronunciation-ipa) |
| Single-GPU TTS reads digits, dates, or currency as characters (for example "2" instead of "two") | Magpie TTS text normalization is off. Re-run `bash scripts/download-nemo-speech-models.sh` so `models/nemo-speech/tn_configs` is present, then restart the `*/single-gpu` profile. The sidecar must receive `--tts.tn-model-dir=/models/tn_configs`. | [Getting Started](01-getting-started.md#docker-based-deployment) · [Jetson Thor](03-jetson-thor.md) |
| Long replies are rejected or truncated | The TTS NIM caps a request at **2,000 normalized characters**. The NVIDIA Pipecat TTS service streams replies **sentence-by-sentence with a 200-character hard limit per sentence**, so the cap is not reached in normal use. It mainly affects custom integrations that synthesize large blocks at once. Split long text into sentence or paragraph chunks. | [Configure TTS](how-to/configure-tts.md) · [TTS NIM troubleshooting](https://docs.nvidia.com/nim/speech/latest/troubleshooting/tts.html) |

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

## Runtime Service Failures

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| The voice session ends after an ASR, LLM, or TTS error | The service reported that it cannot recover, for example because a provider rejected its API key, model, or voice, or because reconnection attempts were exhausted. The pipeline ends the session gracefully instead of repeatedly sending work to an unusable service. Correct the credential or service configuration, confirm that the local sidecar is healthy, and then start a new session. | [Configure Services](how-to/configure-services.md) |

## Cloud (NVCF)

The hosted **[build.nvidia.com](https://build.nvidia.com/)** endpoints are for **experimentation and trials only**. For production, and for the most predictable latency and throughput, **self-host the models on-prem** (local NIM / vLLM sidecar).

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `Error code: 429 - {'status': 429, 'title': 'Too Many Requests'}` | The hosted API key hit its rate limit (tied to your `NVIDIA_API_KEY` / account, not the machine). Check the current per-model rate limits on build.nvidia.com and request a higher limit if needed. For production use, self-host with a local NIM / vLLM sidecar. | [build.nvidia.com](https://build.nvidia.com/) · [Configure LLM](how-to/configure-llm.md) |

> Cloud responses for large models are also slower (Nemotron 3 Super 120B is higher latency than Nemotron 3.5 Lightning, especially with reasoning on). High latency on its own is not a rate-limit error. Only an explicit `429` indicates rate limiting.

## Single GPU

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| `RuntimeError: Engine core initialization failed` (vLLM) | Often low available memory from cached pages held by the kernel (the `nvidia-llm-vllm-lightning` logs show the engine-core failure). Reclaim caches with `sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, then re-up the `<example>/single-gpu` profile and re-check `free -h`. | [Jetson Thor](03-jetson-thor.md) · [Configure LLM → VRAM](how-to/configure-llm.md#vram--hardware-support) |
| Choppy / glitchy bot speech (vLLM and the speech sidecar share one GPU) | vLLM is starved of GPU or host memory. For automatically sized recipes, increase `VLLM_VRAM_HEADROOM_MIB` in `.env` or set a lower `VLLM_GPU_MEMORY_UTILIZATION` override. Lightning on DGX Spark and Jetson Thor already uses a fixed value of `0.35`, so inspect host memory and reduce workload or concurrency first. Restart the profile and re-measure with the speech sidecar loaded. | [Jetson Thor](03-jetson-thor.md) |
| Speech models not found | The GGUF weights are missing. Run `bash scripts/download-nemo-speech-models.sh` as your user (not sudo), or point `NEMO_SPEECH_MODEL_LOC` in `.env` at the absolute path holding them. The same script also downloads TTS TN grammars into `tn_configs`. | [Jetson Thor](03-jetson-thor.md) |

## Frontend/Backend Live

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Every `/v1/live` route answers 404 | The Live routes are not served. `LIVE_ENABLED` is `false`, or it is unset while no `LIVE_API_TOKEN` is set and the deployment exposes other examples, for example with `EXAMPLE_SELECTION=all`. The routes are also off when `UVICORN_WORKERS` is greater than 1. Set `LIVE_ENABLED=true` and run one worker. | [Routes and Settings](../src/examples/frontend_backend_live/README.md#routes-and-settings) |
| The server stops at startup and names a `LIVE_` variable | `LIVE_MAX_SESSIONS`, `LIVE_MAX_SESSION_SECONDS`, `LIVE_MAX_APPENDS_PER_MINUTE`, or `LIVE_MAX_CONSOLE_ROUNDS` is not a whole number of at least 1. Fix the value, or remove it to use the default. | [Frontend/Backend Live Architecture → Environment Variables](design/architecture.md#environment-variables) |
| A live session is refused with `server_error` and "is still starting. Try again in a moment." | A speech service or the talker's model is not ready yet. Wait until the sidecars are healthy (`docker compose ps`), then start a new session. If you set the talker to a model that the profile does not start, such as Nemotron 3 Super without the `nemotron-3-super` profile, every session is refused. | [Getting Started](01-getting-started.md#docker-based-deployment) · [Default Models](../src/examples/frontend_backend_live/README.md#default-models) |
| A live session is refused with `server_error` and "could not be created. Check server model access and configuration." | The server cannot load `config.yaml`, `core.yaml`, or a profile: the file is malformed, or a key is unknown, invalid, or in the wrong file. The server log names the key. Fix the file, and then restart the server, because it reads each file only once. | [Configuration](../src/examples/frontend_backend_live/README.md#configuration) |
| `responses_validation_failed` after a session starts or after `session.update` | The backend cannot honor a `delegation.responses` setting, for example a model that the deployment does not serve. The error's `param` names the setting, and the session continues. Choose a model from `backend_models` in `GET /v1/live/info`, or remove the setting. | [Live Sessions](../src/examples/frontend_backend_live/README.md#live-sessions) |
| An `info` event with code `pipeline_error` during a live session | A speech service failed once and the session recovered, for example one text-to-speech request that returned no audio. The message names the service, and the server log has a WARNING `pipeline_error=...`. Check that service if the notices repeat. An `error` with this code means a client command failed, or the session could not go on and `session.closed` follows. | [Live Gateway](../src/live/README.md) |
| `POST /v1/live/delegate` answers 403 and says the host is not allowed | Without `LIVE_API_TOKEN`, this route accepts only requests addressed to an IP address, `localhost`, or a host name in `LIVE_ALLOWED_HOSTS`. Add the host name to `LIVE_ALLOWED_HOSTS`, or set `LIVE_API_TOKEN`. | [Routes and Settings](../src/examples/frontend_backend_live/README.md#routes-and-settings) |
| With the harness adapter, every delegation fails, and the log shows `harness.runtime_open_failed`, `harness.first_session_failed`, or `harness.conductor_open_failed` | The adapter cannot start or reach the agent: `hermes` is not on the server's `PATH`, the `backend.harness.url` endpoint does not answer, or the agent's configuration lacks a model or an API key. The log line ends with the reason, including the agent's own error. Put `hermes` on the `PATH` or set `backend.harness.command` to its full path, confirm that the URL answers, and check the model and the API key in the Hermes home. | [Frontend/Backend Live Architecture → Connect Your Own Harness](design/architecture.md#connect-your-own-harness) |
| With the harness adapter, a delegation fails with `the connection to the agent is lost` | The agent process ended, or its connection broke. A WARNING before the failure names the cause: `harness.runtime_exited` with the exit code and the end of the agent's stderr, `harness.connection_closed` with the WebSocket close code, or `harness.connection_lost`, for example for a line on the agent's stdout that is not a JSON object. Fix that cause, and keep the agent's stdout for Agent Client Protocol (ACP) messages only. | [Frontend/Backend Live Architecture → Connect Your Own Harness](design/architecture.md#connect-your-own-harness) |
| With the harness adapter, the backend ignores the core's instructions or answers in the default Hermes persona, and nothing fails | Hermes does not receive the identity or the instructions that the core relies on. The Hermes home has no `SOUL.md`, or, over WebSocket, Hermes cannot access the session's folder under `work_root` and logs `configured working directory does not exist`. Write the home's `SOUL.md`, mount `work_root` at the same path for the server and Hermes, and run Hermes as the server's user or as root. | [Frontend/Backend Live Architecture → Connect Your Own Harness](design/architecture.md#connect-your-own-harness) |

## Browser access

| Error / symptom | Cause & fix | Reference |
|-----------------|-------------|-----------|
| Microphone / WebRTC blocked | Browsers require a secure context. Keep TLS enabled (default HTTPS). Setting `PIPELINE_TLS=false` serves plain HTTP and is intended for headless / API testing only. | [Getting Started](01-getting-started.md#docker-based-deployment) |
| Need plain HTTP for temporary browser testing | Set `PIPELINE_TLS=false`, then mark the origin secure in your browser. Open `chrome://flags/#unsafely-treat-insecure-origin-as-secure` (or `edge://flags/#unsafely-treat-insecure-origin-as-secure`), enable **Insecure origins treated as secure**, add `http://<machine-ip>:7860`, relaunch the browser, and remove the origin when done. | — |
| Remote client on a different network cannot connect | Deploy a TURN server. | [Enable a TURN Server](how-to/enable-turn-server.md) |
