# Jetson Thor

Read when `preflight.md` classifies the target as `jetson_thor`.

Jetson Thor does **not** use the workstation / DGX NIM deployment path. Run the LLM with
vLLM from Hugging Face weights and serve speech with the Riva Speech Skills L4T stack.
Orin-class Jetsons are unsupported.

## Source of truth

Before generating or running the Thor speech deployment, read the current
[NVIDIA Riva Quick Start Guide](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/quick-start-guide.html).
Use it as the authority for:

- supported Jetson / JetPack versions
- NGC access, licensing, disk, power-mode, and container-runtime prerequisites
- the current ARM64 Quick Start release and setup flow
- Riva service ports and `config.sh` fields
- ASR / TTS deployment and readiness instructions

Follow the current guide, then apply the voice-agent choices below.
Pass `platforms/readiness.md`, then generate the Thor services through
`output-contract.md`. Do not ship a static Thor Compose recipe.

## Platform stack

| Pipeline | LLM | Speech |
| --- | --- | --- |
| Cascaded | [NVIDIA Nemotron 3.5 Lightning NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4) with vLLM | Riva streaming ASR + Riva TTS |
| Omni | [Nemotron 3 Nano Omni Reasoning NVFP4](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) with vLLM | Riva TTS only (Omni performs ASR) |

This changes the model table. Do not present workstation NIM images, NIM profiles,
`NIM_TAGS_SELECTOR`, Magpie NIM, or speech function ids as the Thor deployment.
Resolve the exact Riva ASR / TTS models from the downloaded ARM64 Quick Start `config.sh`
and disclose the platform-specific substitutions before build. Lock supported locales
through `models/language-routing.md`.

## Prerequisites

- Jetson Thor with the JetPack release supported by the current Riva guide
- Docker Engine and Docker Compose
- NGC CLI configured with `NVIDIA_API_KEY`
- `HF_TOKEN` with access to the locked Nemotron Hugging Face repository
- Disk capacity required by the current Riva Quick Start plus the selected models and
  compiled engines

Use the versions compatible with the installed JetPack release. Resolve the current Riva
ARM64 Quick Start release from the guide and NGC resource:

`nvidia/riva/riva_quickstart_arm64`

Do not copy an x86 Riva or NIM image onto Thor.

## One-time Riva model build

Follow the current Riva Quick Start to download the JetPack-compatible ARM64 bundle. Keep
its model repository outside the generated project so it survives project rebuilds.

1. Open the downloaded `config.sh`.
2. Cascaded: enable ASR and TTS. Choose a streaming ASR profile and the required language.
3. Omni: disable ASR and enable TTS only.
4. Set the ASR / TTS languages and models available in that `config.sh`.
5. Clear §Model path and credentials before running anything.
6. Run `riva_init.sh` once. It downloads the selected models and compiles TensorRT engines
   into the resolved model repository.
7. Do not run `riva_start.sh` when the generated deployment will start Riva itself.

Initialization commonly takes 30 to 60 minutes. Preserve the resulting model repository and
mount it into the Riva service.

### Model path and credentials

Three `config.sh` behaviours each cost a full re-initialization when missed. Check all three
before running `riva_init.sh`.

**The tegra path can override `riva_model_loc`.** On the tegra branch the script has been
observed to resolve the model repository under the current working directory instead of the
value set in `config.sh`, without warning. Read the path handling in the release you actually
downloaded, then assert where the models landed after initialization rather than trusting the
configured value. Mount the resolved path, not the intended one.

**Reusing an existing Quick Start carries root ownership.** `riva_init.sh` runs as root, so an
existing `model_repository` is root-owned. Copying that tree into a new Quick Start directory
carries the ownership across and the container user may not be able to read it. Re-initialize
into a fresh path, or mount the original repository where it already lives.

**Enterprise credentials are all three or none.** Setting `RIVA_API_KEY` and `RIVA_EULA`
switches the Quick Start onto its Enterprise path, which then requires `RIVA_API_NGC_ORG`.
Two of the three fails during initialization in a way that reads like a bad key. Set all three
when the current guide documents Enterprise use for this release, otherwise leave all three
unset and use the standard NGC flow.

## LLM model source

Thor reads two sources, and neither one replaces the other:

| Source | Authority over |
| --- | --- |
| Locked Hugging Face model card | vLLM container/version, installation, parser files, request format, and which flags this model requires |
| [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) | what each serve flag means, its exact spelling, and its current default |

| Pipeline | Hugging Face repository id |
| --- | --- |
| Cascaded | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` |
| Omni | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` |

Read the relevant card immediately before generating the deployment. Do not reuse the
cascaded launch command for Omni. Omni has model-specific vLLM and audio requirements.

### What to take from the card

Work through the card's vLLM section and record every item below. Skimming it for the launch
command is how the platform-specific pieces get missed.

| Item | Why it matters |
| --- | --- |
| Platform-specific container | The card links a separate vLLM container for Jetson Thor and DGX Spark. Use that, not the generic install line written for a workstation |
| Minimum vLLM version | The card pins a floor. An older build can accept every flag and still fail on this quantization |
| Linked cookbook | Fuller deployment guidance than the card's own summary |
| Parser plugin file | A separate download that must exist before the server starts, and only when reasoning is on |
| Required environment variables | The NVFP4 kernel path is enabled by environment variables rather than flags |
| Default context length | The repository configuration carries its own default, so omitting the context flag inherits it |
| Sampling settings | The card gives different values for reasoning on, reasoning off, and tool calling |
| Supported response languages | Compare against the approved response language (`models/language-routing.md` §LLM) |

The card's hardware-compatibility summary field is not the deployment authority here. Its
vLLM section is, because that is where the platform container is named.

A card launch command is an example, not a memory budget for this device. The cascaded
card's example passes no GPU memory fraction and serves a very long context, so copying it
verbatim leaves the vLLM default in place and sizes the KV cache for a device that is not
also running Riva. Treat memory and context as values you calculate, in §Memory and
context flags.

Authenticate with `HF_TOKEN`. vLLM can download the repository directly from the Hub, so
a separate model download is not required unless the model card instructs it. Verify
access with `hf auth whoami`. Use `hf auth login` when authentication is missing.

## LLM with vLLM

Use the current vLLM image/version required by the locked model card and serve the exact
Hugging Face repository id above. Generate the launch command from that card's current
deployment guidance, then translate it into the generated `llm` or `omni` Compose service.
Thor uses raw vLLM, so:

- `HF_TOKEN` supplies model access.
- `NIM_MODEL_PROFILE` and `list-model-profiles` do not apply.
- The Omni repository name does not enable reasoning by itself.
- Reasoning off has two halves. In agent requests set
  `extra_body.chat_template_kwargs.enable_thinking` to `false`. In the serve command omit the
  reasoning parser and its plugin, because the card's command assumes reasoning on and a
  parser left in place is what makes a Thor agent go mute with no error in any log
  (`models/llm.md` §Reasoning parser).
- Take the flags this model requires from the current card, and each flag's meaning and
  current default from the vLLM `serve` CLI reference. Add nothing the card does not list
  unless a fatal log names it (`models/llm.md` §Serve flags).
- Confirm `/v1/models` serves the locked model before starting the agent, and put its
  exact served id in the agent rather than the Hugging Face repository name.
- The framework client traps are the same as any local endpoint. Follow
  `frameworks/pipecat.md` §Local LLM wiring or `frameworks/livekit.md` §NVIDIA models.

### Memory and context flags

`--gpu-memory-utilization` and `--max-model-len` are **required in the generated Compose
service** on Thor, because vLLM and Riva share one device. They are not tuning applied
after an OOM.

The memory fraction is per-instance and measured against total device memory. The reference
states it does not account for memory another process already holds, so a running Riva does
not lower it and vLLM will not leave room on its own. Calculate the fraction with the
arithmetic in `models/llm.md` §Tight fit, treating the Riva configuration you selected plus
startup headroom as memory the LLM cannot have.

Set the context to the smallest length that satisfies the approved use case rather than the
card's example length. KV cache grows with context, and a voice turn is short. The
repository configuration also carries its own default context, so omitting the flag inherits
that default instead of something small. When the
current reference documents an absolute KV cache size such as `--kv-cache-memory-bytes`,
prefer it over a fraction here, since an absolute budget is easier to reconcile with Riva
than a fraction of a device both services share.

Raise either value only after the complete vLLM + Riva stack is stable.

### First boot compiles kernels

The NVFP4 path uses FlashInfer kernels that the card enables through environment variables.
On first boot those kernels can be compiled on the device, which has been observed to run
around 45 minutes before the server answers. Rising `nvcc` and `cicc` compile output is the
progress signal, so read the logs instead of treating the wait as a hang.

Mount a persistent cache for the compiled kernels alongside the Hugging Face cache, so later
starts skip the compile. Resolve the current cache location from the vLLM and FlashInfer
documentation for the version the card pins. Without that mount every container replacement
pays the compile again. No serve flag shortens a kernel compile.

### Host memory at startup

vLLM's startup check has been observed to read Linux **free** memory rather than **available**
memory on this platform. Page cache counts against it, so a host with plenty of reclaimable
memory can still fail to start once model downloads and engine builds have filled the cache.

Read both numbers before starting vLLM. When free memory is the blocker, reclaim page cache
or restart the host rather than lowering the model configuration. This failure and a real
device OOM look alike and their fixes are opposite, so confirm which one the log describes
before changing the memory budget.

## Riva services

Use the L4T Riva Speech image compatible with the Quick Start and mount the generated model
repository. Riva serves ASR and TTS through one gRPC endpoint. Read
`riva_speech_api_port` from the current Quick Start `config.sh` and map that port in the
generated `riva` Compose service.

| Service | Project endpoint |
| --- | --- |
| LLM OpenAI-compatible API | mapped port from the generated vLLM service |
| Riva ASR + TTS gRPC | port from the current `riva_speech_api_port` |

Function ids are empty. For cascaded mode, never hand over while the shared Riva gRPC
endpoint is unavailable. For TTS, discover and use a voice exposed by the running Riva
service rather than a Magpie voice id.

For approved domain terms, follow the Jetson Thor branch in
`domain/speech-customization.md`. Use Riva request-time word boosting and pronunciation
only when the selected ARM64 Quick Start model supports them.

### Language on the Quick Start ASR

The streaming ASR model a Quick Start release offers may be a multilingual auto-detecting
model. On that kind of model a requested locale is advisory and does not restrict
recognition, so Hindi audio returns Devanagari and English audio returns English on the same
stream.

Say this in the proposal instead of presenting a fixed locale as a guarantee, and say plainly
when the release contains no single-language model for the requested locale. Resolve what the
selected release actually offers through `models/language-routing.md`.

## Resource sharing

Thor shares unified memory and one GPU between vLLM and Riva. Do not prescribe CPU
pinning or fixed compute splits. Start from the compute and scheduling defaults supported by
the current vLLM, Riva, and JetPack releases. Memory is the exception, because §Memory and
context flags sets those values explicitly. If measured contention causes audio glitches,
consult the current platform guidance before applying CUDA MPS tuning.

If the stack still OOMs with the calculated budget in place, lower `--max-model-len` first,
then the memory fraction, before changing the approved model.

## Start and verify

Assert each result. Do not read a step as done because the previous one produced output.

1. Start the generated vLLM Compose service. Wait for `/v1/models` and confirm it returns
   the locked served id.
2. Send one streaming chat completion with reasoning off. Assert non-empty `delta.content`
   and empty or absent `delta.reasoning_content`.
3. Measure available unified memory. Start Riva only when the remaining budget covers the
   selected Quick Start configuration plus startup headroom. Otherwise stop vLLM and lower
   `--max-model-len` or the memory fraction.
4. Start the generated `riva` service. Query `GetRivaSynthesisConfig` and assert the loaded
   TTS model plus a voice the service actually returned.
5. Cascaded only: query `GetRivaSpeechRecognitionConfig` and assert the loaded ASR model and
   language.
6. Cascaded only: synthesize one sentence in the approved locale and feed that audio back
   through streaming ASR. This is the only check that proves both speech directions on the
   shared Riva endpoint before a microphone is involved.
7. Run `scripts/smoke.sh`.
8. Start the framework agent.
9. Complete a spoken exchange (`operations/run.md`).

Riva's first start after initialization can still spend a long time loading and building
before it reports ready. Treat it like the first-boot window in `platforms/deployment.md`
and read the logs rather than restarting.

## Anti-patterns

- Following build.nvidia.com NIM launch instructions on Thor.
- Using `NIM_MODEL_PROFILE` or Speech `NIM_TAGS_SELECTOR` on the Thor stack.
- Running `riva_start.sh` after the generated deployment already owns Riva startup.
- Selecting offline ASR for a live voice agent.
- Starting from a Magpie NIM voice id instead of querying Riva voices.
- Copying a card launch command verbatim, which leaves the vLLM memory default in place and
  sizes the context for a device that is not also running Riva.
- Keeping the reasoning parser while the reasoning row is off.
- Adding eager mode, scheduling toggles, or CUDA MPS variables to chase a slow first boot.
- Mounting the `riva_model_loc` you configured without asserting where the models landed.
- Copying a root-owned `model_repository` into a new Quick Start directory.
- Setting some of `RIVA_API_KEY`, `RIVA_EULA`, and `RIVA_API_NGC_ORG` but not all three.
- Presenting a fixed locale as single-language recognition on an auto-detecting model.
- Treating unified 128 GB memory as entirely available to vLLM.
