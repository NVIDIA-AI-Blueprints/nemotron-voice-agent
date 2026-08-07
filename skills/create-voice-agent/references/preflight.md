# Preflight

Sections 1-3 are step 0 of `intake.md` and run before the first word to the user. They
establish host class and base credentials. Section 4 runs during proposal after pipeline,
framework, and candidate models are known.

Per-slot footprints and knobs live in `models/llm.md`, `models/asr.md`, and
`models/tts.md`. Section 4 combines them into self-hosted, cloud, or hybrid placement.

| Output | Feeds |
| --- | --- |
| Host class | which platform guide applies |
| Usable VRAM | later deployment fit |
| Compute capability | LLM precision planning (`models/llm.md`) |
| Deployment row | intake proposal |
| A blocking credential, if any | halts everything before the table |

## 1. Probe the host

Collect the operating system, GPU inventory, machine identity, system RAM, and free disk.
Failures on a machine without a GPU are expected and are not errors.

`nvidia-smi` is the detection method. Ask it for everything in one query, including
`compute_cap`, because compute capability is what decides the precision:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader
```

A non-zero exit or empty output means no usable NVIDIA GPU.

Report usable VRAM as `memory.free`, not nameplate `memory.total`. Add VRAM across GPUs
only when the user intends to pin slots to separate devices.

Detect the operating system first, for example with `uname -s`. Self-hosted NIM models
require Linux, so on macOS or Windows the deployment is cloud. The identity paths below
exist only on Linux; their absence on other systems is expected, not an error.

On Linux, read the machine identity from `/sys/class/dmi/id/product_name` and
`/proc/device-tree/model`, which is the only reliable way to tell DGX Spark and Jetson Thor
apart from a generic host.

Re-probe if the user says they will run on a different machine from the one hosting the
agent.

## 2. Classify the host

| Signal | Host class |
| --- | --- |
| OS is macOS or Windows | `no_gpu`, cloud only (NIM needs Linux) |
| `nvidia-smi` fails or names no GPU | `no_gpu`, cloud is the only option |
| DMI product name contains GB10 or Spark | `dgx_spark`, see `platforms/dgx-spark.md` |
| Device tree model names Jetson Thor | `jetson_thor`, see `platforms/jetson-thor.md` |
| x86_64 with an NVIDIA GPU | `workstation` |
| aarch64 with a GPU that is not Thor | `unsupported_edge`, cloud only |

Name the GPU to the user exactly as `nvidia-smi` reports it, for example
`NVIDIA RTX 6000 Ada Generation`. Warn when system RAM is under 32 GiB or free disk is
under 50 GiB, because NIM images are large.

## 3. Check the credentials

`NVIDIA_API_KEY` is needed on every path, for cloud inference, for pulling NIM images from
`nvcr.io`, and for model discovery. Check it now.

The rest depend on choices intake has not made yet, so check them the moment the table's
shape is known, and before writing anything.

| Also required | When |
| --- | --- |
| `HF_TOKEN` | local raw vLLM pulls from Hugging Face, including Jetson Thor and any workstation / DGX Omni path whose locked source is Hugging Face |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | any LiveKit run, cloud or self-hosted |

Check the shell environment first, then `.env`. A key exported in the shell counts even
when there is no `.env` file at all. Parse `.env`, never source it. Count a key as missing
when it is absent, empty, or still carries a placeholder such as `your-key-here`,
`changeme`, or `API_KEY_REQUIRED`.

When a key is missing, stop. Name it, say how to get it, and wait for the user to confirm
it is set. Ask for every missing key in one message rather than one at a time.

| Key | Where the user gets it |
| --- | --- |
| `NVIDIA_API_KEY` | https://build.nvidia.com/settings/api-keys, starts with `nvapi-`, shown once |
| `HF_TOKEN` | Hugging Face settings, access tokens, read access is enough |
| `LIVEKIT_*` | the `lk cloud auth` browser flow, or project settings, API keys, create key |

Fetch the LiveKit steps from the LiveKit documentation MCP before answering, because that
console flow changes. Self-hosted LiveKit is an override the user has to ask for by name.

Write the keys into `.env.example` as placeholders. Never write a real value to disk, and
never put model ids or endpoints in `.env`, which holds secrets only.

## 4. Deployment fit

During proposal, set the intake **Deployment** row from usable VRAM. Model weights are
planning hints, not service footprints. Authoritative speech VRAM comes from the selected
matrix profiles. The LLM support matrix supplies the proposal candidate. Post-approval
`list-model-profiles` confirms the exact profile. Runtime memory also includes KV cache,
activations, CUDA graphs, and engine overhead.

For `no_gpu` or `unsupported_edge`, set Deployment to cloud and skip local fit.
Jetson Thor uses `platforms/jetson-thor.md`. DGX Spark uses
`platforms/dgx-spark.md` and available unified memory. Neither uses a fixed workstation
VRAM threshold.

For a workstation, build a runtime budget before approving co-location:

1. select a candidate Compatible LLM quantization profile from the support matrix.
   Post-approval `list-model-profiles` pins the exact profile
2. reserve the selected ASR matrix GPU memory
3. reserve the selected TTS batch profile GPU memory
4. reserve GPU memory for the framework, CUDA, service startup, and engine build
5. give only the remainder to the LLM runtime, including weights and KV cache

That budget is a hard gate, and the arithmetic must pass before anything starts:

```text
llm_weights + llm_kv_and_activations + tts_vram + asr_vram + startup_overhead
    <= free VRAM measured now
```

Omni omits the ASR term, so its stack is Omni + TTS. Hybrid counts only the slots assigned
to this GPU, because cloud slots consume no local VRAM. If the sum does not fit, change the
profile, the TTS batch profile, or the placement. Do not start the stack and let the last
service discover the shortfall as an OOM.

Recalculate whenever profile, context length, TTS batch size, or slot placement changes.

Weight estimates alone can support only `provisional co-location`, never the final
`self-hosted, co-located` status. Before generating Compose, lock the LLM profile and
memory controls from `models/llm.md` plus the TTS batch profile from `models/tts.md`.
Require the Shared-GPU memory gate in `platforms/deployment.md`.

| Fit result | Deployment row |
| --- | --- |
| Runtime budgets fit one GPU with startup headroom | self-hosted, provisional co-location |
| Slots fit only when pinned across GPUs | self-hosted, show placement (`platforms/deployment.md` §Scaling across GPUs) |
| Only some slots fit | hybrid, name each cloud slot |
| No local layout fits | cloud |

Use a Compatible NVFP4 profile when available. Otherwise, use the next compatible precision from `models/llm.md`. Do not apply a heavier-precision estimate after selecting NVFP4.

For every shared single-GPU layout, use the controls in `models/llm.md` (profile, context,
runtime memory cap) and `models/tts.md` (explicit batch profile). Re-read this section
whenever a slot moves GPU or to cloud, or the user asks why something will not fit.

Host GPU detection is not container readiness. After self-hosting is approved and before
pulling an image, require Docker, Compose, NVIDIA Container Toolkit, in-container GPU
visibility, and writable cache paths through `platforms/readiness.md`.

## Anti-patterns

- Asking whether to use cloud or self-hosted before probing.
- Offering a deployment choice on a machine with no GPU.
- Routing unsupported ARM64 edge hardware to workstation NIM images.
- Reporting nameplate VRAM instead of what is actually free.
- Approving co-location from model weights without budgeting KV cache and service startup.
- Leaving the LLM runtime memory cap or TTS batch profile implicit on a shared GPU.
- Generating files before checking credentials.
- Treating an `.env.example` placeholder as a real key.
