# LLM

Cascaded Nemotron 3.5 Lightning (default), with Nemotron 3 Super / Ultra when named, plus
Nemotron 3 Nano Omni for the Omni pipeline. Resolve before the proposal
table. Self-hosted fit is mandatory. Cloud skips platform fit. Jetson Thor uses raw vLLM
and follows `platforms/jetson-thor.md`. NIM profile selection below applies to workstation
and DGX Spark.

## Discover

| Step | Where |
| --- | --- |
| Family browse | [NVIDIA Nemotron](https://developer.nvidia.com/topics/ai/nemotron#section-nvidia-nemotron-models) |
| Confirm id / run | `build.nvidia.com/nvidia/<model>` · self-hosted: `?nim=self-hosted` |
| Card / API | `…/<slug>/modelcard` · `docs.api.nvidia.com/nim/reference/nvidia-<slug>` |

Examples: [Lightning cloud](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b) ·
[self-hosted](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b?nim=self-hosted) ·
[Super card](https://build.nvidia.com/nvidia/nemotron-3-super-120b-a12b/modelcard) ·
[Super API](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-super-120b-a12b).

**Omni** is the separate **Nemotron 3 Nano Omni** multimodal model (audio-native), not the
cascaded text LLM and not a Lightning build. It is the only slot that keeps a Nano id.
Exclude it from the cascaded LLM slot. Never interchange.

Every card lists the languages the model supports, and that list is usually shorter than the
speech coverage around it. Check the approved response language against it through
`models/language-routing.md` §LLM before locking this slot.

### Resolve Three Names

The catalog slug, the container image, and the served model id are three different
strings. Treating them as one causes `Access Denied` on pull and a wrong model id at
request time.

| Name | Source | Used by |
| --- | --- | --- |
| Container image | the self-hosted build page or NGC command that actually pulls | `compose.yaml` |
| Profile / tags | `list-model-profiles` on the assigned GPU | `NIM_MODEL_PROFILE` |
| Runtime model id | `GET /v1/models` after the service is healthy | agent code |

The slug on `build.nvidia.com/nvidia/<slug>` records the proposal intent only. It is not
proof of the image URI and not proof of the served id. Do not bake it into agent code
until the listing returns a value. A raw vLLM card commonly passes
`--served-model-name`, so the served id can be a short alias that matches neither the slug
nor the repository name.

Query the NIM port for self-hosted, or `https://integrate.api.nvidia.com/v1/models` for
cloud. Drop `embed`, `guard`, and `vlm` entries. If nothing is up yet, use the cloud
listing and re-check after start.

## Reasoning (Voice Default: Off)

Model default is **on** (`enable_thinking: true` when omitted). That burns turn latency
and can leak into TTS (`<think>`, silence, `redacted_thinking`).

| Intake row | When |
| --- | --- |
| off (propose this) | normal voice agent |
| on, budget 8192 | user asked hard multi-step / specialized. Warn if budget > 16384 |

Verify on the locked model's API reference, for example:
[Lightning](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-5-lightning-30b-a3b) ·
[NIM reasoning](https://docs.nvidia.com/nim/large-language-models/latest/reasoning-model.html).

The model's own build.nvidia.com page is the fastest check. Its prototype panel shows a
runnable request for that exact model, including where the reasoning fields sit. For example, use
[Lightning](https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b). Read that example and the
API reference before generating the request, because field names and nesting move between
model generations.

With an OpenAI-compatible SDK these provider fields travel in `extra_body`, which
merges them into the top level of the request body. A raw HTTP call sets them at the
top level directly. Passing `chat_template_kwargs` as a plain SDK keyword argument is
the error, because the client rejects the unknown argument.

```python
# Off (voice default)
extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

# On (table says so)
extra_body = {
    "chat_template_kwargs": {"enable_thinking": True},
    "reasoning_budget": 8192,
}
```

Prefer top-level `reasoning_budget` when the API lists it (some docs also nest it under
`chat_template_kwargs`). When on, size completion tokens from the card's recommendation,
because a thinking trace plus a spoken answer needs far more than a voice turn alone. Treat
≥1024 as a floor rather than a target, and bound thinking with `reasoning_budget` instead of
starving the answer. Pass `extra_body` the way the framework MCP documents. Do not
invent the wrapper. Frameworks rarely accept a raw request dict, so see where the payload
actually goes in `frameworks/pipecat.md` §Local LLM wiring or
`frameworks/livekit.md` §NVIDIA models.

Cloud still needs `enable_thinking` in `extra_body`. Sampling changes with the mode, and a
card commonly gives separate values for reasoning, for reasoning off, and for tool calling,
so take the set that matches the approved row rather than reusing one. Toggle = edit
constants + restart, not mid-session, unless the framework docs say otherwise.

### Reasoning Parser

A model card's serve command can enable the reasoning parser independently of each request's
thinking mode. Keep the parser enabled for both modes. It separates thinking from `content`
when reasoning is on and applies the model's reasoning-off contract when
`enable_thinking:false`. TTS must read only `content`.

| Reasoning Mode | Serve Flags | Request Setting |
| --- | --- | --- |
| Off (voice default) | Enable the parser required by the locked model card | Set `extra_body.chat_template_kwargs.enable_thinking` to `false` |
| On | Enable the parser required by the locked model card | Set `extra_body.chat_template_kwargs.enable_thinking` to `true`, and keep `reasoning_content` away from TTS |

Parser names and plugin requirements are not shared across this family. Read the row for the
exact model and server you run:

| Locked Model and Server | Current Parser Flag |
| --- | --- |
| Cascaded Lightning on vLLM | `--reasoning-parser nemotron_v3` (built-in, no plugin file) |
| Omni NVFP4 on vLLM | `--reasoning-parser nemotron_v3` |

Both add `--enable-auto-tool-choice --tool-call-parser qwen3_coder`. When a card names a
separate parser-plugin file, it must be downloaded and present before the server starts.
Confirm the parser and any plugin from the model card for the server you run, and never
adapt a vLLM row for a different server such as SGLang.

Prove the result on the streaming endpoint in both modes. `scripts/smoke.sh` asserts
non-empty `delta.content` and empty or absent `delta.reasoning_content` when reasoning is
off (`output-contract.md` §Smoke Before Client). One non-streaming answer does not prove it.

## Serve Flags

Every serve-time flag must trace to one of three sources: the locked model card, the memory
controls in §Tight fit, or a fatal log line that names the flag. Nothing else belongs on the
command line.

Performance and scheduling flags are the usual trap. Reaching for eager mode, a scheduling
toggle, or CUDA MPS variables to chase a slow or unstable first boot adds variables while the
real cause is normally an engine build, a kernel compile, or the memory budget. Each
speculative flag then has to be reverted, and a revert is indistinguishable from a fix.

When a flag looks necessary, name the log line that requires it, change that one thing, and
rerun `scripts/smoke.sh` before touching anything else.

## Platform Fit (Self-Hosted)

Hardware first from `preflight.md`. Never propose model / precision / TP until verified
for the probed GPU.

### 1. Support Matrix HTML

Fetch HTML only (not UI dropdowns, not `.html.md`):

`https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html`

Match rows using `data-model`, `data-precision`, `data-tp`, and `data-gpus`. Normalize the
probed name (`NVIDIA H100 80GB HBM3` → `NVIDIA-H100-80GB-HBM3`). Keep a row only when
short name, precision, and SKU all match. No row → unsupported. Pick another precision,
TP, model, or cloud. Do not guess a close SKU.

| Model | Section |
| --- | --- |
| Lightning | [#nemotron-3-5-lightning-30b-a3b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-5-lightning-30b-a3b) |
| Super | [#nemotron-3-super-120b-a12b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-super-120b-a12b) |
| Ultra | [#nemotron-3-ultra-550b-a55b](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html#nemotron-3-ultra-550b-a55b) |

“Verified GPUs” under a section is overall. It does not replace per-row `data-gpus`.

### 2. Select a NIM Model Profile

Before proposal, use the support matrix to select a candidate profile and mark shared-GPU
fit provisional. After the user approves self-hosting and container readiness passes, run
`list-model-profiles` on that NIM for the probed GPU. Pin the exact result before
generating Compose. This is mandatory on every workstation / DGX Spark NIM LLM
deployment. Buckets:

```bash
docker run --rm --gpus=all <nim_llm_image> list-model-profiles
```

Replace `all` with the exact planned GPU assignment when the LLM is pinned. Copy
`<nim_llm_image>` from the current self-hosted build page. Do not infer the image or tag.

| Bucket | Action |
| --- | --- |
| Compatible | use it |
| Low memory | apply the suggested `--max-model-len` / `NIM_MAX_MODEL_LEN` if the user accepts |
| Incompatible | use lower precision on the same image, or a smaller LLM |

Prefer the smallest Compatible precision that fits (nvfp4 > mxfp4 > w4a16 > fp8 > bf16)
unless the user asked for a heavier one. On a local GPU under ~90 GB, prefer Compatible
nvfp4 when listed. Do not default to a heavier precision if a lighter one is Compatible.

Set `NIM_MODEL_PROFILE` in the launch environment to one of:

- a description such as `vllm-nvfp4-tp1-pp1` (pattern:
  `<backend>-<precision>-tp<N>-pp1`, backend `vllm` | `sglang` | `trtllm`)
- the 64-char profile id from the listing
- leave unset / `default` only when you intentionally want manifest auto-pick

Never use `NIM_TAGS_SELECTOR` on the LLM. Cloud LLM skips this section. For same-image
OOM or latency, re-run `list-model-profiles`, pin a lighter profile or shorter max length,
update the launch command, then health-check. A different model id requires discovery
again.

On a shared GPU, never leave profile selection at `default`. Pin the exact Compatible
profile and verify startup logs report the expected backend and precision. A profile being
Compatible means the LLM can run on that GPU. It does not mean LLM + ASR + TTS fit
together.

Docs:
[model profiles and selection](https://docs.nvidia.com/nim/large-language-models/latest/deployment/model-profiles-and-selection.html).

### 3. Precision and Weights for Planning

Matrix / container win over these hints.

| GPU | Precision default |
| --- | --- |
| Blackwell | NVFP4 when supported by the locked model path |
| Ada / Hopper (CC ≥ 8.9) | NVFP4 when Compatible, else the lightest Compatible profile (W4A16 for Lightning, FP8 for Super / Ultra) |
| Ampere and older | W4A16 for Lightning, otherwise BF16 (large, often needs its own GPU) |

Precision is a tag inside `NIM_MODEL_PROFILE`, for example `vllm-bf16-tp1-pp1`, not a separate
env. The Compatible profile listing wins over this planning table. Lightning NVFP4 requires
Blackwell (SM 10.0+), so Ampere and Hopper hosts use W4A16 rather than FP8. Read the row
rather than assuming a precision exists.

| Slot | Weights (est.) |
| --- | --- |
| Lightning NVFP4 | ~19 GB |
| Lightning W4A16 | ~19 GB |
| Lightning BF16 | ~63 GB |
| Omni 30B NVFP4 | ~15 GB (ASR in-process) |

The support matrix lists exact min VRAM per GPU per precision and TP; use it over these
estimates. Super / Ultra: on one workstation GPU, say they will not fit. Offer Lightning
local or that model in the cloud. Do not silently substitute.

### 4. Tight Fit

Apply this section to every shared single-GPU layout, not only after an OOM.

| Deployment path | Knobs | Source |
| --- | --- | --- |
| Workstation / DGX NIM | `NIM_MAX_MODEL_LEN`, documented runtime memory limit | current NIM environment and memory troubleshooting docs |
| Raw vLLM, including Jetson Thor | `--gpu-memory-utilization`, `--max-model-len` | locked Hugging Face model card for the required flags, [vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/) for their meaning and current defaults |

Both server paths fill their runtime budget with KV cache after weights and overhead. The
documented default fraction is 0.9 or higher on each path, which can starve co-located
speech even when quantized weights are small, so read the exact current default from the
source in the table.

On raw vLLM that fraction is per-instance and measured against total device memory. The
reference states it does not account for memory another process already holds, so a
co-located speech container does not lower it. Derive the fraction from the budget below
rather than expecting vLLM to leave room.

Before first start:

1. re-read free VRAM immediately before launch, including other GPU processes
2. calculate `LLM budget = current free VRAM - startup reserve` minus the reserve of every
   speech slot placed on this GPU, which is TTS plus ASR for Cascaded and TTS alone for
   Omni. Slots in the cloud or on another GPU take nothing from this budget
3. reject co-location when that budget cannot hold the selected build's weights, runtime
   overhead, and a usable KV cache. Move a slot instead, or take a lighter build, which
   is a smaller Compatible profile on NIM and a lighter quantization from the locked card
   on raw vLLM
4. set the context to the smallest value that satisfies the approved use case, through
   `NIM_MAX_MODEL_LEN` on NIM or `--max-model-len` on raw vLLM
5. cap runtime memory so the server cannot exceed `LLM budget`, which as a fraction is no
   higher than `LLM budget / total GPU memory`:
   - NIM: pin the exact `NIM_MODEL_PROFILE`, then take exactly one control from the
     selected image's current docs, either `--gpu-memory-utilization` passed through
     `NIM_PASSTHROUGH_ARGS` or `NIM_KVCACHE_PERCENT` when that image documents it
   - raw vLLM: pass `--gpu-memory-utilization` directly, or an absolute KV cache size
     when the current reference documents one

Never set both runtime memory controls. Do not copy either value from another image or
model version.
If the calculated cap leaves no usable KV cache, the layout does not fit. Do not keep
lowering the cap to force co-location.

On a shared single GPU the chosen control is **required in the generated `compose.yaml`**,
not a troubleshooting step applied after an OOM. Derive its value from the budget above
rather than reusing a number. Leaving the container default in place is what lets the LLM
take the whole device and starve TTS and ASR.

At deployment, verify the measured budget before each service starts. NIM uses
`platforms/deployment.md` §Shared-GPU memory gate. Raw vLLM on Jetson Thor uses the
unified-memory check in `platforms/jetson-thor.md` §Start and verify. Do not let TTS
discover an invalid LLM budget through repeated OOM restarts.

Sources:
[environment variables](https://docs.nvidia.com/nim/large-language-models/latest/reference/environment-variables.html) ·
[GPU memory troubleshooting](https://docs.nvidia.com/nim/large-language-models/latest/troubleshooting/memory.html) ·
[vLLM `serve` CLI reference](https://docs.vllm.ai/en/stable/cli/serve/).

Do not choose a knob from the pipeline name. Choose it from the actual server path.

## Anti-Patterns

- Voice agent with reasoning left at model default (on).
- Reasoning parser or plugin left enabled while the reasoning row is off.
- Adding a serve flag the card does not list and no fatal log demands.
- `chat_template_kwargs` passed as a plain SDK argument rather than through `extra_body`.
  Reasoning flags in `.env`.
- Reasoning on locally without the reasoning parser and plugin file the card requires.
- Cascaded slot → omni model. Model ids in `.env`.
- Propose from memory / Verified GPUs list alone. Click matrix UI or use `.html.md` for fit.
- Override a failed matrix / `list-model-profiles` check. Guess a SKU. `NIM_TAGS_SELECTOR` on LLM.
- Treat quantized weight size as total LLM VRAM or leave vLLM at its default memory budget
  while co-locating speech.
