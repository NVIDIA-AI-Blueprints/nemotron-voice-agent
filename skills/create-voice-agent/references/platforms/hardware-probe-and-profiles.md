# Hardware probe + profiles

Use: GPU detect, NIM profiles, disclosure after pick. REF: model-selection,nim-llm-profiles,nemotron-speech,deployment,dgx-spark,jetson-thor,catalog,clarifying/env-secrets.

## Step 0 — mandatory first action (before gate questions)
When the user wants to **create** a voice pipeline, run local probe commands **before any AskQuestion or gate TPL** — including framework/deployment/transport. Parallel OK with `.env` probe (`clarifying/env-secrets.md`).

```
PROC: run Step1+Step2 → classify → store DETECTED_HARDWARE → Step0b fit estimate → tailor deployment row (Local recommended if stack fits) → then gate
FORBID: ask "cloud or local?" before probe | ask deployment when no GPU (auto cloud) | default Cloud rec when GPU fits full stack
```

### Step1 platform (run first)
```bash
cat /sys/class/dmi/id/product_name 2>/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "NO_GPU"
cat /proc/device-tree/model 2>/dev/null | tr -d '\0'
uname -m
```

### Classification rules
| signal | DETECTED_HARDWARE | DEPLOYMENT_PLATFORM if local chosen |
| --- | --- | --- |
| `nvidia-smi` fails / no GPU name | `no_gpu` | **cloud only** — skip deployment ask |
| DMI contains GB10 or product Spark | `dgx_spark` | workstation + LOCAL_HARDWARE_PROFILE=dgx_spark |
| device-tree/model contains Jetson Thor or aarch64+Thor GPU | `jetson_thor` | jetson + LOCAL_HARDWARE_PROFILE=jetson_thor |
| x86_64 + NVIDIA GPU | `workstation` | workstation |
| aarch64 + GPU, not Thor | `workstation` (edge generic) | workstation or jetson per user |

GPU label for MCQ: use `nvidia-smi --query-gpu=name` first line (e.g. `NVIDIA H100 80GB HBM3`).

### Deployment question (row 2) — probe-driven
Run **Step0b fit check** (below) before asking so the option order/tag is correct.

| probe result | action |
| --- | --- |
| `no_gpu` | **Do not ask** deployment. Set `cloud`. Tell user: "No local NVIDIA GPU detected — defaulting to **cloud** deployment." |
| GPU detected + user already stated deployment | use stated value; disclose detected platform in recap |
| GPU detected + **stack fits** (Step0b) | **AskQuestion**: **Local (recommended)** first, **Cloud** second. Local is the default rec when the whole stack fits. |
| GPU detected + **stack does not fit** (Step0b) | **AskQuestion**: **Cloud (recommended)** first, **Local** second; note which slot(s) don't fit. |

#### Step0b — quick fit estimate (before deployment ask)
Rough co-locate check vs probed `memory.total` (single GPU) or summed multi-GPU; refine later in Step2c/Step2b. **Branch on `PIPELINE_MODE` when known:**

```
cascaded fits ≈ reserved(2-4GB) + LLM(smallest nvfp4 for tier, Nano if <90GB) + ASR(~2-4GB) + TTS(~3-6GB) + KV + ≥5GB slack ≤ available
omni fits     ≈ reserved(2-4GB) + Omni vLLM runtime + TTS(~3-6GB) + KV + ≥5GB slack ≤ available   # omit ASR budget
PIPELINE_MODE unknown → defer Local|Cloud recommendation until row 2b (cascaded|omni) is resolved
```

| estimate | recommended | second |
| --- | --- | --- |
| full stack for selected mode ≤ available with slack | **Local** | Cloud |
| only with Nano nvfp4 (tight, cascaded) | **Local** (Nano) | Cloud |
| LLM/Omni alone won't fit even Nano nvfp4, or no slack | **Cloud** | Local (warn slot) |

REQ disclose one line: `GPU <name> <VRAM>GB fits <cascaded|omni> stack → Local recommended (Cloud optional)` or `GPU too tight for <slot> → Cloud recommended`.

TPL (no GPU): `Probed this machine: no NVIDIA GPU. Deployment will be **cloud** (NVCF). Next: …`
TPL (has GPU, fits): MCQ prompt `Deployment?` options `Local — NVIDIA <gpu_name> (recommended)` (or DGX Spark / Jetson Thor label) | `Cloud`
TPL (has GPU, no fit): MCQ prompt `Deployment?` options `Cloud (recommended)` | `Local — NVIDIA <gpu_name> (tight: <slot> may not fit)`

Re-probe if user says they're on a different machine than the agent host.

## Profile gate (before first rec)
FORBID names-only rec. Phase: Probe(nvidia-smi VRAM,compute,count;free -h) → Budget(co-locate slots+KV+TTS slack) → Fetch(matrices if needed) → Filter → Decide(selectors+LLM nim-llm-profiles) → Disclose(same msg: why,precision,profile,VRAM,image,endpoints,GPU pin,combined VRAM)

Local LLM quantization→nim-llm-profiles. Single-GPU: budget all slots first(cascaded 3-slot; omni 2-slot).

## Step2b Local VRAM cap (<90GB) — mandatory before rec/MCQ
When **local** deployment and co-locate GPU `memory.total` **< 90 GiB** (from Step2 `nvidia-smi`):

```
STORE: LOCAL_VRAM_TIER=tight
FORBID: recommend or (recommended)-tag Nemotron3 Super or Ultra
REQ: LLM family = Nemotron3 Nano only
REQ: LLM precision = nvfp4 only (smallest Compatible list-model-profiles id)
DISCLOSE: "GPU <90GB — LLM capped to Nemotron3 Nano NVFP4; ASR+TTS co-located"
```

| cap | rule |
| --- | --- |
| LLM family | **Nano only** — overrides usecase tier (balanced/specialized cannot bump to Super/Ultra) |
| LLM precision | **nvfp4** — FORBID fp8/mxfp4/bf16 unless user explicitly overrides after WARN |
| Budget | Treat VRAM as tighter than `memory.total`: reserve **2–4 GiB** CUDA/OS + **ASR + TTS + KV + ≥5 GiB slack** before claiming fit |
| User picks Super/Ultra | WARN: will not fit on <90GB; offer Nano NVFP4 or cloud |
| Omni local | 30B NVFP4 omni needs ~80GB+ headroom — at 72GB prefer **cascaded Nano** or cloud omni |

`≥90 GiB` on co-locate GPU: tier table applies; still prefer **nvfp4** smallest Compatible profile unless user asks higher precision.

## Omni VRAM
Skip ASR. Budget Omni vLLM+TTS: `required≈2-4GB reserved+omni_runtime+tts_runtime+2-4GB TTS slack; pass if ≤available with ≥5GB slack`
30B NVFP4 omni local ≥~80GB GPU. DGX/Jetson 128GB OK. Orin unsupported. Downsize OMNI_GPU_MEM_UTIL/OMNI_MAX_MODEL_LEN. No nim-llm for omni sidecar→pipeline/omni.md

## Config
| cfg | DEPLOYMENT_PLATFORM | PROFILE | LLM | ASR | TTS | function_id |
| --- | --- | --- | --- | --- | --- | --- |
| Cloud | cloud | default | integrate.api.nvidia.com/v1 | grpc.nvcf.nvidia.com:443 | same | build cards |
| Workstation | workstation | default | :18000\|:8002 | :50152 | :50151 | empty |
| DGX | workstation | dgx_spark | often :8002 | same | same | empty |
| Jetson | jetson | jetson_thor | :18000 | Riva :50052 | :50051 | empty |

## When probe
Before local rec. Skip cloud-only. Reuse: docker ps+curl+inspect profiles.

## Step2 GPU
```bash
nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,memory.free,compute_cap --format=csv,noheader; free -h
df -BG --output=avail / | tail -1
```
Multi-GPU pin LLM/ASR/TTS. RAM<32GB or disk<50GB warn (disk from `df` above).

## Step2c VRAM budget cascaded
`required≈2-4GB+llm+asr+tts+2-4GB TTS slack; pass ≤available ≥5GB slack`
**<90GB GPU:** Nano NVFP4 LLM only (Step2b); budget with conservative `memory.free` not nameplate total.
Fail order: lower KV%/max-model-len → **Nano nvfp4** (already capped) → smaller ASR → lower TTS batch → multi-GPU pin → hybrid cloud slot

## Step3 profiles
| mod | default | backup matrix |
| --- | --- | --- |
| ASR | nemotron-speech/model-selection | docs.nvidia.com/nim/speech/.../asr.html |
| TTS | Magpie Multilingual | .../tts.html |
| LLM pick | model-selection+catalog | build.nvidia.com |
| LLM config | nim-llm-profiles | list-model-profiles |

## Step4 running (pre-rec inventory)
```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
for p in 8002 18000; do curl -sf http://127.0.0.1:$p/v1/models|head -c400; done
```
Health→run.md §Verify. Map ports no assumptions.

## Step4b selective slot reuse (post model-confirm, local only)
After model table confirmed: probe **LLM, ASR, TTS each separately** — image, tags, `/v1/models` or health — vs locked picks. Keep matches; replace only mismatches. Full PROC→`platforms/deployment.md` §Local selective slot reuse. FORBID full `compose down` when one slot wrong.

## Disclosure
Cloud: platform,ids,endpoints,function_ids,NVCF precision. Local: GPU,profile,per-slot image/selector/precision/GPU pin,VRAM line. Named model no fit→warn WAIT.
REQ rec msg: same disclosure as **visible markdown table** (model-selection.md §Rec display) — user must see exact ids before "looks good", not buried in prose only.

## Anti-patterns
pick before probe | catalog best no budget | skip list-model-profiles | fp8 over nvfp4 when nvfp4 listed without user ask | **Super/Ultra or fp8 rec on local GPU <90GB** | **Cloud recommended when GPU fits full stack** | skip Step0b fit estimate before deployment ask
