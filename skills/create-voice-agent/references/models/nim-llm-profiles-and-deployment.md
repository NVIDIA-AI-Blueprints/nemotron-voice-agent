# NIM LLM profile | local LLM config after id locked

NOT model pick→model-selection+catalog. Trigger: precision,TP,OOM,NIM_MODEL_PROFILE,max-model-len,2.x migration.
Hierarchy: this file+list-model-profiles first; external only if unclear.
NIM=https://docs.nvidia.com/nim/large-language-models/latest

## PROC
```
1 LLM image/id locked (model-selection)
2 docker run --rm --gpus=all <locked_image> list-model-profiles  # mandatory every local deploy
3 smallest compatible: nvfp4>mxfp4>fp8>bf16 (Compatible bucket)
3b **Local GPU <90 GiB:** nvfp4 **mandatory** — pick smallest Compatible nvfp4 profile; FORBID fp8/bf16 default
4 hardware-probe 2c budget vs requires>=X GB/gpu
5 set NIM_MODEL_PROFILE + NIM_MAX_MODEL_LEN/KV% compose
6 WebFetch NIM/deployment/model-profiles-and-selection.html only if 2-3 unclear
```
FORBID NIM_TAGS_SELECTOR on LLM(2.x removed). Use NIM_MODEL_PROFILE.

## Docs (NIM base)
prerequisites|configuration|installation|quickstart|model-profiles-and-selection|model-download|model-free-nim|environment-variables|gpu-memory-oom-errors|1.x-migration-guide

## Profile pattern
`<backend>-<precision>-tp<N>-pp1[-lora]` backend=vllm|sglang|trtllm | precision=bf16|fp8|mxfp4|nvfp4

## list-model-profiles buckets
Compatible→use | Low memory→NIM_MAX_MODEL_LEN per hint | Incompatible→lower precision same image OR smaller LLM model-selection

## NIM_MODEL_PROFILE
unset=manifest auto | default=hardware pick | description e.g. vllm-nvfp4-tp1-pp1 | 64-char id
Compose: `NIM_MODEL_PROFILE=${LLM_NIM_PROFILE:-<smallest>}` — do not default fp8 if nvfp4 listed.

## Precision default
smallest from list-model-profiles. User requests higher→honor. Different family/size→model-selection.
**Local <90 GiB:** nvfp4 only unless user explicitly accepts WARN after fit check.

## Iterate faster same image
troubleshoot latency→list-model-profiles→lower precision/max-model-len→update compose→health+curl /v1/models
Different LLM→model-selection then this file.

## 2.x migration LLM
NIM_TAGS_SELECTOR llm_engine→NIM_MODEL_PROFILE | NIM_CUSTOM_SELECTOR_CLASSES→NIM_MODEL_PROFILE

Cloud LLM: NVCF-managed catalog.md. Local only here.

## Anti-patterns
pick LLM via list-model-profiles alone | nim-llm before image locked | profile without list-model-profiles
