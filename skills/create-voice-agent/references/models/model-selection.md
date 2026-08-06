# Model selection | MCQ pick→disclose→implement

Trigger: open LLM/ASR/TTS slot OR user finished per-slot MCQs (`clarifying/model-slot-mcq.md`). FORBID silent pick+scaffold. FORBID `name models or I'll choose`.

**Usecase tier** (`usecase-before-rec.md`) marks which MCQ option gets `(recommended)` — run when tier unknown before model-slot MCQ.

Hierarchy: 1 skill refs 2 local discovery 3 external docs if unclear 4 user MCQ/Other overrides

## When
| user | action |
| --- | --- |
| named all three | use ids; local probe+disclose |
| open slot(s) | model-slot-mcq per open slot → disclosure table → WAIT → implement |
| you choose / pick for me | **same** per-slot MCQs — not a delegate shortcut |
| partial named | MCQ open slots only |
| Other picked | WAIT free-text → validate → disclose |

## PROC
```
0 usecase if tier unknown (marks recommended on MCQ options)
1 model-slot-mcq: AskQuestion LLM → ASR → TTS (skip named slots)
2 context: deployment; language→language-routing; vertical speech→fetch customization
2.5 checklist: nvidia-smi+free -h | VRAM budget | nim-llm profile | co-locate sum
3 disclosure one msg: full stack table (§Rec display) incl. Reasoning row → looks good/swap. No files.
4 reply: confirm→lock; swap→re-pick slot MCQ or name swap
5 implement: cloud function_id; local LLM curl /v1/models
```

## Defaults ((recommended) on MCQ)
| slot | default (EN) | multilingual |
| --- | --- | --- |
| LLM | tier table in usecase-before-rec | same whitelist |
| TTS | Magpie Multilingual (en-US voice) | Magpie Multilingual + locale voice |
| ASR EN only | Nemotron ASR Streaming (cache-aware) | — |
| ASR auto-detect multilingual | — | Parakeet RNNT 1.1B Multilingual |
| ASR single locked non-EN lang | Nemotron ASR Streaming Multilingual | per language-routing |

## Source
| slot | pick | config |
| --- | --- | --- |
| ASR/TTS | nemotron-speech/model-selection+matrices | NIM_TAGS_SELECTOR |
| LLM cloud | catalog+GET /v1/models | NVCF |
| LLM local | NGC+curl :18000\|:8002 | nim-llm-profiles-and-deployment.md |

Local LLM: image/id first → nim-llm profile

## Rec display (REQ — after MCQ, same turn as confirm ask)
FORBID confirm-only ("Does this look good?", "Proceed with these picks?") without printing the actual model ids/names the user is confirming.
REQ: markdown table or bullet block with **every locked slot** before WAIT:

| Slot | Model id / name | Voice (TTS) | Endpoint | function_id / selector | VRAM | Why |
| --- | --- | --- | --- | --- | --- | --- |
| LLM | `nvidia/nemotron-3-nano-...` | — | integrate.api.../v1 | — | X GB | general tier … |
| ASR | `nemotron-asr-streaming-...` | — | grpc.nvcf:443 | uuid | X GB | EN streaming … |
| TTS | `magpie-tts-multilingual` | `Magpie-Multilingual.EN-US.Aria` | grpc.nvcf:443 | uuid | X GB | locale … |
| Reasoning | OFF or ON, budget N | — | — | — | — | tier from usecase-before-rec |

Omni (2 rows): Omni model id + TTS only — FORBID ASR row.
Local: include `NIM_MODEL_PROFILE` / `NIM_TAGS_SELECTOR`, GPU pin, image tag when known.
Close with: `Reply looks good to lock, or name the slot to swap. No files until you confirm.`

Rec TPL (cascaded cloud excerpt):
```
| Slot | Model | Voice | Endpoint | function_id | Why |
| LLM | nvidia/nemotron-3-nano-... | — | https://integrate.api.nvidia.com/v1 | — | low latency voice |
| ASR | nemotron-asr-streaming-en | — | grpc.nvcf.nvidia.com:443 | <uuid> | X GB | EN streaming low-latency |
| TTS | magpie-tts-multilingual | Magpie-Multilingual.EN-US.Aria | grpc.nvcf.nvidia.com:443 | <uuid> | multilingual voice |
Combined VRAM: n/a (cloud). Alternatives: …
Reply looks good to lock, or swap a slot.
```

## ASR/TTS routing
| scenario | ASR | TTS |
| --- | --- | --- |
| EN only | Nemotron ASR Streaming | Magpie Multilingual |
| auto-detect multilingual | Parakeet RNNT Multilingual | Magpie Multilingual |
| single locked lang | Nemotron ASR Streaming Multilingual | Magpie (locale voice) |
| EN+diarization | Nemotron ASR Streaming | Magpie |
| tight VRAM | Parakeet RNNT+smallest LLM+Magpie | hardware-probe |
| Jetson | Riva :50052 | Riva :50051 Magpie |

## LLM routing (cascaded only)
Whitelist ONLY Nemotron3 Nano/Super/Ultra. Match data[].id contains nemotron-3-{nano|super|ultra}; exclude omni in nano.
Cloud: GET /v1/models+catalog filter. Local: curl :18000|:8002 → nim-llm-profiles.
Pick: usecase tier first (general→Nano | balanced→Super | specialized→Ultra) then VRAM downgrade | latency/tight VRAM→drop one tier
**Local GPU `<90 GiB`:** **Nano only** + **nvfp4 only** — hardware-probe Step2b overrides tier; FORBID Super/Ultra in rec table. Precision: smallest Compatible `vllm-nvfp4-*` from list-model-profiles.
Non-EN: whitelist only; verify build card. Reasoning→llm-reasoning.md. Outside whitelist→warn map closest WAIT.
Omni→pipeline/omni.md (not this whitelist).

## Omni pick
| deploy | omni | TTS |
| --- | --- | --- |
| cloud | GET /v1/models filter omni | nemotron-speech/NVCF |
| local | HF/vLLM :8002 | matrix/compose |

VRAM: Omni+TTS only; no nim-llm for omni sidecar. Rec: omni+TTS slots only FORBID ASR.

## Anti-patterns
`name models or I'll choose` | bundled model delegate MCQ | skip per-slot MCQ on you choose | rec without why/precision/profile/VRAM | **confirm ask without inline model table/ids** | promise-only rec with no exact ids | skip probe | skip list-model-profiles local | omni with ASR | cascaded LLM outside whitelist | **Super/Ultra rec on local GPU <90GB**
