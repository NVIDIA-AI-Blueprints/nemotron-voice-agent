# Gate | pre-scaffold mandatory

```
PROC: Step0 probe+secrets → Required rows → open row → STOP WAIT → FORBID scaffold
SEQ: 0 hardware probe + env-secrets 1 main gate rows **2b pipeline when ambiguous** 2 deployment (defer Step0b until 2b resolved) 3 usecase if tier unknown 4 model-slot-mcq LLM→ASR→TTS 5 disclosure 6 llm-reasoning 7 speech wizard 8 scaffold
Main gate open rows: **structured MCQ** when available — **except** deployment row when probe says no_gpu (auto cloud, no ask). Fallback: TPL if MCQ unavailable.
FORBID: gate questions before Step0 probe | deployment MCQ when no_gpu | collapse rec+reasoning+scaffold; "create"/"you pick models" waives nothing; silent transport/reasoning defaults at scaffold
```

## Step 0 — before main gate (mandatory on create)
1. **Hardware probe** — run `platforms/hardware-probe-and-profiles.md` §Step0 locally; classify `no_gpu|workstation|dgx_spark|jetson_thor`; store `DETECTED_HARDWARE`.
2. **Env secrets** — `clarifying/env-secrets.md`: probe `.env`/shell for `NVIDIA_API_KEY`; if missing → guide + STOP WAIT. Local **Omni** (`PIPELINE_MODE=omni` + local compose) → `HF_TOKEN` WAIT. Jetson local cascaded vLLM may also need `HF_TOKEN`. LiveKit → `LIVEKIT_*` WAIT (see frameworks/livekit.md).
3. **Tailor row 2 (Deployment)** per probe (see hardware-probe §Deployment question) — **after row 2b (pipeline) is resolved** when ambiguous; FORBID sizing deployment as cascaded before pipeline mode is known.

Delegate phrases: you choose|pick for me|use defaults|use already deployed

## Required rows
| # | Choice | missing |
| --- | --- | --- |
| 1 | Framework Pipecat\|LiveKit | generic request |
| 2b | Pipeline cascaded\|omni (Pipecat) | ambiguous — **ask before deployment rec**; FORBID default cascaded for Step0b sizing |
| 2 | Deployment local\|cloud (probe-driven; see hardware-probe §Step0) | unspecified AND probe found GPU; **defer until row 2b resolved when pipeline ambiguous** |
| 2f | Deployment rec: **Local (recommended)** when GPU fits full stack for selected pipeline (hardware-probe §Step0b); Cloud recommended only if stack won't fit | order/tag options accordingly |
| 2n | Deployment → auto **cloud** when probe `no_gpu` | inform user; do not ask |
| 3 | Transport WebRTC\|WebSocket\|Both (Pipecat) | Pipecat no transport |
| 4 | LLM — per-slot MCQ (`clarifying/model-slot-mcq.md` §LLM) | open |
| 4o | Omni model — per-slot MCQ (model-slot-mcq §Omni) | omni open |
| 5 | ASR — per-slot MCQ (model-slot-mcq §ASR) | cascaded open |
| 6 | TTS — per-slot MCQ (model-slot-mcq §TTS) | open |

omni: Pipecat only; LiveKit+omni→redirect; skip row 5; rows 4o+6; REF pipeline/omni.md
Separate: speech-customization (vertical) | perf benchmarks None/Later/Scaling | llm-reasoning (Nemotron3, blocks scaffold unless pre-answered)

## Skip main gate (ALL true)
Framework stated | Deployment stated | Transport **explicit** (not inferred) | Pipeline stated | **All** model slots named (cascaded LLM+ASR+TTS or omni+TTS)
Skip main ≠ skip post-gates (rec, reasoning, speech wizard)

Skip lines: named→"All required choices in message — skipping bundled clarification gate — implementing." | models open→run model-slot-mcq per open slot (not delegate bundle)

## Model slots (rows 4–6) — per-slot MCQ
When LLM, ASR, or TTS is open: load `clarifying/model-slot-mcq.md`. **FORBID** `name models or I'll choose` / `you pick vs you name`.

### One-shot delegated routing (canonical predicate)
Use this exact predicate everywhere (`usecase-before-rec.md`, `model-slot-mcq.md`, `prompts.md`):

```
delegate_phrase := you choose | pick for me | use defaults | use already deployed
impl_phrase     := implement now | go ahead | I confirm your top picks | just build | start now
one_shot_delegated := delegate_phrase AND impl_phrase AND open model slot(s)
```

| branch | behavior |
| --- | --- |
| `one_shot_delegated` AND user said **`I confirm your top picks`** | map tier silently → **skip** per-slot MCQs → disclosure table with `(recommended)` pre-locked → confirm → scaffold |
| `one_shot_delegated` AND NOT `I confirm your top picks` | map tier silently → **run** per-slot MCQs with `(recommended)` marked → disclosure → scaffold |
| delegate only (no `impl_phrase`) | usecase WAIT if tier unknown → per-slot MCQs |
| open slots, no delegate phrase | per-slot MCQs after framework/deployment/transport |

```
SEQ models open (non one-shot confirm): usecase (if tier unknown) → AskQuestion LLM → AskQuestion ASR → AskQuestion TTS → disclosure table → reasoning → scaffold
Skip MCQ only for slots user already named, or when `one_shot_delegated` AND `I confirm your top picks`.
Delegate without impl phrase → same per-slot MCQs with (recommended) marked after usecase.
```

## Model rec gate (after MCQ or one-shot pre-lock)
0 **usecase-before-rec.md** — when tier unknown (marks which LLM/ASR option gets `(recommended)` in MCQ); skip on one-shot pre-lock
1 vertical→speech Step0 unless pre-answered
2 read model-slot-mcq + model-selection + hardware-probe (cascaded) OR pipeline/omni (omni)
3 profile: probe→budget→nim-llm-profiles (local LLM)
4 one msg: **print full model table** (slots + **Reasoning** row) per model-selection.md §Rec display → confirm → STOP WAIT
FORBID bot.py/compose in MCQ or disclosure turn | FORBID "looks good?" without exact ids inline | FORBID bundled delegate model ask

## TPL
deployment (GPU fits stack): MCQ `Deployment?` options `Local — NVIDIA <gpu_name> (recommended)` | `Cloud` — see hardware-probe §Step0b
deployment (GPU too tight): MCQ `Deployment?` options `Cloud (recommended)` | `Local — NVIDIA <gpu_name> (tight: <slot>)`
transport: `Transport? WebRTC (:7860/client), WebSocket (UDP-blocked), or Both (recommended — scaffold webrtc+websocket, switch in Pipecat UI). Benchmarks None/Later/Scaling. No files until answered.`
cloud open: `Transport WebRTC|WebSocket|Both; then per-slot model MCQs (LLM, ASR, TTS). No files until answered.`
benchmarks: `Benchmarks: None/Later/Scaling.`
multi open: `Have: <resolved>. Need: Framework? Deployment? Transport? Models? Benchmarks?`

Partial: ask open rows only; FORBID recap resolved; FORBID bot.py for LiveKit

## Bundled ask examples
| user | action |
| --- | --- |
| pipecat cloud you choose | probe→transport→usecase→LLM MCQ→ASR MCQ→TTS MCQ→disclosure; no scaffold turn1 |
| pipecat cloud WebRTC you choose | usecase→LLM→ASR→TTS MCQs→disclosure |
| no GPU host + create | probe→inform cloud→framework/transport→model slot MCQs; skip deployment ask |
| pipecat cloud WebRTC you choose reasoning off | LLM→ASR→TTS MCQs→disclosure; reasoning pre-answered |
| LiveKit local you choose | usecase→LLM→ASR→TTS MCQs→disclosure |
| SSH workstation one-shot | skip main per ssh.md |

**Structured MCQ (REQ for open gate rows):** Use MCQ for each open field when the host agent supports it. **Bundled MCQ** allowed for transport + reasoning + benchmarks only — **never** bundle LLM+ASR+TTS into one form; always three separate model slot questions per model-slot-mcq.md. Options must match gate choices (e.g. WebRTC|WebSocket|Both (recommended), None|Later|Scaling, cascaded|omni). Do not only paste TPL and wait for typed chat if MCQ is available.

## Gate violations
scaffold without explicit transport | LLM_REASONING_* before gate | you choose→scaffold same turn | skip-gate with open transport/reasoning | post-confirm WebRTC default substitutes transport ask | collapse usecase+disclosure one turn | **gate AskQuestion before Step0 probe** | **deployment MCQ when probe=no_gpu** | **scaffold with missing NVIDIA_API_KEY / LIVEKIT_* / Jetson HF_TOKEN** | **`name models or I'll choose` bundled ask** | **single MCQ for all three model slots**

## After reply
Delegated→lock on confirm | Named→lock ids | Scaffold: one py+pyproject+.env.example | Transport Both→`transport_params` webrtc+websocket, pyproject both extras, run **without** `-t` → `networking/transport-selection.md` | WebRTC only→`-t webrtc`+turn if remote | WebSocket only→`-t websocket` | Local: nim.yml or nim.omni.yml | **Local post-confirm:** per-slot NIM probe (LLM/ASR/TTS) — keep matches, replace mismatches only → `platforms/deployment.md` §Local selective slot reuse | SSH per webrtc-turn+ssh
