# Model slot MCQ | per-slot pick (LLM, ASR, TTS)

Trigger: any open model row (4 LLM | 5 ASR | 6 TTS) on cascaded pipeline. REQ when host supports AskQuestion.

```text
FORBID: one question "name models OR I'll choose" | "you pick vs you name" bundled delegate | single MCQ for all three slots
REQ: separate AskQuestion per open slot — LLM, then ASR, then TTS (skip slots user already named)
REQ: every slot lists skill-supported options + Other | mark exactly one option (recommended) per slot
After all slots locked → resolve ids (model-selection §Rec display) → reasoning gate if Nemotron3 LLM → scaffold
```

Omni (Pipecat): skip ASR row. Ask **Omni model** MCQ then **TTS** MCQ (see §Omni below).

## When to run

| situation | action |
| --- | --- |
| user named all open slots | skip MCQ; resolve ids + disclose |
| user named some slots | MCQ **only open** slots |
| user said you choose / pick for me / use defaults | **still** run per-slot MCQs with (recommended) marked — unless `one_shot_delegated` AND `I confirm your top picks` (see gate.md predicate) |
| slots merely omitted (no delegate phrase) | run per-slot MCQs after framework/deployment/transport |
| one-shot delegated + `I confirm your top picks` | map usecase silently → pre-select (recommended) in disclosure table → **skip MCQ** |
| one-shot delegated without `I confirm your top picks` | map usecase silently → per-slot MCQs with (recommended) marked → disclosure |

## Pick (recommended) before MCQ

Load `usecase-before-rec.md` (tier) + `language-routing.md` (ASR) to mark `(recommended)` — run usecase WAIT first when tier unknown and not `one_shot_delegated` (gate.md predicate).

| slot | (recommended) rule |
| --- | --- |
| LLM | general→Nano, balanced→Super, specialized→Ultra; default balanced if unknown; **local GPU <90 GiB → Nano only** (hardware-probe Step2b) |
| ASR | EN only→Nemotron ASR Streaming; auto multilingual→Parakeet RNNT Multilingual; locked non-EN→Nemotron ASR Streaming Multilingual |
| TTS | Magpie Multilingual always (pick locale voice after lock via list_voices / language-routing) |

Jetson local ASR/TTS: options may show Riva sidecar labels; still offer skill families where applicable.

## AskQuestion — LLM (cascaded)

Prompt: `Which LLM? (Nemotron 3 family)`

| option id | label |
| --- | --- |
| nano | Nemotron 3 Nano — add `(recommended)` when tier=general **or local GPU <90 GiB** |
| super | Nemotron 3 Super — add `(recommended)` when tier=balanced **and GPU ≥90 GiB** |
| ultra | Nemotron 3 Ultra — add `(recommended)` when tier=specialized **and GPU ≥90 GiB** |
| other | Other — I'll type a different model id |

Only **one** option carries `(recommended)` — the tier default, **unless local <90 GiB → always nano**. Others omit the tag.

Local `<90 GiB`: prepend note `GPU <90GB — Super/Ultra won't fit; Nano NVFP4 recommended.`

Map on reply: nano→`nvidia/nemotron-3-nano-30b-a3b` (discover exact id via GET /v1/models); super→nemotron-3-super; ultra→nemotron-3-ultra. **Other** → user free-text → validate whitelist catalog; outside whitelist→warn WAIT.

## AskQuestion — ASR

Prompt: `Which ASR / STT model?`

| option id | label |
| --- | --- |
| nemotron_streaming | Nemotron ASR Streaming — `(recommended)` when EN only / diarization |
| nemotron_streaming_multi | Nemotron ASR Streaming Multilingual — `(recommended)` when single locked non-EN |
| parakeet_rnnt | Parakeet RNNT 1.1B |
| parakeet_rnnt_multi | Parakeet RNNT Multilingual — `(recommended)` when auto-detect multilingual |
| other | Other — I'll type a different ASR model |

Mark `(recommended)` per language-routing table above.

## AskQuestion — TTS

Prompt: `Which TTS model?`

| option id | label |
| --- | --- |
| magpie_multilingual | Magpie Multilingual (recommended) |
| other | Other — I'll type model + voice id |

If Magpie locked and locale known, note default voice in post-MCQ disclosure (e.g. `Magpie-Multilingual.EN-US.Aria`).

## AskQuestion — Omni model (Pipecat omni only)

Prompt: `Which Nemotron Omni model?`

Options from `pipeline/omni.md` + cloud/local GET /v1/models discovery + **Other**. Mark tier-appropriate option `(recommended)`.

## Other option

When user picks **Other**, STOP WAIT for free-text in chat: model id (and voice id for TTS). Resolve via catalog/NVCF discovery; warn if outside skill families.

## After all slots answered

1. `hardware-probe-and-profiles` if local — VRAM fit; downgrade if needed (tell user).
2. One message: **full disclosure table** (model-selection §Rec display) with exact ids, voices, endpoints, function_ids, Reasoning row preview.
3. Nemotron3 LLM → `llm-reasoning.md` if not pre-answered.
4. `Reply looks good to lock, or name the slot to swap.` → scaffold on confirm.

FORBID scaffold in same turn as MCQ unless `one_shot_delegated` AND `I confirm your top picks`.

## Anti-patterns

`Models: name them or you choose?` | delegate phrase→skip MCQ→silent rec | one bundled models MCQ | missing Other | no (recommended) tag | all three recommended | MCQ without usecase/language context when tier/locale affects ASR recommended
