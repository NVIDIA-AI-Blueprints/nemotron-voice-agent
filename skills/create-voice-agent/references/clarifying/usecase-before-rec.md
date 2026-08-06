# Use case gate | tier for (recommended) MCQ tags

Trigger: tier unknown before `clarifying/model-slot-mcq.md` — needed to mark which LLM option gets `(recommended)`. Also used when user explicitly delegates with use-case ambiguity.

Do **NOT** skip when models open and tier unknown — even without delegate phrase (omitted slots still need recommended tags).

Do **skip** (map silently) when:

- user already stated tier (general/balanced/specialized) or named exact LLM tier
- **one-shot delegated** per `clarifying/gate.md` §One-shot delegated routing (`delegate_phrase` + `impl_phrase`) — map tier silently (below), no WAIT
- user named exact model ids for all slots (MCQ skipped entirely)

```text
tier unknown → usecase ask STOP WAIT → model-slot-mcq (LLM/ASR/TTS) with (recommended) tags → disclosure → scaffold
one_shot_delegated + I confirm your top picks → map tier silently → skip MCQ → disclosure (recommended pre-locked) → scaffold on confirm
one_shot_delegated without I confirm your top picks → map tier silently → per-slot MCQs (recommended marked) → disclosure → scaffold
FORBID: usecase ask + MCQ same turn | skip usecase when tier unknown and MCQ would lack (recommended)
```

## When to skip (map silently, no question)

| signal in user message | map to |
| --- | --- |
| one-shot delegated (`gate.md` predicate) + `I confirm your top picks` | infer tier below (else **balanced**), skip MCQ, pre-lock disclosure |
| one-shot delegated without `I confirm your top picks` | infer tier below (else **balanced**), no WAIT — then per-slot MCQs with (recommended) |
| simple FAQ, greeting bot, basic assistant, lightweight | general |
| multi-turn, tools, domain agent, clinic, retail, library | balanced |
| complex reasoning, multi-step workflow, heavy tools, research, coding agent | specialized |
| reasoning on / hard tasks / deep analysis | specialized (+ reasoning ON in rec) |
| reasoning off / lowest latency | general or balanced (+ reasoning OFF in rec) |

If ambiguous after mapping rules AND not a one-shot request, ask — do not default silently. One-shot never blocks: map to balanced and proceed.

## Step 1 — one message STOP WAIT

Use **structured MCQ** when available. Fallback TPL:

```text
Before I recommend models (and reasoning settings), what kind of workload is this voice agent for?

- **General** — short interactions: greetings, FAQs, simple commands, light chat
- **Balanced** — everyday assistant: multi-turn dialogue, light tool use, moderate complexity
- **Specialized** — demanding work: complex reasoning, multi-step workflows, heavy tool use, domain expertise

Reply: general | balanced | specialized (or describe your use case in one line).
No files until you answer.
```

FORBID printing LLM/ASR/TTS ids, reasoning constants, or `bot.py` in this turn.

## Step 2 — after usecase reply

Apply tier to mark `(recommended)` on LLM MCQ option, then run `model-slot-mcq.md` (LLM → ASR → TTS). Read `models/model-selection.md` for id resolution and disclosure table.

### LLM (cascaded Nemotron3 whitelist)

| tier | pick | when to bump |
| --- | --- | --- |
| general | **Nano** | default for small/general workloads |
| balanced | **Super** | multi-turn, tools, domain agents, most verticals |
| specialized | **Ultra** | hard reasoning, long plans, many tools, user asked for max quality |

**Local GPU `<90 GiB` (hardware-probe Step2b):** cap **Nano only** — tier does **not** bump LLM family. Disclose: `GPU <90GB — Nemotron3 Nano NVFP4 (tier would be Super/Ultra on cloud)`.

Downgrade one tier if local VRAM cannot fit profile (see hardware-probe). Cloud: still disclose ids from catalog/discovery.

### Reasoning (Nemotron3 only — preview in rec table, lock at confirm)

| tier | default | budget if ON |
| --- | --- | --- |
| general | **OFF** | — |
| balanced | **OFF** (voice latency) | ON only if user said complex/hard OR vertical needs planning → **512–2048** |
| specialized | **ON** recommended | **8192** default; **16384** if heavy tools / long plans; **512** only if user asked ON but wants low latency |

State reasoning row in the rec table: `Reasoning: OFF` or `Reasoning: ON, budget N`. User can override in same confirm reply; else `llm-reasoning.md` gate is pre-answered when rec included reasoning.

### ASR / TTS (overlay language-routing)

| language context | ASR | TTS |
| --- | --- | --- |
| EN only | Nemotron ASR Streaming | Magpie Multilingual (en-US voice) |
| multilingual or unset locale (auto-detect) | Parakeet RNNT 1.1B **Multilingual** | Magpie **Multilingual** (pick voice via list_voices / locale map) |
| single locked non-EN (e.g. Hindi only) | Nemotron ASR Streaming **Multilingual** | Magpie Multilingual matching locale |
| diarization / multi-speaker | Nemotron ASR Streaming | Magpie Multilingual |

If domain/language already known (Hindi supermarket, Spanish library), apply multilingual/locked rules in rec **without** re-asking language.

Omni delegated: recommend omni model tier with same LLM logic (lighter→smaller omni id, specialized→larger); TTS per `pipeline/omni.md`; FORBID ASR row.

## Disclosure turn (after MCQ)

One message: tier rationale (1 line) + **full slot table** per model-selection §Rec display including **Reasoning** row + combined VRAM + `Reply looks good to lock, or name the slot to swap.`

## Anti-patterns

usecase skip when tier unknown before MCQ | model MCQ without (recommended) tag | bundled name-or-delegate ask | collapse usecase+MCQ+scaffold one turn
