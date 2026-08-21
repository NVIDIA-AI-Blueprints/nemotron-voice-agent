# Intake

Propose and confirm. Two pauses for base intake: after the ask, and after the table.
Post-approval profile discovery adds a reconfirmation only when an approved row changes.
Speech customization is optional. When the user accepts its offer, it adds one glossary
approval after the base table.

```
0  probe    hardware + host class + base credential → preflight.md sections 1-3
1  ask      use case + pipeline + framework
2  propose  models + deployment fit + one filled table → preflight.md §Deployment fit
            + models/catalog.md
3  confirm  build, or amend a row
```

Do not ask anything this file does not define. Do not ask what the table can decide.

## 0. Probe

Before speaking. Missing credentials: say which key, how to get it, wait. Everything else
becomes a table row.

## 1. Ask priority questions

Resolve **Pipeline** and **Framework** explicitly. Skip a choice only when the user
already made it or compatibility forces it. Use structured choices when available and
ask all applicable questions in one message.

If the use case is unclear, also ask:

> What are you building? Include who talks to it, what it helps with, and the spoken
> language. Example: "a Hindi drive-through order taker."

Ask: **Which pipeline should the agent use?**

| Choice | Explain |
| --- | --- |
| Cascaded (Recommended) | streaming ASR → text LLM → TTS. Modular, easier to customize and troubleshoot |
| Omni | audio-in model replaces ASR and the text LLM. TTS remains. Pipecat only |
| Recommend for me | choose from the use case and explain the recommendation |

Then ask: **Which framework should implement it?** Ask this when Cascaded is selected or
the pipeline is undecided:

| Choice | Explain |
| --- | --- |
| Pipecat (Recommended) | supports Cascaded and Omni, with direct transport control |
| LiveKit | choose for LiveKit rooms, workers, or an existing LiveKit deployment. Cascaded only |
| Recommend for me | choose from the use case and explain the recommendation |

Omni locks Pipecat. LiveKit locks Cascaded. State the forced choice instead of asking an
irrelevant question. If the user explicitly selects LiveKit + Omni, ask one conflict
question: keep LiveKit with Cascaded, or keep Omni with Pipecat.

As soon as the framework is resolved, read its framework file and verify that its docs MCP
is available. If it is unavailable, give the setup steps from that file and wait. Complete
this check before proposing models or offering speech customization.

Infer language, vertical, reasoning hardness, and latency-vs-quality from the use case.
Derive the system instruction through `domain/agent-behavior.md`. Do not run a separate
persona wizard. One follow-up for missing use-case context is allowed. Never two.

## 2. Propose

One message. One table. Every row filled with a decision and a short reason. Exact model
ids from `models/llm.md`, `models/asr.md`, `models/tts.md` (via `models/catalog.md`), not
family names. These are catalog ids that record the choice, and two values stay open until
the services answer: the LLM's served model id and the exact TTS voice (`models/llm.md`
§Resolve three names). Lock the TTS locale now and present neither of those as final here.
For shared self-hosting, show the candidate LLM profile and runtime cap, speech profile
reserves, startup reserve, and usable VRAM from `preflight.md` §Deployment fit. Mark
provisional co-location until post-approval profile discovery succeeds and
`platforms/deployment.md` §Shared-GPU memory gate passes.
No code, compose, or layout in this turn.

The example below assumes W4A16 is the lightest candidate supported for this H100 in the
matrix (Lightning NVFP4 needs Blackwell). Post-approval profile discovery must confirm it.

```
Framework    Pipecat                 supports both pipeline shapes
Pipeline     cascaded                spoken Q&A
Deployment   self-hosted             H100 80GB, provisional co-location
Transport    both                    WebRTC + WebSocket
Language     English (en-US)         fixed, lowest latency
LLM          Nemotron 3.5 Lightning  nvidia/nemotron-3.5-lightning-30b-a3b, candidate vllm-w4a16-tp1-pp1
ASR          Nemotron ASR Streaming  nvidia/nemotron-asr-streaming, streaming, batch_size=32
TTS          Magpie TTS Multilingual nvidia/magpie-tts-multilingual, batch_size=8, en-US. Voice after startup
Reasoning    off                     low turn latency
Memory       runtime budget planned  context, LLM cap, speech reserves, startup reserve

Reply "go" to build, or name a row to change.
```

### Shape changes

- Omni: drop ASR, add the Omni model, and recalculate deployment fit
  (`frameworks/omni.md`).
- LiveKit: drop transport. LiveKit cannot run Omni. If both are requested, ask which to
  keep.
- DGX Spark: follow `platforms/dgx-spark.md`. Budget available unified memory and disclose
  any hybrid slot.
- Jetson Thor: use the vLLM + Riva rows from `platforms/jetson-thor.md`. Disclose the
  platform substitutions.
- Remote Pipecat WebRTC: route `networking/remote-webrtc.md`.
- Several languages: route fixed vs automatic detection through
  `models/language-routing.md`.

### Defaults

| Row | Default | Override when |
| --- | --- | --- |
| Framework | explicit intake choice, otherwise Pipecat recommendation | user chooses LiveKit |
| Pipeline | explicit intake choice, otherwise Cascaded recommendation | user chooses Omni or audio-native |
| Deployment | self-hosted when routed platform fit passes. Provisional when slots share one GPU | otherwise hybrid/cloud, name each cloud slot |
| Transport | both | user named one, or LiveKit (omit this row) |
| Language | fixed locale from `models/language-routing.md` | user requested several languages |
| LLM | Nemotron 3.5 Lightning | user named Super, Ultra, or another id |
| ASR | from language via `models/asr.md` | user named a model |
| TTS | Magpie TTS Multilingual, smallest supported batch profile, locale. Voice after runtime discovery | user named a model or voice |
| Reasoning | off | specialized workload → on, budget 8192 |
| Memory | profile reserves + runtime and startup headroom | recalculate whenever profile, batch, context, or placement changes |

Lightning always. Super/Ultra only if named. On one workstation GPU, say they will not fit.
Use `models/llm.md` for reasoning wiring and `models/asr.md` for streaming-first selection
and offline limitations.

## 3. Confirm

After base approval, offer `domain/speech-customization.md` when its Trigger section
applies. If accepted, wait for glossary approval before building. If declined, build
without it and do not ask again.

Amend a row in place. Re-show the table only when the change cascades. LiveKit drops Omni
and transport. Omni replaces ASR and changes fit. Full cloud removes the Memory row.
Hybrid recalculates Memory for local slots. Otherwise just build.

After self-hosting approval, run `platforms/readiness.md`, then discover and pin the exact
Compatible LLM profile. If profile, memory fit, or placement changes, re-show the affected
rows and wait for confirmation before writing files. During deployment, follow
`platforms/deployment.md` §Per-slot reuse before starting services.

## Stop only for

Missing credentials. Still too vague after one follow-up. Conflict (Omni + LiveKit).
Self-hosted on a host with no compatible local platform.

## Anti-patterns

Skip the Pipeline or Framework choice. Ask transport or deployment before proposing.
Ask model slots separately. Offer choices without explaining trade-offs. Family names in
the table. Write files in the proposal turn. Re-ask something already stated.
