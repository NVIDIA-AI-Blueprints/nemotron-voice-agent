---
name: create-voice-agent
description: Scaffold cascaded/omni Pipecat or cascaded LiveKit. Step0 probe+env secrets; per-slot model MCQs (LLM/ASR/TTS) with (recommended)+Other. Gate before scaffold. Omni Pipecat only.
version: "2.0.0"
metadata:
  tags: [pipecat, livekit, voice-agent, nvidia, pipeline, omni, nim, perf]
---

# create-voice-agent | ROUTER

ENC: dense tables; REF=canonical; FORBID/REQ/WAIT; load refs on trigger only; never read all refs.

## Deliverable
bot.py(Pipecat)|agent.py(LiveKit) + pyproject.toml + `.env.example` (placeholders only; user copies to `.env` locally before adding credentials) | local: docker-compose.nim.workstation.yml; WebRTC remote: turn_support.py+docker-compose.turn.yml | PIPELINE_MODE cascaded|omni | omni: **COPY** assets/omni_service.py+audio_only_smart_turn.py then bot.py per bot.omni-runner.md

## Triggers→ref
| when | ref |
| --- | --- |
| create/scaffold (first) | platforms/hardware-probe-and-profiles §Step0, clarifying/env-secrets |
| open rows/scaffold | clarifying/gate.md |
| open model slot (LLM/ASR/TTS) | clarifying/model-slot-mcq |
| you choose models | clarifying/usecase-before-rec → model-slot-mcq → hardware-probe-and-profiles → model-selection |
| vertical | domain/speech-customization |
| language/domain | domain/derive-domain |
| non-EN | models/language-routing |
| omni | pipeline/omni.md, assets/bot.omni-* |
| cascaded Pipecat | frameworks/pipecat.md |
| LiveKit | frameworks/livekit.md |
| cloud/local | platforms/deployment |
| DGX/Jetson/debug | dgx-spark, jetson-thor, common-voice-debug |
| WebRTC/TURN/SSH | first-try-workstation-webrtc, webrtc-turn, ssh, transport-selection |
| LLM catalog/profile/reasoning | models/catalog.md, models/nim-llm-profiles-and-deployment.md, models/llm-reasoning.md |
| speech NIM depth | nemotron-speech/references/ |
| run/iterate/troubleshoot/perf | operations/{run,iterate,troubleshoot,perf}.md |
| skeletons | assets/bot.{workstation,jetson,omni}-runner.md, bot.domain.md |

## Hard gate
gate.md before any file. **Step 0 (create path):** run hardware probe + env-secrets **before** any gate AskQuestion. Open rows: **structured MCQ** when available preferred over chat-only TPL. FORBID scaffold until: transport explicit(Pipecat); models confirmed if delegated; reasoning answered(Nemotron3) unless pre-answered; required `.env` secrets set.

```
Step0 probe+secrets → Main OK → usecase if tier unknown → model-slot-mcq (LLM→ASR→TTS per open slot) → disclosure WAIT → Nemotron3? reasoning WAIT → vertical speech? wizard WAIT → scaffold
```

## Decision tree
```
create→hardware-probe §Step0+env-secrets FIRST | lang/vertical→derive-domain | missing rows→gate | LiveKit+omni→redirect Pipecat omni | LiveKit+WebRTC→console redirect
Pipecat omni→pipeline/omni+bot.omni-* | Pipecat cascaded+WebRTC→first-try FIRST | workstation SSH→ssh+webrtc-turn
DGX/Jetson→platform pre-flight | local GPU→hardware-probe | models open→model-slot-mcq LLM→ASR→TTS then disclosure WAIT
symptom→troubleshoot | change→iterate | perf→perf.md
```

## Platform
| platform | DEPLOYMENT_PLATFORM | extra |
| --- | --- | --- |
| Cloud | cloud | — |
| Workstation | workstation | turn if WebRTC |
| DGX Spark | workstation + LOCAL_HARDWARE_PROFILE=dgx_spark | dgx-spark.md |
| Jetson | jetson + jetson_thor | jetson compose |

## Rules (REQ/FORBID)
1 gate open rows only 2 Pipecat→bot.py LiveKit→agent.py 3 local ids: cascaded curl :18000 then :8002; omni :8002 only; no hardcoded fallback 4 workstation cascaded ASR+LLM+TTS; omni vLLM:8002+TTS only 5 WebRTC→first-try 6 WebSocket grep bot 6b Both→transport_params webrtc+websocket, pyproject both extras, run omit `-t`, UI switch →transport-selection.md 7 local vLLM→bot.workstation-runner supports_developer_role=False 8 TTS 1.3+→first-try 9 DGX auto-detect dgx_spark 10 Jetson DEPLOYMENT_PLATFORM=jetson HF_TOKEN Riva 50052/50051 llm-reasoning 11 run: per-slot NIM probe post-confirm (keep matching LLM/ASR/TTS; replace mismatches only)→NIMs→coturn→agent 12 benchmarks gate None/Later/Scaling 13 domain derive→voice-and-llm-output→fake-data→bot.domain 14 **Step0: hardware probe before any gate Q; no_gpu→cloud no deployment ask; GPU fits full stack (Step0b)→Local (recommended), Cloud second; Cloud rec only if no fit; defer Step0b if PIPELINE_MODE unknown** 15 models open→**three separate slot MCQs** (LLM Nemotron3 family, ASR Nemotron Streaming|Parakeet RNNT, TTS Magpie) each with (recommended)+Other; FORBID name-or-delegate bundle; one-shot delegated routing per gate.md 16 local LLM profile→models/nim-llm-profiles-and-deployment list-model-profiles mandatory; **local GPU <90 GiB: Nano model family only, nvfp4 precision only** (hardware-probe Step2b overrides tier) 17 skill refs first; Pipecat→pipecat-docs MCP; LiveKit→livekit-docs MCP then fetch 18 ASR defaults: EN→Nemotron ASR Streaming; auto multilingual→Parakeet RNNT Multilingual; locked lang→Nemotron ASR Streaming Multilingual 19 disclose endpoints/selectors/GPU on rec 20 rec needs visible table+Why+Precision+Profile+VRAM 21 speech customization REQ full glossary shown inline + user confirms before any speech_glossary.json/bot.py customization wiring; pre-approval text does not count 22 iterate fetch docs first 23 omni→pipeline/omni.md 24 troubleshoot symptom→doc table 25 reasoning gate before LLM_REASONING_* 26 transport must ask not infer 27 FORBID collapse usecase+rec+reasoning+scaffold 28 pre-scaffold: secrets? transport? models? reasoning? speech glossary displayed+confirmed if customization? 29 usecase ask ONLY when models delegated 30 **env-secrets: NVIDIA_API_KEY WAIT if missing; HF_TOKEN WAIT on any local Omni deployment; LIVEKIT_* WAIT for LiveKit** 31 LiveKit default deployment: LiveKit Cloud online server; guide keys via MCP/docs 32 FORBID scaffold a secrets-bearing `.env`; ship `.env.example` only

## Checklist
```
0-probe Step0 hardware-probe+env-secrets BEFORE gate Q | 0 domain|pipeline|open rows→gate|models open→usecase (tier) then LLM→ASR→TTS MCQs→disclosure WAIT|DGX/Jetson/SSH pre-flight
0a-transport Pipecat no transport in msg→WAIT | 0a-reasoning Nemotron3→llm-reasoning WAIT | 0b deployment: no_gpu→cloud skip ask; GPU→Local|Cloud MCQ, **Local (recommended) first when stack fits (Step0b); Cloud rec only if no fit**
1 pipecat|omni|livekit ref (LiveKit: MCP livekit-docs FIRST) | 2 model-slot-mcq per open slot | 2b model-selection resolve ids | 2r reasoning | 2a speech wizard blocks scaffold until full glossary inline + confirmed | 2b domain 2c voice-and-llm-output fake-data 2d glossary
3 bot.py DEPLOYMENT_PLATFORM PIPELINE_MODE TRANSPORT_MODE | 3o COPY assets/omni_service.py+audio_only_smart_turn.py (+turn_support WebRTC) then bot.omni-runner | 3o-greet omni: TTSSpeakFrame+MuteUntilFirstBotComplete FORBID LLMRunFrame connect greeting | 3a WebRTC first-try; Both→transport-selection | 4 pyproject .env.example compose | 4b local post-confirm: probe LLM+ASR+TTS slots individually; keep matches; replace only wrong/missing (deployment.md §selective) | 5 TURN if WebRTC or Both+remote | 6 start: WebRTC `-t webrtc`; WebSocket `-t websocket`; Both omit `-t`
7 verify run.md+first-try | 8 handoff ssh Mac | 8a benchmarks scaling-perf | 9 TCO tco.placeholder | 10 iterate/troubleshoot
```
