# Speech customization | vertical only

Optional derive-domain. Runtime NIM: word boosting,custom_dictionary no retrain. Omni→pipeline/omni.md §speech canonical.
FORBID generic assistant. Trigger: named vertical(medical,food,...). Second gate after main.

Hard gate FORBID auto: speech_glossary.json,boosted_lm_*,custom_dictionary without **visible glossary display + explicit user confirm**.
| user | means | NOT auth |
| Step0 Yes | start wizard | glossary,wiring,score |
| looks good | models locked | scaffold,customization |
| "use your suggested glossary" before inline list | request draft | scaffold,wiring |
| vertical named | domain path | default terms |

Min confirms before Step5: 1 runtime options Step1 WAIT 2 term draft Step3 WAIT 3 boost/IPA Step4 WAIT
FORBID Step0 Yes or model confirm→glossary same turn. FORBID treating pre-approval ("use your suggested glossary", "I approve your suggestions", "implement now") as glossary confirmation unless the **full glossary was already shown in a prior assistant message**.
REQ Step3 display: print full `boost_words[]` + `pronunciations{word:ipa}` inline in chat (markdown table or bullets) before WAIT. FORBID promise-only ("I'll provide a glossary", "here are suggested terms" with no list). Min 3 boost_words when vertical has jargon unless user supplied list.
REQ before scaffold/customization: last assistant response before user confirmation must include every term that will be written to `speech_glossary.json` (ASR boost term, TTS pronunciation/IPA, and boost score if known). No `bot.py`, `speech_glossary.json`, `boosted_lm_words`, or `custom_dictionary` before that confirmation.

Trigger: vertical→Step0 after main. Generic→skip. no speech customization→skip FORBID re-ask.

NIM=https://docs.nvidia.com/nim/speech/latest Fetch FORBID hardcode: asr/customization, tts/customization, tts/phoneme-support, asr/protos, tts/protos, support-matrix asr/tts, pipeline-configuration+asr-custom local deploy-time.

Delegated+vertical→Step0 before model-selection. Named→Step0 before scaffold Step2 may swap WAIT.
Step0 TPL: `Building domain_label. Speech customization (ASR boost,TTS pronunciation)? Yes wizard / No standard.`
Pre-answered speech customization yes=Step0 only Steps1-6 still required.

Wizard one step/turn WAIT announce N/6:
1 fetch ASR+TTS pages runtime options boost,hints,dictionary,SSML phoneme deploy-time local WAIT
2 re-fetch confirm badges unsupported Whisper+boost→Parakeet RNNT WAIT
3 draft boost_words+pronunciations from DomainSpec — **show full list inline** (ASR boost column + TTS IPA column) WAIT
4 score ranges boosted_lm_score CTC 20-40 RNNT 0.5-1.5 — show score with same term list WAIT
5 only after explicit confirm of displayed terms+score→speech_glossary.json+bot.domain wiring no new .env
6 handoff recap boost_words count + pronunciations count + list terms again options models URLs

Wiring:

```python
stt=NvidiaSTTService(...,settings=NvidiaSTTService.Settings(language=...,boosted_lm_words=...,boosted_lm_score=...))
tts=NvidiaTTSService(...,custom_dictionary=...,settings=NvidiaTTSService.Settings(language=...,voice=...))
```

pipecat-docs if field names differ. LiveKit same glossary NVIDIA plugin.

Anti: looks good+Yes no term draft | promise glossary no inline list | "use suggested glossary" without showing terms | medical auto boost | model confirm+full project+customization one turn

Step3 TPL (after draft): show table `| Term | ASR boost | TTS IPA |` with every derived term, then `Edit terms, or reply use your suggestions to lock this draft.`
