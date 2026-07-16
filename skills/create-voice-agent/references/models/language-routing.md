# Language routing | model slots overlay

Trigger: primary_language≠en|multilingual|delegated+locale matters. NOT domain→derive-domain. Overlay model-selection ASR/TTS/LLM BCP47 voices.

| concern | doc |
| vertical prompt/tools | derive-domain+model-selection |
| language/locale Hindi→RNNT+Magpie hi-IN | this file |
| rec flow | model-selection |
| ASR/TTS VRAM | model-selection §ASR/TTS nemotron-speech/model-selection |
| cloud LLM | catalog Nemotron3 only |
| boost/dictionary constraints | speech-customization |

Matrices backup: docs.nvidia.com/nim/speech/.../asr.html tts.html

Decision:
```
en only→Nemotron ASR Streaming+Magpie Multilingual
word boost→Parakeet RNNT EN or Multilingual (explicit user ask)
TTS dictionary→Magpie Multilingual
one lang locked no switch e.g. fr→Nemotron ASR Streaming Multilingual+Magpie locale
multilingual/auto-detect→Parakeet RNNT Multilingual+Magpie
diarization→Nemotron ASR Streaming
```
Translation not domain bot→nemotron-speech/nmt.md

Locale map: hi→hi-IN es→es-US fr→fr-FR de→de-DE ja→ja-JP zh→zh-CN en→en-US

PROC: infer primary_language/locale derive Step1 → decision flow cross-check model-selection + usecase-before-rec multilingual rows → cloud catalog whitelist+function_id TTS voice list_voices runtime → local/Jetson NIM_TAGS_SELECTOR compose jetson-thor reasoning from tier llm-reasoning → language rationale in rec

Constants:
```python
PRIMARY_LANGUAGE="hi"; LOCALE="hi-IN"; ASR_MODEL="..."; TTS_MODEL="magpie-tts-multilingual"
TTS_LANGUAGE_CODE=LOCALE; TTS_VOICE="<list_voices>"; ASR_FUNCTION_ID="<cloud>"; LLM_MODEL_ID="<discovery>"
```
Local function_id empty.

Anti: Parakeet CTC for Hindi when RNNT Multilingual default | hardcode function_id/voice | EN stack for non-EN | skip rec you choose | implement before confirm
