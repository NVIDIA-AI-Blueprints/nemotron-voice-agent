# Language Routing

Read for every build before locking input speech, the LLM, and TTS. This file maps the
user's spoken language to supported model locales. Domain vocabulary remains in
`domain/speech-customization.md`.

## Lock the Route

Record:

- input language and exact BCP-47 locale
- fixed language or automatic detection
- cascaded ASR or Omni model that supports that mode and language
- response language, checked against the LLM's own supported languages
- TTS model and locale. Exact voice remains pending runtime discovery

Resolve locale codes from the selected platform's current source:

| Platform | Language source |
| --- | --- |
| Cloud / workstation / DGX Spark | Speech NIM ASR and TTS matrices + model pages |
| Jetson Thor | current Riva ARM64 Quick Start `config.sh` and model documentation |

For Omni input, use the locked Omni model card and service documentation instead of the
ASR matrix.

For example, Hindi may resolve to `hi-IN` only when both selected speech paths list that
exact locale. Do not invent a regional code from the language name.

## Fixed or Automatic

Use a fixed locale when the user names one language. It is the predictable, low-latency
default.

A fixed locale is a request parameter, not a capability. On a multilingual auto-detecting
model it is advisory, because the model still detects, so audio in another supported
language comes back in that language. Only a single-language model turns a locale into a
constraint. Check which kind the locked row is before promising fixed-language behaviour,
and say so in the proposal when the honest answer is auto-detection.

Use automatic detection only when the user requests several languages and the selected
**streaming** ASR model explicitly supports detection for all of them. Disclose that
detection and code-switching can add latency or errors. Never assume multilingual means
arbitrary code-switching.

For Omni, apply fixed or automatic behavior only when its model documentation supports
the requested audio languages.

If the user wants input in one language and replies in another, lock both sides
separately and reflect the response language in `domain/agent-behavior.md`.

## ASR

Follow `models/asr.md` and require a streaming row that supports the locked locale or
language set. The model card, matrix language table, and profile tags must agree.
Jetson Thor instead uses the streaming Riva model enabled by `platforms/jetson-thor.md`.

Omni bypasses ASR. Verify its supported audio languages and detection behavior from the
locked Omni source. Do not apply an ASR locale or assume multilingual detection.

Do not pass a locale merely because the framework exposes an enum for it. Query the
framework documentation MCP for the current language argument and pass the exact form the
selected NVIDIA service expects.

## LLM

The cascaded LLM carries its own supported-language list, separate from ASR and TTS coverage.
A locale that ASR transcribes and TTS speaks is not automatically a language the LLM answers
well, and this list is often shorter than people expect.

Read the supported languages from the locked model's card or API reference and compare them
with the approved response language. When the response language is outside that list, say so
before building and offer the choice plainly: keep it and accept reduced quality, change the
response language, or select a different LLM. Never let ASR and TTS coverage stand in for LLM
coverage.

For Omni, apply the same check to the response text on its locked source, in addition to the
audio-language check above.

## TTS

Follow `models/tts.md` and require a TTS model that supports the response locale. After
the service starts, query its documented voice-discovery API and choose only a returned
voice compatible with that locale.

On Jetson Thor, use only voices exposed by the selected Riva L4T service.

If no returned voice matches, reopen the TTS row and ask for approval before changing the
model, locale, or response language.

For multilingual conversations, choose the matching approved voice per response locale.
Do not send text in one language to a voice exposed only for another.

## Generated Project

Keep supported locales, routing rules, and voice ids in code or checked-in configuration,
not `.env`. The README must list the supported input and output languages and whether
automatic detection is enabled.

## Verify

For every approved language:

1. speak a short phrase and confirm cascaded ASR transcribes it, or Omni accepts it
2. confirm the agent replies in the approved response language
3. confirm TTS uses a discovered voice for that locale
4. test one unsupported language and verify a clear fallback

For automatic detection, also test consecutive turns in different languages and one
code-switched turn only when the selected model claims that support.

For language failures, use the Language section in `operations/troubleshoot.md`.
