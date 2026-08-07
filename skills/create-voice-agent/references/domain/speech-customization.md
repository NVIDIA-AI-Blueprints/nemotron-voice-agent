# Speech Customization

Use when the agent must recognize or pronounce domain terms such as product names,
people, acronyms, medicines, menu items, or internal vocabulary.

This is runtime customization, not model retraining:

- ASR word boosting biases recognition toward approved words and phrases.
- TTS pronunciation customization maps approved words to supported IPA pronunciations.

Language and locale routing is a separate model-selection concern. Resolve it through
`models/language-routing.md`. This file owns only domain vocabulary.

## Trigger

Offer speech customization when the request names a domain with specialized vocabulary,
or when the user asks for boosting, hotwords, a glossary, pronunciation, or phonemes.
Skip it for a generic assistant unless requested.

After the base model table is approved, ask once:

```text
This use case has domain vocabulary. Should I add ASR word boosting and TTS pronunciation
rules? If yes, send any required terms or I will draft a glossary for approval.
```

## Verify support first

Read the current official documentation before drafting or wiring:

- [ASR customization](https://docs.nvidia.com/nim/speech/latest/asr/customization/customization.html)
- [ASR support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/asr.html)
- [TTS customization](https://docs.nvidia.com/nim/speech/latest/tts/customization.html)
- [TTS support matrix](https://docs.nvidia.com/nim/speech/latest/reference/support-matrix/tts.html)

Confirm the locked model and API support the requested feature. Word-boosting modes,
limits, score ranges, and model support differ by decoder and release. TTS phoneme and
custom-dictionary support also differs by model and language.

If the locked model lacks a required feature, show the affected model row and propose one
supported streaming alternative. Never swap models silently.

### Platform source

| Platform | Customization source |
| --- | --- |
| Cloud / workstation / DGX Spark | Speech NIM docs and matrices above |
| Jetson Thor | current Riva Quick Start plus the Riva ASR / TTS docs below |

Jetson Thor runs Riva L4T, not Speech NIM. For the Riva model selected in
`platforms/jetson-thor.md`, verify against:

- [Riva ASR customization and word boosting](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-customizing.html#word-boosting)
- [Riva TTS custom models and pronunciations](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/public/tts/tts-custom.html)

Prefer request-time Riva customization:

- ASR: add the approved phrases and score to the streaming recognition request through
  `SpeechContext` or the current Riva client helper.
- TTS: send approved custom pronunciations or SSML through the synthesis request when the
  selected Riva model supports them.

Do not apply Speech NIM tags or assume every ARM64 Quick Start model supports the same
decoder or phoneme set. Global ASR lexicon changes and server-side TTS dictionaries require
the current Riva model-build flow and may regenerate the model repository. Treat that as a
deployment change, preserve the current repository, and get approval before rebuilding.

## Glossary approval

Draft one table and show every term before writing files:

| Term | Locale | ASR phrase | Boost score | TTS IPA |
| --- | --- | --- | --- | --- |
| user or domain term | approved locale | exact spoken form | value from current ASR docs | supported IPA |

Use the current docs for score ranges and IPA support. Start with the lowest recommended
boost that solves the ambiguity. Higher scores increase false positives.

Wait for explicit approval of the displayed terms, scores, and pronunciations. “Use your
suggestions” counts only after the complete table has been shown. Do not create a glossary
or wire customization before this confirmation.

## ASR

Keep streaming ASR. Choose the customization mode from the current ASR docs:

- Per-stream boosting for a request-specific glossary.
- Global boosting only when the approved list or deployment requires server-side
  configuration.

Wire the approved phrases and scores through the selected framework's current NVIDIA ASR
API. Query its documentation MCP for field names. Do not reuse score ranges across CTC,
RNNT, TDT, or Nemotron ASR without checking the current docs.

## TTS

Use the current TTS docs to choose request-time SSML phonemes or a custom pronunciation
dictionary. Use only phonemes supported by the locked model and language.

Wire the approved dictionary through the selected framework's current NVIDIA TTS API.
After startup, synthesize every customized term and let the user confirm pronunciation.

## Omni

Omni has no ASR slot, so ASR word boosting does not apply. Do not add ASR just for domain
terms. TTS pronunciation customization still applies.

Domain terms may also be added to the Omni system instruction for response consistency,
but that is not ASR boosting.

## Generated project

When customization is approved:

- Write the confirmed terms to `speech_glossary.json` with this schema:

```json
{
  "version": 1,
  "terms": [
    {
      "term": "approved display form",
      "locale": "en-US",
      "asr": {
        "phrase": "approved spoken form",
        "boost": 4.0
      },
      "tts": {
        "mode": "ipa",
        "grapheme": "approved display form",
        "pronunciation": "approved IPA"
      }
    }
  ]
}
```

- Omit `asr` for Omni or when boosting is not approved. Omit `tts` when pronunciation
  customization is not approved. Do not write placeholder or `null` entries.
- Keep model ids, scores, and pronunciation data out of `.env`.
- Load the glossary from agent code and map it to the framework APIs documented today.
- Include a short README note explaining how to edit and revalidate the glossary.

## Verify

Test each term in a natural sentence:

1. ASR returns the intended spelling without causing nearby false positives.
2. TTS pronounces the term correctly.
3. A normal sentence without glossary terms still works.
4. The full spoken exchange in `operations/run.md` still passes.

Tune one term or score at a time. Use `operations/iterate.md` for later glossary changes.

## Anti-patterns

- Writing or wiring a glossary before showing it and receiving approval.
- Assuming every ASR model supports word boosting.
- Applying ASR boosting to Omni.
- Inventing IPA unsupported by the selected TTS model or language.
- Raising boost scores without checking false positives.
