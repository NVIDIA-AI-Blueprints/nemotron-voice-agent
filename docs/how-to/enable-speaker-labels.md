# Enable Speaker Labels

Use the Generic Assistant when NVIDIA streaming automatic speech recognition
(ASR) can tag who is talking. The conversation panel then labels each speaker,
and the selected prompt decides how the assistant treats them.

## Prerequisites

- Select **Generic Assistant** from the example selector.
- Select an ASR catalog entry with `speaker_diarization_supported: true`, such
  as Nemotron ASR Streaming English (cloud or local NIM). The UI shows the
  control only for entries with this flag.
- In **Speaker Labels**, turn **Diarization** on before you connect. You cannot
  change this setting during a live session. The catalog entry sets the maximum
  speaker count with `speaker_diarization_max_speakers` (`8` for Nemotron ASR
  Streaming). The UI shows it but does not change it.
- For `*/server` deployments, the Compose ASR services run the public
  `nvcr.io/nim/nvidia/nemotron-asr-streaming:1.4.0` NIM, which tags up to eight
  speakers.
- For `generic-assistant/single-gpu`, download the diarization and VAD models
  first. Refer to [Deployment Steps](../01-getting-started.md#deployment-steps).

API clients select `generic-assistant` and send
`"asr_speaker_diarization": true` in the session configuration. A session that
names no ASR uses the default catalog entry. If the ASR does not support
diarization, the server ignores the request. `ENABLE_SPEAKER_DIARIZATION=true`
sets the default for clients that omit the field. The browser always sends its
own choice.

The prompt list shows `multi_speaker_assistant` only while diarization is on and
supported. Otherwise the example's default prompt is used.

## Choose Single-Speaker or Multi-Speaker Behavior

The prompt's `multi_speaker_support` flag selects the mode.

- Single speaker (`generic_assistant`, `generic_assistant_without_tools`, and
  `flowershop`): the first tagged speaker is latched, and later speakers are
  dropped before the large language model (LLM). The UI keeps the **You** label.
- Multi speaker (`multi_speaker_assistant`): every tagged speaker is kept and
  shown as **Speaker 1**, **Speaker 2**, and so on, numbered in the order people
  first speak. The LLM receives one `Speaker N:` prefix per speaker run. The
  assistant answers only when someone calls it by name, Nemotron or assistant.
  Otherwise its reply is `...`, which stays in the history and is not spoken.

While the ASR is still recognizing, the panel shows the text in an
**Identifying speaker** bubble until a tagged final assigns it to a speaker.

## Multi-Speaker Replies

With `multi_speaker_assistant`, the LLM answers each spoken turn with one JSON
object:

```json
{"addressed_to_assistant": true, "reply": "Paris is the capital of France."}
```

The server requests this format with a strict `json_schema` `response_format`,
so the LLM endpoint must support that field. Only the `reply` text reaches the
conversation and text-to-speech. A `reply` of `...`, or
`addressed_to_assistant` set to `false`, stays silent. Chat-history summaries
use plain text.

## Turn Timing

All sessions let the ASR finish each utterance after 320 ms of trailing
silence. This ASR endpointing is independent of the Silero VAD and Smart Turn
signals that determine when the semantic user turn is complete.

## Limitations

- Labels are session-scoped indexes, not names or verified identities.
- In first-speaker mode, the first person to speak claims the latch.
- Overlapping speech can reduce attribution accuracy.
- Labeled text appears on each tagged ASR final. In-progress text stays in
  **Identifying speaker** until a final arrives.
- The multi-speaker prompt decides on its own whether a line called it. Long
  background speech, such as a TV or video, can still get a spoken reply. If
  the ASR misses the name under loud background audio, the call goes unanswered.
  After the assistant answers someone, it can also answer the next question that
  does not use its name.
- Smaller LLMs, such as Nemotron 3.5 Lightning 30B A3B, the default LLM for the
  Generic Assistant, do not always follow the multi-speaker prompt strictly.
  They can answer a question that does not call the assistant by name. Speaker
  labels still work. A larger model, such as Nemotron 3.5 Super VL, can follow
  the prompt more closely. To use it, refer to
  [Run Nemotron 3.5 Super VL](configure-llm.md#run-nemotron-35-super-vl).
