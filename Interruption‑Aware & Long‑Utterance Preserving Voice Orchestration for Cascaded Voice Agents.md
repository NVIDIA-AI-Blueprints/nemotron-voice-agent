# Interruption‑Aware & Long‑Utterance Preserving Voice Orchestration for Cascaded Voice Agents

---

## 1. Problem Overview

Modern cascaded voice agents still underperform their text‑only counterparts on reasoning‑heavy tasks, even when they use the same underlying text LLM. This gap is especially visible on long, pause‑filled user questions where the agent is expected to listen first and reason over the full query.

In typical streaming setups, automatic speech recognition (ASR) emits partial transcripts and a simple end‑of‑utterance heuristic (for example, an 800 ms pause) triggers the LLM, which then generates a full response. The text‑to‑speech (TTS) system starts speaking while the user is still formulating the rest of the question. When the user resumes speaking, the agent interrupts playback, but the dialog state has already advanced on the basis of a partial query and a partial response.

This interaction pattern fragments a single long user question into multiple user–assistant turns because the agent incorrectly treats short pauses as the end of the utterance and interrupts the user mid‑question, a failure mode that is very common in real deployments. The LLM subsequently observes a history of the form “user fragment → assistant reply → user fragment → assistant reply…”, which is not equivalent to receiving one clean prompt containing the complete question:

```text
user: fragment A
assistant: answer to A   ← often full sentences; may be cut mid-reply with no interrupt signal
user: fragment B
assistant: answer to B
```

Studies on interruption handling in conversational robots show that naïvely treating every overlap or pause as a full turn leads to brittle behavior and user frustration, confirming that correct turn‑taking and interruption management is still an open challenge even for LLM‑powered agents.

As a result, speech‑based pipelines show a notable drop in reasoning quality compared to text‑only LLM usage, including on emerging speech reasoning benchmarks such as Big Bench Audio and related audio‑language evaluations, despite using essentially the same language models.

Results from our internal benchmarking: [Artificial Analysis – Nemotron Voice agent | Results.pptx](https://docs.google.com/presentation/d/19jizLySyiyJzQ-3hFoa8hLJOVaKy7dRy/edit?usp=sharing&ouid=104783989951665588727&rtpof=true&sd=true)

---

## 2. Proposed Solution

This proposal introduces an orchestration layer for cascaded voice agents with two core capabilities:

1. **Interruption‑Aware Dialog State Management** — dialog state reflects what the user actually heard, with explicit interruption signals when playback is cut short.  
2. **Long‑Utterance Understanding via Interruption‑Aware Reconstruction** — the LLM, guided by those signals and an updated prompt, reconstructs full user intent from fragmented turns instead of relying on perfect end‑of‑utterance detection.

The design is model‑agnostic and can sit in front of any text LLM, making it suitable as a drop‑in improvement for existing production voice agents.

Delivering both capabilities on our stack requires three POC workstreams (detailed in §8):

| Workstream | Role |
|------------|------|
| **WordTTSService path for `NvidiaTTSService`** | Word‑accurate “what was heard” commits (Capability 1 foundation). |
| **Orchestration changes** | Truncation + interrupt metadata on barge‑in (`on_assistant_turn_stopped`). |
| **Prompt tuning** | Offline harness to teach Nemotron how to interpret barge‑in history (Capability 2). |

---

## 3. Capability 1: Interruption‑Aware Dialog State Management

The first capability ensures that the dialog state always reflects what the user actually heard and intended, not what the LLM or TTS *planned* to say. This is crucial when users frequently interrupt playback to continue or refine their question.

Key design elements:

* **Speculative vs. committed assistant output**  
  The orchestration layer maintains a clear separation between speculative assistant responses (full LLM outputs) and committed dialog state (only the portion of the response that has actually been rendered to the user). This follows the same principle as recent “auto truncation” features, which update the session context to include only the played portion of an interrupted response.

* **Accurate truncation on interruption**  
  When the user starts speaking while TTS is playing, the system immediately stops playback, freezes the assistant turn at the last played boundary, and discards the unplayed tail rather than logging it as if it were delivered. This prevents the LLM from assuming that it has communicated information that the user never heard. On Magpie today that boundary is sentence‑coarse; the POC moves `NvidiaTTSService` to a **WordTTSService‑style** path so the boundary can be word‑accurate (see §5 and §8).

* **Explicit interruption flags and metadata**  
  In addition to truncating the assistant turn, the system explicitly marks that turn as interrupted and passes a short, machine‑readable hint into the next LLM invocation. The working form for experiments is appending `<interrupted>` to the committed assistant text and/or injecting a short developer note, for example:

  ```text
  assistant: The capital of France is <interrupted>
  ```

  or a developer message such as: *The assistant’s last reply was interrupted by the user; do not assume the previous thought was completed—rely on the full user question and prior turns.*

  We will A/B truncation‑only vs `<interrupted>` + developer note before freezing the encoding (§7).

This approach closely mirrors guidance from production systems like Azure Voice Live, which recommend both auto‑truncation and explicit appended text after truncation to keep the LLM’s view of history aligned with the actual audio conversation.

Interruption‑aware state management ensures that partial, unheard responses do not pollute conversation history and that the LLM is made aware of interruptions, enabling it to revisit previous user turns and re‑ground its reasoning when needed.

---

## 4. Capability 2: Long‑Utterance Understanding via Interruption‑Aware Reconstruction

The second capability is to recover long, pause‑filled user questions despite imperfect end‑of‑utterance detection and frequent interruptions, by leveraging the text LLM’s ability to interpret conversational dynamics from flagged, partial responses. Instead of relying on perfect real‑time turn segmentation, the orchestration layer makes interruptions explicit and asks the LLM to reconstruct the user’s full intent from the prior user turns.

Given that the system must append partially heard assistant responses into chat history and does not have a reliable behavioral end‑of‑utterance detector, the design focuses on the following behaviors:

* **Input‑first audio behavior, without “perfect” turns**  
  The agent still keeps listening while speaking and prioritizes user speech over ongoing TTS playback: user audio immediately pauses or stops output at the audio layer. However, the dialog state is allowed to advance with partial assistant turns, because this is how the current pipeline operates in practice.

* **Flagged interrupted turns as conversation signals**  
  Whenever a user starts speaking during TTS playback, the system truncates the assistant turn to the actually played portion and marks that message as interrupted in the conversation history (pending the §7 A/B). A short, standardized note may also be attached for the next LLM call.

* **LLM‑driven reconstruction of long questions**  
  On the next LLM invocation, the model sees:

  * The truncated, flagged assistant message (indicating it was cut off).  
  * The new user input that arrived during the interruption.  
  * The previous user transcripts that contain earlier parts of the same question.

  The prompt instructs the LLM to interpret flagged assistant messages as incomplete and to use the entire sequence of prior user turns, not only the last one, to infer the user’s full, long question before answering. This shifts the “turn stitching” problem into the language model, which is well‑suited to reason over multi‑turn context.

Importantly, the orchestration layer does **not** attempt to infer true end‑of‑utterance boundaries or merge user messages. Instead, it focuses on keeping history faithful to what actually happened (including which assistant replies were interrupted) and relies on the LLM’s conversational competence to re‑assemble long questions from fragmented but accurately labeled interaction history.

```text
User (pause-filled) → early ASR endpoint → LLM answers fragment
  → TTS speaks → user barge-in
  → commit heard text [+ <interrupted>] → next user fragment
  → LLM reconstructs full question → one complete spoken answer
```

---

## 5. Baseline: How Context Is Managed Today

This section grounds Capabilities 1–2 in our Pipecat stack (`NvidiaSTTService` → `NvidiaLLMService` → `NvidiaTTSService`). Assistant history is built from `TTSTextFrame`s after `transport.output()` — what was rendered for playback, not the raw LLM stream.

### 5.1 Magpie / `NvidiaTTSService` (sentence‑level)

`NvidiaTTSService` subclasses Pipecat `TTSService` with `push_text_frames=True` and **no word timestamps**. The TTS aggregator feeds **one sentence per `run_tts`**. Each finished sentence yields one `TTSTextFrame`. Context grain = **sentence only**.

`synthesis_mode` changes Magpie streaming, not that grain:

| Mode | Behavior | Effect on context timing |
|------|----------|---------------------------|
| `per_sentence` | New `SynthesizeOnline` per sentence; `run_tts` waits for that sentence’s audio. | Sentence *N* text is pushed after synthesis finishes; *N+1* starts later → successive sentences enter the context path more slowly. |
| `stitched` | Same sentences, one shared stream; `run_tts` queues text and returns early. | Successive sentence `TTSTextFrame`s can be enqueued faster; still **one frame per sentence**, never a multi‑sentence blob. |

Mid‑sentence barge‑in cannot record “user heard half of sentence 2.”

### 5.2 ElevenLabs WordTTS path (word‑level)

`ElevenLabsTTSService` uses the word‑timestamp path (`push_text_frames=False`, `add_word_timestamps` from alignment). Transport releases per‑word `TTSTextFrame`s in playback order. On barge‑in, context keeps **only spoken words**.

ElevenLabs does **not** append `<interrupted>` (or similar) to LLM messages. Truncation fidelity ≠ barge‑in semantics.

### 5.3 Gap summary

| | Magpie (`TTSService`) | ElevenLabs (word‑timestamp / WordTTS‑style) |
|--|----------------------|-----------------------------------------------|
| Commit grain | Sentence | Word |
| Heard‑text fidelity on barge‑in | Coarse | Fine |
| `<interrupted>` in LLM context | No | No |
| App signal | `on_assistant_turn_stopped(interrupted=True)` only | Same |

**`on_assistant_turn_stopped`:** Pipecat callback when an assistant turn ends. `message.content` is committed text; `message.interrupted` is true on barge‑in. Used today for app side effects (e.g. summarization in `generic/pipeline.py`, omni cleanup). It is **not** injected into Nemotron’s messages unless orchestration mutates context on this hook.

**Implication:** Capability 1 needs word‑accurate commits → move Magpie to a **WordTTSService‑style** path. Capability 2 needs an LLM‑visible interrupt signal + prompt → **orchestration + prompt tuning**. Word‑level TTS alone is necessary but not sufficient.

---

## 6. Prompt Contract (draft from `generic_assistant_without_tools`)

Baseline `generic_assistant_without_tools` has no barge‑in semantics. Working prompt for experiments (refined via §7; not final):

```yaml
generic_assistant_without_tools:
  description: "Generic voice assistant with interruption-aware history interpretation."
  content: |
    You are Nemotron, a helpful voice assistant developed by Nvidia.
    You run inside a cascaded voice system: the user's speech is transcribed to text for you, and your text is spoken back to the user. Do not mention this pipeline, transcription, or text-to-speech to the user. Act as a natural voice assistant.
    Always answer as helpful, friendly, and polite.
    You are knowledgeable about the world and can answer questions about a wide range of topics.
    Respond with one sentence or less than 75 characters.
    Do not use asterisks, markdown, emojis, or special characters.
    Do not respond with bulleted or numbered list.

    Interruption handling:
    - If a prior assistant message in the conversation ends with the token <interrupted>, the user barged in while that reply was being spoken.
    - The text before <interrupted> is only what the user actually heard. Any continuation you may have planned was not delivered.
    - Do not apologize for the interruption, mention the token, or repeat the interrupted fragment unless the user asks you to.
    - Treat <interrupted> as a cue that the latest user message may continue or complete an earlier user turn rather than start an unrelated topic.
    - When that is the case, reconstruct the user's full intent from all relevant prior user turns plus the latest one, then answer that full question in one short spoken reply.
    - If the latest user message is clearly a new topic or a correction, follow the latest user message and do not force a merge.
    - Always produce a complete, well-formed reply suitable for speech. Never emit <interrupted> yourself.
    - Never imitate prior partial or cut-off assistant messages. Truncated history and <interrupted> are metadata about what the user heard, not a style to continue. Do not generate half-finished answers just because earlier assistant turns look incomplete.
```

Rationale: voice framing without leaking pipeline jargon; barge‑in as continuation cue; anti‑pattern rules so next‑token behavior does not copy truncated/`<interrupted>` history into new half‑answers.

---

## 7. Prompt‑Tuning Experiments (before freezing design)

Before freezing orchestration details, we will run **LLM prompt experiments** with measurable outcomes, using an **offline prompt‑eval harness** (fixed message lists → Nemotron; no live ASR/TTS) as the primary iteration loop, then spot‑check live Magpie.

| Axis | Variants |
|------|----------|
| **A. History encoding** | **A0** Truncation only (current Pipecat / ElevenLabs‑style) · **A1** `<interrupted>` on committed assistant text **+** short developer note |
| **B. System prompt** | B0 current `generic_assistant_without_tools` · B1 §6 draft · B2+ refinements from failures |

**Scenarios:** long question + mid‑pause barge‑in; true topic change; “wait, I meant…”; “what did you just say?”; no‑interrupt control; multiple interrupts in one question.

**Scores:** correct merge vs over‑merge; factual completeness vs text gold; format (≤75 chars / one sentence); no marker leakage; no half‑answer imitation.

**Success (draft):** beat B0+A0 on reconstruction without regressing normal turns. Calibrate thresholds after ~20–30 fixtures.

---

## 8. POC Workstreams & Implementation

The POC is an orchestration layer around the existing cascaded stack (ASR → text LLM → TTS). Making **both** capabilities work is not prompt‑only; it requires the workstreams below.

### 8.1 Current baseline behavior (problem pattern)

* Streaming ASR produces partial transcripts.  
* A simple end‑of‑utterance rule (e.g. 800 ms of silence) triggers the LLM with the current partial transcript.  
* The LLM returns a full response; Magpie speaks it (`per_sentence` or `stitched`).  
* Assistant context advances at **sentence** `TTSTextFrame` granularity.  
* On barge‑in, playback stops and spoken‑so‑far text is committed; `on_assistant_turn_stopped(interrupted=True)` fires for app logic, but **no interrupt marker** is added to LLM history.  
* Subsequent LLM calls see fragmented user queries interleaved with unflagged assistant replies — the pattern that degrades reasoning quality.

### 8.2 Target POC behavior

* **Speculative vs committed:** full LLM output buffered separately; only audibly rendered text enters committed history (word‑level once Workstream 1 lands).  
* **Interruption‑aware truncation:** stop TTS on barge‑in; freeze at last played boundary; discard unplayed text; apply winning §7 encoding (`<interrupted>` + developer note or truncation‑only).  
* **Prompt structure:** committed history + interrupt semantics; LLM reconstructs full questions from prior user turns; no code‑side user‑turn merging in v1.

### 8.3 Workstream 1 — WordTTSService path for `NvidiaTTSService`

**Today:** `NvidiaTTSService(TTSService)` → sentence‑level commits.  
**Target:** Move Magpie onto Pipecat’s **WordTTSService‑style** path (word timestamps + `push_text_frames=False` / `add_word_timestamps`), as ElevenLabs does, so barge‑in commits **heard words**.

This is a **required step for Capability 1** and a foundation for Capability 2 (reconstruction quality is bounded by truncation fidelity).

* Evolve `NvidiaTTSService` toward WordTTS (or the equivalent modern audio‑context + word‑timestamp base class).  
* Surface Magpie/Riva word or alignment timing into `add_word_timestamps`.  
* Keep `per_sentence` / `stitched` synthesis compatible where possible; commit grain becomes word‑level regardless.  
* Validate: mid‑sentence barge‑in → assistant context ends at the last spoken word.

### 8.4 Workstream 2 — Orchestration changes

* Hook `on_assistant_turn_stopped`: when `interrupted=True`, apply the winning §7 encoding.  
* For A1: append `<interrupted>` and inject the developer note for the next LLM call.  
* Do not merge user turns in code in v1.  
* Wire into the generic cascaded pipeline (`generic` + `generic_assistant_without_tools`).

### 8.5 Workstream 3 — Prompt tuning

* Build the offline prompt‑eval harness.  
* Run A0 vs A1 × B0/B1/…; iterate §6 wording (especially anti‑half‑response rules).  
* Ship the winning prompt into the catalog.  
* Confirm on a short live Magpie session after WordTTS + orchestration land.

### 8.6 Suggested order

```text
1. Offline prompt harness + A0/A1 experiments          (can start immediately)
2. NvidiaTTSService → WordTTSService-style path        (enables accurate commits)
3. Orchestration on on_assistant_turn_stopped          (marker / developer note)
4. Integrate winning prompt + live smoke / Big Bench Audio subset eval
```

Steps 1 and 2 can proceed in parallel; step 3 depends on a chosen encoding from 1 and benefits from word‑level commits from 2.

---

## 9. Expected Impact

By keeping dialog state strictly consistent with what the user hears and by enabling the LLM to reconstruct long, pause‑filled questions as coherent queries from fragmented turns, this orchestration is expected to:

* Reduce the gap between text‑only and cascaded speech performance on reasoning‑heavy benchmarks such as Big Bench Audio and other audio‑reasoning suites, when using the same underlying text LLM.  
* Improve robustness in realistic, noisy, and interruption‑rich settings where naive turn segmentation fails, leveraging established findings from interruption handling in conversational robots.  
* Provide a reusable, model‑agnostic layer that can be integrated into existing voice products with minimal risk, because it changes orchestration and state management rather than core models.  
* Make the dependency story explicit: **prompt alone cannot fix sentence‑coarse commits; WordTTS alone cannot teach barge‑in semantics** — both plus orchestration are required.

---

## 10. Proposed Next Steps

It is proposed that a focused proof‑of‑concept be initiated on the existing cascaded voice stack. The POC would:

1. Stand up the **offline prompt‑eval harness**; evaluate **A0 (truncation‑only) vs A1 (`<interrupted>` + developer note)** with the §6 draft; iterate until reconstruction is stable and the model does not emit half‑responses or `<interrupted>`.  
2. Plan and implement **`NvidiaTTSService` → WordTTSService‑style** word‑timestamp commits in Pipecat.  
3. Implement **orchestration** on `interrupted=True` using the winning history encoding; update the catalog prompt.  
4. Run controlled evaluations comparing:  
   * Existing voice orchestration vs. the new orchestration.  
   * Speech‑pipeline performance vs. text‑only performance using the same LLM, on a subset of Big Bench Audio and similar speech reasoning tasks.

This POC would provide concrete evidence that better orchestration — paired with word‑level commit fidelity and a tuned prompt — can significantly narrow the speech–text reasoning gap and justify further investment in this direction.

---

## References

- Cao, S. et al., “Interruption Handling for Conversational Robots,” arXiv, 2025. https://arxiv.org/html/2501.01568v1  
- Dasha.AI, “Handling interruptions with LLM – Dasha Quick Start Guide,” accessed 2026. https://docs.dasha.ai/en-us/default/gpt/interruptions  
- Microsoft, “Handle voice interruptions in chat history (preview),” Voice Live API documentation, 2026. https://learn.microsoft.com/en-us/azure/ai-services/speech-service/how-to-voice-live-auto-truncation  
- Pipecat Interruptions / Context Management. https://docs.pipecat.ai/pipecat/fundamentals/interruptions · https://docs.pipecat.ai/pipecat/learn/context-management  
- Pipecat ElevenLabs TTS (word‑level timestamps). https://docs.pipecat.ai/api-reference/server/services/tts/elevenlabs  
