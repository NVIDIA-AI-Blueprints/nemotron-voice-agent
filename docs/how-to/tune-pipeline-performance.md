# Tune Pipeline Performance

This section covers advanced pipeline configurations for optimizing the performance and user experience of the Nemotron Voice Agent.

- [Speculative Speech Processing](#speculative-speech-processing)
- [Chat History Limit](#chat-history-limit)
- [Audio Debugging](#audio-debugging)
- [Audio Output Buffering](#audio-output-buffering)

## Speculative Speech Processing

Speculative speech processing reduces response latency by processing interim ASR transcripts before the user finishes speaking. This feature is enabled by default and provides approximately 300ms reduction in end-to-end latency.

### Configuration

Speculative speech processing is controlled by the `ENABLE_SPECULATIVE_SPEECH` environment variable in [.env](../../config/env.example).

```bash
# Enable speculative speech processing (default)
ENABLE_SPECULATIVE_SPEECH=true

# Or disable speculative speech processing
ENABLE_SPECULATIVE_SPEECH=false
```

**Note:** This feature only works with Nemotron Speech ASR.

### Benefits

When enabled, speculative speech processing provides:

| Benefit | Description |
|---------|-------------|
| **Lower latency** | ~300ms reduction by starting response generation before user finishes speaking |
| **Natural conversation flow** | TTS responses are cached and released at appropriate times |
| **Parallel processing** | LLM and TTS services process interim transcripts while user continues speaking |

### How It Works

1. User speaks. ASR generates interim transcripts with stability scores as audio is processed.
2. Stable interims processed. Transcripts with stability=1.0 are sent to LLM for early response generation.
3. Responses cached. TTS outputs are buffered while the user is still speaking.
4. User stops speaking. Cached responses are released, providing faster perceived response time.
5. Final transcript arrives. If different from interims, the response is updated.

![Workflow](../images/speculative_speech_user_context.png)

### Key Components

The following NVIDIA Pipecat components enable speculative speech processing. For more details, refer to the [NVIDIA Pipecat documentation](../05-nvidia-pipecat.md).

| Component | Purpose |
|-----------|---------|
| `NvidiaUserContextAggregator` | Filters stable interim transcripts and manages conversation context |
| `NvidiaAssistantContextAggregator` | Updates responses as context changes and prevents overlapping turns |
| `NvidiaTTSResponseCacher` | Buffers TTS responses during user speech and coordinates release timing |

### Pipeline Configuration

When speculative speech is enabled, the agent's pipeline includes the TTS response cacher:

```python
pipeline = Pipeline([
    transport.input(),
    stt,                                    # NemotronASRService
    context_aggregator.user(),              # Filters interim transcripts
    llm,
    tts,
    tts_response_cacher,                    # Caches responses during user speech
    transport.output(),
    context_aggregator.assistant()
])
```

When disabled, the response cacher is removed and only final transcripts are processed.

### Advanced: Building Custom Frame Processors

When developing components that work with speculative speech processing, follow these guidelines:

**Handle Interim States**
- Design frames to carry stability information.
- Include mechanisms to update or replace interim content.
- Implement clear state transitions from interim to final.

**Design for Incremental Updates**
- Support partial response processing and cancellation.
- Handle transitions between interim and final states.
- Consider that `TTSRawAudio` frames are cached until release conditions are triggered.

### Technical Foundation

This implementation builds on NVIDIA Nemotron Speech ASR's Two-Pass End of Utterance mechanism:

- Real-time interim transcript generation with stability metrics
- Hypothesis refinement as more audio is processed
- Clear signaling of final transcripts

For more details, refer to the [NVIDIA Nemotron Speech ASR documentation](https://docs.nvidia.com/deeplearning/riva/user-guide/docs/asr/asr-overview.html#two-pass-end-of-utterance).

## Chat History Limit

To control the conversation context window, set the `CHAT_HISTORY_LIMIT` environment variable to the number of conversation turns to retain in the `.env` file. By default, the conversation context window is set to 20.

```bash
# In .env file
CHAT_HISTORY_LIMIT=20  # Number of conversation turns to retain
```

**Recommendations**
- **Standard conversations**: 20 (default)
- **Emotion-aware TTS**: 3-5 (better emotion tracking)
- **Multilingual mode**: 3-5 (better language detection)

## Audio Debugging

You can enable raw audio capture for ASR/TTS debugging and issue reproduction.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ASR_AUDIO_DUMP` | `false` | Capture incoming user audio |
| `ENABLE_TTS_AUDIO_DUMP` | `false` | Capture outgoing synthesized audio |
| `AUDIO_DUMP_PATH` | `./audio_dumps` | Output directory for WAV files |

To enable audio debugging, set the environment variables as follows in the `.env` file.

```bash
ENABLE_ASR_AUDIO_DUMP=true
ENABLE_TTS_AUDIO_DUMP=true
AUDIO_DUMP_PATH=./audio_dumps # Output directory for WAV files.
```

Output files use WAV format with stream IDs for correlation.

> **Note:** If Docker creates the folder with different permissions, you can fix this in one of two ways:
>- Option 1: Pre-create directory before container start
>    ```bash
>    mkdir -p ./audio_dumps
>    ```
>- Option 2: Fix ownership after container creates it
>    ```bash
>    sudo chown -R $(id -u):$(id -g) ./audio_dumps
>    ```

> **Warning:** Disable audio debugging in production to prevent disk exhaustion.

## Audio Output Buffering

To control audio output latency and stability, set the `AUDIO_OUT_10MS_CHUNKS` environment variable to the number of 10ms chunks to buffer for output. By default, the audio output buffer size is set to 5.

```bash
# In .env file
AUDIO_OUT_10MS_CHUNKS=5  # Number of 10ms chunks to buffer
```

The following are the configuration guidelines for the `AUDIO_OUT_10MS_CHUNKS` environment variable.
- **Default WebRTC**: 5 chunks (50ms buffer) - optimized for low latency
- **Default WebSocket**: 10 chunks (100ms buffer) - more stable for network variations
- **High Concurrency**: 10-40 chunks (100-400ms buffer) - prevents audio glitches under high load
