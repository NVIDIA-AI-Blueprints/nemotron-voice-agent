# Best Practices for Building Production-Grade Voice Agents

Building production-grade voice agents requires careful consideration of multiple dimensions: technical performance, user experience, security, and operational excellence. This guide consolidates best practices and lessons learned from deploying voice agents at scale.

---

## Key Success Metrics

The following metrics are key to the success of a voice agent.

- **Latency**: Time from user speech end to bot response start (target: 600-1500ms).
- **Accuracy**: ASR word error rate (WER), factual correctness, LLM generation quality, and so on.
- **Scalability**: Concurrent streams supported without audio glitches or performance degradation, all models scale independently.
- **Availability**: System uptime and fault tolerance (target: 99.9%+).
- **User Satisfaction**: Task completion rate and user feedback scores.

---

## Modular and Event-Driven Pipeline Design

Structure your voice agent as a composable pipeline of independent components:

```
Audio Input → VAD → ASR → Agent → TTS → Audio Output
```

Implement event-driven patterns for:
- Handling real-time transcription updates.
- Processing intermediate results.
- Monitoring system health events.
- Tracking user interaction events.
- Using async/await patterns for non-blocking operations.

**Benefits:**
- Test and scale individual components independently.
- Swap providers without full rewrites.

---

## Optimizing Pipeline Latency

To optimize latency, first measure end-to-end and per-component latency. Voice agent latency comes from multiple pipeline components. Understanding each contributor enables targeted optimization:

### Audio Processing Latency

**Voice Activity Detection (VAD):**
- **Contribution**: 200-500ms (end of speech detection)
- **Optimization**:
  - Use streaming VAD with shorter silence thresholds.
  - Explore shorter EOU detection with Nemotron Speech ASR and open-source smart turn detection models.
  - Implement adaptive VAD sensitivity based on environment noise.

**Audio Buffering:**
- **Contribution**: 50-200ms (network buffering, codec processing)
- **Optimization**:
  - Use lower latency audio codecs (Opus at 20ms frames).
  - Minimize audio buffer sizes while maintaining quality.
  - Implement jitter buffers for network variations.
- **Scaling Audio Output for Concurrency:**
  When scaling to multiple concurrent audio streams using either FastAPI WebSocket transport or WebRTC transport, consider increasing the output audio chunk size using the `AUDIO_OUT_10MS_CHUNKS` parameter up to 400ms to reduce audio glitches and enable smoother playback.

  **Configuration in `.env`:**
  ```bash
  # Number of 10ms chunks to buffer (default: 10)
  AUDIO_OUT_10MS_CHUNKS=10
  ```

### ASR (Automatic Speech Recognition) Latency

**Model Processing:**
- **Contribution**: 50-100 ms for Nemotron Speech ASR
- **Optimization**:
  - Deploy Nemotron Speech ASR NIM locally.
  - Utilize latest GPU hardware and optimized models.
  - Maintain consistent latency performance when handling multiple concurrent requests.
  - Use streaming ASR with interim results for early processing.

### Language Model (LLM) Processing Latency

**Model Inference:**
- **Contribution**: 200-800ms depending on model size and complexity
- **Optimization**:
  - **Model Selection**: Use smaller, faster models (8B compared to 70B parameters).
  - **TRT LLM Optimized**: Use TRT LLM optimized NIM deployments.
  - **Quantization**: Apply INT8/FP16 models for 2-3x speedup.
  - **KV-Cache Optimization**: Enable KV caching for lower TTFB and optimize based on use case.

**Context Management:**
- **Contribution**: 50-200ms for large contexts
- **Optimization**:
  - Implement context truncation strategies.
  - Enable KV caching with adequate cache size.

### TTS (Text-to-Speech) Latency

**Synthesis Time:**
- **Contribution**: 150-300ms for first audio chunk
- **Optimization**:
  - **Streaming TTS**: Start playback before full synthesis.
  - **Local Nemotron Speech TTS**: Achieve 150-200ms with TRT optimized Magpie model.
  - **Chunked Generation**: Process sentences as they are generated.
  - **Batch Size**: Increase the Magpie model batch size (for example, to 64) to boost throughput for high-volume or concurrent workloads.

**Audio Post-processing:**
- **Contribution**: 50-100ms (normalization, encoding)
- **Optimization**:
  - Minimize audio processing pipeline.
  - Use hardware-accelerated audio codecs.

### Network and Infrastructure Latency

- **Geographic Distribution:** Deploy distributed multi-node setups based on user demographics.
- **Load Balancing:** Use sticky sessions to avoid context switching.
- **Monitoring:** Monitor key metrics in production deployment.

### Advanced Latency Reduction Techniques

**Speculative Speech Processing:**
- Process interim ASR transcripts before speech ends.
- Pre-generate likely responses during user speech.
- **Potential Savings**: Achieve 200-400ms reduction in perceived latency.
- Refer to [Speculative Speech Processing](./how-to/tune-pipeline-performance.md#speculative-speech-processing) for details.

**Filler Words or Intermediate Responses:**
- Generate or use random filler words to reduce perceived latency.
- Generate intermediate responses based on function calls or thinking tokens for high latency agents or reasoning models.

---

## Designing User Experience

### Conversation Design Principles

**Natural Turn-Taking:**
- Allow interruptions (barge-in).
- Implement proper silence handling.
- Use conversational markers ("um", "let me check").

**Progressive Disclosure:**
```python
# Don't overwhelm with options
# Bad:
"You can check balance, transfer funds, pay bills, view history,
update profile, set alerts, or lock your card. What would you like?"

# Good:
"What would you like to do today?"
# (Let user guide, offer suggestions if confused)
```
### Persona and Tone Consistency

**Define Agent Personality:**
- Choose professional vs. casual tone.
- Decide proactive vs. reactive behavior.
- Set verbose vs. concise response style.
- Establish empathetic vs. neutral demeanor.

**Maintain Consistency:**
- Document persona guidelines.
- Use system prompts for LLMs.
- Implement tone checkers.
- Conduct regular quality reviews.

### Voice Selection

**Considerations:**
- Match voice to brand and use case.
- Consider user demographics.
- Evaluate regional accent preferences.
- Offer gender neutrality options.
- Create custom IPA dictionary for mispronunciations.

**Quality Metrics:**
- Target naturalness (MOS score > 4.0).
- Evaluate prosody and intonation.
- Assess emotional expressiveness.
- Ensure consistency across sessions.

### Response Optimization for Voice

**Voice-Specific Adaptations:**
- Keep responses concise (1-3 sentences per turn).
- Use conversational language (contractions, simple words).
- Structure information hierarchically.
- Avoid lists with more than 3-4 items.
- Use explicit transitions.

### Prompt Design

**System Prompt Instructions:**
- Include persona and tone guidelines directly in the system prompt.
- Instruct the model to avoid formatting (bullet points, markdown, URLs) that does not translate to voice.
- Define conversation boundaries and scope to prevent rambling.
- Include examples of ideal voice responses for few-shot guidance.
- Add instructions for progressive disclosure and context-aware suggestions.

### ASR Transcript Quality
- Implement custom vocabulary boosting for domain terms.
- Use inverse text normalization (ITN) for proper formatting.
- Ensure user audio quality is good.
- Avoid resampling if possible.
- Skip noise processing since Nemotron Speech ASR models are robust to noise.
- Base critical decisions on final transcripts only.
- Fine-tune ASR model on domain data if needed.

### User-Facing Error Handling

**Error Categories:**

```python
ERROR_MESSAGES = {
    "asr_failure": "I didn't catch that. Could you say that again?",
    "service_unavailable": "I'm having trouble connecting. Let me try again.",
    "timeout": "This is taking longer than expected. Please hold on.",
    "out_of_scope": "I'm not able to help with that, but I can help you with..."
}
```

**Recovery Strategies:**
- Offer alternative input methods (DTMF, transfer to human).
- Provide clear next steps.
- Terminate conversations gracefully.

### Continuous Testing
- Implement unit and integration testing.
- Run load testing to find latency bottlenecks.
- Prepare test data with different conversation scenarios.
- Use A/B testing to improve user experience.

---

## Scalability and Performance

### Horizontal Scaling

**Stateless Services:**
- Deploy ASR/TTS behind load balancers.
- Use container orchestration (Kubernetes).
- Enable auto-scaling based on CPU/memory/queue depth.

**Stateful Services:**
- Use sticky sessions.
- Implement distributed session storage (Redis).

### Resource Optimization

**Model Optimization:**
- Apply quantization (FP16, INT8) and TRT optimization for inference.
- Select smaller models for lower footprint.
- Use batch inference where possible.
- Enable GPU sharing and multiplexing.

### Network Optimization

**WebRTC Best Practices:**
- Use TURN servers for NAT traversal.
- Implement adaptive bitrate.
- Support multiple codecs (Opus preferred).
- Handle network transitions (WiFi to cellular).

---

## Conclusion

Building production voice agents requires a holistic approach balancing technical performance, user experience, and operational excellence. Key takeaways:

1. **Design for Latency**: Every millisecond counts in conversational AI
2. **Handle Errors Gracefully**: Users should never feel lost
3. **Monitor Everything**: You cannot improve what you do not measure
4. **Test Thoroughly**: Automated testing catches issues before users do
5. **Iterate Based on Data**: Use real user feedback to improve
6. **Plan for Scale**: Design for 10x your current load
7. **Prioritize Security**: Protect user data as your top responsibility
