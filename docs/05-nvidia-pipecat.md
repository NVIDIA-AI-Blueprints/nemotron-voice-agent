# NVIDIA Pipecat Services

[NVIDIA Pipecat](https://pypi.org/project/nvidia-pipecat/) provides services for building full-duplex voice agents with NVIDIA ASR, TTS, RAG, and NIM LLM microservices. The services are compatible with the [Pipecat framework](https://github.com/pipecat-ai/pipecat) and you can integrate them into any Pipecat pipeline.

> **Note**
> Some advanced concepts, such as speculative speech processing, require careful integration with existing Pipecat pipelines. You might need to adapt and upgrade your frame processor implementation to work with these concepts and processors.

The following sections give a brief overview of the processors in the [nvidia-pipecat](https://pypi.org/project/nvidia-pipecat/) library and link to the corresponding documentation.

## Core Speech Services

The following table lists the core speech services:

| Pipecat Service | Description |
| --- | --- |
| `NemotronASRService` | Provides streaming speech recognition using NVIDIA [Nemotron Speech ASR models](https://docs.nvidia.com/nim/riva/asr/latest/overview.html). Supports real-time transcription with interim results and interruption handling. |
| `NemotronTTSService` | Provides high-quality speech synthesis using NVIDIA [Nemotron Speech TTS models](https://docs.nvidia.com/nim/riva/tts/latest/overview.html). Supports multiple voices, languages, and custom dictionaries for pronunciation. |

## LLM, RAG, and NAT Services

The following table lists LLM, RAG, and NAT services:

| Pipecat Service | Description |
| --- | --- |
| `NvidiaLLMService` | Extends `LLMService` and serves as the base class for services that connect to [NVIDIA NIM LLMs](https://docs.nvidia.com/nim/large-language-models/latest/introduction.html) using the ChatNvidia client. |
| `NvidiaRAGService` | Use this service when [NVIDIA RAG](https://github.com/NVIDIA-AI-Blueprints/rag/) is the dialog management component in the pipeline. |
| `NATAgentService` | Integrates with NVIDIA [NeMo Agent Toolkit](https://docs.nvidia.com/nemo/agent-toolkit/1.2/api/nat/index.html) to use AI agents in the voice pipeline. |

## Speculative Speech Processing Services

The following table lists speculative speech processing services:

| Pipecat Service | Description |
| --- | --- |
| `NvidiaUserContextAggregator` | Manages NVIDIA-specific user context for speculative speech processing, tracking interim and final transcriptions to enable real-time response generation. |
| `NvidiaAssistantContextAggregator` | Specializes the base LLM assistant context aggregator for NVIDIA, handling assistant responses and maintaining conversation context during speculative speech processing. |
| `NvidiaContextAggregatorPair` | A matched pair of user and assistant context aggregators that collaboratively maintain bidirectional conversation state. |
| `NvidiaTTSResponseCacher` | Manages speculative speech TTS response timing by buffering during user input, coordinating playback with speech state, and queuing to prevent overlap and ensure natural turn-taking. |

## Synchronization and RTVI Processors

The following table lists synchronization and RTVI processors:

| Pipecat Service | Description |
| --- | --- |
| `UserTranscriptSynchronization` | Synchronizes user speech transcripts with the received speech. |
| `BotTranscriptSynchronization` | Synchronizes bot speech transcripts with bot speech playback (TTS playback). |
| `NvidiaRTVIInput` | Extends the base RTVIProcessor to handle WebRTC UI client messages such as context resets, voice changes, and audio uploads. |
| `NvidiaRTVIOutput` | Forwards transcript frames and Nemotron Speech ASR and TTS configuration frames (voice lists, TTS settings, system prompts) to the WebRTC UI client as server messages. |
