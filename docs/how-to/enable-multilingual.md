# Enable Multilingual Voice Agent

This guide explains how to enable multilingual support in the Nemotron Voice Agent. When enabled, the agent detects the user's language and responds in the same language. Unsupported languages fall back to English. Refer to [Supported Languages](#supported-languages) for the list of supported languages.

The following components enable multilingual conversations.

## Key Components

This guide uses the following key components to build the multilingual voice agent:

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **NVIDIA Parakeet RNNT ASR** | Transcribes speech in multiple languages | [Parakeet ASR](https://build.nvidia.com/nvidia/parakeet-1_1b-rnnt-multilingual-asr) |
| **NVIDIA Magpie TTS** | Synthesizes speech in multiple languages | [Magpie TTS](https://build.nvidia.com/nvidia/magpie-tts-multilingual) |
| **NVIDIA Llama Nemotron LLM** | Generates multilingual responses with structured language output | [Llama Nemotron](https://build.nvidia.com/nvidia/llama-3_3-nemotron-super-49b-v1_5) |

## How Multilingual Support Works

The following sections describe how the multilingual pipeline components work together.

### LLM Response Format

The multilingual system uses a structured output format defined in [config/prompt.yaml](../../config/prompt.yaml) to coordinate language detection and TTS routing.

```
Language: <LangCode> Text: <DirectResponse> MetaData: <AdditionalInfo>
```

| Field | Description |
|-------|-------------|
| `Language` | Detected language code (for example, `en-US`, `de-DE`, `fr-FR`) |
| `Text` | The spoken response content—this is what the user hears |
| `MetaData` | Additional context not meant to be spoken (optional) |

The following are some example responses:

```
Language: en-US Text: How can I help you today? MetaData: greeting
Language: de-DE Text: Gerne! Welche Blumen moechten Sie? MetaData: flower inquiry
Language: fr-FR Text: Bonjour! Comment puis-je vous aider? MetaData: none
Language: es-US Text: Hola! Que tipo de flores necesita? MetaData: initial contact
```

### Language Detection Rules

The agent uses these language detection rules.

- **Per-Message Detection**: Language is detected from each user message independently based on LLM analysis of transcripts.
- **Supported Languages Only**: Responses use only languages supported by the TTS model.
- **Graceful Fallback**: Unsupported languages default to `en-US` with English response.

## Deploying the Agent in Multilingual Mode

Follow these steps to deploy the Nemotron Voice Agent in multilingual mode.

1. Copy the example environment file to the root directory.

    ```bash
    cp config/env.example .env
    ```

2. Edit the [.env](../../config/env.example) file and enable multilingual mode.

    ```bash
    ENABLE_MULTILINGUAL=true
    ```

    **Tip:** Set `CHAT_HISTORY_LIMIT` to `3`-`5` in the `.env` file for better language detection accuracy.

3. Configure multilingual ASR by replacing the default ASR settings with the following values.

    ```bash
    # Replace the default English-only ASR with multilingual ASR
    ASR_DOCKER_IMAGE=nvcr.io/nim/nvidia/parakeet-1-1b-rnnt-multilingual:1.4.0
    ASR_MODEL_NAME=parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming
    ASR_NIM_TAGS=mode=str
    ASR_CLOUD_FUNCTION_ID=71203149-d3b7-4460-8231-1be2543a1fca #if using nvcf endpoint
    ```

    **Note:** These values replace the default `parakeet-1-1b-ctc-en-us` configuration. Comment out or remove the existing ASR settings before adding these.

4. Configure the LLM for multilingual by uncommenting OPTION 2 in the [.env](../../config/env.example) file and commenting out OPTION 1.

    Multilingual mode uses the **Llama 3.3 Nemotron Super 49B** model, which is significantly larger than the default Nemotron-3-Nano model. Ensure your system has sufficient VRAM to run the 49B model before enabling multilingual mode. Refer to the [main Requirements section](../../README.md#requirements) for general hardware and software prerequisites.

    ```bash
    # Comment out OPTION 1 (Nemotron-3-Nano) and uncomment OPTION 2:
    NVIDIA_LLM_IMAGE=nvcr.io/nim/nvidia/llama-3.3-nemotron-super-49b-v1.5:1.15.4
    NVIDIA_LLM_MODEL=nvidia/llama-3.3-nemotron-super-49b-v1.5
    TEMPERATURE=0
    TOP_P=1.0
    NIM_ENABLE_KV_CACHE_REUSE=1
    SYSTEM_PROMPT_SELECTOR=llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant
    ```

    > **Note:** Each model has a matching `SYSTEM_PROMPT_SELECTOR` value. Use the prompt selector that corresponds to your chosen model. In this case, the prompt selector for multilingual voice agent is `llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant`.

5. Deploy the Nemotron Voice Agent.

    ```bash
    docker compose up -d
    ```

## Testing the Multilingual Agent

After deployment, open your browser and navigate to `http://<machine-ip>:9000/` to access the voice agent UI.

### Example Conversations

The following examples demonstrate how the agent responds to different languages.

| User Input (Speech) | Agent Response | Language Detected |
|---------------------|----------------|-------------------|
| "Hello, how are you?" | "I'm doing well, thank you for asking!" | en-US |
| "Bonjour, comment ça va?" | "Je vais bien, merci!" | fr-FR |
| "Hallo, wie geht es dir?" | "Mir geht es gut, danke!" | de-DE |
| "Hola, ¿cómo estás?" | "Estoy bien, gracias!" | es-US |

### Supported Languages

The multilingual agent supports the following language codes by default.

| Language | Code |
|----------|------|
| English (US) | `en-US` |
| French | `fr-FR` |
| Spanish (US) | `es-US` |
| German | `de-DE` |
| Mandarin | `zh-CN` |
| Italian | `it-IT` |


You can customize the supported languages by editing the `lang_codes` variable in the `multilingual_voice_assistant` prompt in [config/prompt.yaml](../../config/prompt.yaml). By default, all languages supported by the deployed Magpie TTS model are automatically included as allowed language codes through runtime detection.

### Testing Language Switching

To test mid-conversation language switching:

1. Start a conversation in English: "Hello, I need help."
2. Switch to another language: "Können Sie mir auf Deutsch helfen?"
3. The agent detects the language change and responds in German.

The agent detects language per message, so you can switch languages at any point in the conversation.

## Troubleshooting

You can troubleshoot common issues and known limitations with the multilingual agent by referring to the following table:

| Issue | Cause | Solution |
|-------|-------|----------|
| Wrong language response | LLM not following format | Verify `SYSTEM_PROMPT_SELECTOR` is set to `llama-3.3-nemotron-super-49b-v1.5/multilingual_voice_assistant` |
| TTS speaks wrong language | Language code mismatch | Check LLM is outputting valid language codes |
| No speech output | Format parsing failure | Ensure LLM outputs correct `Language: Text: MetaData:` format |
| ASR not transcribing correctly | Using English-only model | Switch to `parakeet-rnnt-1.1b-unified-ml-cs-universal-multi-asr-streaming` |
| Background noise sensitivity | RNNT Multilingual ASR model interprets background noise as user input | Known limitation of RNNT Multilingual ASR model. Will be fixed in future versions |
| Mandarin and Italian ASR not recognized sometimes | ASR model has lower accuracy for some languages | Known limitation. Will be fixed in future versions |
| LLM not following prompt or adding prompt instruction in response | LLM prompt may require further optimization or finetuning | Optimize or customize the system prompt. Refer to [Customize System Prompts](./customize-system-prompts.md) for guidance |
| LLM appends English sentence to non-English language | LLM language consistency varies by context or prompt | Optimize system prompt for language consistency. Refer to [Customize System Prompts](./customize-system-prompts.md) |
