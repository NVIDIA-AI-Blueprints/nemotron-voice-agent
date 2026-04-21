# Customize System Prompts

You can customize your voice agent's personality, behavior, and response format using system prompts.

To set up a system prompt, set the following environment variables in the `.env` file:
- `PROMPT_FILE_PATH`: The path to the prompt file to use. The prompt file has a collection of system prompts. In this blueprint, the prompt file is [config/prompt.yaml](../../config/prompt.yaml).
- `SYSTEM_PROMPT_SELECTOR`: The name of the system prompt to use from the prompt file. The format is `model-name/prompt-name`.

This blueprint includes several pre-configured prompt samples for various use cases.

## Using System Prompt Samples

The following are examples of calling the system prompt samples available in the blueprint's config file [config/prompt.yaml](../../config/prompt.yaml).

### Generic Voice Assistant

To use the generic voice assistant prompt sample, set the `SYSTEM_PROMPT_SELECTOR` in the `.env` file as follows.

```bash
# In .env file
SYSTEM_PROMPT_SELECTOR=nemotron-3-nano/generic_voice_assistant
# or
SYSTEM_PROMPT_SELECTOR=llama-3.3-nemotron-super-49b-v1.5/generic_voice_assistant
```

### Flowershop Assistant

To use the flowershop assistant prompt sample, set the `SYSTEM_PROMPT_SELECTOR` in the `.env` file as follows.

```bash
# In .env file
SYSTEM_PROMPT_SELECTOR=llama-3.1-8b-instruct/flowershop
```

**Characteristics**
- Persona: Flora from GreenForce Garden
- Handles order management, consultations, delivery coordination

### TTS Emotion Tags

To use the TTS emotion tags prompt sample, set the `SYSTEM_PROMPT_SELECTOR` in the `.env` file as follows. This prompt sample enables dynamic emotional TTS with real-time emotion control.

```bash
# In .env file
SYSTEM_PROMPT_SELECTOR=llama-3.1-8b-instruct/tts_emotion_tags
```

The generated LLM output format with this system prompt is as follows.
```
Emotion: <Happy|Calm|Neutral|Sad|Angry|Fearful> Text: <response>
```

The following are the characteristics of this prompt.
- LLM outputs emotion tags parsed by the pipeline.
- TTS voice changes based on emotion context.
- Supported emotions: `Happy`, `Calm`, `Neutral`, `Sad`, `Angry`, `Fearful`.
- **Requirements**: Supported only with the Magpie Multilingual TTS model in English (en-US).
- **Configuration**: Set `CHAT_HISTORY_LIMIT=3` for best results.

### Multilingual Voice Assistant

To deploy the voice agent with multilingual support, refer to [Enable Multilingual Voice Agent](./enable-multilingual.md). The multilingual prompt uses automatic language detection and responds in the user's language.

## Creating Custom Prompts

1. Add your own system prompt to the prompt file [config/prompt.yaml](../../config/prompt.yaml) by following the format below.

    ```yaml
    your-model-name:
      custom_prompt_name:
        description: "Your prompt description"
        messages:
          - role: system
            content: |
              Your system prompt here...
              Define personality, rules, and response format.
    ```

    When creating your own system prompt for outputting text, follow these best practices:
    - Keep responses concise (1-2 sentences, less than 200 characters).
    - Avoid special characters like `*`, `-`, `/` in output.
    - Avoid bullet points or numbered lists (breaks voice flow).
    - Define clear output format for structured data.
    - Use plain text only.

2. Update the `SYSTEM_PROMPT_SELECTOR` in the `.env` file to use your custom prompt.

    ```bash
    SYSTEM_PROMPT_SELECTOR=your-model-name/custom_prompt_name
    ```
