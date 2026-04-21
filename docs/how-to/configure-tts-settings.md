# Configure TTS Settings

The Text-to-Speech (TTS) system supports multiple voices and languages using the [NVIDIA Magpie TTS model](https://build.nvidia.com/nvidia/magpie-tts-multilingual/modelcard).

## Default Multilingual TTS Voice

```bash
# In .env file
TTS_DOCKER_IMAGE=nvcr.io/nim/nvidia/magpie-tts-multilingual:1.6.0
TTS_VOICE_ID=Magpie-Multilingual.EN-US.Aria
TTS_MODEL_NAME=magpie_tts_ensemble-Magpie-Multilingual
TTS_LANGUAGE=en-US
TTS_NIM_TAGS=name=magpie-tts-multilingual,batch_size=32
```

The voice ID format of the Magpie Multilingual TTS model is `Model.Language.VoiceName`.

**Note:** The available voices depend on your Magpie TTS model version. Refer to the [NVIDIA Magpie TTS documentation](https://docs.nvidia.com/nim/riva/tts/latest/support-matrix.html#available-voices) for the complete voice list.

## Using Cloud TTS Endpoints

1. Set your NVIDIA API key as an environment variable:

    ```bash
    export NVIDIA_API_KEY=<your-nvidia-api-key>
    ```

2. Update the following environment variables in the `.env` file to use the Magpie Multilingual TTS model on NVIDIA's cloud endpoint.

    ```bash
    # In .env file
    TTS_SERVER_URL=grpc.nvcf.nvidia.com:443
    ```

3. Comment out the `tts-service` service in [docker-compose.yml](../../docker-compose.yml) when using cloud endpoints.

4. Remove any dependencies on `tts-service` from the `python-app` service in your [`docker-compose.yml`](../../docker-compose.yml) file.

    ```yaml
    # In docker-compose.yml:
    python-app:
      ...
      environment:
      # Docker services endpoints
      # - TTS_SERVER_URL=${TTS_SERVER_URL:-tts-service:50051} <-- comment out or remove this line
      ...
      depends_on:
      # - tts-service  <-- comment out or remove this line
    ```

5. Restart the services:

    ```bash
    docker compose down
    docker compose up -d
    ```

## Pronunciation Correction

You can customize word pronunciation using International Phonetic Alphabet (IPA).

1. Edit [config/ipa.json](../../config/ipa.json) and add custom word-to-IPA mappings:

    ```json
    {
      "NVIDIA": "ˈɛnˌvɪdiə",
      "GreenForce": "ɡriːn fɔrs",
      "API": "eɪ piː aɪ"
    }
    ```

2. Set the environment variable `TTS_IPA_FILE_PATH` to the path of the IPA file. In the example [.env](../../config/env.example) file, the IPA file path is set to `./config/ipa.json`.

    ```bash
    TTS_IPA_FILE_PATH=./config/ipa.json
    ```

The pipeline automatically applies IPA corrections to TTS output.

## Adding Text Filters

Apply text filters to remove special characters that can cause Magpie TTS failures.

```bash
# In .env file
ENABLE_TTS_TEXT_FILTER=true  # Default: true
```

Consider the following when adding text filters:

- The filter runs only for `TTS_LANGUAGE=en-US` and is skipped for other languages.
- To create custom filters for your use case or language, extend the `BaseTextFilter` class from [pipecat-ai](https://github.com/pipecat-ai/pipecat/blob/v0.0.98/src/pipecat/utils/text/base_text_filter.py).
