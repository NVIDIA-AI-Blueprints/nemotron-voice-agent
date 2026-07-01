# Enable Zero-Shot TTS

Zero-shot TTS allows you to clone any voice using a short audio sample (5+ seconds). This feature uses the Magpie Zero-shot model. Apply for access at the [NVIDIA RIVA TTS Zero-shot models page](https://developer.nvidia.com/riva-tts-zeroshot-models) before you configure the service.

## Steps

1. Prepare your audio sample to meet the following requirements.
   - Audio format is WAV (16-bit PCM recommended).
   - Audio duration is at least 5 seconds of clean speech.
   - Audio quality is clear, no background noise.
   - Language should match your target language.
   - Speaker is single speaker only.

2. Create the `audio_prompts` directory and add your voice sample.

    ```bash
    mkdir -p audio_prompts
    cp <your_audio_file>.wav audio_prompts/<your_audio_file>.wav
    ```

3. Choose how to run the Magpie zero-shot NIM microservice.

   - **Compose-managed service:** Replace the default `tts-service` image and configuration in `.env`, then let `docker compose` manage the service.
   - **External service:** Run the zero-shot NIM outside this compose stack by following the [NVIDIA NIM RIVA TTS documentation](https://docs.nvidia.com/nim/riva/tts/latest/getting-started.html#launching-the-nim), then set `TTS_SERVER_URL` to that endpoint and disable the local `tts-service`.

   Use one path only. Do not run both the compose-managed `tts-service` and an external TTS service for the same deployment.

4. For the compose-managed path, set the environment variables in the `.env` file as follows.

    ```bash
    # Comment out standard TTS configuration
    #TTS_DOCKER_IMAGE=nvcr.io/nim/nvidia/magpie-tts-multilingual:1.6.0
    #TTS_VOICE_ID=Magpie-Multilingual.EN-US.Aria
    #TTS_MODEL_NAME=magpie_tts_ensemble-Magpie-Multilingual
    #TTS_NIM_TAGS=name=magpie-tts-multilingual,batch_size=32

    # Enable Zero-shot TTS
    TTS_DOCKER_IMAGE=<ZEROSHOT_NIM_MICROSERVICE_IMAGE> # Use your version
    TTS_VOICE_ID=Magpie-ZeroShot.Female-1
    TTS_MODEL_NAME=magpie_tts_ensemble-Magpie-ZeroShot
    TTS_NIM_TAGS=name=magpie-tts-zeroshot,batch_size=32
    ZERO_SHOT_AUDIO_PROMPT=audio_prompts/custom_voice.wav
    ```

5. Mount the audio prompts into `python-app` in [docker-compose.yml](../../docker-compose.yml). The Python application reads `ZERO_SHOT_AUDIO_PROMPT` and sends the prompt data to the TTS service.

    ```yaml
    python-app:
      # ... existing configuration ...
      volumes:
        - ./audio_dumps:/app/audio_dumps
        - ./config/:/app/config/
        - ./audio_prompts:/app/audio_prompts  # Add this line
    ```

6. Deploy the services with a rebuild so the compose changes are applied:

    ```bash
    docker compose up --build -d
    ```

7. Verify the deployment.

    ```bash
    docker compose ps tts-service python-app
    docker compose logs -f tts-service python-app
    ```

    Start a conversation in the UI and confirm that the response uses the voice from your audio prompt. If you use the external-service path, verify that `TTS_SERVER_URL` points to the external NIM endpoint and that the local `tts-service` is not running.
