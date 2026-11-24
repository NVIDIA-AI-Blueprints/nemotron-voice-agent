# Enable the Audio Recorder

Capture session audio to diagnose speech recognition, synthesis, and playback issues. This guide applies to Pipecat (`nva-pipecat`) and the Generic Assistant on LiveKit. Recordings can contain sensitive conversation data. Limit access to the output directory and retain only what you need for debugging.

## Configuration

These settings control both frameworks. In 2.4.0, `ENABLE_AUDIO_DUMP` replaces `ENABLE_ASR_AUDIO_DUMP` and `ENABLE_TTS_AUDIO_DUMP`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUDIO_DUMP` | `false` | Turn on session recording |
| `AUDIO_DUMP_PATH` | `audio_dumps` | Output directory (relative to project root, or absolute) |

1. Enable recording in `.env`:

    ```dotenv
    ENABLE_AUDIO_DUMP=true
    AUDIO_DUMP_PATH=audio_dumps
    ```

2. Apply the environment change. For a host-native run, restart the app. For Compose, use `up -d` so Compose can recreate the app service. `docker compose restart` keeps the old environment. For example, the Pipecat `generic-assistant/single-gpu` profile uses the `generic-assistant-single-gpu` service:

    ```bash
    docker compose --profile <recipe> up -d <app-service>
    ```

3. Run a voice session and end it to flush captured audio to disk.

The shipped examples already create and wire the shared recorder, so the `.env` settings are enough to enable capture for those examples. To add the recorder to a **new custom example**, mirror [`src/nemotron_voice_agent/pipecat/examples/generic/pipeline.py`](../../src/nemotron_voice_agent/pipecat/examples/generic/pipeline.py) with three changes to your `pipeline.py`:

1. Import the helper:

    ```python
    from nemotron_voice_agent.pipecat.shared.audio_recorder import create_audio_recorder
    ```

2. Create the recorder and add its three parts to the pipeline. `inputs` goes right after the input transport, `outputs` right before the output transport, and `after` after it. The parts are empty when `ENABLE_AUDIO_DUMP` is off, so the recorder stays a no-op until enabled:

    ```python
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            *audio_recorder.inputs,
            # ... ASR, LLM, TTS ...
            *audio_recorder.outputs,
            transport.output(),
            *audio_recorder.after,
        ]
    )
    ```

3. Start it once the client connects (for example, in your `on_client_connected` handler):

    ```python
    await audio_recorder.start_recording()
    ```

With those in place, the `ENABLE_AUDIO_DUMP` and `AUDIO_DUMP_PATH` settings above control capture for your custom example.

## Output Format

At session end, the recorder writes files for the streams it captured. An empty stream can leave no file. Expect at most one server-side conversation mix per session, in the format for your framework:

```text
audio_dumps/
├── session_<id>.wav       # Pipecat: stereo mix, 16-bit PCM WAV
├── session_<id>.ogg       # LiveKit: stereo mix, Ogg Opus
├── asr_<id>_session.wav   # the audio that ASR received, 16-bit PCM WAV
└── tts_<id>_session.wav   # the audio that TTS produced, 16-bit PCM WAV
```

| File | What it holds |
|------|---------------|
| `session_<id>` | A server-side conversation mix for playback, not a record of what reached the listener. Pipecat uses the stereo output of `AudioBufferProcessor`. LiveKit uses the LiveKit Agents session recorder. Both resample the audio. |
| `asr_<id>_session.wav` | Every audio frame that ASR received in the session, at its own sample rate, before turn handling filters any audio. It also holds the microphone audio from before the user speaks. Use it to replay a session against ASR. In the Omni Assistant, it holds the audio that the Omni model received. |
| `tts_<id>_session.wav` | Every audio frame that TTS sent toward the output in the session, at its own sample rate. |

The ASR and TTS files start together and share one timeline. When a stream pauses for more than 0.2 seconds, the file holds silence for the pause, so the two files line up. The ID is unique per session, so files from concurrent sessions do not collide. The `INFO` log line of the recorder names the session ID. The Frontend/Backend Live example records its live sessions, over WebRTC and WebSocket, each from its start.

> **Note:** For host-native runs, files are written under the project root by default. For Pipecat Docker Compose runs, files are written to `/app/audio_dumps` inside the app container. The stock `docker/pipecat/docker-compose.yaml` does not mount that directory, so the files are lost when the container is recreated. Copy them to the host after capture:
>
> ```bash
> docker compose --profile <recipe> cp <app-service>:/app/audio_dumps ./
> ```
>
> To write the files to the host directly, mount a host directory with a Compose override, and then recreate the app service:
>
> ```yaml
> # docker-compose.override.yaml (example for --profile generic-assistant)
> services:
>   generic-assistant:
>     volumes:
>       - ./audio_dumps:/app/audio_dumps
> ```
>
> A session fails to start when the server cannot write to the folder, and the server log names the folder.
>
> If Docker creates the folder with different permissions, fix ownership:
>
> ```bash
> # Option 1: Pre-create directory before container start
> mkdir -p ./audio_dumps
> # Option 2: Fix ownership after container creates it
> sudo chown -R $(id -u):$(id -g) ./audio_dumps
> ```

## LiveKit Generic Assistant

`docker/livekit/docker-compose.yaml` mounts `./audio_dumps` on the agent services, so with the default `AUDIO_DUMP_PATH` the files appear in `./audio_dumps` on the host. Recreate the agent of your profile after you change `.env`, for example:

```bash
docker compose --profile generic-assistant-livekit/single-gpu up -d livekit-agent-single-gpu
```

The agent writes the files as `root`. To fix ownership, run `sudo chown -R $(id -u):$(id -g) ./audio_dumps`.

> **Warning:** Disable the audio recorder in production to prevent disk exhaustion.
