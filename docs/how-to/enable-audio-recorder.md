# Enable the Audio Recorder

The audio recorder captures raw ASR/TTS audio for debugging and issue reproduction. Each session is saved as one stereo file, with the user on the left channel and the agent on the right channel. This guide applies to the Pipecat examples (`nva-pipecat`) and the LiveKit Generic Assistant.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_AUDIO_DUMP` | `false` | Turn on session recording |
| `AUDIO_DUMP_PATH` | `audio_dumps` | Output directory (relative to project root, or absolute) |

To enable the audio recorder, set the environment variables in the `.env` file:

```bash
ENABLE_AUDIO_DUMP=true
AUDIO_DUMP_PATH=audio_dumps
```

For Docker Compose runs, recreate the app service so that it reads the new `.env` values. `docker compose restart` keeps the old environment. Use the app service of your recipe, for example `generic-assistant-single-gpu` for `--profile generic-assistant/single-gpu`:

```bash
docker compose --profile <recipe> up -d <app-service>
```

The shipped examples already create and wire the shared recorder, so the `.env` settings are enough to enable capture for those examples. To add the recorder to a **new custom example**, mirror [`src/nemotron_voice_agent/pipecat/examples/generic/pipeline.py`](../../src/nemotron_voice_agent/pipecat/examples/generic/pipeline.py) with three changes to your `pipeline.py`:

1. Import the helper:

    ```python
    from nemotron_voice_agent.pipecat.shared.audio_recorder import create_audio_recorder
    ```

2. Create the recorder and add it to the pipeline. `create_audio_recorder()` returns `None` when `ENABLE_AUDIO_DUMP` is off, so it stays a no-op until enabled:

    ```python
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline(
        [
            transport.input(),
            # ... ASR, LLM, TTS, transport.output() ...
            *([audio_recorder] if audio_recorder else []),
        ]
    )
    ```

3. Start it once the client connects (for example, in your `on_client_connected` handler):

    ```python
    if audio_recorder:
        await audio_recorder.start_recording()
    ```

With those in place, the `ENABLE_AUDIO_DUMP` and `AUDIO_DUMP_PATH` settings above control capture for your custom example.

## Output Format

Each session is saved as one stereo file when the session ends.

```text
audio_dumps/
├── session_<stream_id>.wav   # Pipecat: 16-bit PCM WAV
└── session_<job_id>.ogg      # LiveKit: Ogg Opus
```

Pipecat uses the stereo recording of its `AudioBufferProcessor`. LiveKit uses the session recorder of LiveKit Agents. The ID is unique per session, so files from concurrent sessions do not collide. The `INFO` log line of the recorder names the session ID. The Frontend/Backend Live example records its live sessions, over WebRTC and WebSocket, each from its start.

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
