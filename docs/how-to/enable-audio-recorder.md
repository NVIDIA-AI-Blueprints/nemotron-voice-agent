# Enable the Audio Recorder

The audio recorder captures raw ASR/TTS audio for debugging and issue reproduction. Each conversation turn is saved as a separate WAV file for easy analysis.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ASR_AUDIO_DUMP` | `false` | Capture incoming user audio (per turn) |
| `ENABLE_TTS_AUDIO_DUMP` | `false` | Capture outgoing synthesized audio (per turn) |
| `AUDIO_DUMP_PATH` | `audio_dumps` | Output directory (relative to project root, or absolute) |

To enable the audio recorder, set the environment variables in the `.env` file:

```bash
ENABLE_ASR_AUDIO_DUMP=true
ENABLE_TTS_AUDIO_DUMP=true
AUDIO_DUMP_PATH=audio_dumps
```

The shipped examples already create and wire the shared recorder, so the `.env` settings are enough to enable capture for those examples. To add the recorder to a **new custom example**, mirror [`src/examples/generic/pipeline.py`](../../src/examples/generic/pipeline.py) with three changes to your `pipeline.py`:

1. Import the helper:

    ```python
    from examples.shared.audio_recorder import create_audio_recorder
    ```

2. Create the recorder and add it to the pipeline. `create_audio_recorder()` returns `None` when both ASR and TTS dump flags are off, so it stays a no-op until enabled:

    ```python
    audio_recorder = create_audio_recorder()

    pipeline = Pipeline([
        transport.input(),
        # ... ASR, LLM, TTS, transport.output() ...
        *([audio_recorder] if audio_recorder else []),
    ])
    ```

3. Start it once the client connects (for example, in your `on_client_connected` handler):

    ```python
    if audio_recorder:
        await audio_recorder.start_recording()
    ```

With those in place, the `ENABLE_ASR_AUDIO_DUMP` / `ENABLE_TTS_AUDIO_DUMP` / `AUDIO_DUMP_PATH` settings above control capture for your custom example.

## Output Format

Files are saved as 16-bit mono PCM WAV with per-turn indexing:

```text
audio_dumps/
├── asr_<stream_id>_000.wav   # User turn 0
├── asr_<stream_id>_001.wav   # User turn 1
├── tts_<stream_id>_000.wav   # Bot turn 0
├── tts_<stream_id>_001.wav   # Bot turn 1
└── ...
```

The `<stream_id>` is a unique 8-character hex ID per session, so files from concurrent sessions do not collide.

> **Note:** For host-native runs, files are written under the project root by default. For Docker Compose runs, mount a host directory to `/app/audio_dumps` if you want the files to appear on the host, or copy them out of the app container after capture. If Docker creates the folder with different permissions, fix ownership:
>
> ```bash
> # Option 1: Pre-create directory before container start
> mkdir -p ./audio_dumps
> # Option 2: Fix ownership after container creates it
> sudo chown -R $(id -u):$(id -g) ./audio_dumps
> ```

> **Warning:** Disable the audio recorder in production to prevent disk exhaustion.
