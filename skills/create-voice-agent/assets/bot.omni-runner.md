# bot.omni-runner.md — Pipecat omni pipeline skeleton

`PIPELINE_MODE=omni`. Query pipecat-docs before implement. Omni rules → [pipeline/omni.md](../references/pipeline/omni.md).

## REQ: copy reference modules first (do not reimplement from scratch)

Copy from skill `assets/` into project root **before** writing `bot.py`:

| asset | project file |
| --- | --- |
| `assets/omni_service.py` | `omni_service.py` |
| `assets/audio_only_smart_turn.py` | `audio_only_smart_turn.py` |
| `assets/turn_support.py` | `turn_support.py` (WebRTC; required workstation/SSH remote) |

FORBID: subclass `NvidiaLLMService` only; stock Smart Turn stop without `AudioOnlySmartTurnStopStrategy`; omit `omni_service.py`.

## Pipeline order

```text
transport.input()
  → user_aggregator
  → omni (NvidiaOmniMultimodalService)
  → tts (NvidiaTTSService)
  → transport.output()
  → assistant_aggregator
```

NO `NvidiaSTTService`. NO STT between transport and user_aggregator.

## user_aggregator / assistant_aggregator

`LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams(...))`:

- **user_aggregator**: VAD (`SileroVADAnalyzer`), **`MuteUntilFirstBotCompleteUserMuteStrategy`** (REQ WebRTC omni — blocks ambient mic race before greeting finishes), turn strategies
- **assistant_aggregator**: commits assistant text from TTS path into shared `LLMContext`

```python
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy

user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
        user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[AudioOnlySmartTurnStopStrategy()],
        ),
    ),
)
```

## Turn strategies (required for omni)

Stock Smart Turn **stop** waits for `TranscriptionFrame` from STT. Omni has no STT → wire **`AudioOnlySmartTurnStopStrategy`** (from copied `audio_only_smart_turn.py`):

```python
UserTurnStrategies(
    start=[VADUserTurnStartStrategy()],
    stop=[AudioOnlySmartTurnStopStrategy()],
)
```

## TTS

`NvidiaTTSService`, `NvidiaTTSService.Settings(voice=..., language=Language.EN_US)`. Cloud: `grpc.nvcf.nvidia.com:443` + `model_function_map`. Local: `127.0.0.1:50151`.

## Greeting on connect (REQ — not LLMRunFrame)

Omni is **audio-in + JSON** (`{"transcript":"...","response":"..."}`). A text-only `LLMRunFrame` greeting is unreliable on cloud omni (empty/garbage like `"27\n30"`). WebRTC `client-ready` also starts mic streaming immediately — ambient noise can trigger an **audio omni turn** that wins the race before any greeting.

**Use `TTSSpeakFrame`**, not `LLMRunFrame`, for the connect greeting:

```python
from pipecat.frames.frames import TTSSpeakFrame

GREETING_TEXT = (
    "Hi, I'm your voice assistant. I can answer questions and have a conversation with you. "
    "How can I help?"
)

@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    context.add_message({"role": "assistant", "content": GREETING_TEXT})
    await task.queue_frames([TTSSpeakFrame(GREETING_TEXT)])
```

- Add greeting to context as **assistant** (not fake user “Please introduce yourself.”)
- Pair with **`MuteUntilFirstBotCompleteUserMuteStrategy`** above so mic noise cannot fire omni before greeting TTS completes
- `on_client_connected` (Pipecat WebRTC); domain bots may use `on_client_ready` after `set_bot_ready` — see `bot.domain.md`

### Greeting anti-patterns (FORBID)

- `LLMRunFrame` + fake user “Please introduce yourself” as omni connect greeting
- Relying on text-only omni for first utterance when pipeline is audio-in + JSON
- Skipping `MuteUntilFirstBotCompleteUserMuteStrategy` on WebRTC omni bots

## bot.py `__main__`

```python
if __name__ == "__main__":
    from turn_support import apply_turn_patches
    apply_turn_patches()
    from pipecat.runner.run import main
    main()
```

## pyproject

Per gate transport (networking/transport-selection.md):

```toml
# WebRTC only:
dependencies = ["pipecat-ai[nvidia,runner,webrtc,silero]>=1.3.0", "openai>=1.0", "python-dotenv", "loguru"]
# WebSocket only:
dependencies = ["pipecat-ai[nvidia,runner,websocket,silero]>=1.3.0", "openai>=1.0", "python-dotenv", "loguru"]
# Both (recommended):
dependencies = ["pipecat-ai[nvidia,runner,webrtc,websocket,silero]>=1.3.0", "openai>=1.0", "python-dotenv", "loguru"]
```

## Also ship

- `.env.example`: `NVIDIA_API_KEY` and `HF_TOKEN` placeholders only (model ids in `bot.py`, not `.env`)
- Local: `docker-compose.nim.omni.yml` from assets (not cascaded workstation compose); local omni needs `HF_TOKEN`
- WebRTC or Both + remote SSH: `turn_support.py` + `docker-compose.turn.yml`

## Constants in bot.py (not .env)

```python
PIPELINE_MODE = "omni"
DEPLOYMENT_PLATFORM = "cloud" | "workstation" | "jetson"
TRANSPORT_MODE = "webrtc" | "websocket" | "both"  # gate
OMNI_BASE_URL = "https://integrate.api.nvidia.com/v1"  # or http://127.0.0.1:8002/v1
```

```python
# Both (default recommended):
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "websocket": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}
```

## Run

- WebRTC: `uv sync && uv run python bot.py --host ... -t webrtc`
- WebSocket: `uv run python bot.py --host ... -t websocket`
- Both: `uv run python bot.py --host ...` (no `-t`)

## Verify (omni-specific)

```bash
# local vLLM only
curl -sf http://127.0.0.1:8002/health
curl -sf http://127.0.0.1:8002/v1/models
curl -sf http://127.0.0.1:9000/v1/health/ready   # TTS
# NO ASR health on :9001 for omni compose
```

Handoff: state `PIPELINE_MODE=omni`; no ASR slot; omni endpoint + model id disclosed.
