# bot.workstation-runner | workstation/DGX skeleton

Local NIM SSH/remote /client. Checklists dgx-spark first-try-webrtc. pipecat-docs before paste. Transport patterns→networking/transport-selection.md.

REQ local vLLM:
```python
class WorkstationNvidiaLLMService(NvidiaLLMService):
    supports_developer_role = False
```
Greeting role user not developer. Discover LLM GET /v1/models 8002 then 18000.

```python
import json, os
from urllib.request import urlopen
from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.nvidia.llm import NvidiaLLMService
from pipecat.services.nvidia.stt import NvidiaSTTService
from pipecat.services.nvidia.tts import NvidiaTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import TransportParams

load_dotenv(override=True)
DEPLOYMENT_PLATFORM = "workstation"
LOCAL_HARDWARE_PROFILE = os.getenv("LOCAL_HARDWARE_PROFILE", "default")
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "cascaded")  # cascaded|omni from gate
# TRANSPORT_MODE = "webrtc" | "websocket" | "both"  # from gate; drives transport_params + run -t

def _discover_local_llm() -> tuple[str, str]:
    override = os.getenv("LLM_BASE_URL")
    if override:
        candidates = [override]
    elif PIPELINE_MODE == "omni":
        candidates = ["http://127.0.0.1:8002/v1"]
    else:
        candidates = ["http://127.0.0.1:18000/v1"]
    for base in candidates:
        try:
            with urlopen(f"{base.rstrip('/')}/models", timeout=1.5) as r:
                data = json.loads(r.read().decode())
            if data.get("data"): return base, data["data"][0]["id"]
        except Exception: continue
    raise RuntimeError(f"No LLM for {PIPELINE_MODE} — cascaded :18000, omni :8002; curl /v1/models")

class WorkstationNvidiaLLMService(NvidiaLLMService):
    supports_developer_role = False

LLM_BASE_URL, LLM_MODEL = _discover_local_llm()
ASR_SERVER = os.getenv("ASR_SERVER", "127.0.0.1:50152")
TTS_SERVER = os.getenv("TTS_SERVER", "127.0.0.1:50151")
TTS_MODEL = os.getenv("TTS_MODEL", "magpie-tts-multilingual")
TTS_VOICE = os.getenv("TTS_VOICE", "Magpie-Multilingual.EN-US.Aria")

# WebRTC only:
# transport_params = {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
# WebSocket only:
# transport_params = {"websocket": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)}
# Both (recommended — user switches in Pipecat /client UI; run bot.py without -t):
transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    "websocket": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    stt = NvidiaSTTService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        server=ASR_SERVER,
        settings=NvidiaSTTService.Settings(language=Language.EN_US),
    )
    llm = WorkstationNvidiaLLMService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        base_url=LLM_BASE_URL,
        settings=NvidiaLLMService.Settings(model=LLM_MODEL),
    )
    tts = NvidiaTTSService(
        api_key=os.getenv("NVIDIA_API_KEY"),
        server=TTS_SERVER,
        settings=NvidiaTTSService.Settings(model=TTS_MODEL, voice=TTS_VOICE, language=Language.EN_US),
    )

if __name__ == "__main__":
    transport_mode = os.getenv("TRANSPORT_MODE", "both")
    if transport_mode in {"webrtc", "both"}:
        from turn_support import apply_turn_patches
        apply_turn_patches()
    from pipecat.runner.run import main
    main()
```

Ship: pyproject per transport-selection.md | turn_support+docker-compose.turn.yml if WebRTC or Both+remote | .env NVIDIA_API_KEY TURN_*

Run:
- WebRTC: `uv run python bot.py --host 127.0.0.1 -t webrtc`
- WebSocket: `uv run python bot.py --host 127.0.0.1 -t websocket`
- Both: `uv run python bot.py --host 127.0.0.1` (no `-t`; `/client` picks transport)
