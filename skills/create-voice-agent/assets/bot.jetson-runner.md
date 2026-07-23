# bot.jetson-runner | Jetson constants only

Edge vLLM+Riva. Checklist jetson-thor.md. pipecat-docs before paste.
Shared skeleton transport WorkstationNvidiaLLMService turn_support pyproject→bot.workstation-runner.md. This file=Jetson constants.

```python
import os

DEPLOYMENT_PLATFORM="jetson"
LOCAL_HARDWARE_PROFILE=os.getenv("LOCAL_HARDWARE_PROFILE","jetson_thor")
# LLM http://127.0.0.1:18000/v1 discover /v1/models
# ASR 127.0.0.1:50052 TTS 127.0.0.1:50051 function_id empty
```

Jetson: TTS Settings(voice=...) not voice_id= | reasoning extra_body llm-reasoning.md | TURN: `export COTURN_IMAGE=coturn/coturn:4.6.2-r8 VM_IP=LAN IP` then docker-compose.turn.yml (x86 compose default instrumentisto/coturn:4) | ASR probe log if :50052 unreachable | run per transport-selection.md (default Both: omit `-t`)

Ship: jetson compose HF_TOKEN RIVA_* paths pyproject per transport-selection.md. Handoff ssh.md Mac remote if browser Mac.
