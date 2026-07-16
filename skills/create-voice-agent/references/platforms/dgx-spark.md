# DGX Spark runbook

DEPLOYMENT_PLATFORM=workstation LOCAL_HARDWARE_PROFILE=dgx_spark. REF: ssh,webrtc-turn,bot.workstation-runner,run,common-voice-debug,first-try.

Detect: `cat /sys/class/dmi/id/product_name; nvidia-smi --query-gpu=name --format=csv,noheader; grep -i DGX /etc/os-release`
DGX Spark/GB10+local→dgx_spark else default workstation.

## DGX-specific debug (shared→common-voice-debug+first-try)
| symptom | fix |
| --- | --- |
| text OK no voice | ASR :50152 down → start asr curl :9001 |
| VAD speaking no reply | STT silent verify ASR gRPC |
| ASR restart loop | bind ./.nim-cache/asr |
| docker LLM+TTS only | start ASR all three required |

Pre-flight: docker ps; curl -sf http://127.0.0.1:8002/v1/models; curl -sf http://127.0.0.1:18000/v1/models; hostname -I for TURN. Health→run.md. FORBID handoff if ASR gRPC down (cascaded).

Workflow: detect→pre-flight→post-confirm per-slot NIM probe (keep LLM/ASR/TTS matches; replace mismatches only)→scaffold bot+turn+pyproject+compose→uv sync→turn compose→bot --host 127.0.0.1 or VM_IP -t webrtc→verify run+first-try→handoff ssh Mac.

Start: per-slot probe first (deployment.md §selective). ASR gap usual — `mkdir -p .nim-cache/asr&&chmod 755 .nim-cache/asr; docker compose -f docker-compose.nim.workstation.yml --profile full up -d asr-service` only if ASR missing/wrong; ASR pull 15-40m poll :9001.

bot.py: WorkstationNvidiaLLMService; discover LLM by `PIPELINE_MODE` — **cascaded: `:18000` only** (FORBID selecting Omni `:8002` sidecar); **omni: `:8002` only**; ASR probe :50152 (cascaded). Hybrid cloud ASR: grpc.nvcf.nvidia.com:443+function_id+SSL.

Defaults post confirm: nemotron-asr :50152; cascaded LLM :18000; omni LLM :8002; magpie :50151 Aria.
Handoff: ASR :50152+:9001 | LLM /v1/models match | POST /start TURN | Mac ssh TPL.
Ship: bot.py,turn_support,pyproject,.env.example,docker-compose.nim.workstation.yml,docker-compose.turn.yml.
