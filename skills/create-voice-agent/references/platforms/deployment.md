# Deployment platform

Same pipeline single py — change PIPELINE_MODE,DEPLOYMENT_PLATFORM,endpoints.

```python
PIPELINE_MODE="cascaded"  # cascaded|omni Pipecat only
DEPLOYMENT_PLATFORM="cloud"  # cloud|workstation|jetson
LOCAL_HARDWARE_PROFILE="default"  # dgx_spark|jetson_thor
```

| platform | cascaded LLM | omni LLM | ASR/TTS | function_id |
| --- | --- | --- | --- | --- |
| cloud | integrate.api.nvidia.com/v1 | same omni id | grpc.nvcf.nvidia.com:443 | build cards |
| workstation/DGX | :18000/v1 | :8002/v1 vLLM | :50152/:50151 | empty |
| jetson | :18000/v1 | :8002/v1 Thor only | Riva 50052/50051 | empty |

Omni: no ASR docker-compose.nim.omni.yml pipeline/omni.md | Cascaded: docker-compose.nim.workstation.yml nim-llm-profiles after pick

Gate deployment row: Cloud NVCF | local NIM | Jetson edge. **Probe-first:** hardware-probe §Step0 before deployment ask; no_gpu→cloud auto. **Fit-first rec:** GPU fits full stack (hardware-probe §Step0b)→**Local recommended, Cloud second**; stack won't fit→Cloud recommended. Local: GPU+Container Toolkit+NVIDIA_API_KEY nvcr.io. DGX→dgx-spark Jetson→jetson-thor. Secrets gate→clarifying/env-secrets.md

Cloud: no compose catalog+GET /v1/models+function_id in py constants.

Workstation ship: bot.py compose by PIPELINE_MODE cascaded nim.workstation.yml omni nim.omni.yml+HF_TOKEN | turn_support+turn compose if WebRTC remote | first-try before scaffold
DGX: LOCAL_HARDWARE_PROFILE=dgx_spark | Jetson: DEPLOYMENT_PLATFORM=jetson HF_TOKEN riva_init once
FORBID model ids in .env. Start/health/stop→run.md.

## Local selective slot reuse (post model-confirm)

After the model table is **confirmed** and before scaffold or NIM bring-up on local workstation/DGX, probe **each slot independently** (LLM, ASR, TTS; omni skips ASR). Keep healthy containers that already match the chosen models; **replace only slots that are missing, unhealthy, or running a different model**.

```
PROC: post-confirm → per-slot probe (LLM, ASR, TTS) → keep matches → replace mismatches only → verify run.md
FORBID: docker compose down | full-stack restart | start_nim_stack.sh when only one slot mismatches
REQ: disclose per slot — "reusing running LLM" or "replacing ASR (wrong image/model)"
```

### Per-slot probe (workstation/DGX)
| slot | ports | match checks |
| --- | --- | --- |
| LLM | :18000 cascaded; :8002 omni | `docker ps` image == compose `NVIDIA_LLM_IMAGE`; profile `NIM_MODEL_PROFILE` if set; chosen LLM id **in** `curl -sf …/v1/models` `data[].id` list |
| ASR | :50152 gRPC, :9001 HTTP | image == `ASR_DOCKER_IMAGE`; `NIM_TAGS_SELECTOR` == `ASR_NIM_TAGS`; `curl -sf :9001/v1/health/ready` |
| TTS | :50151 gRPC, :9000 HTTP | image == `TTS_DOCKER_IMAGE`; `NIM_TAGS_SELECTOR` == `TTS_NIM_TAGS`; `curl -sf :9000/v1/health/ready` |

```bash
docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}'
curl -sf http://127.0.0.1:18000/v1/models | head -c600   # cascaded LLM
curl -sf http://127.0.0.1:8002/v1/models | head -c600    # omni LLM
curl -sf http://127.0.0.1:9001/v1/health/ready           # ASR
curl -sf http://127.0.0.1:9000/v1/health/ready           # TTS
```

HTTP health alone can be insufficient for speech slots — if image/tags ambiguous, gRPC `GetRivaSpeechConfig` / `GetRivaSynthesisConfig` per nemotron-speech/deployment-readiness-checks.md.

### Slot decision
| probe | action |
| --- | --- |
| image + tags + model id match and healthy | **keep** — do not stop/recreate |
| no container, unhealthy, or wrong image/model id | **replace this slot only** |

### Replace one slot (example: wrong ASR, LLM+TTS OK)
```bash
docker compose -f docker-compose.nim.workstation.yml stop asr-service
docker compose -f docker-compose.nim.workstation.yml rm -f asr-service
docker compose -f docker-compose.nim.workstation.yml --profile full up -d asr-service
# poll :9001 ready; leave nvidia-llm + tts-service running
```

Missing slot only: start the specific service — `docker compose -f docker-compose.nim.workstation.yml up -d nvidia-llm` | `... --profile full up -d tts-service` | `... --profile full up -d asr-service` — without touching other services. All slots missing → `run.md` sequential start (`start_nim_stack.sh` or LLM→TTS→ASR). Jetson Riva :50052/:50051 — same per-slot logic; REF jetson-thor.md.

Local id MUST:
```bash
curl -s http://127.0.0.1:18000/v1/models  # cascaded LLM
curl -s http://127.0.0.1:8002/v1/models   # omni
```
Set LLM_MODEL/OMNI_MODEL_ID from response only.

Branch cascaded:
```python
if DEPLOYMENT_PLATFORM=="cloud":
    LLM_BASE_URL="https://integrate.api.nvidia.com/v1"; ASR_SERVER=TTS_SERVER="grpc.nvcf.nvidia.com:443"; ASR_FUNCTION_ID="<card>"
elif DEPLOYMENT_PLATFORM=="jetson":
    LLM_BASE_URL="http://127.0.0.1:18000/v1"; ASR_SERVER,TTS_SERVER="127.0.0.1:50052","127.0.0.1:50051"; ASR_FUNCTION_ID=TTS_FUNCTION_ID=""
else:
    LLM_BASE_URL="http://127.0.0.1:18000/v1"; ASR_SERVER,TTS_SERVER="127.0.0.1:50152","127.0.0.1:50151"; ASR_FUNCTION_ID=TTS_FUNCTION_ID=""
```
Omni: OMNI_BASE_URL no ASR→bot.omni-runner.md. Verify/stop→run.md
