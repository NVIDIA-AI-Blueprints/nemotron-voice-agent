# Jetson Thor runbook

DEPLOYMENT_PLATFORM=jetson LOCAL_HARDWARE_PROFILE=jetson_thor. REF: ssh,webrtc-turn,bot.jetson-runner,run,common-voice-debug,first-try,llm-reasoning.

Detect: `cat /proc/device-tree/model|tr -d '\0'; nvidia-smi --query-gpu=name; uname -m` Jetson Thor/aarch64→jetson. GB10/DGX→dgx-spark.md.

## Jetson debug (shared→common-voice-debug)
| symptom | fix |
| --- | --- |
| text OK no voice | Riva ASR :50052 |
| vLLM exits | HF_TOKEN |
| Riva restart | riva_init once RIVA_* paths |
| audio glitches | start-mps.sh |
| x86 pull | jetson compose template |
| TTS silent | voice= not voice_id= |
| LLM 400 kwargs | extra_body nest llm-reasoning |
| TTS no spaces | no .strip() LLMTextFrame |
| coturn 255 | coturn/coturn:4.6.2-r8 |

Riva init once:
```bash
ngc registry resource download-version nvidia/riva/riva_quickstart_arm64:2.24.0
cd riva_quickstart_arm64_v2.24.0 && bash riva_init.sh
```
Pre-flight:
```bash
docker ps
curl -sf http://127.0.0.1:18000/v1/models >/dev/null && echo "LLM ok" || echo "LLM down"
ss -ltn | grep -E ':50052|:50051' || true
test -n "${NVIDIA_API_KEY:-}" && test -n "${HF_TOKEN:-}" && echo "keys set" || echo "keys missing"
hostname -I
```
FORBID handoff if :50052 down.

Workflow: detect→riva confirm→pre-flight→post-confirm per-slot probe (keep matching LLM/ASR/TTS; replace mismatches only — deployment.md §selective)→optional MPS→compose up only needed services→scaffold→uv sync→coturn if Mac→bot --host 127.0.0.1 -t webrtc→verify run+first-try.

Ports: LLM :18000 | ASR :50052 | TTS :50051. Skeleton bot.jetson-runner.md.
Defaults: parakeet Riva ASR; smallest LLM :18000; magpie Riva TTS Aria. Unified memory: MPS lower max-model-len if OOM.
Handoff: vLLM+ASR+TTS ports | LLM_MODEL match | first-try if Mac | HF_TOKEN+riva noted.
Ship: bot.py,turn_support,pyproject,.env.example,jetson compose,turn compose.
