# Platform Deployment Reference

Use from repository root after `deploy` picks a recipe profile.

## Common Setup

```bash
test -f .env || cp .env.example .env
export NGC_API_KEY="$NVIDIA_API_KEY"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
```

Required `.env` keys:
- All recipes: `NVIDIA_API_KEY`
- Any recipe ending in `/single-gpu`, plus `omni-assistant/server` and `omni-assistant-subagents/server` (local vLLM downloads the model from HF on first run): `HF_TOKEN`

## Server (NIM, recommended for scaling)

Recipes: `generic-assistant/server`, `multilingual-assistant/server`, `omni-assistant/server`, `frontend-backend-agent/server`.

Services depend on the recipe:
- `generic-assistant/server`: `generic-assistant-server`, `nvidia-llm`, `nemotron-asr-streaming-english`, `tts-service`
- `multilingual-assistant/server`: `multilingual-assistant-server`, `nvidia-llm`, `parakeet-rnnt-asr`, `tts-service`
- `omni-assistant/server`: `omni-assistant-server`, `nvidia-llm-vllm-omni`, `tts-service`
- `frontend-backend-agent/server`: `frontend-backend-agent-server`, `booking-server`, `nvidia-llm`, `nemotron-asr-streaming-english`, `tts-service`

Requires enough GPU VRAM for the selected local NIM services. Single-GPU hosts are valid when capacity is sufficient. Multi-GPU hosts may split speech sidecars and LLM across devices. For the user-facing VRAM, memory-knob, and device-placement matrix, see [VRAM & hardware support](../../../docs/how-to/configure-llm.md#vram--hardware-support).

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
docker compose --profile generic-assistant/server up -d
# or: docker compose --profile multilingual-assistant/server up -d
# or: docker compose --profile omni-assistant/server up -d
# or: docker compose --profile frontend-backend-agent/server up -d
```

## Single GPU (workstations, DGX Spark, and Jetson Thor)

All five examples provide a universal `/single-gpu` recipe. These run the NeMo-Speech.cpp speech stack next to vLLM on one GPU. The LLM service detects DGX Spark and Jetson Thor automatically.

Services depend on the recipe:
- `generic-assistant/single-gpu`: `generic-assistant-single-gpu`, `nvidia-llm-vllm-lightning`, `nemo-speech` (ASR + TTS).
- `multilingual-assistant/single-gpu`: `multilingual-assistant-single-gpu`, `nvidia-llm-vllm-lightning`, `nemo-speech-multilingual` (multilingual ASR + TTS).
- `omni-assistant/single-gpu`: `omni-assistant-single-gpu`, `nvidia-llm-vllm-omni`, `nemo-speech-tts` (TTS only; Omni does its own ASR).
- `omni-assistant-subagents/single-gpu`: `omni-assistant-subagents-single-gpu`, `nvidia-llm-vllm-omni`, `nemo-speech-tts`.
- `frontend-backend-agent/single-gpu`: `frontend-backend-agent-single-gpu`, `booking-server`, `nvidia-llm-vllm-lightning`, `nemo-speech`.

One-time speech model setup, from the repo root, as your user (not sudo):

```bash
bash scripts/download-nemo-speech-models.sh
```

The script reads `HF_TOKEN` from `.env`. If `docker compose` already created `models/nemo-speech` as root, the script reclaims ownership automatically.

Complete the mandatory memory-fit procedure in `../SKILL.md` before deploying, then deploy:

```bash
docker compose --profile generic-assistant/single-gpu up -d
# or: docker compose --profile multilingual-assistant/single-gpu up -d
# Omni Assistant (local Omni vLLM + NeMo-Speech.cpp TTS; requires HF_TOKEN):
# docker compose --profile omni-assistant/single-gpu up -d
# docker compose --profile omni-assistant-subagents/single-gpu up -d
# docker compose --profile frontend-backend-agent/single-gpu up -d
```

The same Lightning service detects DGX Spark and enables its DSpark draft model,
detects a Blackwell workstation and enables DFlash, or detects Jetson Thor and
selects CUTLASS without a draft model. Other hosts use the compute-capability
matrix described above.

If a service fails to start on low memory (e.g. `nvidia-llm-vllm` logs `Engine core initialization failed`), reclaim cached memory and retry:

```bash
free -h
sudo sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
docker compose --profile generic-assistant/single-gpu up -d
```

## TURN

Add `--profile turn` when clients connect from outside the host network.
The bundled coturn profile uses the `instrumentisto/coturn` image, which is
supported on x86_64 (`linux/amd64`) only. On arm64 hosts, such as Jetson Thor,
do not enable the bundled `turn` profile; set `TURN_URL`, `TURN_USERNAME`, and
`TURN_PASSWORD` for an externally hosted TURN server instead.

Before starting TURN, ensure `.env` contains TURN credentials. Coturn has
compose defaults, but the app only publishes ICE servers to clients when
`TURN_USERNAME` and `TURN_PASSWORD` are present in `.env`.

```bash
test -f .env || cp .env.example .env
grep -Eq '^TURN_USERNAME=.+$' .env || printf '\nTURN_USERNAME=turn-%s\n' "$(openssl rand -hex 4)" >> .env
grep -Eq '^TURN_PASSWORD=.+$' .env || printf 'TURN_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> .env
```

Set `TURN_URL=turn:<turn-host-or-ip>:3478` when TURN runs on a different host,
or when the host derived from the incoming request is not reachable by clients.
Open UDP `3478` and UDP `49160-49200` from client networks.

```bash
docker compose --profile generic-assistant --profile turn up -d
docker compose --profile generic-assistant/server --profile turn up -d
```

Verify TURN with:

```bash
docker compose ps coturn
# HTTPS by default; if PIPELINE_TLS=false the HTTPS call fails and the HTTP one returns the config
curl -k https://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers \
  || curl http://localhost:${PIPELINE_APP_PORT:-7860}/api/ice-servers
```

## Verify / Stop

```bash
docker compose ps
docker compose logs --tail 200 <service-name>
docker compose down
```
