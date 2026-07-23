# Speech NIM readiness | verify before pull

Verify host before speech NIM images. Install→setup.md no duplicate install. Announce Step N/6. Fetch per-release driver compute glibc OS WSL2 VRAM setup.md §docs.

Prereq: Linux x86_64 nvidia-smi Docker+Toolkit setup steps 1-3.

Checks:

| check | cmd |
| --- | --- |
| x86_64 | uname -m |
| driver | nvidia-smi vs prerequisites |
| compute | nvidia-smi --query-gpu=compute_cap --format=csv |
| VRAM | nvidia-smi memory.total vs matrix |
| glibc | ldd --version or getconf GNU_LIBC_VERSION vs prerequisites |
| docker | docker info |
| toolkit | docker run --rm --gpus all ubuntu nvidia-smi |
| NGC | NGC_API_KEY set nvcr.io login |
| NVAIE | self-host required |

6-step:

```bash
uname -m; nvidia-smi; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
df -BG --output=avail / | tail -1
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
[ -n "$NGC_API_KEY" ]&&echo set||echo NOT
NIM_TAG="${NIM_TAGS_SELECTOR:-latest}"
CONTAINER_ID="${CONTAINER_ID:?Set CONTAINER_ID from support matrix}"
pull_ok=0
docker pull "nvcr.io/nim/nvidia/${CONTAINER_ID}:${NIM_TAG}" || pull_ok=$?
echo "pull_exit=$pull_ok"
[ "$pull_ok" -eq 0 ] || exit 1
```

Health poll (bounded, structured ready check — default ASR HTTP port **9001**; override with `ASR_HTTP_PORT`):

```bash
ASR_HTTP_PORT="${ASR_HTTP_PORT:-9001}"
ready=0
for i in $(seq 1 30); do
  if curl -sf --max-time 5 "http://127.0.0.1:${ASR_HTTP_PORT}/v1/health/ready" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get("status")=="ready" or d.get("ready") is True else 1)'; then
    ready=1
    break
  fi
  sleep 10
done
[ "$ready" -eq 1 ] || exit 1
```

HTTP alone insufficient pre-existing container→gRPC Get*Config probe asr/tts/nmt. Full deploy→run.md.

GPU compat: matrix row vs nvidia-smi name,memory,compute_cap.

Failures: bad NIM_TAGS_SELECTOR→matrix tags | pull 403→login NGC | download slow→LOCAL_NIM_CACHE volume | nvidia-smi missing container→toolkit | health 503→wait 10-30m | OOM→lower profile | gRPC refused→health poll port map | HTTP 404→/v1/health/ready path

WSL2: Podman subset .wslconfig memory OOM fetch prerequisites.
