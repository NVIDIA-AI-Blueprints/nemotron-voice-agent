#!/usr/bin/env bash
# Download NeMo-Speech.cpp GGUF weights for */single-gpu recipes.
#
# Run as your user, never sudo. The script:
#   - reads HF_TOKEN from the repo .env
#   - creates models/nemo-speech (or $NEMO_SPEECH_MODEL_LOC / $1)
#   - if Docker already created that path or ~/.cache/huggingface as root,
#     reclaims ownership with a one-shot container (no sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this script as root/sudo." >&2
  echo "A missing Docker bind-mount is created as root; sudo then hides hf/uvx on your PATH." >&2
  echo "Re-run as your user. This script reclaims root-owned directories automatically." >&2
  exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi

DEST="${1:-${NEMO_SPEECH_MODEL_LOC:-${ROOT}/models/nemo-speech}}"
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"
export PATH="${HOME}/.local/bin:${PATH}"

if command -v hf >/dev/null 2>&1; then
  HF=(hf)
elif command -v uvx >/dev/null 2>&1; then
  HF=(uvx --from huggingface_hub hf)
else
  echo "Neither 'hf' nor 'uvx' is available. Install uv (https://docs.astral.sh/uv/) or the Hugging Face CLI." >&2
  exit 1
fi

# If Docker created a missing bind-mount, the host path is root-owned. Reclaim
# it with the Docker daemon instead of sudo, so hf/uvx stay on the user PATH.
reclaim_if_unwritable() {
  local path="$1"
  local existing="${path}"
  while [[ ! -e "${existing}" && "${existing}" != "/" && "${existing}" != "." ]]; do
    existing="$(dirname "${existing}")"
  done
  if [[ -e "${existing}" && ! -w "${existing}" ]]; then
    if ! command -v docker >/dev/null 2>&1; then
      echo "Cannot write to ${existing} (often root-owned after docker compose created a missing bind-mount)." >&2
      echo "Fix: sudo chown -R $(id -u):$(id -g) '${existing}'" >&2
      exit 1
    fi
    echo "Reclaiming ownership of ${existing} (created as root by Docker)..."
    docker run --rm -v "${existing}:/fix" alpine:latest chown -R "$(id -u):$(id -g)" /fix
  fi
}

reclaim_if_unwritable "${DEST}"
reclaim_if_unwritable "${HF_CACHE}"
mkdir -p "${DEST}/magpie-tts/extracted" "${DEST}/nano-codec" "${HF_CACHE}"

if [[ ! -w "${DEST}" || ! -w "${HF_CACHE}" ]]; then
  echo "Still cannot write to ${DEST} or ${HF_CACHE}." >&2
  exit 1
fi

"${HF[@]}" download nvidia/nemotron-speech-streaming-en-0.6b \
  nemotron-speech-streaming-en-0.6b.q8_0.gguf --local-dir "${DEST}"

"${HF[@]}" download nvidia/nemotron-3.5-asr-streaming-0.6b \
  nemotron-3.5-asr-streaming-0.6b.q8_0.gguf --local-dir "${DEST}"

"${HF[@]}" download nvidia/magpie_tts_multilingual_357m \
  --include magpie_tts_multilingual_357m.v2602.f16.gguf \
  --include magpie_tts_multilingual_357m.nemo \
  --local-dir "${DEST}/magpie-tts"

tar -xf "${DEST}/magpie-tts/magpie_tts_multilingual_357m.nemo" \
  -C "${DEST}/magpie-tts/extracted"

"${HF[@]}" download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
  nemo_nano_codec_22khz_1.89kbps_21.5fps.decoder.f16.gguf \
  --local-dir "${DEST}/nano-codec"

chmod -R a+rX "${DEST}"
echo "Models ready at ${DEST}"
