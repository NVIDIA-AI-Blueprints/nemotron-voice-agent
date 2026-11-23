#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Download the paired NeMo-Speech.cpp Transcribe GGUF files from Hugging Face.
set -euo pipefail

usage() {
  cat <<USAGE
Usage: $0 [DEST=models/transcribe] [auto|nvfp4|iq4_nl]
Select NVFP4 for compute capability 10.x, 11.x, or 12.x, otherwise IQ4_NL.
DEST defaults to TRANSCRIBE_MODEL_LOC or the repository's models/transcribe.
USAGE
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
if [[ $# -gt 2 ]]; then
  usage >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Do not run this script as root/sudo." >&2
  echo "The model files would be owned by root. Re-run as your user." >&2
  exit 1
fi

if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
DEST="${1:-${TRANSCRIBE_MODEL_LOC:-${ROOT}/models/transcribe}}"
DEST="$(python3 -c 'import os, sys; print(os.path.realpath(os.path.expanduser(sys.argv[1])))' "${DEST}")"

# The script changes permissions on DEST, so refuse paths that hold more than the model files.
case "${DEST}" in
  / | "${HOME}" | "${HOME}/.cache" | "${ROOT}" | "${ROOT}/models")
    echo "Refusing unsafe model destination: ${DEST}" >&2
    echo "Choose a dedicated model directory." >&2
    exit 1
    ;;
esac

if [[ "$(dirname "${DEST}")" == "/" ]]; then
  echo "Refusing top-level model destination: ${DEST}" >&2
  echo "Choose a dedicated model directory below a user-managed path." >&2
  exit 1
fi

PROFILE="${2:-auto}"
if [[ "${PROFILE}" == auto ]]; then
  COMPUTE_CAP="$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true)"
  case "${COMPUTE_CAP//[[:space:]]/}" in
    10.* | 11.* | 12.*) PROFILE=nvfp4 ;;
    *) PROFILE=iq4_nl ;;
  esac
fi
case "${PROFILE}" in
  nvfp4 | iq4_nl) ;;
  *) echo "Invalid quantization: ${PROFILE}. Choose auto, nvfp4, or iq4_nl." >&2; exit 2 ;;
esac

echo "Transcribe decoder quantization: ${PROFILE^^}"
REPO=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
REVISION=cbd72f20b96fab06e8106dba70408210fd789794
BASE_URL="https://huggingface.co/${REPO}/resolve/${REVISION}"
MODEL=NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
DECODER="${MODEL}-${PROFILE^^}.gguf"
ENCODER="${MODEL}-Encoder-F16.gguf"
# SHA-256 of each file at the pinned revision.
declare -A SHA256=(
  ["${MODEL}-NVFP4.gguf"]=eb8c9b338003347b527fbe2e249d4ca6fe9bccac289b6105e55b6a6b08a9c4c6
  ["${MODEL}-IQ4_NL.gguf"]=001bea92194596abe971528cf903efe2970f766ac809e4b46381d8567e2dfa30
  ["${ENCODER}"]=98b97e3b591a268229762fed39b7cbb26a322aa89a15f0bbee49d340bbdf6fe1
)
mkdir -p "${DEST}"

install_file() {
  local name="$1" target="${DEST}/$2" checksum="${SHA256[$1]}" actual
  if [[ -f "${target}" ]]; then
    actual="$(sha256sum "${target}" | awk '{print $1}')"
    if [[ "${actual}" == "${checksum}" ]]; then
      return
    fi
  fi
  local curl_args=(--fail --location --retry 3 --connect-timeout 30 --silent --show-error)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    # Send the token on stdin, not as an argument, so other users cannot read it from the process list.
    printf 'Authorization: Bearer %s\n' "${HF_TOKEN}" |
      curl "${curl_args[@]}" --header @- --output "${target}.part" "${BASE_URL}/${name}"
  else
    curl "${curl_args[@]}" --output "${target}.part" "${BASE_URL}/${name}"
  fi
  actual="$(sha256sum "${target}.part" | awk '{print $1}')"
  if [[ "${actual}" != "${checksum}" ]]; then
    rm -f "${target}.part"
    echo "SHA-256 mismatch for ${name}. Existing file was preserved." >&2
    exit 1
  fi
  mv -f "${target}.part" "${target}"
}

install_file "${DECODER}" transcribe.gguf
install_file "${ENCODER}" "${ENCODER}"
# The container runs as root and reads /models. Open only the directory and the two model files.
chmod a+rx "${DEST}"
chmod a+r "${DEST}/transcribe.gguf" "${DEST}/${ENCODER}"
echo "Models ready in ${DEST}. Mount this directory read-only at /models."
