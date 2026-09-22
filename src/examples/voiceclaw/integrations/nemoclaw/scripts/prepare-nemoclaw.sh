#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
integration_dir="$(cd -- "${script_dir}/.." && pwd)"
example_dir="$(cd -- "${integration_dir}/../.." && pwd)"
version_file="${integration_dir}/compat/version.env"
patch_file="${integration_dir}/compat/0001-response-only-gateway-compatibility.patch"
runtime_root="${VOICECLAW_RUNTIME_ROOT:-${example_dir}/.runtime}"

# shellcheck source=/dev/null
source "${version_file}"

checkout="${1:-${runtime_root}/nemoclaw}"
mkdir -p -- "$(dirname -- "${checkout}")"
fresh_checkout=false

if [[ ! -d "${checkout}/.git" ]]; then
  if [[ -e "${checkout}" ]]; then
    echo "NemoClaw checkout path exists but is not a Git checkout: ${checkout}" >&2
    exit 1
  fi
  git clone --filter=blob:none --no-checkout --no-tags "${NEMOCLAW_REPOSITORY}" "${checkout}"
  fresh_checkout=true
fi

actual_origin="$(git -C "${checkout}" remote get-url origin)"
if [[ "${actual_origin}" != "${NEMOCLAW_REPOSITORY}" ]]; then
  echo "Unexpected NemoClaw origin: ${actual_origin}" >&2
  exit 1
fi

if [[ "$(git -C "${checkout}" rev-parse --is-shallow-repository)" == "true" ]]; then
  git -C "${checkout}" fetch --unshallow --no-tags origin
fi
git -C "${checkout}" fetch --no-tags origin "${NEMOCLAW_REVISION}"
git -C "${checkout}" fetch --no-tags origin \
  "refs/tags/${NEMOCLAW_BASE_TAG}:refs/tags/${NEMOCLAW_BASE_TAG}"

base_tag_revision="$(git -C "${checkout}" rev-parse "refs/tags/${NEMOCLAW_BASE_TAG}^{}")"
if [[ "${base_tag_revision}" != "${NEMOCLAW_BASE_TAG_REVISION}" ]]; then
  echo "Pinned NemoClaw base tag does not resolve to its reviewed revision." >&2
  exit 1
fi

current_revision="$(git -C "${checkout}" rev-parse HEAD 2>/dev/null || true)"
if [[ "${fresh_checkout}" == true ]]; then
  git -C "${checkout}" checkout --detach "${NEMOCLAW_REVISION}"
elif [[ "${current_revision}" != "${NEMOCLAW_REVISION}" ]]; then
  if [[ -n "$(git -C "${checkout}" status --porcelain)" ]]; then
    echo "Refusing to replace a modified NemoClaw checkout: ${checkout}" >&2
    exit 1
  fi
  git -C "${checkout}" checkout --detach "${NEMOCLAW_REVISION}"
fi

actual_describe="$(git -C "${checkout}" describe --tags --match 'v*' "${NEMOCLAW_REVISION}")"
if [[ "${actual_describe}" != "${NEMOCLAW_DESCRIBE}" ]]; then
  echo "Pinned NemoClaw revision no longer has the reviewed build description." >&2
  exit 1
fi

if git -C "${checkout}" apply --reverse --check --unidiff-zero "${patch_file}" >/dev/null 2>&1; then
  if [[ -n "$(git -C "${checkout}" ls-files --others --exclude-standard)" ]] || \
    ! git -C "${checkout}" diff HEAD --binary --no-ext-diff --unified=0 \
      --src-prefix=a/ --dst-prefix=b/ | \
      cmp -s - "${patch_file}"; then
    echo "Refusing a NemoClaw checkout whose changes differ from the reviewed compatibility patch." >&2
    exit 1
  fi
else
  if [[ -n "$(git -C "${checkout}" status --porcelain)" ]]; then
    echo "Refusing to patch a NemoClaw checkout with unrelated modifications." >&2
    exit 1
  fi
  git -C "${checkout}" apply --check --unidiff-zero "${patch_file}"
  git -C "${checkout}" apply --unidiff-zero "${patch_file}"
fi

node "${checkout}/scripts/check-node-version.js"
npm_major="$(npm --version | cut -d. -f1)"
if [[ ! "${npm_major}" =~ ^[0-9]+$ ]] || (( npm_major < 10 )); then
  echo "NemoClaw preparation requires npm 10 or newer." >&2
  exit 1
fi
node "${checkout}/scripts/lib/openshell-sdk-install.mts" prepare
npm --prefix "${checkout}" ci \
  --ignore-scripts \
  --prefer-offline \
  --include=dev \
  --include=optional \
  --@nvidia:registry=https://npm.pkg.github.com
node "${checkout}/scripts/lib/openshell-sdk-install.mts" check
npm --prefix "${checkout}" run clean:cli
npm --prefix "${checkout}" run build:cli
node "${script_dir}/nemoclaw-build-receipt.mjs" \
  write "${checkout}" "${NEMOCLAW_REVISION}" "${NEMOCLAW_DESCRIBE}"

printf 'Prepared NemoClaw %s at %s\n' "${NEMOCLAW_DESCRIBE}" "${checkout}"
printf 'Compatibility patch: accepts status frames and retires per-turn gateway sockets\n'
