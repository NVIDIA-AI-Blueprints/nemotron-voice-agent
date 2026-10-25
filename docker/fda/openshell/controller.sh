#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The `fda-hermes` controller of the FDA OpenShell runtime (the default Hermes runtime). It runs the
# FDA image (FDA_IMAGE) as the OpenShell sandbox "fda-hermes", with docker/fda/gateway.sh as its command
# and the gateway port 8790 exposed through the OpenShell gateway. Compose health-checks the sandbox's
# /health through that route. On stop it saves the sandbox's /app/logs to /out (logs/openshell/<time>/
# on the host) and deletes the sandbox.
#
# Inputs: docker/fda/gateway.env (passed into the sandbox key by key, with the values of this container's
# environment, so an eval overlay value such as FDA_GATEWAY_CONFIG=gateway.eval.yaml wins),
# docker/fda/openshell/controller.env, and NVIDIA_API_KEY from .env. The key goes to the OpenShell
# provider only; the sandbox sees a placeholder. CLI commands: OpenShell crates/openshell-cli at the
# release in docker/fda/openshell/VERSION.
set -euo pipefail

: "${NVIDIA_API_KEY:?NVIDIA_API_KEY is not set (.env): Hermes needs it for the backend LLM}"   # presence only
: "${FDA_IMAGE:?FDA_IMAGE is not set (docker/fda/openshell/controller.env)}"
: "${FDA_GATEWAY_CONFIG:?FDA_GATEWAY_CONFIG is not set (docker/fda/gateway.env)}"
: "${OPENSHELL_GATEWAY_ENDPOINT:?OPENSHELL_GATEWAY_ENDPOINT is not set (docker/fda/openshell/controller.env)}"

SANDBOX=fda-hermes
PROFILE=nvidia-fda

kernel=$(uname -r)
[[ "$(printf '%s\n6.2\n' "${kernel%%-*}" | sort -V | head -1)" == 6.2 ]] \
  || { echo "OpenShell needs Linux 6.2+ (Landlock ABI 3); this host runs $kernel" >&2; exit 1; }
echo "fda-hermes (OpenShell runtime): image $FDA_IMAGE, gateway config $FDA_GATEWAY_CONFIG"
echo "OpenShell CLI: $(openshell --version)"

# `openshell status` exits 0 even when the gateway is down, so read the JSON `status` field:
# "connected" means the gRPC health check passed (openshell-cli/src/commands/gateway.rs).
gateway_connected() { openshell status -o json 2>/dev/null | jq -e '.status == "connected"' >/dev/null; }
for _ in $(seq 120); do gateway_connected && break; sleep 1; done
gateway_connected || { echo "OpenShell gateway at $OPENSHELL_GATEWAY_ENDPOINT not connected after 120 s" >&2; exit 1; }

# Provider profile: import creates it once. An existing profile is updated, which needs the current
# resource_version in the file (the server refuses an update without it).
if ! out=$(openshell profile import -f /cfg/nvidia-fda.yaml 2>&1); then
  grep -q "already exists" <<<"$out" || { echo "$out" >&2; exit 1; }
  version=$(openshell profile export "$PROFILE" -o json | jq -er '.resource_version')
  { cat /cfg/nvidia-fda.yaml; echo "resource_version: $version"; } > /tmp/nvidia-fda.yaml
  openshell profile update "$PROFILE" -f /tmp/nvidia-fda.yaml
fi
# Provider: create it, or update it so that a key changed in .env is used on the next start. A bare
# --credential NAME reads the value from this process's environment (it never appears in argv).
openshell provider create --name "$PROFILE" --type "$PROFILE" --credential NVIDIA_API_KEY >/dev/null 2>&1 \
  || openshell provider update "$PROFILE" --credential NVIDIA_API_KEY >/dev/null

save_and_delete() {
  openshell sandbox get "$SANDBOX" >/dev/null 2>&1 || return 0
  openshell sandbox download "$SANDBOX" /app/logs "/out/$(date +%Y%m%d-%H%M%S)" || true
  openshell sandbox delete "$SANDBOX" || true
  # delete can return while cleanup is pending; a create with the same name would then fail
  for _ in $(seq 60); do openshell sandbox get "$SANDBOX" >/dev/null 2>&1 || return 0; sleep 1; done
  echo "sandbox $SANDBOX still exists 60 s after delete" >&2
  return 1
}
trap 'save_and_delete; exit 0' TERM INT        # set before create: a stop during setup still cleans up
save_and_delete                                # a sandbox left over from a crash

env_args=()
while IFS= read -r line || [[ -n "$line" ]]; do  # `|| [[ -n ... ]]` keeps a last line with no newline
  [[ -z "$line" || "$line" == \#* ]] && continue
  env_args+=(--env-from "${line%%=*}")         # the value of this container's environment
done < /cfg-env/gateway.env

# --detach returns once the sandbox is Ready. ENTRYPOINT is replaced by OpenShell, so the command follows --.
openshell sandbox create --name "$SANDBOX" --from "$FDA_IMAGE" \
  --policy /cfg/policy.yaml --provider "$PROFILE" --no-auto-providers \
  --expose 8790 "${env_args[@]}" \
  --restart-policy on-failure --detach --no-tty \
  -- /app/gateway.sh

# Stay up; Compose health-checks the sandbox through the OpenShell gateway. `wait` returns on a signal,
# so the trap runs at once.
while :; do sleep 3600 & wait $!; done
