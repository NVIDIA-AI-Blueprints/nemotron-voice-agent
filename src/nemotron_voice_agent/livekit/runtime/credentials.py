# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate and share the LiveKit API credentials, so a deployment needs no manual key setup.

The `livekit-init` Compose service runs this module once. It writes a random key and secret (or the `LIVEKIT_API_KEY`
and `LIVEKIT_API_SECRET` you set) and the LiveKit server config to a state volume. The LiveKit server loads the
config. The agent and the UI server read the credentials from the same volume. Standard library only.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from . import telephony

DEFAULT_STATE_DIR = Path("/state")
CREDENTIALS_FILE = "credentials.json"
SERVER_CONFIG_FILE = "livekit.yaml"


def _write_private(path: Path, text: str) -> None:
    """Write a file readable only by its owner."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)


def _write_shared(path: Path, text: str) -> None:
    """Write a config for a third-party container that may run as a non-root user.

    The file holds a secret, but it lives in a Docker volume that only root on the host can open.
    """
    _write_private(path, text)
    path.chmod(0o644)


def server_config(api_key: str, api_secret: str, redis: bool = False) -> str:
    """Return the LiveKit server config with the key pair. Values are JSON-quoted, which is valid YAML.

    With `redis`, the server also uses the optional Redis container, which the SIP service needs to find it.
    """
    text = (
        "port: 7880\n"
        "rtc:\n"
        "  tcp_port: 7881\n"
        "  udp_port: 7882\n"
        "  use_external_ip: false\n"
        "keys:\n"
        f"  {json.dumps(api_key)}: {json.dumps(api_secret)}\n"
    )
    if redis:
        text += f"redis:\n  address: livekit-redis:{telephony.REDIS_PORT}\n"
    return text


def _read_credentials(path: Path) -> dict[str, str] | None:
    """Return the stored key pair, or None when the file is missing or damaged (a new pair is then generated)."""
    try:
        creds = json.loads(path.read_text())
        pair = {"api_key": str(creds["api_key"]), "api_secret": str(creds["api_secret"])}
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return pair if all(pair.values()) else None  # an empty key or secret would lock LiveKit out, so replace it


def truthy(value: str | None) -> bool:
    """Whether an environment flag is on: `1`, `true`, `yes`, or `on`."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _write_overlays(state_dir: Path, creds: Mapping[str, str], env: Mapping[str, str]) -> None:
    """Write (or remove) the config of the optional SIP overlay, so a stale file never outlives its flag."""
    sip_path = state_dir / telephony.SIP_CONFIG_FILE
    if truthy(env.get("LIVEKIT_SIP")):
        _write_shared(sip_path, telephony.sip_config_yaml(creds["api_key"], creds["api_secret"]))
    else:
        sip_path.unlink(missing_ok=True)


def load_or_create(state_dir: Path = DEFAULT_STATE_DIR, env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Write the credentials and the server config to `state_dir`, and return the key pair.

    `LIVEKIT_SIP` (set by `run.sh up --sip`) also writes the SIP config and adds Redis to the server config.

    Precedence: both `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` in the environment (setting only one is an error), then
    credentials from an earlier run, then a newly generated pair. A damaged credentials file counts as missing.
    """
    env = os.environ if env is None else env
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / CREDENTIALS_FILE
    given = [name for name in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if env.get(name)]
    if len(given) == 1:
        raise ValueError(f"Set both LIVEKIT_API_KEY and LIVEKIT_API_SECRET, or neither. Only {given[0]} is set.")
    if given:
        creds = {"api_key": env["LIVEKIT_API_KEY"], "api_secret": env["LIVEKIT_API_SECRET"]}
    else:
        creds = _read_credentials(path) or {
            "api_key": "lk" + secrets.token_hex(6),
            "api_secret": secrets.token_hex(24),
        }
    _write_private(path, json.dumps(creds))
    _write_private(
        state_dir / SERVER_CONFIG_FILE,
        server_config(creds["api_key"], creds["api_secret"], redis=truthy(env.get("LIVEKIT_SIP"))),
    )
    _write_overlays(state_dir, creds, env)
    return creds


def apply_to_environ(state_dir: Path | None = None, environ: MutableMapping[str, str] | None = None) -> None:
    """Set `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET` from the state volume unless the environment already has them."""
    environ = os.environ if environ is None else environ
    if environ.get("LIVEKIT_API_KEY") and environ.get("LIVEKIT_API_SECRET"):
        return
    state_dir = state_dir or Path(environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR))
    path = state_dir / CREDENTIALS_FILE
    if not path.exists():
        raise RuntimeError(f"LiveKit credentials not found at {path}. Start the livekit-init service first.")
    creds = _read_credentials(path)
    if creds is None:  # damaged or empty: livekit-init replaces such a file the next time it runs
        raise RuntimeError(f"LiveKit credentials in {path} are empty or damaged. Run the livekit-init service again.")
    environ["LIVEKIT_API_KEY"] = creds["api_key"]
    environ["LIVEKIT_API_SECRET"] = creds["api_secret"]


def main() -> None:
    """Create the credentials. Never prints the secret."""
    state_dir = Path(os.environ.get("LIVEKIT_STATE_DIR", DEFAULT_STATE_DIR))
    creds = load_or_create(state_dir)
    print(f"LiveKit credentials ready in {state_dir} (key {creds['api_key']})", flush=True)


if __name__ == "__main__":
    main()
