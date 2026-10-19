#!/usr/bin/env python3
"""Rotate the Cisco BYOVA WebSocket datasource JWS and restart the adapter."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NoReturn

DEFAULT_CONFIG = (
    Path.home() / ".config" / "nemotron-voice-agent" / "webex_ws_service_app.json"
)
DEFAULT_SCHEMA_ID = "a38a10b7-43e4-4676-a076-a7d6dce9387d"


def fail(message: str) -> NoReturn:
    """Exit with a concise error suitable for unattended rotation logs."""
    print(f"rotate_jws: {message}", file=sys.stderr)
    raise SystemExit(1)


def request_json(
    url: str,
    *,
    method: str,
    body: bytes = b"",
    headers: dict[str, str] | None = None,
) -> dict:
    """Make an HTTP request and decode its JSON response."""
    request = urllib.request.Request(
        url,
        data=body or None,
        headers=headers or {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        fail(f"HTTP {error.code} from {method} {url}: {detail}")
    except (urllib.error.URLError, TimeoutError) as error:
        fail(f"network error from {method} {url}: {error}")
    except json.JSONDecodeError as error:
        fail(f"invalid JSON from {method} {url}: {error}")


def atomic_write(path: Path, content: str) -> None:
    """Write a file atomically with owner-only permissions."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def write_json(path: Path, data: dict) -> None:
    """Persist JSON credential state atomically."""
    atomic_write(path, json.dumps(data, indent=2) + "\n")


def refresh_access_token(config: dict, config_path: Path) -> str:
    """Mint an access token and persist a rotated refresh token."""
    body = urllib.parse.urlencode(
        {
            "grant_type": "refresh_token",
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "refresh_token": config["refresh_token"],
        }
    ).encode()
    response = request_json(
        "https://webexapis.com/v1/access_token",
        method="POST",
        body=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    access_token = response.get("access_token")
    if not access_token:
        fail("Webex token response did not contain access_token")
    if response.get("refresh_token"):
        config["refresh_token"] = response["refresh_token"]
        write_json(config_path, config)
    return str(access_token)


def rotate_datasource(config: dict, access_token: str, nonce: str) -> dict:
    """Change the datasource nonce so Webex issues a fresh signed JWS."""
    payload = {
        "schemaId": config.get("schema_id", DEFAULT_SCHEMA_ID),
        "url": config["data_source_url"],
        "audience": config.get("audience", "NemotronVoiceAgent"),
        "subject": config.get("subject", "callAudioData"),
        "nonce": nonce,
        "tokenLifetimeMinutes": int(config.get("token_lifetime_minutes", 720)),
        "status": "active",
    }
    return request_json(
        f"https://webexapis.com/v1/dataSources/{config['data_source_id']}",
        method="PUT",
        body=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )


def write_nonce_file(path: Path, nonce: str) -> None:
    """Publish the nonce the adapter watches, keeping the file's inode stable.

    Compose bind-mounts this file into the adapter, and a bind mount follows the
    inode it was created from, so replacing the file would leave the container
    reading the old one. The payload is a fixed-length single write.
    """
    payload = (nonce + "\n").encode("utf-8")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o644)
    with path.open("r+b") as handle:
        handle.write(payload)
        handle.truncate()
        handle.flush()
        os.fsync(handle.fileno())
    # The adapter runs as a non-root user, and the nonce is the rotation marker
    # Cisco echoes back in the token's jti claim, not a signing secret.
    os.chmod(path, 0o644)


def nonce_file(config: dict) -> Path | None:
    """Return the file the adapter watches for rotated nonces, when configured."""
    configured = config.get("nonce_file_path")
    return Path(configured).expanduser() if configured else None


def write_deployment_env(config: dict, nonce: str, watched: Path | None) -> Path:
    """Write non-secret adapter deployment settings consumed by Compose."""
    path = Path(config["deployment_env_path"]).expanduser()
    values = {
        "BYOVA_WS_PUBLISH_ADDRESS": config.get("publish_address", "0.0.0.0"),
        "BYOVA_WS_PUBLISH_PORT": str(config.get("publish_port", 443)),
        "BYOVA_TLS_CERT_HOST": config["tls_cert_host"],
        "BYOVA_TLS_KEY_HOST": config["tls_key_host"],
        "BYOVA_TLS_CERT": "/run/secrets/byova-tls.crt",
        "BYOVA_TLS_KEY": "/run/secrets/byova-tls.key",
        "BYOVA_AUTH_ENABLED": "true",
        "BYOVA_ALLOW_PLAINTEXT_LOCAL": "false",
        "BYOVA_ALLOW_AUTH_DISABLED_LOCAL": "false",
        "NEMOTRON_INSECURE_TLS": str(
            config.get("nemotron_insecure_tls", True)
        ).lower(),
        "CISCO_JWT_ISSUERS": config.get(
            "jwt_issuers", "https://idbroker-b-us.webex.com/idb"
        ),
        "CISCO_JWT_AUDIENCE": config.get("audience", "NemotronVoiceAgent"),
        "CISCO_JWT_SUBJECT": config.get("subject", "callAudioData"),
        "CISCO_JWT_NONCE": nonce,
        "CISCO_DATASOURCE_URL": config["data_source_url"],
        "CISCO_DATASOURCE_SCHEMA_UUID": config.get(
            "schema_id", DEFAULT_SCHEMA_ID
        ),
        "CISCO_ORG_UUID": config["org_id"],
    }
    if watched is not None:
        values["BYOVA_NONCE_FILE_HOST"] = str(watched)
        values["CISCO_JWT_NONCE_FILE"] = "/run/secrets/byova-nonce"
    atomic_write(path, "".join(f"{key}={value}\n" for key, value in values.items()))
    return path


def restart_adapter(config: dict, env_path: Path) -> None:
    """Recreate only the adapter with the newly issued nonce."""
    command = [
        "docker",
        "compose",
        "--env-file",
        str(env_path),
        "--profile",
        "webex-byova-ws-assistant",
        "up",
        "-d",
        "--force-recreate",
        "webex-byova-ws-adapter",
    ]
    try:
        subprocess.run(
            command,
            cwd=Path(config["project_dir"]).expanduser(),
            check=True,
            stdout=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        fail(f"adapter restart failed after datasource rotation: {error}")


def main() -> int:
    """Refresh credentials, rotate the JWS, and deploy the matching nonce."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text())
    except FileNotFoundError:
        fail(f"config not found: {args.config}")
    except json.JSONDecodeError as error:
        fail(f"invalid config JSON: {error}")

    required = (
        "client_id",
        "client_secret",
        "refresh_token",
        "data_source_id",
        "data_source_url",
        "org_id",
        "deployment_env_path",
        "project_dir",
        "tls_cert_host",
        "tls_key_host",
    )
    missing = [field for field in required if not config.get(field)]
    if missing:
        fail(f"missing config fields: {', '.join(missing)}")

    access_token = refresh_access_token(config, args.config)
    nonce = secrets.token_hex(16)
    response = rotate_datasource(config, access_token, nonce)
    watched = nonce_file(config)
    env_path = write_deployment_env(config, nonce, watched)
    if watched is None:
        restart_adapter(config, env_path)
    else:
        # The adapter rereads this file, so rotation never interrupts a call.
        write_nonce_file(watched, nonce)

    config["current_nonce"] = nonce
    config["last_rotation_at"] = dt.datetime.now(
        dt.timezone.utc  # noqa: UP017
    ).isoformat()
    config["last_token_expiry"] = response.get("tokenExpiryTime")
    write_json(args.config, config)

    if not args.quiet:
        print(
            json.dumps(
                {
                    "data_source_id": response.get("id"),
                    "status": response.get("status"),
                    "url": response.get("url"),
                    "token_expiry": response.get("tokenExpiryTime"),
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
