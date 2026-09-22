# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Issue a browser credential without exposing the deployment key in argv."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import ssl
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from voiceclaw.config import ConfigurationError, CredentialReference
from voiceclaw.frontend_runtime import resolve_credential_value

_MAX_RESPONSE_BYTES = 16 * 1024


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if not 1 <= timeout <= 120:
        raise argparse.ArgumentTypeError("must be between 1 and 120 seconds")
    return timeout


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue one short-lived VoiceClaw browser credential from an owner-controlled master-key file.",
    )
    parser.add_argument(
        "--url",
        default="https://127.0.0.1:7860/v1/realtime/client_secrets",
        help="VoiceClaw client-secret endpoint (default: %(default)s)",
    )
    parser.add_argument("--master-key-file", type=Path, required=True, help="absolute deployment master-key file")
    parser.add_argument("--output", type=Path, required=True, help="owner-only file to receive the ek_ credential")
    parser.add_argument("--ca-file", type=Path, help="trusted CA certificate for the VoiceClaw listener")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="disable TLS verification for a literal loopback development endpoint only",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=15.0,
        help="request timeout (default: %(default)s)",
    )
    return parser


def _endpoint(raw_url: str, *, insecure: bool) -> str:
    url = raw_url.strip()
    try:
        parsed = urlsplit(url)
        _ = parsed.port
    except ValueError as error:
        raise ValueError("--url must be a valid HTTPS endpoint") from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/realtime/client_secrets"
    ):
        raise ValueError("--url must be an HTTPS origin followed by /v1/realtime/client_secrets")
    if insecure:
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError as error:
            raise ValueError("--insecure is allowed only for a literal loopback endpoint") from error
        if not address.is_loopback:
            raise ValueError("--insecure is allowed only for a literal loopback endpoint")
    return url


def _private_output_directory(path: Path) -> None:
    parent = path.parent
    with suppress(FileExistsError):
        parent.mkdir(parents=True, mode=0o700)
    metadata = parent.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise PermissionError("client-secret output directory must be an owner-controlled directory")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError("client-secret output directory must have owner-only permissions")


def _write_client_secret(path: Path, value: str) -> None:
    _private_output_directory(path)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii", closefd=True) as destination:
            descriptor = -1
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
        temporary_name = ""
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary_name:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)


def issue_client_secret(
    *,
    url: str,
    master_key_file: Path,
    output: Path,
    ca_file: Path | None,
    insecure: bool,
    timeout: float,
) -> None:
    """Issue one credential and atomically persist only its bounded ``ek_`` value."""
    endpoint = _endpoint(url, insecure=insecure)
    if ca_file is not None and insecure:
        raise ValueError("--ca-file and --insecure are mutually exclusive")
    master_key = resolve_credential_value(
        CredentialReference(file=str(master_key_file)),
        source_environment={},
        label="public Realtime master key",
    )
    if master_key is None:  # pragma: no cover - a file reference always resolves or raises
        raise ConfigurationError("public Realtime master key is required")
    context = ssl._create_unverified_context() if insecure else ssl.create_default_context(cafile=ca_file)
    request = Request(
        endpoint,
        data=b"",
        headers={"Authorization": f"Bearer {master_key}", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout, context=context) as response:  # noqa: S310 - URL policy is validated above
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ValueError("client-secret response exceeded the supported size")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("client-secret endpoint returned invalid JSON") from error
    value = payload.get("value") if isinstance(payload, dict) else None
    if (
        not isinstance(value, str)
        or not value.startswith("ek_")
        or len(value) > 4096
        or any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("client-secret endpoint returned an invalid credential")
    _write_client_secret(output, value)


def main() -> None:
    """Run the credential issuer CLI without printing either credential."""
    arguments = _parser().parse_args()
    try:
        issue_client_secret(
            url=arguments.url,
            master_key_file=arguments.master_key_file,
            output=arguments.output,
            ca_file=arguments.ca_file,
            insecure=arguments.insecure,
            timeout=arguments.timeout,
        )
    except (ConfigurationError, HTTPError, OSError, URLError, ValueError) as error:
        raise SystemExit(f"VoiceClaw client-secret issuance failed: {error}") from error


if __name__ == "__main__":  # pragma: no cover
    main()
