# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Run cloud-TTS service evals one scenario at a time with retries."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = Path("tests/pipecat_evals/service/cloud_tts_manifest.yaml")
HARD_TIMEOUT_PADDING_SECONDS = 60


def _scenario_names(manifest: Path) -> list[str]:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    names: list[str] = []
    for item in data.get("suite", []):
        names.extend(str(name) for name in item.get("scenarios", []))
    return names


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def _load_yaml_mapping(path: Path, description: str) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{description} must be a YAML mapping: {path}")
    return data


def _resolve_relative_path(path: str, *, base: Path) -> str:
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else (base / candidate).resolve())


def _resolve_attachment_path(path: str, *, base: Path) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    root_candidate = ROOT / candidate
    if root_candidate.exists():
        return str(root_candidate.resolve())
    return _resolve_relative_path(path, base=base)


def _runner_body_from_yaml(path: Path, key: str) -> dict:
    data = _load_yaml_mapping(path, "runner body config")
    defaults = data.get("defaults") or {}
    bodies = data.get("bodies") or {}
    if not isinstance(defaults, dict) or not isinstance(bodies, dict):
        raise ValueError(f"{path}: 'defaults:' and 'bodies:' must be mappings")
    body = bodies.get(key)
    if not isinstance(body, dict):
        raise ValueError(f"{path}: missing runner body entry {key!r}")

    merged = {**defaults, **body}
    attachment = merged.get("eval_attachment")
    if isinstance(attachment, dict) and isinstance(attachment.get("path"), str):
        attachment = dict(attachment)
        attachment["path"] = _resolve_attachment_path(attachment["path"], base=path.parent)
        merged["eval_attachment"] = attachment
    return merged


def _materialize_manifest(manifest: Path, work_dir: Path) -> Path:
    data = _load_yaml_mapping(manifest, "manifest")
    manifest_base = manifest.parent
    generated = dict(data)

    for key in ("bots_dir", "scenarios_dir", "runs_dir"):
        value = generated.get(key)
        if value:
            generated[key] = _resolve_relative_path(str(value), base=manifest_base)

    bodies_dir = work_dir / "runner_bodies"
    suite = []
    for index, item in enumerate(data.get("suite", [])):
        if not isinstance(item, dict):
            raise ValueError(f"{manifest}: suite item #{index} must be a mapping")
        generated_item = dict(item)
        runner_body = generated_item.get("runner_body")
        if runner_body and str(runner_body).endswith((".yaml", ".yml")):
            body_key = str(generated_item.pop("runner_body_key", "") or "").strip()
            if not body_key:
                raise ValueError(f"{manifest}: suite item #{index} uses YAML runner_body without runner_body_key")
            body_path = Path(_resolve_relative_path(str(runner_body), base=manifest_base))
            body = _runner_body_from_yaml(body_path, body_key)
            materialized_body = bodies_dir / f"{_slug(body_key)}.json"
            materialized_body.parent.mkdir(parents=True, exist_ok=True)
            materialized_body.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
            generated_item["runner_body"] = str(materialized_body)
        elif "runner_body_key" in generated_item:
            raise ValueError(f"{manifest}: suite item #{index} has runner_body_key without a YAML runner_body")
        suite.append(generated_item)

    generated["suite"] = suite
    generated_manifest = work_dir / "cloud_tts_manifest.yaml"
    generated_manifest.write_text(yaml.safe_dump(generated, sort_keys=False), encoding="utf-8")
    return generated_manifest


def _run_scenario(manifest: Path, scenario: str, *, name_prefix: str, timeout: int, attempt: int) -> int:
    env = os.environ.copy()
    src_path = str(ROOT / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
    name = f"{name_prefix}-{_slug(scenario)}-a{attempt}"
    with tempfile.TemporaryDirectory(prefix="cloud-tts-evals-") as tmp:
        materialized_manifest = _materialize_manifest(manifest, Path(tmp))
        command = [
            sys.executable,
            "-m",
            "pipecat.evals",
            "suite",
            str(materialized_manifest),
            "--scenario",
            scenario,
            "--name",
            name,
            "--timeout",
            str(timeout),
        ]
        print(f"\n=== {scenario} (attempt {attempt}) ===", flush=True)
        hard_timeout = timeout + HARD_TIMEOUT_PADDING_SECONDS
        try:
            return subprocess.run(command, cwd=ROOT, env=env, check=False, timeout=hard_timeout).returncode
        except subprocess.TimeoutExpired:
            print(
                f"{scenario} exceeded hard timeout of {hard_timeout}s (Pipecat suite timeout: {timeout}s)",
                file=sys.stderr,
                flush=True,
            )
            return 124


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--name-prefix", default="cloud-tts")
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=20.0)
    parser.add_argument("--scenario", action="append", help="Run only this scenario. Can be provided multiple times.")
    args = parser.parse_args()
    if args.timeout < 0:
        parser.error("--timeout must be non-negative")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if not math.isfinite(args.retry_delay) or args.retry_delay < 0:
        parser.error("--retry-delay must be finite and non-negative")
    return args


def main() -> int:
    """Run all requested cloud-TTS eval scenarios and return a process status."""
    args = _parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    scenarios = args.scenario or _scenario_names(manifest)
    if not scenarios:
        print(f"No scenarios found in {manifest}", file=sys.stderr)
        return 2

    passed: list[str] = []
    failed: list[str] = []

    for scenario in scenarios:
        for attempt in range(1, args.retries + 2):
            code = _run_scenario(
                manifest,
                scenario,
                name_prefix=args.name_prefix,
                timeout=args.timeout,
                attempt=attempt,
            )
            if code == 0:
                passed.append(scenario)
                break
            if attempt <= args.retries:
                print(f"{scenario} failed; retrying in {args.retry_delay:g}s", flush=True)
                time.sleep(args.retry_delay)
        else:
            failed.append(scenario)

    print("\nCloud TTS eval summary")
    print(f"  passed: {len(passed)}/{len(scenarios)}")
    if failed:
        print("  failed:")
        for scenario in failed:
            print(f"    - {scenario}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
