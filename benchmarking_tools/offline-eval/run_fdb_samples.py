"""Upload individual canonical FDB recordings in bulk and preserve aligned audio.

Each input must already be 16 kHz mono PCM16 WAV. Conversion is deliberately a
separate, provenance-recorded step. This script does not concatenate independent
FDB examples or run the benchmark's scoring models. It requires the experimental
NVA /api/ws/offline endpoint; it never substitutes generated fixture audio.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CLIENT_SPEC = importlib.util.spec_from_file_location("nva_fdb_offline_client", Path(__file__).with_name("client.py"))
assert _CLIENT_SPEC is not None and _CLIENT_SPEC.loader is not None
client = importlib.util.module_from_spec(_CLIENT_SPEC)
sys.modules[_CLIENT_SPEC.name] = client
_CLIENT_SPEC.loader.exec_module(client)


@dataclass(frozen=True)
class Case:
    """One independently scored source example with optional provenance."""

    case_id: str
    input_path: Path
    provenance: dict[str, Any]


def load_cases(inputs: list[Path] | None = None, manifest: Path | None = None) -> list[Case]:
    """Read explicit inputs or a manifest without combining or reordering cases."""
    if bool(inputs) == bool(manifest):
        raise ValueError("Provide exactly one of explicit inputs or a manifest")
    if manifest:
        records = json.loads(manifest.read_text(encoding="utf-8"))["cases"]
        if not isinstance(records, list):
            raise ValueError("Manifest cases must be a list")
        cases = []
        for record in records:
            source = Path(record["input"])
            if not source.is_absolute():
                source = manifest.resolve().parent / source
            provenance = record.get("provenance", {})
            if not isinstance(provenance, dict):
                raise ValueError("Case provenance must be an object")
            cases.append(Case(record["id"], source.resolve(), provenance))
    else:
        cases = [Case(path.stem, path.resolve(), {}) for path in inputs or []]
    if not cases:
        raise ValueError("At least one case is required")
    ids = set()
    for case in cases:
        if not isinstance(case.case_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", case.case_id):
            raise ValueError("Case IDs must be safe directory names of at most 120 characters")
        if case.case_id in ids:
            raise ValueError("Case IDs must be unique")
        ids.add(case.case_id)
        if not case.input_path.is_file():
            raise ValueError(f"Input does not exist: {case.input_path}")
    return cases


def write_comparison(source: Path, response: Path, target: Path) -> dict[str, Any]:
    """Interleave unchanged user/bot PCM at their sample offsets for listening.

    Left channel is user input; right channel is complete server output. The user
    channel is padded with zeros for any final bot tail. No normalization, gain,
    resampling, offset adjustment, or mixing is applied.
    """
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    temporary = target.with_name(target.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite {temporary}")
    try:
        with wave.open(str(source), "rb") as user, wave.open(str(response), "rb") as bot:
            for audio in (user, bot):
                if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth(), audio.getcomptype()) != (
                    client.RATE,
                    1,
                    2,
                    "NONE",
                ):
                    raise ValueError("Comparison requires uncompressed 16 kHz mono PCM16 inputs")
            user_frames, bot_frames = user.getnframes(), bot.getnframes()
            if bot_frames < user_frames:
                raise ValueError("Complete response must cover the entire source horizon")
            with wave.open(str(temporary), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(client.RATE)
                for start in range(0, bot_frames, 65536):
                    count = min(65536, bot_frames - start)
                    left = user.readframes(count)
                    right = bot.readframes(count)
                    expected_user = max(0, min(count, user_frames - start)) * 2
                    if len(left) != expected_user or len(right) != count * 2:
                        raise ValueError("Truncated comparison input")
                    left += bytes(count * 2 - len(left))
                    stereo = bytearray(count * 4)
                    stereo[0::4], stereo[1::4] = left[0::2], left[1::2]
                    stereo[2::4], stereo[3::4] = right[0::2], right[1::2]
                    output.writeframesraw(stereo)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": target.name,
        "sample_rate": client.RATE,
        "channels": {"left": "user", "right": "bot"},
        "source_samples": user_frames,
        "rendered_samples": bot_frames,
        "user_tail_padding_samples": bot_frames - user_frames,
        "alignment_adjustment_samples": 0,
    }


def _write_json(path: Path, value: Any) -> None:
    """Persist a machine-readable case status after each completed step."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


async def run_batch(
    cases: list[Case],
    output_dir: Path,
    *,
    url: str | None,
    config: dict | None = None,
    prepare_only: bool = False,
    chunk_samples: int = 65536,
    watchdog: float = 900,
    max_output_seconds: int = 7200,
    max_cases: int = 5,
    max_input_seconds: int = 300,
) -> dict[str, Any]:
    """Run a bounded batch sequentially, recording failures without false audio."""
    if not cases or len(cases) > max_cases or max_cases <= 0 or max_input_seconds <= 0:
        raise ValueError("Case count or source duration limit is invalid; select a smaller batch or raise bounds")
    if not prepare_only and not url:
        raise ValueError("Provide an offline endpoint or prepare_only=True")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory {output_dir}")
    # Validate the whole selection before beginning any model requests.
    sources = [(case, client.read_pcm(case.input_path)) for case in cases]
    if any(not pcm or len(pcm) // 2 > max_input_seconds * client.RATE for _, pcm in sources):
        raise ValueError("Inputs must be nonempty and fit max_input_seconds; inputs are never truncated")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    summary: dict[str, Any] = {
        "protocol": client.PROTOCOL,
        "upload_mode": "bulk_no_pacing",
        "requested_chunk_samples": chunk_samples,
        "audio_origin": "real_server_provider_output_required",
        "measurement_note": "Observed offline provider and run timings; not real-time FDB latency or scoring.",
        "cases": [],
    }
    for case, pcm in sources:
        directory = output_dir / case.case_id
        directory.mkdir()
        client.write_pcm(directory / "input.wav", pcm)
        entry = {
            "id": case.case_id,
            "input": str(case.input_path),
            "input_samples": len(pcm) // 2,
            "input_duration_s": len(pcm) / 2 / client.RATE,
            "input_pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            "input_file_sha256": hashlib.sha256(case.input_path.read_bytes()).hexdigest(),
            "provenance": case.provenance,
            "status": "prepared_only" if prepare_only else "pending",
        }
        _write_json(directory / "input_manifest.json", entry)
        if not prepare_only:
            try:
                result = await client.run_client(
                    url,
                    pcm,
                    directory,
                    chunk_samples=chunk_samples,
                    upload_speed=0,
                    watchdog=watchdog,
                    max_output_seconds=max_output_seconds,
                    config=config,
                )
                comparison = write_comparison(
                    directory / "input.wav", directory / "full_output.wav", directory / "comparison.wav"
                )
                _write_json(directory / "comparison_manifest.json", comparison)
                entry.update(
                    status="ok",
                    rendered_samples=result["rendered_samples"],
                    output_duration_s=result["output_duration_s"],
                    upload_wall_s=result["upload_wall_s"],
                    total_wall_s=result["total_wall_s"],
                    input_packets=result["input_packets"],
                    chunk_samples=result["chunk_samples"],
                    output_pcm_sha256=result["output_pcm_sha256"],
                    service_metrics=result["server_completion"].get("metrics", {}),
                    configuration_fingerprint=result["server_completion"].get("configuration_fingerprint"),
                )
            except Exception as error:
                # Raw connection/provider messages can include URLs or credentials.
                # The server trace and exception class suffice to identify a failed
                # trial without turning it into a fabricated successful output.
                entry.update(status="failed", error_type=type(error).__name__)
                _write_json(directory / "failure.json", {"status": "failed", "error_type": type(error).__name__})
        summary["cases"].append(entry)
        summary["total_wall_s"] = time.monotonic() - started
        summary["successful_cases"] = sum(item["status"] == "ok" for item in summary["cases"])
        summary["failed_cases"] = sum(item["status"] == "failed" for item in summary["cases"])
        _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Select a small FDB subset and upload each complete recording without sleeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--input", type=Path, nargs="+", help="Canonical FDB WAVs; case IDs use file stems")
    selection.add_argument("--manifest", type=Path, help='JSON {"cases":[{"id":"...","input":"...","provenance":{}}]}')
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", help="Experimental ws(s)://.../api/ws/offline endpoint")
    parser.add_argument("--config", type=Path, help="Generic NVA service/prompt JSON, without credentials")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Preserve canonical inputs without invoking providers"
    )
    parser.add_argument(
        "--chunk-samples", type=int, default=65536, help="Capped to server's negotiated input packet limit"
    )
    parser.add_argument("--watchdog-seconds", type=float, default=900)
    parser.add_argument("--max-output-seconds", type=int, default=7200)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--max-input-seconds", type=int, default=300)
    args = parser.parse_args()
    if not args.prepare_only and not args.url:
        parser.error("Provide --url or --prepare-only")
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(config, dict):
        parser.error("Configuration must be a JSON object")
    result = asyncio.run(
        run_batch(
            load_cases(args.input, args.manifest),
            args.output_dir,
            url=args.url,
            config=config,
            prepare_only=args.prepare_only,
            chunk_samples=args.chunk_samples,
            watchdog=args.watchdog_seconds,
            max_output_seconds=args.max_output_seconds,
            max_cases=args.max_cases,
            max_input_seconds=args.max_input_seconds,
        )
    )
    print(
        json.dumps(
            {
                "summary": str(args.output_dir / "summary.json"),
                "cases": len(result["cases"]),
                "failed_cases": result["failed_cases"],
            }
        )
    )
    if result["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
