"""Client for the experimental nva.offline.v1 server contract.

Use /api/ws/offline. The capability handshake refuses a stock real-time
/api/ws server before uploading PCM.
Python 3.12+, websockets>=14,<16. Other functionality uses the standard library.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import time
import wave
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

PROTOCOL = "nva.offline.v1"
RATE = 16000
HEADER = struct.Struct("!4sQQI")  # magic, sequence, start sample, sample count
REQUIRED = {"offset_audio", "explicit_eof", "aligned_output", "service_trace"}


class ProtocolError(ValueError):
    """The server violated the negotiated offline media contract."""


def seconds_to_samples(value: str, rate: int = RATE) -> int:
    """Require a duration that represents a whole source sample count."""
    samples = Decimal(value) * rate
    if not samples.is_finite() or samples < 0 or samples != samples.to_integral_value():
        raise ValueError(f"Duration {value!r} must represent a nonnegative whole sample count")
    return int(samples)


def read_pcm(path: Path) -> bytes:
    """Read the supported WAV format without changing or trimming samples."""
    with wave.open(str(path), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (
            RATE,
            1,
            2,
            "NONE",
        ):
            raise ValueError(f"{path}: expected uncompressed 16 kHz mono PCM16 WAV")
        pcm = source.readframes(source.getnframes())
        if len(pcm) != source.getnframes() * 2:
            raise ValueError(f"{path}: truncated WAV")
        return pcm


def write_pcm(path: Path, pcm: bytes, rate: int = RATE) -> None:
    """Write mono PCM16 with its declared sample rate."""
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm)


def prepare_session(inputs: list[Path], gap_samples: int, total_samples: int | None) -> tuple[bytes, list[dict]]:
    """Concatenate recordings with gaps; pad the horizon, never truncate speech."""
    if not inputs or gap_samples < 0:
        raise ValueError("Provide at least one input and a nonnegative gap")
    pcm = bytearray()
    segments = []
    for index, path in enumerate(inputs):
        if index:
            pcm.extend(bytes(gap_samples * 2))
        start = len(pcm) // 2
        audio = read_pcm(path)
        pcm.extend(audio)
        segments.append({"file": str(path.resolve()), "start_sample": start, "end_sample": len(pcm) // 2})
    if total_samples is not None:
        if total_samples < len(pcm) // 2:
            raise ValueError("Requested session duration is shorter than the recordings plus gaps")
        pcm.extend(bytes(total_samples * 2 - len(pcm)))
    return bytes(pcm), segments


def encode_packet(magic: bytes, sequence: int, start: int, pcm: bytes) -> bytes:
    """Preserve positions, sequence, and the exact PCM payload."""
    if magic not in (b"NVAI", b"NVAO") or sequence < 0 or start < 0 or not pcm or len(pcm) % 2:
        raise ProtocolError("Invalid PCM packet")
    return HEADER.pack(magic, sequence, start, len(pcm) // 2) + pcm


def decode_packet(data: bytes, magic: bytes) -> tuple[int, int, bytes]:
    """Validate and unpack one positioned PCM packet."""
    if len(data) < HEADER.size:
        raise ProtocolError("Truncated audio header")
    actual_magic, sequence, start, count = HEADER.unpack_from(data)
    pcm = data[HEADER.size :]
    if actual_magic != magic or count == 0 or len(pcm) != count * 2:
        raise ProtocolError("Audio packet magic or sample count mismatch")
    return sequence, start, pcm


def integer_field(message: dict, name: str) -> int:
    """Reject booleans and negative offsets at the wire boundary."""
    value = message.get(name)
    if type(value) is not int or value < 0:
        raise ProtocolError(f"Invalid {name}")
    return value


@dataclass
class PositionedOutput:
    """Collect ordered committed spans without observing arrival timestamps."""

    rate: int
    input_samples: int
    max_samples: int

    def __post_init__(self):
        """Initialize the output extent and expected sequence."""
        self.spans: list[tuple[int, bytes]] = []
        self.end = 0
        self.sequence = 0

    def append(self, packet: bytes) -> None:
        """Reject reordered, overlapping, and oversized committed output."""
        sequence, start, pcm = decode_packet(packet, b"NVAO")
        if sequence != self.sequence:
            raise ProtocolError("Missing, duplicate, or reordered output sequence")
        if start < self.end:
            raise ProtocolError("Overlapping or retroactive committed output")
        end = start + len(pcm) // 2
        if end > self.max_samples:
            raise ProtocolError("Output exceeds configured duration limit")
        self.spans.append((start, pcm))
        self.end = end
        self.sequence += 1

    def finish(self, message: dict) -> int:
        """Require explicit completion with matching input and output counts."""
        extent = integer_field(message, "rendered_samples")
        if integer_field(message, "input_samples") != self.input_samples:
            raise ProtocolError("Server input extent differs from uploaded audio")
        if message.get("status") != "ok" or extent < max(self.end, self.input_samples) or extent > self.max_samples:
            raise ProtocolError("Invalid session completion extent or status")
        if integer_field(message, "output_packets") != self.sequence:
            raise ProtocolError("Completion packet count differs from received output")
        return extent

    def write(self, path: Path, extent: int) -> None:
        """Render committed samples with exact zero padding; no arrival timestamps."""
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.rate)
            cursor = 0
            for start, pcm in self.spans:
                self._zeros(output, start - cursor)
                output.writeframesraw(pcm)
                cursor = start + len(pcm) // 2
            self._zeros(output, extent - cursor)

    @staticmethod
    def _zeros(output, samples: int):
        while samples:
            count = min(samples, 65536)
            output.writeframesraw(bytes(count * 2))
            samples -= count


def benchmark_view(source: Path, target: Path, extent: int) -> None:
    """Produce an exact source-horizon copy for FDB; preserve full output separately."""
    with wave.open(str(source), "rb") as input_file, wave.open(str(target), "wb") as output:
        output.setparams(input_file.getparams())
        remaining = extent
        while remaining:
            count = min(remaining, 65536)
            audio = input_file.readframes(count)
            output.writeframesraw(audio + bytes(count * 2 - len(audio)))
            remaining -= count


async def run_client(
    url: str,
    pcm: bytes,
    output_dir: Path,
    chunk_samples: int = 512,
    upload_speed: float = 0,
    watchdog: float = 600,
    max_output_seconds: int = 7200,
    config: dict | None = None,
) -> dict:
    """Upload concurrently with receiving and render only validated completion."""
    from websockets.asyncio.client import connect

    if chunk_samples <= 0 or upload_speed < 0 or watchdog <= 0:
        raise ValueError("Invalid chunk, speed, or watchdog")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("full_output.wav", "output.wav", "events.jsonl", "result.json"):
        if (output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {output_dir / name}")
    input_samples = len(pcm) // 2
    run_start = time.monotonic()
    result = {}
    async with asyncio.timeout(watchdog):  # Execution watchdog, never a media deadline.
        async with connect(url, max_size=8 * 1024 * 1024, open_timeout=15) as socket:
            await socket.send(
                json.dumps(
                    {
                        "type": "offline.session.start",
                        "protocol": PROTOCOL,
                        "sample_rate": RATE,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                        "input_samples": input_samples,
                        "config": config or {},
                    }
                )
            )
            response = await asyncio.wait_for(socket.recv(), timeout=15)
            if not isinstance(response, str):
                raise ProtocolError("Server did not return an offline capability acknowledgement")
            accepted = json.loads(response)
            if (
                accepted.get("type") != "offline.session.accepted"
                or accepted.get("protocol") != PROTOCOL
                or not REQUIRED.issubset(set(accepted.get("capabilities", [])))
                or accepted.get("output_sample_rate") != RATE
            ):
                raise ProtocolError("Server does not implement the required offline protocol; no PCM uploaded")
            packet_limit = accepted.get("limits", {}).get("max_packet_samples", chunk_samples)
            if type(packet_limit) is not int or packet_limit <= 0:
                raise ProtocolError("Server returned an invalid input packet limit; no PCM uploaded")
            effective_chunk_samples = min(chunk_samples, packet_limit)
            output = PositionedOutput(RATE, input_samples, max_output_seconds * RATE)

            async def upload():
                start_time = time.monotonic()
                sequence = 0
                for start in range(0, input_samples, effective_chunk_samples):
                    if upload_speed:
                        await asyncio.sleep(max(0, start_time + start / RATE / upload_speed - time.monotonic()))
                    chunk = pcm[start * 2 : min(start + effective_chunk_samples, input_samples) * 2]
                    await socket.send(encode_packet(b"NVAI", sequence, start, chunk))
                    sequence += 1
                await socket.send(
                    json.dumps(
                        {
                            "type": "offline.input.complete",
                            "input_samples": input_samples,
                            "input_packets": sequence,
                            "sha256": hashlib.sha256(pcm).hexdigest(),
                        }
                    )
                )
                return time.monotonic() - start_time

            async def receive():
                with (output_dir / "events.jsonl").open("x", encoding="utf-8") as events:
                    async for message in socket:
                        if isinstance(message, bytes):
                            output.append(message)
                            continue
                        event = json.loads(message)
                        events.write(json.dumps(event) + "\n")
                        if event.get("type") == "offline.session.error":
                            raise ProtocolError(str(event.get("message", "Session failed")))
                        if event.get("type") == "offline.session.complete":
                            return event
                raise ProtocolError("Connection closed without explicit session completion")

            async with asyncio.TaskGroup() as group:
                sender = group.create_task(upload())
                receiver = group.create_task(receive())
            extent = output.finish(receiver.result())
            output.write(output_dir / "full_output.wav", extent)
            benchmark_view(output_dir / "full_output.wav", output_dir / "output.wav", input_samples)
            elapsed = time.monotonic() - run_start
            result = {
                "protocol": PROTOCOL,
                "input_samples": input_samples,
                "rendered_samples": extent,
                "source_duration_s": input_samples / RATE,
                "output_duration_s": extent / RATE,
                "upload_wall_s": sender.result(),
                "total_wall_s": elapsed,
                "source_seconds_per_wall_second": input_samples / RATE / elapsed,
                "upload_speed": upload_speed,
                "chunk_samples": effective_chunk_samples,
                "requested_chunk_samples": chunk_samples,
                "input_packets": (input_samples + effective_chunk_samples - 1) // effective_chunk_samples,
                "source_sha256": hashlib.sha256(pcm).hexdigest(),
                "output_pcm_sha256": hashlib.sha256(read_pcm(output_dir / "full_output.wav")).hexdigest(),
                "server_completion": receiver.result(),
                "measurement_note": "Offline wall throughput; not live E2E latency or concurrency capacity.",
            }
            (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    """Prepare a finite source and optionally evaluate it over WebSocket."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="16 kHz mono PCM16 WAVs, in order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gap-seconds", default="5")
    parser.add_argument("--duration-seconds", help="Pad prepared input to this duration; never truncate")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--url", help="Offline WebSocket endpoint, e.g. ws://localhost:7860/api/ws/offline")
    parser.add_argument("--config", type=Path, help="JSON generic service/prompt configuration (omit API keys)")
    parser.add_argument("--chunk-samples", type=int, default=512)
    parser.add_argument("--upload-speed", type=float, default=0, help="0=bulk; 1=real-time upload; 10=10x upload")
    parser.add_argument("--watchdog-seconds", type=float, default=600)
    args = parser.parse_args()
    if not args.prepare_only and not args.url:
        parser.error("Provide --url or --prepare-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "source.wav"
    if source.exists():
        parser.error(f"Refusing to overwrite {source}")
    pcm, segments = prepare_session(
        args.input,
        seconds_to_samples(args.gap_seconds),
        seconds_to_samples(args.duration_seconds) if args.duration_seconds else None,
    )
    write_pcm(source, pcm)
    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "sample_rate": RATE,
                "input_samples": len(pcm) // 2,
                "segments": segments,
                "source_sha256": hashlib.sha256(pcm).hexdigest(),
                "note": "Segment labels describe source files; they are not oracle VAD/turn boundaries.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.prepare_only:
        print(f"Prepared {source}: {len(pcm) / 2 / RATE:.3f} seconds")
    else:
        config = json.loads(args.config.read_text()) if args.config else {}
        print(
            json.dumps(
                asyncio.run(
                    run_client(
                        args.url,
                        pcm,
                        args.output_dir,
                        args.chunk_samples,
                        args.upload_speed,
                        args.watchdog_seconds,
                        config=config,
                    )
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
