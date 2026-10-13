# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-wide system sampler: GPU load and temperature, power, RAM and CPU load.

One sample per second is stored in ``system_samples`` while at least one session
is being recorded, together with the number of live sessions. Everything is
host-wide: there is no per-container or per-process attribution.

The sampler runs in its own daemon thread, so neither file reads nor the
``nvidia-smi`` child process ever touch the pipeline event loop.

Sources are pluggable (``SystemSource``). The built-in ones, in priority order:

==================  ==========================================  =========  ================
Source              Fields                                      Host       App container
==================  ==========================================  =========  ================
``/proc/stat``      ``cpu_load``                                yes        yes (host-wide)
``/proc/meminfo``   ``ram_used_mb``                             yes        yes (host-wide)
``/sys`` thermal    ``gpu_temp_c`` (zone ``gpu-thermal``)       Jetson     Jetson
``/sys`` hwmon      ``power_w`` (INA238 board input, ``VIN``)   Jetson     Jetson
``nvidia-smi``      ``gpu_load`` of GPU 0; temperature and      yes        local recipes
                    power when the sources above are missing               (NVIDIA runtime)
==================  ==========================================  =========  ================

``power_w`` does not mean the same thing everywhere, so every sample carries
``power_source``: ``board`` for the whole-module input power of the INA238, ``gpu``
for the GPU power draw reported by ``nvidia-smi``. Both come from the same source,
so the label always matches the value.

A source that is not available is skipped with a single log line.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Protocol

from loguru import logger

FIELDS = ("gpu_load", "gpu_temp_c", "power_w", "power_source", "ram_used_mb", "cpu_load")
# ``power_source`` values: what ``power_w`` measures depends on the source that provided it.
POWER_BOARD = "board"  # whole-module input power
POWER_GPU = "gpu"  # GPU power draw only
_MAX_FAILURES = 5
# A session counts as live while its recorder heartbeat (every 15 s) is recent.
_LIVE_STALE_SECS = 60.0


class SystemSource(Protocol):
    """One provider of host metrics."""

    name: str

    def available(self) -> bool:
        """Return whether this source can work on this machine."""

    def read(self) -> dict[str, Any]:
        """Return the fields this source knows right now (a subset of ``FIELDS``)."""

    def close(self) -> None:
        """Release resources (called when the last session ends)."""


class ProcStatSource:
    """CPU load in percent of all cores, from ``/proc/stat`` deltas."""

    name = "/proc/stat"

    def __init__(self, path: Path = Path("/proc/stat")):
        """Read CPU counters from ``path``."""
        self._path = path
        self._previous: tuple[int, int] | None = None

    def available(self) -> bool:
        """Return whether ``/proc/stat`` is readable."""
        return self._path.is_file()

    def read(self) -> dict[str, Any]:
        """Return ``cpu_load`` since the previous read (nothing on the first read)."""
        fields = [int(v) for v in self._path.read_text().splitlines()[0].split()[1:]]
        idle, total = fields[3] + (fields[4] if len(fields) > 4 else 0), sum(fields)
        previous, self._previous = self._previous, (idle, total)
        if previous is None or total <= previous[1]:
            return {}
        busy = 1.0 - (idle - previous[0]) / (total - previous[1])
        return {"cpu_load": round(100.0 * min(max(busy, 0.0), 1.0), 1)}

    def close(self) -> None:
        """Forget the previous counters so the next session starts a fresh delta."""
        self._previous = None


class ProcMeminfoSource:
    """RAM in use (``MemTotal - MemAvailable``), which on a Jetson includes GPU memory."""

    name = "/proc/meminfo"

    def __init__(self, path: Path = Path("/proc/meminfo")):
        """Read memory counters from ``path``."""
        self._path = path

    def available(self) -> bool:
        """Return whether ``/proc/meminfo`` is readable."""
        return self._path.is_file()

    def read(self) -> dict[str, Any]:
        """Return ``ram_used_mb``."""
        values = {}
        for line in self._path.read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                values[key] = int(rest.split()[0])
        if len(values) < 2:
            return {}
        return {"ram_used_mb": round((values["MemTotal"] - values["MemAvailable"]) / 1024.0, 1)}

    def close(self) -> None:
        """Nothing to release."""


class ThermalZoneSource:
    """GPU temperature from the ``gpu-thermal`` zone in ``/sys/class/thermal`` (Jetson)."""

    name = "/sys/class/thermal"

    def __init__(self, root: Path = Path("/sys/class/thermal"), zone_type: str = "gpu-thermal"):
        """Look for a thermal zone of ``zone_type`` under ``root``."""
        self._temp: Path | None = None
        try:
            for zone in sorted(root.glob("thermal_zone*")):
                if (zone / "type").read_text().strip() == zone_type:
                    self._temp = zone / "temp"
                    break
        except OSError:
            self._temp = None

    def available(self) -> bool:
        """Return whether a GPU thermal zone exists."""
        return self._temp is not None and self._temp.is_file()

    def read(self) -> dict[str, Any]:
        """Return ``gpu_temp_c``."""
        return {"gpu_temp_c": round(int(self._temp.read_text().strip()) / 1000.0, 1)}

    def close(self) -> None:
        """Nothing to release."""


class HwmonPowerSource:
    """Board input power from an INA238 monitor in ``/sys/class/hwmon`` (Jetson Thor ``VIN``)."""

    name = "/sys/class/hwmon"

    def __init__(self, root: Path = Path("/sys/class/hwmon")):
        """Look for an INA power monitor exposing ``power1_input`` under ``root``."""
        self._power: Path | None = None
        try:
            for device in sorted(root.glob("hwmon*")):
                name = (device / "name").read_text().strip() if (device / "name").is_file() else ""
                if name.startswith("ina") and (device / "power1_input").is_file():
                    self._power = device / "power1_input"
                    break
        except OSError:
            self._power = None

    def available(self) -> bool:
        """Return whether a power monitor exists."""
        return self._power is not None

    def read(self) -> dict[str, Any]:
        """Return ``power_w`` (the sysfs value is in microwatts)."""
        return {"power_w": round(int(self._power.read_text().strip()) / 1e6, 2), "power_source": POWER_BOARD}

    def close(self) -> None:
        """Nothing to release."""


class NvidiaSmiSource:
    """GPU load (and temperature / power draw) from one long-running ``nvidia-smi -l 1``.

    The child process is started on the first read and its output is consumed by
    a reader thread; ``read`` only returns the latest parsed line.
    """

    name = "nvidia-smi"
    _QUERY = ("utilization.gpu", "temperature.gpu", "power.draw")
    _KEYS = ("gpu_load", "gpu_temp_c", "power_w")

    def __init__(self, executable: str = "nvidia-smi", stale_after_secs: float = 5.0):
        """Use ``executable`` (looked up on ``PATH``)."""
        self._executable = shutil.which(executable)
        self._stale_after = stale_after_secs
        self._process: subprocess.Popen | None = None
        self._latest: tuple[float, dict[str, Any]] | None = None

    def available(self) -> bool:
        """Return whether ``nvidia-smi`` is installed."""
        return self._executable is not None

    @staticmethod
    def parse(line: str) -> dict[str, Any]:
        """Parse one CSV line; ``[N/A]`` fields (for example on Jetson) are left out."""
        out: dict[str, Any] = {}
        for key, raw in zip(NvidiaSmiSource._KEYS, line.split(","), strict=False):
            try:
                out[key] = float(raw.strip())
            except ValueError:
                continue
        if "power_w" in out:
            out["power_source"] = POWER_GPU
        return out

    def _pump(self, process: subprocess.Popen) -> None:
        for line in process.stdout:
            values = self.parse(line)
            if values:
                self._latest = (time.monotonic(), values)

    def read(self) -> dict[str, Any]:
        """Return the latest values, starting the child process if needed."""
        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                raise RuntimeError(f"nvidia-smi exited with code {self._process.returncode}")
            self._process = subprocess.Popen(  # noqa: S603 - fixed argument list, no shell
                [
                    self._executable,
                    "--id=0",  # one line per sample: the first GPU
                    f"--query-gpu={','.join(self._QUERY)}",
                    "--format=csv,noheader,nounits",
                    "-l",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            threading.Thread(target=self._pump, args=(self._process,), name="nvidia-smi-reader", daemon=True).start()
        latest = self._latest
        if latest is None or time.monotonic() - latest[0] > self._stale_after:
            return {}
        return dict(latest[1])

    def close(self) -> None:
        """Stop the child process."""
        process, self._process, self._latest = self._process, None, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


def default_sources() -> list[SystemSource]:
    """Built-in sources; earlier ones win when two provide the same field."""
    return [ProcStatSource(), ProcMeminfoSource(), ThermalZoneSource(), HwmonPowerSource(), NvidiaSmiSource()]


class SystemSampler:
    """Samples the sources once per interval while at least one session is live."""

    def __init__(self, sources: list[SystemSource] | None = None, *, interval_secs: float = 1.0):
        """Use ``sources`` (default: the built-in ones, resolved on first use)."""
        self._configured = sources
        self._sources: list[SystemSource] | None = None
        self._failures: dict[str, int] = {}
        self._interval = interval_secs
        self._lock = threading.Lock()
        self._live = 0
        self._stop: threading.Event | None = None
        self._thread: threading.Thread | None = None

    @property
    def live_sessions(self) -> int:
        """Return the number of sessions currently recorded."""
        return self._live

    def _resolve(self) -> list[SystemSource]:
        if self._sources is None:
            candidates = self._configured if self._configured is not None else default_sources()
            self._sources = [source for source in candidates if source.available()]
            missing = [source.name for source in candidates if source not in self._sources]
            active = ", ".join(source.name for source in self._sources) or "none (sampler disabled)"
            unavailable = f"; unavailable: {', '.join(missing)}" if missing else ""
            logger.info(f"System metrics sources: {active}{unavailable}")
        return self._sources

    def sample_once(self, now: float | None = None) -> dict[str, Any] | None:
        """Read every source and return one ``system_samples`` row (None when nothing was read)."""
        values: dict[str, Any] = {}
        for source in list(self._resolve()):
            try:
                reading = source.read()
                self._failures[source.name] = 0
            except Exception as exc:
                self._failures[source.name] = self._failures.get(source.name, 0) + 1
                if self._failures[source.name] >= _MAX_FAILURES:
                    logger.info(f"System metrics source {source.name} disabled after repeated errors: {exc}")
                    self._sources.remove(source)
                continue
            # The power value and its label are taken together, from one source.
            label = reading.pop("power_source", None)
            if "power_w" in reading and "power_w" not in values:
                values["power_w"], values["power_source"] = reading["power_w"], label
            for key, value in reading.items():
                if key in FIELDS:
                    values.setdefault(key, value)
        if not values:
            return None
        return {
            "ts": time.time() if now is None else now,
            "live_sessions": self._live,
            **{field: values.get(field) for field in FIELDS},
        }

    def acquire(self, store) -> None:
        """Count one more live session; start sampling into ``store`` on the first one.

        ``acquire`` and ``release`` can wait for the sampler thread: call them from a
        worker thread (``asyncio.to_thread``), not from the event loop.
        """
        with self._lock:
            self._live += 1
            if self._thread is not None or not self._resolve():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, args=(store, self._stop), name="system-metrics-sampler", daemon=True
            )
            self._thread.start()

    def release(self) -> None:
        """Count one live session less; stop sampling when none is left."""
        with self._lock:
            self._live = max(0, self._live - 1)
            if self._live or self._thread is None:
                return
            self._stop.set()
            # Wait for an in-flight sample so a source is never closed while it is read.
            self._thread.join(timeout=self._interval + 2.0)
            self._thread = None
            for source in self._sources or []:
                try:
                    source.close()
                except Exception as exc:
                    logger.debug(f"System metrics source {source.name} failed to close: {exc}")

    def _run(self, store, stop: threading.Event) -> None:
        while not stop.wait(self._interval):
            try:
                row = self.sample_once()
                if row is None:
                    continue
                # Sessions of other server workers count too (host-wide samples).
                live = store.open_session_count(now=row["ts"], stale_after_secs=_LIVE_STALE_SECS)
                row["live_sessions"] = max(row["live_sessions"], live)
                store.add_system_sample(row, min_gap_secs=self._interval / 2)
            except Exception as exc:  # never let sampling errors escape the thread
                logger.debug(f"System metrics sample failed: {exc}")


_sampler = SystemSampler()


def sampler() -> SystemSampler:
    """Return the process-wide sampler shared by every recorded session."""
    return _sampler
