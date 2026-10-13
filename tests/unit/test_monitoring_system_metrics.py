# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D100, D101, D102, D103, D107

import threading
import time

from monitoring.store import SessionStore
from monitoring.system_metrics import (
    HwmonPowerSource,
    NvidiaSmiSource,
    ProcMeminfoSource,
    ProcStatSource,
    SystemSampler,
    ThermalZoneSource,
)


class FakeSource:
    def __init__(self, name, values, *, available=True, fail=False):
        self.name, self.values, self._available, self.fail = name, values, available, fail
        self.reads = self.closed = 0
        self.threads = set()

    def available(self):
        return self._available

    def read(self):
        self.reads += 1
        self.threads.add(threading.current_thread().name)
        if self.fail:
            raise OSError("gone")
        return dict(self.values)

    def close(self):
        self.closed += 1


def test_sample_merges_sources_in_priority_order_and_skips_unavailable_ones():
    sysfs = FakeSource("sysfs", {"gpu_temp_c": 43.0, "power_w": 21.0, "power_source": "board", "not_a_field": 1.0})
    smi = FakeSource("nvidia-smi", {"gpu_load": 37.0, "gpu_temp_c": 99.0, "power_w": 2.0, "power_source": "gpu"})
    missing = FakeSource("tegrastats", {"gpu_load": 1.0}, available=False)
    sampler = SystemSampler([sysfs, missing, smi])
    row = sampler.sample_once(now=12.5)
    assert row == {
        "ts": 12.5,
        "live_sessions": 0,
        "gpu_load": 37.0,
        "gpu_temp_c": 43.0,  # the earlier source wins
        "power_w": 21.0,
        "power_source": "board",  # the label follows the source that provided the value
        "ram_used_mb": None,
        "cpu_load": None,
    }
    assert missing.reads == 0


def test_power_label_follows_the_source_actually_used():
    # A board monitor that reports no value this time: the fallback's value keeps its own label.
    board = FakeSource("hwmon", {"power_source": "board"})
    smi = FakeSource("nvidia-smi", {"gpu_load": 5.0, "power_w": 2.0, "power_source": "gpu"})
    row = SystemSampler([smi]).sample_once(now=1.0)
    assert (row["power_w"], row["power_source"]) == (2.0, "gpu")
    row = SystemSampler([FakeSource("cpu", {"cpu_load": 1.0})]).sample_once(now=1.0)
    assert (row["power_w"], row["power_source"]) == (None, None)
    row = SystemSampler([board, smi]).sample_once(now=1.0)
    assert (row["power_w"], row["power_source"]) == (2.0, "gpu")


def test_nvidia_smi_that_exits_is_dropped_quietly():
    source = NvidiaSmiSource(executable="false")  # present on PATH, but produces nothing and exits
    assert source.available()
    sampler = SystemSampler([source, FakeSource("cpu", {"cpu_load": 3.0})])
    rows = []
    for _ in range(12):
        rows.append(sampler.sample_once())
        time.sleep(0.02)
    assert all(row["cpu_load"] == 3.0 and row["gpu_load"] is None for row in rows)
    assert source not in sampler._sources  # disabled after repeated errors, the other sources keep working
    source.close()


def test_sampler_without_any_source_is_disabled():
    sampler = SystemSampler([FakeSource("nvidia-smi", {}, available=False)])
    assert sampler.sample_once() is None
    sampler.acquire(store=None)  # cloud profile, or a container without the tools: nothing starts
    assert sampler.live_sessions == 1 and sampler._thread is None
    sampler.release()
    assert sampler.live_sessions == 0


def test_failing_source_is_dropped_after_repeated_errors():
    broken, good = FakeSource("broken", {}, fail=True), FakeSource("good", {"cpu_load": 5.0})
    sampler = SystemSampler([broken, good])
    for _ in range(8):
        assert sampler.sample_once()["cpu_load"] == 5.0
    assert broken.reads == 5 and good.reads == 8


def test_sampler_runs_only_while_sessions_are_live(tmp_path):
    store = SessionStore(f"sqlite:///{tmp_path / 'db.sqlite'}")
    store.create_schema()
    source = FakeSource("fake", {"gpu_load": 50.0, "cpu_load": 10.0})
    sampler = SystemSampler([source], interval_secs=0.02)

    sampler.acquire(store)
    sampler.acquire(store)  # a second concurrent session shares the sampler
    deadline = time.time() + 5
    while len(store.system_samples_between(0, time.time() + 1)) < 3 and time.time() < deadline:
        time.sleep(0.02)
    sampler.release()
    assert sampler._thread is not None and source.closed == 0  # one session is still live
    sampler.release()
    assert sampler._thread is None and source.closed == 1

    samples = store.system_samples_between(0, time.time() + 1)
    assert len(samples) >= 3 and samples[0]["gpu_load"] == 50.0 and samples[0]["live_sessions"] == 2
    assert source.threads == {"system-metrics-sampler"}  # never read from the caller's (event loop) thread
    time.sleep(0.1)
    assert len(store.system_samples_between(0, time.time() + 1)) == len(samples)  # stopped

    sampler.acquire(store)  # the next session starts it again
    assert sampler._thread is not None
    sampler.release()


def test_proc_sources(tmp_path):
    stat = tmp_path / "stat"
    stat.write_text("cpu  100 0 100 700 100 0 0 0 0 0\ncpu0 1 2 3 4\n")
    cpu = ProcStatSource(stat)
    assert cpu.available() and cpu.read() == {}  # needs two readings
    stat.write_text("cpu  150 0 150 750 150 0 0 0 0 0\n")
    assert cpu.read() == {"cpu_load": 50.0}  # 100 busy out of 200 jiffies (iowait counts as idle)

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       2048000 kB\nMemFree:  1 kB\nMemAvailable:   1024000 kB\n")
    assert ProcMeminfoSource(meminfo).read() == {"ram_used_mb": 1000.0}
    assert not ProcStatSource(tmp_path / "missing").available()


def test_sysfs_sources(tmp_path):
    for index, (kind, temp) in enumerate([("cpu-thermal", 40000), ("gpu-thermal", 43250)]):
        zone = tmp_path / "thermal" / f"thermal_zone{index}"
        zone.mkdir(parents=True)
        (zone / "type").write_text(f"{kind}\n")
        (zone / "temp").write_text(f"{temp}\n")
    thermal = ThermalZoneSource(tmp_path / "thermal")
    assert thermal.available() and thermal.read() == {"gpu_temp_c": 43.2}
    assert not ThermalZoneSource(tmp_path / "nothing").available()

    for index, (name, has_power) in enumerate([("nvme", False), ("ina238", True)]):
        device = tmp_path / "hwmon" / f"hwmon{index}"
        device.mkdir(parents=True)
        (device / "name").write_text(f"{name}\n")
        if has_power:
            (device / "power1_input").write_text("21188000\n")
    power = HwmonPowerSource(tmp_path / "hwmon")
    assert power.available() and power.read() == {"power_w": 21.19, "power_source": "board"}
    assert not HwmonPowerSource(tmp_path / "nothing").available()


def test_nvidia_smi_line_parsing():
    assert NvidiaSmiSource.parse("37, 43, 1.98\n") == {
        "gpu_load": 37.0,
        "gpu_temp_c": 43.0,
        "power_w": 1.98,
        "power_source": "gpu",
    }
    assert NvidiaSmiSource.parse("0, [N/A], [N/A]") == {"gpu_load": 0.0}
    assert NvidiaSmiSource.parse("garbage") == {}
    assert not NvidiaSmiSource(executable="definitely-not-installed").available()
