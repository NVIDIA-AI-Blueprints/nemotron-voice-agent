# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment-driven configuration for conversation recording.

Environment variables:
  - MONITORING_ENABLED      Record sessions to the session store (default: false)
  - MONITORING_DATA_DIR     Root for the local DB and artifacts (default: <project_root>/data)
  - MONITORING_DB_URL       SQLAlchemy URL (default: sqlite:///<MONITORING_DATA_DIR>/voice_agent.db)
  - RECORD_AUDIO_TURNS      Per-turn user (16 kHz) and bot WAV clips (default: true)
  - RECORD_AUDIO_STEREO     Whole-call stereo WAV, user left / bot right (default: true)
  - RECORD_VIDEO            off | keyframes | full (default: off)
  - RECORD_VIDEO_FPS        Keyframe sampling rate for RECORD_VIDEO=keyframes (default: 1.0)
  - RECORD_SYSTEM_METRICS   Host GPU/CPU/RAM/power samples while a session is live (default: true)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from utils import PROJECT_ROOT, parse_env_bool, parse_env_float

VIDEO_MODES = ("off", "keyframes", "full")


@dataclass(frozen=True)
class MonitoringConfig:
    """Resolved recording settings."""

    enabled: bool
    data_dir: Path
    db_url: str
    record_audio_turns: bool
    record_audio_stereo: bool
    record_video: str
    video_fps: float
    # Off unless loaded from the environment, so configs built in code and tests do not sample the host.
    record_system_metrics: bool = False

    @property
    def artifacts_dir(self) -> Path:
        """Return the root directory of the local artifact store."""
        return self.data_dir / "artifacts"


def load_monitoring_config() -> MonitoringConfig:
    """Read the monitoring configuration from the environment."""
    raw_dir = Path(os.getenv("MONITORING_DATA_DIR", "") or "data")
    data_dir = raw_dir if raw_dir.is_absolute() else PROJECT_ROOT / raw_dir
    db_url = os.getenv("MONITORING_DB_URL", "") or f"sqlite:///{data_dir / 'voice_agent.db'}"
    record_video = (os.getenv("RECORD_VIDEO", "") or "off").strip().lower()
    if record_video not in VIDEO_MODES:
        record_video = "off"
    return MonitoringConfig(
        enabled=parse_env_bool("MONITORING_ENABLED", default=False),
        data_dir=data_dir,
        db_url=db_url,
        record_audio_turns=parse_env_bool("RECORD_AUDIO_TURNS", default=True),
        record_audio_stereo=parse_env_bool("RECORD_AUDIO_STEREO", default=True),
        record_video=record_video,
        video_fps=parse_env_float("RECORD_VIDEO_FPS", 1.0, min_value=0.1),
        record_system_metrics=parse_env_bool("RECORD_SYSTEM_METRICS", default=True),
    )
