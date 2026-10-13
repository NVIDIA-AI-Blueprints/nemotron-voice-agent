# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational schema of the session store (SQLAlchemy Core, dialect-neutral).

All timestamps are Unix epoch seconds (float). Binary payloads (audio, images,
video) never live in the database: ``media.artifact_key`` points into the
artifact store.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

SCHEMA_VERSION = 4

metadata = MetaData()

schema_version = Table(
    "schema_version",
    metadata,
    Column("version", Integer, primary_key=True),
)

sessions = Table(
    "sessions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("example", String(128)),
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float),
    Column("last_seen_at", Float),
    Column("end_reason", String(64)),
    # Snapshot of everything that defines the pipeline variant (models, recipe,
    # language, voice, turn detection, git sha, client tools): the A/B key.
    Column("config", JSON),
    Column("artifact_prefix", String(256)),
)

turns = Table(
    "turns",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("idx", Integer, primary_key=True),
    Column("user_text", Text),
    Column("user_started_at", Float),
    Column("user_stopped_at", Float),
    Column("bot_text", Text),
    Column("bot_started_at", Float),
    Column("bot_stopped_at", Float),
    # An interruption reached the assistant aggregator while its response was open
    # (LLM response start until its audio finished playing), or the session was
    # cancelled mid-response. Not a barge-in: see ``barge_in``.
    Column("interrupted", Boolean, default=False),
    Column("language", String(32)),
    # --- schema 4 -------------------------------------------------------------
    # ``user_stopped_at`` is when the turn was released to the LLM and
    # ``bot_started_at`` when the LLM response started. The two columns below are
    # the real speech times: end of user speech (VAD stop minus its silence window)
    # and first bot audio sent to the transport.
    Column("user_speech_stopped_at", Float),
    Column("bot_speech_started_at", Float),
    # The user started this turn while the bot was still speaking.
    Column("barge_in", Boolean),
)

# One row per user turn, derived at turn end by ``monitoring.turn_metrics`` (also
# the backfill). Stage columns split ``total_secs`` along the critical path to the
# first response; ``unexplained_secs`` is what no stage accounts for.
turn_metrics = Table(
    "turn_metrics",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("idx", Integer, primary_key=True),
    # plain | tool | intent | vision
    Column("kind", String(16), nullable=False),
    Column("barge_in", Boolean),
    # User stopped speaking -> first bot audio.
    Column("voice_latency", Float),
    # User stopped speaking -> first perceivable response (first bot audio or the
    # send time of the first perceivable tool call, whichever comes first).
    Column("response_latency", Float),
    # audio | tool: what the first perceivable response was.
    Column("response_via", String(16)),
    Column("turn_detection_secs", Float),
    Column("asr_secs", Float),
    Column("intent_match_secs", Float),
    Column("llm_first_secs", Float),
    Column("tool_secs", Float),
    Column("llm_later_secs", Float),
    Column("text_aggregation_secs", Float),
    Column("tts_secs", Float),
    Column("unexplained_secs", Float),
    Column("total_secs", Float),
    # [{"stage", "start", "end"}] in epoch seconds, for the session timeline.
    Column("segments", JSON),
    Column("n_llm_calls", Integer),
    Column("n_tool_calls", Integer),
    Column("n_images", Integer),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("gpu_load_mean", Float),
    Column("gpu_load_peak", Float),
    Column("computed_at", Float, nullable=False),
)

# One row per tool or intent call.
tool_calls = Table(
    "tool_calls",
    metadata,
    Column("session_id", String(64), primary_key=True),
    # The tool_call_id, or ``intent-ha-<event id>`` for Home Assistant intent calls.
    Column("call_id", String(128), primary_key=True),
    Column("turn_idx", Integer),
    Column("name", String(128), nullable=False),
    # llm | intent
    Column("trigger", String(16), nullable=False),
    # client | home_assistant
    Column("target", String(32), nullable=False),
    # Server send time.
    Column("sent_at", Float, nullable=False),
    Column("duration_secs", Float),
    # ok | error | timeout | cancelled
    Column("outcome", String(16), nullable=False),
    Column("perceivable", Boolean, nullable=False, default=False),
    Index("ix_tool_calls_turn", "session_id", "turn_idx"),
)

# Host-wide samples taken once per second while at least one session is live.
system_samples = Table(
    "system_samples",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", Float, nullable=False),
    Column("live_sessions", Integer, nullable=False),
    Column("gpu_load", Float),  # percent
    Column("gpu_temp_c", Float),
    Column("power_w", Float),
    # What ``power_w`` measures: ``board`` (whole-module input power, Jetson INA238)
    # or ``gpu`` (GPU power draw reported by nvidia-smi).
    Column("power_source", String(16)),
    Column("ram_used_mb", Float),
    Column("cpu_load", Float),  # percent of all cores
    Index("ix_system_samples_ts", "ts"),
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    Column("kind", String(64), nullable=False),
    Column("processor", String(128)),
    Column("data", JSON),
    Index("ix_events_session", "session_id", "ts"),
)

metrics = Table(
    "metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    Column("processor", String(128)),
    Column("model", String(256)),
    Column("name", String(128), nullable=False),
    Column("value", Float),
    Index("ix_metrics_session", "session_id", "name"),
)

media = Table(
    "media",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("ts", Float, nullable=False),
    # audio_user | audio_bot | audio_conversation | image | video_keyframe | video_chunk
    Column("modality", String(32), nullable=False),
    # camera | context | tool:<name> | pipeline | ...
    Column("source", String(128)),
    # Free-form link to what produced it (tool_call_id, llm call id, ...).
    Column("ref", String(128)),
    Column("sha256", String(64), nullable=False),
    Column("mime", String(64)),
    Column("width", Integer),
    Column("height", Integer),
    Column("sample_rate", Integer),
    Column("duration_secs", Float),
    Column("artifact_key", String(512), nullable=False),
    UniqueConstraint("session_id", "modality", "sha256", name="uq_media_content"),
    Index("ix_media_session", "session_id", "modality"),
)

llm_calls = Table(
    "llm_calls",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("started_at", Float, nullable=False),
    Column("ended_at", Float),
    Column("processor", String(128)),
    Column("model", String(256)),
    # Full LLM input; inline images are replaced by {"type": "media_ref", "sha256": ...}.
    Column("messages", JSON),
    Column("tools", JSON),
    Column("output_text", Text),
    Column("function_calls", JSON),
    Column("ttfb", Float),
    Column("prompt_tokens", Integer),
    Column("completion_tokens", Integer),
    Column("n_images", Integer, default=0),
    Column("image_pixels", Integer, default=0),
    Column("interrupted", Boolean, default=False),
    Index("ix_llm_calls_session", "session_id", "turn_idx"),
)

annotations = Table(
    "annotations",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(64), nullable=False),
    # session | turn | media | llm_call
    Column("target_type", String(32), nullable=False),
    Column("target_id", String(128), nullable=False),
    # Model name for automatic annotations, ``human:<name>`` for human labels.
    Column("source", String(256), nullable=False),
    Column("source_version", String(128)),
    # transcript | wer | safety | memory | ...
    Column("kind", String(64), nullable=False),
    Column("value", JSON),
    Column("created_at", Float, nullable=False),
    Index("ix_annotations_target", "target_type", "target_id", "kind"),
    Index("ix_annotations_session", "session_id", "kind"),
)

jobs = Table(
    "jobs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("kind", String(64), nullable=False),
    Column("target", String(128), nullable=False),
    # pending | running | done | failed | cancelled
    Column("status", String(16), nullable=False, default="pending"),
    Column("params", JSON),
    Column("progress", JSON),
    Column("attempts", Integer, nullable=False, default=0),
    Column("error", Text),
    Column("created_at", Float, nullable=False),
    Column("started_at", Float),
    Column("finished_at", Float),
    UniqueConstraint("kind", "target", name="uq_jobs_kind_target"),
    Index("ix_jobs_status", "status", "created_at"),
)

# Small shared state between processes (dreamer status heartbeat, pause flag).
kv = Table(
    "kv",
    metadata,
    Column("key", String(128), primary_key=True),
    Column("value", JSON),
    Column("updated_at", Float, nullable=False),
)


# ----------------------------------------------------------------- people
# People the agent talks to. Until voice identification exists, a session is
# attributed to the person picked in the live client, and review can reassign
# whole sessions or single turns (the per-turn rows are voice-ID enrollment data).
people = Table(
    "people",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("name", String(128), nullable=False),
    Column("created_at", Float, nullable=False),
    Column("archived", Boolean, nullable=False, default=False),
)

speaker_assignments = Table(
    "speaker_assignments",
    metadata,
    Column("session_id", String(64), primary_key=True),
    # -1 = the whole session; a turn row overrides it for that turn.
    Column("turn_idx", Integer, primary_key=True),
    Column("person_id", String(32), nullable=False),
    # ``live:picker``, ``live:voice-id:<model key>`` or ``human:<name>``.
    Column("source", String(256), nullable=False),
    Column("created_at", Float, nullable=False),
    Index("ix_speaker_person", "person_id"),
)

# Voice-ID enrollment data: speaker embeddings computed by the live client.
# Vectors are L2-normalised float32 (little-endian) and only comparable within
# one ``model`` key, so every read filters on it.
voice_embeddings = Table(
    "voice_embeddings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("person_id", String(32), nullable=False),
    Column("model", String(128), nullable=False),
    Column("dim", Integer, nullable=False),
    Column("vector", LargeBinary, nullable=False),
    # ``live:enroll`` (the ``enroll_speaker`` tool).
    Column("source", String(256), nullable=False),
    Column("session_id", String(64)),
    Column("created_at", Float, nullable=False),
    Index("ix_voice_embeddings_model_person", "model", "person_id"),
)

# Facts about a person extracted by the ``dream`` job (or written by a reviewer).
# Rows are never deleted: forgetting and superseding are status changes.
memories = Table(
    "memories",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("person_id", String(32), nullable=False),
    Column("text", Text, nullable=False),
    # identity | preference | relationship | routine | event | health | other
    Column("category", String(32)),
    Column("language", String(32)),
    Column("confidence", Float),
    # proposed (waiting for review) | active (used live) | forgotten | superseded
    Column("status", String(16), nullable=False),
    # Model name for extracted memories, ``human:<name>`` for corrections.
    Column("source", String(256), nullable=False),
    Column("source_version", String(256)),
    # Memory this one replaces once approved or used.
    Column("supersedes", Integer),
    Column("superseded_by", Integer),
    Column("reviewed_by", String(256)),
    Column("reviewed_at", Float),
    Column("created_at", Float, nullable=False),
    Column("updated_at", Float, nullable=False),
    Index("ix_memories_person", "person_id", "status"),
)

memory_evidence = Table(
    "memory_evidence",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("memory_id", Integer, nullable=False),
    Column("session_id", String(64), nullable=False),
    Column("turn_idx", Integer),
    Column("quote", Text),
    Index("ix_memory_evidence_memory", "memory_id"),
    Index("ix_memory_evidence_session", "session_id"),
)

# Memories injected into a live session prompt.
memory_uses = Table(
    "memory_uses",
    metadata,
    Column("memory_id", Integer, primary_key=True),
    Column("session_id", String(64), primary_key=True),
    Column("ts", Float, nullable=False),
    Index("ix_memory_uses_session", "session_id"),
)
