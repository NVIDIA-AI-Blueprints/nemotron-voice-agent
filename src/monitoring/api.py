# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review API: browse recorded sessions and run human review activities.

Mounted by ``server.py`` under ``/api/review`` when ``MONITORING_ENABLED`` is
true. Everything here is CPU-only (DB reads, file serving, WER rescoring):
GPU work is only *queued* for the dreamer, which still waits for idle.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from monitoring import memories, schema, voice_id
from monitoring.config import MonitoringConfig
from monitoring.jobs import JOB_REGISTRY
from monitoring.jobs.reasr import live_source_name, reasr_endpoints, score_session
from monitoring.jobs.runner import PAUSE_KEY, STATUS_KEY, declared_services, load_dreamer_config
from monitoring.jobs.wer import error_rates, normalize
from monitoring.report import DEFAULT_GROUP_BY, metrics_json
from monitoring.store import ArtifactStore, SessionStore

_ANNOTATOR_RE = re.compile(r"^[\w .@-]{1,64}$")


class ReferenceIn(BaseModel):
    """A human ASR reference for one user turn."""

    session_id: str
    turn_idx: int
    text: str = Field(max_length=4000)


class ReferenceBatchIn(BaseModel):
    """Human references saved by one annotator."""

    annotator: str
    items: list[ReferenceIn] = Field(min_length=1, max_length=500)


class PauseIn(BaseModel):
    """Pause or resume the dreamer."""

    paused: bool


class EnqueueAllIn(BaseModel):
    """Queue a job kind for every ended session that does not have it yet."""

    kind: str


class JobsIn(BaseModel):
    """(Re)queue a job for sessions."""

    kind: str
    session_ids: list[str] = Field(min_length=1, max_length=1000)


class PersonIn(BaseModel):
    """Create a person."""

    name: str = Field(min_length=1, max_length=128)


class PersonPatchIn(BaseModel):
    """Rename or (un)archive a person."""

    name: str | None = Field(None, min_length=1, max_length=128)
    archived: bool | None = None


class MergeIn(BaseModel):
    """Merge ``source_id`` into the person in the path, who keeps their name."""

    annotator: str
    source_id: str = Field(min_length=1, max_length=32)


class ForgetIdentityIn(BaseModel):
    """Delete a person's voice or face samples for one model key (all keys when omitted)."""

    annotator: str
    model: str | None = Field(None, min_length=1, max_length=128)


class SpeakerIn(BaseModel):
    """Attribute a session (``turn_idx`` omitted) or one turn to a person (``None`` clears it)."""

    annotator: str
    person_id: str | None = None
    turn_idx: int | None = Field(None, ge=0)


class MemoryReviewIn(BaseModel):
    """Review one memory: approve, correct (new text and/or person) or forget."""

    annotator: str
    action: str = Field(pattern="^(approve|correct|forget)$")
    text: str | None = Field(None, max_length=500)
    person_id: str | None = None


def _turn_target(session_id: str, turn_idx: int) -> str:
    return f"{session_id}:{turn_idx}"


def _split_target(target: str) -> tuple[str, int]:
    session_id, _, idx = target.rpartition(":")
    return session_id, int(idx)


def _session_summary(row: dict[str, Any]) -> dict[str, Any]:
    config = row.get("config") or {}
    return {
        "id": row["id"],
        "example": row["example"],
        "started_at": row["started_at"],
        "ended_at": row["ended_at"],
        "end_reason": row["end_reason"],
        "language": config.get("language"),
        "person": config.get("person"),
        "models": {
            "asr": (config.get("asr") or {}).get("model"),
            "llm": (config.get("llm") or {}).get("model"),
            "tts": (config.get("tts") or {}).get("model"),
            "voice": (config.get("tts") or {}).get("voice"),
        },
    }


class ReviewService:
    """Queries and actions behind the review endpoints (synchronous, thread-safe)."""

    def __init__(self, store: SessionStore, artifacts: ArtifactStore, dreamer_config: dict[str, Any]):
        """Bind to the stores; ``dreamer_config`` names the reasr reference/candidates."""
        self.store = store
        self.artifacts = artifacts
        self.dreamer_config = dreamer_config

    # ------------------------------------------------------------- helpers
    @property
    def _reference_name(self) -> str | None:
        reference, _ = reasr_endpoints(self.dreamer_config)
        return reference.name if reference else None

    @property
    def _candidate_names(self) -> list[str]:
        return [c.name for c in reasr_endpoints(self.dreamer_config)[1]]

    def _rows(self, stmt) -> list[dict[str, Any]]:
        with self.store.engine.connect() as conn:
            return [dict(r) for r in conn.execute(stmt).mappings()]

    # ------------------------------------------------------------ sessions
    def list_sessions(self, *, limit: int, offset: int) -> dict[str, Any]:
        """Most recent sessions with turn counts, review progress, WER and job states."""
        s, t, m, a, j = schema.sessions, schema.turns, schema.media, schema.annotations, schema.jobs
        sessions = self._rows(select(s).order_by(s.c.started_at.desc()).limit(limit).offset(offset))
        ids = [row["id"] for row in sessions]
        with self.store.engine.connect() as conn:
            total = conn.execute(select(func.count()).select_from(s)).scalar_one()
        if not ids:
            return {"total": total, "sessions": []}
        user_turns = dict(
            self._rows_pairs(
                select(t.c.session_id, func.count())
                .where(t.c.session_id.in_(ids), t.c.user_text.is_not(None), t.c.user_text != "")
                .group_by(t.c.session_id)
            )
        )
        audio_turns = dict(
            self._rows_pairs(
                select(m.c.session_id, func.count(func.distinct(m.c.turn_idx)))
                .where(m.c.session_id.in_(ids), m.c.modality == "audio_user")
                .group_by(m.c.session_id)
            )
        )
        reviewed = dict(
            self._rows_pairs(
                select(a.c.session_id, func.count(func.distinct(a.c.target_id)))
                .where(a.c.session_id.in_(ids), a.c.kind == "transcript", a.c.source.like("human:%"))
                .group_by(a.c.session_id)
            )
        )
        wer: dict[str, dict[str, float | None]] = defaultdict(dict)
        for row in self._rows(
            select(a.c.session_id, a.c.source, a.c.value).where(a.c.session_id.in_(ids), a.c.kind == "wer_summary")
        ):
            wer[row["session_id"]][row["source"]] = (row["value"] or {}).get("wer")
        jobs: dict[str, dict[str, str]] = defaultdict(dict)
        for row in self._rows(select(j.c.target, j.c.kind, j.c.status).where(j.c.target.in_(ids))):
            jobs[row["target"]][row["kind"]] = row["status"]
        # Current attribution (review can reassign the person picked live).
        sa, p = schema.speaker_assignments, schema.people
        persons = {
            row["session_id"]: {"id": row["id"], "name": row["name"]}
            for row in self._rows(
                select(sa.c.session_id, p.c.id, p.c.name)
                .join(p, p.c.id == sa.c.person_id)
                .where(sa.c.session_id.in_(ids), sa.c.turn_idx == memories.SESSION_TURN)
            )
        }
        return {
            "total": total,
            "sessions": [
                {
                    **_session_summary(row),
                    "person": persons.get(row["id"]),
                    "user_turns": user_turns.get(row["id"], 0),
                    "audio_turns": audio_turns.get(row["id"], 0),
                    "reviewed_turns": reviewed.get(row["id"], 0),
                    "wer": wer.get(row["id"], {}),
                    "jobs": jobs.get(row["id"], {}),
                }
                for row in sessions
            ],
        }

    def _rows_pairs(self, stmt) -> list[tuple[Any, Any]]:
        with self.store.engine.connect() as conn:
            return [tuple(r) for r in conn.execute(stmt)]

    def session_detail(self, session_id: str) -> dict[str, Any]:
        """Everything the session explorer shows, grouped per turn."""
        session = self.store.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        live_source = live_source_name(session.get("config") or {})
        turns = {
            row["idx"]: {**row, "audio": {"user": [], "bot": []}, "images": [], "llm_calls": []}
            for row in self.store.rows("turns", session_id)
        }

        def turn(idx: int | None) -> dict[str, Any] | None:
            if idx is None:
                return None
            return turns.setdefault(idx, {"idx": idx, "audio": {"user": [], "bot": []}, "images": [], "llm_calls": []})

        conversation_audio = None
        for media in self.store.rows("media", session_id):
            entry = {
                "id": media["id"],
                "key": media["artifact_key"],
                "mime": media["mime"],
                "duration_secs": media["duration_secs"],
                "source": media["source"],
                "ts": media["ts"],
            }
            if media["modality"] == "audio_conversation":
                conversation_audio = entry
            elif media["modality"] in ("audio_user", "audio_bot"):
                target = turn(media["turn_idx"])
                if target is not None:
                    target["audio"][media["modality"].removeprefix("audio_")].append(entry)
            elif media["modality"] in ("image", "video_keyframe"):
                target = turn(media["turn_idx"])
                if target is not None:
                    target["images"].append({**entry, "width": media["width"], "height": media["height"]})

        for call in self.store.rows("llm_calls", session_id):
            target = turn(call["turn_idx"])
            if target is not None:
                target["llm_calls"].append(
                    {
                        key: call[key]
                        for key in (
                            "id",
                            "started_at",
                            "model",
                            "ttfb",
                            "prompt_tokens",
                            "completion_tokens",
                            "n_images",
                            "interrupted",
                            "output_text",
                            "function_calls",
                        )
                    }
                )

        turn_metrics = {
            row["idx"]: {k: v for k, v in row.items() if k != "session_id"}
            for row in self.store.rows("turn_metrics", session_id)
        }
        tool_calls: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for call in sorted(self.store.rows("tool_calls", session_id), key=lambda c: c["sent_at"]):
            tool_calls[call["turn_idx"]].append({k: v for k, v in call.items() if k != "session_id"})

        latency: dict[int, float] = {}
        for metric in self.store.rows("metrics", session_id, name="user_bot_latency"):
            if metric["turn_idx"] is not None:
                latency.setdefault(metric["turn_idx"], metric["value"])

        transcripts = self._transcripts_by_turn(session_id)
        wer_by_turn: dict[int, dict[str, Any]] = defaultdict(dict)
        wer_summary = {}
        for annotation in self.store.annotations_for(session_id):
            if annotation["kind"] == "wer":
                wer_by_turn[_split_target(annotation["target_id"])[1]][annotation["source"]] = annotation["value"]
            elif annotation["kind"] == "wer_summary":
                wer_summary[annotation["source"]] = annotation["value"]

        out_turns = []
        for idx in sorted(turns):
            row = turns[idx]
            by_source = transcripts.get(idx, {})
            if row.get("user_text") and row["audio"]["user"]:
                by_source = {live_source: {"text": row["user_text"], "at": 0.0}, **by_source}
            human = self._latest_human(by_source)
            out_turns.append(
                {
                    **row,
                    "transcripts": {k: v["text"] for k, v in by_source.items()},
                    "human_reference": human,
                    "wer": wer_by_turn.get(idx, {}),
                    "user_bot_latency": latency.get(idx),
                    "metrics": turn_metrics.get(idx),
                    "tool_calls": tool_calls.get(idx, []),
                }
            )
        jobs = [job for job in self.store.jobs() if job["target"] == session_id]
        speakers = memories.speakers_for(self.store, session_id)
        for turn_row in out_turns:
            turn_row["speaker"] = speakers["turns"].get(turn_row["idx"])
        return {
            "session": {**_session_summary(session), "config": session.get("config")},
            "speakers": {"session": speakers["session"]},
            "memories_used": self._session_memories(session_id),
            "live_source": live_source,
            "reference_source": self._reference_name,
            "conversation_audio": conversation_audio,
            "turns": out_turns,
            "system_samples": self._system_samples(session),
            "wer_summary": wer_summary,
            "jobs": [{k: job[k] for k in ("kind", "status", "error", "attempts", "finished_at")} for job in jobs],
        }

    def _system_samples(self, session: dict[str, Any], *, max_points: int = 1800) -> list[dict[str, Any]]:
        """Host-wide samples over the session, thinned to ``max_points`` for long sessions."""
        end = session.get("ended_at") or time.time()
        samples = self.store.system_samples_between(session["started_at"], end)
        step = max(1, -(-len(samples) // max_points))
        return [{k: v for k, v in sample.items() if k != "id"} for sample in samples[::step]]

    def _session_memories(self, session_id: str) -> dict[str, Any]:
        """Memories injected in this session and those extracted from it."""
        u, e, m = schema.memory_uses, schema.memory_evidence, schema.memories
        used = self._rows(select(m).join(u, u.c.memory_id == m.c.id).where(u.c.session_id == session_id))
        extracted = self._rows(
            select(m).where(m.c.id.in_(select(e.c.memory_id).where(e.c.session_id == session_id))).order_by(m.c.id)
        )
        return {"used": used, "extracted": extracted}

    def llm_call(self, session_id: str, call_id: int) -> dict[str, Any]:
        """Full stored LLM input/output of one call."""
        calls = self.store.rows("llm_calls", session_id, id=call_id)
        if not calls:
            raise KeyError(call_id)
        return calls[0]

    # -------------------------------------------------------------- ASR queue
    def _transcripts_by_turn(self, session_id: str) -> dict[int, dict[str, dict[str, Any]]]:
        """``{turn_idx: {source: {"text", "at"}}}`` keeping the latest per source."""
        out: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in self.store.annotations_for(session_id, kind="transcript"):
            _, idx = _split_target(row["target_id"])
            out[idx][row["source"]] = {"text": (row["value"] or {}).get("text", ""), "at": row["created_at"]}
        return out

    @staticmethod
    def _latest_human(by_source: dict[str, dict[str, Any] | str]) -> dict[str, Any] | None:
        humans = [
            (value["at"], source, value["text"])
            for source, value in by_source.items()
            if source.startswith("human:") and isinstance(value, dict)
        ]
        if not humans:
            return None
        at, source, text = max(humans)
        return {"source": source, "text": text, "at": at}

    def asr_queue(
        self, *, status: str, limit: int, session_id: str | None, sort: str = "recent", offset: int = 0
    ) -> dict[str, Any]:
        """User turns with audio, most recent first (or most live-vs-reference disagreement first)."""
        reference = self._reference_name
        session_ids = [session_id] if session_id else self.store.ended_session_ids()
        items = []
        for sid in session_ids:
            session = self.store.get_session(sid)
            if session is None:
                continue
            config = session.get("config") or {}
            live_source = live_source_name(config)
            audio: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for media in self.store.rows("media", sid, modality="audio_user"):
                audio[media["turn_idx"]].append({"key": media["artifact_key"], "duration_secs": media["duration_secs"]})
            if not audio:
                continue
            transcripts = self._transcripts_by_turn(sid)
            for turn in self.store.rows("turns", sid):
                idx = turn["idx"]
                if idx not in audio:
                    continue
                by_source = transcripts.get(idx, {})
                human = self._latest_human(by_source)
                models = {src: v["text"] for src, v in by_source.items() if not src.startswith("human:")}
                live_text = turn["user_text"] or ""
                ref_text = models.get(reference) if reference else None
                disagreement = error_rates(ref_text, live_text)["wer"] if ref_text is not None else None
                items.append(
                    {
                        "session_id": sid,
                        "turn_idx": idx,
                        "started_at": turn["user_started_at"] or session["started_at"],
                        "language": turn["language"] or config.get("language"),
                        "audio": audio[idx],
                        "live": {"source": live_source, "text": live_text},
                        "models": models,
                        "reference_source": reference,
                        "human_reference": human,
                        "disagreement": disagreement,
                        "agree": ref_text is not None and normalize(ref_text) == normalize(live_text),
                    }
                )
        # Counts cover every turn; ``status`` only filters the returned items.
        counts = {
            "total": len(items),
            "open": sum(1 for i in items if not i["human_reference"]),
            "agreeing": sum(1 for i in items if i["agree"] and not i["human_reference"]),
        }
        if status == "open":
            items = [i for i in items if not i["human_reference"]]
        elif status == "reviewed":
            items = [i for i in items if i["human_reference"]]
        if sort == "disagreement":
            # Known disagreement first (largest first), then turns still waiting for the reference model.
            items.sort(key=lambda i: (i["disagreement"] is None, -(i["disagreement"] or 0.0), -i["started_at"]))
        else:
            items.sort(key=lambda i: (-i["started_at"], -i["turn_idx"]))
        return {**counts, "matching": len(items), "items": items[offset : offset + limit]}

    def save_references(self, annotator: str, items: list[ReferenceIn]) -> dict[str, Any]:
        """Store human references and rescore the affected sessions inline."""
        rows = [
            {
                "session_id": item.session_id,
                "target_type": "turn",
                "target_id": _turn_target(item.session_id, item.turn_idx),
                "source": f"human:{annotator}",
                "source_version": None,
                "kind": "transcript",
                "value": {"text": item.text.strip()},
            }
            for item in items
        ]
        self.store.add_annotations(rows)
        summaries = {}
        for sid in sorted({item.session_id for item in items}):
            summaries[sid] = {
                s["source"]: s["value"]
                for s in score_session(
                    self.store, sid, reference_name=self._reference_name, candidate_names=self._candidate_names
                )
            }
        return {"saved": len(rows), "wer_summary": summaries}

    # ------------------------------------------------------------- dreamer
    def dreamer_status(self) -> dict[str, Any]:
        """Idle/live state, job counts, running and failed jobs."""
        now = time.time()
        live = self.store.open_session_count(now=now, stale_after_secs=self.dreamer_config["live_stale_secs"])
        last = self.store.last_session_activity()
        jobs = self.store.jobs()
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for job in jobs:
            counts[job["kind"]][job["status"]] += 1

        def brief(job):
            return {k: job[k] for k in ("id", "kind", "target", "status", "attempts", "error", "progress")}

        status_row = self.store.get_kv(STATUS_KEY)
        worker = dict(status_row["value"] or {}) if status_row else None
        if worker is not None:
            # Heartbeats come every ~10 s (also mid-job); allow a few missed polls.
            stale_after = max(3 * float(worker.get("poll_secs") or 10.0), 45.0)
            worker["alive"] = now - float(worker.get("heartbeat_at") or 0.0) <= stale_after
        pause_row = self.store.get_kv(PAUSE_KEY)
        pause = (pause_row["value"] or {}) if pause_row else {}
        reported = (worker or {}).get("services") or {}
        return {
            "live_sessions": live,
            "idle": live == 0 and (last is None or now - last >= self.dreamer_config["idle_grace_secs"]),
            "last_activity": last,
            "paused": bool(pause.get("paused")),
            "paused_at": pause_row["updated_at"] if pause_row and pause.get("paused") else None,
            "worker": worker,
            "services": [
                {"name": name, "state": reported.get(name, "unknown")}
                for name in declared_services(self.dreamer_config)
            ],
            "job_kinds": sorted(JOB_REGISTRY),
            "counts": {kind: dict(v) for kind, v in counts.items()},
            "running": [brief(j) for j in jobs if j["status"] == "running"],
            "failed": [brief(j) for j in jobs if j["status"] == "failed" or (j["error"] and j["status"] != "done")],
            "pending": sum(1 for j in jobs if j["status"] == "pending"),
        }

    def set_paused(self, paused: bool) -> dict[str, Any]:
        """Pause (the running job stops at its next checkpoint) or resume the dreamer."""
        self.store.set_kv(PAUSE_KEY, {"paused": paused})
        return {"paused": paused}

    def list_jobs(self, *, status: str | None, kind: str | None, limit: int, offset: int) -> dict[str, Any]:
        """Job queue page, newest first."""
        total, jobs = self.store.list_jobs(status=status, kind=kind, limit=limit, offset=offset)
        keys = ("id", "kind", "target", "status", "attempts", "error", "progress", "created_at", "started_at")
        return {"total": total, "jobs": [{k: job[k] for k in (*keys, "finished_at")} for job in jobs]}

    def cancel_job(self, job_id: int) -> bool:
        """Cancel a pending job."""
        return self.store.cancel_job(job_id)

    def enqueue_all(self, kind: str) -> int:
        """Queue ``kind`` for every ended session lacking it (existing runs are kept)."""
        return sum(self.store.enqueue_job(kind, sid) for sid in self.store.ended_session_ids())

    def metrics(self, *, since_days: float | None, group_by: list[str]) -> dict[str, Any]:
        """Variant comparison for the metrics charts."""
        since = time.time() - since_days * 86400 if since_days else None
        return metrics_json(self.store, since=since, group_by=group_by)

    def requeue(self, kind: str, session_ids: list[str]) -> int:
        """Queue ``kind`` from scratch for sessions (the dreamer still waits for idle)."""
        for sid in session_ids:
            self.store.reset_job(kind, sid)
        return len(session_ids)

    def activities(self) -> list[dict[str, Any]]:
        """Review activities available, with their open item counts."""
        queue = self.asr_queue(status="open", limit=0, session_id=None)
        return [
            {
                "id": "asr",
                "title": "ASR reference",
                "job": "reasr",
                "description": "Set the human reference transcript of user turns; rescored instantly.",
                "open": queue["open"],
            },
            {
                "id": "memories",
                "title": "Memories",
                "job": "dream",
                "description": "Audit what the agent remembers about people: approve, correct or forget.",
                "open": memories.open_memory_count(self.store),
            },
        ]

    # --------------------------------------------------------------- people
    def people(self, *, include_archived: bool) -> list[dict[str, Any]]:
        """People with session, memory, voice-ID turn and identity sample counts."""
        samples = voice_id.identity_counts(self.store)
        return [
            {**person, "identity_samples": samples.get(person["id"], {})}
            for person in memories.person_summaries(self.store, include_archived=include_archived)
        ]

    def person_identity(self, person_id: str) -> dict[str, Any]:
        """Voice and face samples of one person, and the people they may be a duplicate of."""
        if memories.get_person(self.store, person_id) is None:
            raise KeyError(person_id)
        return {
            "models": voice_id.identity_summary(self.store, person_id),
            "duplicates": voice_id.duplicate_candidates(self.store, person_id=person_id),
        }

    def duplicate_people(self) -> list[dict[str, Any]]:
        """Pairs of people who may be the same person."""
        return voice_id.duplicate_candidates(self.store)

    def merge_people(self, target_id: str, source_id: str, annotator: str) -> dict[str, Any]:
        """Merge ``source_id`` into ``target_id``."""
        merged = memories.merge_people(self.store, source_id, target_id, annotator=annotator)
        logger.info(f"{annotator} merged person {source_id} into {target_id}: {merged['moved']}")
        return merged

    def forget_identity(self, person_id: str, model: str | None, annotator: str) -> dict[str, Any]:
        """Delete a person's voice/face samples."""
        if memories.get_person(self.store, person_id) is None:
            raise KeyError(person_id)
        deleted = voice_id.delete_embeddings(self.store, person_id, model=model)
        logger.info(f"{annotator} deleted {deleted} identity samples of person {person_id} (model={model or 'all'})")
        return {"deleted": deleted}

    def create_person(self, name: str) -> dict[str, Any]:
        """Add a person."""
        return memories.create_person(self.store, name)

    def update_person(self, person_id: str, *, name: str | None, archived: bool | None) -> dict[str, Any]:
        """Rename or (un)archive a person."""
        return memories.update_person(self.store, person_id, name=name, archived=archived)

    def assign_speaker(self, session_id: str, annotator: str, person_id: str | None, turn_idx: int | None) -> dict:
        """Attribute a session or turn, then re-run memory extraction for the session."""
        if self.store.get_session(session_id) is None:
            raise KeyError(session_id)
        if person_id and memories.get_person(self.store, person_id) is None:
            raise ValueError("unknown person")
        memories.assign_speaker(
            self.store,
            session_id,
            person_id,
            source=f"human:{annotator}",
            turn_idx=memories.SESSION_TURN if turn_idx is None else turn_idx,
        )
        self.store.reset_job("dream", session_id)
        return memories.speakers_for(self.store, session_id)

    def list_memories(self, *, person_id: str | None, status: str, limit: int, offset: int) -> dict[str, Any]:
        """Memories page with evidence and usage."""
        return memories.list_memories(self.store, person_id=person_id, status=status, limit=limit, offset=offset)

    def review_memory(self, memory_id: int, annotator: str, payload: MemoryReviewIn) -> dict[str, Any]:
        """Approve, correct or forget a memory."""
        if payload.person_id and memories.get_person(self.store, payload.person_id) is None:
            raise ValueError("unknown person")
        if payload.action == "correct" and not (payload.text or "").strip() and not payload.person_id:
            raise ValueError("correct needs a new text or person")
        row = memories.review_memory(
            self.store,
            memory_id,
            payload.action,
            reviewer=annotator,
            text=payload.text,
            person_id=payload.person_id,
        )
        return memories.attach_details(self.store, [row])[0]


def create_review_router(
    settings: MonitoringConfig,
    store: SessionStore,
    artifacts: ArtifactStore,
    dreamer_config: dict[str, Any] | None = None,
) -> APIRouter:
    """Build the ``/api/review`` router."""
    service = ReviewService(store, artifacts, dreamer_config or load_dreamer_config())
    router = APIRouter(prefix="/api/review", tags=["review"])

    async def run(fn, *args, **kwargs):
        return await asyncio.to_thread(fn, *args, **kwargs)

    def check_annotator(name: str) -> str:
        name = name.strip()
        if not _ANNOTATOR_RE.match(name):
            raise HTTPException(status_code=422, detail="annotator: 1-64 letters, digits, space, . @ - _")
        return name

    @router.get("/activities")
    async def activities():
        return {"activities": await run(service.activities)}

    @router.get("/sessions")
    async def sessions(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0)):
        return await run(service.list_sessions, limit=limit, offset=offset)

    @router.get("/sessions/{session_id}")
    async def session(session_id: str):
        try:
            return await run(service.session_detail, session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found") from None

    @router.get("/sessions/{session_id}/llm-calls/{call_id}")
    async def llm_call(session_id: str, call_id: int):
        try:
            return await run(service.llm_call, session_id, call_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="llm call not found") from None

    @router.get("/artifacts/{key:path}")
    async def artifact(key: str):
        read_path = getattr(artifacts, "read_path", None)
        try:
            if read_path is not None:
                path = read_path(key)
                if not path.is_file():
                    raise FileNotFoundError(key)
                return FileResponse(path)  # honors HTTP Range (audio seeking)
            data = await run(artifacts.get, key)
        except (FileNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="artifact not found") from None
        return Response(data)

    @router.get("/asr/queue")
    async def asr_queue(
        status: str = Query("open", pattern="^(open|reviewed|all)$"),
        limit: int = Query(50, ge=0, le=1000),
        offset: int = Query(0, ge=0),
        sort: str = Query("recent", pattern="^(recent|disagreement)$"),
        session_id: str | None = None,
    ):
        return await run(service.asr_queue, status=status, limit=limit, offset=offset, sort=sort, session_id=session_id)

    @router.post("/asr/references")
    async def save_references(payload: ReferenceBatchIn):
        annotator = check_annotator(payload.annotator)
        return await run(service.save_references, annotator, payload.items)

    @router.get("/dreamer")
    async def dreamer():
        return await run(service.dreamer_status)

    @router.post("/dreamer/pause")
    async def pause(payload: PauseIn):
        return await run(service.set_paused, payload.paused)

    def check_kind(kind: str) -> str:
        if kind not in JOB_REGISTRY:
            raise HTTPException(status_code=422, detail=f"unknown job kind {kind!r}")
        return kind

    @router.get("/jobs")
    async def list_jobs(
        status: str | None = Query(None, pattern="^(pending|running|done|failed|cancelled)$"),
        kind: str | None = None,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        return await run(service.list_jobs, status=status, kind=kind, limit=limit, offset=offset)

    @router.post("/jobs")
    async def jobs(payload: JobsIn):
        check_kind(payload.kind)
        return {"queued": await run(service.requeue, payload.kind, payload.session_ids)}

    @router.post("/jobs/enqueue-all")
    async def enqueue_all(payload: EnqueueAllIn):
        return {"queued": await run(service.enqueue_all, check_kind(payload.kind))}

    @router.post("/jobs/{job_id}/cancel")
    async def cancel(job_id: int):
        if not await run(service.cancel_job, job_id):
            raise HTTPException(status_code=409, detail="only pending jobs can be cancelled")
        return {"cancelled": job_id}

    @router.get("/people")
    async def people(include_archived: bool = False):
        return {"people": await run(service.people, include_archived=include_archived)}

    @router.post("/people")
    async def create_person(payload: PersonIn):
        return await run(service.create_person, payload.name)

    @router.patch("/people/{person_id}")
    async def update_person(person_id: str, payload: PersonPatchIn):
        try:
            return await run(service.update_person, person_id, name=payload.name, archived=payload.archived)
        except KeyError:
            raise HTTPException(status_code=404, detail="person not found") from None

    @router.get("/people/duplicates")
    async def duplicate_people():
        return {"duplicates": await run(service.duplicate_people)}

    @router.get("/people/{person_id}/identity")
    async def person_identity(person_id: str):
        try:
            return await run(service.person_identity, person_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="person not found") from None

    @router.post("/people/{person_id}/merge")
    async def merge_people(person_id: str, payload: MergeIn):
        annotator = check_annotator(payload.annotator)
        try:
            return await run(service.merge_people, person_id, payload.source_id, annotator)
        except KeyError:
            raise HTTPException(status_code=404, detail="person not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @router.post("/people/{person_id}/forget-identity")
    async def forget_identity(person_id: str, payload: ForgetIdentityIn):
        annotator = check_annotator(payload.annotator)
        try:
            return await run(service.forget_identity, person_id, payload.model, annotator)
        except KeyError:
            raise HTTPException(status_code=404, detail="person not found") from None

    @router.post("/sessions/{session_id}/speaker")
    async def assign_speaker(session_id: str, payload: SpeakerIn):
        annotator = check_annotator(payload.annotator)
        try:
            return await run(service.assign_speaker, session_id, annotator, payload.person_id, payload.turn_idx)
        except KeyError:
            raise HTTPException(status_code=404, detail="session not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @router.get("/memories")
    async def list_memories(
        person_id: str | None = None,
        status: str = Query("open", pattern="^(open|usable|all|proposed|active|forgotten|superseded)$"),
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ):
        return await run(service.list_memories, person_id=person_id, status=status, limit=limit, offset=offset)

    @router.post("/memories/{memory_id}/review")
    async def review_memory(memory_id: int, payload: MemoryReviewIn):
        annotator = check_annotator(payload.annotator)
        try:
            return await run(service.review_memory, memory_id, annotator, payload)
        except KeyError:
            raise HTTPException(status_code=404, detail="memory not found") from None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None

    @router.get("/metrics")
    async def metrics(
        since_days: float | None = Query(None, gt=0),
        group_by: str = Query(DEFAULT_GROUP_BY, max_length=500),
    ):
        fields = [f.strip() for f in group_by.split(",") if f.strip()]
        if not all(re.fullmatch(r"[\w.]{1,64}", f) for f in fields) or len(fields) > 8:
            raise HTTPException(status_code=422, detail="group_by: up to 8 dotted config fields")
        return await run(service.metrics, since_days=since_days, group_by=fields)

    return router
