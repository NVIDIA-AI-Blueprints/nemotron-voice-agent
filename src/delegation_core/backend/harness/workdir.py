# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One runtime process's work folder, made under ``backend.harness.work_root``, with a folder for each session.

The folders may hold the composed instructions, so they are deleted when the process ends. A folder's marker is locked
(``flock``) by the server process that owns it, and the system releases a crashed process's lock, so a later start
sweeps every marked folder whose lock it can take, whatever process id the marker names (a restarted container's
server usually has its predecessor's id). The lock holds among the processes of one machine, containers included. An
agent served on another host or in its own container sees these folders only when the operator shares the work root at
the same path and runs the agent as this side's user or as root: each folder has mode 0700.
"""

from __future__ import annotations

import fcntl
import os
import shutil
import tempfile
from pathlib import Path

MARKER = ".voiceclaw-harness-work"  # locked by the process that owns the folder; holds that process's id, as a note

# This process's own work folders: a sweep never takes one, even where a process may take its own lock again (NFS).
_HELD: set[Path] = set()


class WorkFolder:
    """A runtime process's work folder, made under the work root and deleted with the process."""

    def __init__(self, path: Path, lock: int | None = None):
        """Own the folder at ``path`` (made by :meth:`make`), whose marker the open descriptor ``lock`` holds locked."""
        self.path, self._lock = path, lock

    @classmethod
    def make(cls, root: Path) -> WorkFolder:
        """A fresh folder under ``root``, its marker locked until :meth:`remove`."""
        root.mkdir(parents=True, exist_ok=True)
        path = Path(tempfile.mkdtemp(prefix="work-", dir=root))
        _HELD.add(path.resolve())
        lock = None
        try:
            staged = path / f"{MARKER}.new"
            lock = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            fcntl.flock(lock, fcntl.LOCK_EX)
            os.write(lock, f"{os.getpid()}\n".encode())
            staged.rename(path / MARKER)  # in place once locked: a sweep never finds a live folder's marker free
            (path / "sessions").mkdir()
        except BaseException:
            if lock is not None:
                os.close(lock)
            shutil.rmtree(path, ignore_errors=True)
            _HELD.discard(path.resolve())
            raise
        return cls(path, lock)

    def session_folder(self, name: str) -> Path:
        """A new, empty folder for one session."""
        folder = self.path / "sessions" / name
        folder.mkdir(parents=True)
        return folder

    def remove(self) -> None:
        """Delete the folder, only if it carries the marker (so no other folder is ever deleted), and free its lock."""
        if self._lock is not None:
            os.close(self._lock)  # first: a file still open may keep its folder from being deleted (NFS)
            self._lock = None
        if (self.path / MARKER).is_file():
            shutil.rmtree(self.path, ignore_errors=True)
        _HELD.discard(self.path.resolve())

    @staticmethod
    def sweep(root: Path) -> int:
        """Delete the work folders under ``root`` that no process holds (their process has ended); return how many."""
        if not root.is_dir():
            return 0
        removed = 0
        for child in root.iterdir():
            marker = child / MARKER
            if child.is_dir() and marker.is_file() and child.resolve() not in _HELD and not _held(marker):
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        return removed


def _held(marker: Path) -> bool:
    """Whether a process holds ``marker``'s lock (a marker this side cannot open or lock counts as held)."""
    try:
        with open(marker, "r+b") as handle:  # opened to write: an exclusive lock needs it on some file systems (NFS)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    return False
