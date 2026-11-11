"""In-process, server-local session bookkeeping: caller key -> CLI session UUID, with bounded cleanup.

Kept separate from `handler.py` so request shaping stays free of this state's locking and sweeping.
No external store: the CLI's own transcript files under `transcripts_dir` are already the durable
record `--resume` reads, so this module only tracks enough to decide new-vs-resume and when to prune.
"""

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Final
from uuid import UUID, uuid4

verbose_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NewSession:
    session_id: UUID


@dataclass(frozen=True, slots=True)
class ResumeSession:
    session_id: UUID
    turns_already_sent: int
    seen_system_prompt: str = ""
    last_reply_saved: bool = False


SessionDecision = NewSession | ResumeSession


@dataclass(frozen=True, slots=True)
class _Entry:
    session_id: UUID
    last_touched: datetime
    turns_sent: int
    seen_system_prompt: str
    last_reply_saved: bool = False


class SessionStore:
    """One lock for the whole map: expected concurrency is small (~10-12 trainees), per the design notes."""

    def __init__(self) -> None:
        self._lock: Final = asyncio.Lock()
        self._entries: dict[str, _Entry] = {}  # mutable-ok: guarded by _lock, the store's whole reason to exist
        self._retired: list[UUID] = []  # mutable-ok: guarded by _lock, drained by every sweep

    async def resolve_session(self, key: str, *, system_prompt: str = "") -> SessionDecision:
        """Lock-scoped so two near-simultaneous first requests for the same new key cannot mint two UUIDs.

        The CLI replays its first-turn system prompt on every resume, so a prompt that no longer extends
        what the session has seen starts a fresh session rather than resuming one the model reads wrongly.
        """
        async with self._lock:
            entry: Final = self._entries.get(key)
            if entry is not None and system_prompt.startswith(entry.seen_system_prompt):
                return ResumeSession(
                    session_id=entry.session_id,
                    turns_already_sent=entry.turns_sent,
                    seen_system_prompt=entry.seen_system_prompt,
                    last_reply_saved=entry.last_reply_saved,
                )
            if entry is not None:
                self._retired.append(entry.session_id)
            session_id: Final = uuid4()
            self._entries[key] = _Entry(
                session_id=session_id, last_touched=_now(), turns_sent=0, seen_system_prompt=system_prompt
            )
            return NewSession(session_id=session_id)

    async def touch(
        self, key: str, *, turns_sent: int, seen_system_prompt: str | None = None, last_reply_saved: bool = False
    ) -> None:
        """`last_reply_saved` is False when the run was stopped on a caller tool call, not finished by the CLI."""
        async with self._lock:
            if (entry := self._entries.get(key)) is None:
                return
            self._entries[key] = replace(
                entry,
                last_touched=_now(),
                turns_sent=turns_sent,
                seen_system_prompt=entry.seen_system_prompt if seen_system_prompt is None else seen_system_prompt,
                last_reply_saved=last_reply_saved,
            )

    async def discard(self, key: str, session_id: UUID) -> None:
        """Forget a session whose first run failed, so a retry starts fresh instead of resuming a transcript that never existed."""
        async with self._lock:
            entry: Final = self._entries.get(key)
            if entry is None or entry.session_id != session_id:
                return
            del self._entries[key]
            self._retired.append(session_id)

    async def sweep(self, *, now: datetime, inactivity_window: timedelta, transcripts_dir: Path) -> tuple[str, ...]:
        """Delete the on-disk transcript and drop tracking for every session untouched past the window.

        Sessions replaced by a fresh one (their prompt changed) are no longer tracked, so their transcripts
        are deleted on the next sweep regardless of age.
        """
        async with self._lock:
            stale: Final = tuple(
                key for key, entry in self._entries.items() if now - entry.last_touched > inactivity_window
            )
            for key in stale:
                _delete_transcript(transcripts_dir, self._entries[key].session_id)
                del self._entries[key]
            for session_id in self._retired:
                _delete_transcript(transcripts_dir, session_id)
            self._retired.clear()
            return stale

    def snapshot(self) -> MappingProxyType[str, _Entry]:
        return MappingProxyType(dict(self._entries))


def _now() -> datetime:
    return datetime.now()


def _delete_transcript(transcripts_dir: Path, session_id: UUID) -> None:
    (transcripts_dir / f"{session_id}.jsonl").unlink(missing_ok=True)


async def run_sweep_forever(
    store: SessionStore,
    *,
    interval: timedelta,
    inactivity_window: timedelta,
    transcripts_dir: Path,
) -> None:
    """Process-lifetime background loop, started once alongside the provider, mirroring `concurrency_guard`."""
    while True:
        await asyncio.sleep(interval.total_seconds())
        try:
            await store.sweep(now=_now(), inactivity_window=inactivity_window, transcripts_dir=transcripts_dir)
        except OSError:
            verbose_logger.exception("claude_code_cli session sweep failed; will retry next interval")


__all__ = [
    "NewSession",
    "ResumeSession",
    "SessionDecision",
    "SessionStore",
    "run_sweep_forever",
]
