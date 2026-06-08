"""Minimum Viable Memory — the durable episodic store.

This is the "Minimum Viable Memory" build from `memory-architecture.md`: a
single SQLite table of conversation episodes, searched by recency + keyword.
No embeddings, no vector store, no semantic graph yet — those are the v2
north star (see the doc). The point here is that `recall()` works *at all*
for cross-day continuity, cheaply, behind a clean interface the full design
can grow into.

Separation of concerns:
- LangGraph's SqliteSaver (`state/conversations.db`) is *working memory*:
  the within-thread checkpoint chain, owned by the agent graph.
- This store (`state/memory.db`) is the *episodic log*: durable, append-only,
  queryable across threads/days. The source of truth a future consolidation
  pipeline would derive a semantic graph from.

Two separate DB files on purpose — the checkpointer's connection is driven by
LangGraph's threadpool, and we don't want episodic writes contending with it.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Episode:
    """One conversational turn: what the user said and how the robot replied."""

    id: str
    ts: str  # ISO 8601, UTC
    thread_id: str
    user_text: str
    assistant_text: str

    def as_recall_line(self) -> str:
        """One-line rendering for injection into the agent's context."""
        # Date only (not full timestamp) — the LLM reasons better about
        # "yesterday" / "last Tuesday" from a date than a microsecond stamp.
        day = self.ts[:10]
        return f"[{day}] you: {self.user_text} | nemo: {self.assistant_text}"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    ts              TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    user_text       TEXT NOT NULL,
    assistant_text  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(ts);
"""


class MemoryStore:
    """Append-only episodic log backed by SQLite.

    Thread-safe: episode writes arrive from `asyncio.to_thread` on the Edge
    loop, so every connection use is guarded by a lock. `check_same_thread=
    False` is required because the executor threads differ from the opener.
    """

    def __init__(self, db_path: str):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def append(self, user_text: str, assistant_text: str, thread_id: str) -> str:
        """Record one turn. Returns the new episode id.

        Empty turns (no user text) are skipped — a silent wake or aborted
        utterance isn't an episode worth remembering.
        """
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text:
            return ""
        episode_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO episodes (id, ts, thread_id, user_text, assistant_text) "
                "VALUES (?, ?, ?, ?, ?)",
                (episode_id, ts, thread_id, user_text, assistant_text),
            )
            self._conn.commit()
        return episode_id

    def search(self, query: str, limit: int = 5) -> list[Episode]:
        """Keyword search over user + assistant text, most recent first.

        MVM-grade: SQL `LIKE` on whitespace-split terms (OR-matched). This is
        deliberately not vector search — for a single user's own history,
        keyword + recency captures most of the value. The doc says add
        embeddings only when this demonstrably misses things.
        """
        terms = [t for t in (query or "").split() if t]
        with self._lock:
            if not terms:
                rows = self._conn.execute(
                    "SELECT id, ts, thread_id, user_text, assistant_text "
                    "FROM episodes ORDER BY ts DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                clause = " OR ".join(
                    ["(user_text LIKE ? OR assistant_text LIKE ?)"] * len(terms)
                )
                params: list[str] = []
                for t in terms:
                    like = f"%{t}%"
                    params.extend([like, like])
                params.append(str(limit))
                rows = self._conn.execute(
                    "SELECT id, ts, thread_id, user_text, assistant_text "
                    f"FROM episodes WHERE {clause} ORDER BY ts DESC LIMIT ?",
                    params,
                ).fetchall()
        return [Episode(*row) for row in rows]

    def recent(self, limit: int = 5) -> list[Episode]:
        """Most recent episodes regardless of content."""
        return self.search("", limit=limit)

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
