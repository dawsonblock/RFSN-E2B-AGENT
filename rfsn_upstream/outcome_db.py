"""Persistent episode outcome database.

INVARIANTS:
- External to kernel (upstream only)
- Append-only at the logical level
- No mutation of existing episodes
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeRecord:
    """Immutable record of a single repair episode."""
    instance_id: str
    repo_id: str
    arm_id: str
    success: bool
    metrics: dict[str, Any]
    fingerprints: list[dict[str, Any]]
    created_at: str


class OutcomeDB:
    """
    Persistent episode store (SQLite).

    INVARIANTS:
    - External to kernel
    - Append-only at the logical level (we don't mutate existing episodes)
    - All queries are indexed for performance
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instance_id TEXT NOT NULL,
                    repo_id TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    fingerprints_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eps_instance ON episodes(instance_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eps_repo ON episodes(repo_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eps_arm ON episodes(arm_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_eps_success ON episodes(success);")
            conn.commit()

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def add_episode(
        self,
        *,
        instance_id: str,
        repo_id: str,
        arm_id: str,
        success: bool,
        metrics: dict[str, Any],
        fingerprints: list[dict[str, Any]],
        created_at: str | None = None,
    ) -> int:
        """Add an episode record. Returns the episode ID."""
        created_at = created_at or self.now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO episodes(instance_id, repo_id, arm_id, success, metrics_json, fingerprints_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    instance_id,
                    repo_id,
                    arm_id,
                    1 if success else 0,
                    json.dumps(metrics, sort_keys=True),
                    json.dumps(fingerprints, sort_keys=True),
                    created_at,
                ),
            )
            conn.commit()
            return cursor.lastrowid or 0

    def recent_episodes(
        self,
        *,
        instance_id: str | None = None,
        repo_id: str | None = None,
        arm_id: str | None = None,
        success: bool | None = None,
        limit: int = 50,
    ) -> list[EpisodeRecord]:
        """Query recent episodes with optional filters."""
        where = []
        params: list[Any] = []
        if instance_id is not None:
            where.append("instance_id = ?")
            params.append(instance_id)
        if repo_id is not None:
            where.append("repo_id = ?")
            params.append(repo_id)
        if arm_id is not None:
            where.append("arm_id = ?")
            params.append(arm_id)
        if success is not None:
            where.append("success = ?")
            params.append(1 if success else 0)

        sql = "SELECT instance_id, repo_id, arm_id, success, metrics_json, fingerprints_json, created_at FROM episodes"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(int(limit))

        out: list[EpisodeRecord] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                inst, repo, arm, success_i, metrics_j, fps_j, created = row
                out.append(
                    EpisodeRecord(
                        instance_id=inst,
                        repo_id=repo,
                        arm_id=arm,
                        success=bool(success_i),
                        metrics=json.loads(metrics_j),
                        fingerprints=json.loads(fps_j),
                        created_at=created,
                    )
                )
        return out

    def episodes_with_fingerprint(
        self,
        fingerprint_id: str,
        limit: int = 20,
    ) -> list[EpisodeRecord]:
        """Find episodes that contain a specific fingerprint."""
        # Use JSON contains (SQLite JSON1)
        sql = """
            SELECT instance_id, repo_id, arm_id, success, metrics_json, fingerprints_json, created_at
            FROM episodes
            WHERE fingerprints_json LIKE ?
            ORDER BY id DESC
            LIMIT ?
        """
        pattern = f'%"fingerprint_id": "{fingerprint_id}"%'
        
        out: list[EpisodeRecord] = []
        with self._connect() as conn:
            for row in conn.execute(sql, (pattern, limit)):
                inst, repo, arm, success_i, metrics_j, fps_j, created = row
                out.append(
                    EpisodeRecord(
                        instance_id=inst,
                        repo_id=repo,
                        arm_id=arm,
                        success=bool(success_i),
                        metrics=json.loads(metrics_j),
                        fingerprints=json.loads(fps_j),
                        created_at=created,
                    )
                )
        return out

    def arm_success_rate(self, arm_id: str) -> tuple[int, int]:
        """Return (successes, total) for an arm."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT SUM(success), COUNT(*) FROM episodes WHERE arm_id = ?",
                (arm_id,),
            ).fetchone()
            if row and row[1]:
                return (row[0] or 0, row[1])
            return (0, 0)

    def total_episodes(self) -> int:
        """Total episode count."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
            return row[0] if row else 0
