"""Memory Retrieval for Past Experiences.

Stores and retrieves memories of past successes and failures
for context-aware planning.

INVARIANTS:
1. Retrieval is read-only (doesn't affect execution)
2. Memories are used for context only
3. Similar memories are ranked by relevance
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fingerprint import Fingerprint


@dataclass(frozen=True)
class Memory:
    """A memory of a past episode or action.
    
    INVARIANT: Memories are immutable records.
    """
    
    memory_id: str
    memory_type: str  # "success", "failure", "rejection"
    task_id: str
    fingerprint_id: str | None
    content: str  # Summary or description
    metadata: tuple[tuple[str, Any], ...]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "task_id": self.task_id,
            "fingerprint_id": self.fingerprint_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Memory:
        """Deserialize from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = tuple(sorted(metadata.items()))
        
        return cls(
            memory_id=data["memory_id"],
            memory_type=data["memory_type"],
            task_id=data["task_id"],
            fingerprint_id=data.get("fingerprint_id"),
            content=data["content"],
            metadata=metadata,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


class MemoryIndex:
    """Index for storing and retrieving memories.
    
    Uses SQLite for persistence and simple text search.
    
    INVARIANTS:
    1. Retrieval is read-only (context only)
    2. Similar memories ranked by relevance
    3. Memories persist across sessions
    
    Usage:
        index = MemoryIndex("memories.db")
        
        # Store a memory
        index.store(Memory(...))
        
        # Retrieve similar memories
        similar = index.retrieve_similar(fingerprint, k=5)
        
        # Search by text
        results = index.search("pattern", k=10)
    """
    
    def __init__(self, db_path: Path | str | None = None):
        """Initialize memory index.
        
        Args:
            db_path: Path to SQLite database (uses in-memory if None).
        """
        self.db_path = Path(db_path) if db_path else None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
        else:
            conn = sqlite3.connect(":memory:")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT,
                task_id TEXT,
                fingerprint_id TEXT,
                content TEXT,
                metadata TEXT,
                timestamp TEXT
            )
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_fingerprint ON memories(fingerprint_id)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON memories(memory_type)
        """)
        
        # FTS table for text search
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                memory_id, content, tokenize='porter'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self.db_path:
            return sqlite3.connect(self.db_path)
        return sqlite3.connect(":memory:")
    
    def store(self, memory: Memory) -> None:
        """Store a memory in the index.
        
        Args:
            memory: Memory to store.
        """
        conn = self._get_conn()
        try:
            # Store main record
            conn.execute(
                """
                INSERT OR REPLACE INTO memories 
                (memory_id, memory_type, task_id, fingerprint_id, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    memory.memory_type,
                    memory.task_id,
                    memory.fingerprint_id,
                    memory.content,
                    json.dumps(dict(memory.metadata)),
                    memory.timestamp,
                ),
            )
            
            # Update FTS index
            conn.execute(
                """
                INSERT OR REPLACE INTO memories_fts (memory_id, content)
                VALUES (?, ?)
                """,
                (memory.memory_id, memory.content),
            )
            
            conn.commit()
        finally:
            conn.close()
    
    def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.
        
        Args:
            memory_id: Memory identifier.
        
        Returns:
            Memory or None if not found.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                """
                SELECT memory_id, memory_type, task_id, fingerprint_id, content, metadata, timestamp
                FROM memories WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
            
            if row:
                return Memory(
                    memory_id=row[0],
                    memory_type=row[1],
                    task_id=row[2],
                    fingerprint_id=row[3],
                    content=row[4],
                    metadata=tuple(sorted(json.loads(row[5] or "{}").items())),
                    timestamp=row[6],
                )
            return None
        finally:
            conn.close()
    
    def retrieve_by_fingerprint(
        self,
        fingerprint_id: str,
        k: int = 10,
    ) -> list[Memory]:
        """Retrieve memories with matching fingerprint.
        
        Args:
            fingerprint_id: Fingerprint to match.
            k: Maximum number of results.
        
        Returns:
            List of matching memories.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT memory_id, memory_type, task_id, fingerprint_id, content, metadata, timestamp
                FROM memories 
                WHERE fingerprint_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (fingerprint_id, k),
            ).fetchall()
            
            return [
                Memory(
                    memory_id=row[0],
                    memory_type=row[1],
                    task_id=row[2],
                    fingerprint_id=row[3],
                    content=row[4],
                    metadata=tuple(sorted(json.loads(row[5] or "{}").items())),
                    timestamp=row[6],
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def retrieve_similar(
        self,
        fingerprint: Fingerprint,
        k: int = 5,
        min_similarity: float = 0.3,
    ) -> list[tuple[Memory, float]]:
        """Retrieve memories similar to a fingerprint.
        
        Uses fingerprint similarity scoring.
        
        Args:
            fingerprint: Query fingerprint.
            k: Maximum number of results.
            min_similarity: Minimum similarity threshold.
        
        Returns:
            List of (memory, similarity_score) tuples.
        """
        # First, get exact matches
        exact = self.retrieve_by_fingerprint(fingerprint.fingerprint_id, k=k)
        results = [(m, 1.0) for m in exact]
        
        # Then search by patterns
        if fingerprint.patterns and len(results) < k:
            pattern_memories = self.search(" ".join(fingerprint.patterns), k=k * 2)
            for mem in pattern_memories:
                if mem.memory_id not in [r[0].memory_id for r in results]:
                    # Compute similarity (simplified)
                    similarity = 0.5  # Base similarity for pattern match
                    if mem.fingerprint_id:
                        # Could compute actual fingerprint similarity here
                        pass
                    if similarity >= min_similarity:
                        results.append((mem, similarity))
        
        # Sort by similarity and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def search(
        self,
        query: str,
        k: int = 10,
        memory_type: str | None = None,
    ) -> list[Memory]:
        """Search memories by text content.
        
        Args:
            query: Search query.
            k: Maximum number of results.
            memory_type: Optional type filter.
        
        Returns:
            List of matching memories.
        """
        conn = self._get_conn()
        try:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT m.memory_id, m.memory_type, m.task_id, m.fingerprint_id, 
                           m.content, m.metadata, m.timestamp
                    FROM memories m
                    JOIN memories_fts fts ON m.memory_id = fts.memory_id
                    WHERE memories_fts MATCH ? AND m.memory_type = ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, memory_type, k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT m.memory_id, m.memory_type, m.task_id, m.fingerprint_id, 
                           m.content, m.metadata, m.timestamp
                    FROM memories m
                    JOIN memories_fts fts ON m.memory_id = fts.memory_id
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (query, k),
                ).fetchall()
            
            return [
                Memory(
                    memory_id=row[0],
                    memory_type=row[1],
                    task_id=row[2],
                    fingerprint_id=row[3],
                    content=row[4],
                    metadata=tuple(sorted(json.loads(row[5] or "{}").items())),
                    timestamp=row[6],
                )
                for row in rows
            ]
        except sqlite3.OperationalError:
            # FTS query failed, fall back to LIKE search
            return self._search_fallback(query, k, memory_type)
        finally:
            conn.close()
    
    def _search_fallback(
        self,
        query: str,
        k: int,
        memory_type: str | None,
    ) -> list[Memory]:
        """Fallback search using LIKE."""
        conn = self._get_conn()
        try:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT memory_id, memory_type, task_id, fingerprint_id, 
                           content, metadata, timestamp
                    FROM memories 
                    WHERE content LIKE ? AND memory_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", memory_type, k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id, memory_type, task_id, fingerprint_id, 
                           content, metadata, timestamp
                    FROM memories 
                    WHERE content LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", k),
                ).fetchall()
            
            return [
                Memory(
                    memory_id=row[0],
                    memory_type=row[1],
                    task_id=row[2],
                    fingerprint_id=row[3],
                    content=row[4],
                    metadata=tuple(sorted(json.loads(row[5] or "{}").items())),
                    timestamp=row[6],
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def get_recent(
        self,
        k: int = 20,
        memory_type: str | None = None,
    ) -> list[Memory]:
        """Get most recent memories.
        
        Args:
            k: Maximum number of results.
            memory_type: Optional type filter.
        
        Returns:
            List of recent memories.
        """
        conn = self._get_conn()
        try:
            if memory_type:
                rows = conn.execute(
                    """
                    SELECT memory_id, memory_type, task_id, fingerprint_id, 
                           content, metadata, timestamp
                    FROM memories 
                    WHERE memory_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (memory_type, k),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT memory_id, memory_type, task_id, fingerprint_id, 
                           content, metadata, timestamp
                    FROM memories 
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (k,),
                ).fetchall()
            
            return [
                Memory(
                    memory_id=row[0],
                    memory_type=row[1],
                    task_id=row[2],
                    fingerprint_id=row[3],
                    content=row[4],
                    metadata=tuple(sorted(json.loads(row[5] or "{}").items())),
                    timestamp=row[6],
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def count(self, memory_type: str | None = None) -> int:
        """Count memories in index.
        
        Args:
            memory_type: Optional type filter.
        
        Returns:
            Number of memories.
        """
        conn = self._get_conn()
        try:
            if memory_type:
                result = conn.execute(
                    "SELECT COUNT(*) FROM memories WHERE memory_type = ?",
                    (memory_type,),
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            
            return result[0] if result else 0
        finally:
            conn.close()


def create_memory(
    memory_type: str,
    task_id: str,
    content: str,
    fingerprint: Fingerprint | None = None,
    metadata: dict[str, Any] | None = None,
) -> Memory:
    """Create a new memory.
    
    Args:
        memory_type: Type of memory (success, failure, rejection).
        task_id: Associated task ID.
        content: Summary or description.
        fingerprint: Optional fingerprint.
        metadata: Optional metadata.
    
    Returns:
        New Memory instance.
    """
    # Generate memory ID
    memory_id = hashlib.sha256(
        f"{task_id}|{memory_type}|{datetime.now(timezone.utc).isoformat()}".encode()
    ).hexdigest()[:12]
    
    return Memory(
        memory_id=memory_id,
        memory_type=memory_type,
        task_id=task_id,
        fingerprint_id=fingerprint.fingerprint_id if fingerprint else None,
        content=content,
        metadata=tuple(sorted((metadata or {}).items())),
    )
