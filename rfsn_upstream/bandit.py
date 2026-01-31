"""Thompson Sampling Bandit for Prompt Variant Selection.

The bandit learns which prompt variants work best over time.
It uses Thompson Sampling for efficient exploration/exploitation.

State is persisted to SQLite for CI/CD integration.

INVARIANTS:
1. Bandit state is independent of kernel execution
2. Bandit only affects which prompts are selected
3. Bandit cannot modify gate or kernel behavior
"""

from __future__ import annotations


import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ArmStats:
    """Statistics for a bandit arm (prompt variant)."""
    
    arm_id: str
    alpha: float = 1.0  # Successes + 1 (Beta prior)
    beta: float = 1.0   # Failures + 1 (Beta prior)
    pulls: int = 0
    successes: int = 0
    failures: int = 0
    last_updated: str = ""
    
    @property
    def success_rate(self) -> float:
        """Estimated success rate (mean of Beta distribution)."""
        return self.alpha / (self.alpha + self.beta)
    
    @property
    def uncertainty(self) -> float:
        """Uncertainty measure (variance of Beta)."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "arm_id": self.arm_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "pulls": self.pulls,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "last_updated": self.last_updated,
        }


class ThompsonBandit:
    """Thompson Sampling bandit for prompt variant selection.
    
    Uses Beta-Bernoulli Thompson Sampling:
    - Each arm has a Beta(alpha, beta) distribution
    - Selection: sample from each Beta, pick highest
    - Update: increment alpha on success, beta on failure
    
    State is persisted to SQLite for CI/CD integration.
    
    INVARIANTS:
    1. Bandit is external to kernel (coaching layer)
    2. Bandit only influences prompt selection
    3. Bandit cannot bypass gate or modify decisions
    
    Usage:
        bandit = ThompsonBandit("bandit.db")
        
        # Select an arm
        arm_id = bandit.select_arm()
        
        # Run episode with selected arm
        result = run_episode(arm_id)
        
        # Update based on outcome
        bandit.update(arm_id, success=result.success)
    """
    
    def __init__(
        self,
        db_path: Path | str | None = None,
        arms: list[str] | None = None,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
    ):
        """Initialize bandit with SQLite persistence.
        
        Args:
            db_path: Path to SQLite database (uses in-memory if None).
            arms: List of arm IDs to initialize.
            prior_alpha: Prior alpha for Beta distribution.
            prior_beta: Prior beta for Beta distribution.
        """
        self.db_path = Path(db_path) if db_path else None
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        
        # Initialize database
        self._init_db()
        
        # Initialize arms if provided
        if arms:
            for arm_id in arms:
                self._ensure_arm(arm_id)
    
    def _init_db(self) -> None:
        """Initialize SQLite database."""
        if self.db_path:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
        else:
            conn = sqlite3.connect(":memory:")
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arms (
                arm_id TEXT PRIMARY KEY,
                alpha REAL DEFAULT 1.0,
                beta REAL DEFAULT 1.0,
                pulls INTEGER DEFAULT 0,
                successes INTEGER DEFAULT 0,
                failures INTEGER DEFAULT 0,
                last_updated TEXT
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                arm_id TEXT,
                success INTEGER,
                reward REAL,
                task_id TEXT,
                timestamp TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self.db_path:
            return sqlite3.connect(self.db_path)
        return sqlite3.connect(":memory:")
    
    def _ensure_arm(self, arm_id: str) -> None:
        """Ensure arm exists in database."""
        conn = self._get_conn()
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO arms (arm_id, alpha, beta, pulls, successes, failures, last_updated)
                VALUES (?, ?, ?, 0, 0, 0, ?)
                """,
                (arm_id, self.prior_alpha, self.prior_beta, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()
    
    def get_arm_stats(self, arm_id: str) -> ArmStats | None:
        """Get statistics for an arm.
        
        Args:
            arm_id: Arm identifier.
        
        Returns:
            ArmStats or None if arm doesn't exist.
        """
        conn = self._get_conn()
        try:
            row = conn.execute(
                "SELECT arm_id, alpha, beta, pulls, successes, failures, last_updated FROM arms WHERE arm_id = ?",
                (arm_id,),
            ).fetchone()
            
            if row:
                return ArmStats(
                    arm_id=row[0],
                    alpha=row[1],
                    beta=row[2],
                    pulls=row[3],
                    successes=row[4],
                    failures=row[5],
                    last_updated=row[6] or "",
                )
            return None
        finally:
            conn.close()
    
    def get_all_arms(self) -> list[ArmStats]:
        """Get statistics for all arms.
        
        Returns:
            List of ArmStats for all arms.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT arm_id, alpha, beta, pulls, successes, failures, last_updated FROM arms"
            ).fetchall()
            
            return [
                ArmStats(
                    arm_id=row[0],
                    alpha=row[1],
                    beta=row[2],
                    pulls=row[3],
                    successes=row[4],
                    failures=row[5],
                    last_updated=row[6] or "",
                )
                for row in rows
            ]
        finally:
            conn.close()
    
    def select_arm(self, exclude: list[str] | None = None) -> str:
        """Select an arm using Thompson Sampling.
        
        Samples from each arm's Beta distribution and returns
        the arm with the highest sample.
        
        Args:
            exclude: List of arm IDs to exclude from selection.
        
        Returns:
            Selected arm ID.
        
        Raises:
            ValueError: If no arms are available.
        """
        exclude = exclude or []
        arms = [a for a in self.get_all_arms() if a.arm_id not in exclude]
        
        if not arms:
            raise ValueError("No arms available for selection")
        
        # Sample from each arm's Beta distribution
        samples = []
        for arm in arms:
            sample = random.betavariate(arm.alpha, arm.beta)
            samples.append((sample, arm.arm_id))
        
        # Return arm with highest sample
        samples.sort(reverse=True)
        return samples[0][1]
    
    def select_arm_ucb(self, c: float = 2.0) -> str:
        """Select an arm using UCB1 (alternative to Thompson).
        
        Args:
            c: Exploration parameter.
        
        Returns:
            Selected arm ID.
        """
        import math
        
        arms = self.get_all_arms()
        if not arms:
            raise ValueError("No arms available")
        
        total_pulls = sum(a.pulls for a in arms)
        if total_pulls == 0:
            return random.choice(arms).arm_id
        
        ucb_values = []
        for arm in arms:
            if arm.pulls == 0:
                ucb = float("inf")
            else:
                exploitation = arm.success_rate
                exploration = c * math.sqrt(math.log(total_pulls) / arm.pulls)
                ucb = exploitation + exploration
            ucb_values.append((ucb, arm.arm_id))
        
        ucb_values.sort(reverse=True)
        return ucb_values[0][1]
    
    def update(
        self,
        arm_id: str,
        success: bool,
        reward: float | None = None,
        task_id: str = "",
    ) -> None:
        """Update arm statistics based on outcome.
        
        Args:
            arm_id: Arm that was used.
            success: Whether the episode succeeded.
            reward: Optional reward value (uses 1.0/0.0 if not provided).
            task_id: Optional task ID for history.
        """
        self._ensure_arm(arm_id)
        now = datetime.now(timezone.utc).isoformat()
        
        conn = self._get_conn()
        try:
            if success:
                conn.execute(
                    """
                    UPDATE arms 
                    SET alpha = alpha + 1, 
                        pulls = pulls + 1, 
                        successes = successes + 1,
                        last_updated = ?
                    WHERE arm_id = ?
                    """,
                    (now, arm_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE arms 
                    SET beta = beta + 1, 
                        pulls = pulls + 1, 
                        failures = failures + 1,
                        last_updated = ?
                    WHERE arm_id = ?
                    """,
                    (now, arm_id),
                )
            
            # Record in history
            reward_value = reward if reward is not None else (1.0 if success else 0.0)
            conn.execute(
                """
                INSERT INTO history (arm_id, success, reward, task_id, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (arm_id, int(success), reward_value, task_id, now),
            )
            
            conn.commit()
        finally:
            conn.close()
        
        logger.info(f"Updated arm {arm_id}: success={success}")
    
    def get_history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent history of arm selections.
        
        Args:
            limit: Maximum number of entries to return.
        
        Returns:
            List of history entries.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """
                SELECT arm_id, success, reward, task_id, timestamp
                FROM history ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            
            return [
                {
                    "arm_id": row[0],
                    "success": bool(row[1]),
                    "reward": row[2],
                    "task_id": row[3],
                    "timestamp": row[4],
                }
                for row in rows
            ]
        finally:
            conn.close()
    
    def export_state(self) -> dict[str, Any]:
        """Export bandit state for backup/transfer.
        
        Returns:
            Dictionary with all arm stats.
        """
        return {
            "arms": [a.to_dict() for a in self.get_all_arms()],
            "history_count": len(self.get_history(1000)),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
    
    def import_state(self, state: dict[str, Any]) -> None:
        """Import bandit state from backup.
        
        Args:
            state: Exported state dictionary.
        """
        conn = self._get_conn()
        try:
            for arm_data in state.get("arms", []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO arms 
                    (arm_id, alpha, beta, pulls, successes, failures, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        arm_data["arm_id"],
                        arm_data["alpha"],
                        arm_data["beta"],
                        arm_data["pulls"],
                        arm_data["successes"],
                        arm_data["failures"],
                        arm_data.get("last_updated", datetime.now(timezone.utc).isoformat()),
                    ),
                )
            conn.commit()
        finally:
            conn.close()


def create_bandit(
    db_path: Path | str | None = None,
    arms: list[str] | None = None,
) -> ThompsonBandit:
    """Create a bandit with default SWE-bench arms.
    
    Args:
        db_path: Path to SQLite database.
        arms: List of arm IDs (uses defaults if not provided).
    
    Returns:
        Configured ThompsonBandit.
    """
    if arms is None:
        try:
            from rfsn_upstream.strategies import list_arm_ids
            arms = list_arm_ids()
        except ImportError:
            # Fallback if strategies not available (shouldn't happen in prod)
            logger.warning("Could not import strategies. Using minimal default.")
            arms = ["swe_minimal_diff"]
    
    return ThompsonBandit(
        db_path=db_path,
        arms=arms,
    )
