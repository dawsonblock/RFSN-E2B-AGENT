"""Shared utilities for the RFSN dashboard."""

import json
import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any

# Default paths
ARTIFACTS_DIR = Path("artifacts")
BANDIT_DB = ARTIFACTS_DIR / "bandit.db"
OUTCOMES_DB = ARTIFACTS_DIR / "outcomes.db"
LEDGER_DIR = Path("rfsn_ledger")  # Adjust if typical location differs

def get_db_connection(db_path: Path) -> sqlite3.Connection:
    """Get a read-only connection to a SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def load_bandit_stats() -> pd.DataFrame:
    """Load current bandit statistics."""
    if not BANDIT_DB.exists():
        return pd.DataFrame()
        
    conn = get_db_connection(BANDIT_DB)
    try:
        query = "SELECT * FROM arms"
        df = pd.read_sql_query(query, conn)
        # Calculate derived stats like win rate if not explicit
        # (Though ArmStats usually has alpha/beta)
        if not df.empty and "successes" in df.columns and "pulls" in df.columns:
            df["win_rate"] = df["successes"] / df["pulls"].replace(0, 1)
        return df
    finally:
        conn.close()

def load_recent_history(limit: int = 100) -> pd.DataFrame:
    """Load recent bandit history."""
    if not BANDIT_DB.exists():
        return pd.DataFrame()
        
    conn = get_db_connection(BANDIT_DB)
    try:
        query = "SELECT * FROM history ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    finally:
        conn.close()

def load_outcomes() -> pd.DataFrame:
    """Load all outcomes."""
    if not OUTCOMES_DB.exists():
        return pd.DataFrame()
        
    conn = get_db_connection(OUTCOMES_DB)
    try:
        query = "SELECT * FROM episodes ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def parse_ledger(ledger_path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL ledger file."""
    if not ledger_path.exists():
        return []
    
    entries = []
    with open(ledger_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries

def list_ledgers(workspace_path: Path) -> list[Path]:
    """List available ledger files in a workspace."""
    ledger_location = workspace_path / ".rfsn_ledger"
    if not ledger_location.exists():
        return []
    
    return sorted(ledger_location.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
