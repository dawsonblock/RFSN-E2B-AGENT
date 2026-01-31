"""Ledger - Ground Truth Logging.

The Ledger records every step of execution for:
- Replay
- Audit
- Failure analysis
- Training data extraction

INVARIANTS:
1. Every step is logged
2. Ledger entries are immutable
3. Rejected actions produce evidence
4. Ledger supports deterministic replay
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from .controller import ExecutionResult
from .gate import GateDecision
from .proposal import Proposal
from .state import StateSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LedgerEntry:
    """Immutable ledger entry for a single step.
    
    INVARIANT: Entries are frozen and cannot be modified.
    """
    
    entry_id: str
    step_index: int
    entry_type: str  # "proposal" | "decision" | "execution" | "rejection"
    state_hash: str
    proposal_id: str | None
    data: tuple[tuple[str, Any], ...]  # Immutable data dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_id": self.entry_id,
            "step_index": self.step_index,
            "entry_type": self.entry_type,
            "state_hash": self.state_hash,
            "proposal_id": self.proposal_id,
            "data": dict(self.data),
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        """Deserialize from dictionary."""
        entry_data = data.get("data", {})
        if isinstance(entry_data, dict):
            entry_data = tuple(sorted(entry_data.items()))
        
        return cls(
            entry_id=data["entry_id"],
            step_index=data["step_index"],
            entry_type=data["entry_type"],
            state_hash=data["state_hash"],
            proposal_id=data.get("proposal_id"),
            data=entry_data,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> LedgerEntry:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class Ledger:
    """Ground truth logging for all kernel operations.
    
    The Ledger records every proposal, decision, and execution
    in a JSONL file for replay and audit.
    
    INVARIANTS:
    1. Every step is logged
    2. Entries are append-only
    3. Rejected actions produce evidence
    4. Ledger supports deterministic replay
    
    Usage:
        ledger = Ledger("run.jsonl")
        
        # Record step
        ledger.record(state, proposal, result)
        
        # Record rejection
        ledger.record_rejection(state, proposal, decision)
        
        # Replay
        for entry in ledger.replay():
            process(entry)
    """
    
    def __init__(self, path: Path | str):
        """Initialize ledger with file path.
        
        Args:
            path: Path to JSONL file for ledger entries.
        """
        self.path = Path(path)
        self._entry_count = 0
        self._step_index = 0
        
        # Create parent directory if needed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        # Count existing entries if file exists
        if self.path.exists():
            with open(self.path) as f:
                self._entry_count = sum(1 for _ in f)
    
    def record(
        self,
        state: StateSnapshot,
        proposal: Proposal,
        result: ExecutionResult,
    ) -> LedgerEntry:
        """Record a successful execution step.
        
        Args:
            state: State before execution.
            proposal: The executed proposal.
            result: The execution result.
        
        Returns:
            The created LedgerEntry.
        """
        entry = LedgerEntry(
            entry_id=f"e{self._entry_count:06d}",
            step_index=self._step_index,
            entry_type="execution",
            state_hash=state.state_hash,
            proposal_id=proposal.proposal_id,
            data=tuple(sorted({
                "intent": proposal.intent.value,
                "target": proposal.target,
                "success": result.success,
                "return_code": result.return_code,
                "duration_seconds": result.duration_seconds,
                "changed_files": list(result.changed_files),
            }.items())),
        )
        
        self._write_entry(entry)
        self._entry_count += 1
        self._step_index += 1
        
        return entry
    
    def record_rejection(
        self,
        state: StateSnapshot,
        proposal: Proposal,
        decision: GateDecision,
    ) -> LedgerEntry:
        """Record a rejected proposal.
        
        INVARIANT: Rejected actions produce evidence.
        
        Args:
            state: State at rejection time.
            proposal: The rejected proposal.
            decision: The gate decision with reason.
        
        Returns:
            The created LedgerEntry.
        """
        entry = LedgerEntry(
            entry_id=f"e{self._entry_count:06d}",
            step_index=self._step_index,
            entry_type="rejection",
            state_hash=state.state_hash,
            proposal_id=proposal.proposal_id,
            data=tuple(sorted({
                "intent": proposal.intent.value,
                "target": proposal.target,
                "reason": decision.reason,
                "evidence": dict(decision.evidence),
            }.items())),
        )
        
        self._write_entry(entry)
        self._entry_count += 1
        # Don't increment step_index for rejections
        
        return entry
    
    def record_state(self, state: StateSnapshot, note: str = "") -> LedgerEntry:
        """Record a state snapshot.
        
        Args:
            state: State to record.
            note: Optional note about this state.
        
        Returns:
            The created LedgerEntry.
        """
        entry = LedgerEntry(
            entry_id=f"e{self._entry_count:06d}",
            step_index=self._step_index,
            entry_type="state",
            state_hash=state.state_hash,
            proposal_id=None,
            data=tuple(sorted({
                "task_id": state.task_id,
                "step_index": state.task_progress.step_index,
                "patches_applied": state.task_progress.patches_applied,
                "test_status": state.test_result.status.value,
                "note": note,
            }.items())),
        )
        
        self._write_entry(entry)
        self._entry_count += 1
        
        return entry
    
    def record_completion(
        self,
        state: StateSnapshot,
        success: bool,
        reason: str,
    ) -> LedgerEntry:
        """Record task completion.
        
        Args:
            state: Final state.
            success: Whether task succeeded.
            reason: Completion reason.
        
        Returns:
            The created LedgerEntry.
        """
        entry = LedgerEntry(
            entry_id=f"e{self._entry_count:06d}",
            step_index=self._step_index,
            entry_type="completion",
            state_hash=state.state_hash,
            proposal_id=None,
            data=tuple(sorted({
                "success": success,
                "reason": reason,
                "total_steps": self._step_index,
                "total_entries": self._entry_count,
            }.items())),
        )
        
        self._write_entry(entry)
        self._entry_count += 1
        
        return entry
    
    def _write_entry(self, entry: LedgerEntry) -> None:
        """Write entry to JSONL file."""
        with open(self.path, "a") as f:
            f.write(entry.to_json() + "\n")
    
    def replay(self) -> list[LedgerEntry]:
        """Replay all entries from the ledger.
        
        Returns:
            List of all LedgerEntry objects in order.
        """
        entries = []
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(LedgerEntry.from_json(line))
        return entries
    
    def replay_iter(self):
        """Iterate over entries (memory-efficient).
        
        Yields:
            LedgerEntry objects one at a time.
        """
        if self.path.exists():
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield LedgerEntry.from_json(line)
    
    def get_rejections(self) -> list[LedgerEntry]:
        """Get all rejection entries.
        
        Returns:
            List of rejection LedgerEntry objects.
        """
        return [e for e in self.replay() if e.entry_type == "rejection"]
    
    def get_executions(self) -> list[LedgerEntry]:
        """Get all execution entries.
        
        Returns:
            List of execution LedgerEntry objects.
        """
        return [e for e in self.replay() if e.entry_type == "execution"]
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of the ledger.
        
        Returns:
            Dictionary with summary stats.
        """
        entries = self.replay()
        
        executions = [e for e in entries if e.entry_type == "execution"]
        rejections = [e for e in entries if e.entry_type == "rejection"]
        completions = [e for e in entries if e.entry_type == "completion"]
        
        successful_execs = [
            e for e in executions 
            if dict(e.data).get("success", False)
        ]
        
        return {
            "total_entries": len(entries),
            "executions": len(executions),
            "successful_executions": len(successful_execs),
            "rejections": len(rejections),
            "completions": len(completions),
            "completed": len(completions) > 0,
            "success": any(dict(e.data).get("success", False) for e in completions),
        }
    
    def export_for_training(self) -> list[dict[str, Any]]:
        """Export ledger data in format suitable for learning.
        
        Returns:
            List of dictionaries with training-relevant data.
        """
        training_data = []
        
        for entry in self.replay():
            data = dict(entry.data)
            
            if entry.entry_type == "execution":
                training_data.append({
                    "type": "execution",
                    "intent": data.get("intent"),
                    "target": data.get("target"),
                    "success": data.get("success", False),
                    "state_hash": entry.state_hash,
                })
            
            elif entry.entry_type == "rejection":
                training_data.append({
                    "type": "rejection",
                    "intent": data.get("intent"),
                    "target": data.get("target"),
                    "reason": data.get("reason"),
                    "state_hash": entry.state_hash,
                })
        
        return training_data


def create_ledger(task_id: str, base_dir: Path | str = ".") -> Ledger:
    """Create a ledger for a task.
    
    Args:
        task_id: Task identifier.
        base_dir: Base directory for ledger files.
    
    Returns:
        Ledger instance.
    """
    base_dir = Path(base_dir)
    ledger_dir = base_dir / ".rfsn_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger_path = ledger_dir / f"{task_id}_{timestamp}.jsonl"
    
    return Ledger(ledger_path)
