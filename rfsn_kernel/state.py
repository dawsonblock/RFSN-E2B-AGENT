"""State Model - Explicit, typed state snapshots.

The state model provides:
- Immutable state snapshots per step
- Serializable to JSON for replay
- Reproducible state computation
- Filesystem state as hashes only (not content)

INVARIANT: State is immutable - each step produces a new state.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class TestStatus(Enum):
    """Status of the test suite."""
    
    NOT_RUN = "not_run"
    PASSING = "passing"
    FAILING = "failing"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class TestResult:
    """Immutable test execution result."""
    
    status: TestStatus
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_seconds: float = 0.0
    failing_tests: tuple[str, ...] = field(default_factory=tuple)
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "status": self.status.value,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "duration_seconds": self.duration_seconds,
            "failing_tests": list(self.failing_tests),
            "stdout": self.stdout[:1000],  # Truncate for serialization
            "stderr": self.stderr[:1000],
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestResult:
        """Deserialize from dictionary."""
        return cls(
            status=TestStatus(data["status"]),
            passed=data.get("passed", 0),
            failed=data.get("failed", 0),
            errors=data.get("errors", 0),
            skipped=data.get("skipped", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            failing_tests=tuple(data.get("failing_tests", [])),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
        )


@dataclass(frozen=True)
class TaskProgress:
    """Immutable task progress tracking."""
    
    step_index: int = 0
    total_proposals: int = 0
    accepted_proposals: int = 0
    rejected_proposals: int = 0
    patches_applied: int = 0
    tests_run: int = 0
    is_complete: bool = False
    completion_reason: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "step_index": self.step_index,
            "total_proposals": self.total_proposals,
            "accepted_proposals": self.accepted_proposals,
            "rejected_proposals": self.rejected_proposals,
            "patches_applied": self.patches_applied,
            "tests_run": self.tests_run,
            "is_complete": self.is_complete,
            "completion_reason": self.completion_reason,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskProgress:
        """Deserialize from dictionary."""
        return cls(
            step_index=data.get("step_index", 0),
            total_proposals=data.get("total_proposals", 0),
            accepted_proposals=data.get("accepted_proposals", 0),
            rejected_proposals=data.get("rejected_proposals", 0),
            patches_applied=data.get("patches_applied", 0),
            tests_run=data.get("tests_run", 0),
            is_complete=data.get("is_complete", False),
            completion_reason=data.get("completion_reason"),
        )


@dataclass(frozen=True)
class SafetyEnvelope:
    """Immutable safety constraints for the current task."""
    
    max_steps: int = 20
    max_patches: int = 10
    max_diff_lines: int = 500
    max_file_size_bytes: int = 100_000
    allowed_paths: tuple[str, ...] = field(default_factory=lambda: ("**/*.py",))
    forbidden_paths: tuple[str, ...] = field(
        default_factory=lambda: (".git/", "__pycache__/", "node_modules/", ".env")
    )
    allowed_actions: tuple[str, ...] = field(
        default_factory=lambda: ("modify_file", "create_file", "run_test")
    )
    rate_limit_per_minute: int = 60
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_steps": self.max_steps,
            "max_patches": self.max_patches,
            "max_diff_lines": self.max_diff_lines,
            "max_file_size_bytes": self.max_file_size_bytes,
            "allowed_paths": list(self.allowed_paths),
            "forbidden_paths": list(self.forbidden_paths),
            "allowed_actions": list(self.allowed_actions),
            "rate_limit_per_minute": self.rate_limit_per_minute,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyEnvelope:
        """Deserialize from dictionary."""
        return cls(
            max_steps=data.get("max_steps", 20),
            max_patches=data.get("max_patches", 10),
            max_diff_lines=data.get("max_diff_lines", 500),
            max_file_size_bytes=data.get("max_file_size_bytes", 100_000),
            allowed_paths=tuple(data.get("allowed_paths", ["**/*.py"])),
            forbidden_paths=tuple(data.get("forbidden_paths", [".git/", "__pycache__/"])),
            allowed_actions=tuple(data.get("allowed_actions", ["modify_file", "create_file", "run_test"])),
            rate_limit_per_minute=data.get("rate_limit_per_minute", 60),
        )


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable state snapshot at a point in time.
    
    INVARIANTS:
    - State is frozen (immutable)
    - State is serializable to JSON
    - State is reproducible from inputs
    - Filesystem state is hashes only (not content)
    """
    
    state_hash: str
    task_id: str
    workspace_root: str
    filesystem_hashes: tuple[tuple[str, str], ...]  # ((path, hash), ...)
    test_result: TestResult
    task_progress: TaskProgress
    safety_envelope: SafetyEnvelope
    env_fingerprint: str
    timestamp: str
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "state_hash": self.state_hash,
            "task_id": self.task_id,
            "workspace_root": self.workspace_root,
            "filesystem_hashes": dict(self.filesystem_hashes),
            "test_result": self.test_result.to_dict(),
            "task_progress": self.task_progress.to_dict(),
            "safety_envelope": self.safety_envelope.to_dict(),
            "env_fingerprint": self.env_fingerprint,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSnapshot:
        """Deserialize from dictionary."""
        fs_hashes = data.get("filesystem_hashes", {})
        if isinstance(fs_hashes, dict):
            fs_hashes = tuple(sorted(fs_hashes.items()))
        
        return cls(
            state_hash=data["state_hash"],
            task_id=data["task_id"],
            workspace_root=data["workspace_root"],
            filesystem_hashes=fs_hashes,
            test_result=TestResult.from_dict(data["test_result"]),
            task_progress=TaskProgress.from_dict(data["task_progress"]),
            safety_envelope=SafetyEnvelope.from_dict(data["safety_envelope"]),
            env_fingerprint=data["env_fingerprint"],
            timestamp=data["timestamp"],
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> StateSnapshot:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]  # Truncate for readability
    except (OSError, IOError):
        return "ERROR"


def compute_env_fingerprint() -> str:
    """Compute a fingerprint of the execution environment."""
    import platform
    import sys
    
    components = [
        f"python:{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        f"platform:{platform.system().lower()}",
        f"arch:{platform.machine()}",
    ]
    fingerprint = "|".join(components)
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


def compute_state_hash(
    task_id: str,
    filesystem_hashes: dict[str, str],
    test_result: TestResult,
    task_progress: TaskProgress,
) -> str:
    """Compute a deterministic hash of the state."""
    components = [
        task_id,
        json.dumps(dict(sorted(filesystem_hashes.items())), sort_keys=True),
        test_result.status.value,
        str(task_progress.step_index),
        str(task_progress.patches_applied),
    ]
    content = "|".join(components)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def create_initial_state(
    task_id: str,
    workspace_root: Path,
    safety_envelope: SafetyEnvelope | None = None,
    file_patterns: list[str] | None = None,
) -> StateSnapshot:
    """Create an initial state snapshot from a workspace.
    
    Args:
        task_id: Unique identifier for this task.
        workspace_root: Path to the workspace directory.
        safety_envelope: Safety constraints (uses defaults if not provided).
        file_patterns: Glob patterns for files to hash (defaults to *.py).
    
    Returns:
        Initial StateSnapshot.
    """
    workspace_root = Path(workspace_root).resolve()
    file_patterns = file_patterns or ["**/*.py"]
    safety_envelope = safety_envelope or SafetyEnvelope()
    
    # Compute filesystem hashes
    filesystem_hashes: dict[str, str] = {}
    for pattern in file_patterns:
        for path in workspace_root.glob(pattern):
            if path.is_file():
                rel_path = str(path.relative_to(workspace_root))
                # Skip forbidden paths
                if not any(fp in rel_path for fp in safety_envelope.forbidden_paths):
                    filesystem_hashes[rel_path] = compute_file_hash(path)
    
    # Initial test result (not run yet)
    test_result = TestResult(status=TestStatus.NOT_RUN)
    
    # Initial task progress
    task_progress = TaskProgress(step_index=0)
    
    # Compute state hash
    state_hash = compute_state_hash(
        task_id, filesystem_hashes, test_result, task_progress
    )
    
    # Create timestamp
    timestamp = datetime.now(timezone.utc).isoformat()
    
    return StateSnapshot(
        state_hash=state_hash,
        task_id=task_id,
        workspace_root=str(workspace_root),
        filesystem_hashes=tuple(sorted(filesystem_hashes.items())),
        test_result=test_result,
        task_progress=task_progress,
        safety_envelope=safety_envelope,
        env_fingerprint=compute_env_fingerprint(),
        timestamp=timestamp,
    )


def update_state(
    state: StateSnapshot,
    *,
    filesystem_hashes: dict[str, str] | None = None,
    test_result: TestResult | None = None,
    task_progress: TaskProgress | None = None,
) -> StateSnapshot:
    """Create a new state snapshot with updated fields.
    
    INVARIANT: Original state is never modified.
    
    Args:
        state: Current state snapshot.
        filesystem_hashes: Updated filesystem hashes (merges with existing).
        test_result: Updated test result.
        task_progress: Updated task progress.
    
    Returns:
        New StateSnapshot with updates.
    """
    # Merge filesystem hashes
    new_fs_hashes = dict(state.filesystem_hashes)
    if filesystem_hashes:
        new_fs_hashes.update(filesystem_hashes)
    
    # Use updated or existing values
    new_test_result = test_result or state.test_result
    new_task_progress = task_progress or state.task_progress
    
    # Compute new state hash
    new_state_hash = compute_state_hash(
        state.task_id, new_fs_hashes, new_test_result, new_task_progress
    )
    
    # Create new timestamp
    new_timestamp = datetime.now(timezone.utc).isoformat()
    
    return StateSnapshot(
        state_hash=new_state_hash,
        task_id=state.task_id,
        workspace_root=state.workspace_root,
        filesystem_hashes=tuple(sorted(new_fs_hashes.items())),
        test_result=new_test_result,
        task_progress=new_task_progress,
        safety_envelope=state.safety_envelope,
        env_fingerprint=state.env_fingerprint,
        timestamp=new_timestamp,
    )
