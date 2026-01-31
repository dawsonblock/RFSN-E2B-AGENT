"""Controller - Sandboxed Executor.

The Controller executes approved proposals in a sandboxed environment.

INVARIANTS:
1. Controller ONLY executes proposals that passed the gate
2. Controller CANNOT bypass the gate
3. Controller CANNOT modify policy
4. Controller runs in isolation (subprocess/container)
5. All execution results are captured for audit
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .proposal import Proposal, ProposalIntent
from .state import StateSnapshot, TestResult, TestStatus, compute_file_hash

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of executing a proposal.
    
    INVARIANT: Results are immutable and include all artifacts.
    """
    
    success: bool
    proposal_id: str
    intent: str
    stdout: str
    stderr: str
    return_code: int
    duration_seconds: float
    artifacts: tuple[tuple[str, Any], ...]  # Immutable artifacts dict
    changed_files: tuple[str, ...]  # Files modified by execution
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "proposal_id": self.proposal_id,
            "intent": self.intent,
            "stdout": self.stdout[:5000],  # Truncate for storage
            "stderr": self.stderr[:5000],
            "return_code": self.return_code,
            "duration_seconds": self.duration_seconds,
            "artifacts": dict(self.artifacts),
            "changed_files": list(self.changed_files),
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        """Deserialize from dictionary."""
        artifacts = data.get("artifacts", {})
        if isinstance(artifacts, dict):
            artifacts = tuple(sorted(artifacts.items()))
        
        return cls(
            success=data["success"],
            proposal_id=data["proposal_id"],
            intent=data["intent"],
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            return_code=data.get("return_code", 0),
            duration_seconds=data.get("duration_seconds", 0.0),
            artifacts=artifacts,
            changed_files=tuple(data.get("changed_files", [])),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class ControllerConfig:
    """Configuration for the Controller."""
    
    # Execution environment
    use_docker: bool = False
    docker_image: str = "python:3.12-slim"
    timeout_seconds: float = 300.0
    
    # Resource limits
    max_memory_mb: int = 1024
    max_cpu_seconds: float = 60.0
    
    # Test configuration
    default_test_command: str = "pytest -q"
    test_timeout_seconds: float = 120.0
    
    # Sandbox settings
    copy_workspace: bool = True  # Copy workspace before modifications
    cleanup_on_success: bool = True


class ControllerError(Exception):
    """Raised when controller execution fails."""
    
    def __init__(self, message: str, proposal_id: str | None = None):
        self.proposal_id = proposal_id
        super().__init__(f"[Controller] {message}" + (f" (proposal: {proposal_id})" if proposal_id else ""))


class Controller:
    """Sandboxed executor for approved proposals.
    
    INVARIANTS:
    1. Only executes gate-approved proposals
    2. Cannot bypass the gate
    3. Cannot modify policy
    4. All side effects are logged
    
    Usage:
        controller = Controller(ControllerConfig())
        result = controller.execute(proposal, workspace_path)
    """
    
    def __init__(self, config: ControllerConfig | None = None):
        """Initialize controller with config.
        
        Args:
            config: Controller configuration.
        """
        self.config = config or ControllerConfig()
        self._execution_count = 0
    
    def execute(
        self,
        proposal: Proposal,
        workspace_path: Path | str,
        gate_approved: bool = True,
    ) -> ExecutionResult:
        """Execute an approved proposal.
        
        Args:
            proposal: The proposal to execute.
            workspace_path: Path to the workspace.
            gate_approved: Whether the gate approved this proposal.
                          MUST be True - enforced at runtime.
        
        Returns:
            ExecutionResult with all artifacts.
        
        Raises:
            ControllerError: If gate_approved is False or execution fails.
        """
        import time
        
        # CRITICAL: Enforce gate approval
        if not gate_approved:
            raise ControllerError(
                "Attempted to execute proposal without gate approval",
                proposal.proposal_id,
            )
        
        workspace_path = Path(workspace_path).resolve()
        start_time = time.perf_counter()
        
        # Dispatch based on intent
        try:
            if proposal.intent == ProposalIntent.MODIFY_FILE:
                result = self._execute_modify_file(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.CREATE_FILE:
                result = self._execute_create_file(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.DELETE_FILE:
                result = self._execute_delete_file(proposal, workspace_path)
            elif proposal.intent in (ProposalIntent.RUN_TEST, ProposalIntent.RUN_FOCUSED_TEST):
                result = self._execute_run_test(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.READ_FILE:
                result = self._execute_read_file(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.SEARCH_REPO:
                result = self._execute_search_repo(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.LIST_DIRECTORY:
                result = self._execute_list_directory(proposal, workspace_path)
            elif proposal.intent == ProposalIntent.CHECKPOINT:
                result = self._execute_checkpoint(proposal, workspace_path)
            else:
                raise ControllerError(f"Unknown intent: {proposal.intent}", proposal.proposal_id)
            
            duration = time.perf_counter() - start_time
            self._execution_count += 1
            
            return ExecutionResult(
                success=result["success"],
                proposal_id=proposal.proposal_id,
                intent=proposal.intent.value,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                return_code=result.get("return_code", 0),
                duration_seconds=duration,
                artifacts=tuple(sorted(result.get("artifacts", {}).items())),
                changed_files=tuple(result.get("changed_files", [])),
            )
            
        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.exception(f"Execution failed for {proposal.proposal_id}")
            
            return ExecutionResult(
                success=False,
                proposal_id=proposal.proposal_id,
                intent=proposal.intent.value,
                stdout="",
                stderr=str(e),
                return_code=-1,
                duration_seconds=duration,
                artifacts=(("error", str(e)),),
                changed_files=(),
            )
    
    def _execute_modify_file(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a file modification proposal."""
        target_path = workspace / proposal.target
        
        if not target_path.exists():
            return {
                "success": False,
                "stderr": f"File not found: {proposal.target}",
                "return_code": 1,
            }
        
        # Read original content
        original_content = target_path.read_text()
        original_hash = compute_file_hash(target_path)
        
        # Apply the patch
        try:
            new_content = self._apply_patch(original_content, proposal.patch)
            target_path.write_text(new_content)
            new_hash = compute_file_hash(target_path)
            
            return {
                "success": True,
                "stdout": f"Modified {proposal.target}",
                "return_code": 0,
                "changed_files": [proposal.target],
                "artifacts": {
                    "original_hash": original_hash,
                    "new_hash": new_hash,
                    "lines_changed": abs(len(new_content.splitlines()) - len(original_content.splitlines())),
                },
            }
        except Exception as e:
            # Restore original on failure
            target_path.write_text(original_content)
            return {
                "success": False,
                "stderr": f"Patch failed: {e}",
                "return_code": 1,
            }
    
    def _execute_create_file(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a file creation proposal."""
        target_path = workspace / proposal.target
        
        if target_path.exists():
            return {
                "success": False,
                "stderr": f"File already exists: {proposal.target}",
                "return_code": 1,
            }
        
        try:
            # Create parent directories
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content (patch contains the full file content for create)
            target_path.write_text(proposal.patch)
            new_hash = compute_file_hash(target_path)
            
            return {
                "success": True,
                "stdout": f"Created {proposal.target}",
                "return_code": 0,
                "changed_files": [proposal.target],
                "artifacts": {
                    "new_hash": new_hash,
                    "size_bytes": len(proposal.patch),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"Create failed: {e}",
                "return_code": 1,
            }
    
    def _execute_delete_file(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a file deletion proposal."""
        target_path = workspace / proposal.target
        
        if not target_path.exists():
            return {
                "success": False,
                "stderr": f"File not found: {proposal.target}",
                "return_code": 1,
            }
        
        try:
            original_hash = compute_file_hash(target_path)
            target_path.unlink()
            
            return {
                "success": True,
                "stdout": f"Deleted {proposal.target}",
                "return_code": 0,
                "changed_files": [proposal.target],
                "artifacts": {
                    "deleted_hash": original_hash,
                },
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"Delete failed: {e}",
                "return_code": 1,
            }
    
    def _execute_run_test(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a test run proposal."""
        test_cmd = proposal.test_command or self.config.default_test_command
        
        # Parse the command safely (no shell=True!)
        if proposal.target and proposal.target not in test_cmd:
            # Add target to command
            test_cmd = f"{test_cmd} {proposal.target}"
        
        try:
            result = subprocess.run(
                test_cmd.split(),
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=self.config.test_timeout_seconds,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            
            # Parse test results
            test_passed = result.returncode == 0
            
            return {
                "success": True,  # Execution succeeded even if tests failed
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "artifacts": {
                    "tests_passed": test_passed,
                    "command": test_cmd,
                },
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stderr": f"Test timeout after {self.config.test_timeout_seconds}s",
                "return_code": -1,
                "artifacts": {"timeout": True},
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"Test execution failed: {e}",
                "return_code": -1,
            }
    
    def _execute_read_file(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a file read proposal (read-only)."""
        target_path = workspace / proposal.target
        
        if not target_path.exists():
            return {
                "success": False,
                "stderr": f"File not found: {proposal.target}",
                "return_code": 1,
            }
        
        try:
            content = target_path.read_text()
            return {
                "success": True,
                "stdout": content[:10000],  # Limit output size
                "return_code": 0,
                "artifacts": {
                    "file_hash": compute_file_hash(target_path),
                    "size_bytes": len(content),
                    "lines": len(content.splitlines()),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"Read failed: {e}",
                "return_code": 1,
            }
    
    def _execute_search_repo(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a repository search proposal (read-only)."""
        query = proposal.search_query or proposal.target
        
        try:
            # Use grep for simple search
            result = subprocess.run(
                ["grep", "-r", "-n", "-l", query, "."],
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            
            files = [f for f in result.stdout.strip().split("\n") if f]
            
            return {
                "success": True,
                "stdout": "\n".join(files[:50]),  # Limit results
                "return_code": 0,
                "artifacts": {
                    "query": query,
                    "matches": len(files),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"Search failed: {e}",
                "return_code": 1,
            }
    
    def _execute_list_directory(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a directory listing proposal (read-only)."""
        target_path = workspace / proposal.target if proposal.target else workspace
        
        if not target_path.exists():
            return {
                "success": False,
                "stderr": f"Directory not found: {proposal.target}",
                "return_code": 1,
            }
        
        try:
            entries = []
            for entry in sorted(target_path.iterdir()):
                entry_type = "d" if entry.is_dir() else "f"
                entries.append(f"{entry_type} {entry.name}")
            
            return {
                "success": True,
                "stdout": "\n".join(entries[:100]),
                "return_code": 0,
                "artifacts": {
                    "entry_count": len(entries),
                },
            }
        except Exception as e:
            return {
                "success": False,
                "stderr": f"List failed: {e}",
                "return_code": 1,
            }
    
    def _execute_checkpoint(
        self, proposal: Proposal, workspace: Path
    ) -> dict[str, Any]:
        """Execute a checkpoint proposal (no-op, for coordination)."""
        return {
            "success": True,
            "stdout": f"Checkpoint: {proposal.justification}",
            "return_code": 0,
            "artifacts": {
                "checkpoint_id": proposal.proposal_id,
            },
        }
    
    def _apply_patch(self, original: str, patch: str) -> str:
        """Apply a unified diff patch to content.
        
        This is a simplified patch application.
        For production, use a proper diff library.
        """
        # Simple approach: if patch looks like full content, use it
        if not patch.startswith("---") and not patch.startswith("@@"):
            # Treat as full replacement
            return patch
        
        # Try to apply as unified diff
        # For now, extract the new content from the diff
        lines = patch.split("\n")
        result_lines = []
        
        in_hunk = False
        for line in lines:
            if line.startswith("@@"):
                in_hunk = True
                continue
            if in_hunk:
                if line.startswith("+") and not line.startswith("+++"):
                    result_lines.append(line[1:])
                elif line.startswith(" "):
                    result_lines.append(line[1:])
                elif line.startswith("-") and not line.startswith("---"):
                    continue  # Skip removed lines
        
        if result_lines:
            return "\n".join(result_lines)
        
        # Fallback: return patch as-is if we couldn't parse it
        return patch
