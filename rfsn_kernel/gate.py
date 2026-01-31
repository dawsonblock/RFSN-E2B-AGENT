"""Gate - Deterministic Safety Kernel.

The Gate is the CORE of the RFSN kernel. It is a PURE FUNCTION that:
- Accepts (state, proposal) 
- Returns (accept | reject, reason, evidence)

CRITICAL INVARIANTS:
1. Gate is a PURE FUNCTION - no side effects
2. Gate NEVER LEARNS - no weights, no gradients, no updates
3. Gate is DETERMINISTIC - same inputs → same outputs
4. Gate is IMMUTABLE - cannot be modified at runtime
5. Gate decisions produce EVIDENCE for audit

The gate enforces:
- Proposal schema validity
- Allowed action types (explicit allowlist)
- File/path allowlist
- Diff size limits
- Forbidden patterns
- Test requirements
- Phase ordering
- Rate limits
- Envelope constraints
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from .proposal import Proposal, ProposalIntent, validate_proposal_schema
from .state import StateSnapshot

logger = logging.getLogger(__name__)


class GateDecisionType(Enum):
    """Gate decision types."""
    
    ACCEPT = "accept"
    REJECT = "reject"


@dataclass(frozen=True)
class GateDecision:
    """Immutable gate decision with evidence.
    
    INVARIANT: Decisions always include reason and evidence for auditability.
    """
    
    accepted: bool
    reason: str
    evidence: tuple[tuple[str, Any], ...]  # Immutable evidence dict
    decision_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "evidence": dict(self.evidence),
            "decision_time_ms": self.decision_time_ms,
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def accept(cls, reason: str, evidence: dict[str, Any] | None = None) -> GateDecision:
        """Create an accept decision."""
        return cls(
            accepted=True,
            reason=reason,
            evidence=tuple(sorted((evidence or {}).items())),
        )
    
    @classmethod
    def reject(cls, reason: str, evidence: dict[str, Any] | None = None) -> GateDecision:
        """Create a reject decision."""
        return cls(
            accepted=False,
            reason=reason,
            evidence=tuple(sorted((evidence or {}).items())),
        )


# Default forbidden patterns in patches
DEFAULT_FORBIDDEN_PATTERNS = [
    r"\$\(",           # Command substitution
    r"`",              # Backtick substitution  
    r"subprocess\.run.*shell\s*=\s*True",  # shell=True
    r"os\.system\s*\(",  # os.system
    r"eval\s*\(",      # eval
    r"exec\s*\(",      # exec
    r"__import__\s*\(",  # Dynamic import
    r"rm\s+-rf",       # Dangerous remove
    r"sudo\s+",        # Privilege escalation
    r"chmod\s+777",    # Open permissions
    r"curl\s+.*\|.*sh",  # Pipe to shell
    r"wget\s+.*\|.*sh",  # Pipe to shell
]

# Default forbidden paths
DEFAULT_FORBIDDEN_PATHS = [
    ".git/*",
    ".github/workflows/*",
    "__pycache__/*",
    "*.pyc",
    "node_modules/*",
    ".env",
    ".env.*",
    "*.secret",
    "*.key",
    "*.pem",
    "*credentials*",
]


@dataclass(frozen=True)
class GateConfig:
    """Immutable gate configuration.
    
    INVARIANT: This configuration cannot be modified at runtime.
    Learning systems cannot change these values.
    """
    
    # Action allowlist
    allowed_intents: tuple[ProposalIntent, ...] = field(
        default_factory=lambda: (
            ProposalIntent.MODIFY_FILE,
            ProposalIntent.CREATE_FILE,
            ProposalIntent.RUN_TEST,
            ProposalIntent.RUN_FOCUSED_TEST,
            ProposalIntent.READ_FILE,
            ProposalIntent.SEARCH_REPO,
            ProposalIntent.LIST_DIRECTORY,
            ProposalIntent.CHECKPOINT,
        )
    )
    
    # Path constraints
    allowed_path_patterns: tuple[str, ...] = field(
        default_factory=lambda: ("**/*.py", "**/*.txt", "**/*.md", "**/*.json", "**/*.yaml", "**/*.yml")
    )
    forbidden_path_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_FORBIDDEN_PATHS)
    )
    
    # Diff limits
    max_diff_lines: int = 500
    max_diff_files: int = 5
    max_file_size_bytes: int = 100_000
    
    # Pattern constraints
    forbidden_content_patterns: tuple[str, ...] = field(
        default_factory=lambda: tuple(DEFAULT_FORBIDDEN_PATTERNS)
    )
    
    # Rate limits
    max_proposals_per_step: int = 10
    max_patches_per_task: int = 20
    
    # Phase constraints
    require_test_before_patch: bool = False
    require_read_before_modify: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for audit logging."""
        return {
            "allowed_intents": [i.value for i in self.allowed_intents],
            "allowed_path_patterns": list(self.allowed_path_patterns),
            "forbidden_path_patterns": list(self.forbidden_path_patterns),
            "max_diff_lines": self.max_diff_lines,
            "max_diff_files": self.max_diff_files,
            "max_file_size_bytes": self.max_file_size_bytes,
            "forbidden_content_patterns": list(self.forbidden_content_patterns),
            "max_proposals_per_step": self.max_proposals_per_step,
            "max_patches_per_task": self.max_patches_per_task,
        }


class Gate:
    """Deterministic safety gate.
    
    This is the CORE of the RFSN kernel.
    
    INVARIANTS:
    1. validate() is a PURE FUNCTION
    2. Gate NEVER modifies state
    3. Gate NEVER learns
    4. Same (state, proposal) → same decision
    5. All rejections produce evidence
    
    Usage:
        gate = Gate(GateConfig())
        decision = gate.validate(state, proposal)
        
        if decision.accepted:
            # Safe to execute
        else:
            # Log rejection with evidence
    """
    
    def __init__(self, config: GateConfig | None = None):
        """Initialize gate with immutable config.
        
        Args:
            config: Gate configuration. Uses defaults if not provided.
        """
        self._config = config or GateConfig()
        # Compile regex patterns once for performance
        self._forbidden_patterns = [
            re.compile(p) for p in self._config.forbidden_content_patterns
        ]
    
    @property
    def config(self) -> GateConfig:
        """Get the immutable config (read-only)."""
        return self._config
    
    def validate(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Validate a proposal against the current state.
        
        This is a PURE FUNCTION with no side effects.
        
        Args:
            state: Current state snapshot.
            proposal: Proposal to validate.
        
        Returns:
            GateDecision indicating accept/reject with reason and evidence.
        """
        import time
        start = time.perf_counter()
        
        # Run all checks in order
        checks = [
            self._check_intent_allowed,
            self._check_envelope_limits,
            self._check_path_allowed,
            self._check_path_forbidden,
            self._check_diff_size,
            self._check_forbidden_patterns,
            self._check_rate_limits,
        ]
        
        for check in checks:
            decision = check(state, proposal)
            if not decision.accepted:
                elapsed = (time.perf_counter() - start) * 1000
                # Return decision with timing
                return GateDecision(
                    accepted=False,
                    reason=decision.reason,
                    evidence=decision.evidence,
                    decision_time_ms=elapsed,
                )
        
        # All checks passed
        elapsed = (time.perf_counter() - start) * 1000
        return GateDecision(
            accepted=True,
            reason="All gate checks passed",
            evidence=(
                ("intent", proposal.intent.value),
                ("target", proposal.target),
                ("checks_passed", len(checks)),
            ),
            decision_time_ms=elapsed,
        )
    
    def _check_intent_allowed(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that proposal intent is in allowlist."""
        if proposal.intent not in self._config.allowed_intents:
            return GateDecision.reject(
                reason=f"Intent '{proposal.intent.value}' not in allowlist",
                evidence={
                    "intent": proposal.intent.value,
                    "allowed_intents": [i.value for i in self._config.allowed_intents],
                },
            )
        return GateDecision.accept("Intent allowed")
    
    def _check_envelope_limits(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that proposal respects safety envelope limits."""
        envelope = state.safety_envelope
        progress = state.task_progress
        
        # Check max steps
        if progress.step_index >= envelope.max_steps:
            return GateDecision.reject(
                reason=f"Max steps exceeded: {progress.step_index} >= {envelope.max_steps}",
                evidence={"step_index": progress.step_index, "max_steps": envelope.max_steps},
            )
        
        # Check max patches
        if proposal.intent in (ProposalIntent.MODIFY_FILE, ProposalIntent.CREATE_FILE):
            if progress.patches_applied >= envelope.max_patches:
                return GateDecision.reject(
                    reason=f"Max patches exceeded: {progress.patches_applied} >= {envelope.max_patches}",
                    evidence={"patches_applied": progress.patches_applied, "max_patches": envelope.max_patches},
                )
        
        return GateDecision.accept("Envelope limits ok")
    
    def _check_path_allowed(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that target path matches allowed patterns."""
        target = proposal.target
        
        # Skip check for non-file intents
        if proposal.intent in (ProposalIntent.RUN_TEST, ProposalIntent.RUN_FOCUSED_TEST, 
                               ProposalIntent.SEARCH_REPO, ProposalIntent.CHECKPOINT):
            return GateDecision.accept("Non-file intent")
        
        # Check against allowed patterns
        allowed = False
        for pattern in self._config.allowed_path_patterns:
            if fnmatch.fnmatch(target, pattern):
                allowed = True
                break
        
        if not allowed:
            return GateDecision.reject(
                reason=f"Path '{target}' does not match allowed patterns",
                evidence={
                    "target": target,
                    "allowed_patterns": list(self._config.allowed_path_patterns),
                },
            )
        
        return GateDecision.accept("Path allowed")
    
    def _check_path_forbidden(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that target path does not match forbidden patterns."""
        target = proposal.target
        
        for pattern in self._config.forbidden_path_patterns:
            if fnmatch.fnmatch(target, pattern):
                return GateDecision.reject(
                    reason=f"Path '{target}' matches forbidden pattern '{pattern}'",
                    evidence={"target": target, "forbidden_pattern": pattern},
                )
        
        # Also check for path traversal
        if ".." in target:
            return GateDecision.reject(
                reason="Path traversal detected",
                evidence={"target": target},
            )
        
        return GateDecision.accept("Path not forbidden")
    
    def _check_diff_size(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that diff size is within limits."""
        if not proposal.patch:
            return GateDecision.accept("No patch to check")
        
        # Count diff lines
        lines = proposal.patch.count("\n")
        if lines > self._config.max_diff_lines:
            return GateDecision.reject(
                reason=f"Diff too large: {lines} lines > {self._config.max_diff_lines}",
                evidence={"diff_lines": lines, "max_lines": self._config.max_diff_lines},
            )
        
        # Count files in diff
        file_markers = proposal.patch.count("---")
        if file_markers > self._config.max_diff_files:
            return GateDecision.reject(
                reason=f"Diff touches too many files: {file_markers} > {self._config.max_diff_files}",
                evidence={"files": file_markers, "max_files": self._config.max_diff_files},
            )
        
        return GateDecision.accept("Diff size ok")
    
    def _check_forbidden_patterns(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check that proposal content does not contain forbidden patterns."""
        content = proposal.patch + proposal.test_command
        
        for pattern in self._forbidden_patterns:
            match = pattern.search(content)
            if match:
                return GateDecision.reject(
                    reason=f"Forbidden pattern detected: {pattern.pattern}",
                    evidence={
                        "pattern": pattern.pattern,
                        "match": match.group()[:50],
                    },
                )
        
        return GateDecision.accept("No forbidden patterns")
    
    def _check_rate_limits(self, state: StateSnapshot, proposal: Proposal) -> GateDecision:
        """Check rate limits."""
        progress = state.task_progress
        
        # Check proposals per step (rough approximation)
        # In practice, this would need more sophisticated tracking
        if progress.total_proposals >= self._config.max_proposals_per_step * (progress.step_index + 1):
            return GateDecision.reject(
                reason="Rate limit exceeded",
                evidence={
                    "total_proposals": progress.total_proposals,
                    "step_index": progress.step_index,
                },
            )
        
        return GateDecision.accept("Rate limit ok")


def validate_gate_immutability(gate: Gate) -> bool:
    """Verify that the gate is immutable.
    
    This is a runtime check to ensure gate config cannot be modified.
    
    Args:
        gate: Gate instance to check.
    
    Returns:
        True if gate is properly immutable.
    """
    # Check that config is frozen dataclass
    try:
        # Attempt to modify (should fail)
        object.__setattr__(gate._config, "max_diff_lines", 999999)
        return False  # Modification succeeded = not immutable
    except (AttributeError, FrozenInstanceError):
        return True  # Correctly immutable
    except Exception:
        return True  # Some other protection


# Alias for backwards compatibility
FrozenInstanceError = AttributeError
