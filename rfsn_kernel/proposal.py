"""Proposal Schema - Structured proposals from planners.

Proposals are the ONLY way planners can communicate intent.
They are DATA, never executable code.

INVARIANTS:
- Proposals are validated against a strict schema
- Proposals cannot contain executable code
- Proposals must specify expected effects
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class ProposalIntent(Enum):
    """Allowed proposal intents (explicit allowlist)."""
    
    # File operations
    MODIFY_FILE = "modify_file"
    CREATE_FILE = "create_file"
    DELETE_FILE = "delete_file"
    
    # Test operations
    RUN_TEST = "run_test"
    RUN_FOCUSED_TEST = "run_focused_test"
    
    # Read-only operations
    READ_FILE = "read_file"
    SEARCH_REPO = "search_repo"
    LIST_DIRECTORY = "list_directory"
    
    # Control flow
    CHECKPOINT = "checkpoint"
    REPLAN = "replan"


# Shell injection patterns to detect in proposals
SHELL_INJECTION_PATTERNS = [
    r"\$\(",      # Command substitution
    r"`",         # Backtick substitution
    r"&&",        # Command chaining
    r"\|\|",      # Or chaining
    r";",         # Command separator
    r"\|",        # Pipe (contextual)
    r">",         # Redirect
    r"<",         # Redirect
    r"rm\s+-rf",  # Dangerous remove
    r"sudo",      # Privilege escalation
    r"eval",      # Eval
    r"exec",      # Exec
]


@dataclass(frozen=True)
class Proposal:
    """Immutable structured proposal from a planner.
    
    INVARIANTS:
    - proposal_id is unique
    - intent must be in ProposalIntent enum
    - patch cannot contain shell injection patterns
    - expected_effect must be specified
    """
    
    proposal_id: str
    intent: ProposalIntent
    target: str  # File path or test pattern
    justification: str
    expected_effect: str
    patch: str = ""  # Unified diff for file modifications
    test_command: str = ""  # For run_test intents
    search_query: str = ""  # For search intents
    confidence: float = 0.5  # Planner's confidence (0-1)
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def __post_init__(self) -> None:
        """Validate proposal on creation."""
        # Validate intent is a ProposalIntent enum
        if not isinstance(self.intent, ProposalIntent):
            raise ValueError(f"Invalid intent type: {type(self.intent)}")
        
        # Validate confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got: {self.confidence}")
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "intent": self.intent.value,
            "target": self.target,
            "justification": self.justification,
            "expected_effect": self.expected_effect,
            "patch": self.patch,
            "test_command": self.test_command,
            "search_query": self.search_query,
            "confidence": self.confidence,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Proposal:
        """Deserialize from dictionary."""
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = tuple(sorted(metadata.items()))
        
        return cls(
            proposal_id=data["proposal_id"],
            intent=ProposalIntent(data["intent"]),
            target=data["target"],
            justification=data["justification"],
            expected_effect=data["expected_effect"],
            patch=data.get("patch", ""),
            test_command=data.get("test_command", ""),
            search_query=data.get("search_query", ""),
            confidence=data.get("confidence", 0.5),
            metadata=metadata,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> Proposal:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def content_hash(self) -> str:
        """Compute hash of proposal content for deduplication."""
        content = f"{self.intent.value}|{self.target}|{self.patch}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ProposalValidationError(Exception):
    """Raised when a proposal fails validation."""
    
    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(f"[ProposalValidation] {message}" + (f" (field: {field})" if field else ""))


def validate_proposal_schema(data: dict[str, Any]) -> list[str]:
    """Validate proposal against schema, return list of errors.
    
    This is a pure validation function - no side effects.
    
    Args:
        data: Proposal dictionary to validate.
    
    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []
    
    # Required fields
    required = ["proposal_id", "intent", "target", "justification", "expected_effect"]
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return errors  # Can't continue without required fields
    
    # Validate intent
    try:
        ProposalIntent(data["intent"])
    except ValueError:
        errors.append(f"Invalid intent: {data['intent']}. Allowed: {[i.value for i in ProposalIntent]}")
    
    # Validate target is not empty
    if not data.get("target", "").strip():
        errors.append("Target cannot be empty")
    
    # Validate justification
    if not data.get("justification", "").strip():
        errors.append("Justification cannot be empty")
    
    # Validate expected_effect
    if not data.get("expected_effect", "").strip():
        errors.append("Expected effect cannot be empty")
    
    # Validate patch does not contain shell injection
    patch = data.get("patch", "")
    for pattern in SHELL_INJECTION_PATTERNS:
        if re.search(pattern, patch):
            errors.append(f"Patch contains forbidden pattern: {pattern}")
    
    # Validate confidence range
    confidence = data.get("confidence", 0.5)
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        errors.append(f"Confidence must be 0-1, got: {confidence}")
    
    return errors


def create_proposal(
    intent: ProposalIntent | str,
    target: str,
    justification: str,
    expected_effect: str,
    patch: str = "",
    test_command: str = "",
    search_query: str = "",
    confidence: float = 0.5,
    metadata: dict[str, Any] | None = None,
) -> Proposal:
    """Create a validated proposal.
    
    Args:
        intent: What the proposal intends to do.
        target: File path or test pattern.
        justification: Why this proposal is being made.
        expected_effect: What effect this should have.
        patch: Unified diff for file modifications.
        test_command: For run_test intents.
        search_query: For search intents.
        confidence: Planner's confidence (0-1).
        metadata: Additional metadata.
    
    Returns:
        Validated Proposal.
    
    Raises:
        ProposalValidationError: If validation fails.
    """
    # Convert string intent to enum
    if isinstance(intent, str):
        try:
            intent = ProposalIntent(intent)
        except ValueError as e:
            raise ProposalValidationError(f"Invalid intent: {intent}", "intent") from e
    
    # Generate unique ID
    proposal_id = str(uuid.uuid4())[:8]
    
    # Validate patch for shell injection
    for pattern in SHELL_INJECTION_PATTERNS:
        if re.search(pattern, patch):
            raise ProposalValidationError(f"Patch contains forbidden pattern: {pattern}", "patch")
    
    # Create metadata tuple
    metadata_tuple = tuple(sorted((metadata or {}).items()))
    
    return Proposal(
        proposal_id=proposal_id,
        intent=intent,
        target=target,
        justification=justification,
        expected_effect=expected_effect,
        patch=patch,
        test_command=test_command,
        search_query=search_query,
        confidence=confidence,
        metadata=metadata_tuple,
    )


# Example proposal schemas for documentation
PROPOSAL_SCHEMA_EXAMPLES = {
    "modify_file": {
        "proposal_id": "abc123",
        "intent": "modify_file",
        "target": "src/utils.py",
        "patch": "--- a/src/utils.py\n+++ b/src/utils.py\n@@ -10,3 +10,4 @@\n def foo():\n-    return None\n+    return 42",
        "justification": "Fix return value to match expected behavior",
        "expected_effect": "test_utils.py::test_foo will pass",
    },
    "run_test": {
        "proposal_id": "def456",
        "intent": "run_test",
        "target": "tests/test_utils.py::test_foo",
        "test_command": "pytest tests/test_utils.py::test_foo -v",
        "justification": "Verify the fix works",
        "expected_effect": "Test passes",
    },
    "create_file": {
        "proposal_id": "ghi789",
        "intent": "create_file",
        "target": "src/new_module.py",
        "patch": "def new_function():\n    return True",
        "justification": "Add new module for feature X",
        "expected_effect": "New module available for import",
    },
}
