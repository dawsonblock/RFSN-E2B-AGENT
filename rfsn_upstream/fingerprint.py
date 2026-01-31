"""Failure Fingerprinting for Retrieval and Analysis.

Fingerprints capture structured information about failures
for similarity matching and pattern recognition.

INVARIANTS:
1. Fingerprints are deterministic (same failure → same fingerprint)
2. Fingerprints are immutable
3. Fingerprints support similarity comparison
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class Fingerprint:
    """Immutable fingerprint of a failure or rejection.
    
    Captures structured information for:
    - Similarity matching with past failures
    - Pattern recognition
    - Debugging analysis
    
    INVARIANT: Fingerprints are deterministic and immutable.
    """
    
    fingerprint_id: str
    failure_type: str  # "gate_rejection", "test_failure", "execution_error"
    category: str  # High-level category (e.g., "path_violation", "syntax_error")
    subcategory: str  # More specific (e.g., "forbidden_import", "missing_colon")
    patterns: tuple[str, ...]  # Extracted patterns for matching
    context: tuple[tuple[str, Any], ...]  # Additional context
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "fingerprint_id": self.fingerprint_id,
            "failure_type": self.failure_type,
            "category": self.category,
            "subcategory": self.subcategory,
            "patterns": list(self.patterns),
            "context": dict(self.context),
            "timestamp": self.timestamp,
        }
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), sort_keys=True)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        """Deserialize from dictionary."""
        context = data.get("context", {})
        if isinstance(context, dict):
            context = tuple(sorted(context.items()))
        
        return cls(
            fingerprint_id=data["fingerprint_id"],
            failure_type=data["failure_type"],
            category=data["category"],
            subcategory=data.get("subcategory", ""),
            patterns=tuple(data.get("patterns", [])),
            context=context,
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )
    
    def similarity(self, other: Fingerprint) -> float:
        """Compute similarity to another fingerprint.
        
        Args:
            other: Fingerprint to compare with.
        
        Returns:
            Similarity score 0.0-1.0.
        """
        score = 0.0
        max_score = 0.0
        
        # Category match (high weight)
        max_score += 3.0
        if self.category == other.category:
            score += 2.0
            if self.subcategory == other.subcategory:
                score += 1.0
        
        # Failure type match
        max_score += 1.0
        if self.failure_type == other.failure_type:
            score += 1.0
        
        # Pattern overlap
        max_score += 2.0
        if self.patterns and other.patterns:
            self_set = set(self.patterns)
            other_set = set(other.patterns)
            if self_set or other_set:
                overlap = len(self_set & other_set)
                union = len(self_set | other_set)
                score += 2.0 * (overlap / union) if union > 0 else 0.0
        
        return score / max_score if max_score > 0 else 0.0


# Common error patterns for extraction
ERROR_PATTERNS = {
    # Python errors
    r"SyntaxError: (.+)": ("syntax_error", "python"),
    r"IndentationError: (.+)": ("indentation_error", "python"),
    r"NameError: name '(\w+)' is not defined": ("name_error", "undefined"),
    r"TypeError: (.+)": ("type_error", "python"),
    r"AttributeError: '(\w+)' object has no attribute '(\w+)'": ("attribute_error", "missing"),
    r"ImportError: (.+)": ("import_error", "python"),
    r"ModuleNotFoundError: (.+)": ("import_error", "missing_module"),
    r"ValueError: (.+)": ("value_error", "python"),
    r"KeyError: (.+)": ("key_error", "missing_key"),
    r"IndexError: (.+)": ("index_error", "out_of_bounds"),
    r"AssertionError(.*)": ("assertion_error", "test"),
    
    # Gate rejections
    r"Intent '.+' not in allowlist": ("intent_violation", "not_allowed"),
    r"Path '.+' does not match allowed patterns": ("path_violation", "not_allowed"),
    r"Path '.+' matches forbidden pattern": ("path_violation", "forbidden"),
    r"Forbidden pattern detected": ("pattern_violation", "security"),
    r"Max steps exceeded": ("limit_exceeded", "steps"),
    r"Max patches exceeded": ("limit_exceeded", "patches"),
    r"Diff too large": ("limit_exceeded", "diff_size"),
    r"Path traversal detected": ("path_violation", "traversal"),
    
    # Test failures
    r"FAILED (.+)::(.+)": ("test_failure", "assertion"),
    r"ERROR (.+)::(.+)": ("test_failure", "error"),
    r"pytest: error: (.+)": ("test_failure", "collection"),
}


def extract_patterns(text: str) -> list[str]:
    """Extract meaningful patterns from error text.
    
    Args:
        text: Error message or traceback.
    
    Returns:
        List of extracted patterns.
    """
    patterns = []
    
    # Extract file paths
    file_matches = re.findall(r'["\']?([^"\'<>|\s]+\.py)["\']?', text)
    for match in file_matches:
        # Normalize path
        normalized = match.split("/")[-1]  # Just filename
        if normalized not in patterns:
            patterns.append(f"file:{normalized}")
    
    # Extract function/class names from tracebacks
    func_matches = re.findall(r'in (\w+)', text)
    for match in func_matches:
        if len(match) > 2 and match not in ("in", "is", "or", "and"):
            patterns.append(f"func:{match}")
    
    # Extract line numbers
    line_matches = re.findall(r'line (\d+)', text)
    if line_matches:
        patterns.append(f"line:{line_matches[0]}")
    
    # Extract variable names from NameError
    name_matches = re.findall(r"name '(\w+)' is not defined", text)
    for match in name_matches:
        patterns.append(f"name:{match}")
    
    # Extract attribute names from AttributeError
    attr_matches = re.findall(r"has no attribute '(\w+)'", text)
    for match in attr_matches:
        patterns.append(f"attr:{match}")
    
    return patterns[:10]  # Limit patterns


def classify_error(text: str) -> tuple[str, str]:
    """Classify error text into category and subcategory.
    
    Args:
        text: Error message or reason.
    
    Returns:
        Tuple of (category, subcategory).
    """
    for pattern, (category, subcategory) in ERROR_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category, subcategory
    
    # Default classification
    if "error" in text.lower():
        return "unknown_error", "unclassified"
    elif "reject" in text.lower() or "denied" in text.lower():
        return "rejection", "unclassified"
    else:
        return "unknown", "unclassified"


def compute_fingerprint_id(
    failure_type: str,
    category: str,
    subcategory: str,
    patterns: list[str],
) -> str:
    """Compute deterministic fingerprint ID.
    
    Args:
        failure_type: Type of failure.
        category: Category.
        subcategory: Subcategory.
        patterns: Extracted patterns.
    
    Returns:
        Deterministic fingerprint ID.
    """
    content = "|".join([
        failure_type,
        category,
        subcategory,
        ",".join(sorted(patterns)),
    ])
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def compute_fingerprint(
    failure_type: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> Fingerprint:
    """Compute a fingerprint from a failure.
    
    Args:
        failure_type: Type of failure (gate_rejection, test_failure, execution_error).
        message: Error message or failure reason.
        context: Optional additional context.
    
    Returns:
        Fingerprint instance.
    """
    # Classify error
    category, subcategory = classify_error(message)
    
    # Extract patterns
    patterns = extract_patterns(message)
    
    # Compute ID
    fingerprint_id = compute_fingerprint_id(failure_type, category, subcategory, patterns)
    
    # Build context tuple
    context_items = list((context or {}).items())
    context_items.append(("message_preview", message[:100]))
    
    return Fingerprint(
        fingerprint_id=fingerprint_id,
        failure_type=failure_type,
        category=category,
        subcategory=subcategory,
        patterns=tuple(patterns),
        context=tuple(sorted(context_items)),
    )


def fingerprint_from_rejection(
    reason: str,
    evidence: dict[str, Any],
    proposal_intent: str = "",
    proposal_target: str = "",
) -> Fingerprint:
    """Create fingerprint from a gate rejection.
    
    Args:
        reason: Rejection reason from gate.
        evidence: Evidence from gate decision.
        proposal_intent: Intent of rejected proposal.
        proposal_target: Target of rejected proposal.
    
    Returns:
        Fingerprint for the rejection.
    """
    context = {
        "intent": proposal_intent,
        "target": proposal_target,
        **evidence,
    }
    
    return compute_fingerprint(
        failure_type="gate_rejection",
        message=reason,
        context=context,
    )


def fingerprint_from_test_failure(
    stdout: str,
    stderr: str,
    test_name: str = "",
) -> Fingerprint:
    """Create fingerprint from a test failure.
    
    Args:
        stdout: Test stdout.
        stderr: Test stderr.
        test_name: Name of failing test.
    
    Returns:
        Fingerprint for the test failure.
    """
    # Combine outputs for analysis
    combined = f"{stdout}\n{stderr}"
    
    context = {
        "test_name": test_name,
    }
    
    return compute_fingerprint(
        failure_type="test_failure",
        message=combined,
        context=context,
    )


def fingerprint_from_execution_error(
    error: str,
    intent: str = "",
    target: str = "",
) -> Fingerprint:
    """Create fingerprint from an execution error.
    
    Args:
        error: Error message.
        intent: Intent that failed.
        target: Target that failed.
    
    Returns:
        Fingerprint for the execution error.
    """
    context = {
        "intent": intent,
        "target": target,
    }
    
    return compute_fingerprint(
        failure_type="execution_error",
        message=error,
        context=context,
    )
