"""Feature extraction for upstream learning.

Extracts features from test results, tracebacks, and diffs
to enable fingerprint-based learning.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rfsn_kernel.state import TestResult


# Patterns to extract first error from test output
_FIRST_ERROR_RE = re.compile(
    r"^(E\s+|FAILED\s+|ERROR\s+|Traceback \(most recent call last\):)",
    re.MULTILINE
)

_EXCEPTION_RE = re.compile(
    r"((?:(?:\w+Error|\w+Exception|\w+Warning)): .+?)(?:\n|$)"
)

_FILE_LINE_RE = re.compile(
    r'File "([^"]+)", line (\d+)'
)


# Common error patterns for extraction & classification
_ERROR_PATTERNS = {
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


def repo_id_from_path(repo_path: Path | str) -> str:
    """
    Deterministic repo identifier without requiring git metadata.
    Uses path hash for stability.
    """
    p = str(Path(repo_path).resolve()).encode("utf-8")
    return hashlib.sha256(p).hexdigest()[:16]


def extract_first_error(blob: str) -> str:
    """Extract the first error message from test output."""
    m = _FIRST_ERROR_RE.search(blob)
    if m:
        return blob[m.start(): m.start() + 500].strip()
    return blob[:400].strip()


def extract_exception_type(blob: str) -> str:
    """Extract the exception type from error output."""
    m = _EXCEPTION_RE.search(blob)
    if m:
        exc_text = m.group(1)
        # Extract just the exception name
        if ":" in exc_text:
            return exc_text.split(":")[0].strip()
        return exc_text.strip()
    return "unknown"


def classify_error(text: str) -> tuple[str, str]:
    """Classify error text into category and subcategory."""
    for pattern, (category, subcategory) in _ERROR_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return category, subcategory
    
    # Fallback to simple exception extraction
    exc = extract_exception_type(text)
    if exc != "unknown":
        return "exception", exc.lower()
        
    return "unknown", "unclassified"


def extract_failing_files(blob: str) -> list[tuple[str, int]]:
    """Extract (file, line) pairs from traceback."""
    matches = _FILE_LINE_RE.findall(blob)
    return [(f, int(ln)) for f, ln in matches]


def summarize_test_failure(test: "TestResult | None") -> dict[str, Any]:
    """
    Summarize a test result for metrics storage.
    """
    if test is None:
        return {"has_test": False}

    blob = (test.stdout or "") + "\n" + (test.stderr or "")
    first_error = extract_first_error(blob)
    exc_type = extract_exception_type(blob)
    failing_files = extract_failing_files(blob)

    return {
        "has_test": True,
        "status": getattr(test.status, "value", str(test.status)) if hasattr(test, "status") else "unknown",
        "passed": getattr(test, "passed", 0),
        "failed": getattr(test, "failed", 0),
        "errors": getattr(test, "errors", 0),
        "skipped": getattr(test, "skipped", 0),
        "duration_seconds": getattr(test, "duration_seconds", 0.0),
        "failing_tests": list(getattr(test, "failing_tests", [])),
        "first_error_excerpt": first_error[:400],
        "exception_type": exc_type,
        "failing_files": failing_files[:5],  # Limit
    }


def fingerprints_from_test(test: "TestResult | None") -> list[dict[str, Any]]:
    """
    Generate fingerprints from test results.
    
    Fingerprints are used to identify similar failures across runs,
    enabling "don't try this again" logic.
    """
    if test is None:
        return []

    summary = summarize_test_failure(test)
    
    fingerprints = []
    
    # Primary fingerprint: exception type + first error
    exc_type = summary.get("exception_type", "unknown")
    first_error = summary.get("first_error_excerpt", "")
    
    # Use advanced classification
    category, subcategory = classify_error(first_error or exc_type)
    
    # Create deterministic fingerprint ID
    key = f"{category}|{subcategory}|{first_error[:200]}".encode("utf-8")
    fp_id = hashlib.sha256(key).hexdigest()[:16]
    
    fingerprints.append({
        "fingerprint_id": fp_id,
        "failure_type": "test_failure",
        "category": category,
        "subcategory": subcategory,
        "patterns": [first_error[:120]],
        "context": {
            "failing_tests": summary.get("failing_tests", [])[:5],
            "failing_files": summary.get("failing_files", [])[:3],
            "exception_type": exc_type,
        },
    })
    
    # Secondary fingerprint: per failing test
    for test_name in summary.get("failing_tests", [])[:5]:
        test_key = f"test:{test_name}|{exc_type}".encode("utf-8")
        test_fp_id = hashlib.sha256(test_key).hexdigest()[:16]
        fingerprints.append({
            "fingerprint_id": test_fp_id,
            "failure_type": "specific_test",
            "category": "test",
            "subcategory": test_name.split("::")[-1] if "::" in test_name else test_name,
            "patterns": [test_name],
            "context": {"exception_type": exc_type},
        })
    
    return fingerprints


def fingerprints_from_gate_rejection(
    reason: str,
    evidence: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Generate fingerprints from gate rejections.
    
    This enables learning from safety violations too.
    """
    # Use advanced classification
    category, subcategory = classify_error(reason)
    
    key = f"gate:{category}|{subcategory}|{reason[:200]}".encode("utf-8")
    fp_id = hashlib.sha256(key).hexdigest()[:16]
    
    return [{
        "fingerprint_id": fp_id,
        "failure_type": "gate_rejection",
        "category": category,
        "subcategory": subcategory,
        "patterns": [reason[:200]],
        "context": evidence,
    }]


def compute_diff_fingerprint(diff: str) -> str:
    """
    Compute a fingerprint for a diff.
    
    Used to detect "trying the same patch again".
    """
    # Normalize whitespace
    normalized = re.sub(r'\s+', ' ', diff.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
