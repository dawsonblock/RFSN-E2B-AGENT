"""Prompt Variants for SWE-bench.

Different prompting strategies optimized for code repair tasks.
The bandit selects which variant to use.

INVARIANTS:
1. Variants are registered by name
2. Variants don't affect kernel behavior
3. Variants are pure data (no side effects)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromptVariant:
    """A prompt variant configuration."""
    
    name: str
    description: str
    system_prompt: str
    user_prompt_template: str
    temperature: float = 0.2
    max_tokens: int = 4096
    metadata: dict[str, Any] | None = None


# SWE-bench optimized prompt variants
PROMPT_VARIANTS: dict[str, PromptVariant] = {
    "v_minimal_fix": PromptVariant(
        name="v_minimal_fix",
        description="Minimal, focused single-line fixes",
        system_prompt="""You are an expert software engineer. Your goal is to make the MINIMAL change to fix the bug.

RULES:
1. Make the SMALLEST possible change
2. Only modify what is strictly necessary
3. Do not refactor or improve unrelated code
4. Prefer single-line fixes when possible

Respond with a JSON proposal:
```json
{
    "intent": "modify_file",
    "target": "path/to/file.py",
    "patch": "--- a/file.py\n+++ b/file.py\n@@ ...",
    "justification": "Why this minimal change fixes the bug",
    "expected_effect": "Tests will pass"
}
```""",
        user_prompt_template="""## Bug Report
{problem_statement}

## Failing Test
{failing_test}

## Current File Content
```python
{file_content}
```

Make the minimal fix. Respond with a single JSON proposal.""",
        temperature=0.1,
    ),
    
    "v_diagnose_then_patch": PromptVariant(
        name="v_diagnose_then_patch",
        description="First diagnose root cause, then patch",
        system_prompt="""You are an expert software debugger. Follow this process:

1. DIAGNOSE: Identify the root cause of the bug
2. LOCATE: Find the exact line(s) that need to change
3. PATCH: Write the minimal patch

Think step-by-step before proposing a fix.

Respond with your analysis followed by a JSON proposal:
```json
{
    "intent": "modify_file",
    "target": "path/to/file.py",
    "patch": "...",
    "justification": "Root cause: X. Fix: Y",
    "expected_effect": "Tests will pass because Z"
}
```""",
        user_prompt_template="""## Bug Report
{problem_statement}

## Failing Test Output
{test_output}

## Relevant Code
```python
{file_content}
```

First diagnose the bug, then propose a fix.""",
        temperature=0.3,
    ),
    
    "v_test_first": PromptVariant(
        name="v_test_first",
        description="Understand test expectations first, then fix",
        system_prompt="""You are a test-driven developer. Before fixing code, understand what the test expects.

PROCESS:
1. Read the failing test carefully
2. Understand what behavior is expected
3. Find where the code diverges from expectations
4. Make the fix

Focus on making the test pass, not on what seems "right".""",
        user_prompt_template="""## The Failing Test
```python
{test_code}
```

## Test Output
```
{test_output}
```

## Code Under Test
```python
{file_content}
```

What does the test expect? What change makes it pass?""",
        temperature=0.2,
    ),
    
    "v_multi_hypothesis": PromptVariant(
        name="v_multi_hypothesis",
        description="Generate multiple hypotheses, pick best",
        system_prompt="""You are a systematic debugger. Generate multiple hypotheses for the bug.

PROCESS:
1. List 3 possible causes of the bug
2. For each, explain what change would fix it
3. Pick the most likely hypothesis
4. Propose that fix

Be thorough but decisive.""",
        user_prompt_template="""## Problem
{problem_statement}

## Evidence
{test_output}

## Code
```python
{file_content}
```

List 3 hypotheses, then pick the best and propose a fix.""",
        temperature=0.4,
    ),
    
    "v_repair_loop": PromptVariant(
        name="v_repair_loop",
        description="Iterative refinement based on feedback",
        system_prompt="""You are an iterative code repairer. If your previous attempt failed, learn from it.

RULES:
1. Pay attention to rejection feedback
2. Don't repeat rejected approaches
3. Try a different strategy if stuck
4. Be persistent but adaptive

If this is your first attempt, start with the simplest fix.""",
        user_prompt_template="""## Problem
{problem_statement}

## Previous Attempts
{rejection_history}

## Current State
- Tests: {test_status}
- Patches applied: {patches_applied}

## Code
```python
{file_content}
```

Based on previous attempts, try a new approach.""",
        temperature=0.3,
    ),
    
    "v_context_aware": PromptVariant(
        name="v_context_aware",
        description="Use retrieved context from similar bugs",
        system_prompt="""You are an experienced developer with access to past bug fixes.

Use the provided similar bugs and their solutions as guidance.
But adapt the solution to the current context - don't copy blindly.""",
        user_prompt_template="""## Current Bug
{problem_statement}

## Similar Past Bugs and Fixes
{similar_memories}

## Current Code
```python
{file_content}
```

Use the past examples as guidance to fix this bug.""",
        temperature=0.2,
    ),
    
    "v_chain_of_thought": PromptVariant(
        name="v_chain_of_thought",
        description="Explicit reasoning chain before action",
        system_prompt="""You are a methodical software engineer. Think out loud.

For every action, explain your reasoning:
1. What is the symptom?
2. What could cause this?
3. Where should I look?
4. What change would fix it?
5. Why is this the right fix?

Only after reasoning, propose your fix.""",
        user_prompt_template="""## Bug Report
{problem_statement}

## Test Output
{test_output}

## Code
```python
{file_content}
```

Think step by step, then propose a fix.""",
        temperature=0.2,
        max_tokens=6000,
    ),
}


def get_variant(name: str) -> PromptVariant:
    """Get a prompt variant by name.
    
    Args:
        name: Variant name.
    
    Returns:
        PromptVariant instance.
    
    Raises:
        KeyError: If variant not found.
    """
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown variant: {name}. Available: {list(PROMPT_VARIANTS.keys())}")
    return PROMPT_VARIANTS[name]


def list_variants() -> list[str]:
    """List all available variant names.
    
    Returns:
        List of variant names.
    """
    return list(PROMPT_VARIANTS.keys())


def register_variant(variant: PromptVariant) -> None:
    """Register a custom variant.
    
    Args:
        variant: Variant to register.
    """
    PROMPT_VARIANTS[variant.name] = variant


def format_prompt(
    variant: PromptVariant,
    **kwargs: Any,
) -> tuple[str, str]:
    """Format a prompt variant with context.
    
    Args:
        variant: The variant to use.
        **kwargs: Context variables for template.
    
    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    # Fill in missing variables with defaults
    defaults = {
        "problem_statement": "Fix the failing test.",
        "test_output": "",
        "test_code": "",
        "file_content": "",
        "failing_test": "",
        "rejection_history": "None",
        "test_status": "failing",
        "patches_applied": 0,
        "similar_memories": "None available",
    }
    
    context = {**defaults, **kwargs}
    
    user_prompt = variant.user_prompt_template.format(**context)
    
    return variant.system_prompt, user_prompt
