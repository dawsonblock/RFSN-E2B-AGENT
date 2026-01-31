"""SWE-bench specific repair strategy arms.

A strategy arm is NOT "which model" - it's "how we repair".
Each arm defines a distinct repair approach.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyArm:
    """
    A strategy arm defines a repair approach.
    
    This is what the bandit learns over - not models, not prompts,
    but repair strategies.
    """
    arm_id: str
    name: str
    patch_style: str
    planning_instructions: str
    max_steps: int = 10
    max_patches: int = 5


# SWE-bench specific strategy arms
SWE_BENCH_ARMS: list[StrategyArm] = [
    StrategyArm(
        arm_id="swe_minimal_diff",
        name="Minimal diff",
        patch_style="Keep changes small. Prefer local fixes. Avoid refactors.",
        planning_instructions=(
            "1) Identify failing location from traceback.\n"
            "2) Patch the smallest surface.\n"
            "3) Run tests.\n"
            "4) If still failing, adjust only the relevant function."
        ),
        max_steps=8,
        max_patches=3,
    ),
    StrategyArm(
        arm_id="swe_traceback_first",
        name="Traceback-first",
        patch_style="Drive everything from the first error/exception. Do not guess.",
        planning_instructions=(
            "1) Parse stderr/stdout for first error.\n"
            "2) Locate symbol/file/line.\n"
            "3) Patch to satisfy the failing assertion.\n"
            "4) Re-run tests."
        ),
        max_steps=10,
        max_patches=4,
    ),
    StrategyArm(
        arm_id="swe_contract_fix",
        name="API/contract fix",
        patch_style="Assume interface mismatch. Fix signature/return/edge cases.",
        planning_instructions=(
            "1) Find function/class referenced by failing test.\n"
            "2) Infer required contract from test.\n"
            "3) Patch implementation to match.\n"
            "4) Re-run tests."
        ),
        max_steps=10,
        max_patches=5,
    ),
    StrategyArm(
        arm_id="swe_regression_guard",
        name="Regression guard",
        patch_style="Fix bug + add tiny guard/validation. Avoid broad exception swallowing.",
        planning_instructions=(
            "1) Fix root cause.\n"
            "2) Add precise guard where needed.\n"
            "3) Re-run tests.\n"
            "4) Ensure no behavior drift outside failing path."
        ),
        max_steps=12,
        max_patches=4,
    ),
    StrategyArm(
        arm_id="swe_test_driven",
        name="Test-driven repair",
        patch_style="Understand the test first. What does it expect? Then fix to match.",
        planning_instructions=(
            "1) Read the failing test carefully.\n"
            "2) Identify expected vs actual behavior.\n"
            "3) Trace to the source of the discrepancy.\n"
            "4) Patch the implementation, not the test.\n"
            "5) Re-run to verify."
        ),
        max_steps=10,
        max_patches=4,
    ),
    StrategyArm(
        arm_id="swe_import_fix",
        name="Import/dependency fix",
        patch_style="Focus on import errors, missing modules, circular imports.",
        planning_instructions=(
            "1) Check for ImportError, ModuleNotFoundError in traceback.\n"
            "2) Verify import paths and module existence.\n"
            "3) Fix import statement or add missing dependency.\n"
            "4) Check for circular import issues.\n"
            "5) Re-run tests."
        ),
        max_steps=6,
        max_patches=3,
    ),
    StrategyArm(
        arm_id="swe_type_fix",
        name="Type/attribute fix",
        patch_style="Focus on TypeError, AttributeError, KeyError. Fix type mismatches.",
        planning_instructions=(
            "1) Parse TypeError/AttributeError from traceback.\n"
            "2) Identify the mismatched type or missing attribute.\n"
            "3) Trace back to where the wrong type originated.\n"
            "4) Fix the type conversion or add the attribute.\n"
            "5) Re-run tests."
        ),
        max_steps=8,
        max_patches=4,
    ),
]


def get_arm(arm_id: str) -> StrategyArm:
    """Get a strategy arm by ID."""
    for arm in SWE_BENCH_ARMS:
        if arm.arm_id == arm_id:
            return arm
    raise ValueError(f"Unknown arm_id: {arm_id}")


def list_arm_ids() -> list[str]:
    """List all available arm IDs."""
    return [arm.arm_id for arm in SWE_BENCH_ARMS]
