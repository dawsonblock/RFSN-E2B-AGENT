#!/usr/bin/env python3
"""RFSN Agent Runner - Single entrypoint for running tasks.

Now with REAL upstream learning:
- Strategy arm selection via Thompson Sampling
- Outcome recording to persistent database
- Self-critique rubric injection
- Fingerprint extraction for similar failure matching

Usage:
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug"
    
    # With specific strategy arm
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug" \
        --arm-id swe_minimal_diff
    
    # With upstream learning (automatically selects arm)
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug" \
        --artifacts-dir ./artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rfsn_kernel import (
    KernelConfig,
    Ledger,
    SafetyEnvelope,
    run_kernel,
)
from rfsn_kernel.controller import ControllerConfig
from rfsn_kernel.llm_planner import LLMPlanner, LLMPlannerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_agent")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RFSN Agent Runner - Run autonomous code repair tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (selects strategy via Thompson Sampling)
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test"
    
    # With specific LLM
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test" \\
        --provider openai --model gpt-4-turbo
    
    # With specific strategy arm
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test" \\
        --arm-id swe_traceback_first
    
    # Available strategy arms:
    #   swe_minimal_diff    - Keep patches small
    #   swe_traceback_first - Follow the stack trace
    #   swe_contract_fix    - Fix API/interface issues
    #   swe_regression_guard- Add tests before fixing
    #   swe_test_driven     - TDD approach
    #   swe_import_fix      - Fix imports/dependencies
    #   swe_type_fix        - Fix type errors
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--task-id",
        required=True,
        help="Unique identifier for this task",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Path to the workspace/repository",
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task description (what to fix)",
    )
    
    # LLM configuration
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "openai", "anthropic"],
        help="LLM provider (default: deepseek)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: provider default)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)",
    )
    
    # Kernel configuration
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps (default: from strategy arm)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Maximum patches (default: from strategy arm)",
    )
    parser.add_argument(
        "--test-command",
        default="pytest -q",
        help="Test command (default: pytest -q)",
    )
    
    # Upstream learning
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("./artifacts"),
        help="Directory for learner state (default: ./artifacts)",
    )
    parser.add_argument(
        "--arm-id",
        default=None,
        help="Specific strategy arm to use (optional, selects via Thompson Sampling if not provided)",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write outcome JSON (default: stdout)",
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=None,
        help="Directory for ledger files (default: workspace/.rfsn_ledger)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def get_model_default(provider: str) -> str:
    """Get default model for provider."""
    defaults = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4-turbo",
        "anthropic": "claude-3-5-sonnet-20241022",
    }
    return defaults.get(provider, "deepseek-chat")


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting RFSN Agent for task: {args.task_id}")
    logger.info(f"Workspace: {args.workspace}")
    logger.info(f"Provider: {args.provider}")
    
    # Validate workspace
    if not args.workspace.exists():
        logger.error(f"Workspace not found: {args.workspace}")
        return 1
    
    # --- Initialize upstream learner ---
    from rfsn_upstream import (
        UpstreamLearner,
        get_arm,
        list_arm_ids,
        summarize_test_failure,
        fingerprints_from_test,
        SELF_CRITIQUE_RUBRIC,
    )
    
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    bandit_db = args.artifacts_dir / "bandit.db"
    outcomes_db = args.artifacts_dir / "outcomes.db"
    
    learner = UpstreamLearner(bandit_db=bandit_db, outcomes_db=outcomes_db)
    
    # --- Select or honor strategy arm ---
    decision = None
    if args.arm_id:
        # User provided arm - validate and use it
        available = list_arm_ids()
        if args.arm_id not in available:
            logger.error(f"Unknown arm: {args.arm_id}")
            logger.error(f"Available: {', '.join(available)}")
            return 1
        arm = get_arm(args.arm_id)
        logger.info(f"Using specified arm: {args.arm_id} ({arm.name})")
    else:
        # Let learner select via Thompson Sampling
        decision = learner.select(repo_path=args.workspace)
        args.arm_id = decision.arm_id
        arm = decision.arm
        logger.info(f"Learner selected arm: {args.arm_id} ({arm.name})")
    
    # --- Build system prompt with strategy + self-critique ---
    base_system_prompt = f"""You are an expert software engineer tasked with fixing bugs in code repositories.

## Current Task
Task ID: {args.task_id}
Repository: {args.workspace.name}

## Problem Statement
{args.task}

## Instructions
1. Analyze the failing test or error carefully
2. Identify the root cause
3. Propose a minimal, targeted fix
4. Ensure the fix doesn't break other functionality
"""
    
    # Add learner addendum (strategy + self-critique rubric)
    if decision:
        system_addendum = learner.build_system_addendum(decision)
    else:
        # Build addendum manually for user-provided arm
        system_addendum = f"""
---
## SWE-bench Repair Strategy
**Active Arm**: `{arm.arm_id}` ({arm.name})
**Patch Style**: {arm.patch_style}

### Planning Steps
{arm.planning_instructions}

{SELF_CRITIQUE_RUBRIC}
"""
    
    full_system_prompt = base_system_prompt + "\n" + system_addendum
    
    # --- Configure LLM planner ---
    model = args.model or get_model_default(args.provider)
    
    planner_config = LLMPlannerConfig(
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        max_tokens=4096,
        system_prompt=full_system_prompt,
    )
    planner = LLMPlanner(planner_config)
    
    # --- Configure kernel with arm-specific limits ---
    max_steps = args.max_steps or arm.max_steps
    max_patches = args.max_patches or arm.max_patches
    
    safety_envelope = SafetyEnvelope(
        max_steps=max_steps,
        max_patches=max_patches,
    )
    
    controller_config = ControllerConfig(
        default_test_command=args.test_command,
    )
    
    kernel_config = KernelConfig(
        controller_config=controller_config,
        safety_envelope=safety_envelope,
        max_steps=max_steps,
        verbose=args.verbose,
    )
    
    # Create ledger
    ledger_dir = args.ledger_dir or args.workspace / ".rfsn_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger_path = ledger_dir / f"{args.task_id}_{timestamp}.jsonl"
    ledger = Ledger(ledger_path)
    
    # --- Run kernel ---
    logger.info(f"Running kernel: max_steps={max_steps}, max_patches={max_patches}")
    
    try:
        result = run_kernel(
            task_id=args.task_id,
            task=args.task,
            workspace_path=args.workspace,
            planner=planner,
            config=kernel_config,
            ledger=ledger,
        )
        
        logger.info(f"Kernel completed: success={result.success}")
        logger.info(f"Reason: {result.completion_reason}")
        logger.info(f"Steps: {result.total_steps}, "
                   f"Accepted: {result.accepted_proposals}, "
                   f"Rejected: {result.rejected_proposals}")
        
        # --- Extract fingerprints from final test result ---
        final_test = getattr(result.final_state, "last_test_result", None)
        test_summary = summarize_test_failure(final_test)
        fingerprints = fingerprints_from_test(final_test)
        
        # --- Build metrics ---
        metrics: dict[str, Any] = {
            "completion_reason": result.completion_reason,
            "total_steps": result.total_steps,
            "accepted_proposals": result.accepted_proposals,
            "rejected_proposals": result.rejected_proposals,
            "test_status": result.test_status,
            "duration_seconds": result.duration_seconds,
            "test_summary": test_summary,
        }
        
        # --- Update upstream learner ---
        logger.info(f"Recording outcome: success={result.success}, fingerprints={len(fingerprints)}")
        
        learner.update(
            instance_id=args.task_id,
            repo_path=args.workspace,
            arm_id=args.arm_id,
            success=result.success,
            metrics=metrics,
            fingerprints=fingerprints,
        )
        
        # --- Output result ---
        outcome = {
            "task_id": args.task_id,
            "success": result.success,
            "completion_reason": result.completion_reason,
            "total_steps": result.total_steps,
            "accepted_proposals": result.accepted_proposals,
            "rejected_proposals": result.rejected_proposals,
            "test_status": result.test_status,
            "duration_seconds": result.duration_seconds,
            "ledger_path": str(ledger_path),
            "arm_id": args.arm_id,
            "arm_name": arm.name,
            "provider": args.provider,
            "model": model,
            "fingerprint_count": len(fingerprints),
            "artifacts_dir": str(args.artifacts_dir),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        outcome_json = json.dumps(outcome, indent=2)
        
        if args.output:
            args.output.write_text(outcome_json)
            logger.info(f"Wrote outcome to {args.output}")
        else:
            print(outcome_json)
        
        return 0 if result.success else 1
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Kernel failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
