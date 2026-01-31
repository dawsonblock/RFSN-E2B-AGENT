#!/usr/bin/env python3
"""RFSN Agent Runner - Single entrypoint for running tasks.

Usage:
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug"
    
    # With specific planner
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug" \
        --provider deepseek --model deepseek-chat
    
    # With upstream learning
    python run_agent.py --task-id TASK_123 --workspace ./repo --task "Fix the bug" \
        --bandit-db bandit.db --arm-id v_minimal_fix
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from rfsn_kernel import (
    Gate,
    GateConfig,
    Kernel,
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
    # Basic usage
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test"
    
    # With specific LLM
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test" \\
        --provider openai --model gpt-4-turbo
    
    # With bandit integration
    python run_agent.py --task-id BUG_001 --workspace ./my_repo --task "Fix the test" \\
        --bandit-db ./bandit.db --arm-id v_minimal_fix
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
        default=20,
        help="Maximum steps to run (default: 20)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=10,
        help="Maximum patches to apply (default: 10)",
    )
    parser.add_argument(
        "--test-command",
        default="pytest -q",
        help="Test command (default: pytest -q)",
    )
    
    # Upstream learning
    parser.add_argument(
        "--bandit-db",
        type=Path,
        default=None,
        help="Path to bandit SQLite database",
    )
    parser.add_argument(
        "--arm-id",
        default=None,
        help="Specific prompt variant arm to use",
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
    
    # Configure LLM planner
    model = args.model or get_model_default(args.provider)
    
    # Handle prompt variant from bandit
    prompt_variant = args.arm_id
    if args.bandit_db and not args.arm_id:
        try:
            from rfsn_upstream import ThompsonBandit
            bandit = ThompsonBandit(args.bandit_db)
            prompt_variant = bandit.select_arm()
            logger.info(f"Bandit selected arm: {prompt_variant}")
        except Exception as e:
            logger.warning(f"Bandit selection failed: {e}, using default")
            prompt_variant = "v_minimal_fix"
    
    planner_config = LLMPlannerConfig(
        provider=args.provider,
        model=model,
        temperature=args.temperature,
        prompt_variant=prompt_variant or "v_minimal_fix",
    )
    planner = LLMPlanner(planner_config)
    
    # Configure kernel
    safety_envelope = SafetyEnvelope(
        max_steps=args.max_steps,
        max_patches=args.max_patches,
    )
    
    controller_config = ControllerConfig(
        default_test_command=args.test_command,
    )
    
    kernel_config = KernelConfig(
        controller_config=controller_config,
        safety_envelope=safety_envelope,
        max_steps=args.max_steps,
        verbose=args.verbose,
    )
    
    # Create ledger
    ledger_dir = args.ledger_dir or args.workspace / ".rfsn_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger_path = ledger_dir / f"{args.task_id}_{timestamp}.jsonl"
    ledger = Ledger(ledger_path)
    
    # Run kernel
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
        
        # Update bandit if used
        if args.bandit_db and prompt_variant:
            try:
                from rfsn_upstream import ThompsonBandit
                bandit = ThompsonBandit(args.bandit_db)
                bandit.update(prompt_variant, success=result.success, task_id=args.task_id)
                logger.info(f"Updated bandit arm {prompt_variant}: success={result.success}")
            except Exception as e:
                logger.warning(f"Bandit update failed: {e}")
        
        # Output result
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
            "arm_id": prompt_variant,
            "provider": args.provider,
            "model": model,
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
