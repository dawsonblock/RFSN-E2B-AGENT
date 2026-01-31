#!/usr/bin/env python3
"""SWE-bench Episode Runner for CI/CD.

Runs a single SWE-bench episode with REAL upstream learning:
- Strategy arm selection via Thompson Sampling
- Outcome recording to persistent database
- Fingerprint extraction for similar failure matching
- Self-critique rubric injection into prompts

Usage:
    python scripts/run_swebench_episode.py \
        --instance-id django__django-12345 \
        --repo /path/to/repo \
        --arm-id swe_minimal_diff \
        --output outcome.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("swebench_episode")

# Default SWE-bench settings
DEFAULT_TIMEOUT = 1800  # 30 minutes
DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_ARTIFACTS_DIR = "./artifacts"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single SWE-bench episode with upstream learning",
    )
    
    parser.add_argument(
        "--instance-id",
        required=True,
        help="SWE-bench instance ID (e.g., django__django-12345)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the repository",
    )
    parser.add_argument(
        "--problem-statement",
        type=Path,
        help="Path to problem statement file (optional)",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Test command to run (default: auto-detect)",
    )
    
    # Arm selection (optional - if not provided, learner selects)
    parser.add_argument(
        "--arm-id",
        default=None,
        help="Strategy arm ID (if not provided, Thompson Sampling selects)",
    )
    
    # LLM settings
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        help=f"LLM provider (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name",
    )
    
    # Execution settings
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--use-docker",
        action="store_true",
        help="Run in Docker container",
    )
    parser.add_argument(
        "--docker-image",
        default="ghcr.io/swe-bench/swe-bench:latest",
        help="Docker image for execution",
    )
    
    # Artifacts (learner state)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(DEFAULT_ARTIFACTS_DIR),
        help=f"Directory for learner state (default: {DEFAULT_ARTIFACTS_DIR})",
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outcome.json"),
        help="Path to write outcome JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    
    return parser.parse_args()


def get_test_command(repo_path: Path, instance_id: str) -> str:
    """Auto-detect test command based on repository structure."""
    if (repo_path / "manage.py").exists():
        return "python -m pytest --tb=short -q"
    elif (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists():
        return "python -m pytest --tb=short -q"
    else:
        return "python -m pytest -q"


def run_native(
    args: argparse.Namespace,
    problem_statement: str,
) -> dict:
    """Run episode natively with REAL upstream learning."""
    from rfsn_kernel import run_kernel, KernelConfig, SafetyEnvelope
    from rfsn_kernel.controller import ControllerConfig
    from rfsn_kernel.llm_planner import LLMPlanner, LLMPlannerConfig
    from rfsn_kernel.ledger import Ledger
    from rfsn_upstream import (
        UpstreamLearner,
        get_arm,
        list_arm_ids,
        summarize_test_failure,
        fingerprints_from_test,
        repo_id_from_path,
    )
    
    # --- Initialize upstream learner ---
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    bandit_db = args.artifacts_dir / "bandit.db"
    outcomes_db = args.artifacts_dir / "outcomes.db"
    
    learner = UpstreamLearner(bandit_db=bandit_db, outcomes_db=outcomes_db)
    
    # --- Select or honor arm ---
    decision = None
    if args.arm_id:
        # CI provided arm - honor it
        arm = get_arm(args.arm_id)
        logger.info(f"Using provided arm: {args.arm_id}")
    else:
        # Let learner select via Thompson Sampling
        decision = learner.select(repo_path=args.repo)
        args.arm_id = decision.arm_id
        arm = decision.arm
        logger.info(f"Learner selected arm: {args.arm_id} ({arm.name})")
    
    # --- Build system prompt with strategy + self-critique ---
    base_system_prompt = f"""You are an expert software engineer tasked with fixing bugs in code repositories.

## Current Task
Instance: {args.instance_id}
Repository: {args.repo.name}

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
        # Build addendum manually for CI-provided arm
        from rfsn_upstream import SELF_CRITIQUE_RUBRIC
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
    
    # --- Configure planner ---
    model = args.model or (
        "deepseek-chat" if args.provider == "deepseek" else
        "gpt-4-turbo" if args.provider == "openai" else
        "claude-3-5-sonnet-20241022"
    )
    
    planner_config = LLMPlannerConfig(
        provider=args.provider,
        model=model,
        temperature=0.2,  # Lower for more deterministic repairs
        max_tokens=4096,
        system_prompt=full_system_prompt,
    )
    planner = LLMPlanner(planner_config)
    
    # --- Configure kernel with arm-specific limits ---
    test_command = args.test_command or get_test_command(args.repo, args.instance_id)
    
    safety_envelope = SafetyEnvelope(
        max_steps=arm.max_steps,
        max_patches=arm.max_patches,
    )
    
    controller_config = ControllerConfig(
        default_test_command=test_command,
        test_timeout_seconds=300.0,
    )
    
    kernel_config = KernelConfig(
        controller_config=controller_config,
        safety_envelope=safety_envelope,
        verbose=args.verbose,
    )
    
    # --- Create ledger ---
    ledger_dir = args.repo / ".rfsn_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger = Ledger(ledger_dir / f"{args.instance_id}_{timestamp}.jsonl")
    
    # --- Run kernel ---
    logger.info(f"Running kernel with max_steps={arm.max_steps}, max_patches={arm.max_patches}")
    
    result = run_kernel(
        task_id=args.instance_id,
        task=problem_statement,
        workspace_path=args.repo,
        planner=planner,
        config=kernel_config,
        ledger=ledger,
    )
    
    # --- Extract fingerprints from final test result ---
    final_test = getattr(result.final_state, "last_test_result", None)
    test_summary = summarize_test_failure(final_test)
    fingerprints = fingerprints_from_test(final_test)
    
    # --- Build metrics ---
    metrics = {
        "completion_reason": result.completion_reason,
        "total_steps": result.total_steps,
        "accepted_proposals": result.accepted_proposals,
        "rejected_proposals": result.rejected_proposals,
        "test_status": result.test_status,
        "duration_seconds": result.duration_seconds,
        "repo_id": repo_id_from_path(args.repo),
        "test_summary": test_summary,
        "arm_max_steps": arm.max_steps,
        "arm_max_patches": arm.max_patches,
    }
    
    # --- Update upstream learner with outcome ---
    logger.info(f"Recording outcome: success={result.success}, fingerprints={len(fingerprints)}")
    
    learner.update(
        instance_id=args.instance_id,
        repo_path=args.repo,
        arm_id=args.arm_id,
        success=result.success,
        metrics=metrics,
        fingerprints=fingerprints,
    )
    
    return {
        "success": result.success,
        "completion_reason": result.completion_reason,
        "total_steps": result.total_steps,
        "accepted_proposals": result.accepted_proposals,
        "rejected_proposals": result.rejected_proposals,
        "test_status": result.test_status,
        "duration_seconds": result.duration_seconds,
        "ledger_path": str(ledger.path),
        "fingerprint_count": len(fingerprints),
        "arm_name": arm.name,
    }


def run_docker(
    args: argparse.Namespace,
    problem_statement: str,
) -> dict:
    """Run episode in Docker container."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(problem_statement)
        problem_file = f.name
    
    try:
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{args.repo}:/workspace",
            "-v", f"{problem_file}:/problem.txt",
            "-v", f"{args.artifacts_dir.resolve()}:/artifacts",
            "-e", f"DEEPSEEK_API_KEY={os.environ.get('DEEPSEEK_API_KEY', '')}",
            "-e", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}",
            "-e", f"ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', '')}",
            args.docker_image,
            "python", "/app/run_agent.py",
            "--task-id", args.instance_id,
            "--workspace", "/workspace",
            "--task", "@/problem.txt",
            "--provider", args.provider,
            "--artifacts-dir", "/artifacts",
            "--output", "/workspace/outcome.json",
        ]
        
        if args.arm_id:
            docker_cmd.extend(["--arm-id", args.arm_id])
        if args.model:
            docker_cmd.extend(["--model", args.model])
        
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        
        outcome_path = args.repo / "outcome.json"
        if outcome_path.exists():
            return json.loads(outcome_path.read_text())
        else:
            return {
                "success": False,
                "completion_reason": "No outcome file produced",
                "stdout": result.stdout[-2000:] if result.stdout else "",
                "stderr": result.stderr[-2000:] if result.stderr else "",
                "return_code": result.returncode,
            }
            
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "completion_reason": f"Timeout after {args.timeout}s",
        }
    finally:
        os.unlink(problem_file)


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Running SWE-bench episode: {args.instance_id}")
    logger.info(f"Provider: {args.provider}, Artifacts: {args.artifacts_dir}")
    
    # Validate repo
    if not args.repo.exists():
        logger.error(f"Repository not found: {args.repo}")
        return 1
    
    # Get problem statement
    if args.problem_statement:
        problem_statement = args.problem_statement.read_text()
    else:
        problem_file = args.repo / "problem_statement.txt"
        if problem_file.exists():
            problem_statement = problem_file.read_text()
        else:
            problem_statement = f"Fix the failing tests for instance {args.instance_id}"
    
    # Run episode
    start_time = datetime.now(timezone.utc)
    
    try:
        if args.use_docker:
            result = run_docker(args, problem_statement)
        else:
            result = run_native(args, problem_statement)
    except Exception as e:
        logger.exception(f"Episode failed: {e}")
        result = {
            "success": False,
            "completion_reason": f"Exception: {e}",
            "error": str(e),
        }
    
    end_time = datetime.now(timezone.utc)
    
    # Build outcome
    outcome = {
        "instance_id": args.instance_id,
        "arm_id": args.arm_id,
        "provider": args.provider,
        "model": args.model or DEFAULT_MODEL,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "total_duration_seconds": (end_time - start_time).total_seconds(),
        "artifacts_dir": str(args.artifacts_dir),
        **result,
    }
    
    # Write outcome
    args.output.write_text(json.dumps(outcome, indent=2))
    logger.info(f"Wrote outcome to {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Instance: {args.instance_id}")
    print(f"Arm: {args.arm_id} ({result.get('arm_name', 'N/A')})")
    print(f"Success: {result.get('success', False)}")
    print(f"Reason: {result.get('completion_reason', 'Unknown')}")
    print(f"Steps: {result.get('total_steps', 0)}")
    print(f"Fingerprints: {result.get('fingerprint_count', 0)}")
    print(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
    print(f"{'='*60}")
    
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
