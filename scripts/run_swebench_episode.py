#!/usr/bin/env python3
"""SWE-bench Episode Runner for CI/CD.

Runs a single SWE-bench episode with configurable settings.
Outputs a structured outcome.json for the CI pipeline.

Usage:
    python scripts/run_swebench_episode.py \
        --instance-id django__django-12345 \
        --repo /path/to/repo \
        --arm-id v_minimal_fix \
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a single SWE-bench episode",
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
        help="Path to problem statement file (optional, fetches from instance if not provided)",
    )
    parser.add_argument(
        "--test-command",
        default=None,
        help="Test command to run (default: auto-detect)",
    )
    
    # Arm selection
    parser.add_argument(
        "--arm-id",
        required=True,
        help="Prompt variant arm ID",
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
    # Check for common test patterns
    if (repo_path / "manage.py").exists():
        # Django project
        return "python -m pytest --tb=short -q"
    elif (repo_path / "setup.py").exists() or (repo_path / "pyproject.toml").exists():
        return "python -m pytest --tb=short -q"
    else:
        return "python -m pytest -q"


def run_native(
    args: argparse.Namespace,
    problem_statement: str,
) -> dict:
    """Run episode natively (without Docker)."""
    from rfsn_kernel import run_kernel, KernelConfig, SafetyEnvelope
    from rfsn_kernel.controller import ControllerConfig
    from rfsn_kernel.llm_planner import LLMPlanner, LLMPlannerConfig
    from rfsn_kernel.ledger import Ledger
    from rfsn_upstream import get_variant, format_prompt
    
    # Get prompt variant
    variant = get_variant(args.arm_id)
    
    # Configure planner with variant
    model = args.model or (
        "deepseek-chat" if args.provider == "deepseek" else
        "gpt-4-turbo" if args.provider == "openai" else
        "claude-3-5-sonnet-20241022"
    )
    
    planner_config = LLMPlannerConfig(
        provider=args.provider,
        model=model,
        temperature=variant.temperature,
        max_tokens=variant.max_tokens,
        system_prompt=variant.system_prompt,
    )
    planner = LLMPlanner(planner_config)
    
    # Get test command
    test_command = args.test_command or get_test_command(args.repo, args.instance_id)
    
    # Configure kernel
    safety_envelope = SafetyEnvelope(
        max_steps=20,
        max_patches=10,
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
    
    # Create ledger
    ledger_dir = args.repo / ".rfsn_ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ledger = Ledger(ledger_dir / f"{args.instance_id}_{timestamp}.jsonl")
    
    # Run kernel
    result = run_kernel(
        task_id=args.instance_id,
        task=problem_statement,
        workspace_path=args.repo,
        planner=planner,
        config=kernel_config,
        ledger=ledger,
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
    }


def run_docker(
    args: argparse.Namespace,
    problem_statement: str,
) -> dict:
    """Run episode in Docker container."""
    # Write problem statement to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(problem_statement)
        problem_file = f.name
    
    try:
        # Build docker command
        docker_cmd = [
            "docker", "run",
            "--rm",
            "-v", f"{args.repo}:/workspace",
            "-v", f"{problem_file}:/problem.txt",
            "-e", f"DEEPSEEK_API_KEY={os.environ.get('DEEPSEEK_API_KEY', '')}",
            "-e", f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}",
            "-e", f"ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', '')}",
            args.docker_image,
            "python", "/app/run_agent.py",
            "--task-id", args.instance_id,
            "--workspace", "/workspace",
            "--task", "@/problem.txt",
            "--provider", args.provider,
            "--arm-id", args.arm_id,
            "--output", "/workspace/outcome.json",
        ]
        
        if args.model:
            docker_cmd.extend(["--model", args.model])
        
        # Run with timeout
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        
        # Read outcome
        outcome_path = args.repo / "outcome.json"
        if outcome_path.exists():
            return json.loads(outcome_path.read_text())
        else:
            return {
                "success": False,
                "completion_reason": "No outcome file produced",
                "stdout": result.stdout,
                "stderr": result.stderr,
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
    logger.info(f"Arm: {args.arm_id}, Provider: {args.provider}")
    
    # Validate repo
    if not args.repo.exists():
        logger.error(f"Repository not found: {args.repo}")
        return 1
    
    # Get problem statement
    if args.problem_statement:
        problem_statement = args.problem_statement.read_text()
    else:
        # Try to find in repo or use placeholder
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
        **result,
    }
    
    # Write outcome
    args.output.write_text(json.dumps(outcome, indent=2))
    logger.info(f"Wrote outcome to {args.output}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Instance: {args.instance_id}")
    print(f"Arm: {args.arm_id}")
    print(f"Success: {result.get('success', False)}")
    print(f"Reason: {result.get('completion_reason', 'Unknown')}")
    print(f"Duration: {(end_time - start_time).total_seconds():.1f}s")
    print(f"{'='*60}")
    
    return 0 if result.get("success", False) else 1


if __name__ == "__main__":
    sys.exit(main())
