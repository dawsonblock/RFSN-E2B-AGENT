#!/usr/bin/env python3
"""Generate GitHub Actions summary report for SWE-bench runs."""

from __future__ import annotations

import json
import os
from pathlib import Path


def main() -> None:
    """Generate markdown summary."""
    # Get environment variables
    selected_arm = os.environ.get("SELECTED_ARM", "N/A")
    instance_id = os.environ.get("INSTANCE_ID", "N/A")
    provider = os.environ.get("PROVIDER", "N/A")
    run_id = os.environ.get("RUN_ID", "N/A")
    
    summary = """## 🛡️ RFSN Upstream Learning Report

### Episode Summary
"""

    # Load outcome if available
    outcome_path = Path("./outcomes/outcome.json")
    if outcome_path.exists():
        try:
            outcome = json.loads(outcome_path.read_text())
            status = "✅ Success" if outcome.get("success") else "❌ Failed"
            summary += f"""
| Field | Value |
|-------|-------|
| Instance | `{outcome.get('instance_id', 'N/A')}` |
| Arm | `{outcome.get('arm_id', 'N/A')}` ({outcome.get('arm_name', 'N/A')}) |
| Status | {status} |
| Reason | {outcome.get('completion_reason', 'N/A')} |
| Steps | {outcome.get('total_steps', 'N/A')} |
| Duration | {outcome.get('total_duration_seconds', 0):.1f}s |
| Fingerprints | {outcome.get('fingerprint_count', 0)} |
"""
        except Exception as e:
            summary += f"\n*Error reading outcome: {e}*\n"
    else:
        summary += "\n*No outcome file found*\n"

    # Add arm statistics
    try:
        from rfsn_upstream import UpstreamLearner, list_arm_ids
        
        learner = UpstreamLearner(
            bandit_db=Path("./artifacts/bandit.db"),
            outcomes_db=Path("./artifacts/outcomes.db"),
        )
        
        summary += """
### Strategy Arm Performance

| Arm | Success Rate | Pulls | Successes |
|-----|--------------|-------|-----------|
"""
        for arm_id in list_arm_ids():
            stats = learner.bandit.get_arm_stats(arm_id)
            rate = stats.successes / stats.pulls if stats.pulls > 0 else 0
            summary += f"| `{arm_id}` | {rate:.1%} | {stats.pulls} | {stats.successes} |\n"
        
        summary += f"\n**Total Episodes Recorded**: {learner.outcomes.total_episodes()}\n"
        
    except Exception as e:
        summary += f"\n*Could not load learner stats: {e}*\n"

    summary += f"""
### Workflow Info
- **Run ID**: `{run_id}`
- **Provider**: `{provider}`
- **Selected Arm**: `{selected_arm}`
"""

    print(summary)


if __name__ == "__main__":
    main()
